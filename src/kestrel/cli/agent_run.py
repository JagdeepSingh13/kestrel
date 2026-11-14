"""Start one agent run with live output and a log file. Shared by the shell (every message
is a run) and the terminal's `kestrel run`, so both behave the same."""

import asyncio
import contextlib
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import date
from pathlib import Path

from rich.markup import escape

from kestrel.agents.schema import AgentConfig
from kestrel.cli.frontend import BottomBox, CliFrontend
from kestrel.cli.ladder import RungSpec, ShellLadder
from kestrel.cli.output import err_console, window_fix
from kestrel.core.config import ProviderConfig
from kestrel.core.context import NOTE_PREFIX, UNTRUSTED_RULE, ContextManager
from kestrel.core.events import EventBus, JsonlWriter, new_run_id, runs_dir
from kestrel.core.loop import DEFAULT_SYSTEM_PROMPT, Budgets, run_agent
from kestrel.core.types import (
    EventType,
    JSONObject,
    Message,
    Provider,
    Role,
    Rung,
    RunResult,
    StopReason,
)
from kestrel.providers.profiles import Profile, ProfileCache, cache_path, probe
from kestrel.safety.paths import Workspace
from kestrel.safety.permissions import Permissions
from kestrel.safety.redact import Redactor
from kestrel.tools.builtin import BUILTIN_TOOLS
from kestrel.tools.mcp import McpHub, McpStartError, command_line, mcp_tools
from kestrel.tools.registry import FileStamp, Tool, ToolContext, ToolError, ToolRegistry
from kestrel.tools.research import urls_in

type OpenProvider = Callable[[str, ProviderConfig], AbstractAsyncContextManager[Provider]]


async def probe_profile(name: str, base_url: str, model: str, known: Profile | None) -> Profile:
    """Ask the backend about the model (a few metadata calls). Tests replace this."""
    return await probe(name, base_url, model, known=known)


def default_workspace() -> Workspace:
    """The agents' only folder (`<project>/workspace`). Tests replace this."""
    return Workspace.default()


def start_run(
    task: str,
    *,
    agent: AgentConfig,
    workspace: Workspace,
    provider_name: str,
    provider_config: ProviderConfig,
    open_provider: OpenProvider,
    model: str,
    env: Mapping[str, str],
    show_thinking: bool = False,
    conversation: ContextManager | None = None,
    box: BottomBox | None = None,
    think: bool | None = False,
    seen_files: dict[Path, FileStamp] | None = None,
    uppers: Sequence[RungSpec] = (),
    start: int = 0,
    cloud: bool = False,
    session_id: str | None = None,
    mcp_hub: McpHub | None = None,
) -> RunResult | None:
    """Run `task` with `agent` to the end. Returns None if it could not start or the user
    pressed Ctrl-C.

    `conversation` carries earlier turns (the shell); the run adds to it. A turn that is
    stopped or fails is taken back out, so the next message starts from the last good one.
    `think=False` (the default) asks the model not to think first; None leaves it be.
    `seen_files` remembers which files the model has read, across a shell conversation.
    `uppers` are the ladder's rungs above `model` (M4.5); `start` > 0 begins that many
    rungs up (`/up`). `cloud` is the shell's `/cloud on`: cloud models are allowed for
    this session even if the agent file does not say `allow_cloud`.
    """
    # /cloud on only opens the ladder's cloud rung (asked step by step); running the
    # whole conversation on a remote backend still needs the agent file's allow_cloud.
    allow_cloud = agent.allow_cloud or cloud
    if provider_config.is_remote and not agent.allow_cloud:
        err_console.print(
            f"[red]Agent {escape(agent.name)} may not use a remote backend[/red] "
            f"({escape(provider_config.base_url)}). Switch back with /provider, or set "
            "allow_cloud: true in its file. (/cloud on is for stuck steps only.)"
        )
        return None
    cache = ProfileCache(cache_path(env))
    profile = load_profile(provider_name, provider_config, model, cache)
    if agent.tools and profile.native_tools is False:
        err_console.print(
            f"[red]{escape(model)} cannot use tools on this backend[/red] (its chat template "
            "has no tool support). Pick another model with /model, or an agent without tools."
        )
        return None
    window = agent.context.window
    if profile.window is not None and profile.window < window:
        if (model, profile.window) not in _warned_windows:  # once per session
            _warned_windows.add((model, profile.window))
            err_console.print(
                f"[bold yellow]Small context: {escape(model)} is loaded with a "
                f"{profile.window}-token window[/bold yellow] (the agent wants {window}), so "
                "long chats get trimmed early and the model forgets.\n"
                f"  To fix: {escape(window_fix(profile.kind, window))}",
                highlight=False,
            )
        window = profile.window
    run_id = new_run_id()
    writer = JsonlWriter(runs_dir(env) / f"{run_id}.jsonl", Redactor(env).json)
    frontend = CliFrontend(show_thinking=show_thinking, box=box)
    bus = EventBus(run_id)
    bus.subscribe(writer)  # every step goes to the log file...
    bus.subscribe(frontend.on_event)  # ...and to the screen
    step_now = [0]
    """The loop's current step, so events from inside a tool land on it."""
    bus.subscribe(lambda event: step_now.__setitem__(0, event.step))

    def emit(kind: str, data: JSONObject) -> None:
        bus.emit(EventType(kind), step_now[0], data)

    extra = _mcp_tools(agent, mcp_hub, bus)
    if extra is None:
        writer.close()
        return None
    tools = ToolRegistry(
        [*(BUILTIN_TOOLS[name] for name in agent.tools), *extra],
        ToolContext(
            workspace=workspace,
            env=env,
            ask=frontend.ask,
            seen=seen_files if seen_files is not None else {},
            emit=emit,
            user_urls=frozenset(_user_urls(task, conversation)),
            window=window,
        ),
    )
    context = conversation or ContextManager(None)
    notes = [today_note(date.today())]
    if extra:
        notes.append(outside_paths_note(workspace.root))
    context.system_prompt = "\n\n".join(
        [agent.system or DEFAULT_SYSTEM_PROMPT, *notes, UNTRUSTED_RULE]
    )
    context.window = window
    context.max_tool_result_tokens = agent.context.max_tool_result_tokens
    context.tools = tools.specs()  # the agent may have changed since the last turn
    kept = len(context.history)
    seen_before = dict(tools.context.seen)
    limits = agent.limits

    async def go() -> RunResult:
        task = asyncio.current_task()
        assert task is not None

        def stop() -> None:
            if not task.done():
                task.cancel()

        if box is None:  # `kestrel run`: Ctrl-C stops it
            return await work()
        loop = asyncio.get_running_loop()  # the box's Esc stops the turn (from its thread)

        def stop_from_box() -> None:
            with contextlib.suppress(RuntimeError):  # the turn's loop already closed
                loop.call_soon_threadsafe(stop)

        box.set_cancel(stop_from_box)
        try:
            return await work()
        finally:
            box.set_cancel(None)

    async def work() -> RunResult:
        bus.emit(
            EventType.profile,
            0,
            {**profile.model_dump(mode="json"), "wanted_window": agent.context.window},
        )
        async with contextlib.AsyncExitStack() as stack:
            provider = await stack.enter_async_context(
                open_provider(provider_name, provider_config)
            )

            async def complete(system: str, user: str, max_tokens: int) -> str:
                reply = await provider.chat(
                    [
                        Message(role=Role.system, content=system),
                        Message(role=Role.user, content=user),
                    ],
                    None,
                    model=model,
                    temperature=0.1,
                    max_tokens=max_tokens,
                    think=False,
                )
                return reply.message.content or ""

            tools.context = dataclasses.replace(tools.context, complete=complete)
            ladder = ShellLadder(
                Rung(provider, model, model, window),
                uppers,
                stack=stack,
                open_provider=open_provider,
                load_profile=load_profile_async,
                cache=cache,
                env=dict(env),
                allow_cloud=allow_cloud,
                approve=frontend.approve,
                agent_window=agent.context.window,
                start=start,
                emit=emit,
            )
            await ladder.begin()
            return await run_agent(
                task,
                ladder=ladder,
                labels={"session_id": session_id} if session_id else None,
                toolbox=tools,
                bus=bus,
                context=context,
                budgets=Budgets(
                    limits.max_steps,
                    limits.max_tokens_total,
                    limits.max_wall_seconds,
                ),
                gate=Permissions(frontend.confirm, agent.permissions),
                temperature=agent.temperature,
                think=think,
            )

    result: RunResult | None = None
    try:
        result = asyncio.run(go())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass  # the frontend already showed "Stopped"; the run so far is in the log
    finally:
        while context.history[kept:] and _is_note(context.history[-1]):
            context.history.pop()  # a nudge the run ended before answering: the reply stands
        unanswered = context.history[-1:] and context.history[-1].role is Role.user
        if result is None or result.stopped_reason is StopReason.error or unanswered:
            del context.history[kept:]  # never leave the conversation ending on a question
            tools.context.seen.clear()  # reads in the dropped turn are gone from the model's view
            tools.context.seen.update(seen_before)
        frontend.stop()
        writer.close()
    return result


_probed: set[tuple[str, str]] = set()
"""(backend, model) pairs asked this session: once per session keeps profiles fresh
(a re-pulled model, a restarted backend) without asking on every turn."""
_warned_windows: set[tuple[str, int]] = set()


MANY_TOOLS = 12
"""More tools than this and a small model often picks the wrong one: a warning."""
TOO_MANY_TOOLS = 24


def _mcp_tools(agent: AgentConfig, hub: McpHub | None, bus: EventBus) -> list[Tool] | None:
    """The agent's MCP tools, starting their servers if needed; None if the run cannot
    start (a missing or failing server, too many tools)."""
    if not agent.mcp:
        return []
    if hub is None:
        err_console.print("[red]MCP tools are not available here.[/red]")
        return None
    found: list[Tool] = []
    for server, wanted in agent.mcp.items():
        config = hub.servers.get(server)
        if config is None:
            known = ", ".join(hub.servers) or "none configured"
            err_console.print(
                f"[red]Agent {escape(agent.name)} wants MCP server {escape(server)}, which "
                f"is not in your config[/red] (servers: {escape(known)})."
            )
            return None
        if config.is_remote and not agent.allow_cloud:
            err_console.print(
                f"[red]MCP server {escape(server)} is remote ({escape(config.url or '')}); "
                f"agent {escape(agent.name)} needs allow_cloud: true.[/red]"
            )
            return None
        first = server not in hub.running()
        if first:
            err_console.print(
                f"[dim]Starting MCP server {escape(server)} (up to 2 minutes the first "
                f"time; Esc cannot stop it): "
                f"{escape(command_line(server, config, hub.workspace))}[/dim]"
            )
        try:
            listed = hub.tools(server)
            tools = mcp_tools(hub, server, wanted, listed)
        except (McpStartError, ToolError) as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            return None
        if first:
            bus.emit(
                EventType.mcp_start,
                0,
                {
                    "server": server,
                    "command": command_line(server, config, hub.workspace),
                    "tools": [t.name for t in listed],
                },
            )
        found += tools
    total = len(agent.tools) + len(found)
    if total > TOO_MANY_TOOLS:
        err_console.print(
            f"[red]Agent {escape(agent.name)} would have {total} tools; at most "
            f"{TOO_MANY_TOOLS}. List fewer MCP tools by name in its file.[/red]"
        )
        return None
    if total > MANY_TOOLS:
        err_console.print(
            f"[yellow]{total} tools: small models choose better from fewer. Consider "
            "listing only the MCP tools this agent needs.[/yellow]"
        )
    return found


def _user_urls(task: str, conversation: ContextManager | None) -> set[str]:
    """Links the user typed, in this message or earlier ones (never in tool output)."""
    found: set[str] = urls_in(task)
    for message in conversation.history if conversation else []:
        if message.role is Role.user and not _is_note(message):
            found |= urls_in(message.content or "")
    return found


def _is_note(message: Message) -> bool:
    return message.role is Role.user and (message.content or "").startswith(NOTE_PREFIX)


def today_note(today: date) -> str:
    """Small models think it is still their training year; tell them the date."""
    return (
        f"Today is {today:%A}, {today.isoformat()}. Your training data is older than this: "
        "when a year or date is needed and the user gave none, use today's, and say so."
    )


def outside_paths_note(root: Path) -> str:
    """MCP servers know nothing of Kestrel's relative paths; small models guess
    "/workspace". Give them the real folder."""
    return (
        f"The workspace folder is {root}. Tools named server__tool come from outside "
        f"servers: give them full paths inside it, like {root / 'notes.txt'}."
    )


def load_profile(name: str, config: ProviderConfig, model: str, cache: ProfileCache) -> Profile:
    """The cached profile, or a fresh probe of the backend's metadata (a few local calls,
    no prompt). Never fails: an unreachable backend gives an empty profile."""
    try:
        return asyncio.run(load_profile_async(name, config, model, cache))
    except KeyboardInterrupt:
        return cache.get(config.base_url, model) or Profile(backend=config.base_url, model=model)


async def load_profile_async(
    name: str, config: ProviderConfig, model: str, cache: ProfileCache
) -> Profile:
    """`load_profile` from inside a running loop (a ladder rung probed mid-run)."""
    known = cache.get(config.base_url, model)
    if known is not None and (config.base_url, model) in _probed:
        return known  # asked once this session already
    if not known:
        err_console.print(f"[dim]Checking {escape(model)} on {escape(name)}...[/dim]")
    fresh = await probe_profile(name, config.base_url, model, known)
    _probed.add((config.base_url, model))
    cache.save(fresh)
    return fresh
