"""Start one agent run with live output and a log file. Shared by the shell (every message
is a run) and the terminal's `kestrel run`, so both behave the same."""

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from datetime import date
from pathlib import Path

from rich.markup import escape

from kestrel.agents.schema import AgentConfig
from kestrel.cli.frontend import BottomBox, CliFrontend
from kestrel.cli.output import err_console, window_fix
from kestrel.core.config import ProviderConfig
from kestrel.core.context import NOTE_PREFIX, UNTRUSTED_RULE, ContextManager
from kestrel.core.events import EventBus, JsonlWriter, new_run_id, runs_dir
from kestrel.core.loop import DEFAULT_SYSTEM_PROMPT, Budgets, run_agent
from kestrel.core.types import EventType, Message, Provider, Role, RunResult, StopReason
from kestrel.providers.profiles import Profile, ProfileCache, cache_path, probe
from kestrel.safety.paths import Workspace
from kestrel.safety.permissions import Permissions
from kestrel.safety.redact import Redactor
from kestrel.tools.builtin import BUILTIN_TOOLS
from kestrel.tools.registry import FileStamp, ToolContext, ToolRegistry

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
) -> RunResult | None:
    """Run `task` with `agent` to the end. Returns None if it could not start or the user
    pressed Ctrl-C.

    `conversation` carries earlier turns (the shell); the run adds to it. A turn that is
    stopped or fails is taken back out, so the next message starts from the last good one.
    `think=False` (the default) asks the model not to think first; None leaves it be.
    `seen_files` remembers which files the model has read, across a shell conversation.
    """
    if provider_config.is_remote and not agent.allow_cloud:
        err_console.print(
            f"[red]Agent {escape(agent.name)} may not use a remote backend[/red] "
            f"({escape(provider_config.base_url)}). Set allow_cloud: true in its file to allow it."
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
    tools = ToolRegistry(
        [BUILTIN_TOOLS[name] for name in agent.tools],
        ToolContext(
            workspace=workspace,
            env=env,
            ask=frontend.ask,
            seen=seen_files if seen_files is not None else {},
        ),
    )
    context = conversation or ContextManager(None)
    context.system_prompt = (
        f"{agent.system or DEFAULT_SYSTEM_PROMPT}\n\n{today_note(date.today())}\n\n{UNTRUSTED_RULE}"
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
        async with open_provider(provider_name, provider_config) as provider:
            return await run_agent(
                task,
                provider=provider,
                model=model,
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


def _is_note(message: Message) -> bool:
    return message.role is Role.user and (message.content or "").startswith(NOTE_PREFIX)


def today_note(today: date) -> str:
    """Small models think it is still their training year; tell them the date."""
    return (
        f"Today is {today:%A}, {today.isoformat()}. Your training data is older than this: "
        "when a year or date is needed and the user gave none, use today's, and say so."
    )


def load_profile(name: str, config: ProviderConfig, model: str, cache: ProfileCache) -> Profile:
    """The cached profile, or a fresh probe of the backend's metadata (a few local calls,
    no prompt). Never fails: an unreachable backend gives an empty profile."""
    known = cache.get(config.base_url, model)
    if known is not None and (config.base_url, model) in _probed:
        return known  # asked once this session already
    if not known:
        err_console.print(f"[dim]Checking {escape(model)} on {escape(name)}...[/dim]")
    try:
        fresh = asyncio.run(probe_profile(name, config.base_url, model, known))
    except KeyboardInterrupt:
        return known or Profile(backend=config.base_url, model=model)
    _probed.add((config.base_url, model))
    cache.save(fresh)
    return fresh
