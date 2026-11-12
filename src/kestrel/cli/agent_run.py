"""Start one agent run with live output and a log file. Shared by the shell (every message
is a run) and the terminal's `kestrel run`, so both behave the same."""

import asyncio
import contextlib
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager

from rich.markup import escape

from kestrel.agents.schema import AgentConfig
from kestrel.cli.frontend import BottomBox, CliFrontend
from kestrel.cli.keys import watch_stop_keys
from kestrel.cli.output import err_console
from kestrel.core.config import ProviderConfig
from kestrel.core.context import UNTRUSTED_RULE, ContextManager
from kestrel.core.events import EventBus, JsonlWriter, new_run_id, runs_dir
from kestrel.core.loop import DEFAULT_SYSTEM_PROMPT, Budgets, run_agent
from kestrel.core.types import Event, EventType, Provider, Role, RunResult, StopReason
from kestrel.providers.profiles import Profile, ProfileCache, cache_path, probe
from kestrel.reliability.watchdog import WatchLimits
from kestrel.safety.paths import Workspace
from kestrel.safety.permissions import Permissions
from kestrel.safety.redact import Redactor
from kestrel.tools.builtin import BUILTIN_TOOLS
from kestrel.tools.registry import ToolContext, ToolRegistry

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
) -> RunResult | None:
    """Run `task` with `agent` to the end. Returns None if it could not start or the user
    pressed Ctrl-C.

    `conversation` carries earlier turns (the shell); the run adds to it. A turn that is
    stopped or fails is taken back out, so the next message starts from the last good one.
    `think=False` (the default) asks the model not to think first; None leaves it be.
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
            how = {
                "ollama": f"restart Ollama with OLLAMA_CONTEXT_LENGTH={window} ollama serve",
                "llamacpp": f"start llama-server with -c {window}",
            }.get(profile.kind, "raise the context length in the backend's settings")
            err_console.print(
                f"[yellow]{escape(model)} is loaded with a {profile.window}-token window"
                f"[/yellow], so Kestrel uses that instead of {window}. For more: {how}.",
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
        ToolContext(workspace=workspace, env=env, ask=frontend.ask),
    )
    context = conversation or ContextManager(None)
    context.system_prompt = f"{agent.system or DEFAULT_SYSTEM_PROMPT}\n\n{UNTRUSTED_RULE}"
    context.window = window
    context.token_ratio = profile.estimate_ratio
    context.max_tool_result_tokens = agent.context.max_tool_result_tokens
    context.tools = tools.specs()  # the agent may have changed since the last turn
    kept = len(context.history)
    limits = agent.limits
    measured: list[tuple[int, int]] = []
    bus.subscribe(lambda event: _measure(event, measured))

    async def go() -> RunResult:
        task = asyncio.current_task()
        assert task is not None

        def stop() -> None:
            if not task.done():
                task.cancel()

        if box is not None:  # the box owns the keyboard; its Esc stops us (from its thread)
            loop = asyncio.get_running_loop()

            def stop_from_box() -> None:
                with contextlib.suppress(RuntimeError):  # the turn's loop already closed
                    loop.call_soon_threadsafe(stop)

            box.set_cancel(stop_from_box)
            try:
                return await work()
            finally:
                box.set_cancel(None)
        with watch_stop_keys(stop) as pause:
            if pause is not None:
                frontend.pause_keys = pause
                frontend.stop_hint = True
            return await work()

    async def work() -> RunResult:
        bus.emit(EventType.profile, 0, profile.model_dump(mode="json"))
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
                    max_repairs=limits.max_repairs,
                    watch=WatchLimits(
                        max_repeats=limits.max_repeats,
                        max_stall_steps=limits.max_stall_steps,
                        max_tool_errors=limits.max_tool_errors,
                    ),
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
        unanswered = context.history[-1:] and context.history[-1].role is Role.user
        if result is None or result.stopped_reason is StopReason.error or unanswered:
            del context.history[kept:]  # never leave the conversation ending on a question
        frontend.stop()
        writer.close()
    save_profile(profile, measured, provider_name, provider_config, cache)
    return result


_probed: set[tuple[str, str]] = set()
"""(backend, model) pairs asked this session: once per session keeps profiles fresh
(a re-pulled model, a restarted backend) without asking on every turn."""
_warned_windows: set[tuple[str, int]] = set()


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


def save_profile(
    profile: Profile,
    measured: list[tuple[int, int]],
    name: str,
    config: ProviderConfig,
    cache: ProfileCache,
) -> None:
    """Fold this run's token measurements in; learn the loaded window if still unknown
    (the model is loaded now, so the backend can say)."""
    for reported, estimated in measured:
        profile = profile.with_sample(reported, estimated)
    if profile.window is None:
        with contextlib.suppress(KeyboardInterrupt):
            profile = asyncio.run(probe_profile(name, config.base_url, profile.model, profile))
    cache.save(profile)


def _measure(event: Event, measured: list[tuple[int, int]]) -> None:
    """Pair each step's raw estimate with the backend's prompt token count."""
    if event.type is EventType.model_request:
        raw = event.data.get("raw_est_tokens")
        if isinstance(raw, int):
            measured.append((0, raw))
    elif event.type is EventType.model_response and measured:
        usage = event.data.get("usage")
        reported = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if isinstance(reported, int):
            measured[-1] = (reported, measured[-1][1])
