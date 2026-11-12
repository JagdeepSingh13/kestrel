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
from kestrel.core.types import Provider, Role, RunResult, StopReason
from kestrel.safety.paths import Workspace
from kestrel.safety.permissions import Permissions
from kestrel.safety.redact import Redactor
from kestrel.tools.builtin import BUILTIN_TOOLS
from kestrel.tools.registry import ToolContext, ToolRegistry

type OpenProvider = Callable[[str, ProviderConfig], AbstractAsyncContextManager[Provider]]


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
) -> RunResult | None:
    """Run `task` with `agent` to the end. Returns None if it could not start or the user
    pressed Ctrl-C.

    `conversation` carries earlier turns (the shell); the run adds to it. A turn that is
    stopped or fails is taken back out, so the next message starts from the last good one.
    """
    if provider_config.is_remote and not agent.allow_cloud:
        err_console.print(
            f"[red]Agent {escape(agent.name)} may not use a remote backend[/red] "
            f"({escape(provider_config.base_url)}). Set allow_cloud: true in its file to allow it."
        )
        return None
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
    context.window = agent.context.window
    context.max_tool_result_tokens = agent.context.max_tool_result_tokens
    context.tools = tools.specs()  # the agent may have changed since the last turn
    kept = len(context.history)
    limits = agent.limits

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
        async with open_provider(provider_name, provider_config) as provider:
            return await run_agent(
                task,
                provider=provider,
                model=model,
                toolbox=tools,
                bus=bus,
                context=context,
                budgets=Budgets(limits.max_steps, limits.max_tokens_total, limits.max_wall_seconds),
                gate=Permissions(frontend.confirm, agent.permissions),
                temperature=agent.temperature,
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
    return result
