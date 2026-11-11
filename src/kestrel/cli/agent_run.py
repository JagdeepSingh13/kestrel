"""Start one agent run with live output and a log file. Shared by the shell's `/run`
and the terminal's `kestrel run`, so both behave the same."""

import asyncio
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager

from rich.markup import escape

from kestrel.agents.schema import AgentConfig
from kestrel.cli.frontend import CliFrontend
from kestrel.cli.output import err_console
from kestrel.core.config import ProviderConfig
from kestrel.core.context import UNTRUSTED_RULE, ContextManager
from kestrel.core.events import EventBus, JsonlWriter, new_run_id, runs_dir
from kestrel.core.loop import DEFAULT_SYSTEM_PROMPT, Budgets, run_agent
from kestrel.core.types import Provider, RunResult
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
) -> RunResult | None:
    """Run `task` with `agent` to the end. Returns None if it could not start or the user
    pressed Ctrl-C."""
    if provider_config.is_remote and not agent.allow_cloud:
        err_console.print(
            f"[red]Agent {escape(agent.name)} may not use a remote backend[/red] "
            f"({escape(provider_config.base_url)}). Set allow_cloud: true in its file to allow it."
        )
        return None
    run_id = new_run_id()
    writer = JsonlWriter(runs_dir(env) / f"{run_id}.jsonl", Redactor(env).json)
    frontend = CliFrontend(show_thinking=show_thinking)
    bus = EventBus(run_id)
    bus.subscribe(writer)  # every step goes to the log file...
    bus.subscribe(frontend.on_event)  # ...and to the screen
    tools = ToolRegistry(
        [BUILTIN_TOOLS[name] for name in agent.tools],
        ToolContext(workspace=workspace, env=env, ask=frontend.ask),
    )
    system = f"{agent.system or DEFAULT_SYSTEM_PROMPT}\n\n{UNTRUSTED_RULE}"
    context = ContextManager(
        system,
        window=agent.context.window,
        max_tool_result_tokens=agent.context.max_tool_result_tokens,
        tools=tools.specs(),
    )
    limits = agent.limits

    async def go() -> RunResult:
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

    try:
        return asyncio.run(go())
    except KeyboardInterrupt:
        err_console.print("[dim]Stopped. The run so far is in the log.[/dim]")
        return None
    finally:
        frontend.stop()
        writer.close()
        err_console.print(f"[dim]Run id {run_id}. See it with /trace[/dim]")
