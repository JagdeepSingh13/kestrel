"""Start one agent run with live output and a log file. Shared by the shell's `/run`
and the terminal's `kestrel run`, so both behave the same."""

import asyncio
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager

from kestrel.cli.frontend import CliFrontend
from kestrel.cli.output import err_console
from kestrel.core.config import ProviderConfig
from kestrel.core.context import DEFAULT_WINDOW, ContextManager
from kestrel.core.events import EventBus, JsonlWriter, new_run_id, runs_dir
from kestrel.core.loop import DEFAULT_SYSTEM_PROMPT, Budgets, run_agent
from kestrel.core.types import Provider, RunResult
from kestrel.safety.redact import Redactor
from kestrel.tools.demo import DemoToolbox

type OpenProvider = Callable[[str, ProviderConfig], AbstractAsyncContextManager[Provider]]


def start_run(
    task: str,
    *,
    provider_name: str,
    provider_config: ProviderConfig,
    open_provider: OpenProvider,
    model: str,
    env: Mapping[str, str],
    show_thinking: bool = False,
    max_steps: int = 15,
    window: int = DEFAULT_WINDOW,
) -> RunResult | None:
    """Run `task` to the end. Returns None if the user pressed Ctrl-C."""
    run_id = new_run_id()
    writer = JsonlWriter(runs_dir(env) / f"{run_id}.jsonl", Redactor(env).json)
    frontend = CliFrontend(show_thinking=show_thinking)
    bus = EventBus(run_id)
    bus.subscribe(writer)  # every step goes to the log file...
    bus.subscribe(frontend.on_event)  # ...and to the screen
    toolbox = DemoToolbox()
    context = ContextManager(DEFAULT_SYSTEM_PROMPT, window=window, tools=toolbox.specs())

    async def go() -> RunResult:
        async with open_provider(provider_name, provider_config) as provider:
            return await run_agent(
                task,
                provider=provider,
                model=model,
                toolbox=toolbox,
                bus=bus,
                context=context,
                budgets=Budgets(max_steps=max_steps),
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
