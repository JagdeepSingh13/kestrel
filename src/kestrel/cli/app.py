"""Typer application: the `kestrel` command."""

import asyncio
import os
import shutil
import sys
from collections.abc import Sequence
from typing import Annotated

import typer
from rich.markup import escape

from kestrel.__about__ import __app_name__, __version__
from kestrel.agents.loader import AgentError, load_agent
from kestrel.cli import agent_run, output
from kestrel.cli.agent_run import start_run
from kestrel.cli.ladder import RungSpec, ladder_home, parse_rung
from kestrel.cli.output import console, describe_usage, err_console, print_reports
from kestrel.cli.trace import print_run_list, print_trace
from kestrel.core.config import (
    ConfigError,
    KestrelConfig,
    ProviderConfig,
    load_config,
)
from kestrel.core.events import find_run, runs_dir
from kestrel.core.types import Message, ModelResponse, ProviderError, Role, StopReason
from kestrel.providers.discovery import Status, discover

app = typer.Typer(name=__app_name__, help="Local-first agent harness for small local models.")

ModelOption = Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model name. Default: default_model in config."),
]
ProviderOption = Annotated[
    str | None,
    typer.Option("--provider", "-p", help="Provider name. Default: default_provider."),
]


def _version_callback(value: bool) -> None:
    """Print the version and exit when `--version` is passed."""
    if value:
        console.print(f"{__app_name__} {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Kestrel: local-first agent harness for small local models.

    With no command, opens the interactive shell (in a terminal) or shows this help.
    """
    if ctx.invoked_subcommand is not None:
        return
    if sys.stdin.isatty() and sys.stdout.isatty():
        _start_shell(model=None, provider=None)
    else:
        console.print(ctx.get_help())


@app.command()
def chat(model: ModelOption = None, provider: ProviderOption = None) -> None:
    """Open the interactive shell: talk to a model, use /commands."""
    _start_shell(model=model, provider=provider)


@app.command()
def doctor() -> None:
    """Find running local backends and the models they serve."""
    config = _load_config_or_exit()
    reports = asyncio.run(discover(config.providers, env=os.environ))
    print_reports(reports, console)
    if not any(r.status is Status.up for r in reports):
        if shutil.which("ollama"):
            hint = "Ollama is installed but not running: open the Ollama app or run `ollama serve`."
        else:
            hint = "Install one, e.g. `brew install ollama`, then `ollama pull qwen3:8b`."
        console.print(f"No backend is running. {hint}")
        raise typer.Exit(1)


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="What to ask the model.")],
    model: ModelOption = None,
    provider: ProviderOption = None,
    show_thinking: Annotated[
        bool, typer.Option("--show-thinking", help="Print the model's reasoning, dimmed.")
    ] = False,
    think: Annotated[
        bool, typer.Option("--think", help="Let the model think first (slower).")
    ] = False,
) -> None:
    """Send one message to a model and stream the answer. No agent, no tools."""
    provider_name, provider_config, model_name = _resolve_or_exit(provider, model)
    messages = [Message(role=Role.user, content=prompt)]
    try:
        response = asyncio.run(
            _ask(
                provider_name,
                provider_config,
                messages,
                model_name,
                show_thinking,
                None if think else False,
            )
        )
    except ProviderError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        err_console.print("Run `kestrel doctor` to check backends and models.")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        err_console.print("[dim]Stopped.[/dim]")
        raise typer.Exit(130) from None
    if response.usage is not None:
        err_console.print(f"[dim]{describe_usage(response.usage)}[/dim]")


async def _ask(
    name: str,
    config: ProviderConfig,
    messages: Sequence[Message],
    model: str,
    show_thinking: bool,
    think: bool | None,
) -> ModelResponse:
    async with output.open_provider(name, config) as provider:
        return await output.stream_answer(
            provider, messages, model=model, show_thinking=show_thinking, think=think
        )


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="What the agent should do.")],
    model: ModelOption = None,
    provider: ProviderOption = None,
    agent: Annotated[str | None, typer.Option(help="Agent name or .yaml path.")] = None,
    trust_agent: Annotated[
        bool, typer.Option("--trust-agent", help="Allow the agent file to loosen permissions.")
    ] = False,
    show_thinking: Annotated[
        bool, typer.Option("--show-thinking", help="Print the model's reasoning, dimmed.")
    ] = False,
    think: Annotated[
        bool, typer.Option("--think", help="Let the model think first (slower).")
    ] = False,
) -> None:
    """Run an agent on a task, with its tools, inside the workspace folder."""
    config = _load_config_or_exit()
    try:
        chosen = load_agent(agent or config.default_agent, os.environ, trusted=trust_agent)
    except AgentError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from None
    provider_name, provider_config, model_name = _resolve_or_exit(
        provider or chosen.provider, model or (chosen.model[0] if chosen.model else None)
    )
    result = start_run(
        task,
        agent=chosen,
        workspace=agent_run.default_workspace(),
        provider_name=provider_name,
        provider_config=provider_config,
        open_provider=output.open_provider,
        model=model_name,
        env=os.environ,
        uppers=_agent_rungs(chosen.model, model_name, config, provider_name),
        show_thinking=show_thinking,
        think=None if think else False,
    )
    if result is None:
        raise typer.Exit(130)
    if result.stopped_reason is not StopReason.final:
        raise typer.Exit(1)


@app.command()
def trace(
    run: Annotated[str, typer.Argument(help="Run id, a unique prefix, or 'last'.")] = "last",
    full: Annotated[bool, typer.Option("--full", help="Show full text, not previews.")] = False,
    list_runs: Annotated[bool, typer.Option("--list", help="List recent runs.")] = False,
) -> None:
    """Show what happened in a recorded run."""
    directory = runs_dir(os.environ)
    if list_runs:
        print_run_list(directory, console)
        return
    path = find_run(directory, run)
    if path is None:
        err_console.print(f"[red]No single run matches {escape(run)!r}.[/red]")
        print_run_list(directory, err_console)
        raise typer.Exit(2)
    print_trace(path, console, full=full)


def _start_shell(model: str | None, provider: str | None) -> None:
    from kestrel.cli.prompt import input_history
    from kestrel.cli.shell import Session, Shell
    from kestrel.cli.tui import Tui

    config = _load_config_or_exit()
    provider_name, _ = _pick_provider_or_exit(config, provider)
    try:
        agent = load_agent(config.default_agent, os.environ)
    except AgentError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(2) from None
    first_model = agent.model[0] if agent.model else None
    session = Session(
        provider=provider_name, model=model or first_model or config.default_model, agent=agent
    )
    shell = Shell(
        config,
        session,
        open_provider=output.open_provider,
        env=os.environ,
        workspace=agent_run.default_workspace(),
    )
    Tui(shell, input_history()).run()


def _resolve_or_exit(provider: str | None, model: str | None) -> tuple[str, ProviderConfig, str]:
    """Provider name, its config, and the model, from options or config defaults."""
    config = _load_config_or_exit()
    provider_name, provider_config = _pick_provider_or_exit(config, provider)
    model_name = model or config.default_model
    if model_name is None:
        err_console.print(
            "[red]No model given.[/red] Pass --model, or set default_model in config.toml. "
            "Run `kestrel doctor` to see which models are available."
        )
        raise typer.Exit(2)
    if provider_config.is_remote:
        err_console.print(
            f"[bold yellow]REMOTE[/bold yellow] {escape(provider_config.base_url)}: "
            "your prompt leaves this machine."
        )
    return provider_name, provider_config, model_name


def _load_config_or_exit() -> KestrelConfig:
    try:
        return load_config()
    except ConfigError as exc:
        err_console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from None


def _pick_provider_or_exit(config: KestrelConfig, name: str | None) -> tuple[str, ProviderConfig]:
    chosen = name or config.default_provider
    try:
        return chosen, config.provider(chosen)
    except ConfigError as exc:
        err_console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from None


def _agent_rungs(
    models: list[str], first: str, config: KestrelConfig, provider: str
) -> list[RungSpec]:
    """The agent file's ladder above the model the run starts on."""
    try:
        home = ladder_home(config, provider if not config.provider(provider).is_remote else None)
        return [parse_rung(m, config, home) for m in models[1:] if m != first]
    except ValueError as exc:
        err_console.print(f"[red]Ladder:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from None
