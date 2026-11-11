"""Typer application: the `kestrel` command."""

import asyncio
import os
import shutil
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kestrel.__about__ import __app_name__, __version__
from kestrel.core.config import ConfigError, ProviderConfig, load_config
from kestrel.core.types import (
    BadBackendResponseError,
    Message,
    ModelResponse,
    Provider,
    ProviderError,
    Role,
)
from kestrel.providers.discovery import BackendReport, Status, discover
from kestrel.providers.openai_compat import OpenAICompatProvider

app = typer.Typer(
    name=__app_name__,
    help="Local-first agent harness for small local models.",
    no_args_is_help=True,
)
console = Console()
"""Standard output: only the model's answer goes here, so `kestrel ask ... > file` works."""
err_console = Console(stderr=True)
"""Standard error: thinking, progress, token counts, warnings and errors."""


def _version_callback(value: bool) -> None:
    """Print the version and exit when `--version` is passed."""
    if value:
        console.print(f"{__app_name__} {__version__}")
        raise typer.Exit()


@app.callback()
def main(
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
    """Kestrel: local-first agent harness for small local models."""


@app.command()
def doctor() -> None:
    """Find running local backends and the models they serve."""
    try:
        config = load_config()
    except ConfigError as exc:
        console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from None
    reports = asyncio.run(discover(config.providers, env=os.environ))
    print_reports(reports, console)
    if not any(r.status is Status.up for r in reports):
        if shutil.which("ollama"):
            hint = "Ollama is installed but not running: open the Ollama app or run `ollama serve`."
        else:
            hint = "Install one, e.g. `brew install ollama`, then `ollama pull qwen3:8b`."
        console.print(f"No backend is running. {hint}")
        raise typer.Exit(1)


STATUS_STYLE = {Status.up: "green", Status.down: "dim", Status.error: "red"}


def print_reports(reports: list[BackendReport], out: Console) -> None:
    """Print the doctor table, then each backend's warnings and errors."""
    table = Table(title="Backends")
    table.add_column("Provider")
    table.add_column("URL", overflow="fold")
    table.add_column("Status")
    table.add_column("Server")
    table.add_column("Models")
    for r in reports:
        style = STATUS_STYLE[r.status]
        url = escape(r.base_url) + (" [bold yellow]REMOTE[/bold yellow]" if r.remote else "")
        table.add_row(
            r.name,
            url,
            f"[{style}]{r.status.value}[/{style}]",
            escape(r.kind),
            escape(", ".join(r.models)) if r.models else "-",
        )
    out.print(table)
    for r in reports:
        notes = ([r.error] if r.error else []) + (r.warnings if r.status is not Status.down else [])
        for note in notes:
            out.print(f"[yellow]{r.name}:[/yellow] {escape(note)}")


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument(help="What to ask the model.")],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name. Default: default_model in config."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider name. Default: default_provider."),
    ] = None,
    show_thinking: Annotated[
        bool, typer.Option("--show-thinking", help="Print the model's reasoning, dimmed.")
    ] = False,
) -> None:
    """Send one message to a model and stream the answer. No agent, no tools."""
    try:
        config = load_config()
        provider_name = provider or config.default_provider
        provider_config = config.provider(provider_name)
    except ConfigError as exc:
        err_console.print(f"[red]Config error:[/red] {escape(str(exc))}")
        raise typer.Exit(2) from None
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

    messages = [Message(role=Role.user, content=prompt)]
    try:
        response = asyncio.run(
            _ask(provider_name, provider_config, messages, model_name, show_thinking)
        )
    except ProviderError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        err_console.print("Run `kestrel doctor` to check backends and models.")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        err_console.print("[dim]Stopped.[/dim]")
        raise typer.Exit(130) from None
    if (usage := response.usage) is not None:
        tokens_in = "?" if usage.prompt_tokens is None else usage.prompt_tokens
        tokens_out = "?" if usage.completion_tokens is None else usage.completion_tokens
        err_console.print(f"[dim]{tokens_in} tokens in, {tokens_out} tokens out[/dim]")


@asynccontextmanager
async def open_provider(name: str, config: ProviderConfig) -> AsyncGenerator[Provider]:
    """Create the provider for `config` and close it afterwards. Tests replace this."""
    api_key = config.api_key(os.environ)
    async with OpenAICompatProvider(config.base_url, name=name, api_key=api_key) as provider:
        yield provider


async def _ask(
    name: str,
    config: ProviderConfig,
    messages: Sequence[Message],
    model: str,
    show_thinking: bool,
) -> ModelResponse:
    async with open_provider(name, config) as provider:
        return await stream_answer(provider, messages, model=model, show_thinking=show_thinking)


async def stream_answer(
    provider: Provider,
    messages: Sequence[Message],
    *,
    model: str,
    show_thinking: bool = False,
) -> ModelResponse:
    """Print the reply as it arrives and return the complete reply.

    A spinner runs until the first visible output (models can take a while to load or to
    think). Reasoning is printed dimmed to stderr only with `show_thinking`.
    """
    spinner = err_console.status(f"Waiting for {model}...")
    spinner.start()
    spinning = True
    printed_thinking = False
    printed_text = False
    final: ModelResponse | None = None
    try:
        async for chunk in provider.stream(messages, model=model):
            if chunk.reasoning:
                if show_thinking:
                    if spinning:
                        spinner.stop()
                        spinning = False
                    err_console.print(
                        chunk.reasoning,
                        end="",
                        style="dim",
                        markup=False,
                        highlight=False,
                        soft_wrap=True,
                    )
                    printed_thinking = True
                elif spinning:
                    spinner.update("Thinking...")
            if chunk.text:
                if spinning:
                    spinner.stop()
                    spinning = False
                if printed_thinking and not printed_text:
                    err_console.print("\n")  # gap between thinking and answer
                console.print(chunk.text, end="", markup=False, highlight=False, soft_wrap=True)
                printed_text = True
            if chunk.done is not None:
                final = chunk.done
    finally:
        spinner.stop()
    if printed_text:
        console.print()
    else:
        err_console.print("[dim](the model returned no text)[/dim]")
    if final is None:
        raise BadBackendResponseError("the stream ended without a final reply")
    return final
