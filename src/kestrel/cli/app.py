"""Typer application: the `kestrel` command."""

import asyncio
import os
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kestrel.__about__ import __app_name__, __version__
from kestrel.core.config import ConfigError, load_config
from kestrel.providers.discovery import BackendReport, Status, discover

app = typer.Typer(
    name=__app_name__,
    help="Local-first agent harness for small local models.",
    no_args_is_help=True,
)
console = Console()


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
        console.print(
            "No backend is running. Install one, e.g. `brew install ollama`, then "
            "`ollama pull qwen3:8b`."
        )
        raise typer.Exit(1)


STATUS_STYLE = {Status.up: "green", Status.down: "dim", Status.error: "red"}


def print_reports(reports: list[BackendReport], out: Console) -> None:
    """Print the doctor table, then each backend's warnings and errors."""
    table = Table(title="Backends")
    table.add_column("Provider")
    table.add_column("URL")
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
