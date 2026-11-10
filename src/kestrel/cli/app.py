"""Typer application: the `kestrel` command."""

from typing import Annotated

import typer
from rich.console import Console

from kestrel.__about__ import __app_name__, __version__

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
