"""Slash commands for the interactive shell. Add a command: write a handler, add a row."""

from collections.abc import Callable
from dataclasses import dataclass

from rich.markup import escape

from kestrel.cli.output import err_console, print_reports
from kestrel.cli.shell import Shell
from kestrel.core.config import ConfigError
from kestrel.core.types import ProviderError
from kestrel.providers.discovery import discover

type Handler = Callable[[Shell, str], bool]
"""Gets the shell and the text after the command name. Returns False to leave the shell."""


@dataclass(frozen=True)
class Command:
    """One slash command."""

    handler: Handler
    usage: str
    help: str
    hidden: bool = False
    """Left out of /help (aliases)."""


def run_command(shell: Shell, line: str) -> bool:
    """Run a `/command args` line. Returns False when the shell should exit."""
    name, _, arg = line[1:].partition(" ")
    command = COMMANDS.get(name.lower())
    if command is None:
        err_console.print(f"Unknown command /{escape(name)}. Type /help for the list.")
        return True
    return command.handler(shell, arg.strip())


def _help(shell: Shell, arg: str) -> bool:
    for command in COMMANDS.values():
        if not command.hidden:
            usage = escape(f"{command.usage:<18}")  # pad before escaping, or [..] skews it
            err_console.print(f"  [bold]{usage}[/bold] {command.help}")
    err_console.print("  Anything else is sent to the model. Ctrl-C stops a reply, Ctrl-D quits.")
    return True


def _exit(shell: Shell, arg: str) -> bool:
    return False


def _clear(shell: Shell, arg: str) -> bool:
    shell.session.history.clear()
    err_console.print("Conversation cleared.")
    return True


def _model(shell: Shell, arg: str) -> bool:
    if arg:
        shell.session.model = arg
        err_console.print(f"Model: {escape(arg)}")
        if shell.model_names and arg not in shell.model_names:
            err_console.print(
                f"[yellow]{escape(arg)} is not in the list from the backend.[/yellow]"
            )
        return True

    async def list_models() -> list[str]:
        async with shell.provider() as provider:
            return await provider.list_models()

    try:
        shell.model_names = shell.run_async(list_models())
    except ProviderError as exc:
        err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
        return True
    for name in shell.model_names:
        marker = "*" if name == shell.session.model else " "
        err_console.print(f" {marker} {escape(name)}")
    if not shell.model_names:
        err_console.print("The backend lists no models.")
    return True


def _provider(shell: Shell, arg: str) -> bool:
    if not arg:
        for name, config in shell.config.providers.items():
            marker = "*" if name == shell.session.provider else " "
            err_console.print(f" {marker} {escape(name):<10} {escape(config.base_url)}")
        return True
    try:
        config = shell.config.provider(arg)
    except ConfigError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        return True
    shell.session.provider = arg
    shell.model_names = []
    err_console.print(f"Provider: {escape(arg)} ({escape(config.base_url)})")
    if config.is_remote:
        err_console.print("[bold yellow]REMOTE[/bold yellow]: your messages leave this machine.")
    err_console.print("[dim]Model names differ between backends; /model lists this one's.[/dim]")
    return True


def _think(shell: Shell, arg: str) -> bool:
    if arg.lower() in ("on", "off"):
        shell.session.show_thinking = arg.lower() == "on"
    elif arg:
        err_console.print("Usage: /think on, /think off, or /think to toggle.")
        return True
    else:
        shell.session.show_thinking = not shell.session.show_thinking
    err_console.print(f"Thinking: {'shown' if shell.session.show_thinking else 'hidden'}")
    return True


def _doctor(shell: Shell, arg: str) -> bool:
    reports = shell.run_async(discover(shell.config.providers, env=shell.env))
    print_reports(reports, err_console)
    return True


COMMANDS: dict[str, Command] = {
    "help": Command(_help, "/help", "show this list"),
    "model": Command(_model, "/model [name]", "list models, or switch (keeps the conversation)"),
    "provider": Command(_provider, "/provider [name]", "list backends, or switch"),
    "think": Command(_think, "/think [on|off]", "show or hide the model's thinking"),
    "clear": Command(_clear, "/clear", "forget the conversation"),
    "doctor": Command(_doctor, "/doctor", "check which backends are running"),
    "exit": Command(_exit, "/exit", "leave (also /quit, Ctrl-D)"),
    "quit": Command(_exit, "/quit", "leave", hidden=True),
}
