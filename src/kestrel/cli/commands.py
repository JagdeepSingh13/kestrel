"""Slash commands for the interactive shell. Add a command: write a handler, add a row."""

from collections.abc import Callable
from dataclasses import dataclass

from rich.markup import escape

from kestrel.agents.loader import AgentError, list_agents, load_agent
from kestrel.cli.agent_run import start_run
from kestrel.cli.output import console, err_console, print_reports
from kestrel.cli.shell import Shell
from kestrel.cli.trace import print_run_list, print_trace
from kestrel.core.config import ConfigError
from kestrel.core.events import find_run, runs_dir
from kestrel.core.types import ProviderError
from kestrel.providers.discovery import discover
from kestrel.tools.builtin import BUILTIN_TOOLS

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
            usage = escape(f"{command.usage:<22}")  # pad before escaping, or [..] skews it
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


def _run(shell: Shell, arg: str) -> bool:
    if not arg:
        err_console.print("Usage: /run TASK, e.g. /run list the files and summarize them")
        return True
    agent = shell.session.agent
    model = agent.model[0] if agent.model else shell.session.model
    if model is None:
        err_console.print("[red]No model chosen.[/red] /model NAME picks one.")
        return True
    name = agent.provider or shell.session.provider
    try:
        provider_config = shell.config.provider(name)
    except ConfigError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        return True
    start_run(
        arg,
        agent=agent,
        workspace=shell.workspace,
        provider_name=name,
        provider_config=provider_config,
        open_provider=shell.open_provider,
        model=model,
        env=shell.env,
        show_thinking=shell.session.show_thinking,
    )
    return True


def _agents(shell: Shell, arg: str) -> bool:
    for name, where in list_agents(shell.env).items():
        marker = "*" if name == shell.session.agent.name else " "
        err_console.print(f" {marker} {escape(name):<12} [dim]{escape(where)}[/dim]")
    return True


def _agent(shell: Shell, arg: str) -> bool:
    words = arg.split()
    trusted = "--trust-agent" in words
    names = [w for w in words if w != "--trust-agent"]
    if not names:
        agent = shell.session.agent
        err_console.print(f"Agent: {escape(agent.name)}. {escape(agent.description)}")
        err_console.print(
            "[dim]/agent NAME switches; /agents lists them; /tools shows tools.[/dim]"
        )
        return True
    try:
        agent = load_agent(names[0], shell.env, trusted=trusted)
    except AgentError as exc:
        err_console.print(f"[red]{escape(str(exc))}[/red]")
        return True
    shell.session.agent = agent
    model = agent.model[0] if agent.model else "the shell's model"
    err_console.print(f"Agent: {escape(agent.name)} ({escape(model)}). {escape(agent.description)}")
    err_console.print(f"[dim]tools: {escape(', '.join(agent.tools) or 'none')}[/dim]")
    return True


def _tools(shell: Shell, arg: str) -> bool:
    agent = shell.session.agent
    if not agent.tools:
        err_console.print(f"Agent {escape(agent.name)} has no tools.")
    for name in agent.tools:
        spec = BUILTIN_TOOLS[name].spec
        level = agent.permissions.get(name, spec.permission)
        if spec.confirm_if_true:
            level += f" ({spec.confirm_if_true}: confirm)"
        err_console.print(
            f"  [bold]{name:<12}[/bold] [dim]{escape(level):<22}[/dim] {escape(spec.description)}"
        )
    return True


def _workspace(shell: Shell, arg: str) -> bool:
    err_console.print(f"Agents can only use this folder: {shell.workspace.root}")
    if arg:
        err_console.print("[dim]Changing it is turned off for now, for safety.[/dim]")
    return True


def _trace(shell: Shell, arg: str) -> bool:
    words = arg.split()
    full = "--full" in words
    words = [w for w in words if w != "--full"]
    directory = runs_dir(shell.env)
    if words == ["list"]:
        print_run_list(directory, console)
        return True
    wanted = words[0] if words else "last"
    path = find_run(directory, wanted)
    if path is None:
        err_console.print(f"No single run matches {escape(wanted)!r}.")
        print_run_list(directory, err_console)
        return True
    print_trace(path, console, full=full)
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
    "run": Command(_run, "/run TASK", "run the agent on a task, with its tools"),
    "agent": Command(_agent, "/agent [name]", "show or switch the agent /run uses"),
    "agents": Command(_agents, "/agents", "list available agents"),
    "tools": Command(_tools, "/tools", "the agent's tools and their permissions"),
    "workspace": Command(_workspace, "/workspace", "the only folder agents may use"),
    "trace": Command(_trace, "/trace [id] [--full]", "show a run step by step (/trace list)"),
    "clear": Command(_clear, "/clear", "forget the conversation"),
    "doctor": Command(_doctor, "/doctor", "check which backends are running"),
    "exit": Command(_exit, "/exit", "leave (also /quit, Ctrl-D)"),
    "quit": Command(_exit, "/quit", "leave", hidden=True),
}
