"""Slash commands for the interactive shell. Add a command: write a handler, add a row."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from rich.markup import escape
from rich.table import Table

from kestrel.agents.loader import AgentError, list_agents, load_agent
from kestrel.cli import agent_run
from kestrel.cli.ladder import ladder_home, looks_remote, parse_rung
from kestrel.cli.output import console, describe_profile, err_console, print_reports
from kestrel.cli.report import ReportError, build_case, write_case
from kestrel.cli.shell import RETRY_MARK, Shell
from kestrel.cli.trace import print_prompt, print_run_list, print_trace
from kestrel.core.config import ConfigError, ProviderConfig, default_data_dir
from kestrel.core.events import find_run, runs_dir
from kestrel.core.types import ProviderError
from kestrel.providers.discovery import discover
from kestrel.providers.openrouter import OPENROUTER_URL, is_openrouter, openrouter_models
from kestrel.providers.profiles import THINK_STYLES, ProfileCache, backend_kind, cache_path
from kestrel.safety.redact import Redactor
from kestrel.tools.builtin import BUILTIN_TOOLS
from kestrel.tools.mcp import McpStartError, command_line, tool_name

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
    err_console.print(
        "  Anything else goes to the agent, which uses its tools when needed. "
        "Esc stops a turn, Ctrl-D quits."
    )
    return True


def _exit(shell: Shell, arg: str) -> bool:
    return False


def _clear(shell: Shell, arg: str) -> bool:
    shell.session.conversation.clear()
    shell.session.seen_files.clear()  # forgotten with the conversation that held them
    shell.session.session_id = None  # the next message starts a new saved session
    err_console.print("Conversation cleared. The old one stays saved (/sessions).")
    return True


def _mcp(shell: Shell, arg: str) -> bool:
    """MCP servers: list, their tools, restart one."""
    word, _, rest = arg.strip().partition(" ")
    servers = shell.config.mcp_servers
    if not servers:
        err_console.print(
            "No MCP servers configured. Add one to your config file, for example:\n"
            '[dim]  [mcp_servers.fs]\n  command = "npx"\n'
            '  args = ["-y", "@modelcontextprotocol/server-filesystem", "{workspace}"][/dim]'
        )
        return True
    hub = shell.mcp
    if word in ("tools", "restart"):
        name = rest.strip() or (next(iter(servers)) if len(servers) == 1 else "")
        if name not in servers:
            err_console.print(f"Which server? One of: {escape(', '.join(servers))}")
            return True
        if name not in hub.running() or word == "restart":
            err_console.print(
                f"[dim]Starting {escape(name)}: "
                f"{escape(command_line(name, servers[name], hub.workspace))}[/dim]"
            )
        try:
            listed = hub.restart(name) if word == "restart" else hub.tools(name)
        except McpStartError as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            return True
        allow = set(servers[name].allow)
        for t in listed:
            mark = "allowed" if t.name in allow else "asks first"
            first = " ".join((t.description or "").split())[:90]
            err_console.print(
                f"  {escape(tool_name(name, t.name)):<36} [dim]{mark} · {escape(first)}[/dim]"
            )
        err_console.print(
            f"[dim]An agent uses them with  mcp: {{{escape(name)}: [tool, ...]}}  "
            "in its file.[/dim]"
        )
        return True
    agent_uses = shell.session.agent.mcp
    for name, config in servers.items():
        state = "running" if name in hub.running() else "not started"
        used = " · used by this agent" if name in agent_uses else ""
        where = "remote" if config.is_remote else "local"
        err_console.print(
            f"  {escape(name):<10} {state}, {where}{used}  "
            f"[dim]{escape(command_line(name, config, hub.workspace))}[/dim]"
        )
    err_console.print("[dim]/mcp tools NAME lists a server's tools; /mcp restart NAME.[/dim]")
    return True


def _sessions(shell: Shell, arg: str) -> bool:
    try:
        recent = shell.sessions.recent(20)
    except sqlite3.Error as exc:
        err_console.print(f"[red]Could not read saved sessions:[/red] {escape(str(exc))}")
        return True
    if not recent:
        err_console.print("No saved sessions yet. Each conversation is saved after every turn.")
        return True
    table = Table(title="Saved sessions (newest first)")
    for column in ("Id", "Updated", "Msgs", "Agent", "First message"):
        table.add_column(column, overflow="fold")
    for info in recent:
        mark = " *" if info.id == shell.session.session_id else ""
        table.add_row(
            info.id + mark,
            info.updated[:16].replace("T", " "),
            str(info.messages),
            escape(info.agent),
            escape(info.title),
        )
    err_console.print(table)
    err_console.print("[dim]/resume ID (or /resume last) picks one up.[/dim]")
    return True


def _resume(shell: Shell, arg: str) -> bool:
    try:
        return _resume_from(shell, arg.strip())
    except (sqlite3.Error, ValueError) as exc:
        err_console.print(f"[red]Could not resume that session:[/red] {escape(str(exc))}")
        return True


def _resume_from(shell: Shell, wanted: str) -> bool:
    recent = shell.sessions.recent(9)
    if not recent:
        err_console.print("No saved sessions yet.")
        return True
    if not wanted and shell.box is not None:
        options = [(i.id, f"{i.updated[:16].replace('T', ' ')}  {i.title[:50]}") for i in recent]
        options.append(("", "Cancel"))  # Esc picks the last option
        wanted = shell.run_async(shell.box.choose("Resume which session?", options))
        if not wanted:
            return True
    if wanted in ("", "last"):
        wanted = recent[0].id
    matches = [i.id for i in shell.sessions.recent(1000) if i.id.startswith(wanted)]
    if len(matches) != 1:
        err_console.print(
            f"No single saved session matches {escape(wanted)!r}. /sessions lists them."
        )
        return True
    loaded = shell.sessions.load(matches[0])
    if loaded is None:
        return True
    info, messages = loaded
    shell.resume(info, messages)
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

    config = shell.config.provider(shell.session.provider)
    if is_openrouter(config.base_url):
        return _openrouter_models(shell, config.base_url)

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


def _openrouter_models(shell: Shell, base_url: str) -> bool:
    """OpenRouter has hundreds of models: show the free ones that take tools."""
    try:
        models = shell.run_async(openrouter_models(base_url))
    except (httpx.HTTPError, ValueError) as exc:
        err_console.print(f"[red]Error:[/red] could not list OpenRouter models: {escape(str(exc))}")
        return True
    shell.model_names = [m.id for m in models]
    free = [m for m in models if m.free]
    err_console.print(
        f"{len(models)} OpenRouter models take tools; the {len(free)} free ones "
        "(20 requests a minute, 50 a day):"
    )
    for m in free:
        marker = "*" if m.id == shell.session.model else " "
        window = f"{m.window // 1024}K" if m.window else "?"
        err_console.print(f" {marker} {escape(m.id):<50} {window}")
    err_console.print("[dim]Any other id from openrouter.ai/models works too (paid).[/dim]")
    return True


def _ladder(shell: Shell, arg: str) -> bool:
    session = shell.session
    if arg == "off":
        session.ladder = []
        err_console.print("Ladder off: only the current model.")
        return True
    if arg:
        first, *rest = arg.split()
        try:
            home = ladder_home(shell.config, session.agent.provider)
            for name in [first, *rest]:
                parse_rung(name, shell.config, home)
        except (ConfigError, ValueError) as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            return True
        for name in [first, *rest]:
            if (meant := looks_remote(name, shell.config)) is not None:
                err_console.print(
                    f"[red]{escape(name)} has no provider prefix, so it would run on "
                    f"{escape(home)}.[/red] For the cloud write {escape(meant)}"
                )
                return True
        session.model = first
        session.ladder = rest
    uppers = shell.ladder_names()
    home = ladder_home(shell.config, session.agent.provider)
    shown = (
        [_where(n, shell, home) for n in [session.model or "(no model)", *uppers]]
        if uppers
        else [session.model or "(no model)"]
    )
    err_console.print(f"Ladder: {escape(' then '.join(shown))}")
    if not uppers:
        err_console.print("[dim]Only one rung. Set more with /ladder MODEL MODEL ...[/dim]")
    else:
        err_console.print(
            "[dim]A stuck step goes one rung up, then back down after one good step "
            "(at most 2 step-ups per run). Cloud rungs need cloud on (/cloud on) and "
            "your yes for each step.[/dim]"
        )
    return True


def _where(name: str, shell: Shell, home: str) -> str:
    """`qwen3:8b (ollama)`: each rung with the backend it runs on."""
    try:
        spec = parse_rung(name, shell.config, home)
    except ValueError:
        return name
    return spec.label if spec.remote else f"{spec.model} ({spec.provider})"


def _retry(shell: Shell, arg: str) -> bool:
    """Send your last message again, on the same model or (`up`) the next one up."""
    up = arg.strip() == "up"
    if arg.strip() not in ("", "up"):
        err_console.print("Use /retry (same model) or /retry up (the next model up).")
        return True
    if up and not shell.ladder_names():
        err_console.print(
            "No model above the current one. Add one with /cloud setup (a free cloud model) "
            "or /ladder MODEL MODEL."
        )
        return True
    history = shell.session.conversation.history
    before, seen_before = list(history), dict(shell.session.seen_files)
    turn = shell.take_back_last_turn()
    if turn is None:
        err_console.print("Nothing to retry yet.")
        return True
    task, answer = turn
    cut = len(history)
    said = " ".join(answer.split())[:600] or "(no answer)"
    text = (
        f"{task}{RETRY_MARK} because the user judged the previous answer wrong or not good "
        f'enough. That answer was: "{said}". Do not repeat it; check facts with tools.)'
    )
    shell.session.up = 1 if up else 0
    where = shell.ladder_names()[0] if up else (shell.session.model or "the same model")
    err_console.print(f"[dim]Retrying on {escape(where)}: {escape(task[:80])}[/dim]")
    shell.send(text)
    shell.session.up = 0
    if len(history) <= cut:  # the retry did not run or was not kept: keep the old turn
        history[:] = before
        shell.session.seen_files.clear()
        shell.session.seen_files.update(seen_before)
        err_console.print("[dim]The retry did not finish; your earlier turn is kept.[/dim]")
        shell.save_session()
    return True


def _cloud(shell: Shell, arg: str) -> bool:
    """Cloud models for this session: setup, on, off, or status."""
    word, _, rest = arg.strip().partition(" ")
    session = shell.session
    if word == "on":
        session.cloud = True
        if not _cloud_rung(shell):
            err_console.print("Cloud on, but no cloud model is on the ladder: /cloud setup.")
        else:
            err_console.print(
                "Cloud on: a stuck step (or /retry up) may go to the cloud model, after your yes."
            )
        return True
    if word == "off":
        session.cloud = False
        err_console.print("Cloud off: the ladder will not send steps to cloud models.")
        return True
    if word == "setup":
        return _cloud_setup(shell, rest.strip())
    rung = _cloud_rung(shell)
    state = "on" if session.cloud or session.agent.allow_cloud else "off"
    err_console.print(f"Cloud {state}; cloud model: {escape(rung or 'none (/cloud setup)')}")
    return True


def _cloud_rung(shell: Shell) -> str | None:
    for name in shell.ladder_names():
        head = name.partition(":")[0]
        if head in shell.config.providers and shell.config.providers[head].is_remote:
            return name
    return None


def _cloud_setup(shell: Shell, wanted: str) -> bool:
    config = shell.config
    name = next(
        (n for n, c in config.providers.items() if c.is_remote and is_openrouter(c.base_url)),
        None,
    )
    provider = (
        config.providers[name]
        if name is not None
        else ProviderConfig(base_url=OPENROUTER_URL, api_key_env="OPENROUTER_API_KEY")
    )
    if provider.api_key(shell.env) is None:
        err_console.print(
            f"[yellow]No key found.[/yellow] Quit, run export {provider.api_key_env}=<your key> "
            "in this terminal, start kestrel again, then /cloud setup."
        )
        return True
    try:
        models = shell.run_async(openrouter_models(provider.base_url))
    except (httpx.HTTPError, ValueError) as exc:
        err_console.print(f"[red]Could not list OpenRouter models:[/red] {escape(str(exc))}")
        return True
    if wanted:
        chosen = next((m for m in models if m.id == wanted), None)
        if chosen is None:
            err_console.print(
                f"[red]{escape(wanted)} is not an OpenRouter model that takes tools.[/red]"
            )
            return True
    else:
        free = sorted((m for m in models if m.free), key=lambda m: -(m.window or 0))[:9]
        if not free:
            err_console.print("OpenRouter lists no free models with tools right now.")
            return True
        chosen = free[0]
        if shell.box is not None:
            options = [(m.id, f"{m.id}  ({(m.window or 0) // 1024}K)") for m in free]
            options.append(("", "Cancel"))  # Esc and Ctrl-C pick the last option
            picked = shell.run_async(shell.box.choose("Pick a free cloud model:", options))
            if not picked:
                err_console.print("Cancelled; nothing changed.")
                return True
            chosen = next(m for m in free if m.id == picked)
    if name is None:  # for this session only; the config file is yours to change
        name = "openrouter"
        while name in config.providers:
            name += "-cloud"
        config.providers[name] = provider
        err_console.print(
            "[dim]Using OpenRouter for this session. To keep it, add to your config file:\n"
            f'  [providers.{name}]\n  base_url = "{OPENROUTER_URL}"\n'
            '  api_key_env = "OPENROUTER_API_KEY"[/dim]'
        )
    shell.session.ladder = [f"{name}:{chosen.id}"]
    shell.session.cloud = True
    home = ladder_home(config, shell.session.agent.provider)
    local = shell.session.model or "(no model: /model NAME)"
    err_console.print(
        f"Cloud ready: {escape(local)} ({escape(home)}) then {escape(name)}:{escape(chosen.id)}"
        f" ({'free' if chosen.free else 'paid'}).\n"
        "[dim]A stuck step goes up after your yes; /retry up sends your last message there; "
        "/cloud off turns it off.[/dim]"
    )
    return True


def _up(shell: Shell, arg: str) -> bool:
    if not shell.ladder_names():
        err_console.print("No rung above the current model. Set one with /ladder.")
        return True
    shell.session.up = 1
    err_console.print("Your next message starts one rung up.")
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
    session = shell.session
    word = arg.lower()
    if word in ("on", "off"):
        session.think = word == "on"
    elif word in ("show", "hide"):
        session.show_thinking = word == "show"
    elif word:
        err_console.print("Usage: /think on|off (let the model think) or /think show|hide.")
        return True
    thinking = "on" if session.think else "off (faster)"
    shown = "shown" if session.show_thinking else "hidden"
    err_console.print(f"Thinking: {thinking}; display: {shown}")
    config = shell.config.provider(session.provider)
    if not session.think and THINK_STYLES[backend_kind(session.provider, config.base_url)] == (
        "none"
    ):
        err_console.print(
            "[dim]This backend has no known way to switch thinking off; "
            "the model may still think.[/dim]"
        )
    return True


def _profile(shell: Shell, arg: str) -> bool:
    model = shell.session.model
    if model is None:
        err_console.print("No model chosen. /model NAME picks one.")
        return True
    name = shell.session.provider
    config = shell.config.provider(name)
    if config.is_remote:
        err_console.print(
            f"{escape(name)} is a remote backend: Kestrel does not ask it about models."
        )
        return True
    cache = ProfileCache(cache_path(shell.env))
    known = cache.get(config.base_url, model)
    if arg == "probe" or known is None:
        known = shell.run_async(agent_run.probe_profile(name, config.base_url, model, known))
        cache.save(known)
    elif arg:
        err_console.print("Usage: /profile, or /profile probe to ask the backend again.")
        return True
    for line in describe_profile(known, shell.session.agent.context.window):
        err_console.print(line, highlight=False)
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
    if agent.model:
        shell.session.model = agent.model[0]
    model = shell.session.model or "no model"
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
    for server, wanted in agent.mcp.items():
        allow = shell.config.mcp_servers[server].allow if server in shell.config.mcp_servers else []
        for name in wanted:
            level = "allow (config)" if name in allow else "confirm"
            shown = tool_name(server, name) if name != "*" else f"{server}__* (all its tools)"
            err_console.print(f"  [bold]{escape(shown)}[/bold] [dim]{level}[/dim]")
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
    prompt_step: int | None = None
    if "--prompt" in words:
        at = words.index("--prompt")
        if at + 1 >= len(words) or not words[at + 1].isdigit():
            err_console.print("Usage: /trace [id] --prompt STEP")
            return True
        prompt_step = int(words[at + 1])
        del words[at : at + 2]
    directory = runs_dir(shell.env)
    if words == ["list"]:
        print_run_list(directory, console)
        return True
    if words[:1] == ["session"]:
        sid = words[1] if len(words) > 1 else shell.session.session_id
        runs = shell.sessions.runs(sid) if sid else []
        if not runs:
            err_console.print("No runs saved for that session yet.")
            return True
        print_run_list(directory, console, only=runs, title=f"Runs of session {sid}")
        return True
    wanted = words[0] if words else "last"
    path = find_run(directory, wanted)
    if path is None:
        err_console.print(f"No single run matches {escape(wanted)!r}.")
        print_run_list(directory, err_console)
        return True
    if prompt_step is not None:
        if not print_prompt(path, prompt_step, console):
            err_console.print(f"Step {prompt_step} has no logged prompt in run {path.stem}.")
        return True
    print_trace(path, console, full=full)
    return True


def _report_failure(shell: Shell, arg: str) -> bool:
    words = arg.split()
    step: int | None = None
    if "--step" in words:
        at = words.index("--step")
        if at + 1 >= len(words) or not words[at + 1].isdigit():
            err_console.print("Usage: /report-failure [run] [--step N]")
            return True
        step = int(words[at + 1])
        del words[at : at + 2]
    wanted = words[0] if words else "last"
    path = find_run(runs_dir(shell.env), wanted)
    if path is None:
        err_console.print(f"No single run matches {escape(wanted)!r}. /trace list shows runs.")
        return True
    try:
        case = build_case(path, step, Redactor(shell.env).json)
    except ReportError as exc:
        err_console.print(escape(str(exc)))
        return True
    saved = write_case(case, default_data_dir(shell.env) / "reports")
    err_console.print(f"Saved that step as a test case: {saved}")
    err_console.print(
        "[dim]Secrets removed; nothing was uploaded. Fill in its expect block, then copy it "
        "to tests/fixtures/bad_outputs/ to keep it as a test.[/dim]"
    )
    return True


def _doctor(shell: Shell, arg: str) -> bool:
    providers = shell.config.providers
    skipped = [] if shell.session.cloud else [n for n, c in providers.items() if c.is_remote]
    checked = {n: c for n, c in providers.items() if n not in skipped}
    reports = shell.run_async(discover(checked, env=shell.env))
    print_reports(reports, err_console)
    if skipped:
        err_console.print(
            f"[dim]Not contacted (cloud is off): {escape(', '.join(skipped))}. /cloud on "
            "to include them.[/dim]"
        )
    return True


COMMANDS: dict[str, Command] = {
    "help": Command(_help, "/help", "show this list"),
    "model": Command(_model, "/model [name]", "list models, or switch (keeps the conversation)"),
    "provider": Command(_provider, "/provider [name]", "list backends, or switch"),
    "ladder": Command(_ladder, "/ladder [models|off]", "show or set the models to step up to"),
    "up": Command(_up, "/up", "start the next message on the next model up"),
    "retry": Command(_retry, "/retry [up]", "send your last message again (up: on a bigger model)"),
    "cloud": Command(_cloud, "/cloud [setup|on|off]", "use a free cloud model when stuck"),
    "think": Command(_think, "/think [on|off|show|hide]", "let the model think; show it"),
    "profile": Command(_profile, "/profile [probe]", "what Kestrel knows about this model"),
    "agent": Command(_agent, "/agent [name]", "show or switch the agent (keeps the conversation)"),
    "agents": Command(_agents, "/agents", "list available agents"),
    "tools": Command(_tools, "/tools", "the agent's tools and their permissions"),
    "workspace": Command(_workspace, "/workspace", "the only folder agents may use"),
    "trace": Command(
        _trace, "/trace [id] [--full]", "show a run step by step (/trace list, --prompt N)"
    ),
    "report-failure": Command(
        _report_failure, "/report-failure [id]", "save a bad step as a test case (--step N)"
    ),
    "clear": Command(_clear, "/clear", "start a new conversation (the old one stays saved)"),
    "sessions": Command(_sessions, "/sessions", "list saved conversations"),
    "mcp": Command(_mcp, "/mcp [tools|restart] [name]", "MCP servers and their tools"),
    "resume": Command(_resume, "/resume [id|last]", "pick up a saved conversation"),
    "doctor": Command(_doctor, "/doctor", "check which backends are running"),
    "exit": Command(_exit, "/exit", "leave (also /quit, Ctrl-D)"),
    "quit": Command(_exit, "/quit", "leave", hidden=True),
}
