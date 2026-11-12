"""Interactive shell: `kestrel` (no arguments) or `kestrel chat`.

Every message is one agent turn: the current agent gets the conversation so far plus its
tools, and decides itself whether to use them. Lines starting with `/` are commands (see
`commands.py`). Each turn runs in its own `asyncio.run`, so stopping one keeps the shell open.
"""

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path

from rich.box import ROUNDED
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from kestrel.__about__ import __version__
from kestrel.agents.schema import AgentConfig
from kestrel.cli.agent_run import start_run
from kestrel.cli.frontend import BottomBox
from kestrel.cli.output import err_console
from kestrel.core.config import ConfigError, KestrelConfig, ProviderConfig
from kestrel.core.context import ContextManager
from kestrel.core.types import Message, Provider, ProviderError, StopReason
from kestrel.safety.paths import Workspace

type ReadLine = Callable[[str], str]
"""Show a prompt, return the typed line. Raises EOFError (Ctrl-D) or KeyboardInterrupt."""
type OpenProvider = Callable[[str, ProviderConfig], AbstractAsyncContextManager[Provider]]
"""Opens a provider by name and config; `cli.output.open_provider` in real use."""


@dataclass
class Session:
    """What the shell remembers between lines."""

    provider: str
    model: str | None
    agent: AgentConfig = field(default_factory=lambda: AgentConfig(name="default"))
    """The agent every message goes to (`/agent NAME` switches)."""
    show_thinking: bool = False
    conversation: ContextManager = field(default_factory=lambda: ContextManager(None))
    """Every turn so far, tool calls and results included. Trimmed when sent, never here."""

    @property
    def history(self) -> list[Message]:
        """The conversation's messages."""
        return self.conversation.history


class Shell:
    """The read, handle, repeat loop. Input and provider are injected so tests need neither
    a terminal nor a model."""

    def __init__(
        self,
        config: KestrelConfig,
        session: Session,
        *,
        open_provider: OpenProvider,
        env: Mapping[str, str],
        workspace: Workspace,
        read_line: ReadLine | None = None,
    ) -> None:
        """Create a shell; call `run()` to start it."""
        self.config = config
        self.session = session
        self.env = env
        self.model_names: list[str] = []
        """Models last listed by `/model`, used for Tab completion."""
        self.read_line = read_line
        """Set before `run()`; the terminal version needs the shell for Tab completion."""
        self.open_provider = open_provider
        self.workspace = workspace
        """The only folder agents may touch (owner decision: fixed for now)."""
        self.box: BottomBox | None = None
        """The input box fixed at the bottom of the terminal (`cli.tui`), when running."""

    def run(self) -> None:
        """Loop until `/exit` or Ctrl-D, reading lines with `read_line` (tests, no terminal).
        In a terminal the shell runs inside `cli.tui` instead, which calls `handle`."""
        read_line = self.read_line
        if read_line is None:
            raise RuntimeError("Shell.read_line must be set before run()")
        self.start()
        while True:
            try:
                line = read_line(self.prompt())
            except KeyboardInterrupt:
                continue  # Ctrl-C at the prompt only clears the line
            except EOFError:
                break
            if not self.handle(line):
                break
        self.finish()

    def start(self) -> None:
        """Pick a model if none is set, then show the welcome panel."""
        if self.session.model is None:
            self._pick_first_model()
        self._banner()

    def handle(self, line: str) -> bool:
        """Run one typed line: a /command or a message to the agent. False means leave."""
        from kestrel.cli.commands import run_command  # commands imports Shell

        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            return run_command(self, line)
        self.send(line)
        return True

    def finish(self) -> None:
        """Say goodbye."""
        err_console.print("[dim]Bye.[/dim]")

    def prompt(self) -> str:
        """The status line under the input box: model, agent, how full the context is."""
        session = self.session
        window = session.agent.context.window
        used = session.conversation.build().est_tokens if session.history else 0
        return (
            f"{session.model or 'no model'} · agent {session.agent.name} · "
            f"context {min(100, round(100 * used / window))}% of {window // 1024}K · /help"
        )

    def send(self, text: str) -> None:
        """Give one message to the agent, with the conversation so far and its tools."""
        model = self.session.model
        if model is None:
            err_console.print(
                "[red]No model chosen.[/red] Type /model to list models, /model NAME to pick one."
            )
            return
        agent = self.session.agent
        name = agent.provider or self.session.provider
        try:
            provider_config = self.config.provider(name)
        except ConfigError as exc:
            err_console.print(f"[red]{escape(str(exc))}[/red]")
            return
        result = start_run(
            text,
            agent=agent,
            workspace=self.workspace,
            provider_name=name,
            provider_config=provider_config,
            open_provider=self.open_provider,
            model=model,
            env=self.env,
            show_thinking=self.session.show_thinking,
            conversation=self.session.conversation,
            box=self.box,
        )
        if result is None or result.stopped_reason is StopReason.error:
            err_console.print("[dim]That turn was not kept.[/dim]")

    def run_async[T](self, work: Coroutine[object, object, T]) -> T:
        """Run one piece of async work to completion (commands use this too)."""
        return asyncio.run(work)

    def provider(self) -> AbstractAsyncContextManager[Provider]:
        """Open the session's current provider."""
        name = self.session.provider
        return self.open_provider(name, self.config.provider(name))

    def _pick_first_model(self) -> None:
        """No model configured: use the first one the backend lists, if it is running."""

        async def list_models() -> list[str]:
            async with self.provider() as provider:
                return await provider.list_models()

        try:
            self.model_names = self.run_async(list_models())
        except ProviderError:
            return  # the banner says "none"; /doctor explains
        if self.model_names:
            self.session.model = self.model_names[0]

    def _banner(self) -> None:
        provider = self.config.provider(self.session.provider)
        rows = Text.assemble(
            ("✻ ", "bold #d78700"),
            (f"Kestrel {__version__}", "bold"),
            "\n\n",
            ("  /help for commands, Esc stops a turn, Ctrl-D quits\n\n", "dim"),
            ("  model      ", "dim"),
            f"{self.session.model or 'none (/model NAME)'} ({self.session.provider})\n",
            ("  agent      ", "dim"),
            f"{self.session.agent.name}\n",
            ("  workspace  ", "dim"),
            _home(self.workspace.root),
        )
        err_console.print(Panel(rows, box=ROUNDED, border_style="#d78700", expand=False))
        if provider.is_remote:
            err_console.print(
                f"[bold yellow]REMOTE[/bold yellow] {escape(provider.base_url)}: "
                "your messages leave this machine."
            )


def _home(path: Path) -> str:
    """`~/Desktop/kestrel/workspace` instead of the full home path."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)
