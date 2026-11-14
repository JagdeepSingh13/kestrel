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
from kestrel.cli.ladder import ladder_home, parse_rung
from kestrel.cli.output import err_console
from kestrel.core.config import ConfigError, KestrelConfig, ProviderConfig
from kestrel.core.context import NOTE_PREFIX, ContextManager
from kestrel.core.types import Message, Provider, ProviderError, Role, StopReason
from kestrel.safety.paths import Workspace
from kestrel.tools.registry import FileStamp

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
    """Print the model's thinking, dimmed (`/think show|hide`)."""
    think: bool = False
    """Let the model think before answering (`/think on|off`). Off by default: on a
    laptop, thinking can take a minute per step (owner decision, M4)."""
    conversation: ContextManager = field(default_factory=lambda: ContextManager(None))
    """Every turn so far, tool calls and results included. Trimmed when sent, never here."""
    ladder: list[str] = field(default_factory=list[str])
    """Models above `model` to step up to (`/ladder`); empty: the agent file's."""
    up: int = 0
    """Rungs up to start the next message on (`/up`, `/retry up`)."""
    cloud: bool = False
    """`/cloud on`: cloud models may be used this session (each cloud step still asks)."""
    seen_files: dict[Path, FileStamp] = field(default_factory=dict[Path, FileStamp])
    """Files the model has read in this conversation (an edit needs a fresh read)."""

    @property
    def history(self) -> list[Message]:
        """The conversation's messages."""
        return self.conversation.history


RETRY_MARK = "\n\n(Kestrel: you are answering this again"
"""Start of the note `/retry` adds to the message it sends again."""


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
        wanted = session.agent.context.window
        # After a turn the conversation holds the window really used (the backend's
        # loaded one can be smaller than the agent asks for).
        window = min(wanted, session.conversation.window) if session.history else wanted
        used = session.conversation.build().est_tokens if session.history else 0
        small = " (backend loaded it small: /profile)" if window < wanted else ""
        return (
            f"{self.ladder_status()} · agent {session.agent.name} · "
            f"context {min(100, round(100 * used / window))}% of {window // 1024}K{small} · /help"
        )

    def ladder_status(self) -> str:
        """`qwen3:8b ↑ ernie-5.5-flash-preview:free (cloud on)`: where a stuck step goes."""
        model = self.session.model or "no model"
        uppers = self.ladder_names()
        if not uppers:
            return model
        top = uppers[-1]
        head, _, rest = top.partition(":")
        provider = self.config.providers.get(head) if rest else None
        shown = (rest if provider is not None else top).rsplit("/", 1)[-1]
        if provider is None or not provider.is_remote:
            return f"{model} ↑ {shown}"
        cloud_ok = self.session.cloud or self.session.agent.allow_cloud
        return f"{model} ↑ {shown} ({'cloud on' if cloud_ok else 'cloud off'})"

    def take_back_last_turn(self) -> tuple[str, str] | None:
        """Remove the last message you sent and everything after it from the conversation;
        return (that message, the answer it got). None if there is nothing to retry."""
        history = self.session.conversation.history
        task_at = max(
            (
                i
                for i, m in enumerate(history)
                if m.role is Role.user and not (m.content or "").startswith(NOTE_PREFIX)
            ),
            default=None,
        )
        if task_at is None:
            return None
        task = (history[task_at].content or "").split(RETRY_MARK, 1)[0].rstrip()
        answers = [m for m in history[task_at:] if m.role is Role.assistant and m.content]
        final = [m for m in answers if not m.tool_calls]  # the real answer, not "let me look"
        chosen = (final or answers)[-1].content if (final or answers) else ""
        answer = chosen or ""
        del history[task_at:]
        return task, answer or ""

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
            home = ladder_home(self.config, agent.provider)
            uppers = [parse_rung(n, self.config, home) for n in self.ladder_names()]
            if uppers:  # with a ladder, every model name says where it runs
                first = parse_rung(model, self.config, home)
                name, model = first.provider, first.model
            provider_config = self.config.provider(name)
        except (ConfigError, ValueError) as exc:
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
            think=None if self.session.think else False,
            conversation=self.session.conversation,
            seen_files=self.session.seen_files,
            uppers=uppers,
            start=self.session.up,
            cloud=self.session.cloud,
            box=self.box,
        )
        self.session.up = 0  # /up is for one message
        if result is None or result.stopped_reason is StopReason.error:
            err_console.print("[dim]That turn was not kept.[/dim]")

    def ladder_names(self) -> list[str]:
        """The rungs above the current model: `/ladder`'s, else the agent file's."""
        if self.session.ladder:
            return list(self.session.ladder)
        models = self.session.agent.model
        return [m for m in models[1:] if m != self.session.model] if len(models) > 1 else []

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
