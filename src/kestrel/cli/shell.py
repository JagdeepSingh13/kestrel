"""Interactive shell: `kestrel` (no arguments) or `kestrel chat`.

Plain text goes to the model together with the conversation so far; lines starting with
`/` are commands (see `commands.py`). Each model turn runs in its own `asyncio.run`, so
Ctrl-C stops only that turn and the shell stays open.
"""

import asyncio
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field

from rich.markup import escape

from kestrel.__about__ import __version__
from kestrel.cli.output import describe_usage, err_console, stream_answer
from kestrel.core.config import KestrelConfig, ProviderConfig
from kestrel.core.context import DEFAULT_WINDOW, ContextManager
from kestrel.core.types import Message, ModelResponse, Provider, ProviderError, Role

type ReadLine = Callable[[str], str]
"""Show a prompt, return the typed line. Raises EOFError (Ctrl-D) or KeyboardInterrupt."""
type OpenProvider = Callable[[str, ProviderConfig], AbstractAsyncContextManager[Provider]]
"""Opens a provider by name and config; `cli.output.open_provider` in real use."""


@dataclass
class Session:
    """What the shell remembers between lines."""

    provider: str
    model: str | None
    show_thinking: bool = False
    context_window: int = DEFAULT_WINDOW
    history: list[Message] = field(default_factory=list[Message])


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

    def run(self) -> None:
        """Loop until `/exit` or Ctrl-D."""
        from kestrel.cli.commands import run_command  # commands imports Shell

        read_line = self.read_line
        if read_line is None:
            raise RuntimeError("Shell.read_line must be set before run()")
        self._banner()
        while True:
            try:
                line = read_line(self.prompt()).strip()
            except KeyboardInterrupt:
                continue  # Ctrl-C at the prompt only clears the line
            except EOFError:
                break
            if not line:
                continue
            if line.startswith("/"):
                if not run_command(self, line):
                    break
            else:
                self.send(line)
        err_console.print("[dim]Bye.[/dim]")

    def prompt(self) -> str:
        """The prompt text, showing the current model."""
        return f"{self.session.model or 'no model'}> "

    def send(self, text: str) -> None:
        """Send one user message with the history; keep the turn only if it completes."""
        model = self.session.model
        if model is None:
            err_console.print(
                "[red]No model chosen.[/red] Type /model to list models, /model NAME to pick one."
            )
            return
        user = Message(role=Role.user, content=text)
        ctx = ContextManager(None, window=self.session.context_window, pin_first_user=False)
        ctx.history = [*self.session.history, user]
        prompt = ctx.build()
        if prompt.trimmed:
            err_console.print(
                f"[dim]Left out {prompt.dropped_messages} older messages to fit the model's "
                "memory. /clear starts fresh.[/dim]"
            )
        messages = prompt.messages
        try:
            response = self.run_async(self._turn(messages, model))
        except KeyboardInterrupt:
            err_console.print("[dim]Stopped. That turn was not kept.[/dim]")
            return
        except ProviderError as exc:
            err_console.print(f"[red]Error:[/red] {escape(str(exc))}")
            err_console.print("[dim]That turn was not kept. /doctor checks backends.[/dim]")
            return
        reply = response.message.model_copy(update={"reasoning": None})
        self.session.history += [user, reply]
        if response.usage is not None:
            err_console.print(f"[dim]{describe_usage(response.usage)}[/dim]")

    def run_async[T](self, work: Coroutine[object, object, T]) -> T:
        """Run one piece of async work to completion (commands use this too)."""
        return asyncio.run(work)

    def provider(self) -> AbstractAsyncContextManager[Provider]:
        """Open the session's current provider."""
        name = self.session.provider
        return self.open_provider(name, self.config.provider(name))

    async def _turn(self, messages: list[Message], model: str) -> ModelResponse:
        async with self.provider() as provider:
            return await stream_answer(
                provider, messages, model=model, show_thinking=self.session.show_thinking
            )

    def _banner(self) -> None:
        provider = self.config.provider(self.session.provider)
        parts = [f"Kestrel {__version__}", self.session.provider, self.session.model or "no model"]
        err_console.print(" · ".join(parts) + " · /help for commands", highlight=False)
        if provider.is_remote:
            err_console.print(
                f"[bold yellow]REMOTE[/bold yellow] {escape(provider.base_url)}: "
                "your messages leave this machine."
            )
