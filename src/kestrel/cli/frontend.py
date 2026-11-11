"""CliFrontend: shows a run live in the terminal (implements `core.types.Frontend`).

The answer text streams to stdout; tool calls, results and progress go to stderr.
"""

import json

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt
from rich.status import Status

from kestrel.cli.output import console, err_console
from kestrel.core.types import Event, EventType, JSONValue, ToolCall, ToolSpec

PREVIEW_CHARS = 300


class CliFrontend:
    """Prints events as they happen."""

    def __init__(self, *, show_thinking: bool = False) -> None:
        """`show_thinking` prints the model's reasoning, dimmed."""
        self.show_thinking = show_thinking
        self._spinner: Status | None = None
        self._printed_text = False
        self._printed_thinking = False

    def on_event(self, event: Event) -> None:
        """Render one event."""
        data, step = event.data, event.step
        match event.type:
            case EventType.model_request:
                self._printed_text = self._printed_thinking = False
                self._spin(f"Step {step}: waiting for the model...")
            case EventType.model_delta:
                self._delta(step, _str(data.get("text")), _str(data.get("reasoning")))
            case EventType.model_response:
                self.stop()
                if self._printed_text or self._printed_thinking:
                    (console if self._printed_text else err_console).print()
            case EventType.tool_call:
                args = json.dumps(data.get("arguments"), ensure_ascii=False)
                err_console.print(
                    f"[cyan]tool[/cyan] {escape(_str(data.get('name')))} {escape(args)}"
                )
            case EventType.tool_result:
                style = "red" if data.get("is_error") else "dim"
                err_console.print(f"  [{style}]{escape(_preview(data.get('content')))}[/{style}]")
            case EventType.context_trim:
                err_console.print(
                    f"[dim]Context trimmed to fit: {data.get('dropped_tool_results')} old tool "
                    f"results emptied, {data.get('dropped_messages')} old messages dropped.[/dim]"
                )
            case EventType.error:
                self.stop()
                err_console.print(f"[red]Error:[/red] {escape(_str(data.get('message')))}")
            case EventType.run_end:
                self.stop()
                err_console.print(
                    f"[dim]{data.get('stopped_reason')} after {data.get('steps')} steps, "
                    f"{data.get('tool_calls')} tool calls, {data.get('prompt_tokens')} tokens in, "
                    f"{data.get('completion_tokens')} out, {data.get('wall_seconds')}s[/dim]"
                )
            case _:
                pass

    async def confirm(self, call: ToolCall, spec: ToolSpec) -> bool:
        """Ask before a risky tool runs (permissions arrive in M3)."""
        self.stop()
        args = json.dumps(call.arguments, ensure_ascii=False)
        return Confirm.ask(f"Run {call.name} {args}?", console=err_console, default=False)

    async def ask(self, question: str) -> str:
        """Ask the user something on the agent's behalf (M3)."""
        self.stop()
        return Prompt.ask(question, console=err_console)

    def stop(self) -> None:
        """Stop the spinner, if one is running."""
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def _spin(self, message: str) -> None:
        self.stop()
        self._spinner = err_console.status(message)
        self._spinner.start()

    def _delta(self, step: int, text: str, reasoning: str) -> None:
        if reasoning:
            if self.show_thinking:
                self.stop()
                _raw(err_console, reasoning, style="dim")
                self._printed_thinking = True
            elif self._spinner is not None:
                self._spinner.update(f"Step {step}: thinking...")
        if text:
            self.stop()
            if self._printed_thinking and not self._printed_text:
                err_console.print()
            _raw(console, text)
            self._printed_text = True


def _raw(out: Console, text: str, style: str | None = None) -> None:
    out.print(text, end="", style=style, markup=False, highlight=False, soft_wrap=True)


def _str(value: JSONValue) -> str:
    return value if isinstance(value, str) else ""


def _preview(value: JSONValue) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "..."
