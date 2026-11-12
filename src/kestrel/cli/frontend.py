"""CliFrontend: shows a run live in the terminal (implements `core.types.Frontend`).

Claude Code style: the answer after a `⏺`, each tool call as one line `⏺ tool(arg)` with a
short result under `⎿`, a spinner with elapsed time while waiting. The answer text goes to
stdout; everything else to stderr, so `kestrel run ... > file` still captures only the answer.
"""

import json
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from kestrel.cli.output import console, err_console
from kestrel.core.types import ConfirmAnswer, Event, EventType, JSONValue, ToolCall, ToolSpec

PREVIEW_LINES = 3
CONFIRM_PREVIEW_LINES = 15
ARG_CHARS = 60

type PauseKeys = Callable[[], AbstractContextManager[None]]
"""Stops the Esc watcher while the user answers a question (see `cli.keys`)."""


class BottomBox(Protocol):
    """The shell's input box that stays at the bottom while a turn runs (`cli.tui`).
    When there is one, the spinner, menus and questions go through it."""

    def set_activity(self, label: str | None) -> None:
        """Show `label` with a spinner above the box; None hides it."""
        ...

    def set_cancel(self, cancel: Callable[[], None] | None) -> None:
        """What Esc should call to stop the running turn."""
        ...

    async def choose(self, question: str, options: Sequence[tuple[str, str]]) -> str:
        """Arrow-key menu above the box; returns the picked option's value."""
        ...

    async def ask(self, question: str) -> str:
        """Use the box to get a typed answer."""
        ...


class CliFrontend:
    """Prints events as they happen."""

    def __init__(
        self,
        *,
        show_thinking: bool = False,
        interactive: bool | None = None,
        box: BottomBox | None = None,
    ) -> None:
        """`show_thinking` prints the model's reasoning, dimmed. `interactive` (default: a
        terminal is attached) uses arrow-key menus; otherwise plain typed answers. `box`:
        the shell's bottom input box, if it is running."""
        self.show_thinking = show_thinking
        self.box = box
        self.interactive = (
            sys.stdin.isatty() and err_console.is_terminal if interactive is None else interactive
        )
        self.pause_keys: PauseKeys = nullcontext
        self.stop_hint = False
        """Show "esc to stop" in the spinner (set when the Esc watcher runs)."""
        self._spinner: Status | None = None
        self.label = ""
        """What the spinner says (Working, Thinking, Running TOOL)."""
        self.since = 0.0
        self._printed_text = False
        self._printed_thinking = False
        self._answer = ""
        self._thinking = ""
        self._live: Live | None = None

    def on_event(self, event: Event) -> None:
        """Render one event."""
        data = event.data
        match event.type:
            case EventType.model_request:
                self._printed_text = self._printed_thinking = False
                self._answer = self._thinking = ""
                self._spin("Working")
            case EventType.model_delta:
                self._delta(_str(data.get("text")), _str(data.get("reasoning")))
            case EventType.model_response:
                self.stop()  # also ends the live answer
                if self._printed_text and not console.is_terminal:
                    console.print()
                elif self._printed_thinking:
                    err_console.print()
            case EventType.tool_call:
                self.stop()
                name, args = _str(data.get("name")), data.get("arguments")
                err_console.print(
                    Text.assemble("⏺ ", (name, "bold"), f"({describe_args(args)})"),
                    highlight=False,
                )
                self._spin(f"Running {name}")
            case EventType.tool_result:
                self.stop()
                summary = summarize_result(data.get("content"))
                style = "red" if data.get("is_error") else "dim"
                err_console.print(_under(summary, style), highlight=False)
            case EventType.context_trim:
                err_console.print(
                    f"  [dim]context trimmed to fit: {data.get('dropped_tool_results')} old tool "
                    f"results emptied, {data.get('dropped_messages')} old messages dropped[/dim]"
                )
            case EventType.error:
                self.stop()
                err_console.print(f"[red]⏺ Error:[/red] {escape(_str(data.get('message')))}")
            case EventType.run_end:
                self.stop()
                err_console.print(_run_summary(event), highlight=False)
                err_console.print()
            case _:
                pass

    async def confirm(self, call: ToolCall, spec: ToolSpec) -> ConfirmAnswer:
        """Ask before a risky tool runs: yes, yes for the rest of this run, or no."""
        self.stop()
        err_console.print(
            Panel(
                confirm_preview(call),
                title=f"[bold]{escape(call.name)}[/bold] wants to run",
                title_align="left",
                border_style="yellow",
                expand=False,
            )
        )
        options: list[tuple[ConfirmAnswer, str]] = [
            ("yes", "Yes"),
            ("always", f"Yes, and don't ask again for {call.name} this turn"),
            ("no", "No"),
        ]
        if self.box is not None:  # the shell's bottom box draws the menu
            answers: dict[str, ConfirmAnswer] = {value: value for value, _ in options}
            picked = answers[await self.box.choose("Allow it?", options)]
        elif not self.interactive:
            answer = Prompt.ask(
                "Allow? [y]es / [n]o / [a]lways this run",
                choices=["y", "n", "a"],
                default="n",
                console=err_console,
                show_choices=False,
            )
            typed: dict[str, ConfirmAnswer] = {"y": "yes", "a": "always"}
            return typed.get(answer, "no")
        else:
            from kestrel.cli.prompt import choose  # prompt_toolkit only when interactive

            with self.pause_keys():
                picked = await choose("Allow it?", options)
        err_console.print(
            _under(
                {"yes": "allowed", "always": "allowed for this turn", "no": "refused"}[picked],
                "dim",
            )
        )
        return picked

    async def ask(self, question: str) -> str:
        """Ask the user something on the agent's behalf (`ask_user`)."""
        self.stop()
        err_console.print(Text.assemble(("⏺ ", "yellow"), question), highlight=False)
        if self.box is not None:
            answer = await self.box.ask(question)
            err_console.print(_under(answer, ""), highlight=False)
            return answer
        if not self.interactive:
            return Prompt.ask("  ", console=err_console)
        from prompt_toolkit import PromptSession

        with self.pause_keys():
            session: PromptSession[str] = PromptSession()
            return await session.prompt_async("  ⎿ ")

    def stop(self) -> None:
        """Stop the spinner and the live answer, if running."""
        self._stop_spinner()
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self.box is not None:  # with a bottom box the answer is printed once, when done
            if self._thinking:
                err_console.print(Text("∴ " + self._thinking.strip(), style="dim italic"))
                self._thinking = ""
            if self._answer:
                console.print(_bulleted(Markdown(self._answer)))
                self._answer = ""

    def _stop_spinner(self) -> None:
        if self.box is not None:
            self.box.set_activity(None)
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def _spin(self, label: str) -> None:
        self.stop()
        self.label = label
        self.since = time.monotonic()
        if self.box is not None:
            self.box.set_activity(label)
            return
        self._spinner = err_console.status(_Elapsed(self), spinner="dots")
        self._spinner.start()

    def _delta(self, text: str, reasoning: str) -> None:
        if self.box is not None:  # collect; `stop` prints it above the box when done
            self._thinking += reasoning if self.show_thinking else ""
            self._answer += text
            self.box.set_activity("Writing" if self._answer else "Thinking")
            self._printed_text = self._printed_text or bool(text)
            return
        if reasoning:
            if self.show_thinking:
                self._stop_spinner()
                if not self._printed_thinking:
                    err_console.print("∴ ", end="", style="dim")
                _raw(err_console, reasoning, style="dim italic")
                self._printed_thinking = True
            else:
                self.label = "Thinking"
        if text:
            self._stop_spinner()
            if not self._printed_text and self._printed_thinking:
                err_console.print("\n")
            self._printed_text = True
            if not console.is_terminal:  # redirected: just the plain answer
                _raw(console, text)
                return
            self._answer += text
            if self._live is None:
                self._live = Live(
                    console=console, refresh_per_second=8, vertical_overflow="visible"
                )
                self._live.start()
            self._live.update(_bulleted(Markdown(self._answer)))


def _bulleted(body: RenderableType) -> Table:
    """`⏺ ` in front of the answer, with its lines indented under the text."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column(width=1)
    grid.add_column()
    grid.add_row("⏺", body)
    return grid


class _Elapsed:
    """Spinner text that recomputes the elapsed seconds on every refresh."""

    def __init__(self, frontend: CliFrontend) -> None:
        self.frontend = frontend

    def __rich__(self) -> Text:
        seconds = int(time.monotonic() - self.frontend.since)
        hint = " · esc to stop" if self.frontend.stop_hint else ""
        return Text(f"{self.frontend.label}… {seconds}s{hint}", style="dim")


def describe_args(arguments: JSONValue) -> str:
    """The one thing worth showing about a call: `data/sales.csv`, the command, the code."""
    if not isinstance(arguments, dict) or not arguments:
        return ""
    for key in ("path", "command", "code", "question"):
        value = arguments.get(key)
        if isinstance(value, str):
            first = value.strip().splitlines()[0] if value.strip() else ""
            more = "…" if len(first) > ARG_CHARS or first != value.strip() else ""
            return first[:ARG_CHARS] + more
    text = json.dumps(arguments, ensure_ascii=False)
    return text if len(text) <= ARG_CHARS else text[:ARG_CHARS] + "…"


def summarize_result(content: JSONValue) -> str:
    """First few lines of a tool result, and how many more there are."""
    text = content if isinstance(content, str) else json.dumps(content)
    lines = text.strip().splitlines() or ["(no output)"]
    shown = "\n".join(line[:120] for line in lines[:PREVIEW_LINES])
    if len(lines) > PREVIEW_LINES:
        shown += f"\n… +{len(lines) - PREVIEW_LINES} lines"
    return shown


def confirm_preview(call: ToolCall) -> RenderableType:
    """Exactly what will run: the command, the code, or the file about to be written."""
    args = call.arguments
    code, command, path = args.get("code"), args.get("command"), args.get("path")
    if isinstance(code, str):
        return Syntax(_clip(code), "python", theme="ansi_dark", background_color="default")
    if isinstance(command, str):
        return Syntax(_clip(command), "bash", theme="ansi_dark", background_color="default")
    if isinstance(path, str) and isinstance(args.get("content"), str):
        content = str(args["content"])
        verb = "replace" if args.get("overwrite") else "create"
        head = Text(f"{verb} {path} ({len(content)} characters)", style="bold")
        return Group(head, Text(_clip(content), style="dim"))
    return Text(json.dumps(args, ensure_ascii=False, indent=2))


def _clip(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= CONFIRM_PREVIEW_LINES:
        return text
    rest = len(lines) - CONFIRM_PREVIEW_LINES
    return "\n".join([*lines[:CONFIRM_PREVIEW_LINES], f"… +{rest} more lines"])


def _under(text: str, style: str) -> Text:
    """`  ⎿  text`, with following lines indented to match."""
    first, *rest = text.splitlines() or [""]
    body = Text.assemble(("  ⎿  ", "dim"), (first, style))
    for line in rest:
        body.append("\n     " + line, style=style)
    return body


def _run_summary(event: Event) -> Text:
    data = event.data
    reason = _str(data.get("stopped_reason"))
    tokens = _int(data.get("prompt_tokens")) + _int(data.get("completion_tokens"))
    parts = [
        f"{data.get('steps')} steps",
        f"{data.get('tool_calls')} tools",
        f"{tokens / 1000:.1f}K tokens" if tokens >= 1000 else f"{tokens} tokens",
        f"{round(_float(data.get('wall_seconds')))}s",
        f"run {event.run_id}",
    ]
    if reason == "final":
        return Text("  " + " · ".join(parts), style="dim")
    labels = {
        "max_steps": "Stopped: hit the step limit",
        "budget": "Stopped: hit a budget limit",
        "cancelled": "Stopped",
        "error": "Stopped by an error",
    }
    return Text.assemble(
        ("  " + labels.get(reason, reason), "yellow"), (" · " + " · ".join(parts), "dim")
    )


def _raw(out: Console, text: str, style: str | None = None) -> None:
    out.print(text, end="", style=style, markup=False, highlight=False, soft_wrap=True)


def _str(value: JSONValue) -> str:
    return value if isinstance(value, str) else ""


def _int(value: JSONValue) -> int:
    return value if isinstance(value, int) else 0


def _float(value: JSONValue) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
