"""The shell in a terminal: one input box fixed at the bottom for the whole session.

Two threads:
- the main thread runs a prompt_toolkit app: the box, the status line under it, a spinner
  line above it while a turn runs, and the permission menu when a tool asks;
- a worker thread takes typed lines from a queue and runs them with `Shell.handle`
  (commands and agent turns, each in its own `asyncio.run`).

All printing goes through prompt_toolkit's `patch_stdout`, which draws it above the app,
so the box never moves away from the bottom. You can type the next message while a turn
runs; it waits in the queue. Esc stops the running turn.
"""

import asyncio
import concurrent.futures
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from rich.markup import escape

from kestrel.cli.output import err_console
from kestrel.cli.prompt import MAX_SUGGESTIONS, POINTER, STYLE, InputHistory, SlashCompleter
from kestrel.cli.shell import Shell

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
QUIT_SECONDS = 2.0
"""A second Ctrl-C within this time leaves the shell."""
MENU_GRACE_SECONDS = 0.5
"""Keys pressed this soon after a permission menu appears are ignored, so an Enter meant
for a typed-ahead message can never approve a command you have not seen."""


@dataclass
class _Menu:
    question: str
    options: Sequence[tuple[str, str]]
    answer: concurrent.futures.Future[str]
    selected: int = 0
    shown_at: float = field(default_factory=time.monotonic)


@dataclass
class _State:
    """What the box shows. Written by the worker thread, read when the app redraws."""

    activity: str | None = None
    since: float = 0.0
    menu: _Menu | None = None
    asking: concurrent.futures.Future[str] | None = None
    cancel: Callable[[], None] | None = None
    busy: bool = False
    closing: bool = False
    """Set when leaving: lines still in the queue are skipped."""
    status: str = ""
    hint: str = ""
    last_ctrl_c: float = 0.0


class Tui:
    """Runs a `Shell` with the input box fixed at the bottom. Implements `BottomBox`."""

    def __init__(self, shell: Shell, history: InputHistory) -> None:
        """`history` is where typed lines are saved (up arrow recalls them)."""
        self.shell = shell
        self.state = _State()
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.buffer = Buffer(
            history=history,
            completer=SlashCompleter(lambda: shell.model_names),
            complete_while_typing=True,
            multiline=False,
            accept_handler=self._submit,
        )
        self.app: Application[None] = self._build()

    # BottomBox, called from the worker thread

    def set_activity(self, label: str | None) -> None:
        """Show `label` with a spinner above the box; None hides it."""
        if label is not None and self.state.activity is None:
            self.state.since = time.monotonic()
        self.state.activity = label
        self.app.invalidate()

    def set_cancel(self, cancel: Callable[[], None] | None) -> None:
        """What Esc calls to stop the running turn."""
        self.state.cancel = cancel

    async def choose(self, question: str, options: Sequence[tuple[str, str]]) -> str:
        """Arrow-key menu above the box."""
        answer: concurrent.futures.Future[str] = concurrent.futures.Future()
        self.state.menu = _Menu(question, options, answer)
        self.app.invalidate()
        try:
            return await asyncio.wrap_future(answer)
        finally:
            self.state.menu = None
            self.app.invalidate()

    async def ask(self, question: str) -> str:
        """The next line typed in the box answers the agent's question."""
        answer: concurrent.futures.Future[str] = concurrent.futures.Future()
        self.state.asking = answer
        self.app.invalidate()
        try:
            return await asyncio.wrap_future(answer)
        finally:
            self.state.asking = None
            self.app.invalidate()

    # running

    def run(self) -> None:
        """Show the welcome panel, then the box, until /exit, Ctrl-D or Ctrl-C twice."""
        self.shell.box = self
        self.shell.start()
        self.state.status = self.shell.prompt()
        worker = threading.Thread(target=self._work, name="kestrel-shell", daemon=True)
        with patch_stdout(raw=True):
            worker.start()
            self.app.run()
            self.state.closing = True
            self._drop_queued()
            self.lines.put(None)
            if self.state.cancel is not None:
                self.state.cancel()
            worker.join(timeout=5)
        self.shell.finish()

    def _work(self) -> None:
        while (line := self.lines.get()) is not None:
            if self.state.closing:
                continue  # leaving: skip whatever was still queued
            self.state.busy = True
            self.app.invalidate()
            gap = "" if line.startswith("/") else "\n"
            err_console.print(f"[bold]>[/bold] {escape(line)}{gap}", highlight=False)
            try:
                keep_going = self.shell.handle(line)
            except Exception as exc:  # a bug in one line must not kill the shell
                err_console.print(f"[red]Error:[/red] {escape(repr(exc))}")
                keep_going = True
            self.state.busy = False
            self.state.activity = None
            self.state.status = self.shell.prompt()
            if not keep_going:
                self.app.loop.call_soon_threadsafe(self._leave)  # type: ignore[union-attr]
                return
            self.app.invalidate()

    def _drop_queued(self) -> None:
        """Forget lines typed ahead but not started yet."""
        while True:
            try:
                self.lines.get_nowait()
            except queue.Empty:
                return

    def _leave(self) -> None:
        if self.app.is_running and not self.app.future.done():  # type: ignore[union-attr]
            self.app.exit()

    def _submit(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        asking = self.state.asking
        if asking is not None:
            buffer.reset()  # an answer to the agent stays out of the saved input history
            if not asking.done():
                asking.set_result(text)
            return False
        if text:
            self.lines.put(text)
        return False  # clear the box; the line is already in the history

    # drawing

    def _build(self) -> Application[None]:
        state = self.state
        keys = KeyBindings()
        in_menu = Condition(lambda: state.menu is not None)

        @keys.add("up", filter=in_menu, eager=True)
        def _(event: KeyPressEvent) -> None:
            if state.menu is not None:
                state.menu.selected = (state.menu.selected - 1) % len(state.menu.options)

        @keys.add("down", filter=in_menu, eager=True)
        def _(event: KeyPressEvent) -> None:
            if state.menu is not None:
                state.menu.selected = (state.menu.selected + 1) % len(state.menu.options)

        @keys.add("enter", filter=in_menu, eager=True)
        def _(event: KeyPressEvent) -> None:
            self._pick(None)

        for n in range(1, 10):

            @keys.add(str(n), filter=in_menu, eager=True)
            def _(event: KeyPressEvent, n: int = n) -> None:
                self._pick(n - 1)

        @keys.add("escape", eager=True)
        def _(event: KeyPressEvent) -> None:
            if state.menu is not None:
                self._pick(len(state.menu.options) - 1)  # the last option is always "No"
            elif state.cancel is not None:
                self._drop_queued()  # stopping means stopping: typed-ahead lines go too
                state.cancel()
            else:
                self.buffer.reset()

        @keys.add("c-c")
        def _(event: KeyPressEvent) -> None:
            if state.menu is not None:
                self._pick(len(state.menu.options) - 1)
            elif state.cancel is not None:
                state.cancel()
            elif self.buffer.text:
                self.buffer.reset()
            elif time.monotonic() - state.last_ctrl_c < QUIT_SECONDS:
                event.app.exit()
            else:
                state.last_ctrl_c = time.monotonic()
                state.hint = "Press Ctrl-C again to leave"
                event.app.invalidate()

        @keys.add("c-d", filter=Condition(lambda: not self.buffer.text and not state.busy))
        def _(event: KeyPressEvent) -> None:
            event.app.exit()

        input_line = Window(
            BufferControl(self.buffer),
            wrap_lines=True,
            dont_extend_height=True,
            get_line_prefix=lambda line, wrap: [("bold", "> " if line == 0 and not wrap else "  ")],
        )
        layout = HSplit(
            [
                Window(height=Dimension(weight=1)),  # takes the free space: box sits at the bottom
                ConditionalContainer(
                    Window(FormattedTextControl(self._activity), height=1),
                    filter=Condition(lambda: state.activity is not None or not self.lines.empty()),
                ),
                ConditionalContainer(
                    Window(FormattedTextControl(self._menu), dont_extend_height=True),
                    filter=in_menu,
                ),
                _rule("╭", "╮"),
                VSplit([_side(), Window(width=1), input_line, Window(width=1), _side()]),
                _rule("╰", "╯"),
                Window(FormattedTextControl(self._footer), height=1),
                ConditionalContainer(
                    Window(FormattedTextControl(self._suggestions), dont_extend_height=True),
                    filter=Condition(lambda: self.buffer.complete_state is not None),
                ),
            ]
        )
        return Application(
            layout=Layout(layout, focused_element=input_line),
            key_bindings=keys,
            style=STYLE,
            erase_when_done=True,
            refresh_interval=0.1,  # keeps the spinner and the seconds moving
        )

    def _pick(self, index: int | None) -> None:
        menu = self.state.menu
        if menu is None or menu.answer.done():
            return
        if time.monotonic() - menu.shown_at < MENU_GRACE_SECONDS:
            return  # too soon to be an answer to this menu
        if index is None:
            index = menu.selected
        if 0 <= index < len(menu.options):
            menu.answer.set_result(menu.options[index][0])

    def _activity(self) -> StyleAndTextTuples:
        state = self.state
        out: StyleAndTextTuples = []
        if state.activity is not None:
            seconds = int(time.monotonic() - state.since)
            frame = SPINNER[int(time.monotonic() * 10) % len(SPINNER)]
            stop = " · esc to stop" if state.cancel is not None else ""
            out += [
                ("class:spinner", f"{frame} "),
                ("class:status", f"{state.activity}… {seconds}s{stop}"),
            ]
        waiting = self.lines.qsize()
        if waiting:
            out.append(("class:status", f"   {waiting} queued"))
        return out

    def _menu(self) -> StyleAndTextTuples:
        menu = self.state.menu
        if menu is None:
            return []
        out: StyleAndTextTuples = [("bold", f"  {menu.question}\n")]
        for i, (_, label) in enumerate(menu.options):
            if i == menu.selected:
                out.append(("class:menu.selected", f"  {POINTER} {i + 1}. {label}\n"))
            else:
                out.append(("", f"    {i + 1}. {label}\n"))
        return out

    def _footer(self) -> StyleAndTextTuples:
        state = self.state
        if state.asking is not None:
            return [("class:spinner", "  type your answer to the agent's question, then Enter")]
        if state.hint and time.monotonic() - state.last_ctrl_c < QUIT_SECONDS:
            return [("class:status", f"  {state.hint}")]
        return [("class:status", f"  {state.status}")]

    def _suggestions(self) -> StyleAndTextTuples:
        complete = self.buffer.complete_state
        if complete is None:
            return []
        out: StyleAndTextTuples = []
        for i, c in enumerate(complete.completions[:MAX_SUGGESTIONS]):
            selected = i == complete.complete_index
            style = "class:suggestion.selected" if selected else "class:suggestion"
            out.append((style, f"  {c.text:<14} {c.display_meta_text}\n"))
        return out


def _rule(left: str, right: str) -> VSplit:
    return VSplit(
        [
            Window(FormattedTextControl([("class:border", left)]), width=1, height=1),
            Window(char="─", height=1, style="class:border"),
            Window(FormattedTextControl([("class:border", right)]), width=1, height=1),
        ]
    )


def _side() -> Window:
    return Window(char="│", width=1, style="class:border")
