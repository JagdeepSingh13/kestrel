"""Terminal input for the shell (prompt_toolkit), in the Claude Code style.

- `InputBox`: a bordered input box with a status line under it, input history, and the
  matching `/commands` listed under the box as you type (Tab or arrows pick one).
- `choose`: a small arrow-key menu, used for permission prompts.

Kept apart from `shell.py` so the shell can be tested without a terminal.
"""

import os
from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from rich.markup import escape

from kestrel.cli.commands import COMMANDS
from kestrel.cli.output import err_console
from kestrel.cli.shell import ReadLine
from kestrel.core.config import default_data_dir

HISTORY_FILE = "shell_history"
MAX_SUGGESTIONS = 8
POINTER = "\u276f"
"""The heavy angle that marks the chosen menu line."""

STYLE = Style.from_dict(
    {
        "border": "#888888",
        "status": "#888888",
        "suggestion": "#888888",
        "suggestion.selected": "bold #5fafff",
        "menu.selected": "bold #5fafff",
    }
)


class SlashCompleter(Completer):
    """Completes `/command` names (with their help text), and model names after `/model `."""

    def __init__(self, model_names: Callable[[], Sequence[str]]) -> None:
        """`model_names` returns the models last listed by `/model`."""
        self._model_names = model_names

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Suggestions for the text before the cursor."""
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if text.startswith("/model "):
            typed = text[len("/model ") :]
            for name in self._model_names():
                if name.startswith(typed):
                    yield Completion(name, start_position=-len(typed))
            return
        if " " not in text:
            for name, command in COMMANDS.items():
                if name.startswith(text[1:]) and not command.hidden:
                    yield Completion(
                        f"/{name}", start_position=-len(text), display_meta=command.help
                    )


class InputBox:
    """The bordered input box. Call it with the status text; it returns the typed line.

    Raises KeyboardInterrupt (Ctrl-C) or EOFError (Ctrl-D on an empty box), like `input()`.
    """

    def __init__(self, model_names: Callable[[], Sequence[str]]) -> None:
        """Input history is saved in the data dir."""
        data_dir = default_data_dir(os.environ)
        data_dir.mkdir(parents=True, exist_ok=True)
        self.history = FileHistory(str(data_dir / HISTORY_FILE))
        self.completer = SlashCompleter(model_names)

    def __call__(self, status: str) -> str:
        """Show the box with `status` under it until Enter."""
        buffer = Buffer(
            history=self.history,
            completer=self.completer,
            complete_while_typing=True,
            multiline=False,
            accept_handler=lambda buf: _exit_with(buf.text),
        )
        keys = KeyBindings()

        @keys.add("c-c")
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=KeyboardInterrupt)

        @keys.add("c-d", filter=Condition(lambda: not buffer.text))
        def _(event: KeyPressEvent) -> None:
            event.app.exit(exception=EOFError)

        def suggestions() -> StyleAndTextTuples:
            state = buffer.complete_state
            if state is None:
                return []
            out: StyleAndTextTuples = []
            for i, c in enumerate(state.completions[:MAX_SUGGESTIONS]):
                style = (
                    "class:suggestion.selected"
                    if i == state.complete_index
                    else ("class:suggestion")
                )
                out.append((style, f"  {c.text:<14} {c.display_meta_text}\n"))
            return out

        input_line = Window(
            BufferControl(buffer),
            wrap_lines=True,
            dont_extend_height=True,
            get_line_prefix=lambda line, wrap: [("bold", "> " if line == 0 and not wrap else "  ")],
        )
        box = HSplit(
            [
                _rule("╭", "╮"),
                VSplit([_side(), Window(width=1), input_line, Window(width=1), _side()]),
                _rule("╰", "╯"),
                Window(FormattedTextControl([("class:status", f"  {status}")]), height=1),
                ConditionalContainer(
                    Window(FormattedTextControl(suggestions), dont_extend_height=True),
                    filter=Condition(lambda: buffer.complete_state is not None),
                ),
            ]
        )
        app: Application[str] = Application(
            layout=Layout(box, focused_element=input_line),
            key_bindings=keys,
            style=STYLE,
            erase_when_done=True,
        )
        text = app.run()
        if text.strip():
            gap = "" if text.startswith("/") else "\n"
            err_console.print(f"[bold]>[/bold] {escape(text)}{gap}", highlight=False)
        return text


async def choose[T](question: str, options: Sequence[tuple[T, str]]) -> T:
    """Arrow-key menu: Up/Down and Enter, a number key, or Esc for the last option."""
    selected = 0
    keys = KeyBindings()

    @keys.add("up")
    def _(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = (selected - 1) % len(options)

    @keys.add("down")
    def _(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = (selected + 1) % len(options)

    @keys.add("enter")
    def _(event: KeyPressEvent) -> None:
        event.app.exit(result=selected)

    @keys.add("escape")
    @keys.add("c-c")
    def _(event: KeyPressEvent) -> None:
        event.app.exit(result=len(options) - 1)

    for n in range(len(options)):

        @keys.add(str(n + 1))
        def _(event: KeyPressEvent, n: int = n) -> None:
            event.app.exit(result=n)

    def menu() -> StyleAndTextTuples:
        out: StyleAndTextTuples = [("bold", f"  {question}\n")]
        for i, (_, label) in enumerate(options):
            if i == selected:
                out.append(("class:menu.selected", f"  {POINTER} {i + 1}. {label}\n"))
            else:
                out.append(("", f"    {i + 1}. {label}\n"))
        return out

    app: Application[int] = Application(
        layout=Layout(Window(FormattedTextControl(menu), dont_extend_height=True)),
        key_bindings=keys,
        style=STYLE,
        erase_when_done=True,
    )
    return options[await app.run_async()][0]


def terminal_read_line(model_names: Callable[[], Sequence[str]]) -> ReadLine:
    """The shell's input in a real terminal."""
    return InputBox(model_names)


def _exit_with(text: str) -> bool:
    get_app().exit(result=text)
    return False  # clear the box (the text is already in the history)


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
