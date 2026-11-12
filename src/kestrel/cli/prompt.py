"""Shared terminal input pieces (prompt_toolkit), in the Claude Code style.

- `SlashCompleter`: the `/commands` (with their help) and model names offered as you type.
- `input_history`: lines typed in the shell, saved between sessions.
- `choose`: a one-off arrow-key menu (`kestrel run` in a terminal; the shell's own box
  in `cli.tui` draws its menu itself).
"""

import os
from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit.application import Application
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.history import FileHistory, History
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from kestrel.cli.commands import COMMANDS
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
        "spinner": "#d78700",
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


type InputHistory = History
"""Lines typed in the shell, saved between sessions."""


def input_history() -> InputHistory:
    """The shell's input history file in the data dir."""
    data_dir = default_data_dir(os.environ)
    data_dir.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(data_dir / HISTORY_FILE))


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
