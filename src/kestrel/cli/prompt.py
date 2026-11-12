"""Shared terminal input pieces (prompt_toolkit), in the Claude Code style.

- `SlashCompleter`: the `/commands` (with their help) and model names offered as you type.
- `input_history`: lines typed in the shell, saved between sessions.
"""

import os
from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory, History
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
