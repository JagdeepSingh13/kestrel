"""Terminal input for the shell (prompt_toolkit): history, Tab completion, Ctrl-C/Ctrl-D.

Kept apart from `shell.py` so the shell can be tested without a terminal.
"""

import os
from collections.abc import Callable, Iterable, Sequence

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory

from kestrel.cli.commands import COMMANDS
from kestrel.cli.shell import ReadLine
from kestrel.core.config import default_data_dir

HISTORY_FILE = "shell_history"


class SlashCompleter(Completer):
    """Completes `/command` names, and model names after `/model `."""

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
            for name in COMMANDS:
                if name.startswith(text[1:]):
                    yield Completion(f"/{name}", start_position=-len(text))


def terminal_read_line(model_names: Callable[[], Sequence[str]]) -> ReadLine:
    """A `ReadLine` backed by prompt_toolkit, with input history saved to the data dir."""
    data_dir = default_data_dir(os.environ)
    data_dir.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(data_dir / HISTORY_FILE)),
        completer=SlashCompleter(model_names),
        complete_while_typing=True,
    )

    def read_line(prompt: str) -> str:
        return session.prompt(prompt)

    return read_line
