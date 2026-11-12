"""Esc (or Ctrl-C) stops a running turn without leaving the shell.

While a turn runs, the terminal is put in raw mode and key presses are read in the event
loop; Esc or Ctrl-C cancels the turn's task. Questions to the user (permission menu,
`ask_user`) pause the watcher so they get the keys. Does nothing without a terminal.
"""

import asyncio
import sys
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager

from prompt_toolkit.input import Input, create_input
from prompt_toolkit.keys import Keys

STOP_KEYS = frozenset({Keys.Escape, Keys.ControlC})
FLUSH_SECONDS = 0.1
"""A lone Esc byte is only known to be Esc (not the start of an arrow key) after a pause."""


@contextmanager
def watch_stop_keys(
    on_stop: Callable[[], None],
) -> Generator[Callable[[], AbstractContextManager[None]] | None]:
    """Call `on_stop` when Esc or Ctrl-C is pressed. Yields a `pause()` function, or None
    when there is no terminal to watch."""
    if not sys.stdin.isatty():
        yield None
        return
    terminal = create_input()
    loop = asyncio.get_running_loop()
    pending: list[asyncio.TimerHandle] = []

    def check(keys: list[Keys | str]) -> None:
        if any(k in STOP_KEYS for k in keys):
            on_stop()

    def flush() -> None:
        check([k.key for k in terminal.flush_keys()])

    def ready() -> None:
        check([k.key for k in terminal.read_keys()])
        for handle in pending:
            handle.cancel()
        pending[:] = [loop.call_later(FLUSH_SECONDS, flush)]

    with terminal.raw_mode(), terminal.attach(ready):
        yield lambda: _paused(terminal)
    for handle in pending:
        handle.cancel()


def _paused(terminal: Input) -> AbstractContextManager[None]:
    return terminal.detach()
