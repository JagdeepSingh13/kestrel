"""Drive the real `kestrel` shell in a pseudo-terminal and print what the screen shows.

Dev tool for live checks (used by the kestrel-live-tester agent). Unix only. Needs a
running backend. pyte (a terminal emulator) is pulled in just for this run:

    uv run --no-project -q --with pyte==0.8.2 python scripts/drive_shell.py STEP...

Steps, run in order:
    wait:SECONDS     read output for this long
    until:TEXT       read until TEXT is on screen (gives up after 300 s)
    send:KEYS        type KEYS; escapes work: \\r Enter, \\x1b Esc, \\x03 Ctrl-C,
                     \\x1b[A Up, \\x1b[B Down
    snap:LABEL       print the screen with row numbers

The emulated terminal answers cursor-position queries like a real one, so the shell's
layout (input box pinned at the bottom) is drawn exactly as in a real terminal.
"""

import fcntl
import os
import pty
import select
import struct
import sys
import termios
import time

import pyte

ROWS, COLS = 30, 100
CURSOR_QUERY = b"\x1b[6n"
UNTIL_LIMIT_SECONDS = 300


def main(steps: list[str]) -> None:
    """Start the shell and run `steps` against it."""
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(project)
        os.environ["TERM"] = "xterm-256color"
        os.execvp("uv", ["uv", "run", "-q", "kestrel"])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen = pyte.HistoryScreen(COLS, ROWS, history=3000)
    stream = pyte.ByteStream(screen)

    def read(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            try:
                data = os.read(fd, 65536)
            except OSError:
                return  # the shell exited
            stream.feed(data.replace(CURSOR_QUERY, b""))
            for _ in range(data.count(CURSOR_QUERY)):
                answer = f"\x1b[{screen.cursor.y + 1};{screen.cursor.x + 1}R"
                os.write(fd, answer.encode())

    def text() -> str:
        return "\n".join(screen.display)

    for step in steps:
        kind, _, value = step.partition(":")
        if kind == "wait":
            read(float(value))
        elif kind == "until":
            started = time.time()
            while value not in text() and time.time() - started < UNTIL_LIMIT_SECONDS:
                read(0.5)
            if value not in text():
                print(f"---- gave up waiting for {value!r} ----")
        elif kind == "snap":
            rows = (f"{i:2}|{line.rstrip()}" for i, line in enumerate(screen.display))
            print(f"---- snap {value} ----\n" + "\n".join(rows))
        elif kind == "send":
            os.write(fd, value.encode().decode("unicode_escape").encode())
        else:
            sys.exit(f"unknown step {step!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
