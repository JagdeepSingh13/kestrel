"""`run_tests` (M5.5): run the project's tests and give the model only what it needs.

The command never comes from the model: it is the agent file's `test_command`, or for
Python it is found here (the repo's `.venv` Python, else Kestrel's; pytest if it can be
imported, else unittest). The raw output goes to the run log as a `test_run` event; the
model gets `shorten(...)`: one line of counts, at most three failures, the rest by name.
"""

import dataclasses
import importlib.util
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path

from kestrel.tools.process import MAX_SECONDS, run_process, safe_env
from kestrel.tools.registry import OutsideError, Tool, ToolContext, ToolError, tool

BUDGET = 2000
"""Characters the model gets for one test run (about 600 tokens)."""
BLOCKS = 3
BLOCK_LINES = 12
UNREAD_LINES = 25
"""Output that is not pytest or unittest: this many lines from each end."""

_CARETS = re.compile(r"^\s*[\^~]+\s*$")
_PYTEST_SECTION = re.compile(r"^=+ (FAILURES|ERRORS) =+$")
_PYTEST_ANY_SECTION = re.compile(r"^=+ .* =+$")
_PYTEST_TITLE = re.compile(r"^_{3,} (.+?) _{3,}$")
_PYTEST_COUNTS = re.compile(r"^=*\s*((?:\d+ \w+(?:, )?)+) in [\d.]+s\b")
_UNITTEST_TITLE = re.compile(r"^(FAIL|ERROR): \S+ \(.+\)$")
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in ")
_UNITTEST_RESULT = re.compile(r"^(OK|FAILED)(?: \((.*)\))?$")
_PYTEST_FRAME = re.compile(r"^((?:[A-Za-z]:)?[^:]+):\d+: in ")
_UNITTEST_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')


PYTEST_ARGS = ["-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider"]


def find_test_command(root: Path, configured: str | None) -> list[str]:
    """The command `run_tests` runs in `root`: the agent file's `test_command`, else
    pytest or unittest with the repo's `.venv` Python if there is one. Decided by reading
    files only: nothing in the repo runs before you approve the command."""
    if configured:
        return split_command(configured)
    venv = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if venv.is_file():
        python, pytest = str(venv), _wants_pytest(root)
    else:
        found = importlib.util.find_spec("pytest") is not None
        python, pytest = sys.executable, found
    return [python, *(PYTEST_ARGS if pytest else ["-m", "unittest"])]


def split_command(command: str) -> list[str]:
    """`test_command` as a list (raises ValueError if it is empty or badly quoted)."""
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise ValueError("test_command is empty")
    return parts


_PYTEST_CONFIG = ["pytest.ini", "conftest.py", "tests/conftest.py"]
_PYTEST_SIGNS = re.compile(r"^(import pytest|from pytest|@pytest|def test_)", re.MULTILINE)


def _wants_pytest(root: Path) -> bool:
    """Signs a repo uses pytest: its config, or test files that import it or have plain
    `def test_` functions (unittest would not find those)."""
    if any((root / name).is_file() for name in _PYTEST_CONFIG):
        return True
    for name, section in [
        ("pyproject.toml", "[tool.pytest"),
        ("setup.cfg", "[tool:pytest]"),
        ("tox.ini", "[pytest]"),
    ]:
        if section in _text(root / name):
            return True
    tests = [*root.glob("test_*.py"), *root.glob("tests/**/test_*.py")][:50]
    return any(_PYTEST_SIGNS.search(_text(t)[:20_000]) for t in tests if not t.is_symlink())


def _text(file: Path) -> str:
    """A file's text, or "" if it is missing or cannot be read."""
    try:
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def shown(command: list[str], root: Path) -> str:
    """The command as you read it: Kestrel's own Python as `python`, the repo's relative."""
    first = command[0]
    if first == sys.executable:
        first = "python"
    elif Path(first).is_relative_to(root):
        first = Path(first).relative_to(root).as_posix()
    return shlex.join([first, *command[1:]])


@tool(
    permission="confirm",
    side_effects=True,
    timeout=MAX_SECONDS + 10,
    max_output_chars=BUDGET + 200,
)
async def run_tests(ctx: ToolContext) -> str:
    """Run the project's tests. Returns the counts, then the failures (file:line, the
    code, the error). Run it before and after each change."""
    return await _run_tests(ctx, find_test_command(ctx.workspace.root, None))


def run_tests_with(command: list[str], root: Path) -> Tool:
    """`run_tests` for one run: its command fixed and written in its description, so the
    confirm prompt shows exactly what will run."""

    async def fixed(ctx: ToolContext) -> str:
        return await _run_tests(ctx, command)

    spec = run_tests.spec.model_copy(
        update={"description": f"{run_tests.spec.description} Runs: {shown(command, root)}"}
    )
    return dataclasses.replace(run_tests, spec=spec, fn=fixed)


async def _run_tests(ctx: ToolContext, command: list[str]) -> str:
    root = ctx.workspace.root
    env = {**safe_env(ctx.env), "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.monotonic()
    program = shutil.which(command[0], path=env.get("PATH")) or command[0]  # npm.cmd too
    try:
        code, raw = await run_process(
            None, [program, *command[1:]], cwd=root, env=env, seconds=MAX_SECONDS
        )
    except OSError as exc:
        raise ToolError(f"cannot start {command[0]!r}: {exc.strerror or exc}") from None
    took = time.monotonic() - started
    if ctx.emit is not None:  # the full output, for the run log; the model gets the short one
        ctx.emit(
            "test_run",
            {
                "command": shown(command, root),
                "exit_code": code,
                "seconds": round(took, 2),
                "output": raw,
            },
        )
    short = shorten(raw, code, took, root)
    if code != 0:
        raise OutsideError(short)  # a failure for the watchdog and nudges; still untrusted
    return short


def shorten(output: str, exit_code: int, seconds: float, root: str | Path) -> str:
    """What the model sees of a test run: counts first, then up to three failures with
    paths relative to `root`, then the other failures by name; at most `BUDGET` chars."""
    prefix = re.escape(str(root).rstrip("/\\")) + r"[/\\]"
    text = re.sub(prefix, "", output, flags=re.IGNORECASE if os.name == "nt" else 0)
    lines = text.splitlines()
    took = f"({seconds:.1f}s)"
    counts, blocks = _pytest(lines)
    if counts is None:
        counts, blocks = _unittest(lines)
    if counts is None:
        if exit_code == 5 and "no tests ran" in text.lower():  # pytest and unittest
            return f"exit 5: no tests found {took}"
        return _cut(f"exit {exit_code} {took}\n{_head_tail(lines)}")
    header = f"exit {exit_code}: {counts} {took}"
    if exit_code == 0:
        return header
    if not blocks:  # failed for another reason (a plugin, coverage): show the output's end
        return _cut(f"{header}\n" + "\n".join(lines[-UNREAD_LINES:]))
    parts = [header]
    shown = blocks[:BLOCKS]
    for block in shown:
        parts.append("\n".join(block))
    rest = [block[0] for block in blocks[BLOCKS:]]
    if rest:
        parts.append("Also failed: " + ", ".join(rest))
    return _cut("\n\n".join(parts))


def _pytest(lines: list[str]) -> tuple[str | None, list[list[str]]]:
    counts = None
    for line in reversed(lines):
        if match := _PYTEST_COUNTS.match(line):
            counts = match.group(1).strip().rstrip(",")
            break
    if counts is None:
        return None, []
    blocks: list[list[str]] = []
    inside = False
    for line in lines:
        if _PYTEST_SECTION.match(line):
            inside = True
            continue
        if _PYTEST_ANY_SECTION.match(line):
            inside = False
            continue
        if not inside:
            continue
        if title := _PYTEST_TITLE.match(line):
            blocks.append([title.group(1)])
        elif blocks:
            blocks[-1].append(line)
    return counts, [_tidy(block) for block in blocks]


def _unittest(lines: list[str]) -> tuple[str | None, list[list[str]]]:
    ran = next((int(m.group(1)) for line in lines if (m := _UNITTEST_RAN.match(line))), None)
    result = next((m for line in reversed(lines) if (m := _UNITTEST_RESULT.match(line))), None)
    if ran is None or result is None:
        return None, []
    tally = dict(
        (key.strip(), int(value))
        for key, _, value in (part.partition("=") for part in (result.group(2) or "").split(","))
        if value.isdigit()
    )
    failed, errors, skipped = (
        tally.get("failures", 0),
        tally.get("errors", 0),
        tally.get("skipped", 0),
    )
    words = [f"{failed} failed"] if failed else []
    words += [f"{errors} error{'s' if errors > 1 else ''}"] if errors else []
    words += [f"{ran - failed - errors - skipped} passed"]
    words += [f"{skipped} skipped"] if skipped else []
    blocks: list[list[str]] = []
    for i, line in enumerate(lines):
        if not _UNITTEST_TITLE.match(line):
            continue
        body: list[str] = []
        for after in lines[i + 2 :]:  # skip the ---- rule under the title
            if after.startswith(("=" * 20, "-" * 20)):
                break
            body.append(after)
        blocks.append(_tidy([line, *body]))
    return ", ".join(words), blocks


def _tidy(block: list[str]) -> list[str]:
    """One failure: no caret lines, no frames outside the repo, unittest frames written
    like pytest's, at most `BLOCK_LINES` lines (the end, where the error is)."""
    title, body = block[0], block[1:]
    kept: list[str] = []
    skip_source = False
    for line in body:
        if skip_source:
            skip_source = False
            if line.startswith(("    ", "\t")) or line.strip() == "???":
                continue
        if not line.strip() or _CARETS.match(line) or line.startswith("Traceback (most recent"):
            continue
        frame = _UNITTEST_FRAME.match(line)
        if frame:
            path, number, where = frame.groups()
            if _outside(path):
                skip_source = True
                continue
            kept.append(f"{path}:{number}: in {where}")
            continue
        pytest_frame = _PYTEST_FRAME.match(line)
        if pytest_frame and _outside(pytest_frame.group(1)):
            skip_source = True  # pytest frame in site-packages or <frozen ...>
            continue
        kept.append(line)
    if len(kept) > BLOCK_LINES - 1:
        kept = ["...", *kept[-(BLOCK_LINES - 2) :]]
    return [title, *kept]


def _outside(path: str) -> bool:
    """A path that was not made relative to the repo, or not a file at all."""
    return path.startswith(("/", "<")) or bool(re.match(r"^[A-Za-z]:[\\/]", path))


def _head_tail(lines: list[str]) -> str:
    if len(lines) <= 2 * UNREAD_LINES:
        return "\n".join(lines)
    cut = len(lines) - 2 * UNREAD_LINES
    return "\n".join([*lines[:UNREAD_LINES], f"[... {cut} lines cut ...]", *lines[-UNREAD_LINES:]])


def _cut(text: str) -> str:
    if len(text) <= BUDGET:
        return text
    note = "\n[... cut to fit ...]"
    return text[: BUDGET - len(note)] + note
