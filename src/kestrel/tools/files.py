"""File tools: read, list, search, write, append and edit inside the workspace (spec 8.3).

Every path goes through `ctx.workspace.resolve`, the path jail, before anything is opened.
Outputs are shaped for small context windows: line numbers and totals up front, limits on
how much comes back at once. Tools that change a file say exactly what changed and show
the changed lines, so the model (and you) can check the result without another call.
"""

import re
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from kestrel.tools.registry import FileStamp, ToolContext, ToolError, tool

MAX_READ_BYTES = 1_000_000
MAX_LIST_ENTRIES = 300
MAX_MATCHES = 50
SHOWN_AROUND = 2
"""Lines shown before and after a change."""
SKIPPED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache"})
"""Listed, but never opened: they are huge and rarely what the model wants."""


@tool(timeout=10)
def read_file(
    ctx: ToolContext,
    path: Annotated[str, Field(description="File path inside the workspace.")],
    start_line: Annotated[int, Field(ge=1, description="First line to show.")] = 1,
    max_lines: Annotated[int, Field(ge=1, le=1000, description="How many lines.")] = 400,
) -> str:
    """Read a text file inside the workspace. Shows numbered lines and the total line count."""
    file = ctx.workspace.resolve(path)
    if not file.is_file():
        raise ToolError(f"{path} is not a file")
    size = file.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(f"{path} is {size} bytes; files over {MAX_READ_BYTES} bytes are not read")
    raw = file.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ToolError(f"{path} looks like a binary file, not text")
    lines = _lines(raw.decode("utf-8", errors="replace"))
    end = min(len(lines), start_line - 1 + max_lines)
    shown = lines[start_line - 1 : end]
    _remember(ctx, file)
    header = f"{ctx.workspace.relative(file)}: {len(lines)} lines total"
    if not lines:
        return f"{header} (empty file)"
    if start_line > len(lines):
        return f"{header}; start_line {start_line} is past the end"
    if (start_line, end) != (1, len(lines)):
        header += f", showing {start_line} to {end}"
    width = len(str(end))
    body = "\n".join(f"{n:>{width}}| {line}" for n, line in enumerate(shown, start_line))
    return f"{header}\n{body}"


@tool(timeout=10)
def list_dir(
    ctx: ToolContext,
    path: Annotated[str, Field(description="Folder inside the workspace.")] = ".",
    depth: Annotated[int, Field(ge=1, le=3, description="How many levels down to show.")] = 1,
) -> str:
    """List a folder inside the workspace as a tree. Folders end with /, files show their size."""
    folder = ctx.workspace.resolve(path)
    if not folder.is_dir():
        raise ToolError(f"{path} is not a folder")
    lines: list[str] = []
    _walk(folder, depth, "", lines)
    if not lines:
        return f"{ctx.workspace.relative(folder)}/ is empty"
    if len(lines) > MAX_LIST_ENTRIES:
        lines = [*lines[:MAX_LIST_ENTRIES], f"[... {len(lines) - MAX_LIST_ENTRIES} more entries]"]
    return f"{ctx.workspace.relative(folder)}/\n" + "\n".join(lines)


def _walk(folder: Path, depth: int, indent: str, out: list[str]) -> None:
    entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    for entry in entries:
        if len(out) > MAX_LIST_ENTRIES:
            return
        if entry.is_dir():
            skipped = entry.name in SKIPPED_DIRS
            out.append(f"{indent}{entry.name}/" + ("  (not opened)" if skipped else ""))
            if depth > 1 and not skipped:
                _walk(entry, depth - 1, indent + "  ", out)
        else:
            out.append(f"{indent}{entry.name}  ({entry.stat().st_size} bytes)")


def _read_first(ctx: ToolContext, args: BaseModel) -> str | None:
    """Changing an existing file needs a read of its current text in this conversation,
    so the model works from what is there, not from memory. Checked before asking you."""
    path = getattr(args, "path", "")
    if getattr(args, "overwrite", True) is False:
        return None  # write_file without overwrite never replaces anything
    file = ctx.workspace.resolve(path)
    if not file.is_file():
        return None  # the tool itself reports a missing file
    seen = ctx.seen.get(file)
    if seen is None:
        return f"read {path} with read_file first; then change it using its current text."
    if seen != _stamp(file):
        return f"{path} changed since you last read it. read_file it again first."
    return None


def _remember(ctx: ToolContext, file: Path) -> None:
    """The model now knows this file's current text."""
    ctx.seen[file] = _stamp(file)


def _stamp(file: Path) -> FileStamp:
    info = file.stat()
    return info.st_mtime_ns, info.st_size


@tool(
    side_effects=True,
    confirm_if_true="overwrite",
    text_args=["content"],
    timeout=10,
    precheck=_read_first,
)
def write_file(
    ctx: ToolContext,
    path: Annotated[str, Field(description="File path inside the workspace.")],
    content: Annotated[str, Field(description="The full text to write.")],
    overwrite: Annotated[
        bool, Field(description="Must be true to replace a file that already exists.")
    ] = False,
) -> str:
    """Write a text file inside the workspace. Creates folders as needed. Replacing an
    existing file needs overwrite=true (and the user's approval)."""
    file = ctx.workspace.resolve(path)
    if file.is_dir():
        raise ToolError(f"{path} is a folder")
    existed = file.exists()
    if existed and not overwrite:
        raise ToolError(
            f"{path} already exists. To add lines at the end use append_file; to change "
            "part of it use edit_file; to replace all of it set overwrite=true."
        )
    before = _line_count(file) if existed else 0
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    _remember(ctx, file)
    name = ctx.workspace.relative(file)
    lines = len(_lines(content))
    if existed:
        return f"replaced {name}: was {before} lines, now {lines} lines"
    return f"created {name} ({lines} lines)"


@tool(side_effects=True, text_args=["text"], timeout=10)
def append_file(
    ctx: ToolContext,
    path: Annotated[str, Field(description="File path inside the workspace.")],
    text: Annotated[str, Field(description="The lines to add at the end.")],
) -> str:
    """Add text at the end of a file (created if missing), on a new line. Keeps what is
    already there."""
    file = ctx.workspace.resolve(path)
    if file.is_dir():
        raise ToolError(f"{path} is a folder")
    if not text.strip():
        raise ToolError("text is empty: nothing to append")
    old = _read_text(file) if file.exists() else ""
    if old is None:
        raise ToolError(f"{path} is not a text file")
    fresh = file.exists() and ctx.seen.get(file) == _stamp(file)
    eol = "\r\n" if "\r\n" in old else "\n"  # match the file's line endings
    joiner = eol if old and not old.endswith("\n") else ""
    tail = eol.join(text.splitlines()) + eol
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8", newline="") as handle:  # no newline translation
        handle.write(joiner + tail)
    if fresh or not old:
        _remember(ctx, file)  # still known: it only grew by what the model just wrote
    lines = _lines(old + joiner + tail)
    added = len(text.splitlines())
    first = len(lines) - added + 1
    name = ctx.workspace.relative(file)
    shown = _numbered(lines, max(1, first - 1), len(lines))  # short: the new lines matter
    return f"appended {added} line(s) to {name}, now {len(lines)} lines. End of file:\n{shown}"


@tool(
    side_effects=True,
    permission="confirm",
    text_args=["old_text", "new_text"],
    timeout=10,
    precheck=_read_first,
)
def edit_file(
    ctx: ToolContext,
    path: Annotated[str, Field(description="File path inside the workspace.")],
    old_text: Annotated[str, Field(description="Exact text to replace; must appear once.")],
    new_text: Annotated[str, Field(description="Text to put in its place.")],
) -> str:
    """Replace one exact piece of text in a file. Copy old_text from read_file output;
    include enough lines that it appears only once."""
    file = ctx.workspace.resolve(path)
    if not file.is_file():
        raise ToolError(f"{path} is not a file")
    raw = file.read_bytes()
    if len(raw) > MAX_READ_BYTES or b"\x00" in raw[:8192]:
        raise ToolError(f"{path} is not a text file Kestrel can edit")
    try:
        old = raw.decode("utf-8")  # strict: a lossy decode would damage the file on write
    except UnicodeDecodeError:
        raise ToolError(f"{path} is not UTF-8 text, so it was not edited") from None
    if not old_text.strip():
        raise ToolError("old_text is empty. To add at the end use append_file.")
    lines = _lines(old, keep_ends=True)
    # Small models copy read_file's "12| " prefixes and lose trailing spaces: match by
    # whole lines with those ignored, but only ever replace a single place.
    unnumbered = _without_numbers(old_text)
    wanted = [line.rstrip() for line in _lines(unnumbered)]
    plain = [line.rstrip() for line in lines]
    places = [
        i for i in range(len(plain) - len(wanted) + 1) if plain[i : i + len(wanted)] == wanted
    ]
    exact = old.count(old_text)
    if exact == 1:
        at = old.index(old_text)
        stop = at + len(old_text)
        if not new_text and (at == 0 or old[at - 1] == "\n"):
            # Deleting whole lines: take their line break too, so no blank line is left.
            stop += next((len(e) for e in ("\r\n", "\n") if old.startswith(e, stop)), 0)
        new = old[:at] + new_text + old[stop:]
        first = old[:at].count("\n") + 1
        removed = len(_lines(old_text)) or 1
    elif exact == 0 and len(places) == 1:
        if unnumbered != old_text:
            new_text = _without_numbers(new_text)  # it copied the numbers into both
        at_line, removed = places[0], len(wanted)
        last_old = lines[at_line + removed - 1]
        ending = last_old[len(last_old.rstrip("\r\n")) :]
        eol = "\r\n" if "\r\n" in old else "\n"
        block = eol.join(_lines(new_text)) + ending if new_text else ""
        new = "".join([*lines[:at_line], block, *lines[at_line + removed :]])
        first = at_line + 1
    elif exact > 1 or len(places) > 1:
        where = ", ".join(str(i + 1) for i in places[:5]) if places else "several places"
        count = max(exact, len(places))
        raise ToolError(
            f"old_text matches {count} places in {path} (lines {where}). Include more "
            "lines around it so it matches once."
        )
    else:
        raise ToolError(
            f"old_text was not found in {path}. read_file it and copy the text exactly."
        )
    file.write_bytes(new.encode("utf-8"))  # bytes: line endings stay exactly as they were
    _remember(ctx, file)
    added = len(_lines(new_text))
    now = _lines(new)
    last = first + max(added, 1) - 1
    shown = _numbered(now, max(1, first - SHOWN_AROUND), min(len(now), last + SHOWN_AROUND))
    name = ctx.workspace.relative(file)
    return (
        f"edited {name}: {removed} line(s) at line {first} became {added} line(s), now "
        f"{len(now)} lines. Around the change:\n{shown}"
    )


def _lines(text: str, *, keep_ends: bool = False) -> list[str]:
    """Split on line breaks only (str.splitlines also splits on form feeds and other
    rare characters, which would shift line numbers and change files)."""
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts.pop()
    if keep_ends:
        return [p + "\n" for p in parts[:-1]] + (
            [parts[-1] + ("\n" if text.endswith("\n") else "")] if parts else []
        )
    return [p.removesuffix("\r") for p in parts]


_NUMBERED = re.compile(r"^\s*\d+\| ?")


def _without_numbers(text: str) -> str:
    """Drop read_file's line numbers, when every line has them."""
    lines = _lines(text)
    if lines and all(_NUMBERED.match(line) for line in lines):
        return "\n".join(_NUMBERED.sub("", line, count=1) for line in lines)
    return text


@tool(timeout=20)
def search_files(
    ctx: ToolContext,
    text: Annotated[str, Field(description="Text to look for (not case-sensitive).")],
    path: Annotated[str, Field(description="File or folder to search in.")] = ".",
) -> str:
    """Find lines containing some text in the workspace's files. Shows file, line number
    and the line."""
    where = ctx.workspace.resolve(path)
    if not where.exists():
        raise ToolError(f"{path} does not exist")
    if not text.strip():
        raise ToolError("text is empty")
    needle = text.lower()
    files = [where] if where.is_file() else sorted(_files(where))
    hits: list[str] = []
    total = 0
    for file in files:
        content = _read_text(file)
        if content is None:
            continue
        for n, line in enumerate(_lines(content), 1):
            if needle in line.lower():
                total += 1
                if len(hits) < MAX_MATCHES:
                    hits.append(f"{ctx.workspace.relative(file)}:{n}: {line.strip()[:200]}")
    if not hits:
        return f"no lines contain {text!r} in {ctx.workspace.relative(where)}"
    more = f"\n[... {total - MAX_MATCHES} more matches]" if total > MAX_MATCHES else ""
    return f"{total} match(es) for {text!r}:\n" + "\n".join(hits) + more


def _files(folder: Path) -> list[Path]:
    out: list[Path] = []
    for entry in folder.iterdir():
        if entry.is_symlink() or entry.is_junction():
            continue  # could point outside the workspace
        if entry.is_dir():
            if entry.name not in SKIPPED_DIRS:
                out += _files(entry)
        elif entry.is_file():
            out.append(entry)
    return out


def _read_text(file: Path) -> str | None:
    """The file as text; None for binary or oversized files."""
    if file.stat().st_size > MAX_READ_BYTES:
        return None
    raw = file.read_bytes()
    if b"\x00" in raw[:8192]:
        return None
    return raw.decode("utf-8", errors="replace")


def _line_count(file: Path) -> int:
    text = _read_text(file)
    return 0 if text is None else len(_lines(text))


def _numbered(lines: list[str], first: int, last: int) -> str:
    width = len(str(last))
    return "\n".join(f"{n:>{width}}| {lines[n - 1]}" for n in range(first, last + 1))
