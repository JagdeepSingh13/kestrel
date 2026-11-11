"""File tools: read, list and write inside the workspace (spec section 8.3).

Every path goes through `ctx.workspace.resolve`, the path jail, before anything is opened.
Outputs are shaped for small context windows: line numbers and totals up front, limits on
how much comes back at once.
"""

from pathlib import Path
from typing import Annotated

from pydantic import Field

from kestrel.tools.registry import ToolContext, tool

MAX_READ_BYTES = 1_000_000
MAX_LIST_ENTRIES = 300
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
        return f"ERROR: {path} is not a file"
    size = file.stat().st_size
    if size > MAX_READ_BYTES:
        return f"ERROR: {path} is {size} bytes; files over {MAX_READ_BYTES} bytes are not read"
    raw = file.read_bytes()
    if b"\x00" in raw[:8192]:
        return f"ERROR: {path} looks like a binary file, not text"
    lines = raw.decode("utf-8", errors="replace").splitlines()
    end = min(len(lines), start_line - 1 + max_lines)
    shown = lines[start_line - 1 : end]
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
        return f"ERROR: {path} is not a folder"
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


@tool(side_effects=True, confirm_if_true="overwrite", timeout=10)
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
        return f"ERROR: {path} is a folder"
    existed = file.exists()
    if existed and not overwrite:
        return f"ERROR: {path} already exists. Set overwrite=true to replace it."
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    verb = "replaced" if existed else "created"
    return f"{verb} {ctx.workspace.relative(file)} ({len(content)} characters)"
