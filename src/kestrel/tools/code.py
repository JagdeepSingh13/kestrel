"""`repo_map` (M5.5): an outline of a code folder that fits a small window.

Python files are read with `ast` (nothing is run): top-level functions with their
argument names, classes with their method names. Other files are listed by name. If the
outline is too long it gets plainer, in steps: names only, then file names, then a count
per folder. A Python file that does not parse says so, with the line: that helps a model
fixing it. Folders like `.git`, `.venv` and `node_modules`, hidden files and symlinks
are skipped.
"""

import ast
import os
from pathlib import Path
from typing import Annotated

from pydantic import Field

from kestrel.tools.files import SKIPPED_DIRS
from kestrel.tools.registry import ToolContext, ToolError, tool

BUDGET = 4200
"""Characters (about 1,200 tokens)."""
MAX_FILES = 2000
MAX_PY_BYTES = 300_000
"""Bigger Python files are listed by name only."""
MAX_METHODS = 12


@tool(timeout=20, max_output_chars=BUDGET + 200)
def repo_map(
    ctx: ToolContext,
    path: Annotated[str, Field(description="Folder to outline.")] = ".",
) -> str:
    """Outline a code folder: its files, and in Python files the functions and classes
    with their arguments and methods. Use it first to find where things are."""
    folder = ctx.workspace.resolve(path)
    if not folder.is_dir():
        raise ToolError(f"{path} is not a folder")
    files = _files(folder)
    if not files:
        return f"{ctx.workspace.relative(folder)}/ has no files"
    head = f"Outline of {ctx.workspace.relative(folder)}/ ({len(files)} files)"
    if len(files) >= MAX_FILES:
        head += f", stopped at {MAX_FILES}"
    outlines = {f: _outline(folder / f) for f in files}
    for render in (_full, _names, _file_names, _per_folder):
        body = render(files, outlines)
        if len(head) + 1 + len(body) <= BUDGET:
            return f"{head}\n{body}"
    return f"{head}\n{body[: BUDGET - len(head) - 30]}\n[... cut to fit ...]"


def _files(folder: Path) -> list[str]:
    """Relative paths of the files worth showing, sorted, at most `MAX_FILES`."""
    found: list[str] = []
    for here, dirs, names in os.walk(folder):  # does not follow symlinked folders
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in SKIPPED_DIRS
            and not d.startswith(".")
            and not d.endswith(".egg-info")
            and not (Path(here) / d).is_symlink()
        )
        for name in sorted(names):
            full = Path(here) / name
            if name.startswith(".") or full.is_symlink() or name.endswith((".pyc", ".pyo")):
                continue
            found.append(full.relative_to(folder).as_posix())
            if len(found) >= MAX_FILES:
                return found
    return found


def _outline(file: Path) -> list[str] | str | None:
    """A Python file's top-level functions and classes; a short note if it does not
    parse; None for other files."""
    if file.suffix != ".py":
        return None
    try:
        if file.stat().st_size > MAX_PY_BYTES:
            return "(too big to outline)"
        tree = ast.parse(file.read_bytes(), filename=file.name)
    except SyntaxError as exc:
        return f"(syntax error line {exc.lineno})"
    except (OSError, ValueError, MemoryError, RecursionError):
        return "(cannot read)"
    items: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            items.append(f"def {node.name}({_args(node.args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            shown = ", ".join(methods[:MAX_METHODS])
            more = f", +{len(methods) - MAX_METHODS}" if len(methods) > MAX_METHODS else ""
            items.append(f"class {node.name}" + (f": {shown}{more}" if methods else ""))
    return items


def _args(args: ast.arguments) -> str:
    names = [a.arg for a in [*args.posonlyargs, *args.args] if a.arg not in ("self", "cls")]
    if args.vararg:
        names.append(f"*{args.vararg.arg}")
    names += [a.arg for a in args.kwonlyargs]
    if args.kwarg:
        names.append(f"**{args.kwarg.arg}")
    return ", ".join(names)


Outlines = dict[str, list[str] | str | None]


def _full(files: list[str], outlines: Outlines) -> str:
    lines: list[str] = []
    for f in files:
        outline = outlines[f]
        if isinstance(outline, str):
            lines.append(f"{f}  {outline}")
            continue
        lines.append(f)
        lines += [f"  {item}" for item in outline or []]
    return "\n".join(lines)


def _names(files: list[str], outlines: Outlines) -> str:
    lines: list[str] = []
    for f in files:
        outline = outlines[f]
        if isinstance(outline, list) and outline:
            names = [item.split("(")[0].split(":")[0].split(" ", 1)[1] for item in outline]
            lines.append(f"{f}: {', '.join(names)}")
        else:
            lines.append(f"{f}  {outline}" if isinstance(outline, str) else f)
    return "\n".join(lines)


def _file_names(files: list[str], outlines: Outlines) -> str:
    by_folder: dict[str, list[str]] = {}
    for f in files:
        parent, _, name = f.rpartition("/")
        by_folder.setdefault(parent or ".", []).append(name)
    return "\n".join(f"{folder}/: {', '.join(names)}" for folder, names in by_folder.items())


def _per_folder(files: list[str], outlines: Outlines) -> str:
    counts: dict[str, int] = {}
    for f in files:
        parent = f.rpartition("/")[0] or "."
        counts[parent] = counts.get(parent, 0) + 1
    return "\n".join(f"{folder}/: {n} files" for folder, n in counts.items())
