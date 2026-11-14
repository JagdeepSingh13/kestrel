"""Read-only git tools (M5.5): `git_status` and `git_diff`.

A repo's own git settings can run programs (`core.fsmonitor`, external diff and textconv
drivers, filters named in `.gitattributes`), so every call turns those off. Git looks
for the repo only inside the workspace: `GIT_CEILING_DIRECTORIES` stops it climbing out
into some other repo, and `GIT_*` variables from your environment are dropped. Nothing
here writes: no add, commit, checkout or reset.
"""

import os
from typing import Annotated

from pydantic import Field

from kestrel.core.context import head_tail
from kestrel.tools.process import run_process, safe_env
from kestrel.tools.registry import ToolContext, ToolError, tool

GIT_SECONDS = 20
MAX_DIFF_CHARS = 6000
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
SAFE_FLAGS = [
    "--no-pager",
    # Read .gitattributes from the empty tree: a repo's own attributes could name a
    # "filter" program that git runs on status and diff (needs git 2.40 or newer).
    f"--attr-source={EMPTY_TREE}",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.untrackedCache=false",
    "-c",
    f"core.hooksPath={os.devnull}",
    "-c",
    "diff.external=",
    "-c",
    "color.ui=false",
]


@tool(timeout=GIT_SECONDS + 5)
async def git_status(
    ctx: ToolContext,
    path: Annotated[str, Field(description="Folder inside the repo.")] = ".",
) -> str:
    """Show which files are changed, added or new in the git repo (read-only)."""
    out = await _git(ctx, path, ["status", "--short", "--branch"])
    lines = out.splitlines()
    if len(lines) <= 1:
        return f"{out}\nnothing changed".strip()
    return out


@tool(timeout=GIT_SECONDS + 5, max_output_chars=MAX_DIFF_CHARS + 300)
async def git_diff(
    ctx: ToolContext,
    path: Annotated[str, Field(description="Folder or file inside the repo.")] = ".",
) -> str:
    """Show the changed lines in the git repo since the last commit (read-only): a
    summary per file, then the changes."""
    target = ctx.workspace.resolve(path)
    folder = target if target.is_dir() else target.parent
    only = [] if target.is_dir() else ["--", target.name]
    flags = ["--no-ext-diff", "--no-textconv", "--no-color"]
    stat = await _git(ctx, ctx.workspace.relative(folder), ["diff", "--stat", *flags, *only])
    if not stat:
        return "no changes since the last commit (new files show in git_status)"
    patch = await _git(ctx, ctx.workspace.relative(folder), ["diff", *flags, *only])
    return head_tail(f"{stat}\n\n{patch}", MAX_DIFF_CHARS)


async def _git(ctx: ToolContext, path: str, args: list[str]) -> str:
    folder = ctx.workspace.resolve(path)
    if not folder.is_dir():
        raise ToolError(f"{path} is not a folder")
    env = {k: v for k, v in safe_env(ctx.env).items() if not k.upper().startswith("GIT_")}
    env["GIT_CEILING_DIRECTORIES"] = str(ctx.workspace.root.parent)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_OPTIONAL_LOCKS"] = "0"  # status must not write the index
    try:
        code, out = await run_process(
            None, ["git", *SAFE_FLAGS, *args], cwd=folder, env=env, seconds=GIT_SECONDS
        )
    except FileNotFoundError:
        raise ToolError("git is not installed") from None
    if code != 0:
        if "attr-source" in out:
            raise ToolError("git 2.40 or newer is needed (it can turn off repo filters)")
        if "not a git repository" in out:
            where = ctx.workspace.relative(folder)
            raise ToolError(f"{where} is not inside a git repository (in the workspace)")
        raise ToolError(f"git failed (exit {code}): {out.strip()[:500]}")
    return out.strip()
