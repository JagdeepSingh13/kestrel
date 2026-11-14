"""Seam: `git_status` and `git_diff`, read-only and hardened against the repo's own config."""

import os
import shutil
import subprocess
from collections.abc import Awaitable
from pathlib import Path
from typing import cast

import pytest

from kestrel.safety.paths import Workspace
from kestrel.tools.git import git_diff, git_status
from kestrel.tools.registry import Tool, ToolContext, ToolError

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def git(repo: Path, *args: str) -> None:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)


def make_repo(root: Path) -> Path:
    repo = root / "calc"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "stats.py").write_text("def mean(v):\n    return sum(v) / (len(v) - 1)\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "first")
    return repo


async def call(tool: Tool, root: Path, path: str = ".") -> str:
    ctx = ToolContext(workspace=Workspace(root), env=dict(os.environ))
    return await cast(Awaitable[str], tool.fn(ctx, path))


async def test_status_and_diff_show_the_change(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    assert (await call(git_status, repo)).endswith("nothing changed")
    assert (await call(git_diff, repo)).startswith("no changes")
    (repo / "stats.py").write_text("def mean(v):\n    return sum(v) / len(v)\n")
    (repo / "new.py").write_text("")
    status = await call(git_status, repo)
    assert " M stats.py" in status and "?? new.py" in status
    diff = await call(git_diff, repo)
    assert "stats.py | 2 +-" in diff
    assert "-    return sum(v) / (len(v) - 1)" in diff and "+    return sum(v) / len(v)" in diff
    assert "stats.py" in await call(git_diff, repo, "stats.py")


async def test_the_repos_own_config_cannot_run_programs(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    marker = tmp_path / "ran"
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n")
    script.chmod(0o755)
    git(repo, "config", "core.fsmonitor", str(script))
    git(repo, "config", "diff.external", str(script))
    (repo / "stats.py").write_text("changed\n")
    await call(git_status, repo)
    await call(git_diff, repo)
    assert not marker.exists()


async def test_a_folder_with_no_repo_never_reaches_a_repo_above_the_workspace(
    tmp_path: Path,
) -> None:
    make_repo(tmp_path)  # tmp_path/calc is a repo; the workspace sits inside it
    inner = tmp_path / "calc" / "sub"
    inner.mkdir()
    workspace = inner / "workspace"
    workspace.mkdir()
    with pytest.raises(ToolError, match="not inside a git repository"):
        await call(git_status, workspace)


async def test_filters_named_in_the_repos_attributes_never_run(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    marker = tmp_path / "filtered"
    (repo / ".gitattributes").write_text("* filter=x\n")
    git(repo, "config", "filter.x.clean", f"touch {marker}; cat")
    (repo / "stats.py").write_text("changed\n")
    await call(git_status, repo)
    await call(git_diff, repo)
    assert not marker.exists()
