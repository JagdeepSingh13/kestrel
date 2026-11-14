"""Seam: `repo_map`, the outline the coder agent reads first."""

from pathlib import Path

from kestrel.safety.paths import Workspace
from kestrel.tools.code import BUDGET, repo_map
from kestrel.tools.registry import ToolContext


def outline(root: Path, path: str = ".") -> str:
    result = repo_map.fn(ToolContext(workspace=Workspace(root), env={}), path)
    assert isinstance(result, str)
    return result


def test_python_files_show_functions_with_arguments_and_classes_with_methods(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "shop.py").write_text(
        "import os\n"
        "def total(items, *, tax=0.2):\n    return 1\n"
        "class Cart:\n    def add(self, item): ...\n    def clear(self): ...\n"
        "async def fetch(url, **kw): ...\n"
    )
    (tmp_path / "notes.txt").write_text("hi")
    assert outline(tmp_path) == (
        "Outline of ./ (2 files)\n"
        "notes.txt\n"
        "pkg/shop.py\n"
        "  def total(items, tax)\n"
        "  class Cart: add, clear\n"
        "  def fetch(url, **kw)"
    )


def test_a_file_that_does_not_parse_says_where(tmp_path: Path) -> None:
    (tmp_path / "broken.py").write_text("x = 1\ndef f(:\n")
    assert "broken.py  (syntax error line 2)" in outline(tmp_path)


def test_noise_hidden_files_and_symlinks_are_skipped(tmp_path: Path) -> None:
    for folder in [".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "build"]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "x.py").write_text("def hidden(): ...\n")
    (tmp_path / ".env").write_text("SECRET=1")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "far.py").write_text("def far(): ...\n")
    (tmp_path / "link").symlink_to(outside)
    (tmp_path / "main.py").write_text("def main(): ...\n")
    assert outline(tmp_path) == "Outline of ./ (1 files)\nmain.py\n  def main()"


def test_big_repos_get_plainer_but_stay_within_the_budget(tmp_path: Path) -> None:
    for i in range(40):
        body = "".join(f"def function_number_{j}(alpha, beta, gamma): ...\n" for j in range(4))
        (tmp_path / f"module_{i}.py").write_text(body)
    short = outline(tmp_path)
    assert len(short) <= BUDGET
    assert "module_0.py: function_number_0" in short  # names only, still every file
    for i in range(400):
        (tmp_path / f"extra_{i}.py").write_text("")
    assert len(outline(tmp_path)) <= BUDGET
