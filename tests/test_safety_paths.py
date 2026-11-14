"""Path jail: every path a tool receives must stay inside the workspace (CLAUDE.md security)."""

import os
from pathlib import Path

import pytest

from kestrel.safety.paths import PathNotAllowedError, Workspace


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("outside\n", encoding="utf-8")
    return Workspace(root)


def symlink(link: Path, target: Path) -> None:
    """Make a symlink, or skip the test where the OS does not allow it (Windows without
    developer mode)."""
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        pytest.skip("symlinks not permitted on this system")


def test_paths_inside_are_resolved(ws: Workspace) -> None:
    assert ws.resolve("src/app.py") == ws.root / "src" / "app.py"
    assert ws.resolve(".") == ws.root
    assert ws.resolve("src/../src/app.py") == ws.root / "src" / "app.py"
    assert ws.resolve("new/report.md") == ws.root / "new" / "report.md"  # need not exist


def test_absolute_path_inside_is_allowed(ws: Workspace) -> None:
    assert ws.resolve(str(ws.root / "src" / "app.py")) == ws.root / "src" / "app.py"


@pytest.mark.parametrize("path", ["..", "../secret.txt", "src/../../secret.txt", "a/b/../../../x"])
def test_dot_dot_escapes_are_refused(ws: Workspace, path: str) -> None:
    with pytest.raises(PathNotAllowedError, match="outside the workspace"):
        ws.resolve(path)


def test_absolute_paths_are_rooted_at_the_workspace(ws: Workspace) -> None:
    # The workspace is the model's whole world: "/data" is <workspace>/data, never /data.
    assert ws.resolve("/data/sales.csv") == ws.root / "data" / "sales.csv"
    assert ws.resolve(os.path.abspath(os.sep)) == ws.root  # the filesystem root, any OS
    outside = ws.root.parent / "secret.txt"
    assert ws.resolve(str(outside)).is_relative_to(ws.root)  # never the real file
    with pytest.raises(PathNotAllowedError):
        ws.resolve("/../secret.txt")


def test_symlinked_file_pointing_outside_is_refused(ws: Workspace) -> None:
    symlink(ws.root / "innocent.txt", ws.root.parent / "secret.txt")
    with pytest.raises(PathNotAllowedError):
        ws.resolve("innocent.txt")


def test_symlinked_folder_pointing_outside_is_refused(ws: Workspace) -> None:
    symlink(ws.root / "docs", ws.root.parent)
    with pytest.raises(PathNotAllowedError):
        ws.resolve("docs/secret.txt")
    with pytest.raises(PathNotAllowedError):
        ws.resolve("docs/new-file.txt")  # even a file that does not exist yet


def test_symlink_that_stays_inside_is_fine(ws: Workspace) -> None:
    symlink(ws.root / "alias.py", ws.root / "src" / "app.py")
    assert ws.resolve("alias.py") == ws.root / "src" / "app.py"


@pytest.mark.parametrize("path", ["", "src/\x00app.py"])
def test_empty_and_null_byte_paths_are_refused(ws: Workspace, path: str) -> None:
    with pytest.raises(PathNotAllowedError):
        ws.resolve(path)


def test_workspace_given_through_a_symlink_still_works(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    symlink(tmp_path / "link", real)
    ws = Workspace(tmp_path / "link")
    assert ws.root == real.resolve()
    assert ws.resolve("x.txt") == real.resolve() / "x.txt"


def test_workspace_must_be_an_existing_folder(tmp_path: Path) -> None:
    with pytest.raises(PathNotAllowedError, match="not a folder"):
        Workspace(tmp_path / "missing")


def test_relative_shows_paths_as_the_model_should_see_them(ws: Workspace) -> None:
    assert ws.relative(ws.root / "src" / "app.py") == "src/app.py"
    assert ws.relative(ws.root) == "."


def test_open_creates_the_folder(tmp_path: Path) -> None:
    ws = Workspace.open(tmp_path / "a" / "kestrel-workspace")
    assert ws.root.is_dir()
    with pytest.raises(PathNotAllowedError):
        ws.resolve("../secret.txt")


def test_open_refuses_home_above_home_root_and_relative_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home" / "me"
    home.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    (tmp_path / "link").symlink_to(home)
    for folder in ["~", str(home.parent), "/", "relative/folder", str(tmp_path / "link")]:
        with pytest.raises(PathNotAllowedError):
            Workspace.open(folder)
    assert (
        Workspace.open("~/Documents/kestrel-workspace").root
        == (home / "Documents" / "kestrel-workspace").resolve()
    )


def test_open_refuses_home_typed_in_another_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "Home"
    home.mkdir()
    if not (tmp_path / "home").exists():
        pytest.skip("case-sensitive file system")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    with pytest.raises(PathNotAllowedError):
        Workspace.open(str(tmp_path / "home"))
