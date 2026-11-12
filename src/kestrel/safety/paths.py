"""Path jail: tools may only touch files inside the workspace folder (CLAUDE.md security).

Every path a model gives a tool goes through `Workspace.resolve` first. It follows `..`
and symlinks to find where the path really points, and refuses anything that lands
outside the workspace.

Known limit: a symlink could be swapped between this check and the file being opened
(a "time of check, time of use" race). That needs an attacker already writing inside
the workspace while a run is going; out of scope for M3.
"""

from pathlib import Path

WORKSPACE_FOLDER = "workspace"
"""Name of the only folder agents may use (owner decision, 2026-11-11)."""


class PathNotAllowedError(Exception):
    """A path points outside the workspace, or is not a usable path at all."""


class Workspace:
    """The one folder an agent's tools may read and write."""

    def __init__(self, root: Path) -> None:
        """`root` must be an existing folder. Symlinks in it are resolved once, here."""
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise PathNotAllowedError(f"workspace {root} is not a folder")
        self.root = resolved

    def resolve(self, path: str) -> Path:
        """The real location of `path`, inside the workspace.

        The workspace is the model's whole world: `/` is the workspace itself, so `/data`
        means `<workspace>/data` (small models assume this). An absolute path that already
        points inside the workspace is kept. Raises `PathNotAllowedError` if the path is
        empty, contains a null byte, or ends up outside after following `..` and symlinks.
        """
        if not path:
            raise PathNotAllowedError("empty path")
        if "\x00" in path:
            raise PathNotAllowedError("path contains a null byte")
        candidate = Path(path)
        # `anchor`, not is_absolute(): on Windows "/data" has a root but no drive, so it is
        # not "absolute", yet joining it to the workspace would land on C:\\data.
        if candidate.anchor and not candidate.resolve().is_relative_to(self.root):
            candidate = Path(*candidate.parts[1:])  # drop "/" (or "C:\\"): workspace-rooted
        if not candidate.is_absolute():
            candidate = self.root / candidate
        # resolve() follows every symlink that exists along the way and collapses `..`;
        # parts that do not exist yet (a new file) are kept as written.
        real = candidate.resolve()
        if not real.is_relative_to(self.root):
            raise PathNotAllowedError(
                f"{path!r} is outside the workspace. Paths are relative to the workspace "
                "folder, for example 'data/sales.csv'."
            )
        return real

    @classmethod
    def default(cls, start: Path | None = None) -> "Workspace":
        """The agents' folder: `<project>/workspace`, created if missing.

        For now this is the only workspace there is (no way to point agents elsewhere).
        `<project>` is the Kestrel checkout this code runs from: the nearest folder above
        it holding a `pyproject.toml` for `kestrel`. `start` exists for tests.
        """
        project = find_project_root(start or Path(__file__))
        if project is None:
            raise PathNotAllowedError(
                "cannot find the Kestrel project folder to create the agents' workspace in"
            )
        folder = project / WORKSPACE_FOLDER
        folder.mkdir(exist_ok=True)
        return cls(folder)

    def relative(self, path: Path) -> str:
        """`path` as the model should see it: relative, with forward slashes."""
        rel = path.relative_to(self.root).as_posix()
        return rel or "."


def find_project_root(start: Path) -> Path | None:
    """The nearest folder at or above `start` whose pyproject.toml names `kestrel`."""
    for folder in [start.resolve(), *start.resolve().parents]:
        pyproject = folder / "pyproject.toml"
        if pyproject.is_file() and 'name = "kestrel"' in pyproject.read_text(encoding="utf-8"):
            return folder
    return None
