"""File tools, called the way the model calls them: through the registry."""

from pathlib import Path

import pytest

from kestrel.core.types import JSONObject, ToolCall
from kestrel.safety.paths import Workspace
from kestrel.tools.files import list_dir, read_file, write_file
from kestrel.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "app.py").write_bytes(b"a = 1\nb = 2\nc = 3\n")
    (root / "src" / "pkg" / "deep.py").write_bytes(b"x\n")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG\x00\x00binary")
    (tmp_path / "outside.txt").write_text("secret\n", encoding="utf-8")
    return root


async def call(ws: Path, name: str, arguments: JSONObject) -> str:
    registry = ToolRegistry(
        [read_file, list_dir, write_file], ToolContext(workspace=Workspace(ws), env={})
    )
    result = await registry.execute(ToolCall(id="c1", name=name, arguments=arguments))
    return result.content


async def test_read_file_numbers_lines_and_gives_the_total(ws: Path) -> None:
    assert await call(ws, "read_file", {"path": "src/app.py"}) == (
        "src/app.py: 3 lines total\n1| a = 1\n2| b = 2\n3| c = 3"
    )
    assert await call(ws, "read_file", {"path": "src/app.py", "start_line": 2, "max_lines": 1}) == (
        "src/app.py: 3 lines total, showing 2 to 2\n2| b = 2"
    )


async def test_read_file_refuses_what_it_should(ws: Path) -> None:
    assert "binary" in await call(ws, "read_file", {"path": "image.png"})
    assert "not a file" in await call(ws, "read_file", {"path": "missing.txt"})
    assert "outside the workspace" in await call(ws, "read_file", {"path": "../outside.txt"})


async def test_list_dir_shows_a_tree_and_skips_noise(ws: Path) -> None:
    assert await call(ws, "list_dir", {"depth": 3}) == (
        "./\n"
        ".git/  (not opened)\n"
        "src/\n"
        "  pkg/\n"
        "    deep.py  (2 bytes)\n"
        "  app.py  (18 bytes)\n"
        "image.png  (12 bytes)"
    )
    assert await call(ws, "list_dir", {}) == "./\n.git/  (not opened)\nsrc/\nimage.png  (12 bytes)"


async def test_write_file_creates_but_never_silently_overwrites(ws: Path) -> None:
    assert await call(ws, "write_file", {"path": "out/report.md", "content": "hi"}) == (
        "created out/report.md (2 characters)"
    )
    assert (ws / "out" / "report.md").read_text(encoding="utf-8") == "hi"
    refused = await call(ws, "write_file", {"path": "out/report.md", "content": "bye"})
    assert refused.startswith("ERROR: out/report.md already exists")
    assert (
        await call(ws, "write_file", {"path": "out/report.md", "content": "bye", "overwrite": True})
        == "replaced out/report.md (3 characters)"
    )
    assert "outside the workspace" in await call(
        ws, "write_file", {"path": "../evil.txt", "content": "x"}
    )
    assert not (ws.parent / "evil.txt").exists()
