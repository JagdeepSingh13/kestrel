"""File tools, called the way the model calls them: through the registry."""

from pathlib import Path

import pytest

from kestrel.core.types import JSONObject, ToolCall, ToolResult
from kestrel.safety.paths import Workspace
from kestrel.tools.files import (
    append_file,
    edit_file,
    list_dir,
    read_file,
    search_files,
    write_file,
)
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
    return (await result_of(ws, name, arguments)).content


async def result_of(ws: Path, name: str, arguments: JSONObject) -> ToolResult:
    registry = ToolRegistry(
        [read_file, list_dir, write_file, append_file, edit_file, search_files],
        ToolContext(workspace=Workspace(ws), env={}),
    )
    return await registry.execute(ToolCall(id="c1", name=name, arguments=arguments))


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
        "created out/report.md (1 lines)"
    )
    assert (ws / "out" / "report.md").read_text(encoding="utf-8") == "hi"
    refused = await call(ws, "write_file", {"path": "out/report.md", "content": "bye"})
    assert refused.startswith("ERROR: out/report.md already exists")
    assert (
        await call(ws, "write_file", {"path": "out/report.md", "content": "bye", "overwrite": True})
        == "replaced out/report.md: was 1 lines, now 1 lines"
    )
    assert "outside the workspace" in await call(
        ws, "write_file", {"path": "../evil.txt", "content": "x"}
    )
    assert not (ws.parent / "evil.txt").exists()


async def test_append_file_adds_on_a_new_line_and_shows_the_end(ws: Path) -> None:
    (ws / "sales.csv").write_text("12 nov, iphone, 5, 1200", encoding="utf-8")  # no newline
    result = await call(ws, "append_file", {"path": "sales.csv", "text": "13 nov, mac, 10, 1300"})
    assert (ws / "sales.csv").read_text(encoding="utf-8") == (
        "12 nov, iphone, 5, 1200\n13 nov, mac, 10, 1300\n"
    )
    assert result == (
        "appended 1 line(s) to sales.csv, now 2 lines. End of file:\n"
        "1| 12 nov, iphone, 5, 1200\n2| 13 nov, mac, 10, 1300"
    )
    await call(ws, "append_file", {"path": "new/log.txt", "text": "first"})
    assert (ws / "new" / "log.txt").read_text(encoding="utf-8") == "first\n"


async def test_edit_file_replaces_one_exact_match_only(ws: Path) -> None:
    result = await call(
        ws, "edit_file", {"path": "src/app.py", "old_text": "b = 2", "new_text": "b = 20"}
    )
    assert (ws / "src" / "app.py").read_text(encoding="utf-8") == "a = 1\nb = 20\nc = 3\n"
    assert result.startswith("edited src/app.py: 1 line(s) at line 2 became 1 line(s)")
    assert result.endswith("1| a = 1\n2| b = 20\n3| c = 3")
    missing = await call(
        ws, "edit_file", {"path": "src/app.py", "old_text": "zzz", "new_text": "y"}
    )
    assert missing.startswith("ERROR: old_text was not found")
    numbered: JSONObject = {"path": "src/app.py", "old_text": "3| c = 3  ", "new_text": "c = 30"}
    assert (await call(ws, "edit_file", numbered)).startswith("edited")  # copied from read_file
    twice = await call(ws, "edit_file", {"path": "src/app.py", "old_text": " = ", "new_text": "="})
    assert twice.startswith("ERROR: old_text matches 3 places")
    assert (ws / "src" / "app.py").read_text(encoding="utf-8") == "a = 1\nb = 20\nc = 30\n"


async def test_search_files_finds_lines_and_stays_inside(ws: Path) -> None:
    assert await call(ws, "search_files", {"text": "B = 2"}) == (
        "1 match(es) for 'B = 2':\nsrc/app.py:2: b = 2"
    )
    assert "no lines contain" in await call(ws, "search_files", {"text": "secret"})
    assert "outside the workspace" in await call(ws, "search_files", {"text": "x", "path": ".."})


async def test_a_refused_write_is_a_failed_call(ws: Path) -> None:
    refused = await result_of(ws, "write_file", {"path": "src/app.py", "content": "x"})
    assert refused.is_error  # the watchdog counts it; it is not outside data


async def test_edit_file_keeps_line_endings_and_refuses_what_it_would_damage(ws: Path) -> None:
    (ws / "win.txt").write_bytes(b"a\r\nb\r\nc\r\n")
    edit: JSONObject = {"path": "win.txt", "old_text": "2| b", "new_text": "B\nB2"}
    assert (await call(ws, "edit_file", edit)).startswith("edited")
    assert (ws / "win.txt").read_bytes() == b"a\r\nB\r\nB2\r\nc\r\n"
    (ws / "latin.txt").write_bytes(b"caf\xe9\nb\n")
    latin: JSONObject = {"path": "latin.txt", "old_text": "b", "new_text": "c"}
    assert "not UTF-8" in await call(ws, "edit_file", latin)
    assert (ws / "latin.txt").read_bytes() == b"caf\xe9\nb\n"


async def test_edit_file_deletes_whole_lines_without_leaving_a_gap(ws: Path) -> None:
    gone: JSONObject = {"path": "src/app.py", "old_text": "b = 2\n", "new_text": ""}
    await call(ws, "edit_file", gone)
    assert (ws / "src" / "app.py").read_text(encoding="utf-8") == "a = 1\nc = 3\n"
