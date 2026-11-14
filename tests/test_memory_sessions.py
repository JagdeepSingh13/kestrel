"""Seam: SessionStore (saved conversations in SQLite)."""

from pathlib import Path

from kestrel.core.types import Message, Role, ToolCall
from kestrel.memory.sessions import SessionStore

SECRET = "sk-test-0123456789abcdef0123"


def test_a_session_round_trips_with_secrets_removed(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "s.db", lambda text: text.replace(SECRET, "[REDACTED]"))
    call = ToolCall(id="c", name="read_file", arguments={"path": "a.txt", "key": SECRET})
    messages = [
        Message(role=Role.user, content=f"use key {SECRET} to total data/sales.csv"),
        Message(role=Role.assistant, tool_calls=[call]),
        Message(role=Role.tool, content="1| x", tool_call_id="c"),
        Message(role=Role.assistant, content="done"),
    ]
    store.save(
        "s1",
        workspace="/w",
        agent="default",
        provider="ollama",
        model="qwen3:8b",
        ladder=["openrouter:m:free"],
        messages=messages,
        run_id="r1",
    )
    store.save(
        "s1",
        workspace="/w",
        agent="default",
        provider="ollama",
        model="qwen3:8b",
        ladder=[],
        messages=messages[:2],
        run_id="r2",
    )  # a later save replaces the messages (a /retry took a turn back)
    loaded = store.load("s1")
    assert loaded is not None
    info, back = loaded
    assert [m.role for m in back] == [Role.user, Role.assistant]
    assert (
        SECRET not in (back[0].content or "")
        and back[1].tool_calls[0].arguments["key"] == "[REDACTED]"
    )
    assert info.title.startswith("use key [REDACTED]") and info.messages == 2 and info.ladder == []
    assert store.runs("s1") == ["r1", "r2"]
    assert [s.id for s in store.recent()] == ["s1"]
    assert store.load("nope") is None


def test_a_secret_cut_by_the_title_length_is_still_removed(tmp_path: Path) -> None:
    secret = "sk-A1b2C3d4E5f6G7h8I9j0"
    store = SessionStore(tmp_path / "s.db", lambda text: text.replace(secret, "[REDACTED]"))
    message = Message(role=Role.user, content="x" * 70 + " " + secret)
    store.save(
        "s", workspace="/w", agent="a", provider="p", model="m", ladder=[], messages=[message]
    )
    assert "sk-A1b2" not in store.recent()[0].title
