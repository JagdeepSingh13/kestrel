"""Saved shell conversations (M5), in one SQLite file: `<data dir>/sessions.db`.

A session is one shell conversation: its messages (tool calls and results included),
which agent, model and ladder it used, and the runs it was made of. It is saved after
every kept turn, with secrets removed the same way as in run logs, so `/resume` can pick
it up later. Not saved: cloud on or off, `/up`, and which files were read (after
resuming, the model reads a file again before editing it).
"""

import json
import sqlite3
import uuid
from collections.abc import Callable, Generator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kestrel.core.types import JSONValue, Message, Role

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    workspace TEXT NOT NULL,
    title TEXT NOT NULL,
    agent TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    ladder TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS runs (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL,
    PRIMARY KEY (session_id, run_id)
);
"""


@dataclass(frozen=True)
class SessionInfo:
    """One saved session, as listed by `/sessions`."""

    id: str
    created: str
    updated: str
    workspace: str
    title: str
    """The first thing you asked, cut short."""
    agent: str
    provider: str
    model: str
    ladder: list[str]
    messages: int


def new_session_id(now: datetime | None = None) -> str:
    """`20261114-120102-a3f9`: sortable by time, like run ids."""
    now = now or datetime.now(UTC).astimezone()
    return f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def sessions_db(data_dir: Path) -> Path:
    """Where sessions are saved."""
    return data_dir / "sessions.db"


class SessionStore:
    """Read and write saved sessions. Each call opens and closes the database, so the
    shell's worker thread and its main thread never share a connection."""

    def __init__(self, path: Path, redact: Callable[[str], str] = lambda text: text) -> None:
        """Sessions in `path` (created on first save); `redact` removes secrets."""
        self.path = path
        self.redact = redact

    def save(
        self,
        session_id: str,
        *,
        workspace: str,
        agent: str,
        provider: str,
        model: str,
        ladder: list[str],
        messages: list[Message],
        run_id: str | None = None,
    ) -> None:
        """Store the session as it is now (its messages replace the saved ones)."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        first = next((m.content or "" for m in messages if m.role is Role.user), "")
        title = " ".join(self.redact(first).split())[:80]  # redact before cutting
        rows = [(session_id, i, self._dump(m)) for i, m in enumerate(messages)]
        with self._db() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated=excluded.updated, agent=excluded.agent, "
                "provider=excluded.provider, model=excluded.model, ladder=excluded.ladder",
                (
                    session_id,
                    now,
                    now,
                    workspace,
                    title,
                    agent,
                    provider,
                    model,
                    json.dumps(ladder),
                ),
            )
            db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            db.executemany("INSERT INTO messages VALUES (?, ?, ?)", rows)
            if run_id is not None:
                db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?)", (session_id, run_id))

    def add_run(self, session_id: str, run_id: str) -> None:
        """Remember that a run (kept or not) belonged to this session."""
        if not self.exists(session_id):
            return  # the session is created by its first kept turn
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?)", (session_id, run_id))

    def exists(self, session_id: str) -> bool:
        """True once the session has been saved."""
        if not self.path.exists():
            return False
        with self._db() as db:
            row = db.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def recent(self, limit: int = 20) -> list[SessionInfo]:
        """The most recently updated sessions, newest first."""
        if not self.path.exists():
            return []
        with self._db() as db:
            rows = db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) "
                "FROM sessions s ORDER BY updated DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_info(row) for row in rows]

    def load(self, session_id: str) -> tuple[SessionInfo, list[Message]] | None:
        """A saved session and its messages; None if there is no such session."""
        if not self.path.exists():
            return None
        with self._db() as db:
            row = db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) "
                "FROM sessions s WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            texts = db.execute(
                "SELECT message FROM messages WHERE session_id = ? ORDER BY idx", (session_id,)
            ).fetchall()
        return _info(row), [Message.model_validate_json(t[0]) for t in texts]

    def runs(self, session_id: str) -> list[str]:
        """The run ids of a session, oldest first."""
        if not self.path.exists():
            return []
        with self._db() as db:
            rows = db.execute(
                "SELECT run_id FROM runs WHERE session_id = ? ORDER BY run_id", (session_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def _dump(self, message: Message) -> str:
        data: JSONValue = json.loads(message.model_dump_json())
        return json.dumps(_redact_json(data, self.redact), ensure_ascii=False)

    @contextmanager
    def _db(self) -> Generator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript(SCHEMA)
            with db:  # one transaction: committed, or rolled back on error
                yield db


def _info(row: tuple[object, ...]) -> SessionInfo:
    sid, created, updated, workspace, title, agent, provider, model, ladder, count = row
    return SessionInfo(
        id=str(sid),
        created=str(created),
        updated=str(updated),
        workspace=str(workspace),
        title=str(title),
        agent=str(agent),
        provider=str(provider),
        model=str(model),
        ladder=[str(x) for x in json.loads(str(ladder))],
        messages=int(str(count)),
    )


def _redact_json(value: JSONValue, redact: Callable[[str], str]) -> JSONValue:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_json(v, redact) for v in value]
    if isinstance(value, dict):
        return {k: _redact_json(v, redact) for k, v in value.items()}
    return value
