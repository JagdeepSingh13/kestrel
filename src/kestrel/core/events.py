"""Event bus and JSONL run log (spec section 11, gaps 11 and 13).

Everything that happens in a run is emitted on the bus. Subscribers (the CLI, the log
writer, tests) each get every event. The log writer redacts each event, writes it as
one JSON line and flushes at once, so a crash keeps everything up to that moment.
"""

import json
import secrets
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from kestrel.core.config import default_data_dir
from kestrel.core.types import LIVE_ONLY_EVENTS, Event, EventType, JSONObject, JSONValue

SIDE_FILE_BYTES = 64 * 1024
"""Strings longer than this go to a side file next to the log (gap 13)."""

type Subscriber = Callable[[Event], None]
type Redact = Callable[[JSONValue], JSONValue]
"""Scrubs secrets from a JSON value; `safety.redact.Redactor.json` in real use."""
type Clock = Callable[[], datetime]


def utc_now() -> datetime:
    """The current time in UTC."""
    return datetime.now(UTC)


def new_run_id(now: datetime | None = None) -> str:
    """`YYYYMMDD-HHMMSS-xxxx`: sorts by start time; 4 random hex chars avoid clashes."""
    return f"{(now or utc_now()):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def runs_dir(env: Mapping[str, str]) -> Path:
    """Where run logs live: `<data dir>/runs`."""
    return default_data_dir(env) / "runs"


class EventBus:
    """Hands each emitted event to every subscriber, in the order they subscribed."""

    def __init__(self, run_id: str, *, clock: Clock = utc_now) -> None:
        """A bus for one run."""
        self.run_id = run_id
        self._clock = clock
        self._subscribers: list[Subscriber] = []

    def subscribe(self, subscriber: Subscriber) -> None:
        """Receive every future event."""
        self._subscribers.append(subscriber)

    def emit(self, type: EventType, step: int, data: JSONObject | None = None) -> Event:
        """Create an event and deliver it."""
        event = Event(ts=self._clock(), run_id=self.run_id, step=step, type=type, data=data or {})
        for subscriber in self._subscribers:
            subscriber(event)
        return event


class JsonlWriter:
    """Bus subscriber that appends redacted events to `<runs dir>/<run_id>.jsonl`."""

    def __init__(self, path: Path, redact: Redact) -> None:
        """Open (create) the log file. Call `close()` when the run ends."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._side_dir = path.with_suffix(".files")
        self._redact = redact
        self._file: TextIO = path.open("a", encoding="utf-8")
        self._count = 0

    def __call__(self, event: Event) -> None:
        """Write one event (live-only events are skipped)."""
        if event.type in LIVE_ONLY_EVENTS:
            return
        self._count += 1
        data = self._redact(event.data)
        assert isinstance(data, dict)
        data = {key: self._maybe_side_file(key, value) for key, value in data.items()}
        line = event.model_copy(update={"data": data}).model_dump_json()
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        self._file.close()

    def _maybe_side_file(self, key: str, value: JSONValue) -> JSONValue:
        """Values over 64KB go to a file next to the log (text as .txt, lists and objects
        as .json); the event keeps the path (spec gap 13)."""
        if value is None or isinstance(value, bool | int | float):
            return value
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if len(text.encode("utf-8")) <= SIDE_FILE_BYTES:
            return value
        self._side_dir.mkdir(exist_ok=True)
        suffix = "txt" if isinstance(value, str) else "json"
        side = self._side_dir / f"{self._count:04d}-{key}.{suffix}"
        side.write_text(text, encoding="utf-8")
        return {"side_file": f"{self._side_dir.name}/{side.name}", "chars": len(text)}


def load_value(log: Path, value: JSONValue) -> JSONValue:
    """A logged value, read back from its side file if it was moved to one."""
    if isinstance(value, dict) and isinstance(value.get("side_file"), str):
        side = log.parent / str(value["side_file"])
        text = side.read_text(encoding="utf-8")
        return text if side.suffix == ".txt" else json.loads(text)
    return value


def read_events(path: Path) -> list[Event]:
    """Load a run log written by `JsonlWriter`."""
    with path.open(encoding="utf-8") as f:
        return [Event.model_validate_json(line) for line in f if line.strip()]


def find_run(directory: Path, run: str) -> Path | None:
    """Resolve `last`, a full run id, or a unique prefix of one to its log file."""
    logs = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    if run == "last":
        return logs[-1] if logs else None
    matches = [p for p in logs if p.stem.startswith(run)]
    return matches[0] if len(matches) == 1 else None


def first_task(path: Path) -> str:
    """The task text from a log's `run_start` event, for run listings."""
    with path.open(encoding="utf-8") as f:
        first = f.readline()
    try:
        task = json.loads(first)["data"]["task"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "?"
    return task if isinstance(task, str) else "?"
