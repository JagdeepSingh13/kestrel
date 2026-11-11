"""Event bus, JSONL run log, and replaying a recorded run."""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from kestrel.core.context import ContextManager
from kestrel.core.events import (
    SIDE_FILE_BYTES,
    EventBus,
    JsonlWriter,
    find_run,
    first_task,
    new_run_id,
    read_events,
)
from kestrel.core.loop import run_agent
from kestrel.core.types import Event, EventType
from kestrel.providers.fake import FakeProvider
from kestrel.safety.redact import REDACTED, Redactor
from kestrel.tools.demo import DemoToolbox

NOW = datetime(2026, 11, 11, 10, 15, 2, tzinfo=UTC)


def test_run_ids_sort_by_time() -> None:
    run_id = new_run_id(NOW)
    assert re.fullmatch(r"20261111-101502-[0-9a-f]{4}", run_id)
    assert new_run_id(datetime(2026, 11, 12, tzinfo=UTC)) > run_id


def test_bus_delivers_to_every_subscriber_in_order() -> None:
    seen: list[str] = []
    bus = EventBus("r1", clock=lambda: NOW)
    bus.subscribe(lambda e: seen.append(f"a:{e.type}"))
    bus.subscribe(lambda e: seen.append(f"b:{e.type}"))
    event = bus.emit(EventType.final, 3, {"answer": "42"})
    assert seen == ["a:final", "b:final"]
    assert (event.run_id, event.step, event.ts, event.data) == ("r1", 3, NOW, {"answer": "42"})


def write(tmp_path: Path, *events: tuple[EventType, dict[str, str]]) -> Path:
    path = tmp_path / "runs" / "r1.jsonl"
    writer = JsonlWriter(path, Redactor({"MY_API_KEY": "hunter2secret"}).json)
    bus = EventBus("r1", clock=lambda: NOW)
    bus.subscribe(writer)
    for type, data in events:
        bus.emit(type, 1, dict(data))
    writer.close()
    return path


def test_log_has_one_line_per_event_and_skips_live_pieces(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        (EventType.run_start, {"task": "hi"}),
        (EventType.model_delta, {"text": "H"}),
        (EventType.final, {"answer": "Hi"}),
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["type"] for line in lines] == ["run_start", "final"]
    assert json.loads(lines[0])["ts"].startswith("2026-11-11T10:15:02")
    assert [e.type for e in read_events(path)] == [EventType.run_start, EventType.final]


def test_secrets_never_reach_the_file(tmp_path: Path) -> None:
    path = write(
        tmp_path, (EventType.tool_call, {"cmd": "login hunter2secret sk-abcdefghij0123456789"})
    )
    raw = path.read_text(encoding="utf-8")
    assert "hunter2secret" not in raw
    assert "sk-abc" not in raw
    assert raw.count(REDACTED) == 2


def test_huge_values_go_to_a_side_file(tmp_path: Path) -> None:
    big = "x" * (SIDE_FILE_BYTES + 1)
    path = write(tmp_path, (EventType.tool_result, {"content": big, "id": "c1"}))
    [event] = read_events(path)
    ref = event.data["content"]
    assert isinstance(ref, dict)
    assert ref["chars"] == len(big)
    side = path.parent / str(ref["side_file"])
    assert side.read_text(encoding="utf-8") == big
    assert event.data["id"] == "c1"


def test_find_run_by_last_and_prefix(tmp_path: Path) -> None:
    for run_id in ["20261110-090000-aaaa", "20261111-090000-bbbb", "20261111-100000-cccc"]:
        (tmp_path / f"{run_id}.jsonl").write_text(
            json.dumps({"data": {"task": f"task {run_id[-4:]}"}}) + "\n", encoding="utf-8"
        )
    last = find_run(tmp_path, "last")
    assert last is not None
    assert last.stem == "20261111-100000-cccc"
    assert first_task(last) == "task cccc"
    found = find_run(tmp_path, "20261110")
    assert found is not None
    assert found.stem.endswith("aaaa")
    assert find_run(tmp_path, "20261111") is None  # ambiguous
    assert find_run(tmp_path / "missing", "last") is None


async def test_recorded_run_replays_to_the_same_result(tmp_path: Path) -> None:
    """Spec 13: feed a log's model replies back through the loop; same events come out."""

    async def run(provider: FakeProvider, log: Path) -> tuple[str | None, list[EventType]]:
        writer = JsonlWriter(log, Redactor({}).json)
        bus = EventBus(log.stem)
        bus.subscribe(writer)
        result = await run_agent(
            "What is 17 + 25?",
            provider=provider,
            model="m",
            toolbox=DemoToolbox(),
            bus=bus,
            context=ContextManager("system"),
        )
        writer.close()
        return result.answer, [e.type for e in read_events(log)]

    original = FakeProvider(
        [FakeProvider.tool_call("add", {"a": 17, "b": 25}), FakeProvider.text("42")]
    )
    first = await run(original, tmp_path / "first.jsonl")
    replayed = await run(
        FakeProvider.replay(read_events(tmp_path / "first.jsonl")), tmp_path / "again.jsonl"
    )
    assert replayed == first
    assert first[0] == "42"


def test_events_round_trip_through_json() -> None:
    event = Event(ts=NOW, run_id="r", step=1, type=EventType.final, data={"answer": "x"})
    assert Event.model_validate_json(event.model_dump_json()) == event
