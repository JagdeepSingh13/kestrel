"""Seam: turning a run's step into a failure case (`/report-failure`, `kestrel report-failure`)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kestrel.cli.report import ReportError, build_case, pick_step
from kestrel.core.events import EventBus, JsonlWriter, read_events
from kestrel.core.types import EventType, JSONObject, JSONValue
from kestrel.safety.redact import Redactor

SPEC: JSONObject = {"name": "read_file", "description": "Read.", "parameters": {"type": "object"}}


def write_run(directory: Path, *events: tuple[int, EventType, JSONObject]) -> Path:
    path = directory / "runs" / "20261112-100000-abcd.jsonl"
    writer = JsonlWriter(path, Redactor({}).json)
    bus = EventBus(path.stem, clock=lambda: datetime(2026, 11, 12, tzinfo=UTC))
    bus.subscribe(writer)
    bus.emit(EventType.run_start, 0, {"model": "m", "provider": "ollama", "tool_specs": [SPEC]})
    for step, type, data in events:
        bus.emit(type, step, data)
    writer.close()
    return path


def reply(calls: list[JSONObject] | None = None) -> JSONObject:
    listed: list[JSONValue] = list(calls or [])
    return {"content": None, "raw_text": "", "tool_calls": listed, "finish_reason": "stop"}


def test_a_failed_request_is_skipped_for_the_last_reply(tmp_path: Path) -> None:
    log = write_run(
        tmp_path,
        (1, EventType.model_response, reply([{"name": "read_file", "arguments": {}}])),
        (2, EventType.model_request, {"message_count": 3}),
        (2, EventType.error, {"kind": "BackendUnreachableError", "message": "down"}),
    )
    assert pick_step(read_events(log)) == 1


def test_no_reply_at_all_is_explained(tmp_path: Path) -> None:
    log = write_run(tmp_path)
    with pytest.raises(ReportError, match="no model reply"):
        build_case(log, None, Redactor({}).json)


def test_secrets_are_removed_from_the_case(tmp_path: Path) -> None:
    call: JSONObject = {"name": "read_file", "arguments_text": '{"path": "MY-SECRET-VALUE"}'}
    log = write_run(tmp_path, (1, EventType.model_response, reply([call])))
    case = build_case(log, 1, Redactor({"SERVICE_TOKEN": "MY-SECRET-VALUE"}).json)
    assert "MY-SECRET-VALUE" not in json.dumps(case)
