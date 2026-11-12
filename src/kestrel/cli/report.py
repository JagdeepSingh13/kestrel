"""`/report-failure`: turn one step of a run into a failure-corpus case (spec 5.6).

The case has the shape of `tests/fixtures/bad_outputs/*.json`: where it came from, the
tools involved, the model's reply exactly as received, and an `expect` block for you to
fill in. It is written locally with secrets removed; nothing is ever uploaded.
"""

import json
from collections.abc import Callable
from pathlib import Path

from kestrel.__about__ import __version__
from kestrel.core.events import load_value, read_events
from kestrel.core.types import Event, EventType, JSONObject, JSONValue

PROBLEM_EVENTS = frozenset({EventType.repair, EventType.loop_warning, EventType.error})


class ReportError(Exception):
    """The run has no step that can be reported."""


def pick_step(events: list[Event]) -> int:
    """The last replied step where something went wrong: a repair, a warning, an error,
    a broken call or a failed tool. Otherwise the last step with a model reply. (A step
    whose request failed has no reply to report.)"""
    replied = [e.step for e in events if e.type is EventType.model_response]
    troubled = {
        e.step
        for e in events
        if e.type in PROBLEM_EVENTS
        or (e.type is EventType.tool_result and e.data.get("is_error"))
        or (e.type is EventType.tool_call and e.data.get("arguments_error"))
    }
    candidates = [s for s in replied if s in troubled] or replied
    if not candidates:
        raise ReportError("this run has no model reply to report")
    return candidates[-1]


def build_case(log: Path, step: int | None, redact: Callable[[JSONValue], JSONValue]) -> JSONObject:
    """A corpus case for `step` (default: `pick_step`) of the run in `log`."""
    events = read_events(log)
    chosen = pick_step(events) if step is None else step
    start = next((e for e in events if e.type is EventType.run_start), None)
    reply = next(
        (e for e in events if e.type is EventType.model_response and e.step == chosen), None
    )
    if start is None or reply is None:
        raise ReportError(f"step {chosen} has no model reply in this run")
    calls = [c for c in _list(load_value(log, reply.data.get("tool_calls"))) if isinstance(c, dict)]
    called = {str(c.get("name")) for c in calls}
    specs = [s for s in _list(load_value(log, start.data.get("tool_specs"))) if isinstance(s, dict)]
    involved = [s for s in specs if str(s.get("name")) in called] or specs
    case: JSONObject = {
        "id": f"{log.stem}-step{chosen}",
        "source": {
            "model": start.data.get("model"),
            "provider": start.data.get("provider"),
            "kestrel_version": __version__,
            "date": reply.ts.date().isoformat(),
            "run_id": log.stem,
            "step": chosen,
        },
        "tools": [
            {k: s[k] for k in ("name", "description", "parameters") if k in s} for s in involved
        ],
        "response": {
            "content": load_value(log, reply.data.get("content")),
            "raw_text": load_value(log, reply.data.get("raw_text")),
            "finish_reason": reply.data.get("finish_reason"),
            "tool_calls": [
                {"name": c.get("name"), "arguments_text": _arguments_text(c)} for c in calls
            ],
        },
        "expect": {
            "kind": "TODO: calls, final or reject",
            "note": "TODO: what a correct reading of this reply is",
        },
    }
    cleaned = redact(case)
    assert isinstance(cleaned, dict)
    return cleaned


def write_case(case: JSONObject, directory: Path) -> Path:
    """Save the case as `<directory>/<id>.json` and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{case['id']}.json"
    path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _arguments_text(call: JSONObject) -> JSONValue:
    raw = call.get("arguments_text")
    return raw if isinstance(raw, str) else json.dumps(call.get("arguments"), ensure_ascii=False)


def _list(value: JSONValue) -> list[JSONValue]:
    return value if isinstance(value, list) else []
