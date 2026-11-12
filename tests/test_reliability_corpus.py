"""Seam: `repair()` against the failure corpus, one real or typical broken reply per file.

Each case in tests/fixtures/bad_outputs/ holds the tools, the model's reply as received,
and what a correct reading is: the calls to run (and which repairs got there), a final
answer, or a rejection with a note for the model. New cases come from /report-failure.
"""

import json
from pathlib import Path

import pytest

from kestrel.core.types import JSONObject, Message, Role, ToolSpec
from kestrel.providers.wire import tool_call_from_text
from kestrel.reliability.repair import repair

CASES = sorted((Path(__file__).parent / "fixtures" / "bad_outputs").glob("*.json"))


def load(path: Path) -> JSONObject:
    data: JSONObject = json.loads(path.read_text(encoding="utf-8"))
    return data


def test_corpus_is_big_enough() -> None:
    assert len(CASES) >= 30  # M4 done criterion


@pytest.mark.parametrize("path", CASES, ids=[p.stem for p in CASES])
def test_case(path: Path) -> None:
    case = load(path)
    specs = [ToolSpec.model_validate(t) for t in case["tools"]]  # type: ignore[union-attr]
    response = case["response"]
    assert isinstance(response, dict)
    raw_calls = response.get("tool_calls") or []
    assert isinstance(raw_calls, list)
    message = Message(
        role=Role.assistant,
        content=response.get("content"),  # type: ignore[arg-type]
        tool_calls=[
            tool_call_from_text(f"c{i}", str(c["name"]), str(c["arguments_text"]))  # type: ignore[index]
            for i, c in enumerate(raw_calls)
        ],
    )
    result = repair(message, specs, truncated=response.get("finish_reason") == "length")
    expect = case["expect"]
    assert isinstance(expect, dict)
    kinds = [f.kind for f in result.fixes]
    match expect["kind"]:
        case "calls":
            assert result.problem is None, result.problem
            got = [{"name": c.name, "arguments": c.arguments} for c in result.calls]
            assert got == expect["calls"]
            assert kinds == expect["repairs"]
        case "final":
            assert result.calls == [] and result.problem is None
        case "reject":
            assert result.calls == []
            assert result.problem is not None
            assert str(expect["error_contains"]) in result.problem
        case other:
            pytest.fail(f"unknown expect kind {other!r}")
