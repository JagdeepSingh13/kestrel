"""Seams: what `kestrel run` does when a risky tool asks first; what the shell prints."""

import asyncio
from datetime import UTC, datetime
from typing import cast

import pytest

from kestrel.cli import frontend
from kestrel.cli.frontend import BottomBox, CliFrontend
from kestrel.core.types import Event, EventType, JSONValue, ToolCall, ToolSpec


def test_confirm_shows_the_command_then_takes_a_typed_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def typed_a(*args: object, **kwargs: object) -> str:
        return "a"

    monkeypatch.setattr(frontend.Prompt, "ask", typed_a)
    call = ToolCall(id="1", name="run_shell", arguments={"command": "rm -rf build"})
    spec = ToolSpec(name="run_shell", description="d", parameters={}, permission="confirm")
    assert asyncio.run(CliFrontend().confirm(call, spec)) == "always"
    assert "rm -rf build" in capsys.readouterr().err  # the exact command is shown first


class _Box:
    def set_activity(self, label: str | None) -> None: ...


def _event(what: EventType, **data: JSONValue) -> Event:
    return Event(ts=datetime.now(UTC), run_id="r", step=1, type=what, data=data)


def test_shell_does_not_print_an_answer_twice_when_a_nudge_gets_it_again(
    capsys: pytest.CaptureFixture[str],
) -> None:
    shown = CliFrontend(box=cast(BottomBox, _Box()))
    for _ in range(2):
        shown.on_event(_event(EventType.model_request))
        shown.on_event(_event(EventType.model_delta, text="I cannot access it.", reasoning=""))
        shown.on_event(_event(EventType.model_response))
        shown.on_event(_event(EventType.nudge, kind="after_failure", said=""))
    out = capsys.readouterr()
    assert out.out.count("I cannot access it.") == 1
    assert "the same answer again" in out.err
