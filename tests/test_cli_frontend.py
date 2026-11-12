"""Seam: what `kestrel run` (no bottom box) does when a risky tool asks first."""

import asyncio

import pytest

from kestrel.cli import frontend
from kestrel.cli.frontend import CliFrontend
from kestrel.core.types import ToolCall, ToolSpec


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
