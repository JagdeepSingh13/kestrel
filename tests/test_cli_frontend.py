"""Seam: what the terminal shows for a run, and the arrow-key menu (driven by piped keys)."""

import asyncio

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from kestrel.cli import frontend
from kestrel.cli.frontend import CliFrontend, describe_args, summarize_result
from kestrel.cli.prompt import choose
from kestrel.core.types import ToolCall, ToolSpec


def test_call_line_shows_the_one_argument_that_matters() -> None:
    assert describe_args({"path": "data/sales.csv", "max_lines": 5}) == "data/sales.csv"
    assert describe_args({"command": "ls -la"}) == "ls -la"
    assert describe_args({"code": "import csv\nprint(1)"}) == "import csv…"
    assert describe_args({}) == ""
    assert describe_args({"a": 1}) == '{"a": 1}'
    assert describe_args({"command": "x" * 100}).endswith("…")


def test_result_preview_is_a_few_lines_and_a_count() -> None:
    assert summarize_result("one line") == "one line"
    text = "\n".join(f"line {n}" for n in range(10))
    assert summarize_result(text) == "line 0\nline 1\nline 2\n… +7 lines"
    assert summarize_result("") == "(no output)"


def test_without_a_terminal_confirm_takes_a_typed_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def typed_a(*args: object, **kwargs: object) -> str:
        return "a"

    monkeypatch.setattr(frontend.Prompt, "ask", typed_a)
    ui = CliFrontend(interactive=False)
    call = ToolCall(id="1", name="run_shell", arguments={"command": "rm -rf build"})
    spec = ToolSpec(name="run_shell", description="d", parameters={}, permission="confirm")
    assert asyncio.run(ui.confirm(call, spec)) == "always"
    assert "rm -rf build" in capsys.readouterr().err  # the exact command is shown first


@pytest.mark.parametrize(
    ("keys", "expected"),
    [("\r", "yes"), ("\x1b[B\r", "always"), ("\x1b[A\r", "no"), ("3", "no"), ("2", "always")],
)
def test_menu_picks_with_arrows_numbers_and_enter(keys: str, expected: str) -> None:
    options = [("yes", "Yes"), ("always", "Yes, always"), ("no", "No")]
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(keys)
        assert asyncio.run(choose("Allow it?", options)) == expected
