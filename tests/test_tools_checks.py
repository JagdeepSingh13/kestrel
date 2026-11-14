"""Seam: `shorten`, what the model sees of a test run (recorded pytest and unittest output)."""

import os
import shutil
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from kestrel.agents.schema import AgentConfig
from kestrel.core.types import JSONObject
from kestrel.safety.paths import Workspace
from kestrel.tools.checks import BUDGET, find_test_command, run_tests_with, shorten
from kestrel.tools.registry import OutsideError, Tool, ToolContext

OUTPUTS = Path(__file__).parent / "fixtures" / "test_output"


def recorded(name: str) -> str:
    return (OUTPUTS / f"{name}.txt").read_text(encoding="utf-8")


def test_pytest_failure_keeps_the_location_line_and_error() -> None:
    assert shorten(recorded("pytest_fail"), 1, 0.4, "/ROOT") == (
        "exit 1: 1 failed, 2 passed (0.4s)\n"
        "\n"
        "TestStats.test_mean\n"
        "tests/test_stats.py:8: in test_mean\n"
        "    self.assertEqual(mean([2, 4, 6]), 4)\n"
        "E   AssertionError: 6.0 != 4"
    )


def test_unittest_failure_reads_like_pytest_with_relative_paths() -> None:
    assert shorten(recorded("unittest_fail"), 1, 0.0, "/ROOT") == (
        "exit 1: 1 failed, 2 passed (0.0s)\n"
        "\n"
        "FAIL: test_mean (tests.test_stats.TestStats.test_mean)\n"
        "tests/test_stats.py:8: in test_mean\n"
        "    self.assertEqual(mean([2, 4, 6]), 4)\n"
        "AssertionError: 6.0 != 4"
    )


def test_passing_runs_are_one_line() -> None:
    assert shorten(recorded("pytest_pass"), 0, 0.2, "/ROOT") == "exit 0: 3 passed (0.2s)"
    assert shorten(recorded("unittest_pass"), 0, 0.2, "/ROOT") == "exit 0: 3 passed (0.2s)"


def test_no_tests_found() -> None:
    for name in ["pytest_none", "unittest_none"]:
        assert shorten(recorded(name), 5, 0.1, "/ROOT") == "exit 5: no tests found (0.1s)"


def test_many_failures_show_three_blocks_then_names() -> None:
    for name in ["pytest_many", "unittest_many"]:
        short = shorten(recorded(name), 1, 0.3, "/ROOT")
        first = short.splitlines()[0]
        assert first.startswith("exit 1: ") and first.endswith("2 passed (0.3s)")
        assert short.count("\n\n") == 4  # header, three blocks, the rest by name
        assert short.splitlines()[-1].startswith("Also failed: ")
        assert "^^^" not in short and "~~~" not in short  # caret lines are noise
        assert "/ROOT" not in short


def test_import_errors_drop_frames_outside_the_repo() -> None:
    for name in ["pytest_syntax", "unittest_syntax"]:
        short = shorten(recorded(name), 1, 0.1, "/ROOT")
        assert "SyntaxError: '(' was never closed" in short
        assert "calc/stats.py" in short
        assert "site-packages" not in short and "importlib" not in short
        assert "<frozen" not in short and "???" not in short


def test_output_it_cannot_read_keeps_head_and_tail() -> None:
    noise = "\n".join(f"line {i}" for i in range(200))
    short = shorten(noise, 2, 1.0, "/ROOT")
    assert short.startswith("exit 2 (1.0s)\nline 0\n")
    assert "line 199" in short and "line 100" not in short
    assert len(short) <= BUDGET


def test_everything_stays_within_the_budget() -> None:
    huge = recorded("pytest_many").replace("E   AssertionError", "E   " + "x" * 3000)
    assert len(shorten(huge, 1, 0.1, "/ROOT")) <= BUDGET


FIXTURE = Path(__file__).parent / "fixtures" / "coder" / "calc"


async def call(tool: Tool, ctx: ToolContext) -> str:
    return await cast(Awaitable[str], tool.fn(ctx))


def calc_copy(tmp_path: Path) -> Path:
    repo = tmp_path / "calc"
    shutil.copytree(FIXTURE, repo)
    return repo


def test_the_command_comes_from_the_agent_file_or_is_found(tmp_path: Path) -> None:
    assert find_test_command(tmp_path, "npm test") == ["npm", "test"]
    found = find_test_command(tmp_path, None)
    assert found[0] == sys.executable and found[1:3] in (["-m", "pytest"], ["-m", "unittest"])


async def test_a_failing_run_is_an_untrusted_error_and_the_log_gets_everything(
    tmp_path: Path,
) -> None:
    repo = calc_copy(tmp_path)
    events: list[tuple[str, JSONObject]] = []
    ctx = ToolContext(
        workspace=Workspace(repo), env=dict(os.environ), emit=lambda k, d: events.append((k, d))
    )
    tool = run_tests_with([sys.executable, "-m", "unittest"], repo)
    assert "Runs: python -m unittest" in tool.spec.description  # what the confirm shows
    with pytest.raises(OutsideError) as failed:
        await call(tool, ctx)
    assert str(failed.value).startswith("exit 1: 1 failed, 2 passed")
    assert str(repo) not in str(failed.value)  # paths are the repo's own
    kind, data = events[0]
    assert kind == "test_run" and data["exit_code"] == 1
    assert str(repo) in str(data["output"])  # the raw output, full paths and all

    stats = repo / "calc" / "stats.py"
    stats.write_text(stats.read_text().replace("count(values) - 1", "count(values)"))
    assert (await call(tool, ctx)).startswith("exit 0: 3 passed")


def test_finding_the_command_runs_nothing_in_the_repo(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    fake = tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    fake.parent.mkdir(parents=True)
    fake.write_text(f"#!/bin/sh\ntouch {marker}\n")
    fake.chmod(0o755)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x():\n    assert True\n")
    command = find_test_command(tmp_path, None)
    assert command[0] == str(fake) and "pytest" in command  # plain test functions: pytest
    assert not marker.exists()


def test_a_failure_with_counts_but_no_blocks_shows_the_end_of_the_output() -> None:
    raw = "...\n3 passed in 0.10s\nFAIL Required test coverage of 90% not reached."
    assert "Required test coverage" in shorten(raw, 1, 0.1, "/ROOT")


def test_exit_5_from_another_tool_keeps_its_output() -> None:
    assert "lint failed: real reason" in shorten("lint failed: real reason", 5, 0.1, "/ROOT")


def test_bad_test_commands_are_refused_in_the_agent_file() -> None:
    for bad in ["   ", "npm 'test"]:
        with pytest.raises(ValidationError):
            AgentConfig(name="x", test_command=bad)
