"""Seam: `kestrel run` and `kestrel trace`, with FakeProvider and a temp data dir."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kestrel.cli import agent_run, output
from kestrel.cli import app as cli
from kestrel.core.config import ProviderConfig
from kestrel.core.events import read_events
from kestrel.core.types import EventType, Provider
from kestrel.providers.fake import FakeProvider
from kestrel.safety.paths import Workspace

runner = CliRunner()


def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeProvider) -> Path:
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEST_API_KEY", "verysecretvalue")

    @asynccontextmanager
    async def fake_open(name: str, cfg: ProviderConfig) -> AsyncGenerator[Provider]:
        yield fake

    monkeypatch.setattr(output, "open_provider", fake_open)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(agent_run, "default_workspace", lambda: Workspace(workspace))
    return tmp_path / "data" / "runs"


def adding_run() -> FakeProvider:
    return FakeProvider([FakeProvider.tool_call("list_dir", {}), FakeProvider.text("It is 42.")])


def test_run_answers_and_writes_a_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs = setup(monkeypatch, tmp_path, adding_run())
    result = runner.invoke(cli.app, ["run", "add 17 and 25 verysecretvalue", "-m", "m"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "It is 42.\n"
    assert "⏺ list_dir" in result.stderr
    assert " · run " in result.stderr  # the run id, for /trace
    [log] = runs.glob("*.jsonl")
    assert [e.type for e in read_events(log)][-2:] == [EventType.final, EventType.run_end]
    assert "verysecretvalue" not in log.read_text(encoding="utf-8")


def test_run_that_hits_max_steps_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeProvider([FakeProvider.tool_call("list_dir", {}) for _ in range(3)])
    setup(monkeypatch, tmp_path, fake)
    agent = tmp_path / "short.yaml"
    agent.write_text("name: short\ntools: [list_dir]\nlimits:\n  max_steps: 2\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["run", "loop", "-m", "m", "--agent", str(agent)])
    assert result.exit_code == 1
    assert "hit the step limit · 2 steps" in result.stderr


def test_trace_shows_the_last_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(monkeypatch, tmp_path, adding_run())
    runner.invoke(cli.app, ["run", "add 17 and 25", "-m", "m"])
    result = runner.invoke(cli.app, ["trace"])
    assert result.exit_code == 0
    for text in ["run_start", "tool_call", "list_dir", "final", "It is 42.", "run_end"]:
        assert text in result.stdout
    listing = runner.invoke(cli.app, ["trace", "--list"])
    assert "add 17 and 25" in listing.stdout


def test_trace_unknown_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(monkeypatch, tmp_path, FakeProvider())
    result = runner.invoke(cli.app, ["trace", "nope"])
    assert result.exit_code == 2
    assert "No single run matches" in result.stderr
    assert "No runs yet" in result.stderr


def test_loosening_agent_needs_trust(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(monkeypatch, tmp_path, FakeProvider([FakeProvider.text("done")]))
    agent = tmp_path / "loose.yaml"
    agent.write_text(
        "name: loose\ntools: [run_shell]\npermissions:\n  run_shell: allow\n", encoding="utf-8"
    )
    refused = runner.invoke(cli.app, ["run", "x", "-m", "m", "--agent", str(agent)])
    assert refused.exit_code == 2
    assert "run_shell: confirm to allow" in " ".join(refused.stderr.split())
    trusted = runner.invoke(
        cli.app, ["run", "x", "-m", "m", "--agent", str(agent), "--trust-agent"]
    )
    assert trusted.exit_code == 0
