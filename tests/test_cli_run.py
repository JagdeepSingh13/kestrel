"""Seam: `kestrel run` and `kestrel trace`, with FakeProvider and a temp data dir."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kestrel.cli import app as cli
from kestrel.cli import output
from kestrel.core.config import ProviderConfig
from kestrel.core.events import read_events
from kestrel.core.types import EventType, Provider
from kestrel.providers.fake import FakeProvider

runner = CliRunner()


def setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeProvider) -> Path:
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TEST_API_KEY", "verysecretvalue")

    @asynccontextmanager
    async def fake_open(name: str, cfg: ProviderConfig) -> AsyncGenerator[Provider]:
        yield fake

    monkeypatch.setattr(output, "open_provider", fake_open)
    return tmp_path / "data" / "runs"


def adding_run() -> FakeProvider:
    return FakeProvider(
        [FakeProvider.tool_call("add", {"a": 17, "b": 25}), FakeProvider.text("It is 42.")]
    )


def test_run_answers_and_writes_a_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runs = setup(monkeypatch, tmp_path, adding_run())
    result = runner.invoke(cli.app, ["run", "add 17 and 25 verysecretvalue", "-m", "m"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "It is 42.\n"
    assert "tool add" in result.stderr
    assert "/trace" in result.stderr
    [log] = runs.glob("*.jsonl")
    assert [e.type for e in read_events(log)][-2:] == [EventType.final, EventType.run_end]
    assert "verysecretvalue" not in log.read_text(encoding="utf-8")


def test_run_that_hits_max_steps_exits_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakeProvider([FakeProvider.tool_call("echo", {"text": "x"}) for _ in range(3)])
    setup(monkeypatch, tmp_path, fake)
    result = runner.invoke(cli.app, ["run", "loop", "-m", "m", "--max-steps", "2"])
    assert result.exit_code == 1
    assert "max_steps after 2 steps" in result.stderr


def test_trace_shows_the_last_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(monkeypatch, tmp_path, adding_run())
    runner.invoke(cli.app, ["run", "add 17 and 25", "-m", "m"])
    result = runner.invoke(cli.app, ["trace"])
    assert result.exit_code == 0
    for text in ["run_start", "tool_call", "add", "42", "final", "It is 42.", "run_end"]:
        assert text in result.stdout
    listing = runner.invoke(cli.app, ["trace", "--list"])
    assert "add 17 and 25" in listing.stdout


def test_trace_unknown_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(monkeypatch, tmp_path, FakeProvider())
    result = runner.invoke(cli.app, ["trace", "nope"])
    assert result.exit_code == 2
    assert "No single run matches" in result.stderr
    assert "No runs yet" in result.stderr
