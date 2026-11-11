"""Seam: the `kestrel doctor` command, with discovery replaced by canned reports."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kestrel.cli import app as cli
from kestrel.core.config import ProviderConfig
from kestrel.providers.discovery import BackendReport, Status

runner = CliRunner()


def use_reports(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *reports: BackendReport) -> None:
    monkeypatch.setenv("KESTREL_CONFIG", str(tmp_path / "missing.toml"))  # built-in defaults

    async def fake_discover(
        providers: Mapping[str, ProviderConfig], **_: object
    ) -> list[BackendReport]:
        return list(reports)

    monkeypatch.setattr(cli, "discover", fake_discover)


def test_doctor_shows_running_backend_and_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_reports(
        monkeypatch,
        tmp_path,
        BackendReport(
            "ollama",
            "http://localhost:11434/v1",
            Status.up,
            "Ollama",
            False,
            models=["qwen3:8b"],
            warnings=["context is 4K"],
        ),
        BackendReport("vllm", "http://localhost:8000/v1", Status.down, "vLLM", False),
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output
    assert "context is 4K" in result.output
    assert "down" in result.output


def test_doctor_labels_remote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_reports(
        monkeypatch,
        tmp_path,
        BackendReport("cloud", "https://api.example.com/v1", Status.up, "x", True, models=["m"]),
    )
    assert "REMOTE" in runner.invoke(cli.app, ["doctor"]).output


def test_doctor_exits_1_when_nothing_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    use_reports(
        monkeypatch,
        tmp_path,
        BackendReport("ollama", "http://localhost:11434/v1", Status.down, "Ollama", False),
    )
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "brew install ollama" in result.output


def test_doctor_reports_bad_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "config.toml"
    bad.write_text("this is not toml = = =", encoding="utf-8")
    monkeypatch.setenv("KESTREL_CONFIG", str(bad))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 2
    assert "Config error" in result.output
