"""Seam: the `kestrel ask` command, with the real provider replaced by FakeProvider."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kestrel.cli import app as cli
from kestrel.cli import output
from kestrel.core.config import ProviderConfig
from kestrel.core.types import (
    BackendUnreachableError,
    Message,
    ModelResponse,
    Provider,
    Role,
    Usage,
)
from kestrel.providers.fake import FakeProvider

runner = CliRunner()


def setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake: FakeProvider, config: str = ""
) -> list[tuple[str, ProviderConfig]]:
    """Point Kestrel at a temp config and make every provider the given fake."""
    path = tmp_path / "config.toml"
    path.write_text(config, encoding="utf-8")
    monkeypatch.setenv("KESTREL_CONFIG", str(path))
    opened: list[tuple[str, ProviderConfig]] = []

    @asynccontextmanager
    async def fake_open(name: str, cfg: ProviderConfig) -> AsyncGenerator[Provider]:
        opened.append((name, cfg))
        yield fake

    monkeypatch.setattr(output, "open_provider", fake_open)
    return opened


def test_streams_answer_to_stdout_and_sends_the_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeProvider([FakeProvider.text("Hi there friend")])
    opened = setup(monkeypatch, tmp_path, fake)
    result = runner.invoke(cli.app, ["ask", "hello", "--model", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "Hi there friend\n"
    assert opened[0][0] == "ollama"  # default provider
    assert fake.calls[0].model == "qwen3:8b"
    assert fake.calls[0].messages == [Message(role=Role.user, content="hello")]


def test_thinking_hidden_by_default_and_shown_on_stderr_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reply = FakeProvider.text("Hi", reasoning="user greets me")
    setup(monkeypatch, tmp_path, FakeProvider([reply, reply]))
    hidden = runner.invoke(cli.app, ["ask", "hello", "-m", "m"])
    assert "user greets me" not in hidden.output
    shown = runner.invoke(cli.app, ["ask", "hello", "-m", "m", "--show-thinking"])
    assert "user greets me" in shown.stderr
    assert shown.stdout == "Hi\n"  # stdout stays answer-only


def test_uses_default_model_and_provider_from_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeProvider([FakeProvider.text("ok")])
    opened = setup(
        monkeypatch,
        tmp_path,
        fake,
        'default_provider = "lmstudio"\ndefault_model = "gemma3:4b"\n',
    )
    assert runner.invoke(cli.app, ["ask", "hello"]).exit_code == 0
    assert opened[0][0] == "lmstudio"
    assert fake.calls[0].model == "gemma3:4b"


def test_token_counts_go_to_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reply = ModelResponse(
        message=Message(role=Role.assistant, content="ok"),
        raw_text="ok",
        usage=Usage(prompt_tokens=9, completion_tokens=2),
    )
    setup(monkeypatch, tmp_path, FakeProvider([reply]))
    result = runner.invoke(cli.app, ["ask", "hello", "-m", "m"])
    assert "9 tokens in, 2 tokens out" in result.stderr


def test_no_model_anywhere_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    setup(monkeypatch, tmp_path, FakeProvider())
    result = runner.invoke(cli.app, ["ask", "hello"])
    assert result.exit_code == 2
    assert "No model given" in result.stderr


def test_unknown_provider_lists_known_ones(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(monkeypatch, tmp_path, FakeProvider())
    result = runner.invoke(cli.app, ["ask", "hello", "-m", "m", "-p", "nope"])
    assert result.exit_code == 2
    assert "Known providers" in result.stderr


def test_backend_down_gives_friendly_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(
        monkeypatch, tmp_path, FakeProvider([BackendUnreachableError("http://localhost:11434/v1")])
    )
    result = runner.invoke(cli.app, ["ask", "hello", "-m", "m"])
    assert result.exit_code == 1
    assert "Nothing is listening" in result.stderr
    assert "kestrel doctor" in result.stderr
    assert "Traceback" not in result.output


def test_remote_provider_is_labelled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup(
        monkeypatch,
        tmp_path,
        FakeProvider([FakeProvider.text("ok")]),
        '[providers.cloud]\nbase_url = "https://api.example.com/v1"\n',
    )
    result = runner.invoke(cli.app, ["ask", "hello", "-m", "m", "-p", "cloud"])
    assert "REMOTE" in result.stderr
    assert "REMOTE" not in result.stdout
