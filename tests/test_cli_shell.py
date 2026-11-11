"""Seam: the interactive shell, driven by scripted input lines and a FakeProvider."""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from kestrel.cli.prompt import SlashCompleter
from kestrel.cli.shell import CONTEXT_WARNING_TOKENS, Session, Shell
from kestrel.core.config import KestrelConfig, ProviderConfig
from kestrel.core.types import (
    BackendUnreachableError,
    JSONObject,
    Message,
    ModelResponse,
    Provider,
    Role,
    StreamChunk,
    ToolSpec,
    Usage,
)
from kestrel.providers.fake import FakeProvider

type Line = str | type[BaseException]


def make_shell(
    lines: list[Line], provider: Provider, model: str | None = "m"
) -> tuple[Shell, list[str]]:
    """A shell whose input is `lines` (an exception class is raised instead of read)."""
    opened: list[str] = []

    @asynccontextmanager
    async def open_provider(name: str, config: ProviderConfig) -> AsyncGenerator[Provider]:
        opened.append(name)
        yield provider

    def read_line(prompt: str) -> str:
        if not lines:
            raise EOFError
        item = lines.pop(0)
        if isinstance(item, str):
            return item
        raise item

    shell = Shell(
        KestrelConfig(),
        Session(provider="ollama", model=model),
        open_provider=open_provider,
        env={},
        read_line=read_line,
    )
    return shell, opened


def history_text(shell: Shell) -> list[str | None]:
    return [m.content for m in shell.session.history]


def test_conversation_is_remembered_across_turns() -> None:
    fake = FakeProvider([FakeProvider.text("Hi Jag"), FakeProvider.text("Jag")])
    shell, _ = make_shell(["my name is Jag", "what is my name?"], fake)
    shell.run()
    assert history_text(shell) == ["my name is Jag", "Hi Jag", "what is my name?", "Jag"]
    assert [m.content for m in fake.calls[1].messages] == history_text(shell)[:3]


def test_reasoning_is_not_kept_in_history(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider([FakeProvider.text("Hi", reasoning="greeting")])
    shell, _ = make_shell(["/think on", "hello"], fake)
    shell.run()
    assert "greeting" in capsys.readouterr().err  # shown with /think on
    assert shell.session.history[-1].reasoning is None


def test_clear_forgets_the_conversation() -> None:
    fake = FakeProvider([FakeProvider.text("a"), FakeProvider.text("b")])
    shell, _ = make_shell(["one", "/clear", "two"], fake)
    shell.run()
    assert history_text(shell) == ["two", "b"]


def test_model_switch_keeps_history_and_listing_shows_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeProvider([FakeProvider.text("a"), FakeProvider.text("b")], models=["m", "big"])
    shell, _ = make_shell(["one", "/model", "/model big", "two"], fake)
    shell.run()
    assert [c.model for c in fake.calls] == ["m", "big"]
    assert len(shell.session.history) == 4
    assert shell.model_names == ["m", "big"]
    assert "* m" in capsys.readouterr().err


def test_provider_switch_uses_the_new_backend() -> None:
    fake = FakeProvider([FakeProvider.text("a")])
    shell, opened = make_shell(["/provider lmstudio", "hi"], fake)
    shell.run()
    assert opened == ["lmstudio"]


def test_unknown_command_and_provider_are_explained(capsys: pytest.CaptureFixture[str]) -> None:
    shell, _ = make_shell(["/nope", "/provider nope"], FakeProvider())
    shell.run()
    err = capsys.readouterr().err
    assert "Unknown command /nope" in err
    assert "Known providers" in err
    assert shell.session.provider == "ollama"


def test_backend_error_drops_the_turn_and_shell_continues() -> None:
    fake = FakeProvider([BackendUnreachableError("http://x"), FakeProvider.text("ok")])
    shell, _ = make_shell(["first", "second"], fake)
    shell.run()
    assert history_text(shell) == ["second", "ok"]


class InterruptedProvider(FakeProvider):
    """Starts a reply, then the user presses Ctrl-C."""

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="Hel")
        raise KeyboardInterrupt


def test_ctrl_c_during_reply_drops_the_turn_and_shell_continues() -> None:
    shell, _ = make_shell(["hello", "/exit"], InterruptedProvider())
    shell.run()
    assert shell.session.history == []


def test_ctrl_c_at_prompt_is_ignored_and_ctrl_d_exits(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider([FakeProvider.text("a")])
    shell, _ = make_shell([KeyboardInterrupt, "hi", EOFError, "never read"], fake)
    shell.run()
    assert history_text(shell) == ["hi", "a"]
    assert "Bye" in capsys.readouterr().err


def test_exit_stops_reading() -> None:
    lines: list[Line] = ["/exit", "never sent"]
    shell, _ = make_shell(lines, FakeProvider())
    shell.run()
    assert lines == ["never sent"]


def test_no_model_asks_to_pick_one(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider()
    shell, _ = make_shell(["hi"], fake, model=None)
    shell.run()
    assert "No model chosen" in capsys.readouterr().err
    assert fake.calls == []


def test_long_conversation_warns(capsys: pytest.CaptureFixture[str]) -> None:
    reply = ModelResponse(
        message=Message(role=Role.assistant, content="ok"),
        raw_text="ok",
        usage=Usage(prompt_tokens=CONTEXT_WARNING_TOKENS + 1, completion_tokens=1),
    )
    shell, _ = make_shell(["hi"], FakeProvider([reply]))
    shell.run()
    assert "/clear starts fresh" in capsys.readouterr().err


def test_completer_suggests_commands_and_models() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    completer = SlashCompleter(lambda: ["qwen3:8b", "gemma3:4b"])

    def complete(text: str) -> list[str]:
        return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]

    assert complete("/mo") == ["/model"]
    assert complete("/model q") == ["qwen3:8b"]
    assert complete("hello") == []
