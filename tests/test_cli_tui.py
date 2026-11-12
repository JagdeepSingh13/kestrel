"""Seam: the shell with its bottom input box, driven by piped key presses."""

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from kestrel.agents.schema import AgentConfig
from kestrel.cli.shell import Session, Shell
from kestrel.cli.tui import MENU_GRACE_SECONDS, Tui
from kestrel.core.config import KestrelConfig, ProviderConfig
from kestrel.core.types import JSONObject, Message, Provider, Role, StreamChunk, ToolSpec
from kestrel.providers.fake import FakeProvider
from kestrel.safety.paths import Workspace


def run_tui(tmp_path: Path, fake: FakeProvider, keys: Callable[[PipeInput, Tui], None]) -> Tui:
    """Run the box in this thread while `keys` types into it from another."""

    @asynccontextmanager
    async def open_provider(name: str, config: ProviderConfig) -> AsyncGenerator[Provider]:
        yield fake

    env = {"KESTREL_DATA_DIR": str(tmp_path)}
    shell = Shell(
        KestrelConfig(),
        Session(provider="ollama", model="m", agent=AgentConfig(name="t", tools=["run_shell"])),
        open_provider=open_provider,
        env=env,
        workspace=Workspace(tmp_path),
    )
    failures: list[BaseException] = []
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        tui = Tui(shell, InMemoryHistory())

        def type_keys() -> None:
            try:
                wait_for(lambda: tui.app.is_running)
                keys(pipe, tui)
            except BaseException as exc:  # report it, and still let the box close
                failures.append(exc)
                pipe.send_text("\x1b")  # stop a stuck turn first
                time.sleep(0.2)
            pipe.send_text("/exit\r")

        typer = threading.Thread(target=type_keys, daemon=True)
        typer.start()
        tui.run()
        typer.join(timeout=5)
    if failures:
        raise failures[0]
    return tui


def wait_for(check: Callable[[], bool], seconds: float = 5) -> None:
    end = time.monotonic() + seconds
    while not check():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


def test_messages_typed_in_the_box_reach_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path))
    fake = FakeProvider([FakeProvider.text("Hi."), FakeProvider.text("Still here.")])

    def keys(pipe: PipeInput, tui: Tui) -> None:
        pipe.send_text("hello\r")
        pipe.send_text("again\r")  # typed while the first turn may still run: queued
        wait_for(lambda: len(tui.shell.session.history) == 4)

    tui = run_tui(tmp_path, fake, keys)
    texts = [m.content for m in tui.shell.session.history]
    assert texts == ["hello", "Hi.", "again", "Still here."]


def test_permission_menu_in_the_box(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path))
    fake = FakeProvider(
        [
            FakeProvider.tool_call("run_shell", {"command": "echo hi"}),
            FakeProvider.text("You said no."),
        ]
    )

    def keys(pipe: PipeInput, tui: Tui) -> None:
        pipe.send_text("run echo hi\r")
        wait_for(lambda: tui.state.menu is not None)
        pipe.send_text("\r")  # an Enter right away is ignored: too soon to be an answer
        time.sleep(MENU_GRACE_SECONDS + 0.1)
        assert tui.state.menu is not None
        pipe.send_text("3")  # No
        wait_for(lambda: len(tui.shell.session.history) == 4)

    tui = run_tui(tmp_path, fake, keys)
    tool = next(m for m in tui.shell.session.history if m.role is Role.tool)
    assert tool.content is not None and "was not run" in tool.content


class SlowProvider(FakeProvider):
    """Never finishes a reply; records each message list it was asked about."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.asked: list[str] = []

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[StreamChunk]:
        self.asked.append(messages[-1].content or "")
        self.started.set()
        await asyncio.sleep(30)
        yield StreamChunk(text="")  # never reached


def test_esc_stops_the_turn_drops_queued_lines_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path))
    slow = SlowProvider()

    def keys(pipe: PipeInput, tui: Tui) -> None:
        pipe.send_text("slow question\r")
        wait_for(slow.started.is_set)
        pipe.send_text("queued one\r")
        wait_for(lambda: tui.lines.qsize() == 1)
        pipe.send_text("\x1b")
        wait_for(lambda: not tui.state.busy and tui.lines.empty())

    tui = run_tui(tmp_path, slow, keys)
    assert slow.asked == ["slow question"]  # the queued line never ran
    assert tui.shell.session.history == []  # the stopped turn was rolled back


def test_leaving_skips_lines_still_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path))
    slow = SlowProvider()

    def keys(pipe: PipeInput, tui: Tui) -> None:
        pipe.send_text("slow question\r")
        wait_for(slow.started.is_set)
        pipe.send_text("never run\r")
        wait_for(lambda: tui.lines.qsize() == 1)
        loop = tui.app.loop
        assert loop is not None
        loop.call_soon_threadsafe(tui.app.exit)

    run_tui(tmp_path, slow, keys)
    assert slow.asked == ["slow question"]
