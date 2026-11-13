"""Seam: the shell's ladder (rungs, cloud approval, what leaves the machine)."""

import contextlib
from collections.abc import AsyncGenerator
from typing import Literal

import pytest

from kestrel.cli.ladder import CloudView, RungSpec, ShellLadder, parse_rung
from kestrel.core.config import KestrelConfig, ProviderConfig
from kestrel.core.context import NOTE_PREFIX
from kestrel.core.types import Message, Provider, Role, Rung, ToolCall
from kestrel.providers.fake import FakeProvider
from kestrel.providers.profiles import Profile, ProfileCache
from kestrel.safety.redact import Redactor

CLOUD = ProviderConfig(base_url="https://cloud.example/v1", api_key_env="CLOUD_API_KEY")
LOCAL = ProviderConfig(base_url="http://localhost:11434/v1")
SECRET = "sk-or-v1-abcdefabcdefabcdefabcdef"


def config() -> KestrelConfig:
    return KestrelConfig(providers={"ollama": LOCAL, "cloud": CLOUD})


def test_rung_names() -> None:
    assert parse_rung("qwen3:8b", config(), "ollama") == RungSpec("ollama", LOCAL, "qwen3:8b")
    cloud = parse_rung("cloud:google/gemma:free", config(), "ollama")
    assert (cloud.provider, cloud.model, cloud.label) == (
        "cloud",
        "google/gemma:free",
        "cloud:google/gemma:free",
    )


class Answers:
    def __init__(self, *answers: Literal["yes", "no", "always"]) -> None:
        self.answers = list(answers)
        self.asked = 0

    async def __call__(self, question: str, lines: list[str]) -> Literal["yes", "no", "always"]:
        self.asked += 1
        return self.answers.pop(0)


def ladder(
    stack: contextlib.AsyncExitStack,
    uppers: list[RungSpec],
    *,
    cloud: bool,
    approve: Answers,
    tmp: str,
) -> ShellLadder:
    @contextlib.asynccontextmanager
    async def open_provider(name: str, cfg: ProviderConfig) -> AsyncGenerator[Provider]:
        yield FakeProvider()

    async def load_profile(
        name: str, cfg: ProviderConfig, model: str, cache: ProfileCache
    ) -> Profile:
        return Profile(
            backend=cfg.base_url, model=model, native_tools=True, thinking=True, window=8192
        )

    from pathlib import Path

    return ShellLadder(
        Rung(FakeProvider(), "small", "small", 16384),
        uppers,
        stack=stack,
        open_provider=open_provider,
        load_profile=load_profile,
        cache=ProfileCache(Path(tmp) / "p.json"),
        env={"CLOUD_API_KEY": SECRET},
        allow_cloud=cloud,
        approve=approve,
        agent_window=16384,
    )


async def test_local_then_cloud_with_a_yes_then_back_down(tmp_path: object) -> None:
    approve = Answers("yes")
    async with contextlib.AsyncExitStack() as stack:
        rungs = [RungSpec("ollama", LOCAL, "big"), RungSpec("cloud", CLOUD, "m:free")]
        lad = ladder(stack, rungs, cloud=True, approve=approve, tmp=str(tmp_path))
        up = await lad.step_up("repeat", "add called 4 times")
        assert up is not None and up.name == "big" and up.think is True and up.window == 8192
        top = await lad.step_up("stall", "no progress")
        assert (
            top is not None and top.name == "cloud:m:free" and isinstance(top.provider, CloudView)
        )
        assert approve.asked == 1
        assert lad.after_step(ok=True) is None  # 2 step-ups used: it stays up
        assert await lad.step_up("repeat", "again") is None


@pytest.mark.parametrize(("allow", "answers", "asked"), [(False, (), 0), (True, ("no",), 1)])
async def test_no_cloud_without_permission_or_without_a_yes(
    tmp_path: object, allow: bool, answers: tuple[Literal["yes", "no", "always"], ...], asked: int
) -> None:
    approve = Answers(*answers)
    async with contextlib.AsyncExitStack() as stack:
        lad = ladder(
            stack,
            [RungSpec("cloud", CLOUD, "m:free")],
            cloud=allow,
            approve=approve,
            tmp=str(tmp_path),
        )
        assert await lad.step_up("repeat", "x") is None
        assert approve.asked == asked and lad.current().name == "small"


async def test_one_good_step_up_high_steps_back_down(tmp_path: object) -> None:
    async with contextlib.AsyncExitStack() as stack:
        lad = ladder(
            stack,
            [RungSpec("ollama", LOCAL, "big")],
            cloud=False,
            approve=Answers(),
            tmp=str(tmp_path),
        )
        await lad.step_up("repeat", "x")
        assert lad.after_step(ok=False) is None  # a failed step: stay up
        down = lad.after_step(ok=True)
        assert down is not None and down.name == "small"


def test_the_cloud_sees_only_the_stuck_step_without_secrets() -> None:
    call = ToolCall(id="c", name="read_file", arguments={"path": "a.txt"})
    history = [
        Message(role=Role.system, content="rules"),
        Message(role=Role.user, content="an older question"),
        Message(role=Role.assistant, content="an older answer"),
        Message(role=Role.user, content=f"the task, key {SECRET}"),
        Message(role=Role.assistant, tool_calls=[call]),
        Message(role=Role.tool, content="early result", tool_call_id="c"),
        Message(role=Role.assistant, tool_calls=[call]),
        Message(role=Role.tool, content="same result", tool_call_id="c"),
        Message(
            role=Role.user, content=f"{NOTE_PREFIX} a smaller model (small) got stuck here: repeat"
        ),
    ]
    sent = CloudView(FakeProvider(), Redactor({"CLOUD_API_KEY": SECRET})).payload(history)
    assert [m.content for m in sent] == [
        "rules",
        "the task, key [REDACTED]",
        None,
        "same result",
        f"{NOTE_PREFIX} a smaller model (small) got stuck here: repeat",
    ]


async def test_every_cloud_step_asks_again_unless_always(tmp_path: object) -> None:
    approve = Answers("yes", "always")
    async with contextlib.AsyncExitStack() as stack:
        lad = ladder(
            stack,
            [RungSpec("cloud", CLOUD, "m:free")],
            cloud=True,
            approve=approve,
            tmp=str(tmp_path),
        )
        assert await lad.step_up("repeat", "x") is not None
        assert lad.after_step(ok=True) is not None  # back down
        assert await lad.step_up("stall", "y") is not None
        assert approve.asked == 2  # a second cloud step is a second question


def test_without_a_handoff_only_the_latest_message_goes_to_the_cloud() -> None:
    history = [
        Message(role=Role.system, content="rules"),
        Message(role=Role.user, content="an older private question"),
        Message(role=Role.assistant, content="an older answer"),
        Message(role=Role.user, content="the new task"),
    ]
    sent = CloudView(FakeProvider(), Redactor({})).payload(history)
    assert [m.content for m in sent] == ["rules", "the new task"]
