"""Seam: the interactive shell, driven by scripted input lines and a FakeProvider."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from kestrel.agents.schema import AgentConfig
from kestrel.cli.prompt import SlashCompleter
from kestrel.cli.shell import Session, Shell
from kestrel.core.config import KestrelConfig, ProviderConfig
from kestrel.core.types import (
    BackendUnreachableError,
    JSONObject,
    Message,
    ModelResponse,
    Provider,
    Role,
    StreamChunk,
    ToolCall,
    ToolSpec,
)
from kestrel.providers.fake import FakeProvider
from kestrel.safety.paths import Workspace

type Line = str | type[BaseException]


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run logs and the workspace go to a temp folder, never the real home."""
    monkeypatch.setenv("KESTREL_DATA_DIR", str(tmp_path))
    return tmp_path


def make_shell(
    lines: list[Line],
    provider: Provider,
    model: str | None = "m",
    env: dict[str, str] | None = None,
) -> tuple[Shell, list[str]]:
    """A shell whose input is `lines` (an exception class is raised instead of read)."""
    import os

    env = env or {"KESTREL_DATA_DIR": os.environ["KESTREL_DATA_DIR"]}
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
        env=env,
        workspace=Workspace(Path(env["KESTREL_DATA_DIR"])),
        read_line=read_line,
    )
    return shell, opened


def history_text(shell: Shell) -> list[str | None]:
    return [m.content for m in shell.session.history]


def sent(fake: FakeProvider, call: int) -> list[str | None]:
    """What the model got on one call, without the system prompt."""
    return [m.content for m in fake.calls[call].messages if m.role is not Role.system]


def test_conversation_is_remembered_across_turns() -> None:
    fake = FakeProvider([FakeProvider.text("Hi Jag"), FakeProvider.text("Jag")])
    shell, _ = make_shell(["my name is Jag", "what is my name?"], fake)
    shell.run()
    assert history_text(shell) == ["my name is Jag", "Hi Jag", "what is my name?", "Jag"]
    assert sent(fake, 1) == history_text(shell)[:3]


def test_reasoning_is_not_kept_in_history(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider([FakeProvider.text("Hi", reasoning="greeting")])
    shell, _ = make_shell(["/think show", "hello"], fake)
    shell.run()
    assert "greeting" in capsys.readouterr().err  # shown with /think show
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
        think: bool | None = None,
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


def test_no_model_uses_the_first_one_the_backend_lists() -> None:
    fake = FakeProvider([FakeProvider.text("hi")], models=["qwen3:8b", "other"])
    shell, _ = make_shell(["hello"], fake, model=None)
    shell.run()
    assert [c.model for c in fake.calls] == ["qwen3:8b"]


def test_no_model_anywhere_asks_to_pick_one(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider(models=[])
    shell, _ = make_shell(["hi"], fake, model=None)
    shell.run()
    assert "No model chosen" in capsys.readouterr().err
    assert fake.calls == []


def test_long_conversation_is_trimmed_to_fit(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider([FakeProvider.text("x" * 2000) for _ in range(4)])
    shell, _ = make_shell(["one", "two", "three", "four"], fake)
    agent = shell.session.agent
    shell.session.agent = agent.model_copy(
        update={"context": agent.context.model_copy(update={"window": 2600})}
    )
    shell.run()
    assert len(shell.session.history) == 8  # the shell still remembers everything
    last_sent = sent(fake, -1)
    assert last_sent[-1] == "four"
    assert "one" not in last_sent  # but older turns were left out of the prompt
    assert "context trimmed" in capsys.readouterr().err


def test_completer_suggests_commands_and_models() -> None:
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document

    completer = SlashCompleter(lambda: ["qwen3:8b", "gemma3:4b"])

    def complete(text: str) -> list[str]:
        return [c.text for c in completer.get_completions(Document(text), CompleteEvent())]

    assert complete("/mo") == ["/model"]
    assert complete("/model q") == ["qwen3:8b"]
    assert complete("hello") == []


def test_the_agent_decides_to_use_tools_and_trace_shows_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "notes.txt").write_text("hi")
    fake = FakeProvider([FakeProvider.tool_call("list_dir", {}), FakeProvider.text("One file.")])
    shell, _ = make_shell(["What is here?", "/trace", "/trace list"], fake)
    shell.run()
    out = capsys.readouterr()
    assert "One file." in out.out  # the agent's answer
    assert "list_dir" in out.err  # tool calls shown live
    for text in ["run_start", "tool_call", "final", "run_end"]:
        assert text in out.out  # /trace timeline
    assert "What is here?" in out.out  # /trace list
    assert len(list((tmp_path / "runs").glob("*.jsonl"))) == 1
    roles = [m.role for m in shell.session.history]
    assert roles == [Role.user, Role.assistant, Role.tool, Role.assistant]


def test_tool_results_carry_over_to_the_next_message(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("secret plan")
    fake = FakeProvider(
        [
            FakeProvider.tool_call("read_file", {"path": "notes.txt"}),
            FakeProvider.text("It says secret plan."),
            FakeProvider.text("Two words."),
        ]
    )
    shell, _ = make_shell(["read notes.txt", "how many words is it?"], fake)
    shell.run()
    later = " ".join(t or "" for t in sent(fake, 2))
    assert "secret plan" in later  # the file read in turn 1 is still there in turn 2
    assert len(list((tmp_path / "runs").glob("*.jsonl"))) == 2  # one run per message


def test_run_command_is_gone_and_trace_needs_a_run(capsys: pytest.CaptureFixture[str]) -> None:
    shell, _ = make_shell(["/run x", "/trace"], FakeProvider())
    shell.run()
    err = capsys.readouterr().err
    assert "Unknown command /run" in err
    assert "No runs yet" in err


def test_agent_commands(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env = {"KESTREL_DATA_DIR": str(tmp_path), "KESTREL_CONFIG": str(tmp_path / "c.toml")}
    lines: list[Line] = [
        "/agents",
        "/agent counter",
        "/tools",
        "/workspace elsewhere",
        "/agent nope",
    ]
    shell, _ = make_shell(lines, FakeProvider(), env=env)
    shell.run()
    err = capsys.readouterr().err
    assert "default" in err and "counter" in err  # /agents
    assert shell.session.agent.name == "counter"
    assert "write_file" in err and "overwrite: confirm" in err  # /tools
    assert "Changing it is turned off" in err  # /workspace refuses to move
    assert "No agent named 'nope'" in err


def test_status_line_shows_model_agent_and_context_use() -> None:
    fake = FakeProvider([FakeProvider.text("x" * 3500)])
    shell, _ = make_shell(["hi"], fake)
    assert "context 0%" in shell.prompt()
    shell.run()
    assert "context 7% of 16K" in shell.prompt()  # system prompt, tools and the reply


def test_report_failure_saves_the_broken_step_and_trace_shows_its_prompt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = ToolCall(
        id="c1",
        name="list_dir",
        arguments_text="{'path': '.',}",
        arguments_error="not valid JSON",
    )
    fake = FakeProvider(
        [
            ModelResponse(message=Message(role=Role.assistant, tool_calls=[broken]), raw_text=""),
            FakeProvider.text("Gave up."),
        ]
    )
    shell, _ = make_shell(
        ["look around", "/report-failure", "/trace last --prompt 1", "/trace last --prompt 9"],
        fake,
    )
    shell.session.agent = AgentConfig(name="t", tools=["list_dir"])
    shell.run()
    out = capsys.readouterr()
    [report] = list((tmp_path / "reports").glob("*.json"))
    case = json.loads(report.read_text(encoding="utf-8"))
    assert case["source"]["step"] == 1  # the step with the broken call
    assert case["response"]["tool_calls"] == [
        {"name": "list_dir", "arguments_text": "{'path': '.',}"}
    ]
    assert [t["name"] for t in case["tools"]] == ["list_dir"]
    assert case["expect"]["kind"].startswith("TODO")
    assert "look around" in out.out  # the prompt of step 1, as sent
    assert "Step 9 has no logged prompt" in out.err


def test_thinking_is_off_unless_turned_on(capsys: pytest.CaptureFixture[str]) -> None:
    fake = FakeProvider([FakeProvider.text("a"), FakeProvider.text("b")])
    shell, _ = make_shell(["one", "/think on", "two", "/think"], fake)
    shell.run()
    assert [c.think for c in fake.calls] == [False, None]  # off, then left to the model
    assert "Thinking: on" in capsys.readouterr().err


def test_status_line_shows_the_window_the_backend_really_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kestrel.cli import agent_run
    from kestrel.providers.profiles import Profile

    async def small(name: str, base_url: str, model: str, known: Profile | None) -> Profile:
        return Profile(backend=base_url, model=model, kind="ollama", window=4096, probed_at="t")

    monkeypatch.setattr(agent_run, "probe_profile", small)
    shell, _ = make_shell(["hi"], FakeProvider([FakeProvider.text("hello")]))
    shell.run()
    assert "of 4K (backend loaded it small" in shell.prompt()


def test_only_links_the_user_typed_count_as_theirs() -> None:
    from kestrel.cli.agent_run import _user_urls  # pyright: ignore[reportPrivateUsage]
    from kestrel.core.context import ContextManager
    from kestrel.core.types import Message, Role
    from kestrel.reliability.nudge import NOTE

    talk = ContextManager(None)
    talk.add(Message(role=Role.user, content="see https://mine.example/a."))
    talk.add(Message(role=Role.assistant, content="try https://model.example/b"))
    talk.add(Message(role=Role.tool, content="https://page.example/c", tool_call_id="t"))
    talk.add(Message(role=Role.user, content=NOTE + " https://note.example/d"))
    assert _user_urls("and https://now.example/e", talk) == {
        "https://mine.example/a",
        "https://now.example/e",
    }
