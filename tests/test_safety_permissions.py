"""Permissions: allow, confirm (ask the user) or deny, per tool call."""

from kestrel.core.types import ConfirmAnswer, ToolCall, ToolSpec
from kestrel.safety.permissions import Permissions, loosened
from kestrel.tools.files import write_file

SHELL = ToolSpec(name="run_shell", description="d", parameters={}, permission="confirm")
READ = ToolSpec(name="read_file", description="d", parameters={})


class User:
    """Answers confirm prompts from a script and counts how often it was asked."""

    def __init__(self, *answers: ConfirmAnswer) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    async def __call__(self, call: ToolCall, spec: ToolSpec) -> ConfirmAnswer:
        self.asked.append(call.name)
        return self.answers.pop(0)


def call(name: str, **args: bool | str) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=dict(args))


async def test_allow_runs_without_asking() -> None:
    user = User()
    decision = await Permissions(user).check(call("read_file"), READ)
    assert decision.allowed
    assert user.asked == []


async def test_confirm_asks_and_respects_yes_and_no() -> None:
    user = User("yes", "no")
    gate = Permissions(user)
    assert (await gate.check(call("run_shell"), SHELL)).allowed
    no = await gate.check(call("run_shell"), SHELL)
    assert not no.allowed
    assert no.reason == "user said no"
    assert user.asked == ["run_shell", "run_shell"]


async def test_always_stops_asking_for_that_tool_only() -> None:
    user = User("always", "no")
    gate = Permissions(user)
    other = SHELL.model_copy(update={"name": "python_exec"})
    assert (await gate.check(call("run_shell"), SHELL)).allowed
    assert (await gate.check(call("run_shell"), SHELL)).reason == "allowed always for this run"
    assert not (await gate.check(call("python_exec"), other)).allowed
    assert user.asked == ["run_shell", "python_exec"]


async def test_agent_overrides_apply_and_deny_never_asks() -> None:
    user = User()
    gate = Permissions(user, {"read_file": "deny", "run_shell": "allow"})
    assert not (await gate.check(call("read_file"), READ)).allowed
    assert (await gate.check(call("run_shell"), SHELL)).allowed
    assert user.asked == []


async def test_overwriting_a_file_needs_confirmation() -> None:
    user = User("no")
    gate = Permissions(user)
    assert (await gate.check(call("write_file", path="new.md"), write_file.spec)).allowed
    replace = call("write_file", path="old.md", overwrite=True)
    assert not (await gate.check(replace, write_file.spec)).allowed
    assert user.asked == ["write_file"]


def test_loosened_lists_only_less_strict_overrides() -> None:
    defaults = {"run_shell": "confirm", "read_file": "allow", "write_file": "allow"}
    overrides = {"run_shell": "allow", "read_file": "deny", "write_file": "confirm"}
    assert loosened(defaults, overrides) == ["run_shell"]
