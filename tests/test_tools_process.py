"""run_shell, python_exec and ask_user, run for real (tiny local processes, no network)."""

import os
from pathlib import Path

import pytest

from kestrel.core.types import JSONObject, ToolCall
from kestrel.safety.paths import Workspace
from kestrel.tools.builtin import BUILTIN_TOOLS
from kestrel.tools.process import MAX_OUTPUT_BYTES, TEMP_FOLDER, safe_env
from kestrel.tools.registry import ToolContext, ToolRegistry


async def call(
    ws: Path, name: str, arguments: JSONObject, env: dict[str, str] | None = None
) -> str:
    async def answer(question: str) -> str:
        return f"you asked: {question}"

    context = ToolContext(workspace=Workspace(ws), env=env or dict(os.environ), ask=answer)
    registry = ToolRegistry(list(BUILTIN_TOOLS.values()), context)
    result = await registry.execute(ToolCall(id="c1", name=name, arguments=arguments))
    return result.content


def test_secret_looking_variables_are_removed() -> None:
    env = {"PATH": "/bin", "OPENAI_API_KEY": "x", "gh_token": "y", "DB_PASSWORD": "z"}
    assert safe_env(env) == {"PATH": "/bin"}


async def test_run_shell_runs_in_the_workspace(tmp_path: Path) -> None:
    assert await call(tmp_path, "run_shell", {"command": "echo hello"}) == "exit code 0\nhello"


async def test_python_exec_prints_runs_in_workspace_and_cleans_up(tmp_path: Path) -> None:
    expected = f"exit code 0\n{tmp_path.resolve()}"  # noqa: ASYNC240 (test setup)
    out = await call(tmp_path, "python_exec", {"code": "import os\nprint(os.getcwd())"})
    assert out == expected
    assert list((tmp_path / TEMP_FOLDER).iterdir()) == []  # the snippet file is deleted


async def test_commands_cannot_see_secrets(tmp_path: Path) -> None:
    env = {**os.environ, "MY_API_KEY": "s3cret", "PLAIN_VALUE": "ok"}
    code = "import os\nprint(os.environ.get('MY_API_KEY'), os.environ.get('PLAIN_VALUE'))"
    assert await call(tmp_path, "python_exec", {"code": code}, env) == "exit code 0\nNone ok"


async def test_exit_code_and_errors_are_reported(tmp_path: Path) -> None:
    out = await call(tmp_path, "python_exec", {"code": "raise SystemExit(3)"})
    assert out == "exit code 3"
    out = await call(tmp_path, "python_exec", {"code": "1/0"})
    assert out.startswith("exit code 1")
    assert "ZeroDivisionError" in out


async def test_time_limit_stops_the_process(tmp_path: Path) -> None:
    out = await call(tmp_path, "python_exec", {"code": "import time\ntime.sleep(30)", "seconds": 1})
    assert out == "ERROR: stopped after 1s (time limit)"


async def test_huge_output_is_cut(tmp_path: Path) -> None:
    out = await call(tmp_path, "python_exec", {"code": f"print('x' * {MAX_OUTPUT_BYTES * 3})"})
    assert f"output cut at {MAX_OUTPUT_BYTES} bytes" in out
    assert len(out) < MAX_OUTPUT_BYTES + 200


async def test_ask_user(tmp_path: Path) -> None:
    assert await call(tmp_path, "ask_user", {"question": "Which file?"}) == (
        "you asked: Which file?"
    )


@pytest.mark.parametrize("name", ["run_shell", "python_exec"])
def test_risky_tools_ask_first_by_default(name: str) -> None:
    spec = BUILTIN_TOOLS[name].spec
    assert (spec.permission, spec.side_effects) == ("confirm", True)
