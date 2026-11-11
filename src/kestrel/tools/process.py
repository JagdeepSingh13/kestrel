"""Tools that run programs: `run_shell` and `python_exec` (spec 8.3, M3 decisions).

Not a sandbox. What protects you instead:
- both default to `confirm`: you see the exact command or code and answer y/n/a;
- they run with the workspace folder as the current directory;
- a hard time limit kills the whole process group, and output is capped;
- environment variables named like *KEY*, *TOKEN*, *SECRET*, *PASSWORD* are removed,
  so a command the model writes cannot read your API keys.
A command can still name absolute paths outside the workspace; the path jail only
guards Kestrel's own file tools. That is why these tools ask first.
"""

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import Mapping
from typing import Annotated

from pydantic import Field

from kestrel.safety.redact import SECRET_NAME_PARTS
from kestrel.tools.registry import ToolContext, tool

MAX_OUTPUT_BYTES = 20_000
DEFAULT_SECONDS = 60
MAX_SECONDS = 120
TEMP_FOLDER = ".kestrel-tmp"


def safe_env(env: Mapping[str, str]) -> dict[str, str]:
    """`env` without variables whose names look like secrets."""
    return {
        name: value
        for name, value in env.items()
        if not any(part in name.upper() for part in SECRET_NAME_PARTS)
    }


@tool(
    permission="confirm",
    side_effects=True,
    timeout=MAX_SECONDS + 10,
    max_output_chars=MAX_OUTPUT_BYTES + 500,  # room for our own "cut" note
)
async def run_shell(
    ctx: ToolContext,
    command: Annotated[str, Field(description="Shell command to run in the workspace folder.")],
    seconds: Annotated[
        int, Field(ge=1, le=MAX_SECONDS, description="Time limit; it is stopped after this.")
    ] = DEFAULT_SECONDS,
) -> str:
    """Run a shell command in the workspace folder. Returns the exit code and output."""
    return await _run(command, None, ctx, seconds)


@tool(
    permission="confirm",
    side_effects=True,
    timeout=MAX_SECONDS + 10,
    max_output_chars=MAX_OUTPUT_BYTES + 500,  # room for our own "cut" note
)
async def python_exec(
    ctx: ToolContext,
    code: Annotated[str, Field(description="Python code to run. Use print() to show results.")],
    seconds: Annotated[
        int, Field(ge=1, le=MAX_SECONDS, description="Time limit; it is stopped after this.")
    ] = DEFAULT_SECONDS,
) -> str:
    """Run Python code in the workspace folder. Returns the exit code and printed output."""
    folder = ctx.workspace.root / TEMP_FOLDER
    folder.mkdir(exist_ok=True)
    script = folder / f"snippet-{uuid.uuid4().hex[:8]}.py"
    script.write_text(code, encoding="utf-8")
    try:
        # -I: isolated mode; ignores PYTHON* variables and the user's site-packages.
        return await _run(None, [sys.executable, "-I", str(script)], ctx, seconds)
    finally:
        script.unlink(missing_ok=True)


async def _run(command: str | None, argv: list[str] | None, ctx: ToolContext, seconds: int) -> str:
    """Start the process in its own group, collect capped output, kill it if too slow."""
    common: dict[str, object] = {
        "cwd": ctx.workspace.root,
        "env": safe_env(ctx.env),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        common["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        common["start_new_session"] = True  # own process group, so we can kill children too
    if command is not None:
        proc = await asyncio.create_subprocess_shell(command, **common)  # type: ignore[arg-type]
    else:
        assert argv is not None
        proc = await asyncio.create_subprocess_exec(*argv, **common)  # type: ignore[arg-type]

    assert proc.stdout is not None
    try:
        async with asyncio.timeout(seconds):
            output = b""
            while len(output) <= MAX_OUTPUT_BYTES:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break  # the process closed its output
                output += chunk
            if len(output) > MAX_OUTPUT_BYTES:
                _kill(proc)  # too much output: stop it rather than read forever
            await proc.wait()
    except TimeoutError:
        _kill(proc)
        await proc.wait()
        return f"ERROR: stopped after {seconds}s (time limit)"
    text = output[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    if len(output) > MAX_OUTPUT_BYTES:
        text += f"\n[... output cut at {MAX_OUTPUT_BYTES} bytes; the process was stopped]"
    return f"exit code {proc.returncode}\n{text}".rstrip()


def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and everything it started."""
    if proc.returncode is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError):  # it already exited
            os.killpg(proc.pid, signal.SIGKILL)
