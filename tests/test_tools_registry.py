"""@tool and ToolRegistry: functions become tools the model can call safely."""

import asyncio
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field

from kestrel.core.types import JSONObject, ToolCall
from kestrel.safety.paths import Workspace
from kestrel.tools.registry import ToolContext, ToolRegistry, tool


@tool()
def greet(name: Annotated[str, Field(description="Who to greet.")], times: int = 1) -> str:
    """Say hello.

    This second paragraph is for humans and is not sent to the model.
    """
    return " ".join([f"hello {name}"] * times)


@tool(permission="confirm", side_effects=True)
async def where(ctx: ToolContext, path: str) -> str:
    """Show where a path really points."""
    return ctx.workspace.relative(ctx.workspace.resolve(path))


@tool(timeout=0.05)
async def slow() -> str:
    """Never finishes in time."""
    await asyncio.sleep(5)
    return "late"


@tool()
def broken() -> str:
    """Always crashes."""
    raise RuntimeError("disk on fire")


@tool(max_output_chars=10)
def chatty() -> str:
    """Says too much."""
    return "x" * 25


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    ctx = ToolContext(workspace=Workspace(tmp_path), env={})
    return ToolRegistry([greet, where, slow, broken, chatty], ctx)


async def run(registry: ToolRegistry, name: str, arguments: JSONObject) -> tuple[str, bool]:
    result = await registry.execute(ToolCall(id="c1", name=name, arguments=arguments))
    assert result.call_id == "c1"
    return result.content, result.is_error


def test_spec_comes_from_name_docstring_and_type_hints() -> None:
    spec = greet.spec
    assert spec.name == "greet"
    assert spec.description == "Say hello."
    assert spec.parameters == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Who to greet."},
            "times": {"type": "integer", "default": 1},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert (spec.permission, spec.side_effects) == ("allow", False)


def test_context_parameter_is_hidden_from_the_model() -> None:
    assert list(where.spec.parameters["properties"]) == ["path"]  # type: ignore[arg-type]
    assert (where.spec.permission, where.spec.side_effects) == ("confirm", True)


async def test_sync_and_async_tools_run(registry: ToolRegistry) -> None:
    assert await run(registry, "greet", {"name": "Jag", "times": 2}) == (
        "hello Jag hello Jag",
        False,
    )
    assert await run(registry, "where", {"path": "a/../b.txt"}) == ("b.txt", False)


async def test_bad_arguments_are_explained(registry: ToolRegistry) -> None:
    content, is_error = await run(registry, "greet", {"times": "lots", "loud": True})
    assert is_error
    assert content.startswith("ERROR: bad arguments for greet:")
    assert "name: Field required" in content
    assert "times: Input should be a valid integer" in content
    assert "loud: unknown argument" in content


async def test_unknown_tool(registry: ToolRegistry) -> None:
    content, is_error = await run(registry, "delete_everything", {})
    assert is_error
    assert "Available tools: greet, where, slow, broken, chatty" in content


async def test_path_escape_becomes_an_error(registry: ToolRegistry) -> None:
    content, is_error = await run(registry, "where", {"path": "../../etc/passwd"})
    assert is_error
    assert "outside the workspace" in content


async def test_crash_timeout_and_size_cap(registry: ToolRegistry) -> None:
    assert await run(registry, "broken", {}) == (
        "ERROR: broken failed: RuntimeError: disk on fire",
        True,
    )
    content, is_error = await run(registry, "slow", {})
    assert is_error
    assert "took longer than 0.05s" in content
    content, is_error = await run(registry, "chatty", {})
    assert not is_error
    assert content.startswith("x" * 5) and "15 characters cut from the middle" in content


def test_tools_must_document_and_type_their_parameters() -> None:
    with pytest.raises(TypeError, match="docstring"):

        @tool()
        def undocumented(x: int) -> str:
            return ""

    with pytest.raises(TypeError, match="type hint"):

        @tool()
        def untyped(x) -> str:  # type: ignore[no-untyped-def]
            """Untyped."""
            return ""
