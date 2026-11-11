"""Demo toolbox: add and echo, with mistakes returned as ERROR results."""

from kestrel.core.types import JSONObject, ToolCall
from kestrel.tools.demo import DemoToolbox


async def run(name: str, arguments: JSONObject) -> tuple[str, bool]:
    result = await DemoToolbox().execute(ToolCall(id="c1", name=name, arguments=arguments))
    assert result.call_id == "c1"
    return result.content, result.is_error


async def test_add() -> None:
    assert await run("add", {"a": 17, "b": 25}) == ("42", False)
    assert await run("add", {"a": 1.5, "b": 1}) == ("2.5", False)
    assert await run("add", {"a": 2.0, "b": 2.0}) == ("4", False)


async def test_add_rejects_non_numbers() -> None:
    bad: list[JSONObject] = [{"a": "1", "b": 2}, {"a": True, "b": 2}, {"a": 1}]
    for args in bad:
        content, is_error = await run("add", args)
        assert is_error
        assert content.startswith("ERROR: add needs numbers")


async def test_echo() -> None:
    assert await run("echo", {"text": "hi"}) == ("hi", False)
    assert (await run("echo", {}))[1] is True


async def test_unknown_tool() -> None:
    content, is_error = await run("nope", {})
    assert is_error
    assert "Available tools: add, echo" in content
