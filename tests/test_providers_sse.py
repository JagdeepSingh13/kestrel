"""Seam 3: providers/sse.py. Raw `data:` lines in, StreamChunks out."""

import json

import pytest

from kestrel.core.types import (
    BadBackendResponseError,
    JSONValue,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from kestrel.providers.sse import StreamAssembler


def data(payload: JSONValue) -> str:
    return "data: " + json.dumps(payload)


def delta(**fields: JSONValue) -> str:
    return data({"choices": [{"index": 0, "delta": dict(fields)}]})


def run(lines: list[str]) -> tuple[list[StreamChunk], StreamChunk]:
    assembler = StreamAssembler()
    chunks = [chunk for line in lines for chunk in assembler.feed_line(line)]
    return chunks, assembler.finish()


def test_text_pieces_stream_then_assemble() -> None:
    chunks, final = run(
        [
            delta(role="assistant", content="Hi"),
            "",
            delta(content=" there!"),
            data({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            data({"choices": [], "usage": {"prompt_tokens": 14, "completion_tokens": 3}}),
            "data: [DONE]",
        ]
    )
    assert [c.text for c in chunks] == ["Hi", " there!"]
    assert final.done is not None
    assert final.done.message == Message(role=Role.assistant, content="Hi there!")
    assert final.done.raw_text == "Hi there!"
    assert final.done.finish_reason == "stop"
    assert final.done.usage == Usage(prompt_tokens=14, completion_tokens=3)


def test_done_marker_is_recorded_and_later_lines_ignored() -> None:
    assembler = StreamAssembler()
    assembler.feed_line(delta(content="a"))
    assert not assembler.saw_done
    assembler.feed_line("data: [DONE]")
    assert assembler.saw_done
    assert assembler.feed_line(delta(content="late")) == []
    done = assembler.finish().done
    assert done is not None
    assert done.message.content == "a"


def test_comments_events_and_blank_lines_are_ignored() -> None:
    chunks, final = run(
        [
            ": keep-alive",
            "event: message",
            "",
            "data:" + json.dumps({"choices": [{"delta": {"content": "x"}}]}),
        ]
    )  # no space after "data:" is valid SSE too
    assert [c.text for c in chunks] == ["x"]
    assert final.done is not None
    assert final.done.usage is None  # backend sent none


def test_streamed_think_tags_become_reasoning_chunks() -> None:
    chunks, final = run(
        [delta(content="<thi"), delta(content="nk>plan"), delta(content="</think>\n\nOK")]
    )
    assert "".join(c.reasoning for c in chunks) == "plan"
    assert "".join(c.text for c in chunks) == "OK"
    assert final.done is not None
    assert final.done.message.reasoning == "plan"
    assert final.done.message.content == "OK"
    assert final.done.raw_text == "<think>plan</think>\n\nOK"


def tool_delta(index: int, **function: JSONValue) -> dict[str, JSONValue]:
    return {"index": index, "function": dict(function)}


def test_tool_call_split_across_events_is_stitched() -> None:
    first: JSONValue = {"index": 0, "id": "call_1", "type": "function", "function": {"name": "add"}}
    chunks, final = run(
        [
            delta(role="assistant", tool_calls=[first]),
            delta(tool_calls=[tool_delta(0, arguments='{"a": ')]),
            delta(tool_calls=[tool_delta(0, arguments='1, "b"')]),
            delta(tool_calls=[tool_delta(0, arguments=": 2}")]),
            data({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
    )
    assert chunks == []  # nothing to print while a tool call streams
    assert final.done is not None
    assert final.done.message.tool_calls == [
        ToolCall(id="call_1", name="add", arguments={"a": 1, "b": 2})
    ]
    assert final.done.finish_reason == "tool_calls"


def test_parallel_tool_calls_keep_their_order() -> None:
    _, final = run(
        [
            delta(tool_calls=[{"index": 0, "id": "a", "function": {"name": "echo"}}]),
            delta(tool_calls=[{"index": 1, "id": "b", "function": {"name": "add"}}]),
            delta(tool_calls=[tool_delta(1, arguments='{"a": 1, "b": 1}')]),
            delta(tool_calls=[tool_delta(0, arguments='{"text": "hi"}')]),
        ]
    )
    assert final.done is not None
    assert [(c.id, c.name, c.arguments) for c in final.done.message.tool_calls] == [
        ("a", "echo", {"text": "hi"}),
        ("b", "add", {"a": 1, "b": 1}),
    ]


def test_invalid_streamed_arguments_are_reported_at_the_end() -> None:
    assembler = StreamAssembler()
    assembler.feed_line(
        delta(tool_calls=[{"index": 0, "function": {"name": "add", "arguments": "{a:1"}}])
    )
    with pytest.raises(BadBackendResponseError, match=r"\{a:1"):
        assembler.finish()


def test_reasoning_field_streams_as_reasoning() -> None:
    chunks, final = run(
        [delta(reasoning="hmm "), delta(reasoning_content="ok"), delta(content="4")]
    )
    assert [(c.reasoning, c.text) for c in chunks] == [("hmm ", ""), ("ok", ""), ("", "4")]
    assert final.done is not None
    assert final.done.message.reasoning == "hmm ok"
    assert final.done.message.content == "4"


def test_non_json_line_raises() -> None:
    with pytest.raises(BadBackendResponseError, match="not JSON"):
        StreamAssembler().feed_line("data: {oops")


def test_error_event_raises_with_backend_message() -> None:
    with pytest.raises(BadBackendResponseError, match="model is loading"):
        StreamAssembler().feed_line(data({"error": {"message": "model is loading"}}))
