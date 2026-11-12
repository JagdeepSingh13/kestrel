"""Seam 2: providers/wire.py. Kestrel types to OpenAI JSON and back."""

import json

import pytest

from kestrel.core.types import (
    BadBackendResponseError,
    JSONObject,
    JSONValue,
    Message,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from kestrel.providers.wire import from_response_body, to_request_body


def test_simple_messages_become_openai_messages() -> None:
    body = to_request_body(
        [Message(role=Role.system, content="Be brief."), Message(role=Role.user, content="hello")],
        model="qwen3:8b",
    )
    assert body == {
        "model": "qwen3:8b",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hello"},
        ],
        "temperature": 0.2,
        "stream": False,
    }


def test_tool_calls_and_results_use_openai_shapes() -> None:
    body = to_request_body(
        [
            Message(
                role=Role.assistant,
                reasoning="I should add",  # never sent back to the model
                tool_calls=[ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})],
            ),
            Message(role=Role.tool, content="3", tool_call_id="c1", name="add"),
        ],
        model="m",
    )
    assert body["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                }
            ],
        },
        {"role": "tool", "content": "3", "tool_call_id": "c1", "name": "add"},
    ]


def test_tools_streaming_and_limits() -> None:
    spec = ToolSpec(
        name="add",
        description="Add two numbers.",
        parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
        permission="confirm",  # Kestrel-only field, not sent
    )
    body = to_request_body(
        [Message(role=Role.user, content="1+1?")],
        [spec],
        model="m",
        temperature=0.0,
        max_tokens=256,
        stream=True,
    )
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
            },
        }
    ]
    assert body["temperature"] == 0.0
    assert body["max_tokens"] == 256
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_json_schema_becomes_response_format() -> None:
    schema: JSONObject = {"type": "object", "properties": {"final": {"type": "string"}}}
    body = to_request_body([Message(role=Role.user, content="x")], model="m", json_schema=schema)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": True},
    }


# Responses (from a real-shaped OpenAI chat-completions body).


def test_text_reply_becomes_model_response() -> None:
    response = from_response_body(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 14, "completion_tokens": 3, "total_tokens": 17},
        }
    )
    assert response.message == Message(role=Role.assistant, content="Hi there!")
    assert response.raw_text == "Hi there!"
    assert response.finish_reason == "stop"
    assert response.usage == Usage(prompt_tokens=14, completion_tokens=3)


def test_tool_call_arguments_string_is_parsed() -> None:
    response = from_response_body(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert response.message.content is None  # empty string normalized
    assert response.message.tool_calls == [
        ToolCall(
            id="call_1",
            name="add",
            arguments={"a": 1, "b": 2},
            arguments_text='{"a": 1, "b": 2}',  # kept verbatim for repair and reports
        )
    ]
    assert response.usage is None


def test_reasoning_field_is_kept_separately() -> None:
    for field in ("reasoning", "reasoning_content"):  # backends differ
        response = from_response_body(
            {"choices": [{"message": {"role": "assistant", "content": "4", field: "2+2"}}]}
        )
        assert response.message.reasoning == "2+2"
        assert response.message.content == "4"


def test_unparseable_tool_arguments_are_kept_raw_instead_of_failing() -> None:
    # A model's broken call must not end the run: M4's repair pipeline gets the raw text.
    body: JSONValue = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "add", "arguments": "{'a': 1,}"}}
                    ],
                }
            }
        ]
    }
    (call,) = from_response_body(body).message.tool_calls
    assert call.arguments == {}
    assert call.arguments_text == "{'a': 1,}"
    assert call.arguments_error is not None and "not valid JSON" in call.arguments_error


def test_broken_call_goes_back_to_the_model_as_it_wrote_it() -> None:
    broken = ToolCall(id="c", name="add", arguments_text="{'a': 1,}", arguments_error="x")
    body = to_request_body([Message(role=Role.assistant, tool_calls=[broken])], model="m")
    wire_call = body["messages"][0]["tool_calls"][0]  # type: ignore[index]
    # Backends reject invalid JSON in history, so it is wrapped; the model still sees it.
    sent = json.loads(wire_call["function"]["arguments"])  # type: ignore[index]
    assert sent == {"invalid_arguments": "{'a': 1,}"}


@pytest.mark.parametrize(
    ("raw", "error"),
    [("[1]", "a list, not an object"), ("null", "null, not an object"), ('"x"', "a string")],
)
def test_non_object_arguments_are_explained_in_plain_words(raw: str, error: str) -> None:
    body: JSONValue = {
        "choices": [
            {"message": {"tool_calls": [{"id": "c", "function": {"name": "f", "arguments": raw}}]}}
        ]
    }
    (call,) = from_response_body(body).message.tool_calls
    assert call.arguments_error is not None and error in call.arguments_error


def test_missing_arguments_mean_no_arguments() -> None:
    body: JSONValue = {"choices": [{"message": {"tool_calls": [{"function": {"name": "f"}}]}}]}
    (call,) = from_response_body(body).message.tool_calls
    assert (call.arguments, call.arguments_error) == ({}, None)


MALFORMED_BODIES: list[JSONValue] = [
    {},
    {"choices": []},
    {"choices": [{}]},
    [],
    "oops",
    {"choices": [{"message": "x"}]},
]


@pytest.mark.parametrize("body", MALFORMED_BODIES)
def test_malformed_bodies_raise_bad_backend_response(body: JSONValue) -> None:
    with pytest.raises(BadBackendResponseError):
        from_response_body(body)


def test_think_tags_in_content_become_reasoning() -> None:
    raw = "<think>\nGreeting.\n</think>\n\nHello!"
    response = from_response_body({"choices": [{"message": {"role": "assistant", "content": raw}}]})
    assert response.message.content == "Hello!"
    assert response.message.reasoning == "\nGreeting.\n"
    assert response.raw_text == raw  # the exact original is kept
