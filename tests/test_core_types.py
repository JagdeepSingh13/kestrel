"""Seam 1: core types. Validation rules, JSON round trip, error messages."""

import pytest
from pydantic import ValidationError

from kestrel.core.types import (
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnreachableError,
    BadBackendResponseError,
    Message,
    ModelResponse,
    ProviderError,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)


def test_tool_message_requires_tool_call_id() -> None:
    with pytest.raises(ValidationError, match="tool_call_id"):
        Message(role=Role.tool, content="42")


def test_tool_message_with_id_is_valid() -> None:
    msg = Message(role=Role.tool, content="42", tool_call_id="c1", name="add")
    assert msg.tool_call_id == "c1"


def test_only_assistant_may_have_tool_calls() -> None:
    call = ToolCall(id="c1", name="add", arguments={"a": 1})
    with pytest.raises(ValidationError, match="only assistant"):
        Message(role=Role.user, tool_calls=[call])
    assert Message(role=Role.assistant, tool_calls=[call]).tool_calls == [call]


def test_tool_call_arguments_must_be_an_object() -> None:
    with pytest.raises(ValidationError):
        ToolCall.model_validate({"id": "c1", "name": "add", "arguments": [1, 2]})


def test_tool_call_keeps_misspelled_names_for_repair() -> None:
    assert ToolCall(id="c1", name="read-file!").name == "read-file!"


@pytest.mark.parametrize("name", ["read_file", "fs__read_file", "web-search", "a" * 64])
def test_tool_spec_accepts_valid_names(name: str) -> None:
    assert ToolSpec(name=name, description="d", parameters={"type": "object"}).name == name


@pytest.mark.parametrize("name", ["", "read file", "read.file", "a" * 65])
def test_tool_spec_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValidationError):
        ToolSpec(name=name, description="d", parameters={"type": "object"})


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": "user", "content": "hi", "colour": "red"})


def test_response_survives_json_round_trip() -> None:
    # The event log stores responses as JSON and replays them later.
    response = ModelResponse(
        message=Message(
            role=Role.assistant,
            content=None,
            reasoning="need to add",
            tool_calls=[
                ToolCall(id="c1", name="add", arguments={"a": 1, "b": [2.5, None, {"x": True}]})
            ],
        ),
        raw_text='{"tool": "add"}',
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=None),
    )
    assert ModelResponse.model_validate_json(response.model_dump_json()) == response


def test_role_serializes_as_plain_string() -> None:
    assert Message(role=Role.user, content="hi").model_dump()["role"] == "user"


def test_error_messages_are_readable() -> None:
    assert "http://localhost:11434" in str(BackendUnreachableError("http://localhost:11434"))
    assert "reading the reply" in str(BackendTimeoutError("http://x", "reading the reply"))
    assert "missing choices" in str(BadBackendResponseError("missing choices"))


def test_http_error_previews_long_bodies() -> None:
    err = BackendHTTPError(404, "model 'nope' not found" + "x" * 1000)
    assert err.status == 404
    assert "HTTP 404" in str(err)
    assert "model 'nope' not found" in str(err)
    assert str(err).endswith("...")
    assert len(err.body) > 1000  # full body kept for the event log


@pytest.mark.parametrize(
    "err",
    [
        BackendUnreachableError("u"),
        BackendTimeoutError("u", "p"),
        BackendHTTPError(500, "b"),
        BadBackendResponseError("d"),
    ],
)
def test_all_errors_are_provider_errors(err: Exception) -> None:
    assert isinstance(err, ProviderError)
