"""Convert Kestrel types to and from the OpenAI chat-completions JSON format.

Pure functions, no network. This is the only place that knows OpenAI's field names.
"""

import json
from collections.abc import Sequence

from kestrel.core.types import (
    BadBackendResponseError,
    JSONObject,
    JSONValue,
    Message,
    ModelResponse,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from kestrel.providers.think import split_think


def to_request_body(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None = None,
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    json_schema: JSONObject | None = None,
    stream: bool = False,
) -> JSONObject:
    """Build the JSON body for `POST /chat/completions`."""
    body: JSONObject = {
        "model": model,
        "messages": [_message_to_wire(m) for m in messages],
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        body["tools"] = [_tool_to_wire(t) for t in tools]
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if json_schema is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": json_schema, "strict": True},
        }
    if stream:
        # Ask for token counts at the end of the stream (spec gap 10).
        body["stream_options"] = {"include_usage": True}
    return body


def _message_to_wire(message: Message) -> JSONObject:
    # `reasoning` is deliberately not sent back: it is the model's scratchpad.
    wire: JSONObject = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # OpenAI sends and expects arguments as a JSON *string*.
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.role is Role.tool:
        wire["tool_call_id"] = message.tool_call_id
        if message.name is not None:
            wire["name"] = message.name
    return wire


def _tool_to_wire(tool: ToolSpec) -> JSONValue:
    # permission and side_effects are Kestrel-only; the model never sees them.
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


# Responses


def from_response_body(body: JSONValue) -> ModelResponse:
    """Parse a non-streamed `/chat/completions` response body.

    Raises `BadBackendResponseError` if the body is not the expected shape.
    """
    choices = _get(body, "choices")
    if not isinstance(choices, list) or not choices:
        raise BadBackendResponseError("no choices in response")
    choice = choices[0]
    message = _get(choice, "message")
    if not isinstance(message, dict):
        raise BadBackendResponseError("choice has no message object")

    raw_text = _as_str(message.get("content"))
    think, content = split_think(raw_text)
    tool_calls = [
        parse_tool_call(i, raw) for i, raw in enumerate(_as_list(message.get("tool_calls")))
    ]
    return ModelResponse(
        message=Message(
            role=Role.assistant,
            content=content or None,
            reasoning=reasoning_from(message) or think or None,
            tool_calls=tool_calls,
        ),
        raw_text=raw_text,
        finish_reason=_as_str(_get(choice, "finish_reason")) or None,
        usage=parse_usage(_get(body, "usage")),
    )


def parse_tool_call(index: int, raw: JSONValue) -> ToolCall:
    """Parse one OpenAI tool call. `arguments` arrives as a JSON string."""
    function = _get(raw, "function")
    name = _as_str(_get(function, "name"))
    if not name:
        raise BadBackendResponseError(f"tool call {index} has no function name")
    arguments = _get(function, "arguments")
    if isinstance(arguments, str):
        arguments = parse_arguments(arguments, name)
    if not isinstance(arguments, dict):
        raise BadBackendResponseError(f"tool call {name!r}: arguments are not an object")
    # Some backends omit ids; generate a stable one so results can be matched.
    call_id = _as_str(_get(raw, "id")) or f"call_{index}"
    return ToolCall(id=call_id, name=name, arguments=arguments)


def parse_arguments(text: str, tool_name: str) -> JSONObject:
    """Parse a tool call's arguments string into an object.

    M1 reports bad JSON with the raw text; M4's repair pipeline will fix it instead.
    """
    if not text.strip():
        return {}
    try:
        value: JSONValue = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BadBackendResponseError(
            f"tool call {tool_name!r} has invalid JSON arguments ({exc.msg}): {text}"
        ) from exc
    if not isinstance(value, dict):
        raise BadBackendResponseError(f"tool call {tool_name!r}: arguments are not an object")
    return value


def reasoning_from(obj: JSONObject) -> str | None:
    """Thinking text: backends use `reasoning` (Ollama, vLLM) or `reasoning_content`."""
    for field in ("reasoning", "reasoning_content"):
        if text := _as_str(obj.get(field)):
            return text
    return None


def parse_usage(raw: JSONValue) -> Usage | None:
    """Token counts, if the backend sent them."""
    if not isinstance(raw, dict):
        return None
    prompt, completion = raw.get("prompt_tokens"), raw.get("completion_tokens")
    return Usage(
        prompt_tokens=prompt if isinstance(prompt, int) else None,
        completion_tokens=completion if isinstance(completion, int) else None,
    )


# Small helpers for walking untyped JSON safely.


def _get(obj: JSONValue, key: str) -> JSONValue:
    return obj.get(key) if isinstance(obj, dict) else None


def _as_str(value: JSONValue) -> str:
    return value if isinstance(value, str) else ""


def _as_list(value: JSONValue) -> list[JSONValue]:
    return value if isinstance(value, list) else []
