"""Convert Kestrel types to and from the OpenAI chat-completions JSON format.

Pure functions, no network. This is the only place that knows OpenAI's field names.
"""

import json
from collections.abc import Sequence
from typing import Literal

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

type ThinkStyle = Literal["reasoning_effort", "chat_template_kwargs", "none"]
"""How a backend is told not to think (see `providers.profiles`)."""

INVALID_ARGUMENTS_KEY = "invalid_arguments"
"""Wraps a broken call's text when it is sent back to the backend in history."""


def to_request_body(
    messages: Sequence[Message],
    tools: Sequence[ToolSpec] | None = None,
    *,
    model: str,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    json_schema: JSONObject | None = None,
    stream: bool = False,
    think: bool | None = None,
    think_style: ThinkStyle = "none",
) -> JSONObject:
    """Build the JSON body for `POST /chat/completions`. `think=False` is sent the way
    the backend understands (`think_style`); there is no standard field for it."""
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
    if think is False:
        if think_style == "reasoning_effort":  # Ollama, llama.cpp: maps to no thinking
            body["reasoning_effort"] = "none"
        elif think_style == "chat_template_kwargs":  # vLLM, mlx-lm: Qwen-style templates
            body["chat_template_kwargs"] = {"enable_thinking": False}
    if stream:
        # Ask for token counts at the end of the stream (spec gap 10).
        body["stream_options"] = {"include_usage": True}
    return body


def _arguments_to_wire(call: ToolCall) -> str:
    """Arguments as the JSON string OpenAI expects. A broken call's text is wrapped in
    valid JSON: backends reject invalid JSON in history (Ollama answers 400 "invalid tool
    call arguments", checked 2026-11-12), and the model still sees what it wrote."""
    if call.arguments_error is not None and call.arguments_text is not None:
        return json.dumps({INVALID_ARGUMENTS_KEY: call.arguments_text})
    return json.dumps(call.arguments)


def _message_to_wire(message: Message) -> JSONObject:
    # `reasoning` is deliberately not sent back: it is the model's scratchpad.
    wire: JSONObject = {"role": message.role.value, "content": message.content}
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                # OpenAI sends and expects arguments as a JSON *string*.
                "function": {"name": call.name, "arguments": _arguments_to_wire(call)},
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
    # Some backends omit ids; generate a stable one so results can be matched.
    call_id = _as_str(_get(raw, "id")) or f"call_{index}"
    arguments = _get(function, "arguments")
    if isinstance(arguments, dict):
        return ToolCall(id=call_id, name=name, arguments=arguments)
    if arguments is None:
        return ToolCall(id=call_id, name=name)  # no arguments sent: same as {}
    text = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return tool_call_from_text(call_id, name, text)


def tool_call_from_text(call_id: str, name: str, text: str) -> ToolCall:
    """A call whose arguments arrived as a JSON string. Text that is not a JSON object is
    kept, with the reason, for the repair pipeline (M4): a model's broken call must not
    end the run."""
    arguments, error = parse_arguments(text)
    return ToolCall(
        id=call_id, name=name, arguments=arguments, arguments_text=text, arguments_error=error
    )


def parse_arguments(text: str) -> tuple[JSONObject, str | None]:
    """The arguments object, or `{}` and why the text is not a JSON object (in plain
    words, because the model reads it)."""
    if not text.strip():
        return {}, None
    try:
        value: JSONValue = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"not valid JSON ({exc.msg} at character {exc.pos})"
    if not isinstance(value, dict):
        return {}, f"{_kind(value)}, not an object"
    return value, None


def _kind(value: JSONValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a true/false value"
    if isinstance(value, int | float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    return "a list"


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
