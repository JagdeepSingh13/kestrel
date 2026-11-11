"""Assemble a streamed `/chat/completions` reply (Server-Sent Events).

The backend sends one `data: {...}` line per piece of the reply, then `data: [DONE]`.
`StreamAssembler` turns each line into `StreamChunk`s as it arrives and, at the end,
builds the complete `ModelResponse`. No network here: the provider feeds it lines.
"""

import json

from kestrel.core.types import (
    BadBackendResponseError,
    JSONValue,
    Message,
    ModelResponse,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from kestrel.providers.think import ThinkSplitter
from kestrel.providers.wire import parse_arguments, parse_usage, reasoning_from

DONE_MARKER = "[DONE]"


class _ToolCallParts:
    """One tool call being received in pieces (OpenAI streams `arguments` as text)."""

    def __init__(self) -> None:
        self.id = ""
        self.name = ""
        self.arguments: list[str] = []


class StreamAssembler:
    """Feed raw SSE lines with `feed_line`; call `finish` once the stream ends."""

    def __init__(self) -> None:
        self.saw_done = False
        self._raw: list[str] = []
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._think = ThinkSplitter()
        self._finish_reason: str | None = None
        self._usage: Usage | None = None
        self._tool_calls: dict[int, _ToolCallParts] = {}

    def feed_line(self, line: str) -> list[StreamChunk]:
        """Process one line of the stream; return the chunks it produced (maybe none)."""
        if self.saw_done or not line.startswith("data:"):
            return []  # blank keep-alives, `: comments`, `event:` lines
        payload = line[len("data:") :].strip()
        if payload == DONE_MARKER:
            self.saw_done = True
            return []
        try:
            event: JSONValue = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BadBackendResponseError(f"stream line is not JSON: {payload[:200]}") from exc
        if not isinstance(event, dict):
            raise BadBackendResponseError("stream event is not a JSON object")
        if (error := event.get("error")) is not None:
            message = error.get("message") if isinstance(error, dict) else error
            raise BadBackendResponseError(f"backend reported an error: {message}")

        if (usage := parse_usage(event.get("usage"))) is not None:
            self._usage = usage
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return []  # e.g. the final usage-only event
        choice = choices[0]
        if not isinstance(choice, dict):
            return []
        if isinstance(reason := choice.get("finish_reason"), str):
            self._finish_reason = reason
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return []

        self._collect_tool_calls(delta.get("tool_calls"))
        chunks: list[StreamChunk] = []
        if reasoning := reasoning_from(delta):
            chunks += self._chunk(reasoning, "")
        text = delta.get("content")
        if isinstance(text, str) and text:
            self._raw.append(text)
            chunks += self._chunk(*self._think.feed(text))
        return chunks

    def _collect_tool_calls(self, raw: JSONValue) -> None:
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            parts = self._tool_calls.setdefault(
                index if isinstance(index, int) else 0, _ToolCallParts()
            )
            if isinstance(call_id := item.get("id"), str):
                parts.id = call_id
            function = item.get("function")
            if isinstance(function, dict):
                if isinstance(name := function.get("name"), str):
                    parts.name += name
                if isinstance(args := function.get("arguments"), str):
                    parts.arguments.append(args)

    def finish(self) -> StreamChunk:
        """End of stream: return the final chunk, whose `done` is the whole reply."""
        tail = self._chunk(*self._think.flush())
        reasoning = "".join(self._reasoning)
        content = "".join(self._content)
        response = ModelResponse(
            message=Message(
                role=Role.assistant,
                content=content or None,
                reasoning=reasoning or None,
                tool_calls=self._finish_tool_calls(),
            ),
            raw_text="".join(self._raw),
            finish_reason=self._finish_reason,
            usage=self._usage,
        )
        last = tail[0] if tail else StreamChunk()
        return last.model_copy(update={"done": response})

    def _finish_tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(self._tool_calls):
            parts = self._tool_calls[index]
            if not parts.name:
                raise BadBackendResponseError(f"streamed tool call {index} has no name")
            calls.append(
                ToolCall(
                    id=parts.id or f"call_{index}",
                    name=parts.name,
                    arguments=parse_arguments("".join(parts.arguments), parts.name),
                )
            )
        return calls

    def _chunk(self, reasoning: str, text: str) -> list[StreamChunk]:
        if reasoning:
            self._reasoning.append(reasoning)
        if text:
            self._content.append(text)
        if not reasoning and not text:
            return []
        return [StreamChunk(text=text, reasoning=reasoning)]
