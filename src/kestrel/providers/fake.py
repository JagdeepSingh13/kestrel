"""FakeProvider: scripted replies for tests. No network, no model (spec section 4.4)."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from kestrel.core.types import (
    Event,
    EventType,
    JSONObject,
    Message,
    ModelResponse,
    ProviderError,
    Role,
    StreamChunk,
    ToolCall,
    ToolSpec,
    Usage,
)

type Scripted = ModelResponse | ProviderError


@dataclass(frozen=True)
class FakeCall:
    """What the code under test sent to the provider, for assertions."""

    messages: list[Message]
    tools: list[ToolSpec]
    model: str
    temperature: float
    max_tokens: int | None
    json_schema: JSONObject | None
    think: bool | None = None


@dataclass
class FakeProvider:
    """Returns scripted replies in order; raises scripted errors.

    `stream()` replays a scripted reply word by word (reasoning first), then the
    final chunk with `done`, just like a real backend.
    """

    script: list[Scripted] = field(default_factory=list[Scripted])
    models: list[str] = field(default_factory=lambda: ["fake-model"])
    name: str = "fake"
    calls: list[FakeCall] = field(default_factory=list[FakeCall])

    @staticmethod
    def text(content: str, *, reasoning: str | None = None) -> ModelResponse:
        """Shortcut for a plain text reply."""
        return ModelResponse(
            message=Message(role=Role.assistant, content=content, reasoning=reasoning),
            raw_text=content,
            finish_reason="stop",
        )

    @staticmethod
    def tool_call(
        name: str,
        arguments: JSONObject | None = None,
        *,
        call_id: str = "c1",
        content: str | None = None,
        usage: Usage | None = None,
    ) -> ModelResponse:
        """Shortcut for a reply that asks to run one tool."""
        return ModelResponse(
            message=Message(
                role=Role.assistant,
                content=content,
                tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments or {})],
            ),
            raw_text=content or "",
            finish_reason="tool_calls",
            usage=usage,
        )

    @staticmethod
    def replay(events: Sequence[Event]) -> "FakeProvider":
        """A provider that gives back the model replies recorded in a run log, in order."""
        script: list[Scripted] = []
        for event in events:
            if event.type is EventType.model_response:
                script.append(_response_from_event(event.data))
        return FakeProvider(script)

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
        think: bool | None = None,
    ) -> ModelResponse:
        """Record the call and return the next scripted reply."""
        self._record(messages, tools, model, temperature, max_tokens, json_schema, think)
        return self._next()

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Record the call and replay the next scripted reply in pieces."""
        self._record(messages, tools, model, temperature, max_tokens, json_schema, think)
        response = self._next()
        if response.message.reasoning:
            yield StreamChunk(reasoning=response.message.reasoning)
        for word in _pieces(response.message.content or ""):
            yield StreamChunk(text=word)
        yield StreamChunk(done=response)

    async def list_models(self) -> list[str]:
        """The configured model names."""
        return list(self.models)

    def _record(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        model: str,
        temperature: float,
        max_tokens: int | None,
        json_schema: JSONObject | None,
        think: bool | None = None,
    ) -> None:
        self.calls.append(
            FakeCall(
                list(messages),
                list(tools or []),
                model,
                temperature,
                max_tokens,
                json_schema,
                think,
            )
        )

    def _next(self) -> ModelResponse:
        if not self.script:
            raise AssertionError("FakeProvider ran out of scripted replies")
        item = self.script.pop(0)
        if isinstance(item, ProviderError):
            raise item
        return item


def _pieces(text: str) -> list[str]:
    """Split text into word-sized pieces that join back to the exact original."""
    pieces: list[str] = []
    start = 0
    for i, char in enumerate(text):
        if char == " " and i > start:
            pieces.append(text[start:i])
            start = i
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _response_from_event(data: JSONObject) -> ModelResponse:
    """Rebuild a `ModelResponse` from a `model_response` event's data."""
    content, reasoning, raw = data.get("content"), data.get("reasoning"), data.get("raw_text")
    reason, usage = data.get("finish_reason"), data.get("usage")
    calls = data.get("tool_calls")
    return ModelResponse(
        message=Message(
            role=Role.assistant,
            content=content if isinstance(content, str) else None,
            reasoning=reasoning if isinstance(reasoning, str) else None,
            tool_calls=[ToolCall.model_validate(c) for c in calls]
            if isinstance(calls, list)
            else [],
        ),
        raw_text=raw if isinstance(raw, str) else "",
        finish_reason=reason if isinstance(reason, str) else None,
        usage=Usage.model_validate(usage) if isinstance(usage, dict) else None,
    )
