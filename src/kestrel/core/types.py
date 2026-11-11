"""Provider-neutral types shared by every layer.

Providers convert backend wire formats into these types; nothing above
`kestrel.providers` ever sees raw backend JSON (CLAUDE.md rules 5 and 6).
"""

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Any value that can appear in JSON. Used instead of `Any` for tool arguments and
# JSON Schemas, so pyright still checks what we do with them (spec gap 9).
type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None
type JSONObject = dict[str, JSONValue]

# OpenAI's limit for function names; also fits MCP's `server__tool` names.
TOOL_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class _Model(BaseModel):
    """Base for all core types: immutable, and unknown fields are an error."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Role(StrEnum):
    """Who wrote a message."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ToolCall(_Model):
    """A request from the model to run one tool.

    `name` is not validated against a pattern: models misspell tool names, and the
    reliability layer (M4) needs to see the misspelling to repair it.
    """

    id: str
    name: str
    arguments: JSONObject = Field(default_factory=dict[str, JSONValue])


class Message(_Model):
    """One message in a conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list[ToolCall])
    tool_call_id: str | None = None
    """For `role=tool`: which `ToolCall.id` this message answers."""
    name: str | None = None
    """For `role=tool`: the tool's name."""
    reasoning: str | None = None
    """Thinking text from reasoning models, kept apart from `content`."""

    @model_validator(mode="after")
    def _check_role_fields(self) -> Self:
        if self.role is Role.tool and not self.tool_call_id:
            raise ValueError("a tool message needs tool_call_id (the call it answers)")
        if self.tool_calls and self.role is not Role.assistant:
            raise ValueError("only assistant messages can contain tool_calls")
        return self


class ToolSpec(_Model):
    """A tool the model may call, described as a JSON Schema."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str
    parameters: JSONObject
    """JSON Schema for the arguments object."""
    permission: Literal["allow", "confirm", "deny"] = "allow"
    side_effects: bool = False


class Usage(_Model):
    """Token counts reported by the backend. Either may be missing."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelResponse(_Model):
    """One complete model reply, normalized."""

    message: Message
    raw_text: str
    """Exactly the text the backend returned, before any cleanup."""
    finish_reason: str | None = None
    usage: Usage | None = None


class StreamChunk(_Model):
    """One piece of a streamed reply.

    Pieces carry `text` and/or `reasoning`. The final chunk carries the assembled
    reply in `done`, so callers never stitch pieces themselves (spec gap 3).
    """

    text: str = ""
    reasoning: str = ""
    done: ModelResponse | None = None


class Provider(Protocol):
    """What the rest of Kestrel needs from a model backend."""

    name: str

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
    ) -> ModelResponse:
        """Send messages and wait for the whole reply."""
        ...

    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and yield the reply piece by piece; the last chunk has `done`."""
        ...

    async def list_models(self) -> list[str]:
        """Names of the models the backend can serve."""
        ...


# Errors. All subclass ProviderError so callers can catch one type.

BODY_PREVIEW_CHARS = 500


class ProviderError(Exception):
    """Base for every error raised by a provider."""


class BackendUnreachableError(ProviderError):
    """Nothing answered at the backend's address (after retries)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        super().__init__(f"Nothing is listening at {base_url}. Is the backend running?")


class BackendTimeoutError(ProviderError):
    """The backend did not respond in time."""

    def __init__(self, base_url: str, phase: str) -> None:
        self.base_url = base_url
        self.phase = phase
        super().__init__(f"{base_url} timed out while {phase}.")


class BackendHTTPError(ProviderError):
    """The backend answered with an HTTP error status."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        preview = body[:BODY_PREVIEW_CHARS]
        if len(body) > BODY_PREVIEW_CHARS:
            preview += "..."
        super().__init__(f"Backend returned HTTP {status}: {preview}")


class BadBackendResponseError(ProviderError):
    """The backend's reply was not in the expected format."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(f"Unexpected response from backend: {detail}")
