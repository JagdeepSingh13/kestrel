# Core types (M1)

`core/types.py` is the shared language of every layer. Providers convert wire formats into these types; nothing above `providers/` ever sees raw JSON from a backend.

```mermaid
classDiagram
    class Role {
        <<enum>>
        system
        user
        assistant
        tool
    }
    class Message {
        role: Role
        content: str | None
        tool_calls: list~ToolCall~
        tool_call_id: str | None
        name: str | None
        reasoning: str | None
    }
    class ToolCall {
        id: str
        name: str
        arguments: dict~str, JSONValue~
    }
    class ToolSpec {
        name: str
        description: str
        parameters: JSONSchema
        permission: allow | confirm | deny
        side_effects: bool
    }
    class ModelResponse {
        message: Message
        raw_text: str
        finish_reason: str | None
        usage: Usage | None
    }
    class Usage {
        prompt_tokens: int | None
        completion_tokens: int | None
    }
    class StreamChunk {
        text: str
        reasoning: str
        done: ModelResponse | None
    }
    class Provider {
        <<protocol>>
        name: str
        chat(messages, tools, model, ...) ModelResponse
        stream(messages, tools, model, ...) AsyncIterator~StreamChunk~
        list_models() list~str~
    }

    Message --> Role
    Message "1" o-- "*" ToolCall
    ModelResponse --> Message
    ModelResponse --> Usage
    StreamChunk --> ModelResponse : last chunk
    Provider ..> Message : takes
    Provider ..> ToolSpec : takes
    Provider ..> ModelResponse : returns
    Provider ..> StreamChunk : yields
```

Errors (all subclass `ProviderError`, so callers can catch one type):

| Error | When | What the user sees |
|---|---|---|
| `BackendUnreachableError` | connection refused / DNS fails (after retries) | "Nothing is listening at http://localhost:11434. Is Ollama running?" |
| `BackendTimeoutError` | no response within the timeout | which phase timed out |
| `BackendHTTPError` | 4xx/5xx | status + first 500 chars of the body (e.g. "model not found") |
| `BadBackendResponseError` | response isn't the expected JSON | what was missing |

`JSONValue` is a recursive type alias (str, int, float, bool, None, list, dict) used instead of `Any` for tool arguments and JSON Schemas.
