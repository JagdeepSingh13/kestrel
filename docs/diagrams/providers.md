# Provider layer (M1)

The provider layer is the only code that talks to model backends. Everything above it speaks Kestrel's own types (`core/types.py`); wire formats (OpenAI JSON, SSE) exist only inside `providers/`.

## Structure

```mermaid
flowchart TB
    subgraph CLI["cli/app.py"]
        DOC["kestrel doctor"]
        ASK["kestrel ask"]
    end

    subgraph CORE["core/"]
        TYPES["types.py<br/>Message, ToolCall, ToolSpec,<br/>ModelResponse, StreamChunk,<br/>Provider protocol, errors"]
        CFG["config.py<br/>providers from config.toml"]
    end

    subgraph PROV["providers/"]
        OAI["openai_compat.py<br/>OpenAICompatProvider"]
        WIRE["wire.py<br/>Kestrel types to/from OpenAI JSON"]
        SSE["sse.py<br/>parse streamed events"]
        DISC["discovery.py<br/>probe known ports"]
        FAKE["fake.py<br/>FakeProvider (tests)"]
    end

    BACK[("Backend<br/>Ollama / LM Studio / llama.cpp / mlx-lm / vLLM")]

    ASK --> CFG
    ASK --> OAI
    DOC --> DISC
    OAI --> WIRE
    OAI --> SSE
    OAI -- "httpx" --> BACK
    DISC -- "httpx" --> BACK
    OAI -. implements .-> TYPES
    FAKE -. implements .-> TYPES
```

## `kestrel ask "hello"` (streaming)

```mermaid
sequenceDiagram
    actor You
    participant CLI as kestrel ask
    participant P as OpenAICompatProvider
    participant W as wire.py + sse.py
    participant O as Ollama :11434

    You->>CLI: kestrel ask "hello" --model qwen3:8b
    CLI->>P: stream([Message(user, "hello")], model)
    P->>W: to_wire(messages)
    W-->>P: {"model":..., "messages":[...], "stream":true}
    P->>O: POST /v1/chat/completions
    loop each SSE event
        O-->>P: data: {"choices":[{"delta":{"content":"Hi"}}]}
        P->>W: parse event
        W-->>P: StreamChunk(text="Hi")
        P-->>CLI: StreamChunk(text="Hi")
        CLI-->>You: prints "Hi" immediately
    end
    O-->>P: data: [DONE]
    P-->>CLI: StreamChunk(done=ModelResponse(...))
    CLI-->>You: newline + token usage
```

Retries: only when the connection fails **before** any byte arrives (backend not up yet). A generation that started is never retried silently.

## `kestrel doctor`

```mermaid
flowchart TB
    START["kestrel doctor"] --> LIST["Known backends<br/>+ providers from config.toml"]
    LIST --> PROBE["Probe all in parallel<br/>GET /v1/models, 1.5s timeout"]
    PROBE --> UP{"Responds?"}
    UP -- no --> DOWN["not running"]
    UP -- yes --> PORT{"Port 8080?"}
    PORT -- yes --> FP{"GET /props works?"}
    FP -- yes --> LCPP["llama.cpp"]
    FP -- no --> MLX["mlx-lm (probably)"]
    PORT -- no --> NAMED["named by its port"]
    LCPP --> MODELS["list models"]
    MLX --> MODELS
    NAMED --> MODELS
    MODELS --> OLL{"Ollama?"}
    OLL -- yes --> CTX["warn: default context 4K on 16GB;<br/>show how to raise it"]
    OLL -- no --> REMOTE
    CTX --> REMOTE{"Non-local URL?"}
    REMOTE -- yes --> LBL["label REMOTE:<br/>data leaves this machine"]
    REMOTE -- no --> TABLE["print table"]
    LBL --> TABLE
    DOWN --> TABLE
```

## Data, before and after

What Kestrel holds internally:

```python
[Message(role=Role.system, content="Be brief."),
 Message(role=Role.user, content="hello")]
```

What `wire.py` sends (`POST /v1/chat/completions`):

```json
{"model": "qwen3:8b",
 "messages": [{"role": "system", "content": "Be brief."},
              {"role": "user", "content": "hello"}],
 "temperature": 0.2,
 "stream": true,
 "stream_options": {"include_usage": true}}
```

What streams back (Server-Sent Events, one `data:` line per piece):

```
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}
data: {"choices":[{"index":0,"delta":{"content":" there!"}}]}
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: {"choices":[],"usage":{"prompt_tokens":14,"completion_tokens":3}}
data: [DONE]
```

What Kestrel turns it into:

```python
StreamChunk(text="Hi")
StreamChunk(text=" there!")
StreamChunk(done=ModelResponse(
    message=Message(role=Role.assistant, content="Hi there!"),
    raw_text="Hi there!", finish_reason="stop",
    usage=Usage(prompt_tokens=14, completion_tokens=3)))
```

Tool calls arrive in pieces too (`delta.tool_calls[i].function.arguments` is a string split across events); `sse.py` stitches them back into one `ToolCall` with parsed `arguments`.
