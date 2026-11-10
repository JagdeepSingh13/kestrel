# Architecture

Layers, top to bottom. Each layer depends only on layers below it plus `core/types.py` (SPEC section 2). Every layer reports what it does to the event log.

```mermaid
flowchart TB
    FE["Frontends<br/><small>CLI now; web, Telegram later</small>"]
    LOOP["Agent loop + context manager<br/><small>plan, act, observe; token budget</small>"]
    REL["Reliability layer<br/><small>parse + repair tool calls, loop detection</small>"]
    TOOLS["Tools + safety<br/><small>registry, MCP, permissions, path jail</small>"]
    PROV["Provider layer<br/><small>HTTP client, discovery, model profiles</small>"]
    BACK[("Local model backends<br/><small>Ollama, LM Studio, llama.cpp, mlx-lm, vLLM</small>")]
    LOG[["Event log<br/><small>JSONL per run</small>"]]

    FE --> LOOP
    LOOP --> REL
    LOOP --> TOOLS
    REL --> PROV
    PROV --> BACK

    LOOP -.-> LOG
    TOOLS -.-> LOG
    PROV -.-> LOG
```

| Layer | Code | Milestone |
|---|---|---|
| Frontends | `cli/` | M0 (stub), M1+ |
| Agent loop + context | `core/loop.py`, `core/context.py` | M2 |
| Reliability | `reliability/` | M4 |
| Tools + safety | `tools/`, `safety/` | M3, M5 (MCP) |
| Providers | `providers/` | M1 |
| Event log | `core/events.py` | M2 |
