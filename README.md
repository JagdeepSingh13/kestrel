# Kestrel

[![CI](https://github.com/JagdeepSingh13/kestrel/actions/workflows/ci.yml/badge.svg)](https://github.com/JagdeepSingh13/kestrel/actions/workflows/ci.yml)

**The agent harness that makes small local models work.**

Kestrel is a local-first, model-agnostic agent harness for 3B to 14B models on an ordinary 16GB laptop. It plugs into any OpenAI-compatible backend (Ollama, LM Studio, llama.cpp, mlx-lm, vLLM), defines agents as YAML files, and fixes the tool-calling mistakes small models make.

> **Status: early development (v0.0.1).** Kestrel can find local backends and chat with a model in its own shell (`kestrel`), or one-shot (`kestrel ask`). No agent loop or tools yet. See the [roadmap](#roadmap).

## Planned features
- **Reliability:** repairs broken JSON, wrong tool names and bad arguments; detects loops; constrained decoding where the backend supports it.
- **Autotuning:** `kestrel tune` finds the best harness settings for your model.
- **Recipes:** turns a successful run into a fast, replayable recipe.
- **Safe by default:** per-tool permissions, workspace path jail, confirmation before side effects, web content treated as untrusted.
- **Inspectable:** every run writes a JSONL event log you can trace, replay and evaluate.
- **Coder agent (planned):** a built-in agent for software development on small local models: reads the repo, makes small edits, runs tests, fixes failures.

## How Kestrel works

This section explains every part of Kestrel in plain words. It grows with each milestone; parts marked *planned* don't exist yet.

### What is an agent harness?
- **Model** (Qwen, Gemma, Llama) is the engine. Kestrel doesn't build or run models; a backend like Ollama does.
- **Harness** (Kestrel) is the car around the engine: it sends the model a prompt, reads its answer, runs the tools it asks for, feeds the results back, and repeats until the task is done. It also catches the model's mistakes and keeps you safe.
- **Agent** is one trip setup: a YAML file naming the model, the prompt, the tools and the limits.
- **Tools** are attachments: files, shell, Python, web, or any MCP server.

Small models (3B to 14B) make predictable mistakes: broken JSON, wrong tool names, endless loops. Kestrel's job is to make them work anyway.

### The layers

```mermaid
flowchart TB
    FE["Frontends<br/>CLI now; web, Telegram later"]
    LOOP["Agent loop + context manager<br/>plan, act, observe; token budget"]
    REL["Reliability layer<br/>parse + repair tool calls, loop detection"]
    TOOLS["Tools + safety<br/>registry, MCP, permissions, path jail"]
    PROV["Provider layer<br/>HTTP client, discovery, model profiles"]
    BACK[("Local model backends<br/>Ollama, LM Studio, llama.cpp, mlx-lm, vLLM")]
    LOG[["Event log<br/>JSONL per run"]]

    FE --> LOOP
    LOOP --> REL
    LOOP --> TOOLS
    REL --> PROV
    PROV --> BACK
    LOOP -.-> LOG
    TOOLS -.-> LOG
    PROV -.-> LOG
```

Each layer only talks to the layers below it, so any piece can be swapped or tested alone.

| Layer | What it does, in plain words | Status |
|---|---|---|
| Frontends | Where you type. Today: the `kestrel` command and its shell. | shell, `doctor`, `ask` (M1, M1.5) |
| Agent loop + context | The repeat-until-done cycle; decides what fits in the model's small memory each turn. | *planned, M2* |
| Reliability | Turns messy model output into valid tool calls; stops loops. | *planned, M4* |
| Tools + safety | Runs tools; asks before risky actions; blocks files outside the workspace. | *planned, M3* |
| Providers | Talks HTTP to the model program on your machine. | done (M1); model profiles later |
| Event log | Writes every step to a file so runs can be inspected and replayed. | *planned, M2* |

### Talking to the model (M1)
Kestrel never runs a model itself. A **backend** (Ollama, LM Studio, llama.cpp, mlx-lm, vLLM) loads the model and listens on a local port. All of them speak the same HTTP format, called "OpenAI-compatible" (OpenAI invented it; local programs copied it). So one small client covers them all.

```mermaid
flowchart LR
    YOU["You"] --> K["Kestrel"]
    K -- "your conversation" --> B[("Backend on<br/>your machine")]
    B -- "reply, piece by piece" --> K
    K --> SCREEN["Answer on screen<br/>thinking kept apart"]
```

How a reply travels:
1. Kestrel turns your conversation into the backend's JSON format and sends it.
2. The reply streams back in small pieces, so words appear as soon as the model writes them.
3. Kestrel stitches the pieces into one complete reply, and keeps the model's thinking apart from its answer.
4. If the backend is not up yet, Kestrel retries the connection twice. A reply that already started is never silently repeated.
5. Every failure becomes a plain message ("Nothing is listening at ... Is the backend running?") instead of a crash.

Finding backends (`kestrel doctor`): Kestrel knocks on every known port at once and asks each one for its model list. Each backend comes back as running, not running, or answering wrongly (bad key, wrong address).

```mermaid
flowchart LR
    D["kestrel doctor"] --> P1["Ollama port"] & P2["LM Studio port"] & P3["llama.cpp / mlx-lm port"] & P4["vLLM port"]
    P1 & P2 & P3 & P4 --> T["Table: running?<br/>which models?<br/>warnings"]
```

Things worth knowing:
- **Thinking.** qwen3 thinks before answering. Kestrel keeps that apart from the answer and hides it unless you ask (`--show-thinking`, or `/think` in the shell). Add `/no_think` to a prompt to skip it; much faster on a laptop.
- **Ollama context.** On machines with under 24GB of GPU memory (all 16GB Macs), Ollama gives models only 4K tokens of memory. Kestrel needs 8K to 16K. Raise it in the Ollama app (Settings, Context length) or run `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`. `kestrel doctor` reminds you; `ollama ps` shows the current value.
- **REMOTE.** Any backend that is not on this machine is labelled `REMOTE`: your prompt leaves your computer.
- **Settings.** Optional file `~/.config/kestrel/config.toml`: default model, default backend, extra backends. API keys never go in the file, only the name of the environment variable holding them.

### The interactive shell (M1.5)
Type `kestrel` and you are inside Kestrel's own prompt. Plain text goes to the model; lines starting with `/` control the session.

```mermaid
flowchart TB
    READ["You type a line"] --> Q{"Starts with / ?"}
    Q -- yes --> CMD["Run the command<br/>/model /think /doctor ..."]
    Q -- no --> SEND["Send it with the<br/>conversation so far"]
    SEND --> REPLY["Reply streams in"]
    REPLY --> KEEP["Turn added to<br/>the conversation"]
    CMD --> READ
    KEEP --> READ
```

- The model "remembers" because every turn resends the whole conversation. `/clear` starts fresh.
- Ctrl-C stops a reply; that unfinished turn is dropped so the conversation stays clean. Ctrl-D or `/exit` leaves.
- `/model` lists the backend's models; `/model NAME` switches and keeps the conversation. Tab completes commands and model names. The up arrow recalls earlier lines.
- The conversation has to fit the model's memory (16K tokens). Ollama quietly forgets the oldest part when it does not, so the shell warns above 12K. Proper trimming comes with the agent loop (M2).

| Command | Does |
|---|---|
| `/help` | list commands |
| `/model [name]` | list models, or switch |
| `/provider [name]` | list backends, or switch |
| `/think [on\|off]` | show or hide the model's thinking |
| `/clear` | forget the conversation |
| `/doctor` | check which backends are running |
| `/exit` | leave (also `/quit`, Ctrl-D) |

### What exists today
- `kestrel`: the interactive shell (also `kestrel chat --model M`).
- `kestrel doctor`: finds running backends and their models.
- `kestrel ask "..." --model M [--provider P] [--show-thinking]`: one message, one streamed answer, for scripts.
- No agent, no tools yet: those start in M2 and M3.

### Diagrams
All diagrams live in [docs/diagrams/](docs/diagrams/) and render on GitHub.

| Diagram | Shows |
|---|---|
| [architecture.md](docs/diagrams/architecture.md) | The layers above, with code folders and milestones |
| [core-types.md](docs/diagrams/core-types.md) | Kestrel's message and error types |
| [providers.md](docs/diagrams/providers.md) | Provider structure, `ask` sequence, `doctor` flowchart, data before and after |
| [shell.md](docs/diagrams/shell.md) | Shell structure, how a line is handled, session data |

## Install (development)
Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/JagdeepSingh13/kestrel.git
cd kestrel
uv sync
uv run kestrel --version
uv run kestrel doctor
uv run kestrel          # interactive shell
```

For a real chat you need a backend with a model, e.g. [Ollama](https://ollama.com) and `ollama pull qwen3:8b`.

## Roadmap
| Milestone | Scope | Status |
|---|---|---|
| M0 | Scaffold, CI on macOS/Ubuntu/Windows | done |
| M1 | Providers, `kestrel doctor`, `kestrel ask` | done |
| M1.5 | Interactive shell (`kestrel`) | done |
| M2 | Agent loop, event log, `kestrel trace` | next |
| M3 | Agent YAML, file/shell/python tools, safety | |
| M4 | Reliability layer, web tools | |
| M5 | Interactive chat, MCP client | |
| M5.5 | Coder agent: repo map, small edits, run tests (tentative) | |
| M6+ | Memory, evals, autotuner, recipes | |

Full design: [docs/SPEC.md](docs/SPEC.md).

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## License
[Apache-2.0](LICENSE)
