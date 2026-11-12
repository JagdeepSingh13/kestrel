# Kestrel: Specification

Local-first, model-agnostic agent harness for small models.

Status: draft v0.8. Owner edits this file; Claude Code proposes changes via PR.

v0.2 (2026-11-11): applied the milestone changes agreed in the project handoff (section 1 identity + pillars, 5.5 constrained decoding, M4/M7 changes, new M7.5/M8.5/M9.5, backlog) and logged spec gaps in section 14.

v0.3 (2026-11-11): all spec gaps decided (section 14), open decisions tagged with their milestone; sections 8.3, 10, 11 updated to match.

v0.8 (2026-11-12): new M4.3, ideas from two harness sources (CMU "Better Harnesses, Smaller Models", arXiv 2607.08938; the Hookele Terminal-Bench write-up); M4.1 reshaped as one research tool.

v0.7 (2026-11-12): new M4.2, fixes from the owner's own chats (window, file tools, nudge, trace summary); tool roadmap (small core, MCP, model-proposed tools) noted.

v0.6 (2026-11-12): M4 planned as four pieces; web tools split into M4.1; text tool-call modes moved to the backlog.

v0.5 (2026-11-12): new M3.5, the shell becomes an agent conversation with a Claude Code style look.

v0.4 (2026-11-11): headline feature is the model ladder ("step up only when stuck", section 1); its basic version moves from M9.5 to a new M4.5.

---

## 1. Goals and non-goals

### Goals
1. Run useful multi-step agents on 3B to 14B local models on a 16GB laptop.
2. Plug into any OpenAI-compatible endpoint with zero code changes.
3. Define new agents with one YAML file.
4. Make tool calling reliable on weak models (fallback formats, repair, loop detection).
5. Be safe by default: permissions, path jail, untrusted-content handling.
6. Make every run inspectable and replayable (JSONL event log).
7. Measure reliability per model with a built-in eval suite (later milestone).
8. Use the smallest model that can do each step, stepping up to a bigger one only when stuck (model ladder, M4.5).

### Non-goals (for now)
- Training or fine-tuning models.
- Hosting models (we call backends, we do not run them).
- A polished GUI before the harness is solid.
- Multi-user / server deployment.
- Replacing Claude Code or Cursor for strong cloud models. (A coder agent for small local models is in scope: M5.5.)

### Positioning
Existing tools are chat-UI-first (Open WebUI, AnythingLLM) or assume strong cloud models (most agent frameworks). Kestrel is harness-first and small-model-first.

| Tool type | Examples | Gap |
|---|---|---|
| Chat apps with agents | Open WebUI, AnythingLLM | Chat-first, agents bolted on |
| Frameworks | LangChain, CrewAI, smolagents | Agents written in code, assume strong models |
| Coding harnesses | Claude Code, Codex CLI, Aider | One job, built for strong models |
| Visual builders | Dify, Flowise, n8n | Workflow-first, weak small-model reliability |

Kestrel is an **inner harness** (the runtime: loop, tools, context, safety), not an outer harness (AGENTS.md, linters, CI around a coding agent).

### Model, harness, context (framing, 2026-11-11)
An agent is three parts: the **model** (reasons and decides; the user picks it, a backend runs it), the **harness** (Kestrel: what the model sees, what it may do, what gets checked and recorded) and the **context** (the agent file and the workspace files). Harness parts, used as shared vocabulary in README, docs and later `/trace` output: context, control, safety gate, action, results back, run log, inspect and replay. See `GLOSSARY.md`.

Each part can improve by hand, with the user's OK, or automatically. Kestrel improves the harness (autotune, repair, model ladder) and the context (tests from runs, recipes, memory), always with the user's OK for lasting changes. Model weights are out of scope on a laptop, except a possible later export of good runs as fine-tuning data (mlx-lm LoRA). The improvement loop: run, feedback, look at the trace, keep it as a test (`/keep`), change the agent or settings, rerun all tests. Traces and tests are the base of it, which is why the run log must hold exact inputs.

### Identity and pillars
Identity: **the harness that makes small models work.**

Pitch: Kestrel runs each step on the smallest model that can do it and steps up to a bigger one only when stuck, sharing the conversation and what failed. It also fixes small models' tool-calling mistakes, tunes itself to your model, and turns successful runs into fast recipes. Agents that actually work on a 16GB laptop.

**Headline feature (2026-11-11): the model ladder, "step up only when stuck".** Fast and cheap by default (a 4B model), strong only when needed (8B, a bigger local model, then cloud only if allowed), private by default.
- **Triggers:** broken tool calls twice in a row after repair, a repeated call (loop), repeated tool errors with no change of approach, the conversation outgrowing the small model's window, or the user typing `/up`.
- **Shared context, three levels:** (1) one conversation in Kestrel's neutral format, so models can switch mid-run, and a bigger model with a bigger window receives more of the untrimmed history; (2) a handoff note telling the bigger model what was tried and how it failed (reasoning text is not passed between models); (3) after the hard step, the small model continues with that step in its history as a worked example.
- **Other splits:** planner and doer (big model plans once, small model executes), checker (big model reviews only the final answer), the user as a rung before cloud.
- **Cloud rung:** sends only the failing step plus a summary, secrets redacted, exact payload shown for approval, spending cap, per-agent "never cloud".
- **Learning:** every step-up is logged with its reason; task kinds that always escalate start on the bigger model next time; big-model solutions become examples for the small model.
- **Visible:** the trace marks which model did each step and why it stepped up.
- **Limit on 16GB:** a 4B and an 8B fit together, but each switch re-reads the whole conversation (seconds). Step up only when needed.

Supporting pillars:

1. **Reliability:** repair pipeline (JSON repair, fuzzy tool names, schema coercion, corrective re-prompts), loop detection, plus schema-driven constrained decoding (section 5.5). M4.
2. **Autotuning:** `kestrel tune --model X` runs the eval suite across harness configs (tool mode, prompt template, tool description style, max tools, temperature, context strategy), picks the best, saves it as the model profile. M7.5.
3. **Recipes (trace-to-recipe compilation):** after a successful run, compile its trace into a recipe. Deterministic steps replay without the model; the model is called only at real decision points. `kestrel recipe save last-run --name weekly-report`. M8.5.

Target proof chart for README: same task, same 4B model. Plain harness about 30% success; Kestrel with repair + tuned profile about 75%; Kestrel recipe replay about 98% and 8x faster.

Three sharp pillars beat fifteen shallow features. Do not: build an early chat UI, make agents Python classes, chase every feature.

**Flagship use case (tentative, 2026-11-11):** software development. Kestrel stays a general harness, but its first serious agent is a **coder agent** that reads a repo, makes small edits, runs tests and fixes failures on a local 8B model (M5.5). Direction to be revisited after the M4 demo (section 14).

---

## 2. Architecture

Layers, bottom to top:

1. **Backend** (external): Ollama, LM Studio, llama.cpp, mlx-lm, vLLM, cloud API.
2. **Provider layer** (`providers/`): HTTP client, wire format conversion, streaming, capability profile.
3. **Reliability layer** (`reliability/`): turns raw model output into validated tool calls, regardless of model quality.
4. **Agent loop** (`core/loop.py`): plan, act, observe, repeat, with budgets and stop conditions.
5. **Context manager** (`core/context.py`): builds each prompt within token budget.
6. **Tools + safety** (`tools/`, `safety/`): registry, execution, permissions, path jail.
7. **Event bus + log** (`core/events.py`): every action recorded.
8. **Frontends** (`cli/`, later web, Telegram): consume events, collect permission answers.

Rule: each layer depends only on layers below it plus `core/types.py`.

---

## 3. Core types (`core/types.py`)

Sketches. Final field names may change during M1 planning.

```python
class Role(StrEnum):
    system = "system"; user = "user"; assistant = "assistant"; tool = "tool"

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None      # for role=tool
    name: str | None = None              # tool name for role=tool

class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]           # JSON Schema
    permission: Literal["allow", "confirm", "deny"] = "allow"
    side_effects: bool = False

class ModelResponse(BaseModel):
    message: Message                      # normalized assistant message
    raw_text: str                         # exact text returned
    finish_reason: str | None
    usage: Usage | None

class Usage(BaseModel):
    prompt_tokens: int | None
    completion_tokens: int | None
```

Provider interface:

```python
class Provider(Protocol):
    name: str
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: dict | None = None,   # constrained output if backend supports
    ) -> ModelResponse: ...
    async def stream(self, ...) -> AsyncIterator[StreamChunk]: ...
    async def list_models(self) -> list[str]: ...
```

---

## 4. Provider layer

### 4.1 OpenAI-compatible provider (M1)
- `POST {base_url}/chat/completions`, `GET {base_url}/models`.
- Supports `tools` param when profile says native tool calling works.
- Supports `response_format` with JSON schema when available.
- SSE streaming.
- Timeouts, retries with backoff on connection errors only (never retry a completed generation silently).
- Optional `api_key` (local backends usually ignore it).

### 4.2 Backend discovery (`kestrel doctor`)
Probe these defaults, report which respond and which models they list:

| Backend | Default base URL |
|---|---|
| Ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |
| llama.cpp server | `http://localhost:8080/v1` |
| mlx-lm server | `http://localhost:8080/v1` (port clash with llama.cpp, user config wins) |
| vLLM | `http://localhost:8000/v1` |

### 4.3 Model profiles + capability probe
Built in M4 (`providers/profiles.py`). No test prompt: backend metadata only, and only for backends on this machine (a remote backend is never asked).

- Ollama: `/api/show` (capabilities `tools`, `thinking`; the model's maximum context), `/api/ps` (the window it is loaded with; an unloaded model is loaded first, without a prompt), `/api/tags` (digest).
- llama.cpp: `/props` (`n_ctx`); tools always work there.
- Others: nothing asked; unknown stays unknown.
- Cached in `<cache dir>/profiles.json` (one per backend and model; the digest refreshes it), asked once per session.
- Effect on a run: a model without tool support is refused for agents with tools; the context window is lowered to the loaded one (warned once per session, with the fix); thinking is switched off (`reasoning_effort: "none"` for Ollama and llama.cpp, `chat_template_kwargs.enable_thinking: false` for vLLM and mlx-lm) unless `/think on` or `--think`; mlx-lm gets `max_tokens` 4096.

### 4.4 FakeProvider (M1, tests)
Returns scripted `ModelResponse`s in order. Can also script malformed outputs to test the reliability layer. All unit tests use this.

---

## 5. Reliability layer

The core differentiator. Small models fail tool calls in predictable ways; handle each.

### 5.1 Tool-call modes
| Mode | When | How |
|---|---|---|
| `native` | profile.native_tools | Send `tools`, read `tool_calls` |
| `json` | native fails, json_mode works | System prompt instructs a single JSON object `{"tool": ..., "arguments": {...}}` or `{"final": ...}`, enforced by JSON schema output if available |
| `react` | weakest models | Text format: `Thought:` / `Action:` / `Action Input:` / `Final Answer:`, parsed with tolerant regex |

Mode selection: agent config `tool_mode: auto|native|json|react`. `auto` uses profile.

### 5.2 Parsing + repair pipeline
For every model response:
1. Extract candidate tool calls (native field, JSON block, ReAct block, or fenced ```json).
2. Repair JSON: strip trailing commas, fix single quotes, close brackets, unwrap double-encoded strings.
3. Validate tool name exists. Fuzzy-match near misses (`read-file` to `read_file`) only if edit distance is 1 to 2 and unique.
4. Validate arguments against the tool's JSON schema (pydantic). Coerce simple types (`"5"` to `5`).
5. On failure: return a short corrective message to the model as a tool/user message ("Invalid call: X. Expected schema: Y. Try again.") and count it against `max_repairs` (default 2 per step).
6. Emit `repair` events for every fix so evals can measure how often each model needs help.

### 5.3 Loop + stall detection
- Same tool + same args 3 times in a row: inject a "you are repeating yourself, change approach or give final answer" message.
- 5 times: stop run with `stopped_reason="loop"`.
- No progress (no new tool results) for N steps: same treatment.

### 5.4 Tool pruning (M6)
If an agent has more than `max_tools_in_prompt` (default 8) tools, embed tool descriptions once and include only top-k relevant tools per step based on the latest user goal + last observation.

### 5.5 Constrained decoding (M4)
Prevent bad calls instead of repairing them. Per step, generate a grammar from the tool schemas so the model cannot produce an unknown tool name or a wrong argument type.
- llama.cpp server: GBNF grammar (`grammar` request field).
- Backends with JSON schema output (`response_format`, Ollama `format`): a JSON schema that is a `oneOf` over `{"tool": <const name>, "arguments": <tool schema>}` plus `{"final": string}`.
- No support: fall back to the repair pipeline (5.2). Capability probe (4.3) records which applies.
- Repair pipeline still runs on constrained output (models can still be semantically wrong).

### 5.6 Failure corpus (M4)
- `tests/fixtures/bad_outputs/`: real malformed outputs, one file per case, with expected parse result.
- Shell `/report-failure [run] [--step N]`: packages the failed step(s) from a run log, redacted via `safety/redact.py`, into a fixture-shaped file for the user to review. Opt-in, local only; never uploads. Sharing upstream is a manual user action.

---

## 6. Agent loop (`core/loop.py`)

```
run(agent, user_input):
    emit run_start
    ctx = ContextManager(agent, profile)
    ctx.add_user(user_input)
    for step in 1..agent.max_steps:
        prompt = ctx.build()                         # within token budget
        resp = provider.chat(prompt, tools_for_mode) # emit model_request/model_response
        parsed = reliability.parse(resp, mode)       # may emit repair events
        if parsed.final:
            emit final; return
        if parsed.error and repairs_left:
            ctx.add_correction(parsed.error); continue
        for call in parsed.tool_calls:
            decision = permissions.check(agent, call) # may ask frontend; emit permission
            result = tools.execute(call) if allowed else denied_result
            ctx.add_tool_result(call, result)         # emit tool_call/tool_result
        loop_detector.observe(parsed.tool_calls)
    emit run_end(stopped_reason="max_steps")
```

Budgets per run: `max_steps` (default 15), `max_tokens_total`, `max_wall_seconds`. Any budget hit ends the run with a clear `stopped_reason` and a partial summary.

Cancel: Ctrl+C in CLI cancels the current step cleanly and logs `run_end(stopped_reason="cancelled")`.

---

## 7. Context manager (`core/context.py`)

Small windows make this critical.

Prompt assembly order:
1. System prompt (agent's prompt + tool-mode instructions + rules about untrusted content).
2. Tool specs (native mode) or tool descriptions in system prompt (json/react modes).
3. Pinned items (user goal, key facts).
4. Recent history.

Budget rules:
- Token counting: approximate (chars / 3.5) in v0.1; pluggable tokenizer later.
- Reserve `response_reserve` tokens (default 1024).
- Tool results longer than `max_tool_result_tokens` (default 1500) are truncated with a marker and the full result stored in the event log, retrievable via a `read_tool_output(id, offset)` tool (M4).
- When history overflows: drop oldest tool results first, then oldest turns. M6 adds LLM summarization of dropped turns.
- Never drop: system prompt, the original user goal, the most recent assistant message and its tool results.

---

## 8. Tools

### 8.1 Definition
```python
@tool(permission="confirm", side_effects=True)
async def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write text to a file inside the workspace."""
```
Decorator builds `ToolSpec` from signature + docstring (pydantic model from type hints). Sync and async functions both supported.

### 8.2 Execution contract
- Every tool returns a string (or object serialized to string) for the model.
- Errors return `ERROR: <message>` as the result, never crash the loop. A tool signals one by raising `ToolError`, so the result is marked `is_error` (the watchdog counts it, and it is not wrapped as untrusted).
- Timeouts per tool (default 30s, shell/python 60s).
- Output cap (default 20k chars before context truncation).

### 8.3 Built-in tools
| Tool | Milestone | Default permission | Notes |
|---|---|---|---|
| `read_file` | M3 | allow | jailed, size cap, text only |
| `list_dir` | M3 | allow | jailed, depth limit |
| `write_file` | M3 | allow (new file), confirm (overwrite) | jailed, no overwrite unless flag (gap 5); overwrite needs a fresh `read_file` first (M4.2); its "exists" error points to append_file and edit_file |
| `search_files` | M4.2 | allow | literal, case-insensitive; `path:line: text`, 50 matches, skips binary, big files, symlinks and the usual noise folders |
| `append_file` | M4.2 | allow | adds at the end on a new line, creates the file if missing; shows the last lines |
| `edit_file` | M4.2 | confirm | needs a fresh `read_file` of the file first (checked before asking); replace one exact `old_text`; falls back to whole lines with `12| ` prefixes and trailing spaces ignored, only when that matches one place; shows the lines around the change |
| `run_shell` | M3 | confirm | cwd = workspace, timeout, output cap |
| `python_exec` | M3 | confirm | subprocess with `-I`, timeout, temp dir in workspace |
| `web_search` | M4 | allow | pluggable backend: SearXNG URL in config, else DuckDuckGo via small library (ask before adding dep) |
| `fetch_url` | M4 | allow | HTML to readable text/markdown, result wrapped as untrusted |
| `read_tool_output` | M4 | allow | page through truncated results |
| `ask_user` | M3 | allow | lets agent ask a clarifying question via frontend |

### 8.4 MCP (M5)
- Config lists MCP servers (stdio command or HTTP URL).
- On agent start, connect, list tools, convert to `ToolSpec` with namespaced names (`server__tool`).
- Permission for MCP tools defaults to `confirm` unless overridden in agent YAML.

---

## 9. Milestones

Each milestone ends with: passing tests, ruff + pyright clean, README section updated, demo command that works against a real local model.

### M0: Scaffold
- `uv init`, src layout, pyproject, ruff/pyright/pytest config, GitHub Actions CI (lint, type, test on macOS + Ubuntu + Windows), Apache-2.0 LICENSE, README stub, CONTRIBUTING stub.
- **Done when:** `uv run pytest` passes with one trivial test, CI green.

### M1: Types + providers
- `core/types.py`, `providers/openai_compat.py`, `providers/fake.py`, `providers/discovery.py`.
- `kestrel doctor` lists reachable backends and models.
- `kestrel ask "hello" --model qwen3:8b` does a single non-agent chat call, streams to terminal.
- **Done when:** unit tests cover request building, response parsing, streaming, errors (with mocked httpx). Live test passes against Ollama.
- **Status: done (v0.0.1).** Verified against Ollama 0.40 + qwen3:8b: reasoning arrives in the `reasoning` field (content is `""` while thinking); recorded in `tests/fixtures/ollama/`. `ask` writes the answer to stdout and everything else to stderr. Exit codes: 0 ok, 1 backend error / no backend (doctor), 2 usage or config error, 130 Ctrl-C.

### M1.5: Interactive shell
- `kestrel` (no args, in a terminal) and `kestrel chat [--model] [--provider]` open a prompt_toolkit shell. Plain text is sent with the in-memory conversation; `/help /model /provider /think /clear /doctor /exit`.
- Ctrl-C stops a reply and drops that turn; backend errors drop the turn and keep the shell. Input history in `<data dir>/shell_history` (gap 4 paths). Warns above 12K prompt tokens until M2 trimming.
- **Status: done.** Dependency added: `prompt_toolkit` (approved by owner).

### M2: Agent loop + events (no real tools yet)
- `core/loop.py`, `core/events.py` (in-process bus + JSONL writer to `~/.local/share/kestrel/runs/<run_id>.jsonl`), `core/context.py` (basic budget).
- Native tool mode only. Test tools: `add(a, b)`, `echo(text)`.
- `kestrel trace <run_id>` pretty-prints a run.
- **Done when:** FakeProvider-driven tests cover: final answer, single tool call, multi-step, max_steps stop, tool error returned to model.
- **Decisions (2026-11-11):** the loop streams; live pieces are `model_delta` events shown on screen but never written to the log. `kestrel run "task"` ships in M2 with only the `add`/`echo` demo tools. The shell uses the context manager to trim long chats (shell chats are not logged as runs). Run ids are `YYYYMMDD-HHMMSS-xxxx`; `kestrel trace last` opens the newest. Context window defaults to 16384 (`--context`), moving to agent YAML in M3 and profiles in M4. Reasoning is logged in full, never sent back to the model.
- **Status: done (v0.0.2).** Live run on qwen3:8b: answered in 2 steps. Findings for M4: qwen3 issued `add` and `echo` in one step and passed `echo` the sum before seeing `add`'s result (it guessed); the chars/3.5 estimate was about 20% below the backend's prompt token count (198 vs 247).

### M3: Agent YAML + core tools + safety
- `agents/schema.py`, loader, built-in `default` agent.
- Tools: `read_file`, `list_dir`, `write_file`, `run_shell`, `python_exec`, `ask_user`.
- `safety/permissions.py` (allow/confirm/deny, CLI prompt for confirm), `safety/paths.py` (jail), `safety/redact.py`.
- `kestrel run "task" --agent default` (workspace is always `<project>/workspace/`, see decisions below)
- **Done when:** path-escape tests (`../`, absolute paths, symlinks) all rejected; confirm flow works; demo: "count lines in all .py files and write summary to report.md" succeeds on qwen3:8b.
- **Plan (2026-11-11), six pieces, each walked through before the next:** (1) path jail, (2) tool registry + `@tool`, (3) file tools, (4) permissions + confirm, (5) `run_shell`, `python_exec`, `ask_user`, (6) agent YAML + shell `/agent`, `/agents`, `/workspace` (shows the folder), `/tools` + untrusted wrapping. Diagrams: `docs/diagrams/tools-and-safety.md`.
- **Status: done (v0.0.3).** Live demo on qwen3:8b with the `counter` agent: correct counts (19, 81, 60, total 160) in 4 steps, 157 s; it asked `read_file` for one line per file to get the "N lines total" header. Finding for M4: the report was written with literal `\n` text, because the model double-escaped newlines inside the JSON arguments (a double-encoded string, section 5.2 step 2).
- **Decisions (2026-11-11):** no OS sandbox in M3 for `run_shell`/`python_exec` (confirm by default, workspace cwd, timeout, output cap, secret env vars removed; README says plainly it is not a sandbox; real sandbox in backlog). Confirm answer "a" allows that tool for the rest of the run only, never saved. Subprocesses get the environment minus vars named like `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`. Workspace (changed by owner, 2026-11-11, for safety): agents only ever use one folder, `workspace/` inside the Kestrel project folder (created on first use, ignored by git). For now there is no way to point agents anywhere else; `/workspace` only shows the path. `python_exec` uses Kestrel's own interpreter with `-I` and a temp dir inside the workspace.

### M3.5: Shell as an agent (Claude Code style)
- Every message typed in the shell is one agent turn with the current agent's tools; the model decides when to use them. `/run` is removed. The conversation (tool calls and results included) carries over between messages, trimmed by the context manager; the latest user message is the pinned task. `/clear` forgets it; `/agent` keeps it but swaps prompt, tools and permissions.
- Each message is one run with its own log file, so `/trace` works as before. A turn that fails or is stopped (Ctrl-C, Esc, backend error) is rolled back and not kept; a turn that hits a budget is kept.
- New look, no new dependency (prompt_toolkit + rich): welcome panel; bordered input box with a status line under it (model, agent, context used); `/` shows matching commands with their help under the box; tool calls as one line `⏺ tool(arg)` with a short result under `⎿`; spinner with elapsed time; Esc stops a turn; permission prompt as an arrow-key menu (Yes / Yes for the rest of this turn / No). Falls back to plain prompts when not attached to a terminal.
- `kestrel run "task"` keeps working with the same output.
- **Plan (2026-11-12), two pieces:** (1) conversation turns in the shell, (2) the new look. Diagram: `docs/diagrams/shell.md`.
- **Status: done (v0.0.4).** Live on qwen3:8b: plain questions answered without tools, tool use when needed, confirm menu, Esc rolls a turn back. Also: with no model configured, the shell uses the first model the backend lists. After owner feedback the input box is fixed at the bottom for the whole session (one prompt_toolkit app, a worker thread runs each line, output printed above the box); lines typed during a turn are queued. The context manager now pins the latest user message instead of the first (same thing for one-shot runs).

### M4: Reliability layer
- **Plan (2026-11-12, from the kestrel-architect brief; owner decisions below), four pieces:**
  1. **Capture.** Providers never raise on a broken `arguments` string: `ToolCall` gains `arguments_text` (raw) and `arguments_error`. `model_request` events log the exact messages sent after trimming (side file when over 64KB), plus tool spec hash, mode and think setting. Failure corpus in `tests/fixtures/bad_outputs/` (one JSON file per case: source, tools, response, expected result) with one parametrized test. Shell `/report-failure [run] [--step N]` (run defaults to `last`; step defaults to the last replied step where something went wrong) writes a redacted case file to `<data dir>/reports/`; `/trace ID --prompt N` shows one step's prompt. A broken call sent back in history is wrapped as `{"invalid_arguments": "<raw text>"}`, because Ollama rejects invalid JSON there (checked live 2026-11-12). Every step logs its whole prompt, so logs grow with the square of the steps (about 1MB for a 15-step run at 16K); revisit with deltas if logs pass about 10MB.
  2. **Repair** (`reliability/repair.py`, pure function called by the loop between model and tools): find calls (native field, or ```json blocks, `<tool_call>` tags, bare JSON in the text), fix JSON (trailing commas, single quotes, missing brackets, arguments sent as a JSON string), fix tool names (case, `-`/`_`, edit distance 1 to 2 only when unique), check arguments with pydantic, unescape double-escaped text only in arguments a tool marks as text (`ToolSpec.text_args`; only when there is no real new line and 2 or more literal `\n`), one call per step (later calls dropped with a short note). Unusable call: corrective note with that tool's schema; `max_repairs` (default 2) unusable calls in a row, then a `stuck` event and `stopped_reason=loop`. Every fix is a `repair` event. Never repaired (would change a correct call's meaning): a cut-off string or number (`finish_reason=length` gets its own note), JSON words inside strings, escapes inside quoted strings or Windows paths, a fuzzy tool-name guess that leaves arguments unused, a call quoted inside a long answer. Text calls need an exact tool name (up to case and `-`/`_`). Corpus: 48 cases.
  3. **Watchdog** (`reliability/watchdog.py`): `loop_warning` on 3 identical calls in a row or A,B,A,B; `stuck` on 4 identical calls, 5 steps in a row where a call repeats an earlier call and gets its earlier result (different calls with equal results are progress), 3 tool errors in a row, repairs exhausted, or a prompt that trimming could not fit with backend prompt tokens at 95% of the window (a final answer always counts). Warnings re-arm when the pattern stops; after `on_stuck` handles a signal the watchdog resets. M4 policy: warn message once, then stop with `stopped_reason=loop`. `run_agent` takes an `on_stuck` hook so M4.5 can step up instead. Thresholds are constants in code (agent-file settings dropped in the M4 cleanup).
  4. **Profile** (`providers/profiles.py`, section 4.3): capabilities and real window per model; a neutral `think` setting on `Provider.chat/stream`; default `max_tokens` for mlx-lm. Shell `/profile` and `/profile probe`; `/think on|off` controls whether the model thinks, `/think show|hide` the display; `kestrel run --think` and `kestrel ask --think`. (`/profile set` dropped: with text modes in the backlog there is nothing to set.)
- **Decisions (owner, 2026-11-12):** web tools move to M4.1, before M4.5. Thinking is off by default for agent steps. One tool call per step by default. Text tool-call modes (`json`, `react`) for models without native tools are moved to the backlog. Loop stop at 4 repeats (configurable). No GBNF grammars (llama.cpp rejects a grammar sent with tools; JSON schema output covers it). No real tokenizers (would need downloads). No separate Ollama `/api/chat` provider for now.
- **Done when:** 30+ malformed outputs in the corpus are fixed or correctly rejected; both real qwen3 failures (guessed parallel result, double-escaped `write_file` content) are handled in a live run; qwen3:4b finishes a file task natively; thinking off works live on Ollama.
- **Status: done (v0.0.5).** Corpus: 48 cases. Live on qwen3:8b: thinking off cut a two-step task from about 20s to 5s; the M3 line-count demo took 17s (was 157s) and repair's one-call rule fired on two parallel `read_file` calls; a broken call sent back in history is accepted after wrapping (Ollama rejected it raw). Double-escaped content did not recur live; its real case is in the corpus. The token estimate for qwen3:8b with tools came out about 25% high, not low as in M2; it stays a plain estimate (per-model measuring was built, then dropped in the cleanup as it changed nothing). qwen3:4b live check skipped by the owner for now (model not downloaded); run it after `ollama pull qwen3:4b`. Every piece was reviewed by the kestrel-reviewer agent before its commit; its findings (truncated calls being completed, escapes in one-line code, false stuck signals, remote probes, cache races) were fixed with tests.
- **Cleanup (owner, 2026-11-12):** the shell is the only interactive path; `kestrel run` prints plain output and takes typed y/n/a answers (Ctrl-C stops it). Dropped: the separate arrow-key menu and Esc watcher for `kestrel run`, the terminal `kestrel report-failure`, per-model token-ratio measuring, the watchdog and repair limits in agent files, and tests that only checked wording. Paths: the workspace is the model's whole world, so `/data` means `<workspace>/data` (qwen3 kept writing `/data` and giving up).

### M4.1: Web tools
- `read_tool_output` (page through cut results, kept per session in memory), `fetch_url` (stdlib HTML to text, GET only, size and time caps, private and loopback addresses blocked after every redirect, result untrusted), `web_search` (SearXNG preferred; best-effort fallback decided in the M4.1 plan).
- `ToolSpec.network`; web tools only exist for agents that list them; a built-in `researcher` agent.
- Open for the M4.1 plan: confirm fetches to hosts not seen this run (exfiltration risk), and the search fallback.
- **Shape (owner, 2026-11-12; reshaped in M4.3):** web research is one `research(question)` tool. Kestrel runs a fixed pipeline inside it (search, fetch, a separate model call that summarizes with sources), so raw pages never fill the main agent's small window, and the small model never has to manage a sub-agent (the CMU study found sub-agents never helped small models).
- **Done when:** a research task works on a small local model, with every fetched page wrapped as untrusted.

### M4.2: Fixes from the owner's chats
Found by the owner chatting with qwen3:8b: Ollama ran it with a 4096-token window (its default under 23 GiB of VRAM) while the status line said 16K; the model could not append to a file (no tool for it, `write_file` refused) and twice ended a turn with "Let me do that for you." and no call; tool `ERROR:` results were not flagged as errors.
- **Window:** the status line shows the window really used ("of 4K (backend loaded it small: /profile)"); the first-message warning (from a fresh probe; a banner warning from the cached profile was dropped after a live test showed it stale once Ollama was restarted), `/profile` and the trace summary give the fix (`OLLAMA_CONTEXT_LENGTH=16384 ollama serve`). Kestrel cannot raise it itself: Ollama's `/v1` never sends `num_ctx`, so a model loaded bigger would be reloaded at the default.
- **File tools:** `search_files`, `append_file`, `edit_file` (section 8.3); `append_file` and `edit_file` results show the changed lines numbered, `write_file` reports line counts (was, now). Edits keep line endings (CRLF stays CRLF), refuse files that are not UTF-8, and deleting whole lines leaves no gap.
- **Nudge** (`reliability/nudge.py`): a reply with no tool call whose last three sentences announce an action ("let me", "I'll", "I will", "I'm going to", "now I", "next I"; not "let me know", "I'll be happy", "I hope", not a question) gets one note per run and another try. `nudge` event.
- **Trace summary** above the timeline: how the run ended and why, model and backend, window and fullest prompt, trims, failed calls, repairs, warnings, nudges, tokens and time. The `profile` event carries `wanted_window`.
- **Read before changing** (after the owner's chat, where qwen3 edited `sales.csv` from memory and the owner approved an edit that could not match): `edit_file`, and `write_file` with `overwrite=true`, need the file read in this conversation (any read, even part of it) and unchanged since (modification time and size). A read whose result was later trimmed from the prompt still counts; a wrong `old_text` then fails as before. `Toolbox.precheck(call)` runs before the permission gate, so a call that cannot work is answered with an error and the user is never asked. The memory of read files lives for the shell conversation, is cleared by `/clear`, and rolls back with a dropped turn.
- **Today's date** is added to the system prompt each turn ("Today is Thursday, 2026-11-12 ..."): the model wrote 2023 for dates given without a year.
- Default agent: the three new tools; prompt says which file tool is for what, to act instead of announcing, to use details already given, and to make a sensible choice when a request is unclear.
- **Status: built.**

### M4.3: Lessons from harness research
Sources: CMU, "Better Harnesses, Smaller Models" (arXiv 2607.08938): most gains came from added instructions (86% of winning harnesses), new tools (43%) and fewer tools (29%); sub-agents never helped small models; gains were smallest for the weakest (8B) model and for varied tasks. Hookele (a small model on Terminal-Bench 2.0): delete scaffolding before adding, head-and-tail output caps, failures shown verbatim, one nudge when the model stalls after a failure, a step countdown injected near the limit, fuzzy edit matching.
- **Head and tail:** long tool output keeps its start and its end with a "cut from the middle" marker (context cut, registry cap, and subprocess output, which now reads up to 5 MB keeping 10 KB of each end before stopping the process).
- **Steps left:** with 2 steps or fewer left, each tool result ends with a countdown; the last says to answer without tools.
- **After-failure nudge:** a reply with no tool call right after a failed tool gets one note ("try another way, or say why it cannot be done"). Shares the once-per-run budget with the promise nudge; never after a permission refusal (those results start `ERROR: (not allowed)`); never on the last step. `nudge` events carry `kind` (`promise`, `after_failure`).
- **Indentation drift in `edit_file`:** a last matching level ignores leading whitespace too, only when that matches one place; the new text is re-indented from the model's indent steps to the file's (learned from the matched lines, so spaces become tabs level for level).
- **Default agent keeps `python_exec`** (tried dropping it, following Hookele's "fewer tools"; live, qwen3:8b then crammed scripts into `python3 -c` one-liners, repeated the same syntax error five times and invented a total). Its prompt now says standard library only, and to report only results a tool gave; the after-failure note says never to state a result no tool gave.
- Live finding, not yet acted on: a shell command that exits non-zero is not a failed call for the watchdog or the after-failure nudge (its output must stay wrapped as untrusted, so it cannot simply be marked as an error).
- **Status: built.**

### M4.5: Model ladder, basic (headline feature)
- `model:` list in agent YAML (gap 14) becomes a ladder; shell `/ladder MODEL MODEL ...` and `/up`.
- Step-up triggers from M4's failure signals (repairs exhausted, loop warning, repeated tool errors) plus window overflow.
- Shared conversation across models (neutral format, per-model tool mode), handoff note, step back down after the hard step.
- `step_up` / `step_down` events with reason; `/trace` shows which model did each step.
- **Basic OpenRouter rung (moved up from M9.5 by owner, 2026-11-12):** the top rung may be an OpenRouter model (the `openrouter` provider, key from `OPENROUTER_API_KEY`), used only when the agent has `allow_cloud: true` and only for the stuck step. Before anything is sent, the user sees that the step is going to the cloud and approves it (y/n/always this run). It sends the system prompt, the current task and the failing step with a short handoff note, never the whole history; secrets are redacted as in the event log. `/model` marks OpenRouter's free models; its rate-limit errors show as a plain message; the remote model's window and tool support come from OpenRouter's model list. The full privacy filter (exact text shown, spending cap, per-agent "never cloud"), planner/checker splits and learning stay in M9.5.
- **Done when:** FakeProvider tests cover each trigger, handoff note content, stepping back down, trace marks, and that a cloud step is never taken without `allow_cloud` and the user's yes; a live run on qwen3:4b plus qwen3:8b steps up at least once on a task the 4B fails and finishes it; a live run steps up to an OpenRouter model once with approval.

### M5: Interactive chat + MCP client
- `kestrel chat --agent X`: multi-turn session, `/agent`, `/model`, `/tools`, `/trace`, `/exit` slash commands.
- Session persistence (SQLite) and `kestrel chat --resume <id>`.
- Session id: every run started from one shell conversation carries the session id, so `/trace` can list a session's runs (trace levels: step, run, session).
- MCP client: stdio + HTTP servers from config.
- **Done when:** filesystem MCP server and one third-party MCP server work end to end.

### M5.5: Coder agent (software development, tentative)
- Built-in `coder` agent YAML plus code tools:
  - `repo_map`: compact outline of the repo (files, top-level symbols) that fits an 8K window.
  - `search_code`: text/regex search with file and line results.
  - `edit_file`: apply a small search/replace edit instead of rewriting whole files; confirm by default, shows a diff.
  - `run_tests` / `run_check`: run the project's test or lint command (from agent YAML), return failures compactly.
  - `git_status` / `git_diff`: read-only git view. No commits or pushes by the agent unless explicitly allowed.
- Loop: read, edit, test, read failures, fix, until tests pass or budget hit.
- **Done when:** on qwen3:8b, the coder agent fixes a failing test in a small fixture repo end to end, with every edit confirmed and logged.

### M6: Memory + context upgrades
- Summarization of dropped turns.
- Long-term memory store (SQLite + sqlite-vec or similar, ask before choosing), `remember` / `recall` tools.
- Tool pruning via embeddings.
- Folder RAG: `kestrel index <folder>` + `search_notes` tool.

### M7: Eval harness
- `evals/tasks/*.yaml`: task prompt, workspace fixture, success checker (python function or file assertion).
- Checks can judge the final answer (exact or criteria), the workspace end state (files left behind), or the sequence of tool calls. Each task starts from a fresh copy of its fixture folder as the workspace.
- Shell: `/keep [run] "check"` turns a run (usually a failed one) into a task; `/eval` runs the agent's tasks.
- `kestrel eval --models qwen3:8b,gemma4:4b --suite core` outputs table: success rate, avg steps, repairs per run, tokens, wall time.
- Results saved as JSON + markdown for README leaderboard.
- Designed as the foundation for the autotuner: the runner takes a harness config (tool mode, prompt template, tool description style, max tools, temperature, context strategy) as an input, not hardcoded, and is callable as a library, not just a CLI.

### M7.5: Autotuner (pillar 2)
- `kestrel tune --model X [--suite core] [--budget 30m]`: search harness configs with the M7 runner, pick the best by success rate (tie-break on steps, tokens, wall time), save as the model profile.
- Search must fit a 16GB laptop time budget (see open decisions).

### M8: Router + sub-agents
- Supervisor agent delegates to specialist agents with isolated context, results merged as tool results.

### M8.5: Recipes (pillar 3)
- Compile a successful run trace into a recipe: deterministic steps replay without the model; model called only at marked decision points.
- `kestrel recipe save last-run --name X`, `kestrel recipe run X`.
- Replay verifies each step's preconditions; on mismatch, fall back to the agent loop from that step.

### M9: Hooks + scheduler
- Hooks: `pre_tool`, `post_tool`, `on_error`, `on_final` (python callables or shell commands in config).
- Scheduler: cron-style jobs run agents in background, results to file / notification.

### M9.5: Model ladder, full: cloud rung, privacy filter, learning
- Builds on M4.5 (which has the basic OpenRouter rung). Cloud rung only if the agent allows it (`allow_cloud`): send only the failing step plus a summary, never the whole history.
- Privacy filter: before anything leaves the machine, redact sensitive data and show the user exactly what will be sent; spending cap per run; per-agent "never cloud".
- Planner/doer and checker splits; the user as a rung before cloud.
- Learning: step-up history per task kind decides the starting model; big-model solutions become examples for the small model.

### M10+: Frontends
- Local web UI (FastAPI + SSE, simple frontend).
- Telegram bot.
- Expose Kestrel as OpenAI-compatible endpoint and as MCP server.
- Voice (whisper.cpp + Piper).

### Backlog (unscheduled)
- **Tool strategy** (owner, 2026-11-12): a small, sharp built-in core (files, search, shell, Python); breadth through MCP servers (M5), switched on per agent; each agent lists only the tools its job needs. Later (with memory, M6): the model may propose saving a Python snippet that worked as a named tool; the owner reviews and approves it, and it runs under the same jail and confirm rules.
- `/profile fix`: create a copy of the model in Ollama with `num_ctx` set, so the bigger window needs no restart (changes the user's Ollama setup, so it would ask first).
- **Text tool-call modes** (moved from M4 by owner, 2026-11-12): `json` mode (one JSON object per reply, forced by the backend's JSON schema output where supported) and `react` mode (`Thought` / `Action` / `Action Input` / `Final Answer` text, generation stopped at `Observation:` so the model cannot invent results). For models without native tool calls (gemma3, phi4). History stays in the neutral format; a renderer turns it into each mode's text.
- Hardware-aware runtime: knows RAM, loaded models, battery, thermals; avoids loading two 8B models on 16GB; laptop mode on battery; warns on fanless throttling.
- Agent tests: each agent YAML ships test cases; `kestrel test agents/researcher.yaml --model qwen3:8b` gives a pass rate.
- Consistency probe: run the same task phrased two ways; different answers mean one is wrong. Candidate stuck signal for the model ladder.
- Export good runs as fine-tuning data (mlx-lm LoRA on a 16GB Mac).
- Review agent: an agent that reads recent run logs and proposes agent-file changes, applied only with the user's OK.
- Time-travel debugging: fork a past run at step N, change model, prompt or tool, rerun, diff.
- Multi-model team: tiny model for routing and summarizing tool output, 8B for reasoning, embeddings for memory, scheduled to minimize model swapping.
- Tool output shaping: compact, paged, structured outputs for 8K windows.
- OS sandbox for `run_shell` / `python_exec` (bubblewrap on Linux; macOS and Windows options to research).
- Agent trust list: mark specific agent files trusted by content hash so `--trust-agent` is not needed each run; changing the file revokes trust.
- Niche: offline knowledge packs (Kiwix, dev docs), Indian language support, memory as editable markdown, energy and cost meter, portable open agent spec.

---

## 10. Agent YAML schema

```yaml
name: researcher                 # unique, [a-z0-9_-]
description: Researches a topic and writes a cited summary.
model: qwen3:8b                  # or an ordered list: [qwen3:4b, qwen3:8b] (later = fallback)
provider: ollama                 # name from user config providers
allow_cloud: false               # cloud models need true (gap 14)
tool_mode: auto                  # auto | native | json | react
temperature: 0.2
system: |
  You research topics using web_search and fetch_url.
  Cite every claim with its URL. Write results with write_file.
tools:
  - web_search
  - fetch_url
  - write_file
permissions:                     # overrides tool defaults; loosening needs --trust-agent (gap 7)
  write_file: confirm
limits:
  max_steps: 15
  max_tokens_total: 60000
  max_wall_seconds: 600
context:
  window: 16384                  # lowered to the model's loaded window when smaller (M4)
  max_tool_result_tokens: 1500
```

Loader validates with pydantic, reports line-precise errors. Search order: `./agents/`, `~/.config/kestrel/agents/`, built-ins.

---

## 11. Event log

One JSON object per line:

```json
{"ts": "2026-11-06T10:15:02.123Z", "run_id": "r_8f3a", "step": 3,
 "type": "tool_call", "data": {"id": "c1", "name": "read_file", "arguments": {"path": "notes.md"}}}
```

Events store full tool arguments, full results and verbatim model output; results over 64KB go to a side file referenced by path (gap 13). Every line is redacted before writing (gaps 6, 11).

Event types: `run_start`, `profile` (what was known about the model, M4), `model_request` (with the exact messages sent, M4), `model_response`, `repair`, `tool_call`, `permission`, `tool_result`, `context_trim`, `loop_warning`, `stuck` (M4), `nudge` (M4.2), `error`, `final`, `run_end`.

`run_end.data` includes `stopped_reason` (`final|max_steps|budget|loop|cancelled|error`), totals for steps, tokens, repairs, wall time.

Uses: `kestrel trace`, replay tests (feed recorded `model_response`s into FakeProvider), eval scoring, bug reports.

---

## 12. Config

`~/.config/kestrel/config.toml` (Windows: `%APPDATA%\kestrel\config.toml`):

```toml
default_agent = "default"
default_provider = "ollama"          # used by `kestrel ask` when --provider is not given
default_model = "qwen3:8b"           # used when --model is not given (optional)

[providers.ollama]
base_url = "http://localhost:11434/v1"

[providers.lmstudio]
base_url = "http://localhost:1234/v1"

[providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"   # never store keys in file

[tools.web_search]
backend = "searxng"
searxng_url = "http://localhost:8888"

[[mcp_servers]]
name = "fs"
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]
```

Env overrides: `KESTREL_CONFIG`, `KESTREL_DATA_DIR`. Without `KESTREL_CONFIG`, non-Windows systems honour `XDG_CONFIG_HOME`.

Rules (implemented in `core/config.py`, M1):
- No file: built-in providers `ollama` (11434), `lmstudio` (1234), `llamacpp` (8080), `vllm` (8000).
- `[providers.X]` replaces the built-in of the same name; other built-ins stay.
- Unknown keys are errors, reported with their location (e.g. `providers.ollama.port`).
- An `api_key` written in the file is refused; use `api_key_env`.
- A provider is `remote` when its host is not loopback (`localhost`, `127.0.0.1`, `::1`); LAN hosts count as remote.

---

## 13. Testing strategy

- **Unit (default):** FakeProvider, mocked httpx, temp-dir workspaces. Fast, offline, run in CI on 3 OSes.
- **Fixtures:** `tests/fixtures/bad_outputs/` collects real malformed outputs from small models. Every new failure seen in the wild becomes a fixture.
- **Replay:** recorded JSONL runs replayed through the loop to catch regressions in harness logic.
- **Live (`-m live`):** against a real local model, run manually or nightly, not required in CI.
- **Security tests:** path traversal, symlink escape, oversized output, shell timeout, injected instructions inside fetched pages must not change permissions.

---

## 14. Open decisions (resolve when reached)

Each is resolved at the start of the named milestone's plan, with evidence from real runs.

- Final project name (M3/M4, before going public). `kestrel` and `kestrel-agent` are taken on PyPI (the former by an inference engine); `kestrel-harness` is free (checked 2026-11-11).
- Web search default when no SearXNG: which library (M4).
- Vector store: sqlite-vec vs LanceDB vs Chroma (M6).
- Tokenizer: keep approximation or use per-model tokenizers (M4).
- Constrained decoding coverage across backends: which support GBNF, JSON schema, neither (M4).
- Windows sandboxing story for `run_shell` / `python_exec`: decided for M3 (no OS sandbox on any OS, see M3); a real sandbox is in the backlog.
- Recipe format: how decision points are marked and replayed (M8.5).
- Project direction: stay a general harness with coding as flagship, or narrow further towards coding agents (revisit after the M4 demo, with real small-model results).
- Autotuner search space size vs time budget on a 16GB laptop (M7.5).

### Spec gap decisions (raised and decided 2026-11-11)
1. **Ollama context size (M1).** Ollama allocates context by VRAM: under 24 GiB (all 16GB Macs) the default is 4K tokens, below our 8K to 16K target, and `/v1` cannot change it per request. **Decision:** M1 uses only the OpenAI-compatible endpoint; `kestrel doctor` warns about the 4K default and prints the fix (Ollama app context slider, or `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`). An Ollama-native provider (`/api/chat` with `num_ctx`) is added in M3/M4 only if real runs need it.
2. **Port 8080 (M1).** **Decision:** `doctor` calls `GET /props`; success means llama.cpp, otherwise "mlx-lm (probably)". User config wins.
3. **Streaming types (M1).** **Decision:** `stream()` yields `StreamChunk(text, reasoning, done)`; the final chunk carries the assembled `ModelResponse` in `done`, so callers never assemble deltas.
4. **Data dirs (M2).** **Decision:** XDG-style paths on every OS: data `~/.local/share/kestrel`, cache `~/.cache/kestrel`, config `~/.config/kestrel` (Windows config: `%APPDATA%\kestrel`, as section 12). Stdlib only. `KESTREL_DATA_DIR` / `KESTREL_CONFIG` override.
5. **`write_file` default (M3).** **Decision:** creating a new file inside the workspace is `allow`; overwriting an existing file is `confirm`.
6. **Redaction meaning (M3).** **Decision:** redact (a) the exact values of env vars whose names match `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, and (b) well-known secret formats (`sk-...`, `ghp_...`, `Bearer ...`, and similar), replaced with `[REDACTED]`.
7. **Loosening permissions (M3).** **Decision:** an agent YAML may loosen a tool's default (e.g. `run_shell: allow`) only when the run is started with `--trust-agent`; otherwise Kestrel refuses to start and explains which permissions the file tries to loosen. Applies to every agent file regardless of location. Content-hash trust list is in the backlog.
8. **License holder (M0). Resolved:** Apache-2.0, official text unmodified in `LICENSE`; `pyproject.toml` authors = "Kestrel contributors".
9. **JSON typing (M1).** **Decision:** a recursive `JSONValue` type alias in `core/types.py` replaces `Any` for tool arguments and JSON Schemas.
10. **Token usage when streaming (M1).** **Decision:** requests send `stream_options: {"include_usage": true}`; if the backend still omits usage, `usage` is `None`. The chars/3.5 estimate lives in the context manager (M2).
11. **Redaction before M3 (M2).** **Decision:** the redactor (gap 6) is built in M2 inside the JSONL event writer, so no log line is ever written unredacted. M3 adds tool-output tests.
12. **Frontend interface (M2).** **Decision:** a `Frontend` protocol in `core/types.py` (`confirm(call) -> bool`, `ask(question) -> str`, `on_event(event)`) defined in M2; the CLI implements it, tests use a fake.
13. **Recipe-ready events (M2).** **Decision:** events record full tool arguments, full tool results and verbatim model output. Results larger than 64KB are written to a side file next to the run log and the event stores its path.
14. **Model ladder in agent YAML (M3).** **Decision:** `model` accepts one name or an ordered list (`model: [qwen3:4b, qwen3:8b]`); `fallback_model` is removed (later list entries are fallbacks). Cloud models require `allow_cloud: true`.
15. **Embeddings (M6).** **Decision:** add `embed(texts, model) -> list[list[float]]` (`POST /v1/embeddings`) to the provider interface in M6, not before.
16. **Cloud before privacy filter (M1 to M9.5).** **Decision:** `doctor` and `ask` label any non-local base URL `REMOTE` (data leaves this machine); README states data is sent as-is until M9.5.
17. **Name.** See open decisions above.

---

## 15. Definition of done for v1.0

- Install with `uv tool install kestrel` or `pipx install kestrel`.
- 5-minute quickstart works on a 16GB laptop with a free local model.
- Eval leaderboard for at least 6 local models in README.
- Docs site, plugin (tool) authoring guide, agent authoring guide.
- Stable public API for tools, agents, providers, hooks.
