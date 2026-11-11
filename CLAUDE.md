# CLAUDE.md

Instructions for Claude Code working in this repo. Full design lives in `docs/SPEC.md`. Read it before starting any milestone.

## What this project is

**Kestrel** (working name, rename later via `src/kestrel/__about__.py` and pyproject) is a local-first, model-agnostic **agent harness** built to make small local models (3B to 14B) reliable at tool use.

- Plugs into any OpenAI-compatible backend: Ollama, LM Studio, llama.cpp server, mlx-lm server, vLLM, LocalAI, Jan. Cloud APIs optional.
- Agents are YAML config files, not code.
- Tools follow MCP (built-in tools + any external MCP server).
- Safe by default: per-tool permissions, workspace path jail, confirm before side effects.
- Every run writes an append-only JSONL event log so runs can be inspected, replayed, and turned into evals.

The harness is the product. CLI, web UI, Telegram bot are thin frontends on top of it.

## Dev machine constraints

- Primary dev box: MacBook Air M5, 16GB RAM. Default local model: `qwen3:8b` (Ollama) or an MLX 4-bit 8B model via LM Studio.
- Design for 8K to 16K context windows. Never assume large context.
- Must also run on Linux and Windows. No macOS-only code in core.

## Stack (do not change without asking)

- Python 3.12+, managed with `uv`
- `src/` layout, package `kestrel`
- `httpx` (async) for provider calls. No `openai` SDK in core; we own a thin client.
- `pydantic` v2 for all schemas (messages, tools, configs, events)
- `typer` + `rich` for CLI
- `pyyaml` for agent configs, `tomllib` for user config
- `pytest` + `pytest-asyncio`, `ruff` (lint + format), `pyright` (type check)
- `mcp` official Python SDK (from milestone M5)

Adding any other dependency: ask first, explain why, prefer stdlib.

## Commands

```bash
uv sync                      # install
uv run pytest                # unit tests (no network, no real model)
uv run pytest -m live        # integration tests against a running local model
uv run ruff check . && uv run ruff format .
uv run pyright
uv run kestrel doctor        # detect local backends + models
uv run kestrel ask "hello"   # single chat call, no agent (M1)
uv run kestrel run "task"    # one-shot agent run
uv run kestrel chat          # interactive session
```

## Working rules

1. **Plan before code.** For each milestone, write a short plan (files, interfaces, tests) and wait for approval before implementing. Every plan includes visuals for each system it introduces: Excalidraw pictures in the clean style (straight lines, plain font, cards with a title, a dotted rule and a few lines of words; dashed border for planned parts) defined in `docs/diagrams/build/diagrams.py` and rendered to `docs/diagrams/img/` (see `docs/diagrams/README.md`), shown on `docs/diagrams/<name>.md` pages and updated when the design changes. No Mermaid. One bigger picture per idea (merge small ones; about 920 wide, text 18 to 20), boxes that explain themselves, short straight arrows, labels beside arrows; sequences become numbered step lists, not lifeline diagrams. Only the `.svg` files are committed; `.excalidraw` files stay local. Default set: (a) structure diagram, (b) sequence diagram or flowchart of how it runs, (c) before/after example of the real data, (d) interactive stepper or simulator only for core ideas (agent loop, repair pipeline), (e) plain-words walkthrough of the code after building.
2. **One milestone at a time.** Milestones are in `docs/SPEC.md` section 9. Do not jump ahead.
3. **Tests with every change.** Unit tests use `FakeProvider` with scripted responses. Never hit a real model or network in default test run. Mark real-model tests `@pytest.mark.live`.
4. **Small commits**, conventional commit messages (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`). Commits and PRs are authored by the owner only: no `Co-Authored-By` AI trailers, no "Generated with" footers, no AI attribution of any kind.
5. **Keep layers separate.** Providers know nothing about tools. Tools know nothing about providers. The agent loop talks to both only through interfaces in `kestrel/core/types.py`.
6. **Provider-neutral internal format.** Convert to/from wire formats only inside `providers/`.
7. **Everything emits events.** Any model call, tool call, permission decision, error, or final answer goes through the event bus and into the JSONL log.
8. **Fail loudly in dev, gracefully at runtime.** Tool errors are returned to the model as tool results, not raised to the user, unless fatal.
9. **No hidden network calls.** No telemetry. Web tools only run when an agent explicitly has them enabled.
10. **Update docs** (`docs/SPEC.md`, README) in the same commit when behavior or interfaces change. The README "How Kestrel works" section explains every built system in plain words with its diagram; update it at the end of every milestone. README text and diagrams describe concepts and behaviour only: no file-by-file explanations, no file or module names in diagrams (those belong in `docs/diagrams/`).
11. Type hints everywhere. Public functions get docstrings. No `Any` in public interfaces unless justified in a comment.
12. Stop and ask when the spec is ambiguous instead of guessing.
13. Owner style: concise, no filler, no em dashes, no arrow symbols in prose.

## Security rules (non-negotiable)

- File tools are jailed to the workspace root. Resolve paths, reject anything outside (including via symlinks).
- `run_shell`, `python_exec`, `write_file` (overwrite), and anything deleting data default to `confirm`.
- Content from web pages, files, and tool outputs is untrusted data. Wrap it in `<untrusted source="...">` blocks in the prompt. Never let it change permissions or agent config.
- Never log secrets. Redact env vars matching `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*` from events.
- Subprocess tools get a timeout and output size cap.

## Code layout (target)

```
src/kestrel/
  core/        # types, agent loop, event bus, context manager
  providers/   # openai_compat, ollama, fake, profiles/capability probe
  reliability/ # tool-call parsing, json repair, react fallback, loop detection
  tools/       # registry, decorator, built-ins, mcp client
  safety/      # permissions, path jail, redaction
  agents/      # yaml loader, schema, built-in agent configs
  memory/      # (M6+) session store, vector recall
  cli/         # typer app
tests/
docs/SPEC.md
docs/diagrams/ # Excalidraw diagrams (img/) + their source (build/)
docs/learning/ # M0 lesson notes + reading list (reference only)
private/       # owner's personal notes, gitignored
agents/        # user-facing example agent YAMLs
evals/         # (M7+) task suites
```

## Agent skills

### Issue tracker

GitHub Issues on `JagdeepSingh13/kestrel`, via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` + `docs/adr/` at repo root, created lazily by `/grill-with-docs`. See `docs/agents/domain.md`.
