# Contributing

Kestrel is in early development. Issues and ideas are welcome on [GitHub Issues](https://github.com/JagdeepSingh13/kestrel/issues).

## Setup
```bash
uv sync
```

## Before you push
All four must pass (CI runs them on macOS, Ubuntu and Windows):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

`uv run ruff format .` fixes formatting. Tests that need a real local model are marked `@pytest.mark.live` and run only with `uv run pytest -m live`.

## Rules
- Read [docs/SPEC.md](docs/SPEC.md) before changing behavior; update it in the same change if an interface moves.
- Tests with every change. Unit tests never hit a real model or the network.
- Type hints everywhere (pyright strict).
- No new dependency without discussing it first.
- Conventional commit messages: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`.
