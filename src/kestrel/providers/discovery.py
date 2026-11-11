"""Find which backends are running and what they serve (`kestrel doctor`, spec 4.2).

Every configured provider is probed in parallel with `GET {base_url}/models` and a short
timeout. Nothing here changes any backend; it only asks.
"""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlsplit

import httpx

from kestrel.core.config import ProviderConfig
from kestrel.core.types import BackendUnreachableError, ProviderError
from kestrel.providers.openai_compat import OpenAICompatProvider

PROBE_TIMEOUT_SECONDS = 1.5

# What usually listens on each default port. 8080 is shared by llama.cpp and mlx-lm,
# so it is fingerprinted instead (gap 2).
KIND_BY_PORT: dict[int, str] = {11434: "Ollama", 1234: "LM Studio", 8000: "vLLM"}
SHARED_PORT = 8080
OLLAMA_PORT = 11434

OLLAMA_CONTEXT_WARNING = (
    "Ollama gives models only 4K tokens of context on machines with under 24GB of GPU "
    "memory (all 16GB Macs). Kestrel needs 8K to 16K. Raise it in the Ollama app "
    "(Settings, Context length) or start the server with "
    "`OLLAMA_CONTEXT_LENGTH=16384 ollama serve`. Check with `ollama ps`."
)
REMOTE_WARNING = "REMOTE: prompts sent here leave this machine."


class Status(StrEnum):
    """Outcome of probing one backend."""

    up = "up"
    down = "down"
    """Nothing is listening."""
    error = "error"
    """Something answered, but not with a usable model list (bad key, wrong URL, ...)."""


@dataclass(frozen=True)
class BackendReport:
    """What `doctor` found out about one configured provider."""

    name: str
    base_url: str
    status: Status
    kind: str
    """Best guess of the server software, e.g. "Ollama" or "mlx-lm (probably)"."""
    remote: bool
    models: list[str] = field(default_factory=list[str])
    error: str | None = None
    warnings: list[str] = field(default_factory=list[str])


async def discover(
    providers: Mapping[str, ProviderConfig],
    *,
    env: Mapping[str, str] | None = None,
    request_timeout: float = PROBE_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[BackendReport]:
    """Probe every provider at once; reports come back in the input order.

    `env` supplies API keys (`api_key_env`). `transport` exists for tests.
    """
    env = env or {}
    return list(
        await asyncio.gather(
            *(
                probe(name, config, env=env, request_timeout=request_timeout, transport=transport)
                for name, config in providers.items()
            )
        )
    )


async def probe(
    name: str,
    config: ProviderConfig,
    *,
    env: Mapping[str, str] | None = None,
    request_timeout: float = PROBE_TIMEOUT_SECONDS,
    transport: httpx.AsyncBaseTransport | None = None,
) -> BackendReport:
    """Probe one provider: list its models, guess what it is, collect warnings."""
    remote = config.is_remote
    port = urlsplit(config.base_url).port
    warnings = [REMOTE_WARNING] if remote else []
    provider = OpenAICompatProvider(
        config.base_url,
        name=name,
        api_key=config.api_key(env or {}),
        timeout=httpx.Timeout(request_timeout),
        retries=0,  # doctor should be quick; a backend that is starting shows as down
        transport=transport,
    )
    async with provider:
        try:
            models = await provider.list_models()
        except BackendUnreachableError:
            return BackendReport(
                name, config.base_url, Status.down, _kind_by_port(port), remote, warnings=warnings
            )
        except ProviderError as exc:
            return BackendReport(
                name,
                config.base_url,
                Status.error,
                _kind_by_port(port),
                remote,
                error=str(exc),
                warnings=warnings,
            )

    kind = (
        await _fingerprint_shared_port(config.base_url, request_timeout, transport)
        if port == SHARED_PORT
        else _kind_by_port(port)
    )
    if port == OLLAMA_PORT:
        warnings.append(OLLAMA_CONTEXT_WARNING)
    if not models:
        hint = " Try `ollama pull qwen3:8b`." if port == OLLAMA_PORT else ""
        warnings.append(f"No models listed.{hint}")
    return BackendReport(
        name, config.base_url, Status.up, kind, remote, models=models, warnings=warnings
    )


async def _fingerprint_shared_port(
    base_url: str, request_timeout: float, transport: httpx.AsyncBaseTransport | None
) -> str:
    """Tell llama.cpp from mlx-lm: only llama.cpp serves `GET /props` (at the root)."""
    parts = urlsplit(base_url)
    root = f"{parts.scheme}://{parts.netloc}"
    async with httpx.AsyncClient(timeout=request_timeout, transport=transport) as client:
        try:
            response = await client.get(f"{root}/props")
        except httpx.HTTPError:
            return "mlx-lm (probably)"
    return "llama.cpp" if response.is_success else "mlx-lm (probably)"


def _kind_by_port(port: int | None) -> str:
    if port == SHARED_PORT:
        return "llama.cpp or mlx-lm"
    return KIND_BY_PORT.get(port or 0, "OpenAI-compatible")
