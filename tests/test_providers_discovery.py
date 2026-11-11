"""Seam: discover() over a fake network. Each test describes which backends are 'running'."""

import httpx

from kestrel.core.config import BUILTIN_PROVIDERS, ProviderConfig
from kestrel.providers.discovery import (
    OLLAMA_CONTEXT_WARNING,
    REMOTE_WARNING,
    Status,
    discover,
)

type Routes = dict[str, httpx.Response]


def network(routes: Routes) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    """Fake network: `routes` maps "host:port/path" to a reply; anything else refuses."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        key = f"{request.url.host}:{request.url.port}{request.url.path}"
        if key in routes:
            return routes[key]
        raise httpx.ConnectError("refused", request=request)

    return httpx.MockTransport(handler), seen


def models(*ids: str) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": [{"id": i} for i in ids]})


async def test_nothing_running_reports_every_builtin_down() -> None:
    transport, _ = network({})
    reports = await discover(BUILTIN_PROVIDERS, transport=transport)
    assert [r.name for r in reports] == ["ollama", "lmstudio", "llamacpp", "vllm"]
    assert all(r.status is Status.down for r in reports)
    assert all(r.warnings == [] for r in reports)


async def test_ollama_up_lists_models_and_warns_about_context() -> None:
    transport, _ = network({"localhost:11434/v1/models": models("qwen3:8b", "gemma3:4b")})
    [ollama] = await discover({"ollama": BUILTIN_PROVIDERS["ollama"]}, transport=transport)
    assert ollama.status is Status.up
    assert ollama.kind == "Ollama"
    assert ollama.models == ["qwen3:8b", "gemma3:4b"]
    assert OLLAMA_CONTEXT_WARNING in ollama.warnings


async def test_ollama_with_no_models_suggests_a_pull() -> None:
    transport, _ = network({"localhost:11434/v1/models": models()})
    [ollama] = await discover({"ollama": BUILTIN_PROVIDERS["ollama"]}, transport=transport)
    assert any("ollama pull qwen3:8b" in w for w in ollama.warnings)


async def test_port_8080_with_props_is_llama_cpp() -> None:
    transport, _ = network(
        {
            "localhost:8080/v1/models": models("model.gguf"),
            "localhost:8080/props": httpx.Response(200, json={"n_ctx": 8192}),
        }
    )
    [report] = await discover({"llamacpp": BUILTIN_PROVIDERS["llamacpp"]}, transport=transport)
    assert report.kind == "llama.cpp"


async def test_port_8080_without_props_is_probably_mlx() -> None:
    transport, seen = network(
        {
            "localhost:8080/v1/models": models("mlx-community/Qwen3-8B-4bit"),
            "localhost:8080/props": httpx.Response(404),
        }
    )
    [report] = await discover({"llamacpp": BUILTIN_PROVIDERS["llamacpp"]}, transport=transport)
    assert report.kind == "mlx-lm (probably)"
    assert [r.url.path for r in seen] == ["/v1/models", "/props"]


async def test_remote_provider_is_labelled_and_gets_its_key() -> None:
    transport, seen = network({"api.example.com:443/v1/models": models("big-model")})
    providers = {
        "cloud": ProviderConfig(base_url="https://api.example.com/v1", api_key_env="CLOUD_KEY")
    }
    [report] = await discover(providers, env={"CLOUD_KEY": "sk-test"}, transport=transport)
    assert report.remote
    assert REMOTE_WARNING in report.warnings
    assert seen[0].headers["authorization"] == "Bearer sk-test"


async def test_backend_that_answers_badly_is_an_error_not_down() -> None:
    transport, _ = network({"localhost:1234/v1/models": httpx.Response(401, text="no key")})
    [report] = await discover({"lmstudio": BUILTIN_PROVIDERS["lmstudio"]}, transport=transport)
    assert report.status is Status.error
    assert report.error is not None
    assert "HTTP 401" in report.error


async def test_probes_are_not_retried() -> None:
    transport, seen = network({})
    await discover({"ollama": BUILTIN_PROVIDERS["ollama"]}, transport=transport)
    assert len(seen) == 1  # doctor stays fast when nothing is running
