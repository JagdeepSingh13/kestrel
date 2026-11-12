"""Seam: model profiles from backend metadata (mocked HTTP), the cache, and their effect."""

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from kestrel.agents.schema import AgentConfig
from kestrel.cli import agent_run
from kestrel.core.config import ProviderConfig
from kestrel.core.types import Message, Provider, Role, Usage
from kestrel.providers.fake import FakeProvider
from kestrel.providers.profiles import Profile, ProfileCache, backend_kind, probe
from kestrel.providers.wire import to_request_body
from kestrel.safety.paths import Workspace

OLLAMA = "http://localhost:11434/v1"


def ollama_backend(loaded_window: int | None) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "tools", "thinking"],
                    "model_info": {"qwen3.context_length": 40960},
                },
            )
        if request.url.path == "/api/ps":
            models = (
                []
                if loaded_window is None
                else [{"name": "qwen3:8b", "context_length": loaded_window}]
            )
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "digest": "abc" * 8}]})
        return httpx.Response(404)

    return httpx.MockTransport(handle)


def test_ollama_metadata_fills_the_profile() -> None:
    profile = asyncio.run(probe("ollama", OLLAMA, "qwen3:8b", transport=ollama_backend(4096)))
    assert (profile.kind, profile.native_tools, profile.thinking) == ("ollama", True, True)
    assert (profile.max_window, profile.window) == (40960, 4096)
    assert profile.digest == ("abc" * 8)[:16]
    assert profile.think_style == "reasoning_effort"


def test_llamacpp_reports_its_window() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 8192}})

    profile = asyncio.run(
        probe("llamacpp", "http://localhost:8080/v1", "x", transport=httpx.MockTransport(handle))
    )
    assert (profile.kind, profile.window, profile.native_tools) == ("llamacpp", 8192, True)


def test_an_unreachable_backend_gives_an_empty_profile() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    profile = asyncio.run(probe("ollama", OLLAMA, "m", transport=httpx.MockTransport(down)))
    assert profile.native_tools is None and profile.window is None


def test_backend_kind_from_name_then_port() -> None:
    assert backend_kind("ollama", "http://x:9999/v1") == "ollama"
    assert backend_kind("mine", "http://localhost:1234/v1") == "lmstudio"
    assert backend_kind("mlx-server", "http://localhost:8081/v1") == "mlx"
    assert backend_kind("cloud", "https://api.example.com/v1") == "other"


def test_token_ratio_is_measured_and_kept_in_range() -> None:
    profile = Profile(backend=OLLAMA, model="m")
    assert profile.estimate_ratio == 1.1  # safety margin before any measurement
    for _ in range(3):
        profile = profile.with_sample(reported=1200, estimated=1000)
    assert profile.samples == 3 and profile.estimate_ratio == pytest.approx(1.2)
    wild = Profile(backend=OLLAMA, model="m").with_sample(reported=9000, estimated=1000)
    assert wild.token_ratio == 1.5  # clamped


def test_cache_keeps_one_profile_per_backend_and_model(tmp_path: Path) -> None:
    cache = ProfileCache(tmp_path / "profiles.json")
    cache.save(Profile(backend=OLLAMA, model="m", digest="old", probed_at="1"))
    cache.save(Profile(backend=OLLAMA, model="m", digest="new", probed_at="2"))
    found = cache.get(OLLAMA, "m")
    assert found is not None and found.digest == "new"
    assert len(json.loads((tmp_path / "profiles.json").read_text())) == 1
    (tmp_path / "profiles.json").write_text("not json")
    assert cache.get(OLLAMA, "m") is None  # a broken file is ignored, not fatal


def test_thinking_off_is_sent_the_way_each_backend_understands() -> None:
    hi = [Message(role=Role.user, content="hi")]
    ollama = to_request_body(hi, model="m", think=False, think_style="reasoning_effort")
    assert ollama["reasoning_effort"] == "none"
    mlx = to_request_body(hi, model="m", think=False, think_style="chat_template_kwargs")
    assert mlx["chat_template_kwargs"] == {"enable_thinking": False}
    left = to_request_body(hi, model="m", think=None, think_style="reasoning_effort")
    assert "reasoning_effort" not in left


@pytest.fixture(autouse=True)
def fresh_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test is a new session: nothing probed or warned yet."""
    monkeypatch.setattr(agent_run, "_probed", set[tuple[str, str]]())
    monkeypatch.setattr(agent_run, "_warned_windows", set[tuple[str, int]]())


def run_with(
    profile: Profile, fake: FakeProvider, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[agent_run.RunResult | None, Path]:
    async def known_probe(name: str, base_url: str, model: str, known: Profile | None) -> Profile:
        return profile

    monkeypatch.setattr(agent_run, "probe_profile", known_probe)

    @asynccontextmanager
    async def open_fake(name: str, config: ProviderConfig) -> AsyncGenerator[Provider]:
        yield fake

    env = {"KESTREL_DATA_DIR": str(tmp_path), "KESTREL_CACHE_DIR": str(tmp_path / "cache")}
    result = agent_run.start_run(
        "list files",
        agent=AgentConfig(name="t", tools=["list_dir"]),
        workspace=Workspace(tmp_path),
        provider_name="ollama",
        provider_config=ProviderConfig(base_url=OLLAMA),
        open_provider=open_fake,
        model="qwen3:8b",
        env=env,
    )
    return result, tmp_path / "cache" / "profiles.json"


def test_a_model_without_tool_support_is_refused_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    no_tools = Profile(backend=OLLAMA, model="qwen3:8b", native_tools=False)
    fake = FakeProvider()
    result, _ = run_with(no_tools, fake, tmp_path, monkeypatch)
    assert result is None and fake.calls == []
    assert "cannot use tools" in capsys.readouterr().err


def test_the_real_window_and_measured_ratio_are_used_and_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    small = Profile(backend=OLLAMA, model="qwen3:8b", native_tools=True, window=4096)
    reply = FakeProvider.text("done").model_copy(update={"usage": Usage(prompt_tokens=500)})
    fake = FakeProvider([reply])
    result, cache_file = run_with(small, fake, tmp_path, monkeypatch)
    assert result is not None
    assert "loaded with a 4096-token window" in capsys.readouterr().err
    assert fake.calls[0].think is False  # thinking off by default
    saved = ProfileCache(cache_file).get(OLLAMA, "qwen3:8b")
    assert saved is not None and saved.samples == 1 and saved.window == 4096


def test_a_remote_backend_is_never_asked() -> None:
    asked: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return httpx.Response(200, json={})

    profile = asyncio.run(
        probe("ollama", "https://ollama.example.com/v1", "m", transport=httpx.MockTransport(record))
    )
    assert asked == [] and profile.kind == "other"


def test_an_unloaded_ollama_model_is_loaded_to_learn_its_window() -> None:
    state = {"loaded": False}
    inner = ollama_backend(16_384)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            state["loaded"] = True
            return httpx.Response(200, json={"done": True})
        if request.url.path == "/api/ps" and not state["loaded"]:
            return httpx.Response(200, json={"models": []})
        return inner.handle_request(request)

    profile = asyncio.run(
        probe("ollama", OLLAMA, "qwen3:8b", transport=httpx.MockTransport(handle))
    )
    assert state["loaded"] and profile.window == 16_384


def test_low_counts_never_loosen_the_estimate() -> None:
    cached = Profile(backend=OLLAMA, model="m")
    for _ in range(5):  # Ollama leaves reused (cached) prompt tokens out of its count
        cached = cached.with_sample(reported=300, estimated=1000)
    assert cached.estimate_ratio == 1.0


def test_a_cache_that_cannot_be_written_is_ignored(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cache = ProfileCache(blocker / "profiles.json")  # its folder is a file
    cache.save(Profile(backend=OLLAMA, model="m"))  # no exception
    assert cache.get(OLLAMA, "m") is None


def test_the_window_warning_is_shown_once_per_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    small = Profile(backend=OLLAMA, model="qwen3:8b", kind="ollama", window=4096)
    for _ in range(2):
        run_with(small, FakeProvider([FakeProvider.text("ok")]), tmp_path, monkeypatch)
    assert capsys.readouterr().err.count("loaded with a 4096-token window") == 1
