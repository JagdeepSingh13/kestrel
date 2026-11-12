"""Model profiles (spec 4.3, M4 piece 4): what one model on one backend can do.

Read from backend metadata, so no test prompt is needed:
- Ollama: `/api/show` (capabilities: tools, thinking; the model's own maximum context),
  `/api/ps` (the window actually loaded, which Ollama sets to 4096 on small machines
  unless `OLLAMA_CONTEXT_LENGTH` says otherwise), `/api/tags` (digest: a re-pulled
  model gets a fresh profile).
- llama.cpp: `/props` (`n_ctx`, the loaded window). Tools always work there.
- Other backends: nothing to ask; unknown fields stay None and nothing is assumed.

Measured over time: the ratio between the backend's real prompt token count and
Kestrel's estimate, so trimming fits the real window. Profiles are cached in
`<cache dir>/profiles.json`, keyed by backend URL, model and digest.
"""

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict

from kestrel.core.config import LOOPBACK_HOSTS
from kestrel.core.types import JSONObject, JSONValue
from kestrel.providers.wire import ThinkStyle

type BackendKind = Literal["ollama", "llamacpp", "lmstudio", "vllm", "mlx", "other"]

PORT_KINDS: dict[int, BackendKind] = {
    11434: "ollama",
    1234: "lmstudio",
    8080: "llamacpp",
    8000: "vllm",
}
THINK_STYLES: dict[BackendKind, ThinkStyle] = {
    "ollama": "reasoning_effort",  # maps to think=false (checked live on 0.40, 2026-11-12)
    "llamacpp": "reasoning_effort",  # per the llama.cpp server docs; not tried live
    "vllm": "chat_template_kwargs",
    "mlx": "chat_template_kwargs",
    "lmstudio": "none",
    "other": "none",
}
DEFAULT_MAX_TOKENS: dict[BackendKind, int] = {"mlx": 4096}
"""Backends whose own default cuts replies short (mlx-lm stops at 512 tokens)."""
METADATA_TIMEOUT = 3.0
LOAD_TIMEOUT = 120.0
"""Loading a model into memory can take a while on a laptop; it is needed for the run."""
RATIO_MIN, RATIO_MAX = 1.0, 1.5
"""The estimate only ever gets stricter: a backend's count can be low (Ollama leaves out
prompt tokens it reused from its cache), and an overfilled window is cut silently."""
RATIO_SMOOTHING = 0.3
"""Weight of the newest sample in the running ratio."""
SAFETY_RATIO = 1.1
"""Used until a model has `CALIBRATED_SAMPLES` measurements: assume 10% more tokens."""
CALIBRATED_SAMPLES = 3


class Profile(BaseModel):
    """What Kestrel knows about one model on one backend."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    backend: str
    model: str
    kind: BackendKind = "other"
    digest: str | None = None
    native_tools: bool | None = None
    """None: unknown (the backend did not say); False: requests with tools will fail."""
    thinking: bool | None = None
    max_window: int | None = None
    """The most the model itself supports."""
    window: int | None = None
    """The window the backend actually runs it with, when it says."""
    token_ratio: float = 1.0
    samples: int = 0
    probed_at: str = ""

    @property
    def key(self) -> str:
        """Cache key: backend URL, model, digest."""
        return f"{self.backend}|{self.model}|{self.digest or ''}"

    @property
    def think_style(self) -> ThinkStyle:
        """How this backend is told not to think."""
        return THINK_STYLES[self.kind]

    @property
    def estimate_ratio(self) -> float:
        """Multiplier for Kestrel's token estimate (a safety margin until calibrated)."""
        if self.samples < CALIBRATED_SAMPLES:
            return max(self.token_ratio, SAFETY_RATIO)
        return self.token_ratio

    def with_sample(self, reported: int, estimated: int) -> "Profile":
        """Fold one measurement (backend prompt tokens vs Kestrel's raw estimate) in."""
        if reported <= 0 or estimated <= 0:
            return self
        sample = min(RATIO_MAX, max(RATIO_MIN, reported / estimated))
        ratio = (
            sample
            if self.samples == 0
            else ((1 - RATIO_SMOOTHING) * self.token_ratio + RATIO_SMOOTHING * sample)
        )
        return self.model_copy(update={"token_ratio": round(ratio, 3), "samples": self.samples + 1})


NAME_KINDS: tuple[tuple[str, BackendKind], ...] = (
    ("ollama", "ollama"),
    ("llamacpp", "llamacpp"),
    ("llama.cpp", "llamacpp"),
    ("llama-cpp", "llamacpp"),
    ("lmstudio", "lmstudio"),
    ("vllm", "vllm"),
    ("mlx", "mlx"),
)


def backend_kind(name: str, base_url: str) -> BackendKind:
    """Guess the backend from the provider's name, else its port."""
    lowered = name.lower()
    for part, kind in NAME_KINDS:
        if part in lowered:
            return kind
    port = urlsplit(base_url).port
    return PORT_KINDS.get(port or 0, "other")


def is_local(base_url: str) -> bool:
    """Only backends on this machine are asked for metadata (no hidden network calls)."""
    return (urlsplit(base_url).hostname or "").lower() in LOOPBACK_HOSTS


def cache_path(env: Mapping[str, str]) -> Path:
    """`$KESTREL_CACHE_DIR`, `$XDG_CACHE_HOME/kestrel`, else `~/.cache/kestrel` (gap 4)."""
    if override := env.get("KESTREL_CACHE_DIR"):
        root = Path(override).expanduser()
    elif xdg := env.get("XDG_CACHE_HOME"):
        root = Path(xdg) / "kestrel"
    else:
        root = Path.home() / ".cache" / "kestrel"
    return root / "profiles.json"


class ProfileCache:
    """The profiles file. Small and rewritten whole; a broken file is ignored."""

    def __init__(self, path: Path) -> None:
        """Profiles live in `path` (created on the first save)."""
        self.path = path

    def get(self, backend: str, model: str) -> Profile | None:
        """The newest profile for this backend and model, whatever its digest."""
        matches = [p for p in self._load().values() if p.backend == backend and p.model == model]
        return max(matches, key=lambda p: p.probed_at) if matches else None

    def save(self, profile: Profile) -> None:
        """Store `profile`, replacing older ones for the same backend and model."""
        profiles = {
            k: p
            for k, p in self._load().items()
            if not (p.backend == profile.backend and p.model == profile.model)
        }
        profiles[profile.key] = profile
        data = {k: p.model_dump(mode="json") for k, p in profiles.items()}
        temp: Path | None = None
        try:  # a cache: losing a write costs one more probe, never the run
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                "w", dir=self.path.parent, suffix=".tmp", delete=False, encoding="utf-8"
            ) as handle:
                temp = Path(handle.name)  # unique, so two Kestrel processes never clash
                handle.write(json.dumps(data, indent=1))
            os.replace(temp, self.path)
        except OSError:
            if temp is not None:
                with contextlib.suppress(OSError):
                    temp.unlink()

    def _load(self) -> dict[str, Profile]:
        try:
            raw: JSONValue = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        profiles: dict[str, Profile] = {}
        for key, value in _obj(raw).items():
            try:
                profiles[key] = Profile.model_validate(value)
            except ValueError:
                continue
        return profiles


async def probe(
    name: str,
    base_url: str,
    model: str,
    *,
    known: Profile | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Profile:
    """Ask the backend about `model`. Never raises: whatever cannot be learned stays
    unknown. `known` keeps its token measurements when the digest is unchanged."""
    kind = backend_kind(name, base_url)
    fields: dict[str, object] = {}
    if not is_local(base_url):
        kind = "other"  # remote: never asked; nothing is assumed
    async with httpx.AsyncClient(timeout=METADATA_TIMEOUT, transport=transport) as client:
        try:
            if kind == "ollama":
                fields = await _ollama(client, _root(base_url), model)
            elif kind == "llamacpp":
                fields = await _llamacpp(client, _root(base_url))
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            fields = {}  # metadata is a bonus; the run works without it
    profile = Profile(
        backend=base_url,
        model=model,
        kind=kind,
        probed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        **fields,  # type: ignore[arg-type]
    )
    if known is not None and known.digest == profile.digest:
        profile = profile.model_copy(
            update={"token_ratio": known.token_ratio, "samples": known.samples}
        )
        if profile.window is None:  # not loaded right now: keep what was seen before
            profile = profile.model_copy(update={"window": known.window})
    return profile


def _root(base_url: str) -> str:
    """`http://localhost:11434/v1` to `http://localhost:11434`."""
    return base_url.removesuffix("/").removesuffix("/v1")


async def _get(client: httpx.AsyncClient, url: str, body: JSONObject | None = None) -> JSONObject:
    response = await (client.post(url, json=body) if body is not None else client.get(url))
    data: JSONValue = response.raise_for_status().json()
    return _obj(data)


def _obj(value: JSONValue) -> JSONObject:
    return value if isinstance(value, dict) else {}


def _named(listing: JSONObject, model: str) -> JSONObject:
    """The entry for `model` in an Ollama listing (`models: [{name: ...}]`)."""
    entries = listing.get("models")
    for entry in entries if isinstance(entries, list) else []:
        item = _obj(entry)
        if item.get("name") in (model, f"{model}:latest"):
            return item
    return {}


async def _ollama(client: httpx.AsyncClient, root: str, model: str) -> dict[str, object]:
    fields: dict[str, object] = {}
    show = await _get(client, f"{root}/api/show", {"model": model})
    capabilities = show.get("capabilities")
    if isinstance(capabilities, list):
        fields["native_tools"] = "tools" in capabilities
        fields["thinking"] = "thinking" in capabilities
    for key, value in _obj(show.get("model_info")).items():
        if key.endswith(".context_length") and isinstance(value, int):
            fields["max_window"] = value
    window = _named(await _get(client, f"{root}/api/ps"), model).get("context_length")
    if not isinstance(window, int) and fields.get("native_tools") is not None:
        # Not loaded: load it (no prompt; the run needs it loaded anyway) and ask again,
        # so the first run already knows the real window.
        load = await client.post(
            f"{root}/api/generate", json={"model": model}, timeout=LOAD_TIMEOUT
        )
        load.raise_for_status()
        window = _named(await _get(client, f"{root}/api/ps"), model).get("context_length")
    if isinstance(window, int):
        fields["window"] = window
    digest = _named(await _get(client, f"{root}/api/tags"), model).get("digest")
    if isinstance(digest, str):
        fields["digest"] = digest[:16]
    return fields


async def _llamacpp(client: httpx.AsyncClient, root: str) -> dict[str, object]:
    props = await _get(client, f"{root}/props")
    settings = _obj(props.get("default_generation_settings"))
    window = settings.get("n_ctx", props.get("n_ctx"))
    fields: dict[str, object] = {"native_tools": True}
    if isinstance(window, int):
        fields["window"] = window
    return fields
