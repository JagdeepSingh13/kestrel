"""User configuration: `config.toml` plus environment overrides.

Location (spec section 12, gap 4):
1. `$KESTREL_CONFIG` if set.
2. Windows: `%APPDATA%\\kestrel\\config.toml`.
3. Otherwise `$XDG_CONFIG_HOME/kestrel/config.toml`, default `~/.config/kestrel/config.toml`.

A missing file is fine: the built-in backend list is used. API keys are never stored in
the file; a provider names the environment variable that holds its key (`api_key_env`).
"""

import os
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from kestrel.core.types import JSONObject

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class ConfigError(Exception):
    """The config file exists but cannot be used. The message says where and why."""


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProviderConfig(_Strict):
    """One backend Kestrel can talk to."""

    base_url: str
    api_key_env: str | None = None
    """Name of the environment variable holding the API key. Never the key itself."""

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def is_remote(self) -> bool:
        """True when requests leave this machine (any host other than loopback)."""
        host = urlsplit(self.base_url).hostname or ""
        return host.lower() not in LOOPBACK_HOSTS

    def api_key(self, env: Mapping[str, str]) -> str | None:
        """Read the API key from the environment, if this provider uses one."""
        if self.api_key_env is None:
            return None
        return env.get(self.api_key_env) or None


# Default ports of common local backends (spec section 4.2). mlx-lm also defaults to
# 8080; discovery tells llama.cpp and mlx-lm apart (gap 2).
BUILTIN_PROVIDERS: dict[str, ProviderConfig] = {
    "ollama": ProviderConfig(base_url="http://localhost:11434/v1"),
    "lmstudio": ProviderConfig(base_url="http://localhost:1234/v1"),
    "llamacpp": ProviderConfig(base_url="http://localhost:8080/v1"),
    "vllm": ProviderConfig(base_url="http://localhost:8000/v1"),
}


class KestrelConfig(_Strict):
    """Everything read from `config.toml`, merged over built-in defaults."""

    default_agent: str = "default"
    default_provider: str = "ollama"
    default_model: str | None = None
    providers: dict[str, ProviderConfig] = Field(default_factory=lambda: dict(BUILTIN_PROVIDERS))
    tools: dict[str, JSONObject] = Field(default_factory=dict[str, JSONObject])
    """Per-tool settings (M4+). Kept as plain JSON until those tools exist."""
    mcp_servers: list[JSONObject] = Field(default_factory=list[JSONObject])
    """MCP server definitions (M5). Kept as plain JSON until then."""

    def provider(self, name: str | None = None) -> ProviderConfig:
        """Look up a provider by name (default: `default_provider`)."""
        key = name or self.default_provider
        try:
            return self.providers[key]
        except KeyError:
            known = ", ".join(sorted(self.providers))
            raise ConfigError(f"Unknown provider {key!r}. Known providers: {known}.") from None


def default_config_path(env: Mapping[str, str], platform: str = sys.platform) -> Path:
    """Where Kestrel looks for `config.toml` (see module docstring)."""
    if override := env.get("KESTREL_CONFIG"):
        return Path(override).expanduser()
    if platform == "win32" and (appdata := env.get("APPDATA")):
        return Path(appdata) / "kestrel" / "config.toml"
    if xdg := env.get("XDG_CONFIG_HOME"):
        return Path(xdg) / "kestrel" / "config.toml"
    return Path.home() / ".config" / "kestrel" / "config.toml"


def load_config(path: Path | None = None, env: Mapping[str, str] | None = None) -> KestrelConfig:
    """Load config from `path` (default: `default_config_path`), merged over built-ins.

    Raises `ConfigError` for unreadable TOML, unknown keys, or an API key written into
    the file.
    """
    env = os.environ if env is None else env
    path = path or default_config_path(env)
    if not path.is_file():
        return KestrelConfig()

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc

    providers_raw = raw.get("providers", {})
    if isinstance(providers_raw, dict):
        for name, table in providers_raw.items():  # pyright: ignore[reportUnknownVariableType]
            if isinstance(table, dict) and "api_key" in table:
                raise ConfigError(
                    f"{path}: [providers.{name}] contains api_key. Never store keys in the "
                    f"file: put it in an environment variable and set api_key_env instead."
                )

    try:
        user = KestrelConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {_describe(exc)}") from exc

    # User-defined providers replace built-ins of the same name; others are kept.
    return user.model_copy(update={"providers": {**BUILTIN_PROVIDERS, **user.providers}})


def _describe(exc: ValidationError) -> str:
    """One readable line per problem, e.g. `providers.ollama.base_url: Field required`."""
    lines: list[str] = []
    for err in exc.errors():
        where = ".".join(str(part) for part in err["loc"]) or "(top level)"
        lines.append(f"{where}: {err['msg']}")
    return "; ".join(lines)
