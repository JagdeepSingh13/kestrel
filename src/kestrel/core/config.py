"""User configuration: `config.toml` plus environment overrides.

Location (spec section 12, gap 4):
1. `$KESTREL_CONFIG` if set.
2. Windows: `%APPDATA%\\kestrel\\config.toml`.
3. Otherwise `$XDG_CONFIG_HOME/kestrel/config.toml`, default `~/.config/kestrel/config.toml`.

A missing file is fine: the built-in backend list is used. API keys are never stored in
the file; a provider names the environment variable that holds its key (`api_key_env`).
"""

import os
import re
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

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


class McpServerConfig(_Strict):
    """One MCP server: a command to start (stdio), or a URL (HTTP)."""

    command: str | None = None
    args: list[str] = Field(default_factory=list[str])
    """`{workspace}` is replaced by the workspace folder."""
    url: str | None = None
    env_pass: list[str] = Field(default_factory=list[str])
    """Environment variables passed to the server, by name (values never in the file)."""
    allow: list[str] = Field(default_factory=list[str])
    """Tools that run without asking (your config is trusted); the rest ask first."""
    timeout: float = Field(default=30.0, gt=0, le=600)
    """Seconds per tool call."""

    @model_validator(mode="after")
    def _one_way_in(self) -> "McpServerConfig":
        if (self.command is None) == (self.url is None):
            raise ValueError("give either command (a program to start) or url, not both")
        return self

    @property
    def is_remote(self) -> bool:
        """True for an HTTP server off this machine."""
        if self.url is None:
            return False
        host = urlsplit(self.url).hostname or ""
        return host not in LOOPBACK_HOSTS


MCP_SERVER_NAME = r"^[a-z0-9-]{1,16}$"


class KestrelConfig(_Strict):
    """Everything read from `config.toml`, merged over built-in defaults."""

    default_agent: str = "default"
    default_provider: str = "ollama"
    default_model: str | None = None
    providers: dict[str, ProviderConfig] = Field(default_factory=lambda: dict(BUILTIN_PROVIDERS))
    tools: dict[str, JSONObject] = Field(default_factory=dict[str, JSONObject])
    """Per-tool settings (M4+). Kept as plain JSON until those tools exist."""
    mcp_servers: dict[str, McpServerConfig] = Field(default_factory=dict[str, McpServerConfig])
    """MCP servers (M5), by name. Only defined here, never in agent files."""

    @field_validator("mcp_servers")
    @classmethod
    def _server_names(cls, value: dict[str, McpServerConfig]) -> dict[str, McpServerConfig]:
        bad = [name for name in value if not re.fullmatch(MCP_SERVER_NAME, name)]
        if bad:
            raise ValueError(f"MCP server names must be 1 to 16 of a-z, 0-9 and -: {bad}")
        return value

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


def default_data_dir(env: Mapping[str, str]) -> Path:
    """Where Kestrel keeps its own files (gap 4): `$KESTREL_DATA_DIR`, else
    `$XDG_DATA_HOME/kestrel`, else `~/.local/share/kestrel`, on every OS."""
    if override := env.get("KESTREL_DATA_DIR"):
        return Path(override).expanduser()
    if xdg := env.get("XDG_DATA_HOME"):
        return Path(xdg) / "kestrel"
    return Path.home() / ".local" / "share" / "kestrel"


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
