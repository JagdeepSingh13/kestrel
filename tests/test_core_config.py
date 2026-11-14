"""Seam 6: load_config(path, env) and default_config_path(env)."""

from pathlib import Path

import pytest

from kestrel.core.config import (
    ConfigError,
    KestrelConfig,
    ProviderConfig,
    default_config_path,
    load_config,
)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_file_gives_builtin_backends(tmp_path: Path) -> None:
    config = load_config(tmp_path / "nope.toml", env={})
    assert config.default_provider == "ollama"
    assert config.provider().base_url == "http://localhost:11434/v1"
    assert set(config.providers) == {"ollama", "lmstudio", "llamacpp", "vllm"}


def test_file_overrides_and_adds_providers(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
default_model = "qwen3:8b"

[providers.ollama]
base_url = "http://localhost:11500/v1/"

[providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
""",
    )
    config = load_config(path, env={})
    assert config.default_model == "qwen3:8b"
    assert config.provider("ollama").base_url == "http://localhost:11500/v1"  # slash stripped
    assert config.provider("lmstudio").base_url == "http://localhost:1234/v1"  # built-in kept
    assert config.provider("openrouter").api_key_env == "OPENROUTER_API_KEY"


def test_api_key_is_read_from_named_env_var() -> None:
    provider = ProviderConfig(base_url="https://x/v1", api_key_env="MY_KEY")
    assert provider.api_key({"MY_KEY": "sk-123"}) == "sk-123"
    assert provider.api_key({}) is None
    assert provider.api_key({"MY_KEY": ""}) is None
    assert ProviderConfig(base_url="http://localhost:1/v1").api_key({"MY_KEY": "x"}) is None


def test_api_key_written_in_file_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, '[providers.x]\nbase_url = "https://x/v1"\napi_key = "sk-oops"\n')
    with pytest.raises(ConfigError, match="Never store keys"):
        load_config(path, env={})


def test_invalid_toml_names_the_file(tmp_path: Path) -> None:
    path = write(tmp_path, "default_model = \n")
    with pytest.raises(ConfigError, match=r"config\.toml: invalid TOML"):
        load_config(path, env={})


def test_unknown_keys_are_reported_with_location(tmp_path: Path) -> None:
    path = write(tmp_path, '[providers.ollama]\nbase_url = "http://localhost:11434/v1"\nport = 1\n')
    with pytest.raises(ConfigError, match=r"providers\.ollama\.port"):
        load_config(path, env={})


def test_spec_example_sections_are_accepted(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
[tools.web_search]
backend = "searxng"
searxng_url = "http://localhost:8888"

[mcp_servers.fs]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "{workspace}"]
allow = ["read_text_file", "list_directory"]
""",
    )
    config = load_config(path, env={})
    assert config.tools["web_search"]["backend"] == "searxng"
    assert config.mcp_servers["fs"].command == "npx"


def test_unknown_provider_lists_known_ones() -> None:
    with pytest.raises(ConfigError, match=r"Known providers: .*ollama"):
        KestrelConfig().provider("nope")


@pytest.mark.parametrize(
    ("url", "remote"),
    [
        ("http://localhost:11434/v1", False),
        ("http://127.0.0.1:8080/v1", False),
        ("http://[::1]:8000/v1", False),
        ("https://openrouter.ai/api/v1", True),
        ("http://192.168.1.20:11434/v1", True),  # another machine on the LAN
    ],
)
def test_is_remote_means_data_leaves_this_machine(url: str, remote: bool) -> None:
    assert ProviderConfig(base_url=url).is_remote is remote


def test_config_path_env_override(tmp_path: Path) -> None:
    custom = tmp_path / "custom.toml"
    assert default_config_path({"KESTREL_CONFIG": str(custom)}) == custom


def test_config_path_windows_uses_appdata() -> None:
    path = default_config_path({"APPDATA": "C:/Users/me/AppData/Roaming"}, platform="win32")
    assert path == Path("C:/Users/me/AppData/Roaming") / "kestrel" / "config.toml"


def test_config_path_xdg_then_home() -> None:
    assert default_config_path({"XDG_CONFIG_HOME": "/xdg"}, platform="linux") == Path(
        "/xdg/kestrel/config.toml"
    )
    assert default_config_path({}, platform="darwin") == (
        Path.home() / ".config" / "kestrel" / "config.toml"
    )


def test_env_override_is_used_by_load_config(tmp_path: Path) -> None:
    path = write(tmp_path, 'default_provider = "lmstudio"\n')
    assert load_config(env={"KESTREL_CONFIG": str(path)}).default_provider == "lmstudio"
