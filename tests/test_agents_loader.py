"""Agent files: checked fully, errors point at the exact line (spec section 10, gap 7)."""

from pathlib import Path

import pytest

from kestrel.agents.loader import AgentError, list_agents, load_agent, parse_agent

GOOD = """\
name: counter
description: Counts lines.
model: qwen3:8b
tools: [list_dir, read_file, write_file]
permissions:
  write_file: confirm
limits:
  max_steps: 25
"""


def test_good_file_loads_with_defaults_filled_in() -> None:
    agent = parse_agent(GOOD, "counter.yaml")
    assert agent.model == ["qwen3:8b"]  # one model becomes a one-step ladder
    assert agent.permissions == {"write_file": "confirm"}
    assert agent.limits.max_steps == 25
    assert agent.context.window == 16384
    assert agent.system is None  # Kestrel's default prompt is used


def test_errors_name_the_file_line_and_setting() -> None:
    bad = GOOD.replace("max_steps: 25", "max_steps: lots")
    with pytest.raises(AgentError) as caught:
        parse_agent(bad, "counter.yaml")
    assert str(caught.value).startswith("counter.yaml:8: limits.max_steps:")


def test_unknown_keys_and_tools_are_refused() -> None:
    with pytest.raises(AgentError, match=r"counter.yaml:1: colour: Extra inputs"):
        parse_agent("colour: blue\n" + GOOD, "counter.yaml")
    with pytest.raises(AgentError, match=r"counter.yaml:4: unknown tools \['delete_all'\]"):
        parse_agent(GOOD.replace("write_file]", "delete_all]"), "counter.yaml")
    with pytest.raises(AgentError, match="not valid YAML"):
        parse_agent("name: [unclosed", "x.yaml")


def test_permissions_must_be_for_the_agents_own_tools() -> None:
    with pytest.raises(AgentError, match=r"tools the agent does not use: \['run_shell'\]"):
        parse_agent(GOOD.replace("  write_file: confirm", "  run_shell: deny"), "counter.yaml")


def test_loosening_needs_trust_but_tightening_does_not() -> None:
    loose = "name: x\ntools: [run_shell, read_file]\npermissions:\n  run_shell: allow\n"
    with pytest.raises(AgentError, match=r"run_shell: confirm to allow.*--trust-agent"):
        parse_agent(loose, "x.yaml")
    assert parse_agent(loose, "x.yaml", trusted=True).permissions == {"run_shell": "allow"}
    tight = "name: x\ntools: [read_file]\npermissions:\n  read_file: deny\n"
    assert parse_agent(tight, "x.yaml").permissions == {"read_file": "deny"}


def test_builtin_and_project_agents_are_found(tmp_path: Path) -> None:
    env = {"KESTREL_CONFIG": str(tmp_path / "config.toml")}
    assert load_agent("default", env).tools[0] == "list_dir"
    assert load_agent("counter", env).name == "counter"  # agents/counter.yaml in the repo
    found = list_agents(env)
    assert found["default"] == "built-in"
    assert Path(found["counter"]).parts[-2:] == ("agents", "counter.yaml")
    with pytest.raises(
        AgentError, match=r"No agent named 'ghost'\. Known agents: counter, default"
    ):
        load_agent("ghost", env)


def test_user_agents_folder_and_paths_work(tmp_path: Path) -> None:
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "mine.yaml").write_text("name: mine\n", encoding="utf-8")
    env = {"KESTREL_CONFIG": str(tmp_path / "config.toml")}
    assert load_agent("mine", env).name == "mine"
    assert load_agent(str(tmp_path / "agents" / "mine.yaml"), env).name == "mine"
