"""Find and load agent files, with line numbers in every error (spec section 10).

Search order for a name: `<project>/agents/`, `~/.config/kestrel/agents/`, built-ins.
Loosening a tool's default permission (e.g. `run_shell: allow`) is refused unless the
user starts the run with --trust-agent (gap 7), whichever folder the file came from.
"""

from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from kestrel.agents.schema import AgentConfig
from kestrel.core.config import default_config_path
from kestrel.safety.paths import find_project_root
from kestrel.safety.permissions import loosened
from kestrel.tools.builtin import BUILTIN_TOOLS


class AgentError(Exception):
    """An agent file cannot be used. The message says which file, which line, and why."""


def agent_dirs(env: Mapping[str, str]) -> list[Path]:
    """Folders searched for `<name>.yaml`, in order (built-ins come after these)."""
    dirs: list[Path] = []
    project = find_project_root(Path(__file__))
    if project is not None:
        dirs.append(project / "agents")
    dirs.append(default_config_path(env).parent / "agents")
    return dirs


def list_agents(env: Mapping[str, str]) -> dict[str, str]:
    """Every agent name found, with where it comes from (first match wins)."""
    found: dict[str, str] = {}
    for folder in agent_dirs(env):
        for file in sorted(folder.glob("*.yaml")) if folder.is_dir() else []:
            found.setdefault(file.stem, str(file))
    for name in _builtin_names():
        found.setdefault(name, "built-in")
    return found


def load_agent(name: str, env: Mapping[str, str], *, trusted: bool = False) -> AgentConfig:
    """Load an agent by name (or by path to a .yaml file) and check it completely."""
    if name.endswith((".yaml", ".yml")):
        path = Path(name).expanduser()
        if not path.is_file():
            raise AgentError(f"{name}: no such file")
        return parse_agent(path.read_text(encoding="utf-8"), str(path), trusted=trusted)
    for folder in agent_dirs(env):
        path = folder / f"{name}.yaml"
        if path.is_file():
            return parse_agent(path.read_text(encoding="utf-8"), str(path), trusted=trusted)
    if name in _builtin_names():
        text = (resources.files("kestrel.agents") / "builtin" / f"{name}.yaml").read_text("utf-8")
        return parse_agent(text, f"built-in {name}", trusted=trusted)
    known = ", ".join(sorted(list_agents(env))) or "none"
    raise AgentError(f"No agent named {name!r}. Known agents: {known}")


def parse_agent(text: str, source: str, *, trusted: bool = False) -> AgentConfig:
    """Check agent YAML text. `source` names the file in error messages."""
    try:
        raw: object = yaml.safe_load(text)
        root = _compose(text)
    except yaml.YAMLError as exc:
        raise AgentError(f"{source}: not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise AgentError(f"{source}: an agent file must be a mapping of settings")
    try:
        agent = AgentConfig.model_validate(raw)
    except ValidationError as exc:
        problems = [
            f"{source}{_line(root, err['loc'])}: {'.'.join(str(x) for x in err['loc'])}: "
            f"{err['msg']}"
            for err in exc.errors()
        ]
        raise AgentError("\n".join(problems)) from exc

    unknown = [t for t in agent.tools if t not in BUILTIN_TOOLS]
    if unknown:
        known = ", ".join(BUILTIN_TOOLS)
        raise AgentError(
            f"{source}{_line(root, ('tools',))}: unknown tools {unknown}. Available: {known}"
        )
    stray = [t for t in agent.permissions if t not in agent.tools]
    if stray:
        raise AgentError(
            f"{source}{_line(root, ('permissions',))}: permissions for tools the agent "
            f"does not use: {stray}"
        )
    defaults = {name: BUILTIN_TOOLS[name].spec.permission for name in agent.tools}
    loose = loosened(defaults, agent.permissions)
    if loose and not trusted:
        details = ", ".join(f"{t}: {defaults[t]} to {agent.permissions[t]}" for t in loose)
        raise AgentError(
            f"{source} loosens permissions ({details}). Read the file, and if you trust "
            "it, start the run with --trust-agent."
        )
    return agent


def _compose(text: str) -> yaml.Node | None:
    """The YAML node tree, which remembers line numbers (safe_load forgets them)."""
    loader = yaml.SafeLoader(text)
    try:
        return loader.get_single_node()
    finally:
        loader.dispose()


def _builtin_names() -> list[str]:
    folder = resources.files("kestrel.agents") / "builtin"
    return sorted(
        p.name.removesuffix(".yaml") for p in folder.iterdir() if p.name.endswith(".yaml")
    )


def _line(node: yaml.Node | None, loc: tuple[int | str, ...]) -> str:
    """`:12` for the line of the setting at `loc`, or "" if it cannot be found."""
    found = node
    for key in loc:
        if isinstance(found, yaml.MappingNode):
            found = next((v for k, v in found.value if k.value == key), None)
        elif isinstance(found, yaml.SequenceNode) and isinstance(key, int):
            found = found.value[key] if key < len(found.value) else None
        else:
            break
        if found is None:
            break
    target = found or node
    return f":{target.start_mark.line + 1}" if target is not None else ""
