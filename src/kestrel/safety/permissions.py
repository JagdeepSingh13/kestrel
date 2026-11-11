"""Permissions: allow, confirm or deny each tool call (spec 8.3, CLAUDE.md security).

Each tool has a default level (`ToolSpec.permission`). An agent file may override it;
loosening a default needs `--trust-agent` (checked by the agent loader, gap 7). This
class applies the final levels during a run and asks the user for `confirm` calls.
"""

from collections.abc import Awaitable, Callable, Mapping

from kestrel.core.types import ConfirmAnswer, PermissionDecision, ToolCall, ToolSpec

type Level = str  # "allow" | "confirm" | "deny"
type Confirm = Callable[[ToolCall, ToolSpec], Awaitable[ConfirmAnswer]]

STRICTNESS = {"allow": 0, "confirm": 1, "deny": 2}


class Permissions:
    """A `PermissionGate` for one run."""

    def __init__(self, confirm: Confirm, overrides: Mapping[str, Level] | None = None) -> None:
        """`confirm` asks the user; `overrides` are the agent file's `permissions:`."""
        self._confirm = confirm
        self._overrides = dict(overrides or {})
        self._always: set[str] = set()
        """Tools the user answered "always" for. Forgotten when the run ends."""

    def level(self, call: ToolCall, spec: ToolSpec) -> Level:
        """The level that applies to this exact call."""
        level = self._overrides.get(spec.name, spec.permission)
        flag = spec.confirm_if_true
        if level == "allow" and flag is not None and call.arguments.get(flag) is True:
            return "confirm"
        return level

    async def check(self, call: ToolCall, spec: ToolSpec) -> PermissionDecision:
        """Decide; asks the user only for `confirm` calls not yet allowed "always"."""
        level = self.level(call, spec)
        if level == "deny":
            return PermissionDecision(allowed=False, reason="denied by the agent's settings")
        if level == "allow":
            return PermissionDecision(allowed=True, reason="allowed")
        if spec.name in self._always:
            return PermissionDecision(allowed=True, reason="allowed always for this run")
        answer = await self._confirm(call, spec)
        if answer == "always":
            self._always.add(spec.name)
            return PermissionDecision(allowed=True, reason="user allowed always for this run")
        if answer == "yes":
            return PermissionDecision(allowed=True, reason="user allowed")
        return PermissionDecision(allowed=False, reason="user said no")


def loosened(defaults: Mapping[str, Level], overrides: Mapping[str, Level]) -> list[str]:
    """Tools whose override is less strict than their default, e.g. run_shell: allow."""
    return sorted(
        name
        for name, level in overrides.items()
        if name in defaults and STRICTNESS[level] < STRICTNESS[defaults[name]]
    )
