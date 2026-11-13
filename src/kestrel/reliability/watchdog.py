"""Watchdog (spec 5.3, M4 piece 3): notice when a run goes in circles.

The loop reports every tool call with its result, and the backend's prompt token count.
The watchdog answers with signals:
- `warn`: the model gets a short note after the tool result, once per signal, and a
  `loop_warning` event is logged;
- `stuck`: the run cannot make progress. M4 stops it (`stopped_reason=loop`); M4.5 steps
  up to a bigger model instead, through the run's model ladder.

Signals: `repeat` (same call, same arguments, in a row), `oscillation` (A, B, A, B),
`stall` (the same call keeps giving the same result), `tool_errors` (failures in a
row), `context_overflow` (even after trimming, the prompt does not fit the window;
Ollama would silently drop its start). `repairs_exhausted` comes from the loop.
After the ladder steps up, the loop calls `reset()` so the next model starts
with a clean slate.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

from kestrel.core.types import ToolCall, ToolResult

CONTEXT_STUCK_SHARE = 0.95


@dataclass(frozen=True)
class WatchLimits:
    """Thresholds; the agent file sets them under `limits`."""

    max_repeats: int = 4
    """Identical calls in a row: a warning one before this, stuck at this."""
    max_stall_steps: int = 5
    """Steps in a row that repeat an earlier call and get its earlier result: a warning
    two before this, stuck at this."""
    max_tool_errors: int = 3
    """Failed tool calls in a row: stuck."""


@dataclass(frozen=True)
class Signal:
    """Something the loop should act on."""

    level: Literal["warn", "stuck"]
    signal: str
    """repeat, oscillation, stall, tool_errors, context_overflow, repairs_exhausted."""
    detail: str
    note: str | None = None
    """For warnings: the text added after the tool result, for the model."""


@dataclass
class Watchdog:
    """Per-run state. Not thread-safe; one per run."""

    limits: WatchLimits = field(default_factory=WatchLimits)
    _keys: list[str] = field(default_factory=list[str])
    _seen: set[tuple[str, str]] = field(default_factory=set[tuple[str, str]])
    _stall: int = 0
    _errors: int = 0
    _warned: set[str] = field(default_factory=set[str])

    def after_tool(self, call: ToolCall, result: ToolResult) -> list[Signal]:
        """Record one call and its result; return what it means, worst first."""
        key = f"{call.name} {json.dumps(call.arguments, sort_keys=True)}"
        self._keys.append(key)
        digest = hashlib.sha256(result.content.encode("utf-8")).hexdigest()
        # Nothing new: this exact call already gave this exact result. Different files
        # with equal content, or the same file after a write, are progress.
        self._stall = self._stall + 1 if (key, digest) in self._seen else 0
        self._seen.add((key, digest))
        self._errors = self._errors + 1 if result.is_error else 0

        signals: list[Signal] = []
        repeats = self._trailing_repeats()
        limits = self.limits
        if repeats >= limits.max_repeats:
            signals.append(Signal("stuck", "repeat", f"{call.name} called {repeats} times"))
        elif repeats >= max(2, limits.max_repeats - 1):
            signals += self._warn(
                "repeat",
                f"{call.name} called {repeats} times",
                f"(Kestrel: you called {call.name} with the same arguments {repeats} times "
                "in a row. The result will not change. Try a different approach or give "
                "your final answer.)",
            )
        else:
            self._warned.discard("repeat")
        if self._errors >= limits.max_tool_errors:
            signals.append(
                Signal("stuck", "tool_errors", f"{self._errors} failed tool calls in a row")
            )
        if self._stall >= limits.max_stall_steps:
            signals.append(Signal("stuck", "stall", f"{self._stall} steps without a new result"))
        elif self._stall >= max(2, limits.max_stall_steps - 2):
            signals += self._warn(
                "stall",
                f"{self._stall} steps without a new result",
                f"(Kestrel: the last {self._stall} steps produced nothing new. Try something "
                "different or give your final answer.)",
            )
        else:
            self._warned.discard("stall")
        if pair := self._oscillation():
            first, second = pair
            signals += self._warn(
                "oscillation",
                f"alternating {first} and {second}",
                f"(Kestrel: you are going back and forth between {first} and {second}. "
                "Step back, try a different approach or give your final answer.)",
            )
        else:
            self._warned.discard("oscillation")
        return sorted(signals, key=lambda s: s.level != "stuck")

    def after_prompt(
        self, *, over_budget: bool, prompt_tokens: int | None, window: int
    ) -> Signal | None:
        """Stuck when trimming could not make the prompt fit and the backend confirms it
        (or does not say). A prompt that trimming handled is fine, however full."""
        if not over_budget:
            return None
        if prompt_tokens is not None and prompt_tokens < CONTEXT_STUCK_SHARE * window:
            return None
        used = "unknown" if prompt_tokens is None else str(prompt_tokens)
        return Signal("stuck", "context_overflow", f"prompt used {used} of {window} tokens")

    def reset(self) -> None:
        """Forget the history (after a stuck signal was handled, e.g. a step up)."""
        self._keys.clear()
        self._seen.clear()
        self._stall = 0
        self._errors = 0
        self._warned.clear()

    def _trailing_repeats(self) -> int:
        last = self._keys[-1]
        count = 0
        for key in reversed(self._keys):
            if key != last:
                break
            count += 1
        return count

    def _oscillation(self) -> tuple[str, str] | None:
        if len(self._keys) < 4:
            return None
        a, b, c, d = self._keys[-4:]
        if a == c and b == d and a != b:
            return _short(a), _short(b)
        return None

    def _warn(self, signal: str, detail: str, note: str) -> list[Signal]:
        if signal in self._warned:
            return []
        self._warned.add(signal)
        return [Signal("warn", signal, detail, note)]


def _short(key: str, limit: int = 60) -> str:
    """`read_file {"path": "a.txt"}`, cut to fit a note."""
    return key if len(key) <= limit else key[: limit - 3] + "..."
