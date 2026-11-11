"""Context manager: builds each prompt so it fits a small context window (spec section 7).

The full conversation is kept; `build()` returns a trimmed view of it. Trimming order:
1. Empty old tool results (keep the message, so every tool call still has its answer).
2. Drop the oldest turns.
Never touched: the system prompt, the original task (when pinned), and the latest step.
Tokens are estimated as characters / 3.5; real tokenizers may come in M4.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass

from kestrel.core.types import Message, Role, ToolSpec

CHARS_PER_TOKEN = 3.5
MESSAGE_OVERHEAD_TOKENS = 4
"""Role markers and separators the chat template adds around every message."""
DEFAULT_WINDOW = 16_384
DEFAULT_RESPONSE_RESERVE = 1_024
DEFAULT_MAX_TOOL_RESULT_TOKENS = 1_500
REMOVED_TOOL_RESULT = "[older tool result removed to save space]"


def _tokens(chars: int) -> int:
    return math.ceil(chars / CHARS_PER_TOKEN)


def estimate_tokens(messages: Sequence[Message]) -> int:
    """Rough token count of messages as the model sees them (reasoning is never sent)."""
    total = 0
    for m in messages:
        chars = len(m.content or "")
        if m.tool_calls:
            chars += len(json.dumps([c.model_dump() for c in m.tool_calls]))
        total += _tokens(chars) + MESSAGE_OVERHEAD_TOKENS
    return total


def estimate_tool_tokens(tools: Sequence[ToolSpec]) -> int:
    """Rough token count of tool descriptions sent with every request."""
    if not tools:
        return 0
    specs = [t.model_dump(include={"name", "description", "parameters"}) for t in tools]
    return _tokens(len(json.dumps(specs)))


@dataclass(frozen=True)
class Prompt:
    """What `build()` produced and how much it had to cut."""

    messages: list[Message]
    est_tokens: int
    """Estimated tokens of messages plus tool descriptions."""
    dropped_tool_results: int = 0
    dropped_messages: int = 0
    over_budget: bool = False
    """Still too big after trimming everything allowed; sent anyway."""

    @property
    def trimmed(self) -> bool:
        """True when anything was cut."""
        return bool(self.dropped_tool_results or self.dropped_messages)


class ContextManager:
    """Holds a conversation and builds prompts that fit `window` tokens."""

    def __init__(
        self,
        system_prompt: str | None,
        *,
        window: int = DEFAULT_WINDOW,
        response_reserve: int = DEFAULT_RESPONSE_RESERVE,
        max_tool_result_tokens: int = DEFAULT_MAX_TOOL_RESULT_TOKENS,
        tools: Sequence[ToolSpec] = (),
        pin_first_user: bool = True,
    ) -> None:
        """`pin_first_user` keeps the task forever (agent runs); chats pass False."""
        self.system_prompt = system_prompt
        self.window = window
        self.response_reserve = response_reserve
        self.max_tool_result_tokens = max_tool_result_tokens
        self.tools = list(tools)
        self.pin_first_user = pin_first_user
        self.history: list[Message] = []

    def add(self, message: Message) -> None:
        """Append a message. Tool results over the size limit are cut here."""
        if message.role is Role.tool and message.content:
            message = message.model_copy(update={"content": self._cut(message.content)})
        self.history.append(message)

    def clear(self) -> None:
        """Forget the conversation (the system prompt stays)."""
        self.history.clear()

    def build(self) -> Prompt:
        """The messages to send now, trimmed to fit `window - response_reserve`."""
        budget = self.window - self.response_reserve - estimate_tool_tokens(self.tools)
        system = (
            [Message(role=Role.system, content=self.system_prompt)] if self.system_prompt else []
        )
        rest = list(self.history)
        head: list[Message] = []
        if self.pin_first_user and rest and rest[0].role is Role.user:
            head, rest = rest[:1], rest[1:]
        tail_start = _latest_step_start(rest)
        middle, tail = rest[:tail_start], rest[tail_start:]

        def size() -> int:
            return estimate_tokens([*system, *head, *middle, *tail])

        dropped_results = 0
        for i, m in enumerate(middle):
            if size() <= budget:
                break
            if m.role is Role.tool and m.content != REMOVED_TOOL_RESULT:
                middle[i] = m.model_copy(update={"content": REMOVED_TOOL_RESULT})
                dropped_results += 1

        groups = _turn_groups(middle)
        dropped_messages = 0
        while groups and size() > budget:
            dropped_messages += len(groups.pop(0))
            middle = [m for group in groups for m in group]

        messages = [*system, *head, *middle, *tail]
        used = estimate_tokens(messages)
        return Prompt(
            messages=messages,
            est_tokens=used + estimate_tool_tokens(self.tools),
            dropped_tool_results=dropped_results,
            dropped_messages=dropped_messages,
            over_budget=used > budget,
        )

    def _cut(self, text: str) -> str:
        limit = int(self.max_tool_result_tokens * CHARS_PER_TOKEN)
        if len(text) <= limit:
            return text
        more = len(text) - limit
        return (
            f"{text[:limit]}\n[... truncated: {more} more characters. "
            "The full output is in the run log.]"
        )


def _latest_step_start(messages: list[Message]) -> int:
    """Index where the latest step begins: the last assistant message, else the last one."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role is Role.assistant:
            return i
    return max(len(messages) - 1, 0)


def _turn_groups(messages: list[Message]) -> list[list[Message]]:
    """Split into droppable units: each non-tool message plus the tool results after it,
    so a tool call and its answer are always dropped together."""
    groups: list[list[Message]] = []
    for m in messages:
        if m.role is Role.tool and groups:
            groups[-1].append(m)
        else:
            groups.append([m])
    return groups
