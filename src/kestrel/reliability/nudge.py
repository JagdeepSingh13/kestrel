"""Nudge (M4.2): catch a reply that says it will act but calls no tool.

Small models often end a turn with "Let me do that for you." and stop. When a reply
with no tool call ends by announcing an action, the loop keeps the reply, adds one short
note and asks again, once per run. Questions to the user and plain answers pass through.
"""

import re

from kestrel.core.context import NOTE_PREFIX

NOTE = (
    f"{NOTE_PREFIX} you said you would do something but made no tool call, so nothing "
    "happened. If a tool is needed, call it now. Otherwise give your final answer.)"
)

_PROMISE = re.compile(
    r"\b(let me|i will|i'll|i am going to|i'm going to|next,? i|now i)\b", re.IGNORECASE
)
_POLITE = re.compile(
    r"\b(let me know|i'll be (happy|glad)|i will be (happy|glad)|i hope)\b", re.IGNORECASE
)
LAST_SENTENCES = 3
"""Only the end of the reply counts: where a model announces its next step."""


def promises_action(text: str | None) -> bool:
    """True when the end of `text` announces an action rather than answering or asking."""
    if not text or text.rstrip().endswith("?"):
        return False  # a question to the user is a fine way to end a turn
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip().replace("\u2019", "'"))
    tail = " ".join(sentences[-LAST_SENTENCES:])
    return _PROMISE.search(_POLITE.sub("", tail)) is not None
