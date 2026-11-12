"""Nudges: one short note when a reply with no tool call stops too early.

Two cases (once per run in all):
- it announces an action and stops: "Let me do that for you." (M4.2);
- its last tool call failed and it gives up without trying another way (M4.3, after the
  Hookele harness: show the failure, nudge once if the model stalls).
Questions to the user pass through.
"""

import re

from kestrel.core.context import NOTE_PREFIX

NOTE = (
    f"{NOTE_PREFIX} you said you would do something but made no tool call, so nothing "
    "happened. If a tool is needed, call it now. Otherwise give your final answer.)"
)
AFTER_FAILURE = (
    f"{NOTE_PREFIX} your last tool call failed and you made no new call. If the task is "
    "not done, try another way now. If it cannot be done, say why in your final answer; "
    "never state a result that no tool gave you.)"
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


def nudge_for(reply: str | None, *, last_call_failed: bool) -> tuple[str, str] | None:
    """(kind, note) when a reply with no tool call should get one more try."""
    if promises_action(reply):
        return "promise", NOTE
    if last_call_failed and not (reply or "").rstrip().endswith("?"):
        return "after_failure", AFTER_FAILURE
    return None
