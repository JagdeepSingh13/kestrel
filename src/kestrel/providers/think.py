"""Separate `<think>...</think>` reasoning from answer text.

Some reasoning models (qwen3, deepseek-r1) put their thinking inside the content as
`<think>` tags instead of a separate field. Kestrel keeps reasoning apart so the
answer, and later the tool-call parser, only sees the answer.

Streaming makes this tricky: a tag can arrive split across pieces (`"<thi"` + `"nk>"`).
`ThinkSplitter` holds back only the few characters that could still become a tag.
"""

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


class ThinkSplitter:
    """Incrementally split text into (reasoning, content). Feed pieces, then flush."""

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""
        self._trim_next_content = False

    def feed(self, text: str) -> tuple[str, str]:
        """Add a piece of text; return the (reasoning, content) that is now certain."""
        self._pending += text
        reasoning, content = "", ""
        while True:
            tag = CLOSE_TAG if self._in_think else OPEN_TAG
            index = self._pending.find(tag)
            if index < 0:
                keep = _partial_tag_suffix(self._pending, tag)
                ready = self._pending[: len(self._pending) - keep]
                self._pending = self._pending[len(ready) :]
                r, c = self._emit(ready)
                return reasoning + r, content + c
            r, c = self._emit(self._pending[:index])
            reasoning, content = reasoning + r, content + c
            self._pending = self._pending[index + len(tag) :]
            self._in_think = not self._in_think
            if not self._in_think:
                # Models put blank lines between </think> and the answer; drop them.
                self._trim_next_content = True

    def flush(self) -> tuple[str, str]:
        """End of text: release anything held back. An unclosed <think> is reasoning."""
        rest, self._pending = self._pending, ""
        return self._emit(rest)

    def _emit(self, text: str) -> tuple[str, str]:
        if self._in_think:
            return text, ""
        if self._trim_next_content:
            text = text.lstrip()
            if text:
                self._trim_next_content = False
        return "", text


def split_think(text: str) -> tuple[str, str]:
    """Split a complete text into (reasoning, content)."""
    splitter = ThinkSplitter()
    r1, c1 = splitter.feed(text)
    r2, c2 = splitter.flush()
    return r1 + r2, c1 + c2


def _partial_tag_suffix(text: str, tag: str) -> int:
    """Length of the longest end of `text` that is the start of `tag` (e.g. "<thi")."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-length:]):
            return length
    return 0
