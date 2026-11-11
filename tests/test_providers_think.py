"""`<think>` splitting: reasoning text inside content, possibly split across stream pieces."""

import pytest

from kestrel.providers.think import ThinkSplitter, split_think


def feed_all(pieces: list[str]) -> tuple[str, str]:
    splitter = ThinkSplitter()
    reasoning, content = "", ""
    for piece in pieces:
        r, c = splitter.feed(piece)
        reasoning, content = reasoning + r, content + c
    r, c = splitter.flush()
    return reasoning + r, content + c


def test_no_tags_is_all_content() -> None:
    assert split_think("Hello <b>world</b> 3 < 4") == ("", "Hello <b>world</b> 3 < 4")


def test_qwen_style_block_is_split_and_gap_trimmed() -> None:
    text = "<think>\nUser says hi.\n</think>\n\nHello!"
    assert split_think(text) == ("\nUser says hi.\n", "Hello!")


@pytest.mark.parametrize(
    "pieces",
    [
        ["<think>plan</think>answer"],
        ["<thi", "nk>pl", "an</th", "ink>ans", "wer"],
        list("<think>plan</think>answer"),  # one character at a time
    ],
)
def test_tags_split_across_pieces(pieces: list[str]) -> None:
    assert feed_all(pieces) == ("plan", "answer")


def test_content_is_released_as_soon_as_it_cannot_be_a_tag() -> None:
    splitter = ThinkSplitter()
    assert splitter.feed("Hi ") == ("", "Hi ")
    assert splitter.feed("<") == ("", "")  # might be the start of <think>
    assert splitter.feed("b>") == ("", "<b>")  # it wasn't


def test_unclosed_think_is_reasoning() -> None:
    # Generation cut off mid-thought (max_tokens).
    assert feed_all(["<think>still thin", "king"]) == ("still thinking", "")


def test_text_before_think_is_content() -> None:
    assert split_think("Sure. <think>hmm</think>Done.") == ("hmm", "Sure. Done.")
