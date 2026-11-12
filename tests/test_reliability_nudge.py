"""Seam: promises_action() on replies that end a turn, taken from real runs."""

import pytest

from kestrel.reliability.nudge import promises_action

ANNOUNCED = [
    # qwen3:8b, 2026-11-12, after write_file refused to overwrite
    "I apologize for the confusion. To append the new entry to the `sales.csv` file without "
    "overwriting the existing data, I need to open the file, add the new entry, and save it. "
    "Let me do that for you.",
    "The file has 3 rows. Now I will add the total at the end.",
    "I'll read the config next.",
]
FINISHED = [
    "Done. I added the row; data/sales.csv now has 2 lines.",
    "The total is 160. Let me know if you need anything else.",
    "Which file should I change: sales.csv or totals.csv?",
    "Hello! I'll be happy to help with your files.",
    "",
]


@pytest.mark.parametrize("reply", ANNOUNCED)
def test_a_reply_that_only_announces_an_action_is_caught(reply: str) -> None:
    assert promises_action(reply)


@pytest.mark.parametrize("reply", FINISHED)
def test_answers_and_questions_pass(reply: str) -> None:
    assert not promises_action(reply)
