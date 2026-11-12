"""Context manager: fit the conversation into a small window (spec section 7)."""

from kestrel.core.context import (
    MESSAGE_OVERHEAD_TOKENS,
    REMOVED_TOOL_RESULT,
    ContextManager,
    estimate_tokens,
)
from kestrel.core.types import Message, Role, ToolCall

SYSTEM = "You are a helpful agent."
GOAL = "What is 17 + 25?"


def user(text: str) -> Message:
    return Message(role=Role.user, content=text)


def assistant(text: str | None = None, call: str | None = None) -> Message:
    calls = [ToolCall(id=call, name="add", arguments={"a": 1})] if call else []
    return Message(role=Role.assistant, content=text, tool_calls=calls)


def tool(call: str, text: str) -> Message:
    return Message(role=Role.tool, tool_call_id=call, name="add", content=text)


def contents(messages: list[Message]) -> list[str | None]:
    return [m.content for m in messages]


def test_estimate_is_chars_over_3_5_plus_overhead() -> None:
    assert estimate_tokens([user("x" * 35)]) == 10 + MESSAGE_OVERHEAD_TOKENS
    assert estimate_tokens([user("x" * 36)]) == 11 + MESSAGE_OVERHEAD_TOKENS  # rounds up


def test_small_conversation_is_sent_unchanged() -> None:
    ctx = ContextManager(SYSTEM, window=1000, response_reserve=100)
    ctx.add(user(GOAL))
    prompt = ctx.build()
    assert contents(prompt.messages) == [SYSTEM, GOAL]
    assert prompt.est_tokens == estimate_tokens(prompt.messages)
    assert not prompt.trimmed
    assert not prompt.over_budget


def test_huge_tool_result_is_cut_when_added() -> None:
    ctx = ContextManager(SYSTEM, max_tool_result_tokens=10)
    ctx.add(user(GOAL))
    ctx.add(assistant(call="c1"))
    ctx.add(tool("c1", "y" * 1000))
    text = ctx.build().messages[-1].content
    assert text is not None
    assert text.startswith("y" * 35)  # 10 tokens of 3.5 chars
    assert "truncated" in text
    assert "965 more characters" in text


def build_long_run(ctx: ContextManager) -> None:
    """Goal, then three tool steps with bulky results, then the latest step."""
    ctx.add(user(GOAL))
    for i in range(3):
        ctx.add(assistant(call=f"c{i}"))
        ctx.add(tool(f"c{i}", f"result {i} " + "z" * 200))
    ctx.add(assistant(call="c9"))
    ctx.add(tool("c9", "latest result"))


def test_old_tool_results_go_first_and_the_latest_step_stays() -> None:
    ctx = ContextManager(SYSTEM, window=180, response_reserve=0)
    build_long_run(ctx)
    prompt = ctx.build()
    texts = contents(prompt.messages)
    assert texts[:2] == [SYSTEM, GOAL]
    assert texts[-1] == "latest result"
    assert REMOVED_TOOL_RESULT in texts
    assert len(prompt.messages) == 10  # nothing dropped yet, only results emptied
    assert prompt.dropped_tool_results >= 1
    assert prompt.dropped_messages == 0
    assert prompt.est_tokens <= 180


def test_then_oldest_turns_are_dropped_keeping_tool_pairs_valid() -> None:
    ctx = ContextManager(SYSTEM, window=60, response_reserve=0)
    build_long_run(ctx)
    prompt = ctx.build()
    texts = contents(prompt.messages)
    assert texts[:2] == [SYSTEM, GOAL]
    assert texts[-1] == "latest result"
    assert prompt.dropped_messages > 0
    # Every tool message still follows the assistant call it answers.
    seen_calls: set[str] = set()
    for m in prompt.messages:
        seen_calls.update(c.id for c in m.tool_calls)
        if m.role is Role.tool:
            assert m.tool_call_id in seen_calls


def test_too_big_even_after_trimming_is_flagged_not_broken() -> None:
    ctx = ContextManager(SYSTEM, window=10, response_reserve=0)
    ctx.add(user(GOAL))
    ctx.add(assistant("a long latest answer " * 5))
    prompt = ctx.build()
    assert contents(prompt.messages) == [SYSTEM, GOAL, "a long latest answer " * 5]
    assert prompt.over_budget


def test_in_a_conversation_old_questions_go_but_the_newest_stays() -> None:
    ctx = ContextManager(None, window=40, response_reserve=0)
    for i in range(4):
        ctx.add(user(f"question {i} " + "q" * 40))
        ctx.add(assistant(f"answer {i}"))
    ctx.add(user("newest question"))
    texts = contents(ctx.build().messages)
    assert "question 0" not in " ".join(t or "" for t in texts)
    assert texts[-2:] == ["answer 3", "newest question"]


def test_reply_reserve_counts_against_the_window() -> None:
    ctx = ContextManager(SYSTEM, window=1000, response_reserve=990)
    build_long_run(ctx)
    assert ctx.build().trimmed


def test_clear_keeps_only_the_system_prompt() -> None:
    ctx = ContextManager(SYSTEM)
    ctx.add(user(GOAL))
    ctx.clear()
    assert contents(ctx.build().messages) == [SYSTEM]


def test_newest_question_is_kept_even_with_tool_steps_after_it() -> None:
    ctx = ContextManager(None, window=60, response_reserve=0)
    ctx.add(user("old question " + "o" * 60))
    ctx.add(assistant("old answer " + "a" * 60))
    ctx.add(user("NEW"))
    for n in range(3):
        call = ToolCall(id=f"c{n}", name="read_file", arguments={})
        ctx.add(Message(role=Role.assistant, tool_calls=[call]))
        ctx.add(Message(role=Role.tool, tool_call_id=f"c{n}", content="r" * 80))
    texts = contents(ctx.build().messages)
    assert "NEW" in texts
    assert not any(t and t.startswith("old") for t in texts)
