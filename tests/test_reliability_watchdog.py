"""Seam: the watchdog's signals, and how the loop acts on them."""

import asyncio

from kestrel.core.context import ContextManager
from kestrel.core.events import EventBus
from kestrel.core.loop import Budgets, run_agent
from kestrel.core.types import (
    Event,
    EventType,
    Role,
    Rung,
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
)
from kestrel.providers.fake import FakeProvider
from kestrel.reliability.watchdog import Watchdog, WatchLimits
from kestrel.tools.demo import DemoToolbox


def step(dog: Watchdog, name: str, result: str, *, error: bool = False, **args: int) -> list[str]:
    call = ToolCall(id="c", name=name, arguments=dict(args))
    signals = dog.after_tool(call, ToolResult(call_id="c", content=result, is_error=error))
    return [f"{s.level}:{s.signal}" for s in signals]


def test_same_call_warns_once_then_is_stuck() -> None:
    dog = Watchdog(WatchLimits(max_repeats=4, max_stall_steps=50))
    seen = [step(dog, "add", "3", a=1, b=2) for _ in range(4)]
    assert seen == [[], [], ["warn:repeat"], ["stuck:repeat"]]


def test_a_new_call_resets_the_repeat_count() -> None:
    dog = Watchdog(WatchLimits(max_repeats=3, max_stall_steps=50))
    step(dog, "add", "3", a=1, b=2)
    step(dog, "add", "3", a=1, b=2)
    assert step(dog, "add", "4", a=1, b=3) == []


def test_back_and_forth_warns() -> None:
    dog = Watchdog(WatchLimits(max_stall_steps=50))
    calls = [("list_dir", "a"), ("read_file", "b"), ("list_dir", "a"), ("read_file", "b")]
    seen = [step(dog, name, result) for name, result in calls]
    assert seen[-1] == ["warn:oscillation"]


def test_the_same_call_giving_the_same_result_stalls() -> None:
    dog = Watchdog(WatchLimits(max_repeats=50, max_stall_steps=4))
    seen = [step(dog, "add", "same", a=1, b=0) for _ in range(5)]
    assert seen[2] == ["warn:stall"] and seen[4] == ["stuck:stall"]


def test_different_calls_with_equal_results_are_progress() -> None:
    dog = Watchdog(WatchLimits(max_stall_steps=4))
    seen = [step(dog, "read_file", "(empty file)", a=i) for i in range(8)]
    assert seen == [[]] * 8


def test_lowest_allowed_thresholds_never_warn_on_the_first_call() -> None:
    dog = Watchdog(WatchLimits(max_repeats=3, max_stall_steps=4))
    assert step(dog, "add", "3", a=1, b=2) == []
    assert step(dog, "add", "3", a=1, b=2) == ["warn:repeat"]  # stall waits for a 2nd


def test_a_second_stall_episode_is_warned_again() -> None:
    dog = Watchdog(WatchLimits(max_repeats=50, max_stall_steps=6))
    first = [step(dog, "add", "x", a=1, b=0) for _ in range(5)]
    step(dog, "echo", "new")
    second = [step(dog, "echo", "new") for _ in range(4)]
    assert ["warn:stall"] in first and ["warn:stall"] in second


def test_oscillation_note_names_the_calls() -> None:
    dog = Watchdog(WatchLimits(max_stall_steps=50))
    for path in ["a", "b", "a"]:
        dog.after_tool(
            ToolCall(id="c", name="read_file", arguments={"path": path}),
            ToolResult(call_id="c", content=path),
        )
    [warning] = dog.after_tool(
        ToolCall(id="c", name="read_file", arguments={"path": "b"}),
        ToolResult(call_id="c", content="b2"),
    )
    assert warning.note is not None and '{"path": "a"}' in warning.note


def test_errors_in_a_row_are_stuck() -> None:
    dog = Watchdog(WatchLimits(max_tool_errors=3, max_stall_steps=50))
    seen = [step(dog, "add", f"ERROR {i}", error=True, a=i, b=0) for i in range(3)]
    assert seen[-1] == ["stuck:tool_errors"]


def test_only_a_prompt_trimming_cannot_fit_is_stuck() -> None:
    dog = Watchdog()
    assert dog.after_prompt(over_budget=False, prompt_tokens=16_000, window=16_384) is None
    assert dog.after_prompt(over_budget=True, prompt_tokens=9_000, window=16_384) is None
    full = dog.after_prompt(over_budget=True, prompt_tokens=15_600, window=16_384)
    assert full is not None and full.signal == "context_overflow"


async def run(
    fake: FakeProvider, window: int = 16_384, **kwargs: object
) -> tuple[list[Event], StopReason]:
    events: list[Event] = []
    bus = EventBus("r")
    bus.subscribe(events.append)
    result = await run_agent(
        "add things",
        provider=fake,
        model="m",
        toolbox=DemoToolbox(),
        bus=bus,
        context=ContextManager(None, window=window),
        budgets=Budgets(max_steps=10, watch=WatchLimits(max_repeats=3, max_stall_steps=50)),
        **kwargs,  # type: ignore[arg-type]
    )
    return events, result.stopped_reason


def test_loop_warns_the_model_then_stops_when_stuck() -> None:
    same = FakeProvider.tool_call("add", {"a": 1, "b": 2})
    fake = FakeProvider([same, same, same, FakeProvider.text("never")])
    events, reason = asyncio.run(run(fake))
    assert reason is StopReason.loop
    [warning] = [e for e in events if e.type is EventType.loop_warning]
    assert warning.data["signal"] == "repeat"
    [stuck] = [e for e in events if e.type is EventType.stuck]
    assert stuck.data["signal"] == "repeat"
    third_prompt = fake.calls[2].messages
    last_tool = [m for m in third_prompt if m.role is Role.tool][-1]
    assert last_tool.content is not None and "same arguments 2 times" in last_tool.content


class TwoRungs:
    """A test ladder: `small` until stuck, then `big` for one good step, then back."""

    def __init__(self, small: FakeProvider, big: FakeProvider) -> None:
        self.rungs = [Rung(small, "small"), Rung(big, "big", think=True)]
        self.at = 0
        self.reasons: list[str] = []

    def current(self) -> Rung:
        return self.rungs[self.at]

    async def step_up(self, reason: str, detail: str) -> Rung | None:
        self.reasons.append(reason)
        if self.at + 1 >= len(self.rungs):
            return None
        self.at += 1
        return self.rungs[self.at]

    def after_step(self, ok: bool) -> Rung | None:
        if self.at and ok:
            self.at = 0
            return self.rungs[0]
        return None


def test_a_stuck_step_goes_up_with_a_note_and_comes_back_down() -> None:
    same = FakeProvider.tool_call("add", {"a": 1, "b": 2})
    small = FakeProvider([same, same, same, same, FakeProvider.text("7")])
    big = FakeProvider([FakeProvider.tool_call("add", {"a": 3, "b": 4})])
    ladder = TwoRungs(small, big)
    events, reason = asyncio.run(run(small, ladder=ladder))
    assert reason is StopReason.final
    kinds = [e.type for e in events if e.type in (EventType.step_up, EventType.step_down)]
    assert kinds == [EventType.step_up, EventType.step_down]
    up = next(e for e in events if e.type is EventType.step_up)
    assert (up.data["from"], up.data["to"], up.data["reason"]) == ("small", "big", "repeat")
    handoff = big.calls[0].messages[-1].content or ""
    assert handoff.startswith("(Kestrel: a smaller model (small) got stuck here: repeat")
    assert big.calls[0].think is True  # thinking on for the handed-off step
    end = next(e for e in events if e.type is EventType.run_end)
    assert set(end.data["by_model"]) == {"small", "big"}  # type: ignore[arg-type]


def test_a_prompt_that_cannot_fit_stops_a_tool_step() -> None:
    big = FakeProvider.tool_call("add", {"a": 1, "b": 2}, usage=Usage(prompt_tokens=1_000))
    events, reason = asyncio.run(run(FakeProvider([big, FakeProvider.text("x")]), window=1_024))
    assert reason is StopReason.loop
    assert [e.data["signal"] for e in events if e.type is EventType.stuck] == ["context_overflow"]


def test_a_final_answer_is_kept_even_when_the_prompt_was_too_big() -> None:
    answer = FakeProvider.text("done")
    _, reason = asyncio.run(run(FakeProvider([answer]), window=1_024))
    assert reason is StopReason.final


def test_repairs_exhausted_also_steps_up_and_the_top_rung_stuck_stops() -> None:
    bad = FakeProvider.tool_call("add", {"a": "many", "b": 1})
    small = FakeProvider([bad, bad, bad])
    big = FakeProvider([bad, bad, bad])
    ladder = TwoRungs(small, big)
    _, reason = asyncio.run(run(small, ladder=ladder))
    assert ladder.reasons == ["repairs_exhausted", "repairs_exhausted"]
    assert reason is StopReason.loop  # no rung above the top one


def test_after_an_overflow_step_up_the_bigger_model_answers() -> None:
    big_prompt = FakeProvider.tool_call("add", {"a": 1, "b": 2}, usage=Usage(prompt_tokens=1_000))
    small = FakeProvider([big_prompt])
    big = FakeProvider([FakeProvider.text("3")])
    ladder = TwoRungs(small, big)
    events, reason = asyncio.run(run(small, ladder=ladder, window=1_024))
    assert reason is StopReason.final and len(big.calls) == 1
    assert not [e for e in events if e.type is EventType.step_down]
    assert not [e for e in events if e.type is EventType.tool_call]  # the small call never ran
