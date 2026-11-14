"""The agent loop (spec section 6): ask the model, run the tools it asks for, feed the
results back, repeat until it answers or a budget runs out.

It only knows the `Provider` and `Toolbox` protocols and the event bus; it never imports
a concrete provider, tool or frontend (CLAUDE.md rule 5). Every step is an event.
M4 adds tool-call repair and loop detection; M3 adds permission checks.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from kestrel.core.context import NOTE_PREFIX, ContextManager, wrap_untrusted
from kestrel.core.events import EventBus
from kestrel.core.types import (
    BadBackendResponseError,
    EventType,
    JSONObject,
    JSONValue,
    Ladder,
    Message,
    ModelResponse,
    PermissionGate,
    Provider,
    ProviderError,
    Role,
    Rung,
    RunResult,
    StopReason,
    Toolbox,
    ToolCall,
    ToolResult,
    ToolSpec,
)
from kestrel.reliability.nudge import nudge_for
from kestrel.reliability.repair import Fix, repair
from kestrel.reliability.watchdog import Signal, Watchdog, WatchLimits

HANDOFF_NOTE = (
    f"{NOTE_PREFIX} a smaller model ({{small}}) got stuck here: {{reason}} ({{detail}}). "
    "You are a bigger model taking over this step. Do not repeat what failed; take "
    "another approach.)"
)

DENIED_PREFIX = "ERROR: (not allowed) "
"""Start of the result for a call the permission gate refused."""
STEPS_LEFT_WARNING = 2
"""From this many steps left, each tool result says how many remain."""


def steps_left_note(left: int) -> str:
    """The countdown added after a tool result near the step limit."""
    if left == 1:
        return "(Kestrel: this is your last step. Give your final answer now, without tools.)"
    return f"(Kestrel: {left} steps left. Finish the task or give your final answer soon.)"


ONE_CALL_NOTE = "(Only this call ran: one tool per step. Call others again if still needed.)"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. Use the available tools when they help you complete the "
    "task. Call tools one step at a time and read their results. When you have the "
    "answer, reply with it directly and do not call any more tools."
)


@dataclass(frozen=True)
class Budgets:
    """Limits for one run. Hitting any of them ends the run with a clear reason."""

    max_steps: int = 15
    max_tokens_total: int | None = None
    """Prompt plus completion tokens over the whole run, as reported by the backend."""
    max_wall_seconds: float | None = None
    watch: WatchLimits = field(default_factory=WatchLimits)
    """Watchdog thresholds: repeats, steps without progress, errors in a row."""
    max_repairs: int = 2
    """Unusable tool calls allowed in a row (each gets a note and a retry); one more and
    the run is stuck."""


class FixedLadder:
    """One rung, never moves: a run with a single model."""

    def __init__(self, rung: Rung) -> None:
        """Always `rung`."""
        self.rung = rung

    def current(self) -> Rung:
        """The only rung."""
        return self.rung

    async def step_up(self, reason: str, detail: str) -> Rung | None:
        """Nowhere to go: the run stops."""
        return None

    def after_step(self, ok: bool) -> Rung | None:
        """Never moves."""
        return None


@dataclass
class _Totals:
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0


async def run_agent(
    task: str,
    *,
    provider: Provider | None = None,
    model: str | None = None,
    toolbox: Toolbox,
    bus: EventBus,
    context: ContextManager,
    budgets: Budgets | None = None,
    gate: PermissionGate | None = None,
    temperature: float = 0.2,
    clock: Callable[[], float] = time.monotonic,
    think: bool | None = None,
    ladder: Ladder | None = None,
    labels: JSONObject | None = None,
) -> RunResult:
    """Run one task to the end and return what happened.

    Backend failures end the run with `stopped_reason=error` (not an exception).
    Cancellation (Ctrl-C) logs `run_end` with `cancelled`, then propagates.
    Models come from `ladder` (M4.5), or else the one `provider` and `model`. When the
    run is stuck (watchdog or repairs) the ladder may step up to a bigger model, which
    gets a short handoff note; with no rung left the run stops with `loop`.
    """
    if ladder is None:
        if provider is None or model is None:
            raise ValueError("run_agent needs a ladder, or a provider and a model")
        ladder = FixedLadder(Rung(provider, model))
    by_model: dict[str, list[int]] = {}
    """Steps, prompt tokens and completion tokens per model."""
    budgets = budgets or Budgets()
    specs = toolbox.specs()
    started = clock()
    totals = _Totals()
    answer: str | None = None
    reason = StopReason.max_steps
    repairs_used = 0
    """Unusable calls in a row; past `budgets.max_repairs` the run is stuck."""
    watchdog = Watchdog(budgets.watch)
    nudged = False
    """One nudge per run, for a reply that stops too early (see reliability.nudge)."""
    last_failed = False
    """Whether the latest tool call failed (for the after-failure nudge)."""

    async def stop_for(signal: Signal, step: int) -> bool:
        """Log a stuck signal, then step up if the ladder can; True when the run stops."""
        bus.emit(EventType.stuck, step, {"signal": signal.signal, "detail": signal.detail})
        small = ladder.current().name
        bigger = await ladder.step_up(signal.signal, signal.detail)
        if bigger is None:
            return True
        bus.emit(
            EventType.step_up,
            step,
            {"from": small, "to": bigger.name, "reason": signal.signal, "detail": signal.detail},
        )
        note = HANDOFF_NOTE.format(small=small, reason=signal.signal, detail=signal.detail)
        context.add(Message(role=Role.user, content=note))
        watchdog.reset()  # a new model: judge it afresh
        return False

    def step_done(step: int, ok: bool) -> None:
        before = ladder.current().name
        down = ladder.after_step(ok)
        if down is not None:
            bus.emit(EventType.step_down, step, {"from": before, "to": down.name})

    bus.emit(
        EventType.run_start,
        0,
        {
            "task": task,
            "model": ladder.current().name,
            "provider": ladder.current().provider.name,
            "tools": [s.name for s in specs],
            "tool_specs": [s.model_dump(mode="json") for s in specs],
            "max_steps": budgets.max_steps,
            "window": context.window,
            **(labels or {}),  # e.g. the shell session this run belongs to (M5)
        },
    )
    context.add(Message(role=Role.user, content=task))
    try:
        for step in range(1, budgets.max_steps + 1):
            if (
                budgets.max_wall_seconds is not None
                and clock() - started > budgets.max_wall_seconds
            ):
                reason = StopReason.budget
                break
            totals.steps = step
            rung = ladder.current()
            if rung.window is not None:
                context.window = rung.window  # each model gets the conversation trimmed to fit
            response, over_budget = await _ask_model(
                rung,
                specs,
                context,
                bus,
                step,
                temperature,
                think if rung.think is None else rung.think,
            )
            counts = by_model.setdefault(rung.name, [0, 0, 0])
            counts[0] += 1
            prompt_tokens = None
            if response.usage is not None:
                prompt_tokens = response.usage.prompt_tokens
                totals.prompt_tokens += response.usage.prompt_tokens or 0
                totals.completion_tokens += response.usage.completion_tokens or 0
                counts[1] += response.usage.prompt_tokens or 0
                counts[2] += response.usage.completion_tokens or 0

            message = response.message
            repaired = repair(  # model_response above keeps the verbatim reply
                message, specs, step=step, truncated=response.finish_reason == "length"
            )
            for fix in repaired.fixes:
                bus.emit(EventType.repair, step, _fix_data(fix))

            if repaired.problem is not None and repaired.broken is not None:
                # Show the model its unusable call and what was wrong, then let it retry.
                totals.tool_calls += 1
                broken = repaired.broken
                context.add(_stored(message, broken, repaired.content))
                bus.emit(EventType.tool_call, step, _call_data(broken))
                note = ToolResult(call_id=broken.id, content=repaired.problem, is_error=True)
                bus.emit(EventType.tool_result, step, _result_data(broken, note, 0))
                extra = f"\n{ONE_CALL_NOTE}" if repaired.dropped else ""
                context.add(_tool_message(broken, note.content + extra))
                repairs_used += 1
                last_failed = False  # the repair note already told it what went wrong
                if repairs_used > budgets.max_repairs:
                    exhausted = Signal("stuck", "repairs_exhausted", repaired.problem)
                    if await stop_for(exhausted, step):
                        reason = StopReason.loop
                        break
                    repairs_used = 0
                continue
            repairs_used = 0

            nudge = None
            if not repaired.calls and specs and not nudged and step < budgets.max_steps:
                nudge = nudge_for(message.content, last_call_failed=last_failed)
            if nudge is not None:
                nudged = True
                kind, note = nudge
                context.add(message.model_copy(update={"reasoning": None}))
                context.add(Message(role=Role.user, content=note))
                said = (message.content or "")[-200:]
                bus.emit(EventType.nudge, step, {"kind": kind, "said": said})
                continue
            if not repaired.calls:
                context.add(message.model_copy(update={"reasoning": None}))  # never resent
                answer = message.content or ""
                bus.emit(EventType.final, step, {"answer": answer})
                reason = StopReason.final
                break

            # A final answer always counts; a run that cannot fit its next prompt is stuck.
            full = watchdog.after_prompt(
                over_budget=over_budget, prompt_tokens=prompt_tokens, window=context.window
            )
            if full is not None:
                if await stop_for(full, step):
                    reason = StopReason.loop
                    break
                continue  # the bigger rung answers this step afresh

            call = repaired.calls[0]  # one call per step (repair step 6)
            context.add(_stored(message, call, repaired.content))
            totals.tool_calls += 1
            result = await _run_tool(toolbox, call, bus, step, clock, gate, specs)
            content = (
                result.content
                if result.is_error and not result.outside  # Kestrel's own message
                else wrap_untrusted(_source(call), result.content)
            )
            if repaired.dropped:
                content += f"\n{ONE_CALL_NOTE}"
            signals = watchdog.after_tool(call, result)
            for warning in (s for s in signals if s.level == "warn"):
                bus.emit(
                    EventType.loop_warning,
                    step,
                    {"signal": warning.signal, "detail": warning.detail},
                )
                content += f"\n{warning.note}"
            # A "no" from the user is an answer, not a failure to work around.
            last_failed = result.is_error and not result.content.startswith(DENIED_PREFIX)
            left = budgets.max_steps - step
            if 0 < left <= STEPS_LEFT_WARNING:
                # Felt near the end, not read once at the start (Hookele harness).
                content += f"\n{steps_left_note(left)}"
            context.add(_tool_message(call, content))
            stuck = next((s for s in signals if s.level == "stuck"), None)
            if stuck is not None:
                if await stop_for(stuck, step):
                    reason = StopReason.loop
                    break
            else:
                step_done(step, not result.is_error)

            used = totals.prompt_tokens + totals.completion_tokens
            if budgets.max_tokens_total is not None and used >= budgets.max_tokens_total:
                reason = StopReason.budget
                break
    except ProviderError as exc:
        bus.emit(EventType.error, totals.steps, {"kind": type(exc).__name__, "message": str(exc)})
        reason = StopReason.error
    except (asyncio.CancelledError, KeyboardInterrupt):
        _end(bus, totals, StopReason.cancelled, clock() - started, by_model)
        raise

    wall = clock() - started
    _end(bus, totals, reason, wall, by_model)
    return RunResult(
        run_id=bus.run_id,
        stopped_reason=reason,
        answer=answer,
        steps=totals.steps,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        wall_seconds=round(wall, 3),
    )


async def _ask_model(
    rung: Rung,
    specs: list[ToolSpec],
    context: ContextManager,
    bus: EventBus,
    step: int,
    temperature: float,
    think: bool | None,
) -> tuple[ModelResponse, bool]:
    """Build the prompt, stream the reply (live pieces as `model_delta`), log the result.
    Also says whether the prompt was still over budget after trimming."""
    prompt = context.build()
    if prompt.trimmed or prompt.over_budget:
        bus.emit(
            EventType.context_trim,
            step,
            {
                "dropped_tool_results": prompt.dropped_tool_results,
                "dropped_messages": prompt.dropped_messages,
                "over_budget": prompt.over_budget,
                "est_tokens": prompt.est_tokens,
            },
        )
    bus.emit(
        EventType.model_request,
        step,
        {
            "model": rung.name,
            "message_count": len(prompt.messages),
            "est_tokens": prompt.est_tokens,
            # Exactly what the model is shown this step, after trimming (big ones go to a
            # side file). With `tool_specs` in run_start, a step can be replayed or reported.
            "messages": [_message_data(m) for m in prompt.messages],
        },
    )
    final: ModelResponse | None = None
    async for chunk in rung.provider.stream(
        prompt.messages, specs or None, model=rung.model, temperature=temperature, think=think
    ):
        if chunk.text or chunk.reasoning:
            bus.emit(
                EventType.model_delta, step, {"text": chunk.text, "reasoning": chunk.reasoning}
            )
        if chunk.done is not None:
            final = chunk.done
    if final is None:
        raise BadBackendResponseError("the stream ended without a final reply")

    usage: JSONValue = None
    if final.usage is not None:
        usage = {
            "prompt_tokens": final.usage.prompt_tokens,
            "completion_tokens": final.usage.completion_tokens,
        }
    bus.emit(
        EventType.model_response,
        step,
        {
            "content": final.message.content,
            "reasoning": final.message.reasoning,
            "raw_text": final.raw_text,
            "tool_calls": [_call_data(c) for c in final.message.tool_calls],
            "finish_reason": final.finish_reason,
            "usage": usage,
            "model": rung.name,
        },
    )
    return final, prompt.over_budget


async def _run_tool(
    toolbox: Toolbox,
    call: ToolCall,
    bus: EventBus,
    step: int,
    clock: Callable[[], float],
    gate: PermissionGate | None,
    specs: list[ToolSpec],
) -> ToolResult:
    """Check permission, then execute one call. A refused or crashing call becomes an
    `ERROR:` result the model can read, never a crashed run."""
    bus.emit(EventType.tool_call, step, _call_data(call))
    if call.arguments_error is not None:  # guard: repair() never lets one through
        result = ToolResult(
            call_id=call.id,
            content=(
                f"ERROR: the arguments for {call.name} are {call.arguments_error}. "
                "Send the call again with the arguments as one JSON object."
            ),
            is_error=True,
        )
        bus.emit(EventType.tool_result, step, _result_data(call, result, 0))
        return result
    try:
        early = toolbox.precheck(call)
    except Exception:  # a broken check must not end the run; execute reports the problem
        early = None
    if early is not None:  # cannot work: tell the model, never ask the user about it
        bus.emit(EventType.tool_result, step, _result_data(call, early, 0))
        return early
    spec = next((s for s in specs if s.name == call.name), None)
    if gate is not None and spec is not None:  # unknown tools are reported by the toolbox
        decision = await gate.check(call, spec)
        bus.emit(
            EventType.permission,
            step,
            {
                "id": call.id,
                "name": call.name,
                "allowed": decision.allowed,
                "reason": decision.reason,
            },
        )
        if not decision.allowed:
            result = ToolResult(
                call_id=call.id,
                content=f"{DENIED_PREFIX}{call.name} was not run: {decision.reason}",
                is_error=True,
            )
            bus.emit(EventType.tool_result, step, _result_data(call, result, 0))
            return result
    started = clock()
    try:
        result = await toolbox.execute(call)
    except Exception as exc:  # tools must not take the loop down (CLAUDE.md rule 8)
        result = ToolResult(
            call_id=call.id,
            content=f"ERROR: tool {call.name} crashed: {type(exc).__name__}: {exc}",
            is_error=True,
        )
    bus.emit(EventType.tool_result, step, _result_data(call, result, clock() - started))
    return result


def _stored(message: Message, call: ToolCall, content: str | None) -> Message:
    """The reply as the conversation keeps it: its text (without a call written in it),
    the one call that is handled, no reasoning (never resent)."""
    return message.model_copy(update={"content": content, "tool_calls": [call], "reasoning": None})


def _tool_message(call: ToolCall, content: str) -> Message:
    return Message(role=Role.tool, tool_call_id=call.id, name=call.name, content=content)


def _fix_data(fix: Fix) -> JSONObject:
    return {"kind": fix.kind, "call": fix.call, "before": fix.before, "after": fix.after}


def _result_data(call: ToolCall, result: ToolResult, seconds: float) -> JSONObject:
    return {
        "id": call.id,
        "name": call.name,
        "content": result.content,
        "is_error": result.is_error,
        "ms": round(seconds * 1000),
    }


def _source(call: ToolCall) -> str:
    """`read_file:src/app.py`: the tool plus what it looked at, for the untrusted tag."""
    for key in ("path", "command", "question"):
        value = call.arguments.get(key)
        if isinstance(value, str):
            return f"{call.name}:{value[:80]}"
    return call.name


def _call_data(call: ToolCall) -> JSONObject:
    data: JSONObject = {"id": call.id, "name": call.name, "arguments": call.arguments}
    if call.arguments_text is not None:
        data["arguments_text"] = call.arguments_text  # verbatim, for repair and reports
    if call.arguments_error is not None:
        data["arguments_error"] = call.arguments_error
    return data


def _message_data(message: Message) -> JSONObject:
    """A message as logged: everything the model is shown, nothing it is not."""
    data: JSONObject = {"role": message.role.value}
    if message.content is not None:
        data["content"] = message.content
    if message.tool_calls:
        data["tool_calls"] = [_call_data(c) for c in message.tool_calls]
    if message.tool_call_id is not None:
        data["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        data["name"] = message.name
    return data


def _end(
    bus: EventBus,
    totals: _Totals,
    reason: StopReason,
    wall: float,
    by_model: dict[str, list[int]],
) -> None:
    bus.emit(
        EventType.run_end,
        totals.steps,
        {
            "stopped_reason": reason.value,
            "steps": totals.steps,
            "tool_calls": totals.tool_calls,
            "prompt_tokens": totals.prompt_tokens,
            "completion_tokens": totals.completion_tokens,
            "wall_seconds": round(wall, 3),
            "by_model": {
                name: {"steps": c[0], "prompt_tokens": c[1], "completion_tokens": c[2]}
                for name, c in by_model.items()
            },
        },
    )
