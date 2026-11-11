"""The agent loop (spec section 6): ask the model, run the tools it asks for, feed the
results back, repeat until it answers or a budget runs out.

It only knows the `Provider` and `Toolbox` protocols and the event bus; it never imports
a concrete provider, tool or frontend (CLAUDE.md rule 5). Every step is an event.
M4 adds tool-call repair and loop detection; M3 adds permission checks.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from kestrel.core.context import ContextManager
from kestrel.core.events import EventBus
from kestrel.core.types import (
    BadBackendResponseError,
    EventType,
    JSONObject,
    JSONValue,
    Message,
    ModelResponse,
    PermissionGate,
    Provider,
    ProviderError,
    Role,
    RunResult,
    StopReason,
    Toolbox,
    ToolCall,
    ToolResult,
    ToolSpec,
)

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


@dataclass
class _Totals:
    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: int = 0


async def run_agent(
    task: str,
    *,
    provider: Provider,
    model: str,
    toolbox: Toolbox,
    bus: EventBus,
    context: ContextManager,
    budgets: Budgets | None = None,
    gate: PermissionGate | None = None,
    temperature: float = 0.2,
    clock: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Run one task to the end and return what happened.

    Backend failures end the run with `stopped_reason=error` (not an exception).
    Cancellation (Ctrl-C) logs `run_end` with `cancelled`, then propagates.
    """
    budgets = budgets or Budgets()
    specs = toolbox.specs()
    started = clock()
    totals = _Totals()
    answer: str | None = None
    reason = StopReason.max_steps

    bus.emit(
        EventType.run_start,
        0,
        {
            "task": task,
            "model": model,
            "provider": provider.name,
            "tools": [s.name for s in specs],
            "max_steps": budgets.max_steps,
            "window": context.window,
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
            response = await _ask_model(provider, model, specs, context, bus, step, temperature)
            if response.usage is not None:
                totals.prompt_tokens += response.usage.prompt_tokens or 0
                totals.completion_tokens += response.usage.completion_tokens or 0

            message = response.message
            context.add(message.model_copy(update={"reasoning": None}))  # never resent
            if not message.tool_calls:
                answer = message.content or ""
                bus.emit(EventType.final, step, {"answer": answer})
                reason = StopReason.final
                break

            for call in message.tool_calls:
                totals.tool_calls += 1
                result = await _run_tool(toolbox, call, bus, step, clock, gate, specs)
                context.add(
                    Message(
                        role=Role.tool, tool_call_id=call.id, name=call.name, content=result.content
                    )
                )

            used = totals.prompt_tokens + totals.completion_tokens
            if budgets.max_tokens_total is not None and used >= budgets.max_tokens_total:
                reason = StopReason.budget
                break
    except ProviderError as exc:
        bus.emit(EventType.error, totals.steps, {"kind": type(exc).__name__, "message": str(exc)})
        reason = StopReason.error
    except (asyncio.CancelledError, KeyboardInterrupt):
        _end(bus, totals, StopReason.cancelled, clock() - started)
        raise

    wall = clock() - started
    _end(bus, totals, reason, wall)
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
    provider: Provider,
    model: str,
    specs: list[ToolSpec],
    context: ContextManager,
    bus: EventBus,
    step: int,
    temperature: float,
) -> ModelResponse:
    """Build the prompt, stream the reply (live pieces as `model_delta`), log the result."""
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
        {"messages": len(prompt.messages), "est_tokens": prompt.est_tokens},
    )
    final: ModelResponse | None = None
    async for chunk in provider.stream(
        prompt.messages, specs or None, model=model, temperature=temperature
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
        },
    )
    return final


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
                content=f"ERROR: {call.name} was not run: {decision.reason}",
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


def _result_data(call: ToolCall, result: ToolResult, seconds: float) -> JSONObject:
    return {
        "id": call.id,
        "name": call.name,
        "content": result.content,
        "is_error": result.is_error,
        "ms": round(seconds * 1000),
    }


def _call_data(call: ToolCall) -> JSONObject:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _end(bus: EventBus, totals: _Totals, reason: StopReason, wall: float) -> None:
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
        },
    )
