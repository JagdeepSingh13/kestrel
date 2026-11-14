"""`kestrel trace`: show a recorded run as a readable timeline."""

import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from kestrel.cli.frontend import describe_web
from kestrel.cli.output import window_fix
from kestrel.core.events import first_task, load_value, read_events
from kestrel.core.types import Event, EventType, JSONValue

SHORT_CHARS = 160


def print_trace(path: Path, out: Console, *, full: bool = False) -> None:
    """Print a summary of the run, then every logged event, one row each."""
    events = read_events(path)
    out.print(Panel("\n".join(escape(line) for line in summary(events)), title=f"Run {path.stem}"))
    table = Table(show_lines=full)
    table.add_column("Step", justify="right")
    table.add_column("Event")
    table.add_column("Details", overflow="fold")
    for event in events:
        style = _style(event)
        table.add_row(
            str(event.step),
            f"[{style}]{event.type.value}[/{style}]" if style else event.type.value,
            escape(describe(event, full=full)),
        )
    out.print(table)


def summary(events: list[Event]) -> list[str]:
    """What a reader wants first: how the run ended and why, the model, how full the
    context got, and what went wrong along the way."""
    by_type: dict[EventType, list[Event]] = {}
    for event in events:
        by_type.setdefault(event.type, []).append(event)

    def first(kind: EventType) -> dict[str, JSONValue]:
        found = by_type.get(kind)
        return found[0].data if found else {}

    start, end, profile = (
        first(EventType.run_start),
        first(EventType.run_end),
        first(EventType.profile),
    )
    reason = str(end.get("stopped_reason", "did not finish (no run_end)"))
    why = ""
    if stuck := by_type.get(EventType.stuck):
        why = f": {stuck[-1].data.get('signal')}, {stuck[-1].data.get('detail')}"
    elif errors := by_type.get(EventType.error):
        why = f": {errors[-1].data.get('message')}"
    lines = [
        f"ended      {reason}{why}",
        f"model      {start.get('model')} on {profile.get('kind', start.get('provider'))}",
    ]
    window = start.get("window")
    wanted = profile.get("wanted_window")
    if isinstance(window, int):
        peak = max(
            (
                _int(u.get("prompt_tokens"))
                for e in by_type.get(EventType.model_response, [])
                if isinstance(u := e.data.get("usage"), dict)
            ),
            default=0,
        )
        line = f"window     {window} tokens, fullest prompt {peak} ({round(100 * peak / window)}%)"
        if isinstance(wanted, int) and window < wanted:
            line += f"; the agent wanted {wanted}"
        lines.append(line)
        if isinstance(wanted, int) and window < wanted:
            lines.append(f"  fix      {window_fix(str(profile.get('kind')), wanted)}")
    trims = by_type.get(EventType.context_trim, [])
    if trims:
        emptied = sum(_int(e.data.get("dropped_tool_results")) for e in trims)
        dropped = sum(_int(e.data.get("dropped_messages")) for e in trims)
        lines.append(
            f"trimmed    {len(trims)} times: {emptied} old tool results emptied, "
            f"{dropped} old messages dropped"
        )
    failed = sum(1 for e in by_type.get(EventType.tool_result, []) if e.data.get("is_error"))
    counts = [
        f"{end.get('steps', '?')} steps",
        f"{end.get('tool_calls', '?')} tool calls",
        f"{failed} failed",
        f"{len(by_type.get(EventType.repair, []))} repairs",
        f"{len(by_type.get(EventType.loop_warning, []))} warnings",
        f"{len(by_type.get(EventType.nudge, []))} nudges",
    ]
    lines.append("steps      " + ", ".join(counts))
    split = end.get("by_model")
    if isinstance(split, dict) and len(split) > 1:
        ups = len(by_type.get(EventType.step_up, []))
        per = ", ".join(
            f"{name} {_int(c.get('steps')) if isinstance(c, dict) else 0}"
            for name, c in split.items()
        )
        lines.append(f"models     {per} ({ups} step-ups)")
    lines.append(
        f"tokens     {end.get('prompt_tokens', '?')} in, {end.get('completion_tokens', '?')} "
        f"out, {end.get('wall_seconds', '?')}s"
    )
    return lines


def _int(value: JSONValue) -> int:
    return value if isinstance(value, int) else 0


def print_run_list(
    directory: Path,
    out: Console,
    limit: int = 10,
    *,
    only: list[str] | None = None,
    title: str = "Recent runs",
) -> None:
    """The newest runs, newest first (`only`: just these run ids, all of them)."""
    if only is not None:
        logs = [p for p in (directory / f"{r}.jsonl" for r in reversed(only)) if p.is_file()]
    else:
        logs = sorted(directory.glob("*.jsonl"), reverse=True)[:limit] if directory.is_dir() else []
    if not logs:
        out.print('No runs yet. Start one with `kestrel run "task" --model M`.')
        return
    table = Table(title=title)
    table.add_column("Run id")
    table.add_column("Task", overflow="fold")
    for log in logs:
        table.add_row(log.stem, escape(_cut(first_task(log), SHORT_CHARS)))
    out.print(table)


def describe(event: Event, *, full: bool = False) -> str:
    """One-line summary of an event's data."""
    d = event.data
    limit = None if full else SHORT_CHARS

    def text(value: JSONValue) -> str:
        if isinstance(value, dict) and "side_file" in value:
            return f"[{value.get('chars')} chars in {value.get('side_file')}]"
        raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return _cut(raw, limit)

    match event.type:
        case EventType.run_start:
            listed = d.get("tools")
            tools = ", ".join(str(t) for t in listed) if isinstance(listed, list) else ""
            return f"{text(d.get('task'))}  (model {d.get('model')}, tools: {tools or 'none'})"
        case EventType.profile:
            return (
                f"{d.get('model')} on {d.get('kind')}: tools {d.get('native_tools')}, "
                f"window {d.get('window')}"
            )
        case EventType.model_request:
            count = d.get("message_count", d.get("messages"))  # older logs: a plain count
            return f"{count} messages, ~{d.get('est_tokens')} tokens (/trace ID --prompt N)"
        case EventType.model_response:
            parts: list[str] = []
            if d.get("content"):
                parts.append(text(d.get("content")))
            calls = d.get("tool_calls")
            if isinstance(calls, list) and calls:
                names = [str(c.get("name")) for c in calls if isinstance(c, dict)]
                parts.append("calls " + ", ".join(names))
            reasoning = d.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                parts.append(text(reasoning) if full else f"(thought {len(reasoning)} chars)")
            usage = d.get("usage")
            if isinstance(usage, dict):
                parts.append(
                    f"[{usage.get('prompt_tokens')} in, {usage.get('completion_tokens')} out]"
                )
            return "  ".join(parts) or "(empty reply)"
        case EventType.tool_call:
            return f"{d.get('name')} {json.dumps(d.get('arguments'), ensure_ascii=False)}"
        case EventType.tool_result:
            return f"{text(d.get('content'))}  ({d.get('ms')} ms)"
        case EventType.repair:
            change = f"{text(d.get('before'))} to {text(d.get('after'))}"
            return f"{d.get('kind')} on {d.get('call')}: {change}"
        case EventType.loop_warning:
            return f"{d.get('signal')}: {text(d.get('detail'))}"
        case EventType.stuck:
            return f"{d.get('signal')}: {text(d.get('detail'))}"
        case EventType.web:
            return describe_web(d)
        case EventType.mcp_start:
            tools = d.get("tools")
            count = len(tools) if isinstance(tools, list) else 0
            return f"started MCP server {d.get('server')} ({count} tools): {text(d.get('command'))}"
        case EventType.step_up:
            return f"{d.get('from')} to {d.get('to')}: {d.get('reason')} ({text(d.get('detail'))})"
        case EventType.step_down:
            return f"{d.get('from')} back to {d.get('to')}"
        case EventType.nudge:
            why = (
                "stopped after a failed tool"
                if d.get("kind") == "after_failure"
                else "announced an action without a call"
            )
            return f"{why}: {text(d.get('said'))}"
        case EventType.final:
            return text(d.get("answer"))
        case EventType.error:
            return f"{d.get('kind')}: {text(d.get('message'))}"
        case EventType.run_end:
            return (
                f"{d.get('stopped_reason')}: {d.get('steps')} steps, {d.get('tool_calls')} tool "
                f"calls, {d.get('prompt_tokens')} in, {d.get('completion_tokens')} out, "
                f"{d.get('wall_seconds')}s"
            )
        case _:
            return text(d)


def print_prompt(path: Path, step: int, out: Console) -> bool:
    """Print exactly what the model was shown at `step`. False if that step has none."""
    request = next(
        (e for e in read_events(path) if e.type is EventType.model_request and e.step == step),
        None,
    )
    try:
        messages = load_value(path, request.data.get("messages")) if request else None
    except (OSError, ValueError):  # side file deleted or damaged
        return False
    if not isinstance(messages, list):
        return False
    out.print(f"[bold]Run {path.stem}, step {step}: {len(messages)} messages sent[/bold]")
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role"))
        out.print(f"\n[cyan]{escape(role)}[/cyan]", highlight=False)
        content = message.get("content")
        if isinstance(content, str):
            out.print(content, markup=False, highlight=False)
        calls = message.get("tool_calls")
        for call in calls if isinstance(calls, list) else []:
            if isinstance(call, dict):
                raw = call.get("arguments_text") or json.dumps(call.get("arguments"))
                out.print(f"call {call.get('name')} {raw}", markup=False, highlight=False)
    return True


def _style(event: Event) -> str:
    if event.type is EventType.error or (
        event.type is EventType.tool_result and event.data.get("is_error")
    ):
        return "red"
    if event.type is EventType.final:
        return "green"
    return ""


def _cut(text: str, limit: int | None) -> str:
    text = " ".join(text.split())
    return text if limit is None or len(text) <= limit else text[:limit] + "..."
