"""`kestrel trace`: show a recorded run as a readable timeline."""

import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kestrel.core.events import first_task, read_events
from kestrel.core.types import Event, EventType, JSONValue

SHORT_CHARS = 160


def print_trace(path: Path, out: Console, *, full: bool = False) -> None:
    """Print every logged event of one run, one row each."""
    events = read_events(path)
    table = Table(title=f"Run {path.stem}", show_lines=full)
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


def print_run_list(directory: Path, out: Console, limit: int = 10) -> None:
    """The newest runs, newest first."""
    logs = sorted(directory.glob("*.jsonl"), reverse=True)[:limit] if directory.is_dir() else []
    if not logs:
        out.print('No runs yet. Start one with `kestrel run "task" --model M`.')
        return
    table = Table(title="Recent runs")
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
        case EventType.model_request:
            return f"{d.get('messages')} messages, ~{d.get('est_tokens')} tokens"
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
