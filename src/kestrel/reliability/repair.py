"""Repair pipeline (spec 5.2, M4 piece 2): turn a model reply into tool calls that are safe
to run, fixing the predictable ways small models break them.

`repair(message, specs)` is a pure function: no I/O, no provider, no tools. The loop
calls it between the model's reply and the tools. Steps, each recorded as a `Fix`:

1. Find the calls: the native tool-call field, else a call written in the text
   (```json block, <tool_call> tags, or the whole reply as JSON). A text call must name
   a tool exactly (up to case and `-`/`_`) and stand mostly alone, so JSON data or a
   quoted document inside an answer stays an answer.
2. Fix the JSON: code fences, trailing commas, single quotes, Python literals, missing
   closing brackets, arguments encoded twice, arguments wrapped again. Never complete a
   cut-off string or number: that would run a call with data the model never wrote.
3. Fix the tool name: case, `-` vs `_`, a `functions.` prefix, then a small edit distance
   when exactly one tool is that close. After such a guess, a dropped argument rejects.
4. Check the arguments against the tool's schema: coerce "5" to 5 and "true" to true,
   drop arguments the tool does not take; a missing or uncoercible one is a problem.
5. Unescape double-escaped new lines, only in text arguments (`ToolSpec.text_args`), only
   when there is no real new line, at least two literal `\\n` outside quoted strings (a
   one-line `print("a\\nb")` is left alone) and no Windows path.
6. One call per step: only the first runs, so the model cannot guess a later result.

Repairs that could change what a correct call means are refused: a call that cannot be
used yields `problem`, a short note for the model; the loop sends it back.
"""

import ast
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from kestrel.core.types import JSONObject, JSONValue, Message, ToolCall, ToolSpec

MAX_LITERAL_CHARS = 100_000
"""Text longer than this is not given to `ast.literal_eval` or scanned for escapes."""
MIN_ESCAPED_NEWLINES = 2
MAX_PROSE_AROUND_TEXT_CALL = 200
"""A call written in the text may come with a short lead-in, not a whole answer."""

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\n?(.*?)\n?[ \t]*```", re.DOTALL)
_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_STRING = re.compile(r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_LITERAL_WORDS = {"true": "True", "false": "False", "null": "None"}
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:\\")
_NUMBER = re.compile(r"\s*-?\d+(\.\d+)?([eE][-+]?\d+)?\s*")


@dataclass(frozen=True)
class Fix:
    """One change made to the model's reply, logged as a `repair` event."""

    kind: str
    """call_from_text, json_syntax, arguments_string, unwrap_arguments, tool_name, coerce,
    unknown_argument, unescape_text or one_call."""
    call: str
    """The tool name as the model wrote it."""
    before: JSONValue = None
    after: JSONValue = None


@dataclass(frozen=True)
class Repaired:
    """What `repair` made of a reply."""

    calls: list[ToolCall] = field(default_factory=list[ToolCall])
    """Checked calls to run: none (an answer, or a problem) or exactly one."""
    fixes: list[Fix] = field(default_factory=list[Fix])
    problem: str | None = None
    """A note for the model when its call cannot be used."""
    broken: ToolCall | None = None
    """The unusable call, as close to what the model wrote as valid JSON allows."""
    dropped: int = 0
    """Calls left out by the one-call rule."""
    content: str | None = None
    """The reply's text without a call written in it (set only for text calls), so the
    conversation does not hold the call twice."""


def repair(
    message: Message, specs: Sequence[ToolSpec], *, step: int = 0, truncated: bool = False
) -> Repaired:
    """Read `message` as tool calls for `specs`, fixing what can be fixed safely.

    `truncated`: the backend stopped the reply at its token limit, so a broken call is
    incomplete rather than misformatted, and is never repaired.
    """
    by_name = {s.name: s for s in specs}
    fixes: list[Fix] = []
    content = message.content
    calls = list(message.tool_calls)
    if not calls:
        found = _call_from_text(message.content, by_name, fixes, step)
        if found is None:
            return Repaired(fixes=fixes)
        calls, content = [found[0]], found[1]
    dropped = len(calls) - 1
    if dropped:
        fixes.append(
            Fix("one_call", calls[0].name, before=[c.name for c in calls], after=calls[0].name)
        )
    call = calls[0]
    if truncated and call.arguments_error is not None:
        note = (
            f"ERROR: your call to {call.name} was cut off at the token limit before it was "
            "complete. Send it again, shorter (for long files, write them in parts)."
        )
        return Repaired(fixes=fixes, problem=note, broken=call, dropped=dropped, content=content)
    checked, problem, broken = _check(call, by_name, fixes)
    if problem is not None:
        return Repaired(
            fixes=fixes, problem=problem, broken=broken, dropped=dropped, content=content
        )
    assert checked is not None
    fixes.sort(key=lambda f: f.kind == "one_call")  # about the step, so listed last
    return Repaired(calls=[checked], fixes=fixes, dropped=dropped, content=content)


# Step 1: a call written in the text


def _call_from_text(
    content: str | None, by_name: dict[str, ToolSpec], fixes: list[Fix], step: int
) -> tuple[ToolCall, str | None] | None:
    """(the call, the text around it), or None when the text is an answer."""
    if not content or not by_name:
        return None
    matches = list(_TAGGED.finditer(content)) or list(_FENCE.finditer(content))
    blocks = [(m.group(1), m.start(), m.end()) for m in matches] or [
        (content.strip(), 0, len(content))
    ]
    for block, start, end in blocks:
        prose = (content[:start] + content[end:]).strip()
        if len(prose) > MAX_PROSE_AROUND_TEXT_CALL:
            continue  # a long answer that quotes JSON, not a call
        value, fixed = _read_json(block)
        if not isinstance(value, dict):
            continue
        name = value.get("name", value.get("tool"))
        arguments = value.get("arguments", value.get("parameters", {}))
        if not isinstance(name, str) or _match_name(name, by_name, fuzzy=False) is None:
            continue  # JSON that names no tool is data, not a call
        fixes.append(Fix("call_from_text", name, before=block))
        if fixed:
            fixes.append(Fix("json_syntax", name, before=block, after=json.dumps(value)))
        call = ToolCall(id=f"call_text_{step}", name=name, arguments_text=json.dumps(arguments))
        return call, prose or None
    return None


# Step 2: the JSON


def _read_json(text: str) -> tuple[JSONValue, bool]:
    """Parse `text` as JSON, fixing common syntax slips. Returns (value, was_fixed); the
    value is None when no safe fix works."""
    try:
        return json.loads(text), False
    except json.JSONDecodeError:
        pass
    candidate = text.strip()
    if fenced := _FENCE.fullmatch(candidate):
        candidate = fenced.group(1).strip()
    attempts = [candidate]
    closed = _close_brackets(candidate)
    if closed is not None:
        attempts.append(closed)
    for attempt in attempts:
        no_commas = _outside_strings(attempt, lambda part: _TRAILING_COMMA.sub(r"\1", part))
        for json_text in (attempt, no_commas):
            try:
                return json.loads(json_text), True
            except json.JSONDecodeError:
                pass
        literal = _python_literal(attempt)
        if literal is not None:
            return literal, True
    return None, False


def _python_literal(text: str) -> JSONValue:
    """Single quotes and True/False/None: read as a Python literal (`ast.literal_eval`
    evaluates literals only, never code). JSON words are swapped outside strings only."""
    if len(text) > MAX_LITERAL_CHARS:
        return None
    swapped = _outside_strings(
        text, lambda part: re.sub(r"\b(true|false|null)\b", lambda m: _LITERAL_WORDS[m[1]], part)
    )
    for source in (text, swapped):
        try:
            value = ast.literal_eval(source)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        try:
            return json.loads(json.dumps(value, allow_nan=False))  # JSON values only
        except (TypeError, ValueError):
            return None
    return None


def _outside_strings(text: str, change: Callable[[str], str]) -> str:
    """Apply `change` to the parts of `text` that are not inside quotes."""
    out: list[str] = []
    last = 0
    for string in _STRING.finditer(text):
        out.append(change(text[last : string.start()]))
        out.append(string.group(0))
        last = string.end()
    out.append(change(text[last:]))
    return "".join(out)


def _close_brackets(text: str) -> str | None:
    """Append the closing brackets a reply is missing, but only when the last value is
    complete: a cut-off string or number cannot be finished honestly."""
    stack: list[str] = []
    in_string: str | None = None
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
        elif ch in "\"'":
            in_string = ch
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack:
            stack.pop()
    tail = text.rstrip()
    complete = tail.endswith(('"', "'", "}", "]", "true", "false", "null", "True", "False"))
    if in_string or not stack or not complete:
        return None
    return text + "".join(reversed(stack))


def _arguments(call: ToolCall, fixes: list[Fix]) -> tuple[JSONObject | None, str | None]:
    """The call's arguments as an object, or why there are none."""
    raw = call.arguments_text
    if call.arguments or raw is None or not raw.strip():
        value: JSONValue = call.arguments
    else:
        value, fixed = _read_json(raw)
        if value is None:
            return None, f"the arguments are {call.arguments_error or 'not valid JSON'}"
        if fixed:
            fixes.append(Fix("json_syntax", call.name, before=raw, after=json.dumps(value)))
    if isinstance(value, str):  # encoded twice: a JSON string holding the JSON object
        inner, _ = _read_json(value)
        if isinstance(inner, dict):
            fixes.append(Fix("arguments_string", call.name, before=value, after=inner))
            value = inner
    if not isinstance(value, dict):
        return None, f"the arguments are {_describe(value)}, not an object"
    if set(value) == {"arguments"} and isinstance(value["arguments"], dict):
        fixes.append(Fix("unwrap_arguments", call.name, before=value, after=value["arguments"]))
        value = value["arguments"]
    return value, None


def _describe(value: JSONValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a string"
    return "a single value"


# Steps 3 to 5: one call


def _check(
    call: ToolCall, by_name: dict[str, ToolSpec], fixes: list[Fix]
) -> tuple[ToolCall | None, str | None, ToolCall]:
    """(checked call, None, call) when usable, else (None, note, the call as best read)."""
    name = _match_name(call.name, by_name, fuzzy=True)
    if name is None:
        known = ", ".join(by_name) or "none"
        return None, f"ERROR: unknown tool {call.name!r}. Available tools: {known}.", call
    guessed = name.lower() != _normalize(call.name)
    if name != call.name:
        fixes.append(Fix("tool_name", call.name, before=call.name, after=name))
    spec = by_name[name]
    arguments, error = _arguments(call, fixes)
    if arguments is None:
        return None, _note(spec, error or "the arguments could not be read"), call
    readable = call.model_copy(update={"arguments": arguments, "arguments_error": None})
    fitted, error, dropped = _fit_schema(arguments, spec, call.name, fixes)
    if error is None and guessed and dropped:
        error = f"it was not clear {call.name!r} meant {name} ({', '.join(dropped)} unknown)"
    if error is not None:
        return None, _note(spec, error), readable
    fitted = _unescape(fitted, spec, call.name, fixes)
    return ToolCall(id=call.id, name=name, arguments=fitted), None, readable


def _normalize(name: str) -> str:
    return name.strip().removeprefix("functions.").lower().replace("-", "_").replace(" ", "_")


def _match_name(name: str, by_name: dict[str, ToolSpec], *, fuzzy: bool) -> str | None:
    if name in by_name:
        return name
    plain = _normalize(name)
    for known in by_name:
        if known.lower() == plain:
            return known
    if not fuzzy:
        return None
    close = [k for k in by_name if _distance(plain, k.lower()) <= _max_distance(k)]
    return close[0] if len(close) == 1 else None


def _max_distance(name: str) -> int:
    """Short names are easy to hit by accident: `ls` is one edit from many words."""
    return 0 if len(name) < 4 else 1 if len(name) < 7 else 2


def _distance(a: str, b: str) -> int:
    """Levenshtein edit distance (capped: anything over 3 reports 3)."""
    if abs(len(a) - len(b)) > 2:
        return 3
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _fit_schema(
    arguments: JSONObject, spec: ToolSpec, written: str, fixes: list[Fix]
) -> tuple[JSONObject, str | None, list[str]]:
    """(arguments fitted to the schema, a problem or None, names of dropped arguments)."""
    properties = spec.parameters.get("properties")
    props: JSONObject = properties if isinstance(properties, dict) else {}
    required = spec.parameters.get("required")
    needed = [str(r) for r in required] if isinstance(required, list) else []
    closed = spec.parameters.get("additionalProperties") is False
    result: JSONObject = {}
    dropped: list[str] = []
    for key, value in arguments.items():
        if key not in props:
            if closed:
                fixes.append(Fix("unknown_argument", written, before=key))
                dropped.append(key)
                continue
            result[key] = value
            continue
        schema = props[key]
        wanted = schema.get("type") if isinstance(schema, dict) else None
        fits, coerced = _coerce(value, wanted)
        if not fits:
            return arguments, f"{key}: expected {wanted}, got {json.dumps(value)}", dropped
        if coerced != value or type(coerced) is not type(value):
            fixes.append(Fix("coerce", written, before={key: value}, after={key: coerced}))
        result[key] = coerced
    missing = [k for k in needed if k not in result]
    if missing:
        return arguments, f"{', '.join(missing)}: field required", dropped
    return result, None, dropped


def _coerce(value: JSONValue, wanted: JSONValue) -> tuple[bool, JSONValue]:
    """(fits, value as the JSON Schema type `wanted`)."""
    if wanted is None or isinstance(wanted, list):
        return True, value  # no single type to check against (unions); the tool checks again
    if wanted == "string":
        return isinstance(value, str), value
    if wanted == "boolean":
        if isinstance(value, bool):
            return True, value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return True, value.strip().lower() == "true"
        return False, value
    if wanted == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return True, value
        if isinstance(value, float) and value.is_integer():
            return True, int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
            return True, int(value)
        return False, value
    if wanted == "number":
        if isinstance(value, int | float) and not isinstance(value, bool):
            return math.isfinite(value), value
        if isinstance(value, str) and _NUMBER.fullmatch(value):
            number = float(value)
            return math.isfinite(number), number
        return False, value
    if wanted == "array":
        return isinstance(value, list), value
    if wanted == "object":
        return isinstance(value, dict), value
    return True, value


def _unescape(arguments: JSONObject, spec: ToolSpec, written: str, fixes: list[Fix]) -> JSONObject:
    """Literal backslash-n where new lines were meant, in text arguments only."""
    result = dict(arguments)
    for key in spec.text_args:
        value = result.get(key)
        if isinstance(value, str) and _double_escaped(value):
            fixed = value.replace("\\r\\n", "\n").replace("\\n", "\n")
            fixes.append(Fix("unescape_text", written, before=value, after=fixed))
            result[key] = fixed
    return result


def _double_escaped(text: str) -> bool:
    """Literal backslash-n used as line breaks: at least two of them outside quoted
    strings (`print("a\\nb")` keeps its escapes inside quotes), no real new line, and no
    Windows path (`C:\\new`)."""
    if "\n" in text or len(text) > MAX_LITERAL_CHARS or _WINDOWS_PATH.search(text):
        return False
    outside: list[int] = []
    _outside_strings(text, lambda part: outside.append(part.count("\\n")) or part)
    return sum(outside) >= MIN_ESCAPED_NEWLINES


# The note for the model


def _note(spec: ToolSpec, reason: str) -> str:
    return (
        f"ERROR: your call to {spec.name} could not be used: {reason}.\n"
        f"{spec.name} expects: {_shape(spec)}.\n"
        "Call it again with valid arguments."
    )


def _shape(spec: ToolSpec) -> str:
    """`{"path": string, "overwrite": boolean (optional)}`: short, for a small window."""
    properties = spec.parameters.get("properties")
    props: JSONObject = properties if isinstance(properties, dict) else {}
    required = spec.parameters.get("required")
    needed = set(map(str, required)) if isinstance(required, list) else set[str]()
    parts: list[str] = []
    for key, schema in props.items():
        kind = schema.get("type", "value") if isinstance(schema, dict) else "value"
        optional = "" if key in needed else " (optional)"
        parts.append(f'"{key}": {kind}{optional}')
    return "{" + ", ".join(parts) + "}"
