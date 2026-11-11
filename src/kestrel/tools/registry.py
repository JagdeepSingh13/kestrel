"""Tools as plain Python functions (spec section 8).

    @tool(permission="confirm", side_effects=True)
    def write_file(ctx: ToolContext, path: Annotated[str, Field(description="...")]) -> str:
        '''Write text to a file inside the workspace.'''

`@tool` builds the `ToolSpec` the model sees from the function's name, docstring and type
hints. `ToolRegistry` is a `Toolbox`: it checks the model's arguments against those type
hints, runs the function with a time limit and an output size cap, and turns every
failure into an `ERROR: ...` result instead of an exception (CLAUDE.md rule 8).
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from kestrel.core.types import JSONObject, JSONValue, ToolCall, ToolResult, ToolSpec
from kestrel.safety.paths import PathNotAllowedError, Workspace

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 20_000

type Permission = Literal["allow", "confirm", "deny"]


@dataclass(frozen=True)
class ToolContext:
    """What a tool may use besides its arguments. Never shown to the model."""

    workspace: Workspace
    env: Mapping[str, str]
    ask: Callable[[str], Awaitable[str]] | None = None
    """Ask the user a question (set by the frontend; used by `ask_user`)."""


@dataclass(frozen=True)
class Tool:
    """A function plus everything needed to offer it to a model and call it safely."""

    spec: ToolSpec
    fn: Callable[..., object]
    arguments: type[BaseModel]
    """Pydantic model built from the function's parameters; checks the model's arguments."""
    wants_context: bool
    timeout: float
    max_output_chars: int

    @property
    def name(self) -> str:
        """The tool's name, as the model calls it."""
        return self.spec.name


def tool(
    *,
    permission: Permission = "allow",
    side_effects: bool = False,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> Callable[[Callable[..., object]], Tool]:
    """Turn a function into a `Tool`. A first parameter named `ctx` receives the
    `ToolContext` and is left out of what the model sees."""

    def wrap(fn: Callable[..., object]) -> Tool:
        params = list(inspect.signature(fn).parameters.values())
        wants_context = bool(params) and params[0].name == "ctx"
        if wants_context:
            params = params[1:]
        # Any: pydantic's create_model takes arbitrary (type, default) pairs. Local only.
        fields: dict[str, Any] = {}
        for p in params:
            if p.annotation is inspect.Parameter.empty:
                raise TypeError(f"tool {fn.__name__}: parameter {p.name} needs a type hint")
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[p.name] = (p.annotation, default)
        arguments: type[BaseModel] = create_model(
            f"{fn.__name__}_arguments",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )
        doc = inspect.getdoc(fn)
        if not doc:
            raise TypeError(f"tool {fn.__name__} needs a docstring (it is the model's manual)")
        spec = ToolSpec(
            name=fn.__name__,
            description=doc.split("\n\n")[0].replace("\n", " "),
            parameters=_clean_schema(cast(JSONObject, arguments.model_json_schema())),
            permission=permission,
            side_effects=side_effects,
        )
        return Tool(spec, fn, arguments, wants_context, timeout, max_output_chars)

    return wrap


class ToolRegistry:
    """A `Toolbox` made of `Tool`s, bound to one workspace."""

    def __init__(self, tools: Sequence[Tool], context: ToolContext) -> None:
        """Tools are offered to the model in the order given."""
        self._tools = {t.name: t for t in tools}
        self.context = context

    def specs(self) -> list[ToolSpec]:
        """What the model is told about each tool."""
        return [t.spec for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name (permission checks need its spec)."""
        return self._tools.get(name)

    async def execute(self, call: ToolCall) -> ToolResult:
        """Check the arguments, run the tool with its limits, and report the outcome."""
        found = self._tools.get(call.name)
        if found is None:
            names = ", ".join(self._tools) or "none"
            return _error(call, f"unknown tool {call.name!r}. Available tools: {names}")
        try:
            args = found.arguments.model_validate(call.arguments)
        except ValidationError as exc:
            return _error(call, f"bad arguments for {call.name}: {_describe(exc)}")
        kwargs: dict[str, object] = {name: getattr(args, name) for name in type(args).model_fields}
        if found.wants_context:
            kwargs = {"ctx": self.context, **kwargs}
        try:
            async with asyncio.timeout(found.timeout):
                output = await _call(found.fn, kwargs)
        except TimeoutError:
            return _error(call, f"{call.name} took longer than {found.timeout:g}s and was stopped")
        except PathNotAllowedError as exc:
            return _error(call, str(exc))
        except Exception as exc:  # a broken tool must not take the run down
            return _error(call, f"{call.name} failed: {type(exc).__name__}: {exc}")
        text = output if isinstance(output, str) else str(output)
        return ToolResult(call_id=call.id, content=_cap(text, found.max_output_chars))


async def _call(fn: Callable[..., object], kwargs: dict[str, object]) -> object:
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    # Sync tools run in a thread so the time limit can still fire. (The thread itself
    # cannot be killed; subprocess tools enforce their own timeouts.)
    return await asyncio.to_thread(fn, **kwargs)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... cut: {len(text) - limit} more characters]"


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, content=f"ERROR: {message}", is_error=True)


def _describe(exc: ValidationError) -> str:
    """`path: Field required; max_lines: Input should be a valid integer`."""
    parts: list[str] = []
    for err in exc.errors():
        where = ".".join(str(x) for x in err["loc"]) or "arguments"
        message = "unknown argument" if err["type"] == "extra_forbidden" else err["msg"]
        parts.append(f"{where}: {message}")
    return "; ".join(parts)


def _clean_schema(schema: JSONObject) -> JSONObject:
    """Drop pydantic's `title` keys: they cost tokens and tell the model nothing."""

    def clean(value: JSONValue) -> JSONValue:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k != "title"}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    cleaned = clean(schema)
    assert isinstance(cleaned, dict)
    return cleaned
