"""Demo tools for M2: `add` and `echo`. Enough to exercise the agent loop end to end.

Real tools (files, shell, python) and the `@tool` decorator arrive in M3.
"""

from kestrel.core.types import JSONValue, ToolCall, ToolResult, ToolSpec

ADD = ToolSpec(
    name="add",
    description="Add two numbers and return the sum.",
    parameters={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    },
)
ECHO = ToolSpec(
    name="echo",
    description="Return the given text unchanged.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
)


class DemoToolbox:
    """A `Toolbox` with `add` and `echo`. Mistakes come back as `ERROR: ...` results."""

    def specs(self) -> list[ToolSpec]:
        """Both demo tools."""
        return [ADD, ECHO]

    def precheck(self, call: ToolCall) -> ToolResult | None:
        """Nothing to check before running."""
        return None

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run `add` or `echo`."""
        args = call.arguments
        if call.name == "add":
            a, b = args.get("a"), args.get("b")
            if not (_is_number(a) and _is_number(b)):
                return _error(call, f"add needs numbers a and b, got {args}")
            assert isinstance(a, int | float) and isinstance(b, int | float)
            total = a + b
            text = str(int(total)) if float(total).is_integer() else str(total)
            return ToolResult(call_id=call.id, content=text)
        if call.name == "echo":
            text = args.get("text")
            if not isinstance(text, str):
                return _error(call, f"echo needs a string text, got {args}")
            return ToolResult(call_id=call.id, content=text)
        names = ", ".join(spec.name for spec in self.specs())
        return _error(call, f"unknown tool {call.name!r}. Available tools: {names}")


def _is_number(value: JSONValue) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(call_id=call.id, content=f"ERROR: {message}", is_error=True)
