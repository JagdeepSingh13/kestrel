"""What an agent file may contain (spec section 10)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kestrel.core.context import DEFAULT_MAX_TOOL_RESULT_TOKENS, DEFAULT_WINDOW


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Limits(_Strict):
    """Budgets for one run."""

    max_steps: int = Field(default=15, ge=1, le=100)
    max_tokens_total: int | None = Field(default=None, ge=1)
    max_wall_seconds: float | None = Field(default=None, gt=0)


class ContextSettings(_Strict):
    """How much the model sees."""

    window: int = Field(default=DEFAULT_WINDOW, ge=1024)
    max_tool_result_tokens: int = Field(default=DEFAULT_MAX_TOOL_RESULT_TOKENS, ge=100)


class AgentConfig(_Strict):
    """One agent: model, prompt, tools, permissions and limits."""

    name: str = Field(pattern=r"^[a-z0-9_-]{1,40}$")
    description: str = ""
    model: list[str] = Field(default_factory=list[str])
    """Ordered: the first is used; later ones are the ladder (M4.5). Empty: the shell's model."""
    provider: str | None = None
    """Empty: the shell's current provider."""
    allow_cloud: bool = False
    tool_mode: Literal["auto", "native", "json", "react"] = "auto"
    """Only native tool calls exist until M4; other modes are accepted and ignored."""
    temperature: float = Field(default=0.2, ge=0, le=2)
    system: str | None = None
    """The agent's instructions. Empty: Kestrel's default system prompt."""
    tools: list[str] = Field(default_factory=list[str])
    mcp: dict[str, list[str]] = Field(default_factory=dict[str, list[str]])
    """MCP tools by server (M5): `{fs: [read_text_file, list_directory], git: ["*"]}`.
    Servers are defined in your config only; they appear as `fs__read_text_file`."""
    permissions: dict[str, Literal["allow", "confirm", "deny"]] = Field(
        default_factory=dict[str, Literal["allow", "confirm", "deny"]]
    )
    limits: Limits = Limits()
    context: ContextSettings = ContextSettings()

    @field_validator("model", mode="before")
    @classmethod
    def _one_or_many(cls, value: object) -> object:
        return [value] if isinstance(value, str) else value
