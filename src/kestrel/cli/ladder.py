"""The shell's model ladder (M4.5): local models, then optionally an OpenRouter model.

The loop asks it which model to use each step (`core.types.Ladder`). Rungs above the
first are opened only when a run first steps up to them:
- a local rung is probed then (that also loads the model); a model without tool calls
  is skipped;
- an OpenRouter rung needs cloud allowed (`/cloud on`, or `allow_cloud: true` on the
  agent), an API key in the environment, and your yes before each step is sent. It
  then sees only the system prompt, the task, the failing step and the handoff note,
  with secrets removed (`CloudView`), never the older conversation.
After one good step on a higher rung the run steps back down; after 2 step-ups it stays
up. A "no" to a cloud step stops the run.
"""

import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import httpx
from rich.markup import escape

from kestrel.cli.output import err_console
from kestrel.core.config import ConfigError, KestrelConfig, ProviderConfig
from kestrel.core.context import NOTE_PREFIX
from kestrel.core.types import (
    BackendHTTPError,
    JSONObject,
    JSONValue,
    Message,
    ModelResponse,
    Provider,
    ProviderError,
    Role,
    Rung,
    StreamChunk,
    ToolCall,
    ToolSpec,
)
from kestrel.providers.openrouter import RemoteModel, is_openrouter, openrouter_models
from kestrel.providers.profiles import Profile, ProfileCache
from kestrel.safety.redact import Redactor

MAX_UPS = 2
"""Step-ups per run; after the last one the run stays up (never ping-pongs)."""
HANDOFF_START = f"{NOTE_PREFIX} a smaller model"

type Approve = Callable[[str, list[str]], Awaitable[Literal["yes", "no", "always"]]]
"""(question, detail lines) to the user's answer."""
type OpenProvider = Callable[
    [str, ProviderConfig], contextlib.AbstractAsyncContextManager[Provider]
]
type LoadProfile = Callable[[str, ProviderConfig, str, ProfileCache], Awaitable[Profile]]


@dataclass(frozen=True)
class RungSpec:
    """One rung as written: `qwen3:8b`, or `openrouter:google/gemma-4-26b-a4b-it:free`."""

    provider: str
    config: ProviderConfig
    model: str

    @property
    def remote(self) -> bool:
        """True when the model runs off this machine."""
        return self.config.is_remote

    @property
    def label(self) -> str:
        """How it is shown: remote rungs carry their provider name."""
        return f"{self.provider}:{self.model}" if self.remote else self.model


def ladder_home(config: KestrelConfig, agent_provider: str | None) -> str:
    """Where a ladder model without a prefix runs: the agent's own provider if it is on
    this machine, else the configured default. Never the shell's current provider: a
    `/model` look at OpenRouter must not move `qwen3:8b` there."""
    if agent_provider and not config.provider(agent_provider).is_remote:
        return agent_provider
    return config.default_provider


def looks_remote(name: str, config: KestrelConfig) -> str | None:
    """For a name without a prefix that looks like a cloud model id (`vendor/model`),
    the prefixed name you probably meant, if a remote provider is configured."""
    remote = next((n for n, c in config.providers.items() if c.is_remote), None)
    head = name.partition(":")[0]
    if remote is None or head in config.providers or "/" not in name:
        return None
    return f"{remote}:{name}"


def parse_rung(text: str, config: KestrelConfig, default_provider: str) -> RungSpec:
    """`provider:model` when the part before the first `:` names a configured provider,
    else a model on `default_provider` (so `qwen3:8b` stays one model name). Ladder
    callers pass `ladder_home(...)` as the default."""
    head, sep, rest = text.partition(":")
    if sep and head in config.providers and rest:
        return RungSpec(head, config.providers[head], rest)
    try:
        return RungSpec(default_provider, config.provider(default_provider), text)
    except ConfigError as exc:
        raise ValueError(str(exc)) from None


class ShellLadder:
    """A `Ladder` over `RungSpec`s, opening rungs lazily inside one run."""

    def __init__(
        self,
        base: Rung,
        uppers: Sequence[RungSpec],
        *,
        stack: contextlib.AsyncExitStack,
        open_provider: OpenProvider,
        load_profile: LoadProfile,
        cache: ProfileCache,
        env: dict[str, str],
        allow_cloud: bool,
        approve: Approve,
        agent_window: int,
        start: int = 0,
        emit: Callable[[str, JSONObject], None] | None = None,
    ) -> None:
        """`base` is the open first rung; `uppers` the rungs above it. `start` > 0 begins
        that many rungs up (`/up`)."""
        self.specs: list[RungSpec | None] = [None, *uppers]
        self.rungs: list[Rung | None] = [base, *([None] * len(uppers))]
        self.stack = stack
        self.open_provider = open_provider
        self.load_profile = load_profile
        self.cache = cache
        self.redactor = Redactor(env)
        self.env = env
        self.allow_cloud = allow_cloud
        self.approve = approve
        self.agent_window = agent_window
        self.start = start
        self.at = 0
        self.ups = 0
        self.cloud_always = False
        self.beginning = False
        self.remote: dict[str, RemoteModel] = {}
        """OpenRouter's facts per cloud rung (free or paid, window)."""
        self.emit = emit
        """Logs your cloud yes or no as a `permission` event."""
        self.started = False

    async def begin(self) -> None:
        """Move to the starting rung (after `/up`), opening it."""
        self.beginning = True
        for _ in range(self.start):
            rung = await self.step_up("asked", "/up or /retry up")
            if rung is None:
                _warn(f"staying on {self.current().name} for this message")
                break
        self.beginning = False
        self.start = self.at
        self.ups = 0  # /up is your choice, not a step-up the cap should count

    def current(self) -> Rung:
        """The rung in use."""
        rung = self.rungs[self.at]
        assert rung is not None
        return rung

    async def step_up(self, reason: str, detail: str) -> Rung | None:
        """The next rung that can take the step, or None (none left, or you said no)."""
        if self.ups >= MAX_UPS:
            return None
        window = self.current().window
        for index in range(self.at + 1, len(self.rungs)):
            spec = self.specs[index]
            assert spec is not None
            if reason == "context_overflow" and spec.remote:
                continue  # the cloud gets no history, so it cannot fix an overflow
            rung = self.rungs[index] or await self._open(spec)
            if rung is None:
                continue
            if reason == "context_overflow" and (rung.window or 0) <= (window or 0):
                continue
            self.rungs[index] = rung
            if spec.remote and not await self._cloud_ok(spec, reason, detail):
                return None  # you said no: the run stops
            self.at = index
            self.ups += 1
            return rung
        return None

    def after_step(self, ok: bool) -> Rung | None:
        """Back to the starting rung after one good step, unless the cap was reached."""
        if self.at == self.start or not ok or self.ups >= MAX_UPS:
            return None
        self.at = self.start
        return self.current()

    async def _cloud_ok(self, spec: RungSpec, reason: str, detail: str) -> bool:
        """Your yes before every step that goes to the cloud (unless "always")."""
        if self.cloud_always:
            return True
        info = self.remote.get(spec.label)
        price = "unknown price" if info is None else "free" if info.free else "paid"
        answer = await self.approve(
            f"Send this stuck step to {spec.model} on {spec.provider} ({price})?",
            [
                f"stuck: {reason} ({detail})",
                "sent: the instructions, your message (with the answer you rejected, after",
                "/retry up), the failing step, a short note, and the cloud model's own",
                "steps until it hands back; not sent: the rest of the conversation. Secret",
                "values from your environment are removed; file contents in those steps",
                "are sent as they are.",
            ],
        )
        if self.emit is not None:
            self.emit(
                "permission",
                {
                    "name": "cloud_step",
                    "model": spec.label,
                    "allowed": answer != "no",
                    "reason": f"user said {answer}",
                },
            )
        if answer == "no":
            _warn("cloud step declined" + ("" if self.beginning else ": the run stops here"))
            return False
        self.cloud_always = answer == "always"
        return True

    async def _open(self, spec: RungSpec) -> Rung | None:
        if spec.remote:
            return await self._open_cloud(spec)
        profile = await self.load_profile(spec.provider, spec.config, spec.model, self.cache)
        if profile.native_tools is False:
            _warn(f"{spec.model} cannot use tools; skipped on the ladder")
            return None
        provider = await self.stack.enter_async_context(
            self.open_provider(spec.provider, spec.config)
        )
        window = min(self.agent_window, profile.window or self.agent_window)
        return Rung(provider, spec.model, spec.label, window, True if profile.thinking else None)

    async def _open_cloud(self, spec: RungSpec) -> Rung | None:
        """Checks and opens a cloud rung; nothing is sent to the model here."""
        if not self.allow_cloud:
            _warn(
                f"{spec.label} is a cloud model and cloud is off (/cloud on to allow it); skipped"
            )
            return None
        if spec.config.api_key(self.env) is None:
            _warn(f"{spec.label}: set {spec.config.api_key_env or 'an API key'} to use it")
            return None
        info = await self._remote_info(spec)
        if info is not None and not info.tools:
            _warn(f"{spec.label} cannot use tools; skipped on the ladder")
            return None
        if info is not None:
            self.remote[spec.label] = info
        key = spec.config.api_key(self.env) or ""
        redactor = Redactor({**self.env, "KESTREL_CLOUD_KEY": key})  # the key, whatever its name
        provider = await self.stack.enter_async_context(
            self.open_provider(spec.provider, spec.config)
        )
        window = info.window if info is not None else None
        return Rung(CloudView(provider, redactor), spec.model, spec.label, window)

    async def _remote_info(self, spec: RungSpec) -> RemoteModel | None:
        if not is_openrouter(spec.config.base_url):
            return None
        try:
            models = await openrouter_models(spec.config.base_url)
        except (httpx.HTTPError, ValueError):
            return None  # the list is a bonus; the request itself will say what is wrong
        found = next((m for m in models if m.id == spec.model), None)
        if found is not None:
            self.cache.save(
                Profile(
                    backend=spec.config.base_url,
                    model=spec.model,
                    native_tools=found.tools,
                    window=found.window,
                )
            )
        return found


class CloudView:
    """A `Provider` that only lets the stuck step out: system prompt, task, the failing
    step, the handoff note and what came after it, with secrets removed."""

    def __init__(self, inner: Provider, redactor: Redactor) -> None:
        """Wrap `inner`, sending only `payload(messages)`."""
        self.inner = inner
        self.redactor = redactor
        self.name = inner.name

    def payload(self, messages: Sequence[Message]) -> list[Message]:
        """What leaves the machine for this request."""
        kept = list(messages)
        handoff = max(
            (i for i, m in enumerate(kept) if _is_handoff(m)),
            default=None,
        )
        if handoff is None:  # /up straight to the cloud: just this message and what follows
            task = max((i for i, m in enumerate(kept) if _is_task(m)), default=None)
            system = [m for m in kept if m.role is Role.system]
            kept = system + (kept[task:] if task is not None else [])
        else:
            task = max(
                (i for i in range(handoff) if _is_task(kept[i])),
                default=None,
            )
            failing = max(
                (
                    i
                    for i in range(handoff)
                    if kept[i].role is Role.assistant and kept[i].tool_calls
                ),
                default=None,
            )
            parts = [m for m in kept if m.role is Role.system]
            if task is not None:
                parts.append(kept[task])
            if failing is not None and (task is None or failing > task):
                parts.extend(kept[failing:handoff])
            parts.extend(kept[handoff:])
            kept = parts
        return [self._redact(m) for m in kept]

    async def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
        think: bool | None = None,
    ) -> ModelResponse:
        """`inner.chat` with only the payload."""
        try:
            return await self._chat(
                messages, tools, model, temperature, max_tokens, json_schema, think
            )
        except BackendHTTPError as exc:
            raise _plain(exc) from None

    async def _chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        model: str,
        temperature: float,
        max_tokens: int | None,
        json_schema: JSONObject | None,
        think: bool | None,
    ) -> ModelResponse:
        return await self.inner.chat(
            self.payload(messages),
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_schema=json_schema,
            think=think,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_schema: JSONObject | None = None,
        think: bool | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """`inner.stream` with only the payload."""
        try:
            async for chunk in self.inner.stream(
                self.payload(messages),
                tools,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_schema=json_schema,
                think=think,
            ):
                yield chunk
        except BackendHTTPError as exc:
            raise _plain(exc) from None

    async def list_models(self) -> list[str]:
        """The wrapped provider's models."""
        return await self.inner.list_models()

    def _redact(self, message: Message) -> Message:
        calls = [
            ToolCall(
                id=c.id,
                name=c.name,
                arguments=_as_object(self.redactor.json(c.arguments)),
                arguments_text=None
                if c.arguments_text is None
                else self.redactor.text(c.arguments_text),
                arguments_error=c.arguments_error,
            )
            for c in message.tool_calls
        ]
        content = None if message.content is None else self.redactor.text(message.content)
        return message.model_copy(update={"content": content, "tool_calls": calls})


def _plain(exc: BackendHTTPError) -> ProviderError:
    """Cloud errors in words, with the service's own reason (a free model is often
    rate-limited by the company serving it, whatever your own quota)."""
    words = {
        429: "the cloud model is rate-limited right now. Free models are often limited "
        "by the company serving them, not by your account (yours: 20 a minute, 50 a "
        "day). Try another free model, or later.",
        402: "the cloud account has no credits left for this model.",
        401: "the cloud API key was refused; check it.",
    }.get(exc.status)
    if words is None:
        return exc
    reason = _service_reason(exc.body)
    return ProviderError(f"{words} OpenRouter said: {reason}" if reason else words)


def _service_reason(body: str) -> str:
    """The error message inside an OpenRouter error body, if any (cut short)."""
    try:
        data: JSONValue = json.loads(body)
    except ValueError:
        return body.strip()[:200]
    error = _obj(_obj(data).get("error"))
    raw = _obj(error.get("metadata")).get("raw")
    text = raw if isinstance(raw, str) and raw else error.get("message")
    return str(text or "")[:200]


def _obj(value: JSONValue) -> JSONObject:
    return value if isinstance(value, dict) else {}


def _is_handoff(message: Message) -> bool:
    return message.role is Role.user and (message.content or "").startswith(HANDOFF_START)


def _is_task(message: Message) -> bool:
    return message.role is Role.user and not (message.content or "").startswith(NOTE_PREFIX)


def _as_object(value: object) -> JSONObject:
    return value if isinstance(value, dict) else {}  # type: ignore[return-value]


def _warn(text: str) -> None:
    err_console.print(f"  [yellow]ladder:[/yellow] {escape(text)}", highlight=False)
