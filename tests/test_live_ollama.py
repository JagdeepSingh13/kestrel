"""Live tests against a running Ollama with qwen3:8b. Run with `uv run pytest -m live`."""

import pytest

from kestrel.core.config import BUILTIN_PROVIDERS
from kestrel.core.types import Message, Role
from kestrel.providers.discovery import Status, discover
from kestrel.providers.openai_compat import OpenAICompatProvider

pytestmark = pytest.mark.live

MODEL = "qwen3:8b"
URL = BUILTIN_PROVIDERS["ollama"].base_url
# `/no_think` turns off qwen3's thinking so the live run stays fast.
QUESTION = [Message(role=Role.user, content="What is 2+2? Answer with just the number. /no_think")]


async def test_doctor_finds_ollama_and_model() -> None:
    [report] = await discover({"ollama": BUILTIN_PROVIDERS["ollama"]})
    assert report.status is Status.up, report.error
    assert MODEL in report.models


async def test_chat() -> None:
    async with OpenAICompatProvider(URL) as provider:
        response = await provider.chat(QUESTION, model=MODEL, temperature=0.0)
    assert "4" in (response.message.content or "")


async def test_stream() -> None:
    async with OpenAICompatProvider(URL) as provider:
        chunks = [c async for c in provider.stream(QUESTION, model=MODEL, temperature=0.0)]
    done = chunks[-1].done
    assert done is not None
    assert "4" in (done.message.content or "")
    assert done.usage is not None


async def test_agent_uses_the_add_tool() -> None:
    from kestrel.core.context import ContextManager
    from kestrel.core.events import EventBus
    from kestrel.core.loop import DEFAULT_SYSTEM_PROMPT, run_agent
    from kestrel.core.types import Event, EventType, StopReason
    from kestrel.tools.demo import DemoToolbox

    events: list[Event] = []
    bus = EventBus("live")
    bus.subscribe(events.append)
    async with OpenAICompatProvider(URL) as provider:
        result = await run_agent(
            "What is 17 + 25? Use the add tool. /no_think",
            provider=provider,
            model=MODEL,
            toolbox=DemoToolbox(),
            bus=bus,
            context=ContextManager(DEFAULT_SYSTEM_PROMPT),
        )
    assert result.stopped_reason is StopReason.final
    assert "42" in (result.answer or "")
    assert any(e.type is EventType.tool_call and e.data["name"] == "add" for e in events)
