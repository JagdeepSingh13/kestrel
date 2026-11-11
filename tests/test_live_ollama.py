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
