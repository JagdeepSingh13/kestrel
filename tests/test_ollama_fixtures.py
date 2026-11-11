"""Replays of real Ollama 0.40 replies (qwen3:8b), recorded into tests/fixtures/ollama/.

Ollama sends qwen3's thinking in a `reasoning` field (not `<think>` tags) and streams
`"content": ""` while thinking. These tests lock that in without a running model.
"""

from pathlib import Path

import httpx

from kestrel.core.types import Message, Role
from kestrel.providers.openai_compat import OpenAICompatProvider

FIXTURES = Path(__file__).parent / "fixtures" / "ollama"
HELLO = [Message(role=Role.user, content="What is 2+2? Answer with just the number.")]


def replay(name: str, content_type: str) -> OpenAICompatProvider:
    body = (FIXTURES / name).read_bytes()

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    return OpenAICompatProvider("http://localhost:11434/v1", transport=httpx.MockTransport(handler))


async def test_recorded_chat_splits_answer_from_reasoning() -> None:
    async with replay("qwen3_chat.json", "application/json") as provider:
        response = await provider.chat(HELLO, model="qwen3:8b")
    assert response.message.content == "4"
    assert response.message.reasoning is not None
    assert response.message.reasoning.startswith("Okay")
    assert response.usage is not None
    assert response.usage.completion_tokens == 320


async def test_recorded_stream_yields_reasoning_then_answer() -> None:
    async with replay("qwen3_stream.sse", "text/event-stream") as provider:
        chunks = [c async for c in provider.stream(HELLO, model="qwen3:8b")]
    text = "".join(c.text for c in chunks)
    reasoning = "".join(c.reasoning for c in chunks)
    assert text.strip() == "4"
    assert reasoning.startswith("Okay")
    first_text = next(i for i, c in enumerate(chunks) if c.text)
    assert all(not c.text for c in chunks[:first_text])  # all thinking comes first
    done = chunks[-1].done
    assert done is not None
    assert done.message.content == text
    assert done.usage is not None
    assert done.usage.prompt_tokens == 23


async def test_recorded_models_list() -> None:
    async with replay("models.json", "application/json") as provider:
        assert await provider.list_models() == ["qwen3:8b"]
