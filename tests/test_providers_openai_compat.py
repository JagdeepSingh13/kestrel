"""Seam 4: OpenAICompatProvider over httpx's fake network (MockTransport). No real HTTP."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable

import httpx
import pytest

from kestrel.core.types import (
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnreachableError,
    BadBackendResponseError,
    JSONValue,
    Message,
    Provider,
    Role,
    StreamChunk,
    ToolCall,
)
from kestrel.providers.openai_compat import OpenAICompatProvider

BASE = "http://localhost:11434/v1"
HELLO = [Message(role=Role.user, content="hello")]

type Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class Recorder:
    """Fake backend: records requests, answers with the given handler."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.delays: list[float] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self.handler(request)

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)  # record instead of really waiting

    def provider(self, **kwargs: str) -> OpenAICompatProvider:
        return OpenAICompatProvider(
            kwargs.get("base_url", BASE),
            api_key=kwargs.get("api_key"),
            transport=httpx.MockTransport(self),
            sleep=self.sleep,
        )


def reply(payload: JSONValue, status: int = 200) -> Handler:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def sse(*events: JSONValue | str) -> bytes:
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    return "".join(f"data: {line}\n\n" for line in lines).encode()


TEXT_REPLY: JSONValue = {
    "choices": [{"message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 9, "completion_tokens": 2},
}


def test_satisfies_provider_protocol() -> None:
    provider: Provider = OpenAICompatProvider(BASE)  # pyright checks this line
    assert provider.name == "openai_compat"


async def test_chat_sends_openai_request_and_parses_reply() -> None:
    backend = Recorder(reply(TEXT_REPLY))
    async with backend.provider(base_url=BASE + "/") as provider:
        response = await provider.chat(HELLO, model="qwen3:8b", max_tokens=50)

    request = backend.requests[0]
    assert request.method == "POST"
    assert str(request.url) == f"{BASE}/chat/completions"  # trailing slash handled
    assert json.loads(request.content) == {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "stream": False,
        "max_tokens": 50,
    }
    assert "authorization" not in request.headers  # no key configured
    assert response.message.content == "Hi!"
    assert response.usage is not None
    assert response.usage.completion_tokens == 2


async def test_api_key_is_sent_as_bearer_token() -> None:
    backend = Recorder(reply(TEXT_REPLY))
    async with backend.provider(api_key="sk-test") as provider:
        await provider.chat(HELLO, model="m")
    assert backend.requests[0].headers["authorization"] == "Bearer sk-test"


async def test_stream_yields_pieces_then_done() -> None:
    body = sse(
        {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}]},
        {"choices": [{"delta": {"content": " there"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "c1",
                                "function": {"name": "echo", "arguments": "{}"},
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 4}},
        "[DONE]",
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    backend = Recorder(handler)
    async with backend.provider() as provider:
        chunks = [c async for c in provider.stream(HELLO, model="m")]

    assert json.loads(backend.requests[0].content)["stream_options"] == {"include_usage": True}
    assert [c.text for c in chunks[:-1]] == ["Hi", " there"]
    done = chunks[-1].done
    assert done is not None
    assert done.message.content == "Hi there"
    assert done.message.tool_calls == [ToolCall(id="c1", name="echo", arguments={})]
    assert done.usage is not None
    assert done.usage.prompt_tokens == 5


async def test_list_models_reads_ids() -> None:
    backend = Recorder(reply({"object": "list", "data": [{"id": "qwen3:8b"}, {"id": "gemma3:4b"}]}))
    async with backend.provider() as provider:
        assert await provider.list_models() == ["qwen3:8b", "gemma3:4b"]
    assert backend.requests[0].method == "GET"
    assert str(backend.requests[0].url) == f"{BASE}/models"


async def test_http_error_carries_status_and_body() -> None:
    backend = Recorder(reply({"error": {"message": 'model "nope" not found'}}, status=404))
    async with backend.provider() as provider:
        with pytest.raises(BackendHTTPError) as caught:
            await provider.chat(HELLO, model="nope")
        assert caught.value.status == 404
        assert "not found" in caught.value.body

        with pytest.raises(BackendHTTPError, match="HTTP 404"):
            async for _ in provider.stream(HELLO, model="nope"):
                pass


async def test_connection_refused_is_retried_then_succeeds() -> None:
    failures = 2

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal failures
        if failures:
            failures -= 1
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json=TEXT_REPLY)

    backend = Recorder(handler)
    async with backend.provider() as provider:
        response = await provider.chat(HELLO, model="m")
    assert response.message.content == "Hi!"
    assert len(backend.requests) == 3
    assert backend.delays == [0.5, 1.0]  # backoff doubles


async def test_connection_refused_every_time_gives_friendly_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = Recorder(handler)
    async with backend.provider() as provider:
        with pytest.raises(
            BackendUnreachableError, match="Nothing is listening at http://localhost"
        ):
            await provider.list_models()
    assert len(backend.requests) == 3  # first try + 2 retries


async def test_read_timeout_is_not_retried() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    backend = Recorder(handler)
    async with backend.provider() as provider:
        with pytest.raises(BackendTimeoutError, match="waiting for the reply"):
            await provider.chat(HELLO, model="m")
    assert len(backend.requests) == 1


class DroppingStream(httpx.AsyncByteStream):
    """Sends one piece of the reply, then the connection breaks."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield sse({"choices": [{"delta": {"content": "Hel"}}]})
        raise httpx.ReadError("connection reset")


async def test_stream_dropped_midway_is_reported_not_retried() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=DroppingStream())

    backend = Recorder(handler)
    received: list[StreamChunk] = []
    async with backend.provider() as provider:
        with pytest.raises(BadBackendResponseError, match="connection dropped"):
            async for chunk in provider.stream(HELLO, model="m"):
                received.append(chunk)
    assert [c.text for c in received] == ["Hel"]  # what arrived was still delivered
    assert len(backend.requests) == 1


async def test_non_json_body_is_bad_response() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>proxy error</html>")

    backend = Recorder(handler)
    async with backend.provider() as provider:
        with pytest.raises(BadBackendResponseError, match="not JSON"):
            await provider.chat(HELLO, model="m")
