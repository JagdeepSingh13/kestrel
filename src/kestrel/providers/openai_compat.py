"""HTTP client for any OpenAI-compatible backend (Ollama, LM Studio, llama.cpp, ...).

Spec section 4.1. Retries only when the connection fails before the request is sent
(backend not up yet). A reply that has started is never retried silently.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import httpx

from kestrel.core.types import (
    BackendHTTPError,
    BackendTimeoutError,
    BackendUnreachableError,
    BadBackendResponseError,
    JSONObject,
    JSONValue,
    Message,
    ModelResponse,
    StreamChunk,
    ToolSpec,
)
from kestrel.providers.sse import StreamAssembler
from kestrel.providers.wire import ThinkStyle, from_response_body, to_request_body

# Models can take a long time before the first token (loading into memory), so the
# read timeout is generous; connecting should be quick on localhost.
DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=180.0, write=30.0, pool=5.0)
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF_SECONDS = 0.5

type Sleep = Callable[[float], Awaitable[None]]


class OpenAICompatProvider:
    """Talks to `{base_url}/chat/completions` and `{base_url}/models`.

    Implements `kestrel.core.types.Provider`. Use as an async context manager, or call
    `aclose()` when done, to release the connection pool.
    """

    def __init__(
        self,
        base_url: str,
        *,
        name: str = "openai_compat",
        api_key: str | None = None,
        timeout: httpx.Timeout = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Sleep = asyncio.sleep,
        think_style: ThinkStyle = "none",
        default_max_tokens: int | None = None,
    ) -> None:
        """Create a provider.

        `think_style` says how this backend is told not to think; `default_max_tokens`
        is sent when a call gives none (some backends cut replies short otherwise).
        `transport` and `sleep` exist for tests (fake network, no real waiting).
        """
        self.name = name
        self.think_style: ThinkStyle = think_style
        self.default_max_tokens = default_max_tokens
        self.base_url = base_url.rstrip("/")
        self._retries = retries
        self._backoff = backoff_seconds
        self._sleep = sleep
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(timeout=timeout, headers=headers, transport=transport)

    async def __aenter__(self) -> "OpenAICompatProvider":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP connections."""
        await self._client.aclose()

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
        """Send messages and wait for the whole reply."""
        body = to_request_body(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            json_schema=json_schema,
            think=think,
            think_style=self.think_style,
            stream=False,
        )
        response = await self._send("POST", "/chat/completions", body)
        try:
            await self._raise_for_status(response)
            return from_response_body(_decode_json(await self._read(response)))
        finally:
            await response.aclose()

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
        """Send messages and yield the reply as it arrives; the last chunk has `done`."""
        body = to_request_body(
            messages,
            tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            json_schema=json_schema,
            think=think,
            think_style=self.think_style,
            stream=True,
        )
        response = await self._send("POST", "/chat/completions", body)
        try:
            await self._raise_for_status(response)
            assembler = StreamAssembler()
            try:
                async for line in response.aiter_lines():
                    for chunk in assembler.feed_line(line):
                        yield chunk
            except httpx.TimeoutException as exc:
                raise BackendTimeoutError(self.base_url, "streaming the reply") from exc
            except httpx.TransportError as exc:
                # Never retried: part of the reply was already delivered.
                raise BadBackendResponseError(
                    f"connection dropped while streaming the reply ({type(exc).__name__})"
                ) from exc
            yield assembler.finish()
        finally:
            await response.aclose()

    async def list_models(self) -> list[str]:
        """Names of the models the backend can serve (`GET /models`)."""
        response = await self._send("GET", "/models")
        try:
            await self._raise_for_status(response)
            data = _decode_json(await self._read(response))
        finally:
            await response.aclose()
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise BadBackendResponseError("models response has no data list")
        return [
            model_id
            for item in items
            if isinstance(item, dict) and isinstance(model_id := item.get("id"), str)
        ]

    async def _send(self, method: str, path: str, body: JSONObject | None = None) -> httpx.Response:
        """Send a request, retrying only if the connection could not be made."""
        request = self._client.build_request(method, self.base_url + path, json=body)
        attempt = 0
        while True:
            try:
                return await self._client.send(request, stream=True)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt >= self._retries:
                    if isinstance(exc, httpx.ConnectTimeout):
                        raise BackendTimeoutError(self.base_url, "connecting") from exc
                    raise BackendUnreachableError(self.base_url) from exc
                await self._sleep(self._backoff * 2**attempt)
                attempt += 1
            except httpx.TimeoutException as exc:
                raise BackendTimeoutError(self.base_url, "waiting for the reply") from exc

    async def _read(self, response: httpx.Response) -> bytes:
        try:
            return await response.aread()
        except httpx.TimeoutException as exc:
            raise BackendTimeoutError(self.base_url, "reading the reply") from exc
        except httpx.TransportError as exc:
            raise BadBackendResponseError(
                f"connection dropped while reading the reply ({type(exc).__name__})"
            ) from exc

    async def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code >= 400:
            body = (await self._read(response)).decode("utf-8", errors="replace")
            raise BackendHTTPError(response.status_code, body)


def _decode_json(raw: bytes) -> JSONValue:
    try:
        value: JSONValue = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BadBackendResponseError(f"response is not JSON: {raw[:200]!r}") from exc
    return value
