"""OpenRouter's model list (M4.5): which models take tools, their window, free or paid.

Asked only when you list models on an OpenRouter provider or a run first steps up to
it: one GET of `/models?supported_parameters=tools` (no key needed to read it). Nothing
identifying is sent (no app name or referrer headers).
"""

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from kestrel.core.types import JSONValue


@dataclass(frozen=True)
class RemoteModel:
    """One model OpenRouter serves."""

    id: str
    window: int | None
    tools: bool
    free: bool


def is_openrouter(base_url: str) -> bool:
    """True for an OpenRouter base URL."""
    host = urlsplit(base_url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


async def openrouter_models(
    base_url: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> list[RemoteModel]:
    """Models that support tool calls. Raises `httpx.HTTPError` when unreachable."""
    url = base_url.rstrip("/") + "/models"
    async with httpx.AsyncClient(timeout=15.0, transport=transport, trust_env=False) as client:
        response = await client.get(url, params={"supported_parameters": "tools"})
        response.raise_for_status()
        data: JSONValue = response.json()
    entries = data.get("data") if isinstance(data, dict) else None
    models: list[RemoteModel] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            continue
        pricing = entry.get("pricing")
        prices = pricing if isinstance(pricing, dict) else {}
        params = entry.get("supported_parameters")
        window = entry.get("context_length")
        models.append(
            RemoteModel(
                id=str(entry["id"]),
                window=window if isinstance(window, int) else None,
                tools=isinstance(params, list) and "tools" in params,
                free=prices.get("prompt") == "0" and prices.get("completion") == "0",
            )
        )
    return models
