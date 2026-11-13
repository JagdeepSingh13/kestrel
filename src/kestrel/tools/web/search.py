"""Web search for the research tool (M4.1): DuckDuckGo's plain HTML results page.

No API key, no account, no extra dependency: one POST to html.duckduckgo.com per call,
with Kestrel's honest user agent, parsed with the standard library. DuckDuckGo may answer
automated requests with a rate-limit page or a puzzle instead of results; that becomes a
clear error, never a retry loop. (A self-hosted SearXNG backend is planned for later.)
"""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

from kestrel.tools.web.fetch import Web, fetch

DUCKDUCKGO = "https://html.duckduckgo.com/html/"
MAX_RESULTS = 8
UNSAFE = re.compile(r"[\s<>\"'`\x00-\x1f\x7f]")
BLOCK_SIGNS = ("anomaly", "challenge-form", "bots use duckduckgo")


class SearchFailedError(Exception):
    """The search engine gave no usable results page."""


@dataclass(frozen=True)
class Result:
    """One search hit."""

    title: str
    url: str
    snippet: str


async def search(web: Web, query: str, *, limit: int = MAX_RESULTS) -> list[Result]:
    """The top results for `query`. Raises `SearchFailedError` when DuckDuckGo refuses."""
    page = await fetch(web, DUCKDUCKGO, method="POST", data={"q": query}, max_bytes=1_000_000)
    results = parse_results(page.text) if page.status < 400 else []
    # Only a page without results can be a refusal: a result may well mention "anomaly".
    refused = page.status == 202 or (
        not results and any(sign in page.text.lower() for sign in BLOCK_SIGNS)
    )
    if refused:
        raise SearchFailedError(
            "DuckDuckGo refused the search (it limits automated requests). Try again in a "
            "few minutes."
        )
    if page.status >= 400:
        raise SearchFailedError(f"DuckDuckGo answered with HTTP {page.status}")
    return results[:limit]


def parse_results(html: str) -> list[Result]:
    """Results from a DuckDuckGo HTML page, ads and duplicates left out."""
    parser = _Results()
    parser.feed(html)
    parser.close()
    seen: set[str] = set()
    results: list[Result] = []
    for title, href, snippet in parser.found:
        url = _real_url(href)
        if url is None or url in seen:
            continue
        seen.add(url)
        results.append(Result(" ".join(title.split()), url, " ".join(snippet.split())))
    return results


def _real_url(href: str) -> str | None:
    """DuckDuckGo wraps links as //duckduckgo.com/l/?uddg=<url>; ads go through y.js."""
    if href.startswith("//"):
        href = "https:" + href
    try:
        parts = urlsplit(href)
        if parts.hostname and parts.hostname.endswith("duckduckgo.com"):
            if parts.path.startswith("/y.js"):
                return None  # an ad
            target = parse_qs(parts.query).get("uddg")
            href = target[0] if target else ""
        if urlsplit(href).scheme not in ("http", "https"):
            return None
    except ValueError:  # a malformed link (e.g. a broken IPv6 host): skip it, not the search
        return None
    # A link is shown to the model and the user: refuse ones that could break out of a tag.
    if UNSAFE.search(href):
        return None
    return href


class _Results(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[tuple[str, str, str]] = []
        self._field: str | None = None
        self._title = ""
        self._href = ""
        self._snippet = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        named = dict(attrs)
        classes = (named.get("class") or "").split()
        if "result__a" in classes:
            self._flush()
            self._field, self._href, self._title = "title", named.get("href") or "", ""
        elif "result__snippet" in classes:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field == "title":
            self._title += data
        elif self._field == "snippet":
            self._snippet += data

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._href:
            self.found.append((self._title, self._href, self._snippet))
        self._href, self._title, self._snippet = "", "", ""
