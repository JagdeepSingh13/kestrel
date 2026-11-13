"""Seam: fetch() and page_text(), against a fake network (httpx.MockTransport)."""

import httpx
import pytest

from kestrel.tools.web.fetch import FetchBlockedError, Web, fetch, is_public
from kestrel.tools.web.text import page_text

PUBLIC = "93.184.215.14"
ARTICLE = (
    "<html><head><title>Uv  release</title><script>evil()</script></head><body>"
    "<nav>Home | Blog</nav><article><h1>uv 1.2</h1><p>" + "Released today. " * 20 + "</p>"
    "<ul><li>faster</li><li>smaller</li></ul></article><footer>(c) site</footer></body></html>"
)


def network(
    pages: dict[str, httpx.Response], dns: dict[str, list[str]] | None = None
) -> tuple[Web, list[httpx.Request]]:
    """A fake web: `pages` by Host header, every host resolving to PUBLIC unless `dns`."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return pages.get(request.headers["host"], httpx.Response(404))

    async def resolve(host: str, port: int) -> list[str]:
        return (dns or {}).get(host, [PUBLIC])

    return Web(transport=httpx.MockTransport(handle), resolve=resolve), seen


def html(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(status, text=body, headers={"content-type": "text/html; charset=utf-8"})


async def test_a_public_page_is_fetched_through_its_checked_address() -> None:
    web, seen = network({"example.com": html(ARTICLE)})
    page = await fetch(web, "https://example.com/post")
    assert page.status == 200 and page.url == "https://example.com/post"
    assert seen[0].url.host == PUBLIC  # connected to the address that was checked
    assert seen[0].extensions["sni_hostname"] == "example.com"
    title, text = page_text(page.text)
    assert title == "Uv release"
    assert text.startswith("# uv 1.2\nReleased today.")
    assert "- faster" in text and "evil" not in text and "Home" not in text and "(c)" not in text


@pytest.mark.parametrize(
    ("url", "dns", "why"),
    [
        ("http://localhost/", {"localhost": ["127.0.0.1"]}, "not a public address"),
        ("http://router.lan/", {"router.lan": ["192.168.1.1"]}, "not a public address"),
        ("http://mixed.com/", {"mixed.com": [PUBLIC, "10.0.0.5"]}, "not a public address"),
        ("http://[::ffff:127.0.0.1]/", None, "not a public address"),
        ("http://169.254.169.254/latest/meta-data", None, "not a public address"),
        ("http://example.com:8080/", None, "port 8080"),
        ("file:///etc/passwd", None, "only http and https"),
        ("http://user:pw@example.com/", None, "user name or password"),
    ],
)
async def test_what_is_refused(url: str, dns: dict[str, list[str]] | None, why: str) -> None:
    web, seen = network({}, dns)
    with pytest.raises(FetchBlockedError, match=why):
        await fetch(web, url)
    assert seen == []  # refused before anything was sent


async def test_redirects_are_checked_again() -> None:
    to_local = httpx.Response(302, headers={"location": "http://evil.com/"})
    web, _ = network({"example.com": to_local}, {"evil.com": ["127.0.0.1"]})
    with pytest.raises(FetchBlockedError, match="not a public address"):
        await fetch(web, "http://example.com/")
    hop = httpx.Response(301, headers={"location": "https://www.example.com/a"})
    web, _ = network({"example.com": hop, "www.example.com": html(ARTICLE)})
    assert (await fetch(web, "http://example.com/")).url == "https://www.example.com/a"


async def test_non_pages_and_huge_pages() -> None:
    pdf = httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
    web, _ = network({"example.com": pdf})
    with pytest.raises(FetchBlockedError, match="application/pdf"):
        await fetch(web, "https://example.com/a.pdf")
    web, _ = network({"example.com": html("x" * 5000)})
    page = await fetch(web, "https://example.com/", max_bytes=1000)
    assert page.cut and page.size == 1000


@pytest.mark.parametrize(
    "address", ["::7f00:1", "64:ff9b::a00:1", "::ffff:0:7f00:1", "fec0::1", "fe80::1%en0"]
)
def test_ipv6_forms_hiding_a_private_ipv4_are_not_public(address: str) -> None:
    assert not is_public(address)


def test_a_page_with_an_unclosed_header_still_has_text() -> None:
    assert page_text("<header><div>Menu<p>Real content here.")[1] == "Menu\nReal content here."
