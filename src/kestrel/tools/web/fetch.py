"""A guarded HTTP GET (M4.1): only public web pages, with size and time limits.

What it refuses, so a model (or a page it read) cannot use it to reach your machine or
your network:
- anything but http(s) on ports 80 and 443;
- a host whose address is not public (loopback, private ranges, link-local, ...): DNS is
  resolved first, every address must be public, and the connection then goes to that
  checked address, so a DNS answer that changes mid-request cannot redirect it;
- redirects past 5, and each redirect is checked again the same way;
- answers that are not web pages or text (PDFs, images), and more than `max_bytes`.
No cookies and no proxy settings from the environment are used.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import httpx

from kestrel.__about__ import __version__

type Resolver = Callable[[str, int], Awaitable[list[str]]]
"""Host and port in, IP addresses out."""

MAX_REDIRECTS = 5
PAGE_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
ALLOWED_PORTS = (80, 443)


class FetchBlockedError(Exception):
    """The URL or an answer was refused by a rule above (the message says which)."""


async def system_resolve(host: str, port: int) -> list[str]:
    """The machine's own DNS lookup, off the event loop."""
    infos = await asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


@dataclass(frozen=True)
class Web:
    """How to reach the web. Tests swap the transport and the resolver."""

    transport: httpx.AsyncBaseTransport | None = None
    resolve: Resolver = field(default=system_resolve)
    user_agent: str = f"Kestrel/{__version__} (local agent harness; research tool)"


@dataclass(frozen=True)
class Page:
    """One fetched page."""

    url: str
    """Where it ended up, after redirects."""
    status: int
    content_type: str
    text: str
    size: int
    """Bytes received (at most `max_bytes`)."""
    cut: bool = False
    """True when the page was longer than `max_bytes`."""


EMBEDS_IPV4 = (
    ipaddress.IPv6Network("::/96"),  # IPv4-compatible
    ipaddress.IPv6Network("64:ff9b::/96"),  # NAT64
    ipaddress.IPv6Network("::ffff:0:0/96"),  # IPv4-mapped
    ipaddress.IPv6Network("::ffff:0:0:0/96"),  # IPv4-translated
)
SITE_LOCAL = ipaddress.IPv6Network("fec0::/10")


def is_public(address: str) -> bool:
    """True for an address on the public internet. IPv6 forms that carry an IPv4
    address inside (mapped, compatible, NAT64) are judged by that IPv4 address."""
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in SITE_LOCAL:
            return False
        if any(ip in net for net in EMBEDS_IPV4) and int(ip) & 0xFFFFFFFF:
            ip = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip.is_global and not ip.is_multicast


async def fetch(
    web: Web,
    url: str,
    *,
    max_bytes: int = 2_000_000,
    seconds: float = 10.0,
    method: str = "GET",
    data: dict[str, str] | None = None,
) -> Page:
    """Get one page under the rules above. Raises `FetchBlockedError`, or an
    `httpx.HTTPError` when the network fails."""
    async with httpx.AsyncClient(
        transport=web.transport,
        follow_redirects=False,
        trust_env=False,
        timeout=seconds,
        headers={"User-Agent": web.user_agent},
    ) as client:
        for _ in range(MAX_REDIRECTS + 1):
            request = await _checked_request(web, client, url, method, data)
            response = await client.send(request, stream=True)
            try:
                location = response.headers.get("location")
                if response.is_redirect and location:
                    url = urljoin(url, location)
                    method, data = "GET", None  # a redirect is always followed with GET
                    continue
                return await _read(response, url, max_bytes)
            finally:
                await response.aclose()
    raise FetchBlockedError(f"more than {MAX_REDIRECTS} redirects")


async def _checked_request(
    web: Web, client: httpx.AsyncClient, url: str, method: str, data: dict[str, str] | None
) -> httpx.Request:
    """A request to the checked address of `url`'s host, with the real Host and TLS name."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchBlockedError(f"only http and https links are fetched, not {parts.scheme!r}")
    try:
        host = (parts.hostname or "").encode("idna").decode("ascii")  # bücher.de as xn--...
    except UnicodeError:
        raise FetchBlockedError(f"{parts.hostname!r} is not a valid host name") from None
    if not host:
        raise FetchBlockedError("the link has no host")
    if parts.username or parts.password:
        raise FetchBlockedError("links with a user name or password are not fetched")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise FetchBlockedError(f"port {port} is not fetched (only 80 and 443)")
    try:
        addresses = [str(ipaddress.ip_address(host))]
    except ValueError:
        try:
            addresses = await web.resolve(host, port)
        except OSError as exc:
            raise FetchBlockedError(f"{host} could not be found ({exc})") from None
    if not addresses:
        raise FetchBlockedError(f"{host} could not be found")
    private = [a for a in addresses if not is_public(a)]
    if private:
        raise FetchBlockedError(f"{host} points to {private[0]}, which is not a public address")
    address = addresses[0]
    literal = f"[{address}]" if ":" in address else address
    target = parts._replace(netloc=f"{literal}:{port}").geturl()
    extensions = {"sni_hostname": host} if parts.scheme == "https" else {}
    return client.build_request(
        method,
        target,
        headers={"Host": host if parts.port is None else f"{host}:{parts.port}"},
        data=data,
        extensions=extensions,
    )


async def _read(response: httpx.Response, url: str, max_bytes: int) -> Page:
    kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if response.status_code < 400 and kind and kind not in PAGE_TYPES:
        raise FetchBlockedError(f"not a web page or text ({kind})")
    body = bytearray()
    cut = False
    async for chunk in response.aiter_bytes():
        body += chunk
        if len(body) > max_bytes:
            del body[max_bytes:]
            cut = True
            break
    charset = response.charset_encoding or "utf-8"
    try:
        text = bytes(body).decode(charset, errors="replace")
    except LookupError:  # an unknown charset name
        text = bytes(body).decode("utf-8", errors="replace")
    return Page(url, response.status_code, kind, text, len(body), cut)
