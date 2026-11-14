"""MCP tools (M5): tools from MCP servers you configure, offered to agents that ask.

- Servers are defined only in your config (`[mcp_servers.NAME]`), never in agent files.
- `McpHub` keeps each server running for the whole shell session in one background
  event loop (the shell runs each turn in its own loop; an MCP connection must stay in
  the loop that opened it). A server starts the first time an agent that uses it runs.
- `mcp_tools` turns a server's tools into Kestrel tools named `server__tool`, with short
  descriptions and plain schemas for small models. Output, errors included, is outside
  data (wrapped as untrusted). Tools ask before running, except those in the server's
  `allow` list.
"""

import asyncio
import contextlib
import copy
import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any, TextIO

from pydantic import BaseModel, ConfigDict, Field, create_model

from kestrel.core.config import McpServerConfig
from kestrel.core.types import JSONObject, JSONValue, ToolSpec
from kestrel.tools.registry import OutsideError, Tool, ToolContext, ToolError

DESCRIPTION_CHARS = 300
PARAM_DESCRIPTION_CHARS = 120
START_SECONDS = 120.0
"""A first `npx` start downloads the server; later starts take a second or two."""
NAME_LIMIT = 64
DROPPED_KEYS = frozenset({"$schema", "title", "examples", "additionalProperties", "$defs"})

type Connect = Callable[[str, McpServerConfig, TextIO], Any]
"""(server name, config, its log file) to an MCP `Client` (tests pass an in-process one)."""


class McpStartError(Exception):
    """A server could not be started or listed."""


class McpHub:
    """The running MCP servers of one shell (or one `kestrel run`)."""

    def __init__(
        self,
        servers: Mapping[str, McpServerConfig],
        *,
        workspace: Path,
        env: Mapping[str, str],
        log_dir: Path,
        connect: Connect | None = None,
    ) -> None:
        """`servers` from the config; `{workspace}` in their args becomes `workspace`."""
        self.servers = dict(servers)
        self.workspace = workspace
        self.env = env
        self.log_dir = log_dir
        self.connect = connect or self._connect
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._clients: dict[str, Any] = {}
        self._tools: dict[str, list[Any]] = {}
        self._stops: dict[str, asyncio.Event] = {}
        self._done: dict[str, Future[None]] = {}
        self._lock = threading.Lock()

    # running servers

    def running(self) -> list[str]:
        """Names of the servers running now."""
        return sorted(n for n in self._clients if n in self._tools)

    def tools(self, name: str) -> list[Any]:
        """Start `name` if needed; its tools (MCP `Tool` objects). Raises McpStartError."""
        with self._lock:
            if name in self._tools:
                return self._tools[name]
            if name not in self.servers:
                raise McpStartError(f"no MCP server named {name!r} in your config")
            loop = self._ensure_loop()
            ready: Future[list[Any]] = Future()
            stop = asyncio.Event()
            self._stops[name] = stop
            self._done[name] = asyncio.run_coroutine_threadsafe(self._own(name, ready, stop), loop)
            try:
                tools = ready.result(timeout=START_SECONDS)
            except Exception as exc:
                loop.call_soon_threadsafe(stop.set)  # never leave a half-started server
                done = self._done.pop(name, None)
                if done is not None:
                    done.cancel()
                self._stops.pop(name, None)
                self._clients.pop(name, None)
                why = "no answer in time" if isinstance(exc, TimeoutError) else _why(exc)
                raise McpStartError(f"MCP server {name} did not start: {why}") from None
            self._tools[name] = tools
            return tools

    async def call(self, name: str, tool: str, arguments: JSONObject, seconds: float) -> Any:
        """Call a tool on a running server, from any event loop. Esc (cancel) stops the
        wait and the call."""
        client = self._clients.get(name)
        if client is None or self._loop is None:
            raise ToolError(f"MCP server {name} is not running; /mcp restart {name}")
        future = asyncio.run_coroutine_threadsafe(
            client.call_tool(tool, arguments, read_timeout_seconds=seconds), self._loop
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            if _closed(exc):  # the server process is gone: start it afresh next time
                self._forget(name)
                raise ToolError(
                    f"MCP server {name} stopped. It starts again with your next message "
                    f"(or /mcp restart {name})."
                ) from None
            raise

    def _forget(self, name: str) -> None:
        """Drop a dead server so the next use starts it again."""
        stop = self._stops.pop(name, None)
        if stop is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(stop.set)
        self._done.pop(name, None)
        self._tools.pop(name, None)
        self._clients.pop(name, None)

    def restart(self, name: str) -> list[Any]:
        """Stop and start a server again."""
        self.stop(name)
        return self.tools(name)

    def stop(self, name: str) -> None:
        """Stop one server (no-op if it is not running)."""
        stop, done = self._stops.pop(name, None), self._done.pop(name, None)
        self._tools.pop(name, None)
        if stop is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(stop.set)
            with contextlib.suppress(Exception):
                if done is not None:
                    done.result(timeout=10)
        self._clients.pop(name, None)

    def close(self) -> None:
        """Stop every server and the background loop (when the shell ends)."""
        for name in list(self._stops):
            self.stop(name)
        if self._loop is not None:
            loop = self._loop
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            if not loop.is_running():
                loop.close()
            self._loop = None

    # inside the hub's loop

    async def _own(self, name: str, ready: Future[list[Any]], stop: asyncio.Event) -> None:
        """Owns one server's connection from start to stop (anyio wants one task to open
        and close it)."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            with (self.log_dir / f"{name}.log").open("a", encoding="utf-8") as log:
                async with self.connect(name, self.servers[name], log) as client:
                    listed: list[Any] = []
                    cursor: str | None = None
                    while True:
                        page = await client.list_tools(cursor=cursor)
                        listed.extend(page.tools)
                        cursor = getattr(page, "next_cursor", None)
                        if not cursor:
                            break
                    self._clients[name] = client
                    ready.set_result(listed)
                    await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            if not isinstance(exc, Exception):
                raise
        finally:
            self._clients.pop(name, None)
            if self._stops.get(name) is stop:  # ended on its own (crashed): start afresh
                self._tools.pop(name, None)
                self._stops.pop(name, None)

    def _connect(self, name: str, server: McpServerConfig, log: TextIO) -> Any:
        from mcp import Client, StdioServerParameters, stdio_client

        if server.url is not None:
            return Client(server.url)
        assert server.command is not None
        args = [a.replace("{workspace}", str(self.workspace)) for a in server.args]
        passed = {k: self.env[k] for k in server.env_pass if k in self.env}
        params = StdioServerParameters(
            command=server.command, args=args, env=passed or None, cwd=self.workspace
        )
        return Client(stdio_client(params, errlog=log))

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="kestrel-mcp", daemon=True)
            thread.start()
            self._loop, self._thread = loop, thread
        return self._loop


def command_line(name: str, server: McpServerConfig, workspace: Path) -> str:
    """How a server is started, as shown on screen."""
    if server.url is not None:
        return server.url
    args = [a.replace("{workspace}", str(workspace)) for a in server.args]
    return " ".join([server.command or "", *args])


def mcp_tools(
    hub: McpHub,
    server: str,
    wanted: list[str],
    listed: list[Any],
) -> list[Tool]:
    """Kestrel tools for `wanted` tool names of `server` (`*`: all). Raises ToolError
    naming the real tools when one is unknown."""
    by_name = {t.name: t for t in listed}
    names = list(by_name) if "*" in wanted else wanted
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ToolError(
            f"MCP server {server} has no tool {unknown}. It has: {', '.join(by_name) or 'none'}"
        )
    seen: dict[str, str] = {}
    for n in names:
        other = seen.setdefault(tool_name(server, n), n)
        if other != n:
            raise ToolError(f"MCP tools {other!r} and {n!r} of {server} get the same name")
    config = hub.servers[server]
    return [_tool(hub, server, by_name[n], allowed=n in config.allow) for n in names]


def tool_name(server: str, tool: str) -> str:
    """`fs__read_text_file`: safe characters, at most 64 long."""
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", tool)
    name = f"{server}__{safe}"
    if len(name) > NAME_LIMIT:
        digest = hashlib.sha256(tool.encode()).hexdigest()[:6]
        name = f"{name[: NAME_LIMIT - 7]}_{digest}"
    return name


def _tool(hub: McpHub, server: str, mcp_tool: Any, *, allowed: bool) -> Tool:
    schema = clean_schema(dict(mcp_tool.input_schema or {"type": "object"}))
    props = schema.get("properties")
    properties: dict[str, Any] = dict(props) if isinstance(props, dict) else {}
    required_list = schema.get("required")
    required: set[str] = (
        {str(x) for x in required_list} if isinstance(required_list, list) else set()
    )
    # Field names are safe stand-ins; the real parameter names are aliases (pydantic
    # drops names starting with "_" and warns about ones like "schema").
    real: dict[str, str] = {f"arg_{i}": name for i, name in enumerate(properties)}
    fields: dict[str, Any] = {
        field: (Any, Field(... if name in required else None, alias=name))
        for field, name in real.items()
    }
    arguments: type[BaseModel] = create_model(
        f"{server}_{mcp_tool.name}_arguments",
        __config__=ConfigDict(extra="forbid", validate_by_alias=True, validate_by_name=False),
        **fields,
    )
    timeout = hub.servers[server].timeout
    raw_name = mcp_tool.name

    async def run(ctx: ToolContext, **kwargs: Any) -> str:
        named = {real[k]: v for k, v in kwargs.items()}
        sent = {k: v for k, v in named.items() if v is not None or k in required}
        try:
            result = await hub.call(server, raw_name, sent, timeout)
        except ToolError:
            raise
        except TimeoutError:
            raise ToolError(f"{server} took longer than {timeout:g}s") from None
        except Exception as exc:  # MCPError and transport failures: the server's words
            raise OutsideError(f"{server}/{raw_name} failed: {_why(exc)}") from None
        text = result_text(result)
        if getattr(result, "is_error", False):
            raise OutsideError(text or f"{server}/{raw_name} reported an error")
        return text or "(no output)"

    spec = ToolSpec(
        name=tool_name(server, raw_name),
        description=_short(mcp_tool.description or raw_name, DESCRIPTION_CHARS)
        + f" (MCP server {server})",
        parameters=schema,
        permission="allow" if allowed else "confirm",
        side_effects=not allowed,
    )
    return Tool(spec, run, arguments, True, timeout + 5, None, 20_000)


def result_text(result: Any) -> str:
    """Text of an MCP tool result: text blocks joined; other blocks named, not shown."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(str(getattr(block, "text", "")))
        else:
            mime = getattr(block, "mime_type", None) or kind
            parts.append(f"[{kind} omitted: {mime}]")
    if not any(p.strip() for p in parts):
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return json.dumps(structured, ensure_ascii=False)
    return "\n".join(parts)


def clean_schema(schema: JSONObject) -> JSONObject:
    """A plain schema for small models: local `$ref`s inlined, noise keys dropped,
    descriptions cut short."""
    defs = schema.get("$defs")
    definitions: JSONObject = dict(defs) if isinstance(defs, dict) else {}

    def walk(value: JSONValue, depth: int) -> JSONValue:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/") and depth < 8:
                target = definitions.get(ref.removeprefix("#/$defs/"))
                if isinstance(target, dict):
                    return walk(copy.deepcopy(target), depth + 1)
            out: JSONObject = {}
            for key, item in value.items():
                if key == "properties" and isinstance(item, dict):
                    # Names of parameters, not schema keywords: a parameter may well be
                    # called "title" or "examples".
                    out[key] = {name: walk(sub, depth) for name, sub in item.items()}
                    continue
                if key in DROPPED_KEYS:
                    continue
                if key == "description" and isinstance(item, str):
                    out[key] = _short(item, PARAM_DESCRIPTION_CHARS)
                else:
                    out[key] = walk(item, depth)
            return out
        if isinstance(value, list):
            return [walk(v, depth) for v in value]
        return value

    cleaned = walk(schema, 0)
    assert isinstance(cleaned, dict)
    cleaned.setdefault("type", "object")
    return cleaned


def _short(text: str, limit: int) -> str:
    first = " ".join(text.strip().split("\n\n", 1)[0].split())
    return first if len(first) <= limit else first[: limit - 3] + "..."


def _closed(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        sign in text for sign in ("connection closed", "closedresource", "broken pipe", "eof")
    ) or type(exc).__name__ in ("ClosedResourceError", "BrokenResourceError", "EndOfStream")


def _why(exc: BaseException) -> str:
    """The most useful message inside an exception (task groups wrap the real one)."""
    inner = getattr(exc, "exceptions", None)
    if inner:
        return _why(inner[0])
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
