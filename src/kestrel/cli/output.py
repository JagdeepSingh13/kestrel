"""Terminal output shared by `kestrel ask`, `kestrel doctor` and the interactive shell."""

import os
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from kestrel.core.config import ProviderConfig
from kestrel.core.types import BadBackendResponseError, Message, ModelResponse, Provider, Usage
from kestrel.providers.discovery import BackendReport, Status
from kestrel.providers.openai_compat import OpenAICompatProvider

console = Console()
"""Standard output: only the model's answer goes here, so `kestrel ask ... > file` works."""
err_console = Console(stderr=True)
"""Standard error: thinking, progress, token counts, warnings and errors."""

STATUS_STYLE = {Status.up: "green", Status.down: "dim", Status.error: "red"}


@asynccontextmanager
async def open_provider(name: str, config: ProviderConfig) -> AsyncGenerator[Provider]:
    """Create the provider for `config` and close it afterwards. Tests replace this."""
    api_key = config.api_key(os.environ)
    async with OpenAICompatProvider(config.base_url, name=name, api_key=api_key) as provider:
        yield provider


async def stream_answer(
    provider: Provider,
    messages: Sequence[Message],
    *,
    model: str,
    show_thinking: bool = False,
) -> ModelResponse:
    """Print the reply as it arrives and return the complete reply.

    A spinner runs until the first visible output (models can take a while to load or to
    think). Reasoning is printed dimmed to stderr only with `show_thinking`.
    """
    spinner = err_console.status(f"Waiting for {model}...")
    spinner.start()
    spinning = True
    printed_thinking = False
    printed_text = False
    final: ModelResponse | None = None
    try:
        async for chunk in provider.stream(messages, model=model):
            if chunk.reasoning:
                if show_thinking:
                    if spinning:
                        spinner.stop()
                        spinning = False
                    err_console.print(
                        chunk.reasoning,
                        end="",
                        style="dim",
                        markup=False,
                        highlight=False,
                        soft_wrap=True,
                    )
                    printed_thinking = True
                elif spinning:
                    spinner.update("Thinking...")
            if chunk.text:
                if spinning:
                    spinner.stop()
                    spinning = False
                if printed_thinking and not printed_text:
                    err_console.print("\n")  # gap between thinking and answer
                console.print(chunk.text, end="", markup=False, highlight=False, soft_wrap=True)
                printed_text = True
            if chunk.done is not None:
                final = chunk.done
    finally:
        spinner.stop()
    if printed_text:
        console.print()
    else:
        err_console.print("[dim](the model returned no text)[/dim]")
    if final is None:
        raise BadBackendResponseError("the stream ended without a final reply")
    return final


def describe_usage(usage: Usage) -> str:
    """`9 tokens in, 2 tokens out`, with `?` for counts the backend did not report."""
    tokens_in = "?" if usage.prompt_tokens is None else usage.prompt_tokens
    tokens_out = "?" if usage.completion_tokens is None else usage.completion_tokens
    return f"{tokens_in} tokens in, {tokens_out} tokens out"


def print_reports(reports: list[BackendReport], out: Console) -> None:
    """Print the doctor table, then each backend's warnings and errors."""
    table = Table(title="Backends")
    table.add_column("Provider")
    table.add_column("URL", overflow="fold")
    table.add_column("Status")
    table.add_column("Server")
    table.add_column("Models")
    for r in reports:
        style = STATUS_STYLE[r.status]
        url = escape(r.base_url) + (" [bold yellow]REMOTE[/bold yellow]" if r.remote else "")
        table.add_row(
            r.name,
            url,
            f"[{style}]{r.status.value}[/{style}]",
            escape(r.kind),
            escape(", ".join(r.models)) if r.models else "-",
        )
    out.print(table)
    for r in reports:
        notes = ([r.error] if r.error else []) + (r.warnings if r.status is not Status.down else [])
        for note in notes:
            out.print(f"[yellow]{r.name}:[/yellow] {escape(note)}")
