"""`ask_user`: the agent asks you a question and waits for your answer."""

from typing import Annotated

from pydantic import Field

from kestrel.tools.registry import ToolContext, tool


@tool(timeout=3600)
async def ask_user(
    ctx: ToolContext,
    question: Annotated[str, Field(description="One short, clear question for the user.")],
) -> str:
    """Ask the user a question when you are missing information. Returns their answer."""
    if ctx.ask is None:
        return "ERROR: no user is available to answer questions in this run"
    answer = await ctx.ask(question)
    return answer.strip() or "(the user gave no answer)"
