"""FakeProvider: the scripted provider every later test (agent loop, CLI) relies on."""

import pytest

from kestrel.core.types import BackendUnreachableError, Message, Provider, Role
from kestrel.providers.fake import FakeProvider

HELLO = [Message(role=Role.user, content="hello")]


async def test_returns_scripted_replies_in_order_and_records_calls() -> None:
    fake = FakeProvider([FakeProvider.text("one"), FakeProvider.text("two")])
    provider: Provider = fake
    assert (await provider.chat(HELLO, model="m")).message.content == "one"
    assert (await provider.chat(HELLO, model="m", temperature=0.0)).message.content == "two"
    assert [c.temperature for c in fake.calls] == [0.2, 0.0]
    assert fake.calls[0].messages == HELLO


async def test_stream_replays_reasoning_then_words_then_done() -> None:
    fake = FakeProvider([FakeProvider.text("Hi there friend", reasoning="greet")])
    chunks = [c async for c in fake.stream(HELLO, model="m")]
    assert chunks[0].reasoning == "greet"
    assert "".join(c.text for c in chunks) == "Hi there friend"
    assert [c.text for c in chunks[1:-1]] == ["Hi", " there", " friend"]
    assert chunks[-1].done is not None


async def test_scripted_errors_are_raised() -> None:
    fake = FakeProvider([BackendUnreachableError("http://x")])
    with pytest.raises(BackendUnreachableError):
        await fake.chat(HELLO, model="m")


async def test_running_out_of_script_fails_the_test_loudly() -> None:
    with pytest.raises(AssertionError, match="ran out"):
        await FakeProvider().chat(HELLO, model="m")
