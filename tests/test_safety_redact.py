"""Redactor: no secret ever reaches a log line (spec gaps 6 and 11)."""

from kestrel.core.types import JSONValue
from kestrel.safety.redact import REDACTED, Redactor


def test_values_of_secret_named_env_vars_are_redacted() -> None:
    redact = Redactor(
        {
            "OPENAI_API_KEY": "abc123secretvalue",
            "GH_TOKEN": "tok-9999-zzzz",
            "DB_PASSWORD": "hunter2hunter2",
            "my_secret": "lowercase-name-works",
            "HOME": "/Users/jag",
        }
    )
    text = (
        "key abc123secretvalue, tok-9999-zzzz, hunter2hunter2, lowercase-name-works in /Users/jag"
    )
    assert redact.text(text) == f"key {REDACTED}, {REDACTED}, {REDACTED}, {REDACTED} in /Users/jag"


def test_short_env_values_are_ignored() -> None:
    # A KEY=1 flag must not turn every "1" in the log into [REDACTED].
    assert Redactor({"FEATURE_KEY": "1"}).text("step 1 of 1") == "step 1 of 1"


def test_known_secret_formats_are_redacted_without_env() -> None:
    redact = Redactor({})
    assert redact.text("use sk-proj-AbCdEf1234567890XyZ now") == f"use {REDACTED} now"
    assert redact.text("ghp_" + "a" * 36) == REDACTED
    assert redact.text("github_pat_11ABCDEFG0_abcdefghijklmnop") == REDACTED
    assert redact.text("Authorization: Bearer eyJhbGciOi.payload.sig") == (
        f"Authorization: Bearer {REDACTED}"
    )
    assert redact.text("AKIAABCDEFGHIJKLMNOP") == REDACTED
    assert redact.text("xoxb-1234-5678-abcdefgh") == REDACTED


def test_ordinary_text_is_untouched() -> None:
    redact = Redactor({})
    for text in ["sk-short", "ask-me", "the token count is 42", "Bearer"]:
        assert redact.text(text) == text


def test_json_values_are_redacted_recursively_keys_kept() -> None:
    redact = Redactor({"API_KEY": "supersecret99"})
    value: JSONValue = {
        "args": {"cmd": "curl -H 'x: supersecret99'", "n": 3, "ok": True},
        "list": ["supersecret99", None, 1.5],
        "supersecret99": "key names are not redacted",
    }
    assert redact.json(value) == {
        "args": {"cmd": f"curl -H 'x: {REDACTED}'", "n": 3, "ok": True},
        "list": [REDACTED, None, 1.5],
        "supersecret99": "key names are not redacted",
    }
