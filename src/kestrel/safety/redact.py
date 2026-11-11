"""Scrub secrets from anything written to disk (spec gaps 6 and 11).

Two sources of secrets:
1. The values of environment variables whose names contain KEY, TOKEN, SECRET or PASSWORD.
2. Well-known secret formats (API keys, bearer tokens), even if they are not in the env.
"""

import re
from collections.abc import Mapping

from kestrel.core.types import JSONValue

REDACTED = "[REDACTED]"
SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
MIN_SECRET_LENGTH = 4
"""Shorter env values (flags like `FEATURE_KEY=1`) would redact ordinary text."""

KNOWN_FORMATS = re.compile(
    "|".join(
        [
            r"\bsk-[A-Za-z0-9_-]{16,}",  # OpenAI, Anthropic and similar API keys
            r"\bgh[pousr]_[A-Za-z0-9]{20,}",  # GitHub tokens
            r"\bgithub_pat_[A-Za-z0-9_]{20,}",
            r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key ids
            r"\bxox[abprs]-[A-Za-z0-9-]{10,}",  # Slack tokens
            r"(?<=Bearer )[A-Za-z0-9._~+/=-]+",  # the token after "Bearer "
        ]
    )
)


class Redactor:
    """Replaces secrets with `[REDACTED]` in text and in JSON values."""

    def __init__(self, env: Mapping[str, str]) -> None:
        """Collect secret values from `env` (usually `os.environ`)."""
        values = {
            value
            for name, value in env.items()
            if any(part in name.upper() for part in SECRET_NAME_PARTS)
            and len(value) >= MIN_SECRET_LENGTH
        }
        # Longest first, so a secret that contains another is replaced whole.
        self._env_values = sorted(values, key=len, reverse=True)

    def text(self, text: str) -> str:
        """Return `text` with every secret replaced."""
        for value in self._env_values:
            text = text.replace(value, REDACTED)
        return KNOWN_FORMATS.sub(REDACTED, text)

    def json(self, value: JSONValue) -> JSONValue:
        """Redact every string inside a JSON value. Keys are kept as they are."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.json(item) for item in value]
        if isinstance(value, dict):
            return {key: self.json(item) for key, item in value.items()}
        return value
