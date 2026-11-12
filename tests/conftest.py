"""Shared test setup: no test touches a real backend or the real cache."""

from pathlib import Path

import pytest

from kestrel.cli import agent_run
from kestrel.providers.profiles import Profile


@pytest.fixture(autouse=True)
def no_backend_probe(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Profiles come from a fake probe (nothing known) and live in a temp cache.
    Live tests keep the real probe."""
    monkeypatch.setenv("KESTREL_CACHE_DIR", str(tmp_path / "cache"))
    if request.node.get_closest_marker("live"):
        return

    async def fake_probe(name: str, base_url: str, model: str, known: Profile | None) -> Profile:
        return Profile(backend=base_url, model=model, probed_at="test")

    monkeypatch.setattr(agent_run, "probe_profile", fake_probe)
