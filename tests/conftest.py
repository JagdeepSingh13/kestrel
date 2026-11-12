"""Shared test setup: no test touches a real backend or the real cache."""

from pathlib import Path

import pytest

from kestrel.cli import agent_run
from kestrel.providers.profiles import Profile


@pytest.fixture(autouse=True)
def no_backend_probe(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Profiles come from a fake probe (nothing known) and live in a temp cache; HOME
    points at a temp folder too, so no test can write into the real home folder, even
    one that passes its own environment. Live tests keep the real probe."""
    monkeypatch.setenv("KESTREL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))  # Path.home() on Windows
    # Per-session memory of what was probed and warned: each test is a new session.
    monkeypatch.setattr(agent_run, "_probed", set[tuple[str, str]]())
    monkeypatch.setattr(agent_run, "_warned_windows", set[tuple[str, int]]())
    if request.node.get_closest_marker("live"):
        return

    async def fake_probe(name: str, base_url: str, model: str, known: Profile | None) -> Profile:
        return Profile(backend=base_url, model=model, probed_at="test")

    monkeypatch.setattr(agent_run, "probe_profile", fake_probe)
