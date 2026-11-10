"""Smoke tests: the package imports and the CLI starts."""

import re

from typer.testing import CliRunner

import kestrel
from kestrel.__about__ import __app_name__, __version__
from kestrel.cli.app import app

runner = CliRunner()


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", kestrel.__version__)
    assert kestrel.__version__ == __version__


def test_cli_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"{__app_name__} {__version__}"


def test_cli_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout
