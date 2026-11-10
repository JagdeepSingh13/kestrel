"""Allow `python -m kestrel` as an alias for the `kestrel` command."""

from kestrel.cli.app import app

if __name__ == "__main__":
    app()
