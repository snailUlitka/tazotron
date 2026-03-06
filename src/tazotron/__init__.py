"""Tazotron package entrypoint."""

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint with a lazy import to avoid heavy import-time deps."""
    from tazotron.cli.main import main as cli_main  # noqa: PLC0415

    cli_main(argv)

# Public package API.
__all__ = ["main"]
