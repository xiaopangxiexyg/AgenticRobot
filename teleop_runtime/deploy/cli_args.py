"""Small argument-vector helpers shared by deployment CLI modules."""

from __future__ import annotations

from collections.abc import Sequence


def argument_was_supplied(raw_argv: Sequence[str], option: str) -> bool:
    """Return whether ``option`` was supplied directly or as ``--key=value``."""

    return any(token == option or token.startswith(f"{option}=") for token in raw_argv)


__all__ = ["argument_was_supplied"]
