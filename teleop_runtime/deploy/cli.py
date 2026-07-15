"""Canonical shell wrapper for the supported real policy deployment runtime."""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from .run_policy import main as run_policy_main

    return run_policy_main(None if argv is None else list(argv))


if __name__ == "__main__":
    raise SystemExit(main())
