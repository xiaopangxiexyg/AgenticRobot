from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    from .run_collect import main as run_collect_main

    if argv is None:
        return run_collect_main()

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *list(argv)]
        return run_collect_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

