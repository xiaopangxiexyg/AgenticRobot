from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if resolved_argv and resolved_argv[0] == "umi":
        from .umi.cli import main as umi_main

        return umi_main(resolved_argv[1:])

    from .run_collect import main as run_collect_main

    if argv is None:
        return run_collect_main()

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *resolved_argv]
        return run_collect_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
