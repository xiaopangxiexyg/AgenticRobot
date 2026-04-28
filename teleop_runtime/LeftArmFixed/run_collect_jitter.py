from __future__ import annotations

import sys

from ..collection.run_collect import main as _collect_main


def main() -> int:
    if "--left-arm-jitter" not in sys.argv:
        sys.argv.insert(1, "--left-arm-jitter")
    return _collect_main()


if __name__ == "__main__":
    raise SystemExit(main())
