"""Agent Gateway application dispatcher."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "mujoco":
        from .profiles.mujoco.gateway_cli import main as mujoco_main

        return int(mujoco_main(values[1:]))
    if values and values[0] == "real-rm65":
        from .profiles.real.gateway_cli import main as real_rm65_main

        return int(real_rm65_main(values[1:]))

    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway",
        description="DexTeleop external Agent Gateway application profiles.",
    )
    parser.add_argument(
        "profile",
        nargs="?",
        choices=("mujoco", "real-rm65"),
        help="concrete Gateway deployment profile to run",
    )
    if not values:
        parser.print_help()
        return 0
    parser.parse_args(values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
