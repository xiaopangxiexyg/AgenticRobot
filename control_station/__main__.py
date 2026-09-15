"""Control Station command dispatcher."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "b601-rm65":
        from .sim_hil_profile.rebot_b601_rm65_application import main as b601_rm65_main

        return int(b601_rm65_main(values[1:]))
    if values and values[0] == "sim-hil":
        from .sim_hil import main as sim_hil_main

        return int(sim_hil_main(values[1:]))
    if values and values[0] == "vive-calibration":
        from .vive_calibration_cli import main as vive_calibration_main

        return int(vive_calibration_main(values[1:]))
    if values and values[0] == "vive-deflection":
        from .vive_deflection_cli import main as vive_deflection_main

        return int(vive_deflection_main(values[1:]))
    if values and values[0] == "refinement-annotation":
        from .refinement_annotation.cli import (
            main as refinement_annotation_main,
        )

        return int(refinement_annotation_main(values[1:]))
    if values and values[0] == "agent-gateway-mujoco":
        from agent_gateway.profiles.mujoco.gateway_cli import (
            main as agent_gateway_mujoco_main,
        )

        return int(agent_gateway_mujoco_main(values[1:]))
    parser = argparse.ArgumentParser(
        prog="python -m control_station",
        description="DexTeleop Control Station application entrypoints.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "b601-rm65",
            "sim-hil",
            "vive-calibration",
            "vive-deflection",
            "refinement-annotation",
            "agent-gateway-mujoco",
        ),
        help="application profile to run",
    )
    if not values:
        parser.print_help()
        return 0
    # Keep argparse's standard exit-0 behavior for ``--help`` and its
    # fail-closed behavior for unknown top-level commands.
    parser.parse_args(values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
