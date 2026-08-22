"""Fit and validate T_quest_from_robot from paired 3D points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hardware.quest_dev.features.visualization.alignment import (
    alignment_acceptable,
    fit_quest_from_robot,
)
from hardware.quest_dev.features.visualization.transforms import matrix_to_column_major


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="JSON containing robot_points and quest_points"
    )
    parser.add_argument("--max-rms-mm", type=float, default=10.0)
    parser.add_argument("--max-point-mm", type=float, default=20.0)
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = fit_quest_from_robot(data.get("robot_points"), data.get("quest_points"))
    accepted = alignment_acceptable(
        result,
        max_rms_error_m=args.max_rms_mm / 1000.0,
        max_point_error_m=args.max_point_mm / 1000.0,
    )
    output = result.as_dict()
    output["quest_from_robot_column_major"] = matrix_to_column_major(
        result.quest_from_robot
    )
    output["accepted"] = accepted
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
