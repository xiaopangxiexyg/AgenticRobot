"""Read one or more RM65 feedback samples without issuing motion commands.

This probe deliberately constructs the same ``RealManFeedbackReader`` used by
the Quest visualization runtime.  The reader creates ``RM65ArmBackend`` with
``observation_only=True`` and only calls the SDK state-read path.  It is useful
for separating a RealMan SDK/IP/URDF/FK problem from Quest networking or frame
binding problems.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hardware.quest_dev.features.visualization.realman_feedback import (
    RealManArmSpec,
    RealManFeedbackReader,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-ip", required=True, help="Reviewed RM65 controller IP")
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        required=True,
        help="RM65 side used by the existing FK/tool profile",
    )
    parser.add_argument(
        "--robot-id",
        default=None,
        help="Published robot identity (default: rm65_<side>)",
    )
    parser.add_argument(
        "--arm-urdf",
        required=True,
        type=Path,
        help="RM65 URDF used for measured-qpos forward kinematics",
    )
    parser.add_argument(
        "--tool-profile",
        default=None,
        help="Optional reviewed RM65 tool profile name",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of state samples to read (default: 3)",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=0.2,
        help="Delay between reads in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--max-age-s",
        type=float,
        default=1.0,
        help="Freshness limit for printed samples (default: 1.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.interval_s < 0.0:
        raise SystemExit("--interval-s must be non-negative")
    if not args.arm_urdf.is_file():
        raise SystemExit(f"RM65 URDF does not exist: {args.arm_urdf}")

    robot_id = args.robot_id or f"rm65_{args.side}"
    spec = RealManArmSpec(
        robot_id=robot_id,
        side=args.side,
        arm_ip=args.arm_ip,
        arm_urdf_path=str(args.arm_urdf),
        tool_profile=args.tool_profile,
    )
    reader = RealManFeedbackReader(
        (spec,),
        max_age_s=args.max_age_s,
    )
    all_fresh = True
    try:
        reader.start()
        for sample_index in range(args.samples):
            message = reader.poll()
            record = message["robots"][robot_id]
            all_fresh = all_fresh and record.get("fresh") is True
            print(
                json.dumps(
                    {
                        "sample": sample_index + 1,
                        "observation_only": record.get("mode") == "observation_only",
                        "record": record,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                flush=True,
            )
            if sample_index + 1 < args.samples:
                time.sleep(args.interval_s)
    finally:
        reader.stop()
    return 0 if all_fresh else 2


if __name__ == "__main__":
    raise SystemExit(main())

