from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import LEFT_ARM_IP, RIGHT_ARM_IP  # noqa: E402
from hardware.rm65.utils.init_arm.rm65_init_pose_common import (  # noqa: E402
    DATA_TYPE,
    POSE_UNITS,
    connect_robot,
    default_pose_path,
    format_values,
    mean_pose,
    pose_spread,
    read_state,
    selected_sides,
    side_ip,
    timestamp,
    use_tool_frame,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record the current RM65 bare-end pose as the teleoperation standard pose. "
            "The output stores Cartesian pose only; it does not store qpos targets."
        )
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--tool-frame", default="empty")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--start-delay-sec", type=float, default=0.0)
    parser.add_argument("--max-linear-spread-m", type=float, default=0.003)
    parser.add_argument("--max-angular-spread-rad", type=float, default=0.015)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-json", default=default_pose_path())
    return parser


def _validate_args(args) -> None:
    if args.samples < 1:
        raise RuntimeError("--samples must be >= 1")
    if args.hz <= 0.0:
        raise RuntimeError("--hz must be positive")
    if args.start_delay_sec < 0.0:
        raise RuntimeError("--start-delay-sec must be >= 0")
    if args.max_linear_spread_m <= 0.0:
        raise RuntimeError("--max-linear-spread-m must be positive")
    if args.max_angular_spread_rad <= 0.0:
        raise RuntimeError("--max-angular-spread-rad must be positive")
    if not str(args.tool_frame).strip():
        raise RuntimeError("--tool-frame must be non-empty; use 'empty' for the bare arm end")
    output_path = Path(args.output_json)
    if output_path.exists() and not args.overwrite:
        raise RuntimeError(f"{output_path} already exists; pass --overwrite to replace it")


def _sample_standard_pose(side: str, args) -> dict[str, object]:
    ip = side_ip(side, args)
    robot = connect_robot(side, ip)
    try:
        active_tool_frame = use_tool_frame(robot, side, args.tool_frame)
        if args.start_delay_sec > 0.0:
            print(f"{side}: sampling starts in {args.start_delay_sec:.1f}s")
            time.sleep(args.start_delay_sec)

        interval = 1.0 / args.hz
        poses: list[list[float]] = []
        for index in range(args.samples):
            state = read_state(robot, side, tool_frame=args.tool_frame)
            poses.append(state["pose"])
            if index == 0 or index + 1 == args.samples:
                print(f"{side}: sample {index + 1}/{args.samples} pose={format_values(state['pose'], precision=6)}")
            if index + 1 < args.samples:
                time.sleep(interval)

        pose = mean_pose(poses)
        linear_spread, angular_spread = pose_spread(poses, pose)
        print(
            f"{side}: standard_pose={format_values(pose, precision=6)} "
            f"spread={linear_spread:.6f}m/{angular_spread:.6f}rad"
        )
        if linear_spread > args.max_linear_spread_m:
            raise RuntimeError(
                f"{side}: sampled pose linear spread {linear_spread:.6f}m > "
                f"--max-linear-spread-m {args.max_linear_spread_m:.6f}m"
            )
        if angular_spread > args.max_angular_spread_rad:
            raise RuntimeError(
                f"{side}: sampled pose angular spread {angular_spread:.6f}rad > "
                f"--max-angular-spread-rad {args.max_angular_spread_rad:.6f}rad"
            )
        return {
            "ip": ip,
            "tool_frame": args.tool_frame,
            "active_tool_frame": active_tool_frame,
            "pose": pose,
            "pose_units": POSE_UNITS,
            "sample_count": args.samples,
            "sample_hz": args.hz,
            "sample_spread": {
                "linear_m": linear_spread,
                "angular_rad": angular_spread,
            },
        }
    finally:
        robot.Arm_Socket_Close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        arms: dict[str, object] = {}
        for side in selected_sides(args.arm):
            arms[side] = _sample_standard_pose(side, args)

        payload = {
            "schema_version": 1,
            "data_type": DATA_TYPE,
            "generated_by": "hardware/rm65/utils/init_arm/rm65_record_standard_pose.py",
            "recorded_at": timestamp(),
            "arm_selection": args.arm,
            "tool_frame": args.tool_frame,
            "notes": "Cartesian standard pose in the selected controller tool frame; no qpos target is stored.",
            "arms": arms,
        }
        write_json(args.output_json, payload)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
