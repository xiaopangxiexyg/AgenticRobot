"""Visualize a complete MuJoCo motion to one arbitrary RM65 grasp pose.

With no explicit target, the command deterministically samples a nontrivial,
collision-safe reachable pose from the selected arm's joint space.  An
explicit target is the actual DMTAC tool TCP expressed in that arm's base
frame.  Unreachable and collision-blocked targets produce a typed no-motion
receipt; they are never forwarded to the motion executor.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from typing import Any

from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl
from sim.mujoco.replay.tcp_target_visualization import TcpTargetVisualizer
from sim.mujoco.viewer import passive_viewer_handle

from .arbitrary_grasp_pose_motion import (
    MOTION_STATUS_COMPLETED,
    GraspMotionPlayback,
    GraspMotionTarget,
    execute_grasp_motion,
    pose_from_xyz_quat_xyzw,
    pose_from_xyz_rpy_deg,
    select_random_reachable_demo_target,
)


def _sleep_with_viewer(duration_sec: float, *, viewer_handle: Any) -> bool:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        if not viewer_handle.is_running():
            return False
        viewer_handle.sync()
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return bool(viewer_handle.is_running())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--target-xyz-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Explicit DMTAC tool-TCP position in the selected arm base frame. "
            "It must be accompanied by exactly one orientation option."
        ),
    )
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument(
        "--target-quat-xyzw",
        type=float,
        nargs=4,
        metavar=("QX", "QY", "QZ", "QW"),
        help="Explicit DMTAC tool-TCP quaternion in the selected arm base frame.",
    )
    orientation.add_argument(
        "--target-rpy-deg",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Explicit DMTAC tool-TCP XYZ Euler angles in degrees.",
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--demo-max-trials", type=int, default=256)
    parser.add_argument("--demo-max-joint-offset-deg", type=float, default=55.0)
    parser.add_argument("--demo-min-joint-motion-deg", type=float, default=25.0)
    parser.add_argument("--linear-speed-m-s", type=float, default=0.12)
    parser.add_argument("--angular-speed-deg-s", type=float, default=45.0)
    parser.add_argument("--minimum-duration-sec", type=float, default=2.0)
    parser.add_argument("--terminal-hold-sec", type=float, default=1.5)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Generate and execute the same physical playback without opening a GUI.",
    )
    parser.add_argument("--playback-hz", type=float, default=60.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="GUI playback repetitions; 0 repeats until the viewer is closed.",
    )
    parser.add_argument("--pause-between-repeats-sec", type=float, default=0.5)
    parser.add_argument("--camera-distance", type=float, default=1.55)
    parser.add_argument("--camera-azimuth-deg", type=float, default=135.0)
    parser.add_argument("--camera-elevation-deg", type=float, default=-24.0)
    parser.add_argument(
        "--hide-left-ui",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hide-right-ui",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    has_xyz = args.target_xyz_m is not None
    has_orientation = (
        args.target_quat_xyzw is not None or args.target_rpy_deg is not None
    )
    if has_xyz != has_orientation:
        raise ValueError(
            "explicit target requires --target-xyz-m and exactly one of "
            "--target-quat-xyzw/--target-rpy-deg"
        )
    positive = (
        "demo_max_trials",
        "demo_max_joint_offset_deg",
        "demo_min_joint_motion_deg",
        "linear_speed_m_s",
        "angular_speed_deg_s",
        "minimum_duration_sec",
        "playback_hz",
        "camera_distance",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    nonnegative = (
        "terminal_hold_sec",
        "pause_between_repeats_sec",
    )
    for name in nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    for name in ("camera_azimuth_deg", "camera_elevation_deg"):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if int(args.demo_max_trials) < 1:
        raise ValueError("--demo-max-trials must be a positive integer")
    if int(args.repeat) < 0:
        raise ValueError("--repeat must be non-negative")


def _target_from_args(
    args: argparse.Namespace,
    *,
    control: DeskTcpPositionControl,
) -> GraspMotionTarget:
    if args.target_xyz_m is None:
        return select_random_reachable_demo_target(
            control,
            side=str(args.side),
            seed=int(args.seed),
            maximum_trials=int(args.demo_max_trials),
            maximum_joint_offset_deg=float(args.demo_max_joint_offset_deg),
            minimum_joint_motion_deg=float(args.demo_min_joint_motion_deg),
        )
    if args.target_quat_xyzw is not None:
        pose = pose_from_xyz_quat_xyzw(
            args.target_xyz_m, args.target_quat_xyzw
        )
        orientation_format = "quaternion_xyzw"
    else:
        pose = pose_from_xyz_rpy_deg(args.target_xyz_m, args.target_rpy_deg)
        orientation_format = "rpy_xyz_degrees"
    return GraspMotionTarget(
        side=str(args.side),
        target_pose_base_from_dmtac_tcp=pose,
        source="explicit_cli_dmtac_tcp_in_selected_arm_base",
        metadata={"orientation_format": orientation_format},
    )


def _playback_in_viewer(
    control: DeskTcpPositionControl,
    playback: GraspMotionPlayback,
    args: argparse.Namespace,
) -> int:
    if playback.frame_count < 2:
        return 0
    visualizer = TcpTargetVisualizer()
    repeats_completed = 0
    target_xyz = playback.target_pose_world_by_side[playback.side][:3, 3]
    with passive_viewer_handle(
        control.scene,
        show_left_ui=not bool(args.hide_left_ui),
        show_right_ui=not bool(args.hide_right_ui),
    ) as viewer_handle:
        viewer_handle.cam.lookat[:] = target_xyz
        viewer_handle.cam.distance = float(args.camera_distance)
        viewer_handle.cam.azimuth = float(args.camera_azimuth_deg)
        viewer_handle.cam.elevation = float(args.camera_elevation_deg)
        print(
            "[ArbitraryGraspMotionView] legend: RGB axes=final DMTAC TCP target; "
            "robot geometry=actual MuJoCo qpos captured from 60 Hz IK / 120 Hz "
            "command / 1000 Hz physics execution",
            flush=True,
        )
        print(
            "[ArbitraryGraspMotionView] close the viewer or press Ctrl+C to exit",
            flush=True,
        )
        keep_running = True
        while keep_running and (
            int(args.repeat) == 0 or repeats_completed < int(args.repeat)
        ):
            for qpos_by_side in playback.actual_qpos_by_frame:
                if not viewer_handle.is_running():
                    keep_running = False
                    break
                control.scene.set_qpos_by_side(qpos_by_side)
                visualizer.update(
                    viewer_handle,
                    playback.target_pose_world_by_side,
                )
                viewer_handle.sync()
                if not _sleep_with_viewer(
                    1.0 / float(args.playback_hz), viewer_handle=viewer_handle
                ):
                    keep_running = False
                    break
            if not keep_running:
                break
            repeats_completed += 1
            keep_running = _sleep_with_viewer(
                float(args.pause_between_repeats_sec), viewer_handle=viewer_handle
            )
    return repeats_completed


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    control = DeskTcpPositionControl.build()
    repeats_completed = 0
    try:
        target = _target_from_args(args, control=control)
        print(
            "[ArbitraryGraspMotionView] target semantics="
            "selected_arm_base_from_actual_dmtac_tool_tcp source="
            f"{target.source}",
            flush=True,
        )
        playback = execute_grasp_motion(
            control,
            target,
            linear_speed_m_s=float(args.linear_speed_m_s),
            angular_speed_deg_s=float(args.angular_speed_deg_s),
            minimum_duration_sec=float(args.minimum_duration_sec),
            terminal_hold_sec=float(args.terminal_hold_sec),
        )
        if not bool(args.headless) and playback.frame_count > 0:
            repeats_completed = _playback_in_viewer(control, playback, args)
        summary = playback.to_payload()
        summary["headless"] = bool(args.headless)
        summary["repeats_completed"] = repeats_completed
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        control.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0 if summary["motion_status"] == MOTION_STATUS_COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
