"""Continuously visualize random RM65 grasp poses in MuJoCo.

With no explicit target, the command interleaves one-draw random joint-FK poses
with unfiltered Cartesian/SO(3) probes. Each sample receives exactly one typed
terminal classification. Only reachable, direct-path-safe targets move;
rejected targets remain visible while the robot holds position. An explicit
target is the actual DMTAC tool TCP expressed in that arm's base frame.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

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
from .continuous_random_grasp_pose_motion import (
    ContinuousRandomGraspPoseConfig,
    RandomGraspAttemptOutcome,
    RawCartesianGraspTargetStream,
    classify_random_grasp_attempt,
    outcome_color_rgba,
    sample_random_joint_fk_target,
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
    parser.add_argument(
        "--single-reachable-demo",
        action="store_true",
        help=(
            "Preserve the former one-target demo: select one collision-safe "
            "reachable joint-space target and replay it."
        ),
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
        "--random-position-min-m",
        type=float,
        nargs=3,
        metavar=("X_MIN", "Y_MIN", "Z_MIN"),
        default=(-0.70, -0.80, -0.10),
        help="Inclusive lower bounds for raw random DMTAC TCP positions.",
    )
    parser.add_argument(
        "--random-position-max-m",
        type=float,
        nargs=3,
        metavar=("X_MAX", "Y_MAX", "Z_MAX"),
        default=(0.70, 0.80, 1.20),
        help="Exclusive upper bounds for raw random DMTAC TCP positions.",
    )
    parser.add_argument(
        "--random-sample-count",
        type=int,
        default=0,
        help=(
            "Number of mixed random samples; 0 runs until the GUI closes. "
            "Headless mode requires a positive value."
        ),
    )
    parser.add_argument(
        "--raw-cartesian-probe-interval",
        type=int,
        default=3,
        help=(
            "Emit one unfiltered Cartesian/SO(3) probe every N samples; other "
            "samples are one-draw random joint-FK targets."
        ),
    )
    parser.add_argument(
        "--random-max-joint-offset-deg",
        type=float,
        default=45.0,
        help="Per-joint offset bound for non-probe random FK targets.",
    )
    parser.add_argument("--rejected-target-hold-sec", type=float, default=1.25)
    parser.add_argument("--between-samples-sec", type=float, default=0.25)
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
        help="Single-target GUI repetitions; 0 repeats until the viewer is closed.",
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
        "random_max_joint_offset_deg",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    nonnegative = (
        "terminal_hold_sec",
        "pause_between_repeats_sec",
        "rejected_target_hold_sec",
        "between_samples_sec",
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
    if int(args.random_sample_count) < 0:
        raise ValueError("--random-sample-count must be non-negative")
    if int(args.raw_cartesian_probe_interval) < 1:
        raise ValueError("--raw-cartesian-probe-interval must be positive")
    minimum = np.asarray(args.random_position_min_m, dtype=float).reshape(-1)
    maximum = np.asarray(args.random_position_max_m, dtype=float).reshape(-1)
    if (
        minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(minimum >= maximum)
    ):
        raise ValueError("random position bounds must be finite ordered 3-vectors")
    if bool(args.single_reachable_demo) and has_xyz:
        raise ValueError("--single-reachable-demo cannot accompany an explicit target")
    if _uses_continuous_random_mode(args):
        if int(args.repeat) != 0:
            raise ValueError("--repeat applies only to single-target playback")
        if bool(args.headless) and int(args.random_sample_count) == 0:
            raise ValueError(
                "headless continuous mode requires --random-sample-count > 0"
            )


def _uses_continuous_random_mode(args: argparse.Namespace) -> bool:
    return args.target_xyz_m is None and not bool(args.single_reachable_demo)


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


class _RandomTargetStatusVisualizer:
    """Display-only colored marker for the current random target outcome."""

    def __init__(self, *, radius_m: float = 0.018) -> None:
        self.radius_m = float(radius_m)
        self._geom_index: int | None = None

    def update(
        self,
        viewer_handle: Any,
        *,
        target_xyz: np.ndarray,
        outcome: RandomGraspAttemptOutcome,
    ) -> None:
        import mujoco

        with viewer_handle.lock():
            user_scene = viewer_handle.user_scn
            if self._geom_index is None:
                self._geom_index = int(user_scene.ngeom)
            if self._geom_index >= int(user_scene.maxgeom):
                raise RuntimeError("MuJoCo viewer has no room for random target status")
            geom = user_scene.geoms[self._geom_index]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, self.radius_m, dtype=float),
                np.asarray(target_xyz, dtype=float),
                np.eye(3, dtype=float).reshape(-1),
                outcome_color_rgba(outcome),
            )
            geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
            geom.label = f"random_target:{outcome.terminal_classification}"
            user_scene.ngeom = max(int(user_scene.ngeom), self._geom_index + 1)


def _print_random_attempt(
    target: GraspMotionTarget,
    outcome: RandomGraspAttemptOutcome,
) -> None:
    payload = outcome.to_payload()
    payload["sample_kind"] = str(target.metadata.get("sample_kind", "unknown"))
    payload["target_xyz_m"] = target.target_pose_base_from_dmtac_tcp[:3, 3].tolist()
    print(
        "[ContinuousRandomGrasp] "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _replay_random_attempt(
    control: DeskTcpPositionControl,
    playback: GraspMotionPlayback,
    outcome: RandomGraspAttemptOutcome,
    args: argparse.Namespace,
    *,
    viewer_handle: Any,
    target_visualizer: TcpTargetVisualizer,
    status_visualizer: _RandomTargetStatusVisualizer,
) -> bool:
    target_xyz = playback.target_pose_world_by_side[playback.side][:3, 3]
    inactive_side = "right" if playback.side == "left" else "left"
    inactive_xyz = playback.target_pose_world_by_side[inactive_side][:3, 3]
    viewer_handle.cam.lookat[:] = 0.5 * (target_xyz + inactive_xyz)
    viewer_handle.cam.distance = max(
        float(args.camera_distance),
        1.7 * float(np.linalg.norm(target_xyz - inactive_xyz)),
    )
    target_visualizer.update(viewer_handle, playback.target_pose_world_by_side)
    status_visualizer.update(
        viewer_handle,
        target_xyz=target_xyz,
        outcome=outcome,
    )
    viewer_handle.sync()
    if playback.frame_count == 0:
        return _sleep_with_viewer(
            float(args.rejected_target_hold_sec), viewer_handle=viewer_handle
        )
    for qpos_by_side in playback.actual_qpos_by_frame:
        if not viewer_handle.is_running():
            return False
        with viewer_handle.lock():
            control.scene.set_qpos_by_side(qpos_by_side)
        target_visualizer.update(viewer_handle, playback.target_pose_world_by_side)
        status_visualizer.update(
            viewer_handle,
            target_xyz=target_xyz,
            outcome=outcome,
        )
        viewer_handle.sync()
        if not _sleep_with_viewer(
            1.0 / float(args.playback_hz), viewer_handle=viewer_handle
        ):
            return False
    return _sleep_with_viewer(
        float(args.between_samples_sec), viewer_handle=viewer_handle
    )


def _run_continuous_random(
    control: DeskTcpPositionControl,
    args: argparse.Namespace,
) -> dict[str, object]:
    config = ContinuousRandomGraspPoseConfig(
        seed=int(args.seed),
        position_min_m=tuple(float(v) for v in args.random_position_min_m),
        position_max_m=tuple(float(v) for v in args.random_position_max_m),
    )
    stream = RawCartesianGraspTargetStream(side=str(args.side), config=config)
    maximum_samples = int(args.random_sample_count)
    counts: dict[str, int] = {}
    sample_kind_counts: dict[str, int] = {}
    attempted_samples = 0
    last_attempt: dict[str, object] | None = None
    viewer_closed = False

    def run_one(viewer_handle: Any | None = None) -> bool:
        nonlocal attempted_samples, last_attempt
        sample_index = attempted_samples
        if sample_index % int(args.raw_cartesian_probe_interval) == 0:
            target = stream.target_for_sample_index(sample_index)
        else:
            target = sample_random_joint_fk_target(
                control,
                side=str(args.side),
                seed=int(args.seed),
                sample_index=sample_index,
                maximum_joint_offset_deg=float(args.random_max_joint_offset_deg),
            )
        execute_kwargs = {
            "linear_speed_m_s": float(args.linear_speed_m_s),
            "angular_speed_deg_s": float(args.angular_speed_deg_s),
            "minimum_duration_sec": float(args.minimum_duration_sec),
            "terminal_hold_sec": float(args.terminal_hold_sec),
        }
        if viewer_handle is None:
            playback = execute_grasp_motion(control, target, **execute_kwargs)
        else:
            with viewer_handle.lock():
                playback = execute_grasp_motion(control, target, **execute_kwargs)
        outcome = classify_random_grasp_attempt(playback)
        _print_random_attempt(target, outcome)
        attempted_samples += 1
        counts[outcome.terminal_classification] = (
            counts.get(outcome.terminal_classification, 0) + 1
        )
        sample_kind = str(target.metadata.get("sample_kind", "unknown"))
        sample_kind_counts[sample_kind] = sample_kind_counts.get(sample_kind, 0) + 1
        last_attempt = outcome.to_payload()
        if viewer_handle is None:
            return True
        return _replay_random_attempt(
            control,
            playback,
            outcome,
            args,
            viewer_handle=viewer_handle,
            target_visualizer=target_visualizer,
            status_visualizer=status_visualizer,
        )

    if bool(args.headless):
        while attempted_samples < maximum_samples:
            run_one()
    else:
        target_visualizer = TcpTargetVisualizer()
        status_visualizer = _RandomTargetStatusVisualizer()
        with passive_viewer_handle(
            control.scene,
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
        ) as viewer_handle:
            viewer_handle.cam.distance = float(args.camera_distance)
            viewer_handle.cam.azimuth = float(args.camera_azimuth_deg)
            viewer_handle.cam.elevation = float(args.camera_elevation_deg)
            print(
                "[ContinuousRandomGrasp] green=completed, red=proven unreachable, "
                "orange=collision blocked, yellow=search exhausted, purple=error",
                flush=True,
            )
            while viewer_handle.is_running() and (
                maximum_samples == 0 or attempted_samples < maximum_samples
            ):
                if not run_one(viewer_handle):
                    viewer_closed = True
                    break
            viewer_closed = viewer_closed or not viewer_handle.is_running()

    system_failure_count = sum(
        counts.get(name, 0)
        for name in ("invalid_target", "planner_error", "execution_failed")
    )
    summary = {
        "mode": "continuous_mixed_random_grasp",
        "passed": system_failure_count == 0,
        "headless": bool(args.headless),
        "viewer_closed": viewer_closed,
        "seed": config.seed,
        "side": str(args.side),
        "position_min_m": list(config.position_min_m),
        "position_max_m": list(config.position_max_m),
        "attempted_samples": attempted_samples,
        "raw_cartesian_probe_interval": int(args.raw_cartesian_probe_interval),
        "sample_kind_counts": sample_kind_counts,
        "terminal_classification_counts": counts,
        "last_attempt": last_attempt,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


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
        if _uses_continuous_random_mode(args):
            return _run_continuous_random(control, args)
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
    if summary.get("mode") == "continuous_mixed_random_grasp":
        return 0 if bool(summary.get("passed")) else 2
    return 0 if summary["motion_status"] == MOTION_STATUS_COMPLETED else 2


if __name__ == "__main__":
    raise SystemExit(main())
