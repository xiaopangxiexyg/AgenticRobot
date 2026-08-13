"""Visual-review the three URDF-derived absolute TCP workspace candidates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from project_paths import repo_root
from robot_motion.position_control import ResampledMotionReference
from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl
from sim.mujoco.replay.tcp_pose_absolute_workspace_visualization import (
    AbsoluteWorkspaceVisualizer,
    PREVIEW_PROFILE_NAMES,
    build_review_trajectory,
    derive_absolute_workspace,
    load_absolute_workspace_preview_config,
    trajectory_to_world,
)
from sim.mujoco.viewer import passive_viewer_handle


DEFAULT_CONFIG_PATH = (
    repo_root()
    / "sim/sim_real_tests/configs/tcp_pose_absolute_workspace_preview_v1.json"
)


def _sleep_with_viewer(
    duration_sec: float,
    *,
    viewer_handle: Any,
) -> bool:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        if not viewer_handle.is_running():
            return False
        viewer_handle.sync()
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return bool(viewer_handle.is_running())


def _world_to_desk(control: DeskTcpPositionControl, pose_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(control.world_from_desk) @ np.asarray(pose_world, dtype=float)


def _generate_closed_loop_playback(
    *,
    control: DeskTcpPositionControl,
    workspace,
    samples_per_leg: int,
) -> dict[str, object]:
    initial_qpos = {
        side: control.scene.get_arm_qpos(side).copy()
        for side in ("left", "right")
    }
    initial_world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    initial_desk = {
        side: _world_to_desk(control, initial_world[side])
        for side in ("left", "right")
    }
    raw_desk, effective_desk = build_review_trajectory(
        workspace,
        samples_per_leg=int(samples_per_leg),
        start_pose_desk=initial_desk["left"],
    )
    raw_world = trajectory_to_world(workspace, raw_desk)
    effective_world = trajectory_to_world(workspace, effective_desk)
    physics_ticks_exact = (
        float(control.contract.timing.physics_hz)
        / float(control.contract.timing.ik_hz)
    )
    physics_tick_accumulator = 0.0
    actual_world: list[np.ndarray] = []
    actual_qpos: list[dict[str, np.ndarray]] = []
    status_counts: dict[str, int] = {}
    max_position_residual_m = 0.0
    for index, effective_left in enumerate(effective_desk):
        timestamp = float(index) / float(control.contract.timing.ik_hz)
        reference = ResampledMotionReference(
            sequence=index,
            timestamp=timestamp,
            left_source_index=index,
            right_source_index=index,
            left_source_timestamp=timestamp,
            right_source_timestamp=timestamp,
            interpolation_alpha=0.0,
            pose_by_side={
                "left": effective_left,
                "right": initial_desk["right"],
            },
            metadata={
                "source": "absolute_workspace_visual_review",
                "profile": workspace.profile_name,
            },
        )
        result = control.solve_and_dispatch(
            reference,
            gripper_close_by_side={"left": 0.0, "right": 0.0},
        )
        status_counts[result.status] = int(status_counts.get(result.status, 0)) + 1
        physics_tick_accumulator += physics_ticks_exact
        physics_ticks = int(math.floor(physics_tick_accumulator + 1e-12))
        physics_tick_accumulator -= physics_ticks
        for _ in range(physics_ticks):
            control.step_physics()
        qpos = {
            side: control.scene.get_arm_qpos(side).copy()
            for side in ("left", "right")
        }
        actual = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
        )["left"]
        max_position_residual_m = max(
            max_position_residual_m,
            float(np.linalg.norm(actual[:3, 3] - effective_world[index][:3, 3])),
        )
        actual_world.append(actual)
        actual_qpos.append(qpos)
    max_qpos_motion_deg = max(
        float(
            np.max(
                np.abs(
                    np.rad2deg(
                        qpos_by_side["left"] - actual_qpos[0]["left"]
                    )
                )
            )
        )
        for qpos_by_side in actual_qpos
    )
    if max_qpos_motion_deg <= 0.01:
        raise RuntimeError(
            "absolute workspace preview produced no visible left-arm motion; "
            f"max_joint_motion_deg={max_qpos_motion_deg:.6f}"
        )
    return {
        "raw_world": raw_world,
        "effective_world": effective_world,
        "actual_world": tuple(actual_world),
        "actual_qpos": tuple(actual_qpos),
        "status_counts": status_counts,
        "max_position_residual_m": max_position_residual_m,
        "max_qpos_motion_deg": max_qpos_motion_deg,
    }


def run_viewer(args: argparse.Namespace) -> dict[str, object]:
    config = load_absolute_workspace_preview_config(args.config)
    workspace = derive_absolute_workspace(config, args.profile)
    print(
        "[AbsoluteWorkspaceView] review-only candidate:\n"
        + json.dumps(workspace.to_payload(), ensure_ascii=False, indent=2),
        flush=True,
    )
    print(
        "[AbsoluteWorkspaceView] generating actual 60 Hz IK / 120 Hz command / "
        "1000 Hz physics playback",
        flush=True,
    )
    control = DeskTcpPositionControl.build()
    repeats_completed = 0
    try:
        playback = _generate_closed_loop_playback(
            control=control,
            workspace=workspace,
            samples_per_leg=int(args.samples_per_leg),
        )
        visualizer = AbsoluteWorkspaceVisualizer(
            workspace=workspace,
            raw_pose_world=playback["raw_world"],
            effective_pose_world=playback["effective_world"],
            actual_pose_world=playback["actual_world"],
        )
        print(
            "[AbsoluteWorkspaceView] legend: translucent volume/edges=allowed "
            "tool_tcp region; orange=requested TCP; green=projected TCP; "
            "magenta=actual TCP/arm; yellow=projection residual; "
            "RGB=canonical desk XYZ",
            flush=True,
        )
        print(
            "[AbsoluteWorkspaceView] playback="
            + json.dumps(
                {
                    "ik_status_counts": playback["status_counts"],
                    "max_position_residual_m": playback["max_position_residual_m"],
                    "max_qpos_motion_deg": playback["max_qpos_motion_deg"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        with passive_viewer_handle(
            control.scene,
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
        ) as viewer_handle:
            center_desk = np.ones(4, dtype=float)
            center_desk[:3] = workspace.center_xyz_m
            center_world = (workspace.desk_pose_world @ center_desk)[:3]
            viewer_handle.cam.lookat[:] = center_world
            viewer_handle.cam.distance = float(args.camera_distance)
            viewer_handle.cam.azimuth = float(args.camera_azimuth_deg)
            viewer_handle.cam.elevation = float(args.camera_elevation_deg)
            repeat_index = 0
            keep_running = True
            while keep_running and (
                int(args.repeat) == 0 or repeat_index < int(args.repeat)
            ):
                for frame_index in range(visualizer.frame_count):
                    if not viewer_handle.is_running():
                        keep_running = False
                        break
                    control.scene.set_qpos_by_side(
                        playback["actual_qpos"][frame_index]
                    )
                    visualizer.update(viewer_handle, frame_index=frame_index)
                    viewer_handle.sync()
                    if not _sleep_with_viewer(
                        1.0 / float(args.playback_hz),
                        viewer_handle=viewer_handle,
                    ):
                        keep_running = False
                        break
                if not keep_running:
                    break
                repeat_index += 1
                repeats_completed = repeat_index
                keep_running = _sleep_with_viewer(
                    float(args.pause_between_repeats_sec),
                    viewer_handle=viewer_handle,
                )
    finally:
        control.close()
    summary = {
        "profile": workspace.profile_name,
        "frame_count": visualizer.frame_count,
        "repeats_completed": repeats_completed,
        "workspace": workspace.to_payload(),
        "ik_status_counts": playback["status_counts"],
        "max_position_residual_m": playback["max_position_residual_m"],
        "max_qpos_motion_deg": playback["max_qpos_motion_deg"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--profile",
        choices=PREVIEW_PROFILE_NAMES,
        required=True,
    )
    parser.add_argument("--playback-hz", type=float, default=60.0)
    parser.add_argument("--samples-per-leg", type=int, default=90)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--pause-between-repeats-sec", type=float, default=0.4)
    parser.add_argument("--camera-distance", type=float, default=1.75)
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
    for name in (
        "playback_hz",
        "pause_between_repeats_sec",
        "camera_distance",
        "camera_azimuth_deg",
        "camera_elevation_deg",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if float(args.playback_hz) <= 0.0:
        raise ValueError("--playback-hz must be positive")
    if float(args.pause_between_repeats_sec) < 0.0:
        raise ValueError("--pause-between-repeats-sec must be non-negative")
    if float(args.camera_distance) <= 0.0:
        raise ValueError("--camera-distance must be positive")
    if int(args.samples_per_leg) < 2:
        raise ValueError("--samples-per-leg must be >= 2")
    if int(args.repeat) < 0:
        raise ValueError("--repeat must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    try:
        run_viewer(args)
    except KeyboardInterrupt:
        print("\n[AbsoluteWorkspaceView] interrupted; viewer closed cleanly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
