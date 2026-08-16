"""Visualize both arms executing three absolute level TCP orientations."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    identified_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_SIMULATION_CONTRACT_PATH,
    position_15_60_120_timing,
)
from sim.mujoco.replay.level_tcp_orientation_visualization import (
    LevelTcpOrientationVisualizer,
)
from sim.mujoco.scene import MujocoScene
from sim.mujoco.viewer import passive_viewer_handle

from .tcp_orientation_primitives_closed_loop import (
    LevelTcpOrientationClosedLoopConfig,
    run_level_tcp_orientation_primitives_active_closed_loop,
)


def _sleep_with_viewer(duration_sec: float, *, viewer_handle: Any) -> bool:
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    while time.monotonic() < deadline:
        if not viewer_handle.is_running():
            return False
        viewer_handle.sync()
        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    return bool(viewer_handle.is_running())


def _build_playback_scene() -> MujocoScene:
    base_contract = SimulationContract.load(DEFAULT_SIMULATION_CONTRACT_PATH)
    contract = replace(
        base_contract,
        timing=position_15_60_120_timing(
            base_contract.timing,
            parameter_source="level_tcp_orientation_primitive_visualization",
        ),
    )
    return MujocoScene.from_contract(
        contract,
        position_actuation_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_actuation_profile=identified_bimanual_dmtac_gripper_profile(),
    )


def _visualizer_from_records(
    records: Sequence[dict[str, Any]],
) -> LevelTcpOrientationVisualizer:
    if not records:
        raise ValueError("closed-loop playback records must not be empty")
    anchors = {
        side: np.asarray(
            records[0]["target_pose_world_by_side"][side],
            dtype=float,
        )
        for side in ("left", "right")
    }
    return LevelTcpOrientationVisualizer(
        anchor_pose_world_by_side=anchors,
        world_from_desk=np.asarray(records[0]["world_from_desk"], dtype=float),
        actual_pose_world_by_side={
            side: tuple(
                np.asarray(record["actual_pose_world_by_side"][side], dtype=float)
                for record in records
            )
            for side in ("left", "right")
        },
        target_pose_world_by_frame=tuple(
            {
                side: np.asarray(
                    record["target_pose_world_by_side"][side],
                    dtype=float,
                )
                for side in ("left", "right")
            }
            for record in records
        ),
        active_side_by_frame=tuple(record["active_side"] for record in records),
        heading_by_frame=tuple(record["heading"] for record in records),
    )


def run_viewer(args: argparse.Namespace) -> dict[str, Any]:
    config = LevelTcpOrientationClosedLoopConfig(
        samples_per_leg=int(args.samples_per_leg),
        endpoint_hold_frames=int(args.endpoint_hold_frames),
        anchor_hold_frames=int(args.anchor_hold_frames),
    )
    print(
        "[TcpOrientationView] generating actual 60 Hz IK / 120 Hz qcmd / "
        "1000 Hz MuJoCo playback",
        flush=True,
    )
    result = run_level_tcp_orientation_primitives_active_closed_loop(config)
    if not result.passed:
        raise RuntimeError(
            "level TCP orientation closed-loop scenario failed: " f"{result.failure}"
        )
    records = tuple(dict(record) for record in result.records)
    visualizer = _visualizer_from_records(records)
    scene = _build_playback_scene()
    repeats_completed = 0
    try:
        scene.set_qpos_by_side(records[0]["actual_qpos_by_side"])
        with passive_viewer_handle(
            scene,
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
        ) as viewer_handle:
            anchors = np.stack(
                [
                    np.asarray(
                        records[0]["target_pose_world_by_side"][side],
                        dtype=float,
                    )[:3, 3]
                    for side in ("left", "right")
                ],
                axis=0,
            )
            viewer_handle.cam.lookat[:] = np.mean(anchors, axis=0)
            viewer_handle.cam.distance = float(args.camera_distance)
            viewer_handle.cam.azimuth = float(args.camera_azimuth_deg)
            viewer_handle.cam.elevation = float(args.camera_elevation_deg)
            print(
                "[TcpOrientationView] frame=canonical desk; tool +Z=gripper "
                "forward; tool +Y=gripper left; tool +X=down when level",
                flush=True,
            )
            print(
                "[TcpOrientationView] targets: center=0deg; forward-left=desk "
                "+30deg yaw; forward-right=desk -30deg yaw; TCP position fixed",
                flush=True,
            )
            print(
                "[TcpOrientationView] order: left then right; each executes "
                "center_forward/forward_left_30/forward_right_30 and returns",
                flush=True,
            )
            print(
                "[TcpOrientationView] arrows: red=center; green=left30; "
                "blue=right30; magenta/cyan=actual tool +Z; yellow=target",
                flush=True,
            )
            print(
                "[TcpOrientationView] close the viewer or press Ctrl+C to exit",
                flush=True,
            )
            repeat_index = 0
            keep_running = True
            while keep_running and (
                int(args.repeat) == 0 or repeat_index < int(args.repeat)
            ):
                previous_stage: tuple[Any, ...] | None = None
                for frame_index, record in enumerate(records):
                    if not viewer_handle.is_running():
                        keep_running = False
                        break
                    stage = (
                        record["active_side"],
                        record["heading"],
                        record["phase"],
                    )
                    if stage != previous_stage:
                        print(
                            "[TcpOrientationView] stage="
                            + "/".join(
                                "idle" if item is None else str(item) for item in stage
                            ),
                            flush=True,
                        )
                        previous_stage = stage
                    scene.set_qpos_by_side(record["actual_qpos_by_side"])
                    visualizer.update(viewer_handle, frame_index=frame_index)
                    viewer_handle.sync()
                    if not _sleep_with_viewer(
                        1.0 / (config.reference_hz * float(args.speed)),
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
        close_scene = getattr(scene, "close", None)
        if callable(close_scene):
            close_scene()
    summary = {
        "scenario": result.scenario,
        "passed": result.passed,
        "frame_count": len(records),
        "repeats_completed": repeats_completed,
        "metrics": dict(result.metrics),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-leg", type=int, default=120)
    parser.add_argument("--endpoint-hold-frames", type=int, default=30)
    parser.add_argument("--anchor-hold-frames", type=int, default=30)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Number of repeats; 0 repeats until the viewer is closed.",
    )
    parser.add_argument("--pause-between-repeats-sec", type=float, default=0.5)
    parser.add_argument("--camera-distance", type=float, default=1.45)
    parser.add_argument("--camera-azimuth-deg", type=float, default=135.0)
    parser.add_argument("--camera-elevation-deg", type=float, default=-22.0)
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
        "speed",
        "pause_between_repeats_sec",
        "camera_distance",
        "camera_azimuth_deg",
        "camera_elevation_deg",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if float(args.speed) <= 0.0:
        raise ValueError("--speed must be positive")
    if float(args.pause_between_repeats_sec) < 0.0:
        raise ValueError("--pause-between-repeats-sec must be non-negative")
    if float(args.camera_distance) <= 0.0:
        raise ValueError("--camera-distance must be positive")
    for name in (
        "samples_per_leg",
        "endpoint_hold_frames",
        "anchor_hold_frames",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.repeat) < 0:
        raise ValueError("--repeat must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    try:
        run_viewer(args)
    except KeyboardInterrupt:
        print("\n[TcpOrientationView] interrupted; viewer closed cleanly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
