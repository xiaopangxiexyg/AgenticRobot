"""Visualize Active TCP-pose envelope behavior in the MuJoCo sim-real lab."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from project_paths import repo_root
from sim.contracts import SimulationContract
from sim.mujoco.actuation import (
    identified_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from sim.mujoco.control.desk_tcp_position import (
    DEFAULT_SIMULATION_CONTRACT_PATH,
    position_15_60_120_timing,
)
from sim.mujoco.replay.tcp_pose_envelope_visualization import (
    TcpPoseEnvelopeVisualizer,
)
from sim.mujoco.scene import MujocoScene
from sim.mujoco.viewer import passive_viewer_handle

from .tcp_pose_envelope_closed_loop import (
    DEFAULT_CONFIG_PATH,
    load_tcp_pose_envelope_closed_loop_config,
    run_tcp_pose_envelope_active_closed_loop,
)


RELATIVE_PREVIEW_CONFIG_PATH = (
    repo_root()
    / "sim/sim_real_tests/configs/tcp_pose_relative_workspace_preview_v1.json"
)
RELATIVE_PREVIEW_SCHEMA_VERSION = (
    "sim_real_tests.tcp_pose_relative_workspace_preview.v1"
)


def load_relative_preview_profiles(
    path: str | Path = RELATIVE_PREVIEW_CONFIG_PATH,
) -> Mapping[str, Mapping[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("relative preview config must be an object")
    if payload.get("schema_version") != RELATIVE_PREVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {RELATIVE_PREVIEW_SCHEMA_VERSION!r}"
        )
    if payload.get("role") != "visual_review_only_not_a_runtime_safety_profile":
        raise ValueError("relative preview config must declare its review-only role")
    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) != {
        "relative_tight",
        "relative_standard",
        "relative_loose",
    }:
        raise ValueError("relative preview config must define all three profiles")
    normalized: dict[str, Mapping[str, float]] = {}
    for name, raw in profiles.items():
        if not isinstance(raw, Mapping) or set(raw) != {
            "translation_half_extent_m",
            "max_rotation_deg",
        }:
            raise ValueError(f"profiles.{name} has invalid fields")
        half = float(raw["translation_half_extent_m"])
        angle = float(raw["max_rotation_deg"])
        if not math.isfinite(half) or half <= 0.0:
            raise ValueError(f"profiles.{name}.translation_half_extent_m invalid")
        if not math.isfinite(angle) or not 0.0 < angle < 180.0:
            raise ValueError(f"profiles.{name}.max_rotation_deg invalid")
        normalized[str(name)] = MappingProxyType(
            {
                "translation_half_extent_m": half,
                "max_rotation_deg": angle,
            }
        )
    extents = [
        normalized[name]["translation_half_extent_m"]
        for name in ("relative_tight", "relative_standard", "relative_loose")
    ]
    angles = [
        normalized[name]["max_rotation_deg"]
        for name in ("relative_tight", "relative_standard", "relative_loose")
    ]
    if not (extents[0] < extents[1] < extents[2]):
        raise ValueError("relative translation profiles must be strictly ordered")
    if not (angles[0] < angles[1] < angles[2]):
        raise ValueError("relative rotation profiles must be strictly ordered")
    return MappingProxyType(dict(normalized))


def _closed_loop_config_for_profile(
    config,
    profile: Mapping[str, float] | None,
):
    if profile is None:
        return config
    half = float(profile["translation_half_extent_m"])
    angle = float(profile["max_rotation_deg"])
    return replace(
        config,
        translation_half_extent_m=half,
        max_rotation_deg=angle,
        inside_translation_m=0.35 * half,
        outside_translation_m=1.5 * half,
        inside_rotation_deg=0.25 * angle,
        outside_rotation_deg=1.4 * angle,
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


def _build_playback_scene() -> MujocoScene:
    base_contract = SimulationContract.load(DEFAULT_SIMULATION_CONTRACT_PATH)
    contract = replace(
        base_contract,
        timing=position_15_60_120_timing(
            base_contract.timing,
            parameter_source="tcp_pose_envelope_visualization",
        ),
    )
    return MujocoScene.from_contract(
        contract,
        position_actuation_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_actuation_profile=identified_bimanual_dmtac_gripper_profile(),
    )


def _pose_records(result) -> tuple[dict[str, Any], ...]:
    records = tuple(dict(record) for record in result.records)
    if len(records) < 2:
        raise RuntimeError("closed-loop result has too few frames to visualize")
    required = {
        "raw_left_pose_world",
        "effective_left_pose_world",
        "actual_left_pose_world",
        "actual_qpos_by_side",
    }
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise RuntimeError(
                f"closed-loop frame {index} lacks visualization fields: {missing}"
            )
    return records


def run_viewer(args: argparse.Namespace) -> dict[str, Any]:
    config = load_tcp_pose_envelope_closed_loop_config(args.config)
    relative_profiles = load_relative_preview_profiles(args.relative_profile_config)
    selected_profile = (
        None if args.profile is None else relative_profiles[args.profile]
    )
    config = _closed_loop_config_for_profile(config, selected_profile)
    if selected_profile is not None:
        print(
            "[TCPPoseEnvelopeView] review-only profile="
            + json.dumps(
                {
                    "name": args.profile,
                    "translation_half_extent_m": selected_profile[
                        "translation_half_extent_m"
                    ],
                    "max_rotation_deg": selected_profile["max_rotation_deg"],
                    "relative_axes": "canonical_axes",
                    "anchor": "measured_start_tool_tcp",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(
        "[TCPPoseEnvelopeView] generating closed-loop 15->60->120 Hz trajectory",
        flush=True,
    )
    result = run_tcp_pose_envelope_active_closed_loop(config)
    if not result.passed:
        raise RuntimeError(
            "TCP-pose envelope closed-loop scenario failed: "
            f"{result.failure}"
        )
    records = _pose_records(result)
    paths = {
        "raw": tuple(
            np.asarray(record["raw_left_pose_world"], dtype=float)
            for record in records
        ),
        "effective": tuple(
            np.asarray(record["effective_left_pose_world"], dtype=float)
            for record in records
        ),
        "actual": tuple(
            np.asarray(record["actual_left_pose_world"], dtype=float)
            for record in records
        ),
    }
    translation_frame_world = None
    translation_min_xyz_m = None
    translation_max_xyz_m = None
    if selected_profile is not None:
        half = float(selected_profile["translation_half_extent_m"])
        translation_frame_world = np.eye(4, dtype=float)
        translation_frame_world[:3, 3] = paths["raw"][0][:3, 3]
        translation_min_xyz_m = (-half, -half, -half)
        translation_max_xyz_m = (half, half, half)
    visualizer = TcpPoseEnvelopeVisualizer(
        anchor_pose_world=paths["raw"][0],
        max_direction_angle_rad=math.radians(config.max_rotation_deg),
        trajectory_pose_world_by_kind=paths,
        translation_frame_world=translation_frame_world,
        translation_min_xyz_m=translation_min_xyz_m,
        translation_max_xyz_m=translation_max_xyz_m,
    )
    scene = _build_playback_scene()
    repeats_completed = 0
    try:
        first_qpos = records[0]["actual_qpos_by_side"]
        scene.set_qpos_by_side(first_qpos)
        with passive_viewer_handle(
            scene,
            show_left_ui=not bool(args.hide_left_ui),
            show_right_ui=not bool(args.hide_right_ui),
        ) as viewer_handle:
            anchor_xyz = paths["raw"][0][:3, 3]
            viewer_handle.cam.lookat[:] = anchor_xyz
            viewer_handle.cam.distance = float(args.camera_distance)
            viewer_handle.cam.azimuth = float(args.camera_azimuth_deg)
            viewer_handle.cam.elevation = float(args.camera_elevation_deg)
            print(
                "[TCPPoseEnvelopeView] legend: "
                "orange=raw requested Pose, green=effective clipped Pose, "
                "magenta=actual TCP Pose; RGB spherical-cap wire fans=anchor XYZ "
                "direction limits; all orientation rays share the anchor apex; "
                "translucent blue box="
                "anchor-relative translation limit when a relative profile is selected",
                flush=True,
            )
            print(
                "[TCPPoseEnvelopeView] close the viewer or press Ctrl+C to exit",
                flush=True,
            )
            repeat_index = 0
            keep_running = True
            while keep_running and (
                int(args.repeat) == 0 or repeat_index < int(args.repeat)
            ):
                previous_timestamp = float(records[0]["timestamp"])
                for frame_index, record in enumerate(records):
                    if not viewer_handle.is_running():
                        keep_running = False
                        break
                    scene.set_qpos_by_side(record["actual_qpos_by_side"])
                    visualizer.update(viewer_handle, frame_index=frame_index)
                    viewer_handle.sync()
                    timestamp = float(record["timestamp"])
                    dt = max(0.0, timestamp - previous_timestamp)
                    if frame_index > 0 and not _sleep_with_viewer(
                        dt / float(args.speed),
                        viewer_handle=viewer_handle,
                    ):
                        keep_running = False
                        break
                    previous_timestamp = timestamp
                if not keep_running:
                    break
                repeat_index += 1
                repeats_completed = repeat_index
                if float(args.pause_between_repeats_sec) > 0.0:
                    keep_running = _sleep_with_viewer(
                        float(args.pause_between_repeats_sec),
                        viewer_handle=viewer_handle,
                    )
    finally:
        scene.close()
    summary = {
        "scenario": result.scenario,
        "profile": args.profile or "legacy_acceptance_default",
        "passed": result.passed,
        "frame_count": len(records),
        "repeats_completed": repeats_completed,
        "metrics": dict(result.metrics),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
    )
    parser.add_argument(
        "--relative-profile-config",
        default=str(RELATIVE_PREVIEW_CONFIG_PATH),
    )
    parser.add_argument(
        "--profile",
        choices=("relative_tight", "relative_standard", "relative_loose"),
        default=None,
        help="Review-only relative TCP envelope candidate.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.35,
        help="Playback speed multiplier; 0.35 makes pose changes easier to inspect.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=0,
        help="Number of repeats; 0 repeats until the viewer is closed.",
    )
    parser.add_argument("--pause-between-repeats-sec", type=float, default=0.5)
    parser.add_argument("--camera-distance", type=float, default=0.58)
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
        "camera_distance",
        "pause_between_repeats_sec",
        "camera_azimuth_deg",
        "camera_elevation_deg",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if float(args.speed) <= 0.0:
        raise ValueError("--speed must be positive")
    if float(args.camera_distance) <= 0.0:
        raise ValueError("--camera-distance must be positive")
    if float(args.pause_between_repeats_sec) < 0.0:
        raise ValueError("--pause-between-repeats-sec must be non-negative")
    if int(args.repeat) < 0:
        raise ValueError("--repeat must be non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    try:
        run_viewer(args)
    except KeyboardInterrupt:
        print("\n[TCPPoseEnvelopeView] interrupted; viewer closed cleanly", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
