"""Exercise the three absolute level TCP orientations in MuJoCo.

Each primitive preserves the measured ``tool_tcp`` position while moving to
one absolute horizontal heading in canonical desk axes.  Every case starts at
and returns to the same measured simulation anchor.  The left arm completes
center/left-30/right-30 before the right arm begins.  Targets use the existing
60 Hz coupled IK -> 120 Hz qcmd -> 1000 Hz physics chain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.position_control import ResampledMotionReference
from robot_motion.primitives import (
    LEVEL_TCP_ORIENTATION_PRIMITIVES,
    LevelTCPHeading,
)
from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl

from ..core import SimRealScenarioResult

_SIDES = ("left", "right")
_DESK_REFERENCE_FRAME_SPEC = CoordinateFrameSpec(
    frame_id="desk",
    frame_version="1",
    semantic_parameters={
        "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
    },
)


@dataclass(frozen=True)
class LevelTcpOrientationClosedLoopConfig:
    reference_hz: float = 60.0
    samples_per_leg: int = 120
    endpoint_hold_frames: int = 30
    anchor_hold_frames: int = 30
    maximum_target_translation_drift_m: float = 1e-10
    maximum_endpoint_position_error_m: float = 0.008
    maximum_endpoint_orientation_error_deg: float = 3.0
    maximum_endpoint_level_error_deg: float = 3.0
    maximum_final_anchor_position_error_m: float = 0.005
    maximum_final_anchor_orientation_error_deg: float = 3.0

    def __post_init__(self) -> None:
        reference_hz = float(self.reference_hz)
        if not math.isfinite(reference_hz) or reference_hz <= 0.0:
            raise ValueError("reference_hz must be finite and positive")
        object.__setattr__(self, "reference_hz", reference_hz)
        for name in (
            "samples_per_leg",
            "endpoint_hold_frames",
            "anchor_hold_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in (
            "maximum_target_translation_drift_m",
            "maximum_endpoint_position_error_m",
            "maximum_endpoint_orientation_error_deg",
            "maximum_endpoint_level_error_deg",
            "maximum_final_anchor_position_error_m",
            "maximum_final_anchor_orientation_error_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


def _rotation_error_deg(reference: np.ndarray, actual: np.ndarray) -> float:
    relative = (
        np.asarray(actual[:3, :3], dtype=float)
        @ np.asarray(reference[:3, :3], dtype=float).T
    )
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))


def _interpolate_pose(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    blend = float(alpha)
    rotations = Rotation.from_matrix(np.stack((start[:3, :3], end[:3, :3]), axis=0))
    pose = np.asarray(start, dtype=float).copy()
    pose[:3, 3] = (1.0 - blend) * np.asarray(
        start[:3, 3], dtype=float
    ) + blend * np.asarray(end[:3, 3], dtype=float)
    pose[:3, :3] = Slerp((0.0, 1.0), rotations)((blend,)).as_matrix()[0]
    return pose


def build_level_tcp_orientation_world_trajectory(
    *,
    anchor_pose_world_by_side: Mapping[str, np.ndarray],
    world_from_desk: np.ndarray,
    config: LevelTcpOrientationClosedLoopConfig | None = None,
    reference_frame_spec: CoordinateFrameSpec = _DESK_REFERENCE_FRAME_SPEC,
) -> tuple[dict[str, Any], ...]:
    """Build smooth world targets from absolute level desk orientations."""

    cfg = config or LevelTcpOrientationClosedLoopConfig()
    if set(anchor_pose_world_by_side) != set(_SIDES):
        raise ValueError("anchor_pose_world_by_side must contain left and right")
    anchors = {
        side: np.asarray(anchor_pose_world_by_side[side], dtype=float).copy()
        for side in _SIDES
    }
    world_from_desk = np.asarray(world_from_desk, dtype=float)
    if world_from_desk.shape != (4, 4) or not np.all(np.isfinite(world_from_desk)):
        raise ValueError("world_from_desk must be a finite 4x4 pose")
    desk_from_world = np.linalg.inv(world_from_desk)
    anchors_desk = {side: desk_from_world @ anchors[side] for side in _SIDES}
    frames: list[dict[str, Any]] = [
        {
            "active_side": None,
            "heading": None,
            "heading_yaw_deg": None,
            "phase": "initial_anchor",
            "primitive_id": None,
            "primitive_hash": None,
            "target_pose_world_by_side": {
                side: anchors[side].copy() for side in _SIDES
            },
        }
    ]
    for side in _SIDES:
        for primitive in LEVEL_TCP_ORIENTATION_PRIMITIVES:
            target_desk = primitive.apply(
                anchors_desk[side],
                reference_frame_spec=reference_frame_spec,
            )
            endpoint_world = world_from_desk @ np.asarray(
                target_desk.target_pose,
                dtype=float,
            )
            for phase, start, end, count in (
                ("outbound", anchors[side], endpoint_world, cfg.samples_per_leg),
                ("return", endpoint_world, anchors[side], cfg.samples_per_leg),
            ):
                for sample_index in range(1, count + 1):
                    active_pose = _interpolate_pose(
                        start,
                        end,
                        float(sample_index) / float(count),
                    )
                    frames.append(
                        {
                            "active_side": side,
                            "heading": primitive.heading.value,
                            "heading_yaw_deg": primitive.heading_yaw_deg,
                            "phase": phase,
                            "primitive_id": primitive.primitive_id,
                            "primitive_hash": primitive.semantic_content_hash,
                            "target_pose_world_by_side": {
                                candidate_side: (
                                    active_pose.copy()
                                    if candidate_side == side
                                    else anchors[candidate_side].copy()
                                )
                                for candidate_side in _SIDES
                            },
                        }
                    )
                hold_phase = "target_hold" if phase == "outbound" else "anchor_hold"
                hold_count = (
                    cfg.endpoint_hold_frames
                    if phase == "outbound"
                    else cfg.anchor_hold_frames
                )
                for _ in range(hold_count):
                    frames.append(
                        {
                            "active_side": side,
                            "heading": primitive.heading.value,
                            "heading_yaw_deg": primitive.heading_yaw_deg,
                            "phase": hold_phase,
                            "primitive_id": primitive.primitive_id,
                            "primitive_hash": primitive.semantic_content_hash,
                            "target_pose_world_by_side": {
                                candidate_side: (
                                    np.asarray(end, dtype=float).copy()
                                    if candidate_side == side
                                    else anchors[candidate_side].copy()
                                )
                                for candidate_side in _SIDES
                            },
                        }
                    )
    return tuple(frames)


def run_level_tcp_orientation_primitives_active_closed_loop(
    config: LevelTcpOrientationClosedLoopConfig | None = None,
) -> SimRealScenarioResult:
    """Run all three orientations on both arms and return physical records."""

    cfg = config or LevelTcpOrientationClosedLoopConfig()
    control: DeskTcpPositionControl | None = None
    records: list[dict[str, Any]] = []
    try:
        control = DeskTcpPositionControl.build()
        if not math.isclose(
            cfg.reference_hz,
            float(control.contract.timing.ik_hz),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "reference_hz must match the existing MuJoCo IK rate: "
                f"{control.contract.timing.ik_hz}"
            )
        initial_qpos = {
            side: control.scene.get_arm_qpos(side).copy() for side in _SIDES
        }
        anchors_world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=initial_qpos["left"],
            right_qpos=initial_qpos["right"],
        )
        frames = build_level_tcp_orientation_world_trajectory(
            anchor_pose_world_by_side=anchors_world,
            world_from_desk=control.world_from_desk,
            config=cfg,
        )
        desk_from_world = np.linalg.inv(control.world_from_desk)
        physics_ticks_exact = (
            float(control.contract.timing.physics_hz) / cfg.reference_hz
        )
        physics_tick_accumulator = 0.0
        previous_qcmd = control.rig.accepted_qcmd_by_side()
        status_counts: dict[str, int] = {}
        max_qcmd_step_deg = 0.0
        nonfinite_qcmd_count = 0
        max_target_translation_drift_m = 0.0
        max_inactive_position_drift_m = 0.0
        endpoint_position_error: dict[tuple[str, str], list[float]] = {
            (side, heading.value): [] for side in _SIDES for heading in LevelTCPHeading
        }
        endpoint_orientation_error: dict[tuple[str, str], list[float]] = {
            key: [] for key in endpoint_position_error
        }
        endpoint_level_error: dict[tuple[str, str], list[float]] = {
            key: [] for key in endpoint_position_error
        }

        for sequence, frame in enumerate(frames):
            timestamp = float(sequence) / cfg.reference_hz
            target_world = frame["target_pose_world_by_side"]
            for side in _SIDES:
                max_target_translation_drift_m = max(
                    max_target_translation_drift_m,
                    float(
                        np.linalg.norm(
                            target_world[side][:3, 3] - anchors_world[side][:3, 3]
                        )
                    ),
                )
            reference = ResampledMotionReference(
                sequence=sequence,
                timestamp=timestamp,
                left_source_index=sequence,
                right_source_index=sequence,
                left_source_timestamp=timestamp,
                right_source_timestamp=timestamp,
                interpolation_alpha=0.0,
                pose_by_side={
                    side: desk_from_world @ target_world[side] for side in _SIDES
                },
                metadata={
                    "source": "level_tcp_orientation_primitive_visual_review",
                    "active_side": frame["active_side"],
                    "heading": frame["heading"],
                    "heading_yaw_deg": frame["heading_yaw_deg"],
                    "phase": frame["phase"],
                    "primitive_id": frame["primitive_id"],
                    "primitive_hash": frame["primitive_hash"],
                },
            )
            result = control.solve_and_dispatch(
                reference,
                gripper_close_by_side={"left": 0.0, "right": 0.0},
            )
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            physics_tick_accumulator += physics_ticks_exact
            physics_ticks = math.floor(physics_tick_accumulator + 1e-12)
            physics_tick_accumulator -= physics_ticks
            for _ in range(physics_ticks):
                control.step_physics()
                qcmd = control.rig.accepted_qcmd_by_side()
                for side in _SIDES:
                    if not np.all(np.isfinite(qcmd[side])):
                        nonfinite_qcmd_count += 1
                    max_qcmd_step_deg = max(
                        max_qcmd_step_deg,
                        float(
                            np.max(np.abs(np.rad2deg(qcmd[side] - previous_qcmd[side])))
                        ),
                    )
                previous_qcmd = qcmd

            qpos = {side: control.scene.get_arm_qpos(side).copy() for side in _SIDES}
            actual_world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=qpos["left"],
                right_qpos=qpos["right"],
            )
            active_side = frame["active_side"]
            heading = frame["heading"]
            phase = frame["phase"]
            for side in _SIDES:
                if active_side is not None and side != active_side:
                    max_inactive_position_drift_m = max(
                        max_inactive_position_drift_m,
                        float(
                            np.linalg.norm(
                                actual_world[side][:3, 3] - anchors_world[side][:3, 3]
                            )
                        ),
                    )
            if active_side in _SIDES and heading is not None and phase == "target_hold":
                key = (str(active_side), str(heading))
                endpoint_position_error[key].append(
                    float(
                        np.linalg.norm(
                            actual_world[active_side][:3, 3]
                            - target_world[active_side][:3, 3]
                        )
                    )
                )
                endpoint_orientation_error[key].append(
                    _rotation_error_deg(
                        target_world[active_side],
                        actual_world[active_side],
                    )
                )
                actual_desk = desk_from_world @ actual_world[active_side]
                forward_desk = actual_desk[:3, 2]
                endpoint_level_error[key].append(
                    float(np.rad2deg(math.asin(min(1.0, abs(float(forward_desk[2]))))))
                )
            records.append(
                {
                    "sequence": sequence,
                    "timestamp": timestamp,
                    "active_side": active_side,
                    "heading": heading,
                    "heading_yaw_deg": frame["heading_yaw_deg"],
                    "phase": phase,
                    "primitive_id": frame["primitive_id"],
                    "primitive_hash": frame["primitive_hash"],
                    "reference_frame_id": _DESK_REFERENCE_FRAME_SPEC.frame_id,
                    "reference_axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
                    "world_from_desk": control.world_from_desk.tolist(),
                    "target_pose_world_by_side": {
                        side: np.asarray(target_world[side], dtype=float).tolist()
                        for side in _SIDES
                    },
                    "actual_pose_world_by_side": {
                        side: actual_world[side].tolist() for side in _SIDES
                    },
                    "actual_qpos_by_side": {
                        side: qpos[side].tolist() for side in _SIDES
                    },
                    "ik_status": str(result.status),
                }
            )

        metrics: dict[str, float | int | bool | str | None] = {
            "reference_hz": cfg.reference_hz,
            "command_hz": float(control.contract.timing.command_hz),
            "physics_hz": float(control.contract.timing.physics_hz),
            "reference_count": len(records),
            "primitive_count": len(LEVEL_TCP_ORIENTATION_PRIMITIVES),
            "arm_primitive_execution_count": (
                len(_SIDES) * len(LEVEL_TCP_ORIENTATION_PRIMITIVES)
            ),
            "ik_failure_count": sum(
                count
                for status, count in status_counts.items()
                if status not in {"tracking", "boundary_active"}
            ),
            "nonfinite_qcmd_count": nonfinite_qcmd_count,
            "max_qcmd_step_deg": max_qcmd_step_deg,
            "max_target_translation_drift_m": max_target_translation_drift_m,
            "max_inactive_position_drift_m": max_inactive_position_drift_m,
            "reference_frame_id": _DESK_REFERENCE_FRAME_SPEC.frame_id,
            "reference_axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        }
        failures: list[str] = []
        for key in sorted(endpoint_position_error):
            side, heading = key
            position_samples = endpoint_position_error[key]
            orientation_samples = endpoint_orientation_error[key]
            level_samples = endpoint_level_error[key]
            prefix = f"{side}_{heading}"
            final_position = position_samples[-1] if position_samples else float("inf")
            final_orientation = (
                orientation_samples[-1] if orientation_samples else float("inf")
            )
            final_level = level_samples[-1] if level_samples else float("inf")
            metrics[f"{prefix}_final_position_error_m"] = final_position
            metrics[f"{prefix}_final_orientation_error_deg"] = final_orientation
            metrics[f"{prefix}_final_level_error_deg"] = final_level
            if final_position > cfg.maximum_endpoint_position_error_m:
                failures.append(f"{prefix} did not preserve TCP position")
            if final_orientation > cfg.maximum_endpoint_orientation_error_deg:
                failures.append(f"{prefix} did not reach its target orientation")
            if final_level > cfg.maximum_endpoint_level_error_deg:
                failures.append(f"{prefix} did not remain level")
        final_actual = records[-1]["actual_pose_world_by_side"]
        for side in _SIDES:
            final_pose = np.asarray(final_actual[side], dtype=float)
            position_error = float(
                np.linalg.norm(final_pose[:3, 3] - anchors_world[side][:3, 3])
            )
            orientation_error = _rotation_error_deg(anchors_world[side], final_pose)
            metrics[f"{side}_final_anchor_position_error_m"] = position_error
            metrics[f"{side}_final_anchor_orientation_error_deg"] = orientation_error
            if position_error > cfg.maximum_final_anchor_position_error_m:
                failures.append(f"{side} did not return to its anchor position")
            if orientation_error > cfg.maximum_final_anchor_orientation_error_deg:
                failures.append(f"{side} did not return to its anchor orientation")
        if max_target_translation_drift_m > cfg.maximum_target_translation_drift_m:
            failures.append("an orientation target changed TCP translation")
        if int(metrics["ik_failure_count"]) > 0:
            failures.append("IK returned a non-tracking status")
        if nonfinite_qcmd_count > 0:
            failures.append("the qcmd stream contained non-finite values")
        if max_qcmd_step_deg > float(control.config.max_command_step_deg) + 1e-6:
            failures.append("the qcmd stream exceeded its configured step bound")
        return SimRealScenarioResult(
            scenario="level_tcp_orientation_primitives_active_closed_loop",
            passed=not failures,
            metrics=metrics,
            records=tuple(records),
            failure="; ".join(failures),
        )
    except Exception as exc:  # noqa: BLE001 - bounded scenario result.
        return SimRealScenarioResult(
            scenario="level_tcp_orientation_primitives_active_closed_loop",
            passed=False,
            records=tuple(records),
            failure=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if control is not None:
            control.close()


__all__ = [
    "LevelTcpOrientationClosedLoopConfig",
    "build_level_tcp_orientation_world_trajectory",
    "run_level_tcp_orientation_primitives_active_closed_loop",
]
