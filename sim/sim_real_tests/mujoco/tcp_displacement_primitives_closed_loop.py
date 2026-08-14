"""Exercise the six cardinal TCP primitives through the MuJoCo motion chain.

The scenario runs every 3 cm primitive from one measured ``tool_tcp`` anchor,
returns to that anchor, and then advances to the next direction.  Cardinal
directions use the canonical station ``desk`` frame: +X is forward from the
operator/robot-near boundary into the tabletop, +Y is the operator's left, and
+Z is up.  The left arm completes all six directions before the right arm
begins.  Targets enter the existing 60 Hz coupled IK -> 120 Hz qcmd -> 1000 Hz
physics path; this module does not introduce another controller or production
entrypoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.position_control import ResampledMotionReference
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    CardinalTCPPoseDirection,
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
class CardinalTcpPrimitiveClosedLoopConfig:
    reference_hz: float = 60.0
    samples_per_leg: int = 18
    endpoint_hold_frames: int = 8
    anchor_hold_frames: int = 8
    minimum_actual_projection_m: float = 0.003
    maximum_final_anchor_error_m: float = 0.02

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
            "minimum_actual_projection_m",
            "maximum_final_anchor_error_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


def _world_to_desk(
    control: DeskTcpPositionControl,
    pose_world: np.ndarray,
) -> np.ndarray:
    return np.linalg.inv(control.world_from_desk) @ np.asarray(
        pose_world,
        dtype=float,
    )


def _interpolate_translation(
    start: np.ndarray,
    end: np.ndarray,
    alpha: float,
) -> np.ndarray:
    pose = np.asarray(start, dtype=float).copy()
    pose[:3, 3] = (
        (1.0 - float(alpha)) * np.asarray(start[:3, 3], dtype=float)
        + float(alpha) * np.asarray(end[:3, 3], dtype=float)
    )
    return pose


def build_cardinal_tcp_primitive_world_trajectory(
    *,
    anchor_pose_world_by_side: Mapping[str, np.ndarray],
    world_from_desk: np.ndarray,
    config: CardinalTcpPrimitiveClosedLoopConfig | None = None,
    reference_frame_spec: CoordinateFrameSpec = _DESK_REFERENCE_FRAME_SPEC,
) -> tuple[dict[str, Any], ...]:
    """Build world targets from primitives expressed in the station frame."""

    cfg = config or CardinalTcpPrimitiveClosedLoopConfig()
    if set(anchor_pose_world_by_side) != set(_SIDES):
        raise ValueError("anchor_pose_world_by_side must contain left and right")
    anchors = {
        side: np.asarray(anchor_pose_world_by_side[side], dtype=float).copy()
        for side in _SIDES
    }
    world_from_desk = np.asarray(world_from_desk, dtype=float)
    if world_from_desk.shape != (4, 4) or not np.all(
        np.isfinite(world_from_desk)
    ):
        raise ValueError("world_from_desk must be a finite 4x4 pose")
    desk_from_world = np.linalg.inv(world_from_desk)
    anchors_desk = {
        side: desk_from_world @ anchors[side] for side in _SIDES
    }
    frames: list[dict[str, Any]] = [
        {
            "active_side": None,
            "direction": None,
            "phase": "initial_anchor",
            "primitive_id": None,
            "primitive_hash": None,
            "target_pose_world_by_side": {
                side: anchors[side].copy() for side in _SIDES
            },
        }
    ]
    for side in _SIDES:
        for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES:
            target = primitive.apply(
                anchors_desk[side],
                reference_frame_spec=reference_frame_spec,
            )
            endpoint = world_from_desk @ np.asarray(
                target.target_pose,
                dtype=float,
            )
            for phase, start, end, count in (
                (
                    "outbound",
                    anchors[side],
                    endpoint,
                    cfg.samples_per_leg,
                ),
                (
                    "return",
                    endpoint,
                    anchors[side],
                    cfg.samples_per_leg,
                ),
            ):
                for sample_index in range(1, count + 1):
                    alpha = float(sample_index) / float(count)
                    active_pose = _interpolate_translation(start, end, alpha)
                    frames.append(
                        {
                            "active_side": side,
                            "direction": primitive.direction.value,
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
                hold_phase = (
                    "target_hold" if phase == "outbound" else "anchor_hold"
                )
                hold_count = (
                    cfg.endpoint_hold_frames
                    if phase == "outbound"
                    else cfg.anchor_hold_frames
                )
                for _ in range(hold_count):
                    frames.append(
                        {
                            "active_side": side,
                            "direction": primitive.direction.value,
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


def _rotation_error_deg(reference: np.ndarray, actual: np.ndarray) -> float:
    relative = (
        np.asarray(actual[:3, :3], dtype=float)
        @ np.asarray(reference[:3, :3], dtype=float).T
    )
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))


def run_cardinal_tcp_primitives_active_closed_loop(
    config: CardinalTcpPrimitiveClosedLoopConfig | None = None,
) -> SimRealScenarioResult:
    """Run both arms and return a serializable physical playback trace."""

    cfg = config or CardinalTcpPrimitiveClosedLoopConfig()
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
        anchors = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=initial_qpos["left"],
            right_qpos=initial_qpos["right"],
        )
        frames = build_cardinal_tcp_primitive_world_trajectory(
            anchor_pose_world_by_side=anchors,
            world_from_desk=control.world_from_desk,
            config=cfg,
        )
        physics_ticks_exact = (
            float(control.contract.timing.physics_hz) / cfg.reference_hz
        )
        physics_tick_accumulator = 0.0
        previous_qcmd = control.rig.accepted_qcmd_by_side()
        status_counts: dict[str, int] = {}
        nonfinite_qcmd_count = 0
        max_qcmd_step_deg = 0.0
        max_inactive_drift_m = 0.0
        max_orientation_error_deg = 0.0
        projection_samples: dict[tuple[str, str], list[float]] = {
            (side, direction.value): []
            for side in _SIDES
            for direction in CardinalTCPPoseDirection
        }
        endpoint_error_samples: dict[tuple[str, str], list[float]] = {
            key: [] for key in projection_samples
        }

        for sequence, frame in enumerate(frames):
            timestamp = float(sequence) / cfg.reference_hz
            target_world = frame["target_pose_world_by_side"]
            reference = ResampledMotionReference(
                sequence=sequence,
                timestamp=timestamp,
                left_source_index=sequence,
                right_source_index=sequence,
                left_source_timestamp=timestamp,
                right_source_timestamp=timestamp,
                interpolation_alpha=0.0,
                pose_by_side={
                    side: _world_to_desk(control, target_world[side])
                    for side in _SIDES
                },
                metadata={
                    "source": "cardinal_tcp_displacement_primitive_visual_review",
                    "active_side": frame["active_side"],
                    "direction": frame["direction"],
                    "phase": frame["phase"],
                    "primitive_id": frame["primitive_id"],
                    "primitive_hash": frame["primitive_hash"],
                },
            )
            result = control.solve_and_dispatch(
                reference,
                gripper_close_by_side={"left": 0.0, "right": 0.0},
            )
            status_counts[result.status] = int(
                status_counts.get(result.status, 0)
            ) + 1
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
                            np.max(
                                np.abs(
                                    np.rad2deg(
                                        qcmd[side] - previous_qcmd[side]
                                    )
                                )
                            )
                        ),
                    )
                previous_qcmd = qcmd

            qpos = {
                side: control.scene.get_arm_qpos(side).copy() for side in _SIDES
            }
            actual = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=qpos["left"],
                right_qpos=qpos["right"],
            )
            active_side = frame["active_side"]
            direction = frame["direction"]
            phase = frame["phase"]
            for side in _SIDES:
                max_orientation_error_deg = max(
                    max_orientation_error_deg,
                    _rotation_error_deg(anchors[side], actual[side]),
                )
                if active_side is not None and side != active_side:
                    max_inactive_drift_m = max(
                        max_inactive_drift_m,
                        float(
                            np.linalg.norm(
                                actual[side][:3, 3] - anchors[side][:3, 3]
                            )
                        ),
                    )
            if (
                active_side in _SIDES
                and direction is not None
                and phase == "target_hold"
            ):
                key = (str(active_side), str(direction))
                primitive = next(
                    item
                    for item in CARDINAL_TCP_SHORT_STEP_PRIMITIVES
                    if item.direction.value == direction
                )
                unit_reference = (
                    primitive.delta_translation_reference_m
                    / primitive.distance_m
                )
                unit_world = (
                    control.world_from_desk[:3, :3] @ unit_reference
                )
                actual_delta = (
                    actual[active_side][:3, 3]
                    - anchors[active_side][:3, 3]
                )
                projection_samples[key].append(
                    float(np.dot(actual_delta, unit_world))
                )
                endpoint_error_samples[key].append(
                    float(
                        np.linalg.norm(
                            actual[active_side][:3, 3]
                            - target_world[active_side][:3, 3]
                        )
                    )
                )
            records.append(
                {
                    "sequence": sequence,
                    "timestamp": timestamp,
                    "active_side": active_side,
                    "direction": direction,
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
                        side: actual[side].tolist() for side in _SIDES
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
            "primitive_count": len(CARDINAL_TCP_SHORT_STEP_PRIMITIVES),
            "arm_primitive_execution_count": len(_SIDES)
            * len(CARDINAL_TCP_SHORT_STEP_PRIMITIVES),
            "ik_failure_count": sum(
                count
                for status, count in status_counts.items()
                if status not in {"tracking", "boundary_active"}
            ),
            "nonfinite_qcmd_count": nonfinite_qcmd_count,
            "max_qcmd_step_deg": max_qcmd_step_deg,
            "max_inactive_arm_drift_m": max_inactive_drift_m,
            "max_orientation_error_deg": max_orientation_error_deg,
            "reference_frame_id": _DESK_REFERENCE_FRAME_SPEC.frame_id,
            "reference_axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
        }
        failures: list[str] = []
        for key in sorted(projection_samples):
            side, direction = key
            projections = projection_samples[key]
            endpoint_errors = endpoint_error_samples[key]
            metric_prefix = f"{side}_{direction}"
            peak_projection = max(projections) if projections else float("-inf")
            final_endpoint_error = (
                endpoint_errors[-1] if endpoint_errors else float("inf")
            )
            metrics[f"{metric_prefix}_peak_actual_projection_m"] = (
                peak_projection
            )
            metrics[f"{metric_prefix}_final_endpoint_error_m"] = (
                final_endpoint_error
            )
            if peak_projection < cfg.minimum_actual_projection_m:
                failures.append(
                    f"{metric_prefix} did not visibly move in the requested direction"
                )
        final_actual = records[-1]["actual_pose_world_by_side"]
        for side in _SIDES:
            final_anchor_error = float(
                np.linalg.norm(
                    np.asarray(final_actual[side], dtype=float)[:3, 3]
                    - anchors[side][:3, 3]
                )
            )
            metrics[f"{side}_final_anchor_error_m"] = final_anchor_error
            if final_anchor_error > cfg.maximum_final_anchor_error_m:
                failures.append(f"{side} did not return near its anchor")
        if int(metrics["ik_failure_count"]) > 0:
            failures.append("IK returned a non-tracking status")
        if nonfinite_qcmd_count > 0:
            failures.append("the qcmd stream contained non-finite values")
        if max_qcmd_step_deg > float(control.config.max_command_step_deg) + 1e-6:
            failures.append("the qcmd stream exceeded its configured step bound")
        return SimRealScenarioResult(
            scenario="cardinal_tcp_displacement_primitives_active_closed_loop",
            passed=not failures,
            metrics=metrics,
            records=tuple(records),
            failure="; ".join(failures),
        )
    except Exception as exc:  # noqa: BLE001 - bounded scenario result.
        return SimRealScenarioResult(
            scenario="cardinal_tcp_displacement_primitives_active_closed_loop",
            passed=False,
            records=tuple(records),
            failure=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if control is not None:
            control.close()


__all__ = [
    "CardinalTcpPrimitiveClosedLoopConfig",
    "build_cardinal_tcp_primitive_world_trajectory",
    "run_cardinal_tcp_primitives_active_closed_loop",
]
