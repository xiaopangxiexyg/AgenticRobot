"""Active TCP-envelope acceptance through the physical MuJoCo control chain.

The scenario intentionally exercises the shared pre-IK target seam used by
real position control, while remaining isolated from ``teleop_runtime`` and
all hardware backends.  It runs 15 Hz Cartesian source targets through 60 Hz
coupled IK, 120 Hz joint commands, and 1000 Hz MuJoCo physics.

This scenario certifies target projection and closed-loop behavior at emitted
target samples.  It does not claim swept-TCP certification between samples.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.position_control import (
    TimestampedMotionTarget,
    resample_motion_targets,
)
from robot_motion.target_safety.tcp_pose_envelope import (
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
)
from sim.mujoco.adapters.tcp_pose_envelope import (
    install_mujoco_tcp_pose_envelope_filter,
    mujoco_target_frame_binding_hash,
    mujoco_tool_frame_binding_hashes,
)
from sim.mujoco.control.desk_tcp_position import DeskTcpPositionControl

from ..core import SimRealScenarioResult


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "tcp_pose_envelope_active_closed_loop_v1.json"
)

_REFERENCE_FRAME = "mujoco_world"
_EXECUTION_EPOCH = 1


@dataclass(frozen=True)
class TCPPoseEnvelopeClosedLoopConfig:
    source_hz: float = 15.0
    reference_hz: float = 60.0
    translation_half_extent_m: float = 0.015
    max_rotation_deg: float = 8.0
    inside_translation_m: float = 0.006
    outside_translation_m: float = 0.030
    inside_rotation_deg: float = 2.0
    outside_rotation_deg: float = 15.0
    ramp_source_frames: int = 4
    outside_hold_source_frames: int = 4
    return_source_frames: int = 4
    final_hold_source_frames: int = 4
    minimum_nonzero_residual_m: float = 1e-5
    projection_tolerance_m: float = 1e-8
    projection_tolerance_deg: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("source_hz", "reference_hz"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.reference_hz < self.source_hz:
            raise ValueError("reference_hz must be at least source_hz")
        for name in (
            "translation_half_extent_m",
            "max_rotation_deg",
            "inside_translation_m",
            "outside_translation_m",
            "inside_rotation_deg",
            "outside_rotation_deg",
            "minimum_nonzero_residual_m",
            "projection_tolerance_m",
            "projection_tolerance_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.inside_translation_m >= self.translation_half_extent_m:
            raise ValueError("inside_translation_m must be inside the envelope")
        if self.outside_translation_m <= self.translation_half_extent_m:
            raise ValueError("outside_translation_m must exceed the envelope")
        if self.inside_rotation_deg >= self.max_rotation_deg:
            raise ValueError("inside_rotation_deg must be inside the envelope")
        if self.outside_rotation_deg <= self.max_rotation_deg:
            raise ValueError("outside_rotation_deg must exceed the envelope")
        for name in (
            "ramp_source_frames",
            "outside_hold_source_frames",
            "return_source_frames",
            "final_hold_source_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "TCPPoseEnvelopeClosedLoopConfig":
        values = dict(payload)
        values.pop("schema_version", None)
        values.pop("scenario", None)
        return cls(**values)


def load_tcp_pose_envelope_closed_loop_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> TCPPoseEnvelopeClosedLoopConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return TCPPoseEnvelopeClosedLoopConfig.from_mapping(payload)


def _rotation_error_deg(reference: np.ndarray, actual: np.ndarray) -> float:
    return float(
        np.rad2deg(
            Rotation.from_matrix(
                np.asarray(reference[:3, :3], dtype=float).T
                @ np.asarray(actual[:3, :3], dtype=float)
            ).magnitude()
        )
    )


def _max_basis_direction_error_deg(reference: np.ndarray, actual: np.ndarray) -> float:
    dots = np.sum(reference[:3, :3] * actual[:3, :3], axis=0)
    return float(np.max(np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))))


def _world_to_desk(control: DeskTcpPositionControl, pose_world: np.ndarray) -> np.ndarray:
    return np.linalg.inv(control.world_from_desk) @ np.asarray(pose_world, dtype=float)


def _source_targets(
    *,
    control: DeskTcpPositionControl,
    anchor_tool_world_by_side: Mapping[str, np.ndarray],
    config: TCPPoseEnvelopeClosedLoopConfig,
) -> list[TimestampedMotionTarget]:
    anchor_left = np.asarray(anchor_tool_world_by_side["left"], dtype=float)
    anchor_right_desk = _world_to_desk(control, anchor_tool_world_by_side["right"])
    inside = float(config.inside_translation_m)
    outside = float(config.outside_translation_m)
    half = float(config.translation_half_extent_m)
    anchor_offset = np.zeros(3, dtype=float)
    inside_a = np.array([inside, 0.65 * inside, -0.35 * inside], dtype=float)
    outside_a = np.array([outside, 0.55 * half, 0.25 * half], dtype=float)
    outside_b = np.array([0.75 * outside, -0.80 * outside, 0.60 * half], dtype=float)
    inside_b = np.array([-0.70 * inside, -inside, 0.45 * inside], dtype=float)
    phases = (
        (
            int(config.ramp_source_frames),
            anchor_offset,
            inside_a,
            0.0,
            float(config.inside_rotation_deg),
            "inside_ramp",
        ),
        (
            int(config.ramp_source_frames),
            inside_a,
            outside_a,
            float(config.inside_rotation_deg),
            float(config.outside_rotation_deg),
            "outside_ramp",
        ),
        (
            int(config.outside_hold_source_frames),
            outside_a,
            outside_b,
            float(config.outside_rotation_deg),
            -float(config.outside_rotation_deg),
            "outside_traverse",
        ),
        (
            int(config.return_source_frames),
            outside_b,
            inside_b,
            -float(config.outside_rotation_deg),
            -float(config.inside_rotation_deg),
            "return_inward",
        ),
        (
            int(config.final_hold_source_frames),
            inside_b,
            anchor_offset,
            -float(config.inside_rotation_deg),
            0.0,
            "return_to_anchor",
        ),
    )
    samples: list[TimestampedMotionTarget] = []
    source_index = 0
    for phase_index, (
        count,
        start_offset,
        end_offset,
        start_deg,
        end_deg,
        name,
    ) in enumerate(phases):
        for phase_frame in range(count):
            if phase_index > 0 and phase_frame == 0:
                # Adjacent phases share an endpoint; emit it once.
                continue
            alpha = 1.0 if count == 1 else float(phase_frame) / float(count - 1)
            offset = np.asarray(
                start_offset + alpha * (end_offset - start_offset),
                dtype=float,
            )
            angle_deg = float(start_deg + alpha * (end_deg - start_deg))
            left_world = anchor_left.copy()
            left_world[:3, 3] += offset
            left_world[:3, :3] = (
                anchor_left[:3, :3]
                @ Rotation.from_euler("z", angle_deg, degrees=True).as_matrix()
            )
            samples.append(
                TimestampedMotionTarget(
                    timestamp=float(source_index) / float(config.source_hz),
                    source_index=source_index,
                    pose_by_side={
                        "left": _world_to_desk(control, left_world),
                        "right": anchor_right_desk,
                    },
                    metadata={"phase": name},
                )
            )
            source_index += 1
    return samples


def run_tcp_pose_envelope_active_closed_loop(
    config: TCPPoseEnvelopeClosedLoopConfig | None = None,
) -> SimRealScenarioResult:
    """Run the Active target envelope against the full MuJoCo position chain."""

    cfg = config or load_tcp_pose_envelope_closed_loop_config()
    control: DeskTcpPositionControl | None = None
    records: list[dict[str, Any]] = []
    try:
        control = DeskTcpPositionControl.build()
        initial_qpos = {
            side: control.scene.get_arm_qpos(side).copy()
            for side in ("left", "right")
        }
        anchor_tool_world = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=initial_qpos["left"],
            right_qpos=initial_qpos["right"],
        )
        target_binding_hash = mujoco_target_frame_binding_hash(
            reference_frame=_REFERENCE_FRAME,
            world_from_desk=control.world_from_desk,
            world_to_base_by_side=control.world_to_base_by_side,
        )
        tool_binding_hash_by_side = mujoco_tool_frame_binding_hashes(
            control.retargeter,
            sides=("left", "right"),
        )
        half = float(cfg.translation_half_extent_m)
        profile = TCPPoseEnvelopeProfile(
            profile_id="sim-active-left-relative-envelope-v1",
            target_frame_binding_hash=target_binding_hash,
            tool_frame_binding_hash_by_side=tool_binding_hash_by_side,
            reference_frame_by_side={
                "left": _REFERENCE_FRAME,
                "right": _REFERENCE_FRAME,
            },
            side_config_by_side={
                "left": TCPPoseEnvelopeSideConfig(
                    relative_min_xyz_m=(-half, -half, -half),
                    relative_max_xyz_m=(half, half, half),
                    relative_axes="canonical_axes",
                    max_rotation_rad=math.radians(cfg.max_rotation_deg),
                )
            },
        )
        anchor = TCPPoseEnvelopeAnchorReceipt.capture(
            anchor_id="mujoco_measured_start_tool_tcp",
            profile=profile,
            execution_epoch=_EXECUTION_EPOCH,
            clock_domain_id="mujoco_physics_time",
            tcp_pose_by_side={"left": anchor_tool_world["left"]},
            captured_at_by_side={"left": float(control.scene.data.time)},
        )
        active_filter = install_mujoco_tcp_pose_envelope_filter(
            retargeter=control.retargeter,
            profile=profile,
            target_frame_binding_hash=target_binding_hash,
            execution_epoch=_EXECUTION_EPOCH,
            anchor_receipt=anchor,
            apply_to_target=True,
        )
        source_targets = _source_targets(
            control=control,
            anchor_tool_world_by_side=anchor_tool_world,
            config=cfg,
        )
        references = resample_motion_targets(
            source_targets,
            reference_hz=cfg.reference_hz,
            include_final=True,
        )

        physics_ticks_exact = (
            control.contract.timing.physics_hz / float(cfg.reference_hz)
        )
        physics_tick_accumulator = 0.0
        previous_qcmd = control.rig.accepted_qcmd_by_side()
        max_qcmd_step_deg = 0.0
        max_actual_residual_m = 0.0
        max_actual_rotation_residual_deg = 0.0
        max_actual_translation_component_m = 0.0
        max_actual_anchor_rotation_deg = 0.0
        max_effective_translation_component_m = 0.0
        max_effective_rotation_deg = 0.0
        max_raw_translation_component_m = 0.0
        max_raw_rotation_deg = 0.0
        max_raw_direction_angle_deg = 0.0
        max_effective_direction_angle_deg = 0.0
        max_actual_direction_angle_deg = 0.0
        modified_reference_count = 0
        translation_modified_reference_count = 0
        rotation_modified_reference_count = 0
        inward_release_count = 0
        first_modified_index: int | None = None
        first_inward_release_index: int | None = None
        ik_failure_count = 0
        nonfinite_qcmd_count = 0
        right_projection_change_m = 0.0

        for reference_index, reference in enumerate(references):
            actual_before = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=control.scene.get_arm_qpos("left"),
                right_qpos=control.scene.get_arm_qpos("right"),
            )
            result = control.solve_and_dispatch(
                reference,
                gripper_close_by_side={"left": 0.0, "right": 0.0},
            )
            if result.status not in {"tracking", "boundary_active"}:
                ik_failure_count += 1
            projection = active_filter.last_projection
            if projection is None:
                raise RuntimeError("Active TCP envelope produced no projection")
            side_projection = projection.side_projection_by_side["left"]
            raw_left = projection.raw_tcp_pose_by_side["left"]
            effective_left = projection.effective_tcp_pose_by_side["left"]
            anchor_left = anchor_tool_world["left"]
            raw_relative_translation = raw_left[:3, 3] - anchor_left[:3, 3]
            relative_translation = effective_left[:3, 3] - anchor_left[:3, 3]
            raw_rotation_deg = _rotation_error_deg(anchor_left, raw_left)
            effective_rotation_deg = _rotation_error_deg(anchor_left, effective_left)
            max_raw_translation_component_m = max(
                max_raw_translation_component_m,
                float(np.max(np.abs(raw_relative_translation))),
            )
            max_raw_rotation_deg = max(max_raw_rotation_deg, raw_rotation_deg)
            max_raw_direction_angle_deg = max(
                max_raw_direction_angle_deg,
                _max_basis_direction_error_deg(anchor_left, raw_left),
            )
            max_effective_direction_angle_deg = max(
                max_effective_direction_angle_deg,
                _max_basis_direction_error_deg(anchor_left, effective_left),
            )
            max_effective_translation_component_m = max(
                max_effective_translation_component_m,
                float(np.max(np.abs(relative_translation))),
            )
            max_effective_rotation_deg = max(
                max_effective_rotation_deg,
                effective_rotation_deg,
            )
            right_projection_change_m = max(
                right_projection_change_m,
                float(
                    np.max(
                        np.abs(
                            projection.effective_tcp_pose_by_side["right"]
                            - projection.raw_tcp_pose_by_side["right"]
                        )
                    )
                ),
            )
            if side_projection.modified:
                modified_reference_count += 1
                translation_modified_reference_count += int(
                    side_projection.translation_modified
                )
                rotation_modified_reference_count += int(
                    side_projection.rotation_modified
                )
                if first_modified_index is None:
                    first_modified_index = reference_index
            elif first_modified_index is not None:
                raw_delta = raw_left[:3, 3] - anchor_left[:3, 3]
                raw_rotation_deg = _rotation_error_deg(anchor_left, raw_left)
                if (
                    np.max(np.abs(raw_delta))
                    < cfg.translation_half_extent_m - cfg.projection_tolerance_m
                    and raw_rotation_deg
                    < cfg.max_rotation_deg - cfg.projection_tolerance_deg
                ):
                    inward_release_count += 1
                    if first_inward_release_index is None:
                        first_inward_release_index = reference_index

            physics_tick_accumulator += physics_ticks_exact
            physics_ticks = int(math.floor(physics_tick_accumulator + 1e-12))
            physics_tick_accumulator -= physics_ticks
            for _ in range(physics_ticks):
                control.step_physics()
                qcmd = control.rig.accepted_qcmd_by_side()
                for side in ("left", "right"):
                    if not np.all(np.isfinite(qcmd[side])):
                        nonfinite_qcmd_count += 1
                    max_qcmd_step_deg = max(
                        max_qcmd_step_deg,
                        float(
                            np.max(
                                np.abs(np.rad2deg(qcmd[side] - previous_qcmd[side]))
                            )
                        ),
                    )
                previous_qcmd = qcmd

            actual_after = control.retargeter.tool_tcp_world_targets_for_qpos_pair(
                left_qpos=control.scene.get_arm_qpos("left"),
                right_qpos=control.scene.get_arm_qpos("right"),
            )
            actual_residual_m = float(
                np.linalg.norm(
                    effective_left[:3, 3] - actual_after["left"][:3, 3]
                )
            )
            actual_rotation_residual_deg = _rotation_error_deg(
                effective_left,
                actual_after["left"],
            )
            actual_anchor_delta = (
                actual_after["left"][:3, 3] - anchor_tool_world["left"][:3, 3]
            )
            max_actual_translation_component_m = max(
                max_actual_translation_component_m,
                float(np.max(np.abs(actual_anchor_delta))),
            )
            max_actual_anchor_rotation_deg = max(
                max_actual_anchor_rotation_deg,
                _rotation_error_deg(anchor_tool_world["left"], actual_after["left"]),
            )
            max_actual_direction_angle_deg = max(
                max_actual_direction_angle_deg,
                _max_basis_direction_error_deg(
                    anchor_tool_world["left"], actual_after["left"]
                ),
            )
            max_actual_residual_m = max(max_actual_residual_m, actual_residual_m)
            max_actual_rotation_residual_deg = max(
                max_actual_rotation_residual_deg,
                actual_rotation_residual_deg,
            )
            records.append(
                {
                    "reference_index": reference_index,
                    "timestamp": float(reference.timestamp),
                    "source_indices": [
                        int(reference.left_source_index),
                        int(reference.right_source_index),
                    ],
                    "modified": bool(side_projection.modified),
                    "raw_relative_x_m": float(
                        raw_left[0, 3] - anchor_left[0, 3]
                    ),
                    "effective_relative_x_m": float(relative_translation[0]),
                    "effective_rotation_deg": effective_rotation_deg,
                    "actual_residual_before_m": float(
                        np.linalg.norm(
                            effective_left[:3, 3]
                            - actual_before["left"][:3, 3]
                        )
                    ),
                    "actual_residual_after_m": actual_residual_m,
                    "actual_rotation_residual_after_deg": (
                        actual_rotation_residual_deg
                    ),
                    "ik_status": str(result.status),
                    "raw_left_pose_world": raw_left.tolist(),
                    "effective_left_pose_world": effective_left.tolist(),
                    "actual_left_pose_world": actual_after["left"].tolist(),
                    "actual_qpos_by_side": {
                        side: control.scene.get_arm_qpos(side).tolist()
                        for side in ("left", "right")
                    },
                }
            )

        final_projection = active_filter.last_projection
        if final_projection is None:
            raise RuntimeError("Active TCP envelope produced no final projection")
        final_left = final_projection.effective_tcp_pose_by_side["left"]
        final_anchor_error_m = float(
            np.linalg.norm(final_left[:3, 3] - anchor_tool_world["left"][:3, 3])
        )
        final_anchor_rotation_error_deg = _rotation_error_deg(
            anchor_tool_world["left"],
            final_left,
        )
        max_step_limit_deg = float(control.config.max_command_step_deg)
        failures: list[str] = []
        if modified_reference_count == 0:
            failures.append("outside commands never activated the envelope")
        if translation_modified_reference_count == 0:
            failures.append("translation projection was not exercised")
        if rotation_modified_reference_count == 0:
            failures.append("rotation projection was not exercised")
        if inward_release_count == 0:
            failures.append("inward command was not released after clipping")
        if max_actual_residual_m < cfg.minimum_nonzero_residual_m:
            failures.append("closed loop never exercised a nonzero target residual")
        if (
            max_effective_translation_component_m
            > cfg.translation_half_extent_m + cfg.projection_tolerance_m
        ):
            failures.append("an effective target exceeded the translation envelope")
        if max_effective_rotation_deg > cfg.max_rotation_deg + cfg.projection_tolerance_deg:
            failures.append("an effective target exceeded the rotation envelope")
        if right_projection_change_m > cfg.projection_tolerance_m:
            failures.append("the unconfigured right target was modified")
        if final_anchor_error_m > cfg.projection_tolerance_m:
            failures.append("final target retained translation catch-up state")
        if final_anchor_rotation_error_deg > cfg.projection_tolerance_deg:
            failures.append("final target retained rotation catch-up state")
        if ik_failure_count:
            failures.append(f"IK reported {ik_failure_count} non-success references")
        if nonfinite_qcmd_count:
            failures.append(f"120 Hz qcmd was non-finite {nonfinite_qcmd_count} times")
        if max_qcmd_step_deg > max_step_limit_deg + 1e-6:
            failures.append("120 Hz qcmd exceeded its configured joint-step bound")

        return SimRealScenarioResult(
            scenario="tcp_pose_envelope_active_closed_loop",
            passed=not failures,
            metrics={
                "source_hz": cfg.source_hz,
                "reference_hz": cfg.reference_hz,
                "command_hz": control.contract.timing.command_hz,
                "physics_hz": control.contract.timing.physics_hz,
                "source_target_count": len(source_targets),
                "reference_count": len(references),
                "projection_count": active_filter.projection_count,
                "modified_reference_count": modified_reference_count,
                "translation_modified_reference_count": (
                    translation_modified_reference_count
                ),
                "rotation_modified_reference_count": (
                    rotation_modified_reference_count
                ),
                "inward_release_count": inward_release_count,
                "first_modified_index": first_modified_index,
                "first_inward_release_index": first_inward_release_index,
                "max_effective_translation_component_m": (
                    max_effective_translation_component_m
                ),
                "max_effective_rotation_deg": max_effective_rotation_deg,
                "max_raw_translation_component_m": max_raw_translation_component_m,
                "max_raw_rotation_deg": max_raw_rotation_deg,
                "max_raw_direction_angle_deg": max_raw_direction_angle_deg,
                "max_effective_direction_angle_deg": (
                    max_effective_direction_angle_deg
                ),
                "max_actual_direction_angle_deg": max_actual_direction_angle_deg,
                "max_actual_residual_m": max_actual_residual_m,
                "max_actual_rotation_residual_deg": (
                    max_actual_rotation_residual_deg
                ),
                "max_actual_translation_component_m": (
                    max_actual_translation_component_m
                ),
                "max_actual_anchor_rotation_deg": max_actual_anchor_rotation_deg,
                "max_qcmd_step_deg": max_qcmd_step_deg,
                "configured_max_qcmd_step_deg": max_step_limit_deg,
                "right_projection_change": right_projection_change_m,
                "final_anchor_error_m": final_anchor_error_m,
                "final_anchor_rotation_error_deg": final_anchor_rotation_error_deg,
                "ik_failure_count": ik_failure_count,
                "nonfinite_qcmd_count": nonfinite_qcmd_count,
                "swept_tcp_certified": False,
            },
            records=tuple(records),
            failure="; ".join(failures),
        )
    except Exception as exc:  # noqa: BLE001 - scenario reports a bounded failure.
        return SimRealScenarioResult(
            scenario="tcp_pose_envelope_active_closed_loop",
            passed=False,
            records=tuple(records),
            failure=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if control is not None:
            control.close()


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "TCPPoseEnvelopeClosedLoopConfig",
    "load_tcp_pose_envelope_closed_loop_config",
    "run_tcp_pose_envelope_active_closed_loop",
]
