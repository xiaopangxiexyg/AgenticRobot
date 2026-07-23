"""Temporary dual-arm output limiter used during real-hardware bring-up.

This is a move-only extraction from the real teleop entrypoint.  It preserves
the command, metadata invalidation, and limiting semantics; policy changes
belong in a separate safety change.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.runtime.immutable import deep_freeze

from ..transforms import as_pose_matrix


@dataclass(frozen=True)
class TemporaryDualArmSafetyConfig:
    enabled: bool = True
    active_sides: tuple[str, ...] = ("left", "right")
    max_xyz_step_m: float = 0.03
    max_rot_step_deg: float = 8.0
    max_joint_step_deg: float = 6.0
    command_scale: float = 1.0
    evidence_enabled: bool = False


TEMPORARY_DUAL_ARM_SAFETY_EVALUATION_HASH_DOMAIN_V1 = (
    "dexteleop.temporary_dual_arm_safety_evaluation.v1"
)


@dataclass(frozen=True)
class TemporaryArmSafetyStageEvidence:
    """Immutable evidence for one legacy arm-filter invocation.

    Values are flattened tuples plus an explicit shape so an observe-only
    consumer cannot retain or mutate the limiter's numpy state.
    """

    effector_name: str
    command_type: str
    input_shape: tuple[int, ...] | None
    input_values: tuple[float, ...] | None
    previous_anchor_shape: tuple[int, ...] | None
    previous_anchor_values: tuple[float, ...] | None
    output_shape: tuple[int, ...] | None
    output_values: tuple[float, ...] | None
    initialized: bool
    limited: bool
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        if not str(self.effector_name).strip() or not str(self.command_type).strip():
            raise ValueError("effector_name/command_type must be non-empty.")
        object.__setattr__(self, "details", MappingProxyType(dict(deep_freeze(dict(self.details)))))


@dataclass(frozen=True)
class TemporaryDualArmSafetyEvaluation:
    """Exact legacy filter-stage anchor/input/output evidence.

    This is audit evidence only.  In particular, ``output`` means the limiter
    updated its private compatibility state; it does not mean the command was
    accepted by a guard, backend, or execution target.
    """

    evaluation_id: str
    filter_sequence: int
    evaluated_at: float
    enabled: bool
    active_sides: tuple[str, ...]
    max_xyz_step_m: float
    max_rot_step_deg: float
    max_joint_step_deg: float
    command_scale: float
    stages: tuple[TemporaryArmSafetyStageEvidence, ...]
    semantic_content_hash: str


def _value_evidence(value: object) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None]:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None, None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None, None
    return tuple(int(item) for item in array.shape), tuple(
        float(item) for item in array.reshape(-1)
    )


class TemporaryDualArmSafetyLimiter:
    """Small removable output limiter for first dual-arm hardware bring-up."""

    _STALE_POSE_METADATA_KEYS = (
        "target_pose_base",
        "target_pose_tcp_base",
        "target_pose_desk",
        "solver_output_endpoint_pose_base",
        "solver_output_tcp_pose_base",
        "backend_final_target_qpos",
    )
    _STALE_QPOS_METADATA_KEYS = (
        "solver_output_qpos",
        "backend_final_target_qpos",
        "solver_output_endpoint_pose_base",
        "solver_output_tcp_pose_base",
    )

    def __init__(self, config: TemporaryDualArmSafetyConfig):
        self.config = config
        self._previous_by_effector: dict[str, np.ndarray] = {}
        self._filter_sequence = 0
        self._last_evaluation: TemporaryDualArmSafetyEvaluation | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and set(self.config.active_sides) == {"left", "right"})

    @property
    def last_evaluation(self) -> TemporaryDualArmSafetyEvaluation | None:
        return self._last_evaluation

    def sync_from_command(self, command: TeleopCommand | None) -> None:
        """Observe commands sent outside the session filter, such as recovery ramps."""

        if not self.enabled or command is None:
            return
        for effector_name, effector_command in command.commands.items():
            if effector_name in {"left_arm", "right_arm"}:
                self._sync_effector_command(effector_command)

    def filter_command(
        self,
        command: TeleopCommand,
        _observation,
        _now: float,
    ) -> TeleopCommand:
        if not self.enabled:
            if self.config.evidence_enabled:
                self._record_evaluation(
                    command,
                    command,
                    evaluated_at=float(_now),
                    stages=(),
                )
            return command
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
        stages: list[TemporaryArmSafetyStageEvidence] = []
        summary: dict[str, object] = {
            "enabled": True,
            "limited": False,
            "max_xyz_step_m": float(self.config.max_xyz_step_m),
            "max_rot_step_deg": float(self.config.max_rot_step_deg),
            "max_joint_step_deg": float(self.config.max_joint_step_deg),
            "command_scale": float(self.config.command_scale),
            "effectors": {},
        }
        for effector_name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            if effector_name in {"left_arm", "right_arm"}:
                if self.config.evidence_enabled:
                    previous_shape, previous_values = _value_evidence(
                        self._previous_by_effector.get(effector_name)
                    )
                    input_shape, input_values = _value_evidence(effector_command.value)
                next_command, effector_summary = self._filter_arm_command(next_command)
                if effector_summary:
                    summary["effectors"][effector_name] = effector_summary
                    summary["limited"] = bool(
                        summary["limited"] or effector_summary.get("limited")
                    )
                if self.config.evidence_enabled:
                    output_shape, output_values = _value_evidence(next_command.value)
                    stages.append(
                        TemporaryArmSafetyStageEvidence(
                            effector_name=effector_name,
                            command_type=str(effector_command.command_type),
                            input_shape=input_shape,
                            input_values=input_values,
                            previous_anchor_shape=previous_shape,
                            previous_anchor_values=previous_values,
                            output_shape=output_shape,
                            output_values=output_values,
                            initialized=bool(effector_summary.get("initialized", False)),
                            limited=bool(effector_summary.get("limited", False)),
                            details=dict(effector_summary),
                        )
                    )
            filtered.commands[effector_name] = next_command
        filtered.metadata["temporary_dual_arm_safety"] = summary
        if self.config.evidence_enabled:
            self._record_evaluation(
                command,
                filtered,
                evaluated_at=float(_now),
                stages=tuple(stages),
            )
        return filtered

    def _record_evaluation(
        self,
        input_command: TeleopCommand,
        output_command: TeleopCommand,
        *,
        evaluated_at: float,
        stages: tuple[TemporaryArmSafetyStageEvidence, ...],
    ) -> None:
        timestamp = float(evaluated_at)
        if not np.isfinite(timestamp):
            raise ValueError("temporary safety evaluation time must be finite.")
        payload = {
            "filter_sequence": int(self._filter_sequence),
            "evaluated_at": timestamp,
            "enabled": bool(self.enabled),
            "active_sides": tuple(str(side) for side in self.config.active_sides),
            "max_xyz_step_m": float(self.config.max_xyz_step_m),
            "max_rot_step_deg": float(self.config.max_rot_step_deg),
            "max_joint_step_deg": float(self.config.max_joint_step_deg),
            "command_scale": float(self.config.command_scale),
            "input_effectors": tuple(sorted(str(name) for name in input_command.commands)),
            "output_effectors": tuple(sorted(str(name) for name in output_command.commands)),
            "stages": tuple(
                {
                    "effector_name": stage.effector_name,
                    "command_type": stage.command_type,
                    "input_shape": stage.input_shape,
                    "input_values": stage.input_values,
                    "previous_anchor_shape": stage.previous_anchor_shape,
                    "previous_anchor_values": stage.previous_anchor_values,
                    "output_shape": stage.output_shape,
                    "output_values": stage.output_values,
                    "initialized": stage.initialized,
                    "limited": stage.limited,
                    "details": stage.details,
                }
                for stage in stages
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TEMPORARY_DUAL_ARM_SAFETY_EVALUATION_HASH_DOMAIN_V1,
        )
        self._last_evaluation = TemporaryDualArmSafetyEvaluation(
            evaluation_id=f"temporary-dual-arm-safety:{semantic_hash}",
            filter_sequence=int(self._filter_sequence),
            evaluated_at=timestamp,
            enabled=bool(self.enabled),
            active_sides=tuple(str(side) for side in self.config.active_sides),
            max_xyz_step_m=float(self.config.max_xyz_step_m),
            max_rot_step_deg=float(self.config.max_rot_step_deg),
            max_joint_step_deg=float(self.config.max_joint_step_deg),
            command_scale=float(self.config.command_scale),
            stages=tuple(stages),
            semantic_content_hash=semantic_hash,
        )
        self._filter_sequence += 1

    def _filter_arm_command(
        self,
        command: EffectorCommand,
    ) -> tuple[EffectorCommand, dict[str, object]]:
        if command.command_type == "cartesian_pose":
            return self._filter_cartesian_pose_command(command)
        if command.command_type == "joint_vector":
            return self._filter_joint_vector_command(command)
        return command, {
            "command_type": command.command_type,
            "limited": False,
            "reason": "unsupported_command_type",
        }

    def _filter_cartesian_pose_command(
        self,
        command: EffectorCommand,
    ) -> tuple[EffectorCommand, dict[str, object]]:
        try:
            target = as_pose_matrix(command.value)
        except Exception as exc:  # noqa: BLE001
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "reason": f"invalid_pose:{type(exc).__name__}",
            }
        previous = self._previous_by_effector.get(command.effector_name)
        if previous is None:
            self._previous_by_effector[command.effector_name] = target.copy()
            command.metadata = {
                **dict(command.metadata),
                "temporary_dual_arm_safety_initialized": True,
            }
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "initialized": True,
            }
        limited_pose, limit_metadata = self._limit_pose(previous, target)
        self._previous_by_effector[command.effector_name] = limited_pose.copy()
        out = command.copy()
        out.value = limited_pose
        command_metadata = {
            **dict(out.metadata),
            "temporary_dual_arm_safety": limit_metadata,
            "temporary_dual_arm_safety_limited": bool(limit_metadata.get("limited", False)),
        }
        if bool(limit_metadata.get("limited", False)):
            for key in self._STALE_QPOS_METADATA_KEYS:
                command_metadata.pop(key, None)
            command_metadata.update(
                {
                    "target_pose_base": limited_pose.copy(),
                    "target_pose_tcp_base": limited_pose.copy(),
                    "action_pose_source": "temporary_safety_limited_cartesian_pose",
                }
            )
        out.metadata = command_metadata
        return out, {"command_type": command.command_type, **limit_metadata}

    def _sync_effector_command(self, command: EffectorCommand) -> None:
        try:
            if command.command_type == "cartesian_pose":
                self._previous_by_effector[command.effector_name] = as_pose_matrix(
                    command.value
                ).copy()
            elif command.command_type == "joint_vector":
                self._previous_by_effector[command.effector_name] = np.asarray(
                    command.value,
                    dtype=float,
                ).reshape(-1).copy()
        except Exception:
            return

    def _filter_joint_vector_command(
        self,
        command: EffectorCommand,
    ) -> tuple[EffectorCommand, dict[str, object]]:
        try:
            target = np.asarray(command.value, dtype=float).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "reason": f"invalid_joint_vector:{type(exc).__name__}",
            }
        previous = self._previous_by_effector.get(command.effector_name)
        if previous is None or previous.shape != target.shape:
            self._previous_by_effector[command.effector_name] = target.copy()
            command.metadata = {
                **dict(command.metadata),
                "temporary_dual_arm_safety_initialized": True,
            }
            return command, {
                "command_type": command.command_type,
                "limited": False,
                "initialized": True,
            }
        delta = target - previous
        scale = max(0.0, min(1.0, float(self.config.command_scale)))
        delta = delta * scale
        max_step_rad = np.deg2rad(max(0.0, float(self.config.max_joint_step_deg)))
        if max_step_rad > 0.0:
            delta = np.clip(delta, -max_step_rad, max_step_rad)
        output = previous + delta
        self._previous_by_effector[command.effector_name] = output.copy()
        raw_delta = target - previous
        limited = not np.allclose(output, target, atol=1e-10)
        metadata = {
            "limited": bool(limited),
            "raw_max_step_deg": (
                float(np.rad2deg(np.max(np.abs(raw_delta)))) if raw_delta.size else 0.0
            ),
            "applied_max_step_deg": (
                float(np.rad2deg(np.max(np.abs(delta)))) if delta.size else 0.0
            ),
            "max_joint_step_deg": float(self.config.max_joint_step_deg),
            "command_scale": float(self.config.command_scale),
        }
        out = command.copy()
        out.value = output
        command_metadata = {
            **dict(out.metadata),
            "temporary_dual_arm_safety": metadata,
            "temporary_dual_arm_safety_limited": bool(limited),
        }
        if limited:
            for key in self._STALE_POSE_METADATA_KEYS:
                command_metadata.pop(key, None)
            command_metadata["solver_output_qpos"] = output.copy()
            command_metadata["action_pose_source"] = "limited_solver_output_qpos"
            command_metadata["temporary_dual_arm_safety_action_pose_pending_fk"] = True
        out.metadata = command_metadata
        return out, {"command_type": command.command_type, **metadata}

    def _limit_pose(
        self,
        previous: np.ndarray,
        target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        output = np.asarray(previous, dtype=float).reshape(4, 4).copy()
        target_pose = np.asarray(target, dtype=float).reshape(4, 4)
        scale = max(0.0, min(1.0, float(self.config.command_scale)))
        translation_delta = (target_pose[:3, 3] - output[:3, 3]) * scale
        raw_translation_norm = float(
            np.linalg.norm(target_pose[:3, 3] - output[:3, 3])
        )
        max_xyz = max(0.0, float(self.config.max_xyz_step_m))
        applied_translation_norm = float(np.linalg.norm(translation_delta))
        if max_xyz > 0.0 and applied_translation_norm > max_xyz:
            translation_delta = translation_delta * (
                max_xyz / max(1e-12, applied_translation_norm)
            )
            applied_translation_norm = max_xyz
        prev_rot = SciPyRotation.from_matrix(output[:3, :3])
        target_rot = SciPyRotation.from_matrix(target_pose[:3, :3])
        delta_rot = target_rot * prev_rot.inv()
        raw_rot_deg = float(np.rad2deg(delta_rot.magnitude()))
        rotvec = delta_rot.as_rotvec() * scale
        applied_rot_deg = float(np.rad2deg(np.linalg.norm(rotvec)))
        max_rot_deg = max(0.0, float(self.config.max_rot_step_deg))
        if max_rot_deg > 0.0 and applied_rot_deg > max_rot_deg:
            rotvec = rotvec * (max_rot_deg / max(1e-12, applied_rot_deg))
            applied_rot_deg = max_rot_deg
        output[:3, 3] = output[:3, 3] + translation_delta
        output[:3, :3] = (SciPyRotation.from_rotvec(rotvec) * prev_rot).as_matrix()
        limited = bool(
            not np.allclose(output[:3, 3], target_pose[:3, 3], atol=1e-10)
            or not np.allclose(output[:3, :3], target_pose[:3, :3], atol=1e-10)
        )
        return output, {
            "limited": limited,
            "raw_translation_step_m": raw_translation_norm,
            "applied_translation_step_m": applied_translation_norm,
            "raw_rotation_step_deg": raw_rot_deg,
            "applied_rotation_step_deg": applied_rot_deg,
            "max_xyz_step_m": float(self.config.max_xyz_step_m),
            "max_rot_step_deg": float(self.config.max_rot_step_deg),
            "command_scale": float(self.config.command_scale),
        }


__all__ = [
    "TEMPORARY_DUAL_ARM_SAFETY_EVALUATION_HASH_DOMAIN_V1",
    "TemporaryArmSafetyStageEvidence",
    "TemporaryDualArmSafetyConfig",
    "TemporaryDualArmSafetyEvaluation",
    "TemporaryDualArmSafetyLimiter",
]
