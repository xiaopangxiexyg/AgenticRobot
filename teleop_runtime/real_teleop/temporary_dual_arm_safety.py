"""Temporary dual-arm output limiter used during real-hardware bring-up.

This is a move-only extraction from the real teleop entrypoint.  It preserves
the command, metadata invalidation, and limiting semantics; policy changes
belong in a separate safety change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from robot_motion.runtime import EffectorCommand, TeleopCommand

from ..transforms import as_pose_matrix


@dataclass(frozen=True)
class TemporaryDualArmSafetyConfig:
    enabled: bool = True
    active_sides: tuple[str, ...] = ("left", "right")
    max_xyz_step_m: float = 0.03
    max_rot_step_deg: float = 8.0
    max_joint_step_deg: float = 6.0
    command_scale: float = 1.0


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

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and set(self.config.active_sides) == {"left", "right"})

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
            return command
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
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
                next_command, effector_summary = self._filter_arm_command(next_command)
                if effector_summary:
                    summary["effectors"][effector_name] = effector_summary
                    summary["limited"] = bool(
                        summary["limited"] or effector_summary.get("limited")
                    )
            filtered.commands[effector_name] = next_command
        filtered.metadata["temporary_dual_arm_safety"] = summary
        return filtered

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


__all__ = ["TemporaryDualArmSafetyConfig", "TemporaryDualArmSafetyLimiter"]
