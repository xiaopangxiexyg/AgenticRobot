from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


_HARD_STOP_COLLISION_STATUSES = frozenset(
    {"hold_last_safe", "jump_gate_hold", "unsafe_candidate", "failed"}
)
_HARD_STOP_JUMP_STATUSES = frozenset({"hold_entered", "holding"})
_DEGRADED_JUMP_STATUSES = frozenset({"warning", "recovering", "released"})
_NORMAL_SINGULARITY_STATUSES = frozenset(
    {
        "pass",
        "disabled",
        "not_checked",
        "allow_current_singular",
        "allow_position_wrist_singular",
        "allow_position_elbow_singular",
        "allow_position_shoulder_singular",
    }
)
_REAL_AGENT_MOTION_METADATA_KEY = "real_agent_motion_operation"


@dataclass(frozen=True)
class PositionTrajectoryPolicy:
    mode: str
    speed_scale: float
    quick_stop: bool
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return ",".join(self.reasons) if self.reasons else "normal"


def metadata_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def final_follow_qpos_target(
    planned_qpos: np.ndarray,
    accepted_qpos: np.ndarray | None,
    *,
    alpha: float,
    bypass: bool,
) -> tuple[np.ndarray, bool]:
    """Apply the final transport EMA around the last accepted qpos."""

    planned = np.asarray(planned_qpos, dtype=float).reshape(6)
    if bypass or float(alpha) >= 1.0 or accepted_qpos is None:
        return planned.copy(), False
    accepted = np.asarray(accepted_qpos, dtype=float).reshape(6)
    output = accepted + float(alpha) * (planned - accepted)
    return output, not np.allclose(output, planned, rtol=0.0, atol=1e-12)


def classify_position_trajectory_policy(
    command_metadata: Mapping[str, object],
    singularity_metadata: Mapping[str, object],
    *,
    degraded_speed_scale: float,
) -> PositionTrajectoryPolicy:
    """Classify an already validated qref for sim/real position execution."""

    hard_stop_reasons: list[str] = []
    degraded_reasons: list[str] = []
    speed_limit_reasons: list[str] = []
    requested_speed_scale = 1.0
    agent_operation = metadata_mapping(
        command_metadata.get(_REAL_AGENT_MOTION_METADATA_KEY)
    )
    if "speed_scale" in agent_operation:
        value = agent_operation["speed_scale"]
        try:
            requested_speed_scale = float(value)
        except (TypeError, ValueError):
            hard_stop_reasons.append("agent_speed_scale_invalid")
        else:
            if (
                isinstance(value, bool)
                or not math.isfinite(requested_speed_scale)
                or not 0.0 < requested_speed_scale <= 1.0
            ):
                hard_stop_reasons.append("agent_speed_scale_invalid")
            elif requested_speed_scale < 1.0:
                speed_limit_reasons.append("agent_speed_scale")

    collision_status = str(command_metadata.get("collision_avoidance_status") or "").strip()
    if collision_status in _HARD_STOP_COLLISION_STATUSES:
        hard_stop_reasons.append(f"collision:{collision_status}")
    elif collision_status == "boundary_active":
        degraded_reasons.append("collision:boundary_active")
    elif collision_status and collision_status != "tracking":
        degraded_reasons.append(f"collision:{collision_status}")
    fallback_reason = str(
        command_metadata.get("collision_avoidance_fallback_reason") or ""
    ).strip()
    if fallback_reason and not hard_stop_reasons:
        degraded_reasons.append(f"collision_fallback:{fallback_reason}")

    jump_status = str(command_metadata.get("ik_jump_gate_status") or "").strip()
    if jump_status in _HARD_STOP_JUMP_STATUSES:
        hard_stop_reasons.append(f"ik_jump:{jump_status}")
    elif jump_status in _DEGRADED_JUMP_STATUSES:
        degraded_reasons.append(f"ik_jump:{jump_status}")

    singularity_status = str(
        singularity_metadata.get("singularity_gate_status") or ""
    ).strip()
    if singularity_status and singularity_status not in _NORMAL_SINGULARITY_STATUSES:
        degraded_reasons.append(f"singularity:{singularity_status}")

    execution_step = metadata_mapping(command_metadata.get("execution_step_metadata"))
    if bool(execution_step.get("step_limited", False)):
        degraded_reasons.append("execution_step_limited")
    if bool(command_metadata.get("ik_time_budget_exceeded", False)):
        degraded_reasons.append("ik_time_budget_exceeded")

    pose_error = metadata_mapping(command_metadata.get("ik_solve_pose_error"))
    if pose_error:
        try:
            translation_m = float(pose_error.get("translation_m", 0.0) or 0.0)
            rotation_deg = float(pose_error.get("rotation_deg", 0.0) or 0.0)
        except (TypeError, ValueError):
            hard_stop_reasons.append("ik_pose_error_invalid")
        else:
            if not math.isfinite(translation_m) or not math.isfinite(rotation_deg):
                hard_stop_reasons.append("ik_pose_error_nonfinite")

    def unique(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(value) for value in values if str(value)))

    if hard_stop_reasons:
        return PositionTrajectoryPolicy(
            mode="quick_stop",
            speed_scale=0.0,
            quick_stop=True,
            reasons=unique(hard_stop_reasons),
        )
    if degraded_reasons:
        return PositionTrajectoryPolicy(
            mode="degraded",
            speed_scale=min(float(degraded_speed_scale), requested_speed_scale),
            quick_stop=False,
            reasons=unique((*degraded_reasons, *speed_limit_reasons)),
        )
    if speed_limit_reasons:
        return PositionTrajectoryPolicy(
            mode="tracking",
            speed_scale=requested_speed_scale,
            quick_stop=False,
            reasons=unique(speed_limit_reasons),
        )
    return PositionTrajectoryPolicy(
        mode="tracking",
        speed_scale=1.0,
        quick_stop=False,
        reasons=(),
    )


__all__ = [
    "PositionTrajectoryPolicy",
    "classify_position_trajectory_policy",
    "final_follow_qpos_target",
    "metadata_mapping",
]
