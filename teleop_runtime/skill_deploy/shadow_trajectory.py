"""No-device adapters for simulation-first real Skill trajectory workflows.

The caller must already own and read every lower hardware source.  This module
only combines those typed values into the backend-neutral shadow contracts; it
does not connect hardware, refresh state, dispatch, HOLD, or grant authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    ShadowExecutionIdentity,
)

from .control_runtime import RealSkillMotionObservationCut


def capture_authoritative_shadow_state(
    *,
    arm_observation_cut: RealSkillMotionObservationCut,
    gripper_close_by_side: Mapping[str, float],
    gripper_measured_at_by_side: Mapping[str, float],
    gripper_available_at: float,
    gripper_evidence_refs: Sequence[str],
    identity: ShadowExecutionIdentity,
    gripper_state_assumed: bool = False,
) -> AuthoritativeRobotStateSnapshot:
    """Combine already-read arm/gripper measurements into one causal cut.

    Assumed, commanded, cached-without-evidence, or disconnected fallback
    gripper values are intentionally rejected because their geometry can alter
    both the initial MuJoCo collision state and the planned path clearance.
    """

    if not isinstance(arm_observation_cut, RealSkillMotionObservationCut):
        raise TypeError("arm_observation_cut must be RealSkillMotionObservationCut.")
    if not isinstance(identity, ShadowExecutionIdentity):
        raise TypeError("identity must be ShadowExecutionIdentity.")
    if not isinstance(gripper_state_assumed, bool):
        raise TypeError("gripper_state_assumed must be boolean.")
    if gripper_state_assumed:
        raise ValueError("assumed gripper state cannot initialize shadow execution.")
    if isinstance(gripper_available_at, bool) or not isinstance(
        gripper_available_at, (int, float)
    ):
        raise TypeError("gripper_available_at must be numeric.")
    gripper_available = float(gripper_available_at)
    if not math.isfinite(gripper_available):
        raise ValueError("gripper_available_at must be finite.")
    gripper_times = dict(gripper_measured_at_by_side)
    if set(gripper_times) != {"left", "right"}:
        raise ValueError("gripper_measured_at_by_side must contain left and right.")
    for side, value in gripper_times.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"gripper_measured_at_by_side[{side!r}] must be numeric.")
        measured_at = float(value)
        if not math.isfinite(measured_at):
            raise ValueError(
                f"gripper_measured_at_by_side[{side!r}] must be finite."
            )
        if gripper_available < measured_at:
            raise ValueError(
                "gripper_available_at cannot precede a gripper measurement."
            )
    # The combined cut only becomes available after both independently owned
    # lower reads have completed.  Each component retains its own measured-at
    # time for later freshness evaluation.
    return AuthoritativeRobotStateSnapshot(
        qpos_by_side=arm_observation_cut.qpos_by_side,
        gripper_close_by_side=gripper_close_by_side,
        arm_measured_at=arm_observation_cut.measured_at,
        gripper_measured_at_by_side=gripper_times,
        available_at=max(arm_observation_cut.available_at, gripper_available),
        arm_evidence_refs=arm_observation_cut.evidence_refs,
        gripper_evidence_refs=tuple(gripper_evidence_refs),
        identity=identity,
    )


__all__ = ["capture_authoritative_shadow_state"]
