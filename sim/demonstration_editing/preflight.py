"""Offline identity gate for demonstration-conditioned reference sampling.

This module proves only that the generic edit/sampling layer can reproduce a
source demonstration under a zero edit.  It deliberately does not claim IK,
safety, collision, actuator, or task-success replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import DemonstrationEdit, DemonstrationTrajectory
from .editor import DemonstrationReference, apply_demonstration_edit, sample_edited_trajectory


ZERO_EDIT_REPLAY_SCOPE_V1 = "nominal_reference_identity_only_v1"


@dataclass(frozen=True)
class ZeroEditReplayPreflight:
    """Result of replaying every source keyframe through a zero edit."""

    accepted: bool
    scope: str
    source_trajectory_hash: str
    edit_hash: str
    edited_trajectory_hash: str
    sample_id: str
    references: tuple[DemonstrationReference, ...]
    max_position_error_m: float
    max_rotation_error_rad: float
    max_hand_error: float
    tolerance: float

    def __post_init__(self) -> None:
        if self.scope != ZERO_EDIT_REPLAY_SCOPE_V1:
            raise ValueError(f"Unsupported zero-edit replay scope: {self.scope!r}.")
        if not self.references:
            raise ValueError("Zero-edit replay must contain at least one reference.")
        for name in (
            "max_position_error_m",
            "max_rotation_error_rad",
            "max_hand_error",
            "tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)


def _quaternion_angular_error(left: np.ndarray, right: np.ndarray) -> float:
    dot = float(abs(np.dot(left, right)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def preflight_zero_edit_replay(
    trajectory: DemonstrationTrajectory,
    *,
    producer_id: str = "sim-demonstration-zero-edit-preflight",
    sample_id: str = "deterministic-zero-edit",
    tolerance: float = 1e-10,
) -> ZeroEditReplayPreflight:
    """Replay source keyframes through the nominal zero-edit path.

    ``accepted`` means only that the edit and time-sampling layer preserved the
    keyframe pose/hand values within ``tolerance``.  Downstream MuJoCo replay is
    a separate admission and execution boundary.
    """

    if not isinstance(trajectory, DemonstrationTrajectory):
        raise TypeError("trajectory must be a DemonstrationTrajectory.")
    threshold = float(tolerance)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    edit = DemonstrationEdit(
        edit_id=f"zero-edit:{trajectory.semantic_content_hash}",
        producer_id=producer_id,
        sample_id=sample_id,
        translation_m=np.zeros(3, dtype=np.float64),
        rotation_rotvec_rad=np.zeros(3, dtype=np.float64),
        hand_final_delta=np.zeros(trajectory.hand_dim, dtype=np.float64),
        metadata={"preflight_scope": ZERO_EDIT_REPLAY_SCOPE_V1},
    )
    edited = apply_demonstration_edit(trajectory, edit)
    references = tuple(
        sample_edited_trajectory(edited, float(timestamp))
        for timestamp in trajectory.timestamps_sec
    )

    position_errors = tuple(
        float(
            np.linalg.norm(
                reference.tool_pose_xyzquat_xyzw[:3]
                - trajectory.tool_pose_xyzquat_xyzw[index, :3]
            )
        )
        for index, reference in enumerate(references)
    )
    rotation_errors = tuple(
        _quaternion_angular_error(
            reference.tool_pose_xyzquat_xyzw[3:7],
            trajectory.tool_pose_xyzquat_xyzw[index, 3:7],
        )
        for index, reference in enumerate(references)
    )
    hand_errors = tuple(
        float(
            np.max(
                np.abs(
                    reference.hand_target - trajectory.hand_targets[index]
                )
            )
        )
        for index, reference in enumerate(references)
    )
    maxima = (
        max(position_errors, default=0.0),
        max(rotation_errors, default=0.0),
        max(hand_errors, default=0.0),
    )
    return ZeroEditReplayPreflight(
        accepted=all(value <= threshold for value in maxima),
        scope=ZERO_EDIT_REPLAY_SCOPE_V1,
        source_trajectory_hash=trajectory.semantic_content_hash,
        edit_hash=edit.semantic_content_hash,
        edited_trajectory_hash=edited.semantic_content_hash,
        sample_id=edit.sample_id,
        references=references,
        max_position_error_m=maxima[0],
        max_rotation_error_rad=maxima[1],
        max_hand_error=maxima[2],
        tolerance=threshold,
    )


__all__ = [
    "ZERO_EDIT_REPLAY_SCOPE_V1",
    "ZeroEditReplayPreflight",
    "preflight_zero_edit_replay",
]
