"""Pure selection of a collision-safe RM65 all-IK source pair.

Candidate production belongs to a kinematics provider and collision authority
belongs to the canonical planner.  This module only compares already-produced
dual-arm candidates against an admitted source anchor.  It owns no SDK,
planner state, clock, cursor, authority, or dispatch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .constants import SIDES
from .qpos_layout import as_rm65_qpos, validate_qpos_by_side


def _freeze_qpos_mapping(
    values: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    frozen: dict[str, np.ndarray] = {}
    for side in SIDES:
        value = as_rm65_qpos(values[side], label=f"qpos_by_side.{side}")
        value.setflags(write=False)
        frozen[side] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class RM65SourceIKPairSelection:
    """Receipt for one source-rate dual-arm all-IK selection."""

    accepted: bool
    qpos_by_side: Mapping[str, np.ndarray] | None
    candidate_index_by_side: Mapping[str, int]
    candidate_count_by_side: Mapping[str, int]
    collision_rejection_count: int
    squared_joint_distance_rad2: float | None
    reason: str
    rejection_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        qpos = self.qpos_by_side
        if accepted and qpos is None:
            raise ValueError("accepted source IK selection requires qpos_by_side.")
        if not accepted and qpos is not None:
            raise ValueError("rejected source IK selection cannot carry qpos_by_side.")
        if qpos is not None:
            qpos = _freeze_qpos_mapping(qpos)
        indices = {
            str(side): int(index)
            for side, index in dict(self.candidate_index_by_side).items()
        }
        counts = {
            str(side): int(count)
            for side, count in dict(self.candidate_count_by_side).items()
        }
        if set(counts) != set(SIDES) or any(count < 0 for count in counts.values()):
            raise ValueError("candidate_count_by_side must contain non-negative counts.")
        if accepted and set(indices) != set(SIDES):
            raise ValueError("accepted selection requires both candidate indices.")
        rejection_count = int(self.collision_rejection_count)
        if rejection_count < 0:
            raise ValueError("collision_rejection_count must be non-negative.")
        distance = self.squared_joint_distance_rad2
        if distance is not None:
            distance = float(distance)
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    "squared_joint_distance_rad2 must be finite and non-negative."
                )
        rejection_reasons = {
            str(reason): int(count)
            for reason, count in dict(self.rejection_reason_counts).items()
        }
        if any(count < 0 for count in rejection_reasons.values()):
            raise ValueError("rejection reason counts must be non-negative.")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "qpos_by_side", qpos)
        object.__setattr__(
            self,
            "candidate_index_by_side",
            MappingProxyType(indices),
        )
        object.__setattr__(
            self,
            "candidate_count_by_side",
            MappingProxyType(counts),
        )
        object.__setattr__(self, "collision_rejection_count", rejection_count)
        object.__setattr__(self, "squared_joint_distance_rad2", distance)
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "rejection_reason_counts",
            MappingProxyType(rejection_reasons),
        )


def select_nearest_collision_safe_rm65_ik_pair(
    candidate_qpos_by_side: Mapping[str, Sequence[Any]],
    *,
    anchor_qpos_by_side: Mapping[str, Any],
    validate_pair: Callable[[Mapping[str, np.ndarray]], Any],
) -> RM65SourceIKPairSelection:
    """Select the nearest certified dual-arm pair with deterministic ties.

    ``validate_pair`` is the canonical safety boundary.  Its receipt must expose
    a boolean ``safe`` attribute and may expose a string ``reason`` attribute.
    The function never substitutes the anchor when no candidate is accepted.
    """

    if not callable(validate_pair):
        raise TypeError("validate_pair must be callable.")
    anchor = validate_qpos_by_side(anchor_qpos_by_side)
    if set(anchor) != set(SIDES):
        raise ValueError("anchor_qpos_by_side must contain both RM65 arms.")
    candidates: dict[str, tuple[np.ndarray, ...]] = {}
    for side in SIDES:
        raw_values = tuple(candidate_qpos_by_side.get(side, ()))
        candidates[side] = tuple(
            as_rm65_qpos(value, label=f"candidate_qpos_by_side.{side}[{index}]")
            for index, value in enumerate(raw_values)
        )
    counts = {side: len(candidates[side]) for side in SIDES}
    if any(counts[side] == 0 for side in SIDES):
        return RM65SourceIKPairSelection(
            accepted=False,
            qpos_by_side=None,
            candidate_index_by_side={},
            candidate_count_by_side=counts,
            collision_rejection_count=0,
            squared_joint_distance_rad2=None,
            reason="missing_side_candidates",
            rejection_reason_counts={},
        )

    best_pair: dict[str, np.ndarray] | None = None
    best_indices: dict[str, int] = {}
    best_cost = math.inf
    collision_rejections = 0
    rejection_reason_counts: dict[str, int] = {}
    for left_index, left in enumerate(candidates["left"]):
        for right_index, right in enumerate(candidates["right"]):
            pair = {
                "left": left.copy(),
                "right": right.copy(),
            }
            validation = validate_pair(pair)
            if not bool(getattr(validation, "safe", False)):
                collision_rejections += 1
                reason = str(
                    getattr(validation, "reason", None)
                    or "canonical_validation_rejected"
                )
                rejection_reason_counts[reason] = (
                    rejection_reason_counts.get(reason, 0) + 1
                )
                continue
            cost = sum(
                float(
                    np.dot(
                        pair[side] - anchor[side],
                        pair[side] - anchor[side],
                    )
                )
                for side in SIDES
            )
            # Strict comparison preserves enumeration order as a deterministic
            # tie-break without adding an unrelated branch heuristic.
            if cost < best_cost:
                best_pair = pair
                best_indices = {
                    "left": int(left_index),
                    "right": int(right_index),
                }
                best_cost = cost

    if best_pair is None:
        return RM65SourceIKPairSelection(
            accepted=False,
            qpos_by_side=None,
            candidate_index_by_side={},
            candidate_count_by_side=counts,
            collision_rejection_count=collision_rejections,
            squared_joint_distance_rad2=None,
            reason="no_collision_safe_pair",
            rejection_reason_counts=rejection_reason_counts,
        )
    return RM65SourceIKPairSelection(
        accepted=True,
        qpos_by_side=best_pair,
        candidate_index_by_side=best_indices,
        candidate_count_by_side=counts,
        collision_rejection_count=collision_rejections,
        squared_joint_distance_rad2=best_cost,
        reason="nearest_collision_safe_pair",
        rejection_reason_counts=rejection_reason_counts,
    )


__all__ = [
    "RM65SourceIKPairSelection",
    "select_nearest_collision_safe_rm65_ik_pair",
]
