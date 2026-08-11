"""Backend-neutral position-plan splice contracts and state policy.

The deployment owner detects source-plan boundaries because it owns policy
chunk lineage and RTC plan replacement.  This module owns the motion meaning
of that fact: whether an admitted trajectory may continue from its committed
dynamic state or must restart from an acknowledged Hold state.

The contracts carry no queue, cursor, collision, dispatch, or device
authority.  In particular, a source boundary is not itself permission to
rewrite already committed motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from robot_motion.transforms import (
    interpolate_pose,
    parse_pose_matrix,
    rotation_angle_rad,
    translation_distance,
    validate_pose_matrix,
)

from .horizon_governor import JointHorizonState


POSITION_PLAN_RELATION_STREAM_START = "stream_start"
POSITION_PLAN_RELATION_CONTIGUOUS = "same_plan_contiguous"
POSITION_PLAN_RELATION_SOURCE_GAP = "same_plan_source_gap"
POSITION_PLAN_RELATION_REPLACEMENT = "source_plan_replacement"
POSITION_PLAN_RELATIONS = (
    POSITION_PLAN_RELATION_STREAM_START,
    POSITION_PLAN_RELATION_CONTIGUOUS,
    POSITION_PLAN_RELATION_SOURCE_GAP,
    POSITION_PLAN_RELATION_REPLACEMENT,
)

POSITION_PLAN_GAP_NONE = "none"
POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED = "hold_acknowledged"
POSITION_PLAN_GAPS = (
    POSITION_PLAN_GAP_NONE,
    POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED,
)

POSITION_PLAN_SPLICE_STREAM_START = "stream_start_from_reset_state"
POSITION_PLAN_SPLICE_CONTINUE = "continue_committed_dynamics"
POSITION_PLAN_SPLICE_LIVE = "live_splice_preserve_committed_dynamics"
POSITION_PLAN_SPLICE_HOLD_RESTART = "hold_restart_from_zero_dynamics"

POSITION_PLAN_SPLICE_DYNAMICS_OFF = "off"
POSITION_PLAN_SPLICE_DYNAMICS_CONSERVATIVE_V1 = "conservative_v1"
POSITION_PLAN_SPLICE_DYNAMICS_PROFILE_CHOICES = (
    POSITION_PLAN_SPLICE_DYNAMICS_OFF,
    POSITION_PLAN_SPLICE_DYNAMICS_CONSERVATIVE_V1,
)
POSITION_PLAN_SPLICE_CONSERVATIVE_V1_JERK_DELTA_REGULARIZATION = 1e-4


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


@dataclass(frozen=True)
class PositionSourcePlanCut:
    """Deployment-observed source boundary without motion-policy authority."""

    cut_id: str
    current_source_plan_id: str
    current_source_step: int
    relation: str
    execution_gap: str
    detected_at: float
    previous_source_plan_id: str | None = None
    previous_source_step: int | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cut_id = _required(self.cut_id, name="cut_id")
        current_plan = _required(
            self.current_source_plan_id,
            name="current_source_plan_id",
        )
        current_step = _nonnegative_integer(
            self.current_source_step,
            name="current_source_step",
        )
        relation = str(self.relation).strip()
        if relation not in POSITION_PLAN_RELATIONS:
            raise ValueError(f"unsupported position source-plan relation {relation!r}.")
        gap = str(self.execution_gap).strip()
        if gap not in POSITION_PLAN_GAPS:
            raise ValueError(f"unsupported position execution gap {gap!r}.")
        detected_at = float(self.detected_at)
        if not math.isfinite(detected_at):
            raise ValueError("detected_at must be finite.")

        previous_plan = self.previous_source_plan_id
        previous_step = self.previous_source_step
        if (previous_plan is None) != (previous_step is None):
            raise ValueError(
                "previous_source_plan_id and previous_source_step must appear together."
            )
        if previous_plan is not None:
            previous_plan = _required(
                previous_plan,
                name="previous_source_plan_id",
            )
            previous_step = _nonnegative_integer(
                previous_step,
                name="previous_source_step",
            )
        if relation == POSITION_PLAN_RELATION_STREAM_START:
            if previous_plan is not None:
                raise ValueError("stream_start cannot carry a previous source plan.")
        elif previous_plan is None:
            raise ValueError(f"{relation} requires previous source-plan lineage.")
        if relation in {
            POSITION_PLAN_RELATION_CONTIGUOUS,
            POSITION_PLAN_RELATION_SOURCE_GAP,
        } and previous_plan != current_plan:
            raise ValueError(f"{relation} requires the same source plan.")
        if relation == POSITION_PLAN_RELATION_REPLACEMENT and (
            previous_plan == current_plan
        ):
            raise ValueError("source_plan_replacement requires a new source plan.")
        if relation == POSITION_PLAN_RELATION_CONTIGUOUS and (
            current_step != int(previous_step) + 1
        ):
            raise ValueError(
                "same_plan_contiguous requires the immediate next source step."
            )
        if relation == POSITION_PLAN_RELATION_SOURCE_GAP and (
            current_step == int(previous_step) + 1
        ):
            raise ValueError("same_plan_source_gap cannot be contiguous.")

        object.__setattr__(self, "cut_id", cut_id)
        object.__setattr__(self, "current_source_plan_id", current_plan)
        object.__setattr__(self, "current_source_step", current_step)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "execution_gap", gap)
        object.__setattr__(self, "detected_at", detected_at)
        object.__setattr__(self, "previous_source_plan_id", previous_plan)
        object.__setattr__(self, "previous_source_step", previous_step)
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": "robot_motion.position_source_plan_cut/v1",
            "cut_id": self.cut_id,
            "current_source_plan_id": self.current_source_plan_id,
            "current_source_step": self.current_source_step,
            "previous_source_plan_id": self.previous_source_plan_id,
            "previous_source_step": self.previous_source_step,
            "relation": self.relation,
            "execution_gap": self.execution_gap,
            "detected_at": self.detected_at,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class PositionPlanSpliceDecision:
    """Robot-motion interpretation of one deployment-observed plan cut."""

    cut: PositionSourcePlanCut
    mode: str
    preserve_committed_dynamics: bool
    require_zero_dynamic_restart: bool
    mutable_future_replacement: bool

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": "robot_motion.position_plan_splice_decision/v1",
            "mode": self.mode,
            "preserve_committed_dynamics": self.preserve_committed_dynamics,
            "require_zero_dynamic_restart": self.require_zero_dynamic_restart,
            "mutable_future_replacement": self.mutable_future_replacement,
            "source_cut": self.cut.to_metadata(),
        }


@dataclass(frozen=True)
class PositionPlanSpliceDynamicsConfig:
    """Default-off motion tuning applied after an observed source-plan cut.

    Deployment reports an acknowledged Hold boundary; robot motion decides
    whether the first post-Hold jerk command receives an additional
    continuity cost.  A zero weight is the exact legacy rollback.
    """

    hold_restart_initial_jerk_delta_regularization: float = 0.0
    hold_restart_jerk_delta_regularization: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "hold_restart_initial_jerk_delta_regularization",
            "hold_restart_jerk_delta_regularization",
        ):
            weight = float(getattr(self, name))
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, weight)


def resolve_position_plan_splice_dynamics_config(
    profile: str,
) -> PositionPlanSpliceDynamicsConfig:
    """Resolve a bounded, versioned post-Hold splice dynamics profile.

    ``off`` is the exact legacy behavior.  ``conservative_v1`` adds only the
    offline-validated within-horizon jerk-delta cost; it does not change the
    initial state, IK candidates, collision admission, or source timing.
    """

    normalized = str(profile).strip().lower()
    if normalized == POSITION_PLAN_SPLICE_DYNAMICS_OFF:
        return PositionPlanSpliceDynamicsConfig()
    if normalized == POSITION_PLAN_SPLICE_DYNAMICS_CONSERVATIVE_V1:
        return PositionPlanSpliceDynamicsConfig(
            hold_restart_jerk_delta_regularization=(
                POSITION_PLAN_SPLICE_CONSERVATIVE_V1_JERK_DELTA_REGULARIZATION
            )
        )
    raise ValueError(
        "unsupported position plan-splice dynamics profile "
        f"{profile!r}; expected one of "
        f"{POSITION_PLAN_SPLICE_DYNAMICS_PROFILE_CHOICES!r}."
    )


@dataclass(frozen=True)
class PositionStreamStartAcquisitionConfig:
    """Default-off bounds for acquiring a discontinuous first Pose target.

    The deployment owner identifies the first source cut.  This configuration
    owns only the backend-neutral motion subdivision used to approach that
    target from the measured Pose anchor.  It does not consume policy steps,
    select IK branches, dispatch commands, or authorize hardware motion.
    """

    enabled: bool = False
    max_translation_step_m: float = 0.005
    max_rotation_step_rad: float = math.radians(0.5)
    max_source_intervals: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        for name in ("max_translation_step_m", "max_rotation_step_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)
        intervals = _nonnegative_integer(
            self.max_source_intervals,
            name="max_source_intervals",
        )
        if intervals <= 0:
            raise ValueError("max_source_intervals must be positive.")
        object.__setattr__(self, "max_source_intervals", intervals)


@dataclass(frozen=True)
class PositionStreamStartAcquisition:
    """A measured-anchor prelude preceding the first source-owned target."""

    active: bool
    source_intervals: int
    prelude_pose_by_side: tuple[Mapping[str, np.ndarray], ...]
    max_translation_distance_m: float
    max_rotation_distance_rad: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema": "robot_motion.position_stream_start_acquisition/v1",
            "active": bool(self.active),
            "source_intervals": int(self.source_intervals),
            "prelude_source_count": len(self.prelude_pose_by_side),
            "max_translation_distance_m": float(
                self.max_translation_distance_m
            ),
            "max_rotation_distance_rad": float(self.max_rotation_distance_rad),
        }


def build_position_stream_start_acquisition(
    start_pose_by_side: Mapping[str, Any],
    target_pose_by_side: Mapping[str, Any],
    *,
    config: PositionStreamStartAcquisitionConfig,
) -> PositionStreamStartAcquisition:
    """Subdivide one first-target discontinuity without inventing future input.

    The returned prelude contains the measured anchor at alpha zero followed by
    intermediate shortest-path SE(3) samples.  The source-owned target itself
    is deliberately excluded and remains the final endpoint staged by its
    caller.  Consequently a prelude of ``N`` entries delays that endpoint by
    exactly ``N`` source periods while preserving all later source spacing.
    """

    if not isinstance(config, PositionStreamStartAcquisitionConfig):
        raise TypeError(
            "config must be a PositionStreamStartAcquisitionConfig."
        )
    start_values = {
        str(side): value for side, value in start_pose_by_side.items()
    }
    target_values = {
        str(side): value for side, value in target_pose_by_side.items()
    }
    start_sides = set(start_values)
    target_sides = set(target_values)
    if not start_sides or start_sides != target_sides:
        raise ValueError(
            "start_pose_by_side and target_pose_by_side must contain the same "
            "non-empty sides."
        )
    starts: dict[str, np.ndarray] = {}
    targets: dict[str, np.ndarray] = {}
    for side in sorted(start_sides):
        for stage, value, destination in (
            ("start", start_values[side], starts),
            ("target", target_values[side], targets),
        ):
            valid, reason = validate_pose_matrix(value)
            if not valid:
                raise ValueError(
                    f"{side} {stage} stream-start Pose is invalid: {reason}"
                )
            destination[side] = parse_pose_matrix(value)
    translation_by_side = {
        side: translation_distance(starts[side], targets[side])
        for side in starts
    }
    rotation_by_side = {
        side: rotation_angle_rad(starts[side], targets[side])
        for side in starts
    }
    max_translation = max(translation_by_side.values(), default=0.0)
    max_rotation = max(rotation_by_side.values(), default=0.0)
    required_intervals = max(
        1,
        int(math.ceil(max_translation / config.max_translation_step_m)),
        int(math.ceil(max_rotation / config.max_rotation_step_rad)),
    )
    active = bool(config.enabled and required_intervals > 1)
    if not active:
        return PositionStreamStartAcquisition(
            active=False,
            source_intervals=0,
            prelude_pose_by_side=(),
            max_translation_distance_m=max_translation,
            max_rotation_distance_rad=max_rotation,
        )
    if required_intervals > config.max_source_intervals:
        raise RuntimeError(
            "position stream-start acquisition exceeds its bounded source "
            f"horizon: required={required_intervals} "
            f"limit={config.max_source_intervals}."
        )
    prelude: list[Mapping[str, np.ndarray]] = []
    for interval in range(required_intervals):
        alpha = float(interval) / float(required_intervals)
        prelude.append(
            MappingProxyType(
                {
                    side: interpolate_pose(
                        starts[side],
                        targets[side],
                        alpha,
                    )
                    for side in starts
                }
            )
        )
    return PositionStreamStartAcquisition(
        active=True,
        source_intervals=required_intervals,
        prelude_pose_by_side=tuple(prelude),
        max_translation_distance_m=max_translation,
        max_rotation_distance_rad=max_rotation,
    )


def initial_jerk_regularization_for_position_plan_splice(
    decision: PositionPlanSpliceDecision | None,
    *,
    config: PositionPlanSpliceDynamicsConfig,
) -> float:
    """Return a first-jerk weight only for acknowledged Hold restarts."""

    if decision is not None and not isinstance(
        decision,
        PositionPlanSpliceDecision,
    ):
        raise TypeError("decision must be a PositionPlanSpliceDecision or None.")
    if not isinstance(config, PositionPlanSpliceDynamicsConfig):
        raise TypeError("config must be a PositionPlanSpliceDynamicsConfig.")
    if decision is None or decision.mode != POSITION_PLAN_SPLICE_HOLD_RESTART:
        return 0.0
    return float(config.hold_restart_initial_jerk_delta_regularization)


def jerk_delta_regularization_for_position_plan_splice(
    decision: PositionPlanSpliceDecision | None,
    *,
    config: PositionPlanSpliceDynamicsConfig,
) -> float:
    """Return an intra-horizon jerk-delta weight only after a Hold."""

    if decision is not None and not isinstance(
        decision,
        PositionPlanSpliceDecision,
    ):
        raise TypeError("decision must be a PositionPlanSpliceDecision or None.")
    if not isinstance(config, PositionPlanSpliceDynamicsConfig):
        raise TypeError("config must be a PositionPlanSpliceDynamicsConfig.")
    if decision is None or decision.mode != POSITION_PLAN_SPLICE_HOLD_RESTART:
        return 0.0
    return float(config.hold_restart_jerk_delta_regularization)


def requires_clock_rebase_and_tail_drain_for_position_plan_splice(
    decision: PositionPlanSpliceDecision | None,
) -> bool:
    """Identify a stopped-plan restart that may use a fresh source clock.

    Live RTC replacements and an ordinary stream start preserve the existing
    clock contract.  An explicitly acknowledged Hold may instead start only
    after its new plan is prepared.  Stream-start acquisition is a separate,
    default-off executor contract and requests its own rebase only when an
    actual acquisition prelude was inserted.
    """

    if decision is not None and not isinstance(
        decision,
        PositionPlanSpliceDecision,
    ):
        raise TypeError("decision must be a PositionPlanSpliceDecision or None.")
    return bool(
        decision is not None
        and decision.mode == POSITION_PLAN_SPLICE_HOLD_RESTART
    )


def resolve_position_plan_splice(
    cut: PositionSourcePlanCut,
) -> PositionPlanSpliceDecision:
    """Resolve motion continuity without changing any source-owned plan."""

    if not isinstance(cut, PositionSourcePlanCut):
        raise TypeError("cut must be a PositionSourcePlanCut.")
    if cut.execution_gap == POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED:
        return PositionPlanSpliceDecision(
            cut=cut,
            mode=POSITION_PLAN_SPLICE_HOLD_RESTART,
            preserve_committed_dynamics=False,
            require_zero_dynamic_restart=True,
            mutable_future_replacement=False,
        )
    if cut.relation == POSITION_PLAN_RELATION_STREAM_START:
        return PositionPlanSpliceDecision(
            cut=cut,
            mode=POSITION_PLAN_SPLICE_STREAM_START,
            preserve_committed_dynamics=False,
            require_zero_dynamic_restart=True,
            mutable_future_replacement=False,
        )
    if cut.relation == POSITION_PLAN_RELATION_CONTIGUOUS:
        return PositionPlanSpliceDecision(
            cut=cut,
            mode=POSITION_PLAN_SPLICE_CONTINUE,
            preserve_committed_dynamics=True,
            require_zero_dynamic_restart=False,
            mutable_future_replacement=False,
        )
    return PositionPlanSpliceDecision(
        cut=cut,
        mode=POSITION_PLAN_SPLICE_LIVE,
        preserve_committed_dynamics=True,
        require_zero_dynamic_restart=False,
        mutable_future_replacement=(
            cut.relation == POSITION_PLAN_RELATION_REPLACEMENT
        ),
    )


def initial_state_for_position_plan_splice(
    decision: PositionPlanSpliceDecision,
    *,
    committed_state: JointHorizonState,
    zero_tolerance: float = 1e-9,
) -> JointHorizonState:
    """Validate and return the authoritative state for the next source cut.

    Live replacement keeps the exact committed dynamic state.  A stream start
    or acknowledged Hold restart is accepted only after the owning runtime has
    reset its planner to zero velocity and acceleration; this shared layer does
    not silently erase a non-zero state.
    """

    if not isinstance(decision, PositionPlanSpliceDecision):
        raise TypeError("decision must be a PositionPlanSpliceDecision.")
    if not isinstance(committed_state, JointHorizonState):
        raise TypeError("committed_state must be a JointHorizonState.")
    tolerance = float(zero_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")
    if decision.require_zero_dynamic_restart and (
        np.max(np.abs(committed_state.qd_rad_s)) > tolerance
        or np.max(np.abs(committed_state.qdd_rad_s2)) > tolerance
    ):
        raise RuntimeError(
            "position plan restart requires an acknowledged reset to zero "
            "velocity and acceleration."
        )
    return committed_state


__all__ = [
    "POSITION_PLAN_GAP_HOLD_ACKNOWLEDGED",
    "POSITION_PLAN_GAP_NONE",
    "POSITION_PLAN_RELATION_CONTIGUOUS",
    "POSITION_PLAN_RELATION_REPLACEMENT",
    "POSITION_PLAN_RELATION_SOURCE_GAP",
    "POSITION_PLAN_RELATION_STREAM_START",
    "POSITION_PLAN_SPLICE_CONTINUE",
    "POSITION_PLAN_SPLICE_HOLD_RESTART",
    "POSITION_PLAN_SPLICE_LIVE",
    "POSITION_PLAN_SPLICE_STREAM_START",
    "PositionStreamStartAcquisition",
    "PositionStreamStartAcquisitionConfig",
    "PositionPlanSpliceDynamicsConfig",
    "PositionPlanSpliceDecision",
    "PositionSourcePlanCut",
    "build_position_stream_start_acquisition",
    "initial_jerk_regularization_for_position_plan_splice",
    "jerk_delta_regularization_for_position_plan_splice",
    "requires_clock_rebase_and_tail_drain_for_position_plan_splice",
    "initial_state_for_position_plan_splice",
    "resolve_position_plan_splice",
]
