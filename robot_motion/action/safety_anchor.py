"""Pure lifecycle contracts for canonical keyframe-safety anchors.

The contracts in this module deliberately do not observe a frontend, clock,
planner, dispatcher, or backend.  A caller must provide explicit measured-state
or execution-target admission evidence before an anchor can be rebased or
committed.  This keeps legacy comparison and future authoritative ownership
separate while giving both paths one deterministic state model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from dataset_contracts.canonical_encoding import canonical_content_hash

from .keyframe_safety import CanonicalActionSafetyAnchor
from .hand_spec import hand_action_specs_from_metadata
from .proposal import freeze_robot_action, robot_action_semantic_payload
from .schema import RobotAction


SAFETY_ANCHOR_POLICY_HASH_DOMAIN_V1 = "dexteleop.safety_anchor_policy.v1"
SAFETY_ANCHOR_STATE_HASH_DOMAIN_V1 = "dexteleop.safety_anchor_state.v1"
SAFETY_ANCHOR_TRANSITION_HASH_DOMAIN_V1 = "dexteleop.safety_anchor_transition.v1"
SAFETY_ANCHOR_COMMIT_EVIDENCE_HASH_DOMAIN_V1 = (
    "dexteleop.safety_anchor_commit_evidence.v1"
)
SAFETY_EVALUATION_INPUT_HASH_DOMAIN_V1 = "dexteleop.safety_evaluation_input.v1"
MEASURED_DIVERGENCE_GUARD_HASH_DOMAIN_V1 = (
    "dexteleop.measured_divergence_guard.v1"
)
SAFETY_ANCHOR_ACTION_HASH_DOMAIN_V1 = "dexteleop.safety_anchor_action.v1"

SafetyAnchorStatus = Literal[
    "uninitialized",
    "active",
    "rebase_required",
    "invalidated",
]
SafetyAnchorTransitionKind = Literal[
    "initialize_from_measured",
    "commit_safe_target",
    "invalidate_on_hold",
    "invalidate_on_fault",
    "invalidate_on_backend_reject",
    "invalidate_on_execution_epoch_change",
    "invalidate_on_group_cancel",
    "rebase_after_recovery",
]

_ANCHOR_STATUSES = frozenset(
    {"uninitialized", "active", "rebase_required", "invalidated"}
)
_TRANSITION_KINDS = frozenset(
    {
        "initialize_from_measured",
        "commit_safe_target",
        "invalidate_on_hold",
        "invalidate_on_fault",
        "invalidate_on_backend_reject",
        "invalidate_on_execution_epoch_change",
        "invalidate_on_group_cancel",
        "rebase_after_recovery",
    }
)
_COMMIT_EVIDENCE_KINDS = frozenset(
    {
        "execution_target_admitted",
        "reference_segment_endpoint_effective",
        "direct_backend_command_accepted",
    }
)
_VIOLATION_POLICIES = frozenset({"hold_group", "hold_all", "fault"})


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def safety_anchor_action_hash(action: RobotAction) -> str:
    """Hash one canonical action for anchor commit/snapshot evidence."""

    frozen = freeze_robot_action(action)
    return canonical_content_hash(
        robot_action_semantic_payload(frozen),
        domain=SAFETY_ANCHOR_ACTION_HASH_DOMAIN_V1,
    )


@dataclass(frozen=True)
class SafetyAnchorPolicy:
    """Versioned target lifecycle policy, separate from numeric limits."""

    policy_id: str
    max_measurement_age_sec: float
    normal_anchor_kind: str = "previous_safe_committed"
    recovery_anchor_kind: str = "measured_state"
    on_missing_normal_anchor: str = "rebase_required"
    on_stale_anchor: str = "hold_group"
    invalidate_on_execution_epoch_change: bool = True
    preserve_across_source_handoff: bool = True
    max_anchor_age_sec: float | None = None

    def __post_init__(self) -> None:
        _required_text(self.policy_id, name="policy_id")
        if self.normal_anchor_kind != "previous_safe_committed":
            raise ValueError(
                "Safety anchor policy v1 fixes normal_anchor_kind to "
                "previous_safe_committed."
            )
        if self.recovery_anchor_kind != "measured_state":
            raise ValueError(
                "Safety anchor policy v1 fixes recovery_anchor_kind to measured_state."
            )
        if self.on_missing_normal_anchor != "rebase_required":
            raise ValueError(
                "Safety anchor policy v1 requires missing anchors to request rebase."
            )
        if self.on_stale_anchor not in {"hold_group", "hold_all", "fault"}:
            raise ValueError("Unsupported stale-anchor policy.")
        if self.max_measurement_age_sec is None:
            raise ValueError("Safety anchor policy requires max_measurement_age_sec.")
        for name in ("max_anchor_age_sec", "max_measurement_age_sec"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive when provided.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "policy_id": self.policy_id,
                "normal_anchor_kind": self.normal_anchor_kind,
                "recovery_anchor_kind": self.recovery_anchor_kind,
                "on_missing_normal_anchor": self.on_missing_normal_anchor,
                "on_stale_anchor": self.on_stale_anchor,
                "invalidate_on_execution_epoch_change": bool(
                    self.invalidate_on_execution_epoch_change
                ),
                "preserve_across_source_handoff": bool(
                    self.preserve_across_source_handoff
                ),
                "max_anchor_age_sec": self.max_anchor_age_sec,
                "max_measurement_age_sec": self.max_measurement_age_sec,
            },
            domain=SAFETY_ANCHOR_POLICY_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class MeasuredDivergenceGuardSpec:
    """Typed configuration only; activation requires validated thresholds."""

    guard_id: str
    enabled: bool = False
    max_translation_error_m: float | None = None
    max_rotation_error_deg: float | None = None
    max_joint_error_rad: float | None = None
    max_measurement_age_sec: float | None = None
    violation_policy: str = "hold_group"

    def __post_init__(self) -> None:
        _required_text(self.guard_id, name="guard_id")
        if self.violation_policy not in _VIOLATION_POLICIES:
            raise ValueError("Unsupported measured-divergence violation policy.")
        thresholds = (
            self.max_translation_error_m,
            self.max_rotation_error_deg,
            self.max_joint_error_rad,
        )
        for name in (
            "max_translation_error_m",
            "max_rotation_error_deg",
            "max_joint_error_rad",
            "max_measurement_age_sec",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive when provided.")
        if self.enabled and not any(value is not None for value in thresholds):
            raise ValueError("An enabled measured-divergence guard needs a threshold.")
        if self.enabled and self.max_measurement_age_sec is None:
            raise ValueError(
                "An enabled measured-divergence guard needs max_measurement_age_sec."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "guard_id": self.guard_id,
                "enabled": bool(self.enabled),
                "max_translation_error_m": self.max_translation_error_m,
                "max_rotation_error_deg": self.max_rotation_error_deg,
                "max_joint_error_rad": self.max_joint_error_rad,
                "max_measurement_age_sec": self.max_measurement_age_sec,
                "violation_policy": self.violation_policy,
            },
            domain=MEASURED_DIVERGENCE_GUARD_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class SafetyAnchorCommitEvidence:
    evidence_id: str
    evidence_kind: str
    target_id: str
    safety_result_id: str
    safe_action_hash: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    occurred_at: float
    accepted: bool
    effective_from: float | None = None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "target_id",
            "safety_result_id",
            "safe_action_hash",
            "control_domain_id",
            "selection_group_id",
        ):
            _required_text(getattr(self, name), name=name)
        if self.evidence_kind not in _COMMIT_EVIDENCE_KINDS:
            raise ValueError("Unsupported safety-anchor commit evidence kind.")
        if int(self.execution_epoch) < 0 or int(self.group_segment_generation) < 0:
            raise ValueError("Safety-anchor commit epoch/generation must be non-negative.")
        occurred_at = _finite_time(self.occurred_at, name="occurred_at")
        effective_from = (
            occurred_at
            if self.effective_from is None
            else _finite_time(self.effective_from, name="effective_from")
        )
        if self.accepted and self.rejection_reason is not None:
            raise ValueError("Accepted commit evidence cannot have a rejection reason.")
        if not self.accepted and not str(self.rejection_reason or "").strip():
            raise ValueError("Rejected commit evidence requires a reason.")
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "effective_from", effective_from)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "evidence_id": self.evidence_id,
                "evidence_kind": self.evidence_kind,
                "target_id": self.target_id,
                "safety_result_id": self.safety_result_id,
                "safe_action_hash": self.safe_action_hash,
                "control_domain_id": self.control_domain_id,
                "selection_group_id": self.selection_group_id,
                "execution_epoch": int(self.execution_epoch),
                "group_segment_generation": int(self.group_segment_generation),
                "occurred_at": float(self.occurred_at),
                "effective_from": float(self.effective_from),
                "accepted": bool(self.accepted),
                "rejection_reason": self.rejection_reason,
            },
            domain=SAFETY_ANCHOR_COMMIT_EVIDENCE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class SafetyAnchorState:
    state_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    status: SafetyAnchorStatus
    policy_hash: str
    anchor: CanonicalActionSafetyAnchor | None
    source_stage: str | None
    source_evidence_ids: tuple[str, ...]
    available_at: float | None
    committed_at: float | None
    valid_until: float | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.status not in _ANCHOR_STATUSES:
            raise ValueError("Unsupported safety-anchor state status.")
        if self.status == "active":
            if (
                self.anchor is None
                or self.available_at is None
                or self.committed_at is None
                or self.source_stage is None
            ):
                raise ValueError("An active safety anchor needs anchor/source/commit evidence.")
        elif self.anchor is not None:
            raise ValueError("An inactive safety-anchor state cannot expose an eligible anchor.")
        object.__setattr__(self, "source_evidence_ids", tuple(self.source_evidence_ids))


@dataclass(frozen=True)
class SafetyAnchorTransition:
    transition_id: str
    transition_kind: SafetyAnchorTransitionKind
    previous_state_id: str | None
    new_state_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    occurred_at: float
    causal_evidence_ids: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.transition_kind not in _TRANSITION_KINDS:
            raise ValueError("Unsupported safety-anchor transition kind.")
        object.__setattr__(self, "causal_evidence_ids", tuple(self.causal_evidence_ids))


@dataclass(frozen=True)
class SafetyAnchorResolution:
    eligible: bool
    status: str
    anchor: CanonicalActionSafetyAnchor | None
    state_id: str
    policy_hash: str


@dataclass(frozen=True)
class SafetyEvaluationInputSnapshot:
    safety_evaluation_input_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    anchor_group_segment_generation: int
    target_group_segment_generation: int
    arbitration_result_id: str
    directive_id: str
    selected_action_id: str
    selected_action: RobotAction
    selected_action_hash: str
    anchor_state: SafetyAnchorState
    measured_state_evidence: CanonicalActionSafetyAnchor | None
    safety_cutoff_at: float
    target_effective_at: float
    anchor_policy_hash: str
    keyframe_safety_profile_hash: str
    measured_divergence_guard_hash: str
    action_schema_hash: str
    component_schema_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_action", freeze_robot_action(self.selected_action))


def _build_state(
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_epoch: int,
    group_segment_generation: int,
    status: SafetyAnchorStatus,
    policy_hash: str,
    anchor: CanonicalActionSafetyAnchor | None,
    source_stage: str | None,
    source_evidence_ids: tuple[str, ...],
    available_at: float | None,
    committed_at: float | None,
    valid_until: float | None,
) -> SafetyAnchorState:
    if status not in _ANCHOR_STATUSES:
        raise ValueError("Unsupported safety-anchor state status.")
    domain = _required_text(control_domain_id, name="control_domain_id")
    group = _required_text(selection_group_id, name="selection_group_id")
    policy = _required_text(policy_hash, name="policy_hash")
    epoch = int(execution_epoch)
    generation = int(group_segment_generation)
    if epoch < 0 or generation < 0:
        raise ValueError("Safety-anchor state epoch/generation must be non-negative.")
    availability_time = (
        None if available_at is None else _finite_time(available_at, name="available_at")
    )
    commit_time = (
        None if committed_at is None else _finite_time(committed_at, name="committed_at")
    )
    expiry = None if valid_until is None else _finite_time(valid_until, name="valid_until")
    if expiry is not None and (commit_time is None or expiry <= commit_time):
        raise ValueError("valid_until must be later than committed_at.")
    evidence_ids = tuple(_required_text(value, name="source_evidence_id") for value in source_evidence_ids)
    payload = {
        "control_domain_id": domain,
        "selection_group_id": group,
        "execution_epoch": epoch,
        "group_segment_generation": generation,
        "status": status,
        "policy_hash": policy,
        "anchor_id": None if anchor is None else anchor.anchor_id,
        "anchor_hash": None if anchor is None else anchor.semantic_content_hash,
        "source_stage": source_stage,
        "source_evidence_ids": evidence_ids,
        "available_at": availability_time,
        "committed_at": commit_time,
        "valid_until": expiry,
    }
    semantic_hash = canonical_content_hash(payload, domain=SAFETY_ANCHOR_STATE_HASH_DOMAIN_V1)
    return SafetyAnchorState(
        state_id=f"safety-anchor-state:{semantic_hash}",
        control_domain_id=domain,
        selection_group_id=group,
        execution_epoch=epoch,
        group_segment_generation=generation,
        status=status,
        policy_hash=policy,
        anchor=anchor,
        source_stage=source_stage,
        source_evidence_ids=evidence_ids,
        available_at=availability_time,
        committed_at=commit_time,
        valid_until=expiry,
        semantic_content_hash=semantic_hash,
    )


def _build_transition(
    previous: SafetyAnchorState | None,
    new: SafetyAnchorState,
    *,
    transition_kind: SafetyAnchorTransitionKind,
    occurred_at: float,
    causal_evidence_ids: tuple[str, ...],
) -> SafetyAnchorTransition:
    if transition_kind not in _TRANSITION_KINDS:
        raise ValueError("Unsupported safety-anchor transition kind.")
    timestamp = _finite_time(occurred_at, name="occurred_at")
    evidence = tuple(
        _required_text(value, name="causal_evidence_id") for value in causal_evidence_ids
    )
    payload = {
        "transition_kind": transition_kind,
        "previous_state_id": None if previous is None else previous.state_id,
        "new_state_id": new.state_id,
        "control_domain_id": new.control_domain_id,
        "selection_group_id": new.selection_group_id,
        "execution_epoch": new.execution_epoch,
        "group_segment_generation": new.group_segment_generation,
        "occurred_at": timestamp,
        "causal_evidence_ids": evidence,
    }
    semantic_hash = canonical_content_hash(
        payload, domain=SAFETY_ANCHOR_TRANSITION_HASH_DOMAIN_V1
    )
    return SafetyAnchorTransition(
        transition_id=f"safety-anchor-transition:{semantic_hash}",
        transition_kind=transition_kind,
        previous_state_id=None if previous is None else previous.state_id,
        new_state_id=new.state_id,
        control_domain_id=new.control_domain_id,
        selection_group_id=new.selection_group_id,
        execution_epoch=new.execution_epoch,
        group_segment_generation=new.group_segment_generation,
        occurred_at=timestamp,
        causal_evidence_ids=evidence,
        semantic_content_hash=semantic_hash,
    )


def make_uninitialized_safety_anchor_state(
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_epoch: int,
    group_segment_generation: int,
    policy: SafetyAnchorPolicy,
) -> SafetyAnchorState:
    return _build_state(
        control_domain_id=control_domain_id,
        selection_group_id=selection_group_id,
        execution_epoch=execution_epoch,
        group_segment_generation=group_segment_generation,
        status="uninitialized",
        policy_hash=policy.semantic_content_hash,
        anchor=None,
        source_stage=None,
        source_evidence_ids=(),
        available_at=None,
        committed_at=None,
        valid_until=None,
    )


def rebase_safety_anchor_from_measured(
    previous: SafetyAnchorState,
    *,
    measured_anchor: CanonicalActionSafetyAnchor,
    measured_evidence_id: str,
    occurred_at: float,
    policy: SafetyAnchorPolicy,
    transition_kind: SafetyAnchorTransitionKind = "rebase_after_recovery",
) -> tuple[SafetyAnchorState, SafetyAnchorTransition]:
    if measured_anchor.anchor_kind != "measured_state":
        raise ValueError("Safety rebase requires measured_state anchor evidence.")
    timestamp = _finite_time(occurred_at, name="occurred_at")
    if previous.policy_hash != policy.semantic_content_hash:
        raise ValueError("Safety rebase policy hash mismatch.")
    if measured_anchor.observed_at is None:
        raise ValueError("Measured-state rebase requires observed_at.")
    measured_at = float(measured_anchor.observed_at)
    if measured_at > timestamp:
        raise ValueError("Measured-state evidence cannot come from the future.")
    maximum_age = float(policy.max_measurement_age_sec)
    if timestamp - measured_at > maximum_age:
        raise ValueError("Measured-state evidence is stale for safety rebase.")
    if previous.status == "uninitialized" and transition_kind == "rebase_after_recovery":
        transition_kind = "initialize_from_measured"
    if transition_kind not in {"initialize_from_measured", "rebase_after_recovery"}:
        raise ValueError("Measured rebase needs an initialize/recovery transition kind.")
    next_state = _build_state(
        control_domain_id=previous.control_domain_id,
        selection_group_id=previous.selection_group_id,
        execution_epoch=previous.execution_epoch,
        group_segment_generation=previous.group_segment_generation,
        status="active",
        policy_hash=previous.policy_hash,
        anchor=measured_anchor,
        source_stage="measured_state_rebase",
        source_evidence_ids=(measured_evidence_id,),
        available_at=timestamp,
        committed_at=timestamp,
        valid_until=None,
    )
    transition = _build_transition(
        previous,
        next_state,
        transition_kind=transition_kind,
        occurred_at=timestamp,
        causal_evidence_ids=(measured_evidence_id,),
    )
    return next_state, transition


def commit_safe_action_anchor(
    previous: SafetyAnchorState,
    *,
    safe_action: RobotAction,
    evidence: SafetyAnchorCommitEvidence,
    valid_until: float | None = None,
) -> tuple[SafetyAnchorState, SafetyAnchorTransition]:
    if previous.status != "active" or previous.anchor is None:
        raise ValueError("A safe target can commit only after an active measured/safe anchor.")
    if not evidence.accepted:
        raise ValueError("Rejected execution-target evidence cannot commit a safety anchor.")
    if (
        evidence.control_domain_id != previous.control_domain_id
        or evidence.selection_group_id != previous.selection_group_id
        or evidence.execution_epoch != previous.execution_epoch
        or evidence.group_segment_generation != previous.group_segment_generation + 1
    ):
        raise ValueError(
            "Safety-anchor commit evidence must target the same domain/group/epoch "
            "and the next segment generation."
        )
    frozen_action = freeze_robot_action(safe_action)
    safe_hash = safety_anchor_action_hash(frozen_action)
    if evidence.safe_action_hash != safe_hash:
        raise ValueError("Safety-anchor commit evidence safe-action hash mismatch.")
    if (
        previous.committed_at is not None
        and float(evidence.effective_from) < previous.committed_at
    ):
        raise ValueError("Safety-anchor commit effective time predates the current anchor.")
    if frozen_action.end_effector_targets_by_side:
        raise ValueError(
            "Safety-anchor lifecycle v1 does not commit native end-effector actions."
        )
    if frozen_action.arm_pose_by_side and frozen_action.arm_qpos_by_side:
        raise ValueError(
            "Safety-anchor lifecycle v1 requires at most one arm action modality."
        )
    if (
        not frozen_action.arm_pose_by_side
        and not frozen_action.arm_qpos_by_side
        and not frozen_action.hand_qpos_by_side
    ):
        raise ValueError("Safety-anchor lifecycle requires a non-empty safe action.")
    hand_specs = hand_action_specs_from_metadata(frozen_action.metadata)
    if set(hand_specs) != set(frozen_action.hand_qpos_by_side):
        raise ValueError(
            "Committed hand qpos requires one explicit HandActionSpec per side."
        )
    anchor_kwargs = {
        "anchor_id": f"committed-safe:{evidence.evidence_id}",
        "anchor_kind": "previous_safe_committed",
        "action_space": frozen_action.action_space,
        "hand_qpos_by_side": frozen_action.hand_qpos_by_side,
        "hand_action_specs_by_side": hand_specs,
        "observed_at": float(evidence.effective_from),
    }
    if frozen_action.arm_pose_by_side:
        anchor_kwargs["frame_by_side"] = {
            side: str(frozen_action.arm_frame(side))
            for side in frozen_action.arm_pose_by_side
        }
        anchor_kwargs["arm_pose_by_side"] = frozen_action.arm_pose_by_side
    elif frozen_action.arm_qpos_by_side:
        anchor_kwargs["arm_qpos_by_side"] = frozen_action.arm_qpos_by_side
    anchor = CanonicalActionSafetyAnchor(**anchor_kwargs)
    next_state = _build_state(
        control_domain_id=previous.control_domain_id,
        selection_group_id=previous.selection_group_id,
        execution_epoch=previous.execution_epoch,
        group_segment_generation=evidence.group_segment_generation,
        status="active",
        policy_hash=previous.policy_hash,
        anchor=anchor,
        source_stage="safe_action_committed",
        source_evidence_ids=(evidence.safety_result_id, evidence.evidence_id),
        available_at=evidence.occurred_at,
        committed_at=float(evidence.effective_from),
        valid_until=valid_until,
    )
    transition = _build_transition(
        previous,
        next_state,
        transition_kind="commit_safe_target",
        occurred_at=float(evidence.effective_from),
        causal_evidence_ids=(evidence.safety_result_id, evidence.evidence_id),
    )
    return next_state, transition


def invalidate_safety_anchor(
    previous: SafetyAnchorState,
    *,
    transition_kind: SafetyAnchorTransitionKind,
    occurred_at: float,
    causal_evidence_ids: tuple[str, ...],
    next_execution_epoch: int | None = None,
    next_group_segment_generation: int | None = None,
) -> tuple[SafetyAnchorState, SafetyAnchorTransition]:
    if transition_kind not in {
        "invalidate_on_hold",
        "invalidate_on_fault",
        "invalidate_on_backend_reject",
        "invalidate_on_execution_epoch_change",
        "invalidate_on_group_cancel",
    }:
        raise ValueError("invalidate_safety_anchor requires an invalidation transition.")
    epoch = (
        previous.execution_epoch
        if next_execution_epoch is None
        else int(next_execution_epoch)
    )
    generation = (
        previous.group_segment_generation
        if next_group_segment_generation is None
        else int(next_group_segment_generation)
    )
    if epoch < previous.execution_epoch:
        raise ValueError("Safety-anchor execution epoch cannot move backwards.")
    if epoch == previous.execution_epoch and generation < previous.group_segment_generation:
        raise ValueError("Safety-anchor group generation cannot move backwards.")
    if (
        transition_kind == "invalidate_on_execution_epoch_change"
        and epoch <= previous.execution_epoch
    ):
        raise ValueError("Execution-epoch invalidation requires a newer epoch.")
    next_state = _build_state(
        control_domain_id=previous.control_domain_id,
        selection_group_id=previous.selection_group_id,
        execution_epoch=epoch,
        group_segment_generation=generation,
        status="rebase_required",
        policy_hash=previous.policy_hash,
        anchor=None,
        source_stage=None,
        source_evidence_ids=tuple(causal_evidence_ids),
        available_at=None,
        committed_at=None,
        valid_until=None,
    )
    transition = _build_transition(
        previous,
        next_state,
        transition_kind=transition_kind,
        occurred_at=occurred_at,
        causal_evidence_ids=tuple(causal_evidence_ids),
    )
    return next_state, transition


def resolve_safety_anchor(
    state: SafetyAnchorState,
    *,
    policy: SafetyAnchorPolicy,
    safety_cutoff_at: float,
    target_effective_at: float | None = None,
    execution_epoch: int,
    group_segment_generation: int,
) -> SafetyAnchorResolution:
    cutoff = _finite_time(safety_cutoff_at, name="safety_cutoff_at")
    target_time = (
        cutoff
        if target_effective_at is None
        else _finite_time(target_effective_at, name="target_effective_at")
    )
    if state.policy_hash != policy.semantic_content_hash:
        return SafetyAnchorResolution(
            False, "anchor_policy_mismatch", None, state.state_id, state.policy_hash
        )
    if (
        state.execution_epoch != int(execution_epoch)
        or state.group_segment_generation != int(group_segment_generation)
    ):
        return SafetyAnchorResolution(
            False, "anchor_scope_mismatch", None, state.state_id, state.policy_hash
        )
    if state.status != "active" or state.anchor is None:
        return SafetyAnchorResolution(
            False, "rebase_required", None, state.state_id, state.policy_hash
        )
    if state.available_at is not None and state.available_at > cutoff:
        return SafetyAnchorResolution(
            False, "anchor_not_yet_available", None, state.state_id, state.policy_hash
        )
    if state.committed_at is not None and state.committed_at > target_time:
        return SafetyAnchorResolution(
            False, "anchor_not_yet_effective", None, state.state_id, state.policy_hash
        )
    if state.valid_until is not None and target_time >= state.valid_until:
        return SafetyAnchorResolution(
            False, "anchor_expired", None, state.state_id, state.policy_hash
        )
    if (
        policy.max_anchor_age_sec is not None
        and state.committed_at is not None
        and target_time - state.committed_at > policy.max_anchor_age_sec
    ):
        return SafetyAnchorResolution(
            False, "anchor_stale", None, state.state_id, state.policy_hash
        )
    return SafetyAnchorResolution(
        True, "eligible", state.anchor, state.state_id, state.policy_hash
    )


def build_safety_evaluation_input_snapshot(
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_epoch: int,
    target_group_segment_generation: int,
    arbitration_result_id: str,
    directive_id: str,
    selected_action_id: str,
    selected_action: RobotAction,
    anchor_state: SafetyAnchorState,
    measured_state_evidence: CanonicalActionSafetyAnchor | None,
    safety_cutoff_at: float,
    target_effective_at: float | None = None,
    anchor_policy: SafetyAnchorPolicy,
    keyframe_safety_profile_hash: str,
    measured_divergence_guard: MeasuredDivergenceGuardSpec,
    action_schema_hash: str,
    component_schema_hash: str,
) -> SafetyEvaluationInputSnapshot:
    domain = _required_text(control_domain_id, name="control_domain_id")
    group = _required_text(selection_group_id, name="selection_group_id")
    epoch = int(execution_epoch)
    target_generation = int(target_group_segment_generation)
    if (
        anchor_state.control_domain_id != domain
        or anchor_state.selection_group_id != group
        or anchor_state.execution_epoch != epoch
    ):
        raise ValueError("Safety evaluation anchor state belongs to a different scope.")
    if target_generation != anchor_state.group_segment_generation + 1:
        raise ValueError(
            "Safety evaluation target must use the next group segment generation."
        )
    if anchor_state.policy_hash != anchor_policy.semantic_content_hash:
        raise ValueError("Safety evaluation anchor policy hash mismatch.")
    if (
        measured_state_evidence is not None
        and measured_state_evidence.anchor_kind != "measured_state"
    ):
        raise ValueError("Measured safety evidence must use measured_state anchor kind.")
    action = freeze_robot_action(selected_action)
    action_hash = safety_anchor_action_hash(action)
    cutoff = _finite_time(safety_cutoff_at, name="safety_cutoff_at")
    target_time = (
        cutoff
        if target_effective_at is None
        else _finite_time(target_effective_at, name="target_effective_at")
    )
    if anchor_state.available_at is not None and anchor_state.available_at > cutoff:
        raise ValueError("Safety evaluation cannot consume unavailable anchor evidence.")
    if anchor_state.committed_at is not None and anchor_state.committed_at > target_time:
        raise ValueError("Safety evaluation cannot consume a not-yet-effective anchor.")
    if measured_divergence_guard.enabled:
        if measured_state_evidence is None or measured_state_evidence.observed_at is None:
            raise ValueError(
                "Enabled measured-divergence guard requires timestamped measured evidence."
            )
        measured_at = float(measured_state_evidence.observed_at)
        if measured_at > cutoff:
            raise ValueError("Measured safety evidence cannot come from the future.")
        assert measured_divergence_guard.max_measurement_age_sec is not None
        if cutoff - measured_at > measured_divergence_guard.max_measurement_age_sec:
            raise ValueError("Measured safety evidence is stale for divergence guard.")
    payload = {
        "control_domain_id": domain,
        "selection_group_id": group,
        "execution_epoch": epoch,
        "anchor_group_segment_generation": anchor_state.group_segment_generation,
        "target_group_segment_generation": target_generation,
        "arbitration_result_id": _required_text(
            arbitration_result_id, name="arbitration_result_id"
        ),
        "directive_id": _required_text(directive_id, name="directive_id"),
        "selected_action_id": _required_text(
            selected_action_id, name="selected_action_id"
        ),
        "selected_action_hash": action_hash,
        "anchor_state_id": anchor_state.state_id,
        "anchor_state_hash": anchor_state.semantic_content_hash,
        "measured_state_evidence_id": (
            None if measured_state_evidence is None else measured_state_evidence.anchor_id
        ),
        "measured_state_evidence_hash": (
            None
            if measured_state_evidence is None
            else measured_state_evidence.semantic_content_hash
        ),
        "safety_cutoff_at": cutoff,
        "target_effective_at": target_time,
        "anchor_policy_hash": anchor_policy.semantic_content_hash,
        "keyframe_safety_profile_hash": _required_text(
            keyframe_safety_profile_hash, name="keyframe_safety_profile_hash"
        ),
        "measured_divergence_guard_hash": measured_divergence_guard.semantic_content_hash,
        "action_schema_hash": _required_text(action_schema_hash, name="action_schema_hash"),
        "component_schema_hash": _required_text(
            component_schema_hash, name="component_schema_hash"
        ),
    }
    semantic_hash = canonical_content_hash(
        payload, domain=SAFETY_EVALUATION_INPUT_HASH_DOMAIN_V1
    )
    return SafetyEvaluationInputSnapshot(
        safety_evaluation_input_id=f"safety-evaluation-input:{semantic_hash}",
        control_domain_id=domain,
        selection_group_id=group,
        execution_epoch=epoch,
        anchor_group_segment_generation=anchor_state.group_segment_generation,
        target_group_segment_generation=target_generation,
        arbitration_result_id=payload["arbitration_result_id"],
        directive_id=payload["directive_id"],
        selected_action_id=payload["selected_action_id"],
        selected_action=action,
        selected_action_hash=action_hash,
        anchor_state=anchor_state,
        measured_state_evidence=measured_state_evidence,
        safety_cutoff_at=cutoff,
        target_effective_at=target_time,
        anchor_policy_hash=anchor_policy.semantic_content_hash,
        keyframe_safety_profile_hash=payload["keyframe_safety_profile_hash"],
        measured_divergence_guard_hash=measured_divergence_guard.semantic_content_hash,
        action_schema_hash=payload["action_schema_hash"],
        component_schema_hash=payload["component_schema_hash"],
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "MeasuredDivergenceGuardSpec",
    "SafetyAnchorCommitEvidence",
    "SafetyAnchorPolicy",
    "SafetyAnchorResolution",
    "SafetyAnchorState",
    "SafetyAnchorTransition",
    "SafetyEvaluationInputSnapshot",
    "build_safety_evaluation_input_snapshot",
    "commit_safe_action_anchor",
    "invalidate_safety_anchor",
    "make_uninitialized_safety_anchor_state",
    "rebase_safety_anchor_from_measured",
    "resolve_safety_anchor",
    "safety_anchor_action_hash",
]
