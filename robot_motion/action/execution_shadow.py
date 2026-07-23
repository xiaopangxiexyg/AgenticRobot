"""Pure, non-authoritative evaluation of the shared execution path.

This module owns no execution segment, cancellation epoch, dispatch permit,
Rig, or backend.  The M5.7a evaluator stops at immutable command intent.  The
M5.7c2 helpers additionally model a non-authoritative safety-anchor transition,
but only after the caller supplies typed shadow target-admission evidence; they
never manufacture a real execution receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze

from .arbiter import ArbitrationResult
from .executor import RobotActionResolver, teleop_command_semantic_payload
from .persistence import (
    ActionPersistencePolicy,
    ActiveExecutionSnapshot,
    ExecutionDirective,
    build_execution_directive,
)
from .proposal import robot_action_semantic_payload
from .safety import SharedCanonicalSafetyResult
from .schema import RobotAction
from .execution_target import (
    ExecutionTargetAdmissionReceipt,
    ResolvedExecutionTarget,
    execution_target_anchor_commit_evidence,
)
from .keyframe_safety import (
    CanonicalActionSafetyAnchor,
    CanonicalKeyframeSafetyProfile,
    apply_canonical_keyframe_safety,
)
from .safety_anchor import (
    MeasuredDivergenceGuardSpec,
    SafetyAnchorState,
    SafetyAnchorTransition,
    SafetyEvaluationInputSnapshot,
    SafetyAnchorPolicy,
    build_safety_evaluation_input_snapshot,
    commit_safe_action_anchor,
    resolve_safety_anchor,
)


SHARED_EXECUTION_SHADOW_HASH_DOMAIN_V1 = "dexteleop.shared_execution_shadow.v1"
SHARED_EXECUTION_SHADOW_SAFETY_HASH_DOMAIN_V1 = (
    "dexteleop.shared_execution_shadow.safety.v1"
)
SHARED_EXECUTION_SHADOW_COMMAND_HASH_DOMAIN_V1 = (
    "dexteleop.shared_execution_shadow.command.v1"
)
COUNTERFACTUAL_SAFETY_SHADOW_HASH_DOMAIN_V1 = (
    "dexteleop.counterfactual_safety_shadow.v1"
)
COUNTERFACTUAL_TARGET_COMMIT_HASH_DOMAIN_V1 = (
    "dexteleop.counterfactual_target_commit.v1"
)

SharedSafety = Callable[[RobotAction, str], SharedCanonicalSafetyResult]


@dataclass(frozen=True)
class SharedExecutionShadowEvaluation:
    evaluation_id: str
    source_arbitration_evaluation_id: str | None
    arbitration_result_id: str
    output_status: str
    proposed_directive: ExecutionDirective
    safety_result: SharedCanonicalSafetyResult | None
    safety_result_hash: str | None
    command_intent: Mapping[str, Any] | None
    command_intent_hash: str | None
    non_authoritative: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not self.non_authoritative:
            raise ValueError("Shared execution shadow facts must be non-authoritative.")
        if self.command_intent is not None:
            object.__setattr__(
                self,
                "command_intent",
                MappingProxyType(dict(deep_freeze(self.command_intent))),
            )


@dataclass(frozen=True)
class CounterfactualSafetyShadowEvaluation:
    """One non-authoritative safety evaluation against shadow anchor state."""

    evaluation_id: str
    source_arbitration_evaluation_id: str
    input_snapshot: SafetyEvaluationInputSnapshot
    anchor_state_before: SafetyAnchorState
    safety_result_id: str
    safety_result: SharedCanonicalSafetyResult
    safety_result_hash: str
    non_authoritative: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not self.non_authoritative:
            raise ValueError("Counterfactual safety evaluation must be non-authoritative.")
        if self.input_snapshot.anchor_state.state_id != self.anchor_state_before.state_id:
            raise ValueError("Counterfactual safety input references another anchor state.")


@dataclass(frozen=True)
class CounterfactualTargetCommitEvaluation:
    """Shadow-only target admission and optional anchor transition."""

    evaluation_id: str
    source_safety_evaluation_id: str
    target: ResolvedExecutionTarget
    admission: ExecutionTargetAdmissionReceipt
    anchor_state_before: SafetyAnchorState
    anchor_state_after: SafetyAnchorState
    anchor_transition: SafetyAnchorTransition | None
    committed: bool
    non_authoritative: bool
    status: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if not self.non_authoritative:
            raise ValueError("Counterfactual target commit must be non-authoritative.")
        if self.committed != (self.anchor_transition is not None):
            raise ValueError("Counterfactual committed flag and transition disagree.")


def _safety_hash(result: SharedCanonicalSafetyResult) -> str:
    return canonical_content_hash(
        {
            "input_action_id": result.input_action_id,
            "safe_action": (
                None
                if result.safe_action is None
                else robot_action_semantic_payload(result.safe_action)
            ),
            "modified_components": result.modified_components,
            "reasons": result.reasons,
            "status": result.status,
            "safety_contract_id": result.safety_contract_id,
            "safety_contract_hash": result.safety_contract_hash,
            "anchor_id": result.anchor_id,
            "anchor_kind": result.anchor_kind,
            "anchor_hash": result.anchor_hash,
            "details": result.details,
        },
        domain=SHARED_EXECUTION_SHADOW_SAFETY_HASH_DOMAIN_V1,
    )


def evaluate_counterfactual_safety_shadow(
    *,
    source_arbitration_evaluation_id: str,
    arbitration_result_id: str,
    directive_id: str,
    selected_action_id: str,
    selected_action: RobotAction,
    anchor_state: SafetyAnchorState,
    measured_state_evidence: CanonicalActionSafetyAnchor | None,
    safety_cutoff_at: float,
    anchor_policy: SafetyAnchorPolicy,
    keyframe_safety_profile: CanonicalKeyframeSafetyProfile,
    measured_divergence_guard: MeasuredDivergenceGuardSpec,
    action_schema_hash: str,
    component_schema_hash: str,
) -> CounterfactualSafetyShadowEvaluation:
    """Evaluate shared safety without mutating the supplied shadow state.

    Startup/recovery rebasing remains an explicit caller step.  The currently
    disabled measured-divergence contract is accepted as evidence, but an
    enabled guard is rejected until its numeric evaluator and thresholds are
    validated.
    """

    if measured_divergence_guard.enabled:
        raise ValueError(
            "Counterfactual safety shadow does not yet implement measured divergence."
        )
    resolution = resolve_safety_anchor(
        anchor_state,
        policy=anchor_policy,
        safety_cutoff_at=float(safety_cutoff_at),
        execution_epoch=anchor_state.execution_epoch,
        group_segment_generation=anchor_state.group_segment_generation,
    )
    if not resolution.eligible or resolution.anchor is None:
        raise ValueError(f"Counterfactual safety anchor is unavailable: {resolution.status}.")
    snapshot = build_safety_evaluation_input_snapshot(
        control_domain_id=anchor_state.control_domain_id,
        selection_group_id=anchor_state.selection_group_id,
        execution_epoch=anchor_state.execution_epoch,
        target_group_segment_generation=anchor_state.group_segment_generation + 1,
        arbitration_result_id=arbitration_result_id,
        directive_id=directive_id,
        selected_action_id=selected_action_id,
        selected_action=selected_action,
        anchor_state=anchor_state,
        measured_state_evidence=measured_state_evidence,
        safety_cutoff_at=float(safety_cutoff_at),
        anchor_policy=anchor_policy,
        keyframe_safety_profile_hash=keyframe_safety_profile.semantic_content_hash,
        measured_divergence_guard=measured_divergence_guard,
        action_schema_hash=action_schema_hash,
        component_schema_hash=component_schema_hash,
    )
    safety = apply_canonical_keyframe_safety(
        snapshot.selected_action,
        snapshot.selected_action_id,
        anchor=resolution.anchor,
        profile=keyframe_safety_profile,
    )
    safety_hash = _safety_hash(safety)
    safety_result_id = f"counterfactual-shared-safety:{safety_hash}"
    payload = {
        "source_arbitration_evaluation_id": source_arbitration_evaluation_id,
        "safety_evaluation_input_id": snapshot.safety_evaluation_input_id,
        "safety_evaluation_input_hash": snapshot.semantic_content_hash,
        "anchor_state_before_id": anchor_state.state_id,
        "anchor_state_before_hash": anchor_state.semantic_content_hash,
        "safety_result_id": safety_result_id,
        "safety_result_hash": safety_hash,
        "non_authoritative": True,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=COUNTERFACTUAL_SAFETY_SHADOW_HASH_DOMAIN_V1,
    )
    return CounterfactualSafetyShadowEvaluation(
        evaluation_id=f"counterfactual-safety-shadow:{semantic_hash}",
        source_arbitration_evaluation_id=source_arbitration_evaluation_id,
        input_snapshot=snapshot,
        anchor_state_before=anchor_state,
        safety_result_id=safety_result_id,
        safety_result=safety,
        safety_result_hash=safety_hash,
        non_authoritative=True,
        semantic_content_hash=semantic_hash,
    )


def evaluate_counterfactual_target_commit(
    safety_evaluation: CounterfactualSafetyShadowEvaluation,
    *,
    target: ResolvedExecutionTarget,
    admission: ExecutionTargetAdmissionReceipt,
) -> CounterfactualTargetCommitEvaluation:
    """Advance only the shadow anchor after accepted typed target admission."""

    if not isinstance(safety_evaluation, CounterfactualSafetyShadowEvaluation):
        raise TypeError("safety_evaluation must be CounterfactualSafetyShadowEvaluation.")
    safe_action = safety_evaluation.safety_result.safe_action
    if safe_action is None or safety_evaluation.safety_result.status in {"hold", "fault"}:
        raise ValueError("Blocked counterfactual safety cannot create an execution target.")
    if target.source_safety_result_id != safety_evaluation.safety_result_id:
        raise ValueError("Counterfactual target references another safety evaluation.")
    evidence = execution_target_anchor_commit_evidence(
        target,
        admission,
        safety_result_id=safety_evaluation.safety_result_id,
    )
    previous = safety_evaluation.anchor_state_before
    if evidence.accepted:
        next_state, transition = commit_safe_action_anchor(
            previous,
            safe_action=target.group_safe_action,
            evidence=evidence,
            valid_until=target.scheduled_until,
        )
        committed = True
        status = "shadow_target_committed"
    else:
        next_state = previous
        transition = None
        committed = False
        status = "shadow_target_rejected"
    payload = {
        "source_safety_evaluation_id": safety_evaluation.evaluation_id,
        "target_id": target.target_id,
        "target_hash": target.semantic_content_hash,
        "admission_id": admission.admission_id,
        "admission_hash": admission.semantic_content_hash,
        "anchor_state_before_id": previous.state_id,
        "anchor_state_after_id": next_state.state_id,
        "anchor_transition_id": (
            None if transition is None else transition.transition_id
        ),
        "committed": committed,
        "status": status,
        "non_authoritative": True,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=COUNTERFACTUAL_TARGET_COMMIT_HASH_DOMAIN_V1,
    )
    return CounterfactualTargetCommitEvaluation(
        evaluation_id=f"counterfactual-target-commit:{semantic_hash}",
        source_safety_evaluation_id=safety_evaluation.evaluation_id,
        target=target,
        admission=admission,
        anchor_state_before=previous,
        anchor_state_after=next_state,
        anchor_transition=transition,
        committed=committed,
        non_authoritative=True,
        status=status,
        semantic_content_hash=semantic_hash,
    )


def evaluate_shared_execution_shadow(
    result: ArbitrationResult,
    *,
    active_execution: ActiveExecutionSnapshot,
    persistence_policy: ActionPersistencePolicy,
    shared_safety: SharedSafety,
    resolver: RobotActionResolver | None = None,
    now: float,
    supports_local_group_stop: bool,
    source_arbitration_evaluation_id: str | None = None,
) -> SharedExecutionShadowEvaluation:
    """Evaluate selected→persistence→safety→command without side effects."""

    if not callable(shared_safety):
        raise TypeError("shared_safety must be callable.")
    directive = build_execution_directive(
        result,
        active_execution=active_execution,
        policy=persistence_policy,
        now=float(now),
        supports_local_group_stop=bool(supports_local_group_stop),
    )
    safety_result: SharedCanonicalSafetyResult | None = None
    safety_hash: str | None = None
    command_intent: Mapping[str, Any] | None = None
    command_hash: str | None = None
    output_status = "no_command_intent"

    publishes_selected = any(
        item.directive == "publish_selected"
        for item in directive.directive_by_selection_group.values()
    )
    if publishes_selected:
        if result.output_status != "selected" or result.composed_action is None:
            raise ValueError(
                "A publish_selected shadow directive requires a selected action."
            )
        safety_result = shared_safety(result.composed_action, result.result_id)
        if not isinstance(safety_result, SharedCanonicalSafetyResult):
            raise TypeError("shared_safety must return SharedCanonicalSafetyResult.")
        if safety_result.input_action_id != result.result_id:
            raise ValueError("Shared safety result references another arbitration result.")
        safety_hash = _safety_hash(safety_result)
        if safety_result.safe_action is None or safety_result.status in {"hold", "fault"}:
            output_status = f"safety_{safety_result.status}"
        else:
            command = (resolver or RobotActionResolver()).resolve(
                safety_result.safe_action
            )
            command_payload = teleop_command_semantic_payload(command)
            command_hash = canonical_content_hash(
                command_payload,
                domain=SHARED_EXECUTION_SHADOW_COMMAND_HASH_DOMAIN_V1,
            )
            command_intent = deep_freeze(command_payload)
            output_status = "command_intent_ready"
    elif directive.requires_backend_hold:
        output_status = "proposed_hold"
    elif any(
        item.directive == "keep_backend_hold"
        for item in directive.directive_by_selection_group.values()
    ):
        output_status = "proposed_keep_backend_hold"

    payload = {
        "source_arbitration_evaluation_id": source_arbitration_evaluation_id,
        "arbitration_result_id": result.result_id,
        "arbitration_result_hash": result.semantic_content_hash,
        "output_status": output_status,
        "proposed_directive_id": directive.directive_id,
        "proposed_directive_hash": directive.semantic_content_hash,
        "safety_result_hash": safety_hash,
        "command_intent_hash": command_hash,
        "non_authoritative": True,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=SHARED_EXECUTION_SHADOW_HASH_DOMAIN_V1,
    )
    return SharedExecutionShadowEvaluation(
        evaluation_id=f"shared-execution-shadow:{semantic_hash}",
        source_arbitration_evaluation_id=source_arbitration_evaluation_id,
        arbitration_result_id=result.result_id,
        output_status=output_status,
        proposed_directive=directive,
        safety_result=safety_result,
        safety_result_hash=safety_hash,
        command_intent=command_intent,
        command_intent_hash=command_hash,
        non_authoritative=True,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "COUNTERFACTUAL_SAFETY_SHADOW_HASH_DOMAIN_V1",
    "COUNTERFACTUAL_TARGET_COMMIT_HASH_DOMAIN_V1",
    "CounterfactualSafetyShadowEvaluation",
    "CounterfactualTargetCommitEvaluation",
    "SHARED_EXECUTION_SHADOW_HASH_DOMAIN_V1",
    "SharedExecutionShadowEvaluation",
    "evaluate_counterfactual_safety_shadow",
    "evaluate_counterfactual_target_commit",
    "evaluate_shared_execution_shadow",
]
