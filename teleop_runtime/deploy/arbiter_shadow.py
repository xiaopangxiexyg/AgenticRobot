"""Passive single-policy ActionArbiter shadow coordinator.

This module owns no policy queue, prediction transaction, dispatcher, Rig, or
backend. It accepts only the immutable emission returned by the production
queue-pop seam and produces non-authoritative evaluation facts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ActionArbiter,
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionProposalDraft,
    ArbitrationInputSnapshot,
    ArbitrationResult,
    AuthorityPolicySpec,
    AuthorityStore,
    DecisionScheduleSpec,
    DecisionScheduler,
    ProposalAdmissionReceipt,
    ProposalBuffer,
    RobotAction,
    ComponentAction,
    ComponentActionSchema,
    materialize_robot_action_v1,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.arbitration_input import capture_arbitration_input
from robot_motion.deploy.runtime import PolicyActionEmission
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.execution_gate import (
    EXECUTION_GATE_STATUSES,
    ExecutionGateEvent,
    ExecutionGateEventDraft,
    ExecutionGateStore,
)
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


ARBITER_SHADOW_MODE_OFF = "off"
ARBITER_SHADOW_MODE_SHADOW = "shadow"
ARBITER_SHADOW_MODES_M3 = (ARBITER_SHADOW_MODE_OFF, ARBITER_SHADOW_MODE_SHADOW)
ARBITRATION_EVALUATION_HASH_DOMAIN_V1 = "dexteleop.arbitration_evaluation.v1"
SHADOW_STATUS_HASH_DOMAIN_V1 = "dexteleop.arbiter_shadow_status.v1"
LEGACY_EXECUTION_OBSERVATION_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_execution_observation.v1"
)
ACTION_COMPONENT_PARITY_HASH_DOMAIN_V1 = "dexteleop.action_component_parity.v1"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class PolicyArbiterShadowConfig:
    runtime_session_id: str
    control_domain_id: str
    embodiment_instance_id: str
    producer_id: str
    active_effectors: tuple[str, ...]
    action_space: str
    decision_hz: float
    action_schema: ComponentActionSchema
    component_catalog: ActionComponentCatalog | None = None
    producer_epoch: int = 1
    policy_node_id: str = "single_policy"
    clock_domain_id: str = "local_monotonic"
    boot_id: str = "local-process"
    proposal_capacity: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "producer_id",
            "action_space",
            "policy_node_id",
            "clock_domain_id",
            "boot_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        effectors = tuple(dict.fromkeys(_required(value, name="effector") for value in self.active_effectors))
        if not effectors:
            raise ValueError("active_effectors must not be empty.")
        object.__setattr__(self, "active_effectors", effectors)
        if int(self.producer_epoch) < 0:
            raise ValueError("producer_epoch must be non-negative.")
        if int(self.proposal_capacity) <= 0:
            raise ValueError("proposal_capacity must be positive.")
        object.__setattr__(self, "producer_epoch", int(self.producer_epoch))
        object.__setattr__(self, "proposal_capacity", int(self.proposal_capacity))
        rate = float(self.decision_hz)
        if not math.isfinite(rate) or rate <= 0 or rate > 60.0:
            raise ValueError("M3 shadow decision_hz must be in (0, 60].")
        object.__setattr__(self, "decision_hz", rate)
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        if set(self.active_effectors) != set(
            self.action_schema.component_ids
        ):
            raise ValueError(
                "Shadow active_effectors must match the runtime action schema."
            )
        if self.component_catalog is not None:
            if self.component_catalog.action_space != self.action_space:
                raise ValueError("component_catalog action_space must match shadow action_space.")
            catalog_effectors = tuple(
                item.effector_name for item in self.component_catalog.components
            )
            if set(catalog_effectors) != set(effectors):
                raise ValueError(
                    "component_catalog effectors must match active_effectors."
                )


@dataclass(frozen=True)
class ActionComponentParityReceipt:
    matched: bool
    compared_components: tuple[str, ...]
    mismatched_components: tuple[str, ...]
    max_abs_error: float
    absolute_tolerance: float
    semantic_content_hash: str


@dataclass(frozen=True)
class ArbitrationEvaluationEvent:
    evaluation_id: str
    mode: str
    non_authoritative: bool
    arbitration_input_snapshot_id: str
    arbitration_input_hash: str
    proposal_snapshot_id: str
    computation_result_id: str
    computation_result_hash: str
    output_status: str
    would_select_action: ComponentAction | None
    would_select_action_hash: str | None
    would_select_by_group: Mapping[str, str]
    would_select_by_component: Mapping[str, str]
    evaluated_at: float
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.mode != ARBITER_SHADOW_MODE_SHADOW or not self.non_authoritative:
            raise ValueError("M3 evaluation must be explicitly non-authoritative shadow.")
        object.__setattr__(
            self,
            "would_select_by_group",
            MappingProxyType(dict(self.would_select_by_group)),
        )
        object.__setattr__(
            self,
            "would_select_by_component",
            MappingProxyType(dict(self.would_select_by_component)),
        )

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "mode": self.mode,
            "non_authoritative": self.non_authoritative,
            "arbitration_input_snapshot_id": self.arbitration_input_snapshot_id,
            "arbitration_input_hash": self.arbitration_input_hash,
            "proposal_snapshot_id": self.proposal_snapshot_id,
            "computation_result_id": self.computation_result_id,
            "computation_result_hash": self.computation_result_hash,
            "output_status": self.output_status,
            "would_select_action_hash": self.would_select_action_hash,
            "would_select_by_group": dict(self.would_select_by_group),
            "would_select_by_component": dict(self.would_select_by_component),
            "evaluated_at": self.evaluated_at,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ArbiterShadowStatusEvent:
    status_event_id: str
    status: str
    reason: str
    emission_id: str | None
    occurred_at: float
    non_authoritative: bool
    semantic_content_hash: str

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "status_event_id": self.status_event_id,
            "status": self.status,
            "reason": self.reason,
            "emission_id": self.emission_id,
            "occurred_at": self.occurred_at,
            "non_authoritative": self.non_authoritative,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ArbiterShadowStep:
    status: str
    emission_id: str | None
    emission: PolicyActionEmission | None = None
    admission_receipt: ProposalAdmissionReceipt | None = None
    arbitration_input_snapshot: ArbitrationInputSnapshot | None = None
    computation: ArbitrationResult | None = None
    evaluation: ArbitrationEvaluationEvent | None = None
    status_event: ArbiterShadowStatusEvent | None = None

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "emission_id": self.emission_id,
            "proposal_id": (
                None
                if self.admission_receipt is None
                else self.admission_receipt.proposal_id
            ),
            "proposal_admitted": (
                None
                if self.admission_receipt is None
                else self.admission_receipt.accepted
            ),
            "proposal_rejection_code": (
                None
                if self.admission_receipt is None
                else self.admission_receipt.rejection_code
            ),
            "arbitration_input_snapshot_id": (
                None
                if self.arbitration_input_snapshot is None
                else self.arbitration_input_snapshot.arbitration_input_snapshot_id
            ),
            "evaluation": (
                None if self.evaluation is None else self.evaluation.to_debug_payload()
            ),
            "status_event": (
                None if self.status_event is None else self.status_event.to_debug_payload()
            ),
        }


@dataclass(frozen=True)
class LegacyExecutionObservationEvent:
    event_id: str
    owner_kind: str
    non_authoritative_shadow_reference: bool
    emission_id: str
    evaluation_id: str | None
    legacy_safe_representation_trace_id: str | None
    command_present: bool
    execution_accepted: bool
    execution_mode: str
    dispatch_attempted: bool
    dispatch_completed: bool
    dispatch_error_type: str | None
    component_parity: ActionComponentParityReceipt | None
    observed_at: float
    semantic_content_hash: str

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "owner_kind": self.owner_kind,
            "non_authoritative_shadow_reference": self.non_authoritative_shadow_reference,
            "emission_id": self.emission_id,
            "evaluation_id": self.evaluation_id,
            "legacy_safe_representation_trace_id": self.legacy_safe_representation_trace_id,
            "command_present": self.command_present,
            "execution_accepted": self.execution_accepted,
            "execution_mode": self.execution_mode,
            "dispatch_attempted": self.dispatch_attempted,
            "dispatch_completed": self.dispatch_completed,
            "dispatch_error_type": self.dispatch_error_type,
            "component_parity": (
                None
                if self.component_parity is None
                else {
                    "matched": self.component_parity.matched,
                    "compared_components": self.component_parity.compared_components,
                    "mismatched_components": self.component_parity.mismatched_components,
                    "max_abs_error": self.component_parity.max_abs_error,
                    "absolute_tolerance": self.component_parity.absolute_tolerance,
                    "semantic_content_hash": self.component_parity.semantic_content_hash,
                }
            ),
            "observed_at": self.observed_at,
            "semantic_content_hash": self.semantic_content_hash,
        }


def _component_payload(action: RobotAction, effector: str) -> np.ndarray | None:
    side, kind = effector.split("_", 1)
    if kind == "arm":
        value = action.arm_pose_by_side.get(side)
        if value is None:
            value = action.arm_qpos_by_side.get(side)
    else:
        value = action.hand_qpos_by_side.get(side)
        if value is None and side in action.end_effector_targets_by_side:
            value = action.end_effector_targets_by_side[side].values
    return None if value is None else np.asarray(value).reshape(-1)


def compare_action_components(
    expected: RobotAction,
    observed: RobotAction,
    *,
    atol: float = 1e-6,
) -> ActionComponentParityReceipt:
    compared = tuple(dict.fromkeys((*expected.active_effectors, *observed.active_effectors)))
    mismatched: list[str] = []
    max_error = 0.0
    if expected.action_space != observed.action_space:
        mismatched.append("action_space")
    for component in compared:
        expected_value = _component_payload(expected, component)
        observed_value = _component_payload(observed, component)
        if expected_value is None or observed_value is None or expected_value.shape != observed_value.shape:
            mismatched.append(component)
            continue
        difference = float(np.max(np.abs(expected_value - observed_value), initial=0.0))
        max_error = max(max_error, difference)
        if not np.allclose(expected_value, observed_value, rtol=0.0, atol=atol):
            mismatched.append(component)
        if expected.frame_by_effector.get(component) != observed.frame_by_effector.get(component):
            mismatched.append(f"{component}:frame")
        if (
            expected.controlled_frame_by_effector.get(component)
            != observed.controlled_frame_by_effector.get(component)
        ):
            mismatched.append(f"{component}:controlled_frame")
    mismatched = list(dict.fromkeys(mismatched))
    payload = {
        "matched": not mismatched,
        "compared_components": compared,
        "mismatched_components": tuple(mismatched),
        "max_abs_error": max_error,
        "atol": float(atol),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=ACTION_COMPONENT_PARITY_HASH_DOMAIN_V1,
    )
    return ActionComponentParityReceipt(
        matched=not mismatched,
        compared_components=compared,
        mismatched_components=tuple(mismatched),
        max_abs_error=max_error,
        absolute_tolerance=float(atol),
        semantic_content_hash=semantic_hash,
    )


class PassivePolicyArbiterShadow:
    """Single-policy passive adapter around the shared proposal/Arbiter core."""

    def __init__(
        self,
        config: PolicyArbiterShadowConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        descriptor = ClockDomainDescriptor(
            clock_domain_id=config.clock_domain_id,
            runtime_session_id=config.runtime_session_id,
            boot_id=config.boot_id,
            reset_epoch=0,
        )
        self.coordinator = ControlAdmissionCoordinator(
            config.control_domain_id,
            clock_domain=descriptor,
            clock=clock,
        )
        self.producer_epochs = ProducerEpochRegistry(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
        )
        self.producer_epoch_activation = self.producer_epochs.activate(
            producer_id=config.producer_id,
            producer_epoch=config.producer_epoch,
            produced_at=self.now(),
            produced_clock_domain_id=config.clock_domain_id,
            reason="m3_shadow_startup",
        )
        self.component_catalog = (
            config.component_catalog
            if config.component_catalog is not None
            else ActionComponentCatalog(
                catalog_id="single-policy-shadow-components-v1",
                action_space=config.action_space,
                components=tuple(
                    ActionComponentSpec(
                        component_id=effector,
                        selection_group_id=effector,
                        effector_name=effector,
                        coupled=False,
                    )
                    for effector in config.active_effectors
                ),
            )
        )
        priorities = {
            group: {config.producer_id: 100}
            for group in self.component_catalog.selection_group_ids
        }
        self.authority_policy = AuthorityPolicySpec(
            policy_id="single-policy-shadow-fixed-owner-v1",
            priority_by_group=priorities,
            fallback_by_group={group: "no_action" for group in priorities},
            shadow_owner_by_group={group: config.producer_id for group in priorities},
        )
        self.proposal_buffer = ProposalBuffer(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
            producer_epochs=self.producer_epochs,
            capacity=config.proposal_capacity,
        )
        self.authority_store = AuthorityStore(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
            policy=self.authority_policy,
            initial_mode="shadow",
        )
        self.execution_gate_store = ExecutionGateStore(
            runtime_session_id=config.runtime_session_id,
            coordinator=self.coordinator,
            initial_status="ready",
        )
        self.scheduler = DecisionScheduler(
            DecisionScheduleSpec(
                spec_id="single-policy-shadow-v1",
                schedule_mode="event_limited",
                decision_hz=config.decision_hz,
            ),
            clock=clock,
        )
        self.arbiter = ActionArbiter()
        self._next_proposal_sequence = 0
        self._next_gate_sequence = 0
        self._gate_status = "ready"

    def now(self) -> float:
        timestamp = float(self._clock())
        if not math.isfinite(timestamp):
            raise ValueError("Shadow monotonic clock returned a non-finite value.")
        return timestamp

    def update_execution_gate(
        self,
        status: str,
        *,
        reason: str,
        produced_at: float | None = None,
    ) -> ExecutionGateEvent | None:
        normalized = str(status).strip().lower()
        if normalized not in EXECUTION_GATE_STATUSES:
            raise ValueError("Unsupported shadow execution-gate status.")
        if normalized == self._gate_status:
            return None
        timestamp = self.now() if produced_at is None else float(produced_at)
        event = self.execution_gate_store.admit(
            ExecutionGateEventDraft(
                runtime_session_id=self.config.runtime_session_id,
                control_domain_id=self.config.control_domain_id,
                sequence=self._next_gate_sequence,
                status=normalized,
                produced_at=timestamp,
                produced_clock_domain_id=self.config.clock_domain_id,
                reason=_required(reason, name="gate reason"),
            )
        )
        self._next_gate_sequence += 1
        self._gate_status = normalized
        return event

    def status(
        self,
        status: str,
        *,
        reason: str,
        emission_id: str | None = None,
        emission: PolicyActionEmission | None = None,
        occurred_at: float | None = None,
    ) -> ArbiterShadowStep:
        timestamp = self.now() if occurred_at is None else float(occurred_at)
        if not math.isfinite(timestamp):
            raise ValueError("Shadow status time must be finite.")
        payload = {
            "status": _required(status, name="shadow status"),
            "reason": _required(reason, name="shadow status reason"),
            "emission_id": emission_id,
            "occurred_at": timestamp,
            "non_authoritative": True,
        }
        semantic_hash = canonical_content_hash(payload, domain=SHADOW_STATUS_HASH_DOMAIN_V1)
        event = ArbiterShadowStatusEvent(
            status_event_id=f"arbiter-shadow-status:{semantic_hash}",
            **payload,
            semantic_content_hash=semantic_hash,
        )
        return ArbiterShadowStep(
            status=event.status,
            emission_id=emission_id,
            emission=emission,
            status_event=event,
        )

    def evaluate_emission(self, emission: PolicyActionEmission) -> ArbiterShadowStep:
        if not isinstance(emission, PolicyActionEmission):
            raise TypeError("M3 shadow accepts only PolicyActionEmission.")
        if emission.effective_clock_domain_id != self.config.clock_domain_id:
            return self.status(
                "emission_rejected",
                reason="effective_clock_domain_mismatch",
                emission_id=emission.emission_id,
                emission=emission,
            )
        action = emission.pre_safety_robot_action
        if (
            action is None
            or emission.source_validation is None
            or not emission.source_validation.accepted
            or emission.pre_safety_faults
        ):
            return self.status(
                "emission_rejected",
                reason="pre_safety_action_unavailable_or_invalid",
                emission_id=emission.emission_id,
                emission=emission,
            )
        coverage = tuple(
            component
            for component in action.active_effectors
            if component in self.component_catalog.component_by_id
        )
        if not coverage or len(coverage) != len(action.active_effectors):
            return self.status(
                "emission_rejected",
                reason="component_catalog_mismatch",
                emission_id=emission.emission_id,
                emission=emission,
            )
        sequence = self._next_proposal_sequence
        self._next_proposal_sequence += 1
        draft = ActionProposalDraft(
                runtime_session_id=self.config.runtime_session_id,
                control_domain_id=self.config.control_domain_id,
                embodiment_instance_id=self.config.embodiment_instance_id,
                producer_kind="policy",
                producer_id=self.config.producer_id,
                producer_epoch=self.config.producer_epoch,
                sequence=sequence,
                action=project_robot_action_v1_to_component_action(
                    action,
                    action_schema=self.config.action_schema,
                ),
                produced_at=emission.produced_at,
                produced_clock_domain_id=emission.produced_clock_domain_id,
                effective_from=emission.effective_from,
                effective_until=emission.effective_until,
                target_effective_at=emission.target_effective_at,
                effective_clock_domain_id=emission.effective_clock_domain_id,
                authority_priority=100,
                fallback_policy="no_action",
                policy_node_id=emission.policy_node_id,
                inference_id=emission.inference_id,
                policy_output_id=emission.policy_output_id,
                chunk_id=str(emission.chunk_id),
                chunk_keyframe_index=emission.chunk_keyframe_index,
                representation_trace_id=(
                    None
                    if emission.representation_receipt is None
                    else emission.representation_receipt.representation_trace_id
                ),
                audit_metadata={"policy_action_emission_id": emission.emission_id},
        )
        return self.evaluate_proposal_draft(
            draft,
            source_event_id=emission.emission_id,
            emission=emission,
        )

    def evaluate_proposal_draft(
        self,
        draft: ActionProposalDraft,
        *,
        source_event_id: str,
        emission: PolicyActionEmission | None = None,
    ) -> ArbiterShadowStep:
        """Admit and evaluate a typed draft without acquiring dispatch authority.

        M3 policy emissions and M5 human emissions share this exact decision
        path.  The caller remains responsible for source-specific adaptation
        and source-validation facts.
        """

        source_id = _required(source_event_id, name="source_event_id")
        if draft.runtime_session_id != self.config.runtime_session_id:
            return self.status(
                "proposal_rejected",
                reason="runtime_session_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        if draft.control_domain_id != self.config.control_domain_id:
            return self.status(
                "proposal_rejected",
                reason="control_domain_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        if draft.embodiment_instance_id != self.config.embodiment_instance_id:
            return self.status(
                "proposal_rejected",
                reason="embodiment_instance_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        if draft.producer_id != self.config.producer_id:
            return self.status(
                "proposal_rejected",
                reason="producer_id_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        if draft.producer_epoch != self.config.producer_epoch:
            return self.status(
                "proposal_rejected",
                reason="producer_epoch_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        if (
            draft.action.action_schema.semantic_content_hash
            != self.config.action_schema.semantic_content_hash
        ):
            return self.status(
                "proposal_rejected",
                reason="action_schema_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        coverage = draft.action.component_coverage
        catalog = self.config.action_schema.component_catalog
        if not coverage or any(
            component not in catalog.component_by_id
            for component in coverage
        ):
            return self.status(
                "proposal_rejected",
                reason="component_coverage_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        selection_groups = tuple(
            dict.fromkeys(
                catalog.component_by_id[
                    component_id
                ].selection_group_id
                for component_id in coverage
            )
        )
        if not selection_groups or any(
            group not in catalog.selection_group_ids
            for group in selection_groups
        ):
            return self.status(
                "proposal_rejected",
                reason="selection_group_mismatch",
                emission_id=source_id,
                emission=emission,
            )
        receipt = self.proposal_buffer.admit(draft)
        if not receipt.accepted:
            step = self.status(
                "proposal_rejected",
                reason=str(receipt.rejection_code or "proposal_admission_failed"),
                emission_id=source_id,
                emission=emission,
            )
            return ArbiterShadowStep(
                status=step.status,
                emission_id=source_id,
                emission=emission,
                admission_receipt=receipt,
                status_event=step.status_event,
            )

        assert receipt.proposal_id is not None
        self.scheduler.trigger(receipt.proposal_id, "proposal_admitted")
        decision_now = self.now()
        snapshot = capture_arbitration_input(
            coordinator=self.coordinator,
            proposal_buffer=self.proposal_buffer,
            producer_epochs=self.producer_epochs,
            authority_store=self.authority_store,
            execution_gate_store=self.execution_gate_store,
            scheduler=self.scheduler,
            embodiment_instance_id=self.config.embodiment_instance_id,
            action_schema=self.config.action_schema,
            active_selection_groups=(
                self.config.action_schema.component_catalog.selection_group_ids
            ),
            now=decision_now,
        )
        if snapshot is None:
            step = self.status(
                "coalesced_before_decision",
                reason="decision_rate_gate_not_due",
                emission_id=source_id,
                emission=emission,
                occurred_at=decision_now,
            )
            return ArbiterShadowStep(
                status=step.status,
                emission_id=source_id,
                emission=emission,
                admission_receipt=receipt,
                status_event=step.status_event,
            )

        computation = self.arbiter.arbitrate(snapshot)
        evaluated_at = self.now()
        payload = {
            "mode": ARBITER_SHADOW_MODE_SHADOW,
            "non_authoritative": True,
            "arbitration_input_snapshot_id": snapshot.arbitration_input_snapshot_id,
            "arbitration_input_hash": snapshot.semantic_content_hash,
            "proposal_snapshot_id": snapshot.proposal_snapshot.proposal_snapshot_id,
            "computation_result_id": computation.result_id,
            "computation_result_hash": computation.semantic_content_hash,
            "output_status": computation.output_status,
            "would_select_action_hash": computation.selected_action_hash,
            "would_select_by_group": computation.selected_by_group,
            "would_select_by_component": computation.selected_by_component,
            "evaluated_at": evaluated_at,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=ARBITRATION_EVALUATION_HASH_DOMAIN_V1,
        )
        evaluation = ArbitrationEvaluationEvent(
            evaluation_id=f"arbitration-evaluation:{semantic_hash}",
            **payload,
            would_select_action=computation.composed_candidate,
            semantic_content_hash=semantic_hash,
        )
        self.proposal_buffer.prune_expired(
            at_or_before=snapshot.decision_schedule_tick.target_effective_at
        )
        return ArbiterShadowStep(
            status="evaluated",
            emission_id=source_id,
            emission=emission,
            admission_receipt=receipt,
            arbitration_input_snapshot=snapshot,
            computation=computation,
            evaluation=evaluation,
        )

    def observe_legacy_execution(
        self,
        *,
        emission: PolicyActionEmission,
        shadow_step: ArbiterShadowStep | None,
        canonical_action: Any,
        execution: Any | None,
        dispatch_error: BaseException | None,
        observed_at: float | None = None,
    ) -> LegacyExecutionObservationEvent:
        timestamp = self.now() if observed_at is None else float(observed_at)
        if not math.isfinite(timestamp):
            raise ValueError("Legacy execution observation time must be finite.")
        evaluation = None if shadow_step is None else shadow_step.evaluation
        parity = None
        if evaluation is not None and evaluation.would_select_action is not None:
            if emission.pre_safety_robot_action is not None:
                parity = compare_action_components(
                    emission.pre_safety_robot_action,
                    materialize_robot_action_v1(
                        evaluation.would_select_action,
                        timestamp=emission.pre_safety_robot_action.timestamp,
                        source="arbiter_shadow_expected",
                    ),
                )
        representation = getattr(canonical_action, "representation_receipt", None)
        command = getattr(canonical_action, "command", None)
        payload = {
            "owner_kind": "legacy_control_path",
            "non_authoritative_shadow_reference": True,
            "emission_id": emission.emission_id,
            "evaluation_id": None if evaluation is None else evaluation.evaluation_id,
            "legacy_safe_representation_trace_id": (
                None if representation is None else representation.representation_trace_id
            ),
            "command_present": bool(command is not None and getattr(command, "commands", {})),
            "execution_accepted": bool(
                execution is not None and getattr(execution, "accepted", False)
            ),
            "execution_mode": (
                "not_attempted" if execution is None else str(execution.mode)
            ),
            "dispatch_attempted": bool(
                execution is not None and getattr(execution, "dispatch_attempted", False)
            ),
            "dispatch_completed": bool(
                execution is not None and getattr(execution, "dispatch_completed", False)
            ),
            "dispatch_error_type": (
                None if dispatch_error is None else type(dispatch_error).__name__
            ),
            "component_parity_hash": (
                None if parity is None else parity.semantic_content_hash
            ),
            "observed_at": timestamp,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=LEGACY_EXECUTION_OBSERVATION_HASH_DOMAIN_V1,
        )
        return LegacyExecutionObservationEvent(
            event_id=f"legacy-execution-observation:{semantic_hash}",
            **{key: value for key, value in payload.items() if key != "component_parity_hash"},
            component_parity=parity,
            semantic_content_hash=semantic_hash,
        )

    def observe_legacy_action(
        self,
        *,
        source_event_id: str,
        shadow_step: ArbiterShadowStep | None,
        observed_action: RobotAction | None,
        command_present: bool,
        execution_accepted: bool,
        execution_mode: str,
        dispatch_attempted: bool,
        dispatch_completed: bool,
        dispatch_error_type: str | None = None,
        legacy_safe_representation_trace_id: str | None = None,
        observed_at: float | None = None,
    ) -> LegacyExecutionObservationEvent:
        """Observe a source-neutral legacy result without dispatching anything."""

        source_id = _required(source_event_id, name="source_event_id")
        timestamp = self.now() if observed_at is None else float(observed_at)
        if not math.isfinite(timestamp):
            raise ValueError("Legacy execution observation time must be finite.")
        evaluation = None if shadow_step is None else shadow_step.evaluation
        parity = None
        if (
            evaluation is not None
            and evaluation.would_select_action is not None
            and observed_action is not None
        ):
            parity = compare_action_components(
                materialize_robot_action_v1(
                    evaluation.would_select_action,
                    timestamp=observed_action.timestamp,
                    source="arbiter_shadow_expected",
                ),
                observed_action,
            )
        payload = {
            "owner_kind": "legacy_control_path",
            "non_authoritative_shadow_reference": True,
            "emission_id": source_id,
            "evaluation_id": None if evaluation is None else evaluation.evaluation_id,
            "legacy_safe_representation_trace_id": (
                legacy_safe_representation_trace_id
            ),
            "command_present": bool(command_present),
            "execution_accepted": bool(execution_accepted),
            "execution_mode": _required(execution_mode, name="execution_mode"),
            "dispatch_attempted": bool(dispatch_attempted),
            "dispatch_completed": bool(dispatch_completed),
            "dispatch_error_type": (
                None if dispatch_error_type is None else str(dispatch_error_type)
            ),
            "component_parity_hash": (
                None if parity is None else parity.semantic_content_hash
            ),
            "observed_at": timestamp,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=LEGACY_EXECUTION_OBSERVATION_HASH_DOMAIN_V1,
        )
        return LegacyExecutionObservationEvent(
            event_id=f"legacy-execution-observation:{semantic_hash}",
            **{key: value for key, value in payload.items() if key != "component_parity_hash"},
            component_parity=parity,
            semantic_content_hash=semantic_hash,
        )


__all__ = [
    "ARBITER_SHADOW_MODE_OFF",
    "ARBITER_SHADOW_MODE_SHADOW",
    "ARBITER_SHADOW_MODES_M3",
    "ActionComponentParityReceipt",
    "ArbiterShadowStatusEvent",
    "ArbiterShadowStep",
    "ArbitrationEvaluationEvent",
    "LegacyExecutionObservationEvent",
    "PassivePolicyArbiterShadow",
    "PolicyArbiterShadowConfig",
    "compare_action_components",
]
