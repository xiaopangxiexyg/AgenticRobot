"""Observe legacy position-reference derivation without granting authority.

The legacy position worker does not implement the shared M1C execution epoch
or a canonical reference-segment admission point.  These types therefore keep
worker facts, caller-supplied context, and enforced execution semantics
separate.  Nothing in this module can create a canonical
``ReferenceSegmentAdmissionReceipt`` or safety-anchor commit evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import queue
import threading
from typing import Callable, Mapping, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control import ResampledMotionReference


LEGACY_POSITION_GENERATION_NAMESPACE_V1 = (
    "position_deploy_executor_local_generation_v1"
)
LEGACY_REFERENCE_ENDPOINT_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_endpoint_observation.v1"
)
LEGACY_REFERENCE_SEGMENT_KEY_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_segment_key.v1"
)
LEGACY_REFERENCE_SEGMENT_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_segment_observation.v1"
)
LEGACY_REFERENCE_FRAME_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_frame_observation.v1"
)
LEGACY_REFERENCE_RESOLUTION_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_resolution_observation.v1"
)
LEGACY_REFERENCE_RESOLUTION_FAILURE_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_resolution_failure_observation.v1"
)
LEGACY_REFERENCE_TERMINAL_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_segment_terminal_observation.v1"
)
EXECUTION_CONTEXT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.execution_context_binding_evidence.v1"
)
LEGACY_REFERENCE_PARENT_PROVENANCE_HASH_DOMAIN_V1 = (
    "dexteleop.legacy_reference_parent_provenance.v1"
)
BOUND_LEGACY_REFERENCE_SEGMENT_HASH_DOMAIN_V1 = (
    "dexteleop.bound_legacy_reference_segment_observation.v1"
)

LEGACY_REFERENCE_ENDPOINT_ROLES_V1 = frozenset({"start", "end"})
LEGACY_REFERENCE_MODALITIES_V1 = frozenset({"pose", "qpos"})
LEGACY_REFERENCE_DERIVATION_KINDS_V1 = frozenset(
    {"interpolated_segment", "exact_keyframe", "causal_hold"}
)
EXECUTION_CONTEXT_ENFORCEMENT_LEVELS_V1 = (
    "context_only",
    "creation_checked",
    "dequeue_checked",
    "pre_backend_checked",
)
LEGACY_REFERENCE_PARENT_SOURCE_KINDS_V1 = frozenset(
    {
        "legacy_post_safety_keyframe",
        "legacy_position_keyframe",
        "shared_safe_result",
        "measured_rebase",
        "canonical_execution_target",
    }
)
LEGACY_REFERENCE_FAILURE_STAGES_V1 = frozenset(
    {"planner", "resolver", "command_construction", "final_local_guard"}
)


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


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _optional_hash(value: str | None, *, name: str) -> str | None:
    return None if value is None else _required_text(value, name=name)


@dataclass(frozen=True)
class LegacyReferenceSourceFact:
    """Immutable source fact retained when a keyframe enters the resampler."""

    source_index: int
    source_timestamp: float
    source_stage: str
    target_payload_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_index",
            _non_negative_int(self.source_index, name="source_index"),
        )
        object.__setattr__(
            self,
            "source_timestamp",
            _finite_time(self.source_timestamp, name="source_timestamp"),
        )
        _required_text(self.source_stage, name="source_stage")
        _required_text(self.target_payload_hash, name="target_payload_hash")


@dataclass(frozen=True)
class LegacyReferenceEndpointObservation:
    role: str
    modality: str
    source_index: int
    source_timestamp: float
    source_stage: str
    target_payload_hash: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.role not in LEGACY_REFERENCE_ENDPOINT_ROLES_V1:
            raise ValueError("Unsupported legacy reference endpoint role.")
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy reference modality.")
        _non_negative_int(self.source_index, name="source_index")
        _finite_time(self.source_timestamp, name="source_timestamp")
        _required_text(self.source_stage, name="source_stage")
        _required_text(self.target_payload_hash, name="target_payload_hash")
        _required_text(self.semantic_content_hash, name="semantic_content_hash")


def build_legacy_reference_endpoint_observation(
    *,
    role: str,
    modality: str,
    source_fact: LegacyReferenceSourceFact,
) -> LegacyReferenceEndpointObservation:
    payload = {
        "role": _required_text(role, name="role"),
        "modality": _required_text(modality, name="modality"),
        "source_index": int(source_fact.source_index),
        "source_timestamp": float(source_fact.source_timestamp),
        "source_stage": source_fact.source_stage,
        "target_payload_hash": source_fact.target_payload_hash,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=LEGACY_REFERENCE_ENDPOINT_HASH_DOMAIN_V1,
    )
    return LegacyReferenceEndpointObservation(
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class LegacyPositionReferenceSegmentObservation:
    observation_id: str
    segment_key: str
    worker_instance_id: str
    executor_instance_id: str
    legacy_worker_generation: int
    legacy_generation_namespace: str
    modality: str
    endpoint_observations: tuple[LegacyReferenceEndpointObservation, ...]
    interpolation_contract_hash: str
    scheduled_from: float
    scheduled_until: float
    observed_segment_created_at: float
    formation_semantics: str
    observation_authority: str
    non_authoritative: bool
    anchor_commit_eligible: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "segment_key",
            "worker_instance_id",
            "executor_instance_id",
            "legacy_generation_namespace",
            "interpolation_contract_hash",
            "formation_semantics",
            "observation_authority",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.legacy_generation_namespace != LEGACY_POSITION_GENERATION_NAMESPACE_V1:
            raise ValueError("Legacy generation must keep its executor-local namespace.")
        _non_negative_int(
            self.legacy_worker_generation,
            name="legacy_worker_generation",
        )
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy reference modality.")
        endpoints = tuple(self.endpoint_observations)
        if len(endpoints) != 2 or tuple(item.role for item in endpoints) != (
            "start",
            "end",
        ):
            raise ValueError("Legacy interpolation segments require ordered START/END.")
        if endpoints[0].source_index == endpoints[1].source_index:
            raise ValueError("Exact/hold references must not masquerade as segments.")
        if self.scheduled_until <= self.scheduled_from:
            raise ValueError("Legacy segment interval must be [from, until).")
        if self.formation_semantics != (
            "first_observed_reference_derived_from_exact_parent_pair"
        ):
            raise ValueError("Unsupported legacy segment formation semantics.")
        if self.observation_authority != "legacy_observed":
            raise ValueError("Legacy segment observation authority must remain legacy_observed.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Legacy observations cannot become authority or anchor evidence.")


@dataclass(frozen=True)
class LegacyPositionReferenceFrameObservation:
    frame_observation_id: str
    source_segment_observation_id: str | None
    worker_instance_id: str
    executor_instance_id: str
    legacy_worker_generation: int
    legacy_generation_namespace: str
    modality: str
    legacy_reference_sequence: int
    derivation_kind: str
    scheduled_for: float
    query_timestamp: float
    emitted_at: float
    interpolation_alpha: float
    start_source_index: int
    end_source_index: int
    reference_payload_hash: str
    observation_stage: str
    solver_evidence_status: str
    non_authoritative: bool
    anchor_commit_eligible: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "frame_observation_id",
            "worker_instance_id",
            "executor_instance_id",
            "legacy_generation_namespace",
            "reference_payload_hash",
            "observation_stage",
            "solver_evidence_status",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _optional_hash(
            self.source_segment_observation_id,
            name="source_segment_observation_id",
        )
        if self.legacy_generation_namespace != LEGACY_POSITION_GENERATION_NAMESPACE_V1:
            raise ValueError("Legacy generation must keep its executor-local namespace.")
        _non_negative_int(
            self.legacy_worker_generation,
            name="legacy_worker_generation",
        )
        _non_negative_int(
            self.legacy_reference_sequence,
            name="legacy_reference_sequence",
        )
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy reference modality.")
        if self.derivation_kind not in LEGACY_REFERENCE_DERIVATION_KINDS_V1:
            raise ValueError("Unsupported legacy reference derivation kind.")
        if (
            self.derivation_kind == "interpolated_segment"
            and self.source_segment_observation_id is None
        ):
            raise ValueError("Interpolated frames must reference their observed segment.")
        if (
            self.derivation_kind != "interpolated_segment"
            and self.source_segment_observation_id is not None
        ):
            raise ValueError("Exact/hold frames must not reference a fabricated segment.")
        alpha = float(self.interpolation_alpha)
        if not math.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("interpolation_alpha must be in [0, 1].")
        if self.observation_stage != "resampled_pre_resolver":
            raise ValueError("e2-b frame observations are pre-resolver facts.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Legacy frame observations cannot become anchor evidence.")


@dataclass(frozen=True)
class LegacyPositionReferenceObservationBatch:
    segment_observation: LegacyPositionReferenceSegmentObservation | None
    frame_observation: LegacyPositionReferenceFrameObservation


@dataclass(frozen=True)
class LegacyReferenceResolutionDerivation:
    """Post-resolver evidence joined to the same immutable reference derivation."""

    reference_derivation: LegacyReferenceDerivation
    resolved_at: float
    resolver_evidence_hash: str
    planner_evidence_status: str
    planner_input_hash: str | None
    planner_output_hash: str | None
    planner_evidence_hash: str | None
    measured_state_evidence_status: str
    measured_state_evidence_ids: tuple[str, ...]
    command_intent_hash: str
    local_pre_backend_generation_check_passed: bool
    local_pre_backend_check_observed_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.reference_derivation, LegacyReferenceDerivation):
            raise TypeError("reference_derivation must be a LegacyReferenceDerivation.")
        _finite_time(self.resolved_at, name="resolved_at")
        _required_text(self.resolver_evidence_hash, name="resolver_evidence_hash")
        if self.planner_evidence_status not in {
            "not_applicable",
            "legacy_collision_planner_accepted",
        }:
            raise ValueError("Unsupported planner evidence status.")
        planner_input = _optional_hash(
            self.planner_input_hash,
            name="planner_input_hash",
        )
        planner_output = _optional_hash(
            self.planner_output_hash,
            name="planner_output_hash",
        )
        planner_evidence = _optional_hash(
            self.planner_evidence_hash,
            name="planner_evidence_hash",
        )
        if self.planner_evidence_status == "not_applicable":
            if any(
                value is not None
                for value in (planner_input, planner_output, planner_evidence)
            ):
                raise ValueError("Planner N/A evidence cannot carry planner hashes.")
        elif any(value is None for value in (planner_input, planner_output, planner_evidence)):
            raise ValueError("Accepted collision planner evidence requires input/output hashes.")
        if self.measured_state_evidence_status not in {
            "not_applicable",
            "legacy_planner_state_not_typed",
            "typed_measurement_ids",
        }:
            raise ValueError("Unsupported measured-state evidence status.")
        evidence_ids = tuple(
            _required_text(value, name="measured_state_evidence_id")
            for value in self.measured_state_evidence_ids
        )
        if self.measured_state_evidence_status == "typed_measurement_ids":
            if not evidence_ids:
                raise ValueError("Typed measurement evidence requires IDs.")
        elif evidence_ids:
            raise ValueError("Untyped/N/A measurement evidence cannot carry typed IDs.")
        object.__setattr__(self, "measured_state_evidence_ids", evidence_ids)
        _required_text(self.command_intent_hash, name="command_intent_hash")
        if not self.local_pre_backend_generation_check_passed:
            raise ValueError("A successful resolution observation requires the local final check.")
        checked_at = _finite_time(
            self.local_pre_backend_check_observed_at,
            name="local_pre_backend_check_observed_at",
        )
        if checked_at < float(self.resolved_at):
            raise ValueError("The local pre-backend check cannot precede resolution.")


@dataclass(frozen=True)
class LegacyPositionReferenceResolutionObservation:
    resolution_observation_id: str
    source_frame_observation_id: str
    worker_instance_id: str
    executor_instance_id: str
    legacy_worker_generation: int
    legacy_generation_namespace: str
    modality: str
    resolved_at: float
    observation_stage: str
    resolver_evidence_hash: str
    planner_evidence_status: str
    planner_input_hash: str | None
    planner_output_hash: str | None
    planner_evidence_hash: str | None
    measured_state_evidence_status: str
    measured_state_evidence_ids: tuple[str, ...]
    command_intent_hash: str
    local_pre_backend_generation_check_passed: bool
    local_pre_backend_check_observed_at: float
    execution_context_enforcement: str
    backend_accepted: bool
    non_authoritative: bool
    anchor_commit_eligible: bool
    admission_receipt_id: None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "resolution_observation_id",
            "source_frame_observation_id",
            "worker_instance_id",
            "executor_instance_id",
            "legacy_generation_namespace",
            "observation_stage",
            "resolver_evidence_hash",
            "planner_evidence_status",
            "measured_state_evidence_status",
            "command_intent_hash",
            "execution_context_enforcement",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.legacy_generation_namespace != LEGACY_POSITION_GENERATION_NAMESPACE_V1:
            raise ValueError("Resolution must retain the executor-local namespace.")
        _non_negative_int(
            self.legacy_worker_generation,
            name="legacy_worker_generation",
        )
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy reference modality.")
        if self.observation_stage != "resolved_pre_dispatch":
            raise ValueError("Unsupported legacy resolution observation stage.")
        _finite_time(self.resolved_at, name="resolved_at")
        checked_at = _finite_time(
            self.local_pre_backend_check_observed_at,
            name="local_pre_backend_check_observed_at",
        )
        if checked_at < float(self.resolved_at):
            raise ValueError("The local pre-backend check cannot precede resolution.")
        if not self.local_pre_backend_generation_check_passed:
            raise ValueError("Resolution observation requires a passed local check.")
        planner_hashes = (
            self.planner_input_hash,
            self.planner_output_hash,
            self.planner_evidence_hash,
        )
        if self.planner_evidence_status == "not_applicable":
            if any(value is not None for value in planner_hashes):
                raise ValueError("Planner N/A observation cannot carry planner hashes.")
        elif self.planner_evidence_status == "legacy_collision_planner_accepted":
            if any(value is None for value in planner_hashes):
                raise ValueError("Accepted planner observation requires all hashes.")
        else:
            raise ValueError("Unsupported planner evidence status.")
        evidence_ids = tuple(
            _required_text(value, name="measured_state_evidence_id")
            for value in self.measured_state_evidence_ids
        )
        if self.measured_state_evidence_status == "typed_measurement_ids":
            if not evidence_ids:
                raise ValueError("Typed measurement evidence requires IDs.")
        elif self.measured_state_evidence_status in {
            "not_applicable",
            "legacy_planner_state_not_typed",
        }:
            if evidence_ids:
                raise ValueError("Untyped/N/A measurement evidence cannot carry IDs.")
        else:
            raise ValueError("Unsupported measured-state evidence status.")
        object.__setattr__(self, "measured_state_evidence_ids", evidence_ids)
        if self.execution_context_enforcement != "executor_local_generation_only":
            raise ValueError("Legacy resolution cannot claim shared M1C enforcement.")
        if self.backend_accepted:
            raise ValueError("Pre-dispatch resolution cannot claim backend acceptance.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Legacy resolution cannot become authority or anchor evidence.")
        if self.admission_receipt_id is not None:
            raise ValueError("Legacy resolution cannot carry admission receipts.")


def build_legacy_reference_resolution_observation(
    derivation: LegacyReferenceResolutionDerivation,
    *,
    worker_instance_id: str,
    executor_instance_id: str,
) -> LegacyPositionReferenceResolutionObservation:
    frame = build_legacy_reference_observation_batch(
        derivation.reference_derivation,
        worker_instance_id=worker_instance_id,
        executor_instance_id=executor_instance_id,
        include_segment_observation=False,
    ).frame_observation
    payload = {
        "source_frame_observation_id": frame.frame_observation_id,
        "worker_instance_id": _required_text(
            worker_instance_id,
            name="worker_instance_id",
        ),
        "executor_instance_id": _required_text(
            executor_instance_id,
            name="executor_instance_id",
        ),
        "legacy_worker_generation": (
            derivation.reference_derivation.legacy_worker_generation
        ),
        "legacy_generation_namespace": LEGACY_POSITION_GENERATION_NAMESPACE_V1,
        "modality": derivation.reference_derivation.modality,
        "resolved_at": derivation.resolved_at,
        "observation_stage": "resolved_pre_dispatch",
        "resolver_evidence_hash": derivation.resolver_evidence_hash,
        "planner_evidence_status": derivation.planner_evidence_status,
        "planner_input_hash": derivation.planner_input_hash,
        "planner_output_hash": derivation.planner_output_hash,
        "planner_evidence_hash": derivation.planner_evidence_hash,
        "measured_state_evidence_status": derivation.measured_state_evidence_status,
        "measured_state_evidence_ids": derivation.measured_state_evidence_ids,
        "command_intent_hash": derivation.command_intent_hash,
        "local_pre_backend_generation_check_passed": (
            derivation.local_pre_backend_generation_check_passed
        ),
        "local_pre_backend_check_observed_at": (
            derivation.local_pre_backend_check_observed_at
        ),
        "execution_context_enforcement": "executor_local_generation_only",
        "backend_accepted": False,
        "non_authoritative": True,
        "anchor_commit_eligible": False,
        "admission_receipt_id": None,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=LEGACY_REFERENCE_RESOLUTION_HASH_DOMAIN_V1,
    )
    return LegacyPositionReferenceResolutionObservation(
        resolution_observation_id=f"legacy-reference-resolution:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class LegacyReferenceResolutionFailureDerivation:
    """Explicit legacy planner/resolver failure for one observed reference."""

    reference_derivation: LegacyReferenceDerivation
    failed_at: float
    failure_stage: str
    error_type: str
    error_message: str
    failure_evidence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference_derivation, LegacyReferenceDerivation):
            raise TypeError("reference_derivation must be a LegacyReferenceDerivation.")
        _finite_time(self.failed_at, name="failed_at")
        if self.failure_stage not in LEGACY_REFERENCE_FAILURE_STAGES_V1:
            raise ValueError("Unsupported legacy reference failure stage.")
        _required_text(self.error_type, name="error_type")
        _required_text(self.error_message, name="error_message")
        _required_text(self.failure_evidence_hash, name="failure_evidence_hash")


@dataclass(frozen=True)
class LegacyPositionReferenceResolutionFailureObservation:
    failure_observation_id: str
    source_frame_observation_id: str
    worker_instance_id: str
    executor_instance_id: str
    legacy_worker_generation: int
    legacy_generation_namespace: str
    modality: str
    failed_at: float
    failure_stage: str
    error_type: str
    error_message: str
    failure_evidence_hash: str
    execution_context_enforcement: str
    backend_accepted: bool
    non_authoritative: bool
    anchor_commit_eligible: bool
    admission_receipt_id: None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "failure_observation_id",
            "source_frame_observation_id",
            "worker_instance_id",
            "executor_instance_id",
            "legacy_generation_namespace",
            "error_type",
            "error_message",
            "failure_evidence_hash",
            "execution_context_enforcement",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.legacy_generation_namespace != LEGACY_POSITION_GENERATION_NAMESPACE_V1:
            raise ValueError("Failure must retain the executor-local namespace.")
        _non_negative_int(self.legacy_worker_generation, name="legacy_worker_generation")
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy reference modality.")
        _finite_time(self.failed_at, name="failed_at")
        if self.failure_stage not in LEGACY_REFERENCE_FAILURE_STAGES_V1:
            raise ValueError("Unsupported legacy reference failure stage.")
        if self.execution_context_enforcement != (
            "legacy_generation_captured_at_reference_start"
        ):
            raise ValueError("Legacy failure cannot claim shared M1C enforcement.")
        if self.backend_accepted:
            raise ValueError("Resolution failure cannot claim backend acceptance.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Legacy failure cannot become authority or anchor evidence.")
        if self.admission_receipt_id is not None:
            raise ValueError("Legacy failure cannot carry admission receipts.")


def build_legacy_reference_resolution_failure_observation(
    derivation: LegacyReferenceResolutionFailureDerivation,
    *,
    worker_instance_id: str,
    executor_instance_id: str,
) -> LegacyPositionReferenceResolutionFailureObservation:
    frame = build_legacy_reference_observation_batch(
        derivation.reference_derivation,
        worker_instance_id=worker_instance_id,
        executor_instance_id=executor_instance_id,
        include_segment_observation=False,
    ).frame_observation
    payload = {
        "source_frame_observation_id": frame.frame_observation_id,
        "worker_instance_id": _required_text(
            worker_instance_id, name="worker_instance_id"
        ),
        "executor_instance_id": _required_text(
            executor_instance_id, name="executor_instance_id"
        ),
        "legacy_worker_generation": (
            derivation.reference_derivation.legacy_worker_generation
        ),
        "legacy_generation_namespace": LEGACY_POSITION_GENERATION_NAMESPACE_V1,
        "modality": derivation.reference_derivation.modality,
        "failed_at": derivation.failed_at,
        "failure_stage": derivation.failure_stage,
        "error_type": derivation.error_type,
        "error_message": derivation.error_message,
        "failure_evidence_hash": derivation.failure_evidence_hash,
        "execution_context_enforcement": (
            "legacy_generation_captured_at_reference_start"
        ),
        "backend_accepted": False,
        "non_authoritative": True,
        "anchor_commit_eligible": False,
        "admission_receipt_id": None,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=LEGACY_REFERENCE_RESOLUTION_FAILURE_HASH_DOMAIN_V1,
    )
    return LegacyPositionReferenceResolutionFailureObservation(
        failure_observation_id=f"legacy-reference-failure:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class LegacyReferenceGenerationInvalidation:
    legacy_worker_generation: int
    invalidated_at: float
    reason: str
    terminal_status: str

    def __post_init__(self) -> None:
        _non_negative_int(
            self.legacy_worker_generation,
            name="legacy_worker_generation",
        )
        _finite_time(self.invalidated_at, name="invalidated_at")
        _required_text(self.reason, name="reason")
        if self.terminal_status not in {"canceled", "faulted"}:
            raise ValueError("Legacy invalidation supports canceled/faulted only.")


@dataclass(frozen=True)
class LegacyPositionReferenceSegmentTerminalObservation:
    terminal_observation_id: str
    source_segment_observation_id: str
    source_segment_observation_hash: str
    worker_instance_id: str
    executor_instance_id: str
    legacy_worker_generation: int
    legacy_generation_namespace: str
    terminal_status: str
    reason: str
    recorded_at: float
    effective_at: float
    execution_context_enforcement: str
    backend_stop_acknowledged: bool
    non_authoritative: bool
    anchor_commit_eligible: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "terminal_observation_id",
            "source_segment_observation_id",
            "source_segment_observation_hash",
            "worker_instance_id",
            "executor_instance_id",
            "legacy_generation_namespace",
            "reason",
            "execution_context_enforcement",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        if self.legacy_generation_namespace != LEGACY_POSITION_GENERATION_NAMESPACE_V1:
            raise ValueError("Legacy terminal must retain its local namespace.")
        _non_negative_int(
            self.legacy_worker_generation,
            name="legacy_worker_generation",
        )
        if self.terminal_status not in {"canceled", "faulted"}:
            raise ValueError("Legacy terminal supports canceled/faulted only.")
        recorded = _finite_time(self.recorded_at, name="recorded_at")
        effective = _finite_time(self.effective_at, name="effective_at")
        if recorded < effective:
            raise ValueError("Terminal record time cannot precede effective time.")
        if self.execution_context_enforcement != "executor_local_generation_only":
            raise ValueError("Legacy terminal cannot claim shared M1C enforcement.")
        if self.backend_stop_acknowledged:
            raise ValueError("Generation invalidation is not a backend stop acknowledgement.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Legacy terminal cannot become authority or anchor evidence.")


def build_legacy_reference_segment_terminal_observation(
    segment: LegacyPositionReferenceSegmentObservation,
    invalidation: LegacyReferenceGenerationInvalidation,
) -> LegacyPositionReferenceSegmentTerminalObservation:
    if segment.legacy_worker_generation != invalidation.legacy_worker_generation:
        raise ValueError("Terminal invalidation generation does not match segment.")
    payload = {
        "source_segment_observation_id": segment.observation_id,
        "source_segment_observation_hash": segment.semantic_content_hash,
        "worker_instance_id": segment.worker_instance_id,
        "executor_instance_id": segment.executor_instance_id,
        "legacy_worker_generation": segment.legacy_worker_generation,
        "legacy_generation_namespace": segment.legacy_generation_namespace,
        "terminal_status": invalidation.terminal_status,
        "reason": invalidation.reason,
        "recorded_at": invalidation.invalidated_at,
        "effective_at": invalidation.invalidated_at,
        "execution_context_enforcement": "executor_local_generation_only",
        "backend_stop_acknowledged": False,
        "non_authoritative": True,
        "anchor_commit_eligible": False,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=LEGACY_REFERENCE_TERMINAL_HASH_DOMAIN_V1,
    )
    return LegacyPositionReferenceSegmentTerminalObservation(
        terminal_observation_id=f"legacy-reference-terminal:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class LegacyReferenceDerivation:
    legacy_worker_generation: int
    modality: str
    scheduled_for: float
    emitted_at: float
    reference: ResampledMotionReference
    start_source_fact: LegacyReferenceSourceFact
    end_source_fact: LegacyReferenceSourceFact
    interpolation_contract_hash: str

    def __post_init__(self) -> None:
        _non_negative_int(
            self.legacy_worker_generation,
            name="legacy_worker_generation",
        )
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy reference modality.")
        _finite_time(self.scheduled_for, name="scheduled_for")
        _finite_time(self.emitted_at, name="emitted_at")
        _required_text(
            self.interpolation_contract_hash,
            name="interpolation_contract_hash",
        )
        if self.start_source_fact.source_index != self.reference.left_source_index:
            raise ValueError("START source fact does not match resampler output.")
        if self.end_source_fact.source_index != self.reference.right_source_index:
            raise ValueError("END source fact does not match resampler output.")


def _reference_payload(reference: ResampledMotionReference, modality: str) -> Mapping[str, object]:
    if modality == "pose":
        values = reference.pose_by_side
    elif modality == "qpos":
        values = reference.qpos_by_side
    else:  # pragma: no cover - guarded by caller contracts.
        raise ValueError("Unsupported legacy reference modality.")
    if not values:
        raise ValueError(f"Reference does not contain {modality} values.")
    return {str(side): value for side, value in values.items()}


def _segment_key_payload(
    derivation: LegacyReferenceDerivation,
    *,
    worker_instance_id: str,
    executor_instance_id: str,
) -> dict[str, object]:
    return {
        "worker_instance_id": worker_instance_id,
        "executor_instance_id": executor_instance_id,
        "legacy_worker_generation": derivation.legacy_worker_generation,
        "legacy_generation_namespace": LEGACY_POSITION_GENERATION_NAMESPACE_V1,
        "modality": derivation.modality,
        "start_source_index": derivation.start_source_fact.source_index,
        "end_source_index": derivation.end_source_fact.source_index,
        "start_source_timestamp": derivation.start_source_fact.source_timestamp,
        "end_source_timestamp": derivation.end_source_fact.source_timestamp,
        "interpolation_contract_hash": derivation.interpolation_contract_hash,
    }


def build_legacy_reference_observation_batch(
    derivation: LegacyReferenceDerivation,
    *,
    worker_instance_id: str,
    executor_instance_id: str,
    include_segment_observation: bool,
) -> LegacyPositionReferenceObservationBatch:
    """Build one immutable pre-resolver observation batch.

    ``include_segment_observation`` is controlled by the passive observer's
    per-generation parent-pair registry.  It must be true only for the first
    observed frame derived from an interpolated pair.
    """

    worker_id = _required_text(worker_instance_id, name="worker_instance_id")
    executor_id = _required_text(executor_instance_id, name="executor_instance_id")
    reference = derivation.reference
    interpolated = reference.left_source_index != reference.right_source_index
    mode = str(reference.metadata.get("resampling_mode") or "")
    if interpolated:
        derivation_kind = "interpolated_segment"
    elif mode == "causal_hold":
        derivation_kind = "causal_hold"
    else:
        derivation_kind = "exact_keyframe"
    if include_segment_observation and not interpolated:
        raise ValueError("Exact/hold reference cannot create a segment observation.")

    segment: LegacyPositionReferenceSegmentObservation | None = None
    segment_id: str | None = None
    if interpolated:
        key_payload = _segment_key_payload(
            derivation,
            worker_instance_id=worker_id,
            executor_instance_id=executor_id,
        )
        segment_key_hash = canonical_content_hash(
            key_payload,
            domain=LEGACY_REFERENCE_SEGMENT_KEY_HASH_DOMAIN_V1,
        )
        segment_id = f"legacy-reference-segment:{segment_key_hash}"
        if include_segment_observation:
            start = build_legacy_reference_endpoint_observation(
                role="start",
                modality=derivation.modality,
                source_fact=derivation.start_source_fact,
            )
            end = build_legacy_reference_endpoint_observation(
                role="end",
                modality=derivation.modality,
                source_fact=derivation.end_source_fact,
            )
            segment_payload = {
                **key_payload,
                "segment_key": segment_key_hash,
                "endpoint_observation_hashes": (
                    start.semantic_content_hash,
                    end.semantic_content_hash,
                ),
                "scheduled_from": reference.left_source_timestamp,
                "scheduled_until": reference.right_source_timestamp,
                "observed_segment_created_at": derivation.emitted_at,
                "formation_semantics": (
                    "first_observed_reference_derived_from_exact_parent_pair"
                ),
                "observation_authority": "legacy_observed",
                "non_authoritative": True,
                "anchor_commit_eligible": False,
            }
            segment_hash = canonical_content_hash(
                segment_payload,
                domain=LEGACY_REFERENCE_SEGMENT_HASH_DOMAIN_V1,
            )
            segment = LegacyPositionReferenceSegmentObservation(
                observation_id=segment_id,
                segment_key=segment_key_hash,
                worker_instance_id=worker_id,
                executor_instance_id=executor_id,
                legacy_worker_generation=derivation.legacy_worker_generation,
                legacy_generation_namespace=LEGACY_POSITION_GENERATION_NAMESPACE_V1,
                modality=derivation.modality,
                endpoint_observations=(start, end),
                interpolation_contract_hash=derivation.interpolation_contract_hash,
                scheduled_from=reference.left_source_timestamp,
                scheduled_until=reference.right_source_timestamp,
                observed_segment_created_at=derivation.emitted_at,
                formation_semantics=(
                    "first_observed_reference_derived_from_exact_parent_pair"
                ),
                observation_authority="legacy_observed",
                non_authoritative=True,
                anchor_commit_eligible=False,
                semantic_content_hash=segment_hash,
            )

    reference_payload_hash = canonical_content_hash(
        _reference_payload(reference, derivation.modality),
        domain=f"dexteleop.legacy_reference_payload.{derivation.modality}.v1",
    )
    frame_payload = {
        "source_segment_observation_id": segment_id,
        "worker_instance_id": worker_id,
        "executor_instance_id": executor_id,
        "legacy_worker_generation": derivation.legacy_worker_generation,
        "legacy_generation_namespace": LEGACY_POSITION_GENERATION_NAMESPACE_V1,
        "modality": derivation.modality,
        "legacy_reference_sequence": reference.sequence,
        "derivation_kind": derivation_kind,
        "scheduled_for": derivation.scheduled_for,
        "query_timestamp": reference.timestamp,
        "emitted_at": derivation.emitted_at,
        "interpolation_alpha": reference.interpolation_alpha,
        "start_source_index": reference.left_source_index,
        "end_source_index": reference.right_source_index,
        "reference_payload_hash": reference_payload_hash,
        "observation_stage": "resampled_pre_resolver",
        "solver_evidence_status": "not_observed_by_e2b_pre_resolver_hook",
        "non_authoritative": True,
        "anchor_commit_eligible": False,
    }
    frame_hash = canonical_content_hash(
        frame_payload,
        domain=LEGACY_REFERENCE_FRAME_HASH_DOMAIN_V1,
    )
    frame = LegacyPositionReferenceFrameObservation(
        frame_observation_id=f"legacy-reference-frame:{frame_hash}",
        **frame_payload,
        semantic_content_hash=frame_hash,
    )
    return LegacyPositionReferenceObservationBatch(
        segment_observation=segment,
        frame_observation=frame,
    )


@dataclass(frozen=True)
class ExecutionContextBindingEvidence:
    evidence_id: str
    control_domain_id: str
    selection_group_id: str
    execution_epoch: int
    group_segment_generation: int
    binding_source: str
    bound_at: float
    enforcement_level: str
    guard_evidence_ids: tuple[str, ...]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "evidence_id",
            "control_domain_id",
            "selection_group_id",
            "binding_source",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _non_negative_int(self.execution_epoch, name="execution_epoch")
        _non_negative_int(
            self.group_segment_generation,
            name="group_segment_generation",
        )
        _finite_time(self.bound_at, name="bound_at")
        if self.enforcement_level not in EXECUTION_CONTEXT_ENFORCEMENT_LEVELS_V1:
            raise ValueError("Unsupported execution-context enforcement level.")
        evidence_ids = tuple(
            _required_text(value, name="guard_evidence_id")
            for value in self.guard_evidence_ids
        )
        if self.enforcement_level != "context_only" and not evidence_ids:
            raise ValueError("Checked execution context requires guard evidence IDs.")
        object.__setattr__(self, "guard_evidence_ids", evidence_ids)


def build_execution_context_binding_evidence(
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_epoch: int,
    group_segment_generation: int,
    binding_source: str,
    bound_at: float,
    enforcement_level: str = "context_only",
    guard_evidence_ids: tuple[str, ...] = (),
) -> ExecutionContextBindingEvidence:
    payload = {
        "control_domain_id": _required_text(
            control_domain_id, name="control_domain_id"
        ),
        "selection_group_id": _required_text(
            selection_group_id, name="selection_group_id"
        ),
        "execution_epoch": _non_negative_int(
            execution_epoch, name="execution_epoch"
        ),
        "group_segment_generation": _non_negative_int(
            group_segment_generation,
            name="group_segment_generation",
        ),
        "binding_source": _required_text(binding_source, name="binding_source"),
        "bound_at": _finite_time(bound_at, name="bound_at"),
        "enforcement_level": _required_text(
            enforcement_level, name="enforcement_level"
        ),
        "guard_evidence_ids": tuple(guard_evidence_ids),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=EXECUTION_CONTEXT_BINDING_HASH_DOMAIN_V1,
    )
    return ExecutionContextBindingEvidence(
        evidence_id=f"execution-context-binding:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class LegacyReferenceParentProvenance:
    role: str
    modality: str
    source_index: int
    source_kind: str
    source_fact_id: str
    action_content_hash: str
    action_schema_hash: str
    frame_schema_hash: str
    target_effective_at: float
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.role not in LEGACY_REFERENCE_ENDPOINT_ROLES_V1:
            raise ValueError("Unsupported legacy parent role.")
        if self.modality not in LEGACY_REFERENCE_MODALITIES_V1:
            raise ValueError("Unsupported legacy parent modality.")
        if self.source_kind not in LEGACY_REFERENCE_PARENT_SOURCE_KINDS_V1:
            raise ValueError("Unsupported legacy parent source kind.")
        _non_negative_int(self.source_index, name="source_index")
        for name in (
            "source_fact_id",
            "action_content_hash",
            "action_schema_hash",
            "frame_schema_hash",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        _finite_time(self.target_effective_at, name="target_effective_at")


def build_legacy_reference_parent_provenance(
    *,
    role: str,
    modality: str,
    source_index: int,
    source_kind: str,
    source_fact_id: str,
    action_content_hash: str,
    action_schema_hash: str,
    frame_schema_hash: str,
    target_effective_at: float,
) -> LegacyReferenceParentProvenance:
    payload = {
        "role": _required_text(role, name="role"),
        "modality": _required_text(modality, name="modality"),
        "source_index": _non_negative_int(source_index, name="source_index"),
        "source_kind": _required_text(source_kind, name="source_kind"),
        "source_fact_id": _required_text(source_fact_id, name="source_fact_id"),
        "action_content_hash": _required_text(
            action_content_hash, name="action_content_hash"
        ),
        "action_schema_hash": _required_text(
            action_schema_hash, name="action_schema_hash"
        ),
        "frame_schema_hash": _required_text(
            frame_schema_hash, name="frame_schema_hash"
        ),
        "target_effective_at": _finite_time(
            target_effective_at, name="target_effective_at"
        ),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=LEGACY_REFERENCE_PARENT_PROVENANCE_HASH_DOMAIN_V1,
    )
    return LegacyReferenceParentProvenance(
        **payload,
        semantic_content_hash=semantic_hash,
    )


@dataclass(frozen=True)
class BoundLegacyReferenceSegmentObservation:
    projection_id: str
    source_observation_id: str
    source_observation_hash: str
    control_domain_id: str
    selection_group_id: str
    execution_context_binding_id: str
    execution_context_enforcement: str
    parent_provenance: tuple[LegacyReferenceParentProvenance, ...]
    normalized_segment_hash: str
    projection_status: str
    admission_semantics: str
    non_authoritative: bool
    anchor_commit_eligible: bool
    admission_receipt_id: None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "projection_id",
            "source_observation_id",
            "source_observation_hash",
            "control_domain_id",
            "selection_group_id",
            "execution_context_binding_id",
            "normalized_segment_hash",
            "projection_status",
            "admission_semantics",
            "semantic_content_hash",
        ):
            _required_text(getattr(self, name), name=name)
        parents = tuple(self.parent_provenance)
        if len(parents) != 2 or tuple(item.role for item in parents) != (
            "start",
            "end",
        ):
            raise ValueError("Bound legacy segment requires ordered parent provenance.")
        if self.admission_semantics != "observed_legacy":
            raise ValueError("Bound legacy segments must retain observed_legacy semantics.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Bound legacy segments cannot become authority or anchor evidence.")
        if self.admission_receipt_id is not None:
            raise ValueError("Bound legacy observations cannot carry admission receipts.")


def project_legacy_reference_segment(
    observation: LegacyPositionReferenceSegmentObservation,
    *,
    execution_context: ExecutionContextBindingEvidence,
    parent_provenance: tuple[LegacyReferenceParentProvenance, ...],
) -> BoundLegacyReferenceSegmentObservation:
    """Bind typed context without upgrading a legacy fact to admission."""

    parents = tuple(parent_provenance)
    if len(parents) != 2 or tuple(item.role for item in parents) != (
        "start",
        "end",
    ):
        raise ValueError("parent_provenance must be ordered START then END.")
    for observed, bound in zip(observation.endpoint_observations, parents, strict=True):
        if observed.role != bound.role:
            raise ValueError("Legacy parent role mismatch.")
        if observed.modality != bound.modality or observation.modality != bound.modality:
            raise ValueError("Legacy parent modality mismatch.")
        if observed.source_index != bound.source_index:
            raise ValueError("Legacy parent source index mismatch.")
        if not math.isclose(
            observed.source_timestamp,
            bound.target_effective_at,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Legacy parent target time mismatch.")
        if observed.target_payload_hash != bound.action_content_hash:
            raise ValueError("Legacy parent action hash mismatch.")

    normalized_payload = {
        "control_domain_id": execution_context.control_domain_id,
        "selection_group_id": execution_context.selection_group_id,
        "execution_epoch_context": execution_context.execution_epoch,
        "group_segment_generation_context": (
            execution_context.group_segment_generation
        ),
        "legacy_generation_namespace": observation.legacy_generation_namespace,
        "legacy_worker_generation": observation.legacy_worker_generation,
        "modality": observation.modality,
        "ordered_parent_hashes": tuple(
            item.semantic_content_hash for item in parents
        ),
        "interpolation_contract_hash": observation.interpolation_contract_hash,
        "scheduled_from": observation.scheduled_from,
        "scheduled_until": observation.scheduled_until,
    }
    normalized_hash = canonical_content_hash(
        normalized_payload,
        domain="dexteleop.bound_legacy_normalized_segment.v1",
    )
    payload = {
        "source_observation_id": observation.observation_id,
        "source_observation_hash": observation.semantic_content_hash,
        "control_domain_id": execution_context.control_domain_id,
        "selection_group_id": execution_context.selection_group_id,
        "execution_context_binding_id": execution_context.evidence_id,
        "execution_context_enforcement": execution_context.enforcement_level,
        "parent_provenance_hashes": tuple(
            item.semantic_content_hash for item in parents
        ),
        "normalized_segment_hash": normalized_hash,
        "projection_status": "bound",
        "admission_semantics": "observed_legacy",
        "non_authoritative": True,
        "anchor_commit_eligible": False,
        "admission_receipt_id": None,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=BOUND_LEGACY_REFERENCE_SEGMENT_HASH_DOMAIN_V1,
    )
    return BoundLegacyReferenceSegmentObservation(
        projection_id=f"bound-legacy-reference-segment:{semantic_hash}",
        source_observation_id=observation.observation_id,
        source_observation_hash=observation.semantic_content_hash,
        control_domain_id=execution_context.control_domain_id,
        selection_group_id=execution_context.selection_group_id,
        execution_context_binding_id=execution_context.evidence_id,
        execution_context_enforcement=execution_context.enforcement_level,
        parent_provenance=parents,
        normalized_segment_hash=normalized_hash,
        projection_status="bound",
        admission_semantics="observed_legacy",
        non_authoritative=True,
        anchor_commit_eligible=False,
        admission_receipt_id=None,
        semantic_content_hash=semantic_hash,
    )


LegacyPositionReferenceObserverRecord = (
    LegacyPositionReferenceObservationBatch
    | LegacyPositionReferenceResolutionObservation
    | LegacyPositionReferenceResolutionFailureObservation
    | LegacyPositionReferenceSegmentTerminalObservation
)
LegacyPositionReferenceObservationSink = Callable[
    [LegacyPositionReferenceObserverRecord], None
]


class LegacyPositionReferenceObserver(Protocol):
    def submit_nowait(self, derivation: LegacyReferenceDerivation) -> bool: ...

    def submit_resolution_nowait(
        self,
        derivation: LegacyReferenceResolutionDerivation,
    ) -> bool: ...

    def submit_resolution_failure_nowait(
        self,
        derivation: LegacyReferenceResolutionFailureDerivation,
    ) -> bool: ...

    def submit_generation_invalidation_nowait(
        self,
        invalidation: LegacyReferenceGenerationInvalidation,
    ) -> bool: ...


class LegacyPositionReferenceObserverRuntime:
    """Bounded passive observer; producer calls never block or raise."""

    _STOP = object()

    def __init__(
        self,
        *,
        worker_instance_id: str,
        executor_instance_id: str,
        sink: LegacyPositionReferenceObservationSink,
        queue_capacity: int = 256,
    ) -> None:
        self.worker_instance_id = _required_text(
            worker_instance_id, name="worker_instance_id"
        )
        self.executor_instance_id = _required_text(
            executor_instance_id, name="executor_instance_id"
        )
        if not callable(sink):
            raise TypeError("sink must be callable.")
        if isinstance(queue_capacity, bool) or int(queue_capacity) <= 0:
            raise ValueError("queue_capacity must be positive.")
        self._sink = sink
        self._queue: queue.Queue[
            LegacyReferenceDerivation
            | LegacyReferenceResolutionDerivation
            | LegacyReferenceResolutionFailureDerivation
            | LegacyReferenceGenerationInvalidation
            | object
        ] = queue.Queue(maxsize=int(queue_capacity))
        self._seen_segment_keys: set[str] = set()
        self._seen_frame_ids: set[str] = set()
        self._segments_by_generation: dict[
            int, dict[str, LegacyPositionReferenceSegmentObservation]
        ] = {}
        self._seen_generation: int | None = None
        self._closed = False
        self.drop_count = 0
        self.error_count = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="legacy-position-reference-observer",
            daemon=True,
        )
        self._thread.start()

    def submit_nowait(self, derivation: LegacyReferenceDerivation) -> bool:
        if not isinstance(derivation, LegacyReferenceDerivation):
            return False
        if self._closed:
            self.drop_count += 1
            return False
        try:
            self._queue.put_nowait(derivation)
        except queue.Full:
            self.drop_count += 1
            return False
        return True

    def submit_resolution_nowait(
        self,
        derivation: LegacyReferenceResolutionDerivation,
    ) -> bool:
        if not isinstance(derivation, LegacyReferenceResolutionDerivation):
            return False
        if self._closed:
            self.drop_count += 1
            return False
        try:
            self._queue.put_nowait(derivation)
        except queue.Full:
            self.drop_count += 1
            return False
        return True

    def submit_resolution_failure_nowait(
        self,
        derivation: LegacyReferenceResolutionFailureDerivation,
    ) -> bool:
        if not isinstance(derivation, LegacyReferenceResolutionFailureDerivation):
            return False
        if self._closed:
            self.drop_count += 1
            return False
        try:
            self._queue.put_nowait(derivation)
        except queue.Full:
            self.drop_count += 1
            return False
        return True

    def submit_generation_invalidation_nowait(
        self,
        invalidation: LegacyReferenceGenerationInvalidation,
    ) -> bool:
        if not isinstance(invalidation, LegacyReferenceGenerationInvalidation):
            return False
        if self._closed:
            self.drop_count += 1
            return False
        try:
            self._queue.put_nowait(invalidation)
        except queue.Full:
            self.drop_count += 1
            return False
        return True

    def _segment_key_for(
        self,
        derivation: LegacyReferenceDerivation,
    ) -> str | None:
        if (
            derivation.reference.left_source_index
            == derivation.reference.right_source_index
        ):
            return None
        return canonical_content_hash(
            _segment_key_payload(
                derivation,
                worker_instance_id=self.worker_instance_id,
                executor_instance_id=self.executor_instance_id,
            ),
            domain=LEGACY_REFERENCE_SEGMENT_KEY_HASH_DOMAIN_V1,
        )

    def _emit_reference_derivation(
        self,
        derivation: LegacyReferenceDerivation,
    ) -> LegacyPositionReferenceObservationBatch:
        key_hash = self._segment_key_for(derivation)
        include_segment = bool(
            key_hash is not None and key_hash not in self._seen_segment_keys
        )
        batch = build_legacy_reference_observation_batch(
            derivation,
            worker_instance_id=self.worker_instance_id,
            executor_instance_id=self.executor_instance_id,
            include_segment_observation=include_segment,
        )
        if key_hash is not None:
            self._seen_segment_keys.add(key_hash)
        if batch.segment_observation is not None:
            by_id = self._segments_by_generation.setdefault(
                derivation.legacy_worker_generation,
                {},
            )
            by_id[batch.segment_observation.observation_id] = (
                batch.segment_observation
            )
        self._sink(batch)
        self._seen_frame_ids.add(batch.frame_observation.frame_observation_id)
        return batch

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                if isinstance(item, LegacyReferenceGenerationInvalidation):
                    segments = self._segments_by_generation.pop(
                        item.legacy_worker_generation,
                        {},
                    )
                    for segment_id in sorted(segments):
                        self._sink(
                            build_legacy_reference_segment_terminal_observation(
                                segments[segment_id],
                                item,
                            )
                        )
                    if self._seen_generation == item.legacy_worker_generation:
                        self._seen_segment_keys.clear()
                        self._seen_frame_ids.clear()
                        self._seen_generation = None
                    continue
                if isinstance(
                    item,
                    (
                        LegacyReferenceResolutionDerivation,
                        LegacyReferenceResolutionFailureDerivation,
                    ),
                ):
                    reference_derivation = item.reference_derivation
                else:
                    assert isinstance(item, LegacyReferenceDerivation)
                    reference_derivation = item
                if self._seen_generation != reference_derivation.legacy_worker_generation:
                    self._seen_segment_keys.clear()
                    self._seen_frame_ids.clear()
                    self._seen_generation = reference_derivation.legacy_worker_generation
                if isinstance(
                    item,
                    (
                        LegacyReferenceResolutionDerivation,
                        LegacyReferenceResolutionFailureDerivation,
                    ),
                ):
                    candidate_frame = build_legacy_reference_observation_batch(
                        reference_derivation,
                        worker_instance_id=self.worker_instance_id,
                        executor_instance_id=self.executor_instance_id,
                        include_segment_observation=False,
                    ).frame_observation
                    if candidate_frame.frame_observation_id not in self._seen_frame_ids:
                        self._emit_reference_derivation(reference_derivation)
                    if isinstance(item, LegacyReferenceResolutionDerivation):
                        self._sink(
                            build_legacy_reference_resolution_observation(
                                item,
                                worker_instance_id=self.worker_instance_id,
                                executor_instance_id=self.executor_instance_id,
                            )
                        )
                    else:
                        self._sink(
                            build_legacy_reference_resolution_failure_observation(
                                item,
                                worker_instance_id=self.worker_instance_id,
                                executor_instance_id=self.executor_instance_id,
                            )
                        )
                else:
                    self._emit_reference_derivation(item)
            except Exception as exc:  # noqa: BLE001 - passive diagnostics only.
                self.error_count += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        self._queue.join()

    def close(self, *, timeout_sec: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        timeout = float(timeout_sec)
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_sec must be finite and positive.")
        try:
            self._queue.put(self._STOP, timeout=timeout)
        except queue.Full as exc:
            raise TimeoutError("Legacy reference observer queue did not drain.") from exc
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError("Legacy reference observer did not stop before deadline.")


__all__ = [
    "BOUND_LEGACY_REFERENCE_SEGMENT_HASH_DOMAIN_V1",
    "EXECUTION_CONTEXT_ENFORCEMENT_LEVELS_V1",
    "ExecutionContextBindingEvidence",
    "LEGACY_POSITION_GENERATION_NAMESPACE_V1",
    "LEGACY_REFERENCE_PARENT_SOURCE_KINDS_V1",
    "BoundLegacyReferenceSegmentObservation",
    "LegacyPositionReferenceFrameObservation",
    "LegacyPositionReferenceObservationBatch",
    "LegacyPositionReferenceObserverRecord",
    "LegacyPositionReferenceObserver",
    "LegacyPositionReferenceObserverRuntime",
    "LegacyPositionReferenceResolutionObservation",
    "LegacyPositionReferenceResolutionFailureObservation",
    "LegacyPositionReferenceSegmentTerminalObservation",
    "LegacyPositionReferenceSegmentObservation",
    "LegacyReferenceDerivation",
    "LegacyReferenceGenerationInvalidation",
    "LegacyReferenceEndpointObservation",
    "LegacyReferenceParentProvenance",
    "LegacyReferenceResolutionDerivation",
    "LegacyReferenceResolutionFailureDerivation",
    "LegacyReferenceSourceFact",
    "build_execution_context_binding_evidence",
    "build_legacy_reference_observation_batch",
    "build_legacy_reference_parent_provenance",
    "build_legacy_reference_resolution_observation",
    "build_legacy_reference_resolution_failure_observation",
    "build_legacy_reference_segment_terminal_observation",
    "project_legacy_reference_segment",
]
