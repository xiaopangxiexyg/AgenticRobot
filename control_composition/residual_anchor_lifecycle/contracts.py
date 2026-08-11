"""Typed contracts for receiver-admitted residual-anchor lifecycle facts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.identity import RuntimeFactIdentity

from ..activation import CompositionActivationScope
from ..residual_anchor import ResidualAnchorReceipt, ResidualAnchorRequest


RESIDUAL_ANCHOR_LIFECYCLE_PENDING = "pending"
RESIDUAL_ANCHOR_LIFECYCLE_READY = "ready"
RESIDUAL_ANCHOR_LIFECYCLE_PHASES_DEVELOPMENT = frozenset(
    {RESIDUAL_ANCHOR_LIFECYCLE_PENDING, RESIDUAL_ANCHOR_LIFECYCLE_READY}
)
RESIDUAL_ANCHOR_LIFECYCLE_SPEC_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_lifecycle_spec.development"
)
RESIDUAL_ANCHOR_LIFECYCLE_RECORD_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_lifecycle_record.development"
)
RESIDUAL_ANCHOR_LIFECYCLE_SNAPSHOT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.residual_anchor_lifecycle_snapshot.development"
)


class ResidualAnchorLifecycleAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ResidualAnchorLifecycleAdmissionError(
            "invalid_field",
            f"{name} must be non-empty.",
        )
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ResidualAnchorLifecycleAdmissionError(
            "invalid_time",
            f"{name} must be finite.",
        )
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResidualAnchorLifecycleAdmissionError(
            "invalid_identity",
            f"{name} must be a non-negative integer.",
        )
    return int(value)


@dataclass(frozen=True)
class ResidualAnchorLifecycleSpec:
    """Static anchor protocol required by one composition strategy."""

    spec_id: str
    source_role: str
    component_ids: tuple[str, ...]
    absolute_action_schema_hash: str
    relative_action_schema_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        spec_id = _required(self.spec_id, name="spec_id")
        source_role = _required(self.source_role, name="source_role")
        components = tuple(
            _required(value, name="component_id") for value in self.component_ids
        )
        if not components or len(components) != len(set(components)):
            raise ValueError("component_ids must be non-empty and unique.")
        absolute_hash = _required(
            self.absolute_action_schema_hash,
            name="absolute_action_schema_hash",
        )
        relative_hash = _required(
            self.relative_action_schema_hash,
            name="relative_action_schema_hash",
        )
        payload = {
            "spec_id": spec_id,
            "source_role": source_role,
            "component_ids": components,
            "absolute_action_schema_hash": absolute_hash,
            "relative_action_schema_hash": relative_hash,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=RESIDUAL_ANCHOR_LIFECYCLE_SPEC_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def validate_request(self, request: ResidualAnchorRequest) -> None:
        if not isinstance(request, ResidualAnchorRequest):
            raise TypeError("request must be ResidualAnchorRequest.")
        if (
            request.source_role != self.source_role
            or request.component_ids != self.component_ids
            or request.absolute_action_schema_hash
            != self.absolute_action_schema_hash
            or request.relative_action_schema_hash
            != self.relative_action_schema_hash
        ):
            raise ValueError("Residual anchor request disagrees with lifecycle spec.")


@dataclass(frozen=True)
class ResidualAnchorLifecycleDraft:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    lifecycle_scope_id: str
    lifecycle_spec_hash: str
    phase: str
    effective_for_target_at: float
    observed_at: float
    observed_clock_domain_id: str
    anchor_request: ResidualAnchorRequest
    anchor_receipt: ResidualAnchorReceipt | None = None


@dataclass(frozen=True)
class ResidualAnchorLifecycleRecord:
    lifecycle_record_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    lifecycle_scope_id: str
    lifecycle_spec_hash: str
    phase: str
    effective_for_target_at: float
    observed_at: float
    observed_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    composition_activation_id: str
    composition_activation_hash: str
    anchor_request: ResidualAnchorRequest
    anchor_receipt: ResidualAnchorReceipt | None
    semantic_content_hash: str

    @property
    def anchor_request_generation(self) -> int:
        return self.anchor_request.request_generation


@dataclass(frozen=True)
class ResidualAnchorLifecycleAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    lifecycle_record_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    lifecycle_record: ResidualAnchorLifecycleRecord | None = None
    accepted_existing: bool = False


@dataclass(frozen=True)
class ResidualAnchorLifecycleProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    records: tuple[ResidualAnchorLifecycleRecord, ...]


@dataclass(frozen=True)
class ResidualAnchorLifecycleSnapshot:
    lifecycle_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    lifecycle_scope_id: str
    lifecycle_spec_hash: str
    status: str
    reason: str
    selected_record: ResidualAnchorLifecycleRecord | None
    semantic_content_hash: str

    @property
    def ready_receipt(self) -> ResidualAnchorReceipt | None:
        record = self.selected_record
        if record is None or record.phase != RESIDUAL_ANCHOR_LIFECYCLE_READY:
            return None
        return record.anchor_receipt


@dataclass(frozen=True)
class PreparedResidualAnchorLifecycle:
    record_id: str
    draft: ResidualAnchorLifecycleDraft


def prepare_residual_anchor_lifecycle(
    draft: ResidualAnchorLifecycleDraft,
    *,
    spec: ResidualAnchorLifecycleSpec,
    activation_scope: CompositionActivationScope,
) -> PreparedResidualAnchorLifecycle:
    if not isinstance(draft, ResidualAnchorLifecycleDraft):
        raise TypeError("draft must be ResidualAnchorLifecycleDraft.")
    if not isinstance(spec, ResidualAnchorLifecycleSpec):
        raise TypeError("spec must be ResidualAnchorLifecycleSpec.")
    if not isinstance(activation_scope, CompositionActivationScope):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    session = _required(draft.runtime_session_id, name="runtime_session_id")
    domain = _required(draft.control_domain_id, name="control_domain_id")
    producer = _required(draft.producer_id, name="producer_id")
    producer_epoch = _nonnegative_int(draft.producer_epoch, name="producer_epoch")
    sequence = _nonnegative_int(draft.sequence, name="sequence")
    scope_id = _required(draft.lifecycle_scope_id, name="lifecycle_scope_id")
    spec_hash = _required(draft.lifecycle_spec_hash, name="lifecycle_spec_hash")
    if spec_hash != spec.semantic_content_hash:
        raise ResidualAnchorLifecycleAdmissionError(
            "spec_mismatch", "Lifecycle draft uses another spec."
        )
    phase = _required(draft.phase, name="phase")
    if phase not in RESIDUAL_ANCHOR_LIFECYCLE_PHASES_DEVELOPMENT:
        raise ResidualAnchorLifecycleAdmissionError(
            "invalid_phase", "Unsupported residual anchor lifecycle phase."
        )
    effective_at = _finite(
        draft.effective_for_target_at, name="effective_for_target_at"
    )
    observed_at = _finite(draft.observed_at, name="observed_at")
    observed_clock = _required(
        draft.observed_clock_domain_id, name="observed_clock_domain_id"
    )
    request = draft.anchor_request
    spec.validate_request(request)
    if (
        request.composition_activation_id != activation_scope.activation_id
        or request.composition_activation_hash
        != activation_scope.semantic_content_hash
        or request.clock_domain_id != activation_scope.clock_domain_id
    ):
        raise ResidualAnchorLifecycleAdmissionError(
            "activation_mismatch", "Lifecycle request belongs to another activation."
        )
    if observed_clock != activation_scope.clock_domain_id:
        raise ResidualAnchorLifecycleAdmissionError(
            "clock_domain_mismatch", "Lifecycle observation uses another clock."
        )
    if effective_at < request.requested_at:
        raise ResidualAnchorLifecycleAdmissionError(
            "invalid_target_boundary", "Target boundary predates anchor request."
        )
    receipt = draft.anchor_receipt
    if phase == RESIDUAL_ANCHOR_LIFECYCLE_PENDING:
        if receipt is not None:
            raise ResidualAnchorLifecycleAdmissionError(
                "invalid_phase_payload", "Pending cannot carry an anchor receipt."
            )
    else:
        if not isinstance(receipt, ResidualAnchorReceipt):
            raise ResidualAnchorLifecycleAdmissionError(
                "invalid_phase_payload", "Ready requires an anchor receipt."
            )
        receipt.validate_request(request)
    identity = RuntimeFactIdentity(
        runtime_session_id=session,
        control_domain_id=domain,
        producer_id=producer,
        producer_epoch=producer_epoch,
        sequence=sequence,
        namespace="residual_anchor_lifecycle",
    )
    return PreparedResidualAnchorLifecycle(
        record_id=identity.fact_id,
        draft=ResidualAnchorLifecycleDraft(
            runtime_session_id=session,
            control_domain_id=domain,
            producer_id=producer,
            producer_epoch=producer_epoch,
            sequence=sequence,
            lifecycle_scope_id=scope_id,
            lifecycle_spec_hash=spec_hash,
            phase=phase,
            effective_for_target_at=effective_at,
            observed_at=observed_at,
            observed_clock_domain_id=observed_clock,
            anchor_request=request,
            anchor_receipt=receipt,
        ),
    )


def residual_anchor_lifecycle_record_semantic_payload(
    record: ResidualAnchorLifecycleRecord,
) -> dict[str, object]:
    return {
        "lifecycle_record_id": record.lifecycle_record_id,
        "runtime_session_id": record.runtime_session_id,
        "control_domain_id": record.control_domain_id,
        "producer_id": record.producer_id,
        "producer_epoch": record.producer_epoch,
        "sequence": record.sequence,
        "lifecycle_scope_id": record.lifecycle_scope_id,
        "lifecycle_spec_hash": record.lifecycle_spec_hash,
        "phase": record.phase,
        "effective_for_target_at": record.effective_for_target_at,
        "observed_at": record.observed_at,
        "observed_clock_domain_id": record.observed_clock_domain_id,
        "available_at": record.available_at,
        "available_clock_domain_id": record.available_clock_domain_id,
        "control_admission_sequence": record.control_admission_sequence,
        "composition_activation_id": record.composition_activation_id,
        "composition_activation_hash": record.composition_activation_hash,
        "anchor_request_id": record.anchor_request.request_id,
        "anchor_request_hash": record.anchor_request.semantic_content_hash,
        "anchor_request_generation": record.anchor_request.request_generation,
        "anchor_receipt_id": (
            None if record.anchor_receipt is None else record.anchor_receipt.receipt_id
        ),
        "anchor_receipt_hash": (
            None
            if record.anchor_receipt is None
            else record.anchor_receipt.semantic_content_hash
        ),
    }


__all__ = [name for name in globals() if name.startswith("RESIDUAL_ANCHOR_")] + [
    "PreparedResidualAnchorLifecycle",
    "ResidualAnchorLifecycleAdmissionError",
    "ResidualAnchorLifecycleAdmissionReceipt",
    "ResidualAnchorLifecycleDraft",
    "ResidualAnchorLifecycleProjection",
    "ResidualAnchorLifecycleRecord",
    "ResidualAnchorLifecycleSnapshot",
    "ResidualAnchorLifecycleSpec",
    "prepare_residual_anchor_lifecycle",
    "residual_anchor_lifecycle_record_semantic_payload",
]
