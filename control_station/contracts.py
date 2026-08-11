"""Pure application contracts for the DexTeleop Control Station.

These records contain references to existing control/runtime contracts.  They
do not carry action payloads and cannot authorize motion by themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


CONTROL_STATION_SOURCE_KINDS_V1 = (
    "agent_skill",
    "human",
    "replay",
    "policy",
    "composite",
    "recovery",
)
CONTROL_STATION_SOURCE_USES_V1 = (
    "record_only",
    "shadow",
    "contributor",
    "candidate",
    "authority",
)
CONTROL_STATION_TARGET_KINDS_V1 = ("null", "sim", "real")

CONTROL_STATION_INTENT_KINDS_V1 = (
    "acquire_control_lease",
    "release_control_lease",
    "connect_source",
    "disconnect_source",
    "connect_target",
    "disconnect_target",
    "load_replay",
    "load_policy_artifact",
    "run_preflight",
    "start_shadow",
    "stop_shadow",
    "request_candidate",
    "release_candidate",
    "request_motion_authority",
    "release_motion_authority",
    "request_human_takeover",
    "request_residual_rebase_and_clear",
    "start_composition_session",
    "rearm_composition_session",
    "stop_composition_session",
    "hold",
    "estop",
    "request_recovery",
    "home",
    "rearm",
    "begin_capture",
    "finalize_capture",
    "abort_capture",
    "start_typed_motion_test",
    "stop_runtime",
)

SOURCE_SLOT_SPEC_HASH_DOMAIN_V1 = "dexteleop.control_station.source_slot_spec.v1"
EXECUTION_DOMAIN_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.execution_domain_binding.v1"
)
CONTROL_STATION_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.run_manifest.v1"
)
STATION_INTENT_HASH_DOMAIN_V1 = "dexteleop.control_station.intent.v1"
STATION_INTENT_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.intent_admission_receipt.v1"
)
STATION_OPERATION_RESULT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.operation_result.v1"
)
OPERATOR_CONTROL_LEASE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.operator_control_lease.v1"
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _optional(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


def _unique_sorted(
    values: tuple[str, ...] | list[str],
    *,
    field_name: str,
    allowed: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, field_name=field_name) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} cannot contain duplicates.")
    if allowed is not None:
        unknown = sorted(set(normalized) - set(allowed))
        if unknown:
            raise ValueError(f"{field_name} contains unsupported values: {unknown}.")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SourceSlotSpec:
    source_slot_id: str
    source_kind: str
    producer_id: str
    source_action_contract_hash: str
    producer_epoch_policy_hash: str
    supported_uses: tuple[str, ...]
    expected_emission_seam: str
    required_binding_refs: tuple[str, ...] = ()
    trace_requirement_refs: tuple[str, ...] = ()
    required_for_run: bool = True
    explicit_exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_slot_id",
            _required(self.source_slot_id, field_name="source_slot_id"),
        )
        kind = _required(self.source_kind, field_name="source_kind")
        if kind not in CONTROL_STATION_SOURCE_KINDS_V1:
            raise ValueError(f"Unsupported source_kind: {kind!r}.")
        object.__setattr__(self, "source_kind", kind)
        object.__setattr__(
            self, "producer_id", _required(self.producer_id, field_name="producer_id")
        )
        object.__setattr__(
            self,
            "source_action_contract_hash",
            _required(
                self.source_action_contract_hash,
                field_name="source_action_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "producer_epoch_policy_hash",
            _required(
                self.producer_epoch_policy_hash,
                field_name="producer_epoch_policy_hash",
            ),
        )
        uses = _unique_sorted(
            self.supported_uses,
            field_name="supported_uses",
            allowed=CONTROL_STATION_SOURCE_USES_V1,
        )
        if not uses:
            raise ValueError("supported_uses must not be empty.")
        object.__setattr__(self, "supported_uses", uses)
        object.__setattr__(
            self,
            "expected_emission_seam",
            _required(
                self.expected_emission_seam,
                field_name="expected_emission_seam",
            ),
        )
        object.__setattr__(
            self,
            "required_binding_refs",
            _unique_sorted(
                self.required_binding_refs,
                field_name="required_binding_refs",
            ),
        )
        object.__setattr__(
            self,
            "trace_requirement_refs",
            _unique_sorted(
                self.trace_requirement_refs,
                field_name="trace_requirement_refs",
            ),
        )
        object.__setattr__(self, "required_for_run", bool(self.required_for_run))
        object.__setattr__(
            self,
            "explicit_exclusions",
            _unique_sorted(
                self.explicit_exclusions,
                field_name="explicit_exclusions",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "source_slot_id": self.source_slot_id,
                "source_kind": self.source_kind,
                "producer_id": self.producer_id,
                "source_action_contract_hash": self.source_action_contract_hash,
                "producer_epoch_policy_hash": self.producer_epoch_policy_hash,
                "supported_uses": self.supported_uses,
                "expected_emission_seam": self.expected_emission_seam,
                "required_binding_refs": self.required_binding_refs,
                "trace_requirement_refs": self.trace_requirement_refs,
                "required_for_run": self.required_for_run,
                "explicit_exclusions": self.explicit_exclusions,
            },
            domain=SOURCE_SLOT_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ExecutionDomainBinding:
    binding_id: str
    target_kind: str
    control_domain_id: str
    embodiment_instance_id: str
    target_runtime_action_contract_hash: str
    execution_capability_profile_hash: str
    component_catalog_hash: str
    selection_group_ids: tuple[str, ...]
    owner_registry_id: str
    authority_epoch_source_id: str
    execution_epoch_source_id: str
    trace_requirement_profile_id: str
    backend_inventory_hash: str | None = None
    explicit_exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "binding_id",
            "control_domain_id",
            "embodiment_instance_id",
            "target_runtime_action_contract_hash",
            "execution_capability_profile_hash",
            "component_catalog_hash",
            "owner_registry_id",
            "authority_epoch_source_id",
            "execution_epoch_source_id",
            "trace_requirement_profile_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        target_kind = _required(self.target_kind, field_name="target_kind")
        if target_kind not in CONTROL_STATION_TARGET_KINDS_V1:
            raise ValueError(f"Unsupported target_kind: {target_kind!r}.")
        object.__setattr__(self, "target_kind", target_kind)
        groups = _unique_sorted(
            self.selection_group_ids,
            field_name="selection_group_ids",
        )
        if not groups:
            raise ValueError("selection_group_ids must not be empty.")
        object.__setattr__(self, "selection_group_ids", groups)
        object.__setattr__(
            self,
            "backend_inventory_hash",
            _optional(self.backend_inventory_hash),
        )
        if target_kind == "real" and self.backend_inventory_hash is None:
            raise ValueError("A real target binding requires backend_inventory_hash.")
        object.__setattr__(
            self,
            "explicit_exclusions",
            _unique_sorted(
                self.explicit_exclusions,
                field_name="explicit_exclusions",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "binding_id": self.binding_id,
                "target_kind": self.target_kind,
                "control_domain_id": self.control_domain_id,
                "embodiment_instance_id": self.embodiment_instance_id,
                "target_runtime_action_contract_hash": (
                    self.target_runtime_action_contract_hash
                ),
                "execution_capability_profile_hash": (
                    self.execution_capability_profile_hash
                ),
                "component_catalog_hash": self.component_catalog_hash,
                "selection_group_ids": self.selection_group_ids,
                "owner_registry_id": self.owner_registry_id,
                "authority_epoch_source_id": self.authority_epoch_source_id,
                "execution_epoch_source_id": self.execution_epoch_source_id,
                "trace_requirement_profile_id": self.trace_requirement_profile_id,
                "backend_inventory_hash": self.backend_inventory_hash,
                "explicit_exclusions": self.explicit_exclusions,
            },
            domain=EXECUTION_DOMAIN_BINDING_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ControlStationRunManifest:
    manifest_id: str
    manifest_version: str
    execution_binding: ExecutionDomainBinding
    source_slots: tuple[SourceSlotSpec, ...]
    authority_policy_hash: str
    decision_schedule_hash: str
    trace_requirement_profile_id: str
    capture_policy_id: str
    allowed_intent_kinds: tuple[str, ...]
    operator_control_lease_sec: float = 30.0
    explicit_exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "manifest_id",
            "manifest_version",
            "authority_policy_hash",
            "decision_schedule_hash",
            "trace_requirement_profile_id",
            "capture_policy_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if not isinstance(self.execution_binding, ExecutionDomainBinding):
            raise TypeError("execution_binding must be an ExecutionDomainBinding.")
        slots = tuple(self.source_slots)
        if not slots or any(not isinstance(slot, SourceSlotSpec) for slot in slots):
            raise ValueError("source_slots must contain SourceSlotSpec values.")
        slot_ids = tuple(slot.source_slot_id for slot in slots)
        producer_ids = tuple(slot.producer_id for slot in slots)
        if len(set(slot_ids)) != len(slot_ids):
            raise ValueError("source slot IDs must be unique.")
        if self.execution_binding.binding_id in slot_ids:
            raise ValueError("A source slot ID cannot equal the target binding ID.")
        if len(set(producer_ids)) != len(producer_ids):
            raise ValueError("source producer IDs must be unique.")
        object.__setattr__(
            self,
            "source_slots",
            tuple(sorted(slots, key=lambda slot: slot.source_slot_id)),
        )
        object.__setattr__(
            self,
            "allowed_intent_kinds",
            _unique_sorted(
                self.allowed_intent_kinds,
                field_name="allowed_intent_kinds",
                allowed=CONTROL_STATION_INTENT_KINDS_V1,
            ),
        )
        lease_sec = _finite(
            self.operator_control_lease_sec,
            field_name="operator_control_lease_sec",
        )
        if lease_sec <= 0.0:
            raise ValueError("operator_control_lease_sec must be positive.")
        object.__setattr__(self, "operator_control_lease_sec", lease_sec)
        object.__setattr__(
            self,
            "explicit_exclusions",
            _unique_sorted(
                self.explicit_exclusions,
                field_name="explicit_exclusions",
            ),
        )

    @property
    def control_domain_id(self) -> str:
        return self.execution_binding.control_domain_id

    def source_slot(self, source_slot_id: str) -> SourceSlotSpec:
        requested = str(source_slot_id)
        for slot in self.source_slots:
            if slot.source_slot_id == requested:
                return slot
        raise KeyError(f"Unknown source slot: {requested!r}.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "manifest_id": self.manifest_id,
                "manifest_version": self.manifest_version,
                "execution_binding_hash": self.execution_binding.semantic_content_hash,
                "source_slot_hashes": tuple(
                    slot.semantic_content_hash for slot in self.source_slots
                ),
                "authority_policy_hash": self.authority_policy_hash,
                "decision_schedule_hash": self.decision_schedule_hash,
                "trace_requirement_profile_id": self.trace_requirement_profile_id,
                "capture_policy_id": self.capture_policy_id,
                "allowed_intent_kinds": self.allowed_intent_kinds,
                "operator_control_lease_sec": self.operator_control_lease_sec,
                "explicit_exclusions": self.explicit_exclusions,
            },
            domain=CONTROL_STATION_MANIFEST_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class StationIntentDraft:
    client_id: str
    client_sequence: int
    intent_kind: str
    produced_at: float
    produced_clock_domain_id: str
    expected_state_revision: int
    manifest_hash: str
    subject_id: str | None = None
    resource_binding_id: str | None = None
    operation_spec_id: str | None = None
    reason: str = "operator_request"

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_id", _required(self.client_id, field_name="client_id"))
        object.__setattr__(
            self,
            "client_sequence",
            _nonnegative_integer(self.client_sequence, field_name="client_sequence"),
        )
        kind = _required(self.intent_kind, field_name="intent_kind")
        if kind not in CONTROL_STATION_INTENT_KINDS_V1:
            raise ValueError(f"Unsupported intent_kind: {kind!r}.")
        object.__setattr__(self, "intent_kind", kind)
        object.__setattr__(
            self,
            "produced_at",
            _finite(self.produced_at, field_name="produced_at"),
        )
        object.__setattr__(
            self,
            "produced_clock_domain_id",
            _required(
                self.produced_clock_domain_id,
                field_name="produced_clock_domain_id",
            ),
        )
        object.__setattr__(
            self,
            "expected_state_revision",
            _nonnegative_integer(
                self.expected_state_revision,
                field_name="expected_state_revision",
            ),
        )
        object.__setattr__(
            self,
            "manifest_hash",
            _required(self.manifest_hash, field_name="manifest_hash"),
        )
        object.__setattr__(self, "subject_id", _optional(self.subject_id))
        object.__setattr__(
            self,
            "resource_binding_id",
            _optional(self.resource_binding_id),
        )
        object.__setattr__(
            self,
            "operation_spec_id",
            _optional(self.operation_spec_id),
        )
        object.__setattr__(self, "reason", _required(self.reason, field_name="reason"))


@dataclass(frozen=True)
class StationIntent:
    intent_id: str
    runtime_session_id: str
    control_domain_id: str
    client_id: str
    client_sequence: int
    intent_kind: str
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    admission_sequence: int
    expected_state_revision: int
    manifest_hash: str
    subject_id: str | None
    resource_binding_id: str | None
    operation_spec_id: str | None
    reason: str
    semantic_content_hash: str


@dataclass(frozen=True)
class StationIntentAdmissionReceipt:
    intent_id: str
    runtime_session_id: str
    control_domain_id: str
    client_id: str
    client_sequence: int
    intent_kind: str
    available_at: float
    available_clock_domain_id: str
    admission_sequence: int
    accepted: bool
    status: str
    state_revision_before: int
    state_revision_after: int
    manifest_hash: str
    resulting_state_hash: str
    intent_hash: str | None = None

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "intent_id": self.intent_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "client_id": self.client_id,
            "client_sequence": self.client_sequence,
            "intent_kind": self.intent_kind,
            "available_at": self.available_at,
            "available_clock_domain_id": self.available_clock_domain_id,
            "admission_sequence": self.admission_sequence,
            "accepted": self.accepted,
            "status": self.status,
            "state_revision_before": self.state_revision_before,
            "state_revision_after": self.state_revision_after,
            "manifest_hash": self.manifest_hash,
            "resulting_state_hash": self.resulting_state_hash,
        }
        if self.intent_hash is not None:
            payload["intent_hash"] = self.intent_hash
        return canonical_content_hash(
            payload,
            domain=STATION_INTENT_RECEIPT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class StationOperationResultDraft:
    source_intent_id: str
    operation_kind: str
    subject_id: str
    succeeded: bool
    completed_at: float
    completed_clock_domain_id: str
    status: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_intent_id",
            _required(self.source_intent_id, field_name="source_intent_id"),
        )
        operation_kind = _required(self.operation_kind, field_name="operation_kind")
        if operation_kind not in CONTROL_STATION_INTENT_KINDS_V1:
            raise ValueError(f"Unsupported operation_kind: {operation_kind!r}.")
        object.__setattr__(self, "operation_kind", operation_kind)
        object.__setattr__(
            self,
            "subject_id",
            _required(self.subject_id, field_name="subject_id"),
        )
        object.__setattr__(self, "succeeded", bool(self.succeeded))
        object.__setattr__(
            self,
            "completed_at",
            _finite(self.completed_at, field_name="completed_at"),
        )
        object.__setattr__(
            self,
            "completed_clock_domain_id",
            _required(
                self.completed_clock_domain_id,
                field_name="completed_clock_domain_id",
            ),
        )
        object.__setattr__(self, "status", _required(self.status, field_name="status"))
        object.__setattr__(
            self,
            "evidence_refs",
            _unique_sorted(self.evidence_refs, field_name="evidence_refs"),
        )


@dataclass(frozen=True)
class StationOperationResult:
    operation_result_id: str
    runtime_session_id: str
    control_domain_id: str
    source_intent_id: str
    operation_kind: str
    subject_id: str
    succeeded: bool
    completed_at: float
    completed_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    admission_sequence: int
    status: str
    evidence_refs: tuple[str, ...]
    semantic_content_hash: str


@dataclass(frozen=True)
class OperatorControlLease:
    lease_id: str
    client_id: str
    lease_epoch: int
    acquired_intent_id: str
    acquired_at: float
    expires_at: float

    def __post_init__(self) -> None:
        for field_name in ("lease_id", "client_id", "acquired_intent_id"):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "lease_epoch",
            _nonnegative_integer(self.lease_epoch, field_name="lease_epoch"),
        )
        acquired = _finite(self.acquired_at, field_name="acquired_at")
        expires = _finite(self.expires_at, field_name="expires_at")
        if expires <= acquired:
            raise ValueError("Operator control lease must expire after acquisition.")
        object.__setattr__(self, "acquired_at", acquired)
        object.__setattr__(self, "expires_at", expires)

    def active_at(self, timestamp: float) -> bool:
        current = _finite(timestamp, field_name="timestamp")
        return self.acquired_at <= current < self.expires_at

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "lease_id": self.lease_id,
                "client_id": self.client_id,
                "lease_epoch": self.lease_epoch,
                "acquired_intent_id": self.acquired_intent_id,
                "acquired_at": self.acquired_at,
                "expires_at": self.expires_at,
            },
            domain=OPERATOR_CONTROL_LEASE_HASH_DOMAIN_V1,
        )


__all__ = [
    "CONTROL_STATION_INTENT_KINDS_V1",
    "CONTROL_STATION_SOURCE_KINDS_V1",
    "CONTROL_STATION_SOURCE_USES_V1",
    "CONTROL_STATION_TARGET_KINDS_V1",
    "ControlStationRunManifest",
    "ExecutionDomainBinding",
    "OperatorControlLease",
    "SourceSlotSpec",
    "StationIntent",
    "StationIntentAdmissionReceipt",
    "StationIntentDraft",
    "StationOperationResult",
    "StationOperationResultDraft",
]
