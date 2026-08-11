"""Deterministic CS-0 Control Station lifecycle reducer.

The reducer manages application intent/state only.  An accepted station
intent is not a motion permit, authority lease, backend receipt, or capture
commit.  Later adapters must produce those facts through their existing
owners.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import (
    STATION_OPERATION_RESULT_HASH_DOMAIN_V1,
    STATION_INTENT_HASH_DOMAIN_V1,
    ControlStationRunManifest,
    OperatorControlLease,
    StationIntent,
    StationIntentAdmissionReceipt,
    StationIntentDraft,
    StationOperationResult,
    StationOperationResultDraft,
)


CONTROL_STATION_LIFECYCLE_STATES_V1 = (
    "configured",
    "connected",
    "preflighted",
    "shadow_ready",
    "armed",
    "running",
    "hold",
    "fault",
    "stopping",
    "stopped",
)
SOURCE_SLOT_RUNTIME_STATES_V1 = (
    "disconnected",
    "connect_requested",
    "connected",
    "disconnect_requested",
    "preflight_requested",
    "preflighted",
    "shadow_start_requested",
    "shadow",
    "shadow_stop_requested",
    "candidate",
    "active",
    "faulted",
    "stopped",
)
TARGET_RUNTIME_STATES_V1 = (
    "configured",
    "connect_requested",
    "connected",
    "disconnect_requested",
    "preflight_requested",
    "preflighted",
    "faulted",
    "stopped",
)

CONTROL_STATION_STATE_HASH_DOMAIN_V1 = "dexteleop.control_station.state.v1"
CONTROL_STATION_TRANSITION_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.transition.v1"
)
PENDING_STATION_OPERATION_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.pending_operation.v1"
)
CONTROL_STATION_OPERATION_TRANSITION_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.operation_transition.v1"
)
HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1 = "human_takeover_ready"
HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1 = "human_takeover_held"
HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1 = "human_takeover_faulted"
COMPOSITION_SESSION_OPERATION_READY_STATUS_V1 = "composition_session_ready"
COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1 = "composition_session_rearmed"
COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1 = "composition_session_held"
COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1 = "composition_session_stopped"

_LEASE_BYPASS_INTENTS = frozenset({"acquire_control_lease", "hold", "estop"})
_IMPLEMENTED_INTENTS = frozenset(
    {
        "acquire_control_lease",
        "release_control_lease",
        "connect_source",
        "disconnect_source",
        "connect_target",
        "disconnect_target",
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
        "stop_runtime",
    }
)

_PENDING_OPERATION_STATE_BY_INTENT = {
    "connect_source": "connect_requested",
    "disconnect_source": "disconnect_requested",
    "run_preflight": "preflight_requested",
    "start_shadow": "shadow_start_requested",
    "stop_shadow": "shadow_stop_requested",
    "connect_target": "connect_requested",
    "disconnect_target": "disconnect_requested",
}


@dataclass(frozen=True)
class PendingStationOperation:
    source_intent_id: str
    operation_kind: str
    subject_id: str
    prior_runtime_state: str
    requested_runtime_state: str
    resource_binding_id: str | None = None
    resource_prior_runtime_state: str | None = None
    operation_spec_id: str | None = None

    def __post_init__(self) -> None:
        optional = {
            "resource_binding_id": self.resource_binding_id,
            "resource_prior_runtime_state": self.resource_prior_runtime_state,
            "operation_spec_id": self.operation_spec_id,
        }
        normalized = {
            name: None if value is None else str(value).strip()
            for name, value in optional.items()
        }
        if any(value == "" for value in normalized.values()):
            raise ValueError("Pending operation binding fields must be non-empty.")
        supplied = {name for name, value in normalized.items() if value is not None}
        if supplied and supplied != set(normalized):
            raise ValueError(
                "Multi-source pending operation binding fields must be supplied together."
            )
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "source_intent_id": self.source_intent_id,
            "operation_kind": self.operation_kind,
            "subject_id": self.subject_id,
            "prior_runtime_state": self.prior_runtime_state,
            "requested_runtime_state": self.requested_runtime_state,
        }
        if self.resource_binding_id is not None:
            payload.update(
                {
                    "resource_binding_id": self.resource_binding_id,
                    "resource_prior_runtime_state": (
                        self.resource_prior_runtime_state
                    ),
                    "operation_spec_id": self.operation_spec_id,
                }
            )
        return canonical_content_hash(
            payload,
            domain=PENDING_STATION_OPERATION_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ControlStationState:
    runtime_session_id: str
    manifest_hash: str
    control_domain_id: str
    lifecycle_state: str
    state_revision: int
    source_state_by_slot: Mapping[str, str]
    target_state: str
    pending_operation_by_subject: Mapping[str, PendingStationOperation]
    last_client_sequence_by_id: Mapping[str, int]
    operator_control_lease: OperatorControlLease | None = None
    last_intent_id: str | None = None

    def __post_init__(self) -> None:
        if not str(self.runtime_session_id).strip():
            raise ValueError("runtime_session_id must be non-empty.")
        if not str(self.manifest_hash).strip():
            raise ValueError("manifest_hash must be non-empty.")
        if not str(self.control_domain_id).strip():
            raise ValueError("control_domain_id must be non-empty.")
        if self.lifecycle_state not in CONTROL_STATION_LIFECYCLE_STATES_V1:
            raise ValueError("Unsupported Control Station lifecycle state.")
        revision = int(self.state_revision)
        if isinstance(self.state_revision, bool) or revision != self.state_revision or revision < 0:
            raise ValueError("state_revision must be a non-negative integer.")
        normalized_sources = {
            str(slot_id): str(status)
            for slot_id, status in self.source_state_by_slot.items()
        }
        if any(not slot_id for slot_id in normalized_sources):
            raise ValueError("source_state_by_slot requires non-empty slot IDs.")
        if any(status not in SOURCE_SLOT_RUNTIME_STATES_V1 for status in normalized_sources.values()):
            raise ValueError("Unsupported source slot runtime state.")
        if self.target_state not in TARGET_RUNTIME_STATES_V1:
            raise ValueError("Unsupported target runtime state.")
        normalized_pending = {
            str(subject_key): operation
            for subject_key, operation in self.pending_operation_by_subject.items()
        }
        if any(not subject_key for subject_key in normalized_pending):
            raise ValueError("Pending operation subject keys must be non-empty.")
        if any(
            not isinstance(operation, PendingStationOperation)
            for operation in normalized_pending.values()
        ):
            raise TypeError("Pending operations must be PendingStationOperation values.")
        normalized_client_sequences: dict[str, int] = {}
        for client_id, raw_sequence in self.last_client_sequence_by_id.items():
            normalized_client_id = str(client_id).strip()
            sequence = int(raw_sequence)
            if (
                not normalized_client_id
                or isinstance(raw_sequence, bool)
                or sequence != raw_sequence
                or sequence < 0
            ):
                raise ValueError("Client sequence state is invalid.")
            normalized_client_sequences[normalized_client_id] = sequence
        if self.operator_control_lease is not None and not isinstance(
            self.operator_control_lease, OperatorControlLease
        ):
            raise TypeError("operator_control_lease must be an OperatorControlLease.")
        object.__setattr__(
            self,
            "source_state_by_slot",
            MappingProxyType(dict(sorted(normalized_sources.items()))),
        )
        object.__setattr__(
            self,
            "pending_operation_by_subject",
            MappingProxyType(dict(sorted(normalized_pending.items()))),
        )
        object.__setattr__(
            self,
            "last_client_sequence_by_id",
            MappingProxyType(dict(sorted(normalized_client_sequences.items()))),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "runtime_session_id": self.runtime_session_id,
                "manifest_hash": self.manifest_hash,
                "control_domain_id": self.control_domain_id,
                "lifecycle_state": self.lifecycle_state,
                "state_revision": self.state_revision,
                "source_state_by_slot": self.source_state_by_slot,
                "target_state": self.target_state,
                "pending_operation_hash_by_subject": {
                    subject_key: operation.semantic_content_hash
                    for subject_key, operation in self.pending_operation_by_subject.items()
                },
                "last_client_sequence_by_id": self.last_client_sequence_by_id,
                "operator_control_lease_hash": (
                    None
                    if self.operator_control_lease is None
                    else self.operator_control_lease.semantic_content_hash
                ),
                "last_intent_id": self.last_intent_id,
            },
            domain=CONTROL_STATION_STATE_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ControlStationTransition:
    intent: StationIntent
    state_before: ControlStationState
    state_after: ControlStationState
    receipt: StationIntentAdmissionReceipt

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "intent_hash": self.intent.semantic_content_hash,
                "state_before_hash": self.state_before.semantic_content_hash,
                "state_after_hash": self.state_after.semantic_content_hash,
                "receipt_hash": self.receipt.semantic_content_hash,
            },
            domain=CONTROL_STATION_TRANSITION_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class ControlStationOperationTransition:
    result: StationOperationResult
    state_before: ControlStationState
    state_after: ControlStationState
    accepted: bool
    status: str

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "result_hash": self.result.semantic_content_hash,
                "state_before_hash": self.state_before.semantic_content_hash,
                "state_after_hash": self.state_after.semantic_content_hash,
                "accepted": self.accepted,
                "status": self.status,
            },
            domain=CONTROL_STATION_OPERATION_TRANSITION_HASH_DOMAIN_V1,
        )


def initial_control_station_state(
    manifest: ControlStationRunManifest,
    *,
    runtime_session_id: str,
) -> ControlStationState:
    if not isinstance(manifest, ControlStationRunManifest):
        raise TypeError("manifest must be a ControlStationRunManifest.")
    session_id = str(runtime_session_id).strip()
    if not session_id:
        raise ValueError("runtime_session_id must be non-empty.")
    return ControlStationState(
        runtime_session_id=session_id,
        manifest_hash=manifest.semantic_content_hash,
        control_domain_id=manifest.control_domain_id,
        lifecycle_state="configured",
        state_revision=0,
        source_state_by_slot={slot.source_slot_id: "disconnected" for slot in manifest.source_slots},
        target_state="configured",
        pending_operation_by_subject={},
        last_client_sequence_by_id={},
    )


def admit_station_intent(
    draft: StationIntentDraft,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    available_at: float,
    available_clock_domain_id: str,
    admission_sequence: int,
) -> StationIntent:
    if not isinstance(draft, StationIntentDraft):
        raise TypeError("draft must be a StationIntentDraft.")
    session_id = str(runtime_session_id).strip()
    domain_id = str(control_domain_id).strip()
    clock_id = str(available_clock_domain_id).strip()
    if not session_id or not domain_id or not clock_id:
        raise ValueError("Intent receiver identity/clock must be non-empty.")
    admitted_at = float(available_at)
    if not math.isfinite(admitted_at):
        raise ValueError("available_at must be finite.")
    if (
        draft.produced_clock_domain_id == clock_id
        and admitted_at < draft.produced_at
    ):
        raise ValueError("available_at cannot predate produced_at.")
    sequence = int(admission_sequence)
    if isinstance(admission_sequence, bool) or sequence != admission_sequence or sequence < 0:
        raise ValueError("admission_sequence must be a non-negative integer.")
    identity_payload = {
        "runtime_session_id": session_id,
        "control_domain_id": domain_id,
        "client_id": draft.client_id,
        "client_sequence": draft.client_sequence,
        "admission_sequence": sequence,
    }
    intent_id = "station-intent:" + canonical_content_hash(
        identity_payload,
        domain="dexteleop.control_station.intent_identity.v1",
    )
    payload = {
        "intent_id": intent_id,
        **identity_payload,
        "intent_kind": draft.intent_kind,
        "produced_at": draft.produced_at,
        "produced_clock_domain_id": draft.produced_clock_domain_id,
        "available_at": admitted_at,
        "available_clock_domain_id": clock_id,
        "expected_state_revision": draft.expected_state_revision,
        "manifest_hash": draft.manifest_hash,
        "subject_id": draft.subject_id,
        "resource_binding_id": draft.resource_binding_id,
        "operation_spec_id": draft.operation_spec_id,
        "reason": draft.reason,
    }
    return StationIntent(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=STATION_INTENT_HASH_DOMAIN_V1,
        ),
    )


def admit_station_operation_result(
    draft: StationOperationResultDraft,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    available_at: float,
    available_clock_domain_id: str,
    admission_sequence: int,
) -> StationOperationResult:
    if not isinstance(draft, StationOperationResultDraft):
        raise TypeError("draft must be a StationOperationResultDraft.")
    session_id = str(runtime_session_id).strip()
    domain_id = str(control_domain_id).strip()
    clock_id = str(available_clock_domain_id).strip()
    if not session_id or not domain_id or not clock_id:
        raise ValueError("Operation-result receiver identity/clock must be non-empty.")
    admitted_at = float(available_at)
    if not math.isfinite(admitted_at):
        raise ValueError("available_at must be finite.")
    if (
        draft.completed_clock_domain_id == clock_id
        and admitted_at < draft.completed_at
    ):
        raise ValueError("available_at cannot predate completed_at.")
    sequence = int(admission_sequence)
    if isinstance(admission_sequence, bool) or sequence != admission_sequence or sequence < 0:
        raise ValueError("admission_sequence must be a non-negative integer.")
    identity_payload = {
        "runtime_session_id": session_id,
        "control_domain_id": domain_id,
        "source_intent_id": draft.source_intent_id,
        "admission_sequence": sequence,
    }
    result_id = "station-operation-result:" + canonical_content_hash(
        identity_payload,
        domain="dexteleop.control_station.operation_result_identity.v1",
    )
    payload = {
        "operation_result_id": result_id,
        **identity_payload,
        "operation_kind": draft.operation_kind,
        "subject_id": draft.subject_id,
        "succeeded": draft.succeeded,
        "completed_at": draft.completed_at,
        "completed_clock_domain_id": draft.completed_clock_domain_id,
        "available_at": admitted_at,
        "available_clock_domain_id": clock_id,
        "status": draft.status,
        "evidence_refs": draft.evidence_refs,
    }
    return StationOperationResult(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=STATION_OPERATION_RESULT_HASH_DOMAIN_V1,
        ),
    )


def _derived_lifecycle(
    manifest: ControlStationRunManifest,
    source_states: Mapping[str, str],
    target_state: str,
) -> str:
    if any(status == "active" for status in source_states.values()):
        return "running"
    if any(status == "candidate" for status in source_states.values()):
        return "armed"
    if any(status == "shadow" for status in source_states.values()):
        return "shadow_ready"
    required = tuple(slot for slot in manifest.source_slots if slot.required_for_run)
    required_preflighted = all(
        source_states[slot.source_slot_id] in {"preflighted", "shadow", "candidate", "active"}
        for slot in required
    )
    if target_state == "preflighted" and required_preflighted:
        return "preflighted"
    required_connected = all(
        source_states[slot.source_slot_id]
        in {"connected", "preflighted", "shadow", "candidate", "active"}
        for slot in required
    )
    if target_state in {"connected", "preflighted"} and required_connected:
        return "connected"
    return "configured"


def _rejected_transition(
    state: ControlStationState,
    intent: StationIntent,
    *,
    status: str,
) -> ControlStationTransition:
    receipt = StationIntentAdmissionReceipt(
        intent_id=intent.intent_id,
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        client_id=intent.client_id,
        client_sequence=intent.client_sequence,
        intent_kind=intent.intent_kind,
        available_at=intent.available_at,
        available_clock_domain_id=intent.available_clock_domain_id,
        admission_sequence=intent.admission_sequence,
        accepted=False,
        status=status,
        state_revision_before=state.state_revision,
        state_revision_after=state.state_revision,
        manifest_hash=state.manifest_hash,
        resulting_state_hash=state.semantic_content_hash,
        intent_hash=(
            intent.semantic_content_hash
            if intent.intent_kind == "request_residual_rebase_and_clear"
            else None
        ),
    )
    return ControlStationTransition(intent, state, state, receipt)


def _accepted_transition(
    state: ControlStationState,
    intent: StationIntent,
    *,
    lifecycle_state: str | None = None,
    source_states: Mapping[str, str] | None = None,
    target_state: str | None = None,
    pending_operations: Mapping[str, PendingStationOperation] | None = None,
    lease: OperatorControlLease | None | object = ...,
    status: str = "accepted",
) -> ControlStationTransition:
    next_state = ControlStationState(
        runtime_session_id=state.runtime_session_id,
        manifest_hash=state.manifest_hash,
        control_domain_id=state.control_domain_id,
        lifecycle_state=(
            state.lifecycle_state if lifecycle_state is None else lifecycle_state
        ),
        state_revision=state.state_revision + 1,
        source_state_by_slot=(
            state.source_state_by_slot if source_states is None else source_states
        ),
        target_state=state.target_state if target_state is None else target_state,
        pending_operation_by_subject=(
            state.pending_operation_by_subject
            if pending_operations is None
            else pending_operations
        ),
        last_client_sequence_by_id={
            **state.last_client_sequence_by_id,
            intent.client_id: intent.client_sequence,
        },
        operator_control_lease=(
            state.operator_control_lease if lease is ... else lease
        ),
        last_intent_id=intent.intent_id,
    )
    receipt = StationIntentAdmissionReceipt(
        intent_id=intent.intent_id,
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        client_id=intent.client_id,
        client_sequence=intent.client_sequence,
        intent_kind=intent.intent_kind,
        available_at=intent.available_at,
        available_clock_domain_id=intent.available_clock_domain_id,
        admission_sequence=intent.admission_sequence,
        accepted=True,
        status=status,
        state_revision_before=state.state_revision,
        state_revision_after=next_state.state_revision,
        manifest_hash=state.manifest_hash,
        resulting_state_hash=next_state.semantic_content_hash,
        intent_hash=(
            intent.semantic_content_hash
            if intent.intent_kind == "request_residual_rebase_and_clear"
            else None
        ),
    )
    return ControlStationTransition(intent, state, next_state, receipt)


def apply_station_intent(
    manifest: ControlStationRunManifest,
    state: ControlStationState,
    intent: StationIntent,
) -> ControlStationTransition:
    """Apply one admitted application intent without causing external effects."""

    if not isinstance(manifest, ControlStationRunManifest):
        raise TypeError("manifest must be a ControlStationRunManifest.")
    if not isinstance(state, ControlStationState) or not isinstance(intent, StationIntent):
        raise TypeError("state/intent have invalid types.")
    if state.manifest_hash != manifest.semantic_content_hash:
        return _rejected_transition(state, intent, status="state_manifest_mismatch")
    if intent.manifest_hash != state.manifest_hash:
        return _rejected_transition(state, intent, status="intent_manifest_mismatch")
    if (
        intent.runtime_session_id != state.runtime_session_id
        or intent.control_domain_id != state.control_domain_id
    ):
        return _rejected_transition(state, intent, status="intent_runtime_mismatch")
    last_client_sequence = state.last_client_sequence_by_id.get(intent.client_id)
    if (
        last_client_sequence is not None
        and intent.client_sequence <= last_client_sequence
    ):
        return _rejected_transition(
            state,
            intent,
            status="duplicate_or_out_of_order_client_sequence",
        )
    if intent.expected_state_revision != state.state_revision:
        return _rejected_transition(state, intent, status="stale_state_revision")
    if intent.intent_kind not in manifest.allowed_intent_kinds:
        return _rejected_transition(state, intent, status="intent_not_allowed_by_manifest")
    if intent.intent_kind not in _IMPLEMENTED_INTENTS:
        return _rejected_transition(state, intent, status="intent_not_implemented")
    if state.lifecycle_state == "stopped":
        return _rejected_transition(state, intent, status="runtime_stopped")

    if state.lifecycle_state in {"hold", "fault"} and intent.intent_kind not in {
        "acquire_control_lease",
        "release_control_lease",
        "disconnect_source",
        "disconnect_target",
        "stop_shadow",
        "release_candidate",
        "release_motion_authority",
        "rearm_composition_session",
        "stop_composition_session",
        "hold",
        "estop",
        "stop_runtime",
    }:
        return _rejected_transition(state, intent, status="runtime_hold_or_fault_latched")

    lease = state.operator_control_lease
    if lease is not None and not lease.active_at(intent.available_at):
        lease = None
    if intent.intent_kind not in _LEASE_BYPASS_INTENTS:
        if lease is None or lease.client_id != intent.client_id:
            return _rejected_transition(state, intent, status="operator_control_lease_required")

    kind = intent.intent_kind
    if kind == "acquire_control_lease":
        if lease is not None and lease.client_id != intent.client_id:
            return _rejected_transition(state, intent, status="operator_control_lease_held")
        lease_epoch = 0 if state.operator_control_lease is None else state.operator_control_lease.lease_epoch + 1
        lease_id = "station-lease:" + canonical_content_hash(
            {
                "runtime_session_id": state.runtime_session_id,
                "client_id": intent.client_id,
                "lease_epoch": lease_epoch,
                "intent_id": intent.intent_id,
            },
            domain="dexteleop.control_station.operator_lease_identity.v1",
        )
        new_lease = OperatorControlLease(
            lease_id=lease_id,
            client_id=intent.client_id,
            lease_epoch=lease_epoch,
            acquired_intent_id=intent.intent_id,
            acquired_at=intent.available_at,
            expires_at=intent.available_at + manifest.operator_control_lease_sec,
        )
        return _accepted_transition(state, intent, lease=new_lease)

    if kind == "release_control_lease":
        return _accepted_transition(state, intent, lease=None)

    source_states = dict(state.source_state_by_slot)
    pending_operations = dict(state.pending_operation_by_subject)
    pending_exclusive_kinds = {
        operation.operation_kind
        for operation in pending_operations.values()
        if operation.operation_kind
        in {
            "request_human_takeover",
            "start_composition_session",
            "rearm_composition_session",
            "stop_composition_session",
        }
    }
    exclusive_operation_pending = bool(pending_exclusive_kinds)
    if exclusive_operation_pending and kind not in {
        "hold",
        "estop",
        "stop_runtime",
    }:
        return _rejected_transition(
            state,
            intent,
            status=(
                "human_takeover_operation_pending"
                if pending_exclusive_kinds == {"request_human_takeover"}
                else "composition_session_operation_pending"
            ),
        )
    if kind == "request_human_takeover":
        human_slot_id = intent.subject_id
        previous_slot_id = intent.resource_binding_id
        if (
            human_slot_id is None
            or previous_slot_id is None
            or intent.operation_spec_id is None
        ):
            return _rejected_transition(
                state,
                intent,
                status="human_takeover_binding_incomplete",
            )
        if (
            human_slot_id not in source_states
            or previous_slot_id not in source_states
            or human_slot_id == previous_slot_id
        ):
            return _rejected_transition(
                state,
                intent,
                status="human_takeover_source_binding_mismatch",
            )
        if manifest.execution_binding.target_kind == "null":
            return _rejected_transition(
                state,
                intent,
                status="target_has_no_motion_authority",
            )
        if "station_motion_authority_lifecycle" in manifest.explicit_exclusions:
            return _rejected_transition(
                state,
                intent,
                status="station_motion_authority_lifecycle_excluded",
            )
        human_slot = manifest.source_slot(human_slot_id)
        previous_slot = manifest.source_slot(previous_slot_id)
        if human_slot.source_kind != "human" or "authority" not in (
            human_slot.supported_uses
        ):
            return _rejected_transition(
                state,
                intent,
                status="takeover_target_is_not_authoritative_human",
            )
        if previous_slot.source_kind == "human":
            return _rejected_transition(
                state,
                intent,
                status="takeover_previous_source_must_be_nonhuman",
            )
        if state.target_state != "preflighted":
            return _rejected_transition(
                state,
                intent,
                status="target_not_preflighted",
            )
        if state.lifecycle_state != "running":
            return _rejected_transition(
                state,
                intent,
                status="human_takeover_requires_running_runtime",
            )
        if source_states[previous_slot_id] != "active":
            return _rejected_transition(
                state,
                intent,
                status="takeover_previous_source_not_active",
            )
        if source_states[human_slot_id] != "candidate":
            return _rejected_transition(
                state,
                intent,
                status="takeover_human_source_not_candidate",
            )
        active_slots = {
            slot_id
            for slot_id, runtime_state in source_states.items()
            if runtime_state == "active"
        }
        if active_slots != {previous_slot_id}:
            return _rejected_transition(
                state,
                intent,
                status="takeover_requires_one_exact_active_source",
            )
        if pending_operations:
            return _rejected_transition(
                state,
                intent,
                status="operation_pending_prevents_human_takeover",
            )
        subject_key = f"handoff:{previous_slot_id}->{human_slot_id}"
        pending_operations[subject_key] = PendingStationOperation(
            source_intent_id=intent.intent_id,
            operation_kind=kind,
            subject_id=human_slot_id,
            prior_runtime_state=source_states[human_slot_id],
            requested_runtime_state="active",
            resource_binding_id=previous_slot_id,
            resource_prior_runtime_state=source_states[previous_slot_id],
            operation_spec_id=intent.operation_spec_id,
        )
        return _accepted_transition(
            state,
            intent,
            pending_operations=pending_operations,
            lease=lease,
            status="operation_requested",
        )

    if kind == "request_residual_rebase_and_clear":
        composer_slot_id = intent.subject_id
        anchor_binding_id = intent.resource_binding_id
        if (
            composer_slot_id is None
            or anchor_binding_id is None
            or intent.operation_spec_id is None
        ):
            return _rejected_transition(
                state,
                intent,
                status="residual_rebase_binding_incomplete",
            )
        if composer_slot_id not in source_states:
            return _rejected_transition(
                state,
                intent,
                status="residual_rebase_composer_unknown",
            )
        composer_slot = manifest.source_slot(composer_slot_id)
        if (
            composer_slot.source_kind != "composite"
            or source_states[composer_slot_id] != "active"
        ):
            return _rejected_transition(
                state,
                intent,
                status="residual_rebase_requires_active_composer",
            )
        if state.lifecycle_state != "running":
            return _rejected_transition(
                state,
                intent,
                status="residual_rebase_requires_running_session",
            )
        return _accepted_transition(
            state,
            intent,
            lease=lease,
            status="residual_rebase_intent_accepted",
        )

    if kind in {
        "start_composition_session",
        "rearm_composition_session",
        "stop_composition_session",
    }:
        composer_slot_id = intent.subject_id
        target_binding_id = intent.resource_binding_id
        if (
            composer_slot_id is None
            or target_binding_id is None
            or intent.operation_spec_id is None
        ):
            return _rejected_transition(
                state,
                intent,
                status="composition_session_binding_incomplete",
            )
        if (
            composer_slot_id not in source_states
            or target_binding_id != manifest.execution_binding.binding_id
        ):
            return _rejected_transition(
                state,
                intent,
                status="composition_session_binding_mismatch",
            )
        if manifest.execution_binding.target_kind == "null":
            return _rejected_transition(
                state,
                intent,
                status="target_has_no_motion_authority",
            )
        if "station_motion_authority_lifecycle" in manifest.explicit_exclusions:
            return _rejected_transition(
                state,
                intent,
                status="station_motion_authority_lifecycle_excluded",
            )
        composer_slot = manifest.source_slot(composer_slot_id)
        if (
            composer_slot.source_kind != "composite"
            or "authority" not in composer_slot.supported_uses
        ):
            return _rejected_transition(
                state,
                intent,
                status="composition_session_subject_is_not_composer_authority",
            )
        if state.target_state != "preflighted":
            return _rejected_transition(
                state,
                intent,
                status="target_not_preflighted",
            )
        current = source_states[composer_slot_id]
        expected = (
            "candidate"
            if kind == "start_composition_session"
            else "active"
        )
        if current != expected:
            return _rejected_transition(
                state,
                intent,
                status=f"composition_session_composer_not_{expected}",
            )
        if kind == "start_composition_session":
            if state.lifecycle_state != "armed":
                return _rejected_transition(
                    state,
                    intent,
                    status="composition_session_requires_armed_runtime",
                )
            if any(
                slot_id != composer_slot_id and status == "active"
                for slot_id, status in source_states.items()
            ):
                return _rejected_transition(
                    state,
                    intent,
                    status="composition_session_parallel_authority_forbidden",
                )
        elif kind == "rearm_composition_session":
            if state.lifecycle_state != "hold":
                return _rejected_transition(
                    state,
                    intent,
                    status="composition_session_rearm_requires_hold",
                )
            if any(
                slot_id != composer_slot_id and status == "active"
                for slot_id, status in source_states.items()
            ):
                return _rejected_transition(
                    state,
                    intent,
                    status="composition_session_parallel_authority_forbidden",
                )
        elif state.lifecycle_state not in {"running", "hold", "fault"}:
            return _rejected_transition(
                state,
                intent,
                status="composition_session_not_running",
            )
        if pending_operations:
            return _rejected_transition(
                state,
                intent,
                status="operation_pending_prevents_composition_session",
            )
        subject_key = f"composition:{composer_slot_id}"
        pending_operations[subject_key] = PendingStationOperation(
            source_intent_id=intent.intent_id,
            operation_kind=kind,
            subject_id=composer_slot_id,
            prior_runtime_state=current,
            requested_runtime_state=(
                "candidate"
                if kind == "stop_composition_session"
                else "active"
            ),
            resource_binding_id=target_binding_id,
            resource_prior_runtime_state=state.target_state,
            operation_spec_id=intent.operation_spec_id,
        )
        return _accepted_transition(
            state,
            intent,
            pending_operations=pending_operations,
            lease=lease,
            status="operation_requested",
        )

    if kind in {"connect_source", "disconnect_source", "start_shadow", "stop_shadow"}:
        subject = intent.subject_id
        if subject is None or subject not in source_states:
            return _rejected_transition(state, intent, status="unknown_source_slot")
        subject_key = f"source:{subject}"
        if subject_key in pending_operations:
            return _rejected_transition(state, intent, status="source_operation_pending")
        current = source_states[subject]
        if kind == "connect_source":
            if current != "disconnected":
                return _rejected_transition(state, intent, status="source_not_disconnected")
        elif kind == "disconnect_source":
            if current in {"shadow", "candidate", "active"}:
                return _rejected_transition(state, intent, status="source_must_leave_control_first")
        elif kind == "start_shadow":
            slot = manifest.source_slot(subject)
            if "shadow" not in slot.supported_uses:
                return _rejected_transition(state, intent, status="source_shadow_not_supported")
            if current != "preflighted" or state.target_state != "preflighted":
                return _rejected_transition(state, intent, status="source_or_target_not_preflighted")
        else:
            if current != "shadow":
                return _rejected_transition(state, intent, status="source_not_in_shadow")
        requested_state = _PENDING_OPERATION_STATE_BY_INTENT[kind]
        source_states[subject] = requested_state
        pending_operations[subject_key] = PendingStationOperation(
            source_intent_id=intent.intent_id,
            operation_kind=kind,
            subject_id=subject,
            prior_runtime_state=current,
            requested_runtime_state=requested_state,
        )
        return _accepted_transition(
            state,
            intent,
            source_states=source_states,
            pending_operations=pending_operations,
            lease=lease,
            status="operation_requested",
        )

    if kind in {
        "request_candidate",
        "release_candidate",
        "request_motion_authority",
        "release_motion_authority",
    }:
        subject = intent.subject_id
        if subject is None or subject not in source_states:
            return _rejected_transition(state, intent, status="unknown_source_slot")
        if manifest.execution_binding.target_kind == "null":
            return _rejected_transition(
                state,
                intent,
                status="target_has_no_motion_authority",
            )
        if "station_motion_authority_lifecycle" in manifest.explicit_exclusions:
            return _rejected_transition(
                state,
                intent,
                status="station_motion_authority_lifecycle_excluded",
            )
        subject_key = f"source:{subject}"
        if subject_key in pending_operations:
            return _rejected_transition(state, intent, status="source_operation_pending")
        slot = manifest.source_slot(subject)
        current = source_states[subject]
        if kind == "request_candidate":
            if "candidate" not in slot.supported_uses:
                return _rejected_transition(
                    state,
                    intent,
                    status="source_candidate_not_supported",
                )
            if current != "preflighted" or state.target_state != "preflighted":
                return _rejected_transition(
                    state,
                    intent,
                    status="source_or_target_not_preflighted",
                )
            requested_state = "candidate"
        elif kind == "release_candidate":
            if current != "candidate":
                return _rejected_transition(
                    state,
                    intent,
                    status="source_not_candidate",
                )
            requested_state = "preflighted"
        elif kind == "request_motion_authority":
            if "authority" not in slot.supported_uses:
                return _rejected_transition(
                    state,
                    intent,
                    status="source_authority_not_supported",
                )
            if current != "candidate" or state.target_state != "preflighted":
                return _rejected_transition(
                    state,
                    intent,
                    status="source_not_candidate_or_target_not_preflighted",
                )
            other_active_slots = {
                slot_id
                for slot_id, runtime_state in source_states.items()
                if slot_id != subject and runtime_state == "active"
            }
            if other_active_slots:
                return _rejected_transition(
                    state,
                    intent,
                    status="active_source_requires_explicit_handoff",
                )
            requested_state = "active"
        else:
            if current != "active":
                return _rejected_transition(
                    state,
                    intent,
                    status="source_not_active",
                )
            requested_state = "candidate"
        pending_operations[subject_key] = PendingStationOperation(
            source_intent_id=intent.intent_id,
            operation_kind=kind,
            subject_id=subject,
            prior_runtime_state=current,
            requested_runtime_state=requested_state,
        )
        return _accepted_transition(
            state,
            intent,
            source_states=source_states,
            pending_operations=pending_operations,
            lease=lease,
            status="operation_requested",
        )

    if kind in {"connect_target", "disconnect_target"}:
        if intent.subject_id not in {None, manifest.execution_binding.binding_id}:
            return _rejected_transition(state, intent, status="target_binding_mismatch")
        subject = manifest.execution_binding.binding_id
        subject_key = f"target:{subject}"
        if subject_key in pending_operations:
            return _rejected_transition(state, intent, status="target_operation_pending")
        if kind == "connect_target":
            if state.target_state != "configured":
                return _rejected_transition(state, intent, status="target_not_configured")
        else:
            if any(
                value
                in {
                    "shadow_start_requested",
                    "shadow",
                    "shadow_stop_requested",
                    "candidate",
                    "active",
                }
                for value in source_states.values()
            ):
                return _rejected_transition(state, intent, status="active_source_prevents_target_disconnect")
            if any(key.startswith("source:") for key in pending_operations):
                return _rejected_transition(
                    state,
                    intent,
                    status="pending_source_operation_prevents_target_disconnect",
                )
            if state.target_state not in {"connected", "preflighted"}:
                return _rejected_transition(state, intent, status="target_not_connected")
        requested_state = _PENDING_OPERATION_STATE_BY_INTENT[kind]
        pending_operations[subject_key] = PendingStationOperation(
            source_intent_id=intent.intent_id,
            operation_kind=kind,
            subject_id=subject,
            prior_runtime_state=state.target_state,
            requested_runtime_state=requested_state,
        )
        return _accepted_transition(
            state,
            intent,
            target_state=requested_state,
            pending_operations=pending_operations,
            lease=lease,
            status="operation_requested",
        )

    if kind == "run_preflight":
        subject = intent.subject_id
        if subject == manifest.execution_binding.binding_id:
            if state.target_state != "connected":
                return _rejected_transition(state, intent, status="target_not_connected")
            subject_key = f"target:{subject}"
            if subject_key in pending_operations:
                return _rejected_transition(state, intent, status="target_operation_pending")
            pending_operations[subject_key] = PendingStationOperation(
                source_intent_id=intent.intent_id,
                operation_kind=kind,
                subject_id=subject,
                prior_runtime_state=state.target_state,
                requested_runtime_state="preflight_requested",
            )
            next_target = "preflight_requested"
        elif subject in source_states:
            if source_states[subject] != "connected":
                return _rejected_transition(state, intent, status="source_not_connected")
            subject_key = f"source:{subject}"
            if subject_key in pending_operations:
                return _rejected_transition(state, intent, status="source_operation_pending")
            pending_operations[subject_key] = PendingStationOperation(
                source_intent_id=intent.intent_id,
                operation_kind=kind,
                subject_id=subject,
                prior_runtime_state=source_states[subject],
                requested_runtime_state="preflight_requested",
            )
            source_states[subject] = "preflight_requested"
            next_target = state.target_state
        else:
            return _rejected_transition(state, intent, status="unknown_preflight_subject")
        return _accepted_transition(
            state,
            intent,
            source_states=source_states,
            target_state=next_target,
            pending_operations=pending_operations,
            lease=lease,
            status="operation_requested",
        )

    if kind == "hold":
        retained_operations = {
            key: operation
            for key, operation in pending_operations.items()
            if operation.operation_kind
            not in {
                "request_human_takeover",
                "start_composition_session",
                "rearm_composition_session",
                "stop_composition_session",
            }
        }
        return _accepted_transition(
            state,
            intent,
            lifecycle_state="hold",
            pending_operations=retained_operations,
            lease=lease,
        )
    if kind == "estop":
        retained_operations = {
            key: operation
            for key, operation in pending_operations.items()
            if operation.operation_kind
            not in {
                "request_human_takeover",
                "start_composition_session",
                "rearm_composition_session",
                "stop_composition_session",
            }
        }
        return _accepted_transition(
            state,
            intent,
            lifecycle_state="fault",
            pending_operations=retained_operations,
            lease=lease,
        )
    if kind == "stop_runtime":
        stopped_sources = {slot_id: "stopped" for slot_id in source_states}
        return _accepted_transition(
            state,
            intent,
            lifecycle_state="stopped",
            source_states=stopped_sources,
            target_state="stopped",
            pending_operations={},
            lease=None,
        )

    raise AssertionError(f"Unhandled Control Station intent: {kind!r}.")


def apply_station_operation_result(
    manifest: ControlStationRunManifest,
    state: ControlStationState,
    result: StationOperationResult,
) -> ControlStationOperationTransition:
    """Apply typed adapter completion without treating a request as success."""

    if not isinstance(manifest, ControlStationRunManifest):
        raise TypeError("manifest must be a ControlStationRunManifest.")
    if not isinstance(state, ControlStationState) or not isinstance(
        result, StationOperationResult
    ):
        raise TypeError("state/result have invalid types.")

    def rejected(status: str) -> ControlStationOperationTransition:
        return ControlStationOperationTransition(
            result=result,
            state_before=state,
            state_after=state,
            accepted=False,
            status=status,
        )

    if state.manifest_hash != manifest.semantic_content_hash:
        return rejected("state_manifest_mismatch")
    if (
        result.runtime_session_id != state.runtime_session_id
        or result.control_domain_id != state.control_domain_id
    ):
        return rejected("operation_result_runtime_mismatch")
    if state.lifecycle_state == "stopped":
        return rejected("runtime_stopped")

    if result.operation_kind in {
        "start_composition_session",
        "rearm_composition_session",
        "stop_composition_session",
    }:
        matching = tuple(
            (key, operation)
            for key, operation in state.pending_operation_by_subject.items()
            if operation.operation_kind == result.operation_kind
            and operation.subject_id == result.subject_id
        )
        if len(matching) != 1:
            return rejected("unknown_or_resolved_operation")
        subject_key, pending = matching[0]
        if pending.source_intent_id != result.source_intent_id:
            return rejected("operation_result_binding_mismatch")
        if (
            result.subject_id not in state.source_state_by_slot
            or pending.resource_binding_id
            != manifest.execution_binding.binding_id
            or state.target_state != pending.resource_prior_runtime_state
            or state.source_state_by_slot[result.subject_id]
            != pending.prior_runtime_state
        ):
            return rejected("composition_session_pending_state_changed")
        if result.succeeded and not result.evidence_refs:
            return rejected("operation_result_missing_control_evidence")
        expected_status = (
            COMPOSITION_SESSION_OPERATION_READY_STATUS_V1
            if result.operation_kind == "start_composition_session"
            else (
                COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1
                if result.operation_kind == "rearm_composition_session"
                else COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1
            )
        )
        if result.succeeded and result.status != expected_status:
            return rejected("composition_session_result_status_invalid")
        source_states = dict(state.source_state_by_slot)
        source_states[result.subject_id] = (
            pending.requested_runtime_state
            if result.succeeded
            else pending.prior_runtime_state
        )
        pending_operations = dict(state.pending_operation_by_subject)
        pending_operations.pop(subject_key)
        if result.operation_kind in {
            "start_composition_session",
            "rearm_composition_session",
        }:
            lifecycle = (
                "running"
                if result.succeeded
                else (
                    "hold"
                    if result.status
                    == COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1
                    else "fault"
                )
            )
        else:
            lifecycle = "hold" if result.succeeded else "fault"
        next_state = ControlStationState(
            runtime_session_id=state.runtime_session_id,
            manifest_hash=state.manifest_hash,
            control_domain_id=state.control_domain_id,
            lifecycle_state=lifecycle,
            state_revision=state.state_revision + 1,
            source_state_by_slot=source_states,
            target_state=state.target_state,
            pending_operation_by_subject=pending_operations,
            last_client_sequence_by_id=state.last_client_sequence_by_id,
            operator_control_lease=state.operator_control_lease,
            last_intent_id=state.last_intent_id,
        )
        return ControlStationOperationTransition(
            result=result,
            state_before=state,
            state_after=next_state,
            accepted=True,
            status=(
                "operation_succeeded"
                if result.succeeded
                else f"operation_failed:{result.status}"
            ),
        )

    if result.operation_kind == "request_human_takeover":
        if state.lifecycle_state in {"hold", "fault"}:
            return rejected("human_takeover_result_after_latched_safety_state")
        matching = tuple(
            (key, operation)
            for key, operation in state.pending_operation_by_subject.items()
            if operation.operation_kind == "request_human_takeover"
            and operation.subject_id == result.subject_id
        )
        if len(matching) != 1:
            return rejected("unknown_or_resolved_operation")
        subject_key, pending = matching[0]
        if pending.source_intent_id != result.source_intent_id:
            return rejected("operation_result_binding_mismatch")
        previous_slot_id = pending.resource_binding_id
        if (
            previous_slot_id is None
            or previous_slot_id not in state.source_state_by_slot
            or result.subject_id not in state.source_state_by_slot
        ):
            return rejected("human_takeover_pending_binding_invalid")
        if (
            state.target_state != "preflighted"
            or state.source_state_by_slot[previous_slot_id]
            != pending.resource_prior_runtime_state
            or state.source_state_by_slot[result.subject_id]
            != pending.prior_runtime_state
        ):
            return rejected("human_takeover_pending_state_changed")
        if result.succeeded and not result.evidence_refs:
            return rejected("operation_result_missing_control_evidence")
        if result.succeeded and result.status != HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1:
            return rejected("human_takeover_result_not_ready")
        source_states = dict(state.source_state_by_slot)
        source_states[previous_slot_id] = "preflighted"
        source_states[result.subject_id] = (
            "active" if result.succeeded else pending.prior_runtime_state
        )
        pending_operations = dict(state.pending_operation_by_subject)
        pending_operations.pop(subject_key)
        lifecycle = (
            "running"
            if result.succeeded
            else (
                "hold"
                if result.status == HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1
                else "fault"
            )
        )
        next_state = ControlStationState(
            runtime_session_id=state.runtime_session_id,
            manifest_hash=state.manifest_hash,
            control_domain_id=state.control_domain_id,
            lifecycle_state=lifecycle,
            state_revision=state.state_revision + 1,
            source_state_by_slot=source_states,
            target_state=state.target_state,
            pending_operation_by_subject=pending_operations,
            last_client_sequence_by_id=state.last_client_sequence_by_id,
            operator_control_lease=state.operator_control_lease,
            last_intent_id=state.last_intent_id,
        )
        return ControlStationOperationTransition(
            result=result,
            state_before=state,
            state_after=next_state,
            accepted=True,
            status=(
                "operation_succeeded"
                if result.succeeded
                else f"operation_failed:{result.status}"
            ),
        )

    source_operation = result.operation_kind in {
        "connect_source",
        "disconnect_source",
        "start_shadow",
        "stop_shadow",
        "request_candidate",
        "release_candidate",
        "request_motion_authority",
        "release_motion_authority",
    } or (
        result.operation_kind == "run_preflight"
        and result.subject_id in state.source_state_by_slot
    )
    subject_key = (
        f"source:{result.subject_id}"
        if source_operation
        else f"target:{result.subject_id}"
    )
    pending = state.pending_operation_by_subject.get(subject_key)
    if pending is None:
        return rejected("unknown_or_resolved_operation")
    if (
        pending.source_intent_id != result.source_intent_id
        or pending.operation_kind != result.operation_kind
        or pending.subject_id != result.subject_id
    ):
        return rejected("operation_result_binding_mismatch")

    source_states = dict(state.source_state_by_slot)
    target_state = state.target_state
    pending_operations = dict(state.pending_operation_by_subject)
    pending_operations.pop(subject_key)

    if result.succeeded:
        if result.operation_kind in {
            "request_candidate",
            "release_candidate",
            "request_motion_authority",
            "release_motion_authority",
        } and not result.evidence_refs:
            return rejected("operation_result_missing_control_evidence")
        success_state = {
            "connect_source": "connected",
            "disconnect_source": "disconnected",
            "start_shadow": "shadow",
            "stop_shadow": "preflighted",
            "connect_target": "connected",
            "disconnect_target": "configured",
            "run_preflight": "preflighted",
            "request_candidate": "candidate",
            "release_candidate": "preflighted",
            "request_motion_authority": "active",
            "release_motion_authority": "candidate",
        }[result.operation_kind]
    else:
        success_state = pending.prior_runtime_state

    if source_operation:
        source_states[result.subject_id] = success_state
    else:
        if result.subject_id != manifest.execution_binding.binding_id:
            return rejected("target_binding_mismatch")
        target_state = success_state

    lifecycle = state.lifecycle_state
    if result.operation_kind == "release_motion_authority":
        # Releasing the data-plane authority is a safety transition, not an
        # implicit return to an armed/runnable station.  Failure to establish
        # the lower HOLD is a latched fault; it must never derive RUNNING from
        # the source's prior active state.
        lifecycle = "hold" if result.succeeded else "fault"
    elif lifecycle not in {"hold", "fault", "stopping", "stopped"}:
        lifecycle = _derived_lifecycle(manifest, source_states, target_state)
    next_state = ControlStationState(
        runtime_session_id=state.runtime_session_id,
        manifest_hash=state.manifest_hash,
        control_domain_id=state.control_domain_id,
        lifecycle_state=lifecycle,
        state_revision=state.state_revision + 1,
        source_state_by_slot=source_states,
        target_state=target_state,
        pending_operation_by_subject=pending_operations,
        last_client_sequence_by_id=state.last_client_sequence_by_id,
        operator_control_lease=state.operator_control_lease,
        last_intent_id=state.last_intent_id,
    )
    status = "operation_succeeded" if result.succeeded else f"operation_failed:{result.status}"
    return ControlStationOperationTransition(
        result=result,
        state_before=state,
        state_after=next_state,
        accepted=True,
        status=status,
    )


__all__ = [
    "CONTROL_STATION_LIFECYCLE_STATES_V1",
    "SOURCE_SLOT_RUNTIME_STATES_V1",
    "TARGET_RUNTIME_STATES_V1",
    "HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1",
    "HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1",
    "HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_HELD_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_READY_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1",
    "COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1",
    "ControlStationState",
    "ControlStationOperationTransition",
    "ControlStationTransition",
    "PendingStationOperation",
    "admit_station_intent",
    "admit_station_operation_result",
    "apply_station_operation_result",
    "apply_station_intent",
    "initial_control_station_state",
]
