"""Common Station binding for a backend-neutral authoritative target.

Concrete sim/real targets supply a thin runtime wrapper around the shared
``AuthoritativeControlRuntime``.  This module owns no backend or source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    AuthoritativeControlRuntime,
    ExecutionGateTransitionOutcome,
)
from robot_motion.runtime.execution_gate import ExecutionGateSnapshot

from ..contracts import StationIntent, StationOperationResultDraft


MUJOCO_STATION_EVIDENCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.cs2a2b_mujoco_station_operation_evidence.v1"
)


class StationTargetBindingReceipt(Protocol):
    station_authority_lifecycle_bound: bool
    owner_token_id: str
    semantic_content_hash: str


class StationTargetRuntimeConfig(Protocol):
    runtime_session_id: str
    control_domain_id: str


class StationAuthoritativeRuntime(Protocol):
    core: AuthoritativeControlRuntime
    config: StationTargetRuntimeConfig

    def transition_execution_gate(
        self,
        status: str,
        *,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        force_cancellation: bool = False,
    ) -> ExecutionGateTransitionOutcome: ...


class StationAuthoritativeDataPlane(Protocol):
    """Reusable lifecycle surface shared by sim and real target profiles."""

    runtime: StationAuthoritativeRuntime
    binding_receipt: StationTargetBindingReceipt
    started: bool

    def start(self) -> None: ...

    def close(self) -> None: ...


def authoritative_owner_registry_id(runtime: StationAuthoritativeRuntime) -> str:
    return (
        "dispatch-owner-registry:"
        f"{runtime.config.runtime_session_id}:{runtime.config.control_domain_id}"
    )


def authoritative_authority_epoch_source_id(
    runtime: StationAuthoritativeRuntime,
) -> str:
    """Canonical authority epoch source; dispatch ownership is a mirror."""

    return (
        "authority-store:"
        f"{runtime.config.runtime_session_id}:{runtime.config.control_domain_id}"
    )


def authoritative_execution_epoch_source_id(
    runtime: StationAuthoritativeRuntime,
) -> str:
    return (
        "execution-epoch-registry:"
        f"{runtime.config.runtime_session_id}:{runtime.config.control_domain_id}"
    )


@dataclass(frozen=True)
class StationTargetOperationEvidence:
    source_intent_id: str
    operation_kind: str
    subject_id: str
    binding_receipt_hash: str
    owner_token_id: str
    authority_epoch: int
    execution_epoch: int
    execution_gate_status: str
    execution_gate_snapshot_hash: str
    publication_enabled: bool
    lower_event_ids: tuple[str, ...]
    evidence_status: str
    semantic_content_hash: str


def capture_authoritative_execution_gate_snapshot(
    composition: StationAuthoritativeDataPlane,
) -> ExecutionGateSnapshot:
    """Read the receiver-owned gate through its public authoritative seam."""

    return composition.runtime.core.capture_execution_gate_snapshot()


class StationAuthoritativeTargetAdapterBase:
    """Common Station evidence/liveness behavior for one authoritative target."""

    def __init__(
        self,
        *,
        adapter_id: str,
        subject_id: str,
        supported_operations: tuple[str, ...],
        composition: StationAuthoritativeDataPlane,
        clock: Callable[[], float],
        clock_domain_id: str,
        expected_control_mode: str = "human",
    ) -> None:
        self._adapter_id = str(adapter_id).strip()
        self._subject_id = str(subject_id).strip()
        self._supported_operations = tuple(
            sorted(str(value).strip() for value in supported_operations)
        )
        self.composition = composition
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        self.expected_control_mode = str(expected_control_mode).strip()
        if (
            not self._adapter_id
            or not self._subject_id
            or not self.clock_domain_id
            or not self.expected_control_mode
            or not self._supported_operations
            or any(not value for value in self._supported_operations)
        ):
            raise ValueError("Station adapter identity/operations must be non-empty.")
        if len(set(self._supported_operations)) != len(self._supported_operations):
            raise ValueError("Station adapter operations cannot contain duplicates.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._result_by_intent_id: dict[str, StationOperationResultDraft] = {}
        self._evidence_by_intent_id: dict[str, StationTargetOperationEvidence] = {}

    @property
    def adapter_id(self) -> str:
        return self._adapter_id

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def supported_operations(self) -> tuple[str, ...]:
        return self._supported_operations

    @property
    def evidence_by_intent_id(self) -> dict[str, StationTargetOperationEvidence]:
        return dict(self._evidence_by_intent_id)

    def _validate_intent(self, intent: StationIntent) -> None:
        if not isinstance(intent, StationIntent):
            raise TypeError("intent must be a StationIntent.")
        if intent.subject_id != self.subject_id:
            raise ValueError("Station operation belongs to another adapter subject.")
        if intent.intent_kind not in self.supported_operations:
            raise ValueError("Station operation is not supported by this adapter.")

    def _validate_live_binding(self, *, require_started: bool) -> None:
        if not self.composition.binding_receipt.station_authority_lifecycle_bound:
            raise RuntimeError("Composition lacks a Station lifecycle binding.")
        if require_started and not self.composition.started:
            raise RuntimeError("Authoritative target has not been connected and started.")
        receiver = self.composition.runtime.core.capture_receiver_state()
        if (
            receiver.runtime_session_id
            != self.composition.runtime.config.runtime_session_id
            or receiver.control_domain_id
            != self.composition.runtime.config.control_domain_id
            or receiver.current_owner_token_id
            != self.composition.binding_receipt.owner_token_id
        ):
            raise RuntimeError("Live target owner binding no longer matches Station.")
        if (
            self.composition.runtime.core.config.initial_control_mode
            != self.expected_control_mode
        ):
            raise RuntimeError("Scoped Station binding control mode no longer matches.")

    def _build_result(
        self,
        intent: StationIntent,
        *,
        publication_enabled: bool,
        lower_event_ids: tuple[str, ...] = (),
        evidence_status: str,
    ) -> StationOperationResultDraft:
        receiver = self.composition.runtime.core.capture_receiver_state()
        gate = capture_authoritative_execution_gate_snapshot(self.composition)
        payload = {
            "source_intent_id": intent.intent_id,
            "operation_kind": intent.intent_kind,
            "subject_id": self.subject_id,
            "binding_receipt_hash": (
                self.composition.binding_receipt.semantic_content_hash
            ),
            "owner_token_id": str(receiver.current_owner_token_id),
            "authority_epoch": receiver.current_authority_epoch,
            "execution_epoch": receiver.current_execution_epoch,
            "execution_gate_status": gate.status,
            "execution_gate_snapshot_hash": gate.semantic_content_hash,
            "publication_enabled": bool(publication_enabled),
            "lower_event_ids": tuple(lower_event_ids),
            "evidence_status": str(evidence_status).strip(),
        }
        evidence = StationTargetOperationEvidence(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=MUJOCO_STATION_EVIDENCE_HASH_DOMAIN_V1,
            ),
        )
        self._evidence_by_intent_id[intent.intent_id] = evidence
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=self.subject_id,
            succeeded=True,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=evidence.evidence_status,
            evidence_refs=(
                evidence.semantic_content_hash,
                evidence.binding_receipt_hash,
                *evidence.lower_event_ids,
            ),
        )
        self._result_by_intent_id[intent.intent_id] = result
        return result

    def _build_failure(
        self,
        intent: StationIntent,
        *,
        status: str,
    ) -> StationOperationResultDraft:
        result = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=self.subject_id,
            succeeded=False,
            completed_at=float(self.clock()),
            completed_clock_domain_id=self.clock_domain_id,
            status=str(status).strip(),
            evidence_refs=(self.composition.binding_receipt.semantic_content_hash,),
        )
        self._result_by_intent_id[intent.intent_id] = result
        return result


__all__ = [
    "MUJOCO_STATION_EVIDENCE_HASH_DOMAIN_V1",
    "StationAuthoritativeDataPlane",
    "StationAuthoritativeRuntime",
    "StationAuthoritativeTargetAdapterBase",
    "StationTargetBindingReceipt",
    "StationTargetOperationEvidence",
    "StationTargetRuntimeConfig",
    "authoritative_authority_epoch_source_id",
    "authoritative_execution_epoch_source_id",
    "authoritative_owner_registry_id",
    "capture_authoritative_execution_gate_snapshot",
]
