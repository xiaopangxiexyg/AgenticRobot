"""Source-agnostic Station publication linearization barrier.

The barrier does not carry an action and is not a dispatch permit.  It only
serializes source publication against authority release at the application
boundary; the lower runtime remains responsible for proposal admission,
Arbitration, execution epochs, and final dispatch authorization.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, TypeVar

from dataset_contracts.canonical_encoding import canonical_content_hash

from .contracts import StationOperationResult


STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.publication_barrier_snapshot.v1"
)
T = TypeVar("T")


@dataclass(frozen=True)
class StationPublicationBarrierSnapshot:
    source_slot_id: str
    generation: int
    enabled: bool
    authority_operation_result_id: str | None
    authority_evidence_refs: tuple[str, ...]
    transition_reason: str
    semantic_content_hash: str


class StationPublicationBarrier:
    """Linearize a persistent source authority lease against publication."""

    def __init__(self, *, source_slot_id: str) -> None:
        self.source_slot_id = str(source_slot_id).strip()
        if not self.source_slot_id:
            raise ValueError("source_slot_id must be non-empty.")
        self._lock = threading.RLock()
        self._generation = 0
        self._enabled = False
        self._authority_operation_result_id: str | None = None
        self._authority_evidence_refs: tuple[str, ...] = ()
        self._transition_reason = "initial_disabled"

    def _snapshot_locked(self) -> StationPublicationBarrierSnapshot:
        payload = {
            "source_slot_id": self.source_slot_id,
            "generation": self._generation,
            "enabled": self._enabled,
            "authority_operation_result_id": self._authority_operation_result_id,
            "authority_evidence_refs": self._authority_evidence_refs,
            "transition_reason": self._transition_reason,
        }
        return StationPublicationBarrierSnapshot(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1,
            ),
        )

    def snapshot(self) -> StationPublicationBarrierSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def enable_from_authority_result(
        self,
        result: StationOperationResult,
    ) -> StationPublicationBarrierSnapshot:
        return self._enable_from_result(
            result,
            operation_kind="request_motion_authority",
            required_status=None,
            transition_reason="typed_authority_completion",
        )

    def enable_from_human_takeover_result(
        self,
        result: StationOperationResult,
    ) -> StationPublicationBarrierSnapshot:
        """Open only the target human slot after the aggregate READY result."""

        return self._enable_from_result(
            result,
            operation_kind="request_human_takeover",
            required_status="human_takeover_ready",
            transition_reason="typed_human_takeover_completion",
        )

    def _enable_from_result(
        self,
        result: StationOperationResult,
        *,
        operation_kind: str,
        required_status: str | None,
        transition_reason: str,
    ) -> StationPublicationBarrierSnapshot:
        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        if (
            not result.succeeded
            or result.operation_kind != operation_kind
            or result.subject_id != self.source_slot_id
            or not result.evidence_refs
            or (required_status is not None and result.status != required_status)
        ):
            raise ValueError(
                "Publication requires a successful, evidenced authority result."
            )
        with self._lock:
            if self._enabled:
                if self._authority_operation_result_id == result.operation_result_id:
                    return self._snapshot_locked()
                raise RuntimeError("Publication authority is already enabled.")
            self._generation += 1
            self._enabled = True
            self._authority_operation_result_id = result.operation_result_id
            self._authority_evidence_refs = tuple(result.evidence_refs)
            self._transition_reason = transition_reason
            return self._snapshot_locked()

    def disable(self, *, reason: str) -> StationPublicationBarrierSnapshot:
        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("Publication disable reason must be non-empty.")
        with self._lock:
            if self._enabled:
                self._generation += 1
            self._enabled = False
            self._authority_operation_result_id = None
            self._authority_evidence_refs = ()
            self._transition_reason = normalized_reason
            return self._snapshot_locked()

    def execute_if_enabled(self, operation: Callable[[], T]) -> T:
        if not callable(operation):
            raise TypeError("operation must be callable.")
        with self._lock:
            if not self._enabled:
                raise RuntimeError("Station publication barrier is disabled.")
            # Hold the barrier through lower proposal admission.  A concurrent
            # release linearizes after this admission and then cancels/holds
            # the lower execution epoch; it cannot interleave halfway through.
            return operation()


__all__ = [
    "STATION_PUBLICATION_BARRIER_HASH_DOMAIN_V1",
    "StationPublicationBarrier",
    "StationPublicationBarrierSnapshot",
]
