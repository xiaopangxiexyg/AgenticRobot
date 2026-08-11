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


class StationPublicationBarrierObserverPort:
    """Deduplicate and deliver passive publication-barrier observations.

    The port owns no barrier state and cannot enable or disable publication.
    Callers keep responsibility for deciding when a transition is observed and
    whether observer failures are diagnostic-only or operation-fatal.
    """

    def __init__(self, *, suppress_callback_errors: bool) -> None:
        self._suppress_callback_errors = bool(suppress_callback_errors)
        self._lock = threading.RLock()
        self._observer: Callable[..., object] | None = None
        self._observed_keys: set[tuple[str, str]] = set()
        self._failure_count = 0

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def bind(self, observer: Callable[..., object]) -> None:
        if not callable(observer):
            raise TypeError("Publication barrier observer must be callable.")
        with self._lock:
            if self._observer is not None:
                raise RuntimeError("Publication barrier observer is already bound.")
            self._observer = observer

    def observe(
        self,
        snapshot: StationPublicationBarrierSnapshot,
        *,
        causal_station_fact_id: str,
        occurred_at: float,
        clock_domain_id: str,
    ) -> None:
        if not isinstance(snapshot, StationPublicationBarrierSnapshot):
            raise TypeError("snapshot must be a StationPublicationBarrierSnapshot.")
        key = (snapshot.semantic_content_hash, str(causal_station_fact_id))
        with self._lock:
            observer = self._observer
            if observer is None or key in self._observed_keys:
                return
            try:
                observer(
                    snapshot,
                    causal_station_fact_id=causal_station_fact_id,
                    occurred_at=occurred_at,
                    clock_domain_id=clock_domain_id,
                )
            except Exception:
                if not self._suppress_callback_errors:
                    raise
                self._failure_count += 1
                return
            self._observed_keys.add(key)


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

    def enable_from_composition_session_result(
        self,
        result: StationOperationResult,
    ) -> StationPublicationBarrierSnapshot:
        """Open only after the registered composition session is READY."""

        return self._enable_from_result(
            result,
            operation_kind="start_composition_session",
            required_status="composition_session_ready",
            transition_reason="typed_composition_session_completion",
        )

    def enable_from_composition_session_rearm_result(
        self,
        result: StationOperationResult,
    ) -> StationPublicationBarrierSnapshot:
        """Open only after a held session receives a typed rearm completion."""

        return self._enable_from_result(
            result,
            operation_kind="rearm_composition_session",
            required_status="composition_session_rearmed",
            transition_reason="typed_composition_session_rearm_completion",
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
    "StationPublicationBarrierObserverPort",
    "StationPublicationBarrierSnapshot",
]
