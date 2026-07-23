"""Receiver-owned producer epoch activation registry."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .control_admission import ControlAdmissionCoordinator, ControlCut, admitted_at_or_before
from .identity import RuntimeFactIdentity


class ProducerEpochError(ValueError):
    pass


PRODUCER_EPOCH_ACTIVATION_HASH_DOMAIN_V1 = "dexteleop.producer_epoch_activation.v1"


@dataclass(frozen=True)
class ProducerEpochActivation:
    activation_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    previous_epoch: int | None
    activation_sequence: int
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    activated_at: float
    control_admission_sequence: int
    reason: str
    semantic_content_hash: str


@dataclass(frozen=True)
class ProducerEpochSnapshot:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    included_admission_sequence_min: int | None
    included_admission_sequence_max: int | None
    active_epoch_by_producer: Mapping[str, int]
    activation_id_by_producer: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "active_epoch_by_producer",
            MappingProxyType(dict(self.active_epoch_by_producer)),
        )
        object.__setattr__(
            self,
            "activation_id_by_producer",
            MappingProxyType(dict(self.activation_id_by_producer)),
        )


class ProducerEpochRegistry:
    """Track the single active epoch of each producer in a control domain."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
    ) -> None:
        session = str(runtime_session_id).strip()
        if not session:
            raise ValueError("runtime_session_id must be non-empty.")
        if session != coordinator.clock_domain.runtime_session_id:
            raise ValueError("ProducerEpochRegistry session must match the control clock session.")
        self.runtime_session_id = session
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self._records: list[ProducerEpochActivation] = []
        self._active_epoch: dict[str, int] = {}
        self._activation_count: dict[str, int] = {}
        self._lock = threading.RLock()

    def activate(
        self,
        *,
        producer_id: str,
        producer_epoch: int,
        produced_at: float,
        produced_clock_domain_id: str,
        reason: str,
    ) -> ProducerEpochActivation:
        producer = str(producer_id).strip()
        normalized_reason = str(reason).strip()
        epoch = int(producer_epoch)
        produced = float(produced_at)
        produced_clock = str(produced_clock_domain_id).strip()
        if not producer:
            raise ProducerEpochError("producer_id must be non-empty.")
        if epoch < 0:
            raise ProducerEpochError("producer_epoch must be non-negative.")
        if not math.isfinite(produced):
            raise ProducerEpochError("produced_at must be finite.")
        if not produced_clock:
            raise ProducerEpochError("produced_clock_domain_id must be non-empty.")
        if not normalized_reason:
            raise ProducerEpochError("activation reason must be non-empty.")

        def build(available_at: float, admission_sequence: int) -> ProducerEpochActivation:
            with self._lock:
                previous = self._active_epoch.get(producer)
                if previous is not None and epoch <= previous:
                    raise ProducerEpochError(
                        f"Producer {producer!r} epoch must increase beyond {previous}, got {epoch}."
                    )
                activation_sequence = self._activation_count.get(producer, 0)
                identity = RuntimeFactIdentity(
                    runtime_session_id=self.runtime_session_id,
                    control_domain_id=self.control_domain_id,
                    producer_id=producer,
                    producer_epoch=epoch,
                    sequence=activation_sequence,
                    namespace="producer_epoch_activation",
                )
                payload = {
                    "activation_id": identity.fact_id,
                    "runtime_session_id": self.runtime_session_id,
                    "control_domain_id": self.control_domain_id,
                    "producer_id": producer,
                    "producer_epoch": epoch,
                    "previous_epoch": previous,
                    "activation_sequence": activation_sequence,
                    "produced_at": produced,
                    "produced_clock_domain_id": produced_clock,
                    "available_at": available_at,
                    "available_clock_domain_id": self.coordinator.clock_domain.clock_domain_id,
                    "activated_at": available_at,
                    "control_admission_sequence": admission_sequence,
                    "reason": normalized_reason,
                }
                record = ProducerEpochActivation(
                    activation_id=identity.fact_id,
                    runtime_session_id=self.runtime_session_id,
                    control_domain_id=self.control_domain_id,
                    producer_id=producer,
                    producer_epoch=epoch,
                    previous_epoch=previous,
                    activation_sequence=activation_sequence,
                    produced_at=produced,
                    produced_clock_domain_id=produced_clock,
                    available_at=available_at,
                    available_clock_domain_id=self.coordinator.clock_domain.clock_domain_id,
                    activated_at=available_at,
                    control_admission_sequence=admission_sequence,
                    reason=normalized_reason,
                    semantic_content_hash=canonical_content_hash(
                        payload,
                        domain=PRODUCER_EPOCH_ACTIVATION_HASH_DOMAIN_V1,
                    ),
                )
                return record

        return self.coordinator.commit(store=self, build=build)

    def _insert_admitted(self, record: ProducerEpochActivation) -> None:
        with self._lock:
            previous = self._active_epoch.get(record.producer_id)
            if previous != record.previous_epoch:
                raise ProducerEpochError("Producer epoch state changed before admission.")
            expected_sequence = self._activation_count.get(record.producer_id, 0)
            if record.activation_sequence != expected_sequence:
                raise ProducerEpochError("Producer activation sequence changed before admission.")
            self._records.append(record)
            self._active_epoch[record.producer_id] = record.producer_epoch
            self._activation_count[record.producer_id] = record.activation_sequence + 1

    def assert_active(self, *, producer_id: str, producer_epoch: int) -> None:
        with self._lock:
            active = self._active_epoch.get(str(producer_id))
            if active is None:
                raise ProducerEpochError(f"Producer {producer_id!r} has no active epoch.")
            if active != int(producer_epoch):
                raise ProducerEpochError(
                    f"Producer {producer_id!r} active epoch is {active}, got {producer_epoch}."
                )

    def current_epoch(self, producer_id: str) -> int | None:
        with self._lock:
            return self._active_epoch.get(str(producer_id))

    def records_at_or_before(self, cut: ControlCut) -> tuple[ProducerEpochActivation, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(record for record in self._records if admitted_at_or_before(record, cut))

    def snapshot(self, cut: ControlCut) -> ProducerEpochSnapshot:
        active: dict[str, int] = {}
        activation_ids: dict[str, str] = {}
        records = self.records_at_or_before(cut)
        for record in records:
            active[record.producer_id] = record.producer_epoch
            activation_ids[record.producer_id] = record.activation_id
        sequences = tuple(record.control_admission_sequence for record in records)
        return ProducerEpochSnapshot(
            control_domain_id=self.control_domain_id,
            cutoff_at=cut.cutoff_at,
            cutoff_admission_sequence=cut.cutoff_admission_sequence,
            included_admission_sequence_min=min(sequences) if sequences else None,
            included_admission_sequence_max=max(sequences) if sequences else None,
            active_epoch_by_producer=active,
            activation_id_by_producer=activation_ids,
        )


__all__ = [
    "PRODUCER_EPOCH_ACTIVATION_HASH_DOMAIN_V1",
    "ProducerEpochActivation",
    "ProducerEpochError",
    "ProducerEpochRegistry",
    "ProducerEpochSnapshot",
]
