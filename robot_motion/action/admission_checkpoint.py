"""Complete source-admission state at a caller-owned quiescent control cut.

This trusted internal component does not grant authority or restore a backend.
The enclosing runtime must capture its other stores at the same cut and bind
source/profile provenance. Diagnostic projections and replayed admissions are
not substitutes for the retained histories and independent high-water marks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor, RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze
from robot_motion.runtime.producer_epoch import (
    PRODUCER_EPOCH_ACTIVATION_HASH_DOMAIN_V1,
    ProducerEpochActivation,
    ProducerEpochRegistry,
)

from .component_action import ComponentAction, ComponentActionValue
from .proposal import (
    ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3,
    ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4,
    ActionProposal,
    ProposalBuffer,
    action_proposal_semantic_payload,
)

_COORDINATOR_LAYOUT = {
    "control_domain_id",
    "clock_domain",
    "_clock",
    "_lock",
    "_committed_high_watermark",
}
_PRODUCER_LAYOUT = {
    "runtime_session_id",
    "coordinator",
    "control_domain_id",
    "_records",
    "_active_epoch",
    "_activation_count",
    "_lock",
}
_PROPOSAL_LAYOUT = {
    "runtime_session_id",
    "coordinator",
    "control_domain_id",
    "producer_epochs",
    "capacity",
    "_records",
    "_by_id",
    "_last_sequence",
    "_lock",
}


@dataclass(frozen=True)
class SourceAdmissionCheckpoint:
    """Detached typed facts and exact source watermarks, not a file format."""

    control_domain_id: str
    clock_domain: ClockDomainDescriptor
    captured_at: float
    committed_high_watermark: int
    proposal_capacity: int
    producer_records: tuple[ProducerEpochActivation, ...]
    active_epoch_by_producer: Mapping[str, int]
    activation_count_by_producer: Mapping[str, int]
    proposal_records: tuple[ActionProposal, ...]
    last_proposal_sequences: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class RestoredSourceAdmission:
    coordinator: ControlAdmissionCoordinator
    producer_epochs: ProducerEpochRegistry
    proposals: ProposalBuffer


def _integer(value, *, minimum=0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid source-admission counter.")
    return value


def _scope(record, state):
    if (
        record.runtime_session_id != state.clock_domain.runtime_session_id
        or record.control_domain_id != state.control_domain_id
        or record.available_clock_domain_id != state.clock_domain.clock_domain_id
        or type(record.available_at) is not float
        or not math.isfinite(record.available_at)
        or record.available_at > state.captured_at
        or not 1
        <= _integer(record.control_admission_sequence, minimum=1)
        <= state.committed_high_watermark
    ):
        raise ValueError("Source fact does not belong to the saved admission cut.")


def _copy_proposal(record: ActionProposal) -> ActionProposal:
    if (
        type(record) is not ActionProposal
        or type(record.action) is not ComponentAction
        or record.semantic_hash_domain
        not in {
            ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V3,
            ACTION_PROPOSAL_SEMANTIC_HASH_DOMAIN_V4,
        }
        or any(
            type(value) is not ComponentActionValue
            for value in record.action.values_by_component.values()
        )
    ):
        raise ValueError("Unsupported proposal fact in source checkpoint.")
    # The action schema is an immutable contract. Existing value constructors
    # detach/freeze actual payloads; no generic object/pickle copier is used.
    action = replace(
        record.action,
        values_by_component={
            name: replace(value)
            for name, value in record.action.values_by_component.items()
        },
    )
    copied = replace(
        record, action=action, audit_metadata=deep_freeze(record.audit_metadata)
    )
    if (
        action.semantic_content_hash != record.action.semantic_content_hash
        or record.action_schema_hash != action.action_schema.semantic_content_hash
        or record.component_coverage != action.component_coverage
        or canonical_content_hash(
            action_proposal_semantic_payload(copied), domain=record.semantic_hash_domain
        )
        != record.semantic_content_hash
    ):
        raise ValueError("Proposal fact hash or action identity differs.")
    identity = RuntimeFactIdentity(
        runtime_session_id=record.runtime_session_id,
        control_domain_id=record.control_domain_id,
        producer_id=record.producer_id,
        producer_epoch=record.producer_epoch,
        sequence=record.sequence,
        namespace="action_proposal",
    )
    if record.proposal_id != identity.fact_id:
        raise ValueError("Proposal identity differs.")
    return copied


def _validated(state: SourceAdmissionCheckpoint) -> SourceAdmissionCheckpoint:
    if (
        type(state) is not SourceAdmissionCheckpoint
        or type(state.clock_domain) is not ClockDomainDescriptor
        or state.clock_domain.monotonic is not True
        or type(state.control_domain_id) is not str
        or not state.control_domain_id.strip()
        or type(state.captured_at) is not float
        or not math.isfinite(state.captured_at)
        or type(state.producer_records) is not tuple
        or type(state.proposal_records) is not tuple
        or not isinstance(state.active_epoch_by_producer, Mapping)
        or not isinstance(state.activation_count_by_producer, Mapping)
        or type(state.last_proposal_sequences) is not tuple
    ):
        raise ValueError("Invalid source-admission checkpoint.")
    _integer(state.committed_high_watermark)
    _integer(state.proposal_capacity, minimum=1)
    if len(state.proposal_records) > state.proposal_capacity:
        raise ValueError("Saved proposals exceed the configured capacity.")
    active, counts, known_epochs = {}, {}, set()
    epoch_start, epoch_end = {}, {}
    producer_records = []
    seen_admission = set()
    last_admission = 0
    for record in state.producer_records:
        if type(record) is not ProducerEpochActivation:
            raise ValueError("Unsupported producer activation fact.")
        _scope(record, state)
        _integer(record.producer_epoch)
        _integer(record.activation_sequence)
        if (
            record.control_admission_sequence <= last_admission
            or record.previous_epoch != active.get(record.producer_id)
            or record.activation_sequence != counts.get(record.producer_id, 0)
            or (
                record.previous_epoch is not None
                and record.producer_epoch <= record.previous_epoch
            )
        ):
            raise ValueError("Invalid producer activation history.")
        identity = RuntimeFactIdentity(
            runtime_session_id=record.runtime_session_id,
            control_domain_id=record.control_domain_id,
            producer_id=record.producer_id,
            producer_epoch=record.producer_epoch,
            sequence=record.activation_sequence,
            namespace="producer_epoch_activation",
        )
        payload = {
            field.name: getattr(record, field.name)
            for field in fields(record)
            if field.name != "semantic_content_hash"
        }
        if (
            record.activation_id != identity.fact_id
            or canonical_content_hash(
                payload, domain=PRODUCER_EPOCH_ACTIVATION_HASH_DOMAIN_V1
            )
            != record.semantic_content_hash
        ):
            raise ValueError("Producer activation identity/hash differs.")
        if record.previous_epoch is not None:
            epoch_end[(record.producer_id, record.previous_epoch)] = (
                record.control_admission_sequence
            )
        epoch_start[(record.producer_id, record.producer_epoch)] = (
            record.control_admission_sequence
        )
        active[record.producer_id] = record.producer_epoch
        counts[record.producer_id] = record.activation_sequence + 1
        known_epochs.add((record.producer_id, record.producer_epoch))
        seen_admission.add(record.control_admission_sequence)
        last_admission = record.control_admission_sequence
        producer_records.append(replace(record))
    if (
        dict(state.active_epoch_by_producer) != active
        or dict(state.activation_count_by_producer) != counts
    ):
        raise ValueError("Producer live state differs from its complete history.")
    sequences = {}
    for item in state.last_proposal_sequences:
        if type(item) is not tuple or len(item) != 3:
            raise ValueError("Invalid proposal sequence high-water mark.")
        producer, epoch, sequence = item
        _integer(epoch)
        _integer(sequence)
        key = (producer, epoch)
        if type(producer) is not str or key not in known_epochs or key in sequences:
            raise ValueError(
                "Proposal sequence references an unknown or duplicate producer epoch."
            )
        sequences[key] = sequence
    proposals, retained_ids = [], set()
    last_admission = 0
    last_retained_sequence = {}
    for record in state.proposal_records:
        record = _copy_proposal(record)
        _scope(record, state)
        _integer(record.producer_epoch)
        _integer(record.sequence)
        key = (record.producer_id, record.producer_epoch)
        if key in known_epochs and (
            record.control_admission_sequence <= epoch_start[key]
            or (
                key in epoch_end and record.control_admission_sequence >= epoch_end[key]
            )
        ):
            raise ValueError("Proposal epoch was not active at its admission sequence.")
        if (
            key not in known_epochs
            or key not in sequences
            or record.sequence > sequences[key]
            or record.sequence <= last_retained_sequence.get(key, -1)
            or record.proposal_id in retained_ids
            or record.control_admission_sequence in seen_admission
            or record.control_admission_sequence <= last_admission
        ):
            raise ValueError("Retained proposal history and watermarks disagree.")
        retained_ids.add(record.proposal_id)
        seen_admission.add(record.control_admission_sequence)
        last_admission = record.control_admission_sequence
        last_retained_sequence[key] = record.sequence
        proposals.append(record)
    return replace(
        state,
        producer_records=tuple(producer_records),
        proposal_records=tuple(proposals),
        active_epoch_by_producer=MappingProxyType(active),
        activation_count_by_producer=MappingProxyType(counts),
        last_proposal_sequences=tuple(
            (producer, epoch, sequence)
            for (producer, epoch), sequence in sorted(sequences.items())
        ),
    )


def capture_source_admission(
    coordinator: ControlAdmissionCoordinator,
    producer_epochs: ProducerEpochRegistry,
    proposals: ProposalBuffer,
) -> SourceAdmissionCheckpoint:
    """Detach all source state; parent first excludes whole runtime transactions."""
    if (
        type(coordinator) is not ControlAdmissionCoordinator
        or type(producer_epochs) is not ProducerEpochRegistry
        or type(proposals) is not ProposalBuffer
        or set(vars(coordinator)) != _COORDINATOR_LAYOUT
        or set(vars(producer_epochs)) != _PRODUCER_LAYOUT
        or set(vars(proposals)) != _PROPOSAL_LAYOUT
        or producer_epochs.coordinator is not coordinator
        or proposals.coordinator is not coordinator
        or proposals.producer_epochs is not producer_epochs
        or producer_epochs.runtime_session_id
        != coordinator.clock_domain.runtime_session_id
        or proposals.runtime_session_id != producer_epochs.runtime_session_id
        or producer_epochs.control_domain_id != coordinator.control_domain_id
        or proposals.control_domain_id != coordinator.control_domain_id
    ):
        raise ValueError(
            "Source owners do not share the exact supported admission scope/layout."
        )
    with coordinator._lock, proposals._lock, producer_epochs._lock:
        if set(proposals._by_id) != {
            record.proposal_id for record in proposals._records
        } or any(
            proposals._by_id.get(record.proposal_id) is not record
            for record in proposals._records
        ):
            raise ValueError("Proposal retained index differs from its records.")
        state = SourceAdmissionCheckpoint(
            control_domain_id=coordinator.control_domain_id,
            clock_domain=coordinator.clock_domain,
            captured_at=float(coordinator._clock()),
            committed_high_watermark=coordinator._committed_high_watermark,
            proposal_capacity=proposals.capacity,
            producer_records=tuple(producer_epochs._records),
            active_epoch_by_producer=dict(producer_epochs._active_epoch),
            activation_count_by_producer=dict(producer_epochs._activation_count),
            proposal_records=tuple(proposals._records),
            last_proposal_sequences=tuple(
                (producer, epoch, sequence)
                for (producer, epoch), sequence in proposals._last_sequence.items()
            ),
        )
        return _validated(state)


def restore_source_admission(
    state: SourceAdmissionCheckpoint,
    *,
    control_domain_id: str,
    clock_domain: ClockDomainDescriptor,
    proposal_capacity: int,
    clock: Callable[[], float],
) -> RestoredSourceAdmission:
    """Build unpublished source owners; never commit, activate or re-admit facts.

    The saved global watermark can exceed all source records: authority/gate
    and other stores share the coordinator. Their state belongs to the parent
    bundle and must be restored before publishing a complete control runtime.
    """
    state = _validated(state)
    if (
        state.control_domain_id != control_domain_id
        or state.clock_domain != clock_domain
        or type(proposal_capacity) is not int
        or state.proposal_capacity != proposal_capacity
        or not callable(clock)
        or clock() != state.captured_at
    ):
        raise ValueError("Destination admission scope, clock or capacity differs.")
    coordinator = ControlAdmissionCoordinator(
        control_domain_id, clock_domain=clock_domain, clock=clock
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id=clock_domain.runtime_session_id, coordinator=coordinator
    )
    proposals = ProposalBuffer(
        runtime_session_id=clock_domain.runtime_session_id,
        coordinator=coordinator,
        producer_epochs=epochs,
        capacity=proposal_capacity,
    )
    epochs._records = list(state.producer_records)
    epochs._active_epoch = dict(state.active_epoch_by_producer)
    epochs._activation_count = dict(state.activation_count_by_producer)
    proposals._records = list(state.proposal_records)
    proposals._by_id = {record.proposal_id: record for record in state.proposal_records}
    proposals._last_sequence = {
        (producer, epoch): sequence
        for producer, epoch, sequence in state.last_proposal_sequences
    }
    coordinator._committed_high_watermark = state.committed_high_watermark
    return RestoredSourceAdmission(coordinator, epochs, proposals)
