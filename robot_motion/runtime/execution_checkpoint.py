"""Trusted logical dispatch/epoch/ack state for an isolated simulation fork.

This module neither migrates a live backend owner nor provides a distributed
lock. A simulation profile must prove a fresh independent backend and source
provenance before consuming it. Real owner changes continue to use the existing
HOLD/cancellation-acknowledgement transition. No token or permit is re-granted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.capability import CANCELLATION_ACK_LEVELS

from .cancellation import (
    EXECUTION_CANCELLATION_ACK_HASH_DOMAIN_V1,
    EXECUTION_CANCELLATION_HASH_DOMAIN_V1,
    CancellationAcknowledgement,
    CancellationAcknowledgementTracker,
    ExecutionCancellation,
    ExecutionEpochRegistry,
)
from .dispatch_owner import (
    DISPATCH_OWNER_KINDS,
    DISPATCH_OWNER_TOKEN_HASH_DOMAIN_V1,
    DispatchOwnerRegistry,
    DispatchOwnerToken,
)
from .identity import RuntimeFactIdentity

_DISPATCH_FIELDS = {
    "_current_token",
    "_authority_epoch",
    "_execution_epoch",
    "_owner_generation",
    "_next_fact_sequence",
    "_ever_granted",
    "_issued_permit_hash_by_id",
    "_consumed_permit_ids",
}
_DISPATCH_BINDINGS = {
    "runtime_session_id",
    "control_domain_id",
    "_clock",
    "_lock",
    "_pending_transition",
}
_EPOCH_FIELDS = {
    "_execution_epoch",
    "_segment_generation_by_group",
    "_fault_latched",
    "_last_cancellation_id",
    "_next_sequence",
}
_EPOCH_BINDINGS = {
    "runtime_session_id",
    "control_domain_id",
    "clock_domain_id",
    "_clock",
    "_lock",
}
_ACK_LAYOUT = {"_clock", "_events", "_acks", "_lock"}
_PERMIT_NAMESPACES = (
    "motion_dispatch_permit",
    "safety_dispatch_permit",
    "lifecycle_dispatch_permit",
)


@dataclass(frozen=True)
class ExecutionAuthorityCheckpoint:
    """Complete logical registry history; enclosing store owns persistence."""

    scope: tuple[str, str, str]
    captured_at: float
    dispatch: Mapping
    epochs: Mapping
    cancellations: tuple[ExecutionCancellation, ...]
    acknowledgements: tuple[CancellationAcknowledgement, ...]


@dataclass(frozen=True)
class RestoredExecutionAuthority:
    dispatch_owners: DispatchOwnerRegistry
    execution_epochs: ExecutionEpochRegistry
    cancellation_acknowledgements: CancellationAcknowledgementTracker


def _counter(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("Invalid execution registry counter.")
    return value


def _record(record, record_type, domain, *, excluded=()):
    if type(record) is not record_type:
        raise ValueError("Unsupported execution registry record.")
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if field.name not in ("semantic_content_hash", *excluded)
    }
    if canonical_content_hash(payload, domain=domain) != record.semantic_content_hash:
        raise ValueError("Execution registry record hash differs.")
    return replace(record)


def _fact_sequence(value, scope, producer_id, namespace):
    if type(value) is not str:
        raise ValueError("Invalid execution fact identity.")
    try:
        sequence = int(value.rsplit(":", 1)[-1])
    except ValueError as error:
        raise ValueError("Invalid execution fact sequence.") from error
    if (
        value
        != RuntimeFactIdentity(
            runtime_session_id=scope[0],
            control_domain_id=scope[1],
            producer_id=producer_id,
            producer_epoch=0,
            sequence=sequence,
            namespace=namespace,
        ).fact_id
    ):
        raise ValueError("Execution fact identity belongs to another scope.")
    return sequence


def _validated(state):
    if (
        type(state) is not ExecutionAuthorityCheckpoint
        or type(state.scope) is not tuple
        or len(state.scope) != 3
        or any(type(item) is not str or not item.strip() for item in state.scope)
        or type(state.captured_at) is not float
        or not math.isfinite(state.captured_at)
        or not isinstance(state.dispatch, Mapping)
        or set(state.dispatch) != _DISPATCH_FIELDS
        or not isinstance(state.epochs, Mapping)
        or set(state.epochs) != _EPOCH_FIELDS
        or type(state.cancellations) is not tuple
        or type(state.acknowledgements) is not tuple
    ):
        raise ValueError("Invalid execution authority checkpoint.")
    dispatch, epochs = dict(state.dispatch), dict(state.epochs)
    for name in (
        "_authority_epoch",
        "_execution_epoch",
        "_owner_generation",
        "_next_fact_sequence",
    ):
        _counter(
            dispatch[name],
            minimum=1 if name in ("_owner_generation", "_next_fact_sequence") else 0,
        )
    _counter(epochs["_execution_epoch"])
    _counter(epochs["_next_sequence"])
    if (
        dispatch["_ever_granted"] is not True
        or type(epochs["_fault_latched"]) is not bool
    ):
        raise ValueError("Invalid execution lifecycle/fault state.")
    if dispatch["_execution_epoch"] != epochs["_execution_epoch"]:
        raise ValueError("Dispatch and execution registries disagree on live epoch.")
    token = _record(
        dispatch["_current_token"],
        DispatchOwnerToken,
        DISPATCH_OWNER_TOKEN_HASH_DOMAIN_V1,
    )
    _counter(token.owner_generation, minimum=1)
    if (
        (token.runtime_session_id, token.control_domain_id) != state.scope[:2]
        or type(token.owner_runtime_id) is not str
        or not token.owner_runtime_id.strip()
        or token.owner_kind not in DISPATCH_OWNER_KINDS
        or token.process_scope != "in_process_only"
        or token.revoked_at is not None
        or token.owner_generation != dispatch["_owner_generation"]
        or not isinstance(token.acquired_at, float)
        or not math.isfinite(token.acquired_at)
        or token.acquired_at > state.captured_at
    ):
        raise ValueError("Checkpoint does not contain an active local owner token.")
    for acquired, live in (
        (token.authority_epoch_at_acquire, dispatch["_authority_epoch"]),
        (token.execution_epoch_at_acquire, dispatch["_execution_epoch"]),
    ):
        if _counter(acquired) > live:
            raise ValueError("Owner acquisition epochs exceed live epochs.")
    sequences = {
        _fact_sequence(
            token.token_id,
            state.scope,
            "dispatch-owner-registry",
            "dispatch_owner_token",
        )
    }
    if (
        not isinstance(dispatch["_issued_permit_hash_by_id"], Mapping)
        or type(dispatch["_consumed_permit_ids"]) is not frozenset
    ):
        raise ValueError("Invalid dispatch permit history.")
    issued = dict(dispatch["_issued_permit_hash_by_id"])
    consumed = dispatch["_consumed_permit_ids"]
    if set(issued) != consumed:
        raise ValueError("Checkpoint requires all issued permits to be consumed.")
    for permit_id, digest in issued.items():
        if type(permit_id) is not str:
            raise ValueError("Invalid issued permit identity.")
        namespace = next(
            (
                name
                for name in _PERMIT_NAMESPACES
                if permit_id.startswith("dexfact:v1:" + name + ":")
            ),
            None,
        )
        if (
            namespace is None
            or type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Invalid issued permit identity/hash.")
        sequence = _fact_sequence(
            permit_id, state.scope, "dispatch-owner-registry", namespace
        )
        if sequence in sequences:
            raise ValueError("Duplicate execution registry fact sequence.")
        sequences.add(sequence)
    if max(sequences) >= dispatch["_next_fact_sequence"]:
        raise ValueError("Dispatch fact history exceeds its next sequence.")
    generations = epochs["_segment_generation_by_group"]
    if not isinstance(generations, Mapping) or any(
        type(key) is not str or not key.strip() for key in generations
    ):
        raise ValueError("Invalid execution group generations.")
    for generation in generations.values():
        _counter(generation)
    cancellations, by_id = [], {}
    for record in state.cancellations:
        record = _record(
            record,
            ExecutionCancellation,
            EXECUTION_CANCELLATION_HASH_DOMAIN_V1,
            excluded=("audit_metadata",),
        )
        if (
            (
                record.runtime_session_id,
                record.control_domain_id,
                record.effective_clock_domain_id,
            )
            != state.scope
            or record.cancellation_id in by_id
            or _fact_sequence(
                record.cancellation_id,
                state.scope,
                "execution-epoch-registry",
                "execution_cancellation",
            )
            != record.sequence
            or not 0 <= _counter(record.sequence) < epochs["_next_sequence"]
            or record.execution_epoch > epochs["_execution_epoch"]
            or record.effective_at > state.captured_at
        ):
            raise ValueError("Cancellation does not belong to the saved execution cut.")
        by_id[record.cancellation_id] = record
        cancellations.append(record)
    last = epochs["_last_cancellation_id"]
    if last is not None and (type(last) is not str or last not in by_id):
        raise ValueError("Missing last cancellation evidence.")
    if epochs["_fault_latched"] and last is None:
        raise ValueError("Fault latch has no cancellation evidence.")
    acknowledgements, seen = [], set()
    for ack in state.acknowledgements:
        ack = _record(
            ack, CancellationAcknowledgement, EXECUTION_CANCELLATION_ACK_HASH_DOMAIN_V1
        )
        event = by_id.get(ack.cancellation_id)
        key = (ack.cancellation_id, ack.acknowledgement_level)
        if (
            event is None
            or key in seen
            or ack.acknowledgement_level not in CANCELLATION_ACK_LEVELS
            or type(ack.accepted) is not bool
            or not math.isfinite(ack.acknowledged_at)
            or ack.acknowledged_at < event.effective_at
        ):
            raise ValueError("Invalid or disconnected cancellation acknowledgement.")
        seen.add(key)
        acknowledgements.append(ack)
    dispatch["_current_token"] = token
    dispatch["_issued_permit_hash_by_id"] = MappingProxyType(issued)
    epochs["_segment_generation_by_group"] = MappingProxyType(dict(generations))
    return replace(
        state,
        dispatch=MappingProxyType(dispatch),
        epochs=MappingProxyType(epochs),
        cancellations=tuple(cancellations),
        acknowledgements=tuple(acknowledgements),
    )


def capture_execution_authority(
    dispatch: DispatchOwnerRegistry,
    epochs: ExecutionEpochRegistry,
    acknowledgements: CancellationAcknowledgementTracker,
) -> ExecutionAuthorityCheckpoint:
    """Capture under external whole-operation exclusion; do not HOLD or mutate."""
    if (
        type(dispatch) is not DispatchOwnerRegistry
        or set(vars(dispatch)) != _DISPATCH_FIELDS | _DISPATCH_BINDINGS
        or type(epochs) is not ExecutionEpochRegistry
        or set(vars(epochs)) != _EPOCH_FIELDS | _EPOCH_BINDINGS
        or type(acknowledgements) is not CancellationAcknowledgementTracker
        or set(vars(acknowledgements)) != _ACK_LAYOUT
        or (dispatch.runtime_session_id, dispatch.control_domain_id)
        != (epochs.runtime_session_id, epochs.control_domain_id)
        or dispatch._clock is not epochs._clock
        or dispatch._clock is not acknowledgements._clock
    ):
        raise ValueError("Execution registry layout or shared scope/clock differs.")
    with dispatch._lock, epochs._lock, acknowledgements._lock:
        if dispatch._pending_transition is not None:
            raise ValueError("Checkpoint cannot capture an owner transition.")
        if (
            set(acknowledgements._events) != set(acknowledgements._acks)
            or any(
                key != record.cancellation_id
                for key, record in acknowledgements._events.items()
            )
            or any(
                key != record.acknowledgement_level or event != record.cancellation_id
                for event, acks in acknowledgements._acks.items()
                for key, record in acks.items()
            )
        ):
            raise ValueError("Cancellation history indexes differ.")
        owner_state = {name: getattr(dispatch, name) for name in _DISPATCH_FIELDS}
        owner_state["_issued_permit_hash_by_id"] = dict(
            dispatch._issued_permit_hash_by_id
        )
        owner_state["_consumed_permit_ids"] = frozenset(dispatch._consumed_permit_ids)
        epoch_state = {name: getattr(epochs, name) for name in _EPOCH_FIELDS}
        epoch_state["_segment_generation_by_group"] = dict(
            epochs._segment_generation_by_group
        )
        return _validated(
            ExecutionAuthorityCheckpoint(
                scope=(
                    dispatch.runtime_session_id,
                    dispatch.control_domain_id,
                    epochs.clock_domain_id,
                ),
                captured_at=float(dispatch._clock()),
                dispatch=owner_state,
                epochs=epoch_state,
                cancellations=tuple(acknowledgements._events.values()),
                acknowledgements=tuple(
                    ack
                    for values in acknowledgements._acks.values()
                    for ack in values.values()
                ),
            )
        )


def restore_execution_authority(
    state: ExecutionAuthorityCheckpoint,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    clock_domain_id: str,
    owner_runtime_id: str,
    owner_kind: str,
    clock: Callable[[], float],
) -> RestoredExecutionAuthority:
    """Build fresh logical owners; caller binds only an independent simulator.

    The token's acquisition epochs may precede the live epochs after normal
    cancellation/recovery. Restore their exact history, never re-grant a token
    or reset consumed permits to force a match. This is not real-backend auth.
    """
    state = _validated(state)
    token = state.dispatch["_current_token"]
    if (
        state.scope != (runtime_session_id, control_domain_id, clock_domain_id)
        or token.owner_runtime_id != owner_runtime_id
        or token.owner_kind != owner_kind
        or not callable(clock)
        or clock() != state.captured_at
    ):
        raise ValueError("Destination execution scope, owner or logical clock differs.")
    dispatch = DispatchOwnerRegistry(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        initial_authority_epoch=state.dispatch["_authority_epoch"],
        initial_execution_epoch=state.dispatch["_execution_epoch"],
        clock=clock,
    )
    epochs = ExecutionEpochRegistry(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        clock_domain_id=clock_domain_id,
        clock=clock,
    )
    acknowledgements = CancellationAcknowledgementTracker(clock=clock)
    for name, value in state.dispatch.items():
        if name == "_issued_permit_hash_by_id":
            value = dict(value)
        elif name == "_consumed_permit_ids":
            value = set(value)
        setattr(dispatch, name, value)
    for name, value in state.epochs.items():
        if name == "_segment_generation_by_group":
            value = dict(value)
        setattr(epochs, name, value)
    acknowledgements._events = {
        event.cancellation_id: event for event in state.cancellations
    }
    acknowledgements._acks = {
        event.cancellation_id: {} for event in state.cancellations
    }
    for ack in state.acknowledgements:
        acknowledgements._acks[ack.cancellation_id][ack.acknowledgement_level] = ack
    return RestoredExecutionAuthority(dispatch, epochs, acknowledgements)
