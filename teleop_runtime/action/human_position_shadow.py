"""Pure legacy human-position receipt/audit binding.

The canonical GUI sends qpos through ``RM65ArmBackend`` rather than through
``PositionDeployExecutor``.  This module joins the backend's queue receipt to
its existing high-rate ``PositionCommandAuditEvent`` without upgrading either
fact to shared execution admission, M1C enforcement, or anchor evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.position_control.audit import (
    POSITION_TRANSPORT_TARGET_SEQUENCE_KEY,
    PositionCommandAuditEvent,
    position_audit_event_to_dict,
)

from .human_proposal import HumanLegacyCommandObservation


HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.human_position_backend_binding.v1"
)


@dataclass(frozen=True)
class HumanPositionBackendBindingEvidence:
    evidence_id: str
    emission_sequence: int
    command_name: str
    side: str
    target_sequence: int
    target_qpos_hash: str
    dispatch_receipt_operation: str
    dispatch_receipt_status: str
    planner_event_hashes: tuple[str, ...]
    planner_observed: bool
    backend_accepted_observed: bool
    status: str
    non_authoritative: bool
    anchor_commit_eligible: bool
    semantic_content_hash: str

    def __post_init__(self) -> None:
        if self.status not in {
            "target_admitted",
            "planner_observed",
            "backend_accepted_observed",
        }:
            raise ValueError("Unsupported human position binding status.")
        if not self.non_authoritative or self.anchor_commit_eligible:
            raise ValueError("Legacy position binding cannot grant authority/anchor state.")


def _target_sequence(metadata: object) -> int | None:
    if not isinstance(metadata, dict) and not hasattr(metadata, "get"):
        return None
    raw = metadata.get(POSITION_TRANSPORT_TARGET_SEQUENCE_KEY)
    if raw is None or isinstance(raw, bool):
        return None
    value = int(raw)
    return value if value >= 0 and value == raw else None


def bind_human_position_backend_evidence(
    observation: HumanLegacyCommandObservation,
    *,
    audit_events: Iterable[PositionCommandAuditEvent] = (),
) -> tuple[HumanPositionBackendBindingEvidence, ...]:
    """Join an accepted legacy qpos target to later backend planner ticks."""

    if not observation.command_dispatched or observation.outcome != "dispatched":
        return ()
    receipts = {item.command_name: item for item in observation.dispatch_receipts}
    events = tuple(audit_events)
    results: list[HumanPositionBackendBindingEvidence] = []
    for command in observation.final_commands:
        if command.command_type != "joint_vector" or command.side is None:
            continue
        receipt = receipts.get(command.command_name)
        if (
            receipt is None
            or receipt.receipt_kind != "typed_command_dispatch_receipt"
            or receipt.accepted is not True
            or receipt.operation != "queue_background_joint_target"
        ):
            continue
        command_sequence = _target_sequence(command.metadata)
        receipt_sequence = _target_sequence(receipt.metadata)
        if command_sequence is None or receipt_sequence is None:
            continue
        if command_sequence != receipt_sequence:
            raise ValueError("Command and dispatch receipt target sequences disagree.")
        qpos = np.asarray(command.value, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError("Human position binding requires finite 6D qpos.")
        target_hash = canonical_content_hash(
            qpos,
            domain="dexteleop.human_position_backend_target_qpos.v1",
        )
        matching: list[PositionCommandAuditEvent] = []
        for event in events:
            if not isinstance(event, PositionCommandAuditEvent):
                raise TypeError("audit_events must contain PositionCommandAuditEvent.")
            if event.side != command.side or event.target_sequence != command_sequence:
                continue
            if event.segment_target_qpos is not None and not np.array_equal(
                np.asarray(event.segment_target_qpos, dtype=float),
                qpos,
            ):
                raise ValueError("Position audit target does not match dispatched qpos.")
            matching.append(event)
        event_hashes = tuple(
            canonical_content_hash(
                position_audit_event_to_dict(event),
                domain="dexteleop.position_command_audit_event.v1",
            )
            for event in matching
        )
        backend_accepted = any(event.sdk_accepted is True for event in matching)
        status = (
            "backend_accepted_observed"
            if backend_accepted
            else "planner_observed"
            if matching
            else "target_admitted"
        )
        payload = {
            "emission_sequence": int(observation.emission_sequence),
            "command_name": command.command_name,
            "side": command.side,
            "target_sequence": command_sequence,
            "target_qpos_hash": target_hash,
            "dispatch_receipt_operation": receipt.operation,
            "dispatch_receipt_status": receipt.status,
            "planner_event_hashes": event_hashes,
            "planner_observed": bool(matching),
            "backend_accepted_observed": backend_accepted,
            "status": status,
            "non_authoritative": True,
            "anchor_commit_eligible": False,
        }
        digest = canonical_content_hash(
            payload,
            domain=HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1,
        )
        results.append(
            HumanPositionBackendBindingEvidence(
                evidence_id=f"human-position-backend-binding:{digest}",
                **payload,
                semantic_content_hash=digest,
            )
        )
    return tuple(results)


__all__ = [
    "HUMAN_POSITION_BACKEND_BINDING_HASH_DOMAIN_V1",
    "HumanPositionBackendBindingEvidence",
    "bind_human_position_backend_evidence",
]
