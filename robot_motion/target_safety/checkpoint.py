"""Private continuation of the existing TCP envelope owner.

The caller fences the application transition around this cut and authenticates
its provenance. This is data for an independent simulator branch, not a public
Artifact or a grant to enter READY. No lifecycle event is replayed here; an
enclosing application must separately restore its operation and HOLD owners.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

from robot_motion.action.checkpoint_codec import (
    decode_checkpoint_value,
    encode_checkpoint_value,
)

from .runtime import (
    TCPPoseEnvelopeActivationReceipt,
    TCPPoseEnvelopeCatalogEntry,
    TCPPoseEnvelopeSafetyRuntime,
    TCPPoseEnvelopeSelection,
    _ActiveEnvelope,
)
from .tcp_pose_envelope import (
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeProjectionResult,
    TCPPoseEnvelopeSideConfig,
    TCPPoseEnvelopeSideProjection,
    TCPPoseSafetyReferenceReceipt,
    project_tcp_pose_envelope,
)

_CONFIG = {
    "catalog_entries",
    "target_frame_binding_hash",
    "tool_frame_binding_hash_by_side",
    "reference_frame_by_side",
    "controlled_frame_id",
    "clock_domain_id",
}
_STATE = {
    "_safety_reference_epoch",
    "_reference",
    "_active",
    "_last_projection",
}
_LAYOUT = _CONFIG | _STATE | {"_entry_by_hash", "_entry_by_id", "_lock"}
_RECORDS = (
    TCPPoseEnvelopeSideConfig,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeCatalogEntry,
    TCPPoseEnvelopeSelection,
    TCPPoseEnvelopeActivationReceipt,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseSafetyReferenceReceipt,
    TCPPoseEnvelopeSideProjection,
    TCPPoseEnvelopeProjectionResult,
    _ActiveEnvelope,
)
_LOCK_TYPE = type(threading.RLock())


@dataclass(frozen=True)
class TCPPoseEnvelopeCheckpoint:
    configuration: dict
    captured_at: float
    state: dict
    content_hash: str


def _encoded(value):
    return encode_checkpoint_value(value, extra_record_types=_RECORDS)


def _bytes(value):
    return json.dumps(_encoded(value), sort_keys=True, separators=(",", ":")).encode()


def _copy(value):
    encoded = _encoded(value)
    copied = decode_checkpoint_value(encoded, extra_record_types=_RECORDS)
    if _encoded(copied) != encoded:
        raise ValueError("Noncanonical envelope checkpoint data.")

    def check(original, clone):
        if type(original) in _RECORDS:
            record_fields = fields(original)
            if set(vars(original)) != {field.name for field in record_fields}:
                raise ValueError("Unknown envelope record field layout.")
            if getattr(original, "semantic_content_hash", None) != getattr(
                clone, "semantic_content_hash", None
            ):
                raise ValueError(
                    "Envelope record cached identity differs from its data."
                )
            for field in record_fields:
                if field.init:
                    check(getattr(original, field.name), getattr(clone, field.name))
        elif isinstance(original, Mapping):
            for key, part in original.items():
                check(part, clone[key])
        elif type(original) in (tuple, list):
            for part, cloned_part in zip(original, clone):
                check(part, cloned_part)

    check(value, copied)
    return copied


def _configuration(runtime):
    return {name: getattr(runtime, name) for name in sorted(_CONFIG)}


def _hash(configuration, captured_at, state):
    return hashlib.sha256(
        _bytes(("tcp_pose_envelope.checkpoint.v1", configuration, captured_at, state))
    ).hexdigest()


def _cut(value):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError("Envelope checkpoint requires an exact finite cut time.")


def _new_runtime(configuration):
    if type(configuration) is not dict or set(configuration) != _CONFIG:
        raise ValueError("Envelope configuration fields differ.")
    runtime = TCPPoseEnvelopeSafetyRuntime(**configuration)
    if _bytes(_configuration(runtime)) != _bytes(configuration):
        raise ValueError(
            "Envelope configuration is not in its original canonical form."
        )
    return runtime


def _validate_state(runtime, state, captured_at):
    if type(state) is not dict or set(state) != _STATE:
        raise ValueError("Envelope state fields differ.")
    epoch = state["_safety_reference_epoch"]
    reference = state["_reference"]
    active = state["_active"]
    projection = state["_last_projection"]
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Invalid safety reference epoch.")
    if reference is not None and (
        type(reference) is not TCPPoseSafetyReferenceReceipt
        or (
            reference.safety_reference_epoch != epoch
            or reference.target_frame_binding_hash != runtime.target_frame_binding_hash
            or reference.controlled_frame_id != runtime.controlled_frame_id
            or reference.clock_domain_id != runtime.clock_domain_id
            or reference.tool_frame_binding_hash_by_side
            != runtime.tool_frame_binding_hash_by_side
            or reference.reference_frame_by_side
            != {
                side: runtime.reference_frame_by_side.get(side)
                for side in reference.tcp_pose_by_side
            }
            or any(t > captured_at for t in reference.captured_at_by_side.values())
        )
    ):
        raise ValueError("Safety reference does not belong to the saved envelope cut.")
    if active is None:
        if projection is not None:
            raise ValueError("Inactive envelope cannot retain an active projection.")
        return
    if type(active) is not _ActiveEnvelope or (
        type(active.execution_epoch) is not int
        or active.execution_epoch < 0
        or type(active.entry) is not TCPPoseEnvelopeCatalogEntry
        or type(active.selection) is not TCPPoseEnvelopeSelection
        or type(active.receipt) is not TCPPoseEnvelopeActivationReceipt
    ):
        raise ValueError("Invalid active envelope record.")
    selection = active.selection
    receipt = active.receipt
    entry = runtime._entry_by_hash.get(selection.deployment_envelope_hash)
    if entry is None or (
        _bytes(active.entry) != _bytes(entry)
        or selection.deployment_envelope_id != entry.envelope_id
        or selection.constraint_set_hash != entry.constraint_set_hash
        or receipt.route_id not in entry.allowed_route_ids
    ):
        raise ValueError("Active envelope selection, catalog or Route differs.")
    anchor = None
    if entry.profile.anchor_required_sides:
        if reference is None:
            raise ValueError("Relative active envelope has no stable reference.")
        anchor = reference.bind_profile(
            anchor_id=(
                f"operation-anchor:{active.operation_id}:"
                f"reference-{reference.safety_reference_epoch}"
            ),
            profile=entry.profile,
            execution_epoch=active.execution_epoch,
        )
    if _bytes(anchor) != _bytes(active.anchor):
        raise ValueError("Active envelope anchor differs from its stable reference.")
    expected_receipt = TCPPoseEnvelopeActivationReceipt(
        activation_id=(
            f"tcp-envelope-activation:{active.operation_id}:epoch-{active.execution_epoch}"
        ),
        operation_id=active.operation_id,
        route_id=receipt.route_id,
        execution_epoch=active.execution_epoch,
        selection_hash=selection.semantic_content_hash,
        deployment_envelope_hash=entry.semantic_content_hash,
        tcp_pose_envelope_profile_hash=entry.profile.semantic_content_hash,
        safety_reference_hash=None
        if anchor is None
        else reference.semantic_content_hash,
        operation_anchor_hash=None if anchor is None else anchor.semantic_content_hash,
        source_event_ids=receipt.source_event_ids,
    )
    if _bytes(receipt) != _bytes(expected_receipt):
        raise ValueError("Active envelope receipt and original operation links differ.")
    if projection is not None:
        if type(projection) is not TCPPoseEnvelopeProjectionResult:
            raise ValueError("Invalid saved envelope projection.")
        expected = project_tcp_pose_envelope(
            projection.raw_tcp_pose_by_side,
            profile=entry.profile,
            target_frame_binding_hash=runtime.target_frame_binding_hash,
            tool_frame_binding_hash_by_side=runtime.tool_frame_binding_hash_by_side,
            controlled_frame_id=runtime.controlled_frame_id,
            reference_frame_by_side=runtime.reference_frame_by_side,
            execution_epoch=active.execution_epoch,
            anchor_receipt=anchor,
        )
        if _bytes(projection) != _bytes(expected):
            raise ValueError("Saved projection differs from its original envelope.")


def capture_safety_envelope(runtime, *, captured_at):
    """Copy a caller-fenced original owner without changing reference or epochs."""
    _cut(captured_at)
    if (
        type(runtime) is not TCPPoseEnvelopeSafetyRuntime
        or set(vars(runtime)) != _LAYOUT
        or type(runtime._lock) is not _LOCK_TYPE
    ):
        raise ValueError("Unsupported safety envelope owner or field layout.")
    with runtime._lock:
        configuration = _copy(_configuration(runtime))
        probe = _new_runtime(configuration)
        for index, field in (
            (runtime._entry_by_hash, "semantic_content_hash"),
            (runtime._entry_by_id, "envelope_id"),
        ):
            if (
                type(index) is not MappingProxyType
                or set(index)
                != {getattr(item, field) for item in runtime.catalog_entries}
                or any(
                    index[getattr(item, field)] is not item
                    for item in runtime.catalog_entries
                )
            ):
                raise ValueError(
                    "Envelope catalog index no longer owns the exact entries."
                )
        if runtime._active is not None and (
            type(runtime._active) is not _ActiveEnvelope
            or type(runtime._active.selection) is not TCPPoseEnvelopeSelection
            or runtime._active.entry
            is not runtime._entry_by_hash.get(
                runtime._active.selection.deployment_envelope_hash
            )
        ):
            raise ValueError("Active envelope entry is not the original catalog entry.")
        state = _copy({name: getattr(runtime, name) for name in sorted(_STATE)})
        _validate_state(probe, state, captured_at)
        return TCPPoseEnvelopeCheckpoint(
            configuration, captured_at, state, _hash(configuration, captured_at, state)
        )


def restore_safety_envelope(checkpoint, *, captured_at):
    """Return a fresh original owner; activation data alone cannot authorize motion."""
    _cut(captured_at)
    if type(checkpoint) is not TCPPoseEnvelopeCheckpoint or (
        type(checkpoint.captured_at) is not float
        or checkpoint.captured_at != captured_at
        or checkpoint.content_hash
        != _hash(checkpoint.configuration, checkpoint.captured_at, checkpoint.state)
    ):
        raise ValueError("Envelope checkpoint content or logical cut differs.")
    configuration = _copy(checkpoint.configuration)
    state = _copy(checkpoint.state)
    runtime = _new_runtime(configuration)
    _validate_state(runtime, state, captured_at)
    active = state["_active"]
    if active is not None:
        state["_active"] = replace(
            active,
            entry=runtime._entry_by_hash[active.selection.deployment_envelope_hash],
        )
    for name, value in state.items():
        setattr(runtime, name, value)
    return runtime
