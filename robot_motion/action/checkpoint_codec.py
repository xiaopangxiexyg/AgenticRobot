"""Closed data encoding for the logical core component, without live bindings.

An enclosing checkpoint store authenticates provenance and atomically persists
the returned bytes. Decoding does not restore owners; the ordinary constructor's
checkpoint path still validates configuration and the complete causal history.
No module name or executable factory is accepted from the payload.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Mapping
from dataclasses import fields

import numpy as np

from dataset_contracts.action_schema import (
    ActionPayloadSchema,
    ComponentActionSpaceSchema,
)
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.runtime.cancellation import (
    CancellationAcknowledgement,
    ExecutionCancellation,
)
from robot_motion.runtime.dispatch_owner import DispatchOwnerToken
from robot_motion.runtime.execution_checkpoint import ExecutionAuthorityCheckpoint
from robot_motion.runtime.execution_gate import ExecutionGateEvent
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochActivation

from .admission_checkpoint import SourceAdmissionCheckpoint
from .authoritative_checkpoint import AuthoritativeControlCheckpoint
from .authority import AuthorityEvent
from .candidate_authorization import CandidateAuthorizationRecord
from .component_action import (
    ComponentAction,
    ComponentActionSchema,
    ComponentActionValue,
)
from .components import (
    ActionComponentCatalog,
    ActionComponentCatalogV2,
    ActionComponentSpec,
    ControlComponentSpecV2,
)
from .decision_checkpoint import DecisionCheckpoint
from .persistence import ActiveExecutionSegment
from .proposal import ActionProposal

_RECORDS = (
    ActionPayloadSchema,
    ComponentActionSpaceSchema,
    ControlTraceFact,
    ExecutionCancellation,
    CancellationAcknowledgement,
    DispatchOwnerToken,
    ExecutionAuthorityCheckpoint,
    ExecutionGateEvent,
    ClockDomainDescriptor,
    ProducerEpochActivation,
    SourceAdmissionCheckpoint,
    AuthoritativeControlCheckpoint,
    AuthorityEvent,
    CandidateAuthorizationRecord,
    ComponentAction,
    ComponentActionSchema,
    ComponentActionValue,
    ActionComponentCatalog,
    ActionComponentSpec,
    ActionComponentCatalogV2,
    ControlComponentSpecV2,
    DecisionCheckpoint,
    ActiveExecutionSegment,
    ActionProposal,
)
_TYPES = {kind.__name__: kind for kind in _RECORDS}
_SCHEMA = "agenticrobot.authoritative_control.checkpoint.v1"


def _encode(value, depth=0):
    if depth > 64:
        raise ValueError("Checkpoint data nesting is too deep.")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Checkpoint data must be finite.")
        return ["float", value.hex()]
    if type(value) in _RECORDS:
        return [
            "record",
            type(value).__name__,
            {
                f.name: _encode(getattr(value, f.name), depth + 1)
                for f in fields(value)
                if f.init
            },
        ]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("Checkpoint mappings require string keys.")
        return [
            "mapping",
            {key: _encode(part, depth + 1) for key, part in value.items()},
        ]
    if type(value) in (list, tuple, set, frozenset):
        values = list(value)
        if type(value) in (set, frozenset):
            if any(type(item) is not str for item in values):
                raise ValueError("Checkpoint sets contain only IDs.")
            values.sort()
        return [type(value).__name__, [_encode(part, depth + 1) for part in values]]
    if type(value) is bytes:
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, np.ndarray):
        if (
            value.dtype.kind not in "biuf"
            or value.dtype.hasobject
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(
                "Checkpoint arrays must be finite primitive numeric values."
            )
        return [
            "array",
            value.dtype.str,
            list(value.shape),
            base64.b64encode(value.tobytes(order="C")).decode("ascii"),
        ]
    raise ValueError(f"Unsupported checkpoint data: {type(value).__name__}.")


def _decode(value, depth=0):
    if depth > 64:
        raise ValueError("Checkpoint data nesting is too deep.")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is not list or not value:
        raise ValueError("Invalid checkpoint value encoding.")
    tag = value[0]
    if tag == "float" and len(value) == 2 and type(value[1]) is str:
        result = float.fromhex(value[1])
        if math.isfinite(result):
            return result
    elif (
        tag == "record"
        and len(value) == 3
        and type(value[1]) is str
        and type(value[2]) is dict
    ):
        kind = _TYPES.get(value[1])
        if kind is None or set(value[2]) != {f.name for f in fields(kind) if f.init}:
            raise ValueError("Unknown checkpoint record or field layout.")
        return kind(**{key: _decode(part, depth + 1) for key, part in value[2].items()})
    elif tag == "mapping" and len(value) == 2 and type(value[1]) is dict:
        return {key: _decode(part, depth + 1) for key, part in value[1].items()}
    elif (
        tag in ("list", "tuple", "set", "frozenset")
        and len(value) == 2
        and type(value[1]) is list
    ):
        result = [_decode(part, depth + 1) for part in value[1]]
        if tag in ("set", "frozenset") and (
            any(type(item) is not str for item in result)
            or len(set(result)) != len(result)
        ):
            raise ValueError("Invalid checkpoint set values.")
        return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag](
            result
        )
    elif tag == "bytes" and len(value) == 2 and type(value[1]) is str:
        return base64.b64decode(value[1], validate=True)
    elif (
        tag == "array"
        and len(value) == 4
        and type(value[1]) is str
        and type(value[2]) is list
        and type(value[3]) is str
    ):
        dtype = np.dtype(value[1])
        if (
            dtype.kind not in "biuf"
            or dtype.hasobject
            or len(value[2]) > 32
            or any(type(size) is not int or size < 0 for size in value[2])
        ):
            raise ValueError("Invalid checkpoint array descriptor.")
        raw = base64.b64decode(value[3], validate=True)
        if math.prod(value[2]) * dtype.itemsize != len(raw):
            raise ValueError("Checkpoint array byte count differs.")
        result = np.frombuffer(raw, dtype=dtype).reshape(value[2])
        if not np.all(np.isfinite(result)):
            raise ValueError("Checkpoint array contains nonfinite data.")
        return result
    raise ValueError("Invalid checkpoint value encoding.")


def encode_control_checkpoint(state: AuthoritativeControlCheckpoint) -> bytes:
    if type(state) is not AuthoritativeControlCheckpoint:
        raise ValueError("Expected a complete logical core checkpoint.")
    return json.dumps(
        {"schema": _SCHEMA, "state": _encode(state)},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_control_checkpoint(
    payload: bytes, *, max_bytes: int
) -> AuthoritativeControlCheckpoint:
    if (
        type(max_bytes) is not int
        or max_bytes <= 0
        or type(payload) is not bytes
        or len(payload) > max_bytes
    ):
        raise ValueError("Invalid checkpoint byte budget or payload.")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate checkpoint JSON key.")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique)
    if (
        type(value) is not dict
        or set(value) != {"schema", "state"}
        or value["schema"] != _SCHEMA
    ):
        raise ValueError("Unknown logical core checkpoint encoding.")
    result = _decode(value["state"])
    if type(result) is not AuthoritativeControlCheckpoint:
        raise ValueError("Invalid checkpoint root record.")
    return result
