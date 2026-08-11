"""Pure history-context contract shared by arm-hold training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .arm_mode import ARM_MODE_SIDE_ORDER_V1
from .canonical_encoding import canonical_content_hash


ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1 = "dexteleop.arm_hold_history_context.v1"
ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1 = (
    "dexteleop.arm_hold_history_sequence_context.v1"
)
ARM_HOLD_HISTORY_SEQUENCE_STEPS_V1 = 2
ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1 = (
    "previous_canonical_collected_mode_v1"
)
ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1 = "previous_resolved_mode_v1"
ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1 = (2, 3, 4, 8, 16)
ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1 = frozenset(
    {
        ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1,
        ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1,
    }
)
POLICY_ARM_HOLD_HISTORY_INPUT_SCHEMA_V1 = (
    "dexteleop_policy_arm_hold_history_input_v1"
)
POLICY_ARM_HOLD_HISTORY_INPUT_METADATA_KEY = "arm_hold_history_input_contract"
POLICY_ARM_HOLD_HISTORY_INPUT_HASH_KEY = "arm_hold_history_input_contract_hash"
POLICY_ARM_HOLD_HISTORY_INPUT_HASH_DOMAIN_V1 = (
    "dexteleop.policy_arm_hold_history_input.semantic.v1"
)
POLICY_ARM_HOLD_HISTORY_REPRESENTATION_DWELL_BUCKET_V1 = (
    "previous_dwell_bucket_v1"
)


class ArmHoldHistoryContextError(ValueError):
    """Raised when an arm-hold history context is ambiguous or malformed."""


def _nonempty(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ArmHoldHistoryContextError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class ArmHoldHistoryContext:
    """Per-side previous hold mode and dwell with explicit availability."""

    previous_mode: np.ndarray
    available_mask: np.ndarray
    dwell_steps: np.ndarray
    provenance: str = ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1
    schema: str = ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_mode, dtype=np.float32)
        available = np.asarray(self.available_mask, dtype=bool)
        dwell = np.asarray(self.dwell_steps, dtype=np.int64)
        if previous.shape != (2,) or available.shape != (2,) or dwell.shape != (2,):
            raise ArmHoldHistoryContextError(
                "Arm-hold history context must be per-side."
            )
        if self.schema != ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1:
            raise ArmHoldHistoryContextError(
                "Unknown arm-hold history context schema."
            )
        provenance = str(self.provenance).strip()
        if provenance not in ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1:
            raise ArmHoldHistoryContextError(
                "Unknown arm-hold history context provenance."
            )
        if np.any(dwell < 0) or np.any((dwell > 0) != available):
            raise ArmHoldHistoryContextError(
                "History dwell must be positive exactly where context is available."
            )
        if np.any(~np.isin(previous[available], (0.0, 1.0))):
            raise ArmHoldHistoryContextError(
                "Available previous modes must be binary."
            )
        if np.any(previous[~available] != 0.0):
            raise ArmHoldHistoryContextError(
                "Unavailable previous modes must use zero storage."
            )
        frozen_previous = previous.copy()
        frozen_available = available.copy()
        frozen_dwell = dwell.copy()
        frozen_previous.setflags(write=False)
        frozen_available.setflags(write=False)
        frozen_dwell.setflags(write=False)
        object.__setattr__(self, "previous_mode", frozen_previous)
        object.__setattr__(self, "available_mask", frozen_available)
        object.__setattr__(self, "dwell_steps", frozen_dwell)
        object.__setattr__(self, "provenance", provenance)

    def to_training_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provenance": self.provenance,
            "previous_mode": self.previous_mode.copy(),
            "available_mask": self.available_mask.copy(),
            "dwell_steps": self.dwell_steps.copy(),
        }


@dataclass(frozen=True)
class ArmHoldHistorySequenceContext:
    """Two causal per-side label/dwell snapshots, oldest to newest."""

    previous_mode: np.ndarray
    available_mask: np.ndarray
    dwell_steps: np.ndarray
    provenance: str = ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1
    schema: str = ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1

    def __post_init__(self) -> None:
        previous = np.asarray(self.previous_mode, dtype=np.float32)
        available = np.asarray(self.available_mask, dtype=bool)
        dwell = np.asarray(self.dwell_steps, dtype=np.int64)
        expected = (ARM_HOLD_HISTORY_SEQUENCE_STEPS_V1, 2)
        if previous.shape != expected or available.shape != expected or dwell.shape != expected:
            raise ArmHoldHistoryContextError(
                "Arm-hold history sequence must have shape [2,2]."
            )
        if self.schema != ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1:
            raise ArmHoldHistoryContextError(
                "Unknown arm-hold history sequence context schema."
            )
        provenance = str(self.provenance).strip()
        if provenance not in ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1:
            raise ArmHoldHistoryContextError(
                "Unknown arm-hold history sequence context provenance."
            )
        if np.any(dwell < 0) or np.any((dwell > 0) != available):
            raise ArmHoldHistoryContextError(
                "History sequence dwell must be positive exactly where available."
            )
        if np.any(~np.isin(previous[available], (0.0, 1.0))):
            raise ArmHoldHistoryContextError(
                "Available history sequence modes must be binary."
            )
        if np.any(previous[~available] != 0.0):
            raise ArmHoldHistoryContextError(
                "Unavailable history sequence modes must use zero storage."
            )
        if np.any(available[0] & ~available[1]):
            raise ArmHoldHistoryContextError(
                "History sequence availability must be a contiguous suffix."
            )
        both = available[0] & available[1]
        same = previous[0] == previous[1]
        invalid_same = both & same & (dwell[1] != dwell[0] + 1)
        invalid_switch = both & ~same & (dwell[1] != 1)
        if np.any(invalid_same | invalid_switch):
            raise ArmHoldHistoryContextError(
                "History sequence mode/dwell progression is not causal."
            )
        frozen_previous = previous.copy()
        frozen_available = available.copy()
        frozen_dwell = dwell.copy()
        frozen_previous.setflags(write=False)
        frozen_available.setflags(write=False)
        frozen_dwell.setflags(write=False)
        object.__setattr__(self, "previous_mode", frozen_previous)
        object.__setattr__(self, "available_mask", frozen_available)
        object.__setattr__(self, "dwell_steps", frozen_dwell)
        object.__setattr__(self, "provenance", provenance)

    def to_training_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "provenance": self.provenance,
            "order": "oldest_to_newest_v1",
            "previous_mode": self.previous_mode.copy(),
            "available_mask": self.available_mask.copy(),
            "dwell_steps": self.dwell_steps.copy(),
        }


@dataclass(frozen=True)
class PolicyArmHoldHistoryInputContract:
    """Artifact-bound encoding of canonical history for one policy request.

    The canonical context retains mode, availability, dwell, and provenance.
    This v1 model input deliberately exposes only the per-side dwell bucket so
    a learner cannot directly copy its preceding label prediction.
    """

    input_key: str
    representation: str = (
        POLICY_ARM_HOLD_HISTORY_REPRESENTATION_DWELL_BUCKET_V1
    )
    side_order: tuple[str, ...] = ARM_MODE_SIDE_ORDER_V1
    dtype: str = "int64"
    shape: tuple[int, ...] = (2,)
    dwell_bucket_thresholds: tuple[int, ...] = (
        ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1
    )
    unavailable_bucket: int = 0
    training_provenance: str = (
        ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1
    )
    deployment_provenance: str = ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1
    context_schema: str = ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1
    schema: str = POLICY_ARM_HOLD_HISTORY_INPUT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema != POLICY_ARM_HOLD_HISTORY_INPUT_SCHEMA_V1:
            raise ArmHoldHistoryContextError(
                f"Unsupported policy arm-hold history schema {self.schema!r}."
            )
        if (
            self.representation
            != POLICY_ARM_HOLD_HISTORY_REPRESENTATION_DWELL_BUCKET_V1
        ):
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 requires previous_dwell_bucket_v1."
            )
        if tuple(self.side_order) != ARM_MODE_SIDE_ORDER_V1:
            raise ArmHoldHistoryContextError(
                f"side_order must be {ARM_MODE_SIDE_ORDER_V1!r}."
            )
        if str(self.dtype).strip().lower() != "int64":
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 dtype must be int64."
            )
        if tuple(self.shape) != (2,):
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 shape must be [2]."
            )
        if tuple(self.dwell_bucket_thresholds) != (
            ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1
        ):
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 dwell thresholds differ."
            )
        if isinstance(self.unavailable_bucket, bool) or int(
            self.unavailable_bucket
        ) != 0:
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 unavailable bucket must be zero."
            )
        if (
            self.training_provenance
            != ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1
            or self.deployment_provenance
            != ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1
        ):
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 provenance semantics differ."
            )
        if self.context_schema != ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1:
            raise ArmHoldHistoryContextError(
                "Policy arm-hold history v1 requires the canonical context schema."
            )
        object.__setattr__(
            self,
            "input_key",
            _nonempty(self.input_key, name="input_key"),
        )
        object.__setattr__(self, "side_order", ARM_MODE_SIDE_ORDER_V1)
        object.__setattr__(self, "dtype", "int64")
        object.__setattr__(self, "shape", (2,))
        object.__setattr__(
            self,
            "dwell_bucket_thresholds",
            ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1,
        )
        object.__setattr__(self, "unavailable_bucket", 0)

    def to_dict(self, *, include_hash: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "input_key": self.input_key,
            "representation": self.representation,
            "side_order": list(self.side_order),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "dwell_bucket_thresholds": list(self.dwell_bucket_thresholds),
            "unavailable_bucket": self.unavailable_bucket,
            "training_provenance": self.training_provenance,
            "deployment_provenance": self.deployment_provenance,
            "context_schema": self.context_schema,
        }
        if include_hash:
            payload["contract_hash"] = self.contract_hash
        return payload

    @property
    def contract_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(),
            domain=POLICY_ARM_HOLD_HISTORY_INPUT_HASH_DOMAIN_V1,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "PolicyArmHoldHistoryInputContract":
        if not isinstance(payload, Mapping):
            raise ArmHoldHistoryContextError(
                "arm_hold_history_input_contract must be a mapping."
            )
        if "input_key" not in payload:
            raise ArmHoldHistoryContextError(
                "arm_hold_history_input_contract is missing input_key."
            )
        contract = cls(
            input_key=str(payload["input_key"]),
            representation=str(payload.get("representation", "")),
            side_order=tuple(payload.get("side_order", ())),
            dtype=str(payload.get("dtype", "")),
            shape=tuple(payload.get("shape", ())),
            dwell_bucket_thresholds=tuple(
                payload.get("dwell_bucket_thresholds", ())
            ),
            unavailable_bucket=payload.get("unavailable_bucket", -1),
            training_provenance=str(payload.get("training_provenance", "")),
            deployment_provenance=str(
                payload.get("deployment_provenance", "")
            ),
            context_schema=str(payload.get("context_schema", "")),
            schema=str(payload.get("schema", "")),
        )
        declared_hash = payload.get("contract_hash")
        if declared_hash is not None and str(declared_hash) != contract.contract_hash:
            raise ArmHoldHistoryContextError(
                "arm_hold_history_input_contract contract_hash does not match."
            )
        return contract


def encode_policy_arm_hold_history_input(
    context: ArmHoldHistoryContext,
    contract: PolicyArmHoldHistoryInputContract,
) -> np.ndarray:
    """Encode one canonical context into the artifact-declared model tensor."""

    if not isinstance(context, ArmHoldHistoryContext):
        raise TypeError("context must be an ArmHoldHistoryContext.")
    if not isinstance(contract, PolicyArmHoldHistoryInputContract):
        raise TypeError(
            "contract must be a PolicyArmHoldHistoryInputContract."
        )
    encoded = arm_hold_dwell_bucket_indices(
        context.dwell_steps,
        context.available_mask,
        thresholds=contract.dwell_bucket_thresholds,
    )
    encoded.setflags(write=False)
    return encoded


def arm_hold_dwell_bucket_indices(
    dwell_steps: np.ndarray,
    available_mask: np.ndarray,
    *,
    thresholds: tuple[int, ...] = ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1,
) -> np.ndarray:
    """Map canonical dwell arrays to unavailable/1/2/3/4-7/8-15/16+."""

    dwell = np.asarray(dwell_steps)
    available = np.asarray(available_mask)
    resolved_thresholds = tuple(int(value) for value in thresholds)
    if (
        dwell.shape != available.shape
        or dwell.ndim == 0
        or dwell.shape[-1] != 2
        or not np.issubdtype(dwell.dtype, np.integer)
        or available.dtype != np.bool_
        or np.any(dwell < 0)
        or np.any((dwell > 0) != available)
        or resolved_thresholds != ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1
    ):
        raise ArmHoldHistoryContextError(
            "Policy arm-hold dwell bucket source arrays are invalid."
        )
    bucket = available.astype(np.int64)
    for threshold in resolved_thresholds:
        bucket = bucket + (dwell >= threshold).astype(np.int64)
    return np.asarray(bucket, dtype=np.int64)


__all__ = [
    "ARM_HOLD_CANONICAL_COLLECTED_CONTEXT_PROVENANCE_V1",
    "ARM_HOLD_DWELL_BUCKET_THRESHOLDS_V1",
    "ARM_HOLD_HISTORY_CONTEXT_PROVENANCES_V1",
    "ARM_HOLD_HISTORY_CONTEXT_SCHEMA_V1",
    "ARM_HOLD_HISTORY_SEQUENCE_CONTEXT_SCHEMA_V1",
    "ARM_HOLD_HISTORY_SEQUENCE_STEPS_V1",
    "ARM_HOLD_RESOLVED_CONTEXT_PROVENANCE_V1",
    "POLICY_ARM_HOLD_HISTORY_INPUT_HASH_DOMAIN_V1",
    "POLICY_ARM_HOLD_HISTORY_INPUT_HASH_KEY",
    "POLICY_ARM_HOLD_HISTORY_INPUT_METADATA_KEY",
    "POLICY_ARM_HOLD_HISTORY_INPUT_SCHEMA_V1",
    "POLICY_ARM_HOLD_HISTORY_REPRESENTATION_DWELL_BUCKET_V1",
    "ArmHoldHistoryContext",
    "ArmHoldHistoryContextError",
    "ArmHoldHistorySequenceContext",
    "PolicyArmHoldHistoryInputContract",
    "arm_hold_dwell_bucket_indices",
    "encode_policy_arm_hold_history_input",
]
