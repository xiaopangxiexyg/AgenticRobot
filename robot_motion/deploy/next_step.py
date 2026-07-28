"""Read-only certainty signal for an already-owned policy chunk successor.

This contract does not reserve, commit, pop, prepare, or execute an action.
The source frontend may publish it only when its existing queue semantics
already ensure that the exact same-chunk successor will be the next source
step unless the active lifecycle is explicitly invalidated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from .actions import NormalizedAction


POLICY_NEXT_STEP_EXECUTION_SIGNAL_HASH_DOMAIN_V1 = (
    "dexteleop.policy_next_step_execution_signal.v1"
)
POLICY_NEXT_STEP_EXECUTION_SEMANTICS_V1 = (
    "same_chunk_successor_will_execute_next_v1"
)
POLICY_NEXT_STEP_EXECUTION_BASIS_V1 = (
    "existing_non_replacing_queue_interval_v1"
)

_CHUNK_ID_METADATA_KEY = "deploy_action_chunk_id"
_CHUNK_INDEX_METADATA_KEY = "deploy_action_chunk_index"
_CHUNK_SIZE_METADATA_KEY = "deploy_action_chunk_size"
_SEMANTIC_ACTION_METADATA_KEYS = (
    "arm_action_specs_by_side",
    "hand_action_specs_by_side",
    "hand_active_mask_by_side",
    "end_effector_enabled_sides",
    "pose_action_source_mode",
    "pose_action_mode",
    "pose_action_source_absolute_frame",
    "pose_action_output_absolute_frame",
    "pose_anchor_policy",
    "pose_anchor_sides",
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


def _semantic_action_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in _SEMANTIC_ACTION_METADATA_KEYS
        if key in metadata
    }


def normalized_policy_action_hash(action: NormalizedAction) -> str:
    """Hash the decoded action value independently of mutable queue storage."""

    if not isinstance(action, NormalizedAction):
        raise TypeError("action must be a NormalizedAction.")
    return canonical_content_hash(
        {
            "vector": np.asarray(action.vector, dtype=np.float32),
            "arm_joint_targets": {
                str(side): np.asarray(value, dtype=np.float32)
                for side, value in action.arm_joint_targets.items()
            },
            "semantic_metadata": _semantic_action_metadata(action.metadata),
        },
        domain="dexteleop.policy_next_step_decoded_action.v1",
    )


def sync_queue_preserves_same_chunk_successor(
    *,
    queue_policy: object,
    min_action_queue: object,
) -> bool:
    """Return whether existing sync scheduling prevents successor replacement.

    The result describes only the interval between the current source pop and
    its next source pop.  Lifecycle invalidation remains outside this helper.
    """

    policy = str(queue_policy).strip().lower()
    if policy not in {"append", "replace"}:
        raise ValueError("queue_policy must be 'append' or 'replace'.")
    threshold = _nonnegative_integer(
        min_action_queue,
        name="min_action_queue",
    )
    return policy == "append" or threshold == 0


@dataclass(frozen=True)
class PolicyNextStepExecutionSignal:
    """Proof that one exposed same-chunk successor is execution-certain.

    The guarantee is conditional on the current source lifecycle remaining
    active.  HOLD, FAULT, cancellation, shutdown, or producer-epoch rotation
    invalidate the stream rather than replacing the promised successor.
    """

    current_emission_id: str
    current_emission_hash: str
    policy_node_id: str
    inference_id: str
    chunk_id: int
    current_chunk_keyframe_index: int
    next_chunk_keyframe_index: int
    chunk_size: int
    next_source_fact_id: str
    next_action_hash: str
    next_target_effective_at: float
    effective_clock_domain_id: str
    execution_semantics: str
    guarantee_basis: str
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "current_emission_id",
            "current_emission_hash",
            "policy_node_id",
            "inference_id",
            "next_source_fact_id",
            "next_action_hash",
            "effective_clock_domain_id",
            "semantic_content_hash",
        ):
            _required(getattr(self, name), name=name)
        chunk_id = _nonnegative_integer(self.chunk_id, name="chunk_id")
        current_index = _nonnegative_integer(
            self.current_chunk_keyframe_index,
            name="current_chunk_keyframe_index",
        )
        next_index = _nonnegative_integer(
            self.next_chunk_keyframe_index,
            name="next_chunk_keyframe_index",
        )
        chunk_size = _nonnegative_integer(self.chunk_size, name="chunk_size")
        if chunk_size <= 1:
            raise ValueError("A guaranteed successor requires chunk_size > 1.")
        if next_index != current_index + 1 or next_index >= chunk_size:
            raise ValueError(
                "Guaranteed next step must be the immediate in-chunk successor."
            )
        target = float(self.next_target_effective_at)
        if not np.isfinite(target):
            raise ValueError("next_target_effective_at must be finite.")
        if self.execution_semantics != POLICY_NEXT_STEP_EXECUTION_SEMANTICS_V1:
            raise ValueError("Unsupported next-step execution semantics.")
        if self.guarantee_basis != POLICY_NEXT_STEP_EXECUTION_BASIS_V1:
            raise ValueError("Unsupported next-step guarantee basis.")
        payload = {
            "current_emission_id": self.current_emission_id,
            "current_emission_hash": self.current_emission_hash,
            "policy_node_id": self.policy_node_id,
            "inference_id": self.inference_id,
            "chunk_id": chunk_id,
            "current_chunk_keyframe_index": current_index,
            "next_chunk_keyframe_index": next_index,
            "chunk_size": chunk_size,
            "next_source_fact_id": self.next_source_fact_id,
            "next_action_hash": self.next_action_hash,
            "next_target_effective_at": target,
            "effective_clock_domain_id": self.effective_clock_domain_id,
            "execution_semantics": self.execution_semantics,
            "guarantee_basis": self.guarantee_basis,
        }
        expected_hash = canonical_content_hash(
            payload,
            domain=POLICY_NEXT_STEP_EXECUTION_SIGNAL_HASH_DOMAIN_V1,
        )
        if self.semantic_content_hash != expected_hash:
            raise ValueError("Next-step execution signal hash mismatch.")
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "current_chunk_keyframe_index", current_index)
        object.__setattr__(self, "next_chunk_keyframe_index", next_index)
        object.__setattr__(self, "chunk_size", chunk_size)
        object.__setattr__(self, "next_target_effective_at", target)

    @property
    def guarantee_id(self) -> str:
        return f"policy-next-step:{self.semantic_content_hash}"


def build_policy_next_step_execution_signal(
    *,
    current_emission_id: str,
    current_emission_hash: str,
    policy_node_id: str,
    inference_id: str,
    current_chunk_id: int,
    current_chunk_keyframe_index: int,
    current_chunk_size: int,
    current_effective_until: float,
    effective_clock_domain_id: str,
    next_action: NormalizedAction,
) -> PolicyNextStepExecutionSignal:
    """Bind an already-exposed immediate successor to its current emission."""

    if not isinstance(next_action, NormalizedAction):
        raise TypeError("next_action must be a NormalizedAction.")
    metadata = dict(next_action.metadata)
    try:
        next_chunk_id = int(metadata[_CHUNK_ID_METADATA_KEY])
        next_index = int(metadata[_CHUNK_INDEX_METADATA_KEY])
        next_chunk_size = int(metadata[_CHUNK_SIZE_METADATA_KEY])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Next action is missing chunk lineage.") from exc
    current_chunk = _nonnegative_integer(
        current_chunk_id,
        name="current_chunk_id",
    )
    current_index = _nonnegative_integer(
        current_chunk_keyframe_index,
        name="current_chunk_keyframe_index",
    )
    chunk_size = _nonnegative_integer(
        current_chunk_size,
        name="current_chunk_size",
    )
    if (
        next_chunk_id != current_chunk
        or next_index != current_index + 1
        or next_chunk_size != chunk_size
    ):
        raise ValueError("Next action is not the immediate same-chunk successor.")
    node = _required(policy_node_id, name="policy_node_id")
    inference = _required(inference_id, name="inference_id")
    next_action_hash = normalized_policy_action_hash(next_action)
    next_source_fact_id = (
        f"policy-chunk-step:{node}:{inference}:{current_chunk}:{next_index}"
    )
    payload = {
        "current_emission_id": _required(
            current_emission_id,
            name="current_emission_id",
        ),
        "current_emission_hash": _required(
            current_emission_hash,
            name="current_emission_hash",
        ),
        "policy_node_id": node,
        "inference_id": inference,
        "chunk_id": current_chunk,
        "current_chunk_keyframe_index": current_index,
        "next_chunk_keyframe_index": next_index,
        "chunk_size": chunk_size,
        "next_source_fact_id": next_source_fact_id,
        "next_action_hash": next_action_hash,
        "next_target_effective_at": float(current_effective_until),
        "effective_clock_domain_id": _required(
            effective_clock_domain_id,
            name="effective_clock_domain_id",
        ),
        "execution_semantics": POLICY_NEXT_STEP_EXECUTION_SEMANTICS_V1,
        "guarantee_basis": POLICY_NEXT_STEP_EXECUTION_BASIS_V1,
    }
    digest = canonical_content_hash(
        payload,
        domain=POLICY_NEXT_STEP_EXECUTION_SIGNAL_HASH_DOMAIN_V1,
    )
    return PolicyNextStepExecutionSignal(
        **payload,
        semantic_content_hash=digest,
    )


__all__ = [
    "POLICY_NEXT_STEP_EXECUTION_BASIS_V1",
    "POLICY_NEXT_STEP_EXECUTION_SEMANTICS_V1",
    "POLICY_NEXT_STEP_EXECUTION_SIGNAL_HASH_DOMAIN_V1",
    "PolicyNextStepExecutionSignal",
    "build_policy_next_step_execution_signal",
    "normalized_policy_action_hash",
    "sync_queue_preserves_same_chunk_successor",
]
