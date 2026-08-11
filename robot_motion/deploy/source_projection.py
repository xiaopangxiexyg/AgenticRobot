"""Policy-source action semantics before canonical composition ingress.

Every policy source may use its own action representation, but it must project
that representation to an absolute deployment action before composition.  The
contracts here preserve the source values and the exact anchor used for that
projection.  They own no inference, queue, composition, safety, IK, reference,
or dispatch state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    normalize_action_pose_mode,
)


POLICY_SOURCE_ACTION_ABSOLUTE_TARGET = "absolute_target"
POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK = (
    "observation_relative_chunk"
)
POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA = "sequential_target_delta"
POLICY_SOURCE_ACTION_SEMANTICS = frozenset(
    {
        POLICY_SOURCE_ACTION_ABSOLUTE_TARGET,
        POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK,
        POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA,
    }
)
POLICY_SOURCE_ACTION_PROJECTION_SCHEMA_V1 = (
    "policy_source_action_projection_v1"
)
POLICY_SOURCE_ACTION_PROJECTION_HASH_DOMAIN_V1 = (
    "dexteleop.policy_source_action_projection.v1"
)
POLICY_SOURCE_ACTION_PROJECTION_METADATA_KEY = (
    "policy_source_action_projection"
)
POLICY_SOURCE_ANCHOR_NONE = "none"
POLICY_SOURCE_ANCHOR_OBSERVATION_TCP_FIXED_FOR_CHUNK = (
    "observation_tcp_fixed_for_chunk"
)
POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET = "previous_policy_target"


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite_values(
    values: Mapping[str, Sequence[float]],
    *,
    name: str,
    allow_empty: bool = False,
) -> Mapping[str, tuple[float, ...]]:
    normalized: dict[str, tuple[float, ...]] = {}
    for raw_key, raw_values in dict(values).items():
        key = _required(raw_key, name=f"{name} key")
        array = np.asarray(raw_values, dtype=float).reshape(-1)
        if array.size == 0 or np.any(~np.isfinite(array)):
            raise ValueError(f"{name}[{key!r}] must contain finite values.")
        normalized[key] = tuple(float(value) for value in array)
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class PolicySourceActionProjection:
    """One immutable source-action to absolute-action projection record.

    ``observation_relative_chunk`` fixes one measured observation anchor for
    the whole chunk. ``sequential_target_delta`` instead uses the previous
    policy target and therefore carries an explicit monotonically increasing
    target generation.  The two relative semantics are deliberately distinct.
    """

    source_semantics: str
    source_pose_mode: str
    source_action_schema_id: str
    source_action_by_channel: Mapping[str, Sequence[float]]
    conditioned_action_by_channel: Mapping[str, Sequence[float]]
    decoded_absolute_action_hash: str
    anchor_policy: str = POLICY_SOURCE_ANCHOR_NONE
    anchor_value_by_component: Mapping[str, Sequence[float]] = field(
        default_factory=dict
    )
    target_generation: int | None = None
    schema_id: str = POLICY_SOURCE_ACTION_PROJECTION_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_id != POLICY_SOURCE_ACTION_PROJECTION_SCHEMA_V1:
            raise ValueError(
                f"Unsupported policy source projection schema: {self.schema_id!r}."
            )
        semantics = _required(self.source_semantics, name="source_semantics")
        if semantics not in POLICY_SOURCE_ACTION_SEMANTICS:
            raise ValueError(f"Unsupported policy source semantics: {semantics!r}.")
        pose_mode = normalize_action_pose_mode(self.source_pose_mode)
        source = _finite_values(
            self.source_action_by_channel,
            name="source_action_by_channel",
        )
        conditioned = _finite_values(
            self.conditioned_action_by_channel,
            name="conditioned_action_by_channel",
        )
        if set(source) != set(conditioned):
            raise ValueError(
                "Source and conditioned action channels must match exactly."
            )
        anchors = _finite_values(
            self.anchor_value_by_component,
            name="anchor_value_by_component",
            allow_empty=True,
        )
        anchor_policy = _required(self.anchor_policy, name="anchor_policy")
        generation = self.target_generation
        if semantics == POLICY_SOURCE_ACTION_ABSOLUTE_TARGET:
            if pose_mode != ACTION_POSE_MODE_ABSOLUTE:
                raise ValueError("Absolute target semantics require absolute pose mode.")
            if anchor_policy != POLICY_SOURCE_ANCHOR_NONE or anchors:
                raise ValueError("Absolute target semantics cannot carry an anchor.")
            if generation is not None:
                raise ValueError("Absolute target semantics cannot carry a generation.")
        elif semantics == POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK:
            if pose_mode != ACTION_POSE_MODE_RELATIVE:
                raise ValueError("Observation-relative semantics require relative pose mode.")
            if anchor_policy != POLICY_SOURCE_ANCHOR_OBSERVATION_TCP_FIXED_FOR_CHUNK:
                raise ValueError("Observation-relative semantics require the fixed chunk anchor.")
            if not anchors:
                raise ValueError("Observation-relative semantics require anchor values.")
            if generation is not None:
                raise ValueError("Observation-relative chunks do not use target generation.")
        else:
            if pose_mode != ACTION_POSE_MODE_RELATIVE:
                raise ValueError("Sequential target deltas require relative pose mode.")
            if anchor_policy != POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET:
                raise ValueError("Sequential target deltas require the previous target anchor.")
            if not anchors:
                raise ValueError("Sequential target deltas require anchor values.")
            if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
                raise ValueError(
                    "Sequential target deltas require a non-negative target_generation."
                )
        object.__setattr__(self, "source_semantics", semantics)
        object.__setattr__(self, "source_pose_mode", pose_mode)
        object.__setattr__(
            self,
            "source_action_schema_id",
            _required(self.source_action_schema_id, name="source_action_schema_id"),
        )
        object.__setattr__(self, "source_action_by_channel", source)
        object.__setattr__(self, "conditioned_action_by_channel", conditioned)
        object.__setattr__(self, "anchor_policy", anchor_policy)
        object.__setattr__(self, "anchor_value_by_component", anchors)
        object.__setattr__(self, "target_generation", generation)
        object.__setattr__(
            self,
            "decoded_absolute_action_hash",
            _required(
                self.decoded_absolute_action_hash,
                name="decoded_absolute_action_hash",
            ),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_dict(include_hash=False),
            domain=POLICY_SOURCE_ACTION_PROJECTION_HASH_DOMAIN_V1,
        )

    @property
    def projection_id(self) -> str:
        return f"policy-source-action-projection:{self.semantic_content_hash}"

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_id": self.schema_id,
            "source_semantics": self.source_semantics,
            "source_pose_mode": self.source_pose_mode,
            "source_action_schema_id": self.source_action_schema_id,
            "source_action_by_channel": dict(self.source_action_by_channel),
            "conditioned_action_by_channel": dict(
                self.conditioned_action_by_channel
            ),
            "decoded_absolute_action_hash": self.decoded_absolute_action_hash,
            "anchor_policy": self.anchor_policy,
            "anchor_value_by_component": dict(self.anchor_value_by_component),
            "target_generation": self.target_generation,
        }
        if include_hash:
            payload.update(
                {
                    "projection_id": self.projection_id,
                    "semantic_content_hash": self.semantic_content_hash,
                }
            )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PolicySourceActionProjection":
        payload = dict(value)
        expected_hash = payload.pop("semantic_content_hash", None)
        expected_id = payload.pop("projection_id", None)
        projection = cls(**payload)
        if expected_hash is not None and expected_hash != projection.semantic_content_hash:
            raise ValueError("Policy source projection hash does not match its payload.")
        if expected_id is not None and expected_id != projection.projection_id:
            raise ValueError("Policy source projection ID does not match its payload.")
        return projection


def policy_source_projection_from_metadata(
    metadata: Mapping[str, Any],
) -> PolicySourceActionProjection | None:
    value = metadata.get(POLICY_SOURCE_ACTION_PROJECTION_METADATA_KEY)
    if value is None:
        return None
    if isinstance(value, PolicySourceActionProjection):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("policy_source_action_projection metadata must be a mapping.")
    return PolicySourceActionProjection.from_dict(value)


__all__ = [
    "POLICY_SOURCE_ACTION_ABSOLUTE_TARGET",
    "POLICY_SOURCE_ACTION_OBSERVATION_RELATIVE_CHUNK",
    "POLICY_SOURCE_ACTION_PROJECTION_HASH_DOMAIN_V1",
    "POLICY_SOURCE_ACTION_PROJECTION_METADATA_KEY",
    "POLICY_SOURCE_ACTION_PROJECTION_SCHEMA_V1",
    "POLICY_SOURCE_ACTION_SEMANTICS",
    "POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA",
    "POLICY_SOURCE_ANCHOR_NONE",
    "POLICY_SOURCE_ANCHOR_OBSERVATION_TCP_FIXED_FOR_CHUNK",
    "POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET",
    "PolicySourceActionProjection",
    "policy_source_projection_from_metadata",
]
