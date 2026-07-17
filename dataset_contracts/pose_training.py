"""Typed provenance for the pose view used to train a policy.

The policy artifact declares v2.3 model semantics even when its source rows
came from a v2.2 dataset.  This contract keeps that compatibility case
distinct from a model trained on a native v2.3 canonical-desk dataset so a
deployment cannot silently apply a new desk transform to legacy raw state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .pose_frames import CANONICAL_DESK_FRAME, legacy_rm65_source_id
from .v22 import DEXTELEOP_DATASET_SCHEMA_V22, DatasetLayoutBindingError
from .v23 import DEXTELEOP_DATASET_SCHEMA_V23


POSE_TRAINING_DATA_METADATA_KEY = "pose_training_data_contract"
POSE_TRAINING_DATA_SCHEMA_V1 = "dexteleop_pose_training_data_v1"
POSE_TRAINING_ORIGIN_NATIVE_V23 = "native_v23_dataset"
POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY = "v22_raw_identity_compatibility"
POSE_TRAINING_ORIGINS = (
    POSE_TRAINING_ORIGIN_NATIVE_V23,
    POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY,
)

RUNTIME_POSE_BINDING_EXPLICIT_CANONICAL_DESK = (
    "explicit_runtime_raw_to_canonical_desk"
)
RUNTIME_POSE_BINDING_V22_IDENTITY = "same_source_identity_raw_to_desk"


@dataclass(frozen=True)
class PoseTrainingDataContract:
    """Describe where a v2.3 model-facing pose view came from.

    ``native_v23_dataset`` means the training reader used an explicit v2.3
    pose-frame contract and produced a real canonical-desk state view.

    ``v22_raw_identity_compatibility`` means legacy per-side RM65 raw poses
    were numerically preserved by the default identity compatibility view.
    Relative actions remain valid, but the absolute state distribution is
    tied to that same runtime source and identity mapping.
    """

    origin: str
    source_dataset_schema: str
    expected_runtime_source_id: str | None = None
    schema: str = POSE_TRAINING_DATA_SCHEMA_V1

    def __post_init__(self) -> None:
        if str(self.schema) != POSE_TRAINING_DATA_SCHEMA_V1:
            raise DatasetLayoutBindingError(
                f"Unsupported pose training data schema {self.schema!r}."
            )
        origin = str(self.origin).strip().lower()
        if origin not in POSE_TRAINING_ORIGINS:
            raise DatasetLayoutBindingError(
                f"pose training origin must be one of {POSE_TRAINING_ORIGINS}, "
                f"got {self.origin!r}."
            )
        source_schema = str(self.source_dataset_schema).strip()
        expected_source = (
            None
            if self.expected_runtime_source_id is None
            else str(self.expected_runtime_source_id).strip() or None
        )
        if origin == POSE_TRAINING_ORIGIN_NATIVE_V23:
            if source_schema != DEXTELEOP_DATASET_SCHEMA_V23:
                raise DatasetLayoutBindingError(
                    "native_v23_dataset pose training requires source dataset schema "
                    f"{DEXTELEOP_DATASET_SCHEMA_V23!r}."
                )
            if expected_source is not None:
                raise DatasetLayoutBindingError(
                    "native v2.3 pose training must not pin one runtime source_id; "
                    "the explicit canonical-desk runtime contract provides the binding."
                )
        else:
            if source_schema != DEXTELEOP_DATASET_SCHEMA_V22:
                raise DatasetLayoutBindingError(
                    "v22_raw_identity_compatibility requires source dataset schema "
                    f"{DEXTELEOP_DATASET_SCHEMA_V22!r}."
                )
            if expected_source is None:
                raise DatasetLayoutBindingError(
                    "v22 raw identity pose training requires expected_runtime_source_id."
                )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "source_dataset_schema", source_schema)
        object.__setattr__(self, "expected_runtime_source_id", expected_source)
        object.__setattr__(self, "schema", POSE_TRAINING_DATA_SCHEMA_V1)

    @classmethod
    def native_v23(cls) -> "PoseTrainingDataContract":
        return cls(
            origin=POSE_TRAINING_ORIGIN_NATIVE_V23,
            source_dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        )

    @classmethod
    def v22_raw_identity(
        cls,
        *,
        active_sides: Sequence[str] | None = None,
    ) -> "PoseTrainingDataContract":
        return cls(
            origin=POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY,
            source_dataset_schema=DEXTELEOP_DATASET_SCHEMA_V22,
            expected_runtime_source_id=legacy_rm65_source_id(active_sides),
        )

    @property
    def runtime_pose_binding(self) -> str:
        if self.origin == POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY:
            return RUNTIME_POSE_BINDING_V22_IDENTITY
        return RUNTIME_POSE_BINDING_EXPLICIT_CANONICAL_DESK

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "origin": self.origin,
            "source_dataset_schema": self.source_dataset_schema,
            "model_state_pose_frame": CANONICAL_DESK_FRAME,
            "runtime_pose_binding": self.runtime_pose_binding,
            "expected_runtime_source_id": self.expected_runtime_source_id,
            "relative_action_world_transform_invariant": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseTrainingDataContract":
        contract = cls(
            origin=str(payload.get("origin", "")),
            source_dataset_schema=str(payload.get("source_dataset_schema", "")),
            expected_runtime_source_id=(
                None
                if payload.get("expected_runtime_source_id") is None
                else str(payload["expected_runtime_source_id"])
            ),
            schema=str(payload.get("schema", "")),
        )
        expected_constants = {
            "model_state_pose_frame": CANONICAL_DESK_FRAME,
            "runtime_pose_binding": contract.runtime_pose_binding,
            "relative_action_world_transform_invariant": True,
        }
        for key, expected in expected_constants.items():
            if key in payload and payload[key] != expected:
                raise DatasetLayoutBindingError(
                    f"Pose training data {key} must be {expected!r}."
                )
        return contract


__all__ = [
    "POSE_TRAINING_DATA_METADATA_KEY",
    "POSE_TRAINING_DATA_SCHEMA_V1",
    "POSE_TRAINING_ORIGIN_NATIVE_V23",
    "POSE_TRAINING_ORIGIN_V22_RAW_IDENTITY",
    "POSE_TRAINING_ORIGINS",
    "RUNTIME_POSE_BINDING_EXPLICIT_CANONICAL_DESK",
    "RUNTIME_POSE_BINDING_V22_IDENTITY",
    "PoseTrainingDataContract",
]
