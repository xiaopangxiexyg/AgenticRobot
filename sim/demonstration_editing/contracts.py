"""Pure contracts for demonstration-conditioned simulation trajectories.

This package deliberately stops before policy inference, task reward, collision
admission, IK, or simulator execution.  It represents one already-selected
demonstration and one immutable high-level edit sample.  Downstream adapters
may render the resulting nominal references into their own typed component
actions, but must retain policy-sample and safety-admission identities
separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash


DEMONSTRATION_TRAJECTORY_SCHEMA_V1 = (
    "dexteleop.sim.demonstration_trajectory.v1"
)
DEMONSTRATION_EDIT_SCHEMA_V1 = "dexteleop.sim.demonstration_edit.v1"
EDITED_DEMONSTRATION_TRAJECTORY_SCHEMA_V1 = (
    "dexteleop.sim.edited_demonstration_trajectory.v1"
)

DEMONSTRATION_TRAJECTORY_HASH_DOMAIN_V1 = (
    "dexteleop.sim.demonstration_trajectory.semantic.v1"
)
DEMONSTRATION_EDIT_HASH_DOMAIN_V1 = (
    "dexteleop.sim.demonstration_edit.semantic.v1"
)
EDITED_DEMONSTRATION_TRAJECTORY_HASH_DOMAIN_V1 = (
    "dexteleop.sim.edited_demonstration_trajectory.semantic.v1"
)

POSE_EDIT_LEFT_MULTIPLY_RIGID_V1 = "left_multiply_rigid_se3_v1"
HAND_EDIT_FINAL_DELTA_RAMP_V1 = "final_target_delta_linear_ramp_v1"


def _required_text(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _optional_sha256(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    result = _required_text(value, name=name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest.")
    return result


def _required_sha256(value: object, *, name: str) -> str:
    result = _optional_sha256(value, name=name)
    if result is None:
        raise ValueError(f"{name} must be provided.")
    return result


def _index(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _metadata(values: Mapping[str, object], *, name: str) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for raw_key, raw_value in sorted(values.items()):
        key = _required_text(raw_key, name=f"{name} key")
        if key in result:
            raise ValueError(f"{name} keys must be unique.")
        result[key] = _required_text(raw_value, name=f"{name}[{key!r}]")
    return MappingProxyType(result)


def _text_tuple(values: object, *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings.")
    result = tuple(
        _required_text(value, name=f"{name} item") for value in values  # type: ignore[union-attr]
    )
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _finite_vector(value: object, *, name: str, size: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if size is not None and result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result.setflags(write=False)
    return result


def _canonical_pose_array(value: object, *, name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64).copy()
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] < 2:
        raise ValueError(f"{name} must have shape (N, 7) with N >= 2.")
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{name} must contain finite values.")
    quaternion = poses[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1)
    if np.any(norms <= 1e-12) or not np.allclose(norms, 1.0, rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} quaternions must be unit xyzw quaternions.")
    quaternion /= norms[:, None]
    # q and -q represent the same rotation.  Canonicalise signs so equivalent
    # imported trajectories receive one stable semantic identity.
    first = quaternion[0]
    if first[int(np.argmax(np.abs(first)))] < 0.0:
        quaternion[0] *= -1.0
    for index in range(1, quaternion.shape[0]):
        if float(np.dot(quaternion[index - 1], quaternion[index])) < 0.0:
            quaternion[index] *= -1.0
    poses[:, 3:7] = quaternion
    poses.setflags(write=False)
    return poses


def _hand_array(value: object, *, name: str, step_count: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[0] != step_count or result.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape ({step_count}, H) with H >= 1."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values.")
    result.setflags(write=False)
    return result


def _trajectory_arrays(
    *,
    timestamps_sec: object,
    tool_pose_xyzquat_xyzw: object,
    hand_targets: object,
    hand_edit_start_index: object,
    hand_edit_end_index: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    timestamps = _finite_vector(timestamps_sec, name="timestamps_sec")
    if timestamps.shape[0] < 2:
        raise ValueError("timestamps_sec must contain at least two samples.")
    if not math.isclose(float(timestamps[0]), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("timestamps_sec must start at zero.")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("timestamps_sec must be strictly increasing.")
    poses = _canonical_pose_array(
        tool_pose_xyzquat_xyzw,
        name="tool_pose_xyzquat_xyzw",
    )
    if poses.shape[0] != timestamps.shape[0]:
        raise ValueError("tool poses and timestamps must have the same length.")
    hand = _hand_array(
        hand_targets,
        name="hand_targets",
        step_count=timestamps.shape[0],
    )
    start = _index(hand_edit_start_index, name="hand_edit_start_index")
    end = _index(hand_edit_end_index, name="hand_edit_end_index")
    if start > end or end >= timestamps.shape[0]:
        raise ValueError(
            "hand edit indices must satisfy 0 <= start <= end < step_count."
        )
    return timestamps, poses, hand, start, end


@dataclass(frozen=True)
class DemonstrationTrajectory:
    """One successful nominal trajectory expressed in an explicit frame.

    ``hand_edit_start_index`` and ``hand_edit_end_index`` define only the
    generic interval over which a final hand-target delta is blended in.  They
    do not imply a task phase, success predicate, or contact state.
    """

    trajectory_id: str
    reference_frame: str
    controlled_frame: str
    hand_component_id: str
    hand_target_representation: str
    hand_target_names: tuple[str, ...]
    hand_target_units: tuple[str, ...]
    timestamps_sec: np.ndarray
    tool_pose_xyzquat_xyzw: np.ndarray
    hand_targets: np.ndarray
    hand_edit_start_index: int
    hand_edit_end_index: int
    source_kind: str
    source_id: str
    source_content_hash: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = DEMONSTRATION_TRAJECTORY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != DEMONSTRATION_TRAJECTORY_SCHEMA_V1:
            raise ValueError(
                f"Unsupported demonstration trajectory schema: {self.schema_version!r}."
            )
        for name in (
            "trajectory_id",
            "reference_frame",
            "controlled_frame",
            "hand_component_id",
            "hand_target_representation",
            "source_kind",
            "source_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_content_hash",
            _optional_sha256(self.source_content_hash, name="source_content_hash"),
        )
        arrays = _trajectory_arrays(
            timestamps_sec=self.timestamps_sec,
            tool_pose_xyzquat_xyzw=self.tool_pose_xyzquat_xyzw,
            hand_targets=self.hand_targets,
            hand_edit_start_index=self.hand_edit_start_index,
            hand_edit_end_index=self.hand_edit_end_index,
        )
        for name, value in zip(
            (
                "timestamps_sec",
                "tool_pose_xyzquat_xyzw",
                "hand_targets",
                "hand_edit_start_index",
                "hand_edit_end_index",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        names = _text_tuple(self.hand_target_names, name="hand_target_names")
        units = _text_tuple(self.hand_target_units, name="hand_target_units")
        if len(names) != self.hand_dim or len(units) != self.hand_dim:
            raise ValueError(
                "hand target names and units must match the hand target dimension."
            )
        if len(names) != len(set(names)):
            raise ValueError("hand_target_names must be unique.")
        object.__setattr__(self, "hand_target_names", names)
        object.__setattr__(self, "hand_target_units", units)
        object.__setattr__(self, "metadata", _metadata(self.metadata, name="metadata"))

    @property
    def step_count(self) -> int:
        return int(self.timestamps_sec.shape[0])

    @property
    def hand_dim(self) -> int:
        return int(self.hand_targets.shape[1])

    @property
    def duration_sec(self) -> float:
        return float(self.timestamps_sec[-1])

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "reference_frame": self.reference_frame,
            "controlled_frame": self.controlled_frame,
            "hand_component_id": self.hand_component_id,
            "hand_target_representation": self.hand_target_representation,
            "hand_target_names": self.hand_target_names,
            "hand_target_units": self.hand_target_units,
            "timestamps_sec": self.timestamps_sec,
            "tool_pose_xyzquat_xyzw": self.tool_pose_xyzquat_xyzw,
            "hand_targets": self.hand_targets,
            "hand_edit_start_index": self.hand_edit_start_index,
            "hand_edit_end_index": self.hand_edit_end_index,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_content_hash": self.source_content_hash,
            "metadata": dict(self.metadata),
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.semantic_payload(),
            domain=DEMONSTRATION_TRAJECTORY_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        payload["timestamps_sec"] = self.timestamps_sec.tolist()
        payload["tool_pose_xyzquat_xyzw"] = self.tool_pose_xyzquat_xyzw.tolist()
        payload["hand_targets"] = self.hand_targets.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DemonstrationTrajectory":
        if not isinstance(payload, Mapping):
            raise TypeError("demonstration trajectory payload must be a mapping.")
        return cls(
            trajectory_id=payload.get("trajectory_id", ""),
            reference_frame=payload.get("reference_frame", ""),
            controlled_frame=payload.get("controlled_frame", ""),
            hand_component_id=payload.get("hand_component_id", ""),
            hand_target_representation=payload.get("hand_target_representation", ""),
            hand_target_names=payload.get("hand_target_names", ()),
            hand_target_units=payload.get("hand_target_units", ()),
            timestamps_sec=payload.get("timestamps_sec", ()),
            tool_pose_xyzquat_xyzw=payload.get("tool_pose_xyzquat_xyzw", ()),
            hand_targets=payload.get("hand_targets", ()),
            hand_edit_start_index=payload.get("hand_edit_start_index", -1),
            hand_edit_end_index=payload.get("hand_edit_end_index", -1),
            source_kind=payload.get("source_kind", ""),
            source_id=payload.get("source_id", ""),
            source_content_hash=payload.get("source_content_hash"),
            metadata=payload.get("metadata", {}),
            schema_version=payload.get("schema_version", ""),
        )


@dataclass(frozen=True)
class DemonstrationEdit:
    """One immutable high-level edit sample.

    ``sample_id`` links this transformation to policy storage when the edit is
    sampled by PPO.  Log probability and optimizer state intentionally remain
    outside this simulator transformation contract.
    """

    edit_id: str
    producer_id: str
    sample_id: str
    translation_m: np.ndarray
    rotation_rotvec_rad: np.ndarray
    hand_final_delta: np.ndarray
    pose_edit_mode: str = POSE_EDIT_LEFT_MULTIPLY_RIGID_V1
    hand_edit_mode: str = HAND_EDIT_FINAL_DELTA_RAMP_V1
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = DEMONSTRATION_EDIT_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != DEMONSTRATION_EDIT_SCHEMA_V1:
            raise ValueError(
                f"Unsupported demonstration edit schema: {self.schema_version!r}."
            )
        for name in ("edit_id", "producer_id", "sample_id"):
            object.__setattr__(self, name, _required_text(getattr(self, name), name=name))
        if self.pose_edit_mode != POSE_EDIT_LEFT_MULTIPLY_RIGID_V1:
            raise ValueError(f"Unsupported pose_edit_mode: {self.pose_edit_mode!r}.")
        if self.hand_edit_mode != HAND_EDIT_FINAL_DELTA_RAMP_V1:
            raise ValueError(f"Unsupported hand_edit_mode: {self.hand_edit_mode!r}.")
        object.__setattr__(
            self,
            "translation_m",
            _finite_vector(self.translation_m, name="translation_m", size=3),
        )
        object.__setattr__(
            self,
            "rotation_rotvec_rad",
            _finite_vector(
                self.rotation_rotvec_rad,
                name="rotation_rotvec_rad",
                size=3,
            ),
        )
        object.__setattr__(
            self,
            "hand_final_delta",
            _finite_vector(self.hand_final_delta, name="hand_final_delta"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata, name="metadata"))

    @property
    def hand_dim(self) -> int:
        return int(self.hand_final_delta.shape[0])

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "edit_id": self.edit_id,
            "producer_id": self.producer_id,
            "sample_id": self.sample_id,
            "translation_m": self.translation_m,
            "rotation_rotvec_rad": self.rotation_rotvec_rad,
            "hand_final_delta": self.hand_final_delta,
            "pose_edit_mode": self.pose_edit_mode,
            "hand_edit_mode": self.hand_edit_mode,
            "metadata": dict(self.metadata),
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.semantic_payload(),
            domain=DEMONSTRATION_EDIT_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        payload["translation_m"] = self.translation_m.tolist()
        payload["rotation_rotvec_rad"] = self.rotation_rotvec_rad.tolist()
        payload["hand_final_delta"] = self.hand_final_delta.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DemonstrationEdit":
        if not isinstance(payload, Mapping):
            raise TypeError("demonstration edit payload must be a mapping.")
        return cls(
            edit_id=payload.get("edit_id", ""),
            producer_id=payload.get("producer_id", ""),
            sample_id=payload.get("sample_id", ""),
            translation_m=payload.get("translation_m", ()),
            rotation_rotvec_rad=payload.get("rotation_rotvec_rad", ()),
            hand_final_delta=payload.get("hand_final_delta", ()),
            pose_edit_mode=payload.get("pose_edit_mode", ""),
            hand_edit_mode=payload.get("hand_edit_mode", ""),
            metadata=payload.get("metadata", {}),
            schema_version=payload.get("schema_version", ""),
        )


@dataclass(frozen=True)
class EditedDemonstrationTrajectory:
    """Nominal edited references with explicit source/edit lineage.

    This is not an admitted or executed trajectory.  Safety projection and
    measured execution must create their own downstream records.
    """

    trajectory_id: str
    reference_frame: str
    controlled_frame: str
    hand_component_id: str
    hand_target_representation: str
    hand_target_names: tuple[str, ...]
    hand_target_units: tuple[str, ...]
    timestamps_sec: np.ndarray
    tool_pose_xyzquat_xyzw: np.ndarray
    hand_targets: np.ndarray
    hand_edit_start_index: int
    hand_edit_end_index: int
    source_trajectory_hash: str
    edit_hash: str
    edit_sample_id: str
    schema_version: str = EDITED_DEMONSTRATION_TRAJECTORY_SCHEMA_V1

    def __post_init__(self) -> None:
        if self.schema_version != EDITED_DEMONSTRATION_TRAJECTORY_SCHEMA_V1:
            raise ValueError(
                "Unsupported edited demonstration trajectory schema: "
                f"{self.schema_version!r}."
            )
        for name in (
            "trajectory_id",
            "reference_frame",
            "controlled_frame",
            "hand_component_id",
            "hand_target_representation",
            "edit_sample_id",
        ):
            object.__setattr__(self, name, _required_text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "source_trajectory_hash",
            _required_sha256(
                self.source_trajectory_hash,
                name="source_trajectory_hash",
            ),
        )
        object.__setattr__(
            self,
            "edit_hash",
            _required_sha256(self.edit_hash, name="edit_hash"),
        )
        arrays = _trajectory_arrays(
            timestamps_sec=self.timestamps_sec,
            tool_pose_xyzquat_xyzw=self.tool_pose_xyzquat_xyzw,
            hand_targets=self.hand_targets,
            hand_edit_start_index=self.hand_edit_start_index,
            hand_edit_end_index=self.hand_edit_end_index,
        )
        for name, value in zip(
            (
                "timestamps_sec",
                "tool_pose_xyzquat_xyzw",
                "hand_targets",
                "hand_edit_start_index",
                "hand_edit_end_index",
            ),
            arrays,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        names = _text_tuple(self.hand_target_names, name="hand_target_names")
        units = _text_tuple(self.hand_target_units, name="hand_target_units")
        if len(names) != self.hand_dim or len(units) != self.hand_dim:
            raise ValueError(
                "hand target names and units must match the hand target dimension."
            )
        if len(names) != len(set(names)):
            raise ValueError("hand_target_names must be unique.")
        object.__setattr__(self, "hand_target_names", names)
        object.__setattr__(self, "hand_target_units", units)

    @property
    def step_count(self) -> int:
        return int(self.timestamps_sec.shape[0])

    @property
    def hand_dim(self) -> int:
        return int(self.hand_targets.shape[1])

    @property
    def duration_sec(self) -> float:
        return float(self.timestamps_sec[-1])

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "reference_frame": self.reference_frame,
            "controlled_frame": self.controlled_frame,
            "hand_component_id": self.hand_component_id,
            "hand_target_representation": self.hand_target_representation,
            "hand_target_names": self.hand_target_names,
            "hand_target_units": self.hand_target_units,
            "timestamps_sec": self.timestamps_sec,
            "tool_pose_xyzquat_xyzw": self.tool_pose_xyzquat_xyzw,
            "hand_targets": self.hand_targets,
            "hand_edit_start_index": self.hand_edit_start_index,
            "hand_edit_end_index": self.hand_edit_end_index,
            "source_trajectory_hash": self.source_trajectory_hash,
            "edit_hash": self.edit_hash,
            "edit_sample_id": self.edit_sample_id,
        }

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.semantic_payload(),
            domain=EDITED_DEMONSTRATION_TRAJECTORY_HASH_DOMAIN_V1,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        payload["timestamps_sec"] = self.timestamps_sec.tolist()
        payload["tool_pose_xyzquat_xyzw"] = self.tool_pose_xyzquat_xyzw.tolist()
        payload["hand_targets"] = self.hand_targets.tolist()
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "EditedDemonstrationTrajectory":
        if not isinstance(payload, Mapping):
            raise TypeError("edited demonstration payload must be a mapping.")
        return cls(
            trajectory_id=payload.get("trajectory_id", ""),
            reference_frame=payload.get("reference_frame", ""),
            controlled_frame=payload.get("controlled_frame", ""),
            hand_component_id=payload.get("hand_component_id", ""),
            hand_target_representation=payload.get("hand_target_representation", ""),
            hand_target_names=payload.get("hand_target_names", ()),
            hand_target_units=payload.get("hand_target_units", ()),
            timestamps_sec=payload.get("timestamps_sec", ()),
            tool_pose_xyzquat_xyzw=payload.get("tool_pose_xyzquat_xyzw", ()),
            hand_targets=payload.get("hand_targets", ()),
            hand_edit_start_index=payload.get("hand_edit_start_index", -1),
            hand_edit_end_index=payload.get("hand_edit_end_index", -1),
            source_trajectory_hash=payload.get("source_trajectory_hash", ""),
            edit_hash=payload.get("edit_hash", ""),
            edit_sample_id=payload.get("edit_sample_id", ""),
            schema_version=payload.get("schema_version", ""),
        )


__all__ = [
    "DEMONSTRATION_EDIT_SCHEMA_V1",
    "DEMONSTRATION_TRAJECTORY_SCHEMA_V1",
    "EDITED_DEMONSTRATION_TRAJECTORY_SCHEMA_V1",
    "HAND_EDIT_FINAL_DELTA_RAMP_V1",
    "POSE_EDIT_LEFT_MULTIPLY_RIGID_V1",
    "DemonstrationEdit",
    "DemonstrationTrajectory",
    "EditedDemonstrationTrajectory",
]
