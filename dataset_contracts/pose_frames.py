"""Dataset-level absolute pose frame and source contracts.

The model-facing pose frame is deliberately separate from the frame in which a
device reports its native absolute pose.  A dataset may store either native
(``raw_absolute``) or canonical desktop (``desk_absolute``) values; readers
must use this contract instead of guessing from a legacy column name.

This module is hardware independent.  It only validates rigid transforms and
serializes the source/frame metadata required by collection, replay and
deployment adapters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation


POSE_FRAME_CONTRACT_SCHEMA_V1 = "dexteleop_pose_frame_contract_v1"
POSE_SOURCE_CONTRACT_SCHEMA_V1 = "dexteleop_pose_source_contract_v1"
POSE_STORAGE_FRAME_RAW = "raw_absolute"
POSE_STORAGE_FRAME_DESK = "desk_absolute"
POSE_STORAGE_FRAMES = (POSE_STORAGE_FRAME_RAW, POSE_STORAGE_FRAME_DESK)
CANONICAL_DESK_FRAME = "desk"
# Backward-compatible public name.  This is the dataset canonical desk, not
# teleop_runtime's historical Vive retarget frame.
DESK_ABSOLUTE_FRAME = CANONICAL_DESK_FRAME
POSE_FRAME_CONTRACT_METADATA_KEY = "pose_frame_contract"
POSE_SOURCE_CONTRACT_METADATA_KEY = "pose_source_contract"

SIDES = ("left", "right")


class PoseFrameContractError(ValueError):
    """Raised when source/frame metadata is incomplete or inconsistent."""


def _pose_vector_to_matrix(value: Any, *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise PoseFrameContractError(f"{name} must be a finite xyzquat vector.") from exc
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise PoseFrameContractError(f"{name} must have seven finite xyzquat values.")
    quaternion = vector[3:7]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise PoseFrameContractError(f"{name} quaternion norm is zero.")
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = vector[:3]
    matrix[:3, :3] = _Rotation.from_quat(quaternion / norm).as_matrix()
    return matrix


def _matrix_to_pose_vector(value: Any, *, name: str) -> np.ndarray:
    matrix = _finite_matrix(value, name=name)
    quaternion = _Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12 or not np.all(np.isfinite(quaternion)):
        raise PoseFrameContractError(f"{name} produced an invalid quaternion.")
    quaternion /= norm
    if quaternion[3] < 0.0 or (
        abs(quaternion[3]) <= 1e-12
        and next((float(item) for item in quaternion[:3] if abs(item) > 1e-12), 0.0) < 0.0
    ):
        quaternion = -quaternion
    return np.concatenate((matrix[:3, 3], quaternion)).astype(np.float32, copy=False)


def transform_pose_vector(
    pose_xyzquat: Any,
    transform: Any,
    *,
    name: str = "pose",
) -> np.ndarray:
    """Apply a world-frame rigid transform to one xyzquat pose vector."""

    source = _pose_vector_to_matrix(pose_xyzquat, name=name)
    target = _finite_matrix(transform, name="pose frame transform") @ source
    return _matrix_to_pose_vector(target, name=name)


def _normalize_sides(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return SIDES
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    invalid = sorted(set(normalized) - set(SIDES))
    if invalid:
        raise PoseFrameContractError(f"Unsupported pose source side(s): {invalid}.")
    if not normalized:
        raise PoseFrameContractError("A pose source must declare at least one active side.")
    return tuple(side for side in SIDES if side in normalized)


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PoseFrameContractError(f"{name} must be a numeric 4x4 matrix.") from exc
    if matrix.shape != (4, 4):
        raise PoseFrameContractError(f"{name} must have shape (4, 4), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise PoseFrameContractError(f"{name} contains non-finite values.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise PoseFrameContractError(f"{name} last row must be [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise PoseFrameContractError(f"{name} rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise PoseFrameContractError(f"{name} rotation determinant must be +1.")
    return matrix.astype(float, copy=True)


def _identity_matrix() -> np.ndarray:
    return np.eye(4, dtype=float)


def _validate_matrix_alias(
    payload: Mapping[str, Any], *, canonical_key: str, legacy_key: str
) -> None:
    if canonical_key not in payload or legacy_key not in payload:
        return
    canonical = _finite_matrix(payload[canonical_key], name=canonical_key)
    legacy = _finite_matrix(payload[legacy_key], name=legacy_key)
    if not np.allclose(canonical, legacy, atol=1e-9, rtol=0.0):
        raise PoseFrameContractError(
            f"Conflicting pose contract aliases {canonical_key!r} and {legacy_key!r}."
        )


def _validate_side_matrix_alias(
    payload: Mapping[str, Any], *, canonical_key: str, legacy_key: str
) -> None:
    if canonical_key not in payload or legacy_key not in payload:
        return
    canonical = dict(payload[canonical_key])
    legacy = dict(payload[legacy_key])
    if set(canonical) != set(legacy):
        raise PoseFrameContractError(
            f"Conflicting pose contract aliases {canonical_key!r} and {legacy_key!r}."
        )
    for side in canonical:
        canonical_matrix = _finite_matrix(
            canonical[side], name=f"{canonical_key}.{side}"
        )
        legacy_matrix = _finite_matrix(legacy[side], name=f"{legacy_key}.{side}")
        if not np.allclose(canonical_matrix, legacy_matrix, atol=1e-9, rtol=0.0):
            raise PoseFrameContractError(
                f"Conflicting pose contract aliases {canonical_key!r} and {legacy_key!r} "
                f"for side {side!r}."
            )


def normalize_pose_storage_frame(value: str | None) -> str:
    normalized = str(value or POSE_STORAGE_FRAME_DESK).strip().lower()
    aliases = {
        "raw": POSE_STORAGE_FRAME_RAW,
        "raw_absolute": POSE_STORAGE_FRAME_RAW,
        "native": POSE_STORAGE_FRAME_RAW,
        "desk": POSE_STORAGE_FRAME_DESK,
        "desktop": POSE_STORAGE_FRAME_DESK,
        "desk_absolute": POSE_STORAGE_FRAME_DESK,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POSE_STORAGE_FRAMES:
        raise PoseFrameContractError(
            f"pose_storage_frame must be one of {POSE_STORAGE_FRAMES}, got {value!r}."
        )
    return normalized


def legacy_rm65_source_id(active_sides: Sequence[str] | None = None) -> str:
    """Return the stable source id used for pre-v2.3 RM65 datasets."""

    sides = _normalize_sides(active_sides)
    if sides == SIDES:
        return "rm65_real_dual_arm_v1"
    if sides == ("left",):
        return "rm65_real_single_left_v1"
    if sides == ("right",):
        return "rm65_real_single_right_v1"
    raise PoseFrameContractError(f"Unsupported legacy RM65 topology for sides={sides!r}.")


@dataclass(frozen=True)
class PoseSourceProfile:
    """Immutable description of one native pose source and its rig topology."""

    source_id: str
    source_family: str
    source_description: str
    topology: str
    active_sides: tuple[str, ...] = SIDES
    raw_absolute_frame_by_side: Mapping[str, str] = field(default_factory=dict)
    raw_to_desk_transform: np.ndarray = field(default_factory=_identity_matrix)
    raw_to_desk_transform_by_side: Mapping[str, np.ndarray] = field(default_factory=dict)
    desk_frame: str = DESK_ABSOLUTE_FRAME
    profile_version: str = "1"
    config_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        family = str(self.source_family).strip()
        description = str(self.source_description).strip()
        topology = str(self.topology).strip()
        if not source_id or not family or not description or not topology:
            raise PoseFrameContractError(
                "source_id, source_family, source_description and topology are required."
            )
        sides = _normalize_sides(self.active_sides)
        raw_frames = {str(side): str(frame).strip() for side, frame in dict(self.raw_absolute_frame_by_side).items()}
        invalid_frame_sides = set(raw_frames) - set(sides)
        if invalid_frame_sides:
            raise PoseFrameContractError(
                f"raw_absolute_frame_by_side contains inactive side(s): {sorted(invalid_frame_sides)}."
            )
        for side in sides:
            raw_frames.setdefault(side, f"{side}_raw_absolute")
            if not raw_frames[side]:
                raise PoseFrameContractError(f"raw absolute frame for {side} must be non-empty.")
        shared = _finite_matrix(self.raw_to_desk_transform, name="raw_to_desk_transform")
        side_transforms: dict[str, np.ndarray] = {}
        for side, transform in dict(self.raw_to_desk_transform_by_side).items():
            normalized_side = str(side).strip().lower()
            if normalized_side not in sides:
                raise PoseFrameContractError(
                    f"raw_to_desk_transform_by_side contains inactive side {side!r}."
                )
            side_transforms[normalized_side] = _finite_matrix(
                transform,
                name=f"raw_to_desk_transform_by_side.{normalized_side}",
            )
        desk_frame = str(self.desk_frame).strip()
        if not desk_frame:
            raise PoseFrameContractError("desk_frame must be non-empty.")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_family", family)
        object.__setattr__(self, "source_description", description)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "raw_absolute_frame_by_side", raw_frames)
        object.__setattr__(self, "raw_to_desk_transform", shared)
        object.__setattr__(self, "raw_to_desk_transform_by_side", side_transforms)
        object.__setattr__(self, "desk_frame", desk_frame)
        object.__setattr__(self, "profile_version", str(self.profile_version).strip() or "1")
        object.__setattr__(self, "config_hash", None if self.config_hash is None else str(self.config_hash).strip() or None)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def effective_raw_to_desk_transform(self, side: str) -> np.ndarray:
        normalized_side = str(side).strip().lower()
        if normalized_side not in self.active_sides:
            raise PoseFrameContractError(
                f"Source {self.source_id!r} does not declare active side {side!r}."
            )
        transform = self.raw_to_desk_transform_by_side.get(normalized_side, self.raw_to_desk_transform)
        return np.asarray(transform, dtype=float).copy()

    @property
    def canonical_desk_frame(self) -> str:
        """Name of the dataset-wide canonical absolute frame.

        ``desk_frame`` remains the constructor/serialized v1 compatibility
        name.  New code should use this property so it cannot be confused with
        the historical Vive retarget frame.
        """

        return self.desk_frame

    def effective_raw_to_canonical_desk_transform(self, side: str) -> np.ndarray:
        """Return ``T_canonical_desk_raw`` for one source side."""

        return self.effective_raw_to_desk_transform(side)

    def raw_pose_vector_to_canonical_desk(
        self, pose_xyzquat: Any, *, side: str
    ) -> np.ndarray:
        """Convert a source-native absolute xyzquat pose to canonical desk."""

        return transform_pose_vector(
            pose_xyzquat,
            self.effective_raw_to_canonical_desk_transform(side),
            name=f"{side}.raw_pose",
        )

    def raw_frame(self, side: str) -> str:
        normalized_side = str(side).strip().lower()
        if normalized_side not in self.active_sides:
            raise PoseFrameContractError(
                f"Source {self.source_id!r} does not declare active side {side!r}."
            )
        return str(self.raw_absolute_frame_by_side[normalized_side])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POSE_SOURCE_CONTRACT_SCHEMA_V1,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_description": self.source_description,
            "topology": self.topology,
            "active_sides": list(self.active_sides),
            "raw_absolute_frame_by_side": dict(self.raw_absolute_frame_by_side),
            "raw_to_desk_transform": self.raw_to_desk_transform.tolist(),
            "raw_to_desk_transform_by_side": {
                side: transform.tolist()
                for side, transform in self.raw_to_desk_transform_by_side.items()
            },
            "desk_frame": self.desk_frame,
            "raw_to_canonical_desk_transform": self.raw_to_desk_transform.tolist(),
            "raw_to_canonical_desk_transform_by_side": {
                side: transform.tolist()
                for side, transform in self.raw_to_desk_transform_by_side.items()
            },
            "canonical_desk_frame": self.canonical_desk_frame,
            "profile_version": self.profile_version,
            "config_hash": self.config_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseSourceProfile":
        schema = str(payload.get("schema", POSE_SOURCE_CONTRACT_SCHEMA_V1))
        if schema != POSE_SOURCE_CONTRACT_SCHEMA_V1:
            raise PoseFrameContractError(f"Unsupported pose source contract schema {schema!r}.")
        if (
            "canonical_desk_frame" in payload
            and "desk_frame" in payload
            and str(payload["canonical_desk_frame"]) != str(payload["desk_frame"])
        ):
            raise PoseFrameContractError(
                "Conflicting pose contract aliases 'canonical_desk_frame' and 'desk_frame'."
            )
        _validate_matrix_alias(
            payload,
            canonical_key="raw_to_canonical_desk_transform",
            legacy_key="raw_to_desk_transform",
        )
        _validate_side_matrix_alias(
            payload,
            canonical_key="raw_to_canonical_desk_transform_by_side",
            legacy_key="raw_to_desk_transform_by_side",
        )
        return cls(
            source_id=str(payload["source_id"]),
            source_family=str(payload["source_family"]),
            source_description=str(payload.get("source_description", payload["source_id"])),
            topology=str(payload.get("topology", "unknown")),
            active_sides=tuple(payload.get("active_sides", SIDES)),
            raw_absolute_frame_by_side=dict(payload.get("raw_absolute_frame_by_side", {})),
            raw_to_desk_transform=payload.get(
                "raw_to_canonical_desk_transform",
                payload.get("raw_to_desk_transform", _identity_matrix()),
            ),
            raw_to_desk_transform_by_side=dict(
                payload.get(
                    "raw_to_canonical_desk_transform_by_side",
                    payload.get("raw_to_desk_transform_by_side", {}),
                )
            ),
            desk_frame=str(
                payload.get(
                    "canonical_desk_frame",
                    payload.get("desk_frame", DESK_ABSOLUTE_FRAME),
                )
            ),
            profile_version=str(payload.get("profile_version", "1")),
            config_hash=payload.get("config_hash"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PoseFrameContract:
    """Storage-frame contract for one dataset or episode."""

    source: PoseSourceProfile
    pose_storage_frame: str = POSE_STORAGE_FRAME_DESK
    canonical_absolute_frame: str = DESK_ABSOLUTE_FRAME
    schema: str = POSE_FRAME_CONTRACT_SCHEMA_V1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.schema) != POSE_FRAME_CONTRACT_SCHEMA_V1:
            raise PoseFrameContractError(f"Unsupported pose frame contract schema {self.schema!r}.")
        storage = normalize_pose_storage_frame(self.pose_storage_frame)
        canonical = str(self.canonical_absolute_frame).strip()
        if not canonical:
            raise PoseFrameContractError("canonical_absolute_frame must be non-empty.")
        if canonical != self.source.canonical_desk_frame:
            raise PoseFrameContractError(
                "canonical_absolute_frame disagrees with source canonical_desk_frame: "
                f"{canonical!r} != {self.source.canonical_desk_frame!r}."
            )
        object.__setattr__(self, "pose_storage_frame", storage)
        object.__setattr__(self, "canonical_absolute_frame", canonical)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def contract_hash(self) -> str:
        """Canonical hash for dataset/artifact provenance binding."""

        canonical = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    @property
    def is_raw_storage(self) -> bool:
        return self.pose_storage_frame == POSE_STORAGE_FRAME_RAW

    @classmethod
    def v23_default_desk(cls, source: PoseSourceProfile) -> "PoseFrameContract":
        return cls(source=source, pose_storage_frame=POSE_STORAGE_FRAME_DESK)

    @classmethod
    def v23_forced_raw(cls, source: PoseSourceProfile) -> "PoseFrameContract":
        return cls(source=source, pose_storage_frame=POSE_STORAGE_FRAME_RAW)

    @classmethod
    def legacy_v22_raw(
        cls,
        *,
        active_sides: Sequence[str] | None = None,
        raw_to_desk_transform: Any | None = None,
        raw_to_desk_transform_by_side: Mapping[str, Any] | None = None,
    ) -> "PoseFrameContract":
        sides = _normalize_sides(active_sides)
        source_id = legacy_rm65_source_id(sides)
        topology = "dual_arm" if sides == SIDES else "single_arm"
        source = PoseSourceProfile(
            source_id=source_id,
            source_family="real_robot",
            source_description="Legacy RM65 real hardware dataset",
            topology=topology,
            active_sides=sides,
            raw_absolute_frame_by_side={side: f"{side}_rm65_base" for side in sides},
            raw_to_desk_transform=(
                _identity_matrix() if raw_to_desk_transform is None else raw_to_desk_transform
            ),
            raw_to_desk_transform_by_side=(
                {} if raw_to_desk_transform_by_side is None else raw_to_desk_transform_by_side
            ),
            metadata={"compatibility_source": "v2.2_or_earlier"},
        )
        return cls(source=source, pose_storage_frame=POSE_STORAGE_FRAME_RAW)

    def storage_pose_to_desk(self, pose: Any, *, side: str) -> np.ndarray:
        """Compatibility alias for :meth:`storage_pose_to_canonical_desk`."""

        return self.storage_pose_to_canonical_desk(pose, side=side)

    def storage_pose_to_canonical_desk(self, pose: Any, *, side: str) -> np.ndarray:
        """Convert one stored absolute pose matrix into canonical desk."""

        matrix = _finite_matrix(pose, name="stored pose")
        if self.pose_storage_frame == POSE_STORAGE_FRAME_DESK:
            return matrix
        return self.source.effective_raw_to_canonical_desk_transform(side) @ matrix

    def storage_pose_vector_to_desk(self, pose_xyzquat: Any, *, side: str) -> np.ndarray:
        """Compatibility alias for the explicitly named canonical conversion."""

        return self.storage_pose_vector_to_canonical_desk(pose_xyzquat, side=side)

    def storage_pose_vector_to_canonical_desk(
        self, pose_xyzquat: Any, *, side: str
    ) -> np.ndarray:
        """Convert one stored absolute xyzquat pose into canonical desk."""

        if self.pose_storage_frame == POSE_STORAGE_FRAME_DESK:
            return _matrix_to_pose_vector(
                _pose_vector_to_matrix(pose_xyzquat, name=f"{side}.stored_pose"),
                name=f"{side}.desk_pose",
            )
        return transform_pose_vector(
            pose_xyzquat,
            self.source.effective_raw_to_canonical_desk_transform(side),
            name=f"{side}.stored_pose",
        )

    def desk_pose_vector_to_storage(self, pose_xyzquat: Any, *, side: str) -> np.ndarray:
        """Compatibility alias for the explicitly named canonical conversion."""

        return self.canonical_desk_pose_vector_to_storage(pose_xyzquat, side=side)

    def canonical_desk_pose_vector_to_storage(
        self, pose_xyzquat: Any, *, side: str
    ) -> np.ndarray:
        """Convert one canonical desk xyzquat pose into the storage frame."""

        if self.pose_storage_frame == POSE_STORAGE_FRAME_DESK:
            return _matrix_to_pose_vector(
                _pose_vector_to_matrix(pose_xyzquat, name=f"{side}.desk_pose"),
                name=f"{side}.stored_pose",
            )
        return transform_pose_vector(
            pose_xyzquat,
            self._desk_to_raw_transform(side),
            name=f"{side}.desk_pose",
        )

    def _desk_to_raw_transform(self, side: str) -> np.ndarray:
        transform = self.source.effective_raw_to_desk_transform(side)
        inverse = np.eye(4, dtype=float)
        rotation = transform[:3, :3]
        inverse[:3, :3] = rotation.T
        inverse[:3, 3] = -rotation.T @ transform[:3, 3]
        return inverse

    def desk_pose_to_storage(self, pose: Any, *, side: str) -> np.ndarray:
        """Compatibility alias for :meth:`canonical_desk_pose_to_storage`."""

        return self.canonical_desk_pose_to_storage(pose, side=side)

    def canonical_desk_pose_to_storage(self, pose: Any, *, side: str) -> np.ndarray:
        """Convert one canonical desk pose matrix into the storage frame."""

        matrix = _finite_matrix(pose, name="desk pose")
        if self.pose_storage_frame == POSE_STORAGE_FRAME_DESK:
            return matrix
        transform = self.source.effective_raw_to_desk_transform(side)
        rotation = transform[:3, :3].T
        inverse = np.eye(4, dtype=float)
        inverse[:3, :3] = rotation
        inverse[:3, 3] = -rotation @ transform[:3, 3]
        return inverse @ matrix

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pose_storage_frame": self.pose_storage_frame,
            "canonical_absolute_frame": self.canonical_absolute_frame,
            "source": self.source.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PoseFrameContract":
        schema = str(payload.get("schema", POSE_FRAME_CONTRACT_SCHEMA_V1))
        if schema != POSE_FRAME_CONTRACT_SCHEMA_V1:
            raise PoseFrameContractError(f"Unsupported pose frame contract schema {schema!r}.")
        source_payload = payload.get("source")
        if not isinstance(source_payload, Mapping):
            raise PoseFrameContractError("pose frame contract requires a source mapping.")
        return cls(
            source=PoseSourceProfile.from_dict(source_payload),
            pose_storage_frame=str(payload.get("pose_storage_frame", POSE_STORAGE_FRAME_DESK)),
            canonical_absolute_frame=str(payload.get("canonical_absolute_frame", DESK_ABSOLUTE_FRAME)),
            metadata=dict(payload.get("metadata", {})),
        )


def pose_frame_contract_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    active_sides: Sequence[str] | None = None,
    allow_legacy_v22: bool = True,
) -> PoseFrameContract:
    """Resolve an explicit contract, or a marked legacy v2.2 compatibility contract."""

    payload = {} if metadata is None else metadata.get(POSE_FRAME_CONTRACT_METADATA_KEY)
    if isinstance(payload, Mapping):
        return PoseFrameContract.from_dict(payload)
    if not allow_legacy_v22:
        raise PoseFrameContractError(
            f"Metadata is missing {POSE_FRAME_CONTRACT_METADATA_KEY!r}."
        )
    return PoseFrameContract.legacy_v22_raw(active_sides=active_sides)


def load_pose_source_profile(path: str | Path) -> PoseSourceProfile:
    """Load one explicit pose source profile JSON file."""

    source_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PoseFrameContractError(
            f"Cannot read pose source profile {source_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PoseFrameContractError(f"Pose source profile {source_path} must be a JSON object.")
    nested = payload.get(POSE_SOURCE_CONTRACT_METADATA_KEY, payload.get("source", payload))
    if not isinstance(nested, Mapping):
        raise PoseFrameContractError(
            f"Pose source profile {source_path} does not contain a source mapping."
        )
    return PoseSourceProfile.from_dict(nested)


def load_pose_frame_contract(path: str | Path) -> PoseFrameContract:
    """Load an explicit dataset pose-frame contract JSON file."""

    contract_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PoseFrameContractError(
            f"Cannot read pose frame contract {contract_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise PoseFrameContractError(
            f"Pose frame contract {contract_path} must be a JSON object."
        )
    nested = payload.get(POSE_FRAME_CONTRACT_METADATA_KEY, payload)
    if not isinstance(nested, Mapping):
        raise PoseFrameContractError(
            f"Pose frame contract {contract_path} does not contain a contract mapping."
        )
    return PoseFrameContract.from_dict(nested)


__all__ = [
    "CANONICAL_DESK_FRAME",
    "DESK_ABSOLUTE_FRAME",
    "POSE_FRAME_CONTRACT_METADATA_KEY",
    "POSE_FRAME_CONTRACT_SCHEMA_V1",
    "POSE_SOURCE_CONTRACT_METADATA_KEY",
    "POSE_SOURCE_CONTRACT_SCHEMA_V1",
    "POSE_STORAGE_FRAME_DESK",
    "POSE_STORAGE_FRAME_RAW",
    "POSE_STORAGE_FRAMES",
    "PoseFrameContract",
    "PoseFrameContractError",
    "PoseSourceProfile",
    "legacy_rm65_source_id",
    "load_pose_frame_contract",
    "load_pose_source_profile",
    "normalize_pose_storage_frame",
    "pose_frame_contract_from_metadata",
    "transform_pose_vector",
]
