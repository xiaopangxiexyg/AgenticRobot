"""Pure TCP-pose safety-envelope contracts and projection.

This module defines the backend-neutral numeric semantics for limiting a
commanded TCP pose *before* inverse kinematics.  It deliberately owns no
device reads, clocks, Hold policy, IK, FK, collision checks, execution, or
dispatch.  Runtime adapters must provide explicit frame/tool/epoch bindings
and translate validation failures into their authoritative fail-closed action.

Translation limits may combine:

* an absolute axis-aligned box in the declared target reference frame;
* a box relative to a measured activation anchor, with either canonical axes
  or axes fixed to the anchor TCP orientation.

When both boxes are enabled their intersection is projected in one convex
operation.  Sequential clipping is intentionally not used because projecting
onto an AABB and then an oriented box can leave the first box.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash


TCP_POSE_ENVELOPE_PROFILE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.tcp_pose_envelope_profile.development"
)
TCP_POSE_ENVELOPE_ANCHOR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.tcp_pose_envelope_anchor.development"
)
TCP_POSE_ENVELOPE_RESULT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.tcp_pose_envelope_result.development"
)
TCP_POSE_ENVELOPE_FEEDBACK_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.tcp_pose_envelope_feedback.development"
)
TCP_CONTROLLED_FRAME_ID = "tool_tcp"

RelativeTranslationAxes = Literal["canonical_axes", "anchor_local_axes"]
RELATIVE_TRANSLATION_AXES = frozenset(
    {"canonical_axes", "anchor_local_axes"}
)

_AXIS_NAMES = ("x", "y", "z")
_POSE_ATOL = 1e-9
_CONSTRAINT_TOL = 1e-10


class TCPPoseEnvelopeError(ValueError):
    """Base class for deterministic envelope validation failures."""


class TCPPoseEnvelopeBindingError(TCPPoseEnvelopeError):
    """Raised when frame, tool, profile, anchor, or epoch identity differs."""


class TCPPoseEnvelopeInfeasibleError(TCPPoseEnvelopeError):
    """Raised when configured translation constraints have no intersection."""


class TCPPoseEnvelopeRotationCutLocusError(TCPPoseEnvelopeError):
    """Raised when a clipped SO(3) direction is non-unique near 180 degrees."""


def _required_text(value: object, *, name: str) -> str:
    if value is None:
        raise TCPPoseEnvelopeError(f"{name} must be non-empty")
    text = str(value).strip()
    if not text:
        raise TCPPoseEnvelopeError(f"{name} must be non-empty")
    return text


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise TCPPoseEnvelopeError(f"{name} must be finite")
    return result


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TCPPoseEnvelopeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result != value or result < 0:
        raise TCPPoseEnvelopeError(f"{name} must be a non-negative integer")
    return result


def _vector3(
    value: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> tuple[float, float, float]:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise TCPPoseEnvelopeError(f"{name} must contain three finite values") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise TCPPoseEnvelopeError(f"{name} must contain three finite values")
    return tuple(float(item) for item in vector)


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TCPPoseEnvelopeError(f"{name} must be a numeric 4x4 pose") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise TCPPoseEnvelopeError(f"{name} must be a finite 4x4 pose")
    if not np.allclose(
        pose[3],
        (0.0, 0.0, 0.0, 1.0),
        rtol=0.0,
        atol=_POSE_ATOL,
    ):
        raise TCPPoseEnvelopeError(f"{name} must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=_POSE_ATOL,
    ):
        raise TCPPoseEnvelopeError(f"{name} rotation must be orthonormal")
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=_POSE_ATOL,
    ):
        raise TCPPoseEnvelopeError(f"{name} rotation determinant must be +1")
    frozen = np.ascontiguousarray(pose, dtype=np.float64).copy()
    frozen.setflags(write=False)
    return frozen


def _rotation_from_pose(pose: np.ndarray) -> Rotation:
    """Construct SciPy Rotation from an immutable contract pose."""

    # SciPy 1.13's Cython memoryview rejects a read-only source even though the
    # constructor does not semantically mutate it.
    return Rotation.from_matrix(np.array(pose[:3, :3], dtype=float, copy=True))


def _freeze_pose_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, np.ndarray] = {}
    for raw_side, raw_pose in value.items():
        side = _required_text(raw_side, name=f"{name}.side")
        if side in result:
            raise TCPPoseEnvelopeError(f"{name} contains duplicate side {side!r}")
        result[side] = _pose(raw_pose, name=f"{name}.{side}")
    if not result:
        raise TCPPoseEnvelopeError(f"{name} must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


def _freeze_text_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {
        _required_text(side, name=f"{name}.side"): _required_text(
            frame_id,
            name=f"{name}.{side}",
        )
        for side, frame_id in value.items()
    }
    if not result:
        raise TCPPoseEnvelopeError(f"{name} must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


def _freeze_float_mapping(
    value: Mapping[str, object],
    *,
    name: str,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {
        _required_text(side, name=f"{name}.side"): _finite_time(
            timestamp,
            name=f"{name}.{side}",
        )
        for side, timestamp in value.items()
    }
    if not result:
        raise TCPPoseEnvelopeError(f"{name} must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


def _bounds_pair(
    lower: Sequence[float] | np.ndarray | None,
    upper: Sequence[float] | np.ndarray | None,
    *,
    prefix: str,
) -> tuple[
    tuple[float, float, float] | None,
    tuple[float, float, float] | None,
]:
    if (lower is None) != (upper is None):
        raise TCPPoseEnvelopeError(
            f"{prefix}_min_xyz_m and {prefix}_max_xyz_m must be provided together"
        )
    if lower is None:
        return None, None
    normalized_lower = _vector3(lower, name=f"{prefix}_min_xyz_m")
    normalized_upper = _vector3(upper, name=f"{prefix}_max_xyz_m")
    if np.any(np.asarray(normalized_lower) > np.asarray(normalized_upper)):
        raise TCPPoseEnvelopeError(
            f"{prefix}_min_xyz_m must not exceed {prefix}_max_xyz_m"
        )
    return normalized_lower, normalized_upper


@dataclass(frozen=True)
class TCPPoseEnvelopeSideConfig:
    """One side's absolute, anchor-relative, and angular TCP bounds."""

    enabled: bool = True
    absolute_min_xyz_m: tuple[float, float, float] | None = None
    absolute_max_xyz_m: tuple[float, float, float] | None = None
    relative_min_xyz_m: tuple[float, float, float] | None = None
    relative_max_xyz_m: tuple[float, float, float] | None = None
    relative_axes: RelativeTranslationAxes = "canonical_axes"
    max_rotation_rad: float | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TCPPoseEnvelopeError("enabled must be bool")
        absolute_lower, absolute_upper = _bounds_pair(
            self.absolute_min_xyz_m,
            self.absolute_max_xyz_m,
            prefix="absolute",
        )
        relative_lower, relative_upper = _bounds_pair(
            self.relative_min_xyz_m,
            self.relative_max_xyz_m,
            prefix="relative",
        )
        axes = str(self.relative_axes).strip()
        if axes not in RELATIVE_TRANSLATION_AXES:
            raise TCPPoseEnvelopeError(
                "relative_axes must be canonical_axes or anchor_local_axes"
            )
        max_rotation = self.max_rotation_rad
        if max_rotation is not None:
            max_rotation = float(max_rotation)
            if (
                not math.isfinite(max_rotation)
                or max_rotation < 0.0
                or max_rotation > math.pi
            ):
                raise TCPPoseEnvelopeError(
                    "max_rotation_rad must be finite and within [0, pi]"
                )
        if self.enabled and (
            absolute_lower is None
            and relative_lower is None
            and max_rotation is None
        ):
            raise TCPPoseEnvelopeError(
                "an enabled side needs an absolute, relative, or rotation bound"
            )
        object.__setattr__(self, "absolute_min_xyz_m", absolute_lower)
        object.__setattr__(self, "absolute_max_xyz_m", absolute_upper)
        object.__setattr__(self, "relative_min_xyz_m", relative_lower)
        object.__setattr__(self, "relative_max_xyz_m", relative_upper)
        object.__setattr__(self, "relative_axes", axes)
        object.__setattr__(self, "max_rotation_rad", max_rotation)

    @property
    def requires_anchor(self) -> bool:
        return bool(
            self.enabled
            and (
                self.relative_min_xyz_m is not None
                or self.max_rotation_rad is not None
            )
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "absolute_min_xyz_m": self.absolute_min_xyz_m,
            "absolute_max_xyz_m": self.absolute_max_xyz_m,
            "relative_min_xyz_m": self.relative_min_xyz_m,
            "relative_max_xyz_m": self.relative_max_xyz_m,
            "relative_axes": self.relative_axes,
            "max_rotation_rad": self.max_rotation_rad,
        }


@dataclass(frozen=True)
class TCPPoseEnvelopeProfile:
    """Versioned frame/tool-bound TCP envelope selected by an upper layer."""

    profile_id: str
    target_frame_binding_hash: str
    tool_frame_binding_hash_by_side: Mapping[str, str]
    reference_frame_by_side: Mapping[str, str]
    side_config_by_side: Mapping[str, TCPPoseEnvelopeSideConfig]
    controlled_frame_id: str = TCP_CONTROLLED_FRAME_ID
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        profile_id = _required_text(self.profile_id, name="profile_id")
        target_hash = _required_text(
            self.target_frame_binding_hash,
            name="target_frame_binding_hash",
        )
        tool_hashes = _freeze_text_mapping(
            self.tool_frame_binding_hash_by_side,
            name="tool_frame_binding_hash_by_side",
        )
        controlled_frame = _required_text(
            self.controlled_frame_id,
            name="controlled_frame_id",
        )
        if controlled_frame != TCP_CONTROLLED_FRAME_ID:
            raise TCPPoseEnvelopeError(
                "TCP pose envelope v1 requires controlled_frame_id='tool_tcp'"
            )
        references = _freeze_text_mapping(
            self.reference_frame_by_side,
            name="reference_frame_by_side",
        )
        if not isinstance(self.side_config_by_side, Mapping):
            raise TypeError("side_config_by_side must be a mapping")
        configs: dict[str, TCPPoseEnvelopeSideConfig] = {}
        for raw_side, config in self.side_config_by_side.items():
            side = _required_text(raw_side, name="side_config_by_side.side")
            if not isinstance(config, TCPPoseEnvelopeSideConfig):
                raise TypeError(
                    f"side_config_by_side[{side!r}] must be "
                    "TCPPoseEnvelopeSideConfig"
                )
            configs[side] = config
        if not configs:
            raise TCPPoseEnvelopeError("side_config_by_side must not be empty")
        if not any(config.enabled for config in configs.values()):
            raise TCPPoseEnvelopeError(
                "an envelope profile must enable at least one configured side"
            )
        missing_references = sorted(set(configs) - set(references))
        if missing_references:
            raise TCPPoseEnvelopeError(
                "reference_frame_by_side is missing configured sides: "
                f"{missing_references}"
            )
        missing_tool_bindings = sorted(set(configs) - set(tool_hashes))
        if missing_tool_bindings:
            raise TCPPoseEnvelopeError(
                "tool_frame_binding_hash_by_side is missing configured sides: "
                f"{missing_tool_bindings}"
            )
        ordered_configs = MappingProxyType(dict(sorted(configs.items())))
        payload = {
            "profile_id": profile_id,
            "target_frame_binding_hash": target_hash,
            "tool_frame_binding_hash_by_side": tool_hashes,
            "controlled_frame_id": controlled_frame,
            "reference_frame_by_side": references,
            "side_config_by_side": {
                side: config.semantic_payload()
                for side, config in ordered_configs.items()
            },
        }
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "target_frame_binding_hash", target_hash)
        object.__setattr__(self, "tool_frame_binding_hash_by_side", tool_hashes)
        object.__setattr__(self, "controlled_frame_id", controlled_frame)
        object.__setattr__(self, "reference_frame_by_side", references)
        object.__setattr__(self, "side_config_by_side", ordered_configs)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TCP_POSE_ENVELOPE_PROFILE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def enabled_sides(self) -> tuple[str, ...]:
        return tuple(
            side
            for side, config in self.side_config_by_side.items()
            if config.enabled
        )

    @property
    def anchor_required_sides(self) -> tuple[str, ...]:
        return tuple(
            side
            for side, config in self.side_config_by_side.items()
            if config.requires_anchor
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "target_frame_binding_hash": self.target_frame_binding_hash,
            "tool_frame_binding_hash_by_side": dict(
                self.tool_frame_binding_hash_by_side
            ),
            "controlled_frame_id": self.controlled_frame_id,
            "reference_frame_by_side": dict(self.reference_frame_by_side),
            "side_config_by_side": {
                side: config.semantic_payload()
                for side, config in self.side_config_by_side.items()
            },
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TCPPoseEnvelopeAnchorReceipt:
    """Measured target-only TCP anchor bound to one execution epoch."""

    anchor_id: str
    envelope_profile_hash: str
    target_frame_binding_hash: str
    tool_frame_binding_hash_by_side: Mapping[str, str]
    controlled_frame_id: str
    execution_epoch: int
    clock_domain_id: str
    tcp_pose_by_side: Mapping[str, np.ndarray]
    reference_frame_by_side: Mapping[str, str]
    captured_at_by_side: Mapping[str, float]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required_text(getattr(self, name), name=name)
            for name in (
                "anchor_id",
                "envelope_profile_hash",
                "target_frame_binding_hash",
                "controlled_frame_id",
                "clock_domain_id",
            )
        }
        tool_hashes = _freeze_text_mapping(
            self.tool_frame_binding_hash_by_side,
            name="tool_frame_binding_hash_by_side",
        )
        if text["controlled_frame_id"] != TCP_CONTROLLED_FRAME_ID:
            raise TCPPoseEnvelopeError(
                "TCP pose envelope anchor requires controlled_frame_id='tool_tcp'"
            )
        epoch = _non_negative_int(self.execution_epoch, name="execution_epoch")
        poses = _freeze_pose_mapping(
            self.tcp_pose_by_side,
            name="tcp_pose_by_side",
        )
        references = _freeze_text_mapping(
            self.reference_frame_by_side,
            name="reference_frame_by_side",
        )
        timestamps = _freeze_float_mapping(
            self.captured_at_by_side,
            name="captured_at_by_side",
        )
        side_sets = (set(poses), set(references), set(timestamps))
        if not all(side_set == side_sets[0] for side_set in side_sets[1:]):
            raise TCPPoseEnvelopeError(
                "anchor pose, reference-frame, and capture-time sides must match"
            )
        if not set(poses).issubset(tool_hashes):
            raise TCPPoseEnvelopeError(
                "anchor tool-frame bindings are missing captured sides"
            )
        payload = {
            **text,
            "execution_epoch": epoch,
            "tool_frame_binding_hash_by_side": tool_hashes,
            "tcp_pose_by_side": poses,
            "reference_frame_by_side": references,
            "captured_at_by_side": timestamps,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "execution_epoch", epoch)
        object.__setattr__(self, "tool_frame_binding_hash_by_side", tool_hashes)
        object.__setattr__(self, "tcp_pose_by_side", poses)
        object.__setattr__(self, "reference_frame_by_side", references)
        object.__setattr__(self, "captured_at_by_side", timestamps)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TCP_POSE_ENVELOPE_ANCHOR_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @classmethod
    def capture(
        cls,
        *,
        anchor_id: str,
        profile: TCPPoseEnvelopeProfile,
        execution_epoch: int,
        clock_domain_id: str,
        tcp_pose_by_side: Mapping[str, object],
        captured_at_by_side: Mapping[str, float],
    ) -> "TCPPoseEnvelopeAnchorReceipt":
        if not isinstance(profile, TCPPoseEnvelopeProfile):
            raise TypeError("profile must be TCPPoseEnvelopeProfile")
        sides = set(tcp_pose_by_side)
        unknown = sorted(sides - set(profile.reference_frame_by_side))
        if unknown:
            raise TCPPoseEnvelopeBindingError(
                f"anchor contains sides absent from envelope profile: {unknown}"
            )
        return cls(
            anchor_id=anchor_id,
            envelope_profile_hash=profile.semantic_content_hash,
            target_frame_binding_hash=profile.target_frame_binding_hash,
            tool_frame_binding_hash_by_side={
                side: profile.tool_frame_binding_hash_by_side[side]
                for side in sides
            },
            controlled_frame_id=profile.controlled_frame_id,
            execution_epoch=execution_epoch,
            clock_domain_id=clock_domain_id,
            tcp_pose_by_side=tcp_pose_by_side,
            reference_frame_by_side={
                side: profile.reference_frame_by_side[side]
                for side in sides
            },
            captured_at_by_side=captured_at_by_side,
        )

    @property
    def pair_skew_sec(self) -> float:
        timestamps = tuple(self.captured_at_by_side.values())
        return float(max(timestamps) - min(timestamps))

    def to_payload(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "envelope_profile_hash": self.envelope_profile_hash,
            "target_frame_binding_hash": self.target_frame_binding_hash,
            "tool_frame_binding_hash_by_side": dict(
                self.tool_frame_binding_hash_by_side
            ),
            "controlled_frame_id": self.controlled_frame_id,
            "execution_epoch": self.execution_epoch,
            "clock_domain_id": self.clock_domain_id,
            "tcp_pose_by_side": {
                side: pose.tolist()
                for side, pose in self.tcp_pose_by_side.items()
            },
            "reference_frame_by_side": dict(self.reference_frame_by_side),
            "captured_at_by_side": dict(self.captured_at_by_side),
            "pair_skew_sec": self.pair_skew_sec,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TCPPoseEnvelopeSideProjection:
    modified: bool
    translation_modified: bool
    rotation_modified: bool
    translation_adjustment_m: float
    rotation_adjustment_rad: float
    raw_anchor_rotation_rad: float | None
    effective_anchor_rotation_rad: float | None
    violated_translation_constraints: tuple[str, ...] = ()
    active_translation_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("modified", "translation_modified", "rotation_modified"):
            if type(getattr(self, name)) is not bool:
                raise TCPPoseEnvelopeError(f"{name} must be bool")
        if self.modified != (self.translation_modified or self.rotation_modified):
            raise TCPPoseEnvelopeError(
                "modified must equal translation_modified or rotation_modified"
            )
        for name in ("translation_adjustment_m", "rotation_adjustment_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise TCPPoseEnvelopeError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in ("raw_anchor_rotation_rad", "effective_anchor_rotation_rad"):
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0 or value > math.pi + _POSE_ATOL:
                raise TCPPoseEnvelopeError(f"{name} must be within [0, pi]")
            object.__setattr__(self, name, value)
        for name in (
            "violated_translation_constraints",
            "active_translation_constraints",
        ):
            values = tuple(
                _required_text(item, name=f"{name}.item")
                for item in getattr(self, name)
            )
            if len(set(values)) != len(values):
                raise TCPPoseEnvelopeError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "modified": self.modified,
            "translation_modified": self.translation_modified,
            "rotation_modified": self.rotation_modified,
            "translation_adjustment_m": self.translation_adjustment_m,
            "rotation_adjustment_rad": self.rotation_adjustment_rad,
            "raw_anchor_rotation_rad": self.raw_anchor_rotation_rad,
            "effective_anchor_rotation_rad": self.effective_anchor_rotation_rad,
            "violated_translation_constraints": (
                self.violated_translation_constraints
            ),
            "active_translation_constraints": self.active_translation_constraints,
        }


@dataclass(frozen=True)
class TCPPoseEnvelopeProjectionResult:
    """Projected pose set plus immutable per-side diagnostics and lineage."""

    raw_tcp_pose_by_side: Mapping[str, np.ndarray]
    effective_tcp_pose_by_side: Mapping[str, np.ndarray]
    side_projection_by_side: Mapping[str, TCPPoseEnvelopeSideProjection]
    envelope_profile_hash: str
    anchor_hash: str | None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        raw = _freeze_pose_mapping(
            self.raw_tcp_pose_by_side,
            name="raw_tcp_pose_by_side",
        )
        effective = _freeze_pose_mapping(
            self.effective_tcp_pose_by_side,
            name="effective_tcp_pose_by_side",
        )
        if set(raw) != set(effective):
            raise TCPPoseEnvelopeError("raw and effective pose sides must match")
        if not isinstance(self.side_projection_by_side, Mapping):
            raise TypeError("side_projection_by_side must be a mapping")
        projections: dict[str, TCPPoseEnvelopeSideProjection] = {}
        for raw_side, projection in self.side_projection_by_side.items():
            side = _required_text(raw_side, name="side_projection_by_side.side")
            if not isinstance(projection, TCPPoseEnvelopeSideProjection):
                raise TypeError(
                    "side_projection_by_side values must be "
                    "TCPPoseEnvelopeSideProjection"
                )
            projections[side] = projection
        if set(projections) != set(raw):
            raise TCPPoseEnvelopeError("projection diagnostics must cover all sides")
        profile_hash = _required_text(
            self.envelope_profile_hash,
            name="envelope_profile_hash",
        )
        anchor_hash = (
            None
            if self.anchor_hash is None
            else _required_text(self.anchor_hash, name="anchor_hash")
        )
        frozen_projections = MappingProxyType(dict(sorted(projections.items())))
        payload = {
            "raw_tcp_pose_by_side": raw,
            "effective_tcp_pose_by_side": effective,
            "side_projection_by_side": {
                side: projection.semantic_payload()
                for side, projection in frozen_projections.items()
            },
            "envelope_profile_hash": profile_hash,
            "anchor_hash": anchor_hash,
        }
        object.__setattr__(self, "raw_tcp_pose_by_side", raw)
        object.__setattr__(self, "effective_tcp_pose_by_side", effective)
        object.__setattr__(
            self,
            "side_projection_by_side",
            frozen_projections,
        )
        object.__setattr__(self, "envelope_profile_hash", profile_hash)
        object.__setattr__(self, "anchor_hash", anchor_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TCP_POSE_ENVELOPE_RESULT_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def modified_sides(self) -> tuple[str, ...]:
        return tuple(
            side
            for side, projection in self.side_projection_by_side.items()
            if projection.modified
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "raw_tcp_pose_by_side": {
                side: pose.tolist()
                for side, pose in self.raw_tcp_pose_by_side.items()
            },
            "effective_tcp_pose_by_side": {
                side: pose.tolist()
                for side, pose in self.effective_tcp_pose_by_side.items()
            },
            "side_projection_by_side": {
                side: projection.semantic_payload()
                for side, projection in self.side_projection_by_side.items()
            },
            "envelope_profile_hash": self.envelope_profile_hash,
            "anchor_hash": self.anchor_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }


TCPPoseEnvelopeFeedbackState = Literal[
    "unbound",
    "pending",
    "clear",
    "limited",
    "fault",
]


@dataclass(frozen=True)
class TCPPoseEnvelopeSideFeedback:
    """Compact per-side result for low-rate downstream observation."""

    modified: bool
    translation_adjustment_m: float
    rotation_adjustment_rad: float
    violated_translation_constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.modified) is not bool:
            raise TCPPoseEnvelopeError("modified must be bool")
        for name in ("translation_adjustment_m", "rotation_adjustment_rad"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise TCPPoseEnvelopeError(
                    f"{name} must be finite and non-negative"
                )
            object.__setattr__(self, name, value)
        constraints = tuple(
            _required_text(item, name="violated_translation_constraints.item")
            for item in self.violated_translation_constraints
        )
        if len(set(constraints)) != len(constraints):
            raise TCPPoseEnvelopeError(
                "violated_translation_constraints must not contain duplicates"
            )
        object.__setattr__(
            self,
            "violated_translation_constraints",
            constraints,
        )

    @classmethod
    def from_projection(
        cls,
        projection: TCPPoseEnvelopeSideProjection,
    ) -> TCPPoseEnvelopeSideFeedback:
        if not isinstance(projection, TCPPoseEnvelopeSideProjection):
            raise TypeError("projection must be TCPPoseEnvelopeSideProjection")
        return cls(
            modified=projection.modified,
            translation_adjustment_m=projection.translation_adjustment_m,
            rotation_adjustment_rad=projection.rotation_adjustment_rad,
            violated_translation_constraints=(
                projection.violated_translation_constraints
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "modified": self.modified,
            "translation_adjustment_m": self.translation_adjustment_m,
            "rotation_adjustment_rad": self.rotation_adjustment_rad,
            "violated_translation_constraints": list(
                self.violated_translation_constraints
            ),
        }


@dataclass(frozen=True)
class TCPPoseEnvelopeProjectionFeedback:
    """Small immutable receipt exposed by a downstream envelope adapter.

    This is deliberately an observational contract.  It contains no raw or
    effective pose matrices and owns no safety authority.  The Active adapter
    remains the sole owner of projection and fail-closed behavior; an upper
    layer may poll this receipt without participating in the control loop.
    """

    mode: Literal["active", "shadow"]
    state: TCPPoseEnvelopeFeedbackState
    profile_id: str
    profile_hash: str | None
    execution_epoch: int
    anchor_hash: str | None
    frame_count: int
    target_role: str | None
    projection_hash: str | None
    modified_sides: tuple[str, ...]
    side_adjustment_by_side: Mapping[str, TCPPoseEnvelopeSideFeedback]
    reason: str
    error: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"active", "shadow"}:
            raise TCPPoseEnvelopeError("mode must be 'active' or 'shadow'")
        if self.state not in {"unbound", "pending", "clear", "limited", "fault"}:
            raise TCPPoseEnvelopeError("invalid TCP envelope feedback state")
        profile_id = _required_text(self.profile_id, name="profile_id")
        profile_hash = (
            None
            if self.profile_hash is None
            else _required_text(self.profile_hash, name="profile_hash")
        )
        if self.state in {"pending", "clear", "limited"} and profile_hash is None:
            raise TCPPoseEnvelopeError(
                "bound TCP envelope feedback requires profile_hash"
            )
        epoch = _non_negative_int(self.execution_epoch, name="execution_epoch")
        frame_count = _non_negative_int(self.frame_count, name="frame_count")
        anchor_hash = (
            None
            if self.anchor_hash is None
            else _required_text(self.anchor_hash, name="anchor_hash")
        )
        target_role = (
            None
            if self.target_role is None
            else _required_text(self.target_role, name="target_role")
        )
        projection_hash = (
            None
            if self.projection_hash is None
            else _required_text(self.projection_hash, name="projection_hash")
        )
        modified_sides = tuple(
            _required_text(side, name="modified_sides.item")
            for side in self.modified_sides
        )
        if len(set(modified_sides)) != len(modified_sides):
            raise TCPPoseEnvelopeError("modified_sides must not contain duplicates")
        if not isinstance(self.side_adjustment_by_side, Mapping):
            raise TypeError("side_adjustment_by_side must be a mapping")
        side_adjustments: dict[str, TCPPoseEnvelopeSideFeedback] = {}
        for raw_side, adjustment in self.side_adjustment_by_side.items():
            side = _required_text(raw_side, name="side_adjustment_by_side.side")
            if not isinstance(adjustment, TCPPoseEnvelopeSideFeedback):
                raise TypeError(
                    "side_adjustment_by_side values must be "
                    "TCPPoseEnvelopeSideFeedback"
                )
            side_adjustments[side] = adjustment
        expected_modified = tuple(
            side
            for side, adjustment in sorted(side_adjustments.items())
            if adjustment.modified
        )
        if tuple(sorted(modified_sides)) != expected_modified:
            raise TCPPoseEnvelopeError(
                "modified_sides must match modified side adjustments"
            )
        if self.state == "limited" and not modified_sides:
            raise TCPPoseEnvelopeError("limited feedback must name a modified side")
        if self.state == "clear" and modified_sides:
            raise TCPPoseEnvelopeError("clear feedback cannot name modified sides")
        error = (
            None if self.error is None else _required_text(self.error, name="error")
        )
        if self.state == "fault" and error is None:
            raise TCPPoseEnvelopeError("fault feedback requires error")
        if self.state != "fault" and error is not None:
            raise TCPPoseEnvelopeError("non-fault feedback cannot carry error")
        reason = _required_text(self.reason, name="reason")
        frozen_adjustments = MappingProxyType(dict(sorted(side_adjustments.items())))
        payload = {
            "mode": self.mode,
            "state": self.state,
            "controlled_frame": TCP_CONTROLLED_FRAME_ID,
            "profile_id": profile_id,
            "profile_hash": profile_hash,
            "execution_epoch": epoch,
            "anchor_hash": anchor_hash,
            "frame_count": frame_count,
            "target_role": target_role,
            "projection_hash": projection_hash,
            "modified_sides": list(modified_sides),
            "side_adjustments": {
                side: adjustment.to_payload()
                for side, adjustment in frozen_adjustments.items()
            },
            "reason": reason,
            "error": error,
        }
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "profile_hash", profile_hash)
        object.__setattr__(self, "execution_epoch", epoch)
        object.__setattr__(self, "anchor_hash", anchor_hash)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "target_role", target_role)
        object.__setattr__(self, "projection_hash", projection_hash)
        object.__setattr__(self, "modified_sides", modified_sides)
        object.__setattr__(self, "side_adjustment_by_side", frozen_adjustments)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "error", error)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TCP_POSE_ENVELOPE_FEEDBACK_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @classmethod
    def from_evaluation(
        cls,
        *,
        apply_to_target: bool,
        bound: bool,
        profile_id: str,
        profile_hash: str | None,
        execution_epoch: int,
        anchor_hash: str | None,
        frame_count: int,
        target_role: str | None,
        projection: TCPPoseEnvelopeProjectionResult | None,
        error: str | None,
    ) -> TCPPoseEnvelopeProjectionFeedback:
        side_adjustments = (
            {}
            if projection is None
            else {
                side: TCPPoseEnvelopeSideFeedback.from_projection(side_projection)
                for side, side_projection in projection.side_projection_by_side.items()
            }
        )
        modified_sides = (
            () if projection is None else projection.modified_sides
        )
        if error is not None:
            state: TCPPoseEnvelopeFeedbackState = "fault"
            reason = str(error)
        elif not bound:
            state = "unbound"
            reason = "profile_unbound"
        elif projection is None:
            state = "pending"
            reason = "awaiting_first_evaluation"
        elif modified_sides:
            state = "limited"
            reason = "target_projected:" + ",".join(modified_sides)
        else:
            state = "clear"
            reason = "within_envelope"
        return cls(
            mode="active" if apply_to_target else "shadow",
            state=state,
            profile_id=profile_id,
            profile_hash=profile_hash,
            execution_epoch=execution_epoch,
            anchor_hash=anchor_hash,
            frame_count=frame_count,
            target_role=target_role,
            projection_hash=(
                None
                if projection is None
                else projection.semantic_content_hash
            ),
            modified_sides=modified_sides,
            side_adjustment_by_side=side_adjustments,
            reason=reason,
            error=error,
        )

    @property
    def receipt_ref(self) -> str:
        return self.semantic_content_hash

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "robot_motion.tcp_pose_envelope_feedback.v1",
            "mode": self.mode,
            "state": self.state,
            "controlled_frame": TCP_CONTROLLED_FRAME_ID,
            "profile_id": self.profile_id,
            "profile_hash": self.profile_hash,
            "execution_epoch": self.execution_epoch,
            "anchor_hash": self.anchor_hash,
            "frame_count": self.frame_count,
            "target_role": self.target_role,
            "projection_hash": self.projection_hash,
            "modified_sides": list(self.modified_sides),
            "side_adjustments": {
                side: adjustment.to_payload()
                for side, adjustment in self.side_adjustment_by_side.items()
            },
            "reason": self.reason,
            "error": self.error,
            "receipt_ref": self.receipt_ref,
        }


@dataclass(frozen=True)
class _Halfspace:
    normal: np.ndarray
    upper: float
    label: str


def _box_halfspaces(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    origin: np.ndarray,
    axes_world: np.ndarray,
    label_prefix: str,
) -> tuple[_Halfspace, ...]:
    """Return ``lower <= axes.T @ (x-origin) <= upper`` halfspaces."""

    rows: list[_Halfspace] = []
    for axis_index, axis_name in enumerate(_AXIS_NAMES):
        normal = np.asarray(axes_world[:, axis_index], dtype=float)
        origin_coordinate = float(normal @ origin)
        rows.append(
            _Halfspace(
                normal=-normal,
                upper=float(-lower[axis_index] - origin_coordinate),
                label=f"{label_prefix}:{axis_name}:min",
            )
        )
        rows.append(
            _Halfspace(
                normal=normal,
                upper=float(upper[axis_index] + origin_coordinate),
                label=f"{label_prefix}:{axis_name}:max",
            )
        )
    return tuple(rows)


def _project_to_halfspace_intersection(
    point: np.ndarray,
    constraints: Sequence[_Halfspace],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Project one 3-vector onto a closed polyhedron exactly."""

    if not constraints:
        return point.copy(), (), ()
    matrix = np.stack([item.normal for item in constraints], axis=0)
    upper = np.asarray([item.upper for item in constraints], dtype=float)
    labels = tuple(item.label for item in constraints)
    raw_residual = matrix @ point - upper
    violated = tuple(
        labels[index]
        for index in np.flatnonzero(raw_residual > _CONSTRAINT_TOL)
    )

    # Absolute bounds and the default canonical-axis relative bounds form one
    # axis-aligned box intersection.  Project it exactly with component clips
    # instead of enumerating active sets at the safety boundary.  The generic
    # path below remains authoritative for anchor-local rotated boxes and mixed
    # polyhedra.
    axis_by_constraint: list[tuple[int, float]] = []
    for normal in matrix:
        axis = int(np.argmax(np.abs(normal)))
        sign = float(normal[axis])
        residual_normal = normal.copy()
        residual_normal[axis] = 0.0
        if (
            not math.isclose(abs(sign), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or not np.allclose(
                residual_normal,
                np.zeros(3, dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            axis_by_constraint = []
            break
        axis_by_constraint.append((axis, sign))
    if len(axis_by_constraint) == len(constraints):
        lower = np.full(3, -math.inf, dtype=float)
        upper_box = np.full(3, math.inf, dtype=float)
        for index, (axis, sign) in enumerate(axis_by_constraint):
            if sign > 0.0:
                upper_box[axis] = min(upper_box[axis], upper[index])
            else:
                lower[axis] = max(lower[axis], -upper[index])
        if np.any(lower > upper_box + _CONSTRAINT_TOL):
            raise TCPPoseEnvelopeInfeasibleError(
                "absolute and relative TCP translation bounds have an empty "
                "intersection"
            )
        projected = np.minimum(np.maximum(point, lower), upper_box)
        projected_residual = matrix @ projected - upper
        active = tuple(
            labels[index]
            for index in np.flatnonzero(
                np.abs(projected_residual) <= 5.0 * _CONSTRAINT_TOL
            )
        )
        return projected, violated, active

    if not violated:
        active = tuple(
            labels[index]
            for index in np.flatnonzero(np.abs(raw_residual) <= _CONSTRAINT_TOL)
        )
        return point.copy(), violated, active

    best: np.ndarray | None = None
    best_distance_sq = math.inf
    best_key: tuple[str, ...] | None = None
    max_active = min(3, len(constraints))
    for active_count in range(1, max_active + 1):
        for indices in itertools.combinations(range(len(constraints)), active_count):
            active_matrix = matrix[np.asarray(indices, dtype=int)]
            if np.linalg.matrix_rank(active_matrix, tol=1e-11) != active_count:
                continue
            gram = active_matrix @ active_matrix.T
            rhs = active_matrix @ point - upper[np.asarray(indices, dtype=int)]
            try:
                multipliers = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                continue
            if np.any(multipliers < -_CONSTRAINT_TOL):
                continue
            candidate = point - active_matrix.T @ multipliers
            if np.any(matrix @ candidate - upper > 5.0 * _CONSTRAINT_TOL):
                continue
            distance_sq = float(np.dot(candidate - point, candidate - point))
            key = tuple(labels[index] for index in indices)
            if (
                distance_sq < best_distance_sq - 1e-18
                or (
                    math.isclose(
                        distance_sq,
                        best_distance_sq,
                        rel_tol=0.0,
                        abs_tol=1e-18,
                    )
                    and (best_key is None or key < best_key)
                )
            ):
                best = candidate
                best_distance_sq = distance_sq
                best_key = key
    if best is None:
        raise TCPPoseEnvelopeInfeasibleError(
            "absolute and relative TCP translation bounds have an empty intersection"
        )
    residual = matrix @ best - upper
    active = tuple(
        labels[index]
        for index in np.flatnonzero(np.abs(residual) <= 5.0 * _CONSTRAINT_TOL)
    )
    return best, violated, active


def _translation_constraints(
    config: TCPPoseEnvelopeSideConfig,
    *,
    anchor_pose: np.ndarray | None,
) -> tuple[_Halfspace, ...]:
    constraints: list[_Halfspace] = []
    if config.absolute_min_xyz_m is not None:
        constraints.extend(
            _box_halfspaces(
                lower=np.asarray(config.absolute_min_xyz_m, dtype=float),
                upper=np.asarray(config.absolute_max_xyz_m, dtype=float),
                origin=np.zeros(3, dtype=float),
                axes_world=np.eye(3, dtype=float),
                label_prefix="absolute",
            )
        )
    if config.relative_min_xyz_m is not None:
        if anchor_pose is None:
            raise TCPPoseEnvelopeBindingError(
                "anchor pose is required for relative TCP translation bounds"
            )
        axes_world = (
            np.eye(3, dtype=float)
            if config.relative_axes == "canonical_axes"
            else anchor_pose[:3, :3]
        )
        constraints.extend(
            _box_halfspaces(
                lower=np.asarray(config.relative_min_xyz_m, dtype=float),
                upper=np.asarray(config.relative_max_xyz_m, dtype=float),
                origin=anchor_pose[:3, 3],
                axes_world=axes_world,
                label_prefix=(
                    "relative_canonical"
                    if config.relative_axes == "canonical_axes"
                    else "relative_anchor_local"
                ),
            )
        )
    return tuple(constraints)


def _validate_projection_binding(
    *,
    profile: TCPPoseEnvelopeProfile,
    target_frame_binding_hash: str,
    tool_frame_binding_hash_by_side: Mapping[str, str],
    controlled_frame_id: str,
    reference_frame_by_side: Mapping[str, str],
    execution_epoch: int,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
) -> None:
    actual_target_hash = _required_text(
        target_frame_binding_hash,
        name="target_frame_binding_hash",
    )
    actual_tool_hashes = _freeze_text_mapping(
        tool_frame_binding_hash_by_side,
        name="tool_frame_binding_hash_by_side",
    )
    actual_controlled_frame = _required_text(
        controlled_frame_id,
        name="controlled_frame_id",
    )
    epoch = _non_negative_int(execution_epoch, name="execution_epoch")
    actual_references = _freeze_text_mapping(
        reference_frame_by_side,
        name="reference_frame_by_side",
    )
    comparisons = (
        (
            actual_target_hash,
            profile.target_frame_binding_hash,
            "target-frame binding hash mismatch",
        ),
        (
            actual_controlled_frame,
            profile.controlled_frame_id,
            "controlled frame mismatch",
        ),
    )
    for actual, expected, reason in comparisons:
        if actual != expected:
            raise TCPPoseEnvelopeBindingError(reason)
    for side in profile.enabled_sides:
        if actual_references.get(side) != profile.reference_frame_by_side[side]:
            raise TCPPoseEnvelopeBindingError(
                f"target reference frame mismatch for side {side!r}"
            )
        if (
            actual_tool_hashes.get(side)
            != profile.tool_frame_binding_hash_by_side[side]
        ):
            raise TCPPoseEnvelopeBindingError(
                f"tool-frame binding hash mismatch for side {side!r}"
            )

    required_sides = profile.anchor_required_sides
    if not required_sides:
        return
    if anchor_receipt is None:
        raise TCPPoseEnvelopeBindingError(
            f"TCP envelope anchor is required for sides {list(required_sides)}"
        )
    if not isinstance(anchor_receipt, TCPPoseEnvelopeAnchorReceipt):
        raise TypeError("anchor_receipt must be TCPPoseEnvelopeAnchorReceipt")
    anchor_comparisons = (
        (
            anchor_receipt.envelope_profile_hash,
            profile.semantic_content_hash,
            "anchor envelope-profile hash mismatch",
        ),
        (
            anchor_receipt.target_frame_binding_hash,
            actual_target_hash,
            "anchor target-frame binding hash mismatch",
        ),
        (
            anchor_receipt.controlled_frame_id,
            actual_controlled_frame,
            "anchor controlled frame mismatch",
        ),
    )
    for actual, expected, reason in anchor_comparisons:
        if actual != expected:
            raise TCPPoseEnvelopeBindingError(reason)
    if anchor_receipt.execution_epoch != epoch:
        raise TCPPoseEnvelopeBindingError("anchor execution epoch mismatch")
    for side in required_sides:
        if side not in anchor_receipt.tcp_pose_by_side:
            raise TCPPoseEnvelopeBindingError(
                f"anchor is missing required side {side!r}"
            )
        expected_reference = profile.reference_frame_by_side[side]
        if anchor_receipt.reference_frame_by_side.get(side) != expected_reference:
            raise TCPPoseEnvelopeBindingError(
                f"anchor reference frame mismatch for side {side!r}"
            )
        if (
            anchor_receipt.tool_frame_binding_hash_by_side.get(side)
            != actual_tool_hashes[side]
        ):
            raise TCPPoseEnvelopeBindingError(
                f"anchor tool-frame binding hash mismatch for side {side!r}"
            )


def _project_tcp_pose_envelope_side_unchecked(
    *,
    side: str,
    raw_pose: np.ndarray,
    profile: TCPPoseEnvelopeProfile,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
) -> tuple[np.ndarray, TCPPoseEnvelopeSideProjection]:
    """Project one already validated side without changing binding authority."""

    config = profile.side_config_by_side.get(side)
    if config is None or not config.enabled:
        return raw_pose, TCPPoseEnvelopeSideProjection(
            modified=False,
            translation_modified=False,
            rotation_modified=False,
            translation_adjustment_m=0.0,
            rotation_adjustment_rad=0.0,
            raw_anchor_rotation_rad=None,
            effective_anchor_rotation_rad=None,
        )

    anchor_pose = (
        None
        if anchor_receipt is None
        else anchor_receipt.tcp_pose_by_side.get(side)
    )
    constraints = _translation_constraints(config, anchor_pose=anchor_pose)
    projected_translation, violated, active = _project_to_halfspace_intersection(
        raw_pose[:3, 3],
        constraints,
    )
    output = np.asarray(raw_pose, dtype=float).copy()
    output[:3, 3] = projected_translation

    raw_anchor_rotation: float | None = None
    effective_anchor_rotation: float | None = None
    if config.max_rotation_rad is not None:
        if anchor_pose is None:
            raise TCPPoseEnvelopeBindingError(
                f"anchor pose is required for side {side!r} rotation bound"
            )
        anchor_rotation = _rotation_from_pose(anchor_pose)
        raw_rotation = _rotation_from_pose(raw_pose)
        delta = anchor_rotation.inv() * raw_rotation
        raw_anchor_rotation = float(delta.magnitude())
        if raw_anchor_rotation > config.max_rotation_rad + _CONSTRAINT_TOL:
            # At the SO(3) cut locus the shortest rotation axis is not unique.
            if raw_anchor_rotation >= math.pi - 1e-6:
                raise TCPPoseEnvelopeRotationCutLocusError(
                    f"relative TCP rotation for side {side!r} is too close "
                    "to 180 degrees for a unique bounded projection"
                )
            rotvec = delta.as_rotvec()
            if raw_anchor_rotation > 0.0:
                rotvec *= config.max_rotation_rad / raw_anchor_rotation
            output[:3, :3] = (
                anchor_rotation * Rotation.from_rotvec(rotvec)
            ).as_matrix()
            effective_anchor_rotation = float(config.max_rotation_rad)
        else:
            effective_anchor_rotation = raw_anchor_rotation

    output = _pose(output, name=f"effective_tcp_pose_by_side.{side}")
    translation_adjustment = float(
        np.linalg.norm(output[:3, 3] - raw_pose[:3, 3])
    )
    rotation_adjustment = float(
        (
            _rotation_from_pose(raw_pose).inv()
            * _rotation_from_pose(output)
        ).magnitude()
    )
    translation_modified = translation_adjustment > _CONSTRAINT_TOL
    rotation_modified = rotation_adjustment > _CONSTRAINT_TOL
    return output, TCPPoseEnvelopeSideProjection(
        modified=translation_modified or rotation_modified,
        translation_modified=translation_modified,
        rotation_modified=rotation_modified,
        translation_adjustment_m=translation_adjustment,
        rotation_adjustment_rad=rotation_adjustment,
        raw_anchor_rotation_rad=raw_anchor_rotation,
        effective_anchor_rotation_rad=effective_anchor_rotation,
        violated_translation_constraints=violated,
        active_translation_constraints=active,
    )


def _projection_result(
    *,
    raw_poses: Mapping[str, np.ndarray],
    profile: TCPPoseEnvelopeProfile,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
) -> TCPPoseEnvelopeProjectionResult:
    effective: dict[str, np.ndarray] = {}
    diagnostics: dict[str, TCPPoseEnvelopeSideProjection] = {}
    for side, raw_pose in raw_poses.items():
        effective[side], diagnostics[side] = (
            _project_tcp_pose_envelope_side_unchecked(
                side=side,
                raw_pose=raw_pose,
                profile=profile,
                anchor_receipt=anchor_receipt,
            )
        )

    return TCPPoseEnvelopeProjectionResult(
        raw_tcp_pose_by_side=raw_poses,
        effective_tcp_pose_by_side=effective,
        side_projection_by_side=diagnostics,
        envelope_profile_hash=profile.semantic_content_hash,
        anchor_hash=(
            None
            if anchor_receipt is None
            else anchor_receipt.semantic_content_hash
        ),
    )


def project_tcp_pose_envelope(
    raw_tcp_pose_by_side: Mapping[str, object],
    *,
    profile: TCPPoseEnvelopeProfile,
    target_frame_binding_hash: str,
    tool_frame_binding_hash_by_side: Mapping[str, str],
    controlled_frame_id: str,
    reference_frame_by_side: Mapping[str, str],
    execution_epoch: int,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None = None,
) -> TCPPoseEnvelopeProjectionResult:
    """Project final target TCP poses before IK under explicit bindings.

    Unconfigured and disabled sides pass through unchanged.  Every enabled
    configured side must be present.  Binding or numeric validation failures
    raise a typed error so a runtime owner can fail closed before invoking IK.
    """

    if not isinstance(profile, TCPPoseEnvelopeProfile):
        raise TypeError("profile must be TCPPoseEnvelopeProfile")
    raw_poses = _freeze_pose_mapping(
        raw_tcp_pose_by_side,
        name="raw_tcp_pose_by_side",
    )
    missing_sides = sorted(set(profile.enabled_sides) - set(raw_poses))
    if missing_sides:
        raise TCPPoseEnvelopeBindingError(
            f"target pose is missing enabled sides: {missing_sides}"
        )
    _validate_projection_binding(
        profile=profile,
        target_frame_binding_hash=target_frame_binding_hash,
        tool_frame_binding_hash_by_side=tool_frame_binding_hash_by_side,
        controlled_frame_id=controlled_frame_id,
        reference_frame_by_side=reference_frame_by_side,
        execution_epoch=execution_epoch,
        anchor_receipt=anchor_receipt,
    )
    return _projection_result(
        raw_poses=raw_poses,
        profile=profile,
        anchor_receipt=anchor_receipt,
    )


def project_tcp_pose_envelope_side(
    side: str,
    raw_tcp_pose: object,
    *,
    profile: TCPPoseEnvelopeProfile,
    target_frame_binding_hash: str,
    tool_frame_binding_hash_by_side: Mapping[str, str],
    controlled_frame_id: str,
    reference_frame_by_side: Mapping[str, str],
    execution_epoch: int,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None = None,
) -> TCPPoseEnvelopeProjectionResult:
    """Project one independent arm while preserving the full profile binding.

    This entry point exists for independently scheduled arm workers.  It does
    not relax the pair API's requirement that a paired command contain every
    enabled side, and it does not create a side-specific profile identity.
    """

    if not isinstance(profile, TCPPoseEnvelopeProfile):
        raise TypeError("profile must be TCPPoseEnvelopeProfile")
    normalized_side = _required_text(side, name="side")
    if normalized_side not in profile.side_config_by_side:
        raise TCPPoseEnvelopeBindingError(
            f"side {normalized_side!r} is absent from the envelope profile"
        )
    raw_poses = _freeze_pose_mapping(
        {normalized_side: raw_tcp_pose},
        name="raw_tcp_pose_by_side",
    )
    _validate_projection_binding(
        profile=profile,
        target_frame_binding_hash=target_frame_binding_hash,
        tool_frame_binding_hash_by_side=tool_frame_binding_hash_by_side,
        controlled_frame_id=controlled_frame_id,
        reference_frame_by_side=reference_frame_by_side,
        execution_epoch=execution_epoch,
        anchor_receipt=anchor_receipt,
    )
    return _projection_result(
        raw_poses=raw_poses,
        profile=profile,
        anchor_receipt=anchor_receipt,
    )


__all__ = [
    "RELATIVE_TRANSLATION_AXES",
    "TCP_CONTROLLED_FRAME_ID",
    "TCP_POSE_ENVELOPE_ANCHOR_HASH_DOMAIN_DEVELOPMENT",
    "TCP_POSE_ENVELOPE_FEEDBACK_HASH_DOMAIN_DEVELOPMENT",
    "TCP_POSE_ENVELOPE_PROFILE_HASH_DOMAIN_DEVELOPMENT",
    "TCP_POSE_ENVELOPE_RESULT_HASH_DOMAIN_DEVELOPMENT",
    "RelativeTranslationAxes",
    "TCPPoseEnvelopeAnchorReceipt",
    "TCPPoseEnvelopeBindingError",
    "TCPPoseEnvelopeError",
    "TCPPoseEnvelopeInfeasibleError",
    "TCPPoseEnvelopeProfile",
    "TCPPoseEnvelopeProjectionFeedback",
    "TCPPoseEnvelopeProjectionResult",
    "TCPPoseEnvelopeRotationCutLocusError",
    "TCPPoseEnvelopeSideConfig",
    "TCPPoseEnvelopeSideFeedback",
    "TCPPoseEnvelopeSideProjection",
    "project_tcp_pose_envelope",
    "project_tcp_pose_envelope_side",
]
