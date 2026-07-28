"""Extensible, hardware-independent rigid coordinate-frame relationships.

The graph stores immutable ``T_target_source`` edges.  A source adapter may
register one new frame by relating it to any already registered frame; the
graph then resolves transforms to every frame in the same connected component.

This module deliberately does not interpret action payloads.  Pose, twist,
wrench, residual-gain, recenter, and backend adapters remain responsible for
applying a resolved rigid transform according to their representation-specific
semantics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.immutable import deep_freeze, deep_thaw


COORDINATE_FRAME_SPEC_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.coordinate_frame_spec.development"
)
RIGID_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.rigid_frame_transform.development"
)
COORDINATE_FRAME_GRAPH_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.coordinate_frame_graph.development"
)
RESOLVED_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.resolved_frame_transform.development"
)
RESOLVED_POSE_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.resolved_pose_frame_transform.development"
)


class CoordinateFrameContractError(ValueError):
    """Raised when frame identities or rigid relationships are inconsistent."""


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise CoordinateFrameContractError(f"{name} must be non-empty.")
    return normalized


def _semantic_mapping(
    value: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    try:
        return deep_freeze(dict(value))
    except TypeError as exc:
        raise TypeError(f"{name} contains a non-canonical value: {exc}") from exc


def _rigid_transform(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CoordinateFrameContractError(
            f"{name} must be a numeric 4x4 matrix."
        ) from exc
    if matrix.shape != (4, 4):
        raise CoordinateFrameContractError(
            f"{name} must have shape (4, 4), got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise CoordinateFrameContractError(f"{name} contains non-finite values.")
    if not np.allclose(
        matrix[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-10,
    ):
        raise CoordinateFrameContractError(
            f"{name} last row must be [0, 0, 0, 1]."
        )
    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=1e-9,
    ):
        raise CoordinateFrameContractError(
            f"{name} rotation must be orthonormal."
        )
    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        rtol=0.0,
        atol=1e-9,
    ):
        raise CoordinateFrameContractError(
            f"{name} rotation determinant must be +1."
        )
    return deep_freeze(np.ascontiguousarray(matrix, dtype=np.float64))


@dataclass(frozen=True)
class CoordinateFrameSpec:
    """Stable identity and semantics for one spatial coordinate frame."""

    frame_id: str
    frame_version: str = "development"
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = _required(self.frame_id, name="frame_id")
        frame_version = _required(self.frame_version, name="frame_version")
        parameters = _semantic_mapping(
            self.semantic_parameters,
            name="semantic_parameters",
        )
        payload = {
            "frame_id": frame_id,
            "frame_version": frame_version,
            "semantic_parameters": parameters,
        }
        object.__setattr__(self, "frame_id", frame_id)
        object.__setattr__(self, "frame_version", frame_version)
        object.__setattr__(self, "semantic_parameters", parameters)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COORDINATE_FRAME_SPEC_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_version": self.frame_version,
            "semantic_parameters": self.semantic_parameters,
            "semantic_content_hash": self.semantic_content_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_version": self.frame_version,
            "semantic_parameters": deep_thaw(self.semantic_parameters),
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CoordinateFrameSpec":
        value = cls(
            frame_id=payload.get("frame_id", ""),
            frame_version=payload.get("frame_version", "development"),
            semantic_parameters=dict(payload.get("semantic_parameters", {})),
        )
        _verify_optional_hash(
            payload,
            expected=value.semantic_content_hash,
            subject=f"coordinate frame {value.frame_id!r}",
        )
        return value


@dataclass(frozen=True)
class RigidFrameTransformSpec:
    """One immutable ``T_target_source`` rigid transform edge."""

    transform_id: str
    source_frame_id: str
    target_frame_id: str
    matrix_target_source: np.ndarray
    transform_version: str = "development"
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        transform_id = _required(self.transform_id, name="transform_id")
        source = _required(self.source_frame_id, name="source_frame_id")
        target = _required(self.target_frame_id, name="target_frame_id")
        if source == target:
            raise CoordinateFrameContractError(
                "A rigid transform edge must connect two distinct frames."
            )
        version = _required(self.transform_version, name="transform_version")
        matrix = _rigid_transform(
            self.matrix_target_source,
            name="matrix_target_source",
        )
        parameters = _semantic_mapping(
            self.semantic_parameters,
            name="semantic_parameters",
        )
        payload = {
            "transform_id": transform_id,
            "transform_version": version,
            "source_frame_id": source,
            "target_frame_id": target,
            "matrix_convention": "T_target_source",
            "matrix_target_source": matrix,
            "semantic_parameters": parameters,
        }
        object.__setattr__(self, "transform_id", transform_id)
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "target_frame_id", target)
        object.__setattr__(self, "matrix_target_source", matrix)
        object.__setattr__(self, "transform_version", version)
        object.__setattr__(self, "semantic_parameters", parameters)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=RIGID_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "matrix_convention": "T_target_source",
            "matrix_target_source": self.matrix_target_source,
            "semantic_parameters": self.semantic_parameters,
            "semantic_content_hash": self.semantic_content_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "matrix_convention": "T_target_source",
            "matrix_target_source": self.matrix_target_source.tolist(),
            "semantic_parameters": deep_thaw(self.semantic_parameters),
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "RigidFrameTransformSpec":
        convention = str(
            payload.get("matrix_convention", "T_target_source")
        ).strip()
        if convention != "T_target_source":
            raise CoordinateFrameContractError(
                "Rigid frame transform matrix_convention must be "
                "'T_target_source'."
            )
        value = cls(
            transform_id=payload.get("transform_id", ""),
            transform_version=payload.get(
                "transform_version",
                "development",
            ),
            source_frame_id=payload.get("source_frame_id", ""),
            target_frame_id=payload.get("target_frame_id", ""),
            matrix_target_source=payload.get("matrix_target_source"),
            semantic_parameters=dict(payload.get("semantic_parameters", {})),
        )
        _verify_optional_hash(
            payload,
            expected=value.semantic_content_hash,
            subject=f"rigid transform {value.transform_id!r}",
        )
        return value


@dataclass(frozen=True)
class FrameTransformTraversal:
    """One directed use of a registered transform edge."""

    transform_id: str
    transform_hash: str
    direction: str
    source_frame_id: str
    target_frame_id: str

    def __post_init__(self) -> None:
        for name in (
            "transform_id",
            "transform_hash",
            "source_frame_id",
            "target_frame_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        direction = _required(self.direction, name="direction")
        if direction not in {"forward", "inverse"}:
            raise CoordinateFrameContractError(
                "Frame transform traversal direction must be forward or inverse."
            )
        object.__setattr__(self, "direction", direction)

    def semantic_payload(self) -> dict[str, str]:
        return {
            "transform_id": self.transform_id,
            "transform_hash": self.transform_hash,
            "direction": self.direction,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
        }


@dataclass(frozen=True)
class ResolvedFrameTransform:
    """Deterministic graph resolution receipt for one frame pair."""

    source_frame_id: str
    target_frame_id: str
    matrix_target_source: np.ndarray
    traversals: tuple[FrameTransformTraversal, ...]
    frame_graph_hash: str
    semantic_content_hash: str = field(init=False)
    resolution_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = _required(self.source_frame_id, name="source_frame_id")
        target = _required(self.target_frame_id, name="target_frame_id")
        graph_hash = _required(self.frame_graph_hash, name="frame_graph_hash")
        matrix = _rigid_transform(
            self.matrix_target_source,
            name="matrix_target_source",
        )
        traversals = tuple(self.traversals)
        if any(not isinstance(item, FrameTransformTraversal) for item in traversals):
            raise TypeError(
                "traversals must contain FrameTransformTraversal values."
            )
        if source == target:
            if traversals:
                raise CoordinateFrameContractError(
                    "Identity frame resolution must not contain traversals."
                )
            if not np.allclose(matrix, np.eye(4), rtol=0.0, atol=1e-12):
                raise CoordinateFrameContractError(
                    "Identity frame resolution must return an identity matrix."
                )
        elif not traversals:
            raise CoordinateFrameContractError(
                "A non-identity frame resolution requires at least one traversal."
            )
        if traversals:
            if traversals[0].source_frame_id != source:
                raise CoordinateFrameContractError(
                    "Frame resolution path does not start at source_frame_id."
                )
            if traversals[-1].target_frame_id != target:
                raise CoordinateFrameContractError(
                    "Frame resolution path does not end at target_frame_id."
                )
            for previous, current in zip(traversals, traversals[1:]):
                if previous.target_frame_id != current.source_frame_id:
                    raise CoordinateFrameContractError(
                        "Frame resolution traversals are not contiguous."
                    )
        payload = {
            "source_frame_id": source,
            "target_frame_id": target,
            "matrix_convention": "T_target_source",
            "matrix_target_source": matrix,
            "traversals": tuple(item.semantic_payload() for item in traversals),
            "frame_graph_hash": graph_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RESOLVED_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "target_frame_id", target)
        object.__setattr__(self, "matrix_target_source", matrix)
        object.__setattr__(self, "traversals", traversals)
        object.__setattr__(self, "frame_graph_hash", graph_hash)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "resolution_id",
            f"resolved-frame-transform:{semantic_hash}",
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "matrix_convention": "T_target_source",
            "matrix_target_source": self.matrix_target_source,
            "traversals": tuple(
                item.semantic_payload() for item in self.traversals
            ),
            "frame_graph_hash": self.frame_graph_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        payload = self.semantic_payload()
        payload["matrix_target_source"] = self.matrix_target_source.tolist()
        return payload


@dataclass(frozen=True)
class ResolvedPoseFrameTransform:
    """Two-sided frame resolution for one pose representation.

    For an input ``T_source_reference_source_controlled`` the output is:

    ``T_target_reference_source_reference @ input
      @ T_source_controlled_target_controlled``.
    """

    source_reference_frame_id: str
    source_controlled_frame_id: str
    target_reference_frame_id: str
    target_controlled_frame_id: str
    reference_resolution: ResolvedFrameTransform
    controlled_resolution: ResolvedFrameTransform
    frame_graph_hash: str
    semantic_content_hash: str = field(init=False)
    resolution_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "source_reference_frame_id",
                "source_controlled_frame_id",
                "target_reference_frame_id",
                "target_controlled_frame_id",
                "frame_graph_hash",
            )
        }
        if not isinstance(self.reference_resolution, ResolvedFrameTransform):
            raise TypeError(
                "reference_resolution must be ResolvedFrameTransform."
            )
        if not isinstance(self.controlled_resolution, ResolvedFrameTransform):
            raise TypeError(
                "controlled_resolution must be ResolvedFrameTransform."
            )
        reference = self.reference_resolution
        controlled = self.controlled_resolution
        if (
            reference.source_frame_id
            != values["source_reference_frame_id"]
            or reference.target_frame_id
            != values["target_reference_frame_id"]
        ):
            raise CoordinateFrameContractError(
                "Pose reference-frame resolution direction is inconsistent."
            )
        # The controlled-frame transform is the right factor
        # T_source_controlled_target_controlled.
        if (
            controlled.source_frame_id
            != values["target_controlled_frame_id"]
            or controlled.target_frame_id
            != values["source_controlled_frame_id"]
        ):
            raise CoordinateFrameContractError(
                "Pose controlled-frame resolution direction is inconsistent."
            )
        if (
            reference.frame_graph_hash != values["frame_graph_hash"]
            or controlled.frame_graph_hash != values["frame_graph_hash"]
        ):
            raise CoordinateFrameContractError(
                "Pose frame resolutions belong to another coordinate graph."
            )
        payload = {
            **values,
            "reference_resolution_id": reference.resolution_id,
            "reference_resolution_hash": reference.semantic_content_hash,
            "controlled_resolution_id": controlled.resolution_id,
            "controlled_resolution_hash": controlled.semantic_content_hash,
            "pose_transform_equation": (
                "T_target_reference_source_reference"
                "@T_source_reference_source_controlled"
                "@T_source_controlled_target_controlled"
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=RESOLVED_POSE_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT,
        )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "resolution_id",
            f"resolved-pose-frame-transform:{semantic_hash}",
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "resolution_id": self.resolution_id,
            "source_reference_frame_id": self.source_reference_frame_id,
            "source_controlled_frame_id": self.source_controlled_frame_id,
            "target_reference_frame_id": self.target_reference_frame_id,
            "target_controlled_frame_id": self.target_controlled_frame_id,
            "reference_resolution_id": self.reference_resolution.resolution_id,
            "reference_resolution_hash": (
                self.reference_resolution.semantic_content_hash
            ),
            "controlled_resolution_id": (
                self.controlled_resolution.resolution_id
            ),
            "controlled_resolution_hash": (
                self.controlled_resolution.semantic_content_hash
            ),
            "frame_graph_hash": self.frame_graph_hash,
            "pose_transform_equation": (
                "T_target_reference_source_reference"
                "@T_source_reference_source_controlled"
                "@T_source_controlled_target_controlled"
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class _AdjacentTransform:
    neighbor_frame_id: str
    transform: RigidFrameTransformSpec
    direction: str
    matrix_neighbor_current: np.ndarray


@dataclass(frozen=True)
class CoordinateFrameGraph:
    """Immutable, cycle-consistent graph of rigid frame relationships."""

    graph_id: str
    frames: tuple[CoordinateFrameSpec, ...]
    transforms: tuple[RigidFrameTransformSpec, ...] = ()
    graph_version: str = "development"
    consistency_atol: float = 1e-8
    semantic_content_hash: str = field(init=False)
    frame_by_id: Mapping[str, CoordinateFrameSpec] = field(init=False, repr=False)
    transform_by_id: Mapping[str, RigidFrameTransformSpec] = field(
        init=False,
        repr=False,
    )
    _adjacency: Mapping[str, tuple[_AdjacentTransform, ...]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        graph_id = _required(self.graph_id, name="graph_id")
        version = _required(self.graph_version, name="graph_version")
        try:
            tolerance = float(self.consistency_atol)
        except (TypeError, ValueError) as exc:
            raise CoordinateFrameContractError(
                "consistency_atol must be finite and non-negative."
            ) from exc
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise CoordinateFrameContractError(
                "consistency_atol must be finite and non-negative."
            )
        frames = tuple(self.frames)
        if not frames:
            raise CoordinateFrameContractError(
                "A coordinate frame graph requires at least one frame."
            )
        if any(not isinstance(item, CoordinateFrameSpec) for item in frames):
            raise TypeError("frames must contain CoordinateFrameSpec values.")
        frame_by_id = {item.frame_id: item for item in frames}
        if len(frame_by_id) != len(frames):
            raise CoordinateFrameContractError(
                "Coordinate frame IDs must be unique."
            )
        transforms = tuple(self.transforms)
        if any(
            not isinstance(item, RigidFrameTransformSpec)
            for item in transforms
        ):
            raise TypeError(
                "transforms must contain RigidFrameTransformSpec values."
            )
        transform_by_id = {item.transform_id: item for item in transforms}
        if len(transform_by_id) != len(transforms):
            raise CoordinateFrameContractError(
                "Rigid frame transform IDs must be unique."
            )
        unknown = sorted(
            {
                frame_id
                for transform in transforms
                for frame_id in (
                    transform.source_frame_id,
                    transform.target_frame_id,
                )
                if frame_id not in frame_by_id
            }
        )
        if unknown:
            raise CoordinateFrameContractError(
                f"Rigid transforms reference unknown frames: {unknown}."
            )
        ordered_frames = tuple(sorted(frames, key=lambda item: item.frame_id))
        ordered_transforms = tuple(
            sorted(transforms, key=lambda item: item.transform_id)
        )
        adjacency = self._build_adjacency(
            frame_by_id=frame_by_id,
            transforms=ordered_transforms,
        )
        self._validate_cycle_consistency(
            adjacency=adjacency,
            tolerance=tolerance,
        )
        payload = {
            "graph_id": graph_id,
            "graph_version": version,
            "consistency_atol": tolerance,
            "frames": tuple(item.semantic_payload() for item in ordered_frames),
            "transforms": tuple(
                item.semantic_payload() for item in ordered_transforms
            ),
        }
        object.__setattr__(self, "graph_id", graph_id)
        object.__setattr__(self, "graph_version", version)
        object.__setattr__(self, "consistency_atol", tolerance)
        object.__setattr__(self, "frames", ordered_frames)
        object.__setattr__(self, "transforms", ordered_transforms)
        object.__setattr__(
            self,
            "frame_by_id",
            MappingProxyType(dict(frame_by_id)),
        )
        object.__setattr__(
            self,
            "transform_by_id",
            MappingProxyType(dict(transform_by_id)),
        )
        object.__setattr__(
            self,
            "_adjacency",
            MappingProxyType(adjacency),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COORDINATE_FRAME_GRAPH_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @staticmethod
    def _build_adjacency(
        *,
        frame_by_id: Mapping[str, CoordinateFrameSpec],
        transforms: Sequence[RigidFrameTransformSpec],
    ) -> dict[str, tuple[_AdjacentTransform, ...]]:
        mutable: dict[str, list[_AdjacentTransform]] = {
            frame_id: [] for frame_id in frame_by_id
        }
        for transform in transforms:
            mutable[transform.source_frame_id].append(
                _AdjacentTransform(
                    neighbor_frame_id=transform.target_frame_id,
                    transform=transform,
                    direction="forward",
                    matrix_neighbor_current=transform.matrix_target_source,
                )
            )
            mutable[transform.target_frame_id].append(
                _AdjacentTransform(
                    neighbor_frame_id=transform.source_frame_id,
                    transform=transform,
                    direction="inverse",
                    matrix_neighbor_current=_rigid_transform(
                        np.linalg.inv(transform.matrix_target_source),
                        name=f"inverse({transform.transform_id})",
                    ),
                )
            )
        return {
            frame_id: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.neighbor_frame_id,
                        item.transform.transform_id,
                        item.direction,
                    ),
                )
            )
            for frame_id, values in mutable.items()
        }

    @staticmethod
    def _validate_cycle_consistency(
        *,
        adjacency: Mapping[str, tuple[_AdjacentTransform, ...]],
        tolerance: float,
    ) -> None:
        transform_from_root: dict[str, np.ndarray] = {}
        for root in sorted(adjacency):
            if root in transform_from_root:
                continue
            transform_from_root[root] = np.eye(4, dtype=np.float64)
            queue: deque[str] = deque((root,))
            while queue:
                current = queue.popleft()
                current_from_root = transform_from_root[current]
                for adjacent in adjacency[current]:
                    candidate = (
                        adjacent.matrix_neighbor_current @ current_from_root
                    )
                    existing = transform_from_root.get(
                        adjacent.neighbor_frame_id
                    )
                    if existing is None:
                        transform_from_root[
                            adjacent.neighbor_frame_id
                        ] = candidate
                        queue.append(adjacent.neighbor_frame_id)
                        continue
                    if not np.allclose(
                        existing,
                        candidate,
                        rtol=0.0,
                        atol=tolerance,
                    ):
                        raise CoordinateFrameContractError(
                            "Inconsistent rigid-transform cycle involving "
                            f"{current!r}, {adjacent.neighbor_frame_id!r}, and "
                            f"edge {adjacent.transform.transform_id!r}."
                        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "consistency_atol": self.consistency_atol,
            "frames": tuple(item.semantic_payload() for item in self.frames),
            "transforms": tuple(
                item.semantic_payload() for item in self.transforms
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "graph_version": self.graph_version,
            "consistency_atol": self.consistency_atol,
            "frames": [item.to_payload() for item in self.frames],
            "transforms": [item.to_payload() for item in self.transforms],
            "semantic_content_hash": self.semantic_content_hash,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CoordinateFrameGraph":
        raw_frames = payload.get("frames", ())
        raw_transforms = payload.get("transforms", ())
        if not isinstance(raw_frames, Sequence) or isinstance(
            raw_frames,
            (str, bytes),
        ):
            raise CoordinateFrameContractError(
                "Coordinate frame graph frames must be a sequence."
            )
        if not isinstance(raw_transforms, Sequence) or isinstance(
            raw_transforms,
            (str, bytes),
        ):
            raise CoordinateFrameContractError(
                "Coordinate frame graph transforms must be a sequence."
            )
        if any(not isinstance(item, Mapping) for item in raw_frames):
            raise CoordinateFrameContractError(
                "Coordinate frame graph frame payloads must be mappings."
            )
        if any(not isinstance(item, Mapping) for item in raw_transforms):
            raise CoordinateFrameContractError(
                "Coordinate frame graph transform payloads must be mappings."
            )
        value = cls(
            graph_id=payload.get("graph_id", ""),
            graph_version=payload.get("graph_version", "development"),
            consistency_atol=payload.get("consistency_atol", 1e-8),
            frames=tuple(
                CoordinateFrameSpec.from_payload(item) for item in raw_frames
            ),
            transforms=tuple(
                RigidFrameTransformSpec.from_payload(item)
                for item in raw_transforms
            ),
        )
        _verify_optional_hash(
            payload,
            expected=value.semantic_content_hash,
            subject=f"coordinate frame graph {value.graph_id!r}",
        )
        return value

    def connected(self, first_frame_id: str, second_frame_id: str) -> bool:
        first = _required(first_frame_id, name="first_frame_id")
        second = _required(second_frame_id, name="second_frame_id")
        self._require_frame(first)
        self._require_frame(second)
        if first == second:
            return True
        visited = {first}
        queue: deque[str] = deque((first,))
        while queue:
            current = queue.popleft()
            for adjacent in self._adjacency[current]:
                neighbor = adjacent.neighbor_frame_id
                if neighbor == second:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def resolve(
        self,
        source_frame_id: str,
        target_frame_id: str,
    ) -> ResolvedFrameTransform:
        """Resolve deterministic ``T_target_source`` and its exact edge path."""

        source = _required(source_frame_id, name="source_frame_id")
        target = _required(target_frame_id, name="target_frame_id")
        self._require_frame(source)
        self._require_frame(target)
        if source == target:
            return ResolvedFrameTransform(
                source_frame_id=source,
                target_frame_id=target,
                matrix_target_source=np.eye(4, dtype=np.float64),
                traversals=(),
                frame_graph_hash=self.semantic_content_hash,
            )
        queue: deque[str] = deque((source,))
        matrix_from_source: dict[str, np.ndarray] = {
            source: np.eye(4, dtype=np.float64)
        }
        parent: dict[str, tuple[str, _AdjacentTransform]] = {}
        while queue:
            current = queue.popleft()
            for adjacent in self._adjacency[current]:
                neighbor = adjacent.neighbor_frame_id
                if neighbor in matrix_from_source:
                    continue
                matrix_from_source[neighbor] = (
                    adjacent.matrix_neighbor_current
                    @ matrix_from_source[current]
                )
                parent[neighbor] = (current, adjacent)
                if neighbor == target:
                    queue.clear()
                    break
                queue.append(neighbor)
        if target not in matrix_from_source:
            raise CoordinateFrameContractError(
                f"Frames {source!r} and {target!r} are not connected."
            )
        reverse_path: list[FrameTransformTraversal] = []
        cursor = target
        while cursor != source:
            previous, adjacent = parent[cursor]
            reverse_path.append(
                FrameTransformTraversal(
                    transform_id=adjacent.transform.transform_id,
                    transform_hash=adjacent.transform.semantic_content_hash,
                    direction=adjacent.direction,
                    source_frame_id=previous,
                    target_frame_id=cursor,
                )
            )
            cursor = previous
        return ResolvedFrameTransform(
            source_frame_id=source,
            target_frame_id=target,
            matrix_target_source=matrix_from_source[target],
            traversals=tuple(reversed(reverse_path)),
            frame_graph_hash=self.semantic_content_hash,
        )

    def with_connected_frame(
        self,
        *,
        frame: CoordinateFrameSpec,
        transform: RigidFrameTransformSpec,
    ) -> "CoordinateFrameGraph":
        """Return a new graph with one frame linked by one rigid edge."""

        if not isinstance(frame, CoordinateFrameSpec):
            raise TypeError("frame must be CoordinateFrameSpec.")
        if not isinstance(transform, RigidFrameTransformSpec):
            raise TypeError("transform must be RigidFrameTransformSpec.")
        if frame.frame_id in self.frame_by_id:
            raise CoordinateFrameContractError(
                f"Frame {frame.frame_id!r} is already registered."
            )
        endpoints = {
            transform.source_frame_id,
            transform.target_frame_id,
        }
        if frame.frame_id not in endpoints:
            raise CoordinateFrameContractError(
                "The extension transform must reference the new frame."
            )
        other = next(item for item in endpoints if item != frame.frame_id)
        if other not in self.frame_by_id:
            raise CoordinateFrameContractError(
                "The extension transform must connect the new frame to one "
                "already registered frame."
            )
        return CoordinateFrameGraph(
            graph_id=self.graph_id,
            graph_version=self.graph_version,
            consistency_atol=self.consistency_atol,
            frames=(*self.frames, frame),
            transforms=(*self.transforms, transform),
        )

    def with_transform(
        self,
        transform: RigidFrameTransformSpec,
    ) -> "CoordinateFrameGraph":
        """Return a new graph with one additional cycle-checked edge."""

        if not isinstance(transform, RigidFrameTransformSpec):
            raise TypeError("transform must be RigidFrameTransformSpec.")
        return CoordinateFrameGraph(
            graph_id=self.graph_id,
            graph_version=self.graph_version,
            consistency_atol=self.consistency_atol,
            frames=self.frames,
            transforms=(*self.transforms, transform),
        )

    def transform_pose_matrix(
        self,
        pose_source_controlled: object,
        *,
        source_reference_frame_id: str,
        target_reference_frame_id: str,
    ) -> tuple[np.ndarray, ResolvedFrameTransform]:
        """Transform ``T_source_controlled`` into ``T_target_controlled``."""

        pose = _rigid_transform(
            pose_source_controlled,
            name="pose_source_controlled",
        )
        resolution = self.resolve(
            source_reference_frame_id,
            target_reference_frame_id,
        )
        transformed = _rigid_transform(
            resolution.matrix_target_source @ pose,
            name="pose_target_controlled",
        )
        return transformed, resolution

    def transform_pose_between_frames(
        self,
        pose_source_reference_source_controlled: object,
        *,
        source_reference_frame_id: str,
        source_controlled_frame_id: str,
        target_reference_frame_id: str,
        target_controlled_frame_id: str,
    ) -> tuple[np.ndarray, ResolvedPoseFrameTransform]:
        """Transform both the reference and controlled frames of one pose."""

        pose = _rigid_transform(
            pose_source_reference_source_controlled,
            name="pose_source_reference_source_controlled",
        )
        reference = self.resolve(
            source_reference_frame_id,
            target_reference_frame_id,
        )
        controlled = self.resolve(
            target_controlled_frame_id,
            source_controlled_frame_id,
        )
        transformed = _rigid_transform(
            reference.matrix_target_source
            @ pose
            @ controlled.matrix_target_source,
            name="pose_target_reference_target_controlled",
        )
        resolution = ResolvedPoseFrameTransform(
            source_reference_frame_id=source_reference_frame_id,
            source_controlled_frame_id=source_controlled_frame_id,
            target_reference_frame_id=target_reference_frame_id,
            target_controlled_frame_id=target_controlled_frame_id,
            reference_resolution=reference,
            controlled_resolution=controlled,
            frame_graph_hash=self.semantic_content_hash,
        )
        return transformed, resolution

    def _require_frame(self, frame_id: str) -> CoordinateFrameSpec:
        try:
            return self.frame_by_id[frame_id]
        except KeyError as exc:
            raise CoordinateFrameContractError(
                f"Unknown coordinate frame {frame_id!r}."
            ) from exc


def _verify_optional_hash(
    payload: Mapping[str, Any],
    *,
    expected: str,
    subject: str,
) -> None:
    declared = payload.get("semantic_content_hash")
    if declared is None:
        return
    normalized = _required(
        declared,
        name=f"{subject} semantic_content_hash",
    )
    if normalized != expected:
        raise CoordinateFrameContractError(
            f"{subject} semantic_content_hash mismatch."
        )


__all__ = [
    "COORDINATE_FRAME_GRAPH_HASH_DOMAIN_DEVELOPMENT",
    "COORDINATE_FRAME_SPEC_HASH_DOMAIN_DEVELOPMENT",
    "RESOLVED_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT",
    "RESOLVED_POSE_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT",
    "RIGID_FRAME_TRANSFORM_HASH_DOMAIN_DEVELOPMENT",
    "CoordinateFrameContractError",
    "CoordinateFrameGraph",
    "CoordinateFrameSpec",
    "FrameTransformTraversal",
    "ResolvedFrameTransform",
    "ResolvedPoseFrameTransform",
    "RigidFrameTransformSpec",
]
