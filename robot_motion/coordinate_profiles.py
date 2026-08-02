"""Target-coordinate profiles and activation-local pose anchor receipts.

The contracts in this module sit below concrete simulation, hardware, and
teleoperation adapters:

* :class:`TargetFrameProfile` binds one target artifact to its fixed reference
  frames.
* :class:`ToolFrameProfile` binds controlled-frame semantics to exact tool
  artifacts.
* :class:`SessionAnchorReceipt` records the source Home and measured target
  Home selected for one activation.

They do not perform retargeting, composition, safety, IK, or dispatch.
Hash domains remain ``development`` while these additive contracts settle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.runtime.immutable import deep_freeze, deep_thaw


TARGET_FRAME_PROFILE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.target_frame_profile.development"
)
TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.tool_frame_binding.development"
)
TOOL_FRAME_PROFILE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.tool_frame_profile.development"
)
SESSION_POSE_ANCHOR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.session_pose_anchor.development"
)
SESSION_ANCHOR_RECEIPT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.session_anchor_receipt.development"
)
STATION_RETARGET_FRAME_CONTRACT_V1 = (
    "station_retarget_x_forward_y_left_z_up_v1"
)


def _required(value: object, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 pose.") from exc
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 pose.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} must be homogeneous.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} rotation determinant must be +1.")
    return deep_freeze(np.ascontiguousarray(matrix, dtype=np.float64))


def _component_mapping(
    value: Mapping[str, str],
    *,
    name: str,
) -> Mapping[str, str]:
    result = {
        _required(component_id, name=f"{name}.component_id"): _required(
            frame_id,
            name=f"{name}.{component_id}",
        )
        for component_id, frame_id in dict(value).items()
    }
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class TargetFrameProfile:
    """Fixed target reference-frame geometry derived from one exact artifact."""

    profile_id: str
    target_kind: str
    artifact_id: str
    artifact_hash: str
    frame_graph: CoordinateFrameGraph
    canonical_reference_frame_id: str
    reference_frame_by_component: Mapping[str, str]
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_graph, CoordinateFrameGraph):
            raise TypeError("frame_graph must be CoordinateFrameGraph.")
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "profile_id",
                "target_kind",
                "artifact_id",
                "artifact_hash",
                "canonical_reference_frame_id",
            )
        }
        frames = _component_mapping(
            self.reference_frame_by_component,
            name="reference_frame_by_component",
        )
        if text["canonical_reference_frame_id"] not in self.frame_graph.frame_by_id:
            raise ValueError(
                "canonical_reference_frame_id is absent from frame_graph."
            )
        for component_id, frame_id in frames.items():
            if frame_id not in self.frame_graph.frame_by_id:
                raise ValueError(
                    f"Target reference frame {frame_id!r} for "
                    f"{component_id!r} is absent from frame_graph."
                )
            self.frame_graph.resolve(
                frame_id,
                text["canonical_reference_frame_id"],
            )
        parameters = deep_freeze(dict(self.semantic_parameters))
        payload = {
            **text,
            "frame_graph_hash": self.frame_graph.semantic_content_hash,
            "reference_frame_by_component": frames,
            "semantic_parameters": parameters,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "reference_frame_by_component", frames)
        object.__setattr__(self, "semantic_parameters", parameters)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_FRAME_PROFILE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "target_kind": self.target_kind,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "frame_graph_hash": self.frame_graph.semantic_content_hash,
            "canonical_reference_frame_id": self.canonical_reference_frame_id,
            "reference_frame_by_component": dict(
                self.reference_frame_by_component
            ),
            "semantic_parameters": deep_thaw(self.semantic_parameters),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ToolFrameBinding:
    """Bind one action component's controlled frame to a tool artifact."""

    component_id: str
    controlled_frame_id: str
    artifact_id: str
    artifact_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "controlled_frame_id",
                "artifact_id",
                "artifact_hash",
            )
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "component_id": self.component_id,
            "controlled_frame_id": self.controlled_frame_id,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ToolFrameProfile:
    """Exact controlled-frame artifact set selected by one target profile."""

    profile_id: str
    target_frame_profile_hash: str
    bindings: tuple[ToolFrameBinding, ...]
    semantic_content_hash: str = field(init=False)
    binding_by_component: Mapping[str, ToolFrameBinding] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        target_hash = _required(
            self.target_frame_profile_hash,
            name="target_frame_profile_hash",
        )
        bindings = tuple(self.bindings)
        if not bindings or any(
            not isinstance(item, ToolFrameBinding) for item in bindings
        ):
            raise TypeError(
                "bindings must contain at least one ToolFrameBinding."
            )
        by_component = {item.component_id: item for item in bindings}
        if len(by_component) != len(bindings):
            raise ValueError("Tool-frame components must be unique.")
        ordered = tuple(by_component[key] for key in sorted(by_component))
        payload = {
            "profile_id": profile_id,
            "target_frame_profile_hash": target_hash,
            "binding_hash_by_component": {
                item.component_id: item.semantic_content_hash
                for item in ordered
            },
        }
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "target_frame_profile_hash", target_hash)
        object.__setattr__(self, "bindings", ordered)
        object.__setattr__(
            self,
            "binding_by_component",
            MappingProxyType(by_component),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TOOL_FRAME_PROFILE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "target_frame_profile_hash": self.target_frame_profile_hash,
            "bindings": [item.to_payload() for item in self.bindings],
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class SessionPoseAnchor:
    """One source Home paired with one measured target Home."""

    component_id: str
    controlled_frame_id: str
    target_reference_frame_id: str
    source_home_pose_canonical: np.ndarray
    target_home_pose_canonical: np.ndarray
    target_home_pose_reference: np.ndarray
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "component_id",
                "controlled_frame_id",
                "target_reference_frame_id",
            )
        }
        source_home = _pose(
            self.source_home_pose_canonical,
            name="source_home_pose_canonical",
        )
        target_home_canonical = _pose(
            self.target_home_pose_canonical,
            name="target_home_pose_canonical",
        )
        target_home_reference = _pose(
            self.target_home_pose_reference,
            name="target_home_pose_reference",
        )
        payload = {
            **text,
            "source_home_pose_canonical": source_home,
            "target_home_pose_canonical": target_home_canonical,
            "target_home_pose_reference": target_home_reference,
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_home_pose_canonical", source_home)
        object.__setattr__(
            self,
            "target_home_pose_canonical",
            target_home_canonical,
        )
        object.__setattr__(
            self,
            "target_home_pose_reference",
            target_home_reference,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=SESSION_POSE_ANCHOR_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "controlled_frame_id": self.controlled_frame_id,
            "target_reference_frame_id": self.target_reference_frame_id,
            "source_home_pose_canonical": self.source_home_pose_canonical.tolist(),
            "target_home_pose_canonical": self.target_home_pose_canonical.tolist(),
            "target_home_pose_reference": self.target_home_pose_reference.tolist(),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class SessionAnchorReceipt:
    """Activation-local source/target Home binding with exact profile identity."""

    session_id: str
    coordinate_profile_id: str
    coordinate_profile_hash: str
    target_frame_profile_id: str
    target_frame_profile_hash: str
    tool_frame_profile_id: str
    tool_frame_profile_hash: str
    canonical_reference_frame_id: str
    source_anchor_origin: str
    target_anchor_origin: str
    captured_at: float
    clock_domain_id: str
    anchors: tuple[SessionPoseAnchor, ...]
    semantic_content_hash: str = field(init=False)
    receipt_id: str = field(init=False)
    anchor_by_component: Mapping[str, SessionPoseAnchor] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "session_id",
                "coordinate_profile_id",
                "coordinate_profile_hash",
                "target_frame_profile_id",
                "target_frame_profile_hash",
                "tool_frame_profile_id",
                "tool_frame_profile_hash",
                "canonical_reference_frame_id",
                "source_anchor_origin",
                "target_anchor_origin",
                "clock_domain_id",
            )
        }
        captured_at = float(self.captured_at)
        if not math.isfinite(captured_at):
            raise ValueError("captured_at must be finite.")
        anchors = tuple(self.anchors)
        if not anchors or any(
            not isinstance(item, SessionPoseAnchor) for item in anchors
        ):
            raise TypeError(
                "anchors must contain at least one SessionPoseAnchor."
            )
        by_component = {item.component_id: item for item in anchors}
        if len(by_component) != len(anchors):
            raise ValueError("Session anchor components must be unique.")
        ordered = tuple(by_component[key] for key in sorted(by_component))
        payload = {
            **text,
            "captured_at": captured_at,
            "anchor_hash_by_component": {
                item.component_id: item.semantic_content_hash
                for item in ordered
            },
        }
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "captured_at", captured_at)
        object.__setattr__(self, "anchors", ordered)
        object.__setattr__(
            self,
            "anchor_by_component",
            MappingProxyType(by_component),
        )
        semantic_hash = canonical_content_hash(
            payload,
            domain=SESSION_ANCHOR_RECEIPT_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "receipt_id",
            f"session-anchor-receipt:{semantic_hash}",
        )

    @classmethod
    def capture(
        cls,
        *,
        session_id: str,
        coordinate_profile_id: str,
        coordinate_profile_hash: str,
        target_frame_profile: TargetFrameProfile,
        tool_frame_profile: ToolFrameProfile,
        source_anchor_origin: str,
        target_anchor_origin: str,
        captured_at: float,
        clock_domain_id: str,
        anchors: Sequence[SessionPoseAnchor],
    ) -> "SessionAnchorReceipt":
        if not isinstance(target_frame_profile, TargetFrameProfile):
            raise TypeError(
                "target_frame_profile must be TargetFrameProfile."
            )
        if not isinstance(tool_frame_profile, ToolFrameProfile):
            raise TypeError("tool_frame_profile must be ToolFrameProfile.")
        if (
            tool_frame_profile.target_frame_profile_hash
            != target_frame_profile.semantic_content_hash
        ):
            raise ValueError(
                "Tool-frame profile belongs to another target-frame profile."
            )
        entries = tuple(anchors)
        expected = set(target_frame_profile.reference_frame_by_component)
        if {item.component_id for item in entries} != expected:
            raise ValueError(
                "Session anchors must exactly cover target pose components."
            )
        if set(tool_frame_profile.binding_by_component) != expected:
            raise ValueError(
                "Tool-frame profile must exactly cover target pose components."
            )
        for entry in entries:
            expected_reference = (
                target_frame_profile.reference_frame_by_component[
                    entry.component_id
                ]
            )
            if entry.target_reference_frame_id != expected_reference:
                raise ValueError(
                    f"Session anchor reference frame mismatch for "
                    f"{entry.component_id!r}."
                )
            expected_controlled = tool_frame_profile.binding_by_component[
                entry.component_id
            ].controlled_frame_id
            if entry.controlled_frame_id != expected_controlled:
                raise ValueError(
                    f"Session anchor controlled frame mismatch for "
                    f"{entry.component_id!r}."
                )
            resolved = target_frame_profile.frame_graph.resolve(
                entry.target_reference_frame_id,
                target_frame_profile.canonical_reference_frame_id,
            )
            reconstructed = (
                resolved.matrix_target_source
                @ entry.target_home_pose_reference
            )
            if not np.allclose(
                reconstructed,
                entry.target_home_pose_canonical,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"Session target Home transform mismatch for "
                    f"{entry.component_id!r}."
                )
        return cls(
            session_id=session_id,
            coordinate_profile_id=coordinate_profile_id,
            coordinate_profile_hash=coordinate_profile_hash,
            target_frame_profile_id=target_frame_profile.profile_id,
            target_frame_profile_hash=(
                target_frame_profile.semantic_content_hash
            ),
            tool_frame_profile_id=tool_frame_profile.profile_id,
            tool_frame_profile_hash=tool_frame_profile.semantic_content_hash,
            canonical_reference_frame_id=(
                target_frame_profile.canonical_reference_frame_id
            ),
            source_anchor_origin=source_anchor_origin,
            target_anchor_origin=target_anchor_origin,
            captured_at=captured_at,
            clock_domain_id=clock_domain_id,
            anchors=entries,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "session_id": self.session_id,
            "coordinate_profile_id": self.coordinate_profile_id,
            "coordinate_profile_hash": self.coordinate_profile_hash,
            "target_frame_profile_id": self.target_frame_profile_id,
            "target_frame_profile_hash": self.target_frame_profile_hash,
            "tool_frame_profile_id": self.tool_frame_profile_id,
            "tool_frame_profile_hash": self.tool_frame_profile_hash,
            "canonical_reference_frame_id": self.canonical_reference_frame_id,
            "source_anchor_origin": self.source_anchor_origin,
            "target_anchor_origin": self.target_anchor_origin,
            "captured_at": self.captured_at,
            "clock_domain_id": self.clock_domain_id,
            "anchors": [item.to_payload() for item in self.anchors],
            "semantic_content_hash": self.semantic_content_hash,
        }


__all__ = [
    "SESSION_ANCHOR_RECEIPT_HASH_DOMAIN_DEVELOPMENT",
    "SESSION_POSE_ANCHOR_HASH_DOMAIN_DEVELOPMENT",
    "STATION_RETARGET_FRAME_CONTRACT_V1",
    "TARGET_FRAME_PROFILE_HASH_DOMAIN_DEVELOPMENT",
    "TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "TOOL_FRAME_PROFILE_HASH_DOMAIN_DEVELOPMENT",
    "SessionAnchorReceipt",
    "SessionPoseAnchor",
    "TargetFrameProfile",
    "ToolFrameBinding",
    "ToolFrameProfile",
]
