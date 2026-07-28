"""Bind spatial action schemas to exact coordinate-frame identities.

``ActionPayloadSchema`` already declares frame *names*. This additive binding
links those names to immutable ``CoordinateFrameSpec`` hashes in one concrete
frame graph, preventing two unrelated frames with the same label from being
treated as equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import (
    CoordinateFrameGraph,
    CoordinateFrameSpec,
)

from .component_action import ActionPayloadSchema


ACTION_COORDINATE_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.action_coordinate_binding.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class CoordinateFrameReference:
    frame_id: str
    frame_spec_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_id",
            _required(self.frame_id, name="frame_id"),
        )
        object.__setattr__(
            self,
            "frame_spec_hash",
            _required(self.frame_spec_hash, name="frame_spec_hash"),
        )

    @classmethod
    def from_spec(cls, value: CoordinateFrameSpec) -> "CoordinateFrameReference":
        if not isinstance(value, CoordinateFrameSpec):
            raise TypeError("value must be CoordinateFrameSpec.")
        return cls(
            frame_id=value.frame_id,
            frame_spec_hash=value.semantic_content_hash,
        )

    def semantic_payload(self) -> dict[str, str]:
        return {
            "frame_id": self.frame_id,
            "frame_spec_hash": self.frame_spec_hash,
        }


@dataclass(frozen=True)
class ActionCoordinateBinding:
    """Exact spatial-frame binding for one action payload schema."""

    action_payload_schema_hash: str
    representation_family: str
    value_semantics: str
    frame_graph_hash: str
    declared_reference_frame_id: str | None
    declared_controlled_frame_id: str | None
    declared_expressed_in_frame_id: str | None
    reference_frame: CoordinateFrameReference | None
    controlled_frame: CoordinateFrameReference | None
    expressed_in_frame: CoordinateFrameReference | None
    semantic_content_hash: str = field(init=False)
    binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        references = (
            self.reference_frame,
            self.controlled_frame,
            self.expressed_in_frame,
        )
        declared = (
            self.declared_reference_frame_id,
            self.declared_controlled_frame_id,
            self.declared_expressed_in_frame_id,
        )
        if not any(item is not None for item in references):
            raise ValueError(
                "ActionCoordinateBinding requires at least one spatial frame."
            )
        if any(
            item is not None and not isinstance(item, CoordinateFrameReference)
            for item in references
        ):
            raise TypeError(
                "Action coordinate references must be CoordinateFrameReference "
                "values or None."
            )
        normalized_declared = tuple(
            None if item is None else _required(item, name="declared_frame_id")
            for item in declared
        )
        if tuple(item is None for item in normalized_declared) != tuple(
            item is None for item in references
        ):
            raise ValueError(
                "Every declared action frame must have exactly one concrete "
                "coordinate-frame reference."
            )
        payload = {
            "action_payload_schema_hash": _required(
                self.action_payload_schema_hash,
                name="action_payload_schema_hash",
            ),
            "representation_family": _required(
                self.representation_family,
                name="representation_family",
            ),
            "value_semantics": _required(
                self.value_semantics,
                name="value_semantics",
            ),
            "frame_graph_hash": _required(
                self.frame_graph_hash,
                name="frame_graph_hash",
            ),
            "declared_reference_frame_id": normalized_declared[0],
            "declared_controlled_frame_id": normalized_declared[1],
            "declared_expressed_in_frame_id": normalized_declared[2],
            "reference_frame": (
                None
                if self.reference_frame is None
                else self.reference_frame.semantic_payload()
            ),
            "controlled_frame": (
                None
                if self.controlled_frame is None
                else self.controlled_frame.semantic_payload()
            ),
            "expressed_in_frame": (
                None
                if self.expressed_in_frame is None
                else self.expressed_in_frame.semantic_payload()
            ),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=ACTION_COORDINATE_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
        object.__setattr__(
            self,
            "action_payload_schema_hash",
            payload["action_payload_schema_hash"],
        )
        object.__setattr__(
            self,
            "representation_family",
            payload["representation_family"],
        )
        object.__setattr__(self, "value_semantics", payload["value_semantics"])
        object.__setattr__(self, "frame_graph_hash", payload["frame_graph_hash"])
        object.__setattr__(
            self,
            "declared_reference_frame_id",
            payload["declared_reference_frame_id"],
        )
        object.__setattr__(
            self,
            "declared_controlled_frame_id",
            payload["declared_controlled_frame_id"],
        )
        object.__setattr__(
            self,
            "declared_expressed_in_frame_id",
            payload["declared_expressed_in_frame_id"],
        )
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "binding_id",
            f"action-coordinate-binding:{semantic_hash}",
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "binding_id": self.binding_id,
            "action_payload_schema_hash": self.action_payload_schema_hash,
            "representation_family": self.representation_family,
            "value_semantics": self.value_semantics,
            "frame_graph_hash": self.frame_graph_hash,
            "declared_reference_frame_id": self.declared_reference_frame_id,
            "declared_controlled_frame_id": self.declared_controlled_frame_id,
            "declared_expressed_in_frame_id": (
                self.declared_expressed_in_frame_id
            ),
            "reference_frame": (
                None
                if self.reference_frame is None
                else self.reference_frame.semantic_payload()
            ),
            "controlled_frame": (
                None
                if self.controlled_frame is None
                else self.controlled_frame.semantic_payload()
            ),
            "expressed_in_frame": (
                None
                if self.expressed_in_frame is None
                else self.expressed_in_frame.semantic_payload()
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }


def bind_action_payload_coordinates(
    payload_schema: ActionPayloadSchema,
    *,
    frame_graph: CoordinateFrameGraph,
    reference_frame_id: str | None = None,
    controlled_frame_id: str | None = None,
    expressed_in_frame_id: str | None = None,
) -> ActionCoordinateBinding:
    """Bind declared frame semantics to exact, optionally scoped graph frames."""

    if not isinstance(payload_schema, ActionPayloadSchema):
        raise TypeError("payload_schema must be ActionPayloadSchema.")
    if not isinstance(frame_graph, CoordinateFrameGraph):
        raise TypeError("frame_graph must be CoordinateFrameGraph.")

    def reference(
        declared_frame_id: str | None,
        concrete_frame_id: str | None,
        *,
        field_name: str,
    ) -> CoordinateFrameReference | None:
        if declared_frame_id is None:
            if concrete_frame_id is not None:
                raise ValueError(
                    f"{field_name} override is invalid because the action "
                    "payload schema does not declare that frame."
                )
            return None
        frame_id = (
            declared_frame_id
            if concrete_frame_id is None
            else _required(concrete_frame_id, name=field_name)
        )
        try:
            spec = frame_graph.frame_by_id[frame_id]
        except KeyError as exc:
            raise ValueError(
                f"Action payload frame {frame_id!r} is not registered in "
                f"coordinate graph {frame_graph.graph_id!r}."
            ) from exc
        return CoordinateFrameReference.from_spec(spec)

    return ActionCoordinateBinding(
        action_payload_schema_hash=payload_schema.semantic_content_hash,
        representation_family=payload_schema.representation_family,
        value_semantics=payload_schema.value_semantics,
        frame_graph_hash=frame_graph.semantic_content_hash,
        declared_reference_frame_id=payload_schema.reference_frame,
        declared_controlled_frame_id=payload_schema.controlled_frame,
        declared_expressed_in_frame_id=payload_schema.expressed_in_frame,
        reference_frame=reference(
            payload_schema.reference_frame,
            reference_frame_id,
            field_name="reference_frame_id",
        ),
        controlled_frame=reference(
            payload_schema.controlled_frame,
            controlled_frame_id,
            field_name="controlled_frame_id",
        ),
        expressed_in_frame=reference(
            payload_schema.expressed_in_frame,
            expressed_in_frame_id,
            field_name="expressed_in_frame_id",
        ),
    )


__all__ = [
    "ACTION_COORDINATE_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "ActionCoordinateBinding",
    "CoordinateFrameReference",
    "bind_action_payload_coordinates",
]
