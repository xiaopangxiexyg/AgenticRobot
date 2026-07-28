"""Pure projection of canonical component actions into v2.5-p streams.

The projector owns neither action composition nor recorder time.  Collection
code attaches sample/episode identity and timestamps after this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.v25p_native import V25pComponentStreamProfileV2
from robot_motion.action.component_action import ComponentAction, ComponentActionSchema


COMPONENT_ACTION_PROJECTION_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.component_action_projection_spec.v1"
)
COMPONENT_ACTION_PROJECTION_SPEC_SCHEMA_V1 = (
    "dexteleop.component_action_projection_spec.v1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _hash(value: object, *, name: str) -> str:
    normalized = _required(value, name=name).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


@dataclass(frozen=True)
class ComponentActionProjectionSpec:
    """Explicit component-to-stream mapping for one action-space schema."""

    projection_id: str
    projection_version: str
    component_action_space_schema_hash: str
    stream_id_by_action_component: Mapping[str, str]
    required_action_component_ids: tuple[str, ...] = ()
    allow_unmapped_action_components: bool = False
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=COMPONENT_ACTION_PROJECTION_SPEC_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        projection_id = _required(self.projection_id, name="projection_id")
        projection_version = _required(
            self.projection_version,
            name="projection_version",
        )
        schema_hash = _hash(
            self.component_action_space_schema_hash,
            name="component_action_space_schema_hash",
        )
        mapping = {
            _required(component_id, name="action_component_id"): _required(
                stream_id,
                name="stream_id",
            )
            for component_id, stream_id in dict(
                self.stream_id_by_action_component
            ).items()
        }
        if not mapping:
            raise ValueError(
                "stream_id_by_action_component must contain at least one binding."
            )
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Each action stream may be bound only once.")
        required = tuple(
            _required(component_id, name="required_action_component_id")
            for component_id in self.required_action_component_ids
        )
        if len(required) != len(set(required)):
            raise ValueError("required_action_component_ids must be unique.")
        unknown_required = tuple(item for item in required if item not in mapping)
        if unknown_required:
            raise ValueError(
                "Required action components lack stream bindings: "
                f"{unknown_required}."
            )
        payload = {
            "projection_id": projection_id,
            "projection_version": projection_version,
            "component_action_space_schema_hash": schema_hash,
            "stream_id_by_action_component": mapping,
            "required_action_component_ids": required,
            "allow_unmapped_action_components": bool(
                self.allow_unmapped_action_components
            ),
        }
        object.__setattr__(self, "projection_id", projection_id)
        object.__setattr__(self, "projection_version", projection_version)
        object.__setattr__(
            self,
            "component_action_space_schema_hash",
            schema_hash,
        )
        object.__setattr__(
            self,
            "stream_id_by_action_component",
            MappingProxyType(mapping),
        )
        object.__setattr__(self, "required_action_component_ids", required)
        object.__setattr__(
            self,
            "allow_unmapped_action_components",
            bool(self.allow_unmapped_action_components),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_ACTION_PROJECTION_SPEC_HASH_DOMAIN_V1,
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "projection_id": self.projection_id,
            "projection_version": self.projection_version,
            "component_action_space_schema_hash": (
                self.component_action_space_schema_hash
            ),
            "stream_id_by_action_component": dict(
                self.stream_id_by_action_component
            ),
            "required_action_component_ids": self.required_action_component_ids,
            "allow_unmapped_action_components": (
                self.allow_unmapped_action_components
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> "ComponentActionProjectionSpec":
        if payload.get("schema") != COMPONENT_ACTION_PROJECTION_SPEC_SCHEMA_V1:
            raise ValueError("Unsupported component action projection spec schema.")
        result = cls(
            projection_id=payload.get("projection_id", ""),
            projection_version=payload.get("projection_version", ""),
            component_action_space_schema_hash=payload.get(
                "component_action_space_schema_hash",
                "",
            ),
            stream_id_by_action_component=payload.get(
                "stream_id_by_action_component",
                {},
            ),
            required_action_component_ids=tuple(
                payload.get("required_action_component_ids", ())
            ),
            allow_unmapped_action_components=payload.get(
                "allow_unmapped_action_components",
                False,
            ),
        )
        declared = payload.get("semantic_content_hash")
        if declared is not None and str(declared) != result.semantic_content_hash:
            raise ValueError("Component action projection spec hash mismatch.")
        return result


def component_action_catalog_definition(
    schema: ComponentActionSchema,
) -> dict[str, object]:
    """Return the full pure catalog definition behind one runtime schema."""

    if not isinstance(schema, ComponentActionSchema):
        raise TypeError("schema must be a ComponentActionSchema.")
    catalog = schema.component_catalog
    return {
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "components": tuple(
            component.semantic_payload() for component in catalog.components
        ),
    }


@dataclass(frozen=True)
class ProjectedComponentActionValue:
    stream_id: str
    component_id: str
    action_component_id: str
    payload_schema_hash: str
    values: np.ndarray

    def __post_init__(self) -> None:
        for name in ("stream_id", "component_id", "action_component_id"):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "payload_schema_hash",
            _hash(self.payload_schema_hash, name="payload_schema_hash"),
        )
        values = np.asarray(self.values)
        if values.dtype.hasobject:
            raise TypeError("Projected component values must not use object dtype.")
        raw = np.ascontiguousarray(values).tobytes(order="C")
        object.__setattr__(
            self,
            "values",
            np.frombuffer(raw, dtype=values.dtype).reshape(values.shape),
        )


@dataclass(frozen=True)
class ProjectedComponentAction:
    projection_spec_hash: str
    component_action_hash: str
    component_action_space_schema_hash: str
    values_by_stream: Mapping[str, ProjectedComponentActionValue]
    projected_action_component_ids: tuple[str, ...]
    omitted_action_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "projection_spec_hash",
            "component_action_hash",
            "component_action_space_schema_hash",
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), name=name))
        values = dict(self.values_by_stream)
        if any(
            stream_id != value.stream_id
            for stream_id, value in values.items()
        ):
            raise ValueError("Projected value map keys must match value stream IDs.")
        object.__setattr__(self, "values_by_stream", MappingProxyType(values))
        object.__setattr__(
            self,
            "projected_action_component_ids",
            tuple(self.projected_action_component_ids),
        )
        object.__setattr__(
            self,
            "omitted_action_component_ids",
            tuple(self.omitted_action_component_ids),
        )


def validate_component_action_projection_binding(
    *,
    profile: V25pComponentStreamProfileV2,
    spec: ComponentActionProjectionSpec,
) -> None:
    """Fail closed before any sample when a mapping would require inference."""

    if spec.component_action_space_schema_hash != (
        profile.component_action_space_schema.semantic_content_hash
    ):
        raise ValueError("Projection spec and native profile action spaces differ.")
    action_space_components = set(
        profile.component_action_space_schema.component_ids
    )
    unknown_components = sorted(
        set(spec.stream_id_by_action_component) - action_space_components
    )
    if unknown_components:
        raise ValueError(
            f"Projection spec references unknown action components: {unknown_components}."
        )
    action_stream_ids = {
        stream.stream_id
        for stream in profile.streams
        if stream.dataset_role == "action"
    }
    bound_stream_ids = set(spec.stream_id_by_action_component.values())
    if bound_stream_ids != action_stream_ids:
        raise ValueError(
            "Projection spec must bind every action stream exactly once: "
            f"missing={sorted(action_stream_ids - bound_stream_ids)} "
            f"unknown={sorted(bound_stream_ids - action_stream_ids)}."
        )
    for component_id, stream_id in spec.stream_id_by_action_component.items():
        stream = profile.stream_by_id[stream_id]
        if stream.dataset_role != "action" or stream.action_component_id != component_id:
            raise ValueError(
                f"Projection binding {component_id!r}->{stream_id!r} disagrees "
                "with the stream's canonical action component."
            )


def project_component_action_to_v25p(
    *,
    action: ComponentAction,
    profile: V25pComponentStreamProfileV2,
    spec: ComponentActionProjectionSpec,
) -> ProjectedComponentAction:
    """Project values only; recorder-owned identity and time remain absent."""

    if not isinstance(action, ComponentAction):
        raise TypeError("action must be a ComponentAction.")
    validate_component_action_projection_binding(profile=profile, spec=spec)
    action_schema_hash = action.action_schema.semantic_content_hash
    if action_schema_hash != spec.component_action_space_schema_hash:
        raise ValueError("ComponentAction uses another action-space schema.")
    present = set(action.component_coverage)
    missing_required = tuple(
        item for item in spec.required_action_component_ids if item not in present
    )
    if missing_required:
        raise ValueError(
            f"ComponentAction lacks required projected components: {missing_required}."
        )
    unmapped = tuple(
        component_id
        for component_id in action.component_coverage
        if component_id not in spec.stream_id_by_action_component
    )
    if unmapped and not spec.allow_unmapped_action_components:
        raise ValueError(
            f"ComponentAction contains unmapped components: {unmapped}."
        )
    projected: dict[str, ProjectedComponentActionValue] = {}
    projected_components: list[str] = []
    for component_id in action.action_schema.component_ids:
        if component_id not in present or component_id not in (
            spec.stream_id_by_action_component
        ):
            continue
        stream_id = spec.stream_id_by_action_component[component_id]
        stream = profile.stream_by_id[stream_id]
        value = action.value(component_id)
        if value.payload_schema.semantic_content_hash != stream.payload_schema_hash:
            raise ValueError(
                f"Component {component_id!r} payload hash changed after binding."
            )
        normalized = value.payload_schema.validate_values(value.values)
        stream.normalize_values(normalized)
        projected[stream_id] = ProjectedComponentActionValue(
            stream_id=stream_id,
            component_id=stream.component_id,
            action_component_id=component_id,
            payload_schema_hash=stream.payload_schema_hash,
            values=normalized,
        )
        projected_components.append(component_id)
    return ProjectedComponentAction(
        projection_spec_hash=spec.semantic_content_hash,
        component_action_hash=action.semantic_content_hash,
        component_action_space_schema_hash=action_schema_hash,
        values_by_stream=projected,
        projected_action_component_ids=tuple(projected_components),
        omitted_action_component_ids=unmapped,
    )


__all__ = [
    "COMPONENT_ACTION_PROJECTION_SPEC_HASH_DOMAIN_V1",
    "ComponentActionProjectionSpec",
    "ProjectedComponentAction",
    "ProjectedComponentActionValue",
    "component_action_catalog_definition",
    "project_component_action_to_v25p",
    "validate_component_action_projection_binding",
]
