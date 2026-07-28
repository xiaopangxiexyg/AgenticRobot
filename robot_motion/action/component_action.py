"""Arbitrary-topology canonical component action contracts.

``RobotAction`` remains the immutable v1 compatibility carrier.  This module
adds a component-keyed v2 carrier whose values are numerically opaque to
authority/arbitration but bound to exact payload, frame, unit, topology, and
resource contracts.

The carrier represents one canonical action keyframe.  Timing, source,
producer, chunk, authority, safety, and dispatch lineage belong to surrounding
proposal/input/emission contracts and are intentionally absent here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.action_schema import (
    ACTION_PAYLOAD_SCHEMA_HASH_DOMAIN_V1,
    COMPONENT_ACTION_SCHEMA_HASH_DOMAIN_V1,
    STANDARD_REPRESENTATION_FAMILIES_V1,
    STANDARD_VALUE_SEMANTICS_V1,
    SUPPORTED_COMPONENT_DTYPES_V1,
    ActionPayloadSchema,
    ComponentActionSpaceSchema,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

from .components import ActionComponentCatalogV2


COMPONENT_ACTION_VALUE_HASH_DOMAIN_V1 = (
    "dexteleop.component_action_value.v1"
)
COMPONENT_ACTION_HASH_DOMAIN_V1 = "dexteleop.component_action.v1"

def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(
    values: tuple[str, ...] | list[str],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


@dataclass(frozen=True)
class ComponentActionSchema:
    """Bind one topology to one exact payload schema per component."""

    schema_id: str
    schema_version: str
    component_catalog: ActionComponentCatalogV2
    payload_schema_by_component: Mapping[str, ActionPayloadSchema]
    semantic_parameters: Mapping[str, Any] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)
    action_space_schema: ComponentActionSpaceSchema = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.component_catalog, ActionComponentCatalogV2):
            raise TypeError(
                "component_catalog must be an ActionComponentCatalogV2."
            )
        payload_schemas = dict(self.payload_schema_by_component)
        if set(payload_schemas) != set(self.component_catalog.component_ids):
            raise ValueError(
                "Component action schema must bind every catalog component "
                "exactly once."
            )
        ordered: dict[str, ActionPayloadSchema] = {}
        for component_id in self.component_catalog.component_ids:
            payload_schema = payload_schemas[component_id]
            if not isinstance(payload_schema, ActionPayloadSchema):
                raise TypeError(
                    "payload_schema_by_component values must be "
                    "ActionPayloadSchema."
                )
            component = self.component_catalog.component_by_id[component_id]
            if payload_schema.semantic_content_hash not in (
                component.accepted_payload_schema_hashes
            ):
                raise ValueError(
                    f"Component {component_id!r} does not accept payload schema "
                    f"{payload_schema.semantic_content_hash!r}."
                )
            ordered[component_id] = payload_schema
        action_space_schema = ComponentActionSpaceSchema(
            schema_id=self.schema_id,
            schema_version=self.schema_version,
            component_catalog_hash=self.component_catalog.semantic_content_hash,
            payload_schema_hash_by_component={
                component_id: ordered[component_id].semantic_content_hash
                for component_id in self.component_catalog.component_ids
            },
            semantic_parameters=self.semantic_parameters,
        )
        object.__setattr__(self, "schema_id", action_space_schema.schema_id)
        object.__setattr__(self, "schema_version", action_space_schema.schema_version)
        object.__setattr__(
            self,
            "payload_schema_by_component",
            MappingProxyType(ordered),
        )
        object.__setattr__(
            self,
            "semantic_parameters",
            action_space_schema.semantic_parameters,
        )
        object.__setattr__(self, "action_space_schema", action_space_schema)
        object.__setattr__(
            self,
            "semantic_content_hash",
            action_space_schema.semantic_content_hash,
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.component_catalog.component_ids

    def semantic_payload(self) -> dict[str, Any]:
        return self.action_space_schema.semantic_payload()


@dataclass(frozen=True)
class ComponentActionValue:
    """One immutable component value bound to its exact payload schema."""

    component_id: str
    payload_schema: ActionPayloadSchema
    values: Any
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, name="component_id")
        if not isinstance(self.payload_schema, ActionPayloadSchema):
            raise TypeError("payload_schema must be an ActionPayloadSchema.")
        values = self.payload_schema.validate_values(self.values)
        payload = {
            "component_id": component_id,
            "payload_schema_hash": (
                self.payload_schema.semantic_content_hash
            ),
            "values": values,
        }
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_ACTION_VALUE_HASH_DOMAIN_V1,
            ),
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "payload_schema_hash": self.payload_schema.semantic_content_hash,
            "values": self.values,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ComponentAction:
    """Canonical component-keyed action with no source or timing ownership."""

    action_schema: ComponentActionSchema
    values_by_component: Mapping[str, ComponentActionValue | Any]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        raw_values = dict(self.values_by_component)
        unknown = sorted(set(raw_values) - set(self.action_schema.component_ids))
        if unknown:
            raise ValueError(
                f"Component action contains unknown components: {unknown}."
            )
        if not raw_values:
            raise ValueError("Component action must contain at least one value.")
        ordered: dict[str, ComponentActionValue] = {}
        for component_id in self.action_schema.component_ids:
            if component_id not in raw_values:
                continue
            raw = raw_values[component_id]
            expected_schema = self.action_schema.payload_schema_by_component[
                component_id
            ]
            value = (
                raw
                if isinstance(raw, ComponentActionValue)
                else ComponentActionValue(
                    component_id=component_id,
                    payload_schema=expected_schema,
                    values=raw,
                )
            )
            if value.component_id != component_id:
                raise ValueError(
                    "Component action value key and component_id disagree."
                )
            if (
                value.payload_schema.semantic_content_hash
                != expected_schema.semantic_content_hash
            ):
                raise ValueError(
                    f"Component {component_id!r} uses another payload schema."
                )
            ordered[component_id] = value
        payload = {
            "action_schema_hash": self.action_schema.semantic_content_hash,
            "component_coverage": tuple(ordered),
            "component_value_hashes": tuple(
                ordered[component_id].semantic_content_hash
                for component_id in ordered
            ),
        }
        object.__setattr__(
            self,
            "values_by_component",
            MappingProxyType(ordered),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_ACTION_HASH_DOMAIN_V1,
            ),
        )

    @property
    def component_coverage(self) -> tuple[str, ...]:
        return tuple(self.values_by_component)

    def value(self, component_id: str) -> ComponentActionValue:
        return self.values_by_component[_required(component_id, name="component_id")]

    def project(self, component_ids: tuple[str, ...]) -> "ComponentAction":
        requested = _identifiers(
            tuple(component_ids),
            name="component_id",
            allow_empty=False,
        )
        missing = sorted(set(requested) - set(self.values_by_component))
        if missing:
            raise ValueError(
                f"Cannot project absent component values: {missing}."
            )
        return ComponentAction(
            action_schema=self.action_schema,
            values_by_component={
                component_id: self.values_by_component[component_id]
                for component_id in self.action_schema.component_ids
                if component_id in set(requested)
            },
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "action_schema_hash": self.action_schema.semantic_content_hash,
            "component_coverage": self.component_coverage,
            "values_by_component": {
                component_id: value.semantic_payload()
                for component_id, value in self.values_by_component.items()
            },
            "semantic_content_hash": self.semantic_content_hash,
        }


__all__ = [
    "ACTION_PAYLOAD_SCHEMA_HASH_DOMAIN_V1",
    "COMPONENT_ACTION_HASH_DOMAIN_V1",
    "COMPONENT_ACTION_SCHEMA_HASH_DOMAIN_V1",
    "COMPONENT_ACTION_VALUE_HASH_DOMAIN_V1",
    "STANDARD_REPRESENTATION_FAMILIES_V1",
    "STANDARD_VALUE_SEMANTICS_V1",
    "SUPPORTED_COMPONENT_DTYPES_V1",
    "ActionPayloadSchema",
    "ComponentAction",
    "ComponentActionSchema",
    "ComponentActionSpaceSchema",
    "ComponentActionValue",
]
