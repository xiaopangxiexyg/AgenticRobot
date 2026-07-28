"""Pure construction of relative-delta component action schemas.

The helper is source- and baseline-neutral.  It derives a pose/joint-position
delta layout from an exact absolute target schema while preserving topology,
coordinate ordering, units, frames, resource claims, atomic groups, and
dependencies for the selected components.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from dataset_contracts.action_schema import ActionPayloadSchema

from .component_action import ComponentActionSchema
from .components import ActionComponentCatalogV2, ControlComponentSpecV2


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def build_relative_delta_action_schema(
    absolute_action_schema: ComponentActionSchema,
    *,
    component_ids: Sequence[str] | None = None,
    schema_id: str,
    schema_version: str = "development",
    catalog_id: str | None = None,
    catalog_version: str = "development",
    payload_schema_id_by_component: Mapping[str, str] | None = None,
) -> ComponentActionSchema:
    """Derive a relative pose/qpos schema from selected absolute components."""

    if not isinstance(absolute_action_schema, ComponentActionSchema):
        raise TypeError(
            "absolute_action_schema must be ComponentActionSchema."
        )
    selected = (
        tuple(absolute_action_schema.component_ids)
        if component_ids is None
        else tuple(_required(value, name="component_id") for value in component_ids)
    )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("component_ids must be non-empty and unique.")
    unknown = sorted(set(selected) - set(absolute_action_schema.component_ids))
    if unknown:
        raise ValueError(f"Unknown relative components: {unknown}.")
    payload_ids = dict(payload_schema_id_by_component or {})
    unknown_payload_ids = sorted(set(payload_ids) - set(selected))
    if unknown_payload_ids:
        raise ValueError(
            "payload_schema_id_by_component references unselected components: "
            f"{unknown_payload_ids}."
        )
    payloads: dict[str, ActionPayloadSchema] = {}
    for component_id in selected:
        absolute = absolute_action_schema.payload_schema_by_component[
            component_id
        ]
        if (
            absolute.value_semantics != "absolute_target"
            or absolute.representation_family
            not in {"cartesian_pose", "joint_position"}
        ):
            raise ValueError(
                f"Component {component_id!r} is not an absolute pose/qpos target."
            )
        parameters = (
            {
                "translation_application": "reference_frame_additive",
                "rotation_application": "baseline_local_right_multiply",
            }
            if absolute.representation_family == "cartesian_pose"
            else {"application": "elementwise_additive"}
        )
        payloads[component_id] = ActionPayloadSchema(
            schema_id=payload_ids.get(
                component_id,
                f"{absolute.schema_id}.relative_delta",
            ),
            schema_version=_required(
                schema_version,
                name="schema_version",
            ),
            representation_family=absolute.representation_family,
            dtype=absolute.dtype,
            shape=absolute.shape,
            coordinate_names=absolute.coordinate_names,
            units=absolute.units,
            value_semantics="relative_delta",
            reference_frame=absolute.reference_frame,
            controlled_frame=absolute.controlled_frame,
            expressed_in_frame=absolute.expressed_in_frame,
            semantic_parameters=parameters,
        )
    source_catalog = absolute_action_schema.component_catalog
    components = []
    for component_id in selected:
        source = source_catalog.component_by_id[component_id]
        dependencies = tuple(
            dependency
            for dependency in source.dependencies
            if dependency in selected
        )
        if len(dependencies) != len(source.dependencies):
            missing = sorted(set(source.dependencies) - set(selected))
            raise ValueError(
                f"Component {component_id!r} requires unselected dependencies: "
                f"{missing}."
            )
        components.append(
            ControlComponentSpecV2(
                component_id=source.component_id,
                controlled_entity_id=source.controlled_entity_id,
                command_channel_id=source.command_channel_id,
                selection_group_id=source.selection_group_id,
                resource_claims=source.resource_claims,
                accepted_payload_schema_hashes=(
                    payloads[component_id].semantic_content_hash,
                ),
                atomic_group_id=source.atomic_group_id,
                dependencies=dependencies,
            )
        )
    catalog = ActionComponentCatalogV2(
        catalog_id=(
            f"{source_catalog.catalog_id}.relative_delta"
            if catalog_id is None
            else _required(catalog_id, name="catalog_id")
        ),
        catalog_version=_required(
            catalog_version,
            name="catalog_version",
        ),
        components=tuple(components),
    )
    return ComponentActionSchema(
        schema_id=_required(schema_id, name="schema_id"),
        schema_version=_required(schema_version, name="schema_version"),
        component_catalog=catalog,
        payload_schema_by_component=payloads,
        semantic_parameters={
            "derived_from_absolute_action_schema_hash": (
                absolute_action_schema.semantic_content_hash
            ),
            "derivation": "relative_delta_pose_qpos",
        },
    )


__all__ = ["build_relative_delta_action_schema"]
