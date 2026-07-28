"""Deterministic construction of native v2.5-p component profiles.

This module is the pure data-boundary counterpart to a runtime
``ComponentActionSchema``.  Callers declare dataset stream identities and
stages explicitly; the builder copies numerical action semantics from the
canonical payload schemas and never infers robot topology from component
names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dataset_contracts.embodiment import (
    EmbodimentCaptureProfileV2,
    EmbodimentCaptureProfileLike,
    EmbodimentComponentCaptureSpec,
)
from dataset_contracts.v25p_native import (
    V25pComponentStreamProfileV2,
    V25pComponentStreamProfileLike,
    V25pComponentStreamSpecV2,
)
from robot_motion.action import ComponentActionSchema

from .component_action_projection import (
    ComponentActionProjectionSpec,
    component_action_catalog_definition,
    validate_component_action_projection_binding,
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class ActionStreamBinding:
    """Explicit binding from one canonical action component to one stream."""

    action_component_id: str
    dataset_component_id: str
    stream_id: str
    value_stage: str
    required_for_sample: bool = True
    storage_kind: str = "inline_tensor"

    def __post_init__(self) -> None:
        for name in (
            "action_component_id",
            "dataset_component_id",
            "stream_id",
            "value_stage",
            "storage_kind",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "required_for_sample",
            bool(self.required_for_sample),
        )


V25pV2StreamDeclaration = ActionStreamBinding | V25pComponentStreamSpecV2


@dataclass(frozen=True)
class V25pV2CaptureProfileBundle:
    """Pure profile output without producer, time, storage, or episode state."""

    embodiment_profile: EmbodimentCaptureProfileV2
    component_stream_profile: V25pComponentStreamProfileV2
    component_action_projection_spec: ComponentActionProjectionSpec


def validate_v25p_embodiment_stream_binding(
    *,
    embodiment_profile: EmbodimentCaptureProfileLike,
    stream_profile: V25pComponentStreamProfileLike,
) -> None:
    """Validate topology and payload-schema references for V1 or V2 profiles."""

    if stream_profile.embodiment_profile_hash != (
        embodiment_profile.semantic_content_hash
    ):
        raise ValueError("Component stream profile binds another embodiment profile.")
    components = {
        component.component_id: component
        for component in embodiment_profile.components
    }
    for stream in stream_profile.streams:
        component = components.get(stream.component_id)
        if component is None:
            raise ValueError(
                f"Component stream {stream.stream_id!r} references unknown "
                f"component {stream.component_id!r}."
            )
        if stream.dataset_role == "action":
            expected_hash = component.command_payload_schema_hash
        elif stream.dataset_role in {"state", "observation"}:
            expected_hash = component.measurement_payload_schema_hash
        else:
            expected_hash = (
                component.measurement_payload_schema_hash
                or component.command_payload_schema_hash
            )
        if expected_hash is not None and stream.payload_schema_hash != expected_hash:
            raise ValueError(
                f"Stream {stream.stream_id!r} payload schema does not match "
                f"component {stream.component_id!r} role {stream.dataset_role!r}."
            )


def validate_v25p_v2_capture_profile_bundle(
    bundle: V25pV2CaptureProfileBundle,
) -> None:
    """Validate all cross-profile references without touching persistence."""

    if not isinstance(bundle, V25pV2CaptureProfileBundle):
        raise TypeError("bundle must be a V25pV2CaptureProfileBundle.")
    validate_v25p_embodiment_stream_binding(
        embodiment_profile=bundle.embodiment_profile,
        stream_profile=bundle.component_stream_profile,
    )
    validate_component_action_projection_binding(
        profile=bundle.component_stream_profile,
        spec=bundle.component_action_projection_spec,
    )


def build_v25p_v2_capture_profile(
    *,
    component_action_schema: ComponentActionSchema,
    embodiment_profile_id: str,
    embodiment_profile_version: str,
    topology_id: str,
    embodiment_components: Sequence[EmbodimentComponentCaptureSpec],
    stream_profile_id: str,
    stream_profile_version: str,
    stream_declarations: Sequence[V25pV2StreamDeclaration],
    projection_id: str,
    projection_version: str = "1",
    allow_unmapped_action_components: bool = False,
) -> V25pV2CaptureProfileBundle:
    """Build one V2 profile bundle from explicit, ordered declarations.

    ``ActionStreamBinding`` declarations are expanded from the canonical
    payload schema.  Non-action stream specs are accepted verbatim.  Their
    order is preserved so existing profile hashes can migrate without a
    silent representation change.
    """

    if not isinstance(component_action_schema, ComponentActionSchema):
        raise TypeError("component_action_schema must be a ComponentActionSchema.")
    components = tuple(embodiment_components)
    if not components or any(
        not isinstance(component, EmbodimentComponentCaptureSpec)
        for component in components
    ):
        raise TypeError(
            "embodiment_components must contain EmbodimentComponentCaptureSpec values."
        )
    declarations = tuple(stream_declarations)
    if not declarations:
        raise ValueError("stream_declarations must not be empty.")

    embodiment = EmbodimentCaptureProfileV2(
        profile_id=embodiment_profile_id,
        profile_version=embodiment_profile_version,
        topology_id=topology_id,
        component_catalog_contract_id=(
            component_action_schema.component_catalog.catalog_id
        ),
        component_catalog_contract_hash=(
            component_action_schema.component_catalog.semantic_content_hash
        ),
        components=components,
    )
    embodiment_component_by_id = {
        component.component_id: component for component in components
    }
    action_component_ids = set(component_action_schema.component_ids)
    seen_action_components: set[str] = set()
    action_stream_id_by_component: dict[str, str] = {}
    required_action_components: list[str] = []
    streams: list[V25pComponentStreamSpecV2] = []

    for declaration in declarations:
        if isinstance(declaration, ActionStreamBinding):
            component_id = declaration.action_component_id
            if component_id not in action_component_ids:
                raise ValueError(
                    f"Action stream binding references unknown component {component_id!r}."
                )
            if component_id in seen_action_components:
                raise ValueError(
                    f"Action component {component_id!r} has more than one stream binding."
                )
            embodiment_component = embodiment_component_by_id.get(
                declaration.dataset_component_id
            )
            if embodiment_component is None:
                raise ValueError(
                    "Action stream binding references unknown embodiment component "
                    f"{declaration.dataset_component_id!r}."
                )
            payload_schema = component_action_schema.payload_schema_by_component[
                component_id
            ]
            expected_hash = embodiment_component.command_payload_schema_hash
            if (
                expected_hash is not None
                and expected_hash != payload_schema.semantic_content_hash
            ):
                raise ValueError(
                    f"Embodiment command schema disagrees with action component "
                    f"{component_id!r}."
                )
            stream = V25pComponentStreamSpecV2(
                stream_id=declaration.stream_id,
                component_id=declaration.dataset_component_id,
                dataset_role="action",
                value_stage=declaration.value_stage,
                payload_schema_hash=payload_schema.semantic_content_hash,
                representation_family=payload_schema.representation_family,
                dtype=payload_schema.dtype,
                shape=payload_schema.shape,
                coordinate_names=payload_schema.coordinate_names,
                units=payload_schema.units,
                required_for_sample=declaration.required_for_sample,
                storage_kind=declaration.storage_kind,
                reference_frame=payload_schema.reference_frame,
                controlled_frame=payload_schema.controlled_frame,
                expressed_in_frame=payload_schema.expressed_in_frame,
                value_semantics=payload_schema.value_semantics,
                semantic_parameters=payload_schema.semantic_parameters,
                action_component_id=component_id,
            )
            seen_action_components.add(component_id)
            action_stream_id_by_component[component_id] = stream.stream_id
            if declaration.required_for_sample:
                required_action_components.append(component_id)
            streams.append(stream)
            continue

        if not isinstance(declaration, V25pComponentStreamSpecV2):
            raise TypeError(
                "stream_declarations must contain ActionStreamBinding or "
                "V25pComponentStreamSpecV2 values."
            )
        if declaration.dataset_role == "action":
            raise ValueError(
                "Action streams must use ActionStreamBinding so their schema cannot drift."
            )
        streams.append(declaration)

    unmapped = tuple(
        component_id
        for component_id in component_action_schema.component_ids
        if component_id not in seen_action_components
    )
    if unmapped and not allow_unmapped_action_components:
        raise ValueError(
            f"Canonical action components lack stream bindings: {unmapped}."
        )
    if not action_stream_id_by_component:
        raise ValueError("At least one action component stream must be bound.")

    stream_profile = V25pComponentStreamProfileV2(
        profile_id=stream_profile_id,
        profile_version=stream_profile_version,
        embodiment_profile_hash=embodiment.semantic_content_hash,
        streams=tuple(streams),
        component_action_space_schema=component_action_schema.action_space_schema,
        component_catalog_definition=component_action_catalog_definition(
            component_action_schema
        ),
        action_payload_schemas=tuple(
            component_action_schema.payload_schema_by_component.values()
        ),
    )
    projection = ComponentActionProjectionSpec(
        projection_id=projection_id,
        projection_version=projection_version,
        component_action_space_schema_hash=(
            component_action_schema.semantic_content_hash
        ),
        stream_id_by_action_component=action_stream_id_by_component,
        required_action_component_ids=tuple(required_action_components),
        allow_unmapped_action_components=allow_unmapped_action_components,
    )
    bundle = V25pV2CaptureProfileBundle(
        embodiment_profile=embodiment,
        component_stream_profile=stream_profile,
        component_action_projection_spec=projection,
    )
    validate_v25p_v2_capture_profile_bundle(bundle)
    return bundle


__all__ = [
    "ActionStreamBinding",
    "V25pV2CaptureProfileBundle",
    "V25pV2StreamDeclaration",
    "build_v25p_v2_capture_profile",
    "validate_v25p_embodiment_stream_binding",
    "validate_v25p_v2_capture_profile_bundle",
]
