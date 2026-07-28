from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.action_schema import ActionPayloadSchema
from dataset_contracts.embodiment import EmbodimentComponentCaptureSpec
from dataset_contracts.v25p_native import V25pComponentStreamSpecV2
from dataset_interface.component_action_projection import (
    ComponentActionProjectionSpec,
)
from dataset_interface.v25p_component_profile import (
    ActionStreamBinding,
    build_v25p_v2_capture_profile,
    validate_v25p_v2_capture_profile_bundle,
)
from robot_motion.action import (
    ActionComponentCatalogV2,
    ComponentActionSchema,
    ControlComponentSpecV2,
)


MEASUREMENT_HASH = "a" * 64


def _action_schema() -> tuple[
    ComponentActionSchema,
    ActionPayloadSchema,
    ActionPayloadSchema,
]:
    axis = ActionPayloadSchema(
        schema_id="fixture.axis.command",
        schema_version="1",
        representation_family="joint_position",
        dtype="float32",
        shape=(3,),
        coordinate_names=("slide", "tilt", "spin"),
        units=("m", "rad", "rad"),
        value_semantics="absolute_target",
        reference_frame="fixture.origin",
        controlled_frame="fixture.axis.tip",
    )
    mode = ActionPayloadSchema(
        schema_id="fixture.tool.mode",
        schema_version="1",
        representation_family="discrete_mode",
        dtype="int64",
        shape=(1,),
        coordinate_names=("mode",),
        units=("category_id",),
        value_semantics="mode",
        semantic_parameters={"tokens": ("idle", "engaged")},
    )
    catalog = ActionComponentCatalogV2(
        catalog_id="fixture.non_robot_topology",
        catalog_version="2",
        components=(
            ControlComponentSpecV2(
                component_id="axis.primary",
                controlled_entity_id="fixture.axis",
                command_channel_id="fixture.axis.command",
                selection_group_id="fixture_motion",
                resource_claims=("fixture.axis",),
                accepted_payload_schema_hashes=(axis.semantic_content_hash,),
            ),
            ControlComponentSpecV2(
                component_id="tool.mode",
                controlled_entity_id="fixture.tool",
                command_channel_id="fixture.tool.command",
                selection_group_id="fixture_tools",
                resource_claims=("fixture.tool",),
                accepted_payload_schema_hashes=(mode.semantic_content_hash,),
            ),
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="fixture.non_robot_action",
            schema_version="2",
            component_catalog=catalog,
            payload_schema_by_component={
                "axis.primary": axis,
                "tool.mode": mode,
            },
        ),
        axis,
        mode,
    )


def _build_bundle(*, allow_unmapped: bool = False, bind_mode: bool = True):
    action_schema, axis, mode = _action_schema()
    declarations: list[ActionStreamBinding | V25pComponentStreamSpecV2] = [
        ActionStreamBinding(
            action_component_id="axis.primary",
            dataset_component_id="assembly.axis.7",
            stream_id="action.assembly.axis.7",
            value_stage="accepted_reference",
        ),
        V25pComponentStreamSpecV2(
            stream_id="state.assembly.axis.7.temperature",
            component_id="assembly.axis.7",
            dataset_role="state",
            value_stage="causal_measured",
            payload_schema_hash=MEASUREMENT_HASH,
            representation_family="continuous_native",
            dtype="float32",
            shape=(1,),
            coordinate_names=("temperature",),
            units=("degC",),
            value_semantics="measured_state",
        ),
    ]
    if bind_mode:
        declarations.append(
            ActionStreamBinding(
                action_component_id="tool.mode",
                dataset_component_id="assembly.tool.3",
                stream_id="action.assembly.tool.3.mode",
                value_stage="accepted_reference",
                required_for_sample=False,
            )
        )
    return build_v25p_v2_capture_profile(
        component_action_schema=action_schema,
        embodiment_profile_id="fixture.capture",
        embodiment_profile_version="2",
        topology_id="fixture_with_axis_and_tool",
        embodiment_components=(
            EmbodimentComponentCaptureSpec(
                component_id="assembly.axis.7",
                component_kind="actuated_axis_group",
                controlled_entity_id="fixture.axis",
                command_payload_schema_hash=axis.semantic_content_hash,
            ),
            EmbodimentComponentCaptureSpec(
                component_id="assembly.tool.3",
                component_kind="process_tool",
                controlled_entity_id="fixture.tool",
                command_payload_schema_hash=mode.semantic_content_hash,
            ),
        ),
        stream_profile_id="fixture.capture.streams",
        stream_profile_version="2",
        stream_declarations=tuple(declarations),
        projection_id="fixture.action.to.capture",
        allow_unmapped_action_components=allow_unmapped,
    )


def test_builder_is_topology_neutral_and_copies_canonical_action_schemas() -> None:
    bundle = _build_bundle()
    validate_v25p_v2_capture_profile_bundle(bundle)

    axis = bundle.component_stream_profile.stream_by_id[
        "action.assembly.axis.7"
    ]
    assert axis.component_id == "assembly.axis.7"
    assert axis.action_component_id == "axis.primary"
    assert axis.representation_family == "joint_position"
    assert axis.shape == (3,)
    assert axis.coordinate_names == ("slide", "tilt", "spin")
    assert axis.reference_frame == "fixture.origin"
    assert axis.controlled_frame == "fixture.axis.tip"
    assert bundle.component_action_projection_spec.stream_id_by_action_component == {
        "axis.primary": "action.assembly.axis.7",
        "tool.mode": "action.assembly.tool.3.mode",
    }
    assert bundle.component_action_projection_spec.required_action_component_ids == (
        "axis.primary",
    )


def test_projection_spec_round_trip_preserves_hash() -> None:
    spec = _build_bundle().component_action_projection_spec
    restored = ComponentActionProjectionSpec.from_payload(spec.to_payload())
    assert restored == spec
    assert restored.semantic_content_hash == spec.semantic_content_hash

    changed = dict(spec.to_payload())
    changed["projection_id"] = "fixture.changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        ComponentActionProjectionSpec.from_payload(changed)


def test_builder_requires_explicit_complete_action_bindings_by_default() -> None:
    with pytest.raises(ValueError, match="lack stream bindings"):
        _build_bundle(bind_mode=False)

    bundle = _build_bundle(allow_unmapped=True, bind_mode=False)
    assert bundle.component_action_projection_spec.allow_unmapped_action_components
    assert set(
        bundle.component_action_projection_spec.stream_id_by_action_component
    ) == {"axis.primary"}


def test_builder_rejects_action_streams_that_bypass_canonical_schema_copy() -> None:
    bundle = _build_bundle()
    action_stream = bundle.component_stream_profile.stream_by_id[
        "action.assembly.axis.7"
    ]
    non_action_stream = bundle.component_stream_profile.stream_by_id[
        "state.assembly.axis.7.temperature"
    ]
    action_schema, axis, mode = _action_schema()
    with pytest.raises(ValueError, match="must use ActionStreamBinding"):
        build_v25p_v2_capture_profile(
            component_action_schema=action_schema,
            embodiment_profile_id="fixture.capture",
            embodiment_profile_version="2",
            topology_id="fixture",
            embodiment_components=(
                EmbodimentComponentCaptureSpec(
                    component_id="assembly.axis.7",
                    component_kind="actuated_axis_group",
                    controlled_entity_id="fixture.axis",
                    command_payload_schema_hash=axis.semantic_content_hash,
                ),
                EmbodimentComponentCaptureSpec(
                    component_id="assembly.tool.3",
                    component_kind="process_tool",
                    controlled_entity_id="fixture.tool",
                    command_payload_schema_hash=mode.semantic_content_hash,
                ),
            ),
            stream_profile_id="fixture.capture.streams",
            stream_profile_version="2",
            stream_declarations=(
                replace(action_stream, representation_family="cartesian_pose"),
                non_action_stream,
            ),
            projection_id="fixture.action.to.capture",
        )

