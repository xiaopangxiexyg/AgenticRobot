from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dataset_contracts import ActionPayloadSchema as DatasetActionPayloadSchema
from dataset_contracts.action_schema import ComponentActionSpaceSchema
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.embodiment import (
    EmbodimentCaptureProfile,
    EmbodimentCaptureProfileV2,
    EmbodimentComponentCaptureSpec,
    embodiment_capture_profile_from_payload,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p_native import (
    V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V1,
    V25pComponentStreamProfile,
    V25pComponentStreamProfileV2,
    V25pComponentStreamSpec,
    V25pComponentStreamSpecV2,
)
from dataset_contracts.v25p_native_storage import (
    V25P_PARQUET_CHUNK_LAYOUT_V1,
    V25P_RECORD_CJSON_LAYOUT_V1,
)
from dataset_interface.component_action_projection import (
    ComponentActionProjectionSpec,
    component_action_catalog_definition,
    project_component_action_to_v25p,
)
from dataset_interface.v25p_native import load_v25p_native_episode
from robot_motion.action import (
    ActionComponentCatalogV2,
    ActionPayloadSchema,
    ComponentAction,
    ComponentActionSchema,
    ControlComponentSpecV2,
)
from collection_runtime.v25p_component_action import (
    attach_projected_component_action_timing,
)
from collection_runtime.v25p_native_store import (
    V25pComponentInput,
    V25pNativeEpisodeStore,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def _pose_schema(component_id: str, frame: str) -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id=f"fixture:{component_id}:pose",
        schema_version="1",
        representation_family="cartesian_pose",
        dtype="float32",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
        value_semantics="absolute_target",
        reference_frame=frame,
        controlled_frame=f"{component_id}:tcp",
        semantic_parameters={"rotation_representation": "quaternion_xyzw"},
    )


def _runtime_action_schema() -> tuple[
    ComponentActionSchema,
    ActionPayloadSchema,
    ActionPayloadSchema,
]:
    left = _pose_schema("left_arm", "left_base")
    right = _pose_schema("right_arm", "right_base")
    catalog = ActionComponentCatalogV2(
        catalog_id="fixture:bimanual",
        catalog_version="1",
        components=(
            ControlComponentSpecV2(
                component_id="left_arm",
                controlled_entity_id="arm.left",
                command_channel_id="left_arm",
                selection_group_id="arms",
                resource_claims=("arm.left",),
                accepted_payload_schema_hashes=(left.semantic_content_hash,),
                atomic_group_id="arms",
            ),
            ControlComponentSpecV2(
                component_id="right_arm",
                controlled_entity_id="arm.right",
                command_channel_id="right_arm",
                selection_group_id="arms",
                resource_claims=("arm.right",),
                accepted_payload_schema_hashes=(right.semantic_content_hash,),
                atomic_group_id="arms",
            ),
        ),
    )
    return (
        ComponentActionSchema(
            schema_id="fixture:bimanual:action",
            schema_version="1",
            component_catalog=catalog,
            payload_schema_by_component={"left_arm": left, "right_arm": right},
        ),
        left,
        right,
    )


def _action_stream(
    *,
    action_component_id: str,
    dataset_component_id: str,
    schema: ActionPayloadSchema,
) -> V25pComponentStreamSpecV2:
    return V25pComponentStreamSpecV2(
        stream_id=f"action.{action_component_id}",
        component_id=dataset_component_id,
        dataset_role="action",
        value_stage="accepted_reference",
        payload_schema_hash=schema.semantic_content_hash,
        representation_family=schema.representation_family,
        dtype=schema.dtype,
        shape=schema.shape,
        coordinate_names=schema.coordinate_names,
        units=schema.units,
        reference_frame=schema.reference_frame,
        controlled_frame=schema.controlled_frame,
        expressed_in_frame=schema.expressed_in_frame,
        value_semantics=schema.value_semantics,
        semantic_parameters=schema.semantic_parameters,
        action_component_id=action_component_id,
    )


def _profile() -> tuple[V25pComponentStreamProfileV2, ComponentActionSchema]:
    action_schema, left, right = _runtime_action_schema()
    profile = V25pComponentStreamProfileV2(
        profile_id="fixture.native",
        profile_version="2",
        embodiment_profile_hash=HASH_A,
        component_action_space_schema=action_schema.action_space_schema,
        component_catalog_definition=component_action_catalog_definition(
            action_schema
        ),
        action_payload_schemas=(left, right),
        streams=(
            _action_stream(
                action_component_id="left_arm",
                dataset_component_id="manipulator.left",
                schema=left,
            ),
            _action_stream(
                action_component_id="right_arm",
                dataset_component_id="manipulator.right",
                schema=right,
            ),
            V25pComponentStreamSpecV2(
                stream_id="state.tactile_grid",
                component_id="end_effector.left",
                dataset_role="state",
                value_stage="causal_measured",
                payload_schema_hash=HASH_B,
                representation_family="vendor.tactile_grid",
                dtype="uint16",
                shape=(2, 2),
                coordinate_names=("r0c0", "r0c1", "r1c0", "r1c1"),
                units=("count",),
                value_semantics="vendor.measured_state",
                semantic_parameters={"layout": "row_major"},
            ),
        ),
    )
    return profile, action_schema


def _projection_spec(
    action_schema: ComponentActionSchema,
    *,
    required: tuple[str, ...] = ("left_arm",),
) -> ComponentActionProjectionSpec:
    return ComponentActionProjectionSpec(
        projection_id="fixture.component_action.to_v25p",
        projection_version="1",
        component_action_space_schema_hash=action_schema.semantic_content_hash,
        stream_id_by_action_component={
            "left_arm": "action.left_arm",
            "right_arm": "action.right_arm",
        },
        required_action_component_ids=required,
    )


def test_dataset_and_runtime_share_one_payload_schema_class_and_hash() -> None:
    assert DatasetActionPayloadSchema is ActionPayloadSchema
    action_schema, left, _ = _runtime_action_schema()
    expected = canonical_content_hash(
        {
            "schema_id": action_schema.schema_id,
            "schema_version": action_schema.schema_version,
            "component_catalog_hash": action_schema.component_catalog.semantic_content_hash,
            "payload_schema_hash_by_component": {
                component_id: schema.semantic_content_hash
                for component_id, schema in action_schema.payload_schema_by_component.items()
            },
            "semantic_parameters": action_schema.semantic_parameters,
        },
        domain="dexteleop.component_action_schema.v1",
    )
    assert action_schema.semantic_content_hash == expected
    assert action_schema.action_space_schema.semantic_content_hash == expected
    assert ActionPayloadSchema.from_payload(left.semantic_payload()) == left
    assert ComponentActionSpaceSchema.from_payload(
        action_schema.action_space_schema.semantic_payload()
    ) == action_schema.action_space_schema


def test_v2_profile_round_trip_embeds_resolvable_action_definitions() -> None:
    profile, _ = _profile()
    restored = V25pComponentStreamProfileV2.from_payload(profile.to_payload())
    assert restored == profile
    assert set(restored.action_payload_schema_by_hash) == set(
        restored.component_action_space_schema.payload_schema_hash_by_component.values()
    )
    assert restored.stream_by_id["state.tactile_grid"].shape == (2, 2)
    assert restored.stream_by_id["state.tactile_grid"].dtype == "uint16"


def test_v2_profile_rejects_redundant_action_stream_schema_drift() -> None:
    profile, _ = _profile()
    streams = tuple(
        replace(stream, representation_family="xyz_quat_xyzw")
        if stream.stream_id == "action.left_arm"
        else stream
        for stream in profile.streams
    )
    with pytest.raises(ValueError, match="disagrees with payload schema"):
        V25pComponentStreamProfileV2(
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            embodiment_profile_hash=profile.embodiment_profile_hash,
            streams=streams,
            component_action_space_schema=profile.component_action_space_schema,
            component_catalog_definition=profile.component_catalog_definition,
            action_payload_schemas=profile.action_payload_schemas,
        )


def test_partial_component_action_projects_without_recorder_time() -> None:
    profile, action_schema = _profile()
    action = ComponentAction(
        action_schema=action_schema,
        values_by_component={
            "left_arm": np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        },
    )
    projected = project_component_action_to_v25p(
        action=action,
        profile=profile,
        spec=_projection_spec(action_schema),
    )
    assert projected.projected_action_component_ids == ("left_arm",)
    assert projected.omitted_action_component_ids == ()
    assert tuple(projected.values_by_stream) == ("action.left_arm",)
    value = projected.values_by_stream["action.left_arm"]
    assert value.component_id == "manipulator.left"
    assert value.action_component_id == "left_arm"
    assert value.values.dtype == np.dtype("float32")
    assert not hasattr(projected, "sample_index")
    assert not hasattr(projected, "cut_at")


def test_projection_required_coverage_and_action_space_mismatch_fail_closed() -> None:
    profile, action_schema = _profile()
    left_only = ComponentAction(
        action_schema=action_schema,
        values_by_component={"left_arm": [0, 0, 0, 0, 0, 0, 1]},
    )
    with pytest.raises(ValueError, match="lacks required"):
        project_component_action_to_v25p(
            action=left_only,
            profile=profile,
            spec=_projection_spec(action_schema, required=("right_arm",)),
        )
    wrong = replace(
        _projection_spec(action_schema),
        component_action_space_schema_hash="c" * 64,
    )
    with pytest.raises(ValueError, match="action spaces differ"):
        project_component_action_to_v25p(
            action=left_only,
            profile=profile,
            spec=wrong,
        )


def test_frozen_v1_profile_remains_readable_and_is_not_reinterpreted_as_v2() -> None:
    stream = V25pComponentStreamSpec(
        stream_id="legacy.action.left",
        component_id="arm.left",
        dataset_role="action",
        value_stage="accepted_reference",
        payload_schema_hash=HASH_B,
        representation_family="xyz_quat_xyzw",
        dtype="float32",
        shape=(7,),
        coordinate_names=("x", "y", "z", "qx", "qy", "qz", "qw"),
        units=("m", "m", "m", "1", "1", "1", "1"),
    )
    legacy = V25pComponentStreamProfile(
        profile_id="legacy",
        profile_version="1",
        embodiment_profile_hash=HASH_A,
        streams=(stream,),
    )
    payload = legacy.to_payload()
    assert payload["schema"] == V25P_NATIVE_COMPONENT_STREAM_PROFILE_SCHEMA_V1
    assert V25pComponentStreamProfile.from_payload(payload) == legacy
    with pytest.raises(ValueError, match="v2 schema"):
        V25pComponentStreamProfileV2.from_payload(payload)


def test_native_embodiment_v2_omits_target_capability_without_mutating_v1() -> None:
    component = EmbodimentComponentCaptureSpec(
        component_id="manipulator.left",
        component_kind="manipulator",
        controlled_entity_id="arm.left",
    )
    native = EmbodimentCaptureProfileV2(
        profile_id="generic-single-arm",
        profile_version="2",
        topology_id="single_manipulator",
        component_catalog_contract_id="fixture.catalog.v2",
        component_catalog_contract_hash=HASH_A,
        components=(component,),
    )
    native_payload = native.to_payload()
    assert "target_capability_contract_id" not in native_payload
    assert "target_capability_contract_hash" not in native_payload
    assert embodiment_capture_profile_from_payload(native_payload) == native

    legacy = EmbodimentCaptureProfile(
        profile_id="legacy-single-arm",
        profile_version="1",
        topology_id="single_manipulator",
        component_catalog_contract_id="fixture.catalog.v1",
        component_catalog_contract_hash=HASH_A,
        target_capability_contract_id="fixture.target.v1",
        target_capability_contract_hash=HASH_B,
        components=(component,),
    )
    legacy_payload = legacy.to_payload()
    assert legacy_payload["target_capability_contract_id"] == "fixture.target.v1"
    assert embodiment_capture_profile_from_payload(legacy_payload) == legacy


@pytest.mark.parametrize(
    "storage_layout",
    (V25P_RECORD_CJSON_LAYOUT_V1, V25P_PARQUET_CHUNK_LAYOUT_V1),
)
def test_projected_action_gets_owner_time_and_round_trips_v2_storage(
    tmp_path: Path,
    storage_layout: str,
) -> None:
    profile_template, action_schema = _profile()
    embodiment = EmbodimentCaptureProfileV2(
        profile_id="fixture-generic-bimanual",
        profile_version="2",
        topology_id="two_manipulators_one_end_effector",
        component_catalog_contract_id="fixture.catalog.v2",
        component_catalog_contract_hash=(
            action_schema.component_catalog.semantic_content_hash
        ),
        components=(
            EmbodimentComponentCaptureSpec(
                component_id="manipulator.left",
                component_kind="manipulator",
                controlled_entity_id="arm.left",
                command_payload_schema_hash=(
                    action_schema.payload_schema_by_component[
                        "left_arm"
                    ].semantic_content_hash
                ),
            ),
            EmbodimentComponentCaptureSpec(
                component_id="manipulator.right",
                component_kind="manipulator",
                controlled_entity_id="arm.right",
                command_payload_schema_hash=(
                    action_schema.payload_schema_by_component[
                        "right_arm"
                    ].semantic_content_hash
                ),
            ),
            EmbodimentComponentCaptureSpec(
                component_id="end_effector.left",
                component_kind="end_effector",
                controlled_entity_id="gripper.left",
                measurement_payload_schema_hash=HASH_B,
            ),
        ),
    )
    profile = replace(
        profile_template,
        embodiment_profile_hash=embodiment.semantic_content_hash,
    )
    action = ComponentAction(
        action_schema=action_schema,
        values_by_component={
            "left_arm": np.asarray(
                [0.1, 0.2, 0.3, 0, 0, 0, 1], dtype=np.float32
            ),
            "right_arm": np.asarray(
                [-0.1, -0.2, 0.4, 0, 0, 0, 1], dtype=np.float32
            ),
        },
    )
    projected = project_component_action_to_v25p(
        action=action,
        profile=profile,
        spec=_projection_spec(
            action_schema,
            required=("left_arm", "right_arm"),
        ),
    )
    values = dict(
        attach_projected_component_action_timing(
            projected=projected,
            profile=profile,
            effective_at=1.8,
            available_at=1.9,
            effective_clock_domain_id="capture",
            available_clock_domain_id="capture",
        )
    )
    values["state.tactile_grid"] = V25pComponentInput(
        values=np.asarray([[1, 2], [3, 4]], dtype=np.uint16),
        effective_at=1.7,
        available_at=1.9,
        effective_clock_domain_id="capture",
        available_clock_domain_id="capture",
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid=f"v2-action-{storage_layout}",
        legacy_episode_index=0,
        lifecycle_owner_id="fixture.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="capture",
    )
    store = V25pNativeEpisodeStore(
        dataset_root=tmp_path,
        dataset_id=f"fixture-{storage_layout}",
        embodiment_profile=embodiment,
        component_stream_profile=profile,
        physical_storage_layout=storage_layout,
        parquet_chunk_rows=1,
        clock=lambda: 1.0,
        clock_domain_id="capture",
    )
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=1.1,
        bound_clock_domain_id="capture",
    )
    store.append_sample(
        identity,
        sample_index=0,
        cut_at=2.0,
        cut_clock_domain_id="capture",
        values_by_stream=values,
    )
    store.seal_native_episode(
        identity,
        sealed_at=2.1,
        sealed_clock_domain_id="capture",
    )
    store.commit_episode(
        identity,
        committed_at=2.2,
        committed_clock_domain_id="capture",
    )

    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.embodiment_profile == embodiment
    assert loaded.component_stream_profile == profile
    assert loaded.records_by_stream["action.left_arm"][0].values == (
        pytest.approx((0.1, 0.2, 0.3, 0, 0, 0, 1))
    )
    assert loaded.records_by_stream["action.left_arm"][0].source_record_id == (
        action.semantic_content_hash
    )
    assert loaded.records_by_stream["state.tactile_grid"][0].values == (1, 2, 3, 4)
