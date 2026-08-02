from __future__ import annotations

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.v22 import (
    DiscreteCategory,
    DiscreteTokenSpec,
    EndEffectorSpec,
    PolicyIOLayout,
)
from dataset_interface.v25p_projection_plan import plan_v23_compatibility_projection
from teleop_runtime.collection.v25p_capture_profiles import (
    V25P_REAL_TELEOP_SOURCE,
    V25P_UMI_SOURCE,
    build_policy_io_v25p_capture_profiles,
)
from collection_runtime.v25p_native_store import validate_native_stream_profile
from teleop_runtime.collection.v25p_real_teleop_producer import RealTeleopV25pProducer
from teleop_runtime.collection.umi.v25p_producer import UmiV25pProducer


POSE_HASH = canonical_content_hash(
    {"name": "test_pose_frames"},
    domain="test.v25p.capture_profiles.v1",
)

EXPECTED_PROFILE_HASHES = {
    V25P_REAL_TELEOP_SOURCE: {
        "embodiment": "0913b62efa130945ba82992316554bdba05ef9c5878cbf2f48c749a83f654b04",
        "streams": "6abfe496e58bbf42fd28b2b9b03cc95a22d4fa0747aaa7802756fcb7fd6fabef",
        "action": "127a1baa1b4b7cbf7ba3c37521263a7cde83c456da8475a0bedd0a9efae34a8f",
        "projection": "d34c4d0ff9454af3f2105acbb5e5e1474771631d0318d87fef58a315a2c9e5e1",
        "v23": "5fb0edf33afaba3ccf9fc17e9f714e1e04d66166705134fb445ce07161e703dc",
    },
    V25P_UMI_SOURCE: {
        "embodiment": "9d0c2b5d898ac902f5e6386e25e12fad189257b04bbacbe82eb6fb5fc26fd3cf",
        "streams": "b2a916c5d5f543177ea1fbc4583f329e2183db1be0da46fd3c4cdf2a767b28c1",
        "action": "4a2aae8232fcaf47754ac124e25878a404c010604bb03e2942ad10e130504f6a",
        "projection": "0f4c432f0009ee04c99263b290491facc50293b93ddb0bf2d85d790e5a8d6499",
        "v23": "97057a4527e0e043619c09c7f2293e9cfb3e19c2cf996fb507d55eb6abaa9052",
    },
}


def _continuous_layout() -> PolicyIOLayout:
    return PolicyIOLayout(
        name="test_dual_scalar",
        end_effectors={
            side: EndEffectorSpec.continuous(
                side,
                adapter="test_scalar_v1",
                target_dim=1,
                target_names=("opening",),
                target_source_indices=(0,),
                state_names=("opening",),
                state_source_field="gripper_measured_scalar",
                units="normalized",
            )
            for side in ("left", "right")
        },
    )


def test_real_profile_captures_pose_qpos_and_end_effector_without_empty_velocity() -> None:
    layout = _continuous_layout()
    bundle = build_policy_io_v25p_capture_profiles(
        source=V25P_REAL_TELEOP_SOURCE,
        layout=layout,
        pose_frame_contract_hash=POSE_HASH,
    )
    validate_native_stream_profile(bundle.embodiment, bundle.streams)
    stream_ids = set(bundle.streams.stream_by_id)
    assert len(bundle.embodiment.components) == 4
    assert len(stream_ids) == 10
    assert "real_teleop.state.qpos.left" in stream_ids
    assert "real_teleop.state.qpos.right" in stream_ids
    assert not any("velocity" in stream_id for stream_id in stream_ids)
    arm = next(item for item in bundle.embodiment.components if item.component_id == "arm.left")
    assert arm.command_payload_schema_hash is None
    assert arm.measurement_payload_schema_hash is None

    plan = plan_v23_compatibility_projection(
        stream_profile=bundle.streams,
        projection_profile=bundle.v23_projection,
    )
    assert set(plan.omitted_stream_ids) == {
        "real_teleop.state.qpos.left",
        "real_teleop.state.qpos.right",
    }
    assert all(
        set(component.to_payload()) == {"stream_id", "dataset_field", "required"}
        for component in bundle.v23_projection.components
    )
    RealTeleopV25pProducer(
        layout=layout,
        stream_profile=bundle.streams,
        bindings=bundle.producer_bindings,
        component_action_schema=bundle.component_action_schema,
        component_action_projection=bundle.component_action_projection,
    )
    assert (
        bundle.streams.component_action_space_schema.semantic_content_hash
        == bundle.component_action_schema.semantic_content_hash
    )
    assert {
        item.representation_family
        for item in bundle.streams.streams
        if item.dataset_role == "action" and ".pose." in item.stream_id
    } == {"cartesian_pose"}


def test_builtin_profile_builder_refactor_preserves_frozen_profile_hashes() -> None:
    for source, expected in EXPECTED_PROFILE_HASHES.items():
        bundle = build_policy_io_v25p_capture_profiles(
            source=source,
            layout=_continuous_layout(),
            pose_frame_contract_hash=POSE_HASH,
        )
        assert bundle.embodiment.semantic_content_hash == expected["embodiment"]
        assert bundle.streams.semantic_content_hash == expected["streams"]
        assert bundle.component_action_schema.semantic_content_hash == expected["action"]
        assert bundle.component_action_projection.semantic_content_hash == expected[
            "projection"
        ]
        assert bundle.v23_projection.semantic_content_hash == expected["v23"]


def test_umi_profile_has_mixed_action_stages_without_global_stage_duplication() -> None:
    layout = _continuous_layout()
    bundle = build_policy_io_v25p_capture_profiles(
        source=V25P_UMI_SOURCE,
        layout=layout,
        pose_frame_contract_hash=POSE_HASH,
    )
    action_stage_by_stream = {
        item.stream_id: item.value_stage
        for item in bundle.streams.streams
        if item.dataset_role == "action"
    }
    assert action_stage_by_stream["umi.action.pose.left"] == "next_accepted_observation"
    assert action_stage_by_stream["umi.action.end_effector.left"] == "accepted_reference"
    assert not hasattr(bundle.v23_projection, "action_stage")
    assert not any("qpos" in item.stream_id for item in bundle.streams.streams)
    UmiV25pProducer(
        stream_profile=bundle.streams,
        bindings=bundle.producer_bindings,
        component_action_schema=bundle.component_action_schema,
        component_action_projection=bundle.component_action_projection,
    )


def test_absent_or_discrete_end_effectors_only_add_streams_that_exist() -> None:
    no_hand = PolicyIOLayout(name="test_no_end_effector")
    no_hand_bundle = build_policy_io_v25p_capture_profiles(
        source=V25P_REAL_TELEOP_SOURCE,
        layout=no_hand,
        pose_frame_contract_hash=POSE_HASH,
    )
    assert len(no_hand_bundle.embodiment.components) == 2
    assert len(no_hand_bundle.streams.streams) == 6
    assert not any(
        "end_effector" in item.stream_id for item in no_hand_bundle.streams.streams
    )

    token = DiscreteTokenSpec(
        name="grasp_state",
        categories=(
            DiscreteCategory(0, "open"),
            DiscreteCategory(1, "closed"),
        ),
    )
    discrete = PolicyIOLayout(
        name="test_discrete_end_effector",
        end_effectors={
            side: EndEffectorSpec.discrete(
                side,
                adapter="test_discrete_v1",
                target_tokens=(token,),
                target_source_indices=(0,),
                state_source_indices=(0,),
            )
            for side in ("left", "right")
        },
    )
    discrete_bundle = build_policy_io_v25p_capture_profiles(
        source=V25P_REAL_TELEOP_SOURCE,
        layout=discrete,
        pose_frame_contract_hash=POSE_HASH,
    )
    end_effector_streams = tuple(
        item
        for item in discrete_bundle.streams.streams
        if "end_effector" in item.stream_id
    )
    assert end_effector_streams
    assert all(item.dtype == "int64" for item in end_effector_streams)
    assert all(
        item.representation_family == "discrete_mode"
        for item in end_effector_streams
    )
    assert all(
        component.dataset_field.endswith("_token")
        for component in discrete_bundle.v23_projection.components
        if "end_effector" in component.stream_id
    )


def test_real_profile_only_declares_the_active_single_arm_and_end_effector() -> None:
    bundle = build_policy_io_v25p_capture_profiles(
        source=V25P_REAL_TELEOP_SOURCE,
        layout=_continuous_layout(),
        pose_frame_contract_hash=POSE_HASH,
        component_sides=("right",),
    )
    assert bundle.embodiment.topology_id == "single_manipulator"
    assert {item.component_id for item in bundle.embodiment.components} == {
        "arm.right",
        "end_effector.right",
    }
    assert all(not item.stream_id.endswith(".left") for item in bundle.streams.streams)
