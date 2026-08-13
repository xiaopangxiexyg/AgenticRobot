from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import HandActionSpec
from sim.demonstration_editing import (
    DemonstrationComponentActionAdapter,
    DemonstrationComponentActionBinding,
    DemonstrationEdit,
    DemonstrationTrajectory,
    NormalizedGripperCloseToJointPositionProjection,
    apply_demonstration_edit,
    sample_edited_trajectory,
)
from sim.mujoco.config import GripperSpec
from sim.mujoco.deploy import (
    MujocoComponentActionRobotActionV1Materializer,
    build_rm65_right_cube_lift_component_target_profile,
)


def _scene():
    return SimpleNamespace(
        arm_specs={"left": object(), "right": object()},
        gripper_specs={
            "right_gripper": GripperSpec(
                name="right_gripper",
                joint_names=(
                    "right_gripper_left_finger_joint",
                    "right_gripper_right_finger_joint",
                ),
                open_qpos=(0.0, 0.0),
                closed_qpos=(-0.035, -0.035),
                side="right",
            )
        },
    )


def _profile(scene):
    return build_rm65_right_cube_lift_component_target_profile(
        scene,
        SimpleNamespace(
            controlled_side="right",
            right_gripper_name="right_gripper",
        ),
    )


def _trajectory(*, reference_frame: str = "right_arm_base"):
    return DemonstrationTrajectory(
        trajectory_id="committed-zero-edit-fixture",
        reference_frame=reference_frame,
        controlled_frame="tool_tcp",
        hand_component_id="right_gripper",
        hand_target_representation="physical_joint_qpos",
        hand_target_names=(
            "right_gripper_left_finger_joint",
            "right_gripper_right_finger_joint",
        ),
        hand_target_units=("m", "m"),
        timestamps_sec=np.asarray([0.0, 0.1]),
        tool_pose_xyzquat_xyzw=np.asarray(
            [
                [0.4, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [0.41, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        hand_targets=np.asarray(
            [
                [0.0, 0.0],
                [-0.035, -0.035],
            ]
        ),
        hand_edit_start_index=0,
        hand_edit_end_index=1,
        source_kind="committed_mujoco_control_trace_safe_action",
        source_id="episode-1",
        source_content_hash="a" * 64,
    )


def _reference(*, reference_frame: str = "right_arm_base"):
    trajectory = _trajectory(reference_frame=reference_frame)
    edit = DemonstrationEdit(
        edit_id="identity",
        producer_id="fixture-editor",
        sample_id="ppo-sample-1",
        translation_m=np.zeros(3),
        rotation_rotvec_rad=np.zeros(3),
        hand_final_delta=np.zeros(2),
    )
    return sample_edited_trajectory(
        apply_demonstration_edit(trajectory, edit),
        0.1,
    )


def _normalized_close_reference(close: float):
    trajectory = DemonstrationTrajectory(
        trajectory_id="normalized-close-fixture",
        reference_frame="right_arm_base",
        controlled_frame="tool_tcp",
        hand_component_id="right_gripper",
        hand_target_representation="normalized_gripper_close",
        hand_target_names=("gripper_close",),
        hand_target_units=("1",),
        timestamps_sec=np.asarray([0.0, 0.1]),
        tool_pose_xyzquat_xyzw=np.asarray(
            [
                [0.4, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                [0.41, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        hand_targets=np.asarray([[0.0], [close]]),
        hand_edit_start_index=0,
        hand_edit_end_index=1,
        source_kind="committed_mujoco_control_trace_safe_action",
        source_id="episode-1",
        source_content_hash="a" * 64,
    )
    edit = DemonstrationEdit(
        edit_id="identity-normalized-close",
        producer_id="fixture-editor",
        sample_id="ppo-sample-normalized-close",
        translation_m=np.zeros(3),
        rotation_rotvec_rad=np.zeros(3),
        hand_final_delta=np.zeros(1),
    )
    return sample_edited_trajectory(
        apply_demonstration_edit(trajectory, edit),
        0.1,
    )


def _adapter(scene):
    profile = _profile(scene)
    return profile, DemonstrationComponentActionAdapter(
        DemonstrationComponentActionBinding(
            binding_id="fixture.demo-to-cube-lift-component-action.v1",
            action_schema=profile.action_schema,
            pose_component_id="right_arm",
            hand_component_id="right_gripper",
            demonstration_hand_target_representation="physical_joint_qpos",
        )
    )


def test_zero_edit_reference_projects_to_exact_native_pose_hand_schema() -> None:
    scene = _scene()
    profile, adapter = _adapter(scene)

    first = adapter.adapt(_reference())
    second = adapter.adapt(_reference())

    assert first.accepted
    assert first.action is not None
    assert first.action.action_schema is profile.action_schema
    assert first.action.component_coverage == ("right_arm", "right_gripper")
    np.testing.assert_allclose(
        first.action.value("right_arm").values,
        [0.41, -0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(
        first.action.value("right_gripper").values,
        [-0.035, -0.035],
    )
    assert first.receipt.edit_sample_id == "ppo-sample-1"
    assert first.receipt.receipt_id == (
        "demonstration-component-action-receipt:"
        f"{first.receipt.semantic_content_hash}"
    )
    assert first.receipt.output_action_hash == first.action.semantic_content_hash
    assert first.receipt.semantic_content_hash == second.receipt.semantic_content_hash


def test_native_action_materializes_only_at_existing_private_execution_edge() -> None:
    scene = _scene()
    profile, adapter = _adapter(scene)
    result = adapter.adapt(_reference())
    assert result.action is not None
    materializer = MujocoComponentActionRobotActionV1Materializer(
        scene,
        profile,
        hand_action_spec_by_component={
            "right_gripper": HandActionSpec.dmtac_gripper("right")
        },
        support_arm_pose_provider=lambda: {
            "left": np.asarray(
                [0.3, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
        },
        support_arm_sides=("left",),
        support_pose_contract_id="fixture.measured-left-support.v1",
    )

    compatibility = materializer(
        result.action,
        timestamp=result.receipt.resolved_time_sec,
        source=result.receipt.reference_id,
    )

    assert compatibility.active_effectors == (
        "left_arm",
        "right_arm",
        "right_hand",
    )
    np.testing.assert_allclose(
        compatibility.hand_qpos_by_side["right"],
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    assert set(compatibility.metadata) == {
        "arm_action_specs_by_side",
        "hand_action_specs_by_side",
    }


def test_normalized_close_projects_to_scene_declared_joint_positions() -> None:
    scene = _scene()
    profile = _profile(scene)
    gripper = scene.gripper_specs["right_gripper"]
    projection = NormalizedGripperCloseToJointPositionProjection(
        projection_id="fixture.right-gripper-close-to-qpos.v1",
        open_target=gripper.open_qpos,
        closed_target=gripper.closed_qpos,
    )
    adapter = DemonstrationComponentActionAdapter(
        DemonstrationComponentActionBinding(
            binding_id="fixture.normalized-close-to-cube-lift-action.v1",
            action_schema=profile.action_schema,
            pose_component_id="right_arm",
            hand_component_id="right_gripper",
            demonstration_hand_target_representation="normalized_gripper_close",
            hand_target_projection=projection,
        )
    )

    result = adapter.adapt(_normalized_close_reference(0.5))

    assert result.accepted
    assert result.action is not None
    np.testing.assert_allclose(
        result.action.value("right_gripper").values,
        [-0.0175, -0.0175],
    )
    assert adapter.binding.semantic_content_hash != projection.semantic_content_hash


def test_normalized_close_projection_rejects_out_of_range_without_clipping() -> None:
    scene = _scene()
    profile = _profile(scene)
    gripper = scene.gripper_specs["right_gripper"]
    adapter = DemonstrationComponentActionAdapter(
        DemonstrationComponentActionBinding(
            binding_id="fixture.normalized-close-rejects-out-of-range.v1",
            action_schema=profile.action_schema,
            pose_component_id="right_arm",
            hand_component_id="right_gripper",
            demonstration_hand_target_representation="normalized_gripper_close",
            hand_target_projection=NormalizedGripperCloseToJointPositionProjection(
                projection_id="fixture.right-gripper-close-to-qpos.v1",
                open_target=gripper.open_qpos,
                closed_target=gripper.closed_qpos,
            ),
        )
    )

    result = adapter.adapt(_normalized_close_reference(1.1))

    assert not result.accepted
    assert result.action is None
    assert result.receipt.rejection_reasons[0].startswith(
        "target_schema_validation_failed:ValueError:"
    )


def test_adapter_rejects_frame_mismatch_without_partial_action() -> None:
    scene = _scene()
    _profile_value, adapter = _adapter(scene)

    result = adapter.adapt(_reference(reference_frame="object_initial"))

    assert not result.accepted
    assert result.action is None
    assert result.receipt.rejection_reasons == (
        "pose_reference_frame_mismatch",
    )
    assert result.receipt.output_action_hash is None


def test_binding_rejects_unknown_component_before_reference_adaptation() -> None:
    scene = _scene()
    profile = _profile(scene)
    with pytest.raises(ValueError, match="unknown components"):
        DemonstrationComponentActionBinding(
            binding_id="invalid",
            action_schema=profile.action_schema,
            pose_component_id="right_arm",
            hand_component_id="missing_hand",
            demonstration_hand_target_representation="physical_joint_qpos",
        )
