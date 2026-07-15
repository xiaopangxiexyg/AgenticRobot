from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import robot_motion.frames as robot_frames_module
import robot_motion.tcp as robot_tcp_module
import robot_motion.transforms as robot_transforms_module
import robot_motion.action.arm_spec as robot_arm_spec_module
import robot_motion.action.codec as robot_codec_module
import robot_motion.action.executor as robot_executor_module
import robot_motion.action.gripper as robot_gripper_module
import robot_motion.action.hand_mapping as robot_hand_mapping_module
import robot_motion.action.hand_spec as robot_hand_spec_module
import robot_motion.action.schema as robot_schema_module
import robot_motion.action.vector_pipeline as robot_vector_module
from robot_motion.action.codec import (
    EE_POSE_HAND16_VECTOR_DIM,
    EE_POSE_VECTOR_DIM,
    QPOS_HAND16_VECTOR_DIM,
    QPOS_VECTOR_DIM,
    ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION,
    ROBOT_ACTION_VECTOR_LAYOUT_VERSION,
)
from robot_motion.action import (
    ActionVectorEncoding as RobotMotionActionVectorEncoding,
    ArmActionSpec as RobotMotionArmActionSpec,
    HandActionSpec as RobotMotionHandActionSpec,
    RobotAction as RobotMotionRobotAction,
    RobotActionResolver as RobotMotionRobotActionResolver,
    VectorActionLayout as RobotMotionVectorActionLayout,
    decode_robot_action as robot_motion_decode_robot_action,
    encode_robot_action as robot_motion_encode_robot_action,
    robot_action_from_vector_action as robot_motion_robot_action_from_vector_action,
)
from robot_motion.runtime import TeleopCommand as RobotMotionTeleopCommand
from teleop_runtime.action import (
    ActionVectorEncoding as LegacyActionVectorEncoding,
    ArmActionSpec as LegacyArmActionSpec,
    HandActionSpec as LegacyHandActionSpec,
    RobotAction as LegacyRobotAction,
    RobotActionResolver as LegacyRobotActionResolver,
    VectorActionLayout as LegacyVectorActionLayout,
    decode_robot_action as legacy_decode_robot_action,
    encode_robot_action as legacy_encode_robot_action,
    robot_action_from_vector_action as legacy_robot_action_from_vector_action,
)
from teleop_runtime.action import codec as legacy_codec_module
from teleop_runtime import arm_frames as legacy_frames_module
from teleop_runtime import hand_mapping as legacy_hand_mapping_module
from teleop_runtime import tcp as legacy_tcp_module
from teleop_runtime import transforms as legacy_transforms_module
from teleop_runtime.retargeters import scalar_to_gripper as legacy_gripper_module
from teleop_runtime.schemas import TeleopCommand as LegacyTeleopCommand


def _repo_source(relative_path: str) -> str:
    return (Path(__file__).resolve().parents[1] / relative_path).read_text(encoding="utf-8")


def test_legacy_action_contracts_reexport_robot_motion_objects() -> None:
    assert LegacyRobotAction is RobotMotionRobotAction
    assert LegacyActionVectorEncoding is RobotMotionActionVectorEncoding
    assert LegacyArmActionSpec is RobotMotionArmActionSpec
    assert LegacyHandActionSpec is RobotMotionHandActionSpec
    assert LegacyRobotActionResolver is RobotMotionRobotActionResolver
    assert LegacyVectorActionLayout is RobotMotionVectorActionLayout
    assert legacy_encode_robot_action is robot_motion_encode_robot_action
    assert legacy_decode_robot_action is robot_motion_decode_robot_action
    assert legacy_robot_action_from_vector_action is robot_motion_robot_action_from_vector_action


def test_robot_action_encode_decode_resolve_from_new_namespace() -> None:
    action = RobotMotionRobotAction.bimanual_ee_pose(
        left_arm=np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        right_hand=np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        timestamp=12.5,
        source="unit_test",
    )

    encoded = robot_motion_encode_robot_action(action)
    decoded = robot_motion_decode_robot_action(encoded, timestamp=action.timestamp, source=action.source)
    command = RobotMotionRobotActionResolver().resolve(decoded)

    assert isinstance(command, RobotMotionTeleopCommand)
    assert isinstance(command, LegacyTeleopCommand)
    assert set(command.commands) == {"left_arm", "right_hand"}
    assert command.commands["left_arm"].command_type == "cartesian_pose"
    assert command.commands["left_arm"].value.shape == (4, 4)
    assert command.commands["right_hand"].command_type == "joint_vector"
    np.testing.assert_array_equal(command.commands["right_hand"].value, action.hand_qpos_by_side["right"])


def test_robot_action_vector_dimensions_are_explicit() -> None:
    assert EE_POSE_VECTOR_DIM == 26
    assert QPOS_VECTOR_DIM == 24
    assert EE_POSE_HAND16_VECTOR_DIM == 46
    assert QPOS_HAND16_VECTOR_DIM == 44
    assert ROBOT_ACTION_VECTOR_LAYOUT_VERSION == "robot_action_vector.v1"
    assert ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION == "robot_action_vector.hand16.v1"
    assert legacy_codec_module.ROBOT_ACTION_VECTOR_LAYOUT_VERSION == (
        ROBOT_ACTION_VECTOR_LAYOUT_VERSION
    )
    assert legacy_codec_module.ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION == (
        ROBOT_ACTION_VECTOR_HAND16_LAYOUT_VERSION
    )


def test_vector_action_pipeline_from_new_namespace() -> None:
    layout = RobotMotionVectorActionLayout(
        action_dim=26,
        arm_pose_slices={"left": slice(0, 7), "right": slice(7, 14)},
        hand_slices={"left": slice(14, 20), "right": slice(20, 26)},
    )
    vector = np.arange(26, dtype=np.float32)

    result = robot_motion_robot_action_from_vector_action(
        vector=vector,
        layout=layout,
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=True,
        timestamp=3.0,
        source="vector_test",
    )

    assert result.faults == ()
    assert result.robot_action is not None
    assert result.robot_action.active_effectors == ("right_arm", "right_hand")
    np.testing.assert_array_equal(result.robot_action.arm_pose_by_side["right"], vector[7:14])
    np.testing.assert_array_equal(result.robot_action.hand_qpos_by_side["right"], vector[20:26])


def test_robot_motion_frames_match_legacy_arm_frame_conversions() -> None:
    pose_xyzquat = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    assert legacy_frames_module.convert_pose_between_arm_control_frames is (
        robot_frames_module.convert_pose_between_arm_control_frames
    )
    assert legacy_frames_module.canonical_arm_controlled_frame is (
        robot_frames_module.canonical_arm_controlled_frame
    )

    robot_converted = robot_frames_module.convert_pose_between_arm_control_frames(
        pose_xyzquat,
        from_frame=robot_frames_module.ARM_ENDPOINT_FRAME,
        to_frame=robot_frames_module.HAND_BASE_TCP_FRAME,
    )
    legacy_converted = legacy_frames_module.convert_pose_between_arm_control_frames(
        pose_xyzquat,
        from_frame=legacy_frames_module.ARM_ENDPOINT_FRAME,
        to_frame=legacy_frames_module.HAND_BASE_TCP_FRAME,
    )

    np.testing.assert_allclose(robot_converted, legacy_converted)
    np.testing.assert_allclose(
        robot_frames_module.hand_base_to_arm_endpoint_transform(),
        legacy_frames_module.hand_base_to_arm_endpoint_transform(),
    )
    assert robot_frames_module.canonical_arm_controlled_frame("arm_link6") == (
        legacy_frames_module.canonical_arm_controlled_frame("arm_link6")
    )


def test_robot_motion_transforms_match_legacy_pose_formats() -> None:
    pose_matrix = np.eye(4, dtype=np.float32)
    pose_matrix[:3, 3] = [0.1, 0.2, 0.3]
    pose_flat16 = pose_matrix.reshape(16)
    pose_12 = np.concatenate([pose_matrix[:3, 3], pose_matrix[:3, :3].reshape(9)])
    pose_xyzquat = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    assert legacy_transforms_module.as_pose_matrix is robot_transforms_module.as_pose_matrix
    assert legacy_transforms_module.parse_pose_matrix is robot_transforms_module.parse_pose_matrix
    assert robot_transforms_module.as_pose_matrix is robot_transforms_module.parse_pose_matrix
    assert legacy_transforms_module.invert_pose is robot_transforms_module.invert_pose
    assert legacy_transforms_module.interpolate_pose is robot_transforms_module.interpolate_pose

    for value in (pose_matrix, pose_flat16, pose_12, pose_xyzquat):
        np.testing.assert_allclose(
            robot_transforms_module.parse_pose_matrix(value),
            legacy_transforms_module.parse_pose_matrix(value),
        )

    expected_pose = np.eye(4, dtype=float)
    expected_pose[:3, 3] = [0.1, 0.2, 0.3]
    np.testing.assert_allclose(robot_transforms_module.as_pose_matrix(pose_xyzquat), expected_pose)

    np.testing.assert_allclose(
        robot_transforms_module.invert_pose(pose_xyzquat),
        legacy_transforms_module.invert_pose(pose_xyzquat),
    )
    assert robot_transforms_module.validate_pose_matrix(pose_xyzquat) == (
        legacy_transforms_module.validate_pose_matrix(pose_xyzquat)
    )


def test_robot_motion_tcp_matches_legacy_tcp_entrypoint() -> None:
    assert legacy_tcp_module.HAND_BASE_TCP_FRAME == robot_tcp_module.HAND_BASE_TCP_FRAME
    assert legacy_tcp_module.TOOL_TCP_FRAME == robot_tcp_module.TOOL_TCP_FRAME
    assert legacy_tcp_module.LEGACY_RM_EEF_TCP_MODE == robot_tcp_module.LEGACY_RM_EEF_TCP_MODE
    assert legacy_tcp_module.canonical_tcp_frame is robot_tcp_module.canonical_tcp_frame
    assert legacy_tcp_module.legacy_tcp_mode_requested is robot_tcp_module.legacy_tcp_mode_requested
    assert robot_tcp_module.canonical_tcp_frame("rm_eef") == robot_tcp_module.TOOL_TCP_FRAME


def test_robot_motion_gripper_scalar_mapping_matches_legacy_entrypoint() -> None:
    assert legacy_gripper_module.DEFAULT_GRIPPER_INPUT_MIN == (
        robot_gripper_module.DEFAULT_GRIPPER_INPUT_MIN
    )
    assert legacy_gripper_module.DEFAULT_GRIPPER_INPUT_MAX == (
        robot_gripper_module.DEFAULT_GRIPPER_INPUT_MAX
    )
    assert legacy_gripper_module.map_scalar_to_gripper_target is (
        robot_gripper_module.map_scalar_to_gripper_target
    )

    target, normalized = robot_gripper_module.map_scalar_to_gripper_target(
        0.5,
        input_min=0.1,
        input_max=0.9,
        output_min=0.0,
        output_max=1.0,
    )
    inverted_target, inverted_normalized = robot_gripper_module.map_scalar_to_gripper_target(
        0.5,
        input_min=0.1,
        input_max=0.9,
        invert=True,
        output_min=0.0,
        output_max=1.0,
    )

    assert np.isclose(target, 0.5)
    assert np.isclose(normalized, 0.5)
    assert np.isclose(inverted_target, 0.5)
    assert np.isclose(inverted_normalized, 0.5)


def test_robot_motion_hand_q6_mapping_matches_legacy_entrypoint() -> None:
    assert legacy_hand_mapping_module.SOURCE_HAND_Q6_ORDER is (
        robot_hand_mapping_module.SOURCE_HAND_Q6_ORDER
    )
    assert legacy_hand_mapping_module.DEFAULT_INSPIRE_HAND_JOINT_ORDER is (
        robot_hand_mapping_module.DEFAULT_INSPIRE_HAND_JOINT_ORDER
    )
    assert legacy_hand_mapping_module.TARGET_INSPIRE_HAND_Q6_ORDER == (
        "pinky",
        "ring",
        "middle",
        "index",
        "thumb_yaw",
        "thumb_flex",
    )

    result = robot_hand_mapping_module.map_q6_to_dexhand_target(
        np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    )

    np.testing.assert_allclose(result.source_q6, [1.0, 0.8, 0.6, 0.4, 0.0, 0.2])
    np.testing.assert_allclose(result.target_q6, [0.0, 0.2, 0.4, 0.6, 1.0, 0.8])


def test_robot_motion_action_modules_do_not_import_teleop_runtime() -> None:
    for module in (
        robot_frames_module,
        robot_tcp_module,
        robot_transforms_module,
        robot_schema_module,
        robot_arm_spec_module,
        robot_hand_spec_module,
        robot_hand_mapping_module,
        robot_codec_module,
        robot_executor_module,
        robot_gripper_module,
        robot_vector_module,
    ):
        assert "teleop_runtime" not in inspect.getsource(module)


def test_teleop_action_runtime_defaults_use_robot_motion_common_contracts() -> None:
    adapters_source = _repo_source("teleop_runtime/action/adapters.py")
    assert "from robot_motion.action import" in adapters_source
    assert "from robot_motion.runtime import" in adapters_source
    assert "from .schema import" not in adapters_source
    assert "from .arm_spec import" not in adapters_source
    assert "from .hand_spec import" not in adapters_source
    assert "from ..schemas import" not in adapters_source

    synthetic_source = _repo_source("teleop_runtime/action/synthetic.py")
    assert "from robot_motion.action import" in synthetic_source
    assert "from robot_motion.runtime import" in synthetic_source
    assert "from .codec import" not in synthetic_source
    assert "from .executor import" not in synthetic_source
    assert "from .schema import" not in synthetic_source

    kernel_source = _repo_source("teleop_runtime/core/kernel.py")
    assert "from robot_motion.action import" in kernel_source
    assert "from robot_motion.runtime import" in kernel_source
    assert "from ..action." not in kernel_source
    assert "from ..schemas import" not in kernel_source
