from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.action import (
    ArmActionSpec,
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_MIXED_V1,
    BIMANUAL_QPOS_V1,
    EE_POSE_HAND16_VECTOR_DIM,
    QPOS_HAND16_VECTOR_DIM,
    decode_robot_action,
    encode_robot_action,
    HandActionSpec,
    NullCommandDispatcher,
    RobotAction,
    RobotActionResolver,
    TeleopCommandActionAdapterConfig,
    TeleopRetargeterActionSource,
    matrix_to_xyzquat,
    robot_action_from_teleop_command,
    effector_name,
    xyzquat_to_matrix,
)
from teleop_runtime.arm_frames import ARM_ENDPOINT_FRAME, TOOL_TCP_FRAME
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation


class TeleopActionContractTest(unittest.TestCase):
    def test_bimanual_ee_pose_derives_mask_and_frames(self) -> None:
        action = RobotAction.bimanual_ee_pose(
            left_arm=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            right_hand=np.arange(6, dtype=np.float32),
            timestamp=12.3,
            source="vive",
        )

        self.assertEqual(action.action_space, BIMANUAL_EE_POSE_V1)
        self.assertEqual(action.active_effectors, ("left_arm", "right_hand"))
        self.assertEqual(action.active_sides, ("left", "right"))
        self.assertEqual(action.arm_frame("left"), "left_arm_base")
        self.assertEqual(action.arm_controlled_frame("left"), TOOL_TCP_FRAME)
        self.assertTrue(action.is_active("left_arm"))
        self.assertFalse(action.is_active("right_arm"))

    def test_active_mask_can_select_single_arm_from_bimanual_values(self) -> None:
        action = RobotAction.bimanual_ee_pose(
            left_arm=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            right_arm=np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            active_effectors=("right_arm",),
        )
        command = RobotActionResolver().resolve(action)

        self.assertEqual(sorted(command.commands), ["right_arm"])
        self.assertEqual(command.commands["right_arm"].command_type, "cartesian_pose")

    def test_rejects_ambiguous_arm_pose_and_qpos_for_same_side(self) -> None:
        with self.assertRaises(ValueError):
            RobotAction(
                action_space="bad",
                arm_pose_by_side={
                    "left": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                },
                arm_qpos_by_side={"left": np.zeros((6,), dtype=np.float32)},
            )

    def test_rejects_active_effector_without_value(self) -> None:
        with self.assertRaises(ValueError):
            RobotAction.bimanual_ee_pose(
                left_arm=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                active_effectors=("right_arm",),
            )

    def test_qpos_action_resolves_to_joint_commands(self) -> None:
        action = RobotAction.bimanual_qpos(
            left_arm=np.arange(6, dtype=np.float32),
            right_hand=np.arange(6, dtype=np.float32) + 10.0,
            source="policy",
        )
        command = RobotActionResolver().resolve(action)

        self.assertEqual(action.action_space, BIMANUAL_QPOS_V1)
        self.assertEqual(sorted(command.commands), ["left_arm", "right_hand"])
        self.assertEqual(command.commands["left_arm"].command_type, "joint_vector")
        self.assertEqual(command.commands["right_hand"].command_type, "joint_vector")
        np.testing.assert_allclose(command.commands["left_arm"].value, np.arange(6))

    def test_xyzquat_to_matrix_normalizes_quaternion(self) -> None:
        matrix = xyzquat_to_matrix(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 2.0]))

        np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-6)

    def test_null_dispatcher_keeps_resolved_commands_for_audit(self) -> None:
        action = RobotAction.bimanual_qpos(left_arm=np.ones((6,), dtype=np.float32))
        dispatcher = NullCommandDispatcher()
        command = RobotActionResolver().resolve(action)

        dispatcher.dispatch(command)

        self.assertEqual(len(dispatcher.commands), 1)
        self.assertEqual(dispatcher.commands[0].commands["left_arm"].effector_name, effector_name("left", "arm"))

    def test_teleop_cartesian_command_converts_to_pose_action(self) -> None:
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=7.0)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(pose),
                frame_name="left_arm_base",
            )
        )

        action = robot_action_from_teleop_command(command)

        self.assertEqual(action.action_space, BIMANUAL_EE_POSE_V1)
        np.testing.assert_allclose(action.arm_pose_by_side["left"], pose, atol=1e-6)
        self.assertEqual(action.active_effectors, ("left_arm",))
        self.assertEqual(action.arm_controlled_frame("left"), TOOL_TCP_FRAME)
        self.assertEqual(
            action.metadata["arm_action_specs_by_side"]["left"]["controlled_frame"],
            TOOL_TCP_FRAME,
        )

    def test_resolver_preserves_explicit_arm_endpoint_controlled_frame(self) -> None:
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        spec = ArmActionSpec.endpoint("left")
        action = RobotAction.bimanual_ee_pose(
            left_arm=pose,
            metadata={"arm_action_specs_by_side": {"left": spec.to_metadata()}},
        )

        command = RobotActionResolver().resolve(action)

        left = command.commands["left_arm"]
        self.assertEqual(left.metadata["arm_controlled_frame"], ARM_ENDPOINT_FRAME)
        self.assertEqual(left.metadata["arm_action_spec"]["controlled_frame"], ARM_ENDPOINT_FRAME)

    def test_teleop_joint_command_prefers_source_pose_metadata_by_default(self) -> None:
        source_pose = np.array([0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=8.0)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
                metadata={"source_target_pose_base": xyzquat_to_matrix(source_pose)},
            )
        )

        action = robot_action_from_teleop_command(command)

        self.assertEqual(action.action_space, BIMANUAL_EE_POSE_V1)
        np.testing.assert_allclose(action.arm_pose_by_side["right"], source_pose, atol=1e-6)
        self.assertNotIn("right", action.arm_qpos_by_side)

    def test_teleop_joint_command_can_be_kept_as_qpos_action(self) -> None:
        command = TeleopCommand(timestamp=9.0)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
                metadata={"source_target_pose_base": np.eye(4, dtype=np.float32)},
            )
        )

        action = robot_action_from_teleop_command(
            command,
            config=TeleopCommandActionAdapterConfig(arm_joint_policy="qpos"),
        )

        self.assertEqual(action.action_space, BIMANUAL_QPOS_V1)
        np.testing.assert_allclose(action.arm_qpos_by_side["right"], np.arange(6))

    def test_teleop_mixed_command_gets_mixed_action_space(self) -> None:
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=10.0)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(pose),
                frame_name="left_arm_base",
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
            )
        )

        action = robot_action_from_teleop_command(command)

        self.assertEqual(action.action_space, BIMANUAL_MIXED_V1)
        self.assertIn("left", action.arm_pose_by_side)
        self.assertIn("right", action.arm_qpos_by_side)

    def test_matrix_to_xyzquat_round_trips_pose_matrix(self) -> None:
        pose = np.array([0.5, 0.6, 0.7, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        round_trip = matrix_to_xyzquat(xyzquat_to_matrix(pose))

        np.testing.assert_allclose(round_trip, pose, atol=1e-6)

    def test_teleop_retargeter_action_source_wraps_existing_retargeters(self) -> None:
        class FakeRetargeter:
            def retarget(self, observation, calibration=None):
                command = TeleopCommand(timestamp=observation.timestamp)
                command.add_command(
                    EffectorCommand(
                        effector_name="left_hand",
                        side="left",
                        command_type="joint_vector",
                        value=np.arange(6, dtype=np.float32),
                        frame_name=None,
                    )
                )
                return command

        source = TeleopRetargeterActionSource(retargeters=[FakeRetargeter()])
        observation = TeleopObservation(timestamp=11.0)

        action = source.produce_action(observation)

        self.assertEqual(action.action_space, BIMANUAL_QPOS_V1)
        self.assertEqual(action.active_effectors, ("left_hand",))
        np.testing.assert_allclose(action.hand_qpos_by_side["left"], np.arange(6))
        self.assertIsNotNone(source.last_command)

    def test_ee_pose_action_vector_round_trip_preserves_mask(self) -> None:
        action = RobotAction.bimanual_ee_pose(
            left_arm=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            right_hand=np.arange(6, dtype=np.float32),
            timestamp=4.0,
            source="teleop",
        )

        encoded = encode_robot_action(action)
        decoded = decode_robot_action(encoded, timestamp=action.timestamp, source=action.source)

        self.assertEqual(encoded.vector.shape, (26,))
        self.assertTrue(encoded.effector_mask["left_arm"])
        self.assertFalse(encoded.effector_mask["right_arm"])
        self.assertFalse(np.any(encoded.component_mask[7:14]))
        self.assertEqual(decoded.active_effectors, ("left_arm", "right_hand"))
        np.testing.assert_allclose(decoded.arm_pose_by_side["left"], action.arm_pose_by_side["left"])
        np.testing.assert_allclose(decoded.hand_qpos_by_side["right"], action.hand_qpos_by_side["right"])

    def test_qpos_action_vector_round_trip(self) -> None:
        action = RobotAction.bimanual_qpos(
            left_arm=np.arange(6, dtype=np.float32),
            right_arm=np.arange(6, dtype=np.float32) + 6,
            left_hand=np.arange(6, dtype=np.float32) + 12,
        )

        encoded = encode_robot_action(action)
        decoded = decode_robot_action(encoded)

        self.assertEqual(encoded.vector.shape, (24,))
        self.assertEqual(decoded.action_space, BIMANUAL_QPOS_V1)
        np.testing.assert_allclose(decoded.arm_qpos_by_side["right"], np.arange(6) + 6)
        np.testing.assert_allclose(decoded.hand_qpos_by_side["left"], np.arange(6) + 12)

    def test_hand16_ee_pose_action_vector_round_trip(self) -> None:
        action = RobotAction.bimanual_ee_pose_hand16(
            left_arm=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            right_hand=np.arange(16, dtype=np.float32),
            timestamp=4.0,
            source="teleop",
            metadata={"hand_active_dof_by_side": {"right": 6}},
        )

        encoded = encode_robot_action(action)
        decoded = decode_robot_action(encoded, timestamp=action.timestamp, source=action.source)

        self.assertEqual(action.action_space, BIMANUAL_EE_POSE_HAND16_V1)
        self.assertEqual(encoded.vector.shape, (EE_POSE_HAND16_VECTOR_DIM,))
        self.assertEqual(encoded.layout_version, "robot_action_vector.hand16.v1")
        self.assertEqual(decoded.active_effectors, ("left_arm", "right_hand"))
        np.testing.assert_allclose(decoded.hand_qpos_by_side["right"], np.arange(16))

    def test_hand16_qpos_action_vector_round_trip(self) -> None:
        action = RobotAction.bimanual_qpos_hand16(
            left_arm=np.arange(6, dtype=np.float32),
            left_hand=np.arange(16, dtype=np.float32) + 6,
        )

        encoded = encode_robot_action(action)
        decoded = decode_robot_action(encoded)

        self.assertEqual(action.action_space, BIMANUAL_QPOS_HAND16_V1)
        self.assertEqual(encoded.vector.shape, (QPOS_HAND16_VECTOR_DIM,))
        np.testing.assert_allclose(decoded.hand_qpos_by_side["left"], np.arange(16) + 6)

    def test_teleop_hand_q1_can_pad_to_hand16_action(self) -> None:
        command = TeleopCommand(timestamp=12.0)
        command.add_command(
            EffectorCommand(
                effector_name="right_hand",
                side="right",
                command_type="joint_vector",
                value=np.array([0.7], dtype=np.float32),
                frame_name=None,
            )
        )

        action = robot_action_from_teleop_command(
            command,
            config=TeleopCommandActionAdapterConfig(hand_slot_dof=16),
        )

        self.assertEqual(action.action_space, BIMANUAL_QPOS_HAND16_V1)
        right_spec = action.metadata["hand_action_specs_by_side"]["right"]
        self.assertEqual(right_spec["slot_count"], 16)
        self.assertEqual(right_spec["active_dof"], 1)
        self.assertEqual(action.metadata["hand_active_dof_by_side"], {"right": 1})
        self.assertEqual(action.metadata["hand_active_mask_by_side"]["right"], [True] + [False] * 15)
        self.assertEqual(action.metadata["hand_invalid_slot_value_by_side"], {"right": "NaN"})
        expected = np.full((16,), np.nan, dtype=np.float32)
        expected[0] = 0.7
        np.testing.assert_allclose(action.hand_qpos_by_side["right"], expected, equal_nan=True)

        encoded = encode_robot_action(action)
        self.assertTrue(encoded.component_mask[28])
        self.assertFalse(np.any(encoded.component_mask[29:44]))
        decoded = decode_robot_action(encoded)
        np.testing.assert_allclose(decoded.hand_qpos_by_side["right"], expected, equal_nan=True)

    def test_hand16_resolver_trims_invalid_nan_slots(self) -> None:
        hand = np.full((16,), np.nan, dtype=np.float32)
        hand[:2] = [0.2, 0.8]
        spec = HandActionSpec.q16("right", active_dof=2)
        action = RobotAction.bimanual_qpos_hand16(
            right_hand=hand,
            metadata={"hand_action_specs_by_side": {"right": spec.to_metadata()}},
        )

        command = RobotActionResolver().resolve(action)

        np.testing.assert_allclose(command.commands["right_hand"].value, [0.2, 0.8])
        self.assertEqual(command.commands["right_hand"].metadata["hand_active_mask"], [True, True] + [False] * 14)
        self.assertEqual(command.commands["right_hand"].metadata["hand_action_spec"]["active_dof"], 2)
        np.testing.assert_allclose(
            command.commands["right_hand"].metadata["hand_qpos_full"],
            hand,
            equal_nan=True,
        )

    def test_hand16_resolver_requires_hand_spec(self) -> None:
        hand = np.full((16,), np.nan, dtype=np.float32)
        hand[0] = 0.2
        action = RobotAction.bimanual_qpos_hand16(right_hand=hand)

        with self.assertRaises(ValueError):
            RobotActionResolver().resolve(action)

    def test_mixed_action_has_no_stable_vector_codec(self) -> None:
        pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        action = RobotAction(
            action_space=BIMANUAL_MIXED_V1,
            arm_pose_by_side={"left": pose},
            arm_qpos_by_side={"right": np.arange(6, dtype=np.float32)},
        )

        with self.assertRaises(ValueError):
            encode_robot_action(action)


if __name__ == "__main__":
    unittest.main()
