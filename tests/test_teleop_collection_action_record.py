from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.action import BIMANUAL_EE_POSE_V1, xyzquat_to_matrix
from teleop_runtime.collection.action_record import collection_action_record_metadata
from teleop_runtime.collection.sample_assembler import CollectorSampleBuilder
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.session import SessionStepResult


class TeleopCollectionActionRecordTest(unittest.TestCase):
    def test_collection_action_record_metadata_uses_canonical_pose_action(self) -> None:
        pose = np.array([0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=2.0)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
                metadata={"source_target_pose_base": xyzquat_to_matrix(pose)},
            )
        )
        step = SessionStepResult(
            timestamp=2.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=1.9),
            command=command,
            command_dispatched=True,
        )

        metadata = collection_action_record_metadata(step)

        self.assertEqual(metadata["action"]["action_space"], BIMANUAL_EE_POSE_V1)
        self.assertEqual(metadata["action_vector"]["action_space"], BIMANUAL_EE_POSE_V1)
        self.assertEqual(len(metadata["action_vector"]["vector"]), 26)
        self.assertTrue(metadata["action_vector"]["effector_mask"]["left_arm"])
        self.assertFalse(metadata["action_vector"]["effector_mask"]["right_arm"])
        np.testing.assert_allclose(
            metadata["action"]["arm_pose_by_side"]["left"],
            pose,
            atol=1e-6,
        )
        self.assertTrue(metadata["execution"]["sent"])
        self.assertEqual(metadata["execution"]["metadata"]["session_state"], "RUNNING")

    def test_collection_action_record_does_not_mark_unsent_debug_command_as_sent(self) -> None:
        command = TeleopCommand(timestamp=2.5)
        command.add_command(
            EffectorCommand(
                effector_name="left_hand",
                side="left",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
            )
        )
        step = SessionStepResult(
            timestamp=2.5,
            state=TeleopRunState.DEBUG_ONLY,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=2.4),
            command=command,
            command_dispatched=False,
        )

        metadata = collection_action_record_metadata(step)

        self.assertFalse(metadata["execution"]["sent"])
        action_metadata = metadata["action"]["metadata"]
        self.assertIn("hand_action_specs_by_side", action_metadata)
        self.assertEqual(
            action_metadata["hand_action_specs_by_side"]["left"]["active_mask"],
            [True, True, True, True, True, True],
        )

    def test_default_sample_metadata_includes_canonical_action_record(self) -> None:
        pose = np.array([0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=3.0)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(pose),
                frame_name="right_arm_base",
            )
        )
        step = SessionStepResult(
            timestamp=3.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=2.9),
            command=command,
        )

        metadata = CollectorSampleBuilder._default_metadata(
            step,
            None,
            {},
            {},
            {},
        )

        self.assertIn("canonical_action_record", metadata)
        canonical = metadata["canonical_action_record"]["action"]
        self.assertEqual(canonical["active_effectors"], ["right_arm"])
        np.testing.assert_allclose(canonical["arm_pose_by_side"]["right"], pose, atol=1e-6)

    def test_default_sample_metadata_strips_ik_collision_debug_metadata(self) -> None:
        pose = np.array([0.31, 0.32, 0.33, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(
            timestamp=4.0,
            metadata={
                "source_target_pose_base": {"left": xyzquat_to_matrix(pose)},
                "collision_avoidance_status": "tracking",
                "ik_candidate_selection": {"selected": "original"},
            },
        )
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
                metadata={
                    "source_target_pose_base": xyzquat_to_matrix(pose),
                    "collision_avoidance_status": "tracking",
                    "ik_candidate_selection": {"selected": "original"},
                    "joint_limit_margin": {"near_limit": False},
                },
            )
        )
        step = SessionStepResult(
            timestamp=4.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=3.9),
            command=command,
        )

        metadata = CollectorSampleBuilder._default_metadata(
            step,
            None,
            {},
            {},
            {},
        )

        self.assertIn("source_target_pose_base", metadata["command_metadata"])
        self.assertNotIn("collision_avoidance_status", metadata["command_metadata"])
        self.assertNotIn("ik_candidate_selection", metadata["command_metadata"])
        left_arm_metadata = metadata["command_metadata_by_effector"]["left_arm"]
        self.assertIn("source_target_pose_base", left_arm_metadata)
        self.assertNotIn("collision_avoidance_status", left_arm_metadata)
        self.assertNotIn("ik_candidate_selection", left_arm_metadata)
        self.assertNotIn("joint_limit_margin", left_arm_metadata)
        self.assertNotIn("command", metadata["canonical_action_record"]["execution"])

    def test_debug_sample_metadata_keeps_ik_collision_metadata(self) -> None:
        command = TeleopCommand(
            timestamp=5.0,
            metadata={
                "collision_avoidance_status": "tracking",
                "ik_candidate_selection": {"selected": "original"},
            },
        )
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="cartesian_pose",
                value=np.eye(4, dtype=np.float32),
                frame_name="right_arm_base",
                metadata={
                    "collision_avoidance_status": "tracking",
                    "ik_candidate_selection": {"selected": "original"},
                },
            )
        )
        step = SessionStepResult(
            timestamp=5.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=4.9),
            command=command,
        )

        metadata = CollectorSampleBuilder._default_metadata(
            step,
            None,
            {},
            {},
            {},
            include_debug_metadata=True,
        )

        self.assertEqual(metadata["command_metadata"]["collision_avoidance_status"], "tracking")
        self.assertIn("ik_candidate_selection", metadata["command_metadata"])
        right_arm_metadata = metadata["command_metadata_by_effector"]["right_arm"]
        self.assertEqual(right_arm_metadata["collision_avoidance_status"], "tracking")
        self.assertIn("ik_candidate_selection", right_arm_metadata)
        self.assertIn("command", metadata["canonical_action_record"]["execution"])


if __name__ == "__main__":
    unittest.main()
