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


if __name__ == "__main__":
    unittest.main()
