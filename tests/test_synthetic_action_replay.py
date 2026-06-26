from __future__ import annotations

import unittest

import numpy as np

from teleop_runtime.action import RobotAction, encode_robot_action
from teleop_runtime.action.synthetic import (
    SyntheticReplayFrame,
    run_synthetic_action_replay,
)


class SyntheticActionReplayTest(unittest.TestCase):
    def test_replay_dispatches_robot_actions_without_hardware(self) -> None:
        frames = [
            SyntheticReplayFrame(
                observation={"step": 0},
                action=RobotAction.bimanual_ee_pose(
                    left_arm=np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                    left_hand=np.arange(6, dtype=np.float32),
                    timestamp=1.0,
                    source="unit",
                ),
                timestamp=1.0,
            ),
            SyntheticReplayFrame(
                observation={"step": 1},
                action=RobotAction.bimanual_qpos(
                    right_arm=np.arange(6, dtype=np.float32),
                    right_hand=np.arange(6, dtype=np.float32) + 6.0,
                    timestamp=1.1,
                    source="unit",
                ),
                timestamp=1.1,
            ),
        ]

        report = run_synthetic_action_replay(frames)

        self.assertEqual(report.step_count, 2)
        self.assertEqual(report.command_count, 2)
        self.assertEqual(report.sent_command_count, 2)
        self.assertEqual(sorted(report.commands[0].commands), ["left_arm", "left_hand"])
        self.assertEqual(sorted(report.commands[1].commands), ["right_arm", "right_hand"])
        stats = report.phase_stats()
        self.assertIn("action.execute", stats)
        self.assertEqual(stats["action.execute"].count, 2)

    def test_replay_accepts_encoded_action_vectors(self) -> None:
        action = RobotAction.bimanual_ee_pose(
            right_arm=np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            source="teleop",
        )
        encoded = encode_robot_action(action)

        report = run_synthetic_action_replay(
            [SyntheticReplayFrame(action=encoded, observation={"state": "pre"}, timestamp=5.0)]
        )

        self.assertEqual(report.step_count, 1)
        result = report.results[0]
        self.assertEqual(result.action.active_effectors, ("right_arm",))
        self.assertEqual(sorted(result.command.commands), ["right_arm"])
        np.testing.assert_allclose(
            result.command.commands["right_arm"].metadata["target_xyzquat"],
            action.arm_pose_by_side["right"],
        )

    def test_replay_can_run_without_dispatching_commands(self) -> None:
        action = RobotAction.bimanual_qpos(left_arm=np.zeros((6,), dtype=np.float32))

        report = run_synthetic_action_replay(
            [SyntheticReplayFrame(action=action)],
            dispatch_enabled=False,
        )

        self.assertEqual(report.command_count, 0)
        self.assertFalse(report.results[0].execution.sent)
        self.assertEqual(sorted(report.results[0].command.commands), ["left_arm"])


if __name__ == "__main__":
    unittest.main()
