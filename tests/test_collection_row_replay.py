from __future__ import annotations

import json
import unittest

import numpy as np

from teleop_runtime.action import RobotAction, encode_robot_action
from teleop_runtime.action.row_replay import (
    synthetic_frame_from_collection_row,
    synthetic_frames_from_collection_rows,
)
from teleop_runtime.action.synthetic import run_synthetic_action_replay


class CollectionRowReplayTest(unittest.TestCase):
    def test_prefers_canonical_action_vector_from_sample_metadata(self) -> None:
        action = RobotAction.bimanual_qpos(
            left_arm=np.arange(6, dtype=np.float32),
            right_hand=np.arange(6, dtype=np.float32) + 10.0,
        )
        encoded = encode_robot_action(action)
        row = {
            "timestamp": 12.0,
            "index": 5,
            "sample_metadata_json": json.dumps(
                {
                    "canonical_action_record": {
                        "action_vector": {
                            "action_space": encoded.action_space,
                            "vector": encoded.vector.tolist(),
                            "component_mask": encoded.component_mask.tolist(),
                            "effector_mask": dict(encoded.effector_mask),
                            "layout_version": encoded.layout_version,
                        }
                    }
                }
            ),
        }

        frame = synthetic_frame_from_collection_row(row)
        report = run_synthetic_action_replay([frame])

        self.assertEqual(report.step_count, 1)
        self.assertEqual(report.results[0].action.active_effectors, ("left_arm", "right_hand"))
        self.assertEqual(sorted(report.commands[0].commands), ["left_arm", "right_hand"])

    def test_falls_back_to_legacy_26d_action_vector(self) -> None:
        vector = np.full((26,), np.nan, dtype=np.float32)
        vector[7:14] = np.array([0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        vector[20:26] = np.arange(6, dtype=np.float32)
        row = {
            "timestamp": 13.0,
            "active_sides": json.dumps(["right"]),
            "action": vector.tolist(),
        }

        frame = synthetic_frame_from_collection_row(row, prefer_canonical=False)
        report = run_synthetic_action_replay([frame])

        self.assertEqual(report.results[0].action.active_effectors, ("right_arm", "right_hand"))
        self.assertEqual(sorted(report.commands[0].commands), ["right_arm", "right_hand"])

    def test_builds_multiple_synthetic_frames_from_rows(self) -> None:
        rows = [
            {
                "timestamp": 1.0,
                "active_sides": ["left"],
                "action.left_arm.pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            },
            {
                "timestamp": 2.0,
                "active_sides": ["right"],
                "action.right_arm.pose": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
            },
        ]

        frames = synthetic_frames_from_collection_rows(rows, prefer_canonical=False)
        report = run_synthetic_action_replay(frames)

        self.assertEqual(report.step_count, 2)
        self.assertEqual(sorted(report.commands[0].commands), ["left_arm"])
        self.assertEqual(sorted(report.commands[1].commands), ["right_arm"])


if __name__ == "__main__":
    unittest.main()
