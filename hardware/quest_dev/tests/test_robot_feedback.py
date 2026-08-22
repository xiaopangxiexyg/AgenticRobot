from __future__ import annotations

import json
import unittest

from hardware.quest_dev.features.visualization import robot_feedback


class FakeRedis:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.read_keys: list[str] = []

    def get(self, key: str):
        self.read_keys.append(key)
        return self.values.get(key)


def feedback(timestamp: float = 100.0, robot: str = "ar5_1") -> dict:
    return {
        "timestamp": timestamp,
        "robot": robot,
        "control_ready": True,
        "mode": "test",
        "tcp_pose": {
            "frame": "robot_world",
            "position": [0.1, 0.2, 0.3],
            "orientation_quat_xyzw": [0, 0, 0, 1],
        },
    }


class RobotFeedbackTests(unittest.TestCase):
    def test_valid_feedback_becomes_renderable_record(self) -> None:
        record = robot_feedback.feedback_record(
            feedback(),
            expected_robot="ar5_1",
            key="ar5:feedback:ar5_1",
            max_age_s=0.25,
            now=100.1,
        )
        self.assertTrue(record["fresh"])
        self.assertTrue(record["control_ready"])
        self.assertEqual(record["matrix_robot_from_tcp"][12:15], [0.1, 0.2, 0.3])

    def test_stale_feedback_contains_no_pose(self) -> None:
        record = robot_feedback.feedback_record(
            feedback(),
            expected_robot="ar5_1",
            key="ar5:feedback:ar5_1",
            max_age_s=0.25,
            now=101.0,
        )
        self.assertFalse(record["fresh"])
        self.assertEqual(record["error"], "stale")
        self.assertNotIn("matrix_robot_from_tcp", record)

    def test_robot_identity_mismatch_is_rejected(self) -> None:
        record = robot_feedback.feedback_record(
            feedback(robot="ar5_2"),
            expected_robot="ar5_1",
            key="ar5:feedback:ar5_1",
            max_age_s=0.25,
            now=100.0,
        )
        self.assertEqual(record["error"], "robot_mismatch")

    def test_polling_only_reads_configured_keys(self) -> None:
        reader = FakeRedis({"key-1": json.dumps(feedback())})
        message = robot_feedback.poll_robot_feedback(
            reader,
            {"ar5_1": "key-1"},
            max_age_s=0.25,
            now=100.0,
        )
        self.assertEqual(reader.read_keys, ["key-1"])
        self.assertEqual(message["type"], "robot_tcp")
        self.assertTrue(message["robots"]["ar5_1"]["fresh"])


if __name__ == "__main__":
    unittest.main()
