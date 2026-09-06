from __future__ import annotations

import json
import unittest

import numpy as np

from hardware.quest_dev.features.visualization import robot_feedback
from hardware.quest_dev.features.visualization.frame_binding import (
    AcceptedFrameBinding,
    QuestFrameIdentity,
    RobotFrameIdentity,
)


class FakeRedis:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values
        self.read_keys: list[str] = []

    def get(self, key: str):
        self.read_keys.append(key)
        return self.values.get(key)


def make_frame_binding() -> AcceptedFrameBinding:
    return AcceptedFrameBinding(
        binding_id="binding-1",
        acceptance_receipt_sha256="a" * 64,
        acceptance_profile_id="profile-1",
        capture_evidence_semantic_sha256="b" * 64,
        diagnostic_fit_semantic_sha256="c" * 64,
        dataset_sha256="d" * 64,
        fit_semantic_sha256="e" * 64,
        quest=QuestFrameIdentity(
            session_id="xr-session-1",
            reference_space_type="local-floor",
            reference_space_generation=0,
            reference_epoch_id="xr-session-1:local-floor:0",
            controller_source_session_id="controller-source-1",
            reference_frame_id="quest-local-floor",
        ),
        robot=RobotFrameIdentity(
            robot_id="ar5_1",
            source_instance_id="rm65-right-runtime",
            source_epoch_id="rm65-right-epoch-7",
            source_binding_hash="rm65-source-binding-hash",
            component_id="right_arm",
            stream_id="robot-state.right-arm.tool-tcp-pose",
            reference_frame_id="right-arm-base",
            reference_frame_spec_hash="right-base-frame-spec",
            moving_frame_id="tool-tcp",
            tool_profile_hash="sha256:" + "1" * 64,
            pose_schema_hash="pose-schema-xyzquat-v1",
            fk_contract_id="rm65-hardware-qpos-fk-v1",
            fk_contract_hash="rm65-hardware-qpos-fk-hash",
            coordinate_binding_hash="f" * 64,
            source_kind="real_robot_hardware_qpos_fk",
        ),
        matrix_quest_from_robot_reference=np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


def feedback(
    timestamp: float = 100.0,
    robot: str = "ar5_1",
    *,
    pose_binding: dict[str, object] | None = None,
) -> dict:
    payload = {
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
    if pose_binding is not None:
        payload["pose_binding"] = pose_binding
    return payload


class RobotFeedbackTests(unittest.TestCase):
    def test_valid_feedback_is_renderable_but_never_control_ready(self) -> None:
        record = robot_feedback.feedback_record(
            feedback(),
            expected_robot="ar5_1",
            key="ar5:feedback:ar5_1",
            max_age_s=0.25,
            now=100.1,
        )
        self.assertTrue(record["fresh"])
        self.assertFalse(record["control_ready"])
        self.assertFalse(record["alignment_ready"])
        self.assertEqual(record["matrix_robot_from_tcp"][12:15], [0.1, 0.2, 0.3])

    def test_reviewed_binding_adds_quest_tcp_matrix(self) -> None:
        binding = make_frame_binding()
        record = robot_feedback.feedback_record(
            feedback(pose_binding=binding.robot.as_dict()),
            expected_robot="ar5_1",
            key="ar5:feedback:ar5_1",
            max_age_s=0.25,
            now=100.1,
            frame_binding=binding,
        )
        self.assertTrue(record["fresh"])
        self.assertTrue(record["alignment_ready"])
        self.assertEqual(
            record["matrix_quest_from_tcp"][12:15],
            [1.1, 2.2, 3.3],
        )
        self.assertEqual(
            record["frame_binding"]["quest_reference_epoch_id"],
            "xr-session-1:local-floor:0",
        )

    def test_binding_mismatch_never_adds_quest_matrix(self) -> None:
        binding = make_frame_binding()
        metadata = binding.robot.as_dict()
        metadata["source_epoch_id"] = "different-epoch"
        record = robot_feedback.feedback_record(
            feedback(pose_binding=metadata),
            expected_robot="ar5_1",
            key="ar5:feedback:ar5_1",
            max_age_s=0.25,
            now=100.1,
            frame_binding=binding,
        )
        self.assertTrue(record["fresh"])
        self.assertFalse(record["alignment_ready"])
        self.assertEqual(record["alignment_error"], "source_epoch_id_mismatch")
        self.assertNotIn("matrix_quest_from_tcp", record)

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
