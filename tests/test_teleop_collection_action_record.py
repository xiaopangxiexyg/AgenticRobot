from __future__ import annotations

import time
import unittest

import numpy as np

from teleop_runtime.action import BIMANUAL_EE_POSE_V1, xyzquat_to_matrix
from teleop_runtime.collection.action_record import collection_action_record_metadata
from teleop_runtime.collection.arm_state_polling import (
    ArmHardwareStatePoller,
    ArmHardwareStatePollerConfig,
)
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.sample_assembler import CollectorSampleBuilder
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.session import SessionStepResult


class _FakeArmBackend:
    def __init__(self, qpos: np.ndarray, tcp_pose: np.ndarray):
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.tcp_pose = np.asarray(tcp_pose, dtype=np.float32)

    def get_state(self) -> dict:
        return {
            "current_qpos": self.qpos.copy(),
            "last_target_qpos": self.qpos.copy(),
            "last_command_timestamp": 11.0,
        }

    def compute_tcp_pose(self) -> np.ndarray:
        return self.tcp_pose.copy()


class _FakeHandBackend:
    def get_state(self, **_kwargs) -> dict:
        return {"joint_pos": np.zeros(6, dtype=np.float32)}


class _FakeRefreshArmBackend:
    def __init__(self, qpos: np.ndarray, tcp_pose: np.ndarray, timestamp: float):
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
        self.timestamp = float(timestamp)
        self.calls: list[tuple[int | None, float | None]] = []

    def refresh_state_from_hardware(
        self,
        *,
        hardware_read_retries: int | None = None,
        hardware_read_retry_delay_sec: float | None = None,
    ) -> dict:
        self.calls.append((hardware_read_retries, hardware_read_retry_delay_sec))
        return {
            "hardware_qpos": self.qpos.copy(),
            "current_tcp_pose": self.tcp_pose.copy(),
            "hardware_refresh_time": self.timestamp,
            "hardware_result_code": 0,
            "hardware_result_text": "ok",
        }


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

    def test_actual_wrist_camera_pose_is_recorded_without_changing_training_state(self) -> None:
        current_pose_xyzquat = np.array(
            [0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        actual_pose_xyzquat = np.array(
            [0.71, 0.72, 0.73, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        actual_qpos = np.arange(6, dtype=np.float32) + 10.0
        provider_calls: list[float] = []

        def actual_state_provider(_components, sample_timestamp: float) -> dict:
            provider_calls.append(float(sample_timestamp))
            return {
                "left": {
                    "actual_qpos": actual_qpos,
                    "actual_tcp_pose": xyzquat_to_matrix(actual_pose_xyzquat),
                    "wrist_camera_pose_base": xyzquat_to_matrix(actual_pose_xyzquat),
                    "wrist_camera_pose_timestamp": sample_timestamp - 0.01,
                    "wrist_camera_pose_age_sec": 0.01,
                    "wrist_camera_pose_source": "fake_hardware_feedback",
                }
            }

        builder = CollectorSampleBuilder(
            arm_state_metadata_provider=actual_state_provider,
        )
        components = CollectorComponents(
            session=None,
            backends={
                "left_arm": _FakeArmBackend(
                    np.arange(6, dtype=np.float32),
                    xyzquat_to_matrix(current_pose_xyzquat),
                ),
                "left_hand": _FakeHandBackend(),
            },
        )
        step = SessionStepResult(
            timestamp=12.34,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=12.33),
        )

        sample = builder(step, components)
        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertEqual(provider_calls, [12.34])
        np.testing.assert_allclose(
            sample.observation_state[0:6],
            np.arange(6, dtype=np.float32),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            sample.observation_state[6:13],
            current_pose_xyzquat,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row["observation.left_arm.pose"],
            current_pose_xyzquat,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row["observation.arm.left.actual_qpos"],
            actual_qpos,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row["observation.arm.left.actual_tcp_pose"],
            actual_pose_xyzquat,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row["observation.wrist_camera.left.pose_base"],
            actual_pose_xyzquat,
            atol=1e-6,
        )
        self.assertEqual(row["observation.wrist_camera.left.source"], "fake_hardware_feedback")
        self.assertEqual(len(row["observation.state"]), 38)

    def test_actual_wrist_camera_poller_uses_lightweight_feedback_read(self) -> None:
        timestamp = time.time()
        pose_xyzquat = np.array(
            [0.41, 0.42, 0.43, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        qpos = np.arange(6, dtype=np.float32) + 20.0
        backend = _FakeRefreshArmBackend(qpos, xyzquat_to_matrix(pose_xyzquat), timestamp)
        poller = ArmHardwareStatePoller(
            backends_by_side={"left": backend},
            config=ArmHardwareStatePollerConfig(
                poll_hz=30.0,
                stale_after_sec=0.2,
                hardware_read_retries=0,
                hardware_read_retry_delay_sec=0.0,
            ),
        )

        poller._poll_once("left", backend)
        latest = poller.latest_by_side(sample_timestamp=timestamp + 0.02)["left"]

        self.assertEqual(backend.calls, [(0, 0.0)])
        np.testing.assert_allclose(latest["actual_qpos"], qpos, atol=1e-6)
        np.testing.assert_allclose(
            latest["wrist_camera_pose_base"],
            xyzquat_to_matrix(pose_xyzquat),
            atol=1e-6,
        )
        self.assertEqual(latest["actual_hardware_result_code"], 0)
        self.assertAlmostEqual(latest["wrist_camera_pose_sample_time_offset_sec"], 0.02)


if __name__ == "__main__":
    unittest.main()
