from __future__ import annotations

import time
import unittest

import numpy as np

from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    DEPLOY_ACTION_DIM,
    DEPLOY_OBSERVATION_STATE_DIM,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
)
from teleop_runtime.action import BIMANUAL_EE_POSE_V1, xyzquat_to_matrix
from teleop_runtime.collection.action_record import collection_action_record_metadata
from teleop_runtime.collection.arm_state_polling import (
    ArmHardwareStatePoller,
    ArmHardwareStatePollerConfig,
)
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilder,
    CollectorSampleBuilderConfig,
)
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.session import SessionStepResult


class _FakeArmBackend:
    def __init__(
        self,
        qpos: np.ndarray,
        tcp_pose: np.ndarray,
        target_tcp_pose: np.ndarray | None = None,
        last_command_timestamp: float = 11.0,
    ):
        self.qpos = np.asarray(qpos, dtype=np.float32)
        self.tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
        self.target_tcp_pose = (
            self.tcp_pose.copy()
            if target_tcp_pose is None
            else np.asarray(target_tcp_pose, dtype=np.float32)
        )
        self.last_command_timestamp = float(last_command_timestamp)

    def get_state(self) -> dict:
        return {
            "current_qpos": self.qpos.copy(),
            "last_target_qpos": self.qpos.copy(),
            "last_target_tcp_pose": self.target_tcp_pose.copy(),
            "last_command_timestamp": self.last_command_timestamp,
        }

    def compute_tcp_pose(self, **_kwargs) -> np.ndarray:
        return self.tcp_pose.copy()


class _FakeHandBackend:
    def __init__(self, joint_pos: np.ndarray | None = None):
        self.joint_pos = (
            np.zeros(6, dtype=np.float32)
            if joint_pos is None
            else np.asarray(joint_pos, dtype=np.float32)
        )

    def get_state(self, **_kwargs) -> dict:
        return {"joint_pos": self.joint_pos.copy()}


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

    def test_dispatched_joint_action_uses_backend_target_pose_not_source_target(self) -> None:
        source_pose = np.array([0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        target_pose = np.array([0.51, 0.52, 0.53, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=3.5)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
                metadata={"source_target_pose_base": xyzquat_to_matrix(source_pose)},
            )
        )
        step = SessionStepResult(
            timestamp=3.5,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=3.4),
            command=command,
            command_dispatched=True,
        )
        builder = CollectorSampleBuilder()
        components = CollectorComponents(
            session=None,
            backends={
                "left_arm": _FakeArmBackend(
                    np.arange(6, dtype=np.float32),
                    xyzquat_to_matrix(source_pose),
                    target_tcp_pose=xyzquat_to_matrix(target_pose),
                    last_command_timestamp=3.52,
                ),
                "left_hand": _FakeHandBackend(),
            },
        )

        sample = builder(step, components)
        row, _camera_expectations = collector_sample_to_row(sample)

        np.testing.assert_allclose(sample.action[0:7], target_pose, atol=1e-6)
        np.testing.assert_allclose(row["action.left_arm.pose"], target_pose, atol=1e-6)
        canonical = sample.metadata["canonical_action_record"]["action"]
        np.testing.assert_allclose(
            canonical["arm_pose_by_side"]["left"],
            target_pose,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row["teleop.action_source.left_wrist_pose_base"],
            source_pose,
            atol=1e-6,
        )
        self.assertNotIn("action.left_arm.pose_source", row)
        self.assertNotIn("action.left_arm.sample_to_last_command_dt_sec", row)
        self.assertNotIn("arm_action_debug_by_side", sample.metadata)

    def test_dispatched_cartesian_action_uses_limited_solver_target_pose(self) -> None:
        source_pose = np.array([0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        target_pose = np.array([0.31, 0.32, 0.33, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        backend_pose = np.array([0.61, 0.62, 0.63, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        solver_qpos = np.arange(6, dtype=np.float32) / 10.0
        command = TeleopCommand(timestamp=3.6)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(source_pose),
                frame_name="left_arm_base",
                metadata={
                    "source_target_pose_base": xyzquat_to_matrix(source_pose),
                    "target_pose_base": xyzquat_to_matrix(target_pose),
                    "target_pose_tcp_base": xyzquat_to_matrix(target_pose),
                    "solver_output_qpos": solver_qpos,
                    "execution_step_metadata": {
                        "enabled": True,
                        "step_limited": True,
                    },
                    "action_pose_source": "limited_solver_output_qpos",
                },
            )
        )
        step = SessionStepResult(
            timestamp=3.6,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=3.5),
            command=command,
            command_dispatched=True,
        )
        builder = CollectorSampleBuilder()
        components = CollectorComponents(
            session=None,
            backends={
                "left_arm": _FakeArmBackend(
                    np.arange(6, dtype=np.float32),
                    xyzquat_to_matrix(source_pose),
                    target_tcp_pose=xyzquat_to_matrix(backend_pose),
                    last_command_timestamp=3.61,
                ),
                "left_hand": _FakeHandBackend(),
            },
        )

        sample = builder(step, components)
        row, _camera_expectations = collector_sample_to_row(sample)

        np.testing.assert_allclose(sample.action[0:7], target_pose, atol=1e-6)
        np.testing.assert_allclose(row["action.left_arm.pose"], target_pose, atol=1e-6)
        np.testing.assert_allclose(
            row["action.replay.left_arm_joint_target"],
            solver_qpos,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            row["teleop.action_source.left_wrist_pose_base"],
            source_pose,
            atol=1e-6,
        )
        canonical = sample.metadata["canonical_action_record"]["action"]
        np.testing.assert_allclose(
            canonical["arm_pose_by_side"]["left"],
            target_pose,
            atol=1e-6,
        )
        left_metadata = sample.metadata["command_metadata_by_effector"]["left_arm"]
        self.assertEqual(left_metadata["action_pose_source"], "limited_solver_output_qpos")
        self.assertIn("solver_output_qpos", left_metadata)
        self.assertIn("execution_step_metadata", left_metadata)

    def test_collection_action_and_state_follow_deploy_vector_layout(self) -> None:
        left_pose = np.array([0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        right_pose = np.array([0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        left_qpos = np.arange(6, dtype=np.float32) / 10.0
        right_qpos = np.arange(6, dtype=np.float32) / 10.0 + 1.0
        left_hand = np.arange(6, dtype=np.float32) / 20.0
        right_hand = np.arange(6, dtype=np.float32) / 20.0 + 2.0
        command = TeleopCommand(timestamp=3.75)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(left_pose),
                frame_name="left_arm_base",
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(right_pose),
                frame_name="right_arm_base",
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name="left_hand",
                side="left",
                command_type="joint_vector",
                value=left_hand,
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name="right_hand",
                side="right",
                command_type="joint_vector",
                value=right_hand,
            )
        )
        step = SessionStepResult(
            timestamp=3.75,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=3.7),
            command=command,
            command_dispatched=True,
        )
        sample = CollectorSampleBuilder()(
            step,
            CollectorComponents(
                session=None,
                backends={
                    "left_arm": _FakeArmBackend(left_qpos, xyzquat_to_matrix(left_pose)),
                    "right_arm": _FakeArmBackend(right_qpos, xyzquat_to_matrix(right_pose)),
                    "left_hand": _FakeHandBackend(left_hand),
                    "right_hand": _FakeHandBackend(right_hand),
                },
            ),
        )

        self.assertEqual(sample.action.shape, (DEPLOY_ACTION_DIM,))
        np.testing.assert_allclose(sample.action[ACTION_ARM_POSE_SLICES["left"]], left_pose)
        np.testing.assert_allclose(sample.action[ACTION_ARM_POSE_SLICES["right"]], right_pose)
        np.testing.assert_allclose(sample.action[ACTION_HAND_SLICES["left"]], left_hand)
        np.testing.assert_allclose(sample.action[ACTION_HAND_SLICES["right"]], right_hand)

        self.assertEqual(sample.observation_state.shape, (DEPLOY_OBSERVATION_STATE_DIM,))
        np.testing.assert_allclose(sample.observation_state[STATE_ARM_QPOS_SLICES["left"]], left_qpos)
        np.testing.assert_allclose(sample.observation_state[STATE_ARM_QPOS_SLICES["right"]], right_qpos)
        np.testing.assert_allclose(sample.observation_state[STATE_ARM_POSE_SLICES["left"]], left_pose)
        np.testing.assert_allclose(sample.observation_state[STATE_ARM_POSE_SLICES["right"]], right_pose)
        np.testing.assert_allclose(sample.observation_state[STATE_HAND_SLICES["left"]], left_hand)
        np.testing.assert_allclose(sample.observation_state[STATE_HAND_SLICES["right"]], right_hand)

    def test_action_debug_metadata_is_opt_in(self) -> None:
        source_pose = np.array([0.21, 0.22, 0.23, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        target_pose = np.array([0.51, 0.52, 0.53, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        command = TeleopCommand(timestamp=3.5)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.arange(6, dtype=np.float32),
                frame_name=None,
                metadata={"source_target_pose_base": xyzquat_to_matrix(source_pose)},
            )
        )
        step = SessionStepResult(
            timestamp=3.5,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=3.4),
            command=command,
            command_dispatched=True,
        )
        builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(include_action_debug_metadata=True)
        )
        components = CollectorComponents(
            session=None,
            backends={
                "left_arm": _FakeArmBackend(
                    np.arange(6, dtype=np.float32),
                    xyzquat_to_matrix(source_pose),
                    target_tcp_pose=xyzquat_to_matrix(target_pose),
                    last_command_timestamp=3.52,
                ),
                "left_hand": _FakeHandBackend(),
            },
        )

        sample = builder(step, components)
        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertNotIn("action.left_arm.pose_source", row)
        debug = sample.metadata["arm_action_debug_by_side"]["left"]
        self.assertEqual(debug["pose_source"], "backend_last_target_tcp_pose")
        self.assertAlmostEqual(debug["sample_to_last_command_dt_sec"], 0.02)
        self.assertAlmostEqual(debug["last_command_age_sec_at_sample"], -0.02)
        self.assertGreater(debug["target_current_translation_m"], 0.0)
        self.assertIn("arm_actions", sample.metadata["sample_builder_debug"])

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
        pose_quality = metadata["pose_quality"]
        self.assertTrue(pose_quality["available"])
        self.assertEqual(pose_quality["status"], "good")
        self.assertEqual(
            pose_quality["metrics"]["collision_avoidance_status"],
            "tracking",
        )

    def test_default_sample_metadata_records_bad_pose_quality_summary(self) -> None:
        command = TeleopCommand(
            timestamp=4.5,
            metadata={
                "collision_avoidance_status": "hold_last_safe",
                "collision_avoidance_fallback_reason": "collision_margin_low",
                "collision_distance_m": 0.01,
                "collision_min_distance_m": 0.03,
                "ik_solve_pose_error": {
                    "left": {"translation_m": 0.03, "rotation_deg": 6.0},
                    "right": {"translation_m": 0.001, "rotation_deg": 0.2},
                },
                "joint_limit_margin": {
                    "left": {"nearest_margin_deg": 2.0, "near_limit": True},
                    "right": {"nearest_margin_deg": 12.0, "near_limit": False},
                },
            },
        )
        step = SessionStepResult(
            timestamp=4.5,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=4.4),
            command=command,
        )

        metadata = CollectorSampleBuilder._default_metadata(
            step,
            None,
            {},
            {},
            {},
        )

        pose_quality = metadata["pose_quality"]
        self.assertEqual(pose_quality["status"], "bad")
        self.assertTrue(pose_quality["bad"])
        self.assertIn("collision_distance_below_min", pose_quality["bad_reasons"])
        self.assertLess(pose_quality["score"], 0.5)

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

    def test_position_mode_can_use_causal_measured_state_and_keep_commanded_sidecar(self) -> None:
        commanded_pose = np.array(
            [0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        measured_pose = np.array(
            [0.71, 0.72, 0.73, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        commanded_qpos = np.arange(6, dtype=np.float32)
        measured_qpos = commanded_qpos + 10.0

        builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(
                active_sides=("left",),
                prefer_causal_measured_arm_state=True,
            ),
            arm_state_metadata_provider=lambda _components, sample_timestamp: {
                "left": {
                    "actual_qpos": measured_qpos,
                    "actual_tcp_pose": xyzquat_to_matrix(measured_pose),
                    "actual_state_timestamp": sample_timestamp - 0.01,
                }
            },
        )
        sample = builder(
            SessionStepResult(
                timestamp=12.34,
                state=TeleopRunState.RUNNING,
                control_hz=15.0,
                observation=TeleopObservation(timestamp=12.33),
            ),
            CollectorComponents(
                session=None,
                backends={
                    "left_arm": _FakeArmBackend(
                        commanded_qpos,
                        xyzquat_to_matrix(commanded_pose),
                    )
                },
            ),
        )

        self.assertIsNotNone(sample)
        assert sample is not None
        row, _camera_expectations = collector_sample_to_row(sample)
        np.testing.assert_allclose(
            sample.observation_state[STATE_ARM_QPOS_SLICES["left"]],
            measured_qpos,
        )
        np.testing.assert_allclose(
            sample.observation_state[STATE_ARM_POSE_SLICES["left"]],
            measured_pose,
        )
        np.testing.assert_allclose(
            row["observation.arm.left.commanded_qpos"],
            commanded_qpos,
        )
        np.testing.assert_allclose(
            row["observation.arm.left.commanded_tcp_pose"],
            commanded_pose,
        )
        self.assertEqual(sample.arm_state_source, "causal_measured_hardware")

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

    def test_actual_state_poller_selects_latest_snapshot_at_or_before_sample(self) -> None:
        pose = xyzquat_to_matrix(
            np.array([0.41, 0.42, 0.43, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        backend = _FakeRefreshArmBackend(np.ones(6), pose, timestamp=99.0)
        poller = ArmHardwareStatePoller(
            backends_by_side={"left": backend},
            config=ArmHardwareStatePollerConfig(
                stale_after_sec=0.2,
                history_size=4,
            ),
        )
        poller._poll_once("left", backend)
        backend.qpos = np.full(6, 2.0, dtype=np.float32)
        backend.timestamp = 101.0
        poller._poll_once("left", backend)

        selected = poller.latest_at_or_before(100.0)["left"]

        np.testing.assert_allclose(selected["actual_qpos"], np.ones(6), atol=1e-6)
        self.assertEqual(selected["actual_state_timestamp"], 99.0)
        self.assertEqual(selected["actual_sample_age_sec"], 1.0)
        self.assertTrue(selected["actual_state_stale"])
        self.assertNotIn("left", poller.latest_at_or_before(98.0))

    def test_actual_state_poller_readiness_requires_causal_fresh_tcp_for_every_side(self) -> None:
        timestamp = time.time()
        pose = xyzquat_to_matrix(
            np.array([0.41, 0.42, 0.43, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        backend = _FakeRefreshArmBackend(np.ones(6), pose, timestamp=timestamp)
        poller = ArmHardwareStatePoller(
            backends_by_side={"left": backend},
            config=ArmHardwareStatePollerConfig(stale_after_sec=0.2),
        )

        self.assertEqual(
            poller.readiness_by_side(
                required_sides=("left", "right"),
                sample_timestamp=timestamp,
            ),
            {"left": "no_causal_feedback", "right": "no_causal_feedback"},
        )
        poller._poll_once("left", backend)

        self.assertEqual(
            poller.readiness_by_side(
                required_sides=("left", "right"),
                sample_timestamp=timestamp + 0.02,
            ),
            {"left": None, "right": "no_causal_feedback"},
        )

    def test_position_reference_is_single_source_for_fk_pose_and_replay_qref(self) -> None:
        wrong_pose = np.array(
            [0.11, 0.12, 0.13, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        pose_ref = np.array(
            [0.61, 0.62, 0.63, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        q_ref = np.arange(6, dtype=np.float32) + 30.0
        q_ik_raw = q_ref + 0.25
        hand_target = np.arange(6, dtype=np.float32) / 10.0
        reference = {
            "epoch": "episode-7",
            "sequence": 42,
            "source": "teleop_position_worker",
            "source_wall_time": 9.90,
            "generated_wall_time": 9.94,
            "published_wall_time": 9.95,
            "generated_monotonic_time": 109.94,
            "published_monotonic_time": 109.95,
            "pipeline_version": "position_control.v1",
            "references": {
                "left": {
                    "q_ik_raw": q_ik_raw,
                    "q_ref": q_ref,
                    "pose_ref": xyzquat_to_matrix(pose_ref),
                    "valid": True,
                    "hold": False,
                    "modification_reasons": ("joint_rate_limit",),
                }
            },
        }
        provider_calls: list[float] = []

        def reference_provider(sample_timestamp: float) -> dict:
            provider_calls.append(float(sample_timestamp))
            return reference

        command = TeleopCommand(timestamp=10.0)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=xyzquat_to_matrix(wrong_pose),
                frame_name="left_arm_base",
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name="left_hand",
                side="left",
                command_type="joint_vector",
                value=hand_target,
            )
        )
        step = SessionStepResult(
            timestamp=10.0,
            state=TeleopRunState.RUNNING,
            control_hz=15.0,
            observation=TeleopObservation(timestamp=9.99),
            command=command,
            command_dispatched=True,
        )
        builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(
                active_sides=("left",),
                position_reference_max_age_sec=0.2,
            ),
            position_reference_provider=reference_provider,
        )
        sample = builder(
            step,
            CollectorComponents(
                session=None,
                backends={
                    "left_arm": _FakeArmBackend(
                        np.zeros(6),
                        xyzquat_to_matrix(wrong_pose),
                        target_tcp_pose=xyzquat_to_matrix(wrong_pose),
                    ),
                    "left_hand": _FakeHandBackend(hand_target),
                },
            ),
        )

        self.assertIsNotNone(sample)
        assert sample is not None
        row, _camera_expectations = collector_sample_to_row(sample)
        self.assertEqual(provider_calls, [10.0])
        np.testing.assert_allclose(sample.action[ACTION_ARM_POSE_SLICES["left"]], pose_ref)
        np.testing.assert_allclose(sample.action[ACTION_HAND_SLICES["left"]], hand_target)
        np.testing.assert_allclose(row["action.left_arm.pose"], pose_ref)
        np.testing.assert_allclose(row["action.replay.left_arm_joint_target"], q_ref)
        np.testing.assert_allclose(row["action.reference.left_arm.q_ref"], q_ref)
        np.testing.assert_allclose(row["action.reference.left_arm.q_ik_raw"], q_ik_raw)
        np.testing.assert_allclose(row["action.reference.left_arm.pose_ref"], pose_ref)
        self.assertEqual(row["action.reference.epoch"], "episode-7")
        self.assertEqual(row["action.reference.sequence"], 42)
        self.assertAlmostEqual(row["action.reference.age_sec"], 0.05)
        self.assertEqual(row["action.reference.valid"], 1)
        self.assertEqual(row["action.reference.hold"], 0)
        self.assertEqual(
            row["action.reference.modification_reason"],
            "left:joint_rate_limit",
        )
        self.assertEqual(
            row["action.reference.pipeline_version"],
            "position_control.v1",
        )
        canonical_pose = sample.metadata["canonical_action_record"]["action"][
            "arm_pose_by_side"
        ]["left"]
        np.testing.assert_allclose(canonical_pose, pose_ref)

    def test_missing_or_future_position_reference_drops_sample_without_advancing_indices(self) -> None:
        step = SessionStepResult(
            timestamp=10.0,
            state=TeleopRunState.RUNNING,
            control_hz=15.0,
            observation=TeleopObservation(timestamp=9.99),
        )
        components = CollectorComponents(session=None, backends={})
        missing_builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(active_sides=("left",)),
            position_reference_provider=lambda _timestamp: None,
        )

        self.assertIsNone(missing_builder(step, components))
        self.assertEqual(
            missing_builder.last_position_reference_rejection,
            "no_reference_at_or_before_sample",
        )
        self.assertEqual(missing_builder.sample_index, 0)
        self.assertEqual(missing_builder.frame_index, 0)

        future_reference = {
            "epoch": "episode-8",
            "sequence": 1,
            "published_wall_time": 10.01,
            "references": {
                "left": {
                    "q_ref": np.zeros(6),
                    "pose_ref": np.eye(4),
                }
            },
        }
        future_builder = CollectorSampleBuilder(
            CollectorSampleBuilderConfig(active_sides=("left",)),
            position_reference_provider=lambda _timestamp: future_reference,
        )

        self.assertIsNone(future_builder(step, components))
        self.assertEqual(future_builder.last_position_reference_rejection, "future_reference")


if __name__ == "__main__":
    unittest.main()
