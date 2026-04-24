from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from inspire_data_collect.arm.fr3_backend import FR3ArmBackend, FR3ArmBackendConfig
from inspire_data_collect.arm.retargeter import FR3ArmRetargeter, FR3ArmRetargeterConfig
from inspire_data_collect.common import EffectorCommand, ObservationChannel
from inspire_data_collect.utils.writer import (
    ACTION_DIM,
    ARM_DOF,
    HAND_DOF,
    HAND_VECTOR_ORDER,
    HandOnlyLeRobotWriter,
    HandSampleBuilder,
    WriterConfig,
    hand_sample_to_row,
)


class InspireArmIntegrationTests(unittest.TestCase):
    def test_exo_retargeter_outputs_fr3_command_and_metadata(self) -> None:
        source_q = np.deg2rad(np.full(ARM_DOF, 180.0, dtype=np.float32))
        channel = ObservationChannel(
            channel_name="arm_exo_joint_qpos",
            side="right",
            value=source_q,
            timestamp=123.0,
            valid=True,
            source_name="unit_test",
        )
        retargeter = FR3ArmRetargeter(
            FR3ArmRetargeterConfig(side="right", shrink_deg=10.0)
        )

        command = retargeter.retarget(
            {"arm_exo_joint_qpos": channel},
            timestamp=456.0,
        )

        self.assertIsNotNone(command)
        assert command is not None
        self.assertEqual(command.effector_name, "arm")
        self.assertEqual(command.side, "right")
        self.assertEqual(command.value.shape, (ARM_DOF,))
        np.testing.assert_allclose(
            command.value,
            np.asarray([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0], dtype=np.float32),
            atol=1e-5,
        )
        self.assertEqual(command.metadata["mapping"], "dynamixel_to_robot_arm")
        self.assertEqual(command.metadata["source_timestamp"], 123.0)
        self.assertFalse(command.metadata["clipped"])

    def test_fr3_backend_without_hardware_tracks_last_sent_command(self) -> None:
        backend = FR3ArmBackend(
            FR3ArmBackendConfig(host="127.0.0.1", connect_hardware=False)
        )
        command_value = np.linspace(-0.3, 0.3, ARM_DOF, dtype=np.float32)
        backend.send_command(
            EffectorCommand(
                effector_name="arm",
                side="right",
                value=command_value,
                timestamp=time.time(),
                metadata={},
            )
        )

        state = backend.get_state()

        np.testing.assert_allclose(state["last_sent_command"], command_value)
        self.assertIsNone(state["qpos"])
        self.assertFalse(state["arm_enabled"])

    def test_fr3_backend_startup_alignment_ramps_from_current_qpos(self) -> None:
        backend = FR3ArmBackend(
            FR3ArmBackendConfig(
                host="127.0.0.1",
                connect_hardware=True,
                startup_align_enabled=True,
                startup_align_duration_s=10.0,
                startup_align_threshold_rad=0.0,
            )
        )
        start_q = np.zeros(ARM_DOF, dtype=np.float32)
        target_q = np.ones(ARM_DOF, dtype=np.float32)
        with backend._state_lock:
            backend._latest_qpos = start_q.copy()

        backend.send_command(
            EffectorCommand(
                effector_name="arm",
                side="right",
                value=target_q,
                timestamp=time.time(),
                metadata={},
            )
        )
        state = backend.get_state()
        np.testing.assert_allclose(state["last_sent_command"], start_q, atol=1e-4)
        self.assertTrue(state["startup_alignment_active"])
        self.assertFalse(state["startup_alignment_complete"])

        with backend._state_lock:
            backend._startup_align_started_at = time.time() - 20.0
        backend.send_command(
            EffectorCommand(
                effector_name="arm",
                side="right",
                value=target_q,
                timestamp=time.time(),
                metadata={},
            )
        )
        state = backend.get_state()
        np.testing.assert_allclose(state["last_sent_command"], target_q, atol=1e-4)
        self.assertFalse(state["startup_alignment_active"])
        self.assertTrue(state["startup_alignment_complete"])
        self.assertEqual(state["startup_alignment_progress"], 1.0)

    def test_writer_rows_fill_arm_vector_when_command_and_state_exist(self) -> None:
        arm_target = np.linspace(0.1, 0.7, ARM_DOF, dtype=np.float32)
        arm_qpos = np.linspace(-0.1, -0.7, ARM_DOF, dtype=np.float32)
        builder = HandSampleBuilder(active_sides=("right",), episode_index=0)
        sample = builder.build(
            timestamp=1000.0,
            session_state="running",
            control_hz=30.0,
            channels={},
            commands={
                "arm": EffectorCommand(
                    effector_name="arm",
                    side="right",
                    value=arm_target,
                    timestamp=999.9,
                    metadata={
                        "source_qpos": np.zeros(ARM_DOF, dtype=np.float32),
                        "preclip_joint_target": arm_target,
                        "clip_mask": [False] * ARM_DOF,
                        "clipped": False,
                        "mapping": "dynamixel_to_robot_arm",
                        "shrink_deg": 10.0,
                        "arm_side": "right",
                    },
                )
            },
            arm_state={
                "arm_enabled": True,
                "arm_side": "right",
                "state_source": "fr3_joint_control_client.cached",
                "last_sent_command": arm_target,
                "qpos": arm_qpos,
                "startup_alignment_enabled": True,
                "startup_alignment_active": False,
                "startup_alignment_complete": True,
                "startup_alignment_progress": 1.0,
                "startup_alignment_duration_s": 5.0,
                "startup_alignment_start_max_delta_rad": 0.5,
                "startup_alignment_start_qpos": np.zeros(ARM_DOF, dtype=np.float32),
                "startup_alignment_initial_target": arm_target,
                "startup_alignment_latest_target": arm_target,
            },
            hand_states={},
        )

        row, _ = hand_sample_to_row(sample)

        self.assertEqual(len(row["action"]), ACTION_DIM)
        np.testing.assert_allclose(row["action"][:ARM_DOF], arm_target)
        np.testing.assert_allclose(row["observation.state"][:ARM_DOF], arm_qpos)
        np.testing.assert_allclose(row["action.arm.joint_target"], arm_target)
        np.testing.assert_allclose(row["observation.arm.qpos"], arm_qpos)
        self.assertTrue(row["arm_enabled"])
        self.assertEqual(row["arm_state_source"], "fr3_joint_control_client.cached")
        self.assertTrue(row["observation.arm.startup_alignment_complete"])
        np.testing.assert_allclose(
            row["observation.arm.startup_alignment_latest_target"],
            arm_target,
        )
        self.assertEqual(row["teleop.action_source.arm_mapping"], "dynamixel_to_robot_arm")

    def test_writer_info_declares_arm_and_hand_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = HandOnlyLeRobotWriter(
                WriterConfig(
                    root_dir=Path(tmpdir),
                    task_name="unit_task",
                    write_parquet=False,
                    write_video=False,
                )
            )
            info = writer.info_path.read_text(encoding="utf-8")

        self.assertIn('"single_arm_dof": 7', info)
        self.assertEqual(ACTION_DIM, ARM_DOF + HAND_DOF * len(HAND_VECTOR_ORDER))


if __name__ == "__main__":
    unittest.main()
