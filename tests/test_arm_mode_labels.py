from __future__ import annotations

import json
import unittest

import numpy as np

from teleop_runtime.collection.arm_mode_labels import build_arm_mode_metadata
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.schema_v21 import collector_sample_to_row
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _arm_command(side: str) -> EffectorCommand:
    return EffectorCommand(
        effector_name=f"{side}_arm",
        side=side,
        command_type="cartesian_pose",
        value=np.eye(4),
        frame_name="arm_base",
    )


def _command() -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0)
    command.add_command(_arm_command("left"))
    command.add_command(_arm_command("right"))
    return command


class ArmModeLabelTests(unittest.TestCase):
    def test_fix_arm_marks_only_fixed_side(self) -> None:
        command = _command()
        command.metadata["fix_arm_active_side"] = "left"
        command.metadata["fix_arm_mode"] = "fixed"

        payload = build_arm_mode_metadata(
            command=command,
            session_state="RUNNING",
            active_sides=("left", "right"),
        )

        self.assertEqual(payload["sides"]["left"]["mode"], "fixed_hold")
        self.assertEqual(payload["sides"]["left"]["source"], "fix_arm")
        self.assertEqual(payload["sides"]["right"]["mode"], "teleop")

    def test_init_arm_marks_only_active_side(self) -> None:
        command = _command()
        command.commands["right_arm"].metadata["init_arm_active"] = True
        command.commands["right_arm"].metadata["init_arm_side"] = "right"
        command.commands["right_arm"].metadata["init_arm_mode"] = "moving"

        payload = build_arm_mode_metadata(
            command=command,
            session_state="RUNNING",
            active_sides=("left", "right"),
        )

        self.assertEqual(payload["sides"]["left"]["mode"], "teleop")
        self.assertEqual(payload["sides"]["right"]["mode"], "init_pose")
        self.assertEqual(payload["sides"]["right"]["source"], "init_arm")

    def test_hold_or_non_running_state_overrides_special_labels(self) -> None:
        command = _command()
        command.metadata["fix_arm_active_side"] = "left"
        command.metadata["init_arm_active_side"] = "right"

        payload = build_arm_mode_metadata(
            command=command,
            session_state="DEBUG_ONLY",
            active_sides=("left", "right"),
            recording_paused=True,
        )

        self.assertEqual(payload["sides"]["left"]["mode"], "stopped")
        self.assertEqual(payload["sides"]["right"]["mode"], "stopped")
        self.assertEqual(payload["sides"]["left"]["source"], "recording_pause")
        self.assertEqual(payload["active_special_tasks"], [])

    def test_schema_adds_optional_arm_mode_columns_without_changing_vectors(self) -> None:
        payload = build_arm_mode_metadata(
            command=_command(),
            session_state="RUNNING",
            active_sides=("left", "right"),
        )
        sample = CollectorSample(
            index=0,
            episode_index=0,
            frame_index=0,
            timestamp=1.0,
            session_state="RUNNING",
            control_hz=30.0,
            action=np.zeros(26, dtype=np.float32),
            observation_state=np.zeros(38, dtype=np.float32),
            observation_last_action=np.zeros(26, dtype=np.float32),
            arm_state_source="unit",
            active_sides=("left", "right"),
            metadata={"arm_mode": payload},
        )

        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertEqual(len(row["action"]), 26)
        self.assertEqual(len(row["observation.state"]), 38)
        self.assertEqual(row["teleop.arm_mode.left.label"], "teleop")
        self.assertEqual(row["teleop.arm_mode.right.id"], 1)
        self.assertFalse(row["teleop.arm_mode.left.is_special"])
        decoded = json.loads(row["teleop.arm_mode_json"])
        self.assertEqual(decoded["schema"], "dexteleop_arm_mode_v1")

    def test_schema_keeps_missing_legacy_arm_mode_compatible(self) -> None:
        sample = CollectorSample(
            index=0,
            episode_index=0,
            frame_index=0,
            timestamp=1.0,
            session_state="RUNNING",
            control_hz=30.0,
            action=np.zeros(26, dtype=np.float32),
            observation_state=np.zeros(38, dtype=np.float32),
            observation_last_action=np.zeros(26, dtype=np.float32),
            arm_state_source="unit",
            active_sides=("left", "right"),
            metadata={},
        )

        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertEqual(row["teleop.arm_mode.available"], 0)
        self.assertEqual(row["teleop.arm_mode.left.label"], "unknown")
        self.assertEqual(row["teleop.arm_mode.right.label"], "unknown")


if __name__ == "__main__":
    unittest.main()
