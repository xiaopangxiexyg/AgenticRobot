from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from teleop_runtime.collection.arm_mode_labels import (
    InitArmAuditTracker,
    build_arm_mode_metadata,
    flatten_arm_mode_columns,
)
from teleop_runtime.collection.sample_assembler import CollectorSample
from teleop_runtime.collection.schema_v21 import (
    collector_sample_to_row,
    write_rows_to_parquet,
)
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


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


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

    def test_qpos_home_recapture_ends_init_and_marks_target_arm_stopped(self) -> None:
        command = _command()
        command.metadata.update(
            {
                "init_arm_active_side": "left",
                "init_arm_mode": "qpos_recapturing",
                "init_arm_qpos_active": True,
                "init_arm_label_active": False,
                "arm_mode_stopped_sides": ["left"],
                "arm_mode_stop_reason": "init_arm_home_recapture",
            }
        )
        command.commands["left_arm"].metadata.update(
            {
                "arm_mode_stopped": True,
                "arm_mode_stop_reason": "init_arm_home_recapture",
            }
        )

        payload = build_arm_mode_metadata(
            command=command,
            session_state="RUNNING",
            active_sides=("left", "right"),
        )

        self.assertEqual(payload["sides"]["left"]["mode"], "stopped")
        self.assertEqual(payload["sides"]["right"]["mode"], "teleop")
        self.assertEqual(payload["active_special_tasks"], [])
        self.assertEqual(
            payload["sides"]["left"]["detail"]["stop_reason"],
            "init_arm_home_recapture",
        )

    def test_init_audit_uses_actual_feedback_but_model_mode_stays_single_label(self) -> None:
        command = _command()
        target = _pose()
        command.commands["right_arm"].metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": "right",
                "init_arm_mode": "moving",
                "init_arm_target_pose_base": target,
            }
        )
        tracker = InitArmAuditTracker()
        arm_states = {"right": {"actual_tcp_pose": target}}

        first = tracker.observe(command=command, arm_states=arm_states, timestamp=0.0)
        self.assertEqual(first["right"]["init_phase"], "init_move")
        self.assertTrue(first["right"]["init_phase_valid"])

        audit = first
        for timestamp in np.arange(0.1, 3.3, 0.1):
            audit = tracker.observe(
                command=command,
                arm_states=arm_states,
                timestamp=float(timestamp),
            )

        payload = build_arm_mode_metadata(
            command=command,
            session_state="RUNNING",
            active_sides=("left", "right"),
            init_audit_by_side=audit,
        )
        columns = flatten_arm_mode_columns(payload)
        self.assertEqual(payload["sides"]["right"]["mode"], "init_pose")
        self.assertEqual(payload["sides"]["right"]["audit"]["init_phase"], "init_hold")
        self.assertEqual(columns["teleop.arm_mode.right.label"], "init_pose")
        self.assertEqual(columns["teleop.arm_mode.right.init_phase"], "init_hold")

    def test_init_audit_never_guesses_hold_without_actual_feedback(self) -> None:
        command = _command()
        command.commands["right_arm"].metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": "right",
                "init_arm_target_pose_base": _pose(),
            }
        )
        audit = InitArmAuditTracker().observe(
            command=command,
            arm_states={"right": {}},
            timestamp=1.0,
        )
        self.assertEqual(audit["right"]["init_phase"], "unknown")
        self.assertFalse(audit["right"]["init_phase_valid"])

    def test_init_audit_requires_low_actual_speed_before_hold(self) -> None:
        command = _command()
        target = _pose()
        command.commands["right_arm"].metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": "right",
                "init_arm_target_pose_base": target,
            }
        )
        tracker = InitArmAuditTracker()
        tracker.observe(
            command=command,
            arm_states={"right": {"actual_tcp_pose": _pose(0.0014)}},
            timestamp=0.0,
        )
        audit = tracker.observe(
            command=command,
            arm_states={"right": {"actual_tcp_pose": target}},
            timestamp=0.01,
        )

        self.assertEqual(audit["right"]["init_phase"], "init_move")
        self.assertEqual(audit["right"]["reason"], "actual_speed_above_stillness_threshold")

    def test_init_audit_target_change_resets_dwell(self) -> None:
        command = _command()
        initial_target = _pose()
        changed_target = _pose(0.001)
        command.commands["right_arm"].metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": "right",
                "init_arm_target_pose_base": initial_target,
            }
        )
        tracker = InitArmAuditTracker()
        arm_states = {"right": {"actual_tcp_pose": initial_target}}
        tracker.observe(command=command, arm_states=arm_states, timestamp=0.0)
        tracker.observe(command=command, arm_states=arm_states, timestamp=0.1)
        before_change = tracker.observe(command=command, arm_states=arm_states, timestamp=0.4)

        command.commands["right_arm"].metadata["init_arm_target_pose_base"] = changed_target
        changed = tracker.observe(
            command=command,
            arm_states={"right": {"actual_tcp_pose": initial_target}},
            timestamp=0.5,
        )

        self.assertEqual(before_change["right"]["init_phase"], "init_move")
        self.assertGreater(before_change["right"]["dwell_sec"], 0.0)
        self.assertEqual(changed["right"]["init_phase"], "init_move")
        self.assertEqual(changed["right"]["reason"], "init_target_changed")
        self.assertEqual(changed["right"]["dwell_sec"], 0.0)

    def test_init_audit_time_gap_or_reversal_resets_dwell(self) -> None:
        command = _command()
        target = _pose()
        command.commands["right_arm"].metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": "right",
                "init_arm_target_pose_base": target,
            }
        )
        tracker = InitArmAuditTracker()
        arm_states = {"right": {"actual_tcp_pose": target}}
        tracker.observe(command=command, arm_states=arm_states, timestamp=0.0)
        tracker.observe(command=command, arm_states=arm_states, timestamp=0.1)

        after_gap = tracker.observe(command=command, arm_states=arm_states, timestamp=1.0)
        after_reversal = tracker.observe(command=command, arm_states=arm_states, timestamp=0.9)

        for audit in (after_gap, after_reversal):
            self.assertEqual(audit["right"]["init_phase"], "init_move")
            self.assertEqual(
                audit["right"]["reason"],
                "waiting_for_continuous_actual_feedback",
            )
            self.assertEqual(audit["right"]["dwell_sec"], 0.0)

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
            metadata={
                "arm_mode": payload,
                "collection_continuity_segment_index": 3,
            },
        )

        row, _camera_expectations = collector_sample_to_row(sample)

        self.assertEqual(len(row["action"]), 26)
        self.assertEqual(len(row["observation.state"]), 38)
        self.assertEqual(row["teleop.arm_mode.left.label"], "teleop")
        self.assertEqual(row["teleop.arm_mode.right.id"], 1)
        self.assertFalse(row["teleop.arm_mode.left.is_special"])
        decoded = json.loads(row["teleop.arm_mode_json"])
        self.assertEqual(decoded["schema"], "dexteleop_arm_mode_v2")
        self.assertEqual(row["teleop.arm_mode.left.init_phase"], "not_active")
        self.assertEqual(row["teleop.arm_mode.left.init_phase_valid"], 0)
        self.assertEqual(row["collection.continuity_segment_index"], 3)

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

    @pytest.mark.optional
    def test_parquet_uses_stable_vector_and_empty_tag_types(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow is not installed")

        rows = []
        for index in range(2):
            row = {
                "action": [float(index)] * 26,
                "observation.state": [float(index)] * 38,
                "observation.last_action": [float(index)] * 26,
                "action.replay.left_arm_joint_target": [float(index)] * 6,
                "observation.arm.left.actual_qpos": [float(index)] * 6,
                "fault_codes": [],
                "teleop.arm_mode.schema": "dexteleop_arm_mode_v2",
                "teleop.arm_mode.available": 1,
                "teleop.arm_mode.left.id": 1,
                "teleop.arm_mode.left.label": "teleop",
                "teleop.arm_mode.left.is_special": 0,
                "teleop.arm_mode.left.source": "normal",
                "teleop.arm_mode.left.tags": [],
                "teleop.arm_mode.left.scope": "",
                "teleop.arm_mode.left.init_phase": "not_active",
                "teleop.arm_mode.left.init_phase_valid": 0,
                "teleop.arm_mode.left.init_target_position_error_m": None,
                "teleop.arm_mode.left.init_target_rotation_error_deg": None,
                "teleop.arm_mode.left.init_translation_speed_m_s": None,
                "teleop.arm_mode.left.init_rotation_speed_deg_s": None,
                "teleop.arm_mode.left.init_dwell_sec": 0.0,
                "teleop.arm_mode.right.id": 1,
                "teleop.arm_mode.right.label": "teleop",
                "teleop.arm_mode.right.is_special": 0,
                "teleop.arm_mode.right.source": "normal",
                "teleop.arm_mode.right.tags": [],
                "teleop.arm_mode.right.scope": "",
                "teleop.arm_mode.right.init_phase": "not_active",
                "teleop.arm_mode.right.init_phase_valid": 0,
                "teleop.arm_mode.right.init_target_position_error_m": None,
                "teleop.arm_mode.right.init_target_rotation_error_deg": None,
                "teleop.arm_mode.right.init_translation_speed_m_s": None,
                "teleop.arm_mode.right.init_rotation_speed_deg_s": None,
                "teleop.arm_mode.right.init_dwell_sec": 0.0,
                "teleop.arm_mode_json": "{}",
                "observation.hand.left.tactile_process_cpu_affinity": [],
            }
            if index == 1:
                row["teleop.channels.late.valid"] = True
            rows.append(row)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episode.parquet"
            write_rows_to_parquet(rows, path)
            table = pq.read_table(path)

        self.assertEqual(table.schema.field("action").type, pa.list_(pa.float32()))
        self.assertEqual(
            table.schema.field("observation.state").type,
            pa.list_(pa.float32()),
        )
        self.assertEqual(
            table.schema.field("observation.last_action").type,
            pa.list_(pa.float32()),
        )
        self.assertEqual(
            table.schema.field("action.replay.left_arm_joint_target").type,
            pa.list_(pa.float32()),
        )
        self.assertEqual(
            table.schema.field("observation.arm.left.actual_qpos").type,
            pa.list_(pa.float32()),
        )
        self.assertEqual(
            table.schema.field("fault_codes").type,
            pa.list_(pa.string()),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.left.id").type,
            pa.int64(),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.left.label").type,
            pa.string(),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.left.tags").type,
            pa.list_(pa.string()),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.right.tags").type,
            pa.list_(pa.string()),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.left.init_phase").type,
            pa.string(),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.left.init_phase_valid").type,
            pa.int64(),
        )
        self.assertEqual(
            table.schema.field("teleop.arm_mode.left.init_dwell_sec").type,
            pa.float64(),
        )
        self.assertEqual(
            table.schema.field(
                "observation.hand.left.tactile_process_cpu_affinity"
            ).type,
            pa.list_(pa.int64()),
        )
        self.assertEqual(
            table.column("teleop.channels.late.valid").to_pylist(),
            [None, True],
        )


if __name__ == "__main__":
    unittest.main()
