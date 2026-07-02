from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hardware.rm65.calibration_profiles import make_profile
from teleop_runtime.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from teleop_runtime.replay.checks import run_replay_preflight
from teleop_runtime.replay.executor import ReplayExecutionController
from teleop_runtime.replay.hardware import ReplayHardwareConfig
from teleop_runtime.replay.session import ReplayGuiSession
from teleop_runtime.replay.trajectory import ReplayTrajectoryGroup


def _write_profile(path: Path, *, side: str, tcp: list[float]) -> Path:
    profile = make_profile(name=f"test_{side}", arm=side, tool_name="test_tool")
    profile["tcp"]["pose_flange_m_rad"] = list(tcp)
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def _group(profile_path: Path, *, qpos_targets: bool = True) -> ReplayTrajectoryGroup:
    actions = np.full((2, ACTION_DIM), np.nan, dtype=np.float32)
    actions[:, ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [
            [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.21, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ReplayTrajectoryGroup(
        key="g",
        label="test",
        source_path="episode_000000.parquet",
        source_episode_index=0,
        start_index=0,
        end_index=2,
        frame_count=2,
        action_hz=10.0,
        actions=actions,
        arm_joint_targets={}
        if not qpos_targets
        else {"right": np.zeros((2, 6), dtype=np.float32)},
        metadata={
            "source_metadata": {
                "episode_metadata": {
                    "right_tool_profile": str(profile_path),
                }
            }
        },
    )


class ReplayExecutionChecksTest(unittest.TestCase):
    def test_preflight_accepts_matching_tcp_and_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            report = run_replay_preflight(
                group=_group(profile),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.asarray([0.01, 0, 0, 0, 0, 0])},
            )
            self.assertTrue(report.ok)

    def test_preflight_blocks_tcp_mismatch_and_missing_qpos_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            mismatch = run_replay_preflight(
                group=_group(profile),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.zeros(6)},
            )
            self.assertFalse(mismatch.ok)
            failed = {check.name for check in mismatch.checks if not check.ok}
            self.assertIn("right_tcp_numeric_match", failed)

            missing_qpos = run_replay_preflight(
                group=_group(profile, qpos_targets=False),
                active_sides=("right",),
                current_qpos_by_side={"right": np.zeros(6)},
                current_tcp_pose_by_side={"right": np.eye(4)},
                current_tcp_flange_by_side={"right": np.asarray([0.01, 0, 0, 0, 0, 0])},
            )
            self.assertFalse(missing_qpos.ok)
            failed = {check.name for check in missing_qpos.checks if not check.ok}
            self.assertIn("right_qpos_targets_present", failed)

    def test_executor_blocks_motion_until_output_is_enabled(self) -> None:
        controller = ReplayExecutionController()
        with self.assertRaisesRegex(RuntimeError, "Arm output is disabled"):
            controller.move_to_start()

    def test_session_hardware_config_supports_both_sides(self) -> None:
        session = ReplayGuiSession(
            hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False)
        )
        try:
            state = session.dispatch(
                {
                    "type": "configure_hardware",
                    "active_sides": "both",
                    "left_arm_ip": "192.168.1.10",
                    "right_arm_ip": "192.168.1.11",
                    "left_tool_profile": "/tmp/left.json",
                    "right_tool_profile": "/tmp/right.json",
                    "connect_hardware": False,
                    "send_actions": False,
                }
            )
            execution = state["execution"]
            self.assertEqual(execution["active_sides"], ["left", "right"])
            self.assertFalse(execution["settings"]["send_actions_enabled"])
            self.assertFalse(execution["hardware_config"]["connect_hardware"])
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
