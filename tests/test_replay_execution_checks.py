from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from hardware.rm65.calibration_profiles import make_profile
from teleop_runtime.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM
from teleop_runtime.replay.checks import run_replay_preflight
from teleop_runtime.replay.executor import ReplayExecutionController, ReplayExecutionSettings
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


class _FakeReplayRig:
    connected = True

    def __init__(self) -> None:
        self.hold_reasons: list[str] = []
        self.restore_count = 0

    def hold_motion(self, reason: str = "hardware replay hold") -> None:
        self.hold_reasons.append(reason)

    def restore_replay_speed(self) -> None:
        self.restore_count += 1


class _FakeReplayWorkerRig(_FakeReplayRig):
    def __init__(self) -> None:
        super().__init__()
        self.begin_count = 0
        self.sent_joint_targets: list[dict[str, np.ndarray]] = []

    def begin_replay_speed(self) -> None:
        self.begin_count += 1

    def send_joint_targets(self, qpos_by_side, *, source: str) -> None:
        self.sent_joint_targets.append(
            {
                str(side): np.asarray(qpos, dtype=float).copy()
                for side, qpos in dict(qpos_by_side).items()
            }
        )

    def send_action_poses(self, action_by_side, *, source: str) -> None:
        raise AssertionError("hardware replay must use recorded qpos joint targets, not pose action dispatch")


@dataclass(frozen=True)
class _FakePlanStep:
    index: int
    qpos_by_side: dict[str, np.ndarray]


class _FakeQposInitPlan:
    completed = True

    def __init__(self, steps: list[_FakePlanStep]) -> None:
        self.steps = tuple(steps)

    def summary(self) -> dict[str, object]:
        return {"completed": True, "step_count": len(self.steps), "metadata": {"planner": "fake"}}


class _FakeQposInitRig(_FakeReplayWorkerRig):
    def __init__(self) -> None:
        super().__init__()
        self.current = {"right": np.zeros(6, dtype=float)}

    def current_qpos_by_side(self) -> dict[str, np.ndarray]:
        return {side: qpos.copy() for side, qpos in self.current.items()}

    def send_joint_targets(self, qpos_by_side, *, source: str) -> None:
        super().send_joint_targets(qpos_by_side, source=source)
        for side, qpos in dict(qpos_by_side).items():
            self.current[str(side)] = np.asarray(qpos, dtype=float).copy()

    def qpos_reached(self, qpos_by_side, *, tolerance: float) -> bool:
        for side, target in dict(qpos_by_side).items():
            current = self.current.get(str(side))
            if current is None:
                return False
            if float(np.max(np.abs(current - np.asarray(target, dtype=float).reshape(6)))) > float(tolerance):
                return False
        return True


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

    def test_executor_abort_holds_hardware_and_restores_speed(self) -> None:
        controller = ReplayExecutionController()
        rig = _FakeReplayRig()
        controller._rig = rig  # type: ignore[assignment]
        controller._state = "replaying"

        status = controller.abort()

        self.assertEqual(status["state"], "aborted")
        self.assertEqual(rig.hold_reasons, ["hardware replay abort"])
        self.assertEqual(rig.restore_count, 1)

    def test_replay_worker_dispatches_recorded_qpos_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(send_actions_enabled=True),
            )
            rig = _FakeReplayWorkerRig()

            controller._replay_worker(rig, group)  # type: ignore[arg-type]

            self.assertEqual(controller.status()["state"], "finished")
            self.assertEqual(rig.begin_count, 1)
            self.assertEqual(rig.restore_count, 1)
            self.assertEqual(len(rig.sent_joint_targets), 2)
            np.testing.assert_allclose(rig.sent_joint_targets[0]["right"], np.zeros(6))
            np.testing.assert_allclose(rig.sent_joint_targets[1]["right"], np.zeros(6))

    def test_move_to_start_executes_qpos_init_plan_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile = _write_profile(Path(tmpdir) / "right.json", side="right", tcp=[0.01, 0, 0, 0, 0, 0])
            group = _group(profile)
            target = {"right": np.full(6, 0.2, dtype=float)}
            plan = _FakeQposInitPlan(
                [
                    _FakePlanStep(index=1, qpos_by_side={"right": np.full(6, 0.1, dtype=float)}),
                    _FakePlanStep(index=2, qpos_by_side=target),
                ]
            )
            controller = ReplayExecutionController(
                hardware_config=ReplayHardwareConfig(active_sides=("right",), connect_hardware=False),
                settings=ReplayExecutionSettings(
                    send_actions_enabled=True,
                    qpos_init_dt_sec=0.001,
                ),
            )
            rig = _FakeQposInitRig()

            with mock.patch("teleop_runtime.replay.executor.build_rm65_qpos_init_plan", return_value=plan):
                controller._move_to_start_worker(rig, group, target)  # type: ignore[arg-type]

            status = controller.status()
            self.assertEqual(status["state"], "start_ready")
            self.assertEqual(status["qpos_init"]["step_index"], 2)
            self.assertEqual(status["qpos_init"]["step_count"], 2)
            self.assertEqual(len(rig.sent_joint_targets), 2)
            np.testing.assert_allclose(rig.sent_joint_targets[0]["right"], np.full(6, 0.1))
            np.testing.assert_allclose(rig.sent_joint_targets[1]["right"], np.full(6, 0.2))

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
