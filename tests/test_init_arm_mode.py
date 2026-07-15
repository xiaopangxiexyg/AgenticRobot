from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from teleop_runtime.diagnostics import FaultEvent, FaultSeverity
from teleop_runtime.schemas import EffectorCommand, TeleopCommand
from teleop_runtime.special_tasks.init_arm import (
    InitArmCommandController,
    load_init_arm_targets,
)


def _arm_command(side: str, value: float) -> EffectorCommand:
    pose = np.eye(4)
    pose[0, 3] = value
    return EffectorCommand(
        effector_name=f"{side}_arm",
        side=side,
        command_type="cartesian_pose",
        value=pose,
        frame_name="arm_base",
        metadata={"target_pose_base": pose.copy()},
    )


def _command() -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0)
    command.add_command(_arm_command("left", 0.1))
    command.add_command(_arm_command("right", 0.2))
    return command


class _FakeCalibration:
    def desk_to_base(self, pose, *, side: str):
        del side
        return np.asarray(pose, dtype=float).reshape(4, 4)

    def get_arm_for_side(self, side: str):
        del side
        return SimpleNamespace(T_D_B=np.eye(4))


class _FakeIkSolver:
    def __init__(self) -> None:
        self.current_q = np.zeros(12)

    def compute_frame_pose(self, frame_name: str, qpos):
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        pose = np.eye(4)
        pose[0, 3] = float(qpos[0] if frame_name == "left_tcp" else qpos[6])
        return pose


class _FakeDualArmRetargeter:
    def __init__(self) -> None:
        self.ik_solver = _FakeIkSolver()
        self.profile = SimpleNamespace(left_frame_name="left_tcp", right_frame_name="right_tcp")
        self._tool_tcp_from_solver = {"left": np.eye(4), "right": np.eye(4)}

    def _split_solution(self, qpos):
        qpos = np.asarray(qpos, dtype=float).reshape(12)
        return qpos[0:6].copy(), qpos[6:12].copy()

    def _embed_arm_qpos(self, *, current_full_q, left_qpos, right_qpos):
        qpos = np.asarray(current_full_q, dtype=float).reshape(12).copy()
        qpos[0:6] = np.asarray(left_qpos, dtype=float).reshape(6)
        qpos[6:12] = np.asarray(right_qpos, dtype=float).reshape(6)
        return qpos


class InitArmModeTests(unittest.TestCase):
    def test_load_profile_resolves_qpos_to_base_pose_and_dual_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = Path(tmp_dir) / "init_pose.json"
            profile.write_text(
                json.dumps(
                    {
                        "sides": {
                            "left": {"arm_qpos_deg": [10, 0, 0, 0, 0, 0]},
                            "right": {"arm_qpos_deg": [-20, 0, 0, 0, 0, 0]},
                        }
                    }
                ),
                encoding="utf-8",
            )

            targets = load_init_arm_targets(
                profile,
                calibration=_FakeCalibration(),
                dual_arm_retargeter=_FakeDualArmRetargeter(),
            )

        self.assertEqual(set(targets), {"left", "right"})
        np.testing.assert_allclose(targets["left"].arm_qpos[0], np.deg2rad(10))
        np.testing.assert_allclose(targets["right"].arm_qpos[0], np.deg2rad(-20))
        np.testing.assert_allclose(targets["left"].tcp_pose_base[0, 3], np.deg2rad(10))
        np.testing.assert_allclose(targets["right"].solver_pose_desk[0, 3], np.deg2rad(-20))

    def test_filter_command_overrides_only_active_side_without_dual_hook(self) -> None:
        left_target_pose = np.eye(4)
        left_target_pose[1, 3] = 0.42
        controller = InitArmCommandController(
            targets_by_side={
                "left": SimpleNamespace(
                    side="left",
                    tcp_pose_base=left_target_pose,
                    tcp_pose_desk=left_target_pose,
                    solver_pose_desk=None,
                    arm_qpos=np.arange(6, dtype=float) * 0.1,
                    source_path="profile.json",
                ),
                "right": SimpleNamespace(
                    side="right",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=None,
                    arm_qpos=None,
                    source_path="profile.json",
                ),
            },
            selected_side="left",
            use_dual_target_override=False,
        )
        self.assertTrue(controller.start_init(now=1.0).ok)

        filtered = controller.filter_command(_command(), None, 1.1)

        np.testing.assert_allclose(filtered.commands["left_arm"].value, left_target_pose)
        np.testing.assert_allclose(filtered.commands["right_arm"].value, _command().commands["right_arm"].value)
        self.assertEqual(filtered.commands["left_arm"].command_type, "cartesian_pose")
        self.assertEqual(filtered.commands["left_arm"].frame_name, "arm_base")
        self.assertTrue(filtered.commands["left_arm"].metadata["init_arm_active"])
        np.testing.assert_allclose(
            filtered.commands["left_arm"].metadata["solver_output_qpos"],
            np.arange(6, dtype=float) * 0.1,
        )

    def test_dual_target_override_replaces_only_active_solver_target(self) -> None:
        target_pose = np.eye(4)
        target_pose[2, 3] = 0.33
        controller = InitArmCommandController(
            targets_by_side={
                "left": SimpleNamespace(
                    side="left",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=target_pose,
                    arm_qpos=None,
                    source_path="profile.json",
                ),
                "right": SimpleNamespace(
                    side="right",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=np.eye(4),
                    arm_qpos=None,
                    source_path="profile.json",
                ),
            },
            selected_side="left",
            use_dual_target_override=True,
        )
        self.assertTrue(controller.start_init(now=1.0).ok)
        current_targets = {"left": np.eye(4), "right": np.eye(4) * 2.0}

        filtered = controller.filter_dual_arm_targets(current_targets, None, None)

        np.testing.assert_allclose(filtered["left"], target_pose)
        np.testing.assert_allclose(filtered["right"], current_targets["right"])

    def test_filter_faults_suppresses_active_side_tracker_fault_only(self) -> None:
        controller = InitArmCommandController(
            targets_by_side={
                "left": SimpleNamespace(
                    side="left",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=np.eye(4),
                    arm_qpos=None,
                    source_path="profile.json",
                ),
                "right": SimpleNamespace(
                    side="right",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=np.eye(4),
                    arm_qpos=None,
                    source_path="profile.json",
                ),
            },
            selected_side="left",
        )
        self.assertTrue(controller.start_init(now=1.0).ok)
        faults = [
            FaultEvent(
                code="translation_jump",
                message="left jump",
                severity=FaultSeverity.ERROR,
                timestamp=1.0,
                channel_name="left_wrist_pose",
            ),
            FaultEvent(
                code="translation_jump",
                message="right jump",
                severity=FaultSeverity.ERROR,
                timestamp=1.0,
                channel_name="right_wrist_pose",
            ),
        ]

        filtered = controller.filter_faults(faults, {}, 1.1)

        self.assertEqual([fault.channel_name for fault in filtered], ["right_wrist_pose"])

    def test_cancel_active_releases_init_side_for_hold(self) -> None:
        controller = InitArmCommandController(
            targets_by_side={
                "left": SimpleNamespace(
                    side="left",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=np.eye(4),
                    arm_qpos=None,
                    source_path="profile.json",
                ),
                "right": SimpleNamespace(
                    side="right",
                    tcp_pose_base=np.eye(4),
                    tcp_pose_desk=np.eye(4),
                    solver_pose_desk=np.eye(4),
                    arm_qpos=None,
                    source_path="profile.json",
                ),
            },
            selected_side="right",
        )
        self.assertTrue(controller.start_init(now=1.0).ok)

        transition = controller.cancel_active(now=1.1, reason="hold_requested")

        self.assertTrue(transition.ok)
        self.assertFalse(controller.active)
        self.assertEqual(controller.active_side, "")
        self.assertIn("stopped", controller.status_message)

    def test_arrival_status_is_display_only_and_requires_measured_stillness(self) -> None:
        target = np.eye(4)
        controller = InitArmCommandController(
            targets_by_side={
                "left": SimpleNamespace(
                    side="left",
                    tcp_pose_base=target,
                    tcp_pose_desk=target,
                    solver_pose_desk=target,
                    arm_qpos=None,
                    source_path="profile.json",
                ),
                "right": SimpleNamespace(
                    side="right",
                    tcp_pose_base=target,
                    tcp_pose_desk=target,
                    solver_pose_desk=target,
                    arm_qpos=None,
                    source_path="profile.json",
                ),
            },
            selected_side="left",
        )
        self.assertTrue(controller.start_init(now=0.0).ok)

        controller.observe_actual_feedback(None, timestamp=0.0, valid=False)
        status = controller.status_payload()
        self.assertEqual(status["init_arm_arrival_label"], "未到位")

        for timestamp in np.arange(0.0, 3.3, 0.1):
            controller.observe_actual_feedback(target, timestamp=float(timestamp), valid=True)

        status = controller.status_payload()
        self.assertEqual(status["init_arm_arrival_label"], "到位稳定")
        self.assertEqual(status["init_arm_arrival_status"], "stable")
        self.assertTrue(controller.active)  # Display status must not release or gate init-arm.


if __name__ == "__main__":
    unittest.main()
