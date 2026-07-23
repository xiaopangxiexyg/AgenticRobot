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
    advance_init_arm_home_capture,
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


class _FakeDualArmRetargeter:
    def __init__(self) -> None:
        self._current = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }

    def current_qpos_by_side(self):
        return {side: qpos.copy() for side, qpos in self._current.items()}

    @staticmethod
    def solver_world_targets_for_qpos_pair(*, left_qpos, right_qpos):
        targets = {}
        for side, qpos in (("left", left_qpos), ("right", right_qpos)):
            pose = np.eye(4)
            pose[0, 3] = float(np.asarray(qpos, dtype=float).reshape(6)[0])
            targets[side] = pose
        return targets

    @staticmethod
    def tool_tcp_world_targets_for_qpos_pair(*, left_qpos, right_qpos):
        return _FakeDualArmRetargeter.solver_world_targets_for_qpos_pair(
            left_qpos=left_qpos,
            right_qpos=right_qpos,
        )

    @staticmethod
    def solver_world_target_from_tool_tcp_world_target(
        *, side, tool_tcp_world_target
    ):
        assert side in {"left", "right"}
        return np.asarray(tool_tcp_world_target, dtype=float).reshape(4, 4).copy()


class InitArmModeTests(unittest.TestCase):
    @staticmethod
    def _qpos_controller(*, collision_checker=None) -> InitArmCommandController:
        targets = {}
        for side, first_joint in (("left", 0.1), ("right", -0.1)):
            qpos = np.zeros(6, dtype=float)
            qpos[0] = first_joint
            targets[side] = SimpleNamespace(
                side=side,
                tcp_pose_base=np.eye(4),
                tcp_pose_desk=np.eye(4),
                solver_pose_desk=np.eye(4),
                arm_qpos=qpos,
                source_path="profile.json",
            )
        return InitArmCommandController(
            targets_by_side=targets,
            selected_side="left",
            qpos_final_collision_checker=collision_checker,
        )

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

    def test_load_profile_accepts_rm65_default_arms_qpos_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile = Path(tmp_dir) / "bench_init_pose.json"
            profile.write_text(
                json.dumps(
                    {
                        "arms": {
                            "left": {"qpos": [0, 1, 2, 3, 4, 5], "qpos_units": "rad"},
                            "right": {"qpos": [5, 4, 3, 2, 1, 0], "qpos_units": "rad"},
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

        np.testing.assert_allclose(targets["left"].arm_qpos, np.arange(6, dtype=float))
        np.testing.assert_allclose(
            targets["right"].arm_qpos,
            np.arange(5, -1, -1, dtype=float),
        )

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

    def test_exact_qpos_plan_rewrites_both_arms_and_requires_measured_arrival(self) -> None:
        controller = self._qpos_controller()
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        midpoint = {"left": np.array([0.05, 0, 0, 0, 0, 0]), "right": np.zeros(6)}
        target = {"left": np.array([0.1, 0, 0, 0, 0, 0]), "right": np.zeros(6)}
        plan = SimpleNamespace(
            completed=True,
            steps=(
                SimpleNamespace(index=1, qpos_by_side=midpoint),
                SimpleNamespace(index=2, qpos_by_side=target),
            ),
            final_qpos_by_side=target,
            summary=lambda: {"duration_sec": 0.1, "step_count": 2},
        )
        measured = {side: qpos.copy() for side, qpos in start.items()}

        transition = controller.start_qpos_planning(
            start_qpos_by_side=start,
            target_qpos_by_side=target,
            feedback_provider=lambda: measured,
            plan_builder=lambda: plan,
            now=1.0,
            arrival_dwell_sec=0.5,
        )
        self.assertTrue(transition.ok)
        controller._qpos_operation.plan_future.result(timeout=1.0)

        seed = controller.qpos_seed_qpos_by_side()
        self.assertEqual(set(seed), {"left", "right"})
        np.testing.assert_allclose(seed["left"], start["left"])
        np.testing.assert_allclose(seed["right"], start["right"])

        first = controller.filter_command(_command(), None, 1.1)
        self.assertEqual(first.commands["left_arm"].command_type, "joint_vector")
        self.assertEqual(first.commands["right_arm"].command_type, "joint_vector")
        np.testing.assert_allclose(first.commands["left_arm"].value, midpoint["left"])
        np.testing.assert_allclose(first.commands["right_arm"].value, midpoint["right"])

        second = controller.filter_command(_command(), None, 1.2)
        np.testing.assert_allclose(second.commands["left_arm"].value, target["left"])
        self.assertEqual(controller.mode, "qpos_settling")
        self.assertIsNone(controller.consume_qpos_terminal_event())

        measured.update({side: qpos.copy() for side, qpos in target.items()})
        controller.filter_command(_command(), None, 1.3)
        controller.filter_command(_command(), None, 1.9)
        terminal = controller.consume_qpos_terminal_event()
        self.assertEqual(terminal["status"], "completed")
        self.assertTrue(controller.active)
        self.assertTrue(controller.fixed)
        self.assertEqual(controller.fixed_sides, ("left",))
        self.assertTrue(controller.cancel_active(now=2.0, reason="hold_requested").ok)
        self.assertFalse(controller.active)
        controller.close()

    def test_exact_qpos_timed_plan_is_independent_of_caller_loop_rate(self) -> None:
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        midpoint = {
            "left": np.array([0.05, 0, 0, 0, 0, 0]),
            "right": np.zeros(6),
        }
        target = {
            "left": np.array([0.1, 0, 0, 0, 0, 0]),
            "right": np.zeros(6),
        }
        plan = SimpleNamespace(
            completed=True,
            steps=(
                SimpleNamespace(index=1, time_sec=0.05, qpos_by_side=midpoint),
                SimpleNamespace(index=2, time_sec=0.10, qpos_by_side=target),
            ),
            final_qpos_by_side=target,
            summary=lambda: {"duration_sec": 0.1, "step_count": 2},
        )

        def started_controller():
            controller = self._qpos_controller()
            self.assertTrue(
                controller.start_qpos_planning(
                    start_qpos_by_side=start,
                    target_qpos_by_side=target,
                    feedback_provider=lambda: start,
                    plan_builder=lambda: plan,
                    now=1.0,
                ).ok
            )
            controller._qpos_operation.plan_future.result(timeout=1.0)
            initial = controller.filter_command(_command(), None, 2.0)
            np.testing.assert_allclose(initial.commands["left_arm"].value, start["left"])
            return controller

        fast_caller = started_controller()
        slow_caller = started_controller()
        fast_caller.filter_command(_command(), None, 2.02)
        fast_caller.filter_command(_command(), None, 2.04)
        fast = fast_caller.filter_command(_command(), None, 2.075)
        slow = slow_caller.filter_command(_command(), None, 2.075)

        expected = np.array([0.075, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(fast.commands["left_arm"].value, expected)
        np.testing.assert_allclose(slow.commands["left_arm"].value, expected)
        np.testing.assert_allclose(
            fast.commands["left_arm"].value,
            slow.commands["left_arm"].value,
        )
        final = slow_caller.filter_command(_command(), None, 2.10)
        np.testing.assert_allclose(final.commands["left_arm"].value, target["left"])
        self.assertEqual(slow_caller.mode, "qpos_settling")
        fast_caller.close()
        slow_caller.close()

    def test_exact_qpos_plan_rejects_any_motion_of_non_target_support_arm(self) -> None:
        controller = self._qpos_controller()
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        invalid = {
            "left": np.array([0.05, 0, 0, 0, 0, 0]),
            "right": np.array([0.01, 0, 0, 0, 0, 0]),
        }
        plan = SimpleNamespace(
            completed=True,
            steps=(SimpleNamespace(index=1, qpos_by_side=invalid),),
            final_qpos_by_side=invalid,
            summary=lambda: {"duration_sec": 0.05, "step_count": 1},
        )
        self.assertTrue(
            controller.start_qpos_planning(
                start_qpos_by_side=start,
                target_qpos_by_side=invalid,
                feedback_provider=lambda: start,
                plan_builder=lambda: plan,
                now=1.0,
            ).ok
        )
        controller._qpos_operation.plan_future.result(timeout=1.0)
        controller.filter_command(_command(), None, 1.1)
        terminal = controller.consume_qpos_terminal_event()
        self.assertEqual(terminal["status"], "failed")
        self.assertIn("moves non-target support arm right", terminal["message"])
        controller.close()

    def test_exact_qpos_interrupt_fails_to_hold_if_support_arm_drifted(self) -> None:
        controller = self._qpos_controller()
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        midpoint = {"left": np.array([0.05, 0, 0, 0, 0, 0]), "right": np.zeros(6)}
        plan = SimpleNamespace(
            completed=True,
            steps=(SimpleNamespace(index=1, qpos_by_side=midpoint),),
            final_qpos_by_side=midpoint,
            summary=lambda: {"duration_sec": 0.05, "step_count": 1},
        )
        measured = {side: qpos.copy() for side, qpos in start.items()}
        self.assertTrue(
            controller.start_qpos_planning(
                start_qpos_by_side=start,
                target_qpos_by_side=midpoint,
                feedback_provider=lambda: measured,
                plan_builder=lambda: plan,
                now=1.0,
            ).ok
        )
        controller._qpos_operation.plan_future.result(timeout=1.0)
        controller.filter_command(_command(), None, 1.1)
        measured["right"] = np.array([0.02, 0, 0, 0, 0, 0])
        transition = controller.start_recalibration(
            now=1.2,
            timeout_sec=5.0,
            min_samples=2,
        )
        self.assertFalse(transition.ok)
        terminal = controller.consume_qpos_terminal_event()
        self.assertEqual(terminal["status"], "failed")
        self.assertIn("support arm drifted", terminal["message"])
        controller.close()

    def test_second_exact_qpos_trigger_pauses_at_feedback_and_ends_init_label(self) -> None:
        controller = self._qpos_controller()
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        midpoint = {
            "left": np.array([0.05, 0, 0, 0, 0, 0]),
            "right": np.zeros(6),
        }
        target = {
            "left": np.array([0.1, 0, 0, 0, 0, 0]),
            "right": np.zeros(6),
        }
        plan = SimpleNamespace(
            completed=True,
            steps=(
                SimpleNamespace(index=1, qpos_by_side=midpoint),
                SimpleNamespace(index=2, qpos_by_side=target),
            ),
            final_qpos_by_side=target,
            summary=lambda: {"duration_sec": 0.1, "step_count": 2},
        )
        measured = {side: qpos.copy() for side, qpos in start.items()}
        self.assertTrue(
            controller.start_qpos_planning(
                start_qpos_by_side=start,
                target_qpos_by_side=target,
                feedback_provider=lambda: measured,
                plan_builder=lambda: plan,
                now=1.0,
            ).ok
        )
        controller._qpos_operation.plan_future.result(timeout=1.0)
        controller.filter_command(_command(), None, 1.1)
        measured.update({side: qpos.copy() for side, qpos in midpoint.items()})

        transition = controller.start_recalibration(
            now=1.2,
            timeout_sec=5.0,
            min_samples=2,
        )
        paused = controller.filter_command(_command(), None, 1.2)

        self.assertTrue(transition.ok)
        self.assertTrue(controller.recapturing)
        self.assertEqual(controller.recalibration_sides, ("left",))
        self.assertNotIn("init_arm_active_side", paused.metadata)
        self.assertFalse(paused.metadata["init_arm_label_active"])
        self.assertEqual(paused.metadata["arm_mode_stopped_sides"], ["left"])
        np.testing.assert_allclose(paused.commands["left_arm"].value, midpoint["left"])
        self.assertEqual(paused.commands["right_arm"].command_type, "cartesian_pose")
        np.testing.assert_allclose(paused.commands["right_arm"].value, _arm_command("right", 0.2).value)
        controller.close()

    def test_exact_qpos_home_recapture_reanchors_target_tracker_only_before_live(self) -> None:
        controller = self._qpos_controller()
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        plan = SimpleNamespace(
            completed=True,
            steps=(SimpleNamespace(index=1, qpos_by_side=start),),
            final_qpos_by_side=start,
            summary=lambda: {"duration_sec": 0.05, "step_count": 1},
        )
        self.assertTrue(
            controller.start_qpos_planning(
                start_qpos_by_side=start,
                target_qpos_by_side=start,
                feedback_provider=lambda: start,
                plan_builder=lambda: plan,
                now=1.0,
            ).ok
        )
        self.assertTrue(
            controller.start_recalibration(
                now=1.1,
                timeout_sec=5.0,
                min_samples=2,
            ).ok
        )

        class Calibration:
            def __init__(self) -> None:
                self.poses = []
                self.qpos = []

            def capture_home_pose(self, name, pose, **_kwargs):
                self.poses.append((name, np.asarray(pose).copy()))

            def capture_home_qpos(self, name, qpos):
                self.qpos.append((name, np.asarray(qpos).copy()))

        class Retargeter:
            def __init__(self) -> None:
                self.anchors = []

            def set_home_anchors(self, **kwargs):
                self.anchors.append(kwargs)

        calibration = Calibration()
        retargeters = {"left": Retargeter(), "right": Retargeter()}
        channels = {
            side + "_wrist_pose": SimpleNamespace(
                valid=True,
                value=np.eye(4),
                timestamp=1.2,
            )
            for side in ("left", "right")
        }
        progress = None
        for timestamp in (1.2, 1.5, 1.6):
            channels["left_wrist_pose"].timestamp = timestamp
            progress = advance_init_arm_home_capture(
                controller=controller,
                channels=channels,
                calibration=calibration,
                now=timestamp,
                pose_from_channel=lambda channel, _calibration: channel.value,
                mean_pose=lambda samples: np.asarray(samples).mean(axis=0),
                resolve_robot_home=lambda side: (np.eye(4), np.zeros(6), side + "_state"),
                side_retargeter=lambda side: retargeters[side],
            )

        assert progress is not None
        self.assertTrue(progress.completed)
        self.assertEqual(progress.sides, ("left",))
        self.assertFalse(controller.active)
        self.assertEqual([name for name, _pose in calibration.poses], ["left_arm"])
        self.assertEqual(len(retargeters["left"].anchors), 1)
        self.assertEqual(len(retargeters["right"].anchors), 0)
        controller.close()

    def test_exact_qpos_final_collision_guard_fails_closed(self) -> None:
        collision_result = SimpleNamespace(
            safe=False,
            reason="distance_below_minimum",
            min_distance_m=0.01,
            collision_pair_count=1,
            required_min_distance_m=0.025,
        )
        controller = self._qpos_controller(
            collision_checker=lambda **_kwargs: collision_result
        )
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        plan = SimpleNamespace(
            completed=True,
            steps=(SimpleNamespace(index=1, qpos_by_side=start),),
            final_qpos_by_side=start,
            summary=lambda: {"duration_sec": 0.05, "step_count": 1},
        )
        self.assertTrue(
            controller.start_qpos_planning(
                start_qpos_by_side=start,
                target_qpos_by_side=start,
                feedback_provider=lambda: start,
                plan_builder=lambda: plan,
                now=1.0,
            ).ok
        )
        controller._qpos_operation.plan_future.result(timeout=1.0)
        command = controller.filter_command(_command(), None, 1.1)

        fault = controller.qpos_final_collision_fault(command, now=1.1)

        self.assertEqual(fault.code, "init_arm_qpos_final_collision_check_failed")
        self.assertEqual(controller.consume_qpos_terminal_event()["status"], "failed")
        controller.close()

    def test_exact_qpos_feedback_loss_transitions_to_failed_hold(self) -> None:
        controller = self._qpos_controller()
        start = {"left": np.zeros(6), "right": np.zeros(6)}
        steps = tuple(
            SimpleNamespace(index=index, qpos_by_side=start)
            for index in range(1, 3)
        )
        plan = SimpleNamespace(
            completed=True,
            steps=steps,
            final_qpos_by_side=start,
            summary=lambda: {"duration_sec": 0.1, "step_count": 2},
        )
        calls = 0

        def feedback():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("disconnected")
            return start

        self.assertTrue(
            controller.start_qpos_planning(
                start_qpos_by_side=start,
                target_qpos_by_side=start,
                feedback_provider=feedback,
                plan_builder=lambda: plan,
                now=1.0,
                tracking_check_every=1,
            ).ok
        )
        controller._qpos_operation.plan_future.result(timeout=1.0)
        controller.filter_command(_command(), None, 1.1)
        held = controller.filter_command(_command(), None, 1.2)

        terminal = controller.consume_qpos_terminal_event()
        self.assertEqual(terminal["status"], "failed")
        self.assertIn("lost fresh hardware feedback", terminal["message"])
        np.testing.assert_allclose(held.commands["left_arm"].value, start["left"])
        controller.close()


if __name__ == "__main__":
    unittest.main()
