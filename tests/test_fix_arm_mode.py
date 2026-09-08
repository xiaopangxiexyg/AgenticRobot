from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from teleop_runtime.calibration import ArmCalibration, TeleopCalibration
from teleop_runtime.diagnostics import FaultEvent, FaultSeverity
from control_station.web.calibration_web_ui import CalibrationWebUi
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopCommand
from teleop_runtime.special_tasks.fix_arm import (
    FixArmCommandController,
    FixArmPoseSnapshotRecorder,
    build_fix_arm_pose_snapshot_payload,
    advance_fix_arm_home_capture,
)
from teleop_runtime.special_tasks.fix_hold import SpecialArmFixHoldCoordinator


def _command(
    *,
    left_arm: float,
    right_arm: float,
    left_hand: float,
    right_hand: float,
) -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, left_arm, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="right_arm",
            side="right",
            command_type="joint_vector",
            value=np.full(6, right_arm, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=np.full(6, left_hand, dtype=float),
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="right_hand",
            side="right",
            command_type="joint_vector",
            value=np.full(6, right_hand, dtype=float),
        )
    )
    return command


def _controller(selected_side: str = "left") -> FixArmCommandController:
    return FixArmCommandController(
        active_sides=("left", "right"),
        selected_side=selected_side,
        required_effectors_by_side={
            "left": ("left_arm", "left_hand"),
            "right": ("right_arm", "right_hand"),
        },
    )


def _channel(name: str, side: str, value: float, timestamp: float = 1.0) -> ObservationChannel:
    return ObservationChannel(
        channel_name=name,
        capability="pose6d" if name.endswith("_wrist_pose") else "joint_vector",
        side=side,
        value=np.eye(4) * value if name.endswith("_wrist_pose") else np.full(6, value),
        frame_name="desk" if name.endswith("_wrist_pose") else None,
        timestamp=timestamp,
        source_name="test",
    )


def _channels(
    *,
    left_wrist: float,
    right_wrist: float,
    left_hand: float,
    right_hand: float,
    timestamp: float = 1.0,
) -> dict[str, ObservationChannel]:
    return {
        "left_wrist_pose": _channel("left_wrist_pose", "left", left_wrist, timestamp),
        "right_wrist_pose": _channel("right_wrist_pose", "right", right_wrist, timestamp),
        "left_hand_q6": _channel("left_hand_q6", "left", left_hand, timestamp),
        "right_hand_q6": _channel("right_hand_q6", "right", right_hand, timestamp),
    }


class FixArmCommandControllerTests(unittest.TestCase):
    def test_arm_only_fix_can_start_from_current_runtime_home_without_input(self) -> None:
        controller = FixArmCommandController(
            active_sides=("left", "right"),
            selected_side="left",
            required_effectors_by_side={
                "left": ("left_arm",),
                "right": ("right_arm",),
            },
        )
        left_home = np.linspace(0.1, 0.6, 6)
        controller.bind_runtime_home_arm_snapshot(
            {"left": left_home, "right": np.zeros(6)},
            epoch=17,
        )

        transition = controller.freeze_selected_from_runtime_home(
            now=2.0,
            expected_epoch=17,
        )

        self.assertTrue(transition.ok)
        self.assertEqual(controller.fixed_sides, ("left",))
        self.assertEqual(controller.scope_for_side("left"), "arm_only")
        command = TeleopCommand(timestamp=2.1)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 9.0),
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.full(6, 3.0),
            )
        )
        filtered = controller.filter_command(command, None, 2.1)
        np.testing.assert_allclose(filtered.commands["left_arm"].value, left_home)
        np.testing.assert_allclose(
            filtered.commands["right_arm"].value,
            np.full(6, 3.0),
        )
        self.assertEqual(
            filtered.commands["left_arm"].metadata["runtime_home_epoch"],
            17,
        )
        self.assertNotIn("left_hand", filtered.commands)

    def test_runtime_home_fix_rejects_stale_epoch_and_hand_scope(self) -> None:
        qpos = {"left": np.zeros(6), "right": np.ones(6)}
        arm_only = FixArmCommandController(
            active_sides=("left", "right"),
            selected_side="left",
            required_effectors_by_side={
                "left": ("left_arm",),
                "right": ("right_arm",),
            },
        )
        arm_only.bind_runtime_home_arm_snapshot(qpos, epoch=4)
        stale = arm_only.freeze_selected_from_runtime_home(
            now=1.0,
            expected_epoch=5,
        )
        self.assertFalse(stale.ok)
        self.assertIn("epoch mismatch", stale.message)
        self.assertFalse(arm_only.fixed)
        arm_only.cancel_active(now=1.1, clear_snapshots=True)
        cleared = arm_only.freeze_selected_from_runtime_home(
            now=1.2,
            expected_epoch=4,
        )
        self.assertFalse(cleared.ok)
        self.assertIn("cached=None", cleared.message)

        arm_and_hand = _controller("left")
        arm_and_hand.bind_runtime_home_arm_snapshot(qpos, epoch=4)
        unsupported = arm_and_hand.freeze_selected_from_runtime_home(
            now=1.0,
            expected_epoch=4,
        )
        self.assertFalse(unsupported.ok)
        self.assertIn("arm_only", unsupported.message)
        self.assertFalse(arm_and_hand.fixed)

    def test_two_pedal_switch_replaces_fixed_side_atomically(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertEqual(controller.fixed_sides, ("right",))

        transition = controller.start_two_pedal_transition(
            fixed_side="left",
            now=1.2,
            timeout_sec=1.0,
            min_samples=1,
        )
        self.assertTrue(transition.ok)
        self.assertEqual(controller.fixed_sides, ("right",))

        self.assertTrue(controller.complete_recalibration(now=1.3).ok)
        self.assertEqual(controller.fixed_sides, ("left",))
        self.assertNotEqual(controller.fixed_sides, ("left", "right"))

    def test_two_pedal_switch_to_bimanual_releases_only_after_home(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side=None,
                now=1.2,
                timeout_sec=1.0,
                min_samples=1,
            ).ok
        )
        self.assertEqual(controller.fixed_sides, ("right",))
        self.assertTrue(controller.complete_recalibration(now=1.3).ok)
        self.assertEqual(controller.fixed_sides, ())

    def test_two_pedal_transition_reuses_fresh_passive_home_cache(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        pose = np.eye(4)
        for timestamp in (1.10, 1.20, 1.36):
            controller.observe_two_pedal_home_cache(
                "right",
                pose,
                source_timestamp=timestamp,
            )
        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side="left",
                now=1.4,
                timeout_sec=1.0,
                min_samples=1,
            ).ok
        )
        self.assertEqual(controller.sample_count, 1)
        controller.observe_recalibration_sample(
            pose,
            source_timestamp=1.40,
        )
        self.assertEqual(
            controller.recalibration_stability_reason,
            "stable_tracker_sample_accepted",
        )

    def test_profile_transition_can_rebase_from_latest_fresh_moving_sample(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        first_pose = np.eye(4)
        moving_pose = np.eye(4)
        moving_pose[0, 3] = 0.02
        controller.observe_two_pedal_home_cache(
            "right",
            first_pose,
            source_timestamp=1.10,
        )
        controller.observe_two_pedal_home_cache(
            "right",
            moving_pose,
            source_timestamp=1.20,
        )
        stable_only = controller.passive_home_cache_status("right", now=1.22)
        self.assertFalse(stable_only["ready"])
        transition_status = controller.passive_home_cache_status(
            "right",
            now=1.22,
            allow_fresh_rebase=True,
        )
        self.assertTrue(transition_status["ready"])
        self.assertEqual(transition_status["quality"], "fresh_rebase")

        transition = controller.start_two_pedal_transition(
            fixed_side="left",
            now=1.22,
            timeout_sec=1.0,
            min_samples=1,
            allow_fresh_home_rebase=True,
        )

        self.assertTrue(transition.ok)
        self.assertEqual(controller.sample_count, 1)
        np.testing.assert_allclose(
            controller.recalibration_samples()[0],
            moving_pose,
        )
        self.assertEqual(
            controller.recalibration_stability_reason,
            "fresh_tracker_rebase_reused",
        )

    def test_profile_transition_rejects_stale_moving_sample(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        controller.observe_two_pedal_home_cache(
            "right",
            np.eye(4),
            source_timestamp=1.10,
        )

        status = controller.passive_home_cache_status(
            "right",
            now=1.30,
            allow_fresh_rebase=True,
        )

        self.assertFalse(status["ready"])
        self.assertFalse(status["fresh_rebase_ready"])

    def test_single_arm_mode_caches_both_sides_for_immediate_followup_switch(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.05).ok)
        left_pose = np.eye(4)
        right_pose = np.eye(4)
        right_pose[1, 3] = 0.1
        for timestamp in (1.10, 1.20, 1.36):
            controller.observe_two_pedal_home_cache(
                "left",
                left_pose,
                source_timestamp=timestamp,
            )
            controller.observe_two_pedal_home_cache(
                "right",
                right_pose,
                source_timestamp=timestamp,
            )

        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side="left",
                now=1.4,
            ).ok
        )
        self.assertEqual(controller.sample_count, 1)
        self.assertTrue(controller.complete_recalibration(now=1.41).ok)
        self.assertEqual(controller.fixed_sides, ("left",))

        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side=None,
                now=1.42,
            ).ok
        )
        self.assertEqual(controller.sample_count, 1)
        self.assertTrue(controller.complete_recalibration(now=1.43).ok)
        self.assertEqual(controller.fixed_sides, ())

    def test_high_rate_tracker_quantization_can_form_passive_home_cache(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(
                left_wrist=1.0,
                right_wrist=2.0,
                left_hand=3.0,
                right_hand=4.0,
            ),
            now=1.0,
        )
        controller.filter_command(
            _command(
                left_arm=1.0,
                right_arm=2.0,
                left_hand=3.0,
                right_hand=4.0,
            ),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.01).ok)

        accepted = False
        for index in range(80):
            pose = np.eye(4)
            pose[0, 3] = 0.0001 if index % 2 else 0.0
            accepted = (
                controller.observe_two_pedal_home_cache(
                    "right",
                    pose,
                    source_timestamp=1.02 + 0.004 * index,
                )
                or accepted
            )

        self.assertTrue(accepted)
        transition = controller.start_two_pedal_transition(
            fixed_side="left",
            now=1.34,
        )
        self.assertTrue(transition.ok)
        self.assertEqual(controller.sample_count, 1)

    def test_passive_home_cache_reports_raw_tracker_unavailability(self) -> None:
        controller = _controller("right")
        controller.note_two_pedal_home_unavailable(
            "right",
            reason="tracker_channel_invalid",
            observed_at=10.0,
        )

        status = controller.passive_home_cache_status("right", now=10.1)

        self.assertFalse(status["ready"])
        self.assertEqual(status["reason"], "tracker_channel_invalid")

    def test_two_pedal_transition_uses_actual_fixed_side_after_selection_drift(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(controller.select_side("right").ok)

        transition = controller.start_two_pedal_transition(
            fixed_side="right",
            now=1.2,
        )

        self.assertTrue(transition.ok)
        self.assertTrue(controller.recapturing)
        self.assertEqual(controller.fixed_side, "left")
        self.assertEqual(controller.selected_side, "left")
        self.assertTrue(controller.complete_recalibration(now=1.3).ok)
        self.assertEqual(controller.fixed_sides, ("right",))

    def test_new_two_pedal_request_replaces_pending_target(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side="right",
                now=1.2,
            ).ok
        )

        replacement = controller.start_two_pedal_transition(
            fixed_side=None,
            now=1.21,
        )

        self.assertTrue(replacement.ok)
        self.assertTrue(controller.two_pedal_transition_pending)
        self.assertIsNone(controller.two_pedal_transition_target_fixed_side)
        self.assertEqual(controller.recalibration_started_at, 1.21)
        self.assertEqual(controller.recalibration_timeout_sec, 1.0)
        self.assertEqual(controller.fixed_sides, ("left",))
        self.assertTrue(controller.complete_recalibration(now=1.3).ok)
        self.assertEqual(controller.fixed_sides, ())

    def test_unavailable_new_target_cancels_older_pending_destination(self) -> None:
        controller = FixArmCommandController(
            active_sides=("left", "right"),
            selected_side="left",
            required_effectors_by_side={
                "left": ("left_arm",),
                "right": ("right_arm",),
            },
        )
        controller.filter_channels(
            {
                "left_wrist_pose": _channel(
                    "left_wrist_pose",
                    "left",
                    1.0,
                )
            },
            now=1.0,
        )
        command = TeleopCommand(timestamp=1.0)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.ones(6),
            )
        )
        controller.filter_command(command, None, 1.0)
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side=None,
                now=1.2,
            ).ok
        )

        unavailable = controller.start_two_pedal_transition(
            fixed_side="right",
            now=1.21,
        )

        self.assertFalse(unavailable.ok)
        self.assertFalse(controller.recapturing)
        self.assertFalse(controller.two_pedal_transition_pending)
        self.assertEqual(controller.fixed_sides, ("left",))

    def test_pedal_for_current_single_arm_cancels_older_pending_switch(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side="right",
                now=1.2,
            ).ok
        )

        cancelled = controller.start_two_pedal_transition(
            fixed_side="left",
            now=1.21,
        )

        self.assertTrue(cancelled.ok)
        self.assertFalse(controller.recapturing)
        self.assertFalse(controller.two_pedal_transition_pending)
        self.assertEqual(controller.fixed_sides, ("left",))

    def test_two_pedal_continuous_single_arm_to_single_arm_to_bimanual(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side="right",
                now=1.2,
            ).ok
        )
        self.assertTrue(controller.complete_recalibration(now=1.3).ok)
        self.assertEqual(controller.fixed_sides, ("right",))

        self.assertTrue(
            controller.start_two_pedal_transition(
                fixed_side=None,
                now=1.4,
            ).ok
        )
        self.assertTrue(controller.complete_recalibration(now=1.5).ok)
        self.assertEqual(controller.fixed_sides, ())

    def test_both_sides_can_be_fixed_and_releasing_one_keeps_the_other_fixed(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(controller.select_side("right").ok)
        self.assertTrue(controller.freeze_selected(now=1.2).ok)
        self.assertEqual(controller.fixed_sides, ("left", "right"))

        filtered = controller.filter_command(
            _command(left_arm=10.0, right_arm=20.0, left_hand=30.0, right_hand=40.0),
            None,
            1.3,
        )
        np.testing.assert_allclose(filtered.commands["left_arm"].value, np.full(6, 1.0))
        np.testing.assert_allclose(filtered.commands["right_arm"].value, np.full(6, 2.0))

        self.assertTrue(controller.select_side("left").ok)
        self.assertTrue(
            controller.start_recalibration(now=1.4, timeout_sec=5.0, min_samples=1).ok
        )
        self.assertTrue(controller.complete_recalibration(now=1.5).ok)
        self.assertEqual(controller.fixed_sides, ("right",))

    def test_fix_hold_requires_both_fixed_like_sides_and_auto_releases(self) -> None:
        coordinator = SpecialArmFixHoldCoordinator()
        fix = SimpleNamespace(fixed=True, fixed_sides=("left",), fixed_side="left")
        init = SimpleNamespace(fixed=False, fixed_sides=(), active_side="")
        transition = coordinator.update(fix_controller=fix, init_controller=init)
        self.assertFalse(transition.active)

        init.fixed = True
        init.fixed_sides = ("right",)
        init.active_side = "right"
        transition = coordinator.update(fix_controller=fix, init_controller=init)
        self.assertTrue(transition.entered)
        self.assertEqual(transition.source_by_side, {"left": "fix_arm", "right": "init_fixed"})

        # Starting target-side Home recapture is still a pending release; the
        # episode must remain paused until recapture completes.
        init.fix_hold_sides = ("right",)
        init.fixed = False
        init.fixed_sides = ()
        transition = coordinator.update(fix_controller=fix, init_controller=init)
        self.assertTrue(transition.active)
        self.assertFalse(transition.exited)

        fix.fixed = False
        fix.fixed_sides = ()
        init.fix_hold_sides = ()
        transition = coordinator.update(fix_controller=fix, init_controller=init)
        self.assertTrue(transition.exited)
        self.assertFalse(transition.active)

    def test_regular_hold_escalation_is_not_an_auto_release(self) -> None:
        coordinator = SpecialArmFixHoldCoordinator()
        both = SimpleNamespace(
            fixed=True,
            fixed_sides=("left", "right"),
            fixed_side="left",
        )
        coordinator.update(fix_controller=both, init_controller=None)
        transition = coordinator.escalate_to_regular_hold()
        self.assertEqual(transition.reason, "escalated_to_regular_hold")
        self.assertEqual(transition.locked_sides, ("left", "right"))
        self.assertFalse(coordinator.active)

    def test_freeze_reuses_fixed_side_input_channels_and_leaves_other_side_live(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )

        transition = controller.freeze_selected(now=1.1)
        self.assertTrue(transition.ok)

        filtered_channels = controller.filter_channels(
            _channels(left_wrist=10.0, right_wrist=20.0, left_hand=30.0, right_hand=40.0),
            now=1.2,
        )

        np.testing.assert_allclose(filtered_channels["left_wrist_pose"].value, np.eye(4) * 1.0)
        np.testing.assert_allclose(filtered_channels["left_hand_q6"].value, np.full(6, 3.0))
        np.testing.assert_allclose(filtered_channels["right_wrist_pose"].value, np.eye(4) * 20.0)
        np.testing.assert_allclose(filtered_channels["right_hand_q6"].value, np.full(6, 40.0))
        self.assertEqual(filtered_channels["left_wrist_pose"].timestamp, 1.2)
        self.assertTrue(filtered_channels["left_wrist_pose"].metadata["fix_arm_frozen"])

    def test_freeze_overrides_pipeline_command_when_present(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        filtered = controller.filter_command(
            _command(left_arm=10.0, right_arm=20.0, left_hand=30.0, right_hand=40.0),
            None,
            1.2,
        )

        self.assertEqual(
            set(filtered.commands),
            {"left_arm", "right_arm", "left_hand", "right_hand"},
        )
        self.assertEqual(filtered.metadata["fix_arm_active_side"], "left")
        self.assertEqual(filtered.metadata["fix_arm_mode"], "fixed")
        self.assertEqual(filtered.metadata["fix_arm_scope"], "arm_and_end_effector")
        self.assertEqual(filtered.metadata["fix_arm_warning"], "")
        self.assertEqual(filtered.metadata["fix_arm_effectors"], ["left_arm", "left_hand"])
        np.testing.assert_allclose(filtered.commands["left_arm"].value, np.full(6, 1.0))
        np.testing.assert_allclose(filtered.commands["left_hand"].value, np.full(6, 3.0))
        np.testing.assert_allclose(filtered.commands["right_arm"].value, np.full(6, 20.0))
        np.testing.assert_allclose(filtered.commands["right_hand"].value, np.full(6, 40.0))
        self.assertTrue(filtered.commands["left_arm"].metadata["fix_arm_active"])
        self.assertEqual(filtered.commands["left_arm"].metadata["fix_arm_side"], "left")
        for effector_command in filtered.commands.values():
            np.testing.assert_equal(np.asarray(effector_command.value).shape, (6,))

    def test_arm_only_fix_emits_a_warning_when_no_end_effector_is_configured(self) -> None:
        controller = FixArmCommandController(
            active_sides=("left", "right"),
            required_effectors_by_side={
                "left": ("left_arm",),
                "right": ("right_arm",),
            },
            selected_side="left",
        )
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )

        transition = controller.freeze_selected(now=1.1)
        self.assertTrue(transition.ok)
        self.assertIn("warning", transition.message)

        filtered = controller.filter_command(
            _command(left_arm=10.0, right_arm=20.0, left_hand=30.0, right_hand=40.0),
            None,
            1.2,
        )
        self.assertEqual(filtered.metadata["fix_arm_scope"], "arm_only")
        self.assertIn("end_effector_not_configured", filtered.metadata["fix_arm_warning"])
        np.testing.assert_allclose(filtered.commands["left_arm"].value, np.full(6, 1.0))
        np.testing.assert_allclose(filtered.commands["left_hand"].value, np.full(6, 30.0))

    def test_freeze_does_not_inject_fixed_command_when_pipeline_omits_it_by_default(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        right_only = TeleopCommand(timestamp=1.2)
        right_only.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.full(6, 20.0),
            )
        )

        filtered = controller.filter_command(right_only, None, 1.2)

        self.assertEqual(set(filtered.commands), {"right_arm"})
        np.testing.assert_allclose(filtered.commands["right_arm"].value, np.full(6, 20.0))

    def test_freeze_can_inject_allowed_fixed_effector_for_out_of_band_dispatch(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        hand_only = controller.filter_command(
            TeleopCommand(timestamp=1.2),
            None,
            1.2,
            include_missing_fixed_effectors=True,
            allowed_effectors=("left_hand",),
        )

        self.assertEqual(set(hand_only.commands), {"left_hand"})
        np.testing.assert_allclose(hand_only.commands["left_hand"].value, np.full(6, 3.0))

    def test_freeze_requires_previous_commands_for_all_required_effectors(self) -> None:
        controller = _controller("right")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        command = TeleopCommand()
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.zeros(6),
            )
        )
        controller.filter_command(command, None, 1.0)

        transition = controller.freeze_selected(now=1.1)

        self.assertFalse(transition.ok)
        self.assertEqual(transition.missing_effectors, ("right_hand",))
        self.assertFalse(controller.fixed)

    def test_fault_filter_ignores_only_fixed_side_tracker_faults(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        controller.freeze_selected(now=1.1)
        faults = [
            FaultEvent(
                code="stale_channel",
                message="left stale",
                severity=FaultSeverity.ERROR,
                timestamp=1.2,
                channel_name="left_wrist_pose",
            ),
            FaultEvent(
                code="invalid_channel",
                message="left hand invalid",
                severity=FaultSeverity.ERROR,
                timestamp=1.2,
                channel_name="left_hand_q6",
            ),
            FaultEvent(
                code="invalid_channel",
                message="right invalid",
                severity=FaultSeverity.ERROR,
                timestamp=1.2,
                channel_name="right_wrist_pose",
            ),
            FaultEvent(
                code="low_control_rate",
                message="slow",
                severity=FaultSeverity.WARNING,
                timestamp=1.2,
            ),
        ]

        filtered = controller.filter_faults(faults, {}, 1.2)

        self.assertEqual([fault.message for fault in filtered], ["right invalid", "slow"])

    def test_cancel_active_releases_fixed_side_for_hold(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)

        transition = controller.cancel_active(
            now=1.2,
            reason="hold_requested",
            clear_snapshots=True,
        )

        self.assertTrue(transition.ok)
        self.assertFalse(controller.fixed)
        self.assertEqual(controller.fixed_side, "")
        self.assertIn("stopped", controller.status_message)


class FixArmRuntimeHomeTests(unittest.TestCase):
    def test_incremental_home_capture_updates_only_fixed_side_and_releases(self) -> None:
        controller = _controller("left")
        controller.filter_channels(
            _channels(left_wrist=1.0, right_wrist=2.0, left_hand=3.0, right_hand=4.0),
            now=1.0,
        )
        controller.filter_command(
            _command(left_arm=1.0, right_arm=2.0, left_hand=3.0, right_hand=4.0),
            None,
            1.0,
        )
        self.assertTrue(controller.freeze_selected(now=1.1).ok)
        self.assertTrue(
            controller.start_recalibration(now=1.2, timeout_sec=5.0, min_samples=2).ok
        )
        calibration = TeleopCalibration(
            arms={
                "left_arm": ArmCalibration(name="left_arm", side="left"),
                "right_arm": ArmCalibration(name="right_arm", side="right"),
            }
        )
        channel_pose = np.eye(4)
        channel = ObservationChannel(
            channel_name="left_wrist_pose",
            capability="pose6d",
            side="left",
            value=channel_pose,
            frame_name="desk",
            timestamp=1.3,
            source_name="test",
        )
        robot_home = np.eye(4)
        robot_home[:3, 3] = [0.1, 0.2, 0.3]

        class Retargeter:
            def __init__(self) -> None:
                self.anchors = None

            def set_home_anchors(self, *, human_pose_desk, robot_pose_desk) -> None:
                self.anchors = (human_pose_desk.copy(), robot_pose_desk.copy())

        retargeter = Retargeter()

        progress = advance_fix_arm_home_capture(
            controller=controller,
            channels={"left_wrist_pose": channel},
            calibration=calibration,
            now=1.3,
            pose_from_channel=lambda item, _calibration: np.asarray(item.value, dtype=float),
            mean_pose=lambda samples: samples[0],
            resolve_robot_home=lambda side: (robot_home.copy(), np.arange(6), "test_home"),
            side_retargeter=lambda side: retargeter,
            tcp_mode="tool_tcp",
        )
        self.assertTrue(progress.active)
        self.assertFalse(progress.completed)

        repeated = advance_fix_arm_home_capture(
            controller=controller,
            channels={"left_wrist_pose": channel},
            calibration=calibration,
            now=1.4,
            pose_from_channel=lambda item, _calibration: np.asarray(item.value, dtype=float),
            mean_pose=lambda samples: samples[0],
            resolve_robot_home=lambda side: (robot_home.copy(), np.arange(6), "test_home"),
            side_retargeter=lambda side: retargeter,
            tcp_mode="tool_tcp",
        )
        self.assertEqual(repeated.sample_count, 0)
        self.assertTrue(repeated.waiting_for_stability)
        self.assertIn("fresh", repeated.stability_reason)

        for timestamp in (1.6, 1.7):
            channel = ObservationChannel(
                channel_name="left_wrist_pose",
                capability="pose6d",
                side="left",
                value=channel_pose,
                frame_name="desk",
                timestamp=timestamp,
                source_name="test",
            )
            progress = advance_fix_arm_home_capture(
                controller=controller,
                channels={"left_wrist_pose": channel},
                calibration=calibration,
                now=timestamp,
                pose_from_channel=lambda item, _calibration: np.asarray(item.value, dtype=float),
                mean_pose=lambda samples: samples[0],
                resolve_robot_home=lambda side: (robot_home.copy(), np.arange(6), "test_home"),
                side_retargeter=lambda side: retargeter,
                tcp_mode="tool_tcp",
            )

        self.assertTrue(progress.completed)
        self.assertFalse(controller.fixed)
        np.testing.assert_allclose(
            calibration.arms["left_arm"].home_pose_for_tcp_mode(),
            robot_home,
        )
        np.testing.assert_allclose(calibration.captured_home_qpos["left_arm"], np.arange(6))
        self.assertIsNotNone(retargeter.anchors)
        self.assertIsNone(calibration.arms["right_arm"].home_pose_for_tcp_mode())


class _SnapshotRetargeter:
    home_human_pose_desk = np.eye(4)
    home_robot_pose_desk = np.eye(4) * 2.0
    last_target_pose_desk = np.eye(4) * 3.0


class FixArmPoseSnapshotTests(unittest.TestCase):
    def test_snapshot_recorder_writes_json_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            recorder = FixArmPoseSnapshotRecorder(tmp_dir)
            command = _command(
                left_arm=1.0,
                right_arm=2.0,
                left_hand=3.0,
                right_hand=4.0,
            )
            channels = _channels(
                left_wrist=1.0,
                right_wrist=2.0,
                left_hand=3.0,
                right_hand=4.0,
                timestamp=9.0,
            )
            calibration = TeleopCalibration(
                arms={
                    "left_arm": ArmCalibration(name="left_arm", side="left"),
                    "right_arm": ArmCalibration(name="right_arm", side="right"),
                }
            )

            payload = build_fix_arm_pose_snapshot_payload(
                timestamp=10.0,
                monotonic_time=20.0,
                step_index=7,
                active_sides=("left", "right"),
                teleop_state="running",
                control_hz=30.0,
                movement_multiple=1.0,
                tcp_mode="tool",
                calibration=calibration,
                raw_channels=channels,
                filtered_channels=channels,
                observation=None,
                step_command=command,
                active_command=command,
                hand_command=None,
                gripper_command=None,
                command_dispatched=True,
                faults=[],
                fix_arm_state={"fix_arm_available": True, "fix_arm_fixed_side": "left"},
                left_retargeter=_SnapshotRetargeter(),
                right_retargeter=_SnapshotRetargeter(),
                left_preview_qpos=np.arange(6),
                right_preview_qpos=np.arange(6) + 10,
                left_hardware_state={"current_qpos": np.arange(6)},
                right_hardware_state={"current_qpos": np.arange(6) + 10},
                left_command_state={"last_target_tcp_pose": np.eye(4)},
                right_command_state={"last_target_tcp_pose": np.eye(4) * 2.0},
                gripper_status={"left": {"sent": False}},
            )
            record = recorder.record(payload, now=10.0)

            self.assertEqual(record.sequence, 1)
            self.assertTrue(record.path.exists())
            self.assertTrue(recorder.index_path.exists())
            saved = json.loads(record.path.read_text(encoding="utf-8"))
            self.assertEqual(saved["schema"], "dexteleop_fix_arm_pose_snapshot_v1")
            self.assertEqual(saved["step_index"], 7)
            self.assertEqual(saved["fix_arm"]["fix_arm_fixed_side"], "left")
            self.assertEqual(
                saved["commands"]["active_command"]["commands"]["right_arm"]["value"],
                [2.0] * 6,
            )
            index_rows = recorder.index_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(index_rows), 1)
            self.assertEqual(json.loads(index_rows[0])["path"], str(record.path))


class FixArmWebUiTests(unittest.TestCase):
    def test_fix_arm_select_event_requires_active_control(self) -> None:
        ui = CalibrationWebUi(open_browser=False)
        self.assertFalse(ui.push_event("fix_arm_select:left"))
        ui.publish(active_controls=["fix_arm_select"])

        self.assertTrue(ui.push_event("fix_arm_select:right"))
        self.assertEqual(ui.consume_events(), ["fix_arm_select:right"])

    def test_pose_snapshot_event_requires_active_control(self) -> None:
        ui = CalibrationWebUi(open_browser=False)
        self.assertFalse(ui.push_event("record_pose_snapshot"))
        ui.publish(active_controls=["record_pose_snapshot"])

        self.assertTrue(ui.push_event("record_pose_snapshot"))
        self.assertEqual(ui.consume_events(), ["record_pose_snapshot"])


if __name__ == "__main__":
    unittest.main()
