from __future__ import annotations

import argparse
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_motion.action import HandActionSpec
from robot_motion.deploy.schema import ACTION_HAND_SLICES
from teleop_runtime.deploy.policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitPlan,
    DeployPolicyQposInitSettings,
    _current_qpos_by_side,
    load_deploy_policy_qpos_init_plan,
)
from teleop_runtime.replay.executor import ReplayExecutionSettings
from teleop_runtime.deploy.run_policy import (
    _announce_policy_qpos_init_complete,
    _begin_policy_qpos_init_with_force_guard,
    _finish_policy_qpos_init_reference_cut,
    _reset_startup_initializer,
    add_args,
    validate_args,
)
from teleop_runtime.deploy.schema import ACTION_DIM
from teleop_runtime.deploy.startup_init import (
    StartupInitializer,
    finish_policy_qpos_init_reference_cut,
    reset_startup_initializer,
)


def test_run_policy_preserves_startup_init_compatibility_exports() -> None:
    assert (
        _finish_policy_qpos_init_reference_cut
        is finish_policy_qpos_init_reference_cut
    )
    assert _reset_startup_initializer is reset_startup_initializer


def test_policy_qpos_init_completion_announcement_is_explicit() -> None:
    with patch(
        "teleop_runtime.deploy.run_policy.console_log_event"
    ) as log_event:
        _announce_policy_qpos_init_complete(
            wait_for_ready_confirm=True,
            now=123.0,
        )

    assert log_event.call_count == 2
    assert "G INITIALIZATION COMPLETE" in log_event.call_args_list[0].args[1]
    assert "PRESS A" in log_event.call_args_list[1].args[1]


class _FakeArmBackend:
    def __init__(self, qpos: np.ndarray) -> None:
        self.current_qpos = np.asarray(qpos, dtype=float).reshape(6)
        self.commands = []

    def refresh_state_from_hardware(self) -> dict[str, np.ndarray]:
        return {"hardware_qpos": self.current_qpos.copy()}

    def get_state(self) -> dict[str, np.ndarray]:
        return {"current_qpos": self.current_qpos.copy()}

    def send_command(self, command) -> None:
        self.commands.append(command.copy())
        self.current_qpos = np.asarray(command.value, dtype=float).reshape(6)


class _RefreshFailureBackend(_FakeArmBackend):
    def refresh_state_from_hardware(self) -> dict[str, np.ndarray]:
        raise RuntimeError("hardware refresh failed")

    def get_state(self) -> dict[str, np.ndarray]:
        return {
            "current_qpos": self.current_qpos.copy(),
            "last_target_qpos": self.current_qpos.copy(),
        }


class _FakeHandBackend:
    def __init__(
        self,
        measured: float,
        *,
        apply_target_to_measurement: bool = True,
        send_status: str = "sent",
        recover_after_reconnect: bool = False,
    ) -> None:
        self.measured = float(measured)
        self.timestamp = time.time()
        self.commanded = None
        self.apply_target_to_measurement = bool(apply_target_to_measurement)
        self.send_status = str(send_status)
        self.recover_after_reconnect = bool(recover_after_reconnect)
        self.reconnected = False
        self.reconnect_calls = []
        self.force_sent = []
        self.commands = []

    def get_state(self) -> dict[str, object]:
        return {
            "gripper_measured_scalar": self.measured,
            "gripper_commanded_scalar": self.commanded,
            "gripper_last_send_status": self.send_status,
            "joint_timestamp": self.timestamp,
        }

    def send_command(self, command) -> bool:
        self.commands.append(command.copy())
        target = float(np.asarray(command.value, dtype=float).reshape(-1)[0])
        if self.send_status not in {"sent", "skipped_threshold", "rate_limited"}:
            return False
        self.commanded = target
        if self.apply_target_to_measurement:
            self.measured = target
        self.timestamp = time.time()
        return self.send_status == "sent"

    def force_reconnect_gripper(self, reason: str) -> bool:
        self.reconnect_calls.append(str(reason))
        self.reconnected = True
        return True

    def send_fixed_joint_pos(self, target: float) -> bool:
        value = float(target)
        self.force_sent.append(value)
        self.commanded = value
        self.send_status = "sent"
        if self.recover_after_reconnect:
            self.measured = value
        self.timestamp = time.time()
        return True


class DeployPolicyQposInitTest(unittest.TestCase):
    def test_fresh_hardware_snapshot_avoids_redundant_sdk_refresh(self) -> None:
        class _FreshCachedBackend:
            def get_state(self):
                return {
                    "hardware_qpos": np.arange(6, dtype=float),
                    "measured_qpos": np.arange(6, dtype=float),
                    "measured_timestamp_monotonic": time.monotonic(),
                    "hardware_measurement_valid": True,
                    "hardware_read_success": True,
                    "hardware_measurement_sequence": 7,
                }

            def refresh_state_from_hardware(self):
                raise AssertionError("fresh cached qpos should avoid a second SDK read")

        current = _current_qpos_by_side(
            {"left": _FreshCachedBackend()},
            ("left",),
        )

        np.testing.assert_allclose(current["left"], np.arange(6, dtype=float))

    def test_qpos_init_planning_force_window_rearms_before_return(self) -> None:
        events = []

        class _Controller:
            def begin(self, *, arm_backends, now):
                events.append(("begin", arm_backends, now))
                return "planned"

        class _Monitor:
            def suspend_stale_watchdog_for_non_motion(self):
                events.append(("suspend",))
                return 9

            def resume_stale_watchdog_after_non_motion(self, token):
                events.append(("resume", token))
                return {"last_reading_by_side": {"left": {}, "right": {}}}

        message = _begin_policy_qpos_init_with_force_guard(
            _Controller(),
            monitor=_Monitor(),
            arm_backends={"left": object(), "right": object()},
            now=3.0,
        )

        self.assertEqual(message, "planned")
        self.assertEqual([event[0] for event in events], ["suspend", "begin", "resume"])

    def test_qpos_init_planning_force_window_rearms_after_planning_error(self) -> None:
        events = []

        class _Controller:
            def begin(self, *, arm_backends, now):
                del arm_backends, now
                events.append("begin")
                raise RuntimeError("planning exploded")

        class _Monitor:
            def suspend_stale_watchdog_for_non_motion(self):
                events.append("suspend")
                return 4

            def resume_stale_watchdog_after_non_motion(self, token):
                events.append(("resume", token))
                return {"last_reading_by_side": {"left": {}, "right": {}}}

        with self.assertRaisesRegex(RuntimeError, "planning exploded"):
            _begin_policy_qpos_init_with_force_guard(
                _Controller(),
                monitor=_Monitor(),
                arm_backends={"left": object(), "right": object()},
                now=3.0,
            )

        self.assertEqual(events, ["suspend", "begin", ("resume", 4)])

    def test_hardware_refresh_failure_does_not_fall_back_to_cached_target(self) -> None:
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=0,
            action=np.zeros(ACTION_DIM, dtype=np.float32),
            target_qpos_by_side={"right": np.zeros(6)},
        )
        backend = _RefreshFailureBackend(np.zeros(6))
        controller = DeployPolicyQposInitController(plan, active_sides=("right",))

        message = controller.begin(arm_backends={"right": backend}, now=0.0)

        self.assertEqual(controller.state, "failed")
        self.assertIn("hardware refresh failed", message)
        self.assertEqual(backend.commands, [])

    def test_planned_duration_extends_fixed_qpos_init_timeout_floor(self) -> None:
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=0,
            action=np.zeros(ACTION_DIM, dtype=np.float32),
            target_qpos_by_side={"right": np.zeros(6)},
        )
        backend = _FakeArmBackend(np.zeros(6))
        fake_step = SimpleNamespace(qpos_by_side={"right": np.zeros(6)})
        fake_plan = SimpleNamespace(steps=(fake_step,) * 800)
        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=fake_plan,
            ),
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed"),
        ):
            controller = DeployPolicyQposInitController(
                plan,
                active_sides=("right",),
                settings=DeployPolicyQposInitSettings(
                    move_to_start_timeout_s=30.0,
                    qpos_init_dt_sec=0.05,
                    qpos_init_collision_check=False,
                ),
            )
            controller.begin(arm_backends={"right": backend}, now=0.0)

        self.assertEqual(controller._effective_timeout_s, 50.0)

    def test_qpos_init_completion_defaults_are_0_1_rad(self) -> None:
        self.assertEqual(DeployPolicyQposInitSettings().qpos_reached_tolerance_rad, 0.1)
        self.assertEqual(ReplayExecutionSettings().qpos_reached_tolerance_rad, 0.1)
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.policy_qpos_init_qpos_tolerance, 0.1)
        self.assertTrue(args.policy_qpos_init_hands)
        self.assertEqual(args.policy_qpos_init_hand_tolerance, 0.05)
        self.assertEqual(args.policy_qpos_init_dmtac_tolerance, 0.18)
        self.assertEqual(args.policy_qpos_init_hand_progress_timeout_sec, 2.0)
        self.assertEqual(args.policy_qpos_init_hand_progress_epsilon, 0.02)
        self.assertEqual(args.policy_qpos_init_hand_reconnect_attempts, 1)
        self.assertEqual(args.deploy_fixed_init_qpos_tolerance, 0.1)

    def test_loader_uses_requested_dataset_frame(self) -> None:
        action = np.arange(ACTION_DIM, dtype=np.float32).reshape(1, ACTION_DIM)
        target_qpos = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=float)

        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.resolve_replay_parquet_path",
                return_value=(Path("/tmp/fake_episode.parquet"), {"episode_index": 3}),
            ) as resolve_path,
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_contract_replay_actions_from_parquet",
                return_value=(action, {}, None, False),
            ) as load_contract_actions,
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_arm_joint_targets_from_parquet",
                return_value={"right": target_qpos},
            ) as load_qpos_targets,
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_arm_observation_qpos_from_parquet",
                return_value={},
            ) as load_observed_qpos,
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_arm_qpos_audit_from_parquet",
                return_value={},
            ) as load_qpos_audit,
        ):
            plan = load_deploy_policy_qpos_init_plan(
                dataset_root="/tmp/dataset",
                parquet_path=None,
                episode_index=3,
                frame_index=20,
                action_column="action",
                active_sides=("right",),
                qpos_init_source="replay_qpos",
            )

        resolve_path.assert_called_once()
        load_contract_actions.assert_called_once_with(
            Path("/tmp/fake_episode.parquet"),
            resolved_metadata={"episode_index": 3},
            action_column="action",
            start_index=20,
            end_index=21,
        )
        load_qpos_targets.assert_called_once_with(
            Path("/tmp/fake_episode.parquet"),
            active_sides=("right",),
            start_index=20,
            end_index=21,
            required=False,
        )
        self.assertEqual(load_observed_qpos.call_count, 2)
        load_qpos_audit.assert_called_once_with(
            Path("/tmp/fake_episode.parquet"),
            active_sides=("right",),
            start_index=20,
            end_index=21,
        )
        self.assertEqual(plan.frame_index, 20)
        self.assertEqual(plan.source_episode_index, 3)
        np.testing.assert_allclose(plan.target_qpos_by_side["right"], target_qpos[0])
        np.testing.assert_allclose(plan.action, action[0])

    def test_loader_bridges_v22_compact_action_before_staging_qpos_init(self) -> None:
        legacy_action = np.arange(ACTION_DIM, dtype=np.float32).reshape(1, ACTION_DIM)
        target_qpos = np.zeros((1, 6), dtype=float)
        layout_hash = "sha256:typed-qpos-init-test"
        resolved_metadata = {
            "episode_index": 0,
            "dataset_info": {
                "dexteleop_dataset_schema": "dexteleop_dataset_v2.2",
            },
        }

        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.resolve_replay_parquet_path",
                return_value=(
                    Path("/tmp/v22_episode.parquet"),
                    resolved_metadata,
                ),
            ),
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_contract_replay_actions_from_parquet",
                return_value=(
                    legacy_action,
                    {"bridge": "tested", "policy_io_layout_hash": layout_hash},
                    None,
                    True,
                ),
            ) as load_contract_actions,
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_arm_joint_targets_from_parquet",
                return_value={"right": target_qpos},
            ),
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_arm_observation_qpos_from_parquet",
                return_value={},
            ),
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_arm_qpos_audit_from_parquet",
                return_value={},
            ),
        ):
            plan = load_deploy_policy_qpos_init_plan(
                dataset_root="/tmp/v22_dataset",
                parquet_path=None,
                episode_index=0,
                frame_index=0,
                action_column="action",
                active_sides=("right",),
                qpos_init_source="replay_qpos",
            )

        load_contract_actions.assert_called_once_with(
            Path("/tmp/v22_episode.parquet"),
            resolved_metadata=resolved_metadata,
            action_column="action",
            start_index=0,
            end_index=1,
        )
        np.testing.assert_allclose(plan.action, legacy_action[0])
        self.assertEqual(plan.metadata["action_encoding"], "typed_layout_to_legacy_q6_bridge")
        self.assertEqual(plan.metadata["policy_io_layout_hash"], layout_hash)

    def test_controller_sends_replay_qpos_init_source_and_releases(self) -> None:
        target = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=5,
            action=np.ones(ACTION_DIM, dtype=np.float32),
            target_qpos_by_side={"right": target},
            target_sources_by_side={"right": "replay_qpos"},
        )
        backend = _FakeArmBackend(np.zeros(6, dtype=float))
        fake_plan = SimpleNamespace(steps=(SimpleNamespace(qpos_by_side={"right": target}),))

        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=fake_plan,
            ) as build_plan,
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed") as require_completed,
        ):
            controller = DeployPolicyQposInitController(
                plan,
                active_sides=("right",),
                settings=DeployPolicyQposInitSettings(
                    qpos_init_collision_check=False,
                    qpos_init_max_angvel_rad_s=(0.2, 0.2, 0.2, 0.3, 0.3, 0.3),
                ),
            )
            message = controller.begin(arm_backends={"right": backend}, now=1.0)
            result = controller.step(now=1.0, send_actions=True, arm_backends={"right": backend})

        self.assertIn("frame=5", message)
        build_plan.assert_called_once()
        self.assertEqual(
            build_plan.call_args.kwargs["qpos_init_max_angvel_rad_s"],
            (0.2, 0.2, 0.2, 0.3, 0.3, 0.3),
        )
        require_completed.assert_called_once_with(fake_plan)
        self.assertTrue(result.completed)
        self.assertIsNone(result.last_action)
        self.assertTrue(result.requires_fresh_observation)
        self.assertEqual(controller.state, "released")
        self.assertEqual(len(backend.commands), 1)
        self.assertEqual(backend.commands[0].metadata["source"], "replay_qpos_init")
        np.testing.assert_allclose(backend.commands[0].value, target)

    def test_qpos_init_completion_cuts_pose_anchor_and_skips_current_noop(self) -> None:
        initializer = StartupInitializer(
            duration_s=5.0,
            source="current",
            target_action=None,
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
        initializer.reset()

        last_action, current_satisfied = _finish_policy_qpos_init_reference_cut(
            initializer
        )

        self.assertTrue(np.all(np.isnan(last_action)))
        self.assertTrue(current_satisfied)
        self.assertTrue(initializer.completed)
        self.assertIsNone(initializer.started_at)

    def test_no_init_qpos_current_start_skips_redundant_cartesian_init(self) -> None:
        initializer = StartupInitializer(
            duration_s=5.0,
            source="current",
            target_action=None,
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )

        current_satisfied = _reset_startup_initializer(
            initializer,
            current_pose_start_allowed=True,
        )

        self.assertTrue(current_satisfied)
        self.assertTrue(initializer.completed)
        self.assertIsNone(initializer.started_at)

    def test_no_init_qpos_does_not_skip_explicit_startup_target(self) -> None:
        initializer = StartupInitializer(
            duration_s=5.0,
            source="dataset:/tmp/explicit_target",
            target_action=np.zeros(ACTION_DIM, dtype=np.float32),
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )

        current_satisfied = _reset_startup_initializer(
            initializer,
            current_pose_start_allowed=True,
        )

        self.assertFalse(current_satisfied)
        self.assertFalse(initializer.completed)
        self.assertIsNone(initializer.started_at)

    def test_controller_initializes_dmtac_target_before_release(self) -> None:
        target = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[ACTION_HAND_SLICES["right"]][0] = 0.42
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=5,
            action=action,
            target_qpos_by_side={"right": target},
            target_sources_by_side={"right": "replay_qpos"},
        )
        arm_backend = _FakeArmBackend(np.zeros(6, dtype=float))
        hand_backend = _FakeHandBackend(0.0)
        now = time.time()
        fake_plan = SimpleNamespace(
            steps=(SimpleNamespace(qpos_by_side={"right": target}),)
        )

        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=fake_plan,
            ),
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed"),
        ):
            controller = DeployPolicyQposInitController(
                plan,
                active_sides=("right",),
                hand_action_specs_by_side={
                    "right": HandActionSpec.dmtac_gripper("right").to_metadata()
                },
                settings=DeployPolicyQposInitSettings(
                    qpos_init_collision_check=False,
                ),
            )
            controller.begin(arm_backends={"right": arm_backend}, now=now)
            first = controller.step(
                now=now,
                send_actions=True,
                arm_backends={"right": arm_backend},
                hand_backends={"right": hand_backend},
            )
            second = controller.step(
                now=now + 0.3,
                send_actions=True,
                arm_backends={"right": arm_backend},
                hand_backends={"right": hand_backend},
            )

        self.assertFalse(first.completed)
        self.assertTrue(second.completed)
        self.assertEqual(controller.state, "released")
        self.assertEqual(len(arm_backend.commands), 1)
        self.assertEqual(len(hand_backend.commands), 1)
        self.assertEqual(
            hand_backend.commands[0].metadata["source"],
            "policy_qpos_init_hand_final_hold",
        )
        np.testing.assert_allclose(hand_backend.commands[0].value, [0.42])

    def test_dmtac_init_reconnects_and_force_resends_when_measurement_stalls(self) -> None:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=0,
            action=action,
            target_qpos_by_side={"right": np.zeros(6)},
        )
        arm_backend = _FakeArmBackend(np.zeros(6))
        hand_backend = _FakeHandBackend(
            0.269,
            apply_target_to_measurement=False,
            recover_after_reconnect=True,
        )
        now = time.time()
        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            hand_action_specs_by_side={
                "right": HandActionSpec.dmtac_gripper("right")
            },
            settings=DeployPolicyQposInitSettings(
                qpos_init_collision_check=False,
            ),
        )
        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=SimpleNamespace(steps=()),
            ),
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed"),
        ):
            controller.begin(arm_backends={"right": arm_backend}, now=now)

        first = controller.step(
            now=now,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )
        hand_backend.timestamp = now + 2.0
        second = controller.step(
            now=now + 2.1,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )
        hand_backend.timestamp = now + 2.2
        third = controller.step(
            now=now + 2.3,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )

        self.assertFalse(first.completed)
        self.assertFalse(second.completed)
        self.assertIn("reconnected gripper", second.message)
        self.assertTrue(third.completed)
        self.assertEqual(controller.state, "released")
        self.assertEqual(hand_backend.force_sent, [0.0])
        self.assertEqual(
            hand_backend.reconnect_calls,
            ["policy_qpos_init_no_measured_progress"],
        )

    def test_dmtac_init_fails_closed_if_force_resend_still_does_not_move(self) -> None:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=0,
            action=action,
            target_qpos_by_side={"right": np.zeros(6)},
        )
        arm_backend = _FakeArmBackend(np.zeros(6))
        hand_backend = _FakeHandBackend(
            0.269,
            apply_target_to_measurement=False,
        )
        now = time.time()
        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            hand_action_specs_by_side={
                "right": HandActionSpec.dmtac_gripper("right")
            },
            settings=DeployPolicyQposInitSettings(
                qpos_init_collision_check=False,
            ),
        )
        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=SimpleNamespace(steps=()),
            ),
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed"),
        ):
            controller.begin(arm_backends={"right": arm_backend}, now=now)

        controller.step(
            now=now,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )
        hand_backend.timestamp = now + 2.0
        reconnect_result = controller.step(
            now=now + 2.1,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )
        hand_backend.timestamp = now + 4.1
        failed_result = controller.step(
            now=now + 4.2,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )

        self.assertIn("reconnected gripper", reconnect_result.message)
        self.assertFalse(failed_result.completed)
        self.assertIn("failed closed", failed_result.message)
        self.assertEqual(controller.state, "failed")
        self.assertEqual(hand_backend.force_sent, [0.0])

    def test_dmtac_init_requires_fresh_feedback_after_command_attempt(self) -> None:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=0,
            action=action,
            target_qpos_by_side={"right": np.zeros(6)},
        )
        arm_backend = _FakeArmBackend(np.zeros(6))
        hand_backend = _FakeHandBackend(0.0)
        now = time.time()
        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            hand_action_specs_by_side={
                "right": HandActionSpec.dmtac_gripper("right")
            },
            settings=DeployPolicyQposInitSettings(
                qpos_init_collision_check=False,
            ),
        )
        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=SimpleNamespace(steps=()),
            ),
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed"),
        ):
            controller.begin(arm_backends={"right": arm_backend}, now=now)

        controller.step(
            now=now,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )
        hand_backend.timestamp = now - 0.1
        result = controller.step(
            now=now + 0.3,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )

        self.assertFalse(result.completed)
        self.assertEqual(controller.state, "hand_final_hold")
        self.assertIn("feedback_pre_command", controller.status_text)

    def test_dmtac_init_rejects_unconfirmed_command(self) -> None:
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=0,
            action=action,
            target_qpos_by_side={"right": np.zeros(6)},
        )
        arm_backend = _FakeArmBackend(np.zeros(6))
        hand_backend = _FakeHandBackend(0.0, send_status="send_disabled")
        now = time.time()
        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            hand_action_specs_by_side={
                "right": HandActionSpec.dmtac_gripper("right")
            },
            settings=DeployPolicyQposInitSettings(
                qpos_init_collision_check=False,
            ),
        )
        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
                return_value=SimpleNamespace(steps=()),
            ),
            patch("teleop_runtime.deploy.policy_qpos_init._require_plan_completed"),
        ):
            controller.begin(arm_backends={"right": arm_backend}, now=now)

        controller.step(
            now=now,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )
        result = controller.step(
            now=now + 0.3,
            send_actions=True,
            arm_backends={"right": arm_backend},
            hand_backends={"right": hand_backend},
        )

        self.assertFalse(result.completed)
        self.assertEqual(controller.state, "hand_final_hold")
        self.assertIn("command_unconfirmed:send_disabled", controller.status_text)

    def test_qpos_init_completion_preserves_explicit_startup_target(self) -> None:
        initializer = StartupInitializer(
            duration_s=5.0,
            source="dataset:/tmp/explicit_target",
            target_action=np.zeros(ACTION_DIM, dtype=np.float32),
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=True,
        )
        initializer.reset()

        last_action, current_satisfied = _finish_policy_qpos_init_reference_cut(
            initializer
        )

        self.assertTrue(np.all(np.isnan(last_action)))
        self.assertFalse(current_satisfied)
        self.assertFalse(initializer.completed)
        self.assertIsNone(initializer.started_at)

    def test_controller_planning_failure_blocks_policy_without_sending(self) -> None:
        target = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=5,
            action=np.ones(ACTION_DIM, dtype=np.float32),
            target_qpos_by_side={"right": target},
            target_sources_by_side={"right": "replay_qpos"},
        )
        backend = _FakeArmBackend(np.zeros(6, dtype=float))

        with patch(
            "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
            side_effect=RuntimeError("no collision-safe interpolation step was found"),
        ):
            controller = DeployPolicyQposInitController(
                plan,
                active_sides=("right",),
                settings=DeployPolicyQposInitSettings(qpos_init_collision_check=False),
            )
            message = controller.begin(arm_backends={"right": backend}, now=1.0)

        self.assertIn("planning failed before execution", message)
        self.assertEqual(controller.state, "failed")
        self.assertTrue(controller.blocks_policy)
        self.assertEqual(controller.status_text, "policy_qpos_init=failed:g")
        self.assertEqual(backend.commands, [])

    def test_controller_timeout_blocks_policy_without_throwing(self) -> None:
        target = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        plan = DeployPolicyQposInitPlan(
            source_path="/tmp/fake_episode.parquet",
            source_episode_index=0,
            frame_index=5,
            action=np.ones(ACTION_DIM, dtype=np.float32),
            target_qpos_by_side={"right": target},
            target_sources_by_side={"right": "replay_qpos"},
        )
        backend = _FakeArmBackend(np.zeros(6, dtype=float))

        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            settings=DeployPolicyQposInitSettings(move_to_start_timeout_s=0.001),
        )
        controller.state = "final_hold"
        controller._started_monotonic = 0.0
        result = controller.step(now=1.0, send_actions=True, arm_backends={"right": backend})

        self.assertEqual(controller.state, "timeout")
        self.assertTrue(controller.blocks_policy)
        self.assertIn("timed out", result.message)
        self.assertIn("max_err=right:j1", result.message)
        self.assertEqual(controller.status_text, "policy_qpos_init=timeout:g")
        self.assertEqual(backend.commands, [])

    def test_policy_qpos_init_is_mutually_exclusive_with_legacy_fixed_init(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "right",
                "--deploy-fixed-init-dataset",
                "/tmp/fixed",
                "--policy-qpos-init-dataset",
                "/tmp/policy",
            ]
        )
        with self.assertRaisesRegex(ValueError, "Use only one"):
            validate_args(args)

    def test_real_arm_send_requires_dataset_qpos_init_by_default(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "left",
                "right",
                "--left-arm-ip",
                "192.168.101.21",
                "--right-arm-ip",
                "192.168.1.20",
                "--send-actions",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires dataset-frame qpos"):
            validate_args(args)

    def test_real_arm_send_allows_explicit_start_init_qpos_override(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "left",
                "right",
                "--left-arm-ip",
                "192.168.101.21",
                "--right-arm-ip",
                "192.168.1.20",
                "--send-actions",
                "--start-init_qpos",
            ]
        )
        self.assertEqual(validate_args(args), ("left", "right"))

    def test_no_init_qpos_suppresses_profile_dataset_init_and_allows_current_start(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "left",
                "right",
                "--left-arm-ip",
                "192.168.101.21",
                "--right-arm-ip",
                "192.168.1.20",
                "--policy-qpos-init-dataset",
                "/tmp/profile-default-init",
                "--no-init-qpos",
                "--send-actions",
            ]
        )

        self.assertEqual(validate_args(args), ("left", "right"))
        self.assertTrue(args.no_init_qpos)
        self.assertTrue(args.policy_qpos_init_suppressed)
        self.assertIsNone(args.policy_qpos_init_dataset)
        self.assertIsNone(args.policy_qpos_init_parquet)

    def test_legacy_no_init_qpos_spelling_is_accepted(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)

        args = parser.parse_args(["--no--init-qpos"])

        self.assertTrue(args.no_init_qpos)

    def test_no_init_qpos_rejects_explicit_legacy_fixed_trajectory(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "right",
                "--deploy-fixed-init-dataset",
                "/tmp/fixed",
                "--no-init-qpos",
            ]
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_args(args)

    def test_real_arm_send_requires_explicit_execution_branch_in_real_cli(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        argv = [
            "--active-sides",
            "left",
            "right",
            "--left-arm-ip",
            "192.168.101.21",
            "--right-arm-ip",
            "192.168.1.20",
            "--send-actions",
            "--start-init_qpos",
        ]
        args = parser.parse_args(argv)
        with self.assertRaisesRegex(ValueError, "explicit execution branch"):
            validate_args(args, raw_argv=argv)

    def test_position_collision_epsilon_cannot_exceed_minimum_margin(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--position-collision-min-distance-m",
                "0.01",
                "--position-collision-distance-epsilon-m",
                "0.02",
            ]
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            validate_args(args)

    def test_policy_qpos_init_speed_arg_accepts_single_value_alias(self) -> None:
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args(
            [
                "--active-sides",
                "right",
                "--policy-qpos-init-dataset",
                "/tmp/policy",
                "--qpos-init-arm-max-angvel",
                "0.25",
            ]
        )

        self.assertEqual(validate_args(args), ("right",))
        self.assertEqual(args.policy_qpos_init_arm_max_angvel, (0.25,) * 6)


if __name__ == "__main__":
    unittest.main()
