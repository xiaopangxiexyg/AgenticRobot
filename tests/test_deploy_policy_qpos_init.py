from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from dataset_contracts.v22 import EndEffectorSpec, PolicyIOLayout
from teleop_runtime.deploy.policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitPlan,
    DeployPolicyQposInitSettings,
    load_deploy_policy_qpos_init_plan,
)
from teleop_runtime.replay.executor import ReplayExecutionSettings
from teleop_runtime.deploy.run_policy import add_args, validate_args
from teleop_runtime.deploy.schema import ACTION_DIM


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


class DeployPolicyQposInitTest(unittest.TestCase):
    def test_qpos_init_completion_defaults_are_0_1_rad(self) -> None:
        self.assertEqual(DeployPolicyQposInitSettings().qpos_reached_tolerance_rad, 0.1)
        self.assertEqual(ReplayExecutionSettings().qpos_reached_tolerance_rad, 0.1)
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args([])
        self.assertEqual(args.policy_qpos_init_qpos_tolerance, 0.1)
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
                "teleop_runtime.deploy.policy_qpos_init.load_replay_actions_from_parquet",
                return_value=action,
            ) as load_actions,
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
        load_actions.assert_called_once_with(
            Path("/tmp/fake_episode.parquet"),
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
        layout = PolicyIOLayout(
            name="policy_qpos_init_dual_dmtac",
            end_effectors={
                side: EndEffectorSpec.continuous(
                    side,
                    adapter="dmtac_gripper_v1",
                    target_dim=1,
                    target_names=("gripper_close",),
                    target_legacy_q6_slot_indices=(0,),
                )
                for side in ("left", "right")
            },
        )
        legacy_action = np.arange(ACTION_DIM, dtype=np.float32).reshape(1, ACTION_DIM)
        target_qpos = np.zeros((1, 6), dtype=float)

        with (
            patch(
                "teleop_runtime.deploy.policy_qpos_init.resolve_replay_parquet_path",
                return_value=(
                    Path("/tmp/v22_episode.parquet"),
                    {
                        "episode_index": 0,
                        "dataset_info": {
                            "dexteleop_dataset_schema": "dexteleop_dataset_v2.2",
                        },
                    },
                ),
            ),
            patch(
                "teleop_runtime.deploy.policy_qpos_init.policy_io_layout_from_metadata",
                return_value=layout,
            ),
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_v22_replay_actions_from_parquet",
                return_value=(legacy_action, {"bridge": "tested"}),
            ) as load_v22_actions,
            patch(
                "teleop_runtime.deploy.policy_qpos_init.load_replay_actions_from_parquet",
            ) as load_legacy_actions,
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

        load_v22_actions.assert_called_once_with(
            Path("/tmp/v22_episode.parquet"),
            layout=layout,
            start_index=0,
            end_index=1,
        )
        load_legacy_actions.assert_not_called()
        np.testing.assert_allclose(plan.action, legacy_action[0])
        self.assertEqual(plan.metadata["action_encoding"], "v22_layout_to_legacy_q6_bridge")
        self.assertEqual(plan.metadata["policy_io_layout_hash"], layout.layout_hash)

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
        self.assertEqual(controller.state, "released")
        self.assertEqual(len(backend.commands), 1)
        self.assertEqual(backend.commands[0].metadata["source"], "replay_qpos_init")
        np.testing.assert_allclose(backend.commands[0].value, target)

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
