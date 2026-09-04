from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_motion.deploy.schema import ACTION_DIM
from robot_motion.deploy.async_producer import RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1
from teleop_runtime.deploy import run_policy
from teleop_runtime.deploy.openpi_application import _rm65_bimanual_dmtac_args
from teleop_runtime.deploy.policy_qpos_init import (
    DeployPolicyQposInitController,
    DeployPolicyQposInitPlan,
    DeployPolicyQposInitSettings,
)


class _FakeArmBackend:
    def __init__(self, qpos: np.ndarray) -> None:
        self.qpos = np.asarray(qpos, dtype=float).reshape(6).copy()
        self.commands: list[object] = []

    def get_state(self) -> dict[str, object]:
        return {"current_qpos": self.qpos.copy()}

    def send_command(self, command: object) -> None:
        self.commands.append(command)
        self.qpos = np.asarray(command.value, dtype=float).reshape(6).copy()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser


def test_model_rtc_profile_keeps_qpos_init_and_requires_explicit_dispatch() -> None:
    profile_args = _rm65_bimanual_dmtac_args(rtc_real_shadow=True)
    assert "--send-actions" not in profile_args
    assert "--policy-qpos-init-dataset" in profile_args
    assert "--policy-qpos-init-collision-check" in profile_args

    argv = tuple(
        profile_args
        + [
            "--rtc-profile",
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
            "--policy-artifact-json",
            "/tmp/model-rtc-active-artifact.json",
            "--rtc-real-shadow",
            "--send-actions",
        ]
    )
    args = _parser().parse_args(list(argv))
    active_sides = run_policy.validate_args(args, raw_argv=argv)

    assert active_sides == ("left", "right")
    assert args.rtc_real_shadow is True
    assert args.send_actions is True
    assert args.policy_qpos_init_dataset.name == "bimanual-cube-new-success"
    assert args.policy_qpos_init_collision_check is True
    assert args.arm_control_mode == "impedance"


def test_active_model_rtc_qpos_init_sends_bounded_smooth_steps_then_releases() -> None:
    target = np.full(6, 0.30, dtype=float)
    intermediate = np.full(6, 0.15, dtype=float)
    plan = DeployPolicyQposInitPlan(
        source_path="dataset/frame.parquet",
        source_episode_index=0,
        frame_index=3,
        action=np.zeros(ACTION_DIM, dtype=np.float32),
        target_qpos_by_side={"right": target},
        target_sources_by_side={"right": "replay_qpos"},
    )
    backend = _FakeArmBackend(np.zeros(6, dtype=float))
    fake_motion_plan = SimpleNamespace(
        completed=True,
        steps=(
            SimpleNamespace(qpos_by_side={"right": intermediate}),
            SimpleNamespace(qpos_by_side={"right": target}),
        )
    )

    with patch(
        "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
        return_value=fake_motion_plan,
    ):
        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            settings=DeployPolicyQposInitSettings(
                qpos_init_dt_sec=0.01,
                qpos_init_collision_check=False,
                initialize_hands=False,
            ),
        )
        message = controller.begin(arm_backends={"right": backend}, now=1.0)
        first = controller.step(
            now=1.0,
            send_actions=True,
            arm_backends={"right": backend},
        )
        second = controller.step(
            now=1.02,
            send_actions=True,
            arm_backends={"right": backend},
        )

    assert "frame=3" in message
    assert first.completed is False
    assert second.completed is True
    assert second.requires_fresh_observation is True
    assert controller.state == "released"
    assert len(backend.commands) == 2
    sent_qpos = [np.asarray(command.value, dtype=float) for command in backend.commands]
    np.testing.assert_allclose(sent_qpos[0], intermediate)
    np.testing.assert_allclose(sent_qpos[1], target)
    assert all(command.metadata["source"] == "replay_qpos_init" for command in backend.commands)


def test_passive_model_rtc_shadow_runs_the_same_qpos_plan_without_commands() -> None:
    target = np.full(6, 0.30, dtype=float)
    plan = DeployPolicyQposInitPlan(
        source_path="dataset/frame.parquet",
        source_episode_index=0,
        frame_index=3,
        action=np.zeros(ACTION_DIM, dtype=np.float32),
        target_qpos_by_side={"right": target},
        target_sources_by_side={"right": "replay_qpos"},
    )
    backend = _FakeArmBackend(np.zeros(6, dtype=float))
    fake_motion_plan = SimpleNamespace(
        completed=True,
        steps=(
            SimpleNamespace(qpos_by_side={"right": np.full(6, 0.15)}),
            SimpleNamespace(qpos_by_side={"right": target}),
        )
    )

    with patch(
        "teleop_runtime.deploy.policy_qpos_init.build_rm65_qpos_init_plan",
        return_value=fake_motion_plan,
    ):
        controller = DeployPolicyQposInitController(
            plan,
            active_sides=("right",),
            settings=DeployPolicyQposInitSettings(
                qpos_init_dt_sec=0.01,
                qpos_init_collision_check=False,
                initialize_hands=False,
            ),
        )
        controller.begin(arm_backends={"right": backend}, now=1.0)
        controller.step(
            now=1.0,
            send_actions=False,
            arm_backends={"right": backend},
        )
        result = controller.step(
            now=1.02,
            send_actions=False,
            arm_backends={"right": backend},
        )

    assert result.completed is True
    assert result.requires_fresh_observation is True
    assert controller.state == "released"
    assert backend.commands == []
    np.testing.assert_allclose(backend.qpos, np.zeros(6))
