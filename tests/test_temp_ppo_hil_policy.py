from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from robot_motion.deploy import DeployRobotActionAdapter, build_deployment_observation
from sim.mujoco.tasks.cube_lift.action_conditioning import (
    CubeLiftActionConditioner,
)
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco_temp_pipeline.hil_policy import (
    TempPPOHILObservationAdapter,
    _TempPPOAbsolutePoseDecoder,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_PATH = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
)
TASK_CONFIG_35_PATH = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
)


class _Scene:
    arm_specs = {"right": SimpleNamespace(controlled_body="right_tcp")}

    @staticmethod
    def has_body(name: str) -> bool:
        return name in {"right_arm_base_link", "right_tcp"}

    @staticmethod
    def body_pose(name: str) -> np.ndarray:
        del name
        return np.eye(4, dtype=float)

    @staticmethod
    def get_gripper_close(name: str) -> float:
        assert name == "right_gripper"
        return 0.2


class _ObservationAdapter:
    def __init__(self) -> None:
        self.base = SimpleNamespace(last_action=None, set_last_action=self._set_last_action)
        self.last_compact_action = None

    def _set_last_action(self, action: np.ndarray) -> None:
        self.base.last_action = np.asarray(action).copy()

    def set_last_compact_action(self, action: np.ndarray) -> None:
        self.last_compact_action = np.asarray(action).copy()


class _ObservationScene:
    def __init__(self) -> None:
        self.qpos = np.zeros(6, dtype=np.float32)
        self.close = 0.2
        self.cube_pose = np.asarray(
            [0.4, 0.1, 0.03, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    def get_arm_qpos(self, side: str) -> np.ndarray:
        assert side == "right"
        return self.qpos.copy()

    def get_gripper_close(self, name: str) -> float:
        assert name == "right_gripper"
        return self.close

    def freejoint_qpos(self, name: str) -> np.ndarray:
        assert name == "debug_cube_freejoint"
        return self.cube_pose.copy()


class _BaseObservation:
    def __init__(self) -> None:
        self.last_action = None

    def set_last_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action).copy()

    def capture(self, *, step_index: int | None = None):
        return build_deployment_observation(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
            end_effector_sides=("right",),
            step_index=step_index,
            arm_states={
                "right": {
                    "current_qpos": np.zeros(6),
                    "current_tcp_pose": np.eye(4),
                }
            },
            hand_states={"right": {"joint_pos": np.zeros(6)}},
            timestamp=1.0,
        )


def test_temp_ppo_gripper_delta_uses_the_shared_robot_action_carrier() -> None:
    task_config = CubeLiftTaskConfig.load(TASK_CONFIG_PATH)
    task_config = replace(
        task_config,
        scripted_contact_reset=False,
        scripted_contact_action_scale=(1.0,) * 7,
        scripted_contact_action_bias=(0.0,) * 7,
        scripted_contact_nonnegative_lift_action=False,
    )
    adapter = _ObservationAdapter()
    scene = _Scene()
    capture_count = 0

    def capture_scene(capture):
        nonlocal capture_count
        capture_count += 1
        return capture(scene)

    decoder = _TempPPOAbsolutePoseDecoder(
        scene,
        task_config,
        adapter,
        task_initialization_info={
            "scripted_pregrasp_reset": False,
            "scripted_contact_reset": False,
        },
        scene_capture=capture_scene,
    )
    normalized = decoder.decode(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), None)

    assert capture_count == 1
    assert normalized.metadata["policy_hand_control"] == (
        "right_gripper_delta_to_absolute_qpos_v1"
    )
    assert normalized.metadata["end_effector_enabled_sides"] == ("right",)
    np.testing.assert_allclose(normalized.vector[20:26], [0.3, 0, 0, 0, 0, 0])

    observation = build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=True,
        end_effector_sides=("right",),
        step_index=0,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6),
                "current_tcp_pose": np.eye(4),
            }
        },
        hand_states={"right": {"joint_pos": np.zeros(6)}},
        timestamp=1.0,
    )
    result = DeployRobotActionAdapter(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=True,
    ).adapt(normalized, observation)

    assert result.source_validation.accepted
    assert result.robot_action is not None
    assert result.robot_action.active_effectors == ("right_arm", "right_hand")
    np.testing.assert_allclose(
        result.robot_action.hand_qpos_by_side["right"],
        [0.3, 0, 0, 0, 0, 0],
    )
    assert (
        result.robot_action.metadata["hand_action_specs_by_side"]["right"][
            "hand_model"
        ]
        == "dmtac_gripper"
    )


def test_35d_observation_includes_cube_and_deltas_advance_at_policy_cadence() -> None:
    scene = _ObservationScene()
    adapter = TempPPOHILObservationAdapter(
        scene,
        CubeLiftTaskConfig.load(TASK_CONFIG_35_PATH),
        _BaseObservation(),
    )

    first = adapter.capture(step_index=0, commit_policy_history=True)
    np.testing.assert_allclose(first.state_dict["temp_ppo_state"][7:14], 0.0)
    np.testing.assert_allclose(
        first.state_dict["temp_ppo_state"][28:35],
        scene.cube_pose,
    )

    scene.qpos[:] = 0.1
    scene.close = 0.3
    preview = adapter.capture(step_index=1, commit_policy_history=False)
    np.testing.assert_allclose(
        preview.state_dict["temp_ppo_state"][7:13],
        0.1,
    )
    assert preview.state_dict["temp_ppo_state"][13] == np.float32(0.1)
    assert not preview.metadata["policy_observation_history_committed"]

    repeated_preview = adapter.capture(
        step_index=2,
        commit_policy_history=False,
    )
    np.testing.assert_allclose(
        repeated_preview.state_dict["temp_ppo_state"][7:13],
        0.1,
    )

    committed = adapter.capture(step_index=3, commit_policy_history=True)
    np.testing.assert_allclose(
        committed.state_dict["temp_ppo_state"][7:13],
        0.1,
    )
    scene.qpos[:] = 0.2
    next_policy_sample = adapter.capture(
        step_index=4,
        commit_policy_history=True,
    )
    np.testing.assert_allclose(
        next_policy_sample.state_dict["temp_ppo_state"][7:13],
        0.1,
    )


def test_v5_post_contact_action_semantics_are_shared_with_hil_decode() -> None:
    config = CubeLiftTaskConfig.load(TASK_CONFIG_35_PATH)
    conditioner = CubeLiftActionConditioner(config)
    result = None
    for _ in range(config.post_contact_phase_activation_steps):
        result = conditioner.condition(
            np.asarray([1.0, -1.0, 1.0, 0.5, -0.5, 0.25, 0.0]),
            left_finger_contact=True,
            right_finger_contact=True,
            measured_gripper_close=0.6,
            scripted_pregrasp_reset_applied=True,
            scripted_contact_reset_applied=False,
        )

    assert result is not None
    assert result.post_contact_phase_active
    assert result.post_contact_modified
    np.testing.assert_allclose(
        result.values,
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5],
    )
