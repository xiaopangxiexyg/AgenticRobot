from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np

from robot_motion.deploy import DeployRobotActionAdapter, build_deployment_observation
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco_temp_pipeline.hil_policy import _TempPPOAbsolutePoseDecoder


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_PATH = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
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


def test_temp_ppo_gripper_delta_uses_the_shared_robot_action_carrier() -> None:
    task_config = CubeLiftTaskConfig.load(TASK_CONFIG_PATH)
    adapter = _ObservationAdapter()
    decoder = _TempPPOAbsolutePoseDecoder(_Scene(), task_config, adapter)
    normalized = decoder.decode(np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), None)

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
