from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.deploy import (
    POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA,
    DeployRobotActionAdapter,
    build_deployment_observation,
    policy_source_projection_from_metadata,
    pose_matrix_to_xyzquat,
)
from sim.mujoco.tasks.cube_lift.action_conditioning import (
    CubeLiftActionConditioner,
)
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco_temp_pipeline.hil_policy import (
    TempPPOHILObservationAdapter,
    _TempPPOAbsolutePoseDecoder,
    _TempPPOModel,
    validate_temp_ppo_hil_profile,
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
TASK_CONFIG_38_PATH = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_reach_goal_delta_progress_xyz_v3.json"
)
TASK_CONFIG_38_OVERHEAD_PATH = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_reach_goal_delta_overhead_progress_xyz_v4.json"
)
TASK_CONFIG_PICK_XYZ_GRIPPER_PATH = (
    REPO_ROOT
    / "sim"
    / "configs"
    / "tasks"
    / "rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json"
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
    arm_specs = {"right": SimpleNamespace(controlled_body="right_tcp")}

    def __init__(self) -> None:
        self.qpos = np.zeros(6, dtype=np.float32)
        self.close = 0.2
        self.data = SimpleNamespace(
            xpos=np.asarray([[0.3, -0.1, 0.2], [0.3, -0.08, 0.18]], dtype=float),
            xmat=np.tile(np.eye(3, dtype=float).reshape(1, 9), (2, 1)),
        )
        self.cube_pose = np.asarray(
            [0.4, 0.1, 0.03, 1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    @staticmethod
    def body_id(name: str) -> int:
        return {"right_tcp": 0, "right_gripper_grasp_center_frame": 1}[name]

    def get_arm_qpos(self, side: str) -> np.ndarray:
        assert side == "right"
        return self.qpos.copy()

    def get_gripper_close(self, name: str) -> float:
        assert name == "right_gripper"
        return self.close

    def freejoint_qpos(self, name: str) -> np.ndarray:
        assert name == "debug_cube_freejoint"
        return self.cube_pose.copy()

    @staticmethod
    def has_body(name: str) -> bool:
        return name in {"right_arm_base_link", "right_tcp"}

    @staticmethod
    def body_pose(name: str) -> np.ndarray:
        assert name in {"right_arm_base_link", "right_tcp", "right_gripper_grasp_center_frame"}
        pose = np.eye(4, dtype=float)
        if name == "right_tcp":
            pose[:3, 3] = (0.3, -0.1, 0.2)
        elif name == "right_gripper_grasp_center_frame":
            pose[:3, 3] = (0.3, -0.08, 0.18)
        return pose


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
    projection = policy_source_projection_from_metadata(normalized.metadata)
    assert projection is not None
    assert projection.source_semantics == (
        POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA
    )
    assert projection.target_generation == 0
    assert projection.anchor_policy == "previous_policy_target"
    np.testing.assert_allclose(
        projection.anchor_value_by_component["right_arm"],
        pose_matrix_to_xyzquat(decoder._current_tool_pose_arm_base()),
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


@pytest.mark.parametrize("pose_target_anti_windup", (False, True))
def test_goal_delta_observation_is_deployable_and_uses_pose_action_frame(
    pose_target_anti_windup: bool,
) -> None:
    scene = _ObservationScene()
    config = replace(
        CubeLiftTaskConfig.load(TASK_CONFIG_38_PATH),
        pose_target_anti_windup=pose_target_anti_windup,
    )
    adapter = TempPPOHILObservationAdapter(
        scene,
        config,
        _BaseObservation(),
    )
    policy = SimpleNamespace(
        observation_dim=31,
        action_dim=3,
        environment_action_dim=7,
        active_action_indices=(0, 1, 2),
    )

    if pose_target_anti_windup:
        with pytest.raises(ValueError, match="pose_target_anti_windup=True"):
            validate_temp_ppo_hil_profile(config, policy)
        return
    validate_temp_ppo_hil_profile(config, policy)
    observation = adapter.capture(step_index=0)

    grasp_position = scene.body_pose(config.right_grasp_body_name)[:3, 3]
    expected = scene.cube_pose[:3].astype(float, copy=True)
    expected[2] += config.reach_target_height_offset_m
    expected -= grasp_position
    np.testing.assert_allclose(
        observation.state_dict["temp_ppo_state"][28:35],
        scene.cube_pose,
    )
    np.testing.assert_allclose(
        observation.state_dict["reach_target_delta"],
        expected,
        atol=1e-7,
    )
    assert observation.metadata["policy_observation_dim"] == 38


def test_overhead_goal_delta_has_matching_nonprivileged_hil_semantics() -> None:
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform
    from sim.mujoco.tasks.cube_lift.reach_guidance import (
        build_reach_guidance,
        coarse_to_fine_reach_action_scale,
    )

    scene = _ObservationScene()
    config = CubeLiftTaskConfig.load(TASK_CONFIG_38_OVERHEAD_PATH)
    adapter = TempPPOHILObservationAdapter(
        scene,
        config,
        _BaseObservation(),
    )
    observation = adapter.capture(step_index=0)

    base_pose = scene.body_pose("right_arm_base_link")
    grasp_position = scene.body_pose(config.right_grasp_body_name)[:3, 3]
    expected = build_reach_guidance(
        scene.cube_pose[:3],
        grasp_position,
        base_pose[:3, :3],
        target_height_offset_m=config.reach_target_height_offset_m,
        guidance_height_offset_m=config.reach_guidance_height_offset_m,
        guidance_xy_blend_radius_m=(
            config.reach_guidance_xy_blend_radius_m
        ),
    )
    np.testing.assert_allclose(
        observation.state_dict["reach_target_delta"],
        expected.delta_in_base,
        atol=1e-7,
    )
    compact_state = observation.state_dict["temp_ppo_state"]
    expected_scale = coarse_to_fine_reach_action_scale(
        compact_state[35:38],
        minimum_scale=config.reach_action_min_scale,
        slow_radius_m=config.reach_action_slow_radius_m,
    )
    public_action = build_policy_action_transform(config).transform(
        np.ones(3, dtype=np.float32),
        observation=compact_state,
    )
    np.testing.assert_allclose(public_action[:3], expected_scale)
    np.testing.assert_allclose(public_action[3:], 0.0)
    assert expected.overhead_blend == 1.0
    assert config.reach_guidance_enabled is True
    assert config.reach_action_distance_scaling_enabled is True


def test_temp_ppo_model_decodes_latent_action_from_the_same_compact_state() -> None:
    import torch

    from robot_motion.deploy.observation_view import PolicyObservationView
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform
    from sim.mujoco.tasks.cube_lift.reach_guidance import (
        coarse_to_fine_reach_action_scale,
    )

    class OnesPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def predict_action(self, observation, *, deterministic):
            assert deterministic is True
            return torch.ones(
                (observation.shape[0], 3),
                dtype=observation.dtype,
                device=observation.device,
            ), None

    class IdentityNormalizer:
        def __init__(self) -> None:
            self.seen = None

        def normalize(self, values):
            self.seen = np.asarray(values, dtype=np.float32).copy()
            return self.seen

    config = CubeLiftTaskConfig.load(TASK_CONFIG_38_OVERHEAD_PATH)
    compact_state = np.zeros(38, dtype=np.float32)
    compact_state[35:38] = (0.04, 0.0, 0.0)
    normalizer = IdentityNormalizer()
    model = _TempPPOModel(
        OnesPolicy(),
        normalizer,
        build_policy_action_transform(config),
        task_config=config,
    )
    view = PolicyObservationView(
        contract_id="test.temp_ppo.compact_state.v1",
        contract_hash="test-hash",
        payload={"state": compact_state},
    )

    public_action = model.predict(view)

    expected_scale = coarse_to_fine_reach_action_scale(
        compact_state[35:38],
        minimum_scale=config.reach_action_min_scale,
        slow_radius_m=config.reach_action_slow_radius_m,
    )
    assert normalizer.seen.shape == (31,)
    np.testing.assert_allclose(normalizer.seen[-3:], compact_state[35:38])
    np.testing.assert_allclose(public_action[:3], expected_scale)
    np.testing.assert_allclose(public_action[3:], 0.0)


def test_pick_goal_delta_xyz_gripper_profile_uses_generic_hil_projection() -> None:
    import torch

    from robot_motion.deploy.observation_view import PolicyObservationView
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform

    latent = np.asarray([0.25, -0.5, 0.75, -1.0], dtype=np.float32)

    class FixedPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def predict_action(self, observation, *, deterministic):
            assert deterministic is True
            assert observation.shape == (1, 31)
            return torch.as_tensor(
                latent[None, :],
                dtype=observation.dtype,
                device=observation.device,
            ), None

    class IdentityNormalizer:
        def normalize(self, values):
            values = np.asarray(values, dtype=np.float32)
            assert values.shape == (31,)
            return values

    config = replace(
        CubeLiftTaskConfig.load(TASK_CONFIG_PICK_XYZ_GRIPPER_PATH),
        pose_target_anti_windup=False,
    )
    policy_contract = SimpleNamespace(
        observation_dim=31,
        action_dim=4,
        environment_action_dim=7,
        active_action_indices=(0, 1, 2, 6),
    )
    validate_temp_ppo_hil_profile(config, policy_contract)

    model = _TempPPOModel(
        FixedPolicy(),
        IdentityNormalizer(),
        build_policy_action_transform(config),
        task_config=config,
    )
    view = PolicyObservationView(
        contract_id="test.temp_ppo.pick_xyz_gripper.v1",
        contract_hash="test-hash",
        payload={"state": np.zeros(38, dtype=np.float32)},
    )

    public_action = model.predict(view)

    np.testing.assert_allclose(
        public_action,
        [0.25, -0.5, 0.75, 0.0, 0.0, 0.0, -1.0],
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
