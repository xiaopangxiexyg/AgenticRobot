from pathlib import Path

import numpy as np
import pytest


TASK_CONFIG = Path("sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json")
DEFAULT_PHYSICAL_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_v2.json"
)
CONTACT_CURRICULUM_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v1.json"
)
CONTACT_CURRICULUM_V2_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v2.json"
)
CONTACT_CURRICULUM_V3_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
)
PRIVILEGED_EXPERT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_v1.json"
)
NORMAL_RGB_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v1.json"
)
PRIVILEGED_EXPERT_RGB_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v1.json"
)
NORMAL_RGB_48_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v2.json"
)
NORMAL_RGB_MOTION_TRACKING_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json"
)
NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json"
)
PRIVILEGED_EXPERT_RGB_48_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v2.json"
)
PRIVILEGED_NEAR_LIFT_V2_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_grasp_lift_near_v2.json"
)
PRIVILEGED_GENERIC_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_grasp_lift_near_v7_generic_gripper_v1.json"
)
PRIVILEGED_CONTACT_GATED_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_grasp_lift_near_v8_contact_gated_v1.json"
)
PRIVILEGED_CONTACT_PROGRESS_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_grasp_lift_near_v9_contact_progress_v1.json"
)
PRIVILEGED_CONTACT_BALANCED_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_grasp_lift_near_v10_balanced_v1.json"
)
PRIVILEGED_SECURE_LIFT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_grasp_lift_near_v11_secure_lift_v1.json"
)
NORMAL_PREGRASP_BRIDGE_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v1.json"
)
NORMAL_PREGRASP_BRIDGE_V2_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v2.json"
)
NORMAL_PREGRASP_BRIDGE_V3_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v3.json"
)
NORMAL_PREGRASP_BRIDGE_V4_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v4.json"
)
NORMAL_PREGRASP_BRIDGE_V5_CUBE_POSE_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
)
NORMAL_FULL_GRASP_STAGE1_V4_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_normal_full_grasp_stage1_v4.json"
)
REACH_POSE_XYZ_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_pose_xyz_v1.json"
)
REACH_GOAL_DELTA_XYZ_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_goal_delta_xyz_v2.json"
)
REACH_GOAL_DELTA_PROGRESS_XYZ_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_goal_delta_progress_xyz_v3.json"
)
REACH_GOAL_DELTA_OVERHEAD_PROGRESS_XYZ_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_reach_goal_delta_overhead_progress_xyz_v4.json"
)
CONTACT_POSE_XYZ_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_contact_acquire_pose_xyz_gripper_v1.json"
)
CONTACT_GOAL_DELTA_OVERHEAD_XYZ_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_contact_acquire_goal_delta_overhead_xyz_gripper_v2.json"
)
CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_contact_acquire_goal_delta_overhead_xyz_gripper_v3.json"
)
CONTACT_DESCEND_CLOSE_POSE_Z_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_contact_descend_close_pose_z_gripper_v1.json"
)
LIFT_HOLD_POSE_Z_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_hold_pose_z_v1.json"
)
PICK_POSE_XYZ_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_pick_pose_xyz_gripper_v1.json"
)
REACH_FIXED_POSE_XYZ_PROPRIO_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_reach_fixed_pose_xyz_proprio_v1.json"
)
CONTACT_FIXED_POSE_XYZ_GRIPPER_PROPRIO_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_contact_acquire_fixed_pose_xyz_gripper_proprio_v1.json"
)
LIFT_SHORT_POSE_Z_PROPRIO_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_short_pose_z_proprio_v1.json"
)
LIFT_SHORT_POSITIVE_Z_PROPRIO_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_lift_short_positive_z_proprio_v2.json"
)
PICK_POSE_XYZ_GRIPPER_ESTIMATOR_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_pick_pose_xyz_gripper_estimator_v1.json"
)
PICK_GOAL_DELTA_XYZ_GRIPPER_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_pick_goal_delta_xyz_gripper_v2.json"
)
SAFE_HOVER_GOAL_DELTA_XYZ_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_reach_safe_hover_goal_delta_xyz_v1.json"
)
SAFE_PREGRASP_GOAL_DELTA_XYZ_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_reach_safe_pregrasp_goal_delta_xyz_v1.json"
)
CONTACT_QUALITY_GOAL_DELTA_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_cube_contact_quality_goal_delta_xyz_gripper_v1.json"
)
COLLISION_ROBUST_PICK_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_pick_goal_delta_xyz_gripper_v3.json"
)
COLLISION_UNIFORM_LIGHT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_pick_collision_uniform_light_v1.json"
)
COLLISION_CATEGORY_LIGHT_TASK_CONFIG = Path(
    "sim/configs/tasks/rm65_right_cube_pick_collision_category_light_v1.json"
)
TALL_BLOCK_PICK_TASK_CONFIG = Path(
    "sim/configs/tasks/"
    "rm65_right_tall_block_pick_goal_delta_xyz_gripper_v1.json"
)


def _require_mujoco_runtime() -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("gymnasium")


@pytest.mark.optional
def test_temp_training_defaults_to_fully_actuated_task() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.config import PPOConfig
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert args.task_config == DEFAULT_PHYSICAL_TASK_CONFIG
    assert args.log_interval_updates == 1
    assert args.print_full_metrics is False
    assert args.num_envs == 4
    assert args.critic_observation_mode == "actor"
    assert args.critic_backbone_mode == "shared_actor"
    assert args.value_clip_coef == pytest.approx(0.0)
    assert args.actor_reference_mse_coef == pytest.approx(0.0)
    assert args.goal_delta_bootstrap_transitions == 0
    legacy = PPOConfig.from_dict({})
    assert legacy.critic_backbone_mode == "shared_actor"
    assert legacy.value_clip_coef == pytest.approx(0.0)
    assert legacy.actor_reference_mse_coef == pytest.approx(0.0)
    privileged = PPOConfig(
        critic_observation_mode="privileged_training_state"
    )
    assert legacy.num_envs == 4
    assert privileged.critic_backbone_mode == "separate"


@pytest.mark.optional
def test_goal_delta_actor_bootstrap_is_optional_and_nonprivileged() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_policy_action_transform,
    )
    from sim.mujoco_temp_pipeline.bootstrap import (
        GoalDeltaBootstrapDataset,
        fit_goal_delta_actor_bootstrap,
        goal_delta_teacher_action,
        validate_goal_delta_bootstrap_task,
    )
    from sim.mujoco_temp_pipeline.config import (
        GoalDeltaActorBootstrapConfig,
    )
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    disabled = GoalDeltaActorBootstrapConfig()
    assert disabled.enabled is False
    assert disabled.to_dict()["enabled"] is False
    with pytest.raises(ValueError, match="non-negative"):
        GoalDeltaActorBootstrapConfig(transitions=-1)
    with pytest.raises(ValueError, match="finite and positive"):
        GoalDeltaActorBootstrapConfig(lift_step_m=0.0)

    task = CubeLiftTaskConfig.load(REACH_GOAL_DELTA_PROGRESS_XYZ_TASK_CONFIG)
    validate_goal_delta_bootstrap_task(task)
    observation = np.zeros((2, task.observation_dim), dtype=np.float32)
    observation[:, 35:38] = np.asarray(
        ([0.2, -0.2, 0.001], [-0.001, 0.002, -0.2]),
        dtype=np.float32,
    )
    teacher = goal_delta_teacher_action(task, observation)
    expected = np.clip(
        observation[:, 35:38]
        / np.asarray(task.right_tcp_delta_m, dtype=np.float32),
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(teacher, expected)

    incompatible = CubeLiftTaskConfig.load(DEFAULT_PHYSICAL_TASK_CONFIG)
    with pytest.raises(ValueError, match="policy_observation_fields"):
        validate_goal_delta_bootstrap_task(incompatible)

    contact = CubeLiftTaskConfig.load(
        CONTACT_GOAL_DELTA_OVERHEAD_XYZ_GRIPPER_TASK_CONFIG
    )
    validate_goal_delta_bootstrap_task(contact)
    contact_observation = np.zeros(
        (3, contact.observation_dim),
        dtype=np.float32,
    )
    contact_observation[:, 35:38] = np.asarray(
        ((0.04, 0.0, 0.0), (0.03, 0.0, 0.01), (0.04, 0.0, 0.0)),
        dtype=np.float32,
    )
    contact_observation[2, 6] = 0.2
    contact_teacher = goal_delta_teacher_action(
        contact,
        contact_observation,
    )
    assert contact_teacher.shape == (3, 4)
    np.testing.assert_allclose(contact_teacher[:, 3], (0.0, 1.0, 1.0))
    contact_public = build_policy_action_transform(contact).transform(
        contact_teacher,
        observation=contact_observation,
    )
    assert float(np.max(np.abs(contact_public[1:, :3]))) <= (
        contact.contact_close_translation_scale + 1e-6
    )
    np.testing.assert_allclose(contact_public[:, 6], (0.0, 1.0, 1.0))

    pick = CubeLiftTaskConfig.load(
        PICK_GOAL_DELTA_XYZ_GRIPPER_TASK_CONFIG
    )
    validate_goal_delta_bootstrap_task(pick)
    pick_observation = contact_observation.copy()
    pick_observation[2, 6] = 0.6
    with pytest.raises(ValueError, match="explicit optional teacher lift phase"):
        goal_delta_teacher_action(pick, pick_observation)
    pick_teacher = goal_delta_teacher_action(
        pick,
        pick_observation,
        lift_step_m=0.004,
        pick_lift_active=(False, False, True),
    )
    pick_public = build_policy_action_transform(pick).transform(
        pick_teacher,
        observation=pick_observation,
    )
    assert pick_public[2, 2] == pytest.approx(
        0.004 / pick.right_tcp_delta_m[2]
    )
    assert pick_public[2, 2] > 0.0
    assert pick_public[2, 6] == pytest.approx(0.0)
    pick_observation[2, 27] = 1.0
    settling_teacher = goal_delta_teacher_action(
        pick,
        pick_observation,
        lift_step_m=0.004,
        pick_lift_active=(False, False, True),
    )
    settling_public = build_policy_action_transform(pick).transform(
        settling_teacher,
        observation=pick_observation,
    )
    assert settling_public[2, 2] == pytest.approx(0.0)
    assert settling_public[2, 6] == pytest.approx(0.0)

    anisotropic_contact = CubeLiftTaskConfig.load(
        CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG
    )
    anisotropic_teacher = goal_delta_teacher_action(
        anisotropic_contact,
        contact_observation,
    )
    np.testing.assert_allclose(anisotropic_teacher, contact_teacher)
    anisotropic_public = build_policy_action_transform(
        anisotropic_contact
    ).transform(
        anisotropic_teacher,
        observation=contact_observation,
    )
    np.testing.assert_allclose(
        anisotropic_public[1, :2],
        contact_public[1, :2],
    )
    assert anisotropic_public[1, 2] == pytest.approx(
        contact_public[1, 2] * 0.25
    )
    np.testing.assert_allclose(
        anisotropic_public[:, 6],
        contact_public[:, 6],
    )

    fit_config = GoalDeltaActorBootstrapConfig(
        transitions=128,
        epochs=20,
        minibatch_size=64,
        learning_rate=1e-2,
        target_mse=1e-8,
    )
    actor_observations = np.random.default_rng(4).normal(
        size=(128, 31)
    ).astype(np.float32)
    target_actions = np.tile(
        np.asarray((0.4, -0.2, 0.1), dtype=np.float32),
        (128, 1),
    )
    policy = ActorCritic(
        observation_dim=31,
        action_dim=3,
        hidden_dim=16,
        environment_action_dim=7,
        active_action_indices=(0, 1, 2),
        trainable_log_std=False,
    )
    normalizer, result = fit_goal_delta_actor_bootstrap(
        policy,
        GoalDeltaBootstrapDataset(
            actor_observations=actor_observations,
            target_actions=target_actions,
            collection_metadata={"actor_uses_privileged": False},
        ),
        fit_config,
        device=torch.device("cpu"),
    )
    assert normalizer.mean.shape == (31,)
    assert result["final_mse"] < result["initial_mse"]
    assert result["critic_initialized_from_teacher"] is False


def test_all_cube_task_profiles_pin_a_loadable_base_contract() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    paths = sorted(Path("sim/configs/tasks").glob("rm65_right_cube*.json"))
    assert paths
    for path in paths:
        config = CubeLiftTaskConfig.load(path)
        contract = config.base_contract()
        assert contract.name


def test_pose_task_suite_decouples_latent_actions_from_public_pose_action() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_learning_metadata,
        build_observation_contract,
        build_policy_action_transform,
    )

    expected = {
        REACH_POSE_XYZ_TASK_CONFIG: ("reach", 3, (0, 1, 2)),
        CONTACT_POSE_XYZ_GRIPPER_TASK_CONFIG: (
            "contact",
            4,
            (0, 1, 2, 6),
        ),
        CONTACT_DESCEND_CLOSE_POSE_Z_GRIPPER_TASK_CONFIG: (
            "contact",
            2,
            (2, 6),
        ),
        LIFT_HOLD_POSE_Z_TASK_CONFIG: ("lift", 1, (2,)),
        PICK_POSE_XYZ_GRIPPER_TASK_CONFIG: (
            "lift",
            4,
            (0, 1, 2, 6),
        ),
    }
    for path, (objective, policy_dim, indices) in expected.items():
        config = CubeLiftTaskConfig.load(path)
        transform = build_policy_action_transform(config)
        metadata = build_learning_metadata(config)
        assert config.task_objective == objective
        assert config.policy_action_dim == policy_dim
        assert config.policy_active_action_indices == indices
        assert config.action_dim == 7
        assert transform.policy_dim == policy_dim
        assert transform.task_dim == 7
        assert metadata["model_output_dim"] == policy_dim
        assert metadata["task_action_dim"] == 7
        latent = np.linspace(-0.5, 0.5, policy_dim, dtype=np.float32)
        task_action = transform.transform(latent)
        assert task_action.shape == (7,)
        expected_active = (
            latent * np.asarray(config.resolved_policy_action_scale)
            + np.asarray(config.resolved_policy_action_bias)
        )
        np.testing.assert_allclose(task_action[list(indices)], expected_active)
        inactive = tuple(index for index in range(7) if index not in indices)
        np.testing.assert_allclose(task_action[list(inactive)], 0.0)
        assert len(config.source_dependencies) == 2

    randomized = CubeLiftTaskConfig.load(REACH_POSE_XYZ_TASK_CONFIG)
    first = randomized.sample_cube_spawn_qpos(np.random.default_rng(17))
    repeated = randomized.sample_cube_spawn_qpos(np.random.default_rng(17))
    different = randomized.sample_cube_spawn_qpos(np.random.default_rng(18))
    assert first == repeated
    assert first != different
    contract = build_observation_contract(randomized)
    assert contract.field_names == ("proprio", "object_pose")
    assert contract.field("object_pose").source == (
        "sim_ground_truth_proxy_for_external_object_pose_estimator.v1"
    )


def test_reach_goal_delta_is_deployable_actor_input_not_privileged_state() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_actor_observation_contract,
        build_learning_metadata,
        build_observation_contract,
        encode_policy_observation,
    )

    config = CubeLiftTaskConfig.load(REACH_GOAL_DELTA_XYZ_TASK_CONFIG)
    raw_contract = build_observation_contract(config)
    actor_contract = build_actor_observation_contract(config, raw_contract)
    metadata = build_learning_metadata(config)

    assert config.observation_dim == 38
    assert raw_contract.field_names == (
        "proprio",
        "object_pose",
        "reach_target_delta",
    )
    assert actor_contract.field_names == ("proprio", "reach_target_delta")
    assert actor_contract.field("reach_target_delta").source == (
        "derived_from_external_object_pose_and_rm65_measured_qpos_fk_in_right_base.v1"
    )
    assert metadata["model_input_dim"] == 31
    assert metadata["model_input_fields"] == (
        "proprio",
        "reach_target_delta",
    )
    assert metadata["actor_uses_privileged"] is False

    raw = np.arange(38, dtype=np.float32)
    encoded = encode_policy_observation(config, raw)
    assert encoded.shape == (31,)
    np.testing.assert_allclose(encoded[:28], raw[:28])
    np.testing.assert_allclose(encoded[28:], raw[35:38])


def test_reach_goal_delta_progress_task_removes_stationary_reward_without_privilege() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata

    config = CubeLiftTaskConfig.load(REACH_GOAL_DELTA_PROGRESS_XYZ_TASK_CONFIG)
    metadata = build_learning_metadata(config)
    shaping = config.reward_shaping

    assert config.task_objective == "reach"
    assert config.resolved_policy_observation_fields == (
        "proprio",
        "reach_target_delta",
    )
    assert config.resolved_policy_action_components == (
        "right_tcp_delta_x",
        "right_tcp_delta_y",
        "right_tcp_delta_z",
    )
    assert metadata["actor_uses_privileged"] is False
    assert shaping["staged_grasp"] is False
    assert shaping["reach_reward_scale"] == 0.0
    assert shaping["reach_progress_scale"] > 0.0
    assert shaping["grasp_region_reward_scale"] == 0.0
    assert shaping["grasp_region_progress_scale"] == 0.0
    assert shaping["approach_alignment_reward_scale"] == 0.0
    assert shaping["success_bonus"] > 0.0
    assert shaping["time_limit_penalty"] > 0.0


def test_collision_robust_task_curriculum_is_pose_aligned_and_nonprivileged() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata

    expected = {
        SAFE_HOVER_GOAL_DELTA_XYZ_TASK_CONFIG: ("reach", 3, 5),
        SAFE_PREGRASP_GOAL_DELTA_XYZ_TASK_CONFIG: ("reach", 3, 5),
        CONTACT_QUALITY_GOAL_DELTA_TASK_CONFIG: ("contact", 4, 3),
        COLLISION_ROBUST_PICK_TASK_CONFIG: ("lift", 4, 3),
        TALL_BLOCK_PICK_TASK_CONFIG: ("lift", 4, 3),
    }
    for path, (objective, policy_dim, clear_steps) in expected.items():
        config = CubeLiftTaskConfig.load(path)
        metadata = build_learning_metadata(config)
        assert config.task_objective == objective
        assert config.action_dim == 7
        assert config.policy_action_dim == policy_dim
        assert config.success_min_safety_clear_policy_steps == clear_steps
        assert config.reward_shaping["safety_intervention_penalty"] > 0.0
        assert config.resolved_policy_observation_fields == (
            "proprio",
            "reach_target_delta",
        )
        assert metadata["actor_uses_privileged"] is False

    hover = CubeLiftTaskConfig.load(SAFE_HOVER_GOAL_DELTA_XYZ_TASK_CONFIG)
    assert hover.reach_target_height_offset_m == pytest.approx(0.10)
    assert hover.reward_shaping["grasp_height_offset_m"] == pytest.approx(0.10)
    pregrasp = CubeLiftTaskConfig.load(SAFE_PREGRASP_GOAL_DELTA_XYZ_TASK_CONFIG)
    assert pregrasp.reach_target_height_offset_m == pytest.approx(0.04)
    assert pregrasp.reward_shaping["grasp_height_offset_m"] == pytest.approx(0.04)

    for path in (
        CONTACT_QUALITY_GOAL_DELTA_TASK_CONFIG,
        COLLISION_ROBUST_PICK_TASK_CONFIG,
    ):
        config = CubeLiftTaskConfig.load(path)
        assert config.success_min_bilateral_inner_contact_score == pytest.approx(0.5)
        assert config.success_max_outer_contact_score == pytest.approx(0.25)


def test_collision_quality_config_defaults_are_optional_and_validated() -> None:
    from dataclasses import replace

    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    legacy = CubeLiftTaskConfig.load(TASK_CONFIG)
    assert legacy.success_min_safety_clear_policy_steps == 0
    assert legacy.success_min_bilateral_inner_contact_score == pytest.approx(0.0)
    assert legacy.success_max_outer_contact_score == pytest.approx(1.0)
    with pytest.raises(ValueError, match="must be non-negative"):
        replace(legacy, success_min_safety_clear_policy_steps=-1)
    with pytest.raises(ValueError, match="Contact-quality success gates"):
        replace(legacy, success_min_bilateral_inner_contact_score=0.5)
    mismatched_geometry = replace(legacy, object_size_m=(0.03, 0.03, 0.05))
    with pytest.raises(ValueError, match="object_size_m disagrees"):
        mismatched_geometry.base_contract()

    tall_block = CubeLiftTaskConfig.load(TALL_BLOCK_PICK_TASK_CONFIG)
    assert tall_block.object_size_m == pytest.approx((0.03, 0.03, 0.05))
    assert tall_block.base_contract().scene.objects[0].size_m == pytest.approx(
        (0.03, 0.03, 0.05)
    )


def test_collision_penalty_ablation_matches_baseline_expected_fixed_cost() -> None:
    from sim.mujoco.collision_safety import FORBIDDEN_COLLISION_CATEGORIES
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    uniform = CubeLiftTaskConfig.load(COLLISION_UNIFORM_LIGHT_TASK_CONFIG)
    categorized = CubeLiftTaskConfig.load(COLLISION_CATEGORY_LIGHT_TASK_CONFIG)
    assert uniform.reward_shaping["safety_projection_penalty"] == pytest.approx(
        categorized.reward_shaping["safety_projection_penalty"]
    )
    assert uniform.reward_shaping["safety_intervention_penalty"] == pytest.approx(
        0.095
    )
    assert uniform.reward_shaping["safety_collision_category_penalties"] == {}
    category_penalties = categorized.reward_shaping[
        "safety_collision_category_penalties"
    ]
    assert set(category_penalties) == set(FORBIDDEN_COLLISION_CATEGORIES)
    assert categorized.reward_shaping["safety_intervention_penalty"] == 0.0

    # Frozen seed-20000 baseline action rates from the validated v2 checkpoint.
    baseline_rates = {
        "gripper_environment": 0.2746085011185682,
        "gripper_self": 0.0022371364653243847,
        "robot_environment": 0.14821029082774048,
        "robot_self": 0.12472035794183445,
    }
    uniform_expected = 0.095 * sum(baseline_rates.values())
    category_expected = sum(
        baseline_rates.get(category, 0.0) * float(penalty)
        for category, penalty in category_penalties.items()
    )
    assert category_expected == pytest.approx(uniform_expected, abs=5e-4)


def test_success_requires_configured_collision_free_policy_streak() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.kernel import CubeLiftKernel

    config = replace(
        CubeLiftTaskConfig.load(REACH_GOAL_DELTA_XYZ_TASK_CONFIG),
        success_hold_physics_ticks=1,
        success_min_safety_clear_policy_steps=3,
    )
    kernel = object.__new__(CubeLiftKernel)
    kernel.config = config
    kernel._critical_state_is_finite = lambda: True
    kernel._last_active_collision_check = None
    kernel._cube_qpos_addr = 0
    kernel.data = SimpleNamespace(qpos=np.zeros(7, dtype=np.float64))
    kernel._bounds_lower = np.full(3, -1.0, dtype=np.float64)
    kernel._bounds_upper = np.full(3, 1.0, dtype=np.float64)
    kernel._reach_distance_m = lambda: 0.0
    kernel._success_hold_ticks = 0
    kernel._termination_reason = None
    kernel._safety_clear_policy_steps = 2

    assert kernel._check_termination(check_success=True) is False
    assert kernel._success_hold_ticks == 0
    kernel._safety_clear_policy_steps = 3
    assert kernel._check_termination(check_success=True) is True
    assert kernel._termination_reason == "tcp_reach_success"


def test_success_contact_quality_rejects_outer_surface_substitute() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.kernel import CubeLiftKernel

    kernel = object.__new__(CubeLiftKernel)
    kernel.config = CubeLiftTaskConfig.load(COLLISION_ROBUST_PICK_TASK_CONFIG)
    kernel._gripper_contact_scores = lambda: {
        "gripper_bilateral_inner_contact_score": 0.8,
        "gripper_outer_contact_score": 0.4,
    }
    assert kernel._success_contact_quality_satisfied() is False
    kernel._gripper_contact_scores = lambda: {
        "gripper_bilateral_inner_contact_score": 0.8,
        "gripper_outer_contact_score": 0.1,
    }
    assert kernel._success_contact_quality_satisfied() is True


def test_reach_overhead_guidance_uses_nonprivileged_coarse_to_fine_pose_decode() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_learning_metadata,
        build_policy_action_transform,
    )
    from sim.mujoco.tasks.cube_lift.reach_guidance import (
        build_reach_guidance,
        coarse_to_fine_reach_action_scale,
    )
    from sim.mujoco_temp_pipeline.bootstrap import goal_delta_teacher_action

    config = CubeLiftTaskConfig.load(
        REACH_GOAL_DELTA_OVERHEAD_PROGRESS_XYZ_TASK_CONFIG
    )
    direct_config = CubeLiftTaskConfig.load(
        REACH_GOAL_DELTA_PROGRESS_XYZ_TASK_CONFIG
    )
    metadata = build_learning_metadata(config)
    direct_metadata = build_learning_metadata(direct_config)
    assert config.reach_guidance_enabled is True
    assert config.reach_guidance_height_offset_m == pytest.approx(0.12)
    assert config.reach_guidance_xy_blend_radius_m == pytest.approx(0.08)
    assert config.reach_action_distance_scaling_enabled is True
    assert config.reach_action_min_scale == pytest.approx(0.35)
    assert config.reach_action_slow_radius_m == pytest.approx(0.08)
    assert config.resolved_policy_action_scale == pytest.approx((1.0, 1.0, 1.0))
    assert metadata["actor_uses_privileged"] is False
    assert (
        metadata["task_action_binding_hash"]
        == direct_metadata["task_action_binding_hash"]
    )
    assert (
        metadata["observation_contract_hash"]
        != direct_metadata["observation_contract_hash"]
    )
    assert (
        metadata["policy_action_transform_hash"]
        != direct_metadata["policy_action_transform_hash"]
    )
    assert config.resolved_policy_observation_fields == (
        "proprio",
        "reach_target_delta",
    )
    transform = build_policy_action_transform(config)
    direct_transform = build_policy_action_transform(direct_config)
    assert transform.observation_conditioning_enabled is True
    assert direct_transform.observation_conditioning_enabled is False
    assert (
        direct_transform.semantic_content_hash
        == direct_transform.base_transform.semantic_content_hash
    )

    object_position = np.asarray((0.0, 0.0, 0.0))
    far = build_reach_guidance(
        object_position,
        np.asarray((0.2, 0.0, 0.2)),
        np.eye(3),
        target_height_offset_m=config.reach_target_height_offset_m,
        guidance_height_offset_m=config.reach_guidance_height_offset_m,
        guidance_xy_blend_radius_m=(
            config.reach_guidance_xy_blend_radius_m
        ),
    )
    middle = build_reach_guidance(
        object_position,
        np.asarray((0.04, 0.0, 0.2)),
        np.eye(3),
        target_height_offset_m=config.reach_target_height_offset_m,
        guidance_height_offset_m=config.reach_guidance_height_offset_m,
        guidance_xy_blend_radius_m=(
            config.reach_guidance_xy_blend_radius_m
        ),
    )
    aligned = build_reach_guidance(
        object_position,
        np.asarray((0.0, 0.0, 0.2)),
        np.eye(3),
        target_height_offset_m=config.reach_target_height_offset_m,
        guidance_height_offset_m=config.reach_guidance_height_offset_m,
        guidance_xy_blend_radius_m=(
            config.reach_guidance_xy_blend_radius_m
        ),
    )
    assert far.overhead_blend == pytest.approx(1.0)
    assert far.target_height_offset_m == pytest.approx(0.12)
    assert middle.overhead_blend == pytest.approx(0.5)
    assert middle.target_height_offset_m == pytest.approx(0.075)
    assert aligned.overhead_blend == pytest.approx(0.0)
    assert aligned.target_height_offset_m == pytest.approx(0.03)

    observation = np.zeros((1, config.observation_dim), dtype=np.float32)
    observation[0, 35:38] = (0.006, -0.012, 0.03)
    distance_scale = coarse_to_fine_reach_action_scale(
        observation[:, 35:38],
        minimum_scale=config.reach_action_min_scale,
        slow_radius_m=config.reach_action_slow_radius_m,
    )
    np.testing.assert_allclose(
        transform.action_scale(observation),
        distance_scale,
    )
    np.testing.assert_allclose(
        direct_transform.action_scale(observation),
        1.0,
    )
    desired_task_action = np.clip(
        observation[:, 35:38]
        / np.asarray(config.right_tcp_delta_m, dtype=np.float32),
        -1.0,
        1.0,
    )
    expected_latent = np.clip(
        desired_task_action / distance_scale[:, None],
        -1.0,
        1.0,
    )
    latent = goal_delta_teacher_action(config, observation)
    np.testing.assert_allclose(latent, expected_latent, atol=1e-6)
    with pytest.raises(ValueError, match="current task observation"):
        transform.transform(latent)
    public_action = transform.transform(latent, observation=observation)
    np.testing.assert_allclose(
        public_action[0, :3],
        expected_latent[0] * distance_scale[0],
        atol=1e-6,
    )
    np.testing.assert_allclose(public_action[0, 3:], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        coarse_to_fine_reach_action_scale(
            np.zeros((1, 3), dtype=np.float32),
            minimum_scale=config.reach_action_min_scale,
            slow_radius_m=config.reach_action_slow_radius_m,
        ),
        config.reach_action_min_scale,
    )
    np.testing.assert_allclose(
        coarse_to_fine_reach_action_scale(
            np.asarray(((0.08, 0.0, 0.0),), dtype=np.float32),
            minimum_scale=config.reach_action_min_scale,
            slow_radius_m=config.reach_action_slow_radius_m,
        ),
        1.0,
    )

    # Contract-only point-mass check: the bounded decoder must still be able
    # to reach the unchanged final target well inside the 180-step horizon.
    tcp_position = np.asarray((0.24, 0.0, 0.34), dtype=np.float64)
    final_target = object_position + np.asarray(
        (0.0, 0.0, config.reach_target_height_offset_m)
    )
    maximum_axis_step = max(np.asarray(config.right_tcp_delta_m))
    observed_action_scales = []
    for step in range(config.max_episode_steps):
        guidance = build_reach_guidance(
            object_position,
            tcp_position,
            np.eye(3),
            target_height_offset_m=config.reach_target_height_offset_m,
            guidance_height_offset_m=config.reach_guidance_height_offset_m,
            guidance_xy_blend_radius_m=(
                config.reach_guidance_xy_blend_radius_m
            ),
        )
        observation[0, 35:38] = guidance.delta_in_base
        action_scale = coarse_to_fine_reach_action_scale(
            observation[:, 35:38],
            minimum_scale=config.reach_action_min_scale,
            slow_radius_m=config.reach_action_slow_radius_m,
        )
        observed_action_scales.append(float(action_scale[0]))
        latent = goal_delta_teacher_action(config, observation)
        normalized_xyz = transform.transform(
            latent,
            observation=observation,
        )[0, :3]
        physical_delta = normalized_xyz * np.asarray(
            config.right_tcp_delta_m
        )
        assert np.max(np.abs(physical_delta)) <= maximum_axis_step + 1e-9
        assert np.all(
            np.abs(physical_delta)
            <= np.asarray(config.right_tcp_delta_m) * action_scale[0] + 1e-9
        )
        tcp_position += physical_delta
        final_distance = float(np.linalg.norm(final_target - tcp_position))
        if final_distance <= config.reach_success_distance_m:
            break
    assert observed_action_scales[0] == pytest.approx(1.0)
    assert min(observed_action_scales) < 1.0
    assert step + 1 <= 60
    assert final_distance <= config.reach_success_distance_m


def test_contact_goal_delta_bridge_preserves_reach_actor_and_scales_only_xyz() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_learning_metadata,
        build_policy_action_transform,
    )
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
    )

    reach = CubeLiftTaskConfig.load(
        REACH_GOAL_DELTA_OVERHEAD_PROGRESS_XYZ_TASK_CONFIG
    )
    contact = CubeLiftTaskConfig.load(
        CONTACT_GOAL_DELTA_OVERHEAD_XYZ_GRIPPER_TASK_CONFIG
    )
    reach_metadata = build_learning_metadata(reach)
    contact_metadata = build_learning_metadata(contact)

    assert contact.task_objective == "contact"
    assert contact.policy_class == "normal"
    assert contact.resolved_policy_observation_fields == (
        "proprio",
        "reach_target_delta",
    )
    assert contact.resolved_policy_action_components == (
        "right_tcp_delta_x",
        "right_tcp_delta_y",
        "right_tcp_delta_z",
        "right_gripper_delta",
    )
    assert contact_metadata["actor_uses_privileged"] is False
    assert (
        contact_metadata["observation_contract_hash"]
        == reach_metadata["observation_contract_hash"]
    )
    assert (
        contact_metadata["model_input_binding_hash"]
        == reach_metadata["model_input_binding_hash"]
    )
    assert (
        contact_metadata["task_action_binding_hash"]
        == reach_metadata["task_action_binding_hash"]
    )
    checkpoint_task_metadata = reach.to_metadata()
    checkpoint_task_metadata["learning_contract"] = reach_metadata
    compatibility = check_task_config_compatibility(
        {
            "task_config_sha256": reach.source_sha256,
            "task_metadata": checkpoint_task_metadata,
        },
        CONTACT_GOAL_DELTA_OVERHEAD_XYZ_GRIPPER_TASK_CONFIG,
        allow_mismatch=True,
        allow_policy_action_component_transfer=True,
    )
    transfer = compatibility["policy_action_component_transfer"]
    assert transfer is not None
    assert transfer["shared_components"] == (
        "right_tcp_delta_x",
        "right_tcp_delta_y",
        "right_tcp_delta_z",
    )
    assert transfer["new_components"] == ("right_gripper_delta",)

    observation = np.zeros((1, contact.observation_dim), dtype=np.float32)
    observation[0, 35:38] = (0.04, 0.0, 0.0)
    transform = build_policy_action_transform(contact)
    scale = transform.action_scale(observation)
    latent = np.ones((1, 4), dtype=np.float32)
    public_action = transform.transform(latent, observation=observation)

    np.testing.assert_allclose(public_action[0, :3], scale[0])
    np.testing.assert_allclose(public_action[0, 3:6], 0.0)
    assert public_action[0, 6] == pytest.approx(1.0)
    assert transform.scaled_task_indices == (0, 1, 2)
    assert (
        transform.semantic_content_hash
        == "3aed96d11fb2044033834bae24cfb35aaa559dbeb2aaef4bc8ba0065662937f7"
    )

    closing_observation = observation.copy()
    closing_observation[0, 35:38] = (0.03, 0.0, 0.0)
    closing_scale = transform.action_scale(closing_observation)
    closing_action = transform.transform(
        latent,
        observation=closing_observation,
    )
    assert closing_scale[0] < scale[0]
    np.testing.assert_allclose(closing_action[0, :3], closing_scale[0])
    assert closing_action[0, 6] == pytest.approx(1.0)

    latched_observation = observation.copy()
    latched_observation[0, 6] = 0.2
    latched_scale = transform.action_scale(latched_observation)
    assert latched_scale[0] == pytest.approx(
        scale[0] * contact.contact_close_translation_scale
    )

    anisotropic = CubeLiftTaskConfig.load(
        CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG
    )
    anisotropic_transform = build_policy_action_transform(anisotropic)
    component_scale = anisotropic_transform.translation_action_scale(
        closing_observation
    )
    anisotropic_action = anisotropic_transform.transform(
        latent,
        observation=closing_observation,
    )
    np.testing.assert_allclose(component_scale[0, :2], closing_scale[0])
    assert component_scale[0, 2] == pytest.approx(
        closing_scale[0] * 0.25
    )
    np.testing.assert_allclose(anisotropic_action[0, :3], component_scale[0])
    assert anisotropic_action[0, 6] == pytest.approx(1.0)
    assert anisotropic_transform.action_scale(closing_observation)[0] == pytest.approx(
        component_scale[0, 2]
    )
    assert anisotropic_transform.semantic_content_hash != transform.semantic_content_hash

    contact_checkpoint_metadata = contact.to_metadata()
    contact_checkpoint_metadata["learning_contract"] = contact_metadata
    contact_payload = {
        "task_config_sha256": contact.source_sha256,
        "task_metadata": contact_checkpoint_metadata,
    }
    with pytest.raises(ValueError, match="policy_action_transform_hash"):
        check_task_config_compatibility(
            contact_payload,
            CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG,
            allow_mismatch=True,
        )
    transform_transfer = check_task_config_compatibility(
        contact_payload,
        CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG,
        allow_mismatch=True,
        allow_policy_action_transform_transfer=True,
    )["policy_action_transform_transfer"]
    assert transform_transfer is not None
    assert transform_transfer["critic_reset_required"] is True
    assert transform_transfer["checkpoint_transform_hash"] == (
        transform.semantic_content_hash
    )
    assert transform_transfer["runtime_transform_hash"] == (
        anisotropic_transform.semantic_content_hash
    )


def test_action_diagnostics_keep_latent_and_public_pose_metrics_distinct() -> None:
    from sim.mujoco_temp_pipeline.action_diagnostics import ActionDiagnostics

    diagnostics = ActionDiagnostics()
    policy_action = np.asarray(
        ((1.0, -0.5, 0.0), (0.94, -1.0, 0.2)),
        dtype=np.float32,
    )
    task_action = np.zeros((2, 7), dtype=np.float32)
    task_action[:, :3] = policy_action * np.asarray(((1.0,), (0.5,)))
    diagnostics.update(
        policy_action,
        task_action,
        np.asarray((1.0, 0.5), dtype=np.float32),
    )

    summary = diagnostics.summary()
    assert summary["policy_action_abs_mean"] == pytest.approx(
        float(np.mean(np.abs(policy_action)))
    )
    assert summary["policy_action_saturation_rate"] == pytest.approx(2.0 / 6.0)
    assert summary["task_action_abs_mean"] == pytest.approx(
        float(np.mean(np.abs(task_action)))
    )
    assert summary["action_conditioning_scale_mean"] == pytest.approx(0.75)
    assert summary["action_conditioning_scale_min"] == pytest.approx(0.5)
    assert summary["action_conditioning_active_rate"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="batch shapes must match"):
        diagnostics.update(
            policy_action,
            task_action,
            np.ones(1, dtype=np.float32),
        )


def test_contact_task_exposes_proximity_gated_closure_credit() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(CONTACT_POSE_XYZ_GRIPPER_TASK_CONFIG)
    shaping = config.reward_shaping
    assert shaping["gripper_reward_scale"] > 0.0
    assert shaping["gripper_progress_scale"] > 0.0
    assert shaping["gripper_command_reward_scale"] > 0.0
    assert shaping["gripper_reward_requires_contact"] is False
    assert shaping["premature_close_penalty"] > 0.0
    assert shaping["premature_close_state_penalty"] > 0.0
    assert shaping["grasp_acquire_requires_contact"] is True
    assert config.success_requires_bilateral_contact is True


def test_actor_observation_view_makes_object_pose_optional_and_hashed() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_actor_observation_contract,
        build_learning_metadata,
        build_model_input_binding,
        build_observation_contract,
        encode_policy_observation,
    )

    state_only = CubeLiftTaskConfig.load(
        CONTACT_DESCEND_CLOSE_POSE_Z_GRIPPER_TASK_CONFIG
    )
    raw_contract = build_observation_contract(state_only)
    actor_contract = build_actor_observation_contract(state_only, raw_contract)
    binding = build_model_input_binding(state_only, raw_contract)
    metadata = build_learning_metadata(state_only)

    assert raw_contract.field_names == ("proprio", "object_pose")
    assert actor_contract.field_names == ("proprio",)
    assert binding.input_fields == ("proprio",)
    assert binding.allow_privileged is False
    assert metadata["task_observation_dim"] == 35
    assert metadata["model_input_dim"] == 28
    assert metadata["actor_uses_privileged"] is False

    raw = np.arange(35, dtype=np.float32)
    encoded = encode_policy_observation(state_only, raw)
    assert encoded.shape == (28,)
    np.testing.assert_allclose(encoded, raw[:28])

    lift = CubeLiftTaskConfig.load(LIFT_HOLD_POSE_Z_TASK_CONFIG)
    lift_actor_contract = build_actor_observation_contract(lift)
    assert (
        lift_actor_contract.semantic_content_hash
        == actor_contract.semantic_content_hash
    )


def test_task_suite_pairs_proprio_baselines_with_deployable_estimator_tasks() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata

    proprio_only = (
        REACH_FIXED_POSE_XYZ_PROPRIO_TASK_CONFIG,
        CONTACT_FIXED_POSE_XYZ_GRIPPER_PROPRIO_TASK_CONFIG,
        CONTACT_DESCEND_CLOSE_POSE_Z_GRIPPER_TASK_CONFIG,
        LIFT_SHORT_POSE_Z_PROPRIO_TASK_CONFIG,
        LIFT_SHORT_POSITIVE_Z_PROPRIO_TASK_CONFIG,
        LIFT_HOLD_POSE_Z_TASK_CONFIG,
        PICK_POSE_XYZ_GRIPPER_TASK_CONFIG,
    )
    for path in proprio_only:
        config = CubeLiftTaskConfig.load(path)
        metadata = build_learning_metadata(config)
        assert config.resolved_policy_observation_fields == ("proprio",)
        assert metadata["model_input_fields"] == ("proprio",)
        assert metadata["actor_uses_privileged"] is False

    estimator_tasks = (
        REACH_POSE_XYZ_TASK_CONFIG,
        CONTACT_POSE_XYZ_GRIPPER_TASK_CONFIG,
        PICK_POSE_XYZ_GRIPPER_ESTIMATOR_TASK_CONFIG,
    )
    for path in estimator_tasks:
        config = CubeLiftTaskConfig.load(path)
        metadata = build_learning_metadata(config)
        assert config.resolved_policy_observation_fields == (
            "proprio",
            "object_pose",
        )
        assert metadata["model_input_fields"] == (
            "proprio",
            "object_pose",
        )
        assert metadata["actor_uses_privileged"] is False
        assert config.object_pose_observation_source

    assert CubeLiftTaskConfig.load(
        LIFT_SHORT_POSE_Z_PROPRIO_TASK_CONFIG
    ).lift_delta_m == pytest.approx(0.02)
    assert CubeLiftTaskConfig.load(
        LIFT_SHORT_POSITIVE_Z_PROPRIO_TASK_CONFIG
    ).lift_delta_m == pytest.approx(0.02)


def test_pick_task_keeps_closure_credit_and_policy_owned_phase_change() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PICK_POSE_XYZ_GRIPPER_TASK_CONFIG)
    shaping = config.reward_shaping
    assert shaping["gripper_reward_requires_contact"] is False
    assert shaping["gripper_command_reward_scale"] > 0.0
    assert shaping["lift_progress_scale"] > 0.0
    assert shaping["require_bilateral_contact_for_lift_reward"] is True
    assert config.post_contact_phase_enabled is False
    assert config.policy_active_action_indices == (0, 1, 2, 6)
    assert config.right_tcp_delta_m == pytest.approx((0.01, 0.01, 0.01))
    assert config.resolved_policy_action_scale == pytest.approx(
        (0.2, 0.2, 1.0, 1.0)
    )


def test_goal_delta_pick_aligns_actor_target_with_direct_physical_grasp() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_learning_metadata,
        build_policy_action_transform,
    )
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
    )

    config = CubeLiftTaskConfig.load(
        PICK_GOAL_DELTA_XYZ_GRIPPER_TASK_CONFIG
    )
    metadata = build_learning_metadata(config)
    transform = build_policy_action_transform(config)
    assert config.task_objective == "lift"
    assert config.scripted_pregrasp_reset is False
    assert config.randomizes_cube_spawn is True
    assert config.resolved_policy_observation_fields == (
        "proprio",
        "reach_target_delta",
    )
    assert metadata["actor_uses_privileged"] is False
    assert config.policy_action_components == (
        "right_tcp_delta_x",
        "right_tcp_delta_y",
        "right_tcp_delta_z",
        "right_gripper_delta",
    )
    assert config.post_contact_phase_enabled is True
    assert config.post_contact_action_scale == pytest.approx((1.0,) * 7)
    assert config.post_contact_action_bias == pytest.approx((0.0,) * 7)
    assert config.reach_target_height_offset_m == pytest.approx(
        config.reward_shaping["closed_grasp_height_offset_m"]
    )
    assert config.reach_guidance_enabled is False
    assert config.reach_action_distance_scaling_enabled is False
    assert config.contact_close_translation_scaling_enabled is False
    assert config.contact_close_lift_release_enabled is False

    observation = np.zeros((2, config.observation_dim), dtype=np.float32)
    observation[:, 35:38] = (0.02, 0.0, 0.0)
    observation[0, 6] = 0.2
    observation[1, 6] = 0.6
    scales = transform.translation_action_scale(observation)
    np.testing.assert_allclose(scales, 1.0)
    assert transform.observation_conditioning_enabled is False

    contact = CubeLiftTaskConfig.load(
        CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG
    )
    contact_metadata = contact.to_metadata()
    contact_metadata["learning_contract"] = build_learning_metadata(contact)
    contact_payload = {
        "task_config_sha256": contact.source_sha256,
        "task_metadata": contact_metadata,
    }
    with pytest.raises(ValueError, match="task_action_conditioning_hash"):
        check_task_config_compatibility(
            contact_payload,
            PICK_GOAL_DELTA_XYZ_GRIPPER_TASK_CONFIG,
            allow_mismatch=True,
            allow_policy_action_transform_transfer=True,
        )
    compatibility = check_task_config_compatibility(
        contact_payload,
        PICK_GOAL_DELTA_XYZ_GRIPPER_TASK_CONFIG,
        allow_mismatch=True,
        allow_policy_action_transform_transfer=True,
        allow_task_action_conditioning_transfer=True,
    )
    conditioning_transfer = compatibility[
        "task_action_conditioning_transfer"
    ]
    assert conditioning_transfer is not None
    assert conditioning_transfer["critic_reset_required"] is True


def test_lift_hold_maps_owned_z_to_nonnegative_pose_action() -> None:
    from sim.mujoco.tasks.cube_lift.action_conditioning import (
        CubeLiftActionConditioner,
    )
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform

    config = CubeLiftTaskConfig.load(LIFT_HOLD_POSE_Z_TASK_CONFIG)
    transform = build_policy_action_transform(config)
    conditioner = CubeLiftActionConditioner(config)
    mapped_z = []
    for latent_z in (-1.0, 0.0, 1.0):
        requested = transform.transform(
            np.asarray([latent_z], dtype=np.float32)
        )
        conditioned = conditioner.condition(
            requested,
            left_finger_contact=True,
            right_finger_contact=True,
            measured_gripper_close=0.6,
            scripted_pregrasp_reset_applied=False,
            scripted_contact_reset_applied=True,
        )
        mapped_z.append(float(conditioned.values[2]))
        np.testing.assert_allclose(
            conditioned.values[[0, 1, 3, 4, 5, 6]],
            0.0,
        )
    assert mapped_z == pytest.approx((0.0, 0.5, 1.0))
    assert config.scripted_contact_action_scale == pytest.approx(
        (1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 0.0)
    )
    assert config.scripted_contact_action_bias[2] == pytest.approx(0.5)
    assert config.scripted_contact_nonnegative_lift_action is False
    assert config.reward_shaping["bilateral_contact_reward_scale"] == 0.0


def test_lift_short_v2_requires_positive_policy_owned_z() -> None:
    from sim.mujoco.tasks.cube_lift.action_conditioning import (
        CubeLiftActionConditioner,
    )
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform

    config = CubeLiftTaskConfig.load(LIFT_SHORT_POSITIVE_Z_PROPRIO_TASK_CONFIG)
    transform = build_policy_action_transform(config)
    conditioner = CubeLiftActionConditioner(config)
    mapped_z = []
    for latent_z in (-1.0, 0.0, 1.0):
        requested = transform.transform(
            np.asarray([latent_z], dtype=np.float32)
        )
        conditioned = conditioner.condition(
            requested,
            left_finger_contact=True,
            right_finger_contact=True,
            measured_gripper_close=0.6,
            scripted_pregrasp_reset_applied=False,
            scripted_contact_reset_applied=True,
        )
        mapped_z.append(float(conditioned.values[2]))
        np.testing.assert_allclose(
            conditioned.values[[0, 1, 3, 4, 5, 6]],
            0.0,
        )
    assert mapped_z == pytest.approx((0.0, 0.0, 1.0))
    assert config.scripted_contact_action_bias == pytest.approx((0.0,) * 7)
    assert config.scripted_contact_nonnegative_lift_action is True


@pytest.mark.optional
def test_lift_short_v2_zero_policy_holds_and_positive_policy_lifts() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform

    env = FastCubeLiftEnv(
        LIFT_SHORT_POSITIVE_Z_PROPRIO_TASK_CONFIG,
        include_training_reward_state=False,
        use_model_cache=False,
    )
    transform = build_policy_action_transform(env.config)
    try:
        observation, info = env.reset(seed=1)
        zero_action = transform.transform(np.zeros(1, dtype=np.float32))
        for _ in range(env.config.max_episode_steps):
            observation, _reward, terminated, truncated, info = env.step(
                zero_action
            )
            if terminated or truncated:
                break
        assert info["termination_reason"] != "bilateral_contact_lift_success"

        observation, info = env.reset(seed=2)
        positive_action = transform.transform(np.ones(1, dtype=np.float32))
        for _ in range(env.config.max_episode_steps):
            observation, _reward, terminated, truncated, info = env.step(
                positive_action
            )
            if terminated or truncated:
                break
        assert info["termination_reason"] == "bilateral_contact_lift_success"
    finally:
        env.close()


@pytest.mark.optional
def test_descend_close_task_is_feasible_without_hidden_xy_actions() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform

    env = FastCubeLiftEnv(
        CONTACT_DESCEND_CLOSE_POSE_Z_GRIPPER_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        _observation, info = env.reset(seed=1)
        transform = build_policy_action_transform(env.config)
        for step in range(30):
            latent = np.asarray(
                (-1.0, 0.0) if step < 3 else (0.0, 1.0),
                dtype=np.float32,
            )
            task_action = transform.transform(latent)
            _observation, _reward, terminated, truncated, info = env.step(
                task_action
            )
            np.testing.assert_allclose(task_action[[0, 1, 3, 4, 5]], 0.0)
            if terminated or truncated:
                break
        assert info["termination_reason"] == "bilateral_contact_success"
    finally:
        env.close()


def test_new_task_profiles_reject_hidden_randomization_and_nonidentity_action_phase() -> None:
    from dataclasses import replace

    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    unobserved = CubeLiftTaskConfig.load(DEFAULT_PHYSICAL_TASK_CONFIG)
    with pytest.raises(ValueError, match="unobserved cube spawn"):
        replace(unobserved, cube_spawn_xy_jitter_m=(0.01, 0.01))

    randomized = CubeLiftTaskConfig.load(REACH_POSE_XYZ_TASK_CONFIG)
    with pytest.raises(ValueError, match="unobserved cube spawn"):
        replace(
            randomized,
            policy_observation_fields=("proprio",),
        )

    legacy_debug = CubeLiftTaskConfig.load(TASK_CONFIG)
    with pytest.raises(ValueError, match="direct-qpos debug supports lift only"):
        replace(legacy_debug, task_objective="reach")

    hidden_phase = CubeLiftTaskConfig.load(
        NORMAL_PREGRASP_BRIDGE_V5_CUBE_POSE_TASK_CONFIG
    )
    with pytest.raises(ValueError, match="hidden post-contact action phase"):
        replace(
            hidden_phase,
            policy_action_components=(
                "right_tcp_delta_x",
                "right_tcp_delta_y",
                "right_tcp_delta_z",
                "right_gripper_delta",
            ),
        )

    stationary_phase = replace(
        CubeLiftTaskConfig.load(PICK_POSE_XYZ_GRIPPER_TASK_CONFIG),
        post_contact_phase_enabled=True,
        post_contact_action_scale=(1.0,) * 7,
        post_contact_action_bias=(0.0,) * 7,
        post_contact_nonnegative_lift_action=False,
    )
    assert stationary_phase.policy_action_components
    assert stationary_phase.post_contact_phase_enabled is True
    assert stationary_phase.post_contact_action_scale == pytest.approx((1.0,) * 7)

    distance_scaled = CubeLiftTaskConfig.load(
        REACH_GOAL_DELTA_OVERHEAD_PROGRESS_XYZ_TASK_CONFIG
    )
    with pytest.raises(ValueError, match="positive slow radius"):
        replace(distance_scaled, reach_action_slow_radius_m=0.0)
    with pytest.raises(ValueError, match="actor-visible reach_target_delta"):
        replace(distance_scaled, policy_observation_fields=("proprio",))
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(distance_scaled, policy_action_scale=(0.5, 0.5, 0.5))

    contact_scaled = CubeLiftTaskConfig.load(
        CONTACT_GOAL_DELTA_OVERHEAD_XYZ_GRIPPER_TASK_CONFIG
    )
    with pytest.raises(ValueError, match="Contact task"):
        replace(contact_scaled, task_objective="lift")
    with pytest.raises(ValueError, match="xyz or xyz-plus-gripper"):
        replace(
            contact_scaled,
            policy_action_components=(
                "right_tcp_delta_x",
                "right_tcp_delta_y",
                "right_tcp_delta_z",
                "right_tcp_rotvec_x",
            ),
        )
    with pytest.raises(ValueError, match="must lie in"):
        replace(contact_scaled, contact_close_z_translation_scale=0.2)
    with pytest.raises(ValueError, match="enabled Contact-close gate"):
        replace(
            distance_scaled,
            contact_close_z_translation_scale=0.025,
        )


@pytest.mark.optional
def test_reach_task_kernel_preserves_pose_action_lineage_and_seeded_reset() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.learning import build_policy_action_transform

    env = FastCubeLiftEnv(
        REACH_POSE_XYZ_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        _observation, first_info = env.reset(seed=41)
        _observation, repeated_info = env.reset(seed=41)
        _observation, different_info = env.reset(seed=42)
        assert first_info["cube_spawn_qpos"] == pytest.approx(
            repeated_info["cube_spawn_qpos"]
        )
        assert first_info["cube_spawn_qpos"] != pytest.approx(
            different_info["cube_spawn_qpos"]
        )

        transform = build_policy_action_transform(env.config)
        latent = np.asarray([0.25, -0.5, 0.75], dtype=np.float32)
        task_action = transform.transform(latent)
        _observation, _reward, _terminated, _truncated, info = env.step(
            task_action
        )
        assert info["action_lineage_version"] == "cube_lift_action_lineage.v2"
        np.testing.assert_allclose(info["task_requested_action"], task_action)
        np.testing.assert_allclose(info["task_conditioned_action"], task_action)
        assert np.asarray(info["admitted_action"]).shape == (7,)
        assert info["task_objective"] == "reach"
    finally:
        env.close()


@pytest.mark.optional
def test_randomized_pose_task_repeated_seed_has_identical_closed_loop_trace() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        REACH_GOAL_DELTA_OVERHEAD_PROGRESS_XYZ_TASK_CONFIG,
        include_training_reward_state=False,
        use_model_cache=False,
    )
    try:
        traces = []
        for _ in range(2):
            observation, _info = env.reset(seed=8)
            rows = [observation.copy()]
            for _step in range(12):
                action = np.zeros(7, dtype=np.float32)
                action[:3] = (0.4, -0.2, 0.3)
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                rows.append(
                    np.concatenate(
                        (
                            observation,
                            np.asarray(
                                (
                                    reward,
                                    float(info["safety_projection_alpha"]),
                                ),
                                dtype=np.float32,
                            ),
                        )
                    )
                )
                assert not terminated
                assert not truncated
            traces.append(rows)
        assert len(traces[0]) == len(traces[1])
        for first, repeated in zip(*traces):
            np.testing.assert_array_equal(first, repeated)
    finally:
        env.close()


@pytest.mark.optional
def test_goal_delta_pick_physical_oracle_grasps_from_home() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.scripted_grasp_gate import (
        run_scripted_physical_grasp_gate,
    )

    result = run_scripted_physical_grasp_gate(
        PICK_GOAL_DELTA_XYZ_GRIPPER_TASK_CONFIG,
        seed=1,
        lift_step_m=0.004,
    )
    if result.pose_ik_solver != "robot_motion.rm65.position_collision":
        pytest.skip(
            "Current process loaded the MuJoCo DLS fallback before the canonical "
            "Pinocchio runtime; spawned training workers repair LD_LIBRARY_PATH."
        )
    assert result.success is True
    assert result.termination_reason == "bilateral_contact_lift_success"
    assert result.max_cube_lift_m >= 0.03


@pytest.mark.optional
def test_reach_goal_delta_matches_pose_target_in_public_action_frame() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.learning import encode_policy_observation

    env = FastCubeLiftEnv(
        REACH_GOAL_DELTA_XYZ_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        observation, info = env.reset(seed=43)
        state = np.asarray(info["training_reward_state"], dtype=np.float64)
        cube_position = state[7:10]
        grasp_position = cube_position - state[20:23]
        target_position = cube_position.copy()
        target_position[2] += env.config.reach_target_height_offset_m
        base_rotation = env.kernel.scene.body_pose(
            "right_arm_base_link"
        )[:3, :3]
        expected = base_rotation.T @ (target_position - grasp_position)

        np.testing.assert_allclose(observation[35:38], expected, atol=1e-6)
        encoded = encode_policy_observation(env.config, observation)
        np.testing.assert_allclose(encoded[-3:], expected, atol=1e-6)
        assert encoded.shape == (31,)
    finally:
        env.close()


@pytest.mark.optional
@pytest.mark.parametrize(
    ("task_path", "success_reason"),
    (
        (REACH_POSE_XYZ_TASK_CONFIG, "tcp_reach_success"),
        (REACH_GOAL_DELTA_XYZ_TASK_CONFIG, "tcp_reach_success"),
        (REACH_FIXED_POSE_XYZ_PROPRIO_TASK_CONFIG, "tcp_reach_success"),
        (CONTACT_POSE_XYZ_GRIPPER_TASK_CONFIG, "bilateral_contact_success"),
        (
            CONTACT_GOAL_DELTA_OVERHEAD_XYZ_GRIPPER_TASK_CONFIG,
            "bilateral_contact_success",
        ),
        (
            CONTACT_GOAL_DELTA_OVERHEAD_ANISOTROPIC_XYZ_GRIPPER_TASK_CONFIG,
            "bilateral_contact_success",
        ),
        (
            CONTACT_FIXED_POSE_XYZ_GRIPPER_PROPRIO_TASK_CONFIG,
            "bilateral_contact_success",
        ),
        (
            CONTACT_DESCEND_CLOSE_POSE_Z_GRIPPER_TASK_CONFIG,
            "bilateral_contact_success",
        ),
        (LIFT_HOLD_POSE_Z_TASK_CONFIG, "bilateral_contact_lift_success"),
        (LIFT_SHORT_POSE_Z_PROPRIO_TASK_CONFIG, "bilateral_contact_lift_success"),
        (
            LIFT_SHORT_POSITIVE_Z_PROPRIO_TASK_CONFIG,
            "bilateral_contact_lift_success",
        ),
        (PICK_POSE_XYZ_GRIPPER_TASK_CONFIG, "bilateral_contact_lift_success"),
        (
            PICK_POSE_XYZ_GRIPPER_ESTIMATOR_TASK_CONFIG,
            "bilateral_contact_lift_success",
        ),
    ),
)
def test_pose_task_suite_has_a_scripted_feasibility_gate(
    task_path: Path,
    success_reason: str,
) -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.scripted_grasp_gate import (
        run_scripted_physical_grasp_gate,
    )

    result = run_scripted_physical_grasp_gate(task_path, use_model_cache=True)
    if result.pose_ik_solver != "robot_motion.rm65.position_collision":
        pytest.skip(
            "Current process loaded the MuJoCo DLS fallback before the canonical "
            "Pinocchio runtime; spawned training workers repair LD_LIBRARY_PATH."
        )
    assert result.success is True
    assert result.termination_reason == success_reason


def test_normal_rgb_v4_is_the_no_episode_clock_real_observation_profile() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_observation_contract

    config = CubeLiftTaskConfig.load(NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_TASK_CONFIG)
    contract = build_observation_contract(config)
    assert config.policy_class == "normal"
    assert config.proprio_observation_dim == 28
    assert config.observation_dim == 28 + 3 * 48 * 48
    assert config.wrist_rgb_shape == (3, 48, 48)
    assert "episode_progress" not in contract.field_names
    assert config.observation_layout.endswith("right_wrist_rgb_chw_3x48x48.v4")


def test_normal_v2_disables_rgb_and_requires_physical_grasp_success() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_observation_contract

    config = CubeLiftTaskConfig.load(DEFAULT_PHYSICAL_TASK_CONFIG)
    contract = build_observation_contract(config)
    assert config.policy_class == "normal"
    assert config.wrist_rgb_enabled is False
    assert config.wrist_rgb_shape is None
    assert config.observation_dim == 28
    assert "episode_progress" not in contract.field_names
    assert config.success_requires_bilateral_contact is True
    assert config.success_termination_reason == "bilateral_contact_lift_success"
    assert config.pose_target_anti_windup is True
    assert config.lift_delta_m == pytest.approx(0.07)
    assert config.reward_shaping["closed_grasp_height_offset_m"] == pytest.approx(
        0.022
    )
    assert config.reward_shaping["gripper_reward_scale"] == 0.0
    assert config.reward_shaping["require_bilateral_contact_for_lift_reward"] is True
    assert config.reward_shaping["time_limit_penalty"] == 0.0
    assert config.reward_shaping["failure_termination_penalty"] == 5.0
    assert config.reward_shaping["unsafe_termination_penalty"] == 5.0


@pytest.mark.optional
def test_normal_v2_scripted_physical_grasp_gate_reaches_bilateral_lift() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.scripted_grasp_gate import (
        run_scripted_physical_grasp_gate,
    )

    result = run_scripted_physical_grasp_gate(DEFAULT_PHYSICAL_TASK_CONFIG)
    if result.pose_ik_solver != "robot_motion.rm65.position_collision":
        pytest.skip(
            "Current process loaded the MuJoCo DLS fallback before the canonical "
            "Pinocchio runtime; spawned training workers repair LD_LIBRARY_PATH."
        )
    assert result.success is True
    assert result.termination_reason == "bilateral_contact_lift_success"
    assert result.bilateral_contact_steps >= 5
    assert result.max_cube_lift_m >= 0.07
    assert result.steps <= 150


@pytest.mark.optional
def test_contact_curriculum_reset_starts_from_real_bilateral_contact() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    config = CubeLiftTaskConfig.load(CONTACT_CURRICULUM_TASK_CONFIG)
    assert config.scripted_contact_reset_min_measured_close == pytest.approx(0.55)
    assert config.scripted_contact_reset_max_attempts == 2
    env = FastCubeLiftEnv(
        config,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        observation, info = env.reset(seed=1)
    finally:
        env.close()
    assert observation.shape == (28,)
    assert info["scripted_contact_reset"] is True
    assert info["bilateral_finger_cube_contact"] is True
    assert info["termination_reason"] is None
    assert info["right_gripper_measured_close"] >= 0.55
    assert info["scripted_contact_reset_attempts"] == 1


@pytest.mark.optional
def test_contact_curriculum_reset_retries_recoverable_failure(monkeypatch) -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv
    from sim.mujoco.tasks.cube_lift.kernel import ScriptedContactResetError

    env = FastCubeLiftEnv(
        CONTACT_CURRICULUM_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    original_reset = env.kernel.reset
    calls = 0

    def flaky_reset(*, seed=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ScriptedContactResetError("injected transient reset failure")
        return original_reset(seed=seed)

    monkeypatch.setattr(env.kernel, "reset", flaky_reset)
    try:
        observation, info = env.reset(seed=1)
    finally:
        env.close()
    assert observation.shape == (28,)
    assert info["bilateral_finger_cube_contact"] is True
    assert info["scripted_contact_reset_attempts"] == 2
    assert info["scripted_contact_reset_failures"] == (
        "injected transient reset failure",
    )


def test_contact_curriculum_v2_preserves_task_action_but_versions_latent_action() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata

    v1 = CubeLiftTaskConfig.load(CONTACT_CURRICULUM_TASK_CONFIG)
    v2 = CubeLiftTaskConfig.load(CONTACT_CURRICULUM_V2_TASK_CONFIG)
    v1_learning = build_learning_metadata(v1)
    v2_learning = build_learning_metadata(v2)
    assert (
        v2_learning["observation_contract_hash"]
        == v1_learning["observation_contract_hash"]
    )
    assert (
        v2_learning["task_action_binding_hash"]
        == v1_learning["task_action_binding_hash"]
    )
    assert v1_learning["model_output_dim"] == 7
    assert v2_learning["model_output_dim"] == 6
    assert (
        v2_learning["policy_action_transform_hash"]
        != v1_learning["policy_action_transform_hash"]
    )
    assert v2.scripted_contact_reset_min_measured_close == pytest.approx(0.59)
    assert v2.scripted_contact_reset_max_attempts == 3
    assert v2.scripted_contact_loss_grace_steps == 0
    assert v2.scripted_contact_action_scale == pytest.approx(
        (0.05, 0.05, 1.0, 0.02, 0.02, 0.02, 0.0)
    )
    assert v2.scripted_contact_nonnegative_lift_action is True
    assert v2.reward_shaping["safety_projection_penalty"] == pytest.approx(0.02)
    assert v2.reward_shaping["bilateral_contact_acquire_bonus"] == 0.0
    assert v2.reward_shaping["lift_progress_scale"] == pytest.approx(100.0)
    assert v2.reward_shaping["time_limit_penalty"] == pytest.approx(5.0)


def test_contact_curriculum_v3_is_strict_lift_only_with_loss_aware_reward() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata

    v2 = CubeLiftTaskConfig.load(CONTACT_CURRICULUM_V2_TASK_CONFIG)
    v3 = CubeLiftTaskConfig.load(CONTACT_CURRICULUM_V3_TASK_CONFIG)
    v2_learning = build_learning_metadata(v2)
    v3_learning = build_learning_metadata(v3)
    assert (
        v3_learning["observation_contract_hash"]
        == v2_learning["observation_contract_hash"]
    )
    assert (
        v3_learning["task_action_binding_hash"]
        == v2_learning["task_action_binding_hash"]
    )
    assert v3_learning["model_output_dim"] == 1
    assert v2_learning["model_output_dim"] == 6
    assert v3.scripted_contact_action_scale == pytest.approx(
        (0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0)
    )
    assert v3.scripted_contact_action_bias == pytest.approx(
        (0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0)
    )
    assert v3.policy_active_action_indices == (2,)
    assert v3.lift_delta_m == pytest.approx(0.04)
    assert v3.scripted_contact_nonnegative_lift_action is True
    assert v3.reward_shaping["reach_reward_scale"] == 0.0
    assert v3.reward_shaping["reach_progress_scale"] == 0.0
    assert v3.reward_shaping["grasp_region_reward_scale"] == 0.0
    assert v3.reward_shaping["bilateral_contact_missing_penalty"] == pytest.approx(
        0.1
    )
    assert v3.reward_shaping["ungated_negative_lift_progress"] is True
    assert v3.reward_shaping["safety_projection_penalty"] == pytest.approx(0.5)
    assert v3.reward_shaping["unsafe_termination_penalty"] == pytest.approx(10.0)


@pytest.mark.optional
def test_contact_curriculum_v2_scripted_gate_lifts_from_reset() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.scripted_grasp_gate import (
        run_scripted_physical_grasp_gate,
    )

    result = run_scripted_physical_grasp_gate(CONTACT_CURRICULUM_V2_TASK_CONFIG)
    if result.pose_ik_solver != "robot_motion.rm65.position_collision":
        pytest.skip(
            "Current process loaded the MuJoCo DLS fallback before the canonical "
            "Pinocchio runtime; spawned training workers repair LD_LIBRARY_PATH."
        )
    assert result.success is True
    assert result.termination_reason == "bilateral_contact_lift_success"
    assert result.bilateral_contact_steps >= 20
    assert result.max_cube_lift_m >= 0.07


@pytest.mark.optional
def test_contact_curriculum_v2_repeated_reset_preserves_liftable_grasp() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        CONTACT_CURRICULUM_V2_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    lift_action = np.zeros(7, dtype=np.float32)
    lift_action[2] = 0.4
    try:
        for seed in (1, 2):
            _observation, reset_info = env.reset(seed=seed)
            if (
                reset_info["pose_ik_solver"]
                != "robot_motion.rm65.position_collision"
            ):
                pytest.skip(
                    "Current process loaded the MuJoCo DLS fallback before the "
                    "canonical Pinocchio runtime; spawned training workers repair "
                    "LD_LIBRARY_PATH."
                )
            assert reset_info["bilateral_finger_cube_contact"] is True
            for _step in range(30):
                _observation, _reward, terminated, truncated, info = env.step(
                    lift_action
                )
                if terminated or truncated:
                    break
            assert terminated is True
            assert truncated is False
            assert info["termination_reason"] == "bilateral_contact_lift_success"
    finally:
        env.close()


@pytest.mark.optional
def test_contact_curriculum_v2_applies_lift_first_action_constraints() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        CONTACT_CURRICULUM_V2_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        env.reset(seed=1)
        _observation, _reward, terminated, truncated, info = env.step(
            np.asarray((1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0), dtype=np.float32)
        )
    finally:
        env.close()
    assert terminated is False
    assert truncated is False
    assert info["scripted_contact_action_modified"] is True
    assert info["training_reward_state"][23:30] == pytest.approx(
        (0.05, -0.05, 0.0, 0.02, -0.02, 0.02, 0.0),
        abs=1e-6,
    )


@pytest.mark.optional
def test_contact_curriculum_v2_tracks_contact_loss_without_early_termination() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        CONTACT_CURRICULUM_V2_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        env.reset(seed=1)
        cube_qpos_addr = env.kernel._cube_qpos_addr
        env.kernel.data.qpos[cube_qpos_addr + 1] += 0.20
        env.kernel.data.qvel[env.kernel._cube_dof_addr : env.kernel._cube_dof_addr + 6] = 0.0
        env.kernel.mujoco.mj_forward(env.kernel.model, env.kernel.data)
        for expected_loss_steps in range(1, 6):
            _observation, _reward, terminated, truncated, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            assert info["scripted_contact_loss_steps"] == expected_loss_steps
            assert terminated is False
            assert truncated is False
            assert info["termination_reason"] is None
    finally:
        env.close()


@pytest.mark.optional
def test_contact_curriculum_v3_applies_only_nonnegative_lift_action() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        CONTACT_CURRICULUM_V3_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        env.reset(seed=1)
        _observation, _reward, terminated, truncated, info = env.step(
            np.ones(7, dtype=np.float32)
        )
    finally:
        env.close()
    assert terminated is False
    assert truncated is False
    assert info["scripted_contact_action_modified"] is True
    assert info["training_reward_state"][23:30] == pytest.approx(
        (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
        abs=1e-6,
    )


@pytest.mark.optional
def test_contact_curriculum_v3_projection_preserves_locked_gripper_target() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        CONTACT_CURRICULUM_V3_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    raw_lift_actions = (-0.6057, 0.1739, 0.4411, 0.3184, -0.5249)
    try:
        _observation, reset_info = env.reset(seed=1)
        reset_target_close = reset_info["right_gripper_target_close"]
        safety_modified = False
        for raw_lift_action in raw_lift_actions:
            action = np.zeros(7, dtype=np.float32)
            action[2] = raw_lift_action
            _observation, _reward, terminated, truncated, info = env.step(action)
            safety_modified = safety_modified or info["safety_modified"]
            assert info["right_gripper_target_close"] == pytest.approx(
                reset_target_close
            )
            assert terminated is False
            assert truncated is False
    finally:
        env.close()
    assert safety_modified is True


@pytest.mark.optional
def test_contact_curriculum_v3_scripted_gate_lifts_from_reset() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.scripted_grasp_gate import (
        run_scripted_physical_grasp_gate,
    )

    result = run_scripted_physical_grasp_gate(CONTACT_CURRICULUM_V3_TASK_CONFIG)
    if result.pose_ik_solver != "robot_motion.rm65.position_collision":
        pytest.skip(
            "Current process loaded the MuJoCo DLS fallback before the canonical "
            "Pinocchio runtime; spawned training workers repair LD_LIBRARY_PATH."
        )
    assert result.success is True
    assert result.termination_reason == "bilateral_contact_lift_success"
    assert result.bilateral_contact_steps >= 20
    assert result.max_cube_lift_m >= 0.039


def test_privileged_near_lift_v2_uses_dynamic_post_contact_phase() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PRIVILEGED_NEAR_LIFT_V2_TASK_CONFIG)
    assert config.policy_active_action_indices == (0, 1, 2, 6)
    assert config.post_contact_phase_enabled is True
    assert config.post_contact_phase_activation_steps == 5
    assert config.post_contact_action_scale == pytest.approx(
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert config.post_contact_action_bias == pytest.approx(
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )


@pytest.mark.optional
def test_privileged_near_lift_v2_switches_to_lift_only_after_real_contact() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        PRIVILEGED_NEAR_LIFT_V2_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        observation, info = env.reset(seed=1)
        if info["pose_ik_solver"] != "robot_motion.rm65.position_collision":
            pytest.skip(
                "Current process loaded the MuJoCo DLS fallback; the canonical "
                "Pinocchio runtime is required for this trajectory assertion."
            )
        base_rotation = env.kernel.scene.body_pose("right_arm_base_link")[:3, :3]
        closed_height = 0.022
        for _step in range(80):
            state = np.asarray(info["training_reward_state"])
            cube_position = state[7:10]
            grasp_position = cube_position - state[20:23]
            target = cube_position + np.asarray((0.0, 0.0, closed_height))
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(
                (base_rotation.T @ (target - grasp_position))
                / np.asarray(env.config.right_tcp_delta_m),
                -1.0,
                1.0,
            )
            if np.linalg.norm(target - grasp_position) < 0.035:
                action[6] = 1.0
            observation, _reward, terminated, truncated, info = env.step(action)
            if info["post_contact_phase_active"]:
                constrained = np.asarray(info["requested_action"], dtype=np.float32)
                executed = np.asarray(
                    info["training_reward_state"][23:30], dtype=np.float32
                )
                assert info["post_contact_phase_action_modified"] is True
                np.testing.assert_allclose(
                    executed[[0, 1, 3, 4, 5, 6]],
                    0.0,
                    atol=1e-6,
                )
                assert executed[2] >= 0.0
                assert bool(info["bilateral_finger_cube_contact"])
                assert float(constrained[2]) >= -1.0
                return
            if terminated or truncated:
                break
    finally:
        env.close()
    pytest.fail("near-v2 did not acquire the dynamic post-contact phase")


@pytest.mark.optional
def test_temp_training_progress_line_shows_progress_and_key_metrics() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.train import format_training_progress

    line = format_training_progress(
        {
            "update": 10,
            "global_step": 10240,
            "wall_time_seconds": 50.0,
            "transitions_per_second": 204.8,
            "completed_episodes": 71,
            "successful_episodes": 2,
            "success_rate": 2.0 / 71.0,
            "raw_reward_mean": -0.0022,
            "shaped_reward_mean": 0.0365,
            "recent_episode_raw_return_mean": -0.304,
            "reward_distance_m_mean": 0.334,
            "reward_gripper_close_mean": 0.39,
            "reward_bilateral_contact_mean": 0.12,
            "reward_lift_fraction_mean": 0.0,
            "policy_action_abs_mean": 0.61,
            "policy_action_saturation_rate": 0.33,
            "action_conditioning_scale_mean": 0.74,
            "safety_projection_rate": 0.13,
            "safety_rejection_rate": 0.02,
            "clearance_violation_count": 0,
            "hard_penetration_count": 0,
            "contact_loss_termination_count": 5,
            "joint_limit_termination_count": 3,
            "time_limit_termination_count": 4,
            "arm_tracking_error_rad_mean": 0.0179,
            "gripper_tracking_error_mean": 0.09,
            "policy_loss": -0.004,
            "value_loss": 0.33,
            "approx_kl": 0.0006,
        },
        total_timesteps=2_000_000,
        next_checkpoint=50_000,
    )

    assert "steps=10,240/2,000,000 (  0.5%)" in line
    assert "eta=" in line
    assert "succ=2( 2.8%)" in line
    assert "reach=0.334m" in line
    assert "contact=12.0%" in line
    assert "act=0.610 sat=33.0% ascale=0.740" in line
    assert "proj=13.0%" in line
    assert "clr=0 pen=0 closs=5 jlim=3 tout=4" in line
    assert "armerr=0.0179rad" in line
    assert "griperr=0.090" in line
    assert "kl=6.00e-04" in line
    assert "ckpt=next=50,000" in line
    assert len(line) < 700


@pytest.mark.optional
def test_temp_training_can_restore_full_json_terminal_output() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args(
        ["--log-interval-updates", "10", "--print-full-metrics"]
    )
    assert args.log_interval_updates == 10
    assert args.print_full_metrics is True


@pytest.mark.optional
def test_temp_training_init_checkpoint_and_bootstrap_are_explicitly_ordered() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args(["--init-checkpoint", "/tmp/curriculum.pt"])
    assert args.init_checkpoint == Path("/tmp/curriculum.pt")
    sequential = build_arg_parser().parse_args(
        [
            "--init-checkpoint",
            "/tmp/curriculum.pt",
            "--goal-delta-bootstrap-transitions",
            "32",
        ]
    )
    assert sequential.init_checkpoint == Path("/tmp/curriculum.pt")
    assert sequential.goal_delta_bootstrap_transitions == 32


@pytest.mark.optional
def test_temp_training_exploration_schedule_is_explicit() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.config import PPOConfig
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert args.initial_log_std == pytest.approx(-0.5)
    assert args.final_log_std == pytest.approx(-2.5)
    assert args.std_anneal_fraction == pytest.approx(0.75)
    assert args.warm_start_new_action_log_std_cap == pytest.approx(-1.5)
    config = PPOConfig(
        total_timesteps=16,
        num_envs=2,
        rollout_steps=8,
        update_epochs=1,
        minibatch_size=16,
    )
    assert config.final_log_std < config.initial_log_std
    with pytest.raises(ValueError, match="cannot exceed"):
        PPOConfig(final_log_std=0.0)
    with pytest.raises(ValueError, match="policy clamp"):
        PPOConfig(warm_start_new_action_log_std_cap=-6.0)


def test_warm_start_exploration_preserves_shared_components_and_caps_new() -> None:
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_warm_start_exploration_schedule,
    )

    initial, final = build_warm_start_exploration_schedule(
        source_log_std=np.asarray((-2.5, -2.4, -3.0), dtype=np.float32),
        source_components=(
            "right_tcp_delta_x",
            "right_tcp_delta_y",
            "right_tcp_delta_z",
        ),
        target_components=(
            "right_tcp_delta_x",
            "right_tcp_delta_y",
            "right_tcp_delta_z",
            "right_gripper_delta",
        ),
        configured_initial_log_std=-0.5,
        configured_final_log_std=-2.5,
        new_action_log_std_cap=-1.5,
    )

    np.testing.assert_allclose(initial, (-2.5, -2.4, -3.0, -1.5))
    np.testing.assert_allclose(final, (-2.5, -2.5, -3.0, -2.5))


@pytest.mark.optional
def test_temp_training_rejects_effective_action_credit_rewrite() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.config import PPOConfig
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    assert build_arg_parser().parse_args([]).effective_action_for_ppo is False
    args = build_arg_parser().parse_args(["--effective-action-for-ppo"])
    assert args.effective_action_for_ppo is True
    with pytest.raises(ValueError, match="sampled latent action"):
        PPOConfig(effective_action_for_ppo=True)


@pytest.mark.optional
def test_temp_training_ppo_numerical_stability_controls_are_explicit() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.config import PPOConfig
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args([])
    assert args.target_kl == pytest.approx(0.03)
    assert args.max_log_ratio == pytest.approx(10.0)
    assert args.advantage_clip == pytest.approx(10.0)
    assert args.actor_reference_mse_coef == pytest.approx(0.0)
    config = PPOConfig()
    assert config.to_dict()["target_kl"] == pytest.approx(0.03)
    with pytest.raises(ValueError, match="finite and positive"):
        PPOConfig(target_kl=0.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        PPOConfig(actor_reference_mse_coef=-1.0)


@pytest.mark.optional
def test_actor_reference_mse_uses_only_deployable_actor_outputs() -> None:
    import copy

    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic
    from sim.mujoco_temp_pipeline.ppo import compute_actor_reference_mse

    torch.manual_seed(17)
    reference = ActorCritic(
        observation_dim=31,
        action_dim=4,
        hidden_dim=16,
        proprio_dim=31,
        critic_observation_dim=31,
        separate_critic_backbone=True,
        environment_action_dim=7,
        active_action_indices=(0, 1, 2, 6),
    )
    policy = copy.deepcopy(reference)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    observations = torch.randn((8, 31))
    zero_loss = compute_actor_reference_mse(
        policy,
        reference,
        observations,
    )
    assert float(zero_loss.detach()) == pytest.approx(0.0, abs=1e-10)

    with torch.no_grad():
        policy.actor_mean.bias.add_(0.1)
    loss = compute_actor_reference_mse(policy, reference, observations)
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert policy.actor_mean.bias.grad is not None
    assert all(parameter.grad is None for parameter in reference.parameters())


@pytest.mark.optional
def test_temp_training_rgb_health_guard_is_explicit() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.train import build_arg_parser

    args = build_arg_parser().parse_args(["--require-rgb-health"])
    assert args.require_rgb_health is True


def test_generic_gripper_reward_config_uses_surface_contract_without_observation_drift() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PRIVILEGED_GENERIC_GRIPPER_TASK_CONFIG)
    assert config.observation_dim == 31
    assert len(config.gripper_contact_surface_specs) == 2
    assert {spec.side for spec in config.gripper_contact_surface_specs} == {
        "left",
        "right",
    }
    assert config.reward_shaping["side_contact_reward_scale"] == pytest.approx(0.0)
    assert config.reward_shaping["top_bottom_contact_penalty"] == pytest.approx(0.0)
    assert config.reward_shaping["inner_contact_reward_scale"] == pytest.approx(0.8)
    assert config.reward_shaping["outer_contact_penalty"] == pytest.approx(1.6)


def test_contact_gated_profile_removes_precontact_closure_credit() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PRIVILEGED_CONTACT_GATED_TASK_CONFIG)
    shaping = config.reward_shaping
    assert shaping["gripper_reward_scale"] == pytest.approx(0.0)
    assert shaping["gripper_progress_scale"] == pytest.approx(0.0)
    assert shaping["gripper_command_reward_scale"] == pytest.approx(0.0)
    assert shaping["grasp_acquire_bonus"] == pytest.approx(0.0)
    assert shaping["grasp_region_reward_scale"] == pytest.approx(0.0)
    assert shaping["gripper_reward_requires_contact"] is True
    assert shaping["grasp_acquire_requires_contact"] is True
    assert shaping["contact_reward_onset_only"] is True
    assert shaping["require_generic_contact_for_lift"] is True


def test_contact_progress_profile_uses_potential_shaping_without_hover_credit() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PRIVILEGED_CONTACT_PROGRESS_TASK_CONFIG)
    shaping = config.reward_shaping
    assert shaping["grasp_region_reward_scale"] == pytest.approx(0.0)
    assert shaping["grasp_region_progress_scale"] == pytest.approx(3.0)
    assert shaping["gripper_reward_scale"] == pytest.approx(0.0)
    assert shaping["gripper_progress_scale"] == pytest.approx(0.0)


def test_contact_balanced_profile_removes_far_reach_and_soft_safety_dominance() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PRIVILEGED_CONTACT_BALANCED_TASK_CONFIG)
    shaping = config.reward_shaping
    assert shaping["reach_reward_scale"] == pytest.approx(0.0)
    assert shaping["safety_projection_penalty"] == pytest.approx(0.03)
    assert shaping["premature_close_state_penalty"] == pytest.approx(0.08)
    assert shaping["outer_contact_penalty"] == pytest.approx(0.50)
    assert shaping["approach_alignment_reward_scale"] == pytest.approx(0.15)
    assert config.scripted_pregrasp_distance_m == pytest.approx(0.05)


def test_secure_lift_profile_preserves_grip_force_after_contact() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(PRIVILEGED_SECURE_LIFT_TASK_CONFIG)
    assert config.post_contact_phase_enabled is True
    assert config.post_contact_phase_min_measured_close == pytest.approx(0.55)
    assert config.post_contact_phase_loss_grace_steps == 3
    assert config.post_contact_action_scale == pytest.approx(
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert config.post_contact_action_bias[6] == pytest.approx(0.5)


def test_full_grasp_bridge_and_stage1_share_normal_action_contract() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    bridge = CubeLiftTaskConfig.load(NORMAL_PREGRASP_BRIDGE_TASK_CONFIG)
    full = CubeLiftTaskConfig.load(NORMAL_FULL_GRASP_STAGE1_V4_TASK_CONFIG)
    expected_actions = tuple(range(7))
    assert bridge.policy_class == "normal"
    assert bridge.observation_dim == 28
    assert bridge.policy_active_action_indices == expected_actions
    assert bridge.scripted_pregrasp_reset is True
    assert bridge.post_contact_phase_enabled is False
    assert full.observation_dim == bridge.observation_dim
    assert full.policy_active_action_indices == expected_actions
    assert full.scripted_pregrasp_reset is False
    assert full.post_contact_phase_enabled is False
    assert full.reward_shaping["gripper_reward_requires_contact"] is False
    assert full.reward_shaping["inner_contact_reward_scale"] == pytest.approx(0.60)


def test_pregrasp_bridge_v2_uses_proximity_gated_closure() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(NORMAL_PREGRASP_BRIDGE_V2_TASK_CONFIG)
    assert config.observation_dim == 28
    assert config.policy_active_action_indices == tuple(range(7))
    assert config.scripted_pregrasp_reset is True
    assert config.post_contact_phase_enabled is False
    assert config.reward_shaping["gripper_reward_requires_contact"] is False
    assert config.reward_shaping["grasp_acquire_requires_contact"] is True
    assert config.right_tcp_delta_m == pytest.approx((0.006, 0.006, 0.006))


def test_pregrasp_bridge_v3_has_contact_owned_lift_phase() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(NORMAL_PREGRASP_BRIDGE_V3_TASK_CONFIG)
    assert config.observation_dim == 28
    assert config.policy_active_action_indices == tuple(range(7))
    assert config.scripted_pregrasp_reset is True
    assert config.post_contact_phase_enabled is True
    assert config.post_contact_phase_activation_steps == 5
    assert config.post_contact_action_scale == pytest.approx(
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    )
    assert config.post_contact_action_bias[6] == pytest.approx(0.5)
    assert config.post_contact_nonnegative_lift_action is True


def test_pregrasp_bridge_v4_supplies_positive_lift_and_grip_bias() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(NORMAL_PREGRASP_BRIDGE_V4_TASK_CONFIG)
    assert config.observation_dim == 28
    assert config.policy_active_action_indices == tuple(range(7))
    assert config.post_contact_phase_enabled is True
    assert config.post_contact_action_scale == pytest.approx(
        (0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0)
    )
    assert config.post_contact_action_bias == pytest.approx(
        (0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.5)
    )
    assert config.post_contact_nonnegative_lift_action is True


def test_pregrasp_bridge_v5_adds_cube_pose_and_doubles_pose_step() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_model_input_binding,
        build_observation_contract,
    )

    config = CubeLiftTaskConfig.load(NORMAL_PREGRASP_BRIDGE_V5_CUBE_POSE_TASK_CONFIG)
    contract = build_observation_contract(config)
    assert config.proprio_observation_dim == 35
    assert config.observation_dim == 35
    assert config.normal_proprio_profile.endswith("cube_pose.v1")
    assert config.observation_layout.endswith("last_action7_cube_pose7.v1")
    assert "episode_progress" not in contract.field_names
    assert config.right_tcp_delta_m == pytest.approx((0.012, 0.012, 0.012))
    assert config.right_tcp_rotation_delta_rad == pytest.approx(
        (0.0001, 0.0001, 0.0001)
    )
    assert config.right_gripper_delta == pytest.approx(0.1)
    assert config.wrist_rgb_enabled is False
    assert build_model_input_binding(config, contract).expected_flat_dim(contract) == 35
    shaping = config.reward_shaping
    assert shaping["center_proximity_limit_m"] == pytest.approx(0.03)
    assert shaping["center_proximity_penalty"] == pytest.approx(8.0)
    assert shaping["gripper_reward_requires_contact"] is True
    assert shaping["safety_projection_penalty"] == pytest.approx(2.0)
    assert shaping["outer_contact_penalty"] == pytest.approx(2.0)


@pytest.mark.optional
def test_post_contact_phase_exits_after_configured_contact_loss_grace() -> None:
    _require_mujoco_runtime()
    from sim.mujoco.tasks.cube_lift.fast_env import FastCubeLiftEnv

    env = FastCubeLiftEnv(
        PRIVILEGED_SECURE_LIFT_TASK_CONFIG,
        include_training_reward_state=True,
        use_model_cache=False,
    )
    try:
        env.reset(seed=1)
        env.kernel._post_contact_phase_active = True
        for _ in range(2):
            env.kernel._update_post_contact_phase()
            assert env.kernel._post_contact_phase_active is True
        env.kernel._update_post_contact_phase()
        assert env.kernel._post_contact_phase_active is False
    finally:
        env.close()


def test_normal_contact_profile_preserves_identified_free_motion_actuator() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    normal = CubeLiftTaskConfig.load(DEFAULT_PHYSICAL_TASK_CONFIG)
    expert = CubeLiftTaskConfig.load(PRIVILEGED_EXPERT_TASK_CONFIG)
    assert normal.policy_class == "normal"
    assert normal.observation_mode == "proprioceptive_real_motion_tracking_no_episode_clock"
    assert normal.observation_dim == 28
    assert normal.grasp_mode == "physical_contact"
    assert expert.policy_class == "privileged_expert"
    assert expert.observation_mode == "state_privileged"
    assert expert.observation_dim == 31
    assert expert.grasp_mode == "contact_latch_debug"
    assert normal.position_actuation_profile_name == expert.position_actuation_profile_name
    assert normal.position_actuation_profile_sha256 == expert.position_actuation_profile_sha256
    normal_gripper = normal.gripper_actuation_profile()
    expert_gripper = expert.gripper_actuation_profile()
    assert normal_gripper.name != expert_gripper.name
    assert (
        normal_gripper.normalized_position_profile().to_metadata()
        == expert_gripper.normalized_position_profile().to_metadata()
    )
    assert normal_gripper.position_kp_n_m == expert_gripper.position_kp_n_m
    assert normal_gripper.force_limit_n == expert_gripper.force_limit_n
    assert normal_gripper.contact_tuning.post_settle_geom_friction_by_name[
        "debug_cube_collision"
    ] == pytest.approx((1.0, 0.05, 0.005))


def test_rgb_policy_configs_append_only_active_wrist_camera() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    normal = CubeLiftTaskConfig.load(NORMAL_RGB_TASK_CONFIG)
    expert = CubeLiftTaskConfig.load(PRIVILEGED_EXPERT_RGB_TASK_CONFIG)
    assert normal.wrist_rgb_enabled is True
    assert normal.wrist_rgb_camera_name == "right_wrist"
    assert normal.wrist_rgb_shape == (3, 32, 32)
    assert normal.observation_dim == 15 + 3 * 32 * 32
    assert expert.observation_dim == 31 + 3 * 32 * 32
    assert normal.action_layout == expert.action_layout
    assert normal.position_actuation_profile_sha256 == expert.position_actuation_profile_sha256
    assert normal.gripper_actuation_profile_sha256 == expert.gripper_actuation_profile_sha256


def test_current_rgb_policy_configs_use_versioned_48_by_48_input() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    normal = CubeLiftTaskConfig.load(NORMAL_RGB_48_TASK_CONFIG)
    expert = CubeLiftTaskConfig.load(PRIVILEGED_EXPERT_RGB_48_TASK_CONFIG)
    assert normal.wrist_rgb_shape == (3, 48, 48)
    assert expert.wrist_rgb_shape == (3, 48, 48)
    assert normal.observation_dim == 15 + 3 * 48 * 48
    assert expert.observation_dim == 31 + 3 * 48 * 48
    assert normal.observation_layout.endswith("right_wrist_rgb_chw_3x48x48.v2")
    assert expert.observation_layout.endswith("right_wrist_rgb_chw_3x48x48.v2")


def test_normal_rgb_v3_uses_real_observable_motion_tracking_contract() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    config = CubeLiftTaskConfig.load(NORMAL_RGB_MOTION_TRACKING_TASK_CONFIG)
    assert config.policy_class == "normal"
    assert config.normal_proprio_profile == (
        "qpos_close_motion_tracking_action_progress.v1"
    )
    assert config.proprio_observation_dim == 29
    assert config.observation_dim == 29 + 3 * 48 * 48
    assert config.wrist_rgb_spatial_pool_size == 3
    assert config.observation_layout.endswith(
        "right_wrist_rgb_chw_3x48x48.v3"
    )
    assert config.reward_shaping["staged_grasp"] is True


def test_motion_tracking_observation_spec_has_no_privileged_cube_fields() -> None:
    from sim.mujoco.tasks.cube_lift.spec import CubeLiftObservationSpec

    spec = CubeLiftObservationSpec(
        layout_version=(
            "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
            "right_target_error6_right_gripper_target_error1_last_action7_progress1_"
            "right_wrist_rgb_chw_3x48x48.v3"
        )
    )
    assert spec.proprio_dim == 29
    assert "cube_to_right_grasp_x_m" not in spec.field_names
    assert spec.wrist_rgb_shape == (3, 48, 48)
    assert spec.dim == 29 + 3 * 48 * 48


def test_normal_config_rejects_debug_grasp_latch() -> None:
    import json

    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    payload = json.loads(DEFAULT_PHYSICAL_TASK_CONFIG.read_text())
    payload["grasp_mode"] = "contact_latch_debug"
    with pytest.raises(ValueError, match="cannot use the debug grasp latch"):
        CubeLiftTaskConfig.from_dict(payload)


def test_temp_reward_shaping_preserves_raw_reward_when_disabled() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    next_observations = observations.copy()
    raw_rewards = np.asarray((1.0, -0.5), dtype=np.float32)
    result, terms = shape_rewards(
        observations,
        next_observations,
        raw_rewards,
        RewardShapingConfig(enabled=False),
    )
    np.testing.assert_array_equal(result, raw_rewards)
    assert terms["distance_m"].shape == (2,)


def test_temp_reward_shaping_rewards_reaching_and_near_closure() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = np.zeros((1, 31), dtype=np.float32)
    previous[0, 20] = 0.10
    current[0, 20] = 0.02
    current[0, 6] = 1.0
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert shaped[0] > 0.0
    assert terms["reach_progress_term"][0] > 0.0
    assert terms["gripper_term"][0] > 0.0
    assert terms["grasp_region_term"][0] > 0.0
    assert terms["grasp_gate"][0] > 0.5


def test_temp_dense_reward_penalizes_closing_far_from_cube() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 20] = 0.30
    current[:, 20] = 0.30
    current[1, 6] = 1.0
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["grasp_gate"][1] < 1e-4
    assert terms["premature_close_term"][1] < 0.0
    assert shaped[1] < shaped[0]


def test_temp_dense_reach_reward_has_long_range_ordering() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((3, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 20] = (0.40, 0.20, 0.05)
    current[:, 20] = (0.40, 0.20, 0.05)
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(3, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["reach_term"][0] > 0.05
    assert shaped[0] < shaped[1] < shaped[2]


def test_temp_dense_reach_targets_pregrasp_height_above_cube() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    # cube - grasp z: -0.03 means the grasp frame is 30 mm above the cube.
    previous[:, 22] = (-0.03, 0.0)
    current[:, 22] = (-0.03, 0.0)
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(grasp_height_offset_m=0.03),
    )
    assert terms["vertical_error_m"].tolist() == pytest.approx((0.0, 0.03))
    assert shaped[0] > shaped[1]


def test_temp_dense_reward_prioritizes_grasped_lift() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 9] = 0.025
    current[:, 9] = 0.065
    previous[:, 20] = 0.02
    current[:, 20] = 0.02
    current[0, 6] = 1.0
    shaped, terms = shape_rewards(
        previous,
        current,
        np.asarray((100.0, 100.0), dtype=np.float32),
        RewardShapingConfig(),
        resting_cube_z_m=np.asarray((0.025, 0.025), dtype=np.float32),
        lift_delta_m=0.08,
    )
    assert terms["lift_fraction"].tolist() == pytest.approx((0.5, 0.5))
    assert terms["lift_term"][0] > 0.0
    assert terms["lift_progress_term"][0] > 0.0
    assert terms["lift_term"][1] == pytest.approx(0.0)
    assert shaped[0] > shaped[1]
    # Enabled dense shaping does not inherit the task's repeatedly accumulated
    # absolute-height raw reward.
    zero_raw, _ = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(),
        resting_cube_z_m=np.asarray((0.025, 0.025), dtype=np.float32),
        lift_delta_m=0.08,
    )
    np.testing.assert_allclose(shaped, zero_raw)


def test_temp_dense_reward_credits_closing_progress_near_grasp() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = previous.copy()
    previous[0, 22] = -0.03
    current[0, 22] = -0.027
    current[0, 6] = 0.2
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["gripper_progress_term"][0] > 0.0
    assert shaped[0] > 0.0


def test_temp_dense_reward_does_not_move_reach_target_when_gripper_closes() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    # Both samples are at the same 30 mm pre-grasp pose; only the second closes.
    previous[:, 22] = -0.03
    current[:, 22] = -0.03
    current[1, 6] = 0.8
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(),
    )
    assert terms["distance_m"][0] == pytest.approx(terms["distance_m"][1])
    assert terms["reach_term"][0] == pytest.approx(terms["reach_term"][1])
    assert terms["gripper_term"][1] > terms["gripper_term"][0]
    assert shaped[1] > shaped[0]


def test_temp_grasp_region_progress_rewards_approach_and_penalizes_retreat() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((3, 31), dtype=np.float32)
    current = previous.copy()
    # With the default offset, the closed-grasp distance is smallest at
    # cube_to_grasp=(0, 0, -0.025). Keep z fixed and vary only x: sample 0
    # approaches, sample 1 retreats, and sample 2 stays still.
    previous[:, 22] = -0.025
    current[:, 22] = -0.025
    previous[0, 20], current[0, 20] = 0.06, 0.03
    previous[1, 20], current[1, 20] = 0.03, 0.06
    previous[2, 20], current[2, 20] = 0.03, 0.03
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        grasp_region_progress_scale=3.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        latch_reward_scale=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(3, dtype=np.float32),
        config,
    )
    assert terms["grasp_region_progress_term"][0] > 0.0
    assert terms["grasp_region_progress_term"][1] < 0.0
    assert terms["grasp_region_progress_term"][2] == pytest.approx(0.0)
    np.testing.assert_allclose(
        shaped, terms["grasp_region_progress_term"], atol=1e-6
    )


def test_temp_precontact_close_state_is_penalized_but_contact_is_not() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    current[:, 6] = 0.75
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        grasp_region_progress_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        premature_close_state_penalty=0.2,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        latch_reward_scale=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        inner_contact_score=np.asarray((0.0, 1.0), dtype=np.float32),
        bilateral_inner_contact_score=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert terms["premature_close_state_term"].tolist() == pytest.approx(
        (-0.15, 0.0)
    )
    assert shaped.tolist() == pytest.approx((-0.15, 0.0))


def test_temp_center_proximity_penalty_blocks_closed_insertion_without_inner_grasp() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 20] = 0.04
    current[:, 22] = 0.01
    current[:, 6] = 0.75
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        grasp_region_progress_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        premature_close_state_penalty=0.0,
        center_proximity_limit_m=0.03,
        center_proximity_penalty=8.0,
        center_proximity_close_threshold=0.35,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        latch_reward_scale=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        inner_contact_score=np.asarray((0.0, 1.0), dtype=np.float32),
        bilateral_inner_contact_score=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert terms["center_distance_m"].tolist() == pytest.approx((0.01, 0.01))
    assert terms["center_proximity_term"][0] < -3.0
    assert terms["center_proximity_term"][1] == pytest.approx(0.0)
    assert shaped[0] < -3.0
    assert shaped[1] == pytest.approx(0.0)


def test_temp_approach_alignment_rewards_accepted_safe_direction() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        grasp_region_progress_scale=0.0,
        approach_alignment_reward_scale=0.15,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        premature_close_state_penalty=0.0,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        latch_reward_scale=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    accepted = np.asarray(((1.0, 0.0, 0.0, 0, 0, 0, 0), (-1.0, 0.0, 0.0, 0, 0, 0, 0)), dtype=np.float32)
    teacher = np.asarray(((1.0, 0.0, 0.0, 0, 0, 0, 0), (1.0, 0.0, 0.0, 0, 0, 0, 0)), dtype=np.float32)
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        accepted_action=accepted,
        approach_action=teacher,
    )
    assert terms["approach_alignment_term"].tolist() == pytest.approx((0.05, 0.0))
    assert shaped.tolist() == pytest.approx((0.05, 0.0))


def test_temp_dense_reward_has_explicit_success_bonus() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observation = np.zeros((2, 31), dtype=np.float32)
    shaped, terms = shape_rewards(
        observation,
        observation,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(success_bonus=10.0),
        success=np.asarray((False, True)),
    )
    assert terms["success_term"].tolist() == pytest.approx((0.0, 10.0))
    assert shaped[1] - shaped[0] == pytest.approx(10.0)


def test_staged_reward_opens_close_gate_before_strict_lift_gate() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = previous.copy()
    previous[0, 20] = 0.12
    current[0, 20] = 0.08
    current[0, 6] = 0.5
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(
            staged_grasp=True,
            grasp_acquire_gate=0.0,
            grasp_acquire_bonus=1.0,
        ),
    )
    assert terms["close_gate"][0] > terms["grasp_gate"][0]
    assert terms["reach_phase_weight"][0] < 1.0
    assert terms["grasp_acquire_term"][0] == pytest.approx(1.0)
    assert terms["gripper_progress_term"][0] > 0.0
    assert shaped[0] > terms["gripper_term"][0]


def test_contact_aware_reward_removes_close_hover_and_rewards_bilateral_acquisition() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 9] = 0.025
    current[:, 9] = 0.065
    previous[:, 22] = -0.025
    current[:, 22] = -0.025
    previous[:, 6] = 0.8
    current[:, 6] = 0.8
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        grasp_acquire_bonus=0.0,
        bilateral_contact_reward_scale=0.05,
        bilateral_contact_acquire_bonus=2.0,
        require_bilateral_contact_for_lift_reward=True,
        latch_reward_scale=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
        time_limit_penalty=5.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        resting_cube_z_m=0.025,
        bilateral_contact=np.asarray((0.0, 1.0), dtype=np.float32),
        previous_bilateral_contact=np.zeros(2, dtype=np.float32),
        time_limit=np.asarray((1.0, 0.0), dtype=np.float32),
    )
    assert terms["gripper_term"].tolist() == pytest.approx((0.0, 0.0))
    assert terms["lift_term"][0] == pytest.approx(0.0)
    assert terms["lift_term"][1] > 0.0
    assert terms["bilateral_contact_acquire_term"].tolist() == pytest.approx(
        (0.0, 2.0)
    )
    assert terms["time_limit_term"].tolist() == pytest.approx((-5.0, 0.0))
    assert shaped[1] > shaped[0]


def test_lift_curriculum_penalizes_missing_contact_and_ungated_drop() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 9] = 0.065
    current[:, 9] = 0.025
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        grasp_acquire_bonus=0.0,
        bilateral_contact_reward_scale=0.0,
        bilateral_contact_acquire_bonus=0.0,
        bilateral_contact_missing_penalty=0.1,
        require_bilateral_contact_for_lift_reward=True,
        ungated_negative_lift_progress=True,
        latch_reward_scale=0.0,
        lift_reward_scale=0.0,
        lift_progress_scale=100.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        resting_cube_z_m=0.025,
        lift_delta_m=0.08,
        bilateral_contact=np.asarray((1.0, 0.0), dtype=np.float32),
        previous_bilateral_contact=np.ones(2, dtype=np.float32),
    )
    assert terms["lift_progress_term"].tolist() == pytest.approx((-50.0, -50.0))
    assert terms["bilateral_contact_missing_term"].tolist() == pytest.approx(
        (0.0, -0.1)
    )
    assert shaped.tolist() == pytest.approx((-50.0, -50.1))


def test_failure_reward_makes_unsafe_termination_worse_than_timeout() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    state = np.zeros((3, 31), dtype=np.float32)
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        premature_close_penalty=0.0,
        grasp_acquire_bonus=0.0,
        latch_reward_scale=0.0,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
        time_limit_penalty=0.0,
        failure_termination_penalty=5.0,
        unsafe_termination_penalty=5.0,
    )
    shaped, terms = shape_rewards(
        state,
        state,
        np.zeros(3, dtype=np.float32),
        config,
        time_limit=np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
        failure_termination=np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        unsafe_termination=np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
    )
    assert terms["time_limit_term"].tolist() == pytest.approx((0.0, 0.0, 0.0))
    assert terms["failure_termination_term"].tolist() == pytest.approx(
        (0.0, 0.0, -5.0)
    )
    assert terms["unsafe_termination_term"].tolist() == pytest.approx(
        (0.0, 0.0, -5.0)
    )
    assert shaped.tolist() == pytest.approx((0.0, 0.0, -10.0))


@pytest.mark.optional
def test_compute_gae_bootstraps_time_limit_but_not_true_terminal() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.ppo import compute_gae

    rewards = np.zeros((1, 2), dtype=np.float32)
    values = np.zeros_like(rewards)
    bootstrap_values = np.full_like(rewards, 10.0)
    terminated = np.asarray(((0.0, 1.0),), dtype=np.float32)
    episode_done = np.ones_like(rewards)
    advantages, returns = compute_gae(
        rewards,
        values,
        bootstrap_values,
        terminated,
        episode_done,
        gamma=0.99,
        gae_lambda=0.95,
    )
    assert advantages[0].tolist() == pytest.approx((9.9, 0.0))
    np.testing.assert_array_equal(returns, advantages)


def test_temp_reward_shaping_penalizes_collision_projection() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    shaped, terms = shape_rewards(
        observations,
        observations,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(enabled=False, safety_projection_penalty=0.25),
        safety_projection_alpha=np.asarray((1.0, 0.0), dtype=np.float32),
    )
    assert shaped[0] == pytest.approx(0.0)
    assert shaped[1] == pytest.approx(-0.25)
    assert terms["safety_projection_term"].tolist() == pytest.approx((0.0, -0.25))


def test_temp_reward_charges_any_collision_safety_intervention() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    shaped, terms = shape_rewards(
        observations,
        observations,
        np.zeros(2, dtype=np.float32),
        RewardShapingConfig(
            enabled=False,
            safety_projection_penalty=0.0,
            safety_intervention_penalty=0.4,
        ),
        safety_projection_alpha=np.ones(2, dtype=np.float32),
        safety_intervention=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert shaped.tolist() == pytest.approx((0.0, -0.4))
    assert terms["safety_intervention_term"].tolist() == pytest.approx(
        (0.0, -0.4)
    )


def test_temp_reward_can_weight_semantic_collision_categories() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((3, 31), dtype=np.float32)
    config = RewardShapingConfig(
        enabled=False,
        safety_projection_penalty=0.0,
        safety_collision_category_penalties={
            "robot_self": 0.3,
            "gripper_environment": 0.1,
        },
    )
    shaped, terms = shape_rewards(
        observations,
        observations,
        np.zeros(3, dtype=np.float32),
        config,
        safety_projection_alpha=np.ones(3, dtype=np.float32),
        safety_intervention=np.ones(3, dtype=np.float32),
        safety_collision_category=np.asarray(
            ("robot_self", "gripper_environment", None),
            dtype=object,
        ),
    )
    assert shaped.tolist() == pytest.approx((-0.3, -0.1, 0.0))
    assert terms["safety_collision_category_term"].tolist() == pytest.approx(
        (-0.3, -0.1, 0.0)
    )
    with pytest.raises(ValueError, match="Unknown safety collision categories"):
        RewardShapingConfig(
            safety_collision_category_penalties={"not_a_category": 1.0}
        )


def test_temp_reward_command_credit_uses_accepted_gripper_action() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = previous.copy()
    previous[0, 20] = -0.03
    current[0, 20] = -0.03
    current[0, 6] = 0.8
    current[0, 29] = 1.0  # requested gripper command in the legacy state.
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        RewardShapingConfig(
            staged_grasp=True,
            gripper_command_reward_scale=1.0,
            reach_reward_scale=0.0,
            reach_progress_scale=0.0,
            grasp_region_reward_scale=0.0,
            gripper_reward_scale=0.0,
            gripper_progress_scale=0.0,
            premature_close_penalty=0.0,
            grasp_acquire_bonus=0.0,
            bilateral_contact_reward_scale=0.0,
            bilateral_contact_acquire_bonus=0.0,
            action_penalty=0.0,
            safety_projection_penalty=0.0,
        ),
        accepted_action=np.zeros((1, 7), dtype=np.float32),
    )
    assert terms["gripper_command_term"][0] == pytest.approx(0.0)
    assert shaped[0] == pytest.approx(0.0)


def test_contact_gated_reward_does_not_pay_for_precontact_closure() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((1, 31), dtype=np.float32)
    current = previous.copy()
    previous[0, 20] = -0.03
    current[0, 20] = -0.03
    current[0, 6] = 0.8
    config = RewardShapingConfig(
        staged_grasp=True,
        gripper_reward_scale=1.0,
        gripper_progress_scale=2.0,
        gripper_command_reward_scale=3.0,
        premature_close_penalty=0.5,
        gripper_reward_requires_contact=True,
        grasp_acquire_requires_contact=True,
        grasp_acquire_bonus=4.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(1, dtype=np.float32),
        config,
        bilateral_contact=np.zeros(1, dtype=np.float32),
        inner_contact_score=np.zeros(1, dtype=np.float32),
        bilateral_inner_contact_score=np.zeros(1, dtype=np.float32),
        accepted_action=np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )
    assert terms["gripper_term"][0] == pytest.approx(0.0)
    assert terms["gripper_progress_term"][0] == pytest.approx(0.0)
    assert terms["gripper_command_term"][0] == pytest.approx(0.0)
    assert terms["grasp_acquire_term"][0] == pytest.approx(0.0)
    assert terms["premature_close_term"][0] < 0.0
    assert shaped[0] < 0.0


def test_contact_reward_is_onset_only_and_acquire_requires_bilateral_inner_contact() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 20] = -0.03
    current[:, 20] = -0.03
    previous[:, 6] = (0.2, 0.5)
    current[:, 6] = 0.5
    config = RewardShapingConfig(
        staged_grasp=True,
        inner_contact_reward_scale=1.0,
        bilateral_inner_contact_reward_scale=2.0,
        contact_reward_onset_only=True,
        grasp_acquire_requires_contact=True,
        grasp_acquire_bonus=3.0,
    )
    shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        bilateral_contact=np.ones(2, dtype=np.float32),
        previous_bilateral_contact=np.zeros(2, dtype=np.float32),
        inner_contact_score=np.asarray((0.8, 0.8), dtype=np.float32),
        bilateral_inner_contact_score=np.asarray((0.8, 0.8), dtype=np.float32),
        previous_inner_contact_score=np.asarray((0.0, 0.8), dtype=np.float32),
        previous_bilateral_inner_contact_score=np.asarray((0.0, 0.8), dtype=np.float32),
    )
    assert terms["inner_contact_term"].tolist() == pytest.approx((0.8, 0.0))
    assert terms["bilateral_inner_contact_term"].tolist() == pytest.approx((1.6, 0.0))
    assert terms["grasp_acquire_term"].tolist() == pytest.approx((3.0, 0.0))
    assert shaped[0] > shaped[1]


def test_contact_missing_penalty_only_applies_after_previous_contact() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    state = np.zeros((2, 31), dtype=np.float32)
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        bilateral_contact_missing_penalty=1.0,
        contact_missing_requires_previous=True,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    _shaped, terms = shape_rewards(
        state,
        state,
        np.zeros(2, dtype=np.float32),
        config,
        bilateral_contact=np.zeros(2, dtype=np.float32),
        previous_bilateral_contact=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert terms["bilateral_contact_missing_term"].tolist() == pytest.approx(
        (0.0, -1.0)
    )


def test_generic_contact_is_required_for_lift_confidence() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    previous = np.zeros((2, 31), dtype=np.float32)
    current = previous.copy()
    previous[:, 9] = 0.025
    current[:, 9] = 0.065
    previous[:, 20] = 0.02
    current[:, 20] = 0.02
    current[:, 6] = 0.8
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        require_bilateral_contact_for_lift_reward=True,
        require_generic_contact_for_lift=True,
        lift_reward_scale=5.0,
        lift_progress_scale=100.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    _shaped, terms = shape_rewards(
        previous,
        current,
        np.zeros(2, dtype=np.float32),
        config,
        resting_cube_z_m=np.asarray((0.025, 0.025), dtype=np.float32),
        bilateral_contact=np.ones(2, dtype=np.float32),
        inner_contact_score=np.asarray((0.8, 0.8), dtype=np.float32),
        bilateral_inner_contact_score=np.asarray((0.0, 0.8), dtype=np.float32),
    )
    assert terms["lift_term"][0] == pytest.approx(0.0)
    assert terms["lift_term"][1] > 0.0


def test_cube_contact_position_classifies_side_and_top_bottom_faces() -> None:
    from sim.mujoco.tasks.cube_lift.contacts import (
        classify_cube_contact_position,
    )

    center = np.zeros(3, dtype=np.float32)
    half_extents = np.full(3, 0.015, dtype=np.float32)
    identity = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)
    side = classify_cube_contact_position(
        np.asarray((0.015, 0.0, 0.0)), center, identity, half_extents
    )
    top = classify_cube_contact_position(
        np.asarray((0.0, 0.0, 0.015)), center, identity, half_extents
    )
    bottom = classify_cube_contact_position(
        np.asarray((0.0, 0.0, -0.015)), center, identity, half_extents
    )
    assert side == pytest.approx((1.0, 0.0))
    assert top == pytest.approx((0.0, 1.0))
    assert bottom == pytest.approx((0.0, 1.0))


def test_temp_reward_prefers_side_contact_over_top_bottom_contact() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        gripper_command_reward_scale=0.0,
        premature_close_penalty=0.0,
        grasp_acquire_bonus=0.0,
        bilateral_contact_reward_scale=0.0,
        bilateral_contact_acquire_bonus=0.0,
        bilateral_contact_missing_penalty=0.0,
        side_contact_reward_scale=1.0,
        top_bottom_contact_penalty=2.0,
        latch_reward_scale=0.0,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        success_bonus=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    shaped, terms = shape_rewards(
        observations,
        observations,
        np.zeros(2, dtype=np.float32),
        config,
        side_contact_score=np.asarray((1.0, 0.0), dtype=np.float32),
        top_bottom_contact_score=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert shaped.tolist() == pytest.approx((1.0, -2.0))
    assert terms["side_contact_term"].tolist() == pytest.approx((1.0, 0.0))
    assert terms["top_bottom_contact_penalty_term"].tolist() == pytest.approx(
        (0.0, -2.0)
    )


def test_generic_gripper_contact_classifies_inner_and_outer_half_spaces() -> None:
    from sim.mujoco.grasp_contact import (
        GripperContactSurfaceSpec,
        classify_gripper_contact_point,
    )

    spec = GripperContactSurfaceSpec(
        body_name="finger",
        side="left",
        inner_normal_local=(-1.0, 0.0, 0.0),
        face_margin_m=0.001,
        score_width_m=0.002,
    )
    body_position = np.zeros(3, dtype=np.float32)
    identity = np.eye(3, dtype=np.float32)
    inner = classify_gripper_contact_point(
        np.asarray((-0.005, 0.0, 0.0)), body_position, identity, spec
    )
    outer = classify_gripper_contact_point(
        np.asarray((0.005, 0.0, 0.0)), body_position, identity, spec
    )
    assert inner[0] > 0.8 and inner[1] < 0.1
    assert outer[1] > 0.8 and outer[0] < 0.1


def test_generic_gripper_bilateral_score_requires_all_configured_sides() -> None:
    from types import SimpleNamespace

    from sim.mujoco.grasp_contact import (
        GripperContactSurfaceSpec,
        extract_gripper_contact_scores,
    )

    specs = (
        GripperContactSurfaceSpec(
            body_name="left_finger",
            side="left",
            inner_normal_local=(-1.0, 0.0, 0.0),
        ),
        GripperContactSurfaceSpec(
            body_name="right_finger",
            side="right",
            inner_normal_local=(1.0, 0.0, 0.0),
        ),
    )
    model = SimpleNamespace(geom_bodyid=np.asarray((99, 1, 2), dtype=np.int32))
    data = SimpleNamespace(
        ncon=1,
        contact=[SimpleNamespace(geom1=0, geom2=1, pos=np.asarray((-0.005, 0.0, 0.0)))],
        xpos=np.zeros((100, 3), dtype=np.float32),
        xmat=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (100, 1)),
    )
    one_sided = extract_gripper_contact_scores(
        model,
        data,
        object_body_id=99,
        surface_specs=specs,
        surface_body_ids={"left_finger": 1, "right_finger": 2},
    )
    assert one_sided.inner_contact_score > 0.8
    assert one_sided.bilateral_inner_contact_score == pytest.approx(0.0)

    data.contact.append(
        SimpleNamespace(geom1=0, geom2=2, pos=np.asarray((0.005, 0.0, 0.0)))
    )
    data.ncon = 2
    two_sided = extract_gripper_contact_scores(
        model,
        data,
        object_body_id=99,
        surface_specs=specs,
        surface_body_ids={"left_finger": 1, "right_finger": 2},
    )
    assert two_sided.bilateral_inner_contact_score > 0.8


def test_temp_reward_prefers_generic_inner_contact_and_penalizes_outer_contact() -> None:
    from sim.mujoco_temp_pipeline.config import RewardShapingConfig
    from sim.mujoco_temp_pipeline.reward import shape_rewards

    observations = np.zeros((2, 31), dtype=np.float32)
    config = RewardShapingConfig(
        reach_reward_scale=0.0,
        reach_progress_scale=0.0,
        grasp_region_reward_scale=0.0,
        gripper_reward_scale=0.0,
        gripper_progress_scale=0.0,
        gripper_command_reward_scale=0.0,
        premature_close_penalty=0.0,
        grasp_acquire_bonus=0.0,
        bilateral_contact_reward_scale=0.0,
        bilateral_contact_acquire_bonus=0.0,
        bilateral_contact_missing_penalty=0.0,
        inner_contact_reward_scale=1.0,
        bilateral_inner_contact_reward_scale=1.0,
        outer_contact_penalty=2.0,
        latch_reward_scale=0.0,
        lift_reward_scale=0.0,
        lift_progress_scale=0.0,
        success_bonus=0.0,
        action_penalty=0.0,
        safety_projection_penalty=0.0,
    )
    shaped, terms = shape_rewards(
        observations,
        observations,
        np.zeros(2, dtype=np.float32),
        config,
        inner_contact_score=np.asarray((1.0, 0.0), dtype=np.float32),
        bilateral_inner_contact_score=np.asarray((1.0, 0.0), dtype=np.float32),
        outer_contact_score=np.asarray((0.0, 1.0), dtype=np.float32),
    )
    assert shaped.tolist() == pytest.approx((2.0, -2.0))
    assert terms["inner_contact_term"].tolist() == pytest.approx((1.0, 0.0))
    assert terms["bilateral_inner_contact_term"].tolist() == pytest.approx(
        (1.0, 0.0)
    )
    assert terms["outer_contact_penalty_term"].tolist() == pytest.approx(
        (0.0, -2.0)
    )


@pytest.mark.optional
def test_temp_policy_checkpoint_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_policy_from_checkpoint,
        save_checkpoint,
    )
    from sim.mujoco_temp_pipeline.config import PPOConfig, RewardShapingConfig
    from sim.mujoco_temp_pipeline.normalization import RunningMeanStd
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    torch.manual_seed(4)
    policy = ActorCritic(hidden_dim=16, separate_critic_backbone=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    normalizer = RunningMeanStd.create((31,))
    normalizer.update(np.ones((3, 31), dtype=np.float32))
    path = save_checkpoint(
        tmp_path / "policy.pt",
        policy=policy,
        optimizer=optimizer,
        observation_normalizer=normalizer,
        ppo_config=PPOConfig(
            total_timesteps=16,
            num_envs=2,
            rollout_steps=8,
            update_epochs=1,
            minibatch_size=16,
            hidden_dim=16,
            critic_backbone_mode="separate",
        ),
        reward_config=RewardShapingConfig(),
        global_step=16,
        update=1,
        task_config_path=TASK_CONFIG,
        policy_initialization={
            "type": "goal_delta_actor_bootstrap",
            "actor_uses_privileged": False,
        },
    )
    restored, restored_normalizer, payload = build_policy_from_checkpoint(path)
    observation = torch.zeros((2, 31))
    with torch.no_grad():
        expected = policy.act(observation, deterministic=True)[0]
        actual = restored.act(observation, deterministic=True)[0]
    torch.testing.assert_close(actual, expected)
    np.testing.assert_allclose(restored_normalizer.mean, normalizer.mean)
    assert payload["global_step"] == 16
    assert restored.separate_critic_backbone is True
    assert payload["policy"]["separate_critic_backbone"] is True
    assert payload["policy_initialization"] == {
        "type": "goal_delta_actor_bootstrap",
        "actor_uses_privileged": False,
    }


@pytest.mark.optional
def test_task_config_mismatch_requires_explicit_diagnostic_override() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
    )

    payload = {"task_config_sha256": "0" * 64}
    with pytest.raises(ValueError, match="legacy collision diagnostics"):
        check_task_config_compatibility(payload, TASK_CONFIG)

    result = check_task_config_compatibility(
        payload,
        TASK_CONFIG,
        allow_mismatch=True,
    )
    assert result["task_config_mismatch"] is True
    assert result["checkpoint_task_config_sha256"] == "0" * 64


@pytest.mark.optional
def test_cube_pose_checkpoint_migration_preserves_old_actor() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
        expand_cube_pose_policy_state,
        scale_pose_action_head_for_doubled_step,
    )
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    old_config = CubeLiftTaskConfig.load(NORMAL_PREGRASP_BRIDGE_V4_TASK_CONFIG)
    old_metadata = build_learning_metadata(old_config)
    payload = {
        "task_config_sha256": "0" * 64,
        "task_metadata": {
            "action_layout": old_config.action_layout,
            "observation_layout": old_config.observation_layout,
            "learning_contract": old_metadata,
        },
    }
    result = check_task_config_compatibility(
        payload,
        NORMAL_PREGRASP_BRIDGE_V5_CUBE_POSE_TASK_CONFIG,
        allow_mismatch=True,
        allow_cube_pose_observation_expansion=True,
    )
    assert result["cube_pose_observation_expansion"] is True

    torch.manual_seed(7)
    source = ActorCritic(28, 7, 16, proprio_dim=28, environment_action_dim=7)
    target = ActorCritic(35, 7, 16, proprio_dim=35, environment_action_dim=7)
    expand_cube_pose_policy_state(source, target)
    torch.testing.assert_close(
        target.backbone[0].weight[:, :28], source.backbone[0].weight
    )
    assert torch.count_nonzero(target.backbone[0].weight[:, 28:]) == 0
    arm_mean_before = target.actor_mean.weight[:6].detach().clone()
    grip_mean_before = target.actor_mean.weight[6].detach().clone()
    scale_pose_action_head_for_doubled_step(target)
    torch.testing.assert_close(target.actor_mean.weight[:6], arm_mean_before * 0.25)
    torch.testing.assert_close(target.actor_mean.weight[6], grip_mean_before)


@pytest.mark.optional
def test_legacy_collision_diagnostic_cli_flags_are_explicit() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.evaluate import build_arg_parser as eval_parser
    from sim.mujoco_temp_pipeline.play import build_arg_parser as play_parser

    assert eval_parser().parse_args([]).allow_task_config_mismatch is False
    play_args = play_parser().parse_args(
        ["--allow-task-config-mismatch", "--print-safety-events"]
    )
    assert play_args.allow_task_config_mismatch is True
    assert play_args.print_safety_events is True


@pytest.mark.optional
def test_temp_evaluation_action_offsets_are_explicit() -> None:
    pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.evaluate import build_arg_parser

    args = build_arg_parser().parse_args(
        ["--action-offsets=-0.1,0.0,0.1", "--stochastic"]
    )
    assert args.action_offsets == pytest.approx((-0.1, 0.0, 0.1))
    assert args.stochastic is True


@pytest.mark.optional
def test_temp_policy_actions_and_log_probs_are_finite() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(hidden_dim=16)
    observation = torch.zeros((5, 31))
    action, log_prob, value = policy.act(observation)
    evaluated_log_prob, entropy, evaluated_value = policy.evaluate_actions(
        observation, action
    )
    assert action.shape == (5, 7)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    torch.testing.assert_close(log_prob, evaluated_log_prob)
    torch.testing.assert_close(value, evaluated_value)
    assert torch.isfinite(entropy).all()


@pytest.mark.optional
def test_asymmetric_critic_is_optional_and_cannot_leak_into_actor_backbone() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(
        observation_dim=28,
        action_dim=2,
        hidden_dim=16,
        critic_observation_dim=31,
    )
    actor_observation = torch.zeros((4, 28))
    privileged_critic_observation = torch.ones((4, 31))

    actor_action, actor_log_prob = policy.predict_action(actor_observation)
    assert actor_action.shape == (4, 2)
    assert actor_log_prob.shape == (4,)
    with pytest.raises(ValueError, match="asymmetric critic"):
        policy.act(actor_observation)

    action, log_prob, value = policy.act(
        actor_observation,
        critic_observation=privileged_critic_observation,
    )
    assert action.shape == (4, 2)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)

    policy.zero_grad(set_to_none=True)
    policy.value(
        actor_observation,
        critic_observation=privileged_critic_observation,
    ).sum().backward()
    assert all(parameter.grad is None for parameter in policy.backbone.parameters())
    assert any(
        parameter.grad is not None
        for parameter in policy.critic_backbone.parameters()
    )


@pytest.mark.optional
def test_separate_actor_observation_critic_cannot_update_actor_backbone() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(
        observation_dim=28,
        action_dim=2,
        hidden_dim=16,
        separate_critic_backbone=True,
    )
    observation = torch.randn((8, 28))
    before = policy.predict_action(observation, deterministic=True)[0].detach()

    policy.zero_grad(set_to_none=True)
    policy.value(observation).square().mean().backward()
    assert all(parameter.grad is None for parameter in policy.backbone.parameters())
    assert all(parameter.grad is None for parameter in policy.actor_mean.parameters())
    assert any(
        parameter.grad is not None
        for parameter in policy.critic_backbone.parameters()
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer.step()
    after = policy.predict_action(observation, deterministic=True)[0].detach()
    torch.testing.assert_close(after, before)


@pytest.mark.optional
def test_temp_ppo_value_loss_is_clipped_around_rollout_prediction() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.ppo import compute_clipped_value_loss

    new_values = torch.tensor((0.5, 3.0), dtype=torch.float32)
    old_values = torch.zeros(2, dtype=torch.float32)
    returns = torch.ones(2, dtype=torch.float32)

    clipped_loss, clip_fraction = compute_clipped_value_loss(
        new_values,
        old_values,
        returns,
        clip_coef=0.2,
    )
    legacy_loss, legacy_clip_fraction = compute_clipped_value_loss(
        new_values,
        old_values,
        returns,
        clip_coef=0.0,
    )
    assert float(clipped_loss) == pytest.approx(1.16)
    assert float(clip_fraction) == pytest.approx(1.0)
    assert float(legacy_loss) == pytest.approx(1.0625)
    assert float(legacy_clip_fraction) == pytest.approx(0.0)


@pytest.mark.optional
def test_separate_actor_and_critic_gradients_are_clipped_independently() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic
    from sim.mujoco_temp_pipeline.ppo import clip_actor_critic_grad_norms

    policy = ActorCritic(
        observation_dim=28,
        action_dim=2,
        hidden_dim=8,
        separate_critic_backbone=True,
    )
    actor_parameters = policy.actor_parameters()
    critic_parameters = policy.critic_parameters()
    assert not {id(parameter) for parameter in actor_parameters}.intersection(
        id(parameter) for parameter in critic_parameters
    )
    for parameter in actor_parameters:
        parameter.grad = torch.ones_like(parameter)
    for parameter in critic_parameters:
        parameter.grad = torch.full_like(parameter, 100.0)

    actor_norm, critic_norm = clip_actor_critic_grad_norms(
        policy,
        max_grad_norm=0.5,
    )
    assert float(critic_norm) > float(actor_norm) > 0.5

    def gradient_norm(parameters: tuple[torch.nn.Parameter, ...]) -> float:
        return sum(
            float(parameter.grad.detach().square().sum())
            for parameter in parameters
            if parameter.grad is not None
        ) ** 0.5

    assert gradient_norm(actor_parameters) == pytest.approx(0.5, abs=1e-5)
    assert gradient_norm(critic_parameters) == pytest.approx(0.5, abs=1e-5)


@pytest.mark.optional
def test_privileged_critic_checkpoint_keeps_actor_inference_self_contained(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata
    from sim.mujoco_temp_pipeline.checkpoint import (
        build_policy_from_checkpoint,
        save_checkpoint,
    )
    from sim.mujoco_temp_pipeline.config import PPOConfig, RewardShapingConfig
    from sim.mujoco_temp_pipeline.normalization import RunningMeanStd
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    task = CubeLiftTaskConfig.load(REACH_FIXED_POSE_XYZ_PROPRIO_TASK_CONFIG)
    learning = build_learning_metadata(task)
    policy = ActorCritic(
        observation_dim=int(learning["model_input_dim"]),
        action_dim=task.policy_action_dim,
        hidden_dim=16,
        critic_observation_dim=31,
        environment_action_dim=7,
        active_action_indices=task.policy_active_action_indices,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    actor_normalizer = RunningMeanStd.create((28,))
    critic_normalizer = RunningMeanStd.create((31,))
    path = save_checkpoint(
        tmp_path / "asymmetric.pt",
        policy=policy,
        optimizer=optimizer,
        observation_normalizer=actor_normalizer,
        critic_observation_normalizer=critic_normalizer,
        ppo_config=PPOConfig(critic_observation_mode="privileged_training_state"),
        reward_config=RewardShapingConfig(),
        global_step=0,
        update=0,
        task_config_path=REACH_FIXED_POSE_XYZ_PROPRIO_TASK_CONFIG,
    )
    restored, restored_normalizer, payload = build_policy_from_checkpoint(path)
    assert restored.observation_dim == 28
    assert restored.critic_observation_dim == 31
    assert restored_normalizer.mean.shape == (28,)
    assert payload["critic_observation_normalizer"] is not None
    with torch.no_grad():
        action, _log_prob = restored.predict_action(
            torch.zeros((2, 28)),
            deterministic=True,
        )
    assert action.shape == (2, 3)


@pytest.mark.optional
def test_temp_policy_can_reset_only_the_critic_head() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(hidden_dim=16)
    actor_before = policy.actor_mean.weight.detach().clone()
    critic_before = policy.critic.weight.detach().clone()
    policy.reset_critic()
    torch.testing.assert_close(policy.actor_mean.weight, actor_before)
    assert not torch.equal(policy.critic.weight, critic_before)
    torch.testing.assert_close(policy.critic.bias, torch.zeros_like(policy.critic.bias))


@pytest.mark.optional
def test_temp_policy_expands_compact_action_into_public_layout() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(
        action_dim=1,
        environment_action_dim=7,
        active_action_indices=(2,),
        trainable_log_std=False,
        hidden_dim=16,
    )
    observation = torch.zeros((5, 31))
    action, log_prob, value = policy.act(observation)
    expanded = policy.expand_action(action)
    assert action.shape == (5, 1)
    assert expanded.shape == (5, 7)
    torch.testing.assert_close(expanded[:, 2], action[:, 0])
    torch.testing.assert_close(
        expanded[:, (0, 1, 3, 4, 5, 6)],
        torch.zeros((5, 6)),
    )
    assert log_prob.shape == (5,)
    assert value.shape == (5,)
    assert policy.log_std.requires_grad is False


@pytest.mark.optional
def test_named_action_component_transfer_preserves_new_actor_rows() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.checkpoint import (
        transfer_policy_action_components,
    )
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    source = ActorCritic(
        observation_dim=35,
        action_dim=3,
        environment_action_dim=7,
        active_action_indices=(0, 1, 2),
        hidden_dim=16,
    )
    target = ActorCritic(
        observation_dim=35,
        action_dim=4,
        environment_action_dim=7,
        active_action_indices=(0, 1, 2, 6),
        hidden_dim=16,
    )
    with torch.no_grad():
        source.actor_mean.weight.copy_(
            torch.arange(48, dtype=torch.float32).reshape(3, 16)
        )
        source.actor_mean.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))
        source.log_std.copy_(torch.tensor([-0.1, -0.2, -0.3]))
    new_gripper_weight = target.actor_mean.weight[3].detach().clone()
    new_gripper_bias = target.actor_mean.bias[3].detach().clone()
    shared = transfer_policy_action_components(
        source,
        target,
        source_components=(
            "right_tcp_delta_x",
            "right_tcp_delta_y",
            "right_tcp_delta_z",
        ),
        target_components=(
            "right_tcp_delta_x",
            "right_tcp_delta_y",
            "right_tcp_delta_z",
            "right_gripper_delta",
        ),
    )
    assert shared == (
        "right_tcp_delta_x",
        "right_tcp_delta_y",
        "right_tcp_delta_z",
    )
    torch.testing.assert_close(
        target.actor_mean.weight[:3], source.actor_mean.weight
    )
    torch.testing.assert_close(target.actor_mean.bias[:3], source.actor_mean.bias)
    torch.testing.assert_close(target.log_std[:3], source.log_std)
    torch.testing.assert_close(target.actor_mean.weight[3], new_gripper_weight)
    torch.testing.assert_close(target.actor_mean.bias[3], new_gripper_bias)


@pytest.mark.optional
def test_temp_rgb_policy_uses_cnn_and_keeps_action_contract() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    policy = ActorCritic(
        observation_dim=15 + 3 * 32 * 32,
        action_dim=7,
        hidden_dim=32,
        proprio_dim=15,
        image_shape=(3, 32, 32),
    )
    observation = torch.zeros((4, 15 + 3 * 32 * 32))
    action, log_prob, value = policy.act(observation, deterministic=True)
    assert action.shape == (4, 7)
    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert policy.image_shape == (3, 32, 32)


@pytest.mark.optional
def test_temp_rgb_v3_policy_preserves_spatial_features() -> None:
    torch = pytest.importorskip("torch")
    from sim.mujoco_temp_pipeline.policy import ActorCritic

    observation_dim = 29 + 3 * 48 * 48
    policy = ActorCritic(
        observation_dim=observation_dim,
        action_dim=7,
        hidden_dim=32,
        proprio_dim=29,
        image_shape=(3, 48, 48),
        image_pool_size=3,
    )
    observation = torch.zeros((2, observation_dim))
    action, log_prob, value = policy.act(observation, deterministic=True)
    assert action.shape == (2, 7)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert policy.image_pool_size == 3
    assert policy.image_encoder[-2].in_features == 64 * 3 * 3


def test_temp_run_recorder_writes_machine_and_human_readable_outputs(
    tmp_path: Path,
) -> None:
    import csv
    import json

    from sim.mujoco_temp_pipeline.records import RunRecorder

    recorder = RunRecorder(
        tmp_path,
        run_name="smoke run",
        config={"ppo": {"learning_rate": 3e-4}, "device": "cpu"},
    )
    metrics = {
        "update": 1,
        "global_step": 16,
        "wall_time_seconds": 0.5,
        "transitions_per_second": 32.0,
        "raw_reward_mean": 0.1,
        "shaped_reward_mean": 0.2,
        "completed_episodes": 1,
        "successful_episodes": 1,
        "success_rate": 1.0,
        "policy_loss": -0.01,
        "value_loss": 0.5,
        "entropy": 2.0,
        "approx_kl": 0.001,
        "clip_fraction": 0.0,
        "policy_std_mean": 0.6,
        "learning_rate": 3e-4,
    }
    recorder.record_update(
        metrics,
        (
            {
                "env_id": 0,
                "length": 10,
                "raw_return": 1.0,
                "shaped_return": 2.0,
                "success": True,
                "termination_reason": "height_only_lift_success",
            },
        ),
    )
    recorder.finish({"global_step": 16, "success_rate": 1.0})

    assert json.loads((recorder.directory / "run_config.json").read_text())["device"] == "cpu"
    assert json.loads(recorder.metrics_jsonl.read_text().splitlines()[0])["global_step"] == 16
    assert json.loads(recorder.episodes_jsonl.read_text().splitlines()[0])["success"] is True
    with recorder.metrics_csv.open(newline="") as handle:
        assert next(csv.DictReader(handle))["policy_loss"] == "-0.01"
    assert json.loads(recorder.summary_json.read_text())["success_rate"] == 1.0
    report = recorder.report_html.read_text(encoding="utf-8")
    assert "Cube-Lift temporary PPO" in report
    assert '"global_step": 16' in report
