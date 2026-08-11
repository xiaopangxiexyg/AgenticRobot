import numpy as np
import pytest

from sim.mujoco.learning import (
    EmbodimentRewardConfig,
    ModelInputBinding,
    ObservationContract,
    ObservationFieldSpec,
    ObservationRole,
    PolicyActionBinding,
    PolicyActionTransform,
    compose_embodiment_reward,
    compose_reward,
    dry_run_model_io,
    encode_model_input,
    PolicyModelManifest,
)


def test_policy_action_transform_keeps_latent_and_task_actions_distinct() -> None:
    policy = PolicyActionBinding(
        "latent.xyz.v1",
        "continuous",
        3,
        "right_base",
        "relative",
    )
    task = PolicyActionBinding(
        "task.pose.v1",
        "continuous",
        7,
        "right_base",
        "relative",
    )
    transform = PolicyActionTransform(
        transform_id="latent_xyz_to_pose7.v1",
        policy_action_binding_hash=policy.semantic_content_hash,
        task_action_binding_hash=task.semantic_content_hash,
        policy_components=("x", "y", "z"),
        task_components=("x", "y", "z", "rx", "ry", "rz", "gripper"),
        task_defaults=(0.0,) * 7,
    )
    transform.validate_bindings(policy, task)
    latent = np.asarray([[0.25, -0.5, 0.75]], dtype=np.float32)
    mapped = transform.transform(latent)
    np.testing.assert_allclose(mapped[:, :3], latent)
    np.testing.assert_allclose(mapped[:, 3:], 0.0)
    assert transform.policy_dim == 3
    assert transform.task_dim == 7


def test_policy_action_transform_applies_hashed_affine_scaling() -> None:
    policy = PolicyActionBinding(
        "latent.xy.v1", "continuous", 2, "right_base", "relative"
    )
    task = PolicyActionBinding(
        "task.pose.v1", "continuous", 7, "right_base", "relative"
    )
    identity = PolicyActionTransform(
        transform_id="latent_xy_to_pose7.v1",
        policy_action_binding_hash=policy.semantic_content_hash,
        task_action_binding_hash=task.semantic_content_hash,
        policy_components=("x", "y"),
        task_components=("x", "y", "z", "rx", "ry", "rz", "gripper"),
        task_defaults=(0.0,) * 7,
    )
    scaled = PolicyActionTransform(
        transform_id="latent_xy_to_pose7.affine.v1",
        policy_action_binding_hash=policy.semantic_content_hash,
        task_action_binding_hash=task.semantic_content_hash,
        policy_components=("x", "y"),
        task_components=("x", "y", "z", "rx", "ry", "rz", "gripper"),
        task_defaults=(0.0,) * 7,
        policy_scales=(0.2, 0.5),
        policy_biases=(0.1, 0.0),
    )
    latent = np.asarray((1.0, -1.0), dtype=np.float32)
    np.testing.assert_allclose(scaled.transform(latent)[:2], (0.3, -0.5))
    assert scaled.semantic_content_hash != identity.semantic_content_hash


def _contract() -> ObservationContract:
    return ObservationContract(
        contract_id="fixture.normal.v1",
        fields=(
            ObservationFieldSpec("proprio", shape=(2,), source="rm65_state"),
            ObservationFieldSpec(
                "wrist_rgb", shape=(3, 4, 4), encoding="image_chw", source="wrist_camera"
            ),
            ObservationFieldSpec(
                "object_pose", shape=(3,), role=ObservationRole.PRIVILEGED, source="sim_scene"
            ),
            ObservationFieldSpec(
                "contact_debug", shape=(1,), role=ObservationRole.REWARD_ONLY, source="sim_contact"
            ),
        ),
    )


def test_normal_binding_rejects_privileged_and_reward_only_fields() -> None:
    contract = _contract()
    normal = ModelInputBinding(
        "normal",
        contract.semantic_content_hash,
        ("proprio", "wrist_rgb"),
    )
    normal.validate(contract)
    assert normal.expected_flat_dim(contract) == 50

    with pytest.raises(ValueError, match="privileged"):
        ModelInputBinding(
            "leaky",
            contract.semantic_content_hash,
            ("proprio", "object_pose"),
        ).validate(contract)
    with pytest.raises(ValueError, match="reward_only"):
        ModelInputBinding(
            "leaky_reward",
            contract.semantic_content_hash,
            ("proprio", "contact_debug"),
        ).validate(contract)


def test_model_input_encoding_is_ordered_and_supports_a_batch() -> None:
    contract = _contract()
    binding = ModelInputBinding(
        "normal",
        contract.semantic_content_hash,
        ("proprio", "wrist_rgb"),
    )
    observation = {
        "proprio": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "wrist_rgb": np.ones((2, 3, 4, 4), dtype=np.uint8) * 7,
    }
    encoded = encode_model_input(observation, contract, binding)
    assert encoded.flat.shape == (2, 50)
    np.testing.assert_allclose(encoded.flat[0, :2], [1.0, 2.0])
    np.testing.assert_allclose(encoded.flat[0, 2:], 7.0)
    assert encoded.fields["wrist_rgb"].dtype == np.float32


def test_binding_hash_changes_when_order_or_metadata_changes() -> None:
    contract = _contract()
    first = ModelInputBinding(
        "normal", contract.semantic_content_hash, ("proprio", "wrist_rgb")
    )
    reordered = ModelInputBinding(
        "normal", contract.semantic_content_hash, ("wrist_rgb", "proprio")
    )
    changed = ModelInputBinding(
        "normal",
        contract.semantic_content_hash,
        ("proprio", "wrist_rgb"),
        metadata={"camera_resize": "bilinear"},
    )
    assert first.semantic_content_hash != reordered.semantic_content_hash
    assert first.semantic_content_hash != changed.semantic_content_hash


def test_action_binding_is_explicit_about_pose_semantics() -> None:
    binding = PolicyActionBinding(
        binding_id="rm65.right.pose.v1",
        action_space="continuous",
        output_dim=7,
        frame="right_base",
        mode="relative",
    )
    assert binding.semantic_content_hash
    with pytest.raises(ValueError, match="output_dim"):
        PolicyActionBinding("bad", "continuous", 0, "right_base", "relative")


def test_model_manifest_and_dry_run_fail_closed_on_contract_or_batch_drift() -> None:
    contract = _contract()
    binding = ModelInputBinding(
        "normal", contract.semantic_content_hash, ("proprio", "wrist_rgb")
    )
    action = PolicyActionBinding(
        binding_id="rm65.right.pose.v1",
        action_space="continuous",
        output_dim=7,
        frame="right_base",
        mode="relative",
    )
    manifest = PolicyModelManifest(
        model_id="unit-normal",
        artifact_sha256="a" * 64,
        observation_contract_hash=contract.semantic_content_hash,
        model_input_binding_hash=binding.semantic_content_hash,
        action_binding_hash=action.semantic_content_hash,
        input_dim=50,
        output_dim=7,
    )
    observation = {
        "proprio": np.zeros((2, 2), dtype=np.float32),
        "wrist_rgb": np.zeros((2, 3, 4, 4), dtype=np.uint8),
    }
    receipt = dry_run_model_io(
        manifest,
        contract,
        binding,
        action,
        observation,
        np.zeros((2, 7), dtype=np.float32),
    )
    assert receipt.accepted is True
    assert receipt.input_shape == (2, 50)
    assert receipt.output_shape == (2, 7)
    with pytest.raises(ValueError, match="batched model input"):
        dry_run_model_io(
            manifest,
            contract,
            binding,
            action,
            observation,
            np.zeros(7, dtype=np.float32),
        )
    with pytest.raises(ValueError, match="contract mismatch"):
        PolicyModelManifest(
            **{
                **manifest.to_dict(),
                "observation_contract_hash": "bad",
            }
        ).validate(contract, binding, action)


def test_cube_lift_reward_state_is_named_without_changing_legacy_layout() -> None:
    from sim.mujoco.tasks.cube_lift.reward_adapter import CubeLiftRewardState

    values = np.arange(62, dtype=np.float32).reshape(2, 31)
    state = CubeLiftRewardState.from_flat(values)
    assert state.right_qpos.shape == (2, 6)
    assert state.right_gripper_close.tolist() == [6.0, 37.0]
    assert state.cube_position[0].tolist() == [7.0, 8.0, 9.0]
    assert state.last_action[1].tolist() == list(range(54, 61))
    assert state.episode_progress.tolist() == [30.0, 61.0]


def test_generic_reward_is_separate_from_task_terms() -> None:
    embodiment = compose_embodiment_reward(
        EmbodimentRewardConfig(
            action_effort_scale=1.0,
            action_delta_scale=0.5,
            tracking_error_scale=2.0,
            safety_intervention_scale=3.0,
        ),
        action=np.asarray([1.0, -1.0]),
        previous_action=np.zeros(2),
        tracking_error=np.asarray([0.1, -0.1]),
        safety_intervention=1.0,
    )
    assert embodiment.task_terms == {}
    assert embodiment.total < 0.0
    total = compose_reward(embodiment, task_terms={"task_progress": 0.75})
    assert total.task_terms == {"task_progress": 0.75}
    assert total.total == pytest.approx(embodiment.total + 0.75)


def test_cube_lift_fixture_projection_is_lossless_and_has_a_firewall() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import (
        build_action_binding,
        build_learning_metadata,
        build_model_input_binding,
        build_observation_contract,
        project_flat_observation,
    )

    normal = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json"
    )
    contract = build_observation_contract(normal)
    binding = build_model_input_binding(normal, contract)
    action = build_action_binding(normal)
    flat = np.concatenate(
        (
            np.linspace(-0.2, 0.2, normal.proprio_observation_dim, dtype=np.float32),
            np.linspace(0.0, 1.0, normal.observation_dim - normal.proprio_observation_dim, dtype=np.float32),
        )
    )
    projected = project_flat_observation(normal, flat)
    encoded = encode_model_input(projected, contract, binding)
    np.testing.assert_allclose(encoded.flat, flat)
    assert action.output_dim == normal.action_dim
    assert action.mode == "relative"
    assert all(field.role is ObservationRole.ACTOR for field in contract.fields)
    assert "episode_progress" in binding.input_fields
    assert contract.field("episode_progress").source == "runtime_episode_clock"
    metadata = build_learning_metadata(normal)
    assert metadata["observation_contract_hash"] == contract.semantic_content_hash
    assert metadata["model_manifest_version"]
    assert metadata["model_input_dim"] == normal.observation_dim
    assert metadata["model_output_dim"] == 7
    assert metadata["action_dim"] == 7
    assert metadata["policy_action_dim"] == 7
    assert metadata["task_action_dim"] == 7

    expert = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_cube_lift_rl_privileged_expert_rgb_v2.json"
    )
    expert_contract = build_observation_contract(expert)
    expert_binding = build_model_input_binding(expert, expert_contract)
    assert expert_binding.allow_privileged is True
    assert expert_contract.privileged_fields

    no_clock = CubeLiftTaskConfig.load(
        "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v4.json"
    )
    no_clock_contract = build_observation_contract(no_clock)
    no_clock_binding = build_model_input_binding(no_clock, no_clock_contract)
    assert no_clock.proprio_observation_dim == 28
    assert no_clock.observation_dim == 28 + 3 * 48 * 48
    assert "episode_progress" not in no_clock_contract.field_names
    assert no_clock_binding.input_fields == ("proprio", "wrist_rgb")
    no_clock_flat = np.concatenate(
        (
            np.arange(no_clock.proprio_observation_dim, dtype=np.float32),
            np.linspace(
                0.0,
                1.0,
                no_clock.observation_dim - no_clock.proprio_observation_dim,
                dtype=np.float32,
            ),
        )
    )
    no_clock_projected = project_flat_observation(no_clock, no_clock_flat)
    no_clock_encoded = encode_model_input(
        no_clock_projected, no_clock_contract, no_clock_binding
    )
    np.testing.assert_allclose(no_clock_encoded.flat, no_clock_flat)


def test_checkpoint_compatibility_checks_learning_contract_hashes() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
        file_sha256,
    )

    path = "sim/configs/tasks/rm65_right_cube_lift_rl_normal_rgb_v3.json"
    config = CubeLiftTaskConfig.load(path)
    payload = {
        "task_config_sha256": file_sha256(path),
        "task_metadata": {
            "action_layout": config.action_layout,
            "observation_layout": config.observation_layout,
            "learning_contract": build_learning_metadata(config),
        },
    }
    result = check_task_config_compatibility(payload, path)
    assert result["task_config_mismatch"] is False
    payload["task_metadata"]["learning_contract"]["action_binding_hash"] = "bad"
    with pytest.raises(ValueError, match="Learning contract"):
        check_task_config_compatibility(payload, path)


def test_checkpoint_compatibility_recognizes_legacy_compact_action_metadata() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
        task_config_sha256,
    )

    path = "sim/configs/tasks/rm65_right_cube_lift_rl_normal_contact_curriculum_v3.json"
    config = CubeLiftTaskConfig.load(path)
    learning = build_learning_metadata(config)
    legacy_learning = dict(learning)
    legacy_learning["action_binding_hash"] = learning[
        "task_action_binding_hash"
    ]
    legacy_learning["model_output_dim"] = config.action_dim
    for key in (
        "policy_action_layout",
        "policy_action_components",
        "policy_action_transform_id",
        "policy_action_transform_hash",
        "task_action_binding_id",
        "task_action_binding_hash",
        "task_action_layout",
        "task_action_components",
        "policy_action_dim",
        "task_action_dim",
    ):
        legacy_learning.pop(key, None)
    payload = {
        "task_config_sha256": task_config_sha256(path),
        "policy": {
            "action_dim": 1,
            "environment_action_dim": 7,
            "active_action_indices": [2],
        },
        "task_metadata": {
            "action_layout": config.action_layout,
            "observation_layout": config.observation_layout,
            "learning_contract": legacy_learning,
        },
    }
    result = check_task_config_compatibility(payload, path)
    assert result["legacy_compact_action_contract_migration"] is True


def test_checkpoint_compatibility_allows_explicit_named_action_transfer() -> None:
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from sim.mujoco.tasks.cube_lift.learning import build_learning_metadata
    from sim.mujoco_temp_pipeline.checkpoint import (
        check_task_config_compatibility,
        task_config_sha256,
    )

    source_path = "sim/configs/tasks/rm65_right_cube_reach_pose_xyz_v1.json"
    target_path = (
        "sim/configs/tasks/"
        "rm65_right_cube_contact_acquire_pose_xyz_gripper_v1.json"
    )
    source = CubeLiftTaskConfig.load(source_path)
    payload = {
        "task_config_sha256": task_config_sha256(source_path),
        "policy": {
            "action_dim": source.policy_action_dim,
            "environment_action_dim": source.action_dim,
            "active_action_indices": list(source.policy_active_action_indices),
        },
        "task_metadata": {
            "action_layout": source.action_layout,
            "observation_layout": source.observation_layout,
            "learning_contract": build_learning_metadata(source),
        },
    }
    result = check_task_config_compatibility(
        payload,
        target_path,
        allow_mismatch=True,
        allow_policy_action_component_transfer=True,
    )
    transfer = result["policy_action_component_transfer"]
    assert transfer is not None
    assert transfer["shared_components"] == (
        "right_tcp_delta_x",
        "right_tcp_delta_y",
        "right_tcp_delta_z",
    )
    assert transfer["new_components"] == ("right_gripper_delta",)
