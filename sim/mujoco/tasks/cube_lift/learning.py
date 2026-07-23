"""Cube-Lift adapter for the backend-neutral learning contracts.

This module is a parity fixture, not a second observation implementation. It
only names the vector and image slices already produced by ``CubeLiftKernel``
and records the normal/expert admission policy around them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.mujoco.learning import (
    MODEL_MANIFEST_VERSION,
    ModelInputBinding,
    ObservationContract,
    ObservationFieldSpec,
    ObservationRole,
    PolicyActionBinding,
)

from .config import (
    NORMAL_POLICY_CLASS,
    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
    CubeLiftTaskConfig,
)


def _uses_episode_clock(config: CubeLiftTaskConfig) -> bool:
    return not (
        config.policy_class == NORMAL_POLICY_CLASS
        and config.normal_proprio_profile
        == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2
    )


def build_observation_contract(config: CubeLiftTaskConfig) -> ObservationContract:
    """Describe the existing Cube-Lift policy vector without changing it."""

    if config.policy_class == NORMAL_POLICY_CLASS:
        state_name = "proprio"
        state_role = ObservationRole.ACTOR
        state_source = "rm65_measured_and_command_state"
    else:
        state_name = "privileged_state"
        state_role = ObservationRole.PRIVILEGED
        state_source = "sim_scene_and_execution_debug"
    uses_episode_clock = _uses_episode_clock(config)
    # Historical vectors keep episode_progress as their final scalar. Keep
    # that position in the model input when present, but name it separately:
    # it is a runtime clock feature, not an RM65 sensor measurement.
    state_dim = config.proprio_observation_dim - int(uses_episode_clock)
    fields = [
        ObservationFieldSpec(
            state_name,
            shape=(state_dim,),
            role=state_role,
            source=state_source,
        )
    ]
    if uses_episode_clock:
        fields.append(
            ObservationFieldSpec(
                "episode_progress",
                shape=(1,),
                role=ObservationRole.ACTOR,
                source="runtime_episode_clock",
                unit="fraction",
            )
        )
    if config.wrist_rgb_enabled:
        fields.append(
            ObservationFieldSpec(
                "wrist_rgb",
                shape=config.wrist_rgb_shape,
                role=ObservationRole.ACTOR,
                encoding="image_chw",
                source=f"camera:{config.wrist_rgb_camera_name}",
                normalization="unit_interval",
            )
        )
    return ObservationContract(
        contract_id=f"dexteleop.cube_lift.observation:{config.observation_layout}",
        fields=tuple(fields),
    )


def build_model_input_binding(
    config: CubeLiftTaskConfig,
    contract: ObservationContract | None = None,
) -> ModelInputBinding:
    """Build the normal or privileged binding selected by ``config``."""

    resolved_contract = build_observation_contract(config) if contract is None else contract
    state_name = "proprio" if config.policy_class == NORMAL_POLICY_CLASS else "privileged_state"
    if _uses_episode_clock(config):
        input_fields = (
            (state_name, "episode_progress", "wrist_rgb")
            if config.wrist_rgb_enabled
            else (state_name, "episode_progress")
        )
    else:
        input_fields = (
            (state_name, "wrist_rgb") if config.wrist_rgb_enabled else (state_name,)
        )
    binding = ModelInputBinding(
        binding_id=f"dexteleop.cube_lift.model_input:{config.observation_layout}",
        observation_contract_hash=resolved_contract.semantic_content_hash,
        input_fields=input_fields,
        allow_privileged=config.policy_class != NORMAL_POLICY_CLASS,
        metadata={
            "observation_layout": config.observation_layout,
            "proprio_profile": config.normal_proprio_profile,
            "wrist_rgb_camera_name": config.wrist_rgb_camera_name
            if config.wrist_rgb_enabled
            else None,
            "wrist_rgb_shape": config.wrist_rgb_shape,
        },
    )
    binding.validate(resolved_contract)
    return binding


def build_action_binding(config: CubeLiftTaskConfig) -> PolicyActionBinding:
    """Declare the existing seven-dimensional pose/qpos action semantics."""

    return PolicyActionBinding(
        binding_id=f"dexteleop.cube_lift.action:{config.action_layout}",
        action_space="continuous",
        output_dim=config.action_dim,
        frame=(
            f"{config.controlled_side}_base"
            if config.uses_pose_action
            else f"{config.controlled_side}_joint"
        ),
        mode="relative",
        normalized=True,
        metadata={
            "action_layout": config.action_layout,
            "pose_control_frame": config.pose_control_frame,
        },
    )


def project_flat_observation(
    config: CubeLiftTaskConfig,
    observation: Any,
) -> dict[str, np.ndarray]:
    """Split an existing kernel observation into binding-named fields.

    The final axis is the established flat observation axis; any preceding
    axes are preserved as a batch shape for vectorized PPO and parity tests.
    """

    values = np.asarray(observation, dtype=np.float32)
    if values.ndim == 0 or values.shape[-1] != config.observation_dim:
        raise ValueError(
            f"Cube-Lift observation must end in {config.observation_dim} values; got {values.shape}"
        )
    uses_episode_clock = _uses_episode_clock(config)
    state_dim = config.proprio_observation_dim - int(uses_episode_clock)
    state_name = "proprio" if config.policy_class == NORMAL_POLICY_CLASS else "privileged_state"
    result: dict[str, np.ndarray] = {state_name: np.ascontiguousarray(values[..., :state_dim])}
    if uses_episode_clock:
        result["episode_progress"] = np.ascontiguousarray(
            values[..., state_dim : config.proprio_observation_dim]
        )
    if config.wrist_rgb_enabled:
        assert config.wrist_rgb_shape is not None
        result["wrist_rgb"] = np.ascontiguousarray(
            values[..., config.proprio_observation_dim :].reshape(
                *values.shape[:-1], *config.wrist_rgb_shape
            )
        )
    return result


def build_learning_metadata(config: CubeLiftTaskConfig) -> dict[str, Any]:
    """Return checkpoint-safe identities for the selected model I/O contract."""

    contract = build_observation_contract(config)
    input_binding = build_model_input_binding(config, contract)
    action_binding = build_action_binding(config)
    return {
        "model_manifest_version": MODEL_MANIFEST_VERSION,
        "policy_class": config.policy_class,
        "observation_contract_id": contract.contract_id,
        "observation_contract_hash": contract.semantic_content_hash,
        "model_input_binding_id": input_binding.binding_id,
        "model_input_binding_hash": input_binding.semantic_content_hash,
        "model_input_fields": input_binding.input_fields,
        "allow_privileged": input_binding.allow_privileged,
        "action_binding_id": action_binding.binding_id,
        "action_binding_hash": action_binding.semantic_content_hash,
        "action_space": action_binding.action_space,
        "action_frame": action_binding.frame,
        "action_mode": action_binding.mode,
        "model_input_dim": input_binding.expected_flat_dim(contract),
        "model_output_dim": action_binding.output_dim,
        "action_dim": action_binding.output_dim,
    }


__all__ = [
    "build_action_binding",
    "build_learning_metadata",
    "build_model_input_binding",
    "build_observation_contract",
    "project_flat_observation",
]
