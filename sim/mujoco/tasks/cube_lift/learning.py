"""Cube-Lift adapter for the backend-neutral learning contracts.

This module is a parity fixture, not a second observation implementation. It
only names the vector and image slices already produced by ``CubeLiftKernel``
and records the normal/expert admission policy around them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from sim.mujoco.learning import (
    MODEL_MANIFEST_VERSION,
    ModelInputBinding,
    ObservationContract,
    ObservationFieldSpec,
    ObservationRole,
    PolicyActionBinding,
    PolicyActionTransform,
    encode_model_input,
)

from .config import (
    NORMAL_POLICY_CLASS,
    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
    CubeLiftTaskConfig,
)
from .reach_guidance import (
    coarse_to_fine_reach_action_scale,
    contact_close_translation_multiplier,
)


_CUBE_LIFT_ACTION_TRANSFORM_HASH_DOMAIN = (
    "dexteleop.cube_lift.policy_action_transform.v1"
)
_REACH_TRANSLATION_COMPONENTS = (
    "right_tcp_delta_x",
    "right_tcp_delta_y",
    "right_tcp_delta_z",
)


def _uses_episode_clock(config: CubeLiftTaskConfig) -> bool:
    return not (
        config.policy_class == NORMAL_POLICY_CLASS
        and config.normal_proprio_profile
        in {
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
        }
    )


def _uses_explicit_object_pose(config: CubeLiftTaskConfig) -> bool:
    return bool(
        config.normal_proprio_profile
        in {
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
        }
        and config.object_pose_observation_source
    )


def _uses_explicit_reach_target_delta(config: CubeLiftTaskConfig) -> bool:
    return bool(
        config.normal_proprio_profile
        == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2
        and config.reach_target_delta_observation_source
    )


def _contract_identity_suffix(config: CubeLiftTaskConfig) -> str:
    suffix = ""
    if _uses_explicit_object_pose(config):
        suffix += f":object_pose_source={config.object_pose_observation_source}"
    if _uses_explicit_reach_target_delta(config):
        suffix += (
            ":reach_target_delta_source="
            f"{config.reach_target_delta_observation_source}"
        )
    return suffix


def build_observation_contract(config: CubeLiftTaskConfig) -> ObservationContract:
    """Describe every named field carried by the task observation."""

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
    state_dim = (
        28
        if _uses_explicit_object_pose(config)
        else config.proprio_observation_dim - int(uses_episode_clock)
    )
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
    if _uses_explicit_object_pose(config):
        fields.append(
            ObservationFieldSpec(
                "object_pose",
                shape=(7,),
                role=ObservationRole.ACTOR,
                source=config.object_pose_observation_source,
                unit="m_and_unit_quaternion_wxyz",
            )
        )
    if _uses_explicit_reach_target_delta(config):
        fields.append(
            ObservationFieldSpec(
                "reach_target_delta",
                shape=(3,),
                role=ObservationRole.ACTOR,
                source=config.reach_target_delta_observation_source,
                unit="m_in_controlled_arm_base_xyz",
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
        contract_id=(
            f"dexteleop.cube_lift.observation:{config.observation_layout}"
            f"{_contract_identity_suffix(config)}"
        ),
        fields=tuple(fields),
    )


def build_actor_observation_contract(
    config: CubeLiftTaskConfig,
    contract: ObservationContract | None = None,
) -> ObservationContract:
    """Project the raw task contract to exactly the fields consumed by actor.

    Returning the raw contract for legacy/all-field bindings preserves their
    checkpoint hashes.  A strict subset gets an independent contract identity,
    so an unused simulator object-pose source cannot become an implicit actor
    dependency through the parent observation hash.
    """

    raw_contract = build_observation_contract(config) if contract is None else contract
    selected = config.resolved_policy_observation_fields
    if not config.policy_observation_fields and selected == raw_contract.field_names:
        return raw_contract
    missing = tuple(name for name in selected if name not in raw_contract.field_names)
    if missing:
        raise ValueError(
            f"Actor observation fields are absent from raw contract: {missing}."
        )
    return ObservationContract(
        contract_id=(
            "dexteleop.cube_lift.actor_observation:"
            f"fields={','.join(selected)}.v1"
        ),
        fields=tuple(raw_contract.field(name) for name in selected),
    )


def build_model_input_binding(
    config: CubeLiftTaskConfig,
    contract: ObservationContract | None = None,
) -> ModelInputBinding:
    """Build the normal or privileged binding selected by ``config``."""

    raw_contract = build_observation_contract(config) if contract is None else contract
    resolved_contract = build_actor_observation_contract(config, raw_contract)
    input_fields = config.resolved_policy_observation_fields
    subset = bool(config.policy_observation_fields) or (
        input_fields != raw_contract.field_names
    )
    binding_metadata = (
        {"policy_observation_fields": input_fields}
        if subset
        else {
            "observation_layout": config.observation_layout,
            "proprio_profile": config.normal_proprio_profile,
            "wrist_rgb_camera_name": config.wrist_rgb_camera_name
            if config.wrist_rgb_enabled
            else None,
            "wrist_rgb_shape": config.wrist_rgb_shape,
        }
    )
    if "object_pose" in input_fields:
        binding_metadata["object_pose_observation_source"] = (
            config.object_pose_observation_source
        )
    if "reach_target_delta" in input_fields:
        binding_metadata["reach_target_delta_observation_source"] = (
            config.reach_target_delta_observation_source
        )
    if "wrist_rgb" in input_fields and subset:
        binding_metadata["wrist_rgb_camera_name"] = config.wrist_rgb_camera_name
        binding_metadata["wrist_rgb_shape"] = config.wrist_rgb_shape
    binding = ModelInputBinding(
        binding_id=(
            f"dexteleop.cube_lift.model_input:fields={','.join(input_fields)}.v1"
            if subset
            else (
                f"dexteleop.cube_lift.model_input:{config.observation_layout}"
                f"{_contract_identity_suffix(config)}"
            )
        ),
        observation_contract_hash=resolved_contract.semantic_content_hash,
        input_fields=input_fields,
        allow_privileged=config.policy_class != NORMAL_POLICY_CLASS,
        metadata=binding_metadata,
    )
    binding.validate(resolved_contract)
    return binding


def build_task_action_binding(config: CubeLiftTaskConfig) -> PolicyActionBinding:
    """Declare the complete real-aligned action consumed by the task kernel."""

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


def build_action_binding(config: CubeLiftTaskConfig) -> PolicyActionBinding:
    """Compatibility alias for the complete task/environment action binding."""

    return build_task_action_binding(config)


def build_policy_action_binding(
    config: CubeLiftTaskConfig,
) -> PolicyActionBinding:
    """Declare the PPO latent output independently of the task action."""

    task_binding = build_task_action_binding(config)
    if config.policy_action_dim == config.action_dim and (
        config.resolved_policy_action_components == config.action_components
    ):
        # Preserve the established seven-dimensional binding identity for
        # existing full-pose checkpoints.
        return task_binding
    return PolicyActionBinding(
        binding_id=(
            "dexteleop.cube_lift.policy_action:"
            f"{config.policy_action_layout}"
        ),
        action_space="continuous",
        output_dim=config.policy_action_dim,
        frame=(
            f"{config.controlled_side}_base"
            if config.uses_pose_action
            else f"{config.controlled_side}_joint"
        ),
        mode="relative",
        normalized=True,
        metadata={
            "policy_action_layout": config.policy_action_layout,
            "policy_action_components": (
                config.resolved_policy_action_components
            ),
            "task_action_layout": config.action_layout,
        },
    )


@dataclass(frozen=True)
class CubeLiftPolicyActionTransform:
    """Task wrapper adding optional observation-conditioned pose scaling.

    Disabled profiles preserve the exact generic transform identity and hash.
    Enabled Reach/contact-approach profiles use only the actor-visible target
    delta, so the same decoder is available during training, evaluation, and
    Sim-HIL. Translation is scaled while an optional gripper component remains
    independently policy-owned.
    """

    config: CubeLiftTaskConfig
    base_transform: PolicyActionTransform

    @property
    def observation_conditioning_enabled(self) -> bool:
        return bool(
            self.config.reach_action_distance_scaling_enabled
            or self.config.contact_close_translation_scaling_enabled
        )

    @property
    def transform_id(self) -> str:
        if not self.observation_conditioning_enabled:
            return self.base_transform.transform_id
        transform_id = self.base_transform.transform_id
        if self.config.reach_action_distance_scaling_enabled:
            transform_id += ":reach_distance_coarse_to_fine.v1"
        if self.config.contact_close_translation_scaling_enabled:
            version = (
                "v3"
                if self.config.contact_close_lift_release_enabled
                else (
                    "v2"
                    if self.config.contact_close_anisotropic_translation_scaling_enabled
                    else "v1"
                )
            )
            transform_id += f":contact_close_translation_gate.{version}"
        return transform_id

    @property
    def policy_action_binding_hash(self) -> str:
        return self.base_transform.policy_action_binding_hash

    @property
    def task_action_binding_hash(self) -> str:
        return self.base_transform.task_action_binding_hash

    @property
    def policy_components(self) -> tuple[str, ...]:
        return self.base_transform.policy_components

    @property
    def task_components(self) -> tuple[str, ...]:
        return self.base_transform.task_components

    @property
    def task_defaults(self) -> tuple[float, ...]:
        return self.base_transform.task_defaults

    @property
    def policy_scales(self) -> tuple[float, ...]:
        return self.base_transform.policy_scales

    @property
    def policy_biases(self) -> tuple[float, ...]:
        return self.base_transform.policy_biases

    @property
    def policy_dim(self) -> int:
        return self.base_transform.policy_dim

    @property
    def task_dim(self) -> int:
        return self.base_transform.task_dim

    @property
    def task_indices(self) -> tuple[int, ...]:
        return self.base_transform.task_indices

    @property
    def scaled_task_indices(self) -> tuple[int, ...]:
        return tuple(
            self.task_components.index(component)
            for component in _REACH_TRANSLATION_COMPONENTS
            if component in self.policy_components
        )

    def validate_bindings(
        self,
        policy_binding: PolicyActionBinding,
        task_binding: PolicyActionBinding,
    ) -> None:
        self.base_transform.validate_bindings(policy_binding, task_binding)

    def translation_action_scale(self, observation: Any) -> np.ndarray:
        """Return x/y/z public-translation multipliers per observation."""

        values = np.asarray(observation, dtype=np.float32)
        if values.ndim == 0 or values.shape[-1] != self.config.observation_dim:
            raise ValueError(
                "Cube-Lift action scaling requires task observations ending "
                f"in {self.config.observation_dim} values; got {values.shape}."
            )
        if not self.observation_conditioning_enabled:
            return np.ones((*values.shape[:-1], 3), dtype=np.float32)
        projected = project_flat_observation(
            self.config,
            values,
        )
        goal_delta = projected["reach_target_delta"]
        reach_scale = np.ones(values.shape[:-1], dtype=np.float32)
        if self.config.reach_action_distance_scaling_enabled:
            reach_scale *= coarse_to_fine_reach_action_scale(
                goal_delta,
                minimum_scale=self.config.reach_action_min_scale,
                slow_radius_m=self.config.reach_action_slow_radius_m,
            )
        translation_scale = np.repeat(
            reach_scale[..., None],
            3,
            axis=-1,
        )
        if self.config.contact_close_translation_scaling_enabled:
            # All supported normal proprio profiles expose measured right
            # gripper close at index six. It is actor input, not reward state.
            measured_close = np.asarray(projected["proprio"])[..., 6]
            xy_multiplier = contact_close_translation_multiplier(
                goal_delta,
                measured_close,
                translation_scale=(
                    self.config.contact_close_translation_scale
                ),
                close_distance_m=self.config.contact_close_distance_m,
                keep_measured_close=(
                    self.config.contact_close_keep_measured_close
                ),
            )
            if self.config.contact_close_anisotropic_translation_scaling_enabled:
                z_multiplier = contact_close_translation_multiplier(
                    goal_delta,
                    measured_close,
                    translation_scale=(
                        self.config.resolved_contact_close_translation_scales[2]
                    ),
                    close_distance_m=self.config.contact_close_distance_m,
                    keep_measured_close=(
                        self.config.contact_close_keep_measured_close
                    ),
                )
            else:
                z_multiplier = xy_multiplier
            if self.config.contact_close_lift_release_enabled:
                # This threshold is already actor-visible in proprio and
                # matches the identity post-contact reference phase. Release
                # only z for lifting; retain fine lateral grasp centering.
                z_multiplier = np.where(
                    measured_close
                    >= self.config.post_contact_phase_min_measured_close,
                    np.float32(1.0),
                    z_multiplier,
                )
            translation_scale *= np.stack(
                (xy_multiplier, xy_multiplier, z_multiplier),
                axis=-1,
            )
        return np.asarray(translation_scale, dtype=np.float32)

    def action_scale(self, observation: Any) -> np.ndarray:
        """Return the minimum active translation multiplier per observation."""

        return np.min(self.translation_action_scale(observation), axis=-1)

    def transform(
        self,
        values: Any,
        *,
        observation: Any | None = None,
    ) -> np.ndarray:
        output = self.base_transform.transform(values)
        if not self.observation_conditioning_enabled:
            return output
        if observation is None:
            raise ValueError(
                "Distance-dependent Cube-Lift action scaling requires the "
                "current task observation."
            )
        policy_values = np.asarray(values, dtype=np.float32)
        translation_scale = self.translation_action_scale(observation)
        expected_scale_shape = (*policy_values.shape[:-1], 3)
        if translation_scale.shape != expected_scale_shape:
            raise ValueError(
                "Policy action and action-scaling observation batch shapes "
                f"must match: action={policy_values.shape}, "
                f"translation_scale={translation_scale.shape}."
            )
        output[..., list(self.scaled_task_indices)] *= translation_scale
        return output

    def semantic_payload(self) -> dict[str, Any]:
        if not self.observation_conditioning_enabled:
            return self.base_transform.semantic_payload()
        actor_contract = build_actor_observation_contract(self.config)
        reach_conditioning = {
            "id": "reach_target_delta_l2_smoothstep.v1",
            "observation_field": "reach_target_delta",
            "distance_unit": "m",
            "minimum_scale": self.config.reach_action_min_scale,
            "slow_radius_m": self.config.reach_action_slow_radius_m,
            "scaled_task_components": _REACH_TRANSLATION_COMPONENTS,
        }
        if not self.config.contact_close_translation_scaling_enabled:
            # Preserve the exact Reach-v4 transform hash.
            conditioning: dict[str, Any] = reach_conditioning
        else:
            anisotropic = (
                self.config.contact_close_anisotropic_translation_scaling_enabled
            )
            lift_release = self.config.contact_close_lift_release_enabled
            close_conditioning = {
                "id": (
                    "contact_close_translation_gate.v3"
                    if lift_release
                    else (
                        "contact_close_translation_gate.v2"
                        if anisotropic
                        else "contact_close_translation_gate.v1"
                    )
                ),
                "observation_fields": [
                    "reach_target_delta",
                    "proprio.right_gripper_measured_close",
                ],
                "close_distance_m": self.config.contact_close_distance_m,
                "keep_measured_close": (
                    self.config.contact_close_keep_measured_close
                ),
                "scaled_task_components": _REACH_TRANSLATION_COMPONENTS,
            }
            if anisotropic:
                close_conditioning["translation_scales_xyz"] = (
                    self.config.resolved_contact_close_translation_scales
                )
            else:
                # Preserve the exact isotropic Contact-v2 transform hash.
                close_conditioning["translation_scale"] = (
                    self.config.contact_close_translation_scale
                )
            if lift_release:
                close_conditioning["lift_z_release"] = {
                    "observation_field": (
                        "proprio.right_gripper_measured_close"
                    ),
                    "minimum_measured_close": (
                        self.config.post_contact_phase_min_measured_close
                    ),
                    "released_z_multiplier": 1.0,
                    "lateral_gate_retained": True,
                }
            conditioning = {
                "id": (
                    "coarse_to_fine_then_contact_close_translation.v3"
                    if lift_release
                    else (
                        "coarse_to_fine_then_contact_close_translation.v2"
                        if anisotropic
                        else "coarse_to_fine_then_contact_close_translation.v1"
                    )
                ),
                "stages": [reach_conditioning, close_conditioning],
            }
        return {
            "transform_id": self.transform_id,
            "base_transform_hash": self.base_transform.semantic_content_hash,
            "actor_observation_contract_hash": (
                actor_contract.semantic_content_hash
            ),
            "conditioning": conditioning,
        }

    @property
    def semantic_content_hash(self) -> str:
        if not self.observation_conditioning_enabled:
            return self.base_transform.semantic_content_hash
        return canonical_content_hash(
            self.semantic_payload(),
            domain=_CUBE_LIFT_ACTION_TRANSFORM_HASH_DOMAIN,
        )


def build_policy_action_transform(
    config: CubeLiftTaskConfig,
) -> CubeLiftPolicyActionTransform:
    """Map compact PPO outputs into the complete seven-dimensional action."""

    policy_binding = build_policy_action_binding(config)
    task_binding = build_task_action_binding(config)
    affine = bool(
        config.resolved_policy_action_scale
        != (1.0,) * config.policy_action_dim
        or config.resolved_policy_action_bias
        != (0.0,) * config.policy_action_dim
    )
    base_transform = PolicyActionTransform(
        transform_id=(
            "dexteleop.cube_lift.policy_to_task_action:"
            f"{config.policy_action_layout}->{config.action_layout}"
            f"{':affine.v1' if affine else ''}"
        ),
        policy_action_binding_hash=policy_binding.semantic_content_hash,
        task_action_binding_hash=task_binding.semantic_content_hash,
        policy_components=config.resolved_policy_action_components,
        task_components=config.action_components,
        task_defaults=(0.0,) * config.action_dim,
        policy_scales=config.resolved_policy_action_scale,
        policy_biases=config.resolved_policy_action_bias,
    )
    transform = CubeLiftPolicyActionTransform(config, base_transform)
    transform.validate_bindings(policy_binding, task_binding)
    return transform


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
    state_dim = (
        28
        if _uses_explicit_object_pose(config)
        else config.proprio_observation_dim - int(uses_episode_clock)
    )
    state_name = "proprio" if config.policy_class == NORMAL_POLICY_CLASS else "privileged_state"
    result: dict[str, np.ndarray] = {state_name: np.ascontiguousarray(values[..., :state_dim])}
    if uses_episode_clock:
        result["episode_progress"] = np.ascontiguousarray(
            values[..., state_dim : config.proprio_observation_dim]
        )
    if _uses_explicit_object_pose(config):
        result["object_pose"] = np.ascontiguousarray(
            values[..., 28:35]
        )
    if _uses_explicit_reach_target_delta(config):
        result["reach_target_delta"] = np.ascontiguousarray(
            values[..., 35:38]
        )
    if config.wrist_rgb_enabled:
        assert config.wrist_rgb_shape is not None
        result["wrist_rgb"] = np.ascontiguousarray(
            values[..., config.proprio_observation_dim :].reshape(
                *values.shape[:-1], *config.wrist_rgb_shape
            )
        )
    return result


def encode_policy_observation(
    config: CubeLiftTaskConfig,
    observation: Any,
) -> np.ndarray:
    """Encode a raw scalar or batched task observation for the actor only."""

    raw_contract = build_observation_contract(config)
    actor_contract = build_actor_observation_contract(config, raw_contract)
    binding = build_model_input_binding(config, raw_contract)
    encoded = encode_model_input(
        project_flat_observation(config, observation),
        actor_contract,
        binding,
    )
    return encoded.flat


def build_learning_metadata(config: CubeLiftTaskConfig) -> dict[str, Any]:
    """Return checkpoint-safe identities for the selected model I/O contract."""

    task_contract = build_observation_contract(config)
    contract = build_actor_observation_contract(config, task_contract)
    input_binding = build_model_input_binding(config, task_contract)
    task_action_binding = build_task_action_binding(config)
    policy_action_binding = build_policy_action_binding(config)
    action_transform = build_policy_action_transform(config)
    return {
        "model_manifest_version": MODEL_MANIFEST_VERSION,
        "policy_class": config.policy_class,
        "task_observation_contract_id": task_contract.contract_id,
        "task_observation_contract_hash": task_contract.semantic_content_hash,
        "task_observation_dim": config.observation_dim,
        "observation_contract_id": contract.contract_id,
        "observation_contract_hash": contract.semantic_content_hash,
        "model_input_binding_id": input_binding.binding_id,
        "model_input_binding_hash": input_binding.semantic_content_hash,
        "model_input_fields": input_binding.input_fields,
        "allow_privileged": input_binding.allow_privileged,
        "actor_uses_privileged": any(
            contract.field(name).role is ObservationRole.PRIVILEGED
            for name in input_binding.input_fields
        ),
        "action_binding_id": policy_action_binding.binding_id,
        "action_binding_hash": policy_action_binding.semantic_content_hash,
        "action_space": policy_action_binding.action_space,
        "action_frame": policy_action_binding.frame,
        "action_mode": policy_action_binding.mode,
        "policy_action_layout": config.policy_action_layout,
        "policy_action_components": config.resolved_policy_action_components,
        "policy_action_transform_id": action_transform.transform_id,
        "policy_action_transform_hash": action_transform.semantic_content_hash,
        "task_action_conditioning_hash": config.action_conditioning_sha256,
        "task_action_conditioning": config.action_conditioning_metadata,
        "task_action_binding_id": task_action_binding.binding_id,
        "task_action_binding_hash": task_action_binding.semantic_content_hash,
        "task_action_layout": config.action_layout,
        "task_action_components": config.action_components,
        "model_input_dim": input_binding.expected_flat_dim(contract),
        "model_output_dim": policy_action_binding.output_dim,
        "policy_action_dim": policy_action_binding.output_dim,
        "task_action_dim": task_action_binding.output_dim,
        # Compatibility key: the environment/task action remains 7-D.
        "action_dim": task_action_binding.output_dim,
    }


__all__ = [
    "build_action_binding",
    "build_actor_observation_contract",
    "build_learning_metadata",
    "build_model_input_binding",
    "build_observation_contract",
    "build_policy_action_binding",
    "build_policy_action_transform",
    "build_task_action_binding",
    "CubeLiftPolicyActionTransform",
    "encode_policy_observation",
    "project_flat_observation",
]
