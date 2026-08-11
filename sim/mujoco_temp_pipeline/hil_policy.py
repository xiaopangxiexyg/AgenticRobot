"""Temporary-PPO contributor adapter for MuJoCo Sim-HIL.

The temporary CubeLift PPO checkpoint is not a generic deploy policy: its
latent output can control a task-selected subset such as xyz or z-only.  The
task-owned transform expands that latent sample into seven normalized values
(right TCP xyz/rotvec increments plus a gripper increment) before this module
exposes the shared ``PolicyActionEmission`` seam consumed by
``control_station``.

It intentionally owns no MuJoCo execution, canonical safety, IK, authority, or
dispatch.  The only MuJoCo access here is read-only observation/pose anchoring
needed to decode the task-local action into one absolute right-arm pose.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from robot_motion.action import ArmActionSpec, HandActionSpec
from robot_motion.deploy import (
    NormalizedAction,
    PolicyActionEmission,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    PolicyExecutionFeedback,
    PolicyExecutionFeedbackApplication,
)
from robot_motion.deploy.action_adapter import DeployRobotActionAdapter
from robot_motion.deploy.source_projection import (
    POLICY_SOURCE_ACTION_PROJECTION_METADATA_KEY,
    POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA,
    POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
    PolicySourceActionProjection,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    pose_matrix_to_xyzquat,
)
from sim.mujoco.deploy.deployment_observation import (
    MujocoDeploymentObservationAdapter,
    MujocoDeploymentObservationConfig,
)
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
from sim.mujoco.tasks.cube_lift.action_conditioning import (
    CubeLiftActionConditioner,
    CubeLiftFingerContactObserver,
)
from sim.mujoco.tasks.cube_lift.learning import (
    build_learning_metadata,
    build_policy_action_transform,
    encode_policy_observation,
)
from sim.mujoco.tasks.cube_lift.reach_guidance import build_reach_guidance
from sim.mujoco.tasks.cube_lift.spec import (
    CubeLiftActionSpec,
    CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION,
    CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION,
    CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION,
    CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION,
)

from .checkpoint import (
    build_policy_from_checkpoint,
    check_task_config_compatibility,
)
from .policy import resolve_device


TEMP_PPO_HIL_POLICY_ID_V1 = "sim.mujoco_temp_pipeline.ppo_hil.v1"
TEMP_PPO_HIL_OBSERVATION_ADAPTER_ID_V1 = (
    "sim.mujoco_temp_pipeline.normal_motion_tracking_no_episode_clock.v1"
)


def _finite_time(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _arm_base_body_name(scene: Any, side: str) -> str:
    candidates = (f"{side}_arm_base_link", "arm_base_link", "base_link")
    for name in candidates:
        if scene.has_body(name):
            return name
    raise KeyError(f"Could not resolve {side} arm base body; tried={candidates}.")


def validate_temp_ppo_hil_profile(
    task_config: CubeLiftTaskConfig,
    policy: Any,
) -> None:
    """Validate the task/checkpoint pair without constructing devices or motion."""

    if not task_config.uses_pose_action:
        raise ValueError("Temporary PPO Sim-HIL requires a pose-action task.")
    if task_config.action_layout != CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION:
        raise ValueError("Temporary PPO task action layout is not the pose layout.")
    supported_layout_dims = {
        CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION: 28,
        CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION: 35,
        CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION: 38,
    }
    expected_observation_dim = supported_layout_dims.get(
        task_config.observation_layout
    )
    if expected_observation_dim is None:
        raise ValueError(
            "Temporary PPO Sim-HIL requires a supported no-episode-clock "
            "motion-tracking observation."
        )
    if (
        task_config.observation_dim != expected_observation_dim
        or task_config.action_dim != 7
    ):
        raise ValueError(
            "Temporary PPO task dimensions do not match its declared "
            "observation layout."
        )
    action_transform = build_policy_action_transform(task_config)
    learning_metadata = build_learning_metadata(task_config)
    if (
        int(policy.observation_dim) != int(learning_metadata["model_input_dim"])
        or int(policy.action_dim) != action_transform.policy_dim
    ):
        raise ValueError(
            "Selected PPO checkpoint dimensions do not match the task "
            "observation layout."
        )
    if (
        int(policy.environment_action_dim) != action_transform.task_dim
        or tuple(policy.active_action_indices) != action_transform.task_indices
    ):
        raise ValueError(
            "Temporary PPO checkpoint does not match the task-owned "
            "latent-to-pose action transform."
        )


class TempPPOHILObservationAdapter:
    """Project the shared MuJoCo observation into a supported task-local state.

    The shared deployment observation remains 38-D so it can pass the common
    observation firewall.  The task-local 28-D, 35-D, or 38-D PPO state is
    stored in its prefix and the predictor explicitly consumes that prefix.
    Target-error fields are zeroed because reconstructing the old CubeLift
    internal qpos target before composition would introduce the forbidden
    second IK owner.  The optional reach delta is reconstructed from the
    externally observable object pose and the measured TCP pose in the same
    right-arm-base frame used by the public pose action.  Motion deltas advance
    only on policy-cadence samples, not on every faster Control Station
    composition tick.
    """

    def __init__(
        self,
        scene: Any,
        task_config: CubeLiftTaskConfig,
        base: MujocoDeploymentObservationAdapter,
    ) -> None:
        self.scene = scene
        self.task_config = task_config
        self.base = base
        self._previous_qpos: np.ndarray | None = None
        self._previous_close: float | None = None
        self._last_compact_action = np.zeros(7, dtype=np.float32)
        self._endpoint_to_tool: np.ndarray | None = None
        if (
            task_config.observation_layout
            == CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION
        ):
            from robot_motion.rm65 import (
                endpoint_to_tool_transform,
                load_rm65_tool_profile,
            )

            self._endpoint_to_tool = endpoint_to_tool_transform(
                load_rm65_tool_profile(
                    task_config.right_tool_profile_path,
                    side="right",
                )
            )

    @property
    def last_compact_action(self) -> np.ndarray:
        return self._last_compact_action.copy()

    def reset(self) -> None:
        self._previous_qpos = None
        self._previous_close = None
        self._last_compact_action.fill(0.0)
        self.base.set_last_action(np.zeros(26, dtype=np.float32))

    def set_last_compact_action(self, action: Sequence[float]) -> None:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("Temporary PPO last action must be finite 7-D.")
        self._last_compact_action = values.copy()

    def capture(
        self,
        *,
        step_index: int | None = None,
        commit_policy_history: bool = True,
    ) -> DeploymentObservation:
        base_observation = self.base.capture(step_index=step_index)
        measured_qpos = np.asarray(
            self.scene.get_arm_qpos("right"), dtype=np.float32
        ).reshape(-1)
        measured_close = float(
            self.scene.get_gripper_close(self.task_config.right_gripper_name)
        )
        if measured_qpos.shape != (6,) or not np.all(np.isfinite(measured_qpos)):
            raise ValueError("MuJoCo right qpos is not a finite 6-D vector.")
        if not math.isfinite(measured_close):
            raise ValueError("MuJoCo right gripper close is not finite.")

        previous_qpos = (
            measured_qpos
            if self._previous_qpos is None
            else self._previous_qpos
        )
        previous_close = (
            measured_close
            if self._previous_close is None
            else self._previous_close
        )
        compact_dim = int(self.task_config.observation_dim)
        compact_state = np.zeros(compact_dim, dtype=np.float32)
        compact_state[:6] = measured_qpos
        compact_state[6] = measured_close
        compact_state[7:13] = measured_qpos - previous_qpos
        compact_state[13] = measured_close - previous_close
        # 14:20 and 20 are the historical target-error fields.  They remain
        # explicit zero fallbacks until a post-composition target is available.
        compact_state[21:28] = self._last_compact_action
        cube_pose: np.ndarray | None = None
        if self.task_config.observation_layout in {
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION,
        }:
            cube_pose = np.asarray(
                self.scene.freejoint_qpos(
                    self.task_config.object_freejoint_name
                ),
                dtype=np.float32,
            ).reshape(-1)
            if cube_pose.shape != (7,) or not np.all(np.isfinite(cube_pose)):
                raise ValueError("MuJoCo cube freejoint pose is not finite 7-D.")
            compact_state[28:35] = cube_pose
        if (
            self.task_config.observation_layout
            == CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION
        ):
            if cube_pose is None or self._endpoint_to_tool is None:
                raise RuntimeError("Reach-delta observation dependencies are unavailable.")
            base_pose = np.asarray(
                self.scene.body_pose(_arm_base_body_name(self.scene, "right")),
                dtype=float,
            ).reshape(4, 4)
            endpoint_pose = np.asarray(
                self.scene.body_pose(
                    self.scene.arm_specs["right"].controlled_body
                ),
                dtype=float,
            ).reshape(4, 4)
            if not np.all(np.isfinite(base_pose)) or not np.all(
                np.isfinite(endpoint_pose)
            ):
                raise ValueError("MuJoCo right-arm-base/TCP poses are not finite.")
            tool_pose = endpoint_pose @ self._endpoint_to_tool
            compact_state[35:38] = build_reach_guidance(
                cube_pose[:3],
                tool_pose[:3, 3],
                base_pose[:3, :3],
                target_height_offset_m=(
                    self.task_config.reach_target_height_offset_m
                ),
                guidance_height_offset_m=(
                    self.task_config.reach_guidance_height_offset_m
                ),
                guidance_xy_blend_radius_m=(
                    self.task_config.reach_guidance_xy_blend_radius_m
                ),
            ).delta_in_base
        if commit_policy_history:
            self._previous_qpos = measured_qpos.copy()
            self._previous_close = measured_close

        state = np.zeros(38, dtype=np.float32)
        state[:compact_dim] = compact_state
        state_valid = np.ones(38, dtype=bool)
        metadata = dict(base_observation.metadata)
        metadata.update(
            {
                "policy_observation_adapter_id": TEMP_PPO_HIL_OBSERVATION_ADAPTER_ID_V1,
                "policy_observation_target_error_source": (
                    "measured_qpos_zero_fallback_v1"
                ),
                "policy_observation_compatibility": (
                    "target_error_not_reconstructed_without_precomposition_ik"
                ),
                "policy_observation_layout": (
                    self.task_config.observation_layout
                ),
                "policy_observation_dim": compact_dim,
                "policy_observation_history_committed": bool(
                    commit_policy_history
                ),
                "policy_observation_compact_last_action": (
                    self._last_compact_action.copy()
                ),
            }
        )
        state_dict = dict(base_observation.state_dict)
        state_dict.update(
            {
                "temp_ppo_state": compact_state.copy(),
                "right_target_error": np.zeros(6, dtype=np.float32),
                "right_gripper_target_error": 0.0,
                **(
                    {"cube_pose": compact_state[28:35].copy()}
                    if compact_dim in {35, 38}
                    else {}
                ),
                **(
                    {
                        "reach_target_delta": compact_state[35:38].copy(),
                    }
                    if compact_dim == 38
                    else {}
                ),
            }
        )
        return replace(
            base_observation,
            state=state,
            state_dict=state_dict,
            state_valid=state_valid,
            metadata=metadata,
        )


class _TempPPOModel:
    """Strict model-facing wrapper around the temporary ActorCritic."""

    def __init__(
        self,
        policy: Any,
        normalizer: Any,
        action_transform: Any,
        *,
        task_config: CubeLiftTaskConfig,
    ) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.action_transform = action_transform
        self.task_config = task_config
        self.task_observation_dim = int(task_config.observation_dim)
        self.device = next(policy.parameters()).device

    def predict(self, observation: PolicyObservationView) -> np.ndarray:
        if not isinstance(observation, PolicyObservationView):
            raise TypeError("Temporary PPO requires a PolicyObservationView.")
        state = np.asarray(observation.model_input()["state"], dtype=np.float32).reshape(-1)
        if state.shape != (38,):
            raise ValueError(f"Shared observation state must be 38-D, got {state.shape}.")
        compact_state = state[: self.task_observation_dim]
        actor_state = encode_policy_observation(
            self.task_config,
            compact_state,
        )
        normalized = self.normalizer.normalize(actor_state)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self.policy.predict_action(
                tensor.unsqueeze(0),
                deterministic=True,
            )
        latent = action.detach().cpu().numpy().astype(np.float32, copy=False)
        values = self.action_transform.transform(
            latent,
            observation=compact_state[None, :],
        )[0]
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("Temporary PPO returned a non-finite 7-D action.")
        return values.copy()


class _TempPPOAbsolutePoseDecoder:
    """Decode task-local PPO deltas into one absolute right-arm pose vector."""

    def __init__(
        self,
        scene: Any,
        task_config: CubeLiftTaskConfig,
        observation_adapter: TempPPOHILObservationAdapter,
        *,
        task_initialization_info: Mapping[str, object],
        scene_capture: Callable[[Callable[[Any], object]], object] | None = None,
    ) -> None:
        self.scene = scene
        self.task_config = task_config
        self.observation_adapter = observation_adapter
        if scene_capture is not None and not callable(scene_capture):
            raise TypeError("scene_capture must be callable or None.")
        self._scene_capture = (
            (lambda capture: capture(self.scene))
            if scene_capture is None
            else scene_capture
        )
        from robot_motion.rm65 import endpoint_to_tool_transform, load_rm65_tool_profile

        self._endpoint_to_tool = endpoint_to_tool_transform(
            load_rm65_tool_profile(task_config.right_tool_profile_path, side="right")
        )
        self._action_spec = CubeLiftActionSpec(
            layout_version=task_config.action_layout
        )
        self._right_hand_spec = HandActionSpec.dmtac_gripper("right")
        initialization = dict(task_initialization_info)
        self._scripted_pregrasp_reset_applied = bool(
            initialization.get("scripted_pregrasp_reset", False)
        )
        self._scripted_contact_reset_applied = bool(
            initialization.get("scripted_contact_reset", False)
        )
        if (
            self._scripted_pregrasp_reset_applied
            != bool(task_config.scripted_pregrasp_reset)
            or self._scripted_contact_reset_applied
            != bool(task_config.scripted_contact_reset)
        ):
            raise ValueError(
                "Temporary PPO source initialization does not match the "
                "task-declared scripted reset semantics."
            )
        self._action_conditioner = CubeLiftActionConditioner(task_config)
        self._contact_observer = (
            CubeLiftFingerContactObserver(scene, task_config)
            if task_config.post_contact_phase_enabled
            else None
        )
        self._target_pose_arm_base: np.ndarray | None = None
        self._target_gripper_close: float | None = None
        self._target_generation = 0

    def reset(self) -> None:
        self._action_conditioner.reset()
        self._target_pose_arm_base = None
        self._target_gripper_close = None
        self._target_generation = 0

    @property
    def target_generation(self) -> int:
        return int(self._target_generation)

    def _current_tool_pose_arm_base(self) -> np.ndarray:
        base = self.scene.body_pose(_arm_base_body_name(self.scene, "right"))
        endpoint = self.scene.body_pose(self.scene.arm_specs["right"].controlled_body)
        pose = np.linalg.inv(base) @ endpoint @ self._endpoint_to_tool
        pose = np.asarray(pose, dtype=float).reshape(4, 4)
        if not np.all(np.isfinite(pose)):
            raise ValueError("Current MuJoCo right TCP pose is not finite.")
        return pose

    def decode(
        self,
        action: Sequence[float] | np.ndarray,
        observation: DeploymentObservation,
    ) -> NormalizedAction:
        # The model inference has already completed.  Capture only the short
        # read/conditioning/decode section under the downstream scene owner;
        # never hold the MuJoCo lock while running the policy network.
        return self._scene_capture(
            lambda _scene: self._decode_from_captured_scene(
                action,
                observation,
            )
        )

    def _decode_from_captured_scene(
        self,
        action: Sequence[float] | np.ndarray,
        observation: DeploymentObservation,
    ) -> NormalizedAction:
        del observation
        requested_values = self._action_spec.coerce(action)
        left_contact, right_contact = (
            (False, False)
            if self._contact_observer is None
            else self._contact_observer.observe()
        )
        conditioning = self._action_conditioner.condition(
            requested_values,
            left_finger_contact=left_contact,
            right_finger_contact=right_contact,
            measured_gripper_close=self.scene.get_gripper_close(
                self.task_config.right_gripper_name
            ),
            scripted_pregrasp_reset_applied=(
                self._scripted_pregrasp_reset_applied
            ),
            scripted_contact_reset_applied=(
                self._scripted_contact_reset_applied
            ),
        )
        values = conditioning.values
        if self._target_pose_arm_base is None:
            self._target_pose_arm_base = self._current_tool_pose_arm_base()
        if self._target_gripper_close is None:
            measured_close = float(
                self.scene.get_gripper_close(self.task_config.right_gripper_name)
            )
            if not math.isfinite(measured_close):
                raise ValueError("Current MuJoCo right gripper close is not finite.")
            self._target_gripper_close = float(
                np.clip(measured_close, 0.0, self.task_config.right_gripper_max_close)
            )
        source_anchor_pose = self._target_pose_arm_base.copy()
        source_anchor_gripper = float(self._target_gripper_close)
        target = self._target_pose_arm_base.copy()
        target[:3, 3] += values[:3] * np.asarray(
            self.task_config.right_tcp_delta_m, dtype=float
        )
        rotation_delta = Rotation.from_rotvec(
            values[3:6]
            * np.asarray(self.task_config.right_tcp_rotation_delta_rad, dtype=float)
        ).as_matrix()
        target[:3, :3] = rotation_delta @ target[:3, :3]
        self._target_pose_arm_base = target
        self._target_gripper_close = float(
            np.clip(
                self._target_gripper_close
                + float(values[6]) * self.task_config.right_gripper_delta,
                0.0,
                self.task_config.right_gripper_max_close,
            )
        )
        xyzquat = pose_matrix_to_xyzquat(target)
        if xyzquat is None:
            raise ValueError("Decoded PPO target pose cannot be represented as xyzquat.")
        vector = np.zeros(26, dtype=np.float32)
        vector[ACTION_ARM_POSE_SLICES["right"]] = xyzquat
        hand_qpos = np.zeros(self._right_hand_spec.slot_count, dtype=np.float32)
        hand_qpos[self._right_hand_spec.slot_map["gripper_close"]] = (
            self._target_gripper_close
        )
        self._right_hand_spec.validate_vector(hand_qpos)
        vector[ACTION_HAND_SLICES["right"]] = hand_qpos
        projection = PolicySourceActionProjection(
            source_semantics=POLICY_SOURCE_ACTION_SEQUENTIAL_TARGET_DELTA,
            source_pose_mode="relative",
            source_action_schema_id=self.task_config.action_layout,
            source_action_by_channel={"task_action": requested_values},
            conditioned_action_by_channel={"task_action": values},
            decoded_absolute_action_hash=canonical_content_hash(
                {"vector": vector, "arm_joint_targets": {}},
                domain="dexteleop.policy_decoded_absolute_action.v1",
            ),
            anchor_policy=POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
            anchor_value_by_component={
                "right_arm": pose_matrix_to_xyzquat(source_anchor_pose),
                "right_gripper": (source_anchor_gripper,),
            },
            target_generation=self._target_generation,
        )
        self._target_generation += 1
        arm_spec = ArmActionSpec.tool_tcp(
            "right", reference_frame="right_arm_base"
        )
        metadata = {
            "source": "temp_ppo_hil",
            "arm_action_specs_by_side": {"right": arm_spec.to_metadata()},
            "arm_controlled_frame_by_side": {"right": "tool_tcp"},
            "arm_reference_frame_by_side": {"right": "right_arm_base"},
            "end_effector_enabled_sides": ("right",),
            "hand_action_specs_by_side": {
                "right": self._right_hand_spec.to_metadata()
            },
            "pose_action_source_mode": "relative",
            "pose_action_mode": "absolute",
            "pose_action_source_absolute_frame": "right_arm_base",
            "pose_action_output_absolute_frame": "runtime_raw",
            "pose_anchor_policy": POLICY_SOURCE_ANCHOR_PREVIOUS_POLICY_TARGET,
            "pose_anchor_sides": ("right",),
            POLICY_SOURCE_ACTION_PROJECTION_METADATA_KEY: projection.to_dict(),
            "temp_ppo_hil_profile": TEMP_PPO_HIL_POLICY_ID_V1,
            "unused_action_components": (),
            "policy_hand_control": "right_gripper_delta_to_absolute_qpos_v1",
            "right_gripper_target_close": self._target_gripper_close,
            "action_protocol": "robot_action_v1",
            "policy_action_layout": CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION,
            "policy_latent_action_layout": self.task_config.policy_action_layout,
            "policy_latent_action_components": (
                self.task_config.resolved_policy_action_components
            ),
            "task_action_components": self.task_config.action_components,
            "observation_target_error_source": "measured_qpos_zero_fallback_v1",
            "task_requested_action": requested_values.copy(),
            "task_conditioned_action": values.copy(),
            # Compatibility aliases for existing Sim-HIL log consumers.
            "policy_requested_action": requested_values.copy(),
            "policy_conditioned_action": values.copy(),
            "scripted_pregrasp_action_modified": (
                conditioning.scripted_pregrasp_modified
            ),
            "scripted_contact_action_modified": (
                conditioning.scripted_contact_modified
            ),
            "post_contact_phase_active": (
                conditioning.post_contact_phase_active
            ),
            "post_contact_phase_action_modified": (
                conditioning.post_contact_modified
            ),
            "post_contact_phase_contact_steps": (
                conditioning.post_contact_phase_contact_steps
            ),
            "post_contact_phase_loss_steps": (
                conditioning.post_contact_phase_loss_steps
            ),
        }
        self.observation_adapter.set_last_compact_action(values)
        self.observation_adapter.base.set_last_action(vector)
        return NormalizedAction(vector=vector, metadata=metadata)


class _TempPPOActionPreparer:
    def __init__(self, adapter: DeployRobotActionAdapter) -> None:
        self.action_adapter = adapter

    def adapt_pre_safety(
        self,
        action: NormalizedAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> Any:
        return self.action_adapter.adapt(
            action,
            observation,
            policy_name=policy_name,
            input_stage="decoded_absolute_pre_safety",
        )

    def prepare(
        self,
        action: NormalizedAction,
        observation: DeploymentObservation,
        *,
        policy_name: str = "",
    ) -> Any:
        result = self.adapt_pre_safety(
            action, observation, policy_name=policy_name
        )
        return result.robot_action


class _TempPPOPolicyDeployRuntime(PolicyDeployRuntime):
    def __init__(self, *args: Any, decoder: _TempPPOAbsolutePoseDecoder, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._decoder = decoder

    def normalize_and_decode(
        self,
        policy_output: Any,
        observation: DeploymentObservation,
    ) -> list[NormalizedAction]:
        return [self._decoder.decode(policy_output, observation)]


class TempPPOHILPolicySource:
    """One policy contributor backed by a temporary PPO checkpoint."""

    def __init__(
        self,
        scene: Any,
        *,
        checkpoint_path: str | Path,
        task_config_path: str | Path,
        policy_node_id: str = "sim-hil-temp-ppo",
        policy_clock_domain_id: str = "mujoco.sim_hil.monotonic",
        device: str = "auto",
        monotonic_clock: Callable[[], float] = time.monotonic,
        allow_task_config_mismatch: bool = False,
        task_initialization_info: Mapping[str, object] | None = None,
        downstream_target_lead_sec: float | None = None,
        scene_capture: Callable[[Callable[[Any], object]], object] | None = None,
    ) -> None:
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        if scene_capture is not None and not callable(scene_capture):
            raise TypeError("scene_capture must be callable or None.")
        self.scene = scene
        self._scene_capture = (
            (lambda capture: capture(self.scene))
            if scene_capture is None
            else scene_capture
        )
        self.checkpoint_path = Path(checkpoint_path)
        self.task_config_path = Path(task_config_path)
        self.clock = monotonic_clock
        self.task_config = CubeLiftTaskConfig.load(self.task_config_path)
        resolved_device = resolve_device(device)
        self.policy, self.normalizer, self.checkpoint_payload = build_policy_from_checkpoint(
            self.checkpoint_path, device=resolved_device
        )
        self.compatibility = check_task_config_compatibility(
            self.checkpoint_payload,
            self.task_config_path,
            allow_mismatch=allow_task_config_mismatch,
        )
        self._validate_profile()
        initialization = (
            {
                "scripted_pregrasp_reset": False,
                "scripted_contact_reset": False,
            }
            if task_initialization_info is None
            else dict(task_initialization_info)
        )
        if (
            self.task_config.scripted_pregrasp_reset
            or self.task_config.scripted_contact_reset
        ) and task_initialization_info is None:
            raise ValueError(
                "A task with scripted reset semantics requires explicit "
                "task_initialization_info from the simulator target."
            )
        hand_spec = HandActionSpec.dmtac_gripper("right")
        arm_spec = ArmActionSpec.tool_tcp(
            "right", reference_frame="right_arm_base"
        )
        observation_config = MujocoDeploymentObservationConfig(
            active_sides=("right",),
            tcp_body_by_side={
                "right": scene.arm_specs["right"].controlled_body
            },
            hands_enabled=True,
            end_effector_sides=("right",),
            gripper_name_by_side={
                "right": self.task_config.right_gripper_name
            },
            metadata={
                "arm_action_specs_by_side": {"right": arm_spec.to_metadata()},
                "arm_controlled_frame_by_side": {"right": "tool_tcp"},
                "arm_reference_frame_by_side": {"right": "right_arm_base"},
                "hand_action_specs_by_side": {
                    "right": hand_spec.to_metadata()
                },
                "policy_source": "temp_ppo_hil",
            },
        )
        base_observation = MujocoDeploymentObservationAdapter(
            scene,
            observation_config,
            clock=monotonic_clock,
        )
        self.observation_adapter = TempPPOHILObservationAdapter(
            scene, self.task_config, base_observation
        )
        self.decoder = _TempPPOAbsolutePoseDecoder(
            scene,
            self.task_config,
            self.observation_adapter,
            task_initialization_info=initialization,
            scene_capture=self._scene_capture,
        )
        period = float(self.task_config.policy_period_sec)
        downstream_lead = (
            0.0
            if downstream_target_lead_sec is None
            else _finite_time(
                downstream_target_lead_sec,
                name="downstream_target_lead_sec",
            )
        )
        if downstream_lead < 0.0:
            raise ValueError(
                "downstream_target_lead_sec must be non-negative."
            )
        # This single-step source has no guaranteed successor.  Keep the last
        # admitted action through one bounded CPU inference/IK tail; longer
        # stalls still expire and enter the existing HOLD path.
        self._maximum_hold_sec = max(
            0.25,
            2.0 * period + downstream_lead,
        )
        runtime_config = PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
            policy_action_pose_mode="absolute",
            policy_absolute_frame="runtime_raw",
            policy_node_id=str(policy_node_id),
            policy_clock_domain_id=str(policy_clock_domain_id),
            action_keyframe_period_sec=period,
            # A one-step policy has no guaranteed successor to interpolate.
            # Keep the current step valid across the next inference period
            # plus one downstream scheduling horizon; this is hold-last
            # fallback, not future-action prediction.
            action_maximum_hold_sec=self._maximum_hold_sec,
            extend_final_keyframe_to_maximum_hold=True,
            action_chunk_execution_steps=1,
            policy_observation_binding_id="temp_ppo_hil_v1",
        )
        self.runtime = _TempPPOPolicyDeployRuntime(
            runtime_config,
            monotonic_clock=monotonic_clock,
            decoder=self.decoder,
        )
        self._model = _TempPPOModel(
            self.policy,
            self.normalizer,
            build_policy_action_transform(self.task_config),
            task_config=self.task_config,
        )
        self._preparer = _TempPPOActionPreparer(
            DeployRobotActionAdapter(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=True,
            )
        )
        self._started = False
        self._last_emission: PolicyActionEmission | None = None
        self._last_prediction_result: Any | None = None
        self._last_execution_feedback: PolicyExecutionFeedback | None = None
        self._last_feedback_application: (
            PolicyExecutionFeedbackApplication | None
        ) = None
        self._last_error: str | None = None

    def _validate_profile(self) -> None:
        validate_temp_ppo_hil_profile(self.task_config, self.policy)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def ready(self) -> bool:
        return self._last_error is None and self._started

    @property
    def last_emission(self) -> PolicyActionEmission | None:
        return self._last_emission

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def policy_period_sec(self) -> float:
        return float(self.task_config.policy_period_sec)

    @property
    def maximum_hold_sec(self) -> float:
        return self._maximum_hold_sec

    def start(self) -> None:
        if self._started:
            return
        self.reset_for_activation()
        self._started = True

    def reset_for_activation(self) -> None:
        """Reset policy-local state after a new composition activation barrier.

        This is still the lower-owned PPO queue/schedule seam.  It does not
        touch composition activation, authority, safety, or execution state.
        """

        if not self._started and self._last_emission is not None:
            raise RuntimeError("Cannot reset a stopped PPO source with an emission.")
        self.observation_adapter.reset()
        self.decoder.reset()
        self.runtime.clear_action_queue()
        self.runtime.reset_prediction_schedule()
        self._last_emission = None
        self._last_prediction_result = None
        self._last_execution_feedback = None
        self._last_feedback_application = None
        self._last_error = None

    def close(self) -> None:
        self._started = False
        self.runtime.clear_action_queue()
        self._last_emission = None

    def observe_execution_feedback(
        self,
        feedback: PolicyExecutionFeedback,
    ) -> PolicyExecutionFeedbackApplication:
        """Record exact downstream evidence without changing policy state.

        Sequential-delta anchoring remains ``previous_policy_target``.  A
        future profile may explicitly opt into downstream-reference rebase,
        but doing that implicitly would silently change policy semantics.
        """

        if not isinstance(feedback, PolicyExecutionFeedback):
            raise TypeError("feedback must be PolicyExecutionFeedback.")
        generation = self.decoder.target_generation
        last = self._last_emission
        if (
            last is None
            or feedback.policy_emission_id != last.emission_id
            or feedback.policy_emission_hash != last.semantic_content_hash
        ):
            decision = "ignored_stale_or_foreign_emission"
            reason = "feedback does not bind the source's latest emission"
        else:
            decision = "recorded_no_state_mutation"
            reason = (
                "sequential target anchor is policy-owned; downstream facts "
                "are passive evidence"
            )
            self._last_execution_feedback = feedback
        application = PolicyExecutionFeedbackApplication(
            feedback_id=feedback.feedback_id,
            feedback_hash=feedback.semantic_content_hash,
            decision=decision,
            reason=reason,
            target_generation_before=generation,
            target_generation_after=self.decoder.target_generation,
        )
        self._last_feedback_application = application
        return application

    def step(
        self,
        *,
        now: float | None = None,
        step_index: int | None = None,
    ) -> PolicyActionEmission | None:
        if not self._started:
            raise RuntimeError("TempPPOHILPolicySource.start() is required.")
        timestamp = (
            _finite_time(self.clock(), name="clock")
            if now is None
            else _finite_time(now, name="now")
        )
        prediction_due = self.runtime.should_predict(
            now=timestamp,
            policy_interval_s=self.policy_period_sec,
            min_action_queue=0,
        )
        observation = self._scene_capture(
            lambda _scene: self.observation_adapter.capture(
                step_index=step_index,
                commit_policy_history=prediction_due,
            )
        )
        if prediction_due:
            result = self.runtime.predict_and_queue(
                self._model,
                observation,
                now=timestamp,
                queue_policy="replace",
                action_keyframe_period_sec=self.policy_period_sec,
            )
            self._last_prediction_result = result
            if result.error is not None or result.faults:
                error = result.error or ",".join(result.faults)
                self._last_error = f"{type(error).__name__}:{error}" if isinstance(error, BaseException) else str(error)
                return None
        preparation = self.runtime.prepare_next_action(
            observation,
            preparer=self._preparer,
            policy_name="temp_ppo_hil",
        )
        if preparation.emission is None:
            if preparation.emission_error is not None:
                self._last_error = (
                    f"{type(preparation.emission_error).__name__}:"
                    f"{preparation.emission_error}"
                )
            return None
        if preparation.emission.pre_safety_faults or (
            preparation.emission.source_validation is None
            or not preparation.emission.source_validation.accepted
        ):
            self._last_error = "temporary PPO pre-safety adapter rejected action"
            return None
        self._last_emission = preparation.emission
        self._last_error = None
        return preparation.emission

    def status(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = self.clock() if now is None else now
        age = None
        if self._last_emission is not None:
            age = float(timestamp) - float(self._last_emission.emitted_at)
        lineage = getattr(self._last_prediction_result, "chunk_lineage", None)
        inference_duration = (
            None
            if lineage is None
            else float(lineage.inference_completed_at)
            - float(lineage.inference_started_at)
        )
        return {
            "source_id": TEMP_PPO_HIL_POLICY_ID_V1,
            "started": self._started,
            "ready": self.ready,
            "checkpoint": str(self.checkpoint_path),
            "task_config": str(self.task_config_path),
            "policy_node_id": self.runtime.config.policy_node_id,
            "clock_domain_id": self.runtime.config.policy_clock_domain_id,
            "policy_period_sec": self.policy_period_sec,
            "queue_length": self.runtime.action_queue_length(),
            "last_emission_age_sec": age,
            "last_inference_duration_sec": inference_duration,
            "last_execution_feedback_id": (
                None
                if self._last_execution_feedback is None
                else self._last_execution_feedback.feedback_id
            ),
            "last_feedback_application": (
                None
                if self._last_feedback_application is None
                else {
                    "receipt_id": self._last_feedback_application.receipt_id,
                    "decision": self._last_feedback_application.decision,
                    "reason": self._last_feedback_application.reason,
                }
            ),
            "last_error": self._last_error,
            "unused_action_components": [],
            "policy_hand_control": "right_gripper_delta_to_absolute_qpos_v1",
            "compatibility": dict(self.compatibility),
        }


__all__ = [
    "TEMP_PPO_HIL_OBSERVATION_ADAPTER_ID_V1",
    "TEMP_PPO_HIL_POLICY_ID_V1",
    "TempPPOHILObservationAdapter",
    "TempPPOHILPolicySource",
    "validate_temp_ppo_hil_profile",
]
