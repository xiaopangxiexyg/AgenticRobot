"""Temporary-PPO contributor adapter for MuJoCo Sim-HIL.

The temporary CubeLift PPO checkpoint is not a generic deploy policy: it emits
seven normalized values (right TCP xyz/rotvec increments plus a gripper
increment) and was trained against a 28-value task-local observation.  This
module keeps those semantics in the temporary pipeline while exposing the
shared ``PolicyActionEmission`` seam consumed by ``control_station``.

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
)
from robot_motion.deploy.action_adapter import DeployRobotActionAdapter
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
from sim.mujoco.tasks.cube_lift.spec import (
    CubeLiftActionSpec,
    CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION,
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


class TempPPOHILObservationAdapter:
    """Project the shared MuJoCo observation into the checkpoint's 28-D state.

    The shared deployment observation remains 38-D so it can pass the common
    observation firewall.  The task-local 28-D PPO state is stored in its
    prefix and the predictor explicitly consumes that prefix.  Target-error
    fields are zeroed because reconstructing the old CubeLift internal qpos
    target before composition would introduce the forbidden second IK owner.
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

    def capture(self, *, step_index: int | None = None) -> DeploymentObservation:
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
        state28 = np.zeros(28, dtype=np.float32)
        state28[:6] = measured_qpos
        state28[6] = measured_close
        state28[7:13] = measured_qpos - previous_qpos
        state28[13] = measured_close - previous_close
        # 14:20 and 20 are the historical target-error fields.  They remain
        # explicit zero fallbacks until a post-composition target is available.
        state28[21:28] = self._last_compact_action
        self._previous_qpos = measured_qpos.copy()
        self._previous_close = measured_close

        state = np.zeros(38, dtype=np.float32)
        state[:28] = state28
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
                    CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION
                ),
                "policy_observation_dim": 28,
                "policy_observation_compact_last_action": (
                    self._last_compact_action.copy()
                ),
            }
        )
        state_dict = dict(base_observation.state_dict)
        state_dict.update(
            {
                "temp_ppo_state": state28.copy(),
                "right_target_error": np.zeros(6, dtype=np.float32),
                "right_gripper_target_error": 0.0,
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

    def __init__(self, policy: Any, normalizer: Any, *, observation_dim: int) -> None:
        self.policy = policy
        self.normalizer = normalizer
        self.observation_dim = int(observation_dim)
        self.device = next(policy.parameters()).device

    def predict(self, observation: PolicyObservationView) -> np.ndarray:
        if not isinstance(observation, PolicyObservationView):
            raise TypeError("Temporary PPO requires a PolicyObservationView.")
        state = np.asarray(observation.model_input()["state"], dtype=np.float32).reshape(-1)
        if state.shape != (38,):
            raise ValueError(f"Shared observation state must be 38-D, got {state.shape}.")
        compact_state = state[: self.observation_dim]
        normalized = self.normalizer.normalize(compact_state)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _, _ = self.policy.act(tensor.unsqueeze(0), deterministic=True)
            action = self.policy.expand_action(action)
        values = action[0].detach().cpu().numpy().astype(np.float32, copy=False)
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
    ) -> None:
        self.scene = scene
        self.task_config = task_config
        self.observation_adapter = observation_adapter
        from robot_motion.rm65 import endpoint_to_tool_transform, load_rm65_tool_profile

        self._endpoint_to_tool = endpoint_to_tool_transform(
            load_rm65_tool_profile(task_config.right_tool_profile_path, side="right")
        )
        self._action_spec = CubeLiftActionSpec(
            layout_version=task_config.action_layout
        )
        self._right_hand_spec = HandActionSpec.dmtac_gripper("right")
        self._target_pose_arm_base: np.ndarray | None = None
        self._target_gripper_close: float | None = None

    def reset(self) -> None:
        self._target_pose_arm_base = None
        self._target_gripper_close = None

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
        del observation
        values = self._action_spec.coerce(action)
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
            "pose_action_source_mode": "absolute",
            "pose_action_mode": "absolute",
            "pose_action_source_absolute_frame": "right_arm_base",
            "pose_action_output_absolute_frame": "runtime_raw",
            "pose_anchor_policy": "maintained_absolute_target_from_measured_tcp",
            "pose_anchor_sides": ("right",),
            "temp_ppo_hil_profile": TEMP_PPO_HIL_POLICY_ID_V1,
            "unused_action_components": (),
            "policy_hand_control": "right_gripper_delta_to_absolute_qpos_v1",
            "right_gripper_target_close": self._target_gripper_close,
            "action_protocol": "robot_action_v1",
            "policy_action_layout": CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION,
            "observation_target_error_source": "measured_qpos_zero_fallback_v1",
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
    ) -> None:
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable.")
        self.scene = scene
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
            scene, self.task_config, self.observation_adapter
        )
        period = float(self.task_config.policy_period_sec)
        runtime_config = PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=True,
            policy_action_pose_mode="absolute",
            policy_absolute_frame="runtime_raw",
            policy_node_id=str(policy_node_id),
            policy_clock_domain_id=str(policy_clock_domain_id),
            action_keyframe_period_sec=period,
            action_maximum_hold_sec=period,
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
            observation_dim=int(self.task_config.observation_dim),
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
        self._last_error: str | None = None

    def _validate_profile(self) -> None:
        if not self.task_config.uses_pose_action:
            raise ValueError("Temporary PPO Sim-HIL requires a pose-action task.")
        if self.task_config.action_layout != CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION:
            raise ValueError("Temporary PPO task action layout is not the pose layout.")
        if (
            self.task_config.observation_layout
            != CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION
        ):
            raise ValueError(
                "Temporary PPO Sim-HIL requires the 28-D no-episode-clock observation."
            )
        if self.task_config.observation_dim != 28 or self.task_config.action_dim != 7:
            raise ValueError("Temporary PPO Sim-HIL requires task dimensions 28 -> 7.")
        if self.policy.observation_dim != 28 or self.policy.action_dim != 7:
            raise ValueError(
                "Selected PPO checkpoint must have policy dimensions 28 -> 7."
            )
        if self.policy.environment_action_dim != 7:
            raise ValueError("Temporary PPO checkpoint environment action dim must be 7.")

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
        self._last_error = None

    def close(self) -> None:
        self._started = False
        self.runtime.clear_action_queue()
        self._last_emission = None

    def step(
        self,
        *,
        now: float | None = None,
        step_index: int | None = None,
    ) -> PolicyActionEmission | None:
        if not self._started:
            raise RuntimeError("TempPPOHILPolicySource.start() is required.")
        observation = self.observation_adapter.capture(step_index=step_index)
        timestamp = observation.timestamp if now is None else _finite_time(now, name="now")
        if self.runtime.should_predict(
            now=timestamp,
            policy_interval_s=self.policy_period_sec,
            min_action_queue=0,
        ):
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
]
