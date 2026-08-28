"""MuJoCo Sim-HIL binding for a state-only behavior-cloning policy.

The state-BC learner owns a 16-D desk-frame state/action contract.  This
module only binds that adapter to the already-authoritative Sim-HIL scene and
the backend-neutral threaded emission source.  It does not own source
selection, safety, IK, reference scheduling, or MuJoCo execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_motion.action import ArmActionSpec, HandActionSpec
from robot_motion.deploy import (
    POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
)
from robot_motion.deploy.action_adapter import DeployRobotActionAdapter
from robot_motion.deploy.emission_source import (
    DeployActionPreparer,
    ThreadedPolicyEmissionSource,
    ThreadedPolicyEmissionSourceConfig,
)
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)

from sim.mujoco.deploy.deployment_observation import (
    MujocoDeploymentObservationAdapter,
    MujocoDeploymentObservationConfig,
)
from sim.mujoco.deploy.state_policy_adapter import (
    STATE_BC_ACTION_DIM,
    STATE_BC_LAYOUT_HASH_V1,
    STATE_BC_MODEL_SCHEMA_V1,
    STATE_BC_STATE_DIM,
    StatePolicyAdapter,
    StatePolicyAdapterConfig,
)
from sim.mujoco.deploy.target_measurement import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)


STATE_BC_HIL_POLICY_ID_V1 = "sim.mujoco.state_bc_hil.v1"
STATE_BC_HIL_OBSERVATION_BINDING_ID_V1 = "state_bc_hil_v1"


@dataclass(frozen=True)
class MujocoStateBCPolicySourceProfile:
    """Concrete state-BC artifact and cadence selection for Sim-HIL."""

    policy_artifact_path: Path
    action_hz: float = 15.0
    policy_hz: float = 15.0
    prediction_request_queue_threshold: int = 1
    output_active_sides: tuple[str, ...] = ("right",)
    model_observation_sides: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_artifact_path",
            Path(self.policy_artifact_path).expanduser(),
        )
        for name in ("action_hz", "policy_hz"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        threshold = int(self.prediction_request_queue_threshold)
        if threshold < 0 or threshold != self.prediction_request_queue_threshold:
            raise ValueError(
                "prediction_request_queue_threshold must be a non-negative integer."
            )
        object.__setattr__(self, "prediction_request_queue_threshold", threshold)
        output_sides = tuple(str(side).strip().lower() for side in self.output_active_sides)
        model_sides = tuple(str(side).strip().lower() for side in self.model_observation_sides)
        if output_sides != ("right",):
            raise ValueError(
                "The first state-BC Sim-HIL binding contributes the right side only."
            )
        if model_sides != ("left", "right"):
            raise ValueError(
                "state BC model_observation_sides must be ('left', 'right')."
            )
        object.__setattr__(self, "output_active_sides", output_sides)
        object.__setattr__(self, "model_observation_sides", model_sides)


class _StateBCPolicyInput:
    """Adapt the actor-view firewall output to StatePolicyAdapter's contract."""

    name = "state-bc"

    def __init__(self, adapter: StatePolicyAdapter) -> None:
        self.adapter = adapter

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "state_bc",
            "artifact_hash": self.adapter.artifact_hash,
            "artifact_schema": STATE_BC_MODEL_SCHEMA_V1,
            "state_dim": STATE_BC_STATE_DIM,
            "action_dim": STATE_BC_ACTION_DIM,
        }

    def predict(self, observation: Any) -> Any:
        # Strict PolicyDeployRuntime mode supplies a sanitized immutable view.
        # The state adapter deliberately consumes the legacy deployment shape
        # so this conversion remains at this concrete binding, not in the
        # backend-neutral runtime.
        if isinstance(observation, PolicyObservationView):
            observation = observation.to_deployment_observation()
        return self.adapter.predict(observation)

    def reset(self) -> None:
        self.adapter.reset()


def _build_state_policy_adapter(
    profile: MujocoStateBCPolicySourceProfile,
    *,
    pose_contract: Any | None = None,
) -> StatePolicyAdapter:
    contract = (
        default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
        if pose_contract is None
        else pose_contract
    )
    return StatePolicyAdapter(
        artifact_path=profile.policy_artifact_path,
        config=StatePolicyAdapterConfig(
            active_sides=profile.model_observation_sides,
            # FormalCollection state/action vectors are desk_absolute.  The
            # adapter converts both observations and actions through this
            # reviewed runtime contract; the shared runtime receives raw pose
            # vectors after the adapter boundary.
            policy_absolute_frame="canonical_desk",
            runtime_pose_frame_contract=contract,
            expected_layout_hash=STATE_BC_LAYOUT_HASH_V1,
            source=STATE_BC_HIL_POLICY_ID_V1,
            metadata={
                "policy_source": "state_bc",
                "state_bc_model_state_dim": STATE_BC_STATE_DIM,
                "state_bc_model_action_dim": STATE_BC_ACTION_DIM,
                "state_bc_layout_hash": STATE_BC_LAYOUT_HASH_V1,
            },
        ),
    )


def preflight_mujoco_state_bc_policy_source(
    profile: MujocoStateBCPolicySourceProfile,
) -> dict[str, Any]:
    """Validate a state-BC artifact without constructing a scene or worker."""

    if not isinstance(profile, MujocoStateBCPolicySourceProfile):
        raise TypeError("profile must be a MujocoStateBCPolicySourceProfile.")
    adapter = _build_state_policy_adapter(profile)
    metadata = dict(adapter.artifact_metadata)
    return {
        "artifact_schema": metadata.get("schema"),
        "artifact_hash": adapter.artifact_hash,
        "selector_id": metadata.get("selector_id"),
        "model_type": metadata.get("model_type"),
        "layout_hash": metadata.get("layout_hash"),
        "policy_dimensions": [STATE_BC_STATE_DIM, STATE_BC_ACTION_DIM],
        "policy_topology": {
            "model_observation_sides": list(profile.model_observation_sides),
            "output_active_sides": list(profile.output_active_sides),
        },
        "prepared_artifact_required": False,
        "state_only_capability": True,
        "server_connection_checked": False,
        "full_bimanual_execution": False,
        "projection": "bimanual_state_bc_to_right_arm_and_gripper_contributor",
    }


def build_mujoco_state_bc_policy_source(
    profile: MujocoStateBCPolicySourceProfile,
    context: Any,
) -> ThreadedPolicyEmissionSource:
    """Bind one state-BC artifact to the existing streaming policy seam."""

    if not isinstance(profile, MujocoStateBCPolicySourceProfile):
        raise TypeError("profile must be a MujocoStateBCPolicySourceProfile.")
    sides = profile.model_observation_sides
    pose_contract = default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    arm_specs = {
        side: ArmActionSpec.tool_tcp(side, reference_frame=f"{side}_arm_base")
        for side in sides
    }
    hand_specs = {side: HandActionSpec.dmtac_gripper(side) for side in sides}
    observation_adapter = MujocoDeploymentObservationAdapter(
        context.scene,
        MujocoDeploymentObservationConfig(
            active_sides=sides,
            tcp_body_by_side={
                side: context.scene.arm_specs[side].controlled_body
                for side in sides
            },
            hands_enabled=True,
            end_effector_sides=sides,
            gripper_name_by_side={
                "left": context.task_config.left_gripper_name,
                "right": context.task_config.right_gripper_name,
            },
            pose_frame_contract=pose_contract,
            metadata={
                "arm_action_specs_by_side": {
                    side: spec.to_metadata() for side, spec in arm_specs.items()
                },
                "arm_controlled_frame_by_side": dict.fromkeys(sides, "tool_tcp"),
                "arm_reference_frame_by_side": {
                    side: f"{side}_arm_base" for side in sides
                },
                "hand_action_specs_by_side": {
                    side: spec.to_metadata() for side, spec in hand_specs.items()
                },
                "policy_source": "state_bc",
                "tcp_pose_semantics": "T_arm_base_tool_tcp",
            },
        ),
        tcp_pose_by_side_provider=lambda: capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            context.scene,
            context.task_config,
        ),
        clock=context.target_runtime.clock,
    )
    policy_adapter = _build_state_policy_adapter(
        profile,
        pose_contract=pose_contract,
    )
    policy = _StateBCPolicyInput(policy_adapter)

    def capture(*, step_index: int | None = None):
        return context.scene_capture(
            lambda _scene: observation_adapter.capture(step_index=step_index)
        )

    action_period = 1.0 / profile.action_hz
    maximum_hold = max(
        0.25,
        2.0 / profile.policy_hz + max(0.0, context.downstream_target_lead_sec),
    )
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=profile.output_active_sides,
            model_observation_sides=profile.model_observation_sides,
            arms_enabled=True,
            hands_enabled=True,
            policy_action_pose_mode="absolute",
            # StatePolicyAdapter has already converted desk_absolute actions to
            # runtime_raw before they enter this generic stream runtime.
            policy_absolute_frame=POLICY_ABSOLUTE_FRAME_RUNTIME_RAW,
            runtime_pose_frame_contract=pose_contract,
            policy_node_id=context.target_runtime.config.policy_node_id,
            policy_clock_domain_id=context.target_runtime.config.clock_domain_id,
            action_keyframe_period_sec=action_period,
            action_maximum_hold_sec=maximum_hold,
            extend_final_keyframe_to_maximum_hold=True,
            action_chunk_execution_steps=1,
            policy_observation_binding_id=STATE_BC_HIL_OBSERVATION_BINDING_ID_V1,
        ),
        monotonic_clock=context.target_runtime.clock,
    )
    return ThreadedPolicyEmissionSource(
        ThreadedPolicyEmissionSourceConfig(
            source_id=STATE_BC_HIL_POLICY_ID_V1,
            policy_name=policy.name,
            policy_interval_sec=1.0 / profile.policy_hz,
            action_period_sec=action_period,
            maximum_hold_sec=maximum_hold,
            prediction_request_queue_threshold=(
                profile.prediction_request_queue_threshold
            ),
            status_metadata={
                "backend": "state_bc",
                "artifact_path": str(profile.policy_artifact_path),
                "artifact_hash": policy_adapter.artifact_hash,
                "artifact_schema": STATE_BC_MODEL_SCHEMA_V1,
                "layout_hash": STATE_BC_LAYOUT_HASH_V1,
                "model_state_dim": STATE_BC_STATE_DIM,
                "model_action_dim": STATE_BC_ACTION_DIM,
                "model_observation_sides": profile.model_observation_sides,
                "output_active_sides": profile.output_active_sides,
                "state_only_capability": True,
                "full_bimanual_execution": False,
            },
        ),
        runtime=runtime,
        policy=policy,
        observation_capture=capture,
        observation_reset=observation_adapter.reset,
        last_action_sink=lambda action: observation_adapter.set_last_action(
            action.vector
        ),
        action_preparer=DeployActionPreparer(
            DeployRobotActionAdapter(
                active_sides=profile.output_active_sides,
                arms_enabled=True,
                hands_enabled=True,
            )
        ),
        monotonic_clock=context.target_runtime.clock,
    )


__all__ = [
    "MujocoStateBCPolicySourceProfile",
    "STATE_BC_HIL_OBSERVATION_BINDING_ID_V1",
    "STATE_BC_HIL_POLICY_ID_V1",
    "build_mujoco_state_bc_policy_source",
    "preflight_mujoco_state_bc_policy_source",
]
