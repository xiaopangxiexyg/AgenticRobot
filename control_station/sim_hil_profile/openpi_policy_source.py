"""Sim-HIL application binding for MuJoCo observations and OpenPI emissions.

Only scene sampling is simulator-specific.  Artifact validation, model
transport, chunk scheduling and emission creation remain in their lower-owned
generic packages; this module owns no composition, safety, IK or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deployments.openpi.client import (
    OpenPIEmissionSourceConfig,
    build_openpi_policy_emission_source,
    validate_openpi_state_policy_artifact,
)
from deployments.policy_artifact import load_policy_artifact
from robot_motion.action import ArmActionSpec, HandActionSpec
from robot_motion.pose_frame_profiles import (
    default_rm65_workspace_near_boundary_runtime_pose_frame_contract,
)

from sim.mujoco.deploy.deployment_observation import (
    MujocoDeploymentObservationAdapter,
    MujocoDeploymentObservationConfig,
)
from sim.mujoco.deploy.sim_hil_target import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)


@dataclass(frozen=True)
class MujocoOpenPIStatePolicySourceProfile:
    """Concrete model/scene selection without Station lifecycle ownership."""

    policy_artifact_path: Path
    host: str = "127.0.0.1"
    port: int = 8000
    prompt: str | None = None
    action_hz: float = 15.0
    policy_hz: float = 15.0
    action_chunk_execution_steps: int | None = None
    prediction_request_queue_threshold: int = 4
    output_active_sides: tuple[str, ...] = ("right",)
    model_observation_sides: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "policy_artifact_path",
            Path(self.policy_artifact_path).expanduser(),
        )

    def openpi_config(
        self,
        *,
        downstream_target_lead_sec: float = 0.0,
        policy_node_id: str = "openpi-policy",
        policy_clock_domain_id: str = "local_monotonic",
    ) -> OpenPIEmissionSourceConfig:
        artifact = load_policy_artifact(self.policy_artifact_path)
        validate_openpi_state_policy_artifact(artifact)
        return OpenPIEmissionSourceConfig(
            policy_artifact=artifact,
            runtime_pose_frame_contract=(
                default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
            ),
            output_active_sides=self.output_active_sides,
            model_observation_sides=self.model_observation_sides,
            host=self.host,
            port=self.port,
            prompt=self.prompt,
            action_hz=self.action_hz,
            policy_hz=self.policy_hz,
            action_chunk_execution_steps=self.action_chunk_execution_steps,
            prediction_request_queue_threshold=(
                self.prediction_request_queue_threshold
            ),
            downstream_target_lead_sec=downstream_target_lead_sec,
            policy_node_id=policy_node_id,
            policy_clock_domain_id=policy_clock_domain_id,
        )


def preflight_mujoco_openpi_state_policy_source(
    profile: MujocoOpenPIStatePolicySourceProfile,
) -> dict[str, Any]:
    """Validate artifact/source bindings without a server, scene or worker."""

    if not isinstance(profile, MujocoOpenPIStatePolicySourceProfile):
        raise TypeError("profile must be a MujocoOpenPIStatePolicySourceProfile.")
    source = profile.openpi_config()
    artifact = source.policy_artifact
    return {
        "model_id": artifact.model_id,
        "checkpoint_step": artifact.checkpoint_step,
        "policy_dimensions": [
            artifact.state_input_dim,
            artifact.policy_io_layout.action_dim,
        ],
        "policy_topology": {
            "model_observation_sides": list(source.model_observation_sides),
            "output_active_sides": list(source.output_active_sides),
        },
        "prepared_artifact_required": True,
        "state_only_capability": True,
        "server_connection_checked": False,
        "full_bimanual_execution": False,
        "projection": "bimanual_model_to_right_arm_and_gripper_contributor",
    }


def build_mujoco_openpi_state_policy_source(
    profile: MujocoOpenPIStatePolicySourceProfile,
    context: Any,
) -> Any:
    """Bind one prepared policy to a context-provided MuJoCo scene snapshot."""

    if not isinstance(profile, MujocoOpenPIStatePolicySourceProfile):
        raise TypeError("profile must be a MujocoOpenPIStatePolicySourceProfile.")
    sides = profile.model_observation_sides
    arm_specs = {
        side: ArmActionSpec.tool_tcp(
            side,
            reference_frame=f"{side}_arm_base",
        )
        for side in sides
    }
    hand_specs = {side: HandActionSpec.dmtac_gripper(side) for side in sides}
    pose_contract = (
        default_rm65_workspace_near_boundary_runtime_pose_frame_contract()
    )
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
                "policy_source": "openpi_state",
                "tcp_pose_semantics": "T_arm_base_tool_tcp",
            },
        ),
        tcp_pose_by_side_provider=lambda: (
            capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                context.scene,
                context.task_config,
            )
        ),
        clock=context.target_runtime.clock,
    )

    def capture(*, step_index: int | None = None):
        return context.scene_capture(
            lambda _scene: observation_adapter.capture(step_index=step_index)
        )

    return build_openpi_policy_emission_source(
        profile.openpi_config(
            downstream_target_lead_sec=context.downstream_target_lead_sec,
            policy_node_id=context.target_runtime.config.policy_node_id,
            policy_clock_domain_id=context.target_runtime.config.clock_domain_id,
        ),
        observation_capture=capture,
        observation_reset=observation_adapter.reset,
        last_action_sink=lambda action: observation_adapter.set_last_action(
            action.vector
        ),
        monotonic_clock=context.target_runtime.clock,
    )


__all__ = [
    "MujocoOpenPIStatePolicySourceProfile",
    "build_mujoco_openpi_state_policy_source",
    "preflight_mujoco_openpi_state_policy_source",
]
