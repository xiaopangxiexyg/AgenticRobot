"""MuJoCo target adapters used by the Control Station Sim-HIL profile.

This module contains only target-side measurement binding.  It does not own
composition, authority, canonical safety application, IK, or dispatch.  The
Station application supplies the profile selection and calls this adapter when
the shared safety callback needs a current measured anchor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    CanonicalActionSafetyAnchor,
    ComponentActionSchema,
    ExecutionObservationPlan,
    ExecutionObservationSink,
    HandActionSpec,
)
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE,
    POSITION_45_150_PROFILE,
    POSITION_60_120_PROFILE,
    POSITION_PROFILE,
    PositionControlProfile,
)
from robot_motion.rm65 import endpoint_to_tool_transform, load_rm65_tool_profile


_POSITION_CONTROL_PROFILES_BY_NAME = {
    profile.name: profile
    for profile in (
        POSITION_PROFILE,
        POSITION_15_60_120_PROFILE,
        POSITION_60_120_PROFILE,
        POSITION_45_150_PROFILE,
    )
}


@dataclass(frozen=True)
class MujocoSimHILTargetConfig:
    """Target-side selection for the Station Sim-HIL profile."""

    task_config_path: Path
    realtime_worker: bool = True
    runtime_session_id: str = "control-station-sim-hil-ppo-session"
    control_domain_id: str = "mujoco-sim-hil"
    embodiment_instance_id: str = "mujoco-rm65-sim-hil"
    producer_id: str = "sim-hil-temp-ppo"
    composer_producer_id: str = "control-station-sim-hil-composer"
    policy_node_id: str = "sim-hil-temp-ppo"
    clock_domain_id: str = "mujoco.sim_hil.monotonic"
    # Sim-HIL receives already-composed targets at the downstream reference
    # cadence by default.  The old 15->60->120 contract remains selectable by
    # name and is not changed for replay or ordinary policy deployment.
    position_control_profile_name: str = POSITION_60_120_PROFILE.name
    # Optional explicit final compose/authority cadence.  When omitted, the
    # selected profile's accepted input cadence is used.
    compose_target_hz: float | None = None
    execution_observation_plan: ExecutionObservationPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_config_path", Path(self.task_config_path).expanduser())
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "embodiment_instance_id",
            "producer_id",
            "composer_producer_id",
            "policy_node_id",
            "clock_domain_id",
            "position_control_profile_name",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if self.compose_target_hz is not None:
            compose_target_hz = float(self.compose_target_hz)
            if not np.isfinite(compose_target_hz) or compose_target_hz <= 0.0:
                raise ValueError("compose_target_hz must be finite and positive.")
            object.__setattr__(self, "compose_target_hz", compose_target_hz)
        if self.execution_observation_plan is not None and not isinstance(
            self.execution_observation_plan,
            ExecutionObservationPlan,
        ):
            raise TypeError(
                "execution_observation_plan must be an "
                "ExecutionObservationPlan or None."
            )


@dataclass(frozen=True)
class MujocoSimHILTarget:
    """Already-owned MuJoCo target resources selected by the Station."""

    task_config: Any
    scene: Any
    runtime: Any
    hand_specs_by_side: Mapping[str, HandActionSpec]


def build_mujoco_sim_hil_target(
    *,
    config: MujocoSimHILTargetConfig,
    contracts: Any,
    hand_specs_by_side: Mapping[str, HandActionSpec],
    composition_input_action_schema_by_role: Mapping[
        str, ComponentActionSchema
    ]
    | None = None,
    execution_observation_sink: ExecutionObservationSink | None = None,
) -> MujocoSimHILTarget:
    """Build scene, shared safety, coupled IK and one deploy runtime.

    The composition contract is supplied by the application leaf as an
    opaque binding.  This target factory only uses its already-built strategy
    and compound authority profile; it does not define composition semantics.
    """

    if not isinstance(config, MujocoSimHILTargetConfig):
        raise TypeError("config must be a MujocoSimHILTargetConfig.")
    if not hasattr(contracts, "authority_profile") or not hasattr(contracts, "strategy"):
        raise TypeError("contracts must expose authority_profile and strategy.")
    hand_specs = {
        str(side): (
            spec
            if isinstance(spec, HandActionSpec)
            else HandActionSpec.from_metadata(spec)
        )
        for side, spec in dict(hand_specs_by_side).items()
    }
    if set(hand_specs) != {"left", "right"}:
        raise ValueError("MuJoCo Sim-HIL target requires left and right hand specs.")

    from robot_motion.action import (
        CanonicalKeyframeSafetyProfile,
        HandQposSafetyLimits,
        SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
        apply_canonical_keyframe_safety,
    )
    from robot_motion.position_control import PositionKeyframeSafetyConfig
    from sim.contracts import TimingSpec
    from sim.mujoco.actuation import (
        identified_bimanual_dmtac_gripper_profile,
        rm65_bimanual_udp_feedback_profile,
    )
    from sim.mujoco.adapters import (
        MujocoPositionExecutionConfig,
        MujocoPositionExecutionRig,
    )
    from sim.mujoco.ik import RM65CanonicalIK
    from sim.mujoco.scene import MujocoScene
    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig
    from .pose_resolution import MujocoCoupledPoseToQposResolver
    from .runtime import MujocoDeployConfig, MujocoDeployRuntime

    task_config = CubeLiftTaskConfig.load(config.task_config_path)
    contract = task_config.base_contract()
    try:
        position_profile: PositionControlProfile = _POSITION_CONTROL_PROFILES_BY_NAME[
            str(config.position_control_profile_name).strip()
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported Sim-HIL position control profile: "
            f"{config.position_control_profile_name!r}."
        ) from exc
    # The task contract owns the simulation physics budget, while the selected
    # position profile owns the target/reference/IK/command clocks.  Refuse a
    # silent mixed profile instead of allowing task metadata and the execution
    # capability manifest to disagree.
    if not np.isclose(
        float(task_config.reference_hz),
        float(position_profile.rates.reference_hz),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Sim-HIL task reference_hz does not match the selected position "
            f"profile {position_profile.name!r}."
        )
    if not np.isclose(
        float(task_config.command_hz),
        float(position_profile.rates.command_hz),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "Sim-HIL task command_hz does not match the selected position "
            f"profile {position_profile.name!r}."
        )
    compose_target_hz = position_profile.validate_compose_target_hz(
        position_profile.rates.sample_hz
        if config.compose_target_hz is None
        else config.compose_target_hz
    )
    timing = TimingSpec(
        # ``policy_hz`` remains the model/source clock.  ``sample_hz`` is the
        # final compose target cadence consumed by the downstream controller.
        sample_hz=compose_target_hz,
        policy_hz=task_config.policy_hz,
        reference_hz=position_profile.rates.reference_hz,
        ik_hz=position_profile.rates.ik_hz,
        command_hz=position_profile.rates.command_hz,
        physics_hz=task_config.physics_hz,
        parameter_source=(
            f"sim_mujoco_sim_hil:{position_profile.name}:"
            f"compose_{compose_target_hz:g}hz"
        ),
    )
    arm_profile = (
        task_config.position_actuation_profile()
        or rm65_bimanual_udp_feedback_profile()
    )
    gripper_profile = (
        task_config.gripper_actuation_profile()
        or identified_bimanual_dmtac_gripper_profile()
    )
    scene = MujocoScene.from_contract(
        contract,
        position_actuation_profile=arm_profile,
        gripper_actuation_profile=gripper_profile,
    )
    scene.reset_data()
    scene.set_qpos_by_side(
        {"left": task_config.left_home_qpos, "right": task_config.right_home_qpos}
    )
    for name, close in task_config.initial_gripper_close_by_name.items():
        scene.set_gripper_close(name, float(close))
    scene.set_freejoint_state(
        task_config.object_freejoint_name,
        qpos=task_config.cube_spawn_qpos,
        qvel=(0.0,) * 6,
    )
    scene.forward()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=timing,
            active_sides=("left", "right"),
            actuation_profile=arm_profile,
            gripper_actuation_profile=gripper_profile,
            realtime_worker=config.realtime_worker,
            follow_qpos_alpha=1.0,
        ),
    )
    safety_profile = CanonicalKeyframeSafetyProfile(
        profile_id="control-station-mujoco-sim-hil-safety-v1",
        limits=PositionKeyframeSafetyConfig(workspace_enabled=False),
        hand_limits_by_side={
            side: HandQposSafetyLimits.uniform(spec, lower=0.0, upper=1.0)
            for side, spec in hand_specs.items()
        },
    )

    def shared_safety(action: Any, input_id: str) -> Any:
        anchor = capture_mujoco_sim_hil_safety_anchor(
            scene,
            task_config,
            hand_specs,
            now=time.monotonic(),
        )
        return apply_canonical_keyframe_safety(
            action,
            input_action_id=input_id,
            anchor=anchor,
            profile=safety_profile,
        )

    ik = RM65CanonicalIK.from_tool_profiles(
        scene,
        {
            "left": task_config.left_tool_profile_path,
            "right": task_config.right_tool_profile_path,
        },
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id=config.runtime_session_id,
            control_domain_id=config.control_domain_id,
            embodiment_instance_id=config.embodiment_instance_id,
            producer_id=config.producer_id,
            composer_producer_id=config.composer_producer_id,
            enable_primary_policy_authority=False,
            enable_composer_authority=True,
            enable_replay_authority=False,
            producer_epoch=1,
            policy_node_id=config.policy_node_id,
            # The authoritative compose schedule is independent from the
            # lower policy source/inference schedule.
            decision_hz=compose_target_hz,
            reference_activation_profile_id=SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
            proposal_horizon_sec=0.1,
            clock_domain_id=config.clock_domain_id,
            pose_frame_contract_hash="mujoco-sim-hil-rm65-arm-base-v1",
            compound_hand_action_specs_by_side=hand_specs,
            compound_composer_authority_profile=contracts.authority_profile,
            initial_control_mode="policy",
            enable_component_execution_handoff=True,
            execution_observation_plan=config.execution_observation_plan,
        ),
        pose_resolver=MujocoCoupledPoseToQposResolver(ik),
        shared_safety=shared_safety,
        composition_strategy=contracts.strategy,
        composition_input_action_schema_by_role=(
            composition_input_action_schema_by_role
        ),
        execution_observation_sink=execution_observation_sink,
    )
    return MujocoSimHILTarget(
        task_config=task_config,
        scene=scene,
        runtime=runtime,
        hand_specs_by_side=hand_specs,
    )


def capture_mujoco_sim_hil_safety_anchor(
    scene: Any,
    task_config: Any,
    hand_specs: Mapping[str, HandActionSpec],
    *,
    now: float,
) -> CanonicalActionSafetyAnchor:
    """Capture one measured bimanual pose/hand anchor for shared safety."""

    arm_pose_by_side: dict[str, np.ndarray] = {}
    tool_profiles = {
        "left": load_rm65_tool_profile(task_config.left_tool_profile_path, side="left"),
        "right": load_rm65_tool_profile(task_config.right_tool_profile_path, side="right"),
    }
    for side in ("left", "right"):
        candidates = (f"{side}_arm_base_link", "arm_base_link", "base_link")
        base_name = next((name for name in candidates if scene.has_body(name)), None)
        if base_name is None:
            raise KeyError(f"Could not resolve {side} arm base body.")
        endpoint = scene.body_pose(scene.arm_specs[side].controlled_body)
        pose = np.linalg.inv(scene.body_pose(base_name)) @ endpoint
        pose = pose @ endpoint_to_tool_transform(tool_profiles[side])
        vector = pose_matrix_to_xyzquat(pose)
        if vector is None:
            raise ValueError(f"Current {side} TCP pose cannot be represented as xyzquat.")
        arm_pose_by_side[side] = np.asarray(vector, dtype=np.float32)

    hand_qpos_by_side: dict[str, np.ndarray] = {}
    for side, name in (
        ("left", task_config.left_gripper_name),
        ("right", task_config.right_gripper_name),
    ):
        values = np.zeros(6, dtype=np.float32)
        values[0] = float(scene.get_gripper_close(name))
        hand_qpos_by_side[side] = values
    return CanonicalActionSafetyAnchor(
        anchor_id=f"mujoco-sim-hil-measured-anchor:{float(now):.9f}",
        anchor_kind="measured_state",
        action_space=BIMANUAL_EE_POSE_V1,
        arm_pose_by_side=arm_pose_by_side,
        hand_qpos_by_side=hand_qpos_by_side,
        hand_action_specs_by_side=hand_specs,
        frame_by_side={side: f"{side}_arm_base" for side in ("left", "right")},
        observed_at=float(now),
    )


__all__ = [
    "MujocoSimHILTarget",
    "MujocoSimHILTargetConfig",
    "build_mujoco_sim_hil_target",
    "capture_mujoco_sim_hil_safety_anchor",
]
