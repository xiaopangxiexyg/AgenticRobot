"""Versioned configuration for the minimal right-arm Cube-Lift task."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from project_paths import resolve_path
from robot_motion.position_control import DEFAULT_POSITION_COMMAND_HZ
from sim.contracts import SimulationContract

from ...actuation import (
    cad_bimanual_dmtac_gripper_profile,
    cube_lift_contact_dmtac_gripper_profile,
    identified_bimanual_dmtac_gripper_profile,
    point_grasp_cad_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
    strong_cad_bimanual_dmtac_gripper_profile,
)
from ...control import DIRECT_QPOS_DEBUG_EXECUTION_MODE
from ...grasp_contact import GripperContactSurfaceSpec

CUBE_LIFT_TASK_SCHEMA_VERSION = "dexteleop.sim.mujoco.cube_lift_task.v0"
CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION = "dexteleop.sim.mujoco.cube_lift_task.v1"
CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION = "dexteleop.sim.mujoco.cube_lift_task.v2"
ACTUATED_POSITION_EXECUTION_MODE = "actuated_position_servo"
MUJOCO_POSITION_ACTUATOR_DYNAMICS = "mujoco_position_actuator"
PRESCRIBED_GRIPPER_DYNAMICS = "prescribed_qpos_debug"
MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS = "mujoco_identified_position_actuator"
RM65_ACTUATION_PROFILE_NAME = "rm65_bimanual_udp_feedback_20260721.v4"
DMTAC_GRIPPER_ACTUATION_PROFILE_NAME = "dmtac_bimanual_position_20260722.v1"
CUBE_LIFT_CONTACT_GRIPPER_ACTUATION_PROFILE_NAME = (
    "dmtac_bimanual_position_cube_lift_contact_20260724.v2"
)
CAD_DMTAC_GRIPPER_ACTUATION_PROFILE_NAME = (
    "dmtac_bimanual_position_eg2_4b2_cad_20260818.v1"
)
NORMAL_POLICY_CLASS = "normal"
PRIVILEGED_EXPERT_POLICY_CLASS = "privileged_expert"
LEGACY_DEBUG_POLICY_CLASS = "legacy_debug"
NORMAL_PROPRIO_PROFILE_V0 = "qpos_close_action_progress.v0"
NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1 = (
    "qpos_close_motion_tracking_action_progress.v1"
)
NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2 = (
    "qpos_close_motion_tracking_no_episode_clock.v2"
)
NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1 = (
    "qpos_close_motion_tracking_no_episode_clock_cube_pose.v1"
)
NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2 = (
    "qpos_close_motion_tracking_no_episode_clock_cube_pose_reach_delta.v2"
)
PHYSICAL_CONTACT_GRASP_MODE = "physical_contact"
CONTACT_LATCH_DEBUG_GRASP_MODE = "contact_latch_debug"
HEIGHT_ONLY_LIFT_SUCCESS_REASON = "height_only_lift_success"
BILATERAL_CONTACT_LIFT_SUCCESS_REASON = "bilateral_contact_lift_success"
TCP_REACH_SUCCESS_REASON = "tcp_reach_success"
BILATERAL_CONTACT_SUCCESS_REASON = "bilateral_contact_success"
SCRIPTED_CONTACT_LOST_REASON = "scripted_contact_lost"
REACH_TASK_OBJECTIVE = "reach"
CONTACT_TASK_OBJECTIVE = "contact"
LIFT_TASK_OBJECTIVE = "lift"
DEFAULT_LEFT_TOOL_PROFILE_PATH = (
    "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"
)
DEFAULT_RIGHT_TOOL_PROFILE_PATH = (
    "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json"
)


def _metadata_sha256(values: Mapping[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rm65_position_actuation_profile_sha256() -> str:
    return _metadata_sha256(rm65_bimanual_udp_feedback_profile().to_metadata())


def resolve_dmtac_gripper_actuation_profile(name: str):
    profiles = {
        profile.name: profile
        for profile in (
            identified_bimanual_dmtac_gripper_profile(),
            cube_lift_contact_dmtac_gripper_profile(),
            cad_bimanual_dmtac_gripper_profile(),
            point_grasp_cad_bimanual_dmtac_gripper_profile(),
            strong_cad_bimanual_dmtac_gripper_profile(),
        )
    }
    try:
        return profiles[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unsupported DM-Tac gripper profile {name!r}.") from exc


def dmtac_gripper_actuation_profile_sha256(
    name: str = DMTAC_GRIPPER_ACTUATION_PROFILE_NAME,
) -> str:
    return _metadata_sha256(resolve_dmtac_gripper_actuation_profile(name).to_metadata())


def is_cube_lift_success_reason(reason: object) -> bool:
    return reason in {
        HEIGHT_ONLY_LIFT_SUCCESS_REASON,
        BILATERAL_CONTACT_LIFT_SUCCESS_REASON,
        TCP_REACH_SUCCESS_REASON,
        BILATERAL_CONTACT_SUCCESS_REASON,
    }


def _finite_tuple(
    values: Sequence[float],
    *,
    size: int,
    name: str,
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {size} finite values.")
    return result


def _unit_interval_mapping(
    values: Mapping[str, float], *, name: str
) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key).strip()
        value = float(raw_value)
        if not key or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must map non-empty names to finite [0, 1] values."
            )
        result[key] = value
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return MappingProxyType(result)


@dataclass(frozen=True)
class CubeLiftRewardConfig:
    success_bonus: float = 5.0
    action_penalty: float = 0.01

    def __post_init__(self) -> None:
        success_bonus = float(self.success_bonus)
        action_penalty = float(self.action_penalty)
        if not math.isfinite(success_bonus) or success_bonus < 0.0:
            raise ValueError("success_bonus must be finite and non-negative.")
        if not math.isfinite(action_penalty) or action_penalty < 0.0:
            raise ValueError("action_penalty must be finite and non-negative.")
        object.__setattr__(self, "success_bonus", success_bonus)
        object.__setattr__(self, "action_penalty", action_penalty)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CubeLiftRewardConfig:
        return cls(
            success_bonus=float(payload.get("success_bonus", 5.0)),
            action_penalty=float(payload.get("action_penalty", 0.01)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "success_bonus": self.success_bonus,
            "action_penalty": self.action_penalty,
        }


@dataclass(frozen=True)
class CubeLiftTaskConfig:
    """All task-local parameters; does not alter the real/deploy contracts."""

    name: str
    base_contract_path: str
    base_contract_sha256: str
    execution_mode: str = DIRECT_QPOS_DEBUG_EXECUTION_MODE
    robot_dynamics: str = "not_simulated"
    gripper_dynamics: str = PRESCRIBED_GRIPPER_DYNAMICS
    position_actuation_profile_name: str | None = None
    position_actuation_profile_sha256: str | None = None
    gripper_actuation_profile_name: str | None = None
    gripper_actuation_profile_sha256: str | None = None
    policy_class: str = NORMAL_POLICY_CLASS
    grasp_mode: str = PHYSICAL_CONTACT_GRASP_MODE
    controlled_side: str = "right"
    fixed_side: str = "left"
    # The historical left/right field names below also identify fixed/controlled
    # slots in the CubeLift action contract. Physical side identity is carried
    # by controlled_side/fixed_side and the side-specific scene bindings.
    left_home_qpos: tuple[float, ...] = (0.0,) * 6
    right_home_qpos: tuple[float, ...] = (0.0,) * 6
    right_qpos_lower: tuple[float, ...] = (0.0,) * 6
    right_qpos_upper: tuple[float, ...] = (0.0,) * 6
    right_joint_delta_rad: tuple[float, ...] = (0.0,) * 6
    # Physical-policy actions are normalized TCP pose increments.  The
    # translation and rotation increments are expressed in the controlled
    # arm-base frame; the old joint delta remains only for v0 direct-qpos
    # debug compatibility.
    right_tcp_delta_m: tuple[float, ...] = (0.02, 0.02, 0.02)
    right_tcp_rotation_delta_rad: tuple[float, ...] = (0.15, 0.15, 0.15)
    pose_control_frame: str = "tool_tcp"
    # Explicit model-output components.  An empty tuple preserves historical
    # configs by inferring compact dimensions from their scripted-reset action
    # scales.  New task profiles should always declare this field so policy
    # dimensionality is independent of reset/curriculum mechanics.
    policy_action_components: tuple[str, ...] = ()
    # Optional affine latent-to-task scaling, ordered like
    # policy_action_components. Empty tuples preserve the identity mapping.
    policy_action_scale: tuple[float, ...] = ()
    policy_action_bias: tuple[float, ...] = ()
    left_tool_profile_path: str = DEFAULT_LEFT_TOOL_PROFILE_PATH
    right_tool_profile_path: str = DEFAULT_RIGHT_TOOL_PROFILE_PATH
    # Optional active-side wrist RGB policy input.  The feature is opt-in at
    # the task-config boundary so legacy state-only/debug configs remain
    # loadable; the RGB policy configs select a versioned CHW contract.
    wrist_rgb_enabled: bool = False
    wrist_rgb_camera_name: str = "right_wrist"
    wrist_rgb_width: int = 32
    wrist_rgb_height: int = 32
    wrist_rgb_spatial_pool_size: int = 1
    normal_proprio_profile: str = NORMAL_PROPRIO_PROFILE_V0
    # Optional explicit provenance for the 7-D object-pose suffix. Historical
    # configs leave this empty and retain their legacy flat-state contract.
    # New randomized tasks must name the simulator proxy/real-estimator seam.
    object_pose_observation_source: str = ""
    # Optional deployable goal feature derived from the object-pose estimator
    # and measured-RM65 forward kinematics.  It is expressed in the same arm
    # base frame as the public pose action, and is not simulator-only state.
    reach_target_delta_observation_source: str = ""
    # Ordered allowlist consumed by the actor.  The task/environment may expose
    # additional fields for diagnostics, reward, an optional privileged critic,
    # or sensor-ablation comparisons without silently making them policy inputs.
    # Empty preserves each historical profile's complete model-input binding.
    policy_observation_fields: tuple[str, ...] = ()
    right_gripper_delta: float = 0.2
    right_gripper_max_close: float = 0.8
    left_gripper_name: str = "left_gripper"
    right_gripper_name: str = "right_gripper"
    initial_gripper_close_by_name: Mapping[str, float] = field(default_factory=dict)
    object_body_name: str = "debug_cube"
    object_freejoint_name: str = "debug_cube_freejoint"
    right_grasp_body_name: str = "right_gripper_grasp_center_frame"
    grasp_latch_left_finger_body_name: str = "right_gripper_left_finger"
    grasp_latch_right_finger_body_name: str = "right_gripper_right_finger"
    grasp_latch_acquire_close: float = 0.45
    grasp_latch_release_close: float = 0.20
    grasp_latch_acquire_distance_m: float = 0.06
    # Generic simulator-only gripper surface contract.  Each entry describes
    # the inward-facing half-space of one finger; the opposite half-space is
    # treated as the outer surface by the grasp-contact extractor.
    gripper_contact_surface_specs: tuple[GripperContactSurfaceSpec, ...] = ()
    object_size_m: tuple[float, ...] = (0.03, 0.03, 0.03)
    cube_center_xy_m: tuple[float, ...] = (-0.472, 0.0)
    # Per-episode domain randomization. Values are symmetric half-ranges and
    # remain zero for historical deterministic configs.
    cube_spawn_xy_jitter_m: tuple[float, ...] = (0.0, 0.0)
    cube_spawn_yaw_jitter_rad: float = 0.0
    table_body_name: str = "table"
    table_top_z_m: float = 0.01
    spawn_clearance_m: float = 0.001
    settle_physics_ticks: int = 100
    policy_hz: float = 15.0
    reference_hz: float = 60.0
    command_hz: float = DEFAULT_POSITION_COMMAND_HZ
    physics_hz: float = 1000.0
    max_episode_steps: int = 150
    task_objective: str = LIFT_TASK_OBJECTIVE
    reach_target_height_offset_m: float = 0.03
    # Optional non-privileged geometric guidance for Reach/contact/pick approach.
    # Far from the object, the actor-visible target sits at the higher offset;
    # inside the XY radius it smoothly descends to
    # reach_target_height_offset_m. Both values stay zero for the historical
    # direct-target contract.
    reach_guidance_height_offset_m: float = 0.0
    reach_guidance_xy_blend_radius_m: float = 0.0
    # Optional observation-conditioned coarse-to-fine action scaling. The
    # public pose step stays full-size outside the radius and smoothly falls
    # to the minimum scale as the deployable reach-target delta approaches
    # zero. Defaults preserve the historical stateless transform.
    reach_action_min_scale: float = 1.0
    reach_action_slow_radius_m: float = 0.0
    # Optional stateless precision gate for contact acquisition. Once the
    # actor-visible target delta is inside the close radius (or measured close
    # proves closure has started), only xyz is multiplied by this scale;
    # gripper output remains independent. Defaults disable the gate.
    contact_close_translation_scale: float = 1.0
    # Optional absolute z multiplier during the same close phase. ``None``
    # preserves the isotropic xyz gate; an explicit smaller value lets a task
    # retain lateral centering while reducing table-directed vertical motion.
    contact_close_z_translation_scale: float | None = None
    contact_close_distance_m: float = 0.0
    contact_close_keep_measured_close: float = 1.0
    reach_success_distance_m: float = 0.02
    contact_success_min_measured_close: float = 0.45
    contact_success_min_score: float = 0.5
    lift_delta_m: float = 0.08
    success_hold_physics_ticks: int = 20
    success_requires_bilateral_contact: bool = False
    # Optional task-quality gates. Defaults preserve historical success
    # semantics. New physical tasks can require a short sequence of policy
    # actions that needed no collision-safety intervention, and can reject a
    # binary two-finger contact that does not land on both configured inner
    # surfaces (or also touches an outer surface).
    success_min_safety_clear_policy_steps: int = 0
    success_min_bilateral_inner_contact_score: float = 0.0
    success_max_outer_contact_score: float = 1.0
    pose_target_anti_windup: bool = False
    # Optional simulator-only reset preparation.  This moves the arm to a
    # collision-checked pre-grasp pose while keeping the gripper open; unlike
    # scripted_contact_reset it establishes no contact and exposes all seven
    # action dimensions to PPO after reset.
    scripted_pregrasp_reset: bool = False
    scripted_pregrasp_distance_m: float = 0.035
    scripted_pregrasp_action_scale: tuple[float, ...] = (1.0,) * 7
    scripted_pregrasp_action_bias: tuple[float, ...] = (0.0,) * 7
    # Optional dynamic phase after real bilateral contact.  Unlike the
    # scripted-contact curriculum, this phase is entered by the environment
    # only after the policy has acquired contact for several policy ticks.
    # The public action/learning contract remains the approach contract; the
    # simulator constrains only the executed action once lifting is safe.
    post_contact_phase_enabled: bool = False
    post_contact_phase_activation_steps: int = 3
    post_contact_phase_min_measured_close: float = 0.55
    post_contact_phase_loss_grace_steps: int = 3
    post_contact_action_scale: tuple[float, ...] = (1.0,) * 7
    post_contact_action_bias: tuple[float, ...] = (0.0,) * 7
    post_contact_nonnegative_lift_action: bool = False
    scripted_contact_reset: bool = False
    scripted_contact_reset_hold_steps: int = 5
    scripted_contact_reset_min_measured_close: float = 0.60
    scripted_contact_reset_max_attempts: int = 1
    scripted_contact_loss_grace_steps: int = 0
    scripted_contact_action_scale: tuple[float, ...] = (1.0,) * 7
    scripted_contact_action_bias: tuple[float, ...] = (0.0,) * 7
    scripted_contact_nonnegative_lift_action: bool = False
    out_of_bounds_xyz_lower_m: tuple[float, ...] = (-0.922, -0.65, -0.05)
    out_of_bounds_xyz_upper_m: tuple[float, ...] = (-0.022, 0.65, 0.6)
    reward: CubeLiftRewardConfig = field(default_factory=CubeLiftRewardConfig)
    # Simulator-only dense reward parameters are kept on the task boundary so
    # the selected reward profile is recorded in the task hash and checkpoint
    # metadata.  The policy still receives only normal observations.
    reward_shaping: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CUBE_LIFT_TASK_SCHEMA_VERSION
    source_path: str = ""
    source_sha256: str = ""
    source_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version not in {
            CUBE_LIFT_TASK_SCHEMA_VERSION,
            CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION,
            CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"Unsupported CubeLift task schema {self.schema_version!r}; "
                "expected direct-qpos v0, arm-actuated v1, or fully-actuated v2."
            )
        name = str(self.name).strip()
        if not name:
            raise ValueError("CubeLift task name must be non-empty.")
        source_dependencies = tuple(
            str(value).strip() for value in self.source_dependencies
        )
        if any(not value for value in source_dependencies):
            raise ValueError("source_dependencies cannot contain empty paths.")
        object.__setattr__(self, "source_dependencies", source_dependencies)
        if self.execution_mode == DIRECT_QPOS_DEBUG_EXECUTION_MODE:
            if self.schema_version != CUBE_LIFT_TASK_SCHEMA_VERSION:
                raise ValueError("Direct-qpos CubeLift must use the v0 task schema.")
            if self.robot_dynamics != "not_simulated":
                raise ValueError(
                    "Direct-qpos CubeLift must declare robot_dynamics=not_simulated."
                )
            if self.position_actuation_profile_name is not None:
                raise ValueError(
                    "Direct-qpos CubeLift cannot select an actuation profile."
                )
            if self.gripper_dynamics != PRESCRIBED_GRIPPER_DYNAMICS:
                raise ValueError("Direct-qpos CubeLift requires prescribed grippers.")
            if self.gripper_actuation_profile_name is not None:
                raise ValueError(
                    "Direct-qpos CubeLift cannot select a gripper profile."
                )
        elif self.execution_mode == ACTUATED_POSITION_EXECUTION_MODE:
            if self.schema_version not in {
                CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION,
                CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION,
            }:
                raise ValueError("Actuated CubeLift must use the v1 or v2 task schema.")
            if self.robot_dynamics != MUJOCO_POSITION_ACTUATOR_DYNAMICS:
                raise ValueError(
                    "Actuated CubeLift must declare robot_dynamics=mujoco_position_actuator."
                )
            if self.position_actuation_profile_name != RM65_ACTUATION_PROFILE_NAME:
                raise ValueError(
                    f"Actuated CubeLift requires profile {RM65_ACTUATION_PROFILE_NAME!r}."
                )
            expected_profile_hash = rm65_position_actuation_profile_sha256()
            if self.position_actuation_profile_sha256 != expected_profile_hash:
                raise ValueError(
                    "CubeLift position actuation profile hash mismatch: "
                    f"expected={self.position_actuation_profile_sha256}, "
                    f"actual={expected_profile_hash}."
                )
            if self.schema_version == CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION:
                if self.gripper_dynamics != PRESCRIBED_GRIPPER_DYNAMICS:
                    raise ValueError(
                        "Arm-actuated CubeLift v1 requires prescribed grippers."
                    )
                if self.gripper_actuation_profile_name is not None:
                    raise ValueError(
                        "Arm-actuated CubeLift v1 cannot select a gripper profile."
                    )
            else:
                if self.gripper_dynamics != MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS:
                    raise ValueError(
                        "Fully-actuated CubeLift v2 requires the identified MuJoCo gripper."
                    )
                expected_gripper_hash = dmtac_gripper_actuation_profile_sha256(
                    str(self.gripper_actuation_profile_name)
                )
                if self.gripper_actuation_profile_sha256 != expected_gripper_hash:
                    raise ValueError(
                        "CubeLift gripper actuation profile hash mismatch: "
                        f"expected={self.gripper_actuation_profile_sha256}, "
                        f"actual={expected_gripper_hash}."
                    )
        else:
            raise ValueError(
                f"Unsupported CubeLift execution_mode {self.execution_mode!r}."
            )
        if self.policy_class not in {
            NORMAL_POLICY_CLASS,
            PRIVILEGED_EXPERT_POLICY_CLASS,
            LEGACY_DEBUG_POLICY_CLASS,
        }:
            raise ValueError(
                f"Unsupported CubeLift policy_class {self.policy_class!r}."
            )
        if self.grasp_mode not in {
            PHYSICAL_CONTACT_GRASP_MODE,
            CONTACT_LATCH_DEBUG_GRASP_MODE,
        }:
            raise ValueError(f"Unsupported CubeLift grasp_mode {self.grasp_mode!r}.")
        if (
            self.policy_class
            in {
                NORMAL_POLICY_CLASS,
                PRIVILEGED_EXPERT_POLICY_CLASS,
            }
            and self.schema_version != CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION
        ):
            raise ValueError(
                "Normal and privileged-expert policies require the same "
                "fully-actuated task schema v2."
            )
        if (
            self.policy_class == NORMAL_POLICY_CLASS
            and self.grasp_mode != PHYSICAL_CONTACT_GRASP_MODE
        ):
            raise ValueError("Normal CubeLift policy cannot use the debug grasp latch.")
        if self.uses_pose_action and str(self.pose_control_frame).strip() != "tool_tcp":
            raise ValueError(
                "Physical CubeLift pose actions must control the tool_tcp frame."
            )
        policy_action_components = tuple(
            str(component).strip() for component in self.policy_action_components
        )
        if any(not component for component in policy_action_components):
            raise ValueError("policy_action_components cannot contain empty names.")
        if len(policy_action_components) != len(set(policy_action_components)):
            raise ValueError("policy_action_components must be unique.")
        if policy_action_components:
            unknown_policy_components = tuple(
                component
                for component in policy_action_components
                if component not in self.action_components
            )
            if unknown_policy_components:
                raise ValueError(
                    "policy_action_components are absent from the public task "
                    f"action: {unknown_policy_components}."
                )
        object.__setattr__(
            self,
            "policy_action_components",
            policy_action_components,
        )
        policy_action_scale = tuple(float(value) for value in self.policy_action_scale)
        policy_action_bias = tuple(float(value) for value in self.policy_action_bias)
        if (policy_action_scale or policy_action_bias) and not policy_action_components:
            raise ValueError(
                "policy_action_scale/bias require explicit policy_action_components."
            )
        policy_dim = len(policy_action_components)
        if policy_action_scale and len(policy_action_scale) != policy_dim:
            raise ValueError("policy_action_scale must match policy_action_components.")
        if policy_action_bias and len(policy_action_bias) != policy_dim:
            raise ValueError("policy_action_bias must match policy_action_components.")
        resolved_scale = policy_action_scale or (1.0,) * policy_dim
        resolved_bias = policy_action_bias or (0.0,) * policy_dim
        if any(
            not math.isfinite(scale) or not 0.0 < scale <= 1.0
            for scale in resolved_scale
        ):
            raise ValueError("policy_action_scale values must lie in (0, 1].")
        if any(
            not math.isfinite(bias) or abs(bias) + scale > 1.0
            for scale, bias in zip(resolved_scale, resolved_bias)
        ):
            raise ValueError(
                "policy_action_bias plus scale must keep mapped actions in [-1, 1]."
            )
        object.__setattr__(self, "policy_action_scale", policy_action_scale)
        object.__setattr__(self, "policy_action_bias", policy_action_bias)
        controlled_side = str(self.controlled_side).strip().lower()
        fixed_side = str(self.fixed_side).strip().lower()
        if {controlled_side, fixed_side} != {"left", "right"}:
            raise ValueError(
                "CubeLift controlled_side/fixed_side must assign left and right "
                "exactly once."
            )
        object.__setattr__(self, "controlled_side", controlled_side)
        object.__setattr__(self, "fixed_side", fixed_side)
        task_objective = str(self.task_objective).strip().lower()
        if task_objective not in {
            REACH_TASK_OBJECTIVE,
            CONTACT_TASK_OBJECTIVE,
            LIFT_TASK_OBJECTIVE,
        }:
            raise ValueError("task_objective must be 'reach', 'contact', or 'lift'.")
        if (
            self.execution_mode == DIRECT_QPOS_DEBUG_EXECUTION_MODE
            and task_objective != LIFT_TASK_OBJECTIVE
        ):
            raise ValueError(
                "Reach/contact objectives require the authoritative actuated "
                "CubeLiftKernel; legacy direct-qpos debug supports lift only."
            )
        reach_target_height = float(self.reach_target_height_offset_m)
        reach_guidance_height = float(self.reach_guidance_height_offset_m)
        reach_guidance_radius = float(self.reach_guidance_xy_blend_radius_m)
        reach_action_min_scale = float(self.reach_action_min_scale)
        reach_action_slow_radius = float(self.reach_action_slow_radius_m)
        contact_close_translation_scale = float(self.contact_close_translation_scale)
        contact_close_z_translation_scale = (
            None
            if self.contact_close_z_translation_scale is None
            else float(self.contact_close_z_translation_scale)
        )
        contact_close_distance = float(self.contact_close_distance_m)
        contact_close_keep_measured = float(self.contact_close_keep_measured_close)
        reach_success_distance = float(self.reach_success_distance_m)
        contact_min_close = float(self.contact_success_min_measured_close)
        contact_min_score = float(self.contact_success_min_score)
        success_min_inner_score = float(self.success_min_bilateral_inner_contact_score)
        success_max_outer_score = float(self.success_max_outer_contact_score)
        if not math.isfinite(reach_target_height) or reach_target_height <= 0.0:
            raise ValueError(
                "reach_target_height_offset_m must be positive and finite."
            )
        if not math.isfinite(reach_success_distance) or reach_success_distance <= 0.0:
            raise ValueError("reach_success_distance_m must be positive and finite.")
        if (
            not math.isfinite(reach_guidance_height)
            or not math.isfinite(reach_guidance_radius)
            or reach_guidance_height < 0.0
            or reach_guidance_radius < 0.0
        ):
            raise ValueError(
                "Reach guidance height and XY blend radius must be finite and "
                "non-negative."
            )
        reach_guidance_enabled = bool(
            reach_guidance_height > 0.0 or reach_guidance_radius > 0.0
        )
        if reach_guidance_enabled and (
            reach_guidance_height <= reach_target_height or reach_guidance_radius <= 0.0
        ):
            raise ValueError(
                "Overhead reach guidance requires a height above "
                "reach_target_height_offset_m and a positive XY blend radius."
            )
        if (
            not math.isfinite(reach_action_min_scale)
            or not 0.0 < reach_action_min_scale <= 1.0
        ):
            raise ValueError("reach_action_min_scale must lie in (0, 1].")
        if (
            not math.isfinite(reach_action_slow_radius)
            or reach_action_slow_radius < 0.0
        ):
            raise ValueError(
                "reach_action_slow_radius_m must be finite and non-negative."
            )
        reach_action_scaling_enabled = bool(
            reach_action_min_scale < 1.0 or reach_action_slow_radius > 0.0
        )
        if reach_action_scaling_enabled and (
            reach_action_min_scale >= 1.0 or reach_action_slow_radius <= 0.0
        ):
            raise ValueError(
                "Distance-dependent reach action scaling requires a minimum "
                "scale below 1 and a positive slow radius."
            )
        if (
            not math.isfinite(contact_close_translation_scale)
            or not 0.0 < contact_close_translation_scale <= 1.0
        ):
            raise ValueError("contact_close_translation_scale must lie in (0, 1].")
        if not math.isfinite(contact_close_distance) or contact_close_distance < 0.0:
            raise ValueError(
                "contact_close_distance_m must be finite and non-negative."
            )
        if (
            not math.isfinite(contact_close_keep_measured)
            or not 0.0 <= contact_close_keep_measured <= 1.0
        ):
            raise ValueError("contact_close_keep_measured_close must lie in [0, 1].")
        contact_close_scaling_enabled = bool(
            contact_close_translation_scale < 1.0
            or contact_close_distance > 0.0
            or contact_close_keep_measured < 1.0
        )
        if contact_close_scaling_enabled and (
            contact_close_translation_scale >= 1.0
            or contact_close_distance <= 0.0
            or not 0.0 < contact_close_keep_measured < 1.0
        ):
            raise ValueError(
                "Contact-close translation scaling requires a scale below 1, "
                "a positive close distance, and a measured-close latch in "
                "(0, 1)."
            )
        if contact_close_z_translation_scale is not None and (
            not contact_close_scaling_enabled
            or not math.isfinite(contact_close_z_translation_scale)
            or not 0.0
            < contact_close_z_translation_scale
            <= contact_close_translation_scale
        ):
            raise ValueError(
                "contact_close_z_translation_scale requires an enabled "
                "Contact-close gate and must lie in (0, "
                "contact_close_translation_scale]."
            )
        if not 0.0 <= contact_min_close <= self.right_gripper_max_close:
            raise ValueError(
                "contact_success_min_measured_close must lie within the "
                "configured gripper close range."
            )
        if not 0.0 <= contact_min_score <= 1.0:
            raise ValueError("contact_success_min_score must lie in [0, 1].")
        if not 0.0 <= success_min_inner_score <= 1.0:
            raise ValueError(
                "success_min_bilateral_inner_contact_score must lie in [0, 1]."
            )
        if not 0.0 <= success_max_outer_score <= 1.0:
            raise ValueError("success_max_outer_contact_score must lie in [0, 1].")
        object.__setattr__(self, "task_objective", task_objective)
        object.__setattr__(
            self,
            "reach_target_height_offset_m",
            reach_target_height,
        )
        object.__setattr__(
            self,
            "reach_guidance_height_offset_m",
            reach_guidance_height,
        )
        object.__setattr__(
            self,
            "reach_guidance_xy_blend_radius_m",
            reach_guidance_radius,
        )
        object.__setattr__(
            self,
            "reach_action_min_scale",
            reach_action_min_scale,
        )
        object.__setattr__(
            self,
            "reach_action_slow_radius_m",
            reach_action_slow_radius,
        )
        object.__setattr__(
            self,
            "contact_close_translation_scale",
            contact_close_translation_scale,
        )
        object.__setattr__(
            self,
            "contact_close_z_translation_scale",
            contact_close_z_translation_scale,
        )
        object.__setattr__(
            self,
            "contact_close_distance_m",
            contact_close_distance,
        )
        object.__setattr__(
            self,
            "contact_close_keep_measured_close",
            contact_close_keep_measured,
        )
        object.__setattr__(
            self,
            "reach_success_distance_m",
            reach_success_distance,
        )
        object.__setattr__(
            self,
            "contact_success_min_measured_close",
            contact_min_close,
        )
        object.__setattr__(
            self,
            "contact_success_min_score",
            contact_min_score,
        )
        object.__setattr__(
            self,
            "success_min_bilateral_inner_contact_score",
            success_min_inner_score,
        )
        object.__setattr__(
            self,
            "success_max_outer_contact_score",
            success_max_outer_score,
        )
        wrist_rgb_enabled = bool(self.wrist_rgb_enabled)
        wrist_rgb_camera_name = str(self.wrist_rgb_camera_name).strip()
        wrist_rgb_width = int(self.wrist_rgb_width)
        wrist_rgb_height = int(self.wrist_rgb_height)
        wrist_rgb_spatial_pool_size = int(self.wrist_rgb_spatial_pool_size)
        normal_proprio_profile = str(self.normal_proprio_profile).strip()
        object_pose_observation_source = str(
            self.object_pose_observation_source
        ).strip()
        reach_target_delta_observation_source = str(
            self.reach_target_delta_observation_source
        ).strip()
        policy_observation_fields = tuple(
            str(value).strip() for value in self.policy_observation_fields
        )
        if not wrist_rgb_camera_name:
            raise ValueError("wrist_rgb_camera_name must be non-empty.")
        if wrist_rgb_width <= 0 or wrist_rgb_height <= 0:
            raise ValueError("wrist_rgb_width and wrist_rgb_height must be positive.")
        if wrist_rgb_spatial_pool_size not in {1, 3}:
            raise ValueError("wrist_rgb_spatial_pool_size must be 1 or 3.")
        if wrist_rgb_enabled and (wrist_rgb_width, wrist_rgb_height) not in {
            (32, 32),
            (48, 48),
        }:
            raise ValueError(
                "The versioned wrist RGB observation contract supports only 32x32 or 48x48 pixels."
            )
        if normal_proprio_profile not in {
            NORMAL_PROPRIO_PROFILE_V0,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
        }:
            raise ValueError(
                f"Unsupported normal_proprio_profile: {normal_proprio_profile!r}."
            )
        if (
            normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
            }
            and self.policy_class != NORMAL_POLICY_CLASS
        ):
            raise ValueError(
                "The motion/tracking proprio profile is only valid for normal policies."
            )
        if (
            normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
            }
            and wrist_rgb_enabled
        ):
            raise ValueError(
                "The cube-pose proprio contract currently excludes wrist RGB; "
                "use a dedicated RGB observation layout before enabling both."
            )
        if object_pose_observation_source and normal_proprio_profile not in {
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
            NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
        }:
            raise ValueError(
                "object_pose_observation_source requires the cube-pose "
                "normal proprio profile."
            )
        uses_reach_target_delta = bool(
            normal_proprio_profile
            == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2
        )
        if uses_reach_target_delta != bool(reach_target_delta_observation_source):
            raise ValueError(
                "The cube-pose reach-delta profile and "
                "reach_target_delta_observation_source must be selected "
                "together."
            )
        approach_objectives = {
            REACH_TASK_OBJECTIVE,
            CONTACT_TASK_OBJECTIVE,
            LIFT_TASK_OBJECTIVE,
        }
        if reach_guidance_enabled and (
            task_objective not in approach_objectives or not uses_reach_target_delta
        ):
            raise ValueError(
                "Overhead approach guidance requires a Reach/contact/lift objective "
                "and the deployable reach-target-delta observation profile."
            )
        if reach_action_scaling_enabled and (
            task_objective not in approach_objectives or not uses_reach_target_delta
        ):
            raise ValueError(
                "Distance-dependent approach action scaling requires a "
                "Reach/contact/lift objective and deployable reach-target-delta "
                "observation."
            )
        scalable_action_components = {
            (
                "right_tcp_delta_x",
                "right_tcp_delta_y",
                "right_tcp_delta_z",
            ),
            (
                "right_tcp_delta_x",
                "right_tcp_delta_y",
                "right_tcp_delta_z",
                "right_gripper_delta",
            ),
        }
        if (
            reach_action_scaling_enabled
            and policy_action_components not in scalable_action_components
        ):
            raise ValueError(
                "Distance-dependent approach action scaling requires explicit "
                "xyz or xyz-plus-gripper policy action components."
            )
        if (
            reach_action_scaling_enabled
            and policy_observation_fields
            and "reach_target_delta" not in policy_observation_fields
        ):
            raise ValueError(
                "Distance-dependent reach action scaling requires the "
                "actor-visible reach_target_delta field."
            )
        if reach_action_scaling_enabled and (
            resolved_scale != (1.0,) * policy_dim
            or resolved_bias != (0.0,) * policy_dim
        ):
            raise ValueError(
                "Distance-dependent reach action scaling cannot be combined "
                "with a static policy_action_scale or policy_action_bias."
            )
        contact_close_objective_supported = bool(
            task_objective == CONTACT_TASK_OBJECTIVE
            or (
                task_objective == LIFT_TASK_OBJECTIVE
                and bool(self.post_contact_phase_enabled)
            )
        )
        if contact_close_scaling_enabled and (
            not contact_close_objective_supported
            or not uses_reach_target_delta
            or not reach_action_scaling_enabled
            or policy_action_components
            != (
                "right_tcp_delta_x",
                "right_tcp_delta_y",
                "right_tcp_delta_z",
                "right_gripper_delta",
            )
            or (
                policy_observation_fields
                and "reach_target_delta" not in policy_observation_fields
            )
        ):
            raise ValueError(
                "Contact-close translation scaling requires a Contact task or "
                "a Lift task with an identity post-contact reference phase, "
                "goal-delta xyz-plus-gripper profile with distance-dependent "
                "approach scaling and actor-visible reach_target_delta."
            )
        if (
            contact_close_scaling_enabled
            and task_objective == LIFT_TASK_OBJECTIVE
            and float(self.post_contact_phase_min_measured_close)
            <= contact_close_keep_measured
        ):
            raise ValueError(
                "Lift contact-close z release requires "
                "post_contact_phase_min_measured_close above "
                "contact_close_keep_measured_close."
            )
        if (
            normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
            }
            and wrist_rgb_enabled
            and (wrist_rgb_width, wrist_rgb_height) != (48, 48)
        ):
            raise ValueError(
                "The motion/tracking RGB contract is currently versioned only at 48x48."
            )
        if any(not value for value in policy_observation_fields):
            raise ValueError("policy_observation_fields cannot contain empty names.")
        if len(policy_observation_fields) != len(set(policy_observation_fields)):
            raise ValueError("policy_observation_fields must be unique.")
        state_field = (
            "proprio"
            if self.policy_class == NORMAL_POLICY_CLASS
            else "privileged_state"
        )
        uses_episode_clock = not (
            self.policy_class == NORMAL_POLICY_CLASS
            and normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
            }
        )
        available_observation_fields = [state_field]
        if uses_episode_clock:
            available_observation_fields.append("episode_progress")
        if (
            normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
            }
            and object_pose_observation_source
        ):
            available_observation_fields.append("object_pose")
        if uses_reach_target_delta:
            available_observation_fields.append("reach_target_delta")
        if wrist_rgb_enabled:
            available_observation_fields.append("wrist_rgb")
        available_observation_fields_tuple = tuple(available_observation_fields)
        if policy_observation_fields:
            unknown_fields = tuple(
                field
                for field in policy_observation_fields
                if field not in available_observation_fields_tuple
            )
            if unknown_fields:
                raise ValueError(
                    "policy_observation_fields are absent from the task "
                    f"observation contract: {unknown_fields}."
                )
            if state_field not in policy_observation_fields:
                raise ValueError(
                    "policy_observation_fields must retain the measured state "
                    f"field {state_field!r}."
                )
            canonical_subset = tuple(
                field
                for field in available_observation_fields_tuple
                if field in policy_observation_fields
            )
            if policy_observation_fields != canonical_subset:
                raise ValueError(
                    "policy_observation_fields must follow task observation order: "
                    f"{available_observation_fields_tuple}."
                )
        base_contract_path = str(self.base_contract_path).strip()
        base_contract_hash = str(self.base_contract_sha256).strip().lower()
        if not base_contract_path:
            raise ValueError("base_contract_path must be non-empty.")
        if len(base_contract_hash) != 64 or any(
            char not in "0123456789abcdef" for char in base_contract_hash
        ):
            raise ValueError(
                "base_contract_sha256 must be a lowercase SHA-256 hex digest."
            )

        left_home = _finite_tuple(self.left_home_qpos, size=6, name="left_home_qpos")
        right_home = _finite_tuple(self.right_home_qpos, size=6, name="right_home_qpos")
        lower = _finite_tuple(self.right_qpos_lower, size=6, name="right_qpos_lower")
        upper = _finite_tuple(self.right_qpos_upper, size=6, name="right_qpos_upper")
        joint_delta = _finite_tuple(
            self.right_joint_delta_rad,
            size=6,
            name="right_joint_delta_rad",
        )
        tcp_delta = _finite_tuple(
            self.right_tcp_delta_m,
            size=3,
            name="right_tcp_delta_m",
        )
        tcp_rotation_delta = _finite_tuple(
            self.right_tcp_rotation_delta_rad,
            size=3,
            name="right_tcp_rotation_delta_rad",
        )
        if any(low >= high for low, high in zip(lower, upper)):
            raise ValueError(
                "right_qpos_lower must be strictly below right_qpos_upper."
            )
        if any(delta <= 0.0 for delta in joint_delta):
            raise ValueError("right_joint_delta_rad must be strictly positive.")
        if any(delta <= 0.0 for delta in tcp_delta):
            raise ValueError("right_tcp_delta_m must be strictly positive.")
        if any(delta <= 0.0 for delta in tcp_rotation_delta):
            raise ValueError("right_tcp_rotation_delta_rad must be strictly positive.")
        if any(
            value < low or value > high
            for value, low, high in zip(right_home, lower, upper)
        ):
            raise ValueError("right_home_qpos must lie within right qpos bounds.")

        object_size = _finite_tuple(self.object_size_m, size=3, name="object_size_m")
        if any(value <= 0.0 for value in object_size):
            raise ValueError("object_size_m must be positive.")
        cube_center = _finite_tuple(
            self.cube_center_xy_m, size=2, name="cube_center_xy_m"
        )
        lower_xyz = _finite_tuple(
            self.out_of_bounds_xyz_lower_m,
            size=3,
            name="out_of_bounds_xyz_lower_m",
        )
        upper_xyz = _finite_tuple(
            self.out_of_bounds_xyz_upper_m,
            size=3,
            name="out_of_bounds_xyz_upper_m",
        )
        if any(low >= high for low, high in zip(lower_xyz, upper_xyz)):
            raise ValueError("out_of_bounds lower values must be below upper values.")
        scalar_names = (
            "right_gripper_delta",
            "table_top_z_m",
            "spawn_clearance_m",
            "policy_hz",
            "reference_hz",
            "command_hz",
            "physics_hz",
            "lift_delta_m",
            "grasp_latch_acquire_distance_m",
        )
        scalar_values = {name: float(getattr(self, name)) for name in scalar_names}
        if not all(
            math.isfinite(value) and value > 0.0 for value in scalar_values.values()
        ):
            raise ValueError(
                f"CubeLift positive finite values required: {scalar_names}."
            )
        if scalar_values["physics_hz"] < scalar_values["command_hz"]:
            raise ValueError("physics_hz must be at least command_hz.")
        if scalar_values["command_hz"] < scalar_values["reference_hz"]:
            raise ValueError("command_hz must be at least reference_hz.")
        if scalar_values["reference_hz"] < scalar_values["policy_hz"]:
            raise ValueError("reference_hz must be at least policy_hz.")
        if self.settle_physics_ticks < 0 or self.max_episode_steps <= 0:
            raise ValueError(
                "settle_physics_ticks must be >= 0 and max_episode_steps > 0."
            )
        if self.success_hold_physics_ticks <= 0:
            raise ValueError("success_hold_physics_ticks must be positive.")
        if self.success_min_safety_clear_policy_steps < 0:
            raise ValueError(
                "success_min_safety_clear_policy_steps must be non-negative."
            )
        unit_values = {
            "right_gripper_max_close": float(self.right_gripper_max_close),
            "grasp_latch_acquire_close": float(self.grasp_latch_acquire_close),
            "grasp_latch_release_close": float(self.grasp_latch_release_close),
        }
        if any(not 0.0 <= value <= 1.0 for value in unit_values.values()):
            raise ValueError(
                f"CubeLift latch close values must be in [0, 1]: {unit_values}."
            )
        if (
            unit_values["grasp_latch_release_close"]
            >= unit_values["grasp_latch_acquire_close"]
        ):
            raise ValueError("grasp latch release close must be below acquire close.")
        if (
            unit_values["grasp_latch_acquire_close"]
            > unit_values["right_gripper_max_close"]
        ):
            raise ValueError(
                "grasp latch acquire close cannot exceed gripper max close."
            )

        grippers = _unit_interval_mapping(
            self.initial_gripper_close_by_name,
            name="initial_gripper_close_by_name",
        )
        required_grippers = {str(self.left_gripper_name), str(self.right_gripper_name)}
        if set(grippers) != required_grippers:
            raise ValueError(
                "initial_gripper_close_by_name must define exactly the configured left/right grippers."
            )
        for field_name in (
            "left_gripper_name",
            "right_gripper_name",
            "object_body_name",
            "object_freejoint_name",
            "right_grasp_body_name",
            "grasp_latch_left_finger_body_name",
            "grasp_latch_right_finger_body_name",
            "table_body_name",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty.")

        if not isinstance(self.reward, CubeLiftRewardConfig):
            raise TypeError("reward must be CubeLiftRewardConfig.")
        if not isinstance(self.reward_shaping, Mapping):
            raise TypeError("reward_shaping must be a JSON object mapping.")
        reward_shaping = dict(self.reward_shaping)
        if any(not isinstance(key, str) or not key.strip() for key in reward_shaping):
            raise ValueError("reward_shaping keys must be non-empty strings.")
        if task_objective == REACH_TASK_OBJECTIVE and (
            "grasp_height_offset_m" in reward_shaping
            and not math.isclose(
                float(reward_shaping["grasp_height_offset_m"]),
                reach_target_height,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Reach success and reward must use the same target height: "
                "reach_target_height_offset_m != grasp_height_offset_m."
            )
        surface_specs = tuple(
            spec
            if isinstance(spec, GripperContactSurfaceSpec)
            else GripperContactSurfaceSpec.from_dict(spec)
            for spec in self.gripper_contact_surface_specs
        )
        surface_bodies = [spec.body_name for spec in surface_specs]
        if len(surface_bodies) != len(set(surface_bodies)):
            raise ValueError(
                "gripper_contact_surface_specs must contain one entry per finger body."
            )
        surface_sides = [spec.side for spec in surface_specs]
        if len(surface_sides) != len(set(surface_sides)):
            raise ValueError(
                "gripper_contact_surface_specs must contain unique finger sides."
            )
        if not surface_specs and (
            success_min_inner_score > 0.0 or success_max_outer_score < 1.0
        ):
            raise ValueError(
                "Contact-quality success gates require gripper_contact_surface_specs."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "base_contract_path", base_contract_path)
        object.__setattr__(self, "base_contract_sha256", base_contract_hash)
        object.__setattr__(self, "left_home_qpos", left_home)
        object.__setattr__(self, "right_home_qpos", right_home)
        object.__setattr__(self, "right_qpos_lower", lower)
        object.__setattr__(self, "right_qpos_upper", upper)
        object.__setattr__(self, "right_joint_delta_rad", joint_delta)
        object.__setattr__(self, "right_tcp_delta_m", tcp_delta)
        object.__setattr__(self, "right_tcp_rotation_delta_rad", tcp_rotation_delta)
        object.__setattr__(
            self, "pose_control_frame", str(self.pose_control_frame).strip()
        )
        object.__setattr__(
            self,
            "left_tool_profile_path",
            str(self.left_tool_profile_path).strip(),
        )
        object.__setattr__(
            self,
            "right_tool_profile_path",
            str(self.right_tool_profile_path).strip(),
        )
        object.__setattr__(self, "wrist_rgb_enabled", wrist_rgb_enabled)
        object.__setattr__(self, "wrist_rgb_camera_name", wrist_rgb_camera_name)
        object.__setattr__(self, "wrist_rgb_width", wrist_rgb_width)
        object.__setattr__(self, "wrist_rgb_height", wrist_rgb_height)
        object.__setattr__(
            self, "wrist_rgb_spatial_pool_size", wrist_rgb_spatial_pool_size
        )
        object.__setattr__(self, "normal_proprio_profile", normal_proprio_profile)
        object.__setattr__(self, "object_size_m", object_size)
        object.__setattr__(self, "cube_center_xy_m", cube_center)
        object.__setattr__(self, "out_of_bounds_xyz_lower_m", lower_xyz)
        object.__setattr__(self, "out_of_bounds_xyz_upper_m", upper_xyz)
        object.__setattr__(self, "initial_gripper_close_by_name", grippers)
        object.__setattr__(self, "gripper_contact_surface_specs", surface_specs)
        object.__setattr__(self, "reward_shaping", reward_shaping)
        object.__setattr__(
            self,
            "object_pose_observation_source",
            object_pose_observation_source,
        )
        object.__setattr__(
            self,
            "reach_target_delta_observation_source",
            reach_target_delta_observation_source,
        )
        object.__setattr__(
            self,
            "policy_observation_fields",
            policy_observation_fields,
        )
        for field_name, value in scalar_values.items():
            object.__setattr__(self, field_name, value)
        for field_name, value in unit_values.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "settle_physics_ticks", int(self.settle_physics_ticks))
        object.__setattr__(self, "max_episode_steps", int(self.max_episode_steps))
        object.__setattr__(
            self, "success_hold_physics_ticks", int(self.success_hold_physics_ticks)
        )
        object.__setattr__(
            self,
            "success_min_safety_clear_policy_steps",
            int(self.success_min_safety_clear_policy_steps),
        )
        spawn_xy_jitter = _finite_tuple(
            self.cube_spawn_xy_jitter_m,
            size=2,
            name="cube_spawn_xy_jitter_m",
        )
        if any(value < 0.0 for value in spawn_xy_jitter):
            raise ValueError("cube_spawn_xy_jitter_m values must be non-negative.")
        spawn_yaw_jitter = float(self.cube_spawn_yaw_jitter_rad)
        if (
            not math.isfinite(spawn_yaw_jitter)
            or spawn_yaw_jitter < 0.0
            or spawn_yaw_jitter > math.pi
        ):
            raise ValueError("cube_spawn_yaw_jitter_rad must lie in [0, pi].")
        object.__setattr__(self, "cube_spawn_xy_jitter_m", spawn_xy_jitter)
        object.__setattr__(
            self,
            "cube_spawn_yaw_jitter_rad",
            spawn_yaw_jitter,
        )
        randomizes_cube_spawn = bool(
            any(value > 0.0 for value in spawn_xy_jitter) or spawn_yaw_jitter > 0.0
        )
        if randomizes_cube_spawn and self.policy_class == NORMAL_POLICY_CLASS:
            has_object_derived_actor_input = bool(
                normal_proprio_profile
                in {
                    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
                }
                and object_pose_observation_source
                and (
                    not policy_observation_fields
                    or "object_pose" in policy_observation_fields
                    or "reach_target_delta" in policy_observation_fields
                )
            )
            has_actor_rgb = bool(
                wrist_rgb_enabled
                and (
                    not policy_observation_fields
                    or "wrist_rgb" in policy_observation_fields
                )
            )
            if not has_actor_rgb and not has_object_derived_actor_input:
                raise ValueError(
                    "A normal policy cannot randomize an unobserved cube "
                    "spawn; enable wrist RGB, object pose, or an explicitly "
                    "sourced object-derived target observation."
                )
        object.__setattr__(
            self,
            "success_requires_bilateral_contact",
            bool(self.success_requires_bilateral_contact),
        )
        object.__setattr__(
            self,
            "pose_target_anti_windup",
            bool(self.pose_target_anti_windup),
        )
        scripted_pregrasp_reset = bool(self.scripted_pregrasp_reset)
        scripted_pregrasp_distance = float(self.scripted_pregrasp_distance_m)
        if (
            not math.isfinite(scripted_pregrasp_distance)
            or scripted_pregrasp_distance <= 0.0
        ):
            raise ValueError(
                "scripted_pregrasp_distance_m must be positive and finite."
            )
        if scripted_pregrasp_reset and self.scripted_contact_reset:
            raise ValueError(
                "scripted_pregrasp_reset and scripted_contact_reset are mutually exclusive."
            )
        object.__setattr__(self, "scripted_pregrasp_reset", scripted_pregrasp_reset)
        object.__setattr__(
            self, "scripted_pregrasp_distance_m", scripted_pregrasp_distance
        )
        scripted_pregrasp_action_scale = _finite_tuple(
            self.scripted_pregrasp_action_scale,
            size=7,
            name="scripted_pregrasp_action_scale",
        )
        if any(value < 0.0 or value > 1.0 for value in scripted_pregrasp_action_scale):
            raise ValueError(
                "scripted_pregrasp_action_scale values must lie in [0, 1]."
            )
        scripted_pregrasp_action_bias = _finite_tuple(
            self.scripted_pregrasp_action_bias,
            size=7,
            name="scripted_pregrasp_action_bias",
        )
        if any(
            abs(bias) + scale > 1.0
            for bias, scale in zip(
                scripted_pregrasp_action_bias,
                scripted_pregrasp_action_scale,
            )
        ):
            raise ValueError(
                "scripted_pregrasp_action_bias plus action scale must keep "
                "every constrained action within [-1, 1]."
            )
        object.__setattr__(
            self,
            "scripted_pregrasp_action_scale",
            scripted_pregrasp_action_scale,
        )
        object.__setattr__(
            self,
            "scripted_pregrasp_action_bias",
            scripted_pregrasp_action_bias,
        )
        post_contact_phase_enabled = bool(self.post_contact_phase_enabled)
        post_contact_activation_steps = int(self.post_contact_phase_activation_steps)
        if post_contact_activation_steps <= 0:
            raise ValueError("post_contact_phase_activation_steps must be positive.")
        post_contact_min_close = float(self.post_contact_phase_min_measured_close)
        if not 0.0 <= post_contact_min_close <= self.right_gripper_max_close:
            raise ValueError(
                "post_contact_phase_min_measured_close must lie within the "
                "configured gripper close range."
            )
        post_contact_loss_grace_steps = int(self.post_contact_phase_loss_grace_steps)
        if post_contact_loss_grace_steps < 0:
            raise ValueError(
                "post_contact_phase_loss_grace_steps must be non-negative."
            )
        post_contact_action_scale = _finite_tuple(
            self.post_contact_action_scale,
            size=7,
            name="post_contact_action_scale",
        )
        if any(value < 0.0 or value > 1.0 for value in post_contact_action_scale):
            raise ValueError("post_contact_action_scale values must lie in [0, 1].")
        post_contact_action_bias = _finite_tuple(
            self.post_contact_action_bias,
            size=7,
            name="post_contact_action_bias",
        )
        if any(
            abs(bias) + scale > 1.0
            for bias, scale in zip(post_contact_action_bias, post_contact_action_scale)
        ):
            raise ValueError(
                "post_contact_action_bias plus action scale must keep every "
                "constrained action within [-1, 1]."
            )
        if post_contact_phase_enabled and not any(
            value > 0.0 for value in post_contact_action_scale
        ):
            raise ValueError(
                "An enabled post-contact phase must expose at least one action."
            )
        if post_contact_phase_enabled and bool(self.scripted_contact_reset):
            raise ValueError(
                "post_contact_phase_enabled cannot be combined with "
                "scripted_contact_reset; use one lift-phase owner."
            )
        post_contact_changes_action = bool(
            post_contact_action_scale != (1.0,) * 7
            or post_contact_action_bias != (0.0,) * 7
            or self.post_contact_nonnegative_lift_action
        )
        if (
            post_contact_phase_enabled
            and self.policy_action_components
            and post_contact_changes_action
        ):
            raise ValueError(
                "Explicit latent-action profiles cannot use the hidden "
                "post-contact action phase. Make phase state an actor "
                "observation first, or keep one stationary action mapping."
            )
        if not post_contact_phase_enabled and (
            post_contact_action_scale != (1.0,) * 7
            or post_contact_action_bias != (0.0,) * 7
            or bool(self.post_contact_nonnegative_lift_action)
        ):
            raise ValueError(
                "Post-contact action constraints require "
                "post_contact_phase_enabled=true."
            )
        object.__setattr__(
            self, "post_contact_phase_enabled", post_contact_phase_enabled
        )
        object.__setattr__(
            self,
            "post_contact_phase_activation_steps",
            post_contact_activation_steps,
        )
        object.__setattr__(
            self,
            "post_contact_phase_min_measured_close",
            post_contact_min_close,
        )
        object.__setattr__(
            self,
            "post_contact_phase_loss_grace_steps",
            post_contact_loss_grace_steps,
        )
        object.__setattr__(self, "post_contact_action_scale", post_contact_action_scale)
        object.__setattr__(self, "post_contact_action_bias", post_contact_action_bias)
        object.__setattr__(
            self,
            "post_contact_nonnegative_lift_action",
            bool(self.post_contact_nonnegative_lift_action),
        )
        object.__setattr__(
            self,
            "scripted_contact_reset",
            bool(self.scripted_contact_reset),
        )
        object.__setattr__(
            self,
            "scripted_contact_reset_hold_steps",
            int(self.scripted_contact_reset_hold_steps),
        )
        if self.scripted_contact_reset_hold_steps <= 0:
            raise ValueError("scripted_contact_reset_hold_steps must be positive.")
        object.__setattr__(
            self,
            "scripted_contact_reset_max_attempts",
            int(self.scripted_contact_reset_max_attempts),
        )
        if self.scripted_contact_reset_max_attempts <= 0:
            raise ValueError("scripted_contact_reset_max_attempts must be positive.")
        object.__setattr__(
            self,
            "scripted_contact_loss_grace_steps",
            int(self.scripted_contact_loss_grace_steps),
        )
        if self.scripted_contact_loss_grace_steps < 0:
            raise ValueError("scripted_contact_loss_grace_steps must be non-negative.")
        scripted_action_scale = _finite_tuple(
            self.scripted_contact_action_scale,
            size=7,
            name="scripted_contact_action_scale",
        )
        if any(value < 0.0 or value > 1.0 for value in scripted_action_scale):
            raise ValueError("scripted_contact_action_scale values must lie in [0, 1].")
        object.__setattr__(
            self,
            "scripted_contact_action_scale",
            scripted_action_scale,
        )
        scripted_action_bias = _finite_tuple(
            self.scripted_contact_action_bias,
            size=7,
            name="scripted_contact_action_bias",
        )
        if any(
            abs(bias) + scale > 1.0
            for bias, scale in zip(
                scripted_action_bias,
                scripted_action_scale,
            )
        ):
            raise ValueError(
                "scripted_contact_action_bias plus action scale must keep "
                "every constrained action within [-1, 1]."
            )
        object.__setattr__(
            self,
            "scripted_contact_action_bias",
            scripted_action_bias,
        )
        object.__setattr__(
            self,
            "scripted_contact_nonnegative_lift_action",
            bool(self.scripted_contact_nonnegative_lift_action),
        )
        if not self.scripted_contact_reset and (
            self.scripted_contact_loss_grace_steps > 0
            or scripted_action_scale != (1.0,) * 7
            or scripted_action_bias != (0.0,) * 7
            or self.scripted_contact_nonnegative_lift_action
        ):
            raise ValueError(
                "Scripted-contact curriculum constraints require "
                "scripted_contact_reset=true."
            )
        if not self.scripted_pregrasp_reset and (
            scripted_pregrasp_action_scale != (1.0,) * 7
            or scripted_pregrasp_action_bias != (0.0,) * 7
        ):
            raise ValueError(
                "Scripted-pregrasp action constraints require "
                "scripted_pregrasp_reset=true."
            )
        scripted_min_close = float(self.scripted_contact_reset_min_measured_close)
        if not 0.0 <= scripted_min_close <= self.right_gripper_max_close:
            raise ValueError(
                "scripted_contact_reset_min_measured_close must lie within "
                "the configured gripper close range."
            )
        object.__setattr__(
            self,
            "scripted_contact_reset_min_measured_close",
            scripted_min_close,
        )

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def action_components(self) -> tuple[str, ...]:
        """Names in the real-aligned public task action layout."""

        from .spec import CubeLiftActionSpec

        return CubeLiftActionSpec(layout_version=self.action_layout).components

    @property
    def resolved_policy_action_components(self) -> tuple[str, ...]:
        """Task-owned PPO latent components, with legacy inference isolated."""

        if self.policy_action_components:
            return self.policy_action_components
        return tuple(
            self.action_components[index]
            for index in self._legacy_policy_active_action_indices
        )

    @property
    def policy_action_dim(self) -> int:
        return len(self.resolved_policy_action_components)

    @property
    def resolved_policy_action_scale(self) -> tuple[float, ...]:
        return self.policy_action_scale or (1.0,) * self.policy_action_dim

    @property
    def reach_guidance_enabled(self) -> bool:
        return bool(
            self.reach_guidance_height_offset_m > 0.0
            and self.reach_guidance_xy_blend_radius_m > 0.0
        )

    @property
    def reach_action_distance_scaling_enabled(self) -> bool:
        return bool(
            self.reach_action_min_scale < 1.0 and self.reach_action_slow_radius_m > 0.0
        )

    @property
    def contact_close_translation_scaling_enabled(self) -> bool:
        return bool(
            self.contact_close_translation_scale < 1.0
            and self.contact_close_distance_m > 0.0
            and self.contact_close_keep_measured_close < 1.0
        )

    @property
    def resolved_contact_close_translation_scales(self) -> tuple[float, ...]:
        z_scale = (
            self.contact_close_translation_scale
            if self.contact_close_z_translation_scale is None
            else self.contact_close_z_translation_scale
        )
        return (
            self.contact_close_translation_scale,
            self.contact_close_translation_scale,
            z_scale,
        )

    @property
    def contact_close_anisotropic_translation_scaling_enabled(self) -> bool:
        return bool(
            self.contact_close_translation_scaling_enabled
            and self.contact_close_z_translation_scale is not None
            and self.contact_close_z_translation_scale
            < self.contact_close_translation_scale
        )

    @property
    def contact_close_lift_release_enabled(self) -> bool:
        """Whether measured closure releases z for a stationary pick decoder."""

        return bool(
            self.task_objective == LIFT_TASK_OBJECTIVE
            and self.contact_close_translation_scaling_enabled
            and self.post_contact_phase_enabled
        )

    @property
    def resolved_policy_action_bias(self) -> tuple[float, ...]:
        return self.policy_action_bias or (0.0,) * self.policy_action_dim

    @property
    def available_policy_observation_fields(self) -> tuple[str, ...]:
        """Named fields exposed by the task's raw observation contract."""

        state_field = (
            "proprio"
            if self.policy_class == NORMAL_POLICY_CLASS
            else "privileged_state"
        )
        fields = [state_field]
        if not (
            self.policy_class == NORMAL_POLICY_CLASS
            and self.normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
            }
        ):
            fields.append("episode_progress")
        if (
            self.normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2,
            }
            and self.object_pose_observation_source
        ):
            fields.append("object_pose")
        if (
            self.normal_proprio_profile
            == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2
        ):
            fields.append("reach_target_delta")
        if self.wrist_rgb_enabled:
            fields.append("wrist_rgb")
        return tuple(fields)

    @property
    def resolved_policy_observation_fields(self) -> tuple[str, ...]:
        """Ordered actor input fields, independent of raw task observation."""

        return (
            self.policy_observation_fields or self.available_policy_observation_fields
        )

    @property
    def policy_action_layout(self) -> str:
        components = ".".join(self.resolved_policy_action_components)
        return f"normalized_latent[{components}].v1"

    @property
    def action_conditioning_metadata(self) -> dict[str, Any]:
        """Describe only action-changing task transforms after latent mapping."""

        identity_scale = (1.0,) * self.action_dim
        zero_bias = (0.0,) * self.action_dim
        stages: list[dict[str, Any]] = []
        if self.scripted_pregrasp_reset and (
            self.scripted_pregrasp_action_scale != identity_scale
            or self.scripted_pregrasp_action_bias != zero_bias
        ):
            stages.append(
                {
                    "stage": "scripted_pregrasp_reset",
                    "scale": self.scripted_pregrasp_action_scale,
                    "bias": self.scripted_pregrasp_action_bias,
                    "stateful": False,
                }
            )
        if self.scripted_contact_reset and (
            self.scripted_contact_action_scale != identity_scale
            or self.scripted_contact_action_bias != zero_bias
            or self.scripted_contact_nonnegative_lift_action
        ):
            stages.append(
                {
                    "stage": "scripted_contact_reset",
                    "scale": self.scripted_contact_action_scale,
                    "bias": self.scripted_contact_action_bias,
                    "nonnegative_lift": (self.scripted_contact_nonnegative_lift_action),
                    "stateful": False,
                }
            )
        if self.post_contact_phase_enabled:
            stages.append(
                {
                    "stage": "post_contact_phase",
                    "scale": self.post_contact_action_scale,
                    "bias": self.post_contact_action_bias,
                    "nonnegative_lift": (self.post_contact_nonnegative_lift_action),
                    "activation_steps": (self.post_contact_phase_activation_steps),
                    "minimum_measured_close": (
                        self.post_contact_phase_min_measured_close
                    ),
                    "loss_grace_steps": (self.post_contact_phase_loss_grace_steps),
                    "stateful": True,
                }
            )
        return {
            "version": "cube_lift_action_conditioning.v1",
            "stages": tuple(stages),
        }

    @property
    def action_conditioning_sha256(self) -> str:
        return _metadata_sha256(self.action_conditioning_metadata)

    @property
    def _legacy_policy_active_action_indices(self) -> tuple[int, ...]:
        if not (self.scripted_contact_reset or self.scripted_pregrasp_reset):
            return tuple(range(self.action_dim))
        action_scale = (
            self.scripted_contact_action_scale
            if self.scripted_contact_reset
            else self.scripted_pregrasp_action_scale
        )
        active = tuple(index for index, scale in enumerate(action_scale) if scale > 0.0)
        if not active:
            raise ValueError(
                "A scripted-contact curriculum must expose at least one "
                "policy-controlled action dimension."
            )
        return active

    @property
    def policy_active_action_indices(self) -> tuple[int, ...]:
        """Return policy-controlled indices within the public 7-D action."""

        return tuple(
            self.action_components.index(component)
            for component in self.resolved_policy_action_components
        )

    @property
    def success_termination_reason(self) -> str:
        if self.task_objective == REACH_TASK_OBJECTIVE:
            return TCP_REACH_SUCCESS_REASON
        if self.task_objective == CONTACT_TASK_OBJECTIVE:
            return BILATERAL_CONTACT_SUCCESS_REASON
        return (
            BILATERAL_CONTACT_LIFT_SUCCESS_REASON
            if self.success_requires_bilateral_contact
            else HEIGHT_ONLY_LIFT_SUCCESS_REASON
        )

    @property
    def uses_pose_action(self) -> bool:
        """Whether this task routes policy arm actions through TCP pose IK."""

        return (
            self.execution_mode == ACTUATED_POSITION_EXECUTION_MODE
            and self.schema_version == CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION
            and self.policy_class
            in {NORMAL_POLICY_CLASS, PRIVILEGED_EXPERT_POLICY_CLASS}
        )

    @property
    def action_layout(self) -> str:
        from .spec import (
            CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION,
            CUBE_LIFT_QPOS_ACTION_LAYOUT_VERSION,
        )

        return (
            CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION
            if self.uses_pose_action
            else CUBE_LIFT_QPOS_ACTION_LAYOUT_VERSION
        )

    @property
    def observation_dim(self) -> int:
        image_dim = (
            3 * self.wrist_rgb_width * self.wrist_rgb_height
            if self.wrist_rgb_enabled
            else 0
        )
        return self.proprio_observation_dim + image_dim

    @property
    def proprio_observation_dim(self) -> int:
        if self.policy_class != NORMAL_POLICY_CLASS:
            return 31
        return (
            29
            if self.normal_proprio_profile == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1
            else (
                28
                if self.normal_proprio_profile
                == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2
                else (
                    35
                    if self.normal_proprio_profile
                    == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1
                    else (
                        38
                        if self.normal_proprio_profile
                        == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2
                        else 15
                    )
                )
            )
        )

    @property
    def wrist_rgb_shape(self) -> tuple[int, int, int] | None:
        if not self.wrist_rgb_enabled:
            return None
        return (3, self.wrist_rgb_height, self.wrist_rgb_width)

    @property
    def observation_mode(self) -> str:
        base = (
            (
                "proprioceptive_real_motion_tracking"
                if self.normal_proprio_profile
                == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1
                else (
                    "proprioceptive_real_motion_tracking_no_episode_clock"
                    if self.normal_proprio_profile
                    == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2
                    else (
                        "proprioceptive_real_motion_tracking_no_episode_clock_cube_pose"
                        if self.normal_proprio_profile
                        == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1
                        else (
                            "proprioceptive_real_motion_tracking_no_episode_clock_cube_pose_reach_delta"
                            if self.normal_proprio_profile
                            == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2
                            else "proprioceptive_real"
                        )
                    )
                )
            )
            if self.policy_class == NORMAL_POLICY_CLASS
            else "state_privileged"
        )
        return f"{base}_wrist_rgb" if self.wrist_rgb_enabled else base

    @property
    def observation_layout(self) -> str:
        from .spec import (
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_PRIVILEGED_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_PRIVILEGED_RGB_OBSERVATION_LAYOUT_VERSION,
        )

        if self.policy_class == NORMAL_POLICY_CLASS:
            if not self.wrist_rgb_enabled:
                return (
                    CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION
                    if self.normal_proprio_profile
                    == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1
                    else (
                        CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION
                        if self.normal_proprio_profile
                        == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2
                        else (
                            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION
                            if self.normal_proprio_profile
                            == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1
                            else (
                                CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_OBSERVATION_LAYOUT_VERSION
                                if self.normal_proprio_profile
                                == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2
                                else CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION
                            )
                        )
                    )
                )
            if (
                self.normal_proprio_profile == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1
                and self.wrist_rgb_width == 48
                and self.wrist_rgb_height == 48
            ):
                return (
                    CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION
                )
            if (
                self.normal_proprio_profile
                == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2
                and self.wrist_rgb_width == 48
                and self.wrist_rgb_height == 48
            ):
                return CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION
            return (
                CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION
                if self.wrist_rgb_width == 48 and self.wrist_rgb_height == 48
                else CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION
            )
        if not self.wrist_rgb_enabled:
            return CUBE_LIFT_PRIVILEGED_OBSERVATION_LAYOUT_VERSION
        return (
            CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION
            if self.wrist_rgb_width == 48 and self.wrist_rgb_height == 48
            else CUBE_LIFT_PRIVILEGED_RGB_OBSERVATION_LAYOUT_VERSION
        )

    @property
    def policy_period_sec(self) -> float:
        return 1.0 / self.policy_hz

    @property
    def cube_spawn_qpos(self) -> tuple[float, ...]:
        return (
            self.cube_center_xy_m[0],
            self.cube_center_xy_m[1],
            self.table_top_z_m + self.object_size_m[2] / 2.0 + self.spawn_clearance_m,
            1.0,
            0.0,
            0.0,
            0.0,
        )

    @property
    def randomizes_cube_spawn(self) -> bool:
        return bool(
            any(value > 0.0 for value in self.cube_spawn_xy_jitter_m)
            or self.cube_spawn_yaw_jitter_rad > 0.0
        )

    def sample_cube_spawn_qpos(
        self,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        """Sample one deterministic-from-seed free-joint reset pose."""

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy Generator.")
        jitter = np.asarray(self.cube_spawn_xy_jitter_m, dtype=np.float64)
        xy = np.asarray(self.cube_center_xy_m, dtype=np.float64)
        if np.any(jitter > 0.0):
            xy = xy + rng.uniform(-jitter, jitter)
        yaw = (
            0.0
            if self.cube_spawn_yaw_jitter_rad == 0.0
            else float(
                rng.uniform(
                    -self.cube_spawn_yaw_jitter_rad,
                    self.cube_spawn_yaw_jitter_rad,
                )
            )
        )
        half_yaw = 0.5 * yaw
        return (
            float(xy[0]),
            float(xy[1]),
            self.table_top_z_m + self.object_size_m[2] / 2.0 + self.spawn_clearance_m,
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        )

    def base_contract(self) -> SimulationContract:
        path = resolve_path(self.base_contract_path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != self.base_contract_sha256:
            raise ValueError(
                "CubeLift task base contract hash mismatch: "
                f"expected={self.base_contract_sha256}, actual={digest}, path={path}."
            )
        contract = SimulationContract.load(path)
        timing = contract.timing
        expected_timing = {
            "policy_hz": self.policy_hz,
            "command_hz": self.command_hz,
            "physics_hz": self.physics_hz,
        }
        actual_timing = {
            "policy_hz": timing.policy_hz,
            "command_hz": timing.command_hz,
            "physics_hz": timing.physics_hz,
        }
        timing_mismatches = {
            key: (expected_timing[key], actual_timing[key])
            for key in expected_timing
            if not np.isclose(
                expected_timing[key],
                actual_timing[key],
                rtol=0.0,
                atol=1e-12,
            )
        }
        if timing_mismatches:
            raise ValueError(
                f"CubeLift task clock disagrees with base contract: {timing_mismatches}"
            )
        object_spec = next(
            (
                item
                for item in contract.scene.objects
                if item.name == self.object_body_name
            ),
            None,
        )
        if object_spec is None:
            raise ValueError(
                f"Base contract has no object {self.object_body_name!r} "
                "required by CubeLift."
            )
        if not np.allclose(
            object_spec.size_m,
            self.object_size_m,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "CubeLift object_size_m disagrees with the base contract object."
            )
        return contract

    def to_metadata(self) -> dict[str, Any]:
        return {
            "task_name": self.name,
            "task_schema_version": self.schema_version,
            "task_config_path": self.source_path,
            "task_config_sha256": self.source_sha256,
            "task_config_dependencies": list(self.source_dependencies),
            "base_contract_path": self.base_contract_path,
            "base_contract_sha256": self.base_contract_sha256,
            "execution_mode": self.execution_mode,
            "robot_dynamics": self.robot_dynamics,
            "gripper_dynamics": self.gripper_dynamics,
            "position_actuation_profile_name": self.position_actuation_profile_name,
            "position_actuation_profile_sha256": self.position_actuation_profile_sha256,
            "gripper_actuation_profile_name": self.gripper_actuation_profile_name,
            "gripper_actuation_profile_sha256": self.gripper_actuation_profile_sha256,
            "policy_class": self.policy_class,
            "grasp_mode": self.grasp_mode,
            "controlled_side": self.controlled_side,
            "fixed_side": self.fixed_side,
            "observation_mode": self.observation_mode,
            "action_layout": self.action_layout,
            "pose_control_frame": self.pose_control_frame,
            "task_objective": self.task_objective,
            "success_termination_reason": self.success_termination_reason,
            "reach_target_height_offset_m": self.reach_target_height_offset_m,
            "reach_guidance_enabled": self.reach_guidance_enabled,
            "reach_guidance_height_offset_m": (self.reach_guidance_height_offset_m),
            "reach_guidance_xy_blend_radius_m": (self.reach_guidance_xy_blend_radius_m),
            "reach_action_distance_scaling_enabled": (
                self.reach_action_distance_scaling_enabled
            ),
            "reach_action_min_scale": self.reach_action_min_scale,
            "reach_action_slow_radius_m": (self.reach_action_slow_radius_m),
            "contact_close_translation_scaling_enabled": (
                self.contact_close_translation_scaling_enabled
            ),
            "contact_close_translation_scale": (self.contact_close_translation_scale),
            "contact_close_z_translation_scale": (
                self.contact_close_z_translation_scale
            ),
            "contact_close_translation_scales_xyz": list(
                self.resolved_contact_close_translation_scales
            ),
            "contact_close_distance_m": self.contact_close_distance_m,
            "contact_close_keep_measured_close": (
                self.contact_close_keep_measured_close
            ),
            "contact_close_lift_release_enabled": (
                self.contact_close_lift_release_enabled
            ),
            "contact_close_lift_release_measured_close": (
                self.post_contact_phase_min_measured_close
                if self.contact_close_lift_release_enabled
                else None
            ),
            "reach_success_distance_m": self.reach_success_distance_m,
            "contact_success_min_measured_close": (
                self.contact_success_min_measured_close
            ),
            "contact_success_min_score": self.contact_success_min_score,
            "success_min_safety_clear_policy_steps": (
                self.success_min_safety_clear_policy_steps
            ),
            "success_min_bilateral_inner_contact_score": (
                self.success_min_bilateral_inner_contact_score
            ),
            "success_max_outer_contact_score": (self.success_max_outer_contact_score),
            "policy_action_layout": self.policy_action_layout,
            "policy_action_components": list(self.resolved_policy_action_components),
            "policy_action_components_explicit": bool(self.policy_action_components),
            "policy_action_dim": self.policy_action_dim,
            "policy_action_scale": list(self.resolved_policy_action_scale),
            "policy_action_bias": list(self.resolved_policy_action_bias),
            "action_conditioning": self.action_conditioning_metadata,
            "action_conditioning_sha256": self.action_conditioning_sha256,
            "right_tcp_delta_m": list(self.right_tcp_delta_m),
            "right_tcp_rotation_delta_rad": list(self.right_tcp_rotation_delta_rad),
            "pose_target_anti_windup": self.pose_target_anti_windup,
            "scripted_pregrasp_reset": self.scripted_pregrasp_reset,
            "scripted_pregrasp_distance_m": self.scripted_pregrasp_distance_m,
            "scripted_pregrasp_action_scale": list(self.scripted_pregrasp_action_scale),
            "scripted_pregrasp_action_bias": list(self.scripted_pregrasp_action_bias),
            "post_contact_phase_enabled": self.post_contact_phase_enabled,
            "post_contact_phase_activation_steps": (
                self.post_contact_phase_activation_steps
            ),
            "post_contact_phase_min_measured_close": (
                self.post_contact_phase_min_measured_close
            ),
            "post_contact_phase_loss_grace_steps": (
                self.post_contact_phase_loss_grace_steps
            ),
            "post_contact_action_scale": list(self.post_contact_action_scale),
            "post_contact_action_bias": list(self.post_contact_action_bias),
            "post_contact_nonnegative_lift_action": (
                self.post_contact_nonnegative_lift_action
            ),
            "scripted_contact_reset": self.scripted_contact_reset,
            "scripted_contact_reset_hold_steps": (
                self.scripted_contact_reset_hold_steps
            ),
            "scripted_contact_reset_min_measured_close": (
                self.scripted_contact_reset_min_measured_close
            ),
            "scripted_contact_reset_max_attempts": (
                self.scripted_contact_reset_max_attempts
            ),
            "scripted_contact_loss_grace_steps": (
                self.scripted_contact_loss_grace_steps
            ),
            "scripted_contact_action_scale": list(self.scripted_contact_action_scale),
            "scripted_contact_action_bias": list(self.scripted_contact_action_bias),
            "scripted_contact_nonnegative_lift_action": (
                self.scripted_contact_nonnegative_lift_action
            ),
            "policy_active_action_indices": list(self.policy_active_action_indices),
            "cube_spawn_xy_jitter_m": list(self.cube_spawn_xy_jitter_m),
            "cube_spawn_yaw_jitter_rad": self.cube_spawn_yaw_jitter_rad,
            "randomizes_cube_spawn": self.randomizes_cube_spawn,
            "left_tool_profile_path": self.left_tool_profile_path,
            "right_tool_profile_path": self.right_tool_profile_path,
            "wrist_rgb_enabled": self.wrist_rgb_enabled,
            "wrist_rgb_camera_name": self.wrist_rgb_camera_name,
            "wrist_rgb_width": self.wrist_rgb_width,
            "wrist_rgb_height": self.wrist_rgb_height,
            "wrist_rgb_spatial_pool_size": self.wrist_rgb_spatial_pool_size,
            "normal_proprio_profile": self.normal_proprio_profile,
            "object_pose_observation_source": (
                self.object_pose_observation_source or None
            ),
            "reach_target_delta_observation_source": (
                self.reach_target_delta_observation_source or None
            ),
            "policy_observation_fields": list(self.resolved_policy_observation_fields),
            "policy_observation_fields_explicit": bool(self.policy_observation_fields),
            "proprio_observation_dim": self.proprio_observation_dim,
            "wrist_rgb_shape": (
                None if self.wrist_rgb_shape is None else list(self.wrist_rgb_shape)
            ),
            "observation_layout": self.observation_layout,
            "observation_dim": self.observation_dim,
            "success_requires_bilateral_contact": (
                self.success_requires_bilateral_contact
            ),
            "gripper_contact_surface_specs": [
                spec.to_dict() for spec in self.gripper_contact_surface_specs
            ],
            "reward_shaping": dict(self.reward_shaping),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: str = "",
        source_sha256: str = "",
        source_dependencies: tuple[str, ...] = (),
    ) -> CubeLiftTaskConfig:
        reward_payload = payload.get("reward", {})
        if not isinstance(reward_payload, Mapping):
            raise ValueError(  # noqa: TRY004 - configuration validation contract
                "CubeLift reward must be a JSON object."
            )
        policy_class = str(payload.get("policy_class", NORMAL_POLICY_CLASS))
        default_grasp_mode = (
            PHYSICAL_CONTACT_GRASP_MODE
            if policy_class == NORMAL_POLICY_CLASS
            else CONTACT_LATCH_DEBUG_GRASP_MODE
        )
        return cls(
            schema_version=str(
                payload.get("schema_version", CUBE_LIFT_TASK_SCHEMA_VERSION)
            ),
            name=str(payload["name"]),
            base_contract_path=str(payload["base_contract_path"]),
            base_contract_sha256=str(payload["base_contract_sha256"]),
            execution_mode=str(
                payload.get("execution_mode", DIRECT_QPOS_DEBUG_EXECUTION_MODE)
            ),
            robot_dynamics=str(payload.get("robot_dynamics", "not_simulated")),
            gripper_dynamics=str(
                payload.get("gripper_dynamics", PRESCRIBED_GRIPPER_DYNAMICS)
            ),
            position_actuation_profile_name=(
                None
                if payload.get("position_actuation_profile_name") is None
                else str(payload["position_actuation_profile_name"])
            ),
            position_actuation_profile_sha256=(
                None
                if payload.get("position_actuation_profile_sha256") is None
                else str(payload["position_actuation_profile_sha256"])
            ),
            gripper_actuation_profile_name=(
                None
                if payload.get("gripper_actuation_profile_name") is None
                else str(payload["gripper_actuation_profile_name"])
            ),
            gripper_actuation_profile_sha256=(
                None
                if payload.get("gripper_actuation_profile_sha256") is None
                else str(payload["gripper_actuation_profile_sha256"])
            ),
            policy_class=policy_class,
            grasp_mode=str(payload.get("grasp_mode", default_grasp_mode)),
            controlled_side=str(payload.get("controlled_side", "right")),
            fixed_side=str(payload.get("fixed_side", "left")),
            left_home_qpos=tuple(payload["left_home_qpos"]),
            right_home_qpos=tuple(payload["right_home_qpos"]),
            right_qpos_lower=tuple(payload["right_qpos_lower"]),
            right_qpos_upper=tuple(payload["right_qpos_upper"]),
            right_joint_delta_rad=tuple(payload["right_joint_delta_rad"]),
            right_tcp_delta_m=tuple(
                payload.get("right_tcp_delta_m", (0.02, 0.02, 0.02))
            ),
            right_tcp_rotation_delta_rad=tuple(
                payload.get("right_tcp_rotation_delta_rad", (0.15, 0.15, 0.15))
            ),
            pose_control_frame=str(payload.get("pose_control_frame", "tool_tcp")),
            policy_action_components=tuple(payload.get("policy_action_components", ())),
            policy_action_scale=tuple(payload.get("policy_action_scale", ())),
            policy_action_bias=tuple(payload.get("policy_action_bias", ())),
            left_tool_profile_path=str(
                payload.get("left_tool_profile_path", DEFAULT_LEFT_TOOL_PROFILE_PATH)
            ),
            right_tool_profile_path=str(
                payload.get("right_tool_profile_path", DEFAULT_RIGHT_TOOL_PROFILE_PATH)
            ),
            wrist_rgb_enabled=bool(payload.get("wrist_rgb_enabled", False)),
            wrist_rgb_camera_name=str(
                payload.get("wrist_rgb_camera_name", "right_wrist")
            ),
            wrist_rgb_width=int(payload.get("wrist_rgb_width", 32)),
            wrist_rgb_height=int(payload.get("wrist_rgb_height", 32)),
            wrist_rgb_spatial_pool_size=int(
                payload.get("wrist_rgb_spatial_pool_size", 1)
            ),
            normal_proprio_profile=str(
                payload.get("normal_proprio_profile", NORMAL_PROPRIO_PROFILE_V0)
            ),
            object_pose_observation_source=str(
                payload.get("object_pose_observation_source", "")
            ),
            reach_target_delta_observation_source=str(
                payload.get("reach_target_delta_observation_source", "")
            ),
            policy_observation_fields=tuple(
                payload.get("policy_observation_fields", ())
            ),
            right_gripper_delta=float(payload.get("right_gripper_delta", 0.2)),
            right_gripper_max_close=float(payload.get("right_gripper_max_close", 0.8)),
            left_gripper_name=str(payload.get("left_gripper_name", "left_gripper")),
            right_gripper_name=str(payload.get("right_gripper_name", "right_gripper")),
            initial_gripper_close_by_name=dict(
                payload["initial_gripper_close_by_name"]
            ),
            object_body_name=str(payload.get("object_body_name", "debug_cube")),
            object_freejoint_name=str(
                payload.get("object_freejoint_name", "debug_cube_freejoint")
            ),
            right_grasp_body_name=str(
                payload.get("right_grasp_body_name", "right_gripper_grasp_center_frame")
            ),
            grasp_latch_left_finger_body_name=str(
                payload.get(
                    "grasp_latch_left_finger_body_name", "right_gripper_left_finger"
                )
            ),
            grasp_latch_right_finger_body_name=str(
                payload.get(
                    "grasp_latch_right_finger_body_name", "right_gripper_right_finger"
                )
            ),
            grasp_latch_acquire_close=float(
                payload.get("grasp_latch_acquire_close", 0.45)
            ),
            grasp_latch_release_close=float(
                payload.get("grasp_latch_release_close", 0.20)
            ),
            grasp_latch_acquire_distance_m=float(
                payload.get("grasp_latch_acquire_distance_m", 0.06)
            ),
            gripper_contact_surface_specs=tuple(
                GripperContactSurfaceSpec.from_dict(value)
                for value in payload.get("gripper_contact_surface_specs", ())
            ),
            object_size_m=tuple(payload["object_size_m"]),
            cube_center_xy_m=tuple(payload["cube_center_xy_m"]),
            cube_spawn_xy_jitter_m=tuple(
                payload.get("cube_spawn_xy_jitter_m", (0.0, 0.0))
            ),
            cube_spawn_yaw_jitter_rad=float(
                payload.get("cube_spawn_yaw_jitter_rad", 0.0)
            ),
            table_body_name=str(payload.get("table_body_name", "table")),
            table_top_z_m=float(payload.get("table_top_z_m", 0.01)),
            spawn_clearance_m=float(payload.get("spawn_clearance_m", 0.001)),
            settle_physics_ticks=int(payload.get("settle_physics_ticks", 100)),
            policy_hz=float(payload.get("policy_hz", 15.0)),
            reference_hz=float(payload.get("reference_hz", 60.0)),
            command_hz=float(payload.get("command_hz", DEFAULT_POSITION_COMMAND_HZ)),
            physics_hz=float(payload.get("physics_hz", 1000.0)),
            max_episode_steps=int(payload.get("max_episode_steps", 150)),
            task_objective=str(payload.get("task_objective", LIFT_TASK_OBJECTIVE)),
            reach_target_height_offset_m=float(
                payload.get("reach_target_height_offset_m", 0.03)
            ),
            reach_guidance_height_offset_m=float(
                payload.get("reach_guidance_height_offset_m", 0.0)
            ),
            reach_guidance_xy_blend_radius_m=float(
                payload.get("reach_guidance_xy_blend_radius_m", 0.0)
            ),
            reach_action_min_scale=float(payload.get("reach_action_min_scale", 1.0)),
            reach_action_slow_radius_m=float(
                payload.get("reach_action_slow_radius_m", 0.0)
            ),
            contact_close_translation_scale=float(
                payload.get("contact_close_translation_scale", 1.0)
            ),
            contact_close_z_translation_scale=(
                None
                if payload.get("contact_close_z_translation_scale") is None
                else float(payload["contact_close_z_translation_scale"])
            ),
            contact_close_distance_m=float(
                payload.get("contact_close_distance_m", 0.0)
            ),
            contact_close_keep_measured_close=float(
                payload.get("contact_close_keep_measured_close", 1.0)
            ),
            reach_success_distance_m=float(
                payload.get("reach_success_distance_m", 0.02)
            ),
            contact_success_min_measured_close=float(
                payload.get("contact_success_min_measured_close", 0.45)
            ),
            contact_success_min_score=float(
                payload.get("contact_success_min_score", 0.5)
            ),
            lift_delta_m=float(payload.get("lift_delta_m", 0.08)),
            success_hold_physics_ticks=int(
                payload.get("success_hold_physics_ticks", 20)
            ),
            success_requires_bilateral_contact=bool(
                payload.get("success_requires_bilateral_contact", False)
            ),
            success_min_safety_clear_policy_steps=int(
                payload.get("success_min_safety_clear_policy_steps", 0)
            ),
            success_min_bilateral_inner_contact_score=float(
                payload.get(
                    "success_min_bilateral_inner_contact_score",
                    0.0,
                )
            ),
            success_max_outer_contact_score=float(
                payload.get("success_max_outer_contact_score", 1.0)
            ),
            pose_target_anti_windup=bool(payload.get("pose_target_anti_windup", False)),
            scripted_pregrasp_reset=bool(payload.get("scripted_pregrasp_reset", False)),
            scripted_pregrasp_distance_m=float(
                payload.get("scripted_pregrasp_distance_m", 0.035)
            ),
            scripted_pregrasp_action_scale=tuple(
                payload.get("scripted_pregrasp_action_scale", (1.0,) * 7)
            ),
            scripted_pregrasp_action_bias=tuple(
                payload.get("scripted_pregrasp_action_bias", (0.0,) * 7)
            ),
            post_contact_phase_enabled=bool(
                payload.get("post_contact_phase_enabled", False)
            ),
            post_contact_phase_activation_steps=int(
                payload.get("post_contact_phase_activation_steps", 3)
            ),
            post_contact_phase_min_measured_close=float(
                payload.get("post_contact_phase_min_measured_close", 0.55)
            ),
            post_contact_phase_loss_grace_steps=int(
                payload.get("post_contact_phase_loss_grace_steps", 3)
            ),
            post_contact_action_scale=tuple(
                payload.get("post_contact_action_scale", (1.0,) * 7)
            ),
            post_contact_action_bias=tuple(
                payload.get("post_contact_action_bias", (0.0,) * 7)
            ),
            post_contact_nonnegative_lift_action=bool(
                payload.get("post_contact_nonnegative_lift_action", False)
            ),
            scripted_contact_reset=bool(payload.get("scripted_contact_reset", False)),
            scripted_contact_reset_hold_steps=int(
                payload.get("scripted_contact_reset_hold_steps", 5)
            ),
            scripted_contact_reset_min_measured_close=float(
                payload.get("scripted_contact_reset_min_measured_close", 0.60)
            ),
            scripted_contact_reset_max_attempts=int(
                payload.get("scripted_contact_reset_max_attempts", 1)
            ),
            scripted_contact_loss_grace_steps=int(
                payload.get("scripted_contact_loss_grace_steps", 0)
            ),
            scripted_contact_action_scale=tuple(
                payload.get("scripted_contact_action_scale", (1.0,) * 7)
            ),
            scripted_contact_action_bias=tuple(
                payload.get("scripted_contact_action_bias", (0.0,) * 7)
            ),
            scripted_contact_nonnegative_lift_action=bool(
                payload.get("scripted_contact_nonnegative_lift_action", False)
            ),
            out_of_bounds_xyz_lower_m=tuple(payload["out_of_bounds_xyz_lower_m"]),
            out_of_bounds_xyz_upper_m=tuple(payload["out_of_bounds_xyz_upper_m"]),
            reward=CubeLiftRewardConfig.from_dict(reward_payload),
            reward_shaping=dict(payload.get("reward_shaping", {})),
            source_path=source_path,
            source_sha256=source_sha256,
            source_dependencies=source_dependencies,
        )

    @classmethod
    def load(cls, path: str | Path) -> CubeLiftTaskConfig:
        resolved = resolve_path(path)
        payload, semantic_sha256, dependencies = cls._load_resolved_payload(
            resolved,
            stack=(),
        )
        return cls.from_dict(
            payload,
            source_path=str(resolved),
            source_sha256=semantic_sha256,
            source_dependencies=dependencies,
        )

    @classmethod
    def _load_resolved_payload(
        cls,
        path: Path,
        *,
        stack: tuple[Path, ...],
    ) -> tuple[dict[str, Any], str, tuple[str, ...]]:
        resolved = path.resolve()
        if resolved in stack:
            chain = " -> ".join(str(value) for value in (*stack, resolved))
            raise ValueError(f"CubeLift task config inheritance cycle: {chain}")
        content = resolved.read_bytes()
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(  # noqa: TRY004 - configuration validation contract
                f"CubeLift task config {resolved} must contain a JSON object."
            )
        raw_sha256 = hashlib.sha256(content).hexdigest()
        extends = payload.get("extends")
        if extends is None:
            return dict(payload), raw_sha256, (str(resolved),)
        extends_path = str(extends).strip()
        if not extends_path:
            raise ValueError("CubeLift task config extends must be non-empty.")
        parent = resolve_path(extends_path)
        parent_payload, parent_sha256, parent_dependencies = cls._load_resolved_payload(
            parent, stack=(*stack, resolved)
        )
        overlay = dict(payload)
        overlay.pop("extends", None)
        merged = dict(parent_payload)
        for key, value in overlay.items():
            if (
                key in {"reward", "reward_shaping"}
                and isinstance(merged.get(key), Mapping)
                and isinstance(value, Mapping)
            ):
                merged[key] = {**dict(merged[key]), **dict(value)}
            else:
                merged[key] = value
        semantic_sha256 = hashlib.sha256(
            (
                "dexteleop.cube_lift.task_profile.v1\n"
                f"parent={parent_sha256}\n"
                f"overlay={raw_sha256}\n"
            ).encode()
        ).hexdigest()
        return (
            merged,
            semantic_sha256,
            (*parent_dependencies, str(resolved)),
        )

    @property
    def uses_actuated_position(self) -> bool:
        return self.execution_mode == ACTUATED_POSITION_EXECUTION_MODE

    def position_actuation_profile(self):
        if not self.uses_actuated_position:
            return None
        profile = rm65_bimanual_udp_feedback_profile()
        if profile.name != self.position_actuation_profile_name:
            raise RuntimeError(
                "Resolved RM65 actuation profile name changed after validation."
            )
        return profile

    @property
    def uses_actuated_gripper(self) -> bool:
        return self.gripper_dynamics == MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS

    def gripper_actuation_profile(self):
        if not self.uses_actuated_gripper:
            return None
        profile = resolve_dmtac_gripper_actuation_profile(
            str(self.gripper_actuation_profile_name)
        )
        if profile.name != self.gripper_actuation_profile_name:
            raise RuntimeError(
                "Resolved DM-Tac gripper profile name changed after validation."
            )
        return profile


__all__ = [
    "ACTUATED_POSITION_EXECUTION_MODE",
    "CAD_DMTAC_GRIPPER_ACTUATION_PROFILE_NAME",
    "CONTACT_LATCH_DEBUG_GRASP_MODE",
    "CONTACT_TASK_OBJECTIVE",
    "CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION",
    "CUBE_LIFT_CONTACT_GRIPPER_ACTUATION_PROFILE_NAME",
    "CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION",
    "CUBE_LIFT_TASK_SCHEMA_VERSION",
    "LEGACY_DEBUG_POLICY_CLASS",
    "LIFT_TASK_OBJECTIVE",
    "MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS",
    "MUJOCO_POSITION_ACTUATOR_DYNAMICS",
    "NORMAL_POLICY_CLASS",
    "NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_REACH_DELTA_V2",
    "NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_V1",
    "NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2",
    "NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1",
    "NORMAL_PROPRIO_PROFILE_V0",
    "PHYSICAL_CONTACT_GRASP_MODE",
    "PRESCRIBED_GRIPPER_DYNAMICS",
    "PRIVILEGED_EXPERT_POLICY_CLASS",
    "REACH_TASK_OBJECTIVE",
    "SCRIPTED_CONTACT_LOST_REASON",
    "CubeLiftRewardConfig",
    "CubeLiftTaskConfig",
    "dmtac_gripper_actuation_profile_sha256",
    "resolve_dmtac_gripper_actuation_profile",
    "rm65_position_actuation_profile_sha256",
]
