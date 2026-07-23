"""Versioned configuration for the minimal right-arm Cube-Lift task."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from project_paths import resolve_path
from robot_motion.position_control import DEFAULT_POSITION_COMMAND_HZ
from sim.contracts import SimulationContract

from ...actuation import (
    identified_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from ...control import DIRECT_QPOS_DEBUG_EXECUTION_MODE


CUBE_LIFT_TASK_SCHEMA_VERSION = "dexteleop.sim.mujoco.cube_lift_task.v0"
CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION = "dexteleop.sim.mujoco.cube_lift_task.v1"
CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION = "dexteleop.sim.mujoco.cube_lift_task.v2"
ACTUATED_POSITION_EXECUTION_MODE = "actuated_position_servo"
MUJOCO_POSITION_ACTUATOR_DYNAMICS = "mujoco_position_actuator"
PRESCRIBED_GRIPPER_DYNAMICS = "prescribed_qpos_debug"
MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS = "mujoco_identified_position_actuator"
RM65_ACTUATION_PROFILE_NAME = "rm65_bimanual_udp_feedback_20260721.v4"
DMTAC_GRIPPER_ACTUATION_PROFILE_NAME = "dmtac_bimanual_position_20260722.v1"
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
PHYSICAL_CONTACT_GRASP_MODE = "physical_contact"
CONTACT_LATCH_DEBUG_GRASP_MODE = "contact_latch_debug"
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


def dmtac_gripper_actuation_profile_sha256() -> str:
    return _metadata_sha256(identified_bimanual_dmtac_gripper_profile().to_metadata())


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


def _unit_interval_mapping(values: Mapping[str, float], *, name: str) -> Mapping[str, float]:
    result: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key).strip()
        value = float(raw_value)
        if not key or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must map non-empty names to finite [0, 1] values.")
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "CubeLiftRewardConfig":
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
    object_size_m: tuple[float, ...] = (0.03, 0.03, 0.03)
    cube_center_xy_m: tuple[float, ...] = (-0.472, 0.0)
    table_body_name: str = "table"
    table_top_z_m: float = 0.01
    spawn_clearance_m: float = 0.001
    settle_physics_ticks: int = 100
    policy_hz: float = 15.0
    reference_hz: float = 60.0
    command_hz: float = DEFAULT_POSITION_COMMAND_HZ
    physics_hz: float = 1000.0
    max_episode_steps: int = 150
    lift_delta_m: float = 0.08
    success_hold_physics_ticks: int = 20
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
        if self.execution_mode == DIRECT_QPOS_DEBUG_EXECUTION_MODE:
            if self.schema_version != CUBE_LIFT_TASK_SCHEMA_VERSION:
                raise ValueError("Direct-qpos CubeLift must use the v0 task schema.")
            if self.robot_dynamics != "not_simulated":
                raise ValueError("Direct-qpos CubeLift must declare robot_dynamics=not_simulated.")
            if self.position_actuation_profile_name is not None:
                raise ValueError("Direct-qpos CubeLift cannot select an actuation profile.")
            if self.gripper_dynamics != PRESCRIBED_GRIPPER_DYNAMICS:
                raise ValueError("Direct-qpos CubeLift requires prescribed grippers.")
            if self.gripper_actuation_profile_name is not None:
                raise ValueError("Direct-qpos CubeLift cannot select a gripper profile.")
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
                    raise ValueError("Arm-actuated CubeLift v1 requires prescribed grippers.")
                if self.gripper_actuation_profile_name is not None:
                    raise ValueError("Arm-actuated CubeLift v1 cannot select a gripper profile.")
            else:
                if self.gripper_dynamics != MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS:
                    raise ValueError(
                        "Fully-actuated CubeLift v2 requires the identified MuJoCo gripper."
                    )
                if (
                    self.gripper_actuation_profile_name
                    != DMTAC_GRIPPER_ACTUATION_PROFILE_NAME
                ):
                    raise ValueError(
                        "Fully-actuated CubeLift requires gripper profile "
                        f"{DMTAC_GRIPPER_ACTUATION_PROFILE_NAME!r}."
                    )
                expected_gripper_hash = dmtac_gripper_actuation_profile_sha256()
                if self.gripper_actuation_profile_sha256 != expected_gripper_hash:
                    raise ValueError(
                        "CubeLift gripper actuation profile hash mismatch: "
                        f"expected={self.gripper_actuation_profile_sha256}, "
                        f"actual={expected_gripper_hash}."
                    )
        else:
            raise ValueError(f"Unsupported CubeLift execution_mode {self.execution_mode!r}.")
        if self.policy_class not in {
            NORMAL_POLICY_CLASS,
            PRIVILEGED_EXPERT_POLICY_CLASS,
            LEGACY_DEBUG_POLICY_CLASS,
        }:
            raise ValueError(f"Unsupported CubeLift policy_class {self.policy_class!r}.")
        if self.grasp_mode not in {
            PHYSICAL_CONTACT_GRASP_MODE,
            CONTACT_LATCH_DEBUG_GRASP_MODE,
        }:
            raise ValueError(f"Unsupported CubeLift grasp_mode {self.grasp_mode!r}.")
        if self.policy_class in {
            NORMAL_POLICY_CLASS,
            PRIVILEGED_EXPERT_POLICY_CLASS,
        } and self.schema_version != CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION:
            raise ValueError(
                "Normal and privileged-expert policies require the same "
                "fully-actuated task schema v2."
            )
        if self.policy_class == NORMAL_POLICY_CLASS:
            if self.grasp_mode != PHYSICAL_CONTACT_GRASP_MODE:
                raise ValueError("Normal CubeLift policy cannot use the debug grasp latch.")
        if self.uses_pose_action and str(self.pose_control_frame).strip() != "tool_tcp":
            raise ValueError("Physical CubeLift pose actions must control the tool_tcp frame.")
        if (self.controlled_side, self.fixed_side) != ("right", "left"):
            raise ValueError("CubeLift controls right and holds left.")
        wrist_rgb_enabled = bool(self.wrist_rgb_enabled)
        wrist_rgb_camera_name = str(self.wrist_rgb_camera_name).strip()
        wrist_rgb_width = int(self.wrist_rgb_width)
        wrist_rgb_height = int(self.wrist_rgb_height)
        wrist_rgb_spatial_pool_size = int(self.wrist_rgb_spatial_pool_size)
        normal_proprio_profile = str(self.normal_proprio_profile).strip()
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
        }:
            raise ValueError(
                "Unsupported normal_proprio_profile: "
                f"{normal_proprio_profile!r}."
            )
        if (
            normal_proprio_profile
            in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
            }
            and self.policy_class != NORMAL_POLICY_CLASS
        ):
            raise ValueError(
                "The motion/tracking proprio profile is only valid for normal policies."
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
        base_contract_path = str(self.base_contract_path).strip()
        base_contract_hash = str(self.base_contract_sha256).strip().lower()
        if not base_contract_path:
            raise ValueError("base_contract_path must be non-empty.")
        if len(base_contract_hash) != 64 or any(char not in "0123456789abcdef" for char in base_contract_hash):
            raise ValueError("base_contract_sha256 must be a lowercase SHA-256 hex digest.")

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
            raise ValueError("right_qpos_lower must be strictly below right_qpos_upper.")
        if any(delta <= 0.0 for delta in joint_delta):
            raise ValueError("right_joint_delta_rad must be strictly positive.")
        if any(delta <= 0.0 for delta in tcp_delta):
            raise ValueError("right_tcp_delta_m must be strictly positive.")
        if any(delta <= 0.0 for delta in tcp_rotation_delta):
            raise ValueError("right_tcp_rotation_delta_rad must be strictly positive.")
        if any(value < low or value > high for value, low, high in zip(right_home, lower, upper)):
            raise ValueError("right_home_qpos must lie within right qpos bounds.")

        object_size = _finite_tuple(self.object_size_m, size=3, name="object_size_m")
        if any(value <= 0.0 for value in object_size):
            raise ValueError("object_size_m must be positive.")
        cube_center = _finite_tuple(self.cube_center_xy_m, size=2, name="cube_center_xy_m")
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
        if not all(math.isfinite(value) and value > 0.0 for value in scalar_values.values()):
            raise ValueError(f"CubeLift positive finite values required: {scalar_names}.")
        if scalar_values["physics_hz"] < scalar_values["command_hz"]:
            raise ValueError("physics_hz must be at least command_hz.")
        if scalar_values["command_hz"] < scalar_values["reference_hz"]:
            raise ValueError("command_hz must be at least reference_hz.")
        if scalar_values["reference_hz"] < scalar_values["policy_hz"]:
            raise ValueError("reference_hz must be at least policy_hz.")
        if self.settle_physics_ticks < 0 or self.max_episode_steps <= 0:
            raise ValueError("settle_physics_ticks must be >= 0 and max_episode_steps > 0.")
        if self.success_hold_physics_ticks <= 0:
            raise ValueError("success_hold_physics_ticks must be positive.")
        unit_values = {
            "right_gripper_max_close": float(self.right_gripper_max_close),
            "grasp_latch_acquire_close": float(self.grasp_latch_acquire_close),
            "grasp_latch_release_close": float(self.grasp_latch_release_close),
        }
        if any(not 0.0 <= value <= 1.0 for value in unit_values.values()):
            raise ValueError(f"CubeLift latch close values must be in [0, 1]: {unit_values}.")
        if unit_values["grasp_latch_release_close"] >= unit_values["grasp_latch_acquire_close"]:
            raise ValueError("grasp latch release close must be below acquire close.")
        if unit_values["grasp_latch_acquire_close"] > unit_values["right_gripper_max_close"]:
            raise ValueError("grasp latch acquire close cannot exceed gripper max close.")

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
        object.__setattr__(self, "pose_control_frame", str(self.pose_control_frame).strip())
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
        object.__setattr__(self, "wrist_rgb_spatial_pool_size", wrist_rgb_spatial_pool_size)
        object.__setattr__(self, "normal_proprio_profile", normal_proprio_profile)
        object.__setattr__(self, "object_size_m", object_size)
        object.__setattr__(self, "cube_center_xy_m", cube_center)
        object.__setattr__(self, "out_of_bounds_xyz_lower_m", lower_xyz)
        object.__setattr__(self, "out_of_bounds_xyz_upper_m", upper_xyz)
        object.__setattr__(self, "initial_gripper_close_by_name", grippers)
        object.__setattr__(self, "reward_shaping", reward_shaping)
        for field_name, value in scalar_values.items():
            object.__setattr__(self, field_name, value)
        for field_name, value in unit_values.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "settle_physics_ticks", int(self.settle_physics_ticks))
        object.__setattr__(self, "max_episode_steps", int(self.max_episode_steps))
        object.__setattr__(self, "success_hold_physics_ticks", int(self.success_hold_physics_ticks))

    @property
    def action_dim(self) -> int:
        return 7

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
                else 15
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
                    else "proprioceptive_real"
                )
            )
            if self.policy_class == NORMAL_POLICY_CLASS
            else "state_privileged"
        )
        return f"{base}_wrist_rgb" if self.wrist_rgb_enabled else base

    @property
    def observation_layout(self) -> str:
        from .spec import (
            CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION,
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
                        else CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION
                    )
                )
            if (
                self.normal_proprio_profile
                == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1
                and self.wrist_rgb_width == 48
                and self.wrist_rgb_height == 48
            ):
                return CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION
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

    def base_contract(self) -> SimulationContract:
        path = resolve_path(self.base_contract_path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != self.base_contract_sha256:
            raise ValueError(
                "CubeLift task base contract hash mismatch: "
                f"expected={self.base_contract_sha256}, actual={digest}, path={path}."
            )
        return SimulationContract.load(path)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "task_name": self.name,
            "task_schema_version": self.schema_version,
            "task_config_path": self.source_path,
            "task_config_sha256": self.source_sha256,
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
            "observation_mode": self.observation_mode,
            "action_layout": self.action_layout,
            "pose_control_frame": self.pose_control_frame,
            "right_tcp_delta_m": list(self.right_tcp_delta_m),
            "right_tcp_rotation_delta_rad": list(self.right_tcp_rotation_delta_rad),
            "left_tool_profile_path": self.left_tool_profile_path,
            "right_tool_profile_path": self.right_tool_profile_path,
            "wrist_rgb_enabled": self.wrist_rgb_enabled,
            "wrist_rgb_camera_name": self.wrist_rgb_camera_name,
            "wrist_rgb_width": self.wrist_rgb_width,
            "wrist_rgb_height": self.wrist_rgb_height,
            "wrist_rgb_spatial_pool_size": self.wrist_rgb_spatial_pool_size,
            "normal_proprio_profile": self.normal_proprio_profile,
            "proprio_observation_dim": self.proprio_observation_dim,
            "wrist_rgb_shape": (
                None if self.wrist_rgb_shape is None else list(self.wrist_rgb_shape)
            ),
            "observation_layout": self.observation_layout,
            "observation_dim": self.observation_dim,
            "reward_shaping": dict(self.reward_shaping),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, source_path: str = "", source_sha256: str = "") -> "CubeLiftTaskConfig":
        reward_payload = payload.get("reward", {})
        if not isinstance(reward_payload, Mapping):
            raise ValueError("CubeLift reward must be a JSON object.")
        policy_class = str(payload.get("policy_class", NORMAL_POLICY_CLASS))
        default_grasp_mode = (
            PHYSICAL_CONTACT_GRASP_MODE
            if policy_class == NORMAL_POLICY_CLASS
            else CONTACT_LATCH_DEBUG_GRASP_MODE
        )
        return cls(
            schema_version=str(payload.get("schema_version", CUBE_LIFT_TASK_SCHEMA_VERSION)),
            name=str(payload["name"]),
            base_contract_path=str(payload["base_contract_path"]),
            base_contract_sha256=str(payload["base_contract_sha256"]),
            execution_mode=str(payload.get("execution_mode", DIRECT_QPOS_DEBUG_EXECUTION_MODE)),
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
            right_tcp_delta_m=tuple(payload.get("right_tcp_delta_m", (0.02, 0.02, 0.02))),
            right_tcp_rotation_delta_rad=tuple(
                payload.get("right_tcp_rotation_delta_rad", (0.15, 0.15, 0.15))
            ),
            pose_control_frame=str(payload.get("pose_control_frame", "tool_tcp")),
            left_tool_profile_path=str(
                payload.get("left_tool_profile_path", DEFAULT_LEFT_TOOL_PROFILE_PATH)
            ),
            right_tool_profile_path=str(
                payload.get("right_tool_profile_path", DEFAULT_RIGHT_TOOL_PROFILE_PATH)
            ),
            wrist_rgb_enabled=bool(payload.get("wrist_rgb_enabled", False)),
            wrist_rgb_camera_name=str(payload.get("wrist_rgb_camera_name", "right_wrist")),
            wrist_rgb_width=int(payload.get("wrist_rgb_width", 32)),
            wrist_rgb_height=int(payload.get("wrist_rgb_height", 32)),
            wrist_rgb_spatial_pool_size=int(payload.get("wrist_rgb_spatial_pool_size", 1)),
            normal_proprio_profile=str(
                payload.get("normal_proprio_profile", NORMAL_PROPRIO_PROFILE_V0)
            ),
            right_gripper_delta=float(payload.get("right_gripper_delta", 0.2)),
            right_gripper_max_close=float(payload.get("right_gripper_max_close", 0.8)),
            left_gripper_name=str(payload.get("left_gripper_name", "left_gripper")),
            right_gripper_name=str(payload.get("right_gripper_name", "right_gripper")),
            initial_gripper_close_by_name=dict(payload["initial_gripper_close_by_name"]),
            object_body_name=str(payload.get("object_body_name", "debug_cube")),
            object_freejoint_name=str(payload.get("object_freejoint_name", "debug_cube_freejoint")),
            right_grasp_body_name=str(
                payload.get("right_grasp_body_name", "right_gripper_grasp_center_frame")
            ),
            grasp_latch_left_finger_body_name=str(
                payload.get("grasp_latch_left_finger_body_name", "right_gripper_left_finger")
            ),
            grasp_latch_right_finger_body_name=str(
                payload.get("grasp_latch_right_finger_body_name", "right_gripper_right_finger")
            ),
            grasp_latch_acquire_close=float(payload.get("grasp_latch_acquire_close", 0.45)),
            grasp_latch_release_close=float(payload.get("grasp_latch_release_close", 0.20)),
            grasp_latch_acquire_distance_m=float(
                payload.get("grasp_latch_acquire_distance_m", 0.06)
            ),
            object_size_m=tuple(payload["object_size_m"]),
            cube_center_xy_m=tuple(payload["cube_center_xy_m"]),
            table_body_name=str(payload.get("table_body_name", "table")),
            table_top_z_m=float(payload.get("table_top_z_m", 0.01)),
            spawn_clearance_m=float(payload.get("spawn_clearance_m", 0.001)),
            settle_physics_ticks=int(payload.get("settle_physics_ticks", 100)),
            policy_hz=float(payload.get("policy_hz", 15.0)),
            reference_hz=float(payload.get("reference_hz", 60.0)),
            command_hz=float(payload.get("command_hz", DEFAULT_POSITION_COMMAND_HZ)),
            physics_hz=float(payload.get("physics_hz", 1000.0)),
            max_episode_steps=int(payload.get("max_episode_steps", 150)),
            lift_delta_m=float(payload.get("lift_delta_m", 0.08)),
            success_hold_physics_ticks=int(payload.get("success_hold_physics_ticks", 20)),
            out_of_bounds_xyz_lower_m=tuple(payload["out_of_bounds_xyz_lower_m"]),
            out_of_bounds_xyz_upper_m=tuple(payload["out_of_bounds_xyz_upper_m"]),
            reward=CubeLiftRewardConfig.from_dict(reward_payload),
            reward_shaping=dict(payload.get("reward_shaping", {})),
            source_path=source_path,
            source_sha256=source_sha256,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CubeLiftTaskConfig":
        resolved = resolve_path(path)
        content = resolved.read_bytes()
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"CubeLift task config {resolved} must contain a JSON object.")
        return cls.from_dict(
            payload,
            source_path=str(resolved),
            source_sha256=hashlib.sha256(content).hexdigest(),
        )

    @property
    def uses_actuated_position(self) -> bool:
        return self.execution_mode == ACTUATED_POSITION_EXECUTION_MODE

    def position_actuation_profile(self):
        if not self.uses_actuated_position:
            return None
        profile = rm65_bimanual_udp_feedback_profile()
        if profile.name != self.position_actuation_profile_name:
            raise RuntimeError("Resolved RM65 actuation profile name changed after validation.")
        return profile

    @property
    def uses_actuated_gripper(self) -> bool:
        return self.gripper_dynamics == MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS

    def gripper_actuation_profile(self):
        if not self.uses_actuated_gripper:
            return None
        profile = identified_bimanual_dmtac_gripper_profile()
        if profile.name != self.gripper_actuation_profile_name:
            raise RuntimeError("Resolved DM-Tac gripper profile name changed after validation.")
        return profile


__all__ = [
    "ACTUATED_POSITION_EXECUTION_MODE",
    "CUBE_LIFT_ACTUATED_TASK_SCHEMA_VERSION",
    "CUBE_LIFT_FULLY_ACTUATED_TASK_SCHEMA_VERSION",
    "CUBE_LIFT_TASK_SCHEMA_VERSION",
    "CubeLiftRewardConfig",
    "CubeLiftTaskConfig",
    "MUJOCO_POSITION_ACTUATOR_DYNAMICS",
    "MUJOCO_IDENTIFIED_GRIPPER_DYNAMICS",
    "PRESCRIBED_GRIPPER_DYNAMICS",
    "CONTACT_LATCH_DEBUG_GRASP_MODE",
    "LEGACY_DEBUG_POLICY_CLASS",
    "NORMAL_POLICY_CLASS",
    "NORMAL_PROPRIO_PROFILE_V0",
    "NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1",
    "NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2",
    "PHYSICAL_CONTACT_GRASP_MODE",
    "PRIVILEGED_EXPERT_POLICY_CLASS",
    "rm65_position_actuation_profile_sha256",
    "dmtac_gripper_actuation_profile_sha256",
]
