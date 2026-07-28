"""Task-local RL spaces; intentionally separate from real/deploy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# The retained direct-qpos debug task keeps its historical action contract.
# Physical-policy tasks use the pose contract below; both are seven normalized
# values so the PPO policy head does not need a second shape-specific path.
CUBE_LIFT_QPOS_ACTION_LAYOUT_VERSION = "right_qpos_delta6_right_gripper_delta1.v0"
CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION = (
    "right_tcp_delta_xyz3_rotvec3_right_gripper_delta1.v1"
)
CUBE_LIFT_ACTION_LAYOUT_VERSION = CUBE_LIFT_QPOS_ACTION_LAYOUT_VERSION
CUBE_LIFT_PRIVILEGED_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_cube_pose7_cube_velocity6_"
    "cube_to_grasp3_last_action7_progress1.v0"
)
CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_last_action7_progress1.v0"
)
CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_last_action7_progress1_"
    "right_wrist_rgb_chw_3x32x32.v1"
)
CUBE_LIFT_OBSERVATION_LAYOUT_VERSION = CUBE_LIFT_PRIVILEGED_OBSERVATION_LAYOUT_VERSION
CUBE_LIFT_PRIVILEGED_RGB_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_cube_pose7_cube_velocity6_"
    "cube_to_grasp3_last_action7_progress1_right_wrist_rgb_chw_3x32x32.v1"
)
CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_last_action7_progress1_"
    "right_wrist_rgb_chw_3x48x48.v2"
)
CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
    "right_target_error6_right_gripper_target_error1_last_action7_progress1.v1"
)
CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
    "right_target_error6_right_gripper_target_error1_last_action7_progress1_"
    "right_wrist_rgb_chw_3x48x48.v3"
)
CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
    "right_target_error6_right_gripper_target_error1_last_action7.v2"
)
CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
    "right_target_error6_right_gripper_target_error1_last_action7_cube_pose7.v1"
)
CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_right_qpos_delta6_right_close_delta1_"
    "right_target_error6_right_gripper_target_error1_last_action7_"
    "right_wrist_rgb_chw_3x48x48.v4"
)
CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_cube_pose7_cube_velocity6_"
    "cube_to_grasp3_last_action7_progress1_right_wrist_rgb_chw_3x48x48.v2"
)

# The image contracts are deliberately small and versioned.  The MuJoCo
# renderer produces HWC uint8 RGB; the task observation appends the same
# pixels as CHW float32 values in [0, 1].  Keeping each shape in the layout
# version prevents a checkpoint from silently changing its preprocessing.
CUBE_LIFT_WRIST_RGB_IMAGE_SHAPE = (3, 32, 32)
CUBE_LIFT_WRIST_RGB_IMAGE_SHAPE_48 = (3, 48, 48)

NORMAL_OBSERVATION_FIELDS = (
    "right_qpos_0",
    "right_qpos_1",
    "right_qpos_2",
    "right_qpos_3",
    "right_qpos_4",
    "right_qpos_5",
    "right_gripper_close",
    "last_action_0",
    "last_action_1",
    "last_action_2",
    "last_action_3",
    "last_action_4",
    "last_action_5",
    "last_action_6",
    "episode_progress",
)

NORMAL_MOTION_TRACKING_OBSERVATION_FIELDS = (
    "right_qpos_0",
    "right_qpos_1",
    "right_qpos_2",
    "right_qpos_3",
    "right_qpos_4",
    "right_qpos_5",
    "right_gripper_close",
    "right_qpos_delta_0",
    "right_qpos_delta_1",
    "right_qpos_delta_2",
    "right_qpos_delta_3",
    "right_qpos_delta_4",
    "right_qpos_delta_5",
    "right_gripper_close_delta",
    "right_target_error_0",
    "right_target_error_1",
    "right_target_error_2",
    "right_target_error_3",
    "right_target_error_4",
    "right_target_error_5",
    "right_gripper_target_error",
    "last_action_0",
    "last_action_1",
    "last_action_2",
    "last_action_3",
    "last_action_4",
    "last_action_5",
    "last_action_6",
    "episode_progress",
)

NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_FIELDS = (
    "right_qpos_0",
    "right_qpos_1",
    "right_qpos_2",
    "right_qpos_3",
    "right_qpos_4",
    "right_qpos_5",
    "right_gripper_close",
    "right_qpos_delta_0",
    "right_qpos_delta_1",
    "right_qpos_delta_2",
    "right_qpos_delta_3",
    "right_qpos_delta_4",
    "right_qpos_delta_5",
    "right_gripper_close_delta",
    "right_target_error_0",
    "right_target_error_1",
    "right_target_error_2",
    "right_target_error_3",
    "right_target_error_4",
    "right_target_error_5",
    "right_gripper_target_error",
    "last_action_0",
    "last_action_1",
    "last_action_2",
    "last_action_3",
    "last_action_4",
    "last_action_5",
    "last_action_6",
)

NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_FIELDS = (
    *NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_FIELDS,
    "cube_x_m",
    "cube_y_m",
    "cube_z_m",
    "cube_quat_w",
    "cube_quat_x",
    "cube_quat_y",
    "cube_quat_z",
)


def require_gymnasium():
    """Import Gymnasium lazily so config/physics utilities remain lightweight."""

    try:
        import gymnasium
        from gymnasium import spaces
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CubeLiftEnv requires gymnasium. Install it in dexteleop310 with "
            "`python -m pip install gymnasium`."
        ) from exc
    return gymnasium, spaces


@dataclass(frozen=True)
class CubeLiftActionSpec:
    """Seven normalized action values consumed at the policy rate.

    ``right_qpos_delta6_right_gripper_delta1.v0`` is retained for the explicit
    direct-qpos debug task.  Fitted physical-policy tasks use
    ``right_tcp_delta_xyz3_rotvec3_right_gripper_delta1.v1``: three base-frame
    TCP translations, three base-frame rotation-vector increments, and one
    normalized gripper-close increment.
    """

    layout_version: str = CUBE_LIFT_ACTION_LAYOUT_VERSION
    dtype: np.dtype = np.dtype(np.float32)
    dim: int = 7

    def __post_init__(self) -> None:
        if self.layout_version not in {
            CUBE_LIFT_QPOS_ACTION_LAYOUT_VERSION,
            CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION,
        } or self.dim != 7:
            raise ValueError("Unsupported CubeLift action layout.")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def coerce(self, action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=self.dtype).reshape(-1)
        if values.shape != (self.dim,) or not np.all(np.isfinite(values)):
            raise ValueError("CubeLift action must be a finite 7-D vector.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("CubeLift action values must lie in [-1, 1].")
        return values.copy()

    def gym_space(self):
        _gymnasium, spaces = require_gymnasium()
        return spaces.Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=self.dtype)


@dataclass(frozen=True)
class CubeLiftObservationSpec:
    """Versioned normal or privileged policy observation."""

    layout_version: str = CUBE_LIFT_OBSERVATION_LAYOUT_VERSION
    dtype: np.dtype = np.dtype(np.float32)
    field_names: tuple[str, ...] = (
        "right_qpos_0",
        "right_qpos_1",
        "right_qpos_2",
        "right_qpos_3",
        "right_qpos_4",
        "right_qpos_5",
        "right_gripper_close",
        "cube_x_m",
        "cube_y_m",
        "cube_z_m",
        "cube_quat_w",
        "cube_quat_x",
        "cube_quat_y",
        "cube_quat_z",
        "cube_linear_velocity_x_mps",
        "cube_linear_velocity_y_mps",
        "cube_linear_velocity_z_mps",
        "cube_angular_velocity_x_radps",
        "cube_angular_velocity_y_radps",
        "cube_angular_velocity_z_radps",
        "cube_to_right_grasp_x_m",
        "cube_to_right_grasp_y_m",
        "cube_to_right_grasp_z_m",
        "last_action_0",
        "last_action_1",
        "last_action_2",
        "last_action_3",
        "last_action_4",
        "last_action_5",
        "last_action_6",
        "episode_progress",
    )

    def __post_init__(self) -> None:
        supported = {
            CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION: NORMAL_OBSERVATION_FIELDS,
            CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION: NORMAL_OBSERVATION_FIELDS,
            CUBE_LIFT_PRIVILEGED_OBSERVATION_LAYOUT_VERSION: self.field_names,
            CUBE_LIFT_PRIVILEGED_RGB_OBSERVATION_LAYOUT_VERSION: self.field_names,
            CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION: NORMAL_OBSERVATION_FIELDS,
            CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION: self.field_names,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION: NORMAL_MOTION_TRACKING_OBSERVATION_FIELDS,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION: NORMAL_MOTION_TRACKING_OBSERVATION_FIELDS,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION: NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_FIELDS,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION: NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_FIELDS,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION: NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_FIELDS,
        }
        if self.layout_version not in supported:
            raise ValueError("Unsupported CubeLift observation layout.")
        expected_fields = supported[self.layout_version]
        if self.layout_version in {
            CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION,
        }:
            object.__setattr__(self, "field_names", expected_fields)
        elif len(self.field_names) != 31:
            raise ValueError("Privileged CubeLift observation layout must remain 31-D.")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def proprio_dim(self) -> int:
        return len(self.field_names)

    @property
    def wrist_rgb_enabled(self) -> bool:
        return self.layout_version in {
            CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_PRIVILEGED_RGB_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION,
        }

    @property
    def wrist_rgb_shape(self) -> tuple[int, int, int] | None:
        if not self.wrist_rgb_enabled:
            return None
        if self.layout_version in {
            CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION,
            CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION,
        }:
            return CUBE_LIFT_WRIST_RGB_IMAGE_SHAPE_48
        return CUBE_LIFT_WRIST_RGB_IMAGE_SHAPE

    @property
    def dim(self) -> int:
        image_dim = (
            int(np.prod(self.wrist_rgb_shape, dtype=np.int64))
            if self.wrist_rgb_enabled
            else 0
        )
        return len(self.field_names) + image_dim

    def coerce(self, observation: Any) -> np.ndarray:
        values = np.asarray(observation, dtype=self.dtype).reshape(-1)
        if values.shape != (self.dim,) or not np.all(np.isfinite(values)):
            raise RuntimeError("CubeLift observation became non-finite or changed layout.")
        return values.copy()

    def gym_space(self):
        _gymnasium, spaces = require_gymnasium()
        # Keep finite declared bounds so Gymnasium's API checker can catch
        # non-finite observations rather than accepting them through inf bounds.
        return spaces.Box(low=-1e6, high=1e6, shape=(self.dim,), dtype=self.dtype)


__all__ = [
    "CUBE_LIFT_ACTION_LAYOUT_VERSION",
    "CUBE_LIFT_POSE_ACTION_LAYOUT_VERSION",
    "CUBE_LIFT_QPOS_ACTION_LAYOUT_VERSION",
    "CUBE_LIFT_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_RGB_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_RGB_48_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_MOTION_TRACKING_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_48_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_NORMAL_RGB_MOTION_TRACKING_NO_EPISODE_CLOCK_48_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_PRIVILEGED_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_PRIVILEGED_RGB_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_PRIVILEGED_RGB_48_OBSERVATION_LAYOUT_VERSION",
    "CUBE_LIFT_WRIST_RGB_IMAGE_SHAPE",
    "CUBE_LIFT_WRIST_RGB_IMAGE_SHAPE_48",
    "NORMAL_MOTION_TRACKING_OBSERVATION_FIELDS",
    "NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_OBSERVATION_FIELDS",
    "NORMAL_MOTION_TRACKING_NO_EPISODE_CLOCK_CUBE_POSE_OBSERVATION_FIELDS",
    "CubeLiftActionSpec",
    "CubeLiftObservationSpec",
    "require_gymnasium",
]
