"""Compatibility exports for lower-owned synthetic Human sources.

New code should import ``teleop_runtime.action.synthetic_human`` directly.
The Station path remains temporarily available to preserve existing profiles
and tests while the old MuJoCo composition fixtures are retired.
"""

from teleop_runtime.action.synthetic_human import (
    SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1,
    SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1,
    SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1,
    SyntheticHumanPoseSource,
    SyntheticHumanPoseSourceSpec,
    SyntheticHumanQposSource,
    SyntheticHumanQposSourceSpec,
)


__all__ = [
    "SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1",
    "SyntheticHumanPoseSource",
    "SyntheticHumanPoseSourceSpec",
    "SyntheticHumanQposSource",
    "SyntheticHumanQposSourceSpec",
]
