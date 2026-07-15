"""Compatibility re-export for legacy teleop_runtime.deploy.observation imports.

New common deploy code should import from robot_motion.deploy.observation directly.
"""

from __future__ import annotations

from robot_motion.deploy.observation import (
    DEFAULT_CAMERA_FRAME_MAX_AGE_SEC,
    DeploymentObservation,
    build_deployment_observation,
    build_state_vector,
    build_tactile_vectors,
    latest_camera_frames,
    observation_faults,
)

__all__ = [
    "DEFAULT_CAMERA_FRAME_MAX_AGE_SEC",
    "DeploymentObservation",
    "build_deployment_observation",
    "build_state_vector",
    "build_tactile_vectors",
    "latest_camera_frames",
    "observation_faults",
]
