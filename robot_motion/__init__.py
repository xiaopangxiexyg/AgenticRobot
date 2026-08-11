"""Shared motion APIs with dependency-local lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, str] = {
    "CompositeKinematicsProvider": ".composite_provider",
    "FaultEvent": ".diagnostics",
    "FaultSeverity": ".diagnostics",
    "RateTracker": ".diagnostics",
    "TeleopRunState": ".diagnostics",
    "TeleopStateMachine": ".diagnostics",
    "IKSafetyResolver": ".ik_resolver",
    "JointLimitStep": ".joint_limits",
    "JointRateLimiter": ".joint_limits",
    "JointVelocityLimitProfile": ".joint_limits",
    "limit_joint_step": ".joint_limits",
    "load_joint_velocity_limit_profiles": ".joint_limits",
    "plan_limited_joint_path": ".joint_limits",
    "PinocchioIKProvider": ".pinocchio_provider",
    "IKSafetyProfile": ".profiles",
    "load_ik_safety_profiles": ".profiles",
    "ACTION_POSE_MODES": ".relative_action",
    "ACTION_POSE_MODE_ABSOLUTE": ".relative_action",
    "ACTION_POSE_MODE_RELATIVE": ".relative_action",
    "DEFAULT_ARM_POSE_SLICES": ".relative_action",
    "PoseChunkErrorMetrics": ".relative_action",
    "RelativeActionError": ".relative_action",
    "absolute_pose": ".relative_action",
    "compare_pose_chunks": ".relative_action",
    "convert_action_and_last_action_chunk": ".relative_action",
    "convert_arm_pose_chunk": ".relative_action",
    "convert_last_action_chunk": ".relative_action",
    "normalize_action_pose_mode": ".relative_action",
    "pose_matrix_to_xyzquat": ".relative_action",
    "relative_pose": ".relative_action",
    "xyzquat_to_pose_matrix": ".relative_action",
    "SeedExpansionProfile": ".seed_expansion",
    "SeedExpansionResult": ".seed_expansion",
    "SeedPerturbation": ".seed_expansion",
    "expand_seed_qpos": ".seed_expansion",
    "load_seed_expansion_profiles": ".seed_expansion",
    "BackendDispatchError": ".runtime",
    "BimanualRig": ".runtime",
    "CommandDispatchReceipt": ".runtime",
    "CompositeRig": ".runtime",
    "EffectorBackend": ".runtime",
    "EffectorCommand": ".runtime",
    "MotionCommand": ".runtime",
    "MotionObservation": ".runtime",
    "ObservationChannel": ".runtime",
    "ObservationSource": ".runtime",
    "Retargeter": ".runtime",
    "Rig": ".runtime",
    "RigDispatchError": ".runtime",
    "TeleopCommand": ".runtime",
    "TeleopObservation": ".runtime",
    "latest_channel_timestamp": ".runtime",
    "merge_teleop_commands": ".runtime",
    "MotionSafetyConfig": ".safety",
    "SafetyMonitor": ".safety",
    "SafetyViolation": ".safety",
    "TeleopSafetyConfig": ".safety",
    "IKCandidate": ".types",
    "IKResolveRequest": ".types",
    "IKResolveResult": ".types",
    "IKResolveStatus": ".types",
    "PoseTolerance": ".types",
    "SingularityReport": ".types",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
