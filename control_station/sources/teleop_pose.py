"""Compatibility exports for lower-owned passive TeleopSession sources.

New code should import ``teleop_runtime.action.passive_teleop`` directly.
Station retains this path while concrete Sim-HIL profiles migrate their
imports; no source production or numerical adaptation is implemented here.
"""

from teleop_runtime.action.passive_teleop import (
    AttachablePassiveTeleopComponentSource,
    PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1,
    PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1,
    PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1,
    PassiveTeleopComponentBindingReceipt,
    PassiveTeleopComponentClockBridge,
    PassiveTeleopComponentSessionSource,
    PassiveTeleopComponentSourceSpec,
    PassiveTeleopComponentSpec,
    PassiveTeleopComponentStep,
    PassiveTeleopPoseBindingReceipt,
    PassiveTeleopPoseSessionSource,
    PassiveTeleopPoseSourceSpec,
    PassiveTeleopPoseStep,
    build_passive_teleop_component_session_source,
    build_passive_teleop_pose_session_source,
)


__all__ = [
    "AttachablePassiveTeleopComponentSource",
    "PASSIVE_TELEOP_COMPONENT_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_GENERIC_SOURCE_CONTRACT_ID_V1",
    "PASSIVE_TELEOP_POSE_SOURCE_CONTRACT_ID_V1",
    "PassiveTeleopComponentBindingReceipt",
    "PassiveTeleopComponentClockBridge",
    "PassiveTeleopComponentSessionSource",
    "PassiveTeleopComponentSourceSpec",
    "PassiveTeleopComponentSpec",
    "PassiveTeleopComponentStep",
    "PassiveTeleopPoseBindingReceipt",
    "PassiveTeleopPoseSessionSource",
    "PassiveTeleopPoseSourceSpec",
    "PassiveTeleopPoseStep",
    "build_passive_teleop_component_session_source",
    "build_passive_teleop_pose_session_source",
]
