"""Immutable, non-authoritative contracts for simulation-first real replay."""

from .contracts import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowReplayAdmissionReceipt,
    ShadowReplayAdmissionStatus,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
    evaluate_shadow_replay_admission,
)

__all__ = [
    "AuthoritativeRobotStateSnapshot",
    "RegisteredJointTrajectoryPoint",
    "RegisteredShadowJointTrajectory",
    "ShadowExecutionIdentity",
    "ShadowReplayAdmissionReceipt",
    "ShadowReplayAdmissionStatus",
    "ShadowSimulationReceipt",
    "ShadowSimulationStatus",
    "evaluate_shadow_replay_admission",
]
