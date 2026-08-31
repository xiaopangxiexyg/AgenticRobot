"""Canonical absolute-pose recovery planning for teleop runtimes.

Callers remain responsible for reading their authoritative arm state before
building or advancing a plan. This module owns the shared plan shape,
target-resolution semantics, interpolation, and command construction only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schemas import TeleopCommand, TeleopObservation, merge_teleop_commands
from .transforms import interpolate_pose


@dataclass
class RecoveryPlan:
    mode: str
    started_at: float
    duration_sec: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


def validate_absolute_recovery_control_mode(control_mode: str) -> None:
    if str(control_mode) != "absolute":
        raise ValueError(
            "delta_clutch has been hard-archived; recovery only supports 'absolute'."
        )


def component_side_retargeter(components: object, side: str) -> Any:
    return (
        getattr(components, "left_retargeter")
        if side == "left"
        else getattr(components, "right_retargeter")
    )


def component_tcp_pose_desk(components: object, *, side: str) -> np.ndarray:
    session = getattr(components, "session")
    calibration = session.calibration
    assert calibration is not None
    backend = (
        getattr(components, "left_preview_backend")
        if side == "left"
        else getattr(components, "right_preview_backend")
    )
    pose_base = backend.compute_tcp_pose()
    return calibration.get_arm_for_side(side).T_D_B @ pose_base


def build_absolute_recovery_plan(
    observation: TeleopObservation,
    *,
    calibration: Any,
    active_sides: Sequence[str],
    dual_arm_retargeter: Any | None,
    side_retargeter: Callable[[str], Any],
    current_tcp_pose_desk: Callable[[str], np.ndarray],
    now: float,
    ramp_sec: float,
    channel_name_for_side: Callable[[str], str] | None = None,
) -> RecoveryPlan:
    """Resolve one immutable recovery target set from the current observation."""

    if dual_arm_retargeter is not None:
        context = dual_arm_retargeter.build_target_context(observation, calibration)
        return RecoveryPlan(
            mode="absolute_ramp_dual",
            started_at=now,
            duration_sec=float(ramp_sec),
            payload={
                "context": context,
                "start_targets": dual_arm_retargeter.current_world_targets(),
                "end_targets": {
                    "left": context.left_target.copy(),
                    "right": context.right_target.copy(),
                },
            },
        )

    end_resolutions: dict[str, dict[str, object]] = {}
    for side in active_sides:
        resolution = side_retargeter(side).resolve_absolute_target_from_observation(
            observation,
            calibration,
        )
        if resolution is None:
            channel_name = (
                f"{side}_wrist_pose"
                if channel_name_for_side is None
                else channel_name_for_side(side)
            )
            raise RuntimeError(
                f"Absolute recovery requires active wrist channel '{channel_name}' to be valid."
            )
        end_resolutions[side] = resolution

    return RecoveryPlan(
        mode="absolute_ramp_independent",
        started_at=now,
        duration_sec=float(ramp_sec),
        payload={
            "start_targets": {
                side: current_tcp_pose_desk(side) for side in active_sides
            },
            "end_resolutions": end_resolutions,
        },
    )


def advance_absolute_recovery_plan(
    observation: TeleopObservation,
    plan: RecoveryPlan,
    *,
    calibration: Any,
    dual_arm_retargeter: Any | None,
    side_retargeter: Callable[[str], Any],
    now: float,
) -> tuple[TeleopCommand, bool]:
    """Advance one absolute recovery plan without owning state synchronization."""

    alpha = 1.0
    if plan.duration_sec > 1e-6:
        alpha = float(
            np.clip((now - plan.started_at) / plan.duration_sec, 0.0, 1.0)
        )

    if plan.mode == "absolute_ramp_dual":
        if dual_arm_retargeter is None:
            raise RuntimeError(
                "Dual-arm recovery plan is active but dual retargeter is unavailable."
            )
        context = plan.payload["context"]
        start_targets = plan.payload["start_targets"]
        end_targets = plan.payload["end_targets"]
        command = dual_arm_retargeter.build_command_for_context(
            context,
            targets=dual_arm_retargeter.interpolate_world_targets(
                start_targets,
                end_targets,
                alpha,
            ),
        )
        command.timestamp = observation.timestamp
        return command, alpha >= 1.0

    if plan.mode == "absolute_ramp_independent":
        start_targets = plan.payload["start_targets"]
        end_resolutions = plan.payload["end_resolutions"]
        commands: list[TeleopCommand] = []
        for side, resolution in end_resolutions.items():
            target_pose = interpolate_pose(
                start_targets[side],
                resolution["target_pose_desk"],
                alpha,
            )
            commands.append(
                side_retargeter(side).build_command_from_target_pose_desk(
                    target_pose_desk=target_pose,
                    observation=observation,
                    calibration=calibration,
                    pose_operator=resolution.get("pose_operator"),
                )
            )
        command = merge_teleop_commands(tuple(commands))
        command.timestamp = observation.timestamp
        return command, alpha >= 1.0

    raise ValueError(f"Unsupported recovery plan mode: {plan.mode}")


__all__ = [
    "RecoveryPlan",
    "advance_absolute_recovery_plan",
    "build_absolute_recovery_plan",
    "component_side_retargeter",
    "component_tcp_pose_desk",
    "validate_absolute_recovery_control_mode",
]
