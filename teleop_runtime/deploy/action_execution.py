"""Deployment-owned routing for one already-prepared canonical action.

This module selects the existing direct or position-reference execution path.
It does not own policy queues, canonical safety, position-reference advancement,
backend lifecycle, or hardware authority; those remain with the collaborators
passed to :class:`PreparedActionExecutor`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from robot_motion.deploy.canonical_action import CanonicalDeployActionResult
from robot_motion.deploy.observation import DeploymentObservation
from robot_motion.position_control import PositionSourcePlanCut

from ..interfaces import EffectorBackend
from ..position_control import PositionDeployExecutor, PositionDeployExecutorFault
from .action_dispatch import command_effector_kind_subset
from .dispatch import (
    dispatch_teleop_command,
    require_arm_backends_ready_for_position_dispatch,
)
from .rm65_legacy_worker_binding import RM65LegacyWorkerExecutionBinding


@dataclass(frozen=True)
class PreparedActionExecution:
    accepted: bool
    mode: str
    dispatch_attempted: bool
    dispatch_completed: bool
    asynchronous: bool = False
    position_source_index: int | None = None


class PreparedActionExecutor:
    """Route one canonical action through the configured deployment executor."""

    def __init__(
        self,
        *,
        position_deploy_executor: PositionDeployExecutor | None,
        send_actions: bool,
        arm_backends: Mapping[str, EffectorBackend],
        hand_backends: Mapping[str, EffectorBackend],
        active_sides: tuple[str, ...],
        position_worker_binding: RM65LegacyWorkerExecutionBinding | None,
        submit_legacy_owner_state: Callable[..., Any],
    ) -> None:
        self._position_deploy_executor = position_deploy_executor
        self._send_actions = bool(send_actions)
        self._arm_backends = arm_backends
        self._hand_backends = hand_backends
        self._active_sides = tuple(str(side) for side in active_sides)
        self._position_worker_binding = position_worker_binding
        self._submit_legacy_owner_state = submit_legacy_owner_state

    def execute(
        self,
        canonical_action: CanonicalDeployActionResult,
        observation: DeploymentObservation,
        *,
        policy_name: str,
        source_decision_id: str | None = None,
        source_reference_segment_id: str | None = None,
        lookahead_canonical_action: CanonicalDeployActionResult | None = None,
        lookahead_observation: DeploymentObservation | None = None,
        second_lookahead_canonical_action: CanonicalDeployActionResult | None = None,
        second_lookahead_observation: DeploymentObservation | None = None,
        third_lookahead_canonical_action: CanonicalDeployActionResult | None = None,
        third_lookahead_observation: DeploymentObservation | None = None,
        source_plan_cut: PositionSourcePlanCut | None = None,
        phase_timing: dict[str, Any] | None = None,
    ) -> PreparedActionExecution:
        command = getattr(canonical_action, "command", None)
        if command is None or not command.commands:
            return PreparedActionExecution(
                accepted=False,
                mode="no_command",
                dispatch_attempted=False,
                dispatch_completed=False,
            )
        position_deploy_executor = self._position_deploy_executor
        if position_deploy_executor is None:
            direct_dispatch_started = time.monotonic()
            if self._send_actions:
                dispatch_teleop_command(
                    command,
                    arm_backends=self._arm_backends,
                    hand_backends=self._hand_backends,
                )
            if phase_timing is not None:
                phase_timing["direct_dispatch_ms"] = max(
                    0.0,
                    time.monotonic() - direct_dispatch_started,
                ) * 1000.0
            return PreparedActionExecution(
                accepted=True,
                mode="direct" if self._send_actions else "dry_run",
                dispatch_attempted=bool(self._send_actions),
                dispatch_completed=bool(self._send_actions),
            )

        arm_command = command_effector_kind_subset(command, kind="arm")
        hand_command = command_effector_kind_subset(command, kind="hand")
        lookahead_hand_command = None
        if lookahead_canonical_action is not None:
            lookahead_hand_command = command_effector_kind_subset(
                getattr(lookahead_canonical_action, "command", None),
                kind="hand",
            )
        if arm_command is not None:
            execution_work_item = None
            lookahead_execution_work_item = None
            position_worker_binding = self._position_worker_binding
            if position_worker_binding is not None:
                if not source_decision_id or not source_reference_segment_id:
                    raise RuntimeError(
                        "epoch-mediated position submission requires exact source lineage"
                    )
                self._submit_legacy_owner_state(
                    lifecycle_state="running",
                    execution_gate="ready",
                    reason="legacy_reference_work_admission",
                    source_event_ids=(
                        str(source_decision_id),
                        str(source_reference_segment_id),
                    ),
                )
                execution_work_item = (
                    position_worker_binding.context.create_reference_segment_work_item(
                        source_decision_id=str(source_decision_id),
                        source_reference_segment_id=str(source_reference_segment_id),
                    )
                )
                if lookahead_canonical_action is not None:
                    lookahead_execution_work_item = (
                        position_worker_binding.context.create_reference_segment_work_item(
                            source_decision_id=str(source_decision_id),
                            source_reference_segment_id=(
                                str(source_reference_segment_id) + ":lookahead"
                            ),
                        )
                    )
            position_submit_started = time.monotonic()
            try:
                source_index = position_deploy_executor.submit(
                    canonical_action.filtered,
                    observation,
                    policy_name=str(policy_name),
                    discrete_command=hand_command,
                    lookahead_filtered=(
                        None
                        if lookahead_canonical_action is None
                        else lookahead_canonical_action.filtered
                    ),
                    lookahead_observation=lookahead_observation,
                    lookahead_discrete_command=lookahead_hand_command,
                    second_lookahead_filtered=(
                        None
                        if second_lookahead_canonical_action is None
                        else second_lookahead_canonical_action.filtered
                    ),
                    second_lookahead_observation=second_lookahead_observation,
                    third_lookahead_filtered=(
                        None
                        if third_lookahead_canonical_action is None
                        else third_lookahead_canonical_action.filtered
                    ),
                    third_lookahead_observation=third_lookahead_observation,
                    execution_work_item=execution_work_item,
                    lookahead_execution_work_item=lookahead_execution_work_item,
                    source_plan_cut=source_plan_cut,
                )
            except PositionDeployExecutorFault:
                fault_stage = str(
                    position_deploy_executor.snapshot().get("fault_stage") or ""
                )
                if fault_stage in {
                    "safety_gate",
                    "15hz_keyframe_safety",
                    "reference_anchor_safety",
                    "target_pose_envelope",
                }:
                    return PreparedActionExecution(
                        accepted=False,
                        mode="position_safety_fault_latched",
                        dispatch_attempted=False,
                        dispatch_completed=False,
                    )
                raise
            finally:
                if phase_timing is not None:
                    phase_timing["position_submit_ms"] = max(
                        0.0,
                        time.monotonic() - position_submit_started,
                    ) * 1000.0
            return PreparedActionExecution(
                accepted=True,
                mode=(
                    "position_async_queue"
                    if self._send_actions
                    else "position_dry_run_queue"
                ),
                dispatch_attempted=False,
                dispatch_completed=False,
                asynchronous=True,
                position_source_index=int(source_index),
            )
        if self._send_actions and hand_command is not None:
            # A hand-only command has no arm keyframe whose causal reference
            # clock could carry it. Mixed arm+hand actions are queued together
            # above so both become effective on the same logical reference tick.
            try:
                require_arm_backends_ready_for_position_dispatch(
                    self._arm_backends,
                    require_background_target=False,
                    stage="pre_direct_hand_only_dispatch",
                    required_sides=self._active_sides,
                )
                dispatch_teleop_command(
                    hand_command,
                    arm_backends={},
                    hand_backends=self._hand_backends,
                )
            except Exception:
                position_deploy_executor.clear()
                raise
            return PreparedActionExecution(
                accepted=True,
                mode="position_hand_only_direct",
                dispatch_attempted=True,
                dispatch_completed=True,
            )
        return PreparedActionExecution(
            accepted=bool(hand_command is not None),
            mode="position_hand_only_dry_run",
            dispatch_attempted=False,
            dispatch_completed=False,
        )


__all__ = ["PreparedActionExecution", "PreparedActionExecutor"]
