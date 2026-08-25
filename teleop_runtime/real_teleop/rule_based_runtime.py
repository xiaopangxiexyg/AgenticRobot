"""Lightweight rule-plan extension for the existing RealTeleop owner loop.

This module owns neither hardware lifecycle nor dispatch.  It claims the one
existing command-override slot, consumes the formal recorder's causal arm
feedback, and delegates every command to the normal RealTeleop safety path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.relative_action import pose_matrix_to_xyzquat
from robot_motion.replay import (
    CANONICAL_DESK_ACTION_FRAME,
    HARDWARE_ARM_BASE_ACTION_FRAME,
    ReplayTrajectoryGroup,
    RuleBasedActionSegment,
    RuleBasedBoundaryState,
    RuleBasedCollectionCursor,
    RuleBasedCollectionPlan,
    RuleBasedReplay,
    replay_action_pose_frame,
)

from ..collection.rule_based_adapter import (
    RuleBasedCollectionAdapter,
    RuleBasedCollectionEmission,
)
from ..command_override import TeleopCommandOverride
from .pose_frame_runtime import (
    RealTeleopPoseFrameActionResolver,
    resolve_runtime_pose_frame_contract,
)
from .runtime_extension import RealTeleopDatasetSampleAdmission


RULE_BASED_REAL_TELEOP_SCHEMA_V1 = "dexteleop.rule_based_real_teleop.v1"


@dataclass(frozen=True)
class RuleBasedRealTeleopDeployment:
    """Pre-device-open deployment evidence required by this extension."""

    active_sides: tuple[str, ...]
    collection_pose_frame_contract: PoseFrameContract
    tcp_pose_envelope_profile: str

    def __post_init__(self) -> None:
        sides = tuple(str(side) for side in self.active_sides)
        if set(sides) != {"left", "right"} or len(sides) != 2:
            raise ValueError("rule-based real control requires both active RM65 arms.")
        if not isinstance(self.collection_pose_frame_contract, PoseFrameContract):
            raise TypeError("collection_pose_frame_contract must be PoseFrameContract.")
        if self.collection_pose_frame_contract.is_raw_storage:
            raise ValueError(
                "rule-based real control requires canonical desk collection storage."
            )
        profile = str(self.tcp_pose_envelope_profile).strip()
        if not profile:
            raise ValueError("tcp_pose_envelope_profile must be non-empty.")
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "tcp_pose_envelope_profile", profile)


def preflight_rule_based_real_teleop_deployment(
    args: object,
) -> RuleBasedRealTeleopDeployment:
    """Fail closed on a conservative first real-robot deployment profile."""

    if set(getattr(args, "active_sides", ())) != {"left", "right"}:
        raise ValueError("rule-based real control requires --active-sides both.")
    if str(getattr(args, "arm_control_mode", "")) != "position":
        raise ValueError("rule-based real control currently requires position mode.")
    if bool(getattr(args, "disable_dual_arm_collision_avoidance", False)):
        raise ValueError("rule-based real control refuses disabled collision avoidance.")
    if not bool(getattr(args, "tcp_pose_envelope_active", False)):
        raise ValueError("rule-based real control requires an Active TCP pose envelope.")
    profile = str(getattr(args, "tcp_pose_envelope_profile", "") or "").strip()
    if not profile:
        raise ValueError("rule-based real control requires a TCP envelope profile.")
    if not bool(getattr(args, "position_force_guard", False)):
        raise ValueError("rule-based real control requires the position force guard.")
    if str(getattr(args, "position_force_guard_source", "tool")) != "tool":
        raise ValueError("rule-based real control requires tool-frame force guarding.")
    if bool(getattr(args, "no_collection", False)):
        raise ValueError("rule-based real control requires formal collection.")
    if any(
        bool(getattr(args, name, False))
        for name in (
            "position_admittance_active",
            "position_admittance_rotation_active",
            "position_admittance_combined_active",
        )
    ):
        raise ValueError(
            "the first rule-based real deployment refuses Active admittance overlays."
        )
    contract = getattr(args, "pose_frame_contract", None)
    return RuleBasedRealTeleopDeployment(
        active_sides=tuple(getattr(args, "active_sides", ())),
        collection_pose_frame_contract=contract,
        tcp_pose_envelope_profile=profile,
    )


@dataclass(frozen=True)
class RuleBasedRealTeleopConfig:
    """Only rule pacing and measured-arrival thresholds live here."""

    action_hz: float | None = None
    qpos_tolerance_rad: float = math.radians(2.0)
    terminal_translation_tolerance_m: float = 0.006
    terminal_rotation_tolerance_deg: float = 4.0
    settle_consecutive_samples: int = 3
    target_timeout_sec: float = 30.0

    def __post_init__(self) -> None:
        if self.action_hz is not None and (
            not np.isfinite(float(self.action_hz)) or float(self.action_hz) <= 0.0
        ):
            raise ValueError("action_hz must be finite and positive when provided.")
        for name in (
            "qpos_tolerance_rad",
            "terminal_translation_tolerance_m",
            "terminal_rotation_tolerance_deg",
            "target_timeout_sec",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if int(self.settle_consecutive_samples) <= 0:
            raise ValueError("settle_consecutive_samples must be positive.")


@dataclass(frozen=True)
class RuleBasedRealTeleopStatus:
    started: bool
    execution_begun: bool
    completed: bool
    held: bool
    waiting_reason: str
    step_id: str | None
    directive_kind: str | None
    sequence_index: int | None
    published_boundaries: tuple[str, ...] = field(default_factory=tuple)


class RuleBasedRealTeleopExtension:
    """Exclusive rule-plan source attached to one existing RealTeleop runtime."""

    def __init__(
        self,
        *,
        deployment: RuleBasedRealTeleopDeployment,
        plan: RuleBasedCollectionPlan,
        trajectory_groups: Mapping[str, ReplayTrajectoryGroup],
        config: RuleBasedRealTeleopConfig | None = None,
    ) -> None:
        if not isinstance(deployment, RuleBasedRealTeleopDeployment):
            raise TypeError("deployment must be RuleBasedRealTeleopDeployment.")
        if not isinstance(plan, RuleBasedCollectionPlan):
            raise TypeError("plan must be RuleBasedCollectionPlan.")
        self.deployment = deployment
        self.plan = plan
        self.trajectory_groups = dict(trajectory_groups)
        self.config = config or RuleBasedRealTeleopConfig()
        self._validate_plan()
        self.runtime_pose_frame_contract = resolve_runtime_pose_frame_contract(
            deployment.collection_pose_frame_contract
        )
        self._boundaries: dict[str, RuleBasedBoundaryState] = {}
        self._boundary_frame_by_step = self._relative_boundary_frames()
        cursor = RuleBasedCollectionCursor(
            plan,
            self.trajectory_groups,
            boundary_provider=self._boundary_for,
        )
        self.adapter = RuleBasedCollectionAdapter(
            cursor,
            resolver=RealTeleopPoseFrameActionResolver(
                collection_contract=deployment.collection_pose_frame_contract,
                runtime_contract=self.runtime_pose_frame_contract,
            ),
        )
        rates = [
            float(group.action_hz)
            for group in self.trajectory_groups.values()
            if np.isfinite(float(group.action_hz)) and float(group.action_hz) > 0.0
        ]
        self.action_hz = float(
            self.config.action_hz
            if self.config.action_hz is not None
            else np.median(rates) if rates else 15.0
        )
        self._components: object | None = None
        self._clutch_input: object | None = None
        self._dataset_recorder_provider: Callable[[], object | None] | None = None
        self._recording_active: Callable[[], bool] | None = None
        self._current: RuleBasedCollectionEmission | None = None
        self._frame_emission: RuleBasedCollectionEmission | None = None
        self._target_wait_started_at: float | None = None
        self._next_due_at: float | None = None
        self._current_dispatched = False
        self._settled_samples = 0
        self._started = False
        self._execution_begun = False
        self._completed = False
        self._held = False
        self._closed = False
        self._waiting_reason = "not_started"

    def _validate_plan(self) -> None:
        steps = tuple(self.plan.steps)
        for index, step in enumerate(steps):
            if step.capture.stream_ids:
                raise ValueError(
                    "RealTeleop formal samples are atomic; per-stream rule capture "
                    f"is unsupported for step {step.step_id!r}."
                )
            if isinstance(step, RuleBasedReplay) and step.relative_to is not None:
                if index == 0 or not isinstance(steps[index - 1], RuleBasedActionSegment):
                    raise ValueError(
                        "first real relative replay requires an immediately preceding "
                        "bounded runtime action segment."
                    )
                if step.relative_to != steps[index - 1].step_id:
                    raise ValueError(
                        "relative_to must name the immediately preceding runtime "
                        "segment step_id for measured-boundary publication."
                    )

    def _relative_boundary_frames(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for step in self.plan.steps:
            if not isinstance(step, RuleBasedReplay) or step.relative_to is None:
                continue
            frame = replay_action_pose_frame(self.trajectory_groups[step.group_key])
            previous = result.get(step.relative_to)
            if previous is not None and previous != frame:
                raise ValueError(
                    f"boundary {step.relative_to!r} is requested in conflicting frames."
                )
            result[step.relative_to] = frame
        return result

    def attach_components(self, components: object) -> object:
        if set(getattr(components, "active_sides", ())) != {"left", "right"}:
            raise RuntimeError("rule-based runtime requires coupled dual-arm components.")
        if not bool(getattr(components, "collision_guard_requested", False)):
            raise RuntimeError("rule-based runtime requires collision guarding.")
        retargeter = getattr(components, "dual_arm_retargeter", None)
        if retargeter is None:
            raise RuntimeError("rule-based runtime requires the coupled target solver.")
        target_filter = getattr(retargeter, "pre_ik_target_safety_filter", None)
        if target_filter is None or not bool(
            getattr(target_filter, "apply_to_target", False)
        ):
            raise RuntimeError("rule-based runtime requires an Active final TCP envelope.")
        profile = getattr(target_filter, "profile", None)
        if set(getattr(profile, "enabled_sides", ())) != {"left", "right"}:
            raise RuntimeError("Active TCP envelope must cover both arms.")
        session = components.session
        if session.command_override_provider is not None:
            raise RuntimeError("RealTeleop already has a command override provider.")
        if session.human_command_selection_provider is not None:
            raise RuntimeError("rule-based runtime cannot share a source selector.")
        session.command_override_provider = self.command_for_observation
        self._components = components
        return components

    def bind_runtime(
        self,
        *,
        components: object,
        clutch_input: object,
        camera_sources: Mapping[str, object],
        dataset_recorder: object | None,
        dataset_recorder_provider: Callable[[], object | None],
        episode_recorder: object,
        control_ui: object | None,
        recording_active: Callable[[], bool],
        pause_recording: Callable[..., bool],
    ) -> None:
        del camera_sources, dataset_recorder, episode_recorder, control_ui, pause_recording
        if components is not self._components:
            raise RuntimeError("RealTeleop did not retain rule-based components.")
        if not callable(dataset_recorder_provider) or not callable(recording_active):
            raise TypeError("rule-based runtime requires recorder state providers.")
        self._clutch_input = clutch_input
        self._dataset_recorder_provider = dataset_recorder_provider
        self._recording_active = recording_active

    def start(self, *, components: object) -> None:
        if components is not self._components or self._dataset_recorder_provider is None:
            raise RuntimeError("rule-based runtime is not fully bound.")
        self._started = True
        self._waiting_reason = "waiting_for_formal_episode"

    def after_control_input_poll(self, *, now: float) -> None:
        del now
        self._frame_emission = None

    def command_for_observation(self, observation: object, now: float):
        if self._closed:
            raise RuntimeError("rule-based runtime is closed.")
        if not self._started:
            return TeleopCommandOverride(None)
        ready, reason, snapshots = self._runtime_ready(float(now))
        if not ready:
            self._waiting_reason = reason
            if self._execution_begun:
                raise RuntimeError(
                    "rule-based execution lost formal collection readiness: " + reason
                )
            return TeleopCommandOverride(None)
        self._waiting_reason = ""
        if self._current is not None and self._advance_due(float(now)):
            self._clear_current()
        if self._current is None and not self._completed:
            self._load_next(observation, float(now), snapshots)
        if self._completed:
            return TeleopCommandOverride(None)
        assert self._current is not None
        self._frame_emission = self._current
        self._execution_begun = True
        return TeleopCommandOverride(self._current.command)

    def _runtime_ready(
        self,
        now: float,
    ) -> tuple[bool, str, dict[str, dict[str, Any]]]:
        provider = self._dataset_recorder_provider
        if provider is None:
            return False, "runtime_not_bound", {}
        recorder = provider()
        if recorder is None:
            return False, "formal_recorder_unavailable", {}
        if self._recording_active is None or not self._recording_active():
            return False, "formal_episode_not_active", {}
        if not bool(getattr(recorder, "is_recording", False)):
            return False, "formal_writer_not_recording", {}
        if not bool(getattr(recorder, "can_record_samples", False)):
            return False, "formal_writer_paused", {}
        poller = getattr(recorder, "arm_state_poller", None)
        if poller is None:
            return False, "causal_arm_feedback_poller_unavailable", {}
        readiness = poller.readiness_by_side(
            required_sides=self.deployment.active_sides,
            sample_timestamp=now,
        )
        blocked = {side: value for side, value in readiness.items() if value}
        if blocked:
            details = ",".join(f"{side}={value}" for side, value in sorted(blocked.items()))
            return False, "causal_arm_feedback_not_ready:" + details, {}
        snapshots = poller.latest_at_or_before(now)
        missing = sorted(set(self.deployment.active_sides) - set(snapshots))
        if missing:
            return False, "causal_arm_feedback_missing:" + ",".join(missing), {}
        return True, "", snapshots

    def _load_next(
        self,
        observation: object,
        now: float,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> None:
        try:
            emission = self.adapter.next_runtime(
                observation,
                context={
                    "actual_arm_state_by_side": snapshots,
                    "runtime_pose_frame_contract": self.runtime_pose_frame_contract,
                },
            )
        except StopIteration:
            self._completed = True
            self._waiting_reason = "completed"
            return
        emission.command.metadata.update(
            {
                "rule_based_real_teleop_schema": RULE_BASED_REAL_TELEOP_SCHEMA_V1,
                "rule_based_step_id": emission.directive.step_id,
                "rule_based_directive_kind": emission.directive.kind,
                "rule_based_sequence_index": int(emission.directive.sequence_index),
                "rule_based_capture_enabled": bool(emission.recordable),
            }
        )
        self._current = emission
        self._target_wait_started_at = None
        self._current_dispatched = False
        self._settled_samples = 0
        self._next_due_at = now + self._emission_period(emission)

    def _emission_period(self, emission: RuleBasedCollectionEmission) -> float:
        directive = emission.directive
        if directive.kind == "replay" and directive.group_key is not None:
            group = self.trajectory_groups[directive.group_key]
            index = int(directive.replay_frame_index or 0)
            if index + 1 < int(group.frame_count):
                interval = group.relative_time_sec(index + 1) - group.relative_time_sec(index)
                if np.isfinite(interval) and interval > 0.0:
                    return float(interval)
            return 1.0 / float(group.action_hz)
        return 1.0 / self.action_hz

    def _advance_due(self, now: float) -> bool:
        assert self._current is not None
        if self._current.directive.kind == "initialize":
            return False
        if self._current.directive.segment_terminal:
            return False
        return bool(
            self._current_dispatched
            and self._next_due_at is not None
            and now >= self._next_due_at
        )

    def after_session_step(self, *, step_result: object, now: float) -> None:
        emission = self._frame_emission
        if emission is not None and bool(getattr(step_result, "command_dispatched", False)):
            self._current_dispatched = True
            if emission.directive.kind == "initialize" or emission.directive.segment_terminal:
                if self._target_wait_started_at is None:
                    self._target_wait_started_at = float(now)
                self._observe_target_arrival(emission, float(now))
        if self._completed:
            self._establish_hold("rule-based plan completed")

    def _observe_target_arrival(
        self,
        emission: RuleBasedCollectionEmission,
        now: float,
    ) -> None:
        ready, reason, snapshots = self._runtime_ready(now)
        if not ready:
            raise RuntimeError("target arrival feedback unavailable: " + reason)
        reached = self._target_reached(emission, snapshots)
        self._settled_samples = self._settled_samples + 1 if reached else 0
        if self._settled_samples >= int(self.config.settle_consecutive_samples):
            if emission.directive.segment_terminal:
                self._publish_boundary(emission.directive.step_id, snapshots, now)
            self._clear_current()
            return
        if (
            self._target_wait_started_at is not None
            and now - self._target_wait_started_at > float(self.config.target_timeout_sec)
        ):
            raise TimeoutError(
                f"rule-based target arrival timed out at step {emission.directive.step_id!r}."
            )

    def _target_reached(
        self,
        emission: RuleBasedCollectionEmission,
        snapshots: Mapping[str, Mapping[str, Any]],
    ) -> bool:
        directive = emission.directive
        if directive.kind == "initialize":
            for side, target in directive.qpos_by_side.items():
                actual = np.asarray(
                    snapshots[side].get("actual_qpos"), dtype=float
                ).reshape(-1)
                if actual.shape != (6,) or not np.all(np.isfinite(actual)):
                    return False
                if float(np.max(np.abs(actual - np.asarray(target, dtype=float)))) > float(
                    self.config.qpos_tolerance_rad
                ):
                    return False
            return True
        for side in emission.action.arm_qpos_by_side:
            target = np.asarray(emission.action.arm_qpos_by_side[side], dtype=float)
            actual = np.asarray(
                snapshots[side].get("actual_qpos"), dtype=float
            ).reshape(-1)
            if actual.shape != (6,) or float(np.max(np.abs(actual - target))) > float(
                self.config.qpos_tolerance_rad
            ):
                return False
        for side in emission.action.arm_pose_by_side:
            command = emission.command.commands.get(f"{side}_arm")
            if command is None:
                return False
            target = np.asarray(command.value, dtype=float).reshape(4, 4)
            actual = np.asarray(
                snapshots[side].get("actual_tcp_pose"), dtype=float
            ).reshape(4, 4)
            if float(np.linalg.norm(actual[:3, 3] - target[:3, 3])) > float(
                self.config.terminal_translation_tolerance_m
            ):
                return False
            relative_rotation = actual[:3, :3].T @ target[:3, :3]
            cosine = float(
                np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
            )
            if math.degrees(math.acos(cosine)) > float(
                self.config.terminal_rotation_tolerance_deg
            ):
                return False
        return True

    def _publish_boundary(
        self,
        step_id: str,
        snapshots: Mapping[str, Mapping[str, Any]],
        now: float,
    ) -> None:
        frame = self._boundary_frame_by_step.get(step_id)
        if frame is None:
            return
        poses: dict[str, np.ndarray] = {}
        frames: dict[str, str] = {}
        for side in self.deployment.active_sides:
            raw_pose = pose_matrix_to_xyzquat(snapshots[side]["actual_tcp_pose"])
            if frame == CANONICAL_DESK_ACTION_FRAME:
                poses[side] = (
                    self.runtime_pose_frame_contract.storage_pose_vector_to_canonical_desk(
                        raw_pose,
                        side=side,
                    )
                )
                frames[side] = "canonical_desk"
            elif frame == HARDWARE_ARM_BASE_ACTION_FRAME:
                poses[side] = raw_pose
                frames[side] = f"{side}_arm_base"
            else:
                raise RuntimeError(f"unsupported relative boundary frame {frame!r}.")
        self._boundaries[step_id] = RuleBasedBoundaryState(
            tcp_pose_by_side=poses,
            pose_frame_by_side=frames,
            source="real_teleop_causal_actual_tcp",
            timestamp=now,
            metadata={
                "runtime_pose_frame_contract_hash": (
                    self.runtime_pose_frame_contract.contract_hash
                ),
                "causal_snapshot_contract": (
                    "source_timestamp<=T0_and_available_timestamp<=T0"
                ),
            },
        )

    def _boundary_for(self, boundary_ref: str) -> RuleBasedBoundaryState:
        try:
            return self._boundaries[str(boundary_ref)]
        except KeyError as exc:
            raise RuntimeError(
                f"measured rule-based boundary {boundary_ref!r} is not published."
            ) from exc

    def _clear_current(self) -> None:
        self._current = None
        self._target_wait_started_at = None
        self._next_due_at = None
        self._current_dispatched = False
        self._settled_samples = 0

    def dataset_sample_admission(
        self,
        *,
        step_result: object,
        now: float,
    ) -> RealTeleopDatasetSampleAdmission:
        del now
        emission = self._frame_emission
        allowed = bool(
            emission is not None
            and emission.recordable
            and getattr(step_result, "command_dispatched", False)
        )
        if emission is None:
            return RealTeleopDatasetSampleAdmission(allowed=False)
        directive = emission.directive
        return RealTeleopDatasetSampleAdmission(
            allowed=allowed,
            metadata={
                "rule_based_real_teleop_schema": RULE_BASED_REAL_TELEOP_SCHEMA_V1,
                "rule_based_step_id": directive.step_id,
                "rule_based_directive_kind": directive.kind,
                "rule_based_sequence_index": int(directive.sequence_index),
                "rule_based_capture_enabled": bool(emission.recordable),
                "rule_based_segment_terminal": bool(directive.segment_terminal),
                "rule_based_segment_stop_reason": directive.segment_stop_reason,
            },
        )

    def _establish_hold(self, reason: str) -> None:
        if self._held or self._components is None:
            return
        self._components.session.hold_hardware(str(reason))
        request_hold = getattr(self._clutch_input, "request_hold", None)
        if callable(request_hold):
            request_hold(reason="rule_based_runtime_hold")
        self._held = True

    def close_before_session(
        self,
        *,
        components: object,
        terminal_error: BaseException | None,
    ) -> None:
        del terminal_error
        if components is self._components and self._started:
            self._establish_hold("rule-based runtime closing")
        self._closed = True

    def status(self) -> RuleBasedRealTeleopStatus:
        directive = None if self._current is None else self._current.directive
        return RuleBasedRealTeleopStatus(
            started=self._started,
            execution_begun=self._execution_begun,
            completed=self._completed,
            held=self._held,
            waiting_reason=self._waiting_reason,
            step_id=None if directive is None else directive.step_id,
            directive_kind=None if directive is None else directive.kind,
            sequence_index=None if directive is None else directive.sequence_index,
            published_boundaries=tuple(sorted(self._boundaries)),
        )


__all__ = [
    "RULE_BASED_REAL_TELEOP_SCHEMA_V1",
    "RuleBasedRealTeleopConfig",
    "RuleBasedRealTeleopDeployment",
    "RuleBasedRealTeleopExtension",
    "RuleBasedRealTeleopStatus",
    "preflight_rule_based_real_teleop_deployment",
]
