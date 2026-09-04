from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.action import HandActionSpec
from robot_motion.deploy.schema import (
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ARM_JOINT_DOF,
    HAND_DOF,
)
from robot_motion.replay.trajectory import (
    QPOS_INIT_AUTO,
    ReplayTrajectoryGroup,
    normalize_qpos_init_source,
    select_replay_qpos_init_targets,
)
from robot_motion.rm65.arm_init.qpos_init import DEFAULT_QPOS_TARGET_PATH
from robot_motion.rm65.arm_init.schema import load_qpos_target
from robot_motion.runtime import EffectorCommand

from ..replay.arm_init.qpos_execute import (
    DEFAULT_QPOS_INIT_BRANCH_SEARCH,
    DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES,
    DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_SCENE,
    DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M,
    DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG,
    DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG,
    DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG,
    DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG,
    _require_plan_completed,
    build_rm65_qpos_init_plan,
    normalize_qpos_init_max_angvel,
)
from .replay import (
    load_contract_replay_actions_from_parquet,
    load_replay_arm_joint_targets_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    load_replay_arm_qpos_audit_from_parquet,
    resolve_replay_parquet_path,
)


# Migration note:
# This module is real-launcher staging code. The dataset-frame target selection
# and qpos init planning should move to robot_motion/deploy once deploy common
# code is split out. Backend send_command dispatch and live hardware qpos reads
# should remain in teleop_runtime.


_QPOS_INIT_FRESH_HARDWARE_MAX_AGE_SEC = 0.25
_DMTAC_COMMAND_CONFIRM_TOLERANCE = 0.002
_DMTAC_ACCEPTED_SEND_STATUSES = frozenset(
    {"sent", "skipped_threshold", "rate_limited"}
)


@dataclass(frozen=True)
class DeployPolicyQposInitPlan:
    source_path: str
    source_episode_index: int
    frame_index: int
    action: np.ndarray
    target_qpos_by_side: dict[str, np.ndarray]
    target_sources_by_side: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeployPolicyQposInitSettings:
    # A 0.01 rad completion gate is too strict for the compliant impedance
    # backend.  Keep the same completion contract across deploy/replay paths.
    qpos_reached_tolerance_rad: float = 0.1
    move_to_start_timeout_s: float = 30.0
    qpos_init_dt_sec: float = 0.05
    qpos_init_joint_velocity_profile: str = "rm65_init_slow"
    qpos_init_max_angvel_rad_s: tuple[float, ...] | None = None
    qpos_init_max_steps_per_side: int = 10000
    qpos_init_collision_check: bool = True
    qpos_init_collision_scene_profile: str = DEFAULT_QPOS_INIT_COLLISION_SCENE
    qpos_init_collision_slow_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M
    qpos_init_collision_hold_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M
    qpos_init_collision_stop_distance_m: float = DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M
    qpos_init_max_joint_abs_deg: float = DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG
    qpos_init_joint_slow_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG
    qpos_init_joint_hold_margin_deg: float = DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG
    qpos_init_single_arm_max_qpos_delta_deg: float = DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG
    qpos_init_branch_search: bool = DEFAULT_QPOS_INIT_BRANCH_SEARCH
    qpos_init_branch_search_max_branches: int = DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES
    qpos_init_branch_neutral_qpos_path: str | Path | None = DEFAULT_QPOS_TARGET_PATH
    initialize_hands: bool = True
    hand_reached_tolerance: float = 0.05
    dmtac_hand_reached_tolerance: float = 0.18
    hand_state_max_age_sec: float = 0.5
    hand_progress_timeout_sec: float = 2.0
    hand_progress_epsilon: float = 0.02
    hand_reconnect_attempts: int = 1
    # Observation-only RTC still validates the selected dataset frame, current
    # qpos, branch search, interpolation and collision checks.  It must not
    # pretend that the resulting trajectory was executed, however, because
    # ``send_actions`` is false.  This explicit mode releases policy inference
    # from a fresh live-observation cut after planning instead of waiting for a
    # target pose that a no-action run is intentionally unable to reach.
    validate_plan_only: bool = False


@dataclass(frozen=True)
class PolicyQposInitStepResult:
    # Qpos init never admits a Cartesian action. In particular, plan.action
    # may be stored in canonical_desk while the live deployment observation is
    # runtime_raw, so it must not become the runtime safety/reference anchor.
    last_action: np.ndarray | None = None
    completed: bool = False
    requires_fresh_observation: bool = False
    plan_only: bool = False
    message: str = ""


@dataclass(frozen=True)
class _HandInitEvaluation:
    reached: bool
    status_text: str
    reconnect_sides: tuple[str, ...] = ()
    exhausted_sides: tuple[str, ...] = ()


def _load_policy_qpos_init_actions(
    source_path: Path,
    *,
    resolved_metadata: Mapping[str, Any],
    action_column: str,
    frame_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one display/hold action without assuming a legacy 26D source row.

    Qpos target selection below uses only audited joint-state columns.  The
    action retained on the init plan is diagnostic metadata only and must not
    become the runtime ``last_action`` reference. It must nevertheless remain
    a canonical 26D deployment action. v2.2
    datasets store a compact layout-bound action (16D for dual DMTAC), which
    must take the same explicit bridge used by dataset replay.
    """

    actions, action_metadata, _pose_contract, is_typed = (
        load_contract_replay_actions_from_parquet(
            source_path,
            resolved_metadata=resolved_metadata,
            action_column=action_column,
            start_index=frame_index,
            end_index=frame_index + 1,
        )
    )
    return actions, {
        "action_encoding": (
            "typed_layout_to_legacy_q6_bridge" if is_typed else "legacy_26d"
        ),
        "policy_io_layout_hash": action_metadata.get("policy_io_layout_hash"),
        "action_metadata": dict(action_metadata),
    }


def load_deploy_policy_qpos_init_plan(
    *,
    dataset_root: str | Path | None,
    parquet_path: str | Path | None,
    episode_index: int,
    frame_index: int,
    action_column: str,
    active_sides: Sequence[str],
    qpos_init_source: str = QPOS_INIT_AUTO,
) -> DeployPolicyQposInitPlan | None:
    if not dataset_root and not parquet_path:
        return None
    active = tuple(str(side).strip().lower() for side in active_sides)
    if not active:
        raise ValueError("policy qpos init requires at least one active side.")
    source_path, resolved_metadata = resolve_replay_parquet_path(
        dataset_root=dataset_root,
        parquet_path=parquet_path,
        episode_index=int(episode_index),
    )
    frame = int(frame_index)
    if frame < 0:
        raise ValueError("--policy-qpos-init-frame must be >= 0.")
    actions, action_metadata = _load_policy_qpos_init_actions(
        source_path,
        resolved_metadata=resolved_metadata,
        action_column=str(action_column or "action"),
        frame_index=frame,
    )
    action = np.asarray(actions[0], dtype=np.float32).reshape(ACTION_DIM)
    group = ReplayTrajectoryGroup(
        key="policy_qpos_init",
        label="policy_qpos_init",
        source_path=str(source_path),
        source_episode_index=int(resolved_metadata.get("episode_index", int(episode_index))),
        start_index=frame,
        end_index=frame + 1,
        frame_count=1,
        action_hz=0.0,
        actions=actions.astype(np.float32, copy=True),
        arm_joint_targets=load_replay_arm_joint_targets_from_parquet(
            source_path,
            active_sides=active,
            start_index=frame,
            end_index=frame + 1,
            required=False,
        ),
        arm_observation_current_qpos=load_replay_arm_observation_qpos_from_parquet(
            source_path,
            active_sides=active,
            field_name="current_qpos",
            start_index=frame,
            end_index=frame + 1,
        ),
        arm_observation_actual_qpos=load_replay_arm_observation_qpos_from_parquet(
            source_path,
            active_sides=active,
            field_name="actual_qpos",
            start_index=frame,
            end_index=frame + 1,
        ),
        arm_observation_qpos_audit=load_replay_arm_qpos_audit_from_parquet(
            source_path,
            active_sides=active,
            start_index=frame,
            end_index=frame + 1,
        ),
        metadata={"source_metadata": dict(resolved_metadata)},
    )
    targets, sources = select_replay_qpos_init_targets(
        group,
        active,
        index=0,
        source=normalize_qpos_init_source(qpos_init_source),
    )
    missing = [side for side in active if side not in targets]
    if missing:
        raise RuntimeError(
            "Policy qpos init target frame is missing finite qpos target(s) for "
            f"{missing}; qpos_init_source={qpos_init_source!r}."
        )
    return DeployPolicyQposInitPlan(
        source_path=str(source_path),
        source_episode_index=int(resolved_metadata.get("episode_index", int(episode_index))),
        frame_index=frame,
        action=action.astype(np.float32, copy=True),
        target_qpos_by_side={
            side: np.asarray(targets[side], dtype=float).reshape(ARM_JOINT_DOF).copy()
            for side in active
        },
        target_sources_by_side={str(side): str(sources.get(side, "")) for side in active},
        metadata={
            "action_column": str(action_column or "action"),
            **action_metadata,
            "qpos_init_source": normalize_qpos_init_source(qpos_init_source),
            "source_metadata": dict(resolved_metadata),
        },
    )


class DeployPolicyQposInitController:
    def __init__(
        self,
        plan: DeployPolicyQposInitPlan,
        *,
        active_sides: Sequence[str],
        hand_action_specs_by_side: Mapping[str, Any] | None = None,
        settings: DeployPolicyQposInitSettings | None = None,
    ) -> None:
        self.plan = plan
        self.active_sides = tuple(str(side) for side in active_sides)
        self.settings = settings if settings is not None else DeployPolicyQposInitSettings()
        self.state = "idle"
        self._plan_steps: tuple[Any, ...] = ()
        self._step_index = 0
        self._step_count = 0
        self._next_send_at = 0.0
        self._started_monotonic: float | None = None
        self._effective_timeout_s = float(self.settings.move_to_start_timeout_s)
        self._last_message = ""
        self._last_qpos_error_text = ""
        self._last_hand_error_text = ""
        self._hand_command_attempted_at_by_side: dict[str, float] = {}
        self._hand_best_error_by_side: dict[str, float] = {}
        self._hand_last_progress_at_by_side: dict[str, float] = {}
        self._hand_reconnect_attempts_by_side: dict[str, int] = {}
        self._hand_action_specs_by_side = _normalize_hand_action_specs(
            hand_action_specs_by_side
        )
        self._target_hand_by_side = (
            _hand_targets_from_action(
                self.plan.action,
                specs_by_side=self._hand_action_specs_by_side,
            )
            if bool(self.settings.initialize_hands)
            else {}
        )

    @property
    def blocks_policy(self) -> bool:
        return self.state != "released"

    @property
    def running(self) -> bool:
        return self.state in {
            "planning",
            "plan_validated",
            "moving",
            "final_hold",
            "hand_final_hold",
        }

    @property
    def status_text(self) -> str:
        if self.state == "idle":
            return "policy_qpos_init=wait:g"
        if self.state == "planning":
            return "policy_qpos_init=planning"
        if self.state == "moving":
            return f"policy_qpos_init=qpos:{self._step_index}/{self._step_count}"
        if self.state == "final_hold":
            suffix = "" if not self._last_qpos_error_text else f",err={self._last_qpos_error_text}"
            return f"policy_qpos_init=final_hold{suffix}"
        if self.state == "hand_final_hold":
            suffix = "" if not self._last_hand_error_text else f",err={self._last_hand_error_text}"
            return f"policy_qpos_init=hands{suffix}"
        if self.state == "plan_validated":
            return "policy_qpos_init=plan_validated"
        if self.state == "failed":
            return "policy_qpos_init=failed:g"
        if self.state == "timeout":
            return "policy_qpos_init=timeout:g"
        return "policy_qpos_init=released"

    def begin(
        self,
        *,
        arm_backends: Mapping[str, Any],
        now: float,
    ) -> str:
        if self.state == "released":
            return "[PolicyQposInit] already released."
        if self.running:
            return f"[PolicyQposInit] already {self.state}."
        self.state = "planning"
        self._started_monotonic = time.monotonic()
        self._step_index = 0
        self._plan_steps = ()
        self._step_count = 0
        self._effective_timeout_s = float(self.settings.move_to_start_timeout_s)
        self._last_qpos_error_text = ""
        self._last_hand_error_text = ""
        self._hand_command_attempted_at_by_side.clear()
        self._hand_best_error_by_side.clear()
        self._hand_last_progress_at_by_side.clear()
        self._hand_reconnect_attempts_by_side.clear()
        try:
            self._build_plan(arm_backends)
        except Exception as exc:  # noqa: BLE001 - planning failed before any command is sent.
            self.state = "failed"
            self._started_monotonic = None
            self._next_send_at = 0.0
            message = _format_planning_failure(exc)
            self._last_message = message
            return message
        self._next_send_at = float(now)
        if bool(self.settings.validate_plan_only):
            self.state = "plan_validated"
        else:
            self.state = "moving" if self._step_count > 0 else "final_hold"
        sources = ", ".join(
            f"{side}={self.plan.target_sources_by_side.get(side, '')}"
            for side in self.active_sides
        )
        self._last_message = (
            "[PolicyQposInit] qpos init plan ready "
            f"frame={self.plan.frame_index} steps={self._step_count} sources={sources} "
            f"hand_sides={list(self._target_hand_by_side)}"
        )
        return self._last_message

    def step(
        self,
        *,
        now: float,
        send_actions: bool,
        arm_backends: Mapping[str, Any],
        hand_backends: Mapping[str, Any] | None = None,
    ) -> PolicyQposInitStepResult:
        if self.state in {"idle", "planning", "failed", "timeout", "released"}:
            return PolicyQposInitStepResult()
        if self.state == "plan_validated":
            if send_actions:
                raise RuntimeError(
                    "qpos-init plan-only validation cannot be combined with "
                    "send_actions=true"
                )
            self.state = "released"
            self._started_monotonic = None
            self._next_send_at = 0.0
            message = (
                "[PolicyQposInit] dataset-frame qpos-init plan and collision "
                "checks validated for observation-only RTC; no arm or "
                "end-effector command was sent."
            )
            self._last_message = message
            return PolicyQposInitStepResult(
                completed=True,
                requires_fresh_observation=True,
                plan_only=True,
                message=message,
            )
        hand_backend_map = {} if hand_backends is None else hand_backends
        timeout_message = self._timeout_message(
            arm_backends,
            hand_backends=hand_backend_map,
        )
        if timeout_message:
            self.state = "timeout"
            self._started_monotonic = None
            self._last_message = timeout_message
            return PolicyQposInitStepResult(message=timeout_message)
        if self.state == "moving":
            if float(now) + 1e-9 < float(self._next_send_at):
                return PolicyQposInitStepResult()
            if self._step_index < self._step_count:
                step = self._plan_steps[self._step_index]
                if send_actions:
                    try:
                        _send_qpos_targets(
                            step.qpos_by_side,
                            arm_backends=arm_backends,
                            source="replay_qpos_init",
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.state = "failed"
                        message = _format_execution_failure(
                            "qpos init command send failed",
                            exc,
                        )
                        self._last_message = message
                        return PolicyQposInitStepResult(message=message)
                self._step_index += 1
                self._next_send_at = float(now) + max(0.001, float(self.settings.qpos_init_dt_sec))
                if self._step_index < self._step_count:
                    return PolicyQposInitStepResult()
            self.state = "final_hold"

        if self.state == "final_hold":
            reached, error_text = _qpos_reached_status(
                self.plan.target_qpos_by_side,
                arm_backends=arm_backends,
                tolerance=float(self.settings.qpos_reached_tolerance_rad),
                send_actions=send_actions,
            )
            self._last_qpos_error_text = error_text
            if reached:
                if self._target_hand_by_side:
                    self.state = "hand_final_hold"
                    self._next_send_at = min(float(self._next_send_at), float(now))
                else:
                    self.state = "released"
                    return PolicyQposInitStepResult(
                        completed=True,
                        requires_fresh_observation=True,
                        message=(
                            "[PolicyQposInit] dataset frame qpos reached; "
                            "policy startup is released."
                        ),
                    )
            if send_actions and not reached:
                try:
                    _send_qpos_targets(
                        self.plan.target_qpos_by_side,
                        arm_backends=arm_backends,
                        source="replay_qpos_init_final_hold",
                    )
                except Exception as exc:  # noqa: BLE001
                    self.state = "failed"
                    message = _format_execution_failure(
                        "qpos init final hold command send failed",
                        exc,
                    )
                    self._last_message = message
                    return PolicyQposInitStepResult(message=message)
            if self.state == "final_hold":
                return PolicyQposInitStepResult()

        if self.state == "hand_final_hold":
            evaluation = _hand_reached_status(
                self._target_hand_by_side,
                specs_by_side=self._hand_action_specs_by_side,
                hand_backends=hand_backend_map,
                tolerance=float(self.settings.hand_reached_tolerance),
                dmtac_tolerance=float(
                    self.settings.dmtac_hand_reached_tolerance
                ),
                max_age_sec=float(self.settings.hand_state_max_age_sec),
                progress_timeout_sec=float(
                    self.settings.hand_progress_timeout_sec
                ),
                progress_epsilon=float(self.settings.hand_progress_epsilon),
                reconnect_attempt_limit=int(
                    self.settings.hand_reconnect_attempts
                ),
                command_attempted_at_by_side=(
                    self._hand_command_attempted_at_by_side
                ),
                best_error_by_side=self._hand_best_error_by_side,
                last_progress_at_by_side=(
                    self._hand_last_progress_at_by_side
                ),
                reconnect_attempts_by_side=(
                    self._hand_reconnect_attempts_by_side
                ),
                send_actions=send_actions,
                now=float(now),
            )
            self._last_hand_error_text = evaluation.status_text
            if evaluation.reached:
                self.state = "released"
                return PolicyQposInitStepResult(
                    completed=True,
                    requires_fresh_observation=True,
                    message=(
                        "[PolicyQposInit] dataset frame qpos and end-effector "
                        "targets reached; policy startup is released."
                    ),
                )
            if evaluation.exhausted_sides:
                self.state = "failed"
                sides = ",".join(evaluation.exhausted_sides)
                message = (
                    "[PolicyQposInit] end-effector init failed closed after "
                    f"reconnect and force-resend; sides={sides} "
                    f"status={evaluation.status_text}. Policy remains blocked."
                )
                self._last_message = message
                return PolicyQposInitStepResult(message=message)
            if evaluation.reconnect_sides:
                try:
                    _reconnect_and_force_send_hand_targets(
                        evaluation.reconnect_sides,
                        targets_by_side=self._target_hand_by_side,
                        hand_backends=hand_backend_map,
                        frame_index=int(self.plan.frame_index),
                    )
                except Exception as exc:  # noqa: BLE001
                    self.state = "failed"
                    message = _format_execution_failure(
                        "end-effector reconnect/force-resend failed",
                        exc,
                    )
                    self._last_message = message
                    return PolicyQposInitStepResult(message=message)
                for side in evaluation.reconnect_sides:
                    side_key = str(side)
                    self._hand_reconnect_attempts_by_side[side_key] = (
                        int(
                            self._hand_reconnect_attempts_by_side.get(
                                side_key,
                                0,
                            )
                        )
                        + 1
                    )
                    self._hand_command_attempted_at_by_side[side_key] = float(now)
                    self._hand_best_error_by_side.pop(side_key, None)
                    self._hand_last_progress_at_by_side.pop(side_key, None)
                self._next_send_at = float(now) + max(
                    0.001,
                    float(self.settings.qpos_init_dt_sec),
                )
                attempts_text = ",".join(
                    f"{side}={self._hand_reconnect_attempts_by_side[str(side)]}/"
                    f"{int(self.settings.hand_reconnect_attempts)}"
                    for side in evaluation.reconnect_sides
                )
                return PolicyQposInitStepResult(
                    message=(
                        "[PolicyQposInit] no measured end-effector progress; "
                        "reconnected gripper and force-resent the recorded "
                        f"target ({attempts_text})."
                    )
                )
            if send_actions and float(now) + 1e-9 >= float(self._next_send_at):
                try:
                    _send_hand_targets(
                        self._target_hand_by_side,
                        hand_backends=hand_backend_map,
                        frame_index=int(self.plan.frame_index),
                        source="policy_qpos_init_hand_final_hold",
                    )
                    for side in self._target_hand_by_side:
                        self._hand_command_attempted_at_by_side.setdefault(
                            str(side),
                            float(now),
                        )
                except Exception as exc:  # noqa: BLE001
                    self.state = "failed"
                    message = _format_execution_failure(
                        "end-effector init command send failed",
                        exc,
                    )
                    self._last_message = message
                    return PolicyQposInitStepResult(message=message)
                self._next_send_at = float(now) + max(
                    0.001,
                    float(self.settings.qpos_init_dt_sec),
                )
            return PolicyQposInitStepResult()
        return PolicyQposInitStepResult()

    def _build_plan(self, arm_backends: Mapping[str, Any]) -> None:
        current = _current_qpos_by_side(arm_backends, self.active_sides)
        branch_neutral = _branch_neutral_qpos(
            self.settings.qpos_init_branch_neutral_qpos_path,
            active_sides=self.active_sides,
            enabled=bool(self.settings.qpos_init_branch_search),
        )
        plan = build_rm65_qpos_init_plan(
            start_qpos_by_side=current,
            target_qpos_by_side=self.plan.target_qpos_by_side,
            active_sides=self.active_sides,
            qpos_dt_sec=float(self.settings.qpos_init_dt_sec),
            joint_velocity_profile=str(self.settings.qpos_init_joint_velocity_profile),
            qpos_init_max_angvel_rad_s=normalize_qpos_init_max_angvel(
                self.settings.qpos_init_max_angvel_rad_s
            ),
            max_steps_per_side=int(self.settings.qpos_init_max_steps_per_side),
            collision_check=bool(self.settings.qpos_init_collision_check),
            collision_scene_profile=str(self.settings.qpos_init_collision_scene_profile),
            collision_slow_distance_m=float(self.settings.qpos_init_collision_slow_distance_m),
            collision_hold_distance_m=float(self.settings.qpos_init_collision_hold_distance_m),
            collision_stop_distance_m=float(self.settings.qpos_init_collision_stop_distance_m),
            max_joint_abs_deg=float(self.settings.qpos_init_max_joint_abs_deg),
            joint_slow_margin_deg=float(self.settings.qpos_init_joint_slow_margin_deg),
            joint_hold_margin_deg=float(self.settings.qpos_init_joint_hold_margin_deg),
            single_arm_max_qpos_delta_deg=float(self.settings.qpos_init_single_arm_max_qpos_delta_deg),
            branch_search=bool(self.settings.qpos_init_branch_search),
            branch_search_max_branches=int(self.settings.qpos_init_branch_search_max_branches),
            branch_neutral_qpos_by_side=branch_neutral,
        )
        _require_plan_completed(plan)
        self._plan_steps = tuple(getattr(plan, "steps", ()))
        self._step_count = len(self._plan_steps)
        planned_duration_s = self._step_count * max(0.001, float(self.settings.qpos_init_dt_sec))
        execution_margin_s = max(5.0, planned_duration_s * 0.25)
        self._effective_timeout_s = max(
            float(self.settings.move_to_start_timeout_s),
            planned_duration_s + execution_margin_s,
        )

    def _timeout_message(
        self,
        arm_backends: Mapping[str, Any],
        *,
        hand_backends: Mapping[str, Any],
    ) -> str:
        if self._started_monotonic is None:
            return ""
        if time.monotonic() - float(self._started_monotonic) <= float(self._effective_timeout_s):
            return ""
        _, error_text = _qpos_reached_status(
            self.plan.target_qpos_by_side,
            arm_backends=arm_backends,
            tolerance=float(self.settings.qpos_reached_tolerance_rad),
            send_actions=True,
        )
        self._last_qpos_error_text = error_text
        hand_error_text = ""
        if self._target_hand_by_side:
            evaluation = _hand_reached_status(
                self._target_hand_by_side,
                specs_by_side=self._hand_action_specs_by_side,
                hand_backends=hand_backends,
                tolerance=float(self.settings.hand_reached_tolerance),
                dmtac_tolerance=float(
                    self.settings.dmtac_hand_reached_tolerance
                ),
                max_age_sec=float(self.settings.hand_state_max_age_sec),
                progress_timeout_sec=float(
                    self.settings.hand_progress_timeout_sec
                ),
                progress_epsilon=float(self.settings.hand_progress_epsilon),
                reconnect_attempt_limit=int(
                    self.settings.hand_reconnect_attempts
                ),
                command_attempted_at_by_side=(
                    self._hand_command_attempted_at_by_side
                ),
                best_error_by_side=self._hand_best_error_by_side,
                last_progress_at_by_side=(
                    self._hand_last_progress_at_by_side
                ),
                reconnect_attempts_by_side=(
                    self._hand_reconnect_attempts_by_side
                ),
                send_actions=True,
                now=time.time(),
            )
            hand_error_text = evaluation.status_text
            self._last_hand_error_text = hand_error_text
        return (
            "[PolicyQposInit] timed out while moving to dataset frame; "
            "policy remains blocked and no further qpos-init commands will be sent. "
            f"state={self.state} step={self._step_index}/{self._step_count} "
            f"timeout_s={float(self._effective_timeout_s):.2f} "
            f"tolerance_rad={float(self.settings.qpos_reached_tolerance_rad):.4f} "
            f"{error_text} "
            f"hand={hand_error_text or 'not_configured'}"
        )


def _format_planning_failure(exc: BaseException) -> str:
    text = str(exc).strip()
    if not text:
        text = repr(exc)
    compact = " ".join(text.split())
    if len(compact) > 1400:
        compact = compact[:1400].rstrip() + " ..."
    return (
        "[PolicyQposInit] qpos init planning failed before execution; "
        "no command was sent and policy remains blocked. "
        f"{type(exc).__name__}: {compact} "
        "Try a safer dataset frame, move manually to a safer pre-start qpos, or if the strict "
        "proxy is overly conservative use --policy-qpos-init-collision-scene-profile "
        "dual_arm_hand_proxy_open after checking clearance."
    )


def _format_execution_failure(context: str, exc: BaseException) -> str:
    text = str(exc).strip() or repr(exc)
    compact = " ".join(text.split())
    if len(compact) > 800:
        compact = compact[:800].rstrip() + " ..."
    return (
        f"[PolicyQposInit] {context}; policy remains blocked. "
        f"{type(exc).__name__}: {compact}"
    )


def _current_qpos_by_side(
    arm_backends: Mapping[str, Any],
    active_sides: Sequence[str],
) -> dict[str, np.ndarray]:
    current: dict[str, np.ndarray] = {}
    for side in active_sides:
        backend = arm_backends.get(str(side))
        if backend is None:
            raise RuntimeError(f"Policy qpos init requires {side}_arm backend.")
        state = _backend_state(backend)
        state_source = str(state.get("_qpos_state_source") or "")
        keys = (
            ("hardware_qpos", "measured_qpos")
            if state_source in {"hardware_refresh", "fresh_hardware_snapshot"}
            else ("hardware_qpos", "measured_qpos", "current_qpos")
        )
        for key in keys:
            value = state.get(key)
            if value is None:
                continue
            qpos = np.asarray(value, dtype=float).reshape(-1)
            if qpos.shape == (ARM_JOINT_DOF,) and np.all(np.isfinite(qpos)):
                current[str(side)] = qpos.astype(float, copy=True)
                break
        if str(side) not in current:
            raise RuntimeError(f"{side}_arm current qpos is unavailable for policy qpos init.")
    return current


def _backend_state(backend: Any) -> Mapping[str, Any]:
    get_state = getattr(backend, "get_state", None)
    if callable(get_state):
        cached = dict(get_state())
        qpos = cached.get("hardware_qpos")
        measured_at = cached.get("measured_timestamp_monotonic")
        try:
            vector = np.asarray(qpos, dtype=float).reshape(-1)
            measured_monotonic = float(measured_at)
            age_sec = time.monotonic() - measured_monotonic
            fresh_hardware = bool(
                vector.shape == (ARM_JOINT_DOF,)
                and np.all(np.isfinite(vector))
                and np.isfinite(measured_monotonic)
                and 0.0 <= age_sec <= _QPOS_INIT_FRESH_HARDWARE_MAX_AGE_SEC
                and bool(cached.get("hardware_measurement_valid"))
                and bool(cached.get("hardware_read_success"))
                and int(cached.get("hardware_measurement_sequence") or 0) > 0
            )
        except (TypeError, ValueError, OverflowError):
            fresh_hardware = False
        if fresh_hardware:
            cached["_qpos_state_source"] = "fresh_hardware_snapshot"
            cached["_qpos_hardware_age_sec"] = float(age_sec)
            return cached
    refresh = getattr(backend, "refresh_state_from_hardware", None)
    if callable(refresh):
        state = dict(refresh())
        state["_qpos_state_source"] = "hardware_refresh"
        return state
    if not callable(get_state):
        raise RuntimeError("arm backend exposes neither hardware refresh nor state read.")
    state = cached
    state["_qpos_state_source"] = "compatibility_state"
    return state


def _normalize_hand_action_specs(
    specs_by_side: Mapping[str, Any] | None,
) -> dict[str, HandActionSpec]:
    normalized: dict[str, HandActionSpec] = {}
    for side, payload in dict(specs_by_side or {}).items():
        if isinstance(payload, HandActionSpec):
            spec = payload
        elif isinstance(payload, Mapping):
            spec = HandActionSpec.from_metadata(payload)
        else:
            raise TypeError(
                f"Policy qpos init hand spec for {side!r} must be HandActionSpec or metadata."
            )
        if int(spec.slot_count) != HAND_DOF:
            raise ValueError(
                f"Policy qpos init requires {HAND_DOF}-slot canonical hand actions; "
                f"{side} declares {spec.slot_count}."
            )
        normalized[str(side)] = spec
    return normalized


def _hand_targets_from_action(
    action: Any,
    *,
    specs_by_side: Mapping[str, HandActionSpec],
) -> dict[str, np.ndarray]:
    vector = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    targets: dict[str, np.ndarray] = {}
    for side, spec in specs_by_side.items():
        slots = np.asarray(
            vector[ACTION_HAND_SLICES[str(side)]],
            dtype=np.float32,
        ).reshape(HAND_DOF)
        validated = spec.validate_vector(slots)
        active = spec.active_values(validated)
        if np.any(active < 0.0) or np.any(active > 1.0):
            raise ValueError(
                f"Policy qpos init {side} hand target must be normalized to [0, 1]."
            )
        targets[str(side)] = active.astype(np.float32, copy=True)
    return targets


def _send_hand_targets(
    targets_by_side: Mapping[str, Any],
    *,
    hand_backends: Mapping[str, Any],
    frame_index: int,
    source: str,
) -> None:
    for side, target in targets_by_side.items():
        backend = hand_backends.get(str(side))
        if backend is None:
            raise RuntimeError(
                f"Policy qpos init missing backend for {side}_hand."
            )
        backend.send_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=str(side),
                command_type="joint_vector",
                value=np.asarray(target, dtype=np.float32).reshape(-1).copy(),
                frame_name=None,
                metadata={
                    "source": str(source),
                    "policy_qpos_init_frame": int(frame_index),
                },
            )
        )


def _reconnect_and_force_send_hand_targets(
    sides: Sequence[str],
    *,
    targets_by_side: Mapping[str, Any],
    hand_backends: Mapping[str, Any],
    frame_index: int,
) -> None:
    del frame_index  # Reserved for future backend receipt metadata.
    for side in sides:
        side_key = str(side)
        backend = hand_backends.get(side_key)
        if backend is None:
            raise RuntimeError(
                f"Policy qpos init missing backend for {side_key}_hand."
            )
        reconnect = getattr(backend, "force_reconnect_gripper", None)
        force_send = getattr(backend, "send_fixed_joint_pos", None)
        if not callable(reconnect) or not callable(force_send):
            raise RuntimeError(
                f"{side_key}_hand does not expose DMTAC reconnect/force-send."
            )
        if not bool(reconnect("policy_qpos_init_no_measured_progress")):
            raise RuntimeError(f"{side_key}_hand gripper reconnect returned false.")
        target = np.asarray(targets_by_side[side_key], dtype=float).reshape(-1)
        if target.shape != (1,) or not np.all(np.isfinite(target)):
            raise ValueError(
                f"{side_key}_hand DMTAC target must be one finite scalar."
            )
        if not bool(force_send(float(target[0]))):
            raise RuntimeError(f"{side_key}_hand force-resend returned false.")


def _measured_hand_target(
    state: Mapping[str, Any],
    *,
    spec: HandActionSpec,
) -> np.ndarray | None:
    if int(spec.active_dof) == 1:
        scalar = state.get("gripper_measured_scalar")
        if scalar is not None:
            try:
                value = np.asarray([float(scalar)], dtype=float)
            except (TypeError, ValueError, OverflowError):
                value = np.asarray([], dtype=float)
            if value.shape == (1,) and np.all(np.isfinite(value)):
                return value
    for key in ("measured_joint_pos", "joint_pos"):
        raw = state.get(key)
        if raw is None:
            continue
        try:
            vector = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if vector.shape == (int(spec.active_dof),) and np.all(np.isfinite(vector)):
            return vector.copy()
        if vector.shape == (int(spec.slot_count),) and np.all(np.isfinite(vector)):
            return spec.active_values(vector).astype(float, copy=True)
    return None


def _hand_reached_status(
    targets_by_side: Mapping[str, Any],
    *,
    specs_by_side: Mapping[str, HandActionSpec],
    hand_backends: Mapping[str, Any],
    tolerance: float,
    dmtac_tolerance: float,
    max_age_sec: float,
    progress_timeout_sec: float,
    progress_epsilon: float,
    reconnect_attempt_limit: int,
    command_attempted_at_by_side: Mapping[str, float],
    best_error_by_side: dict[str, float],
    last_progress_at_by_side: dict[str, float],
    reconnect_attempts_by_side: Mapping[str, int],
    send_actions: bool,
    now: float,
) -> _HandInitEvaluation:
    if not send_actions:
        return _HandInitEvaluation(reached=True, status_text="dry-run")
    status_by_side: dict[str, str] = {}
    all_reached = True
    reconnect_sides: list[str] = []
    exhausted_sides: list[str] = []
    for side, target in targets_by_side.items():
        side_key = str(side)
        backend = hand_backends.get(str(side))
        spec = specs_by_side.get(str(side))
        if backend is None or spec is None:
            return _HandInitEvaluation(
                reached=False,
                status_text=f"{side}:backend_or_spec_unavailable",
            )
        try:
            state = dict(backend.get_state())
        except Exception as exc:  # noqa: BLE001
            return _HandInitEvaluation(
                reached=False,
                status_text=f"{side}:state_read_failed:{type(exc).__name__}:{exc}",
            )
        measured_at = state.get("joint_timestamp")
        try:
            measurement_age_sec = float(now) - float(measured_at)
        except (TypeError, ValueError, OverflowError):
            return _HandInitEvaluation(
                reached=False,
                status_text=f"{side}:measurement_timestamp_unavailable",
            )
        if (
            not np.isfinite(measurement_age_sec)
            or measurement_age_sec < -float(max_age_sec)
            or measurement_age_sec > float(max_age_sec)
        ):
            return _HandInitEvaluation(
                reached=False,
                status_text=f"{side}:measurement_stale:{measurement_age_sec:.3f}s",
            )
        measured = _measured_hand_target(state, spec=spec)
        if measured is None:
            return _HandInitEvaluation(
                reached=False,
                status_text=f"{side}:measured_target_unavailable",
            )
        requested = np.asarray(target, dtype=float).reshape(int(spec.active_dof))
        error = float(np.max(np.abs(requested - measured)))
        if str(spec.hand_model) != "dmtac_gripper":
            reached = error <= float(tolerance)
            all_reached = all_reached and reached
            status_by_side[side_key] = (
                f"measured_err={error:.4f}"
                if reached
                else f"measured_wait={error:.4f}"
            )
            continue

        attempted_at = command_attempted_at_by_side.get(side_key)
        if attempted_at is None:
            all_reached = False
            status_by_side[side_key] = "command_not_attempted"
            continue
        send_status = str(state.get("gripper_last_send_status") or "unknown")
        commanded_scalar = state.get("gripper_commanded_scalar")
        try:
            commanded_error = abs(float(commanded_scalar) - float(requested[0]))
        except (TypeError, ValueError, OverflowError):
            commanded_error = float("inf")
        command_confirmed = bool(
            send_status in _DMTAC_ACCEPTED_SEND_STATUSES
            and np.isfinite(commanded_error)
            and commanded_error <= _DMTAC_COMMAND_CONFIRM_TOLERANCE
        )
        if not command_confirmed:
            all_reached = False
            status_by_side[side_key] = (
                f"command_unconfirmed:{send_status}:err={commanded_error:.4f}"
            )
            continue
        feedback_after_attempt = bool(
            float(measured_at) + 1e-6 >= float(attempted_at)
        )
        if not feedback_after_attempt:
            all_reached = False
            status_by_side[side_key] = (
                f"feedback_pre_command:measured_err={error:.4f}"
            )
            continue
        if error <= float(dmtac_tolerance):
            status_by_side[side_key] = (
                f"measured_reached:{send_status}:err={error:.4f}"
            )
            continue
        all_reached = False
        previous_best = best_error_by_side.get(side_key)
        if previous_best is None:
            best_error_by_side[side_key] = error
            last_progress_at_by_side[side_key] = float(attempted_at)
        elif error <= float(previous_best) - max(0.0, float(progress_epsilon)):
            best_error_by_side[side_key] = error
            last_progress_at_by_side[side_key] = float(now)
        best_error = float(best_error_by_side.get(side_key, error))
        last_progress_at = float(
            last_progress_at_by_side.get(side_key, attempted_at)
        )
        stalled_for = max(0.0, float(now) - last_progress_at)
        if stalled_for + 1e-9 < max(0.0, float(progress_timeout_sec)):
            status_by_side[side_key] = (
                f"moving_wait:err={error:.4f}:best={best_error:.4f}"
                f":stall={stalled_for:.2f}s"
            )
            continue
        reconnect_count = int(reconnect_attempts_by_side.get(side_key, 0))
        if reconnect_count < max(0, int(reconnect_attempt_limit)):
            reconnect_sides.append(side_key)
            status_by_side[side_key] = (
                f"reconnect_required:err={error:.4f}:best={best_error:.4f}"
                f":stall={stalled_for:.2f}s"
            )
        else:
            exhausted_sides.append(side_key)
            status_by_side[side_key] = (
                f"reconnect_exhausted:err={error:.4f}:best={best_error:.4f}"
                f":stall={stalled_for:.2f}s"
            )
    return _HandInitEvaluation(
        reached=all_reached,
        status_text=";".join(
            f"{side}={status}" for side, status in status_by_side.items()
        ),
        reconnect_sides=tuple(reconnect_sides),
        exhausted_sides=tuple(exhausted_sides),
    )


def _send_qpos_targets(
    qpos_by_side: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, Any],
    source: str,
) -> None:
    for side, qpos in qpos_by_side.items():
        backend = arm_backends.get(str(side))
        if backend is None:
            raise RuntimeError(f"Policy qpos init missing backend for {side}_arm.")
        target = np.asarray(qpos, dtype=float).reshape(ARM_JOINT_DOF)
        backend.send_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=str(side),
                command_type="joint_vector",
                value=target,
                frame_name=None,
                metadata={"source": str(source)},
            )
        )


def _qpos_reached(
    qpos_by_side: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, Any],
    tolerance: float,
    send_actions: bool,
) -> bool:
    reached, _ = _qpos_reached_status(
        qpos_by_side,
        arm_backends=arm_backends,
        tolerance=tolerance,
        send_actions=send_actions,
    )
    return reached


def _qpos_reached_status(
    qpos_by_side: Mapping[str, Any],
    *,
    arm_backends: Mapping[str, Any],
    tolerance: float,
    send_actions: bool,
) -> tuple[bool, str]:
    if not send_actions:
        return True, "dry-run"
    try:
        current = _current_qpos_by_side(arm_backends, qpos_by_side.keys())
    except Exception as exc:  # noqa: BLE001
        return False, f"qpos_read_unavailable:{type(exc).__name__}:{exc}"
    worst_side = ""
    worst_joint = -1
    worst_error = 0.0
    for side, target in qpos_by_side.items():
        diff = np.abs(
            np.asarray(target, dtype=float).reshape(ARM_JOINT_DOF)
            - current[str(side)]
        )
        joint_index = int(np.argmax(diff))
        error = float(diff[joint_index])
        if error > worst_error:
            worst_side = str(side)
            worst_joint = joint_index
            worst_error = error
    error_text = (
        "max_err=none"
        if worst_joint < 0
        else (
            f"max_err={worst_side}:j{worst_joint + 1}="
            f"{worst_error:.4f}rad/{np.rad2deg(worst_error):.2f}deg"
        )
    )
    return worst_error <= float(tolerance), error_text


def _branch_neutral_qpos(
    path: str | Path | None,
    *,
    active_sides: Sequence[str],
    enabled: bool,
) -> dict[str, np.ndarray] | None:
    sides = tuple(str(side) for side in active_sides)
    if not enabled or sides != ("left", "right") or path is None or not str(path).strip():
        return None
    qpos = load_qpos_target(path).qpos_by_side()
    return {
        side: np.asarray(qpos[side], dtype=float).reshape(ARM_JOINT_DOF).copy()
        for side in sides
    }
