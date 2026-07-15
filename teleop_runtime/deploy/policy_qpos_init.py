from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.v22 import (
    DEXTELEOP_DATASET_SCHEMA_V22,
    policy_io_layout_from_metadata,
)
from robot_motion.deploy.schema import ACTION_DIM, ARM_JOINT_DOF
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
    load_replay_actions_from_parquet,
    load_replay_arm_joint_targets_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    load_replay_arm_qpos_audit_from_parquet,
    load_v22_replay_actions_from_parquet,
    resolve_replay_parquet_path,
)


# Migration note:
# This module is real-launcher staging code. The dataset-frame target selection
# and qpos init planning should move to robot_motion/deploy once deploy common
# code is split out. Backend send_command dispatch and live hardware qpos reads
# should remain in teleop_runtime.


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


@dataclass(frozen=True)
class PolicyQposInitStepResult:
    last_action: np.ndarray | None = None
    completed: bool = False
    message: str = ""


def _load_policy_qpos_init_actions(
    source_path: Path,
    *,
    resolved_metadata: Mapping[str, Any],
    action_column: str,
    frame_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one display/hold action without assuming a legacy 26D source row.

    Qpos target selection below uses only audited joint-state columns.  The
    action retained on the init plan is nevertheless consumed by status/debug
    paths, so it must always be a canonical 26D deployment action.  v2.2
    datasets store a compact layout-bound action (16D for dual DMTAC), which
    must take the same explicit bridge used by dataset replay.
    """

    dataset_info = resolved_metadata.get("dataset_info")
    if (
        isinstance(dataset_info, Mapping)
        and str(dataset_info.get("dexteleop_dataset_schema") or "")
        == DEXTELEOP_DATASET_SCHEMA_V22
    ):
        layout = policy_io_layout_from_metadata(dataset_info, allow_legacy_v21=False)
        actions, action_metadata = load_v22_replay_actions_from_parquet(
            source_path,
            layout=layout,
            start_index=frame_index,
            end_index=frame_index + 1,
        )
        return actions, {
            "action_encoding": "v22_layout_to_legacy_q6_bridge",
            "policy_io_layout_hash": layout.layout_hash,
            "action_metadata": dict(action_metadata),
        }

    actions = load_replay_actions_from_parquet(
        source_path,
        action_column=action_column,
        start_index=frame_index,
        end_index=frame_index + 1,
    )
    return actions, {"action_encoding": "legacy_26d"}


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
        self._last_message = ""
        self._last_qpos_error_text = ""

    @property
    def blocks_policy(self) -> bool:
        return self.state != "released"

    @property
    def running(self) -> bool:
        return self.state in {"planning", "moving", "final_hold"}

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
        self._last_qpos_error_text = ""
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
        self.state = "moving" if self._step_count > 0 else "final_hold"
        sources = ", ".join(
            f"{side}={self.plan.target_sources_by_side.get(side, '')}"
            for side in self.active_sides
        )
        self._last_message = (
            "[PolicyQposInit] qpos init plan ready "
            f"frame={self.plan.frame_index} steps={self._step_count} sources={sources}"
        )
        return self._last_message

    def step(
        self,
        *,
        now: float,
        send_actions: bool,
        arm_backends: Mapping[str, Any],
    ) -> PolicyQposInitStepResult:
        if self.state in {"idle", "planning", "failed", "timeout", "released"}:
            return PolicyQposInitStepResult()
        timeout_message = self._timeout_message(arm_backends)
        if timeout_message:
            self.state = "timeout"
            self._started_monotonic = None
            self._last_message = timeout_message
            return PolicyQposInitStepResult(
                last_action=self.plan.action,
                message=timeout_message,
            )
        if self.state == "moving":
            if float(now) + 1e-9 < float(self._next_send_at):
                return PolicyQposInitStepResult(last_action=self.plan.action)
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
                        return PolicyQposInitStepResult(
                            last_action=self.plan.action,
                            message=message,
                        )
                self._step_index += 1
                self._next_send_at = float(now) + max(0.001, float(self.settings.qpos_init_dt_sec))
                if self._step_index < self._step_count:
                    return PolicyQposInitStepResult(last_action=self.plan.action)
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
                self.state = "released"
                return PolicyQposInitStepResult(
                    last_action=self.plan.action,
                    completed=True,
                    message="[PolicyQposInit] dataset frame qpos reached; policy startup is released.",
                )
            if send_actions:
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
                    return PolicyQposInitStepResult(
                        last_action=self.plan.action,
                        message=message,
                    )
            return PolicyQposInitStepResult(last_action=self.plan.action)
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

    def _timeout_message(self, arm_backends: Mapping[str, Any]) -> str:
        if self._started_monotonic is None:
            return ""
        if time.monotonic() - float(self._started_monotonic) <= float(self.settings.move_to_start_timeout_s):
            return ""
        _, error_text = _qpos_reached_status(
            self.plan.target_qpos_by_side,
            arm_backends=arm_backends,
            tolerance=float(self.settings.qpos_reached_tolerance_rad),
            send_actions=True,
        )
        self._last_qpos_error_text = error_text
        return (
            "[PolicyQposInit] timed out while moving to dataset frame; "
            "policy remains blocked and no further qpos-init commands will be sent. "
            f"state={self.state} step={self._step_index}/{self._step_count} "
            f"timeout_s={float(self.settings.move_to_start_timeout_s):.2f} "
            f"tolerance_rad={float(self.settings.qpos_reached_tolerance_rad):.4f} "
            f"{error_text}"
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
        for key in ("hardware_qpos", "current_qpos", "last_target_qpos"):
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
    try:
        return dict(backend.refresh_state_from_hardware())
    except Exception:
        return dict(backend.get_state())


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
