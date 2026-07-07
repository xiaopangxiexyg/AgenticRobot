from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from ..deploy.schema import ARM_JOINT_DOF
from .arm_init.qpos_execute import (
    DEFAULT_QPOS_INIT_BRANCH_SEARCH,
    DEFAULT_QPOS_INIT_BRANCH_SEARCH_MAX_BRANCHES,
    DEFAULT_QPOS_INIT_COLLISION_SCENE,
    DEFAULT_QPOS_INIT_COLLISION_HOLD_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_SLOW_DISTANCE_M,
    DEFAULT_QPOS_INIT_COLLISION_STOP_DISTANCE_M,
    DEFAULT_QPOS_INIT_JOINT_HOLD_MARGIN_DEG,
    DEFAULT_QPOS_INIT_JOINT_SLOW_MARGIN_DEG,
    DEFAULT_QPOS_INIT_MAX_JOINT_ABS_DEG,
    DEFAULT_SINGLE_ARM_MAX_QPOS_DELTA_DEG,
    DEFAULT_QPOS_TARGET_PATH,
    build_rm65_qpos_init_plan,
    load_qpos_target,
    normalize_qpos_init_max_angvel,
    _require_plan_completed,
)
from .action_frame import replay_teleop_command_at_index
from .checks import ReplayPreflightReport, run_replay_preflight
from .dataset import LoadedReplayEpisode
from .hardware import ReplayHardwareConfig, ReplayHardwareRig, normalize_active_sides
from .trajectory import (
    QPOS_INIT_AUTO,
    REPLAY_TRAJECTORY_ACTION_POSE,
    REPLAY_TRAJECTORY_QPOS_TARGET,
    ReplayGroupSpec,
    ReplayTrajectoryGroup,
    build_replay_group_for_episode,
    normalize_qpos_init_source,
    normalize_replay_trajectory_source,
    select_replay_qpos_init_targets,
)


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def _print_worker_fault(stage: str, exc: BaseException) -> None:
    print(f"[ReplayHardware][ERROR] {stage} failed: {exc}", file=sys.stderr, flush=True)


def _episode_index_or_zero(episode: LoadedReplayEpisode) -> int:
    if episode.summary.episode_index is None:
        return 0
    return int(episode.summary.episode_index)


def _payload_value(payload: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return default


def _episode_group_args(
    episode: LoadedReplayEpisode,
    *,
    active_sides: Sequence[str],
    action_column: str,
) -> argparse.Namespace:
    episode_index = episode.summary.episode_index
    use_dataset = episode_index is not None and episode.dataset_root.exists()
    return argparse.Namespace(
        replay_dataset=str(episode.dataset_root) if use_dataset else "",
        replay_parquet="" if use_dataset else str(episode.summary.parquet_path),
        replay_episode=_episode_index_or_zero(episode),
        replay_action_column=str(action_column or "action"),
        replay_all_episodes=False,
        active_sides=tuple(active_sides),
    )


def build_episode_replay_group(
    episode: LoadedReplayEpisode,
    *,
    active_sides: Sequence[str],
    action_column: str = "action",
) -> ReplayTrajectoryGroup:
    sides = normalize_active_sides(tuple(active_sides))
    episode_index = _episode_index_or_zero(episode)
    args = _episode_group_args(episode, active_sides=sides, action_column=action_column)
    group = build_replay_group_for_episode(
        args,
        episode_index=episode_index,
        group_spec=ReplayGroupSpec(
            key="g",
            start_index=0,
            end_index=-1,
            episode_range=str(episode_index),
        ),
        balanced_episode_count=1,
        require_arm_joint_targets=False,
    )
    source_metadata = dict(group.metadata.get("source_metadata") or {})
    if episode.summary.metadata and not isinstance(source_metadata.get("episode_metadata"), Mapping):
        source_metadata["episode_metadata"] = dict(episode.summary.metadata)
    group.metadata["source_metadata"] = source_metadata
    group.metadata["loaded_episode"] = {
        "episode_id": episode.summary.episode_id,
        "episode_index": episode.summary.episode_index,
        "parquet_path": str(episode.summary.parquet_path),
        "dataset_root": str(episode.dataset_root),
    }
    return group


@dataclass(frozen=True)
class ReplayExecutionSettings:
    send_actions_enabled: bool = False
    action_column: str = "action"
    trajectory_source: str = REPLAY_TRAJECTORY_ACTION_POSE
    qpos_init_source: str = QPOS_INIT_AUTO
    end_effectors_enabled: bool = False
    qpos_reached_tolerance_rad: float = 0.01
    move_to_start_timeout_s: float = 30.0
    tcp_pos_tolerance_m: float = 1e-4
    tcp_rot_tolerance_rad: float = 1e-3
    max_qpos_init_delta_rad: float = float(np.pi)
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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], current: "ReplayExecutionSettings") -> "ReplayExecutionSettings":
        return cls(
            send_actions_enabled=bool(
                payload.get("send_actions")
                if "send_actions" in payload
                else payload.get("sendActions", current.send_actions_enabled)
            ),
            action_column=str(payload.get("action_column") or payload.get("actionColumn") or current.action_column),
            trajectory_source=normalize_replay_trajectory_source(
                payload.get("trajectory_source")
                or payload.get("trajectorySource")
                or current.trajectory_source
            ),
            qpos_init_source=normalize_qpos_init_source(
                payload.get("qpos_init_source")
                or payload.get("qposInitSource")
                or current.qpos_init_source
            ),
            end_effectors_enabled=bool(
                payload.get("end_effectors_enabled")
                if "end_effectors_enabled" in payload
                else payload.get(
                    "endEffectorsEnabled",
                    payload.get(
                        "hands_enabled",
                        payload.get("handsEnabled", current.end_effectors_enabled),
                    ),
                )
            ),
            qpos_reached_tolerance_rad=float(
                payload.get("qpos_reached_tolerance_rad")
                or payload.get("qposReachedToleranceRad")
                or current.qpos_reached_tolerance_rad
            ),
            move_to_start_timeout_s=float(
                payload.get("move_to_start_timeout_s")
                or payload.get("moveToStartTimeoutS")
                or current.move_to_start_timeout_s
            ),
            tcp_pos_tolerance_m=float(
                payload.get("tcp_pos_tolerance_m")
                or payload.get("tcpPosToleranceM")
                or current.tcp_pos_tolerance_m
            ),
            tcp_rot_tolerance_rad=float(
                payload.get("tcp_rot_tolerance_rad")
                or payload.get("tcpRotToleranceRad")
                or current.tcp_rot_tolerance_rad
            ),
            max_qpos_init_delta_rad=float(
                payload.get("max_qpos_init_delta_rad")
                or payload.get("maxQposInitDeltaRad")
                or current.max_qpos_init_delta_rad
            ),
            qpos_init_dt_sec=float(
                payload.get("qpos_init_dt_sec")
                or payload.get("qposInitDtSec")
                or current.qpos_init_dt_sec
            ),
            qpos_init_joint_velocity_profile=str(
                payload.get("qpos_init_joint_velocity_profile")
                or payload.get("qposInitJointVelocityProfile")
                or current.qpos_init_joint_velocity_profile
            ),
            qpos_init_max_angvel_rad_s=normalize_qpos_init_max_angvel(
                _payload_value(
                    payload,
                    (
                        "qpos_init_max_angvel_rad_s",
                        "qposInitMaxAngvelRadS",
                        "qpos_init_arm_max_angvel",
                        "qposInitArmMaxAngvel",
                    ),
                    current.qpos_init_max_angvel_rad_s,
                )
            ),
            qpos_init_max_steps_per_side=int(
                payload.get("qpos_init_max_steps_per_side")
                or payload.get("qposInitMaxStepsPerSide")
                or current.qpos_init_max_steps_per_side
            ),
            qpos_init_collision_check=bool(
                payload.get("qpos_init_collision_check")
                if "qpos_init_collision_check" in payload
                else payload.get("qposInitCollisionCheck", current.qpos_init_collision_check)
            ),
            qpos_init_collision_scene_profile=str(
                payload.get("qpos_init_collision_scene_profile")
                or payload.get("qposInitCollisionSceneProfile")
                or current.qpos_init_collision_scene_profile
            ),
            qpos_init_collision_slow_distance_m=float(
                payload.get("qpos_init_collision_slow_distance_m")
                or payload.get("qposInitCollisionSlowDistanceM")
                or current.qpos_init_collision_slow_distance_m
            ),
            qpos_init_collision_hold_distance_m=float(
                payload.get("qpos_init_collision_hold_distance_m")
                or payload.get("qposInitCollisionHoldDistanceM")
                or current.qpos_init_collision_hold_distance_m
            ),
            qpos_init_collision_stop_distance_m=float(
                payload.get("qpos_init_collision_stop_distance_m")
                or payload.get("qposInitCollisionStopDistanceM")
                or current.qpos_init_collision_stop_distance_m
            ),
            qpos_init_max_joint_abs_deg=float(
                payload.get("qpos_init_max_joint_abs_deg")
                or payload.get("qposInitMaxJointAbsDeg")
                or current.qpos_init_max_joint_abs_deg
            ),
            qpos_init_joint_slow_margin_deg=float(
                payload.get("qpos_init_joint_slow_margin_deg")
                or payload.get("qposInitJointSlowMarginDeg")
                or current.qpos_init_joint_slow_margin_deg
            ),
            qpos_init_joint_hold_margin_deg=float(
                payload.get("qpos_init_joint_hold_margin_deg")
                or payload.get("qposInitJointHoldMarginDeg")
                or current.qpos_init_joint_hold_margin_deg
            ),
            qpos_init_single_arm_max_qpos_delta_deg=float(
                payload.get("qpos_init_single_arm_max_qpos_delta_deg")
                or payload.get("qposInitSingleArmMaxQposDeltaDeg")
                or current.qpos_init_single_arm_max_qpos_delta_deg
            ),
            qpos_init_branch_search=bool(
                payload.get("qpos_init_branch_search")
                if "qpos_init_branch_search" in payload
                else payload.get("qposInitBranchSearch", current.qpos_init_branch_search)
            ),
            qpos_init_branch_search_max_branches=int(
                payload.get("qpos_init_branch_search_max_branches")
                or payload.get("qposInitBranchSearchMaxBranches")
                or current.qpos_init_branch_search_max_branches
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "send_actions_enabled": bool(self.send_actions_enabled),
            "action_column": str(self.action_column),
            "trajectory_source": normalize_replay_trajectory_source(self.trajectory_source),
            "qpos_init_source": normalize_qpos_init_source(self.qpos_init_source),
            "end_effectors_enabled": bool(self.end_effectors_enabled),
            "qpos_reached_tolerance_rad": float(self.qpos_reached_tolerance_rad),
            "move_to_start_timeout_s": float(self.move_to_start_timeout_s),
            "tcp_pos_tolerance_m": float(self.tcp_pos_tolerance_m),
            "tcp_rot_tolerance_rad": float(self.tcp_rot_tolerance_rad),
            "max_qpos_init_delta_rad": float(self.max_qpos_init_delta_rad),
            "qpos_init_dt_sec": float(self.qpos_init_dt_sec),
            "qpos_init_joint_velocity_profile": str(self.qpos_init_joint_velocity_profile),
            "qpos_init_max_angvel_rad_s": (
                None
                if self.qpos_init_max_angvel_rad_s is None
                else [float(value) for value in self.qpos_init_max_angvel_rad_s]
            ),
            "qpos_init_max_steps_per_side": int(self.qpos_init_max_steps_per_side),
            "qpos_init_collision_check": bool(self.qpos_init_collision_check),
            "qpos_init_collision_scene_profile": str(self.qpos_init_collision_scene_profile),
            "qpos_init_collision_slow_distance_m": float(self.qpos_init_collision_slow_distance_m),
            "qpos_init_collision_hold_distance_m": float(self.qpos_init_collision_hold_distance_m),
            "qpos_init_collision_stop_distance_m": float(self.qpos_init_collision_stop_distance_m),
            "qpos_init_max_joint_abs_deg": float(self.qpos_init_max_joint_abs_deg),
            "qpos_init_joint_slow_margin_deg": float(self.qpos_init_joint_slow_margin_deg),
            "qpos_init_joint_hold_margin_deg": float(self.qpos_init_joint_hold_margin_deg),
            "qpos_init_single_arm_max_qpos_delta_deg": float(self.qpos_init_single_arm_max_qpos_delta_deg),
            "qpos_init_branch_search": bool(self.qpos_init_branch_search),
            "qpos_init_branch_search_max_branches": int(self.qpos_init_branch_search_max_branches),
        }


class ReplayExecutionController:
    def __init__(
        self,
        *,
        hardware_config: ReplayHardwareConfig | None = None,
        settings: ReplayExecutionSettings | None = None,
    ):
        self._lock = threading.RLock()
        self._hardware_config = hardware_config or ReplayHardwareConfig()
        self._settings = settings or ReplayExecutionSettings()
        self._rig: ReplayHardwareRig | None = None
        self._group: ReplayTrajectoryGroup | None = None
        self._preflight_report: ReplayPreflightReport | None = None
        self._state = "configured"
        self._frame_index = 0
        self._last_error = ""
        self._last_message = ""
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._qpos_init_plan_summary: dict[str, Any] | None = None
        self._qpos_init_step_index = 0
        self._qpos_init_step_count = 0
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

    def close(self) -> None:
        self.abort(wait=True)
        with self._lock:
            if self._rig is not None:
                self._rig.close()
                self._rig = None
            self._state = "closed"

    def configure(
        self,
        *,
        hardware_config: ReplayHardwareConfig | None = None,
        settings: ReplayExecutionSettings | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked()
            config_changed = hardware_config is not None and hardware_config != self._hardware_config
            if self._rig is not None and config_changed:
                self._rig.close()
                self._rig = None
            if hardware_config is not None:
                self._hardware_config = hardware_config
            if settings is not None:
                self._settings = settings
            self._group = None
            self._preflight_report = None
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._last_error = ""
            self._last_message = "hardware replay configured"
            self._state = "connected" if self._rig is not None and self._rig.connected else "configured"
            return self.status()

    def configure_from_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            current_config = self._hardware_config
            current_settings = self._settings
        hardware_config = ReplayHardwareConfig.from_payload(
            {
                **current_config.to_dict(),
                **dict(payload),
            }
        )
        settings = ReplayExecutionSettings.from_payload(payload, current_settings)
        return self.configure(hardware_config=hardware_config, settings=settings)

    def set_send_actions(self, enabled: bool) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            self._settings = replace(self._settings, send_actions_enabled=bool(enabled))
            self._last_message = f"arm output {'enabled' if enabled else 'disabled'}"
            return self.status()

    def clear_preflight(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            self._group = None
            self._preflight_report = None
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._last_error = ""
            self._last_message = "preflight cleared"
            if self._rig is not None and self._rig.connected:
                self._state = "connected"
            else:
                self._state = "configured"
            return self.status()

    def connect(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            if self._rig is not None:
                self._rig.close()
            self._rig = ReplayHardwareRig(self._hardware_config)
            rig = self._rig
            self._state = "connecting"
            self._last_error = ""
        try:
            rig.connect()
        except Exception as exc:
            try:
                rig.close()
            except Exception:
                pass
            with self._lock:
                if self._rig is rig:
                    self._rig = None
                self._state = "fault"
                self._last_error = str(exc)
                self._last_message = "hardware connect failed"
            raise
        with self._lock:
            self._state = "connected"
            self._last_message = "hardware connected"
            return self.status()

    def preflight(self, episode: LoadedReplayEpisode) -> dict[str, Any]:
        with self._lock:
            rig = self._require_rig_locked()
            sides = self._hardware_config.active_sides
            settings = self._settings
        group = build_episode_replay_group(
            episode,
            active_sides=sides,
            action_column=settings.action_column,
        )
        report = run_replay_preflight(
            group=group,
            active_sides=sides,
            current_qpos_by_side=rig.current_qpos_by_side(),
            current_tcp_pose_by_side=rig.current_tcp_pose_by_side(),
            current_tcp_flange_by_side=rig.current_tcp_flange_by_side(),
            tcp_pos_tolerance_m=float(settings.tcp_pos_tolerance_m),
            tcp_rot_tolerance_rad=float(settings.tcp_rot_tolerance_rad),
            max_qpos_init_delta_rad=float(settings.max_qpos_init_delta_rad),
            trajectory_source=settings.trajectory_source,
            qpos_init_source=settings.qpos_init_source,
            end_effectors_enabled=bool(settings.end_effectors_enabled),
            end_effector_backend_sides=rig.end_effector_backend_sides(),
        )
        with self._lock:
            self._group = group
            self._preflight_report = report
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._last_message = "preflight ok" if report.ok else "preflight failed"
            self._last_error = "" if report.ok else "preflight failed"
            self._state = "preflight_ok" if report.ok else "preflight_failed"
            return self.status()

    def move_to_start(self) -> dict[str, Any]:
        with self._lock:
            self._require_output_enabled_locked()
            rig = self._require_rig_locked()
            group = self._require_checked_group_locked()
            self._ensure_no_worker_locked()
            target_qpos = self._first_qpos_targets_locked()
            self._stop_event.clear()
            self._state = "moving_to_start"
            self._frame_index = 0
            self._clear_qpos_init_status_locked()
            self._last_error = ""
            self._started_at = time.time()
            self._finished_at = None
            worker = threading.Thread(
                target=self._move_to_start_worker,
                args=(rig, group, target_qpos),
                name="replay-move-to-start",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return self.status()

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._require_output_enabled_locked()
            rig = self._require_rig_locked()
            group = self._require_checked_group_locked()
            if self._state != "start_ready":
                raise RuntimeError("Move to the replay start before starting hardware replay.")
            self._ensure_no_worker_locked()
            self._stop_event.clear()
            self._state = "replaying"
            self._frame_index = 0
            self._last_error = ""
            self._started_at = time.time()
            self._finished_at = None
            worker = threading.Thread(
                target=self._replay_worker,
                args=(rig, group),
                name="replay-trajectory",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return self.status()

    def abort(self, *, wait: bool = False) -> dict[str, Any]:
        worker: threading.Thread | None = None
        with self._lock:
            self._stop_event.set()
            worker = self._worker
            if self._state in {"moving_to_start", "start_ready", "replaying", "connecting"}:
                self._state = "aborted"
                self._last_message = "hardware replay aborted"
            rig = self._rig
        hold_error = ""
        if rig is not None:
            try:
                rig.hold_motion("hardware replay abort")
            except Exception as exc:  # noqa: BLE001
                hold_error = str(exc)
            finally:
                rig.restore_replay_speed()
        if wait and worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        if hold_error:
            with self._lock:
                self._state = "fault"
                self._last_error = hold_error
                self._last_message = "hardware replay abort hold failed"
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            group_payload: dict[str, Any] | None = None
            if self._group is not None:
                group_payload = {
                    "key": self._group.key,
                    "label": self._group.label,
                    "source_path": self._group.source_path,
                    "source_episode_index": int(self._group.source_episode_index),
                    "start_index": int(self._group.start_index),
                    "end_index": int(self._group.end_index),
                    "frame_count": int(self._group.frame_count),
                    "action_hz": float(self._group.action_hz),
                    "arm_joint_target_sides": sorted(self._group.arm_joint_targets),
                    "arm_observation_current_qpos_sides": sorted(self._group.arm_observation_current_qpos),
                    "arm_observation_actual_qpos_sides": sorted(self._group.arm_observation_actual_qpos),
                    "trajectory_source": normalize_replay_trajectory_source(self._settings.trajectory_source),
                    "qpos_init_source": normalize_qpos_init_source(self._settings.qpos_init_source),
                    "end_effectors_enabled": bool(self._settings.end_effectors_enabled),
                }
            return {
                "state": self._state,
                "active_sides": list(self._hardware_config.active_sides),
                "connected": bool(self._rig is not None and self._rig.connected),
                "settings": self._settings.to_dict(),
                "hardware_config": self._hardware_config.to_dict(),
                "frame_index": int(self._frame_index),
                "group": group_payload,
                "preflight": None
                if self._preflight_report is None
                else self._preflight_report.to_dict(),
                "qpos_init": {
                    "plan_summary": self._qpos_init_plan_summary,
                    "step_index": int(self._qpos_init_step_index),
                    "step_count": int(self._qpos_init_step_count),
                },
                "last_error": self._last_error,
                "last_message": self._last_message,
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "worker_alive": bool(self._worker is not None and self._worker.is_alive()),
            }

    def _move_to_start_worker(
        self,
        rig: ReplayHardwareRig,
        group: ReplayTrajectoryGroup,
        target_qpos: dict[str, np.ndarray],
    ) -> None:
        del group
        deadline = time.monotonic() + float(self._settings.move_to_start_timeout_s)
        period = float(self._settings.qpos_init_dt_sec)
        try:
            rig.begin_replay_speed()
            plan = self._build_qpos_init_plan(rig, target_qpos)
            steps = tuple(getattr(plan, "steps", ()))
            summary = plan.summary() if hasattr(plan, "summary") else {"type": type(plan).__name__}
            with self._lock:
                self._qpos_init_plan_summary = _serializable(summary)
                self._qpos_init_step_count = len(steps)
                self._qpos_init_step_index = 0
                self._last_message = f"qpos init plan ready steps={len(steps)}"
            next_time = time.monotonic()
            for step in steps:
                if self._stop_event.is_set():
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out while moving to replay start qpos; "
                        "verify robot state and replay_arm_max_angvel."
                    )
                rig.send_joint_targets(step.qpos_by_side, source="replay_qpos_init")
                with self._lock:
                    self._qpos_init_step_index = int(getattr(step, "index", self._qpos_init_step_index + 1))
                next_time += max(0.001, period)
                sleep_s = next_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            while not self._stop_event.is_set():
                if rig.qpos_reached(
                    target_qpos,
                    tolerance=float(self._settings.qpos_reached_tolerance_rad),
                ):
                    with self._lock:
                        self._state = "start_ready"
                        self._finished_at = time.time()
                        self._last_message = "replay start qpos reached"
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Timed out while waiting for replay start qpos after qpos init plan."
                    )
                rig.send_joint_targets(target_qpos, source="replay_qpos_init_final_hold")
                time.sleep(max(0.001, min(period, 0.05)))
            with self._lock:
                self._state = "aborted"
                self._finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            try:
                rig.hold_motion("hardware replay move_to_start fault")
            except Exception as hold_exc:  # noqa: BLE001
                exc = RuntimeError(f"{exc}; hold failed: {hold_exc}")
            _print_worker_fault("move_to_start", exc)
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
                self._finished_at = time.time()
        finally:
            rig.restore_replay_speed()

    def _replay_worker(self, rig: ReplayHardwareRig, group: ReplayTrajectoryGroup) -> None:
        period = 1.0 / max(1e-6, float(group.action_hz))
        next_time = time.monotonic()
        trajectory_source = normalize_replay_trajectory_source(self._settings.trajectory_source)
        current_index: int | None = None
        try:
            rig.begin_replay_speed()
            for index in range(int(group.frame_count)):
                current_index = int(index)
                if self._stop_event.is_set():
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                if trajectory_source == REPLAY_TRAJECTORY_QPOS_TARGET:
                    rig.send_joint_targets(
                        self._qpos_targets_at_index(group, index),
                        source="replay_qpos_trajectory",
                    )
                else:
                    rig.send_commands(
                        replay_teleop_command_at_index(
                            group,
                            self._hardware_config.active_sides,
                            index=index,
                            include_arms=True,
                            include_end_effectors=bool(self._settings.end_effectors_enabled),
                            source="replay_action_pose",
                            resolver=rig.action_resolver,
                        )
                    )
                with self._lock:
                    self._frame_index = int(index)
                next_time += period
                sleep_s = next_time - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            with self._lock:
                self._state = "finished"
                self._frame_index = max(0, int(group.frame_count) - 1)
                self._finished_at = time.time()
                self._last_message = "hardware replay finished"
        except Exception as exc:  # noqa: BLE001
            try:
                rig.hold_motion("hardware replay fault")
            except Exception as hold_exc:  # noqa: BLE001
                exc = RuntimeError(f"{exc}; hold failed: {hold_exc}")
            if current_index is not None:
                exc = RuntimeError(
                    f"replay frame_index={current_index} "
                    f"trajectory_source={trajectory_source} "
                    f"source={group.source_path}: {exc}"
                )
            _print_worker_fault("replay", exc)
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
                self._finished_at = time.time()
        finally:
            rig.restore_replay_speed()

    def _ensure_idle_locked(self, *, allow_ready: bool = False) -> None:
        busy = {"moving_to_start", "replaying", "connecting"}
        if self._state in busy:
            raise RuntimeError(f"Cannot reconfigure while replay state is {self._state!r}.")
        if not allow_ready and self._state == "start_ready":
            raise RuntimeError("Abort or start the ready replay before reconfiguring.")

    def _ensure_no_worker_locked(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError(f"Replay worker is already active in state {self._state!r}.")

    def _require_rig_locked(self) -> ReplayHardwareRig:
        if self._rig is None:
            raise RuntimeError("Connect hardware before hardware replay preflight.")
        if not self._rig.connected:
            raise RuntimeError("Hardware replay rig is not ready.")
        return self._rig

    def _require_checked_group_locked(self) -> ReplayTrajectoryGroup:
        if self._group is None:
            raise RuntimeError("Run preflight before hardware replay.")
        if self._preflight_report is None or not self._preflight_report.ok:
            raise RuntimeError("Hardware replay preflight has not passed.")
        return self._group

    def _require_output_enabled_locked(self) -> None:
        if not self._settings.send_actions_enabled:
            raise RuntimeError("Arm output is disabled. Enable send_actions before robot motion.")

    def _clear_qpos_init_status_locked(self) -> None:
        self._qpos_init_plan_summary = None
        self._qpos_init_step_index = 0
        self._qpos_init_step_count = 0

    def _default_branch_neutral_qpos(self) -> dict[str, np.ndarray] | None:
        sides = tuple(self._hardware_config.active_sides)
        if sides != ("left", "right") or not bool(self._settings.qpos_init_branch_search):
            return None
        target = load_qpos_target(DEFAULT_QPOS_TARGET_PATH).qpos_by_side()
        return {side: np.asarray(target[side], dtype=float).reshape(ARM_JOINT_DOF) for side in sides}

    def _build_qpos_init_plan(
        self,
        rig: ReplayHardwareRig,
        target_qpos: Mapping[str, np.ndarray],
    ) -> Any:
        current_raw = rig.current_qpos_by_side()
        current: dict[str, np.ndarray] = {}
        for side in self._hardware_config.active_sides:
            qpos = current_raw.get(side)
            if qpos is None:
                raise RuntimeError(f"{side} current hardware qpos is unavailable for qpos init.")
            current[side] = np.asarray(qpos, dtype=float).reshape(ARM_JOINT_DOF)
        plan = build_rm65_qpos_init_plan(
            start_qpos_by_side=current,
            target_qpos_by_side=target_qpos,
            active_sides=self._hardware_config.active_sides,
            qpos_dt_sec=float(self._settings.qpos_init_dt_sec),
            joint_velocity_profile=str(self._settings.qpos_init_joint_velocity_profile),
            qpos_init_max_angvel_rad_s=self._settings.qpos_init_max_angvel_rad_s,
            max_steps_per_side=int(self._settings.qpos_init_max_steps_per_side),
            collision_check=bool(self._settings.qpos_init_collision_check),
            collision_scene_profile=str(self._settings.qpos_init_collision_scene_profile),
            collision_slow_distance_m=float(self._settings.qpos_init_collision_slow_distance_m),
            collision_hold_distance_m=float(self._settings.qpos_init_collision_hold_distance_m),
            collision_stop_distance_m=float(self._settings.qpos_init_collision_stop_distance_m),
            max_joint_abs_deg=float(self._settings.qpos_init_max_joint_abs_deg),
            joint_slow_margin_deg=float(self._settings.qpos_init_joint_slow_margin_deg),
            joint_hold_margin_deg=float(self._settings.qpos_init_joint_hold_margin_deg),
            single_arm_max_qpos_delta_deg=float(self._settings.qpos_init_single_arm_max_qpos_delta_deg),
            branch_search=bool(self._settings.qpos_init_branch_search),
            branch_search_max_branches=int(self._settings.qpos_init_branch_search_max_branches),
            branch_neutral_qpos_by_side=self._default_branch_neutral_qpos(),
        )
        _require_plan_completed(plan)
        return plan

    def _first_qpos_targets_locked(self) -> dict[str, np.ndarray]:
        group = self._require_checked_group_locked()
        targets, sources = select_replay_qpos_init_targets(
            group,
            self._hardware_config.active_sides,
            index=0,
            source=self._settings.qpos_init_source,
        )
        missing = [side for side in self._hardware_config.active_sides if side not in targets]
        if missing:
            raise RuntimeError(
                "Replay start is missing finite qpos init target(s) for "
                f"{missing}; qpos_init_source={self._settings.qpos_init_source!r}."
            )
        self._last_message = (
            "qpos init targets selected: "
            + ", ".join(f"{side}={sources.get(side)}" for side in self._hardware_config.active_sides)
        )
        return {side: target.astype(float, copy=True) for side, target in targets.items()}

    def _qpos_targets_at_index(
        self,
        group: ReplayTrajectoryGroup,
        index: int,
    ) -> dict[str, np.ndarray]:
        targets: dict[str, np.ndarray] = {}
        for side in self._hardware_config.active_sides:
            values = group.arm_joint_targets.get(side)
            if values is None or np.asarray(values).shape[0] <= int(index):
                raise RuntimeError(f"{side} replay is missing arm joint targets.")
            target = np.asarray(values[int(index)], dtype=float).reshape(-1)
            if target.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(target)):
                raise RuntimeError(f"{side} replay qpos target at frame {int(index)} is invalid.")
            targets[side] = target.astype(float, copy=True)
        return targets
