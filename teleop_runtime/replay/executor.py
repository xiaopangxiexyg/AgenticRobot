from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..deploy.schema import ACTION_ARM_POSE_SLICES, ARM_JOINT_DOF
from .checks import ReplayPreflightReport, run_replay_preflight
from .dataset import LoadedReplayEpisode
from .hardware import ReplayHardwareConfig, ReplayHardwareRig, normalize_active_sides
from .trajectory import ReplayGroupSpec, ReplayTrajectoryGroup, build_replay_group_for_episode


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


def _episode_index_or_zero(episode: LoadedReplayEpisode) -> int:
    if episode.summary.episode_index is None:
        return 0
    return int(episode.summary.episode_index)


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
    qpos_reached_tolerance_rad: float = 0.01
    move_to_start_timeout_s: float = 30.0
    tcp_pos_tolerance_m: float = 1e-4
    tcp_rot_tolerance_rad: float = 1e-3
    max_qpos_init_delta_rad: float = float(np.pi)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], current: "ReplayExecutionSettings") -> "ReplayExecutionSettings":
        return cls(
            send_actions_enabled=bool(
                payload.get("send_actions")
                if "send_actions" in payload
                else payload.get("sendActions", current.send_actions_enabled)
            ),
            action_column=str(payload.get("action_column") or payload.get("actionColumn") or current.action_column),
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
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "send_actions_enabled": bool(self.send_actions_enabled),
            "action_column": str(self.action_column),
            "qpos_reached_tolerance_rad": float(self.qpos_reached_tolerance_rad),
            "move_to_start_timeout_s": float(self.move_to_start_timeout_s),
            "tcp_pos_tolerance_m": float(self.tcp_pos_tolerance_m),
            "tcp_rot_tolerance_rad": float(self.tcp_rot_tolerance_rad),
            "max_qpos_init_delta_rad": float(self.max_qpos_init_delta_rad),
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
            if self._rig is not None:
                self._rig.close()
                self._rig = None
            if hardware_config is not None:
                self._hardware_config = hardware_config
            if settings is not None:
                self._settings = settings
            self._group = None
            self._preflight_report = None
            self._frame_index = 0
            self._last_error = ""
            self._last_message = "hardware replay configured"
            self._state = "configured"
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
            self._settings = ReplayExecutionSettings(
                send_actions_enabled=bool(enabled),
                action_column=self._settings.action_column,
                qpos_reached_tolerance_rad=self._settings.qpos_reached_tolerance_rad,
                move_to_start_timeout_s=self._settings.move_to_start_timeout_s,
                tcp_pos_tolerance_m=self._settings.tcp_pos_tolerance_m,
                tcp_rot_tolerance_rad=self._settings.tcp_rot_tolerance_rad,
                max_qpos_init_delta_rad=self._settings.max_qpos_init_delta_rad,
            )
            self._last_message = f"arm output {'enabled' if enabled else 'disabled'}"
            return self.status()

    def clear_preflight(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_idle_locked(allow_ready=True)
            self._group = None
            self._preflight_report = None
            self._frame_index = 0
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
            self._state = "connecting"
            self._last_error = ""
        try:
            self._rig.connect()
        except Exception as exc:
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
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
        )
        with self._lock:
            self._group = group
            self._preflight_report = report
            self._frame_index = 0
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
            if self._state in {"moving_to_start", "replaying", "connecting"}:
                self._state = "aborted"
                self._last_message = "hardware replay aborted"
            rig = self._rig
        if rig is not None:
            rig.restore_replay_speed()
        if wait and worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
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
        period = 1.0 / float(self._hardware_config.arm_control_hz)
        try:
            rig.begin_replay_speed()
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
                        "Timed out while moving to replay start qpos; "
                        "verify robot state and replay_arm_max_angvel."
                    )
                rig.send_joint_targets(target_qpos, source="replay_move_to_start")
                time.sleep(max(0.001, period))
            with self._lock:
                self._state = "aborted"
                self._finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state = "fault"
                self._last_error = str(exc)
                self._finished_at = time.time()
        finally:
            rig.restore_replay_speed()

    def _replay_worker(self, rig: ReplayHardwareRig, group: ReplayTrajectoryGroup) -> None:
        period = 1.0 / max(1e-6, float(group.action_hz))
        next_time = time.monotonic()
        try:
            rig.begin_replay_speed()
            for index in range(int(group.frame_count)):
                if self._stop_event.is_set():
                    with self._lock:
                        self._state = "aborted"
                        self._finished_at = time.time()
                    return
                action = group.actions[index]
                action_by_side = {
                    side: action[ACTION_ARM_POSE_SLICES[side]].astype(np.float32, copy=True)
                    for side in self._hardware_config.active_sides
                }
                rig.send_action_poses(action_by_side, source="replay_trajectory")
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

    def _first_qpos_targets_locked(self) -> dict[str, np.ndarray]:
        group = self._require_checked_group_locked()
        targets: dict[str, np.ndarray] = {}
        for side in self._hardware_config.active_sides:
            values = group.arm_joint_targets.get(side)
            if values is None or np.asarray(values).shape[0] <= 0:
                raise RuntimeError(f"{side} replay is missing arm joint targets.")
            target = np.asarray(values[0], dtype=float).reshape(-1)
            if target.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(target)):
                raise RuntimeError(f"{side} first replay qpos target is invalid.")
            targets[side] = target.astype(float, copy=True)
        return targets
