from __future__ import annotations

import bisect
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .dataset import LoadedReplayEpisode, ReplayDatasetIndex
from .executor import ReplayExecutionController, ReplayExecutionSettings
from .hardware import ReplayHardwareConfig
from .media import (
    CameraVideoTrack,
    TactileEpisodeReader,
    build_camera_tracks,
    build_tactile_readers,
    placeholder_jpeg,
    tactile_frame_jpeg,
)


_HARDWARE_BUSY_STATES = frozenset({"connecting", "moving_to_start", "replaying"})


class ReplayGuiSession:
    def __init__(
        self,
        dataset_root: str | Path | None = None,
        *,
        hardware_config: ReplayHardwareConfig | None = None,
        execution_settings: ReplayExecutionSettings | None = None,
        send_actions: bool = False,
        end_effectors_enabled: bool = False,
    ):
        self._lock = threading.RLock()
        self.dataset_root: Path | None = None
        self.index: ReplayDatasetIndex | None = None
        self.episode: LoadedReplayEpisode | None = None
        self.frame_index = 0
        self.playing = False
        self.speed = 1.0
        self._play_start_wall = time.monotonic()
        self._play_start_time_sec = 0.0
        self._camera_tracks: dict[str, CameraVideoTrack] = {}
        self._tactile_readers: dict[str, TactileEpisodeReader] = {}
        self._messages: list[str] = []
        self.executor = ReplayExecutionController(
            hardware_config=hardware_config,
            settings=execution_settings
            if execution_settings is not None
            else ReplayExecutionSettings(end_effectors_enabled=bool(end_effectors_enabled)),
        )
        if send_actions:
            self.executor.set_send_actions(True)
        if dataset_root is not None:
            self.load_dataset(dataset_root)

    def close(self) -> None:
        with self._lock:
            self._close_media_locked()
        self.executor.close()

    def _close_media_locked(self) -> None:
        for track in self._camera_tracks.values():
            track.close()
        for reader in self._tactile_readers.values():
            reader.close()
        self._camera_tracks.clear()
        self._tactile_readers.clear()

    def _log_locked(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self._messages.append(f"[{timestamp}] {message}")
        self._messages = self._messages[-80:]

    @staticmethod
    def _payload_has_hardware_config(payload: Mapping[str, Any]) -> bool:
        keys = {
            "active_sides",
            "activeSides",
            "left_arm_ip",
            "leftArmIp",
            "right_arm_ip",
            "rightArmIp",
            "left_tool_profile",
            "leftToolProfile",
            "right_tool_profile",
            "rightToolProfile",
            "arm_control_hz",
            "armControlHz",
            "connect_hardware",
            "connectHardware",
            "end_effectors_enabled",
            "endEffectorsEnabled",
            "end_effector_backend_by_side",
            "endEffectorBackendBySide",
            "left_end_effector_backend",
            "leftEndEffectorBackend",
            "right_end_effector_backend",
            "rightEndEffectorBackend",
            "left_hand_ip",
            "leftHandIp",
            "right_hand_ip",
            "rightHandIp",
            "left_gripper_port",
            "leftGripperPort",
            "right_gripper_port",
            "rightGripperPort",
        }
        return any(key in payload for key in keys)

    @staticmethod
    def _preflight_log_message(status: Mapping[str, Any]) -> str:
        report = status.get("preflight") if isinstance(status, Mapping) else None
        if not isinstance(report, Mapping):
            return "hardware preflight finished without report"
        checks = report.get("checks")
        checks = checks if isinstance(checks, list) else []
        blocked = [
            check
            for check in checks
            if isinstance(check, Mapping) and not bool(check.get("ok"))
        ]
        warnings = [
            check
            for check in checks
            if isinstance(check, Mapping) and bool(check.get("ok")) and check.get("level") == "warning"
        ]
        if report.get("ok"):
            suffix = f" checks={len(checks)}"
            if warnings:
                suffix += f" warnings={len(warnings)}"
            return "hardware preflight ok" + suffix
        names = ", ".join(str(check.get("name")) for check in blocked[:5])
        if len(blocked) > 5:
            names += f", ... +{len(blocked) - 5}"
        first_message = str(blocked[0].get("message") or "") if blocked else ""
        return f"hardware preflight failed blocked={len(blocked)} [{names}] {first_message}".strip()

    def _configure_connect_for_preflight(self, payload: Mapping[str, Any]) -> None:
        if self._payload_has_hardware_config(payload):
            self.executor.configure_from_payload(payload)
            with self._lock:
                self._log_locked("hardware replay configured for preflight")
        status = self.executor.status()
        if not bool(status.get("connected")):
            connect_status = self.executor.connect()
            with self._lock:
                self._log_locked(connect_status.get("last_message") or "hardware connected")

    def _require_hardware_not_busy_locked(self, action: str) -> None:
        status = self.executor.status()
        state = str(status.get("state") or "")
        if state in _HARDWARE_BUSY_STATES or bool(status.get("worker_alive")):
            raise RuntimeError(f"Abort or wait for hardware replay before {action}; current state is {state!r}.")

    def load_dataset(self, dataset_root: str | Path, *, episode: str | int | None = None) -> dict[str, Any]:
        with self._lock:
            self._require_hardware_not_busy_locked("loading a dataset")
            self._close_media_locked()
            self.playing = False
            self.dataset_root = Path(dataset_root).expanduser().resolve()
            self.index = ReplayDatasetIndex(self.dataset_root)
            episodes = self.index.available_episodes()
            self.episode = None
            self.frame_index = 0
            self._log_locked(f"dataset loaded: {self.dataset_root}")
            if episode is not None or episodes:
                token = episode if episode is not None else ""
                self._select_episode_locked(token)
            return self.state()

    def select_episode(self, episode: str | int) -> dict[str, Any]:
        with self._lock:
            self._select_episode_locked(episode)
            return self.state()

    def _select_episode_locked(self, episode: str | int) -> None:
        if self.index is None:
            raise RuntimeError("Load a replay dataset before selecting an episode.")
        self._require_hardware_not_busy_locked("selecting an episode")
        self._close_media_locked()
        self.episode = self.index.load_episode(episode)
        self.frame_index = 0
        self.playing = False
        self._play_start_time_sec = 0.0
        self._camera_tracks = build_camera_tracks(self.episode)
        self._tactile_readers = build_tactile_readers(self.episode)
        try:
            self.executor.clear_preflight()
        except Exception as exc:  # noqa: BLE001
            self._log_locked(f"hardware preflight clear failed: {exc}")
        self._log_locked(
            f"episode selected: {self.episode.summary.episode_id} "
            f"frames={self.episode.frame_count} hz={self.episode.nominal_hz:.3f}"
        )

    def play(self) -> dict[str, Any]:
        with self._lock:
            if self.episode is not None and self.episode.frame_count > 0:
                self._update_playhead_locked()
                self.playing = True
                self._play_start_wall = time.monotonic()
                self._play_start_time_sec = self._current_time_sec_locked()
            return self.state()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._update_playhead_locked()
            self.playing = False
            return self.state()

    def seek(self, frame_index: int) -> dict[str, Any]:
        with self._lock:
            self._set_frame_index_locked(frame_index)
            self._reset_play_anchor_locked()
            return self.state()

    def step(self, delta: int) -> dict[str, Any]:
        with self._lock:
            self.playing = False
            self._set_frame_index_locked(self.frame_index + int(delta))
            self._reset_play_anchor_locked()
            return self.state()

    def set_speed(self, speed: float) -> dict[str, Any]:
        with self._lock:
            value = float(speed)
            if value <= 0.0:
                raise ValueError("Replay speed must be positive.")
            self._update_playhead_locked()
            self.speed = max(0.05, min(16.0, value))
            self._reset_play_anchor_locked()
            return self.state()

    def dispatch(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        action = str(payload.get("type") or payload.get("action") or "").strip()
        if action == "load_dataset":
            return self.load_dataset(payload.get("dataset_root") or "", episode=payload.get("episode"))
        if action == "select_episode":
            return self.select_episode(payload.get("episode") or "")
        if action == "play":
            return self.play()
        if action == "pause":
            return self.pause()
        if action == "toggle_play":
            with self._lock:
                playing = bool(self.playing)
            return self.pause() if playing else self.play()
        if action == "seek":
            return self.seek(int(payload.get("frame") or payload.get("frame_index") or 0))
        if action == "step":
            return self.step(int(payload.get("delta") or 0))
        if action == "speed":
            return self.set_speed(float(payload.get("speed") or 1.0))
        if action == "configure_hardware":
            status = self.executor.configure_from_payload(payload)
            with self._lock:
                self._log_locked("hardware replay configured")
            return self.state()
        if action == "set_send_actions":
            enabled = bool(payload.get("enabled") if "enabled" in payload else payload.get("send_actions"))
            self.executor.set_send_actions(enabled)
            with self._lock:
                self._log_locked(f"arm output {'enabled' if enabled else 'disabled'}")
            return self.state()
        if action == "connect_hardware":
            status = self.executor.connect()
            with self._lock:
                self._log_locked(status.get("last_message") or "hardware connected")
            return self.state()
        if action == "preflight_hardware":
            with self._lock:
                episode = self.episode
            if episode is None:
                raise RuntimeError("Select an episode before hardware replay preflight.")
            try:
                self._configure_connect_for_preflight(payload)
                status = self.executor.preflight(episode)
            except Exception as exc:
                with self._lock:
                    self._log_locked(f"hardware preflight failed before checks: {exc}")
                raise
            with self._lock:
                self._log_locked(self._preflight_log_message(status))
            return self.state()
        if action == "move_to_start":
            self.executor.move_to_start()
            with self._lock:
                self.playing = False
                self._set_frame_index_locked(0)
                self._log_locked("moving hardware to replay start")
            return self.state()
        if action == "start_hardware_replay":
            self.executor.start()
            with self._lock:
                self.playing = False
                self._set_frame_index_locked(0)
                self._log_locked("hardware replay started")
            return self.state()
        if action == "abort_hardware_replay":
            self.executor.abort()
            with self._lock:
                self.playing = False
                self._log_locked("hardware replay abort requested")
            return self.state()
        raise ValueError(f"Unknown replay GUI action: {action!r}")

    def _sync_execution_playhead_locked(self) -> None:
        status = self.executor.status()
        if status.get("state") not in {"moving_to_start", "start_ready", "replaying", "finished"}:
            return
        if self.episode is None or self.episode.frame_count <= 0:
            return
        frame = int(status.get("frame_index") or 0)
        self._set_frame_index_locked(frame)

    def _set_frame_index_locked(self, frame_index: int) -> None:
        if self.episode is None or self.episode.frame_count <= 0:
            self.frame_index = 0
            return
        self.frame_index = max(0, min(int(frame_index), self.episode.frame_count - 1))

    def _current_time_sec_locked(self) -> float:
        if self.episode is None or not self.episode.frame_times_sec:
            return 0.0
        index = max(0, min(self.frame_index, len(self.episode.frame_times_sec) - 1))
        return float(self.episode.frame_times_sec[index])

    def _reset_play_anchor_locked(self) -> None:
        self._play_start_wall = time.monotonic()
        self._play_start_time_sec = self._current_time_sec_locked()

    def _update_playhead_locked(self) -> None:
        if not self.playing or self.episode is None or self.episode.frame_count <= 0:
            return
        times = self.episode.frame_times_sec
        if not times:
            return
        elapsed = max(0.0, time.monotonic() - self._play_start_wall) * float(self.speed)
        target_time = float(self._play_start_time_sec) + elapsed
        index = bisect.bisect_right(times, target_time) - 1
        self._set_frame_index_locked(index)
        if self.frame_index >= self.episode.frame_count - 1 and target_time >= float(times[-1]):
            self.playing = False
            self._reset_play_anchor_locked()

    def _row_summary_locked(self) -> dict[str, Any]:
        if self.episode is None or not self.episode.rows:
            return {}
        row = self.episode.row_at(self.frame_index)
        summary: dict[str, Any] = {}
        for key in ("frame_index", "timestamp", "control_hz", "session_state", "active_sides"):
            if key in row:
                summary[key] = row.get(key)
        for side in ("left", "right"):
            for key in (
                f"action.replay.{side}_arm_joint_target",
                f"action.replay.{side}_hand_joint_target",
                f"observation.hand.{side}.tactile_frame_index",
            ):
                value = row.get(key)
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    summary[key] = [float(x) for x in value[:8]]
                else:
                    summary[key] = value
        return summary

    def state(self) -> dict[str, Any]:
        with self._lock:
            self._update_playhead_locked()
            self._sync_execution_playhead_locked()
            episodes = self.index.available_episodes() if self.index is not None else []
            episode_payload = self.episode.to_dict() if self.episode is not None else None
            frame_count = self.episode.frame_count if self.episode is not None else 0
            frame_time = self._current_time_sec_locked()
            camera_keys = list(self.episode.summary.camera_keys) if self.episode is not None else []
            if self.episode is not None:
                camera_keys = sorted(set(camera_keys).union(self._camera_tracks))
            tactile_sides = list(self.episode.summary.tactile_sides) if self.episode is not None else []
            if self.episode is not None:
                tactile_sides = sorted(set(tactile_sides).union(self._tactile_readers))
            return {
                "dataset_root": str(self.dataset_root) if self.dataset_root is not None else "",
                "dataset_loaded": self.index is not None,
                "episodes": [summary.to_dict() for summary in episodes],
                "episode": episode_payload,
                "frame_index": int(self.frame_index),
                "frame_count": int(frame_count),
                "frame_time_sec": float(frame_time),
                "playing": bool(self.playing),
                "speed": float(self.speed),
                "camera_keys": camera_keys,
                "tactile_sides": tactile_sides,
                "row": self._row_summary_locked(),
                "execution": self.executor.status(),
                "messages": list(self._messages),
            }

    def camera_jpeg(self, camera_key: str) -> bytes:
        with self._lock:
            self._update_playhead_locked()
            if self.episode is None or not self.episode.rows:
                return placeholder_jpeg(str(camera_key), ["no episode selected"])
            row = self.episode.row_at(self.frame_index)
            track = self._camera_tracks.get(str(camera_key))
            if track is None:
                return placeholder_jpeg(str(camera_key), ["no recorded video path"])
            return track.frame_jpeg(row, replay_index=self.frame_index)

    def tactile_jpeg(self, side: str) -> bytes:
        with self._lock:
            self._update_playhead_locked()
            if self.episode is None or not self.episode.rows:
                return placeholder_jpeg(f"{side} tactile", ["no episode selected"])
            row = self.episode.row_at(self.frame_index)
            reader = self._tactile_readers.get(str(side))
            if reader is None:
                return placeholder_jpeg(f"{side} tactile", ["no tactile stream"])
            frame = reader.frame_for(row, replay_index=self.frame_index)
            return tactile_frame_jpeg(frame)
