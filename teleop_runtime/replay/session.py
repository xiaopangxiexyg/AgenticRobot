from __future__ import annotations

import bisect
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from robot_motion.runtime.lifecycle import run_best_effort_cleanup
from robot_motion.replay.proposal_source import ReplaySourceFrameEmission
from teleop_runtime.action.replay_shadow_frontend import (
    ReplayGuiShadowBinding,
    ReplayGuiShadowConsumer,
    ReplayGuiShadowProfile,
)

from .dataset import LoadedReplayEpisode, ReplayDatasetIndex
from .emission_fanout import ReplaySourceFrameEmissionFanout
from .executor import ReplayExecutionController, ReplayExecutionSettings, _bool_value
from .hardware import ReplayHardwareConfig
from .source_binding import ReplayGuiSourceBinding
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
        action_observer_queue: queue.Queue[ReplaySourceFrameEmission] | None = None,
        replay_action_shadow: bool = False,
        replay_action_shadow_profile: ReplayGuiShadowProfile | None = None,
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
        self.replay_action_shadow = (
            ReplayGuiShadowConsumer(
                profile=(replay_action_shadow_profile or ReplayGuiShadowProfile())
            )
            if replay_action_shadow
            else None
        )
        self.replay_action_fanout: ReplaySourceFrameEmissionFanout | None = None
        if self.replay_action_shadow is not None and action_observer_queue is not None:
            self.replay_action_fanout = ReplaySourceFrameEmissionFanout(
                {
                    "gui_shadow": self.replay_action_shadow.emission_queue,
                    "external_passive_observer": action_observer_queue,
                }
            )
            observer_queue = self.replay_action_fanout
        elif self.replay_action_shadow is not None:
            observer_queue = self.replay_action_shadow.emission_queue
        else:
            observer_queue = action_observer_queue
        self.executor = ReplayExecutionController(
            hardware_config=hardware_config,
            settings=execution_settings
            if execution_settings is not None
            else ReplayExecutionSettings(end_effectors_enabled=bool(end_effectors_enabled)),
            action_observer_queue=observer_queue,
        )
        if send_actions:
            self.executor.set_send_actions(True)
        if dataset_root is not None:
            self.load_dataset(dataset_root)

    def close(self) -> None:
        run_best_effort_cleanup(
            "replay_gui_session.close",
            (
                ("media", self._close_media),
                ("executor", self.executor.close),
                ("replay_action_shadow", self._close_replay_action_shadow),
            ),
        )

    def _close_replay_action_shadow(self) -> None:
        if self.replay_action_shadow is not None:
            self.replay_action_shadow.close()

    def _clear_replay_shadow_binding(self, *, reason: str) -> None:
        shadow = self.replay_action_shadow
        if shadow is None:
            return
        try:
            shadow.clear_binding(reason=reason)
        except Exception as exc:  # passive diagnostics never block GUI lifecycle
            with self._lock:
                self._log_locked(
                    f"replay action shadow clear failed: {type(exc).__name__}: {exc}"
                )

    def _bind_replay_shadow(
        self,
        episode: LoadedReplayEpisode,
        execution_status: Mapping[str, Any],
    ) -> None:
        shadow = self.replay_action_shadow
        if shadow is None:
            return
        preflight = execution_status.get("preflight")
        if not isinstance(preflight, Mapping) or not bool(preflight.get("ok")):
            self._clear_replay_shadow_binding(reason="preflight_not_ready")
            return
        source_binding = self.executor.checked_source_binding()
        if source_binding.source_episode_id != episode.summary.episode_id:
            raise RuntimeError("Replay source binding belongs to another GUI episode.")
        binding = ReplayGuiShadowBinding(
            source_path=source_binding.expected_source_path,
            episode_id=source_binding.source_episode_id,
            action_field=str(source_binding.source_contract.action_field),
            source_hz=source_binding.source_hz,
            dataset_metadata=source_binding.metadata_copy(),
        )
        shadow.bind(binding)
        with self._lock:
            self._log_locked(
                "replay action passive shadow bound; legacy replay remains execution owner"
            )

    def checked_replay_source_binding(self) -> ReplayGuiSourceBinding:
        """Expose lower-owned typed evidence without exposing the replay group."""

        return self.executor.checked_source_binding()

    def _close_media(self) -> None:
        with self._lock:
            self._close_media_locked()

    def position_collision_debug_snapshot(self, *, limit: int = 100) -> dict[str, Any]:
        return self.executor.position_collision_debug_snapshot(limit=limit)

    def _close_media_locked(self) -> None:
        steps = [
            (f"tactile_reader[{side}]", reader.close)
            for side, reader in reversed(tuple(self._tactile_readers.items()))
        ]
        steps.extend(
            (f"camera_track[{camera_key}]", track.close)
            for camera_key, track in reversed(tuple(self._camera_tracks.items()))
        )
        try:
            run_best_effort_cleanup("replay_gui_session.media.close", steps)
        finally:
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
            "arm_control_mode",
            "armControlMode",
            "arm_control_hz",
            "armControlHz",
            "arm_canfd_follow",
            "armCanfdFollow",
            "force_tare_sec",
            "forceTareSec",
            "strict_tool_profile",
            "strictToolProfile",
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
            "gripper_id_by_side",
            "gripperIdBySide",
            "gripper_baudrate_by_side",
            "gripperBaudrateBySide",
            "gripper_timeout_by_side",
            "gripperTimeoutBySide",
            "left_gripper_id",
            "leftGripperId",
            "right_gripper_id",
            "rightGripperId",
            "left_gripper_baudrate",
            "leftGripperBaudrate",
            "right_gripper_baudrate",
            "rightGripperBaudrate",
            "left_gripper_timeout",
            "leftGripperTimeout",
            "right_gripper_timeout",
            "rightGripperTimeout",
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

    def _configure_connect_for_preflight(
        self,
        payload: Mapping[str, Any],
        episode: LoadedReplayEpisode,
    ) -> None:
        if self._payload_has_hardware_config(payload):
            self.executor.configure_from_payload(payload)
            with self._lock:
                self._log_locked("hardware replay configured for preflight")
        mode_validator = getattr(
            self.executor,
            "validate_episode_arm_control_mode",
            None,
        )
        if callable(mode_validator):
            mode_contract = mode_validator(episode)
            if mode_contract.get("warning"):
                with self._lock:
                    self._log_locked(str(mode_contract["warning"]))
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
        self._clear_replay_shadow_binding(reason="dataset_reloaded")
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
        self._clear_replay_shadow_binding(reason="episode_changed")
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
            self._clear_replay_shadow_binding(reason="hardware_configuration_changed")
            status = self.executor.configure_from_payload(payload)
            with self._lock:
                self._log_locked("hardware replay configured")
            return self.state()
        if action == "set_send_actions":
            enabled = _bool_value(
                payload.get("enabled") if "enabled" in payload else payload.get("send_actions"),
                name="send_actions",
            )
            self.executor.set_send_actions(enabled)
            with self._lock:
                self._log_locked(f"arm output {'enabled' if enabled else 'disabled'}")
            return self.state()
        if action == "connect_hardware":
            with self._lock:
                episode = self.episode
            if episode is not None:
                mode_validator = getattr(
                    self.executor,
                    "validate_episode_arm_control_mode",
                    None,
                )
                if callable(mode_validator):
                    mode_validator(episode)
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
                self._configure_connect_for_preflight(payload, episode)
                status = self.executor.preflight(episode)
            except Exception as exc:
                with self._lock:
                    self._log_locked(f"hardware preflight failed before checks: {exc}")
                raise
            with self._lock:
                self._log_locked(self._preflight_log_message(status))
            try:
                self._bind_replay_shadow(episode, status)
            except Exception as exc:  # passive shadow cannot fail legacy preflight
                self._clear_replay_shadow_binding(reason="binding_failed")
                with self._lock:
                    self._log_locked(
                        "replay action shadow binding failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
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
                "replay_action_shadow": (
                    {"enabled": False, "authoritative": False}
                    if self.replay_action_shadow is None
                    else self.replay_action_shadow.status()
                ),
                "replay_action_fanout": (
                    {"enabled": False, "authoritative": False}
                    if self.replay_action_fanout is None
                    else self.replay_action_fanout.status()
                ),
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
