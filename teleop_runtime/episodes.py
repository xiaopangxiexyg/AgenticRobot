from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from project_paths import teleop_records_root
from rm65_inspire_tac.ik_tuning import weighted_joint_distance_score

from .recording import _to_serializable


EPISODE_FORMAT = "teleop_episode"
EPISODE_FORMAT_VERSION = 1


@dataclass
class EpisodeRecorderConfig:
    enabled: bool = True
    save_channels: bool = True
    save_active_command: bool = True
    save_faults: bool = True


@dataclass
class EpisodeValidationReport:
    episode_dir: Path
    manifest: dict[str, Any]
    frames: list[dict[str, Any]]
    events: list[dict[str, Any]]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_sec: float = 0.0
    average_dt_sec: float | None = None
    average_hz: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class EpisodeStartSideCheck:
    side: str
    current_qpos: np.ndarray
    recorded_qpos: np.ndarray
    abs_joint_error_deg: np.ndarray
    max_abs_joint_error_deg: float
    weighted_joint_error_deg: float
    max_abs_threshold_deg: float
    weighted_threshold_deg: float

    @property
    def within_limits(self) -> bool:
        return (
            self.max_abs_joint_error_deg <= float(self.max_abs_threshold_deg)
            and self.weighted_joint_error_deg <= float(self.weighted_threshold_deg)
        )


def _iso_now_from_timestamp(timestamp: float | None = None) -> str:
    current = time.time() if timestamp is None else float(timestamp)
    return datetime.fromtimestamp(current).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _append_jsonl(handle, payload: Any) -> None:
    handle.write(json.dumps(_to_serializable(payload), ensure_ascii=True) + "\n")
    handle.flush()


def _episode_id_from_index(index: int) -> str:
    return f"episode_{index:06d}"


def _coerce_qpos(qpos: Any) -> np.ndarray:
    array = np.asarray(qpos, dtype=float).reshape(-1)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite 6D joint vector, got shape {array.shape}.")
    return array.copy()


def _coerce_pose(pose: Any) -> np.ndarray:
    array = np.asarray(pose, dtype=float)
    if array.shape != (4, 4) or not np.all(np.isfinite(array)):
        raise ValueError(f"Expected a finite (4, 4) pose matrix, got shape {array.shape}.")
    return array.copy()


def _extract_side_frame(frame: dict[str, Any], side: str) -> dict[str, Any]:
    arms = frame.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("Frame is missing 'arms'.")
    side_payload = arms.get(side)
    if not isinstance(side_payload, dict):
        raise ValueError(f"Frame is missing arms['{side}'].")
    return side_payload


def extract_frame_qpos_pair(frame: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        side: _coerce_qpos(_extract_side_frame(frame, side).get("current_qpos"))
        for side in ("left", "right")
    }


def resolve_episode_dir(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        return candidate
    if candidate.name in {"manifest.json", "frames.jsonl", "events.jsonl"}:
        return candidate.parent
    raise FileNotFoundError(f"Could not resolve episode directory from '{candidate}'.")


def load_episode(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    episode_dir = resolve_episode_dir(path)
    manifest = _read_json(episode_dir / "manifest.json")
    frames = _read_jsonl(episode_dir / "frames.jsonl")
    events = _read_jsonl(episode_dir / "events.jsonl")
    return manifest, frames, events


def validate_episode(path: str | Path) -> EpisodeValidationReport:
    episode_dir = resolve_episode_dir(path)
    manifest, frames, events = load_episode(episode_dir)
    report = EpisodeValidationReport(
        episode_dir=episode_dir,
        manifest=manifest,
        frames=frames,
        events=events,
    )

    if manifest.get("format") != EPISODE_FORMAT:
        report.errors.append(
            f"manifest.format={manifest.get('format')!r} does not match {EPISODE_FORMAT!r}."
        )
    if int(manifest.get("format_version", -1)) != EPISODE_FORMAT_VERSION:
        report.errors.append(
            f"manifest.format_version={manifest.get('format_version')!r} does not match {EPISODE_FORMAT_VERSION}."
        )
    if not frames:
        report.errors.append("Episode does not contain any frames.")
        return report

    last_timestamp = None
    last_relative_time = None
    intervals: list[float] = []
    for index, frame in enumerate(frames):
        if "timestamp" not in frame:
            report.errors.append(f"Frame {index} is missing 'timestamp'.")
            continue
        if "relative_time_sec" not in frame:
            report.errors.append(f"Frame {index} is missing 'relative_time_sec'.")
            continue
        timestamp = float(frame["timestamp"])
        relative_time = float(frame["relative_time_sec"])
        if not np.isfinite(timestamp):
            report.errors.append(f"Frame {index} timestamp is not finite.")
        if not np.isfinite(relative_time):
            report.errors.append(f"Frame {index} relative_time_sec is not finite.")
        if last_timestamp is not None and timestamp <= last_timestamp:
            report.errors.append(
                f"Frame timestamps are not strictly increasing at index {index}: "
                f"{timestamp} <= {last_timestamp}."
            )
        if last_relative_time is not None and relative_time < last_relative_time:
            report.errors.append(
                f"Frame relative_time_sec regressed at index {index}: "
                f"{relative_time} < {last_relative_time}."
            )
        if last_timestamp is not None and timestamp > last_timestamp:
            intervals.append(timestamp - last_timestamp)
        last_timestamp = timestamp
        last_relative_time = relative_time

        if "state" not in frame:
            report.errors.append(f"Frame {index} is missing 'state'.")
        for side in ("left", "right"):
            try:
                side_payload = _extract_side_frame(frame, side)
                _coerce_qpos(side_payload.get("current_qpos"))
                tcp_pose_base = side_payload.get("tcp_pose_base")
                tcp_pose_desk = side_payload.get("tcp_pose_desk")
                if tcp_pose_base is not None:
                    _coerce_pose(tcp_pose_base)
                if tcp_pose_desk is not None:
                    _coerce_pose(tcp_pose_desk)
            except Exception as exc:
                report.errors.append(f"Frame {index} side={side} invalid: {exc}")

    report.duration_sec = float(frames[-1]["relative_time_sec"]) if frames else 0.0
    if intervals:
        report.average_dt_sec = float(np.mean(np.asarray(intervals, dtype=float)))
        if report.average_dt_sec > 1e-9:
            report.average_hz = float(1.0 / report.average_dt_sec)

    manifest_frame_count = manifest.get("frame_count")
    if manifest_frame_count is not None and int(manifest_frame_count) != len(frames):
        report.errors.append(
            f"manifest.frame_count={manifest_frame_count} does not match actual frame count {len(frames)}."
        )
    manifest_event_count = manifest.get("event_count")
    if manifest_event_count is not None and int(manifest_event_count) != len(events):
        report.warnings.append(
            f"manifest.event_count={manifest_event_count} does not match actual event count {len(events)}."
        )
    manifest_duration = manifest.get("duration_sec")
    if manifest_duration is not None and abs(float(manifest_duration) - report.duration_sec) > 0.05:
        report.warnings.append(
            f"manifest.duration_sec={manifest_duration} differs from frame-derived duration {report.duration_sec:.3f}."
        )

    try:
        manifest_start_qpos = manifest.get("start_qpos")
        if isinstance(manifest_start_qpos, dict):
            frame_start_qpos = extract_frame_qpos_pair(frames[0])
            for side in ("left", "right"):
                expected = _coerce_qpos(manifest_start_qpos.get(side))
                if not np.allclose(expected, frame_start_qpos[side], atol=1e-8):
                    report.warnings.append(
                        f"manifest.start_qpos[{side}] does not match first frame current_qpos."
                    )
    except Exception as exc:
        report.warnings.append(f"Could not validate manifest.start_qpos: {exc}")

    metadata = manifest.get("metadata", {})
    if isinstance(metadata, dict) and metadata.get("arm_mode") == "virtual":
        report.warnings.append(
            "Episode was recorded with arm_mode=virtual; real replay may not match hardware-feasible joint timing exactly."
        )

    return report


def require_valid_episode(path: str | Path) -> EpisodeValidationReport:
    report = validate_episode(path)
    if not report.ok:
        message = "\n".join(report.errors)
        raise ValueError(f"Episode validation failed for {report.episode_dir}:\n{message}")
    return report


def episode_start_qpos_pair(
    manifest: dict[str, Any],
    frames: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    start_qpos = manifest.get("start_qpos")
    if isinstance(start_qpos, dict):
        try:
            return {
                side: _coerce_qpos(start_qpos.get(side))
                for side in ("left", "right")
            }
        except Exception:
            pass
    if not frames:
        raise ValueError("Episode does not contain any frames.")
    return extract_frame_qpos_pair(frames[0])


def compare_start_qpos_pair(
    *,
    current_qpos_pair: dict[str, Any],
    recorded_qpos_pair: dict[str, Any],
    joint_distance_weights: np.ndarray | tuple[float, ...],
    max_abs_threshold_deg: float,
    weighted_threshold_deg: float,
) -> dict[str, EpisodeStartSideCheck]:
    checks: dict[str, EpisodeStartSideCheck] = {}
    for side in ("left", "right"):
        current_qpos = _coerce_qpos(current_qpos_pair.get(side))
        recorded_qpos = _coerce_qpos(recorded_qpos_pair.get(side))
        abs_joint_error_deg = np.abs(np.rad2deg(current_qpos - recorded_qpos))
        weighted_joint_error_deg = weighted_joint_distance_score(
            reference_q=recorded_qpos,
            candidate_q=current_qpos,
            joint_distance_weights=joint_distance_weights,
        )
        checks[side] = EpisodeStartSideCheck(
            side=side,
            current_qpos=current_qpos,
            recorded_qpos=recorded_qpos,
            abs_joint_error_deg=abs_joint_error_deg,
            max_abs_joint_error_deg=float(np.max(abs_joint_error_deg)),
            weighted_joint_error_deg=float(weighted_joint_error_deg),
            max_abs_threshold_deg=float(max_abs_threshold_deg),
            weighted_threshold_deg=float(weighted_threshold_deg),
        )
    return checks


class EpisodeRecorder:
    def __init__(
        self,
        root_dir: str | Path | None = None,
        config: EpisodeRecorderConfig | None = None,
    ):
        self.config = config if config is not None else EpisodeRecorderConfig()
        self.root_dir = None if root_dir is None else Path(root_dir)
        self._episode_dir: Path | None = None
        self._frames_handle = None
        self._events_handle = None
        self._manifest: dict[str, Any] | None = None
        self._frame_count = 0
        self._event_count = 0
        self._started_at_unix: float | None = None
        self._first_qpos: dict[str, list[float]] | None = None
        self._last_qpos: dict[str, list[float]] | None = None

    @property
    def episode_dir(self) -> Path | None:
        return self._episode_dir

    @property
    def is_recording(self) -> bool:
        return self._episode_dir is not None

    @property
    def frame_count(self) -> int:
        return int(self._frame_count)

    def _ensure_root_dir(self) -> Path:
        if self.root_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self.root_dir = teleop_records_root() / stamp / "trajectories"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        return self.root_dir

    def _next_episode_dir(self) -> Path:
        root_dir = self._ensure_root_dir()
        index = 1
        while True:
            candidate = root_dir / _episode_id_from_index(index)
            if not candidate.exists():
                return candidate
            index += 1

    def start_episode(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> Path | None:
        if not self.config.enabled:
            return None
        if self.is_recording:
            raise RuntimeError("EpisodeRecorder is already recording.")

        started_at = time.time() if timestamp is None else float(timestamp)
        episode_dir = self._next_episode_dir()
        episode_dir.mkdir(parents=True, exist_ok=False)
        self._episode_dir = episode_dir
        self._frames_handle = (episode_dir / "frames.jsonl").open("a", encoding="utf-8")
        self._events_handle = (episode_dir / "events.jsonl").open("a", encoding="utf-8")
        self._started_at_unix = started_at
        self._frame_count = 0
        self._event_count = 0
        self._first_qpos = None
        self._last_qpos = None
        self._manifest = {
            "format": EPISODE_FORMAT,
            "format_version": EPISODE_FORMAT_VERSION,
            "episode_id": episode_dir.name,
            "episode_dir": str(episode_dir),
            "started_at_unix": started_at,
            "started_at_local": _iso_now_from_timestamp(started_at),
            "frame_count": 0,
            "event_count": 0,
            "duration_sec": 0.0,
            "metadata": _to_serializable(metadata or {}),
        }
        _write_json(episode_dir / "manifest.json", self._manifest)
        self.record_event("episode_started", payload={"metadata": metadata or {}}, timestamp=started_at)
        return episode_dir

    def record_frame(self, frame: dict[str, Any], *, timestamp: float | None = None) -> None:
        if not self.is_recording or not self.config.enabled:
            return
        assert self._frames_handle is not None
        assert self._started_at_unix is not None
        current_time = time.time() if timestamp is None else float(timestamp)
        payload = dict(frame)
        payload["timestamp"] = current_time
        payload["relative_time_sec"] = current_time - self._started_at_unix
        if not self.config.save_channels:
            payload.pop("channels", None)
        if not self.config.save_active_command:
            payload.pop("active_command", None)
        if not self.config.save_faults:
            payload.pop("faults", None)
        _append_jsonl(self._frames_handle, payload)
        self._frame_count += 1
        try:
            qpos_pair = {
                side: _coerce_qpos(_extract_side_frame(payload, side).get("current_qpos")).tolist()
                for side in ("left", "right")
            }
        except Exception:
            qpos_pair = None
        if qpos_pair is not None:
            if self._first_qpos is None:
                self._first_qpos = qpos_pair
            self._last_qpos = qpos_pair

    def record_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> None:
        if not self.is_recording or not self.config.enabled:
            return
        assert self._events_handle is not None
        current_time = time.time() if timestamp is None else float(timestamp)
        event_payload = {
            "timestamp": current_time,
            "relative_time_sec": 0.0
            if self._started_at_unix is None
            else current_time - self._started_at_unix,
            "event_type": str(event_type),
            "payload": payload or {},
        }
        _append_jsonl(self._events_handle, event_payload)
        self._event_count += 1

    def stop_episode(
        self,
        *,
        end_reason: str,
        summary: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> Path | None:
        if not self.is_recording:
            return None
        assert self._episode_dir is not None
        stopped_at = time.time() if timestamp is None else float(timestamp)
        self.record_event(
            "episode_stopped",
            payload={
                "end_reason": end_reason,
                "summary": summary or {},
            },
            timestamp=stopped_at,
        )
        if self._manifest is None:
            raise RuntimeError("Episode manifest is missing.")
        self._manifest.update(
            {
                "stopped_at_unix": stopped_at,
                "stopped_at_local": _iso_now_from_timestamp(stopped_at),
                "frame_count": self._frame_count,
                "event_count": self._event_count,
                "duration_sec": 0.0
                if self._started_at_unix is None
                else stopped_at - self._started_at_unix,
                "end_reason": str(end_reason),
                "summary": _to_serializable(summary or {}),
                "start_qpos": _to_serializable(self._first_qpos),
                "end_qpos": _to_serializable(self._last_qpos),
            }
        )
        _write_json(self._episode_dir / "manifest.json", self._manifest)
        self._close_handles()
        episode_dir = self._episode_dir
        self._episode_dir = None
        self._manifest = None
        self._started_at_unix = None
        self._frame_count = 0
        self._event_count = 0
        self._first_qpos = None
        self._last_qpos = None
        return episode_dir

    def _close_handles(self) -> None:
        if self._frames_handle is not None:
            self._frames_handle.close()
            self._frames_handle = None
        if self._events_handle is not None:
            self._events_handle.close()
            self._events_handle = None

    def close(self) -> None:
        if self.is_recording:
            self.stop_episode(end_reason="recorder_closed")
        self._close_handles()
