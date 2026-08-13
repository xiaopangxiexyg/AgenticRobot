"""Read-only VIVE Home-deflection application state.

The controller owns one VIVE receiver and a display-only Home anchor.  It does
not construct a simulator, policy, composition runtime, execution target,
authority, recorder, or robot backend.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from control_composition.observability import (
    project_pose_matrix_deflection,
    projectable_pose_matrix,
)


VIVE_DEFLECTION_STATUS_SCHEMA_V1 = (
    "dexteleop.control_station.vive_deflection_status.v1"
)
VIVE_RETARGET_AXIS_SEMANTICS = {
    "positive_x": "forward",
    "negative_x": "backward",
    "positive_y": "left",
    "negative_y": "right",
    "positive_z": "up",
    "negative_z": "down",
}


def _coordinate_axis_semantics(
    calibration: Any,
) -> tuple[dict[str, str], str, str | None]:
    """Return the physical signed-axis meaning of a stored profile."""

    metadata = getattr(calibration, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        return dict(VIVE_RETARGET_AXIS_SEMANTICS), "canonical", None
    if bool(metadata.get("virtual_desk_scene_alignment_defaulted")):
        return (
            dict(VIVE_RETARGET_AXIS_SEMANTICS),
            "legacy_compatibility_corrected",
            None,
        )
    try:
        yaw_deg = float(metadata.get("virtual_desk_post_yaw_deg", 0.0))
    except (TypeError, ValueError):
        yaw_deg = 0.0
    if not math.isfinite(yaw_deg):
        return (
            {"positive_z": "up", "negative_z": "down"},
            "invalid_post_yaw",
            "profile 的 post-yaw 非有限值；只声明上下，XY 仅显示轴名。",
        )
    normalized = yaw_deg % 360.0
    quarter_turn = int(round(normalized / 90.0)) % 4
    if not math.isclose(
        normalized,
        float(quarter_turn * 90),
        abs_tol=1e-6,
    ):
        return (
            {"positive_z": "up", "negative_z": "down"},
            "non_cardinal_post_yaw",
            "profile 使用非 90° 倍数的 post-yaw；只声明上下，XY 仅显示轴名。",
        )
    horizontal = {
        0: {
            "positive_x": "forward",
            "negative_x": "backward",
            "positive_y": "left",
            "negative_y": "right",
        },
        1: {
            "positive_x": "left",
            "negative_x": "right",
            "positive_y": "backward",
            "negative_y": "forward",
        },
        2: {
            "positive_x": "backward",
            "negative_x": "forward",
            "positive_y": "right",
            "negative_y": "left",
        },
        3: {
            "positive_x": "right",
            "negative_x": "left",
            "positive_y": "forward",
            "negative_y": "backward",
        },
    }[quarter_turn]
    semantics = {
        **horizontal,
        "positive_z": "up",
        "negative_z": "down",
    }
    if quarter_turn == 0:
        return semantics, "canonical", None
    return (
        semantics,
        "explicit_post_yaw",
        (
            f"profile 显式 post-yaw={yaw_deg:g}°；方向标签按实际物理轴显示。"
            "若要求 +X=前、+Y=左，请用修正后的标定流程重新保存。"
        ),
    )


@dataclass(frozen=True)
class ViveDeflectionConfig:
    source_role: str = "left_wrist"
    tracker_name: str = "tracker_1"
    legacy_index: int | None = 0
    vive_host: str = "0.0.0.0"
    vive_port: int = 9999
    vive_clock_sync_port: int | None = 10001
    freshness_sec: float = 0.5
    coordinate_profile_path: Path | None = Path("latest")

    def __post_init__(self) -> None:
        if self.source_role not in {"left_wrist", "right_wrist"}:
            raise ValueError("source_role must be left_wrist or right_wrist")
        if not str(self.tracker_name).strip():
            raise ValueError("tracker_name must be non-empty")
        if self.legacy_index is not None and int(self.legacy_index) < 0:
            raise ValueError("legacy_index must be non-negative or None")
        for name in ("vive_port", "vive_clock_sync_port"):
            value = getattr(self, name)
            if value is not None and not 0 <= int(value) <= 65535:
                raise ValueError(f"{name} must be in [0, 65535]")
        freshness = float(self.freshness_sec)
        if not math.isfinite(freshness) or freshness <= 0.0:
            raise ValueError("freshness_sec must be finite and positive")
        if self.coordinate_profile_path is not None:
            object.__setattr__(
                self,
                "coordinate_profile_path",
                Path(self.coordinate_profile_path).expanduser(),
            )


def _build_vive_source(config: ViveDeflectionConfig) -> Any:
    from teleop_runtime.sources.pose.vive import VivePoseSource, ViveTrackerConfig

    return VivePoseSource(
        tracker_configs=[
            ViveTrackerConfig(
                name=config.tracker_name,
                side=None,
                role=config.source_role,
                channel_name="wrist_pose",
                legacy_index=config.legacy_index,
            )
        ],
        host=config.vive_host,
        port=config.vive_port,
        clock_sync_port=config.vive_clock_sync_port,
        # This passive viewer consumes only the latest pose. It does not need
        # the intent-estimation history used by HIL/control runtimes.
        history_window_sec=0.0,
        motion_window_sec=0.0,
        name="vive_deflection_pose",
    )


def _load_coordinate_profile_for_source(
    name_or_path: str | Path,
    *,
    source_role: str,
) -> tuple[Any, Path]:
    from .vive_calibration_profiles import (
        LATEST_VIVE_COORDINATE_PROFILE,
        list_coordinate_profiles,
        load_coordinate_profile,
    )

    requested = str(name_or_path).strip()
    if requested.lower() != LATEST_VIVE_COORDINATE_PROFILE:
        return load_coordinate_profile(name_or_path)
    source_side = source_role.removesuffix("_wrist")
    for summary in list_coordinate_profiles():
        if summary.valid and source_side in summary.active_sides:
            return load_coordinate_profile(summary.path)
    raise FileNotFoundError(
        "No valid Control Station VIVE coordinate profile is available for "
        f"{source_role}."
    )


def _coordinate_reference_frame(calibration: Any, *, source_role: str) -> str:
    active_sides = {
        str(getattr(arm, "side", "")).strip()
        for arm in getattr(calibration, "arms", {}).values()
    }
    source_side = source_role.removesuffix("_wrist")
    if source_side not in active_sides:
        raise ValueError(f"Coordinate profile does not provide {source_role}.")
    return str(
        getattr(
            getattr(calibration, "desk_frame", None),
            "name",
            "vive_retarget_frame",
        )
        or "vive_retarget_frame"
    )

def _finite_pose(channel: Any) -> Any | None:
    try:
        return projectable_pose_matrix(
            getattr(channel, "value", None),
            name="vive_channel_pose",
        )
    except (TypeError, ValueError):
        return None


def _channel_age(channel: Any, *, now: float) -> float | None:
    metadata = getattr(channel, "metadata", {}) or {}
    timestamp = (
        metadata.get("receiver_timestamp")
        if isinstance(metadata, Mapping)
        else None
    )
    if timestamp is None:
        timestamp = getattr(channel, "timestamp", None)
    try:
        value = float(timestamp)
        return max(0.0, float(now) - value) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class ViveDeflectionController:
    """Lifecycle and Home state for the standalone passive viewer."""

    def __init__(
        self,
        config: ViveDeflectionConfig,
        *,
        source: Any | None = None,
        coordinate_calibration: Any | None = None,
        clock: Any = time.time,
        monotonic_clock: Any = time.perf_counter,
    ) -> None:
        self.config = config
        self.source = _build_vive_source(config) if source is None else source
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self._lock = threading.RLock()
        self.coordinate_calibration = coordinate_calibration
        self.coordinate_profile_path: Path | None = None
        if (
            self.coordinate_calibration is None
            and config.coordinate_profile_path is not None
        ):
            self.coordinate_calibration, self.coordinate_profile_path = (
                _load_coordinate_profile_for_source(
                    config.coordinate_profile_path,
                    source_role=config.source_role,
                )
            )
        self.display_reference_frame = "vive_world"
        self.translation_axis_semantics: dict[str, str] = {}
        self.axis_semantics_status = "raw_vive_world"
        self.axis_semantics_warning: str | None = None
        if self.coordinate_calibration is not None:
            self.display_reference_frame = _coordinate_reference_frame(
                self.coordinate_calibration,
                source_role=config.source_role,
            )
            (
                self.translation_axis_semantics,
                self.axis_semantics_status,
                self.axis_semantics_warning,
            ) = _coordinate_axis_semantics(self.coordinate_calibration)
        self._started = False
        self._home_pose: Any | None = None
        self._home_generation = 0
        self._home_sample_seq: Any = None
        self._cached_source_token: tuple[Any, Any] | None = None
        self._cached_channel: Any | None = None
        self._cached_pose: Any | None = None

    def apply_coordinate_profile(
        self,
        name_or_path: str | Path,
    ) -> dict[str, Any]:
        calibration, path = _load_coordinate_profile_for_source(
            name_or_path,
            source_role=self.config.source_role,
        )
        reference_frame = _coordinate_reference_frame(
            calibration,
            source_role=self.config.source_role,
        )
        axis_semantics, axis_status, axis_warning = _coordinate_axis_semantics(
            calibration
        )
        with self._lock:
            self.coordinate_calibration = calibration
            self.coordinate_profile_path = path
            self.display_reference_frame = reference_frame
            self.translation_axis_semantics = axis_semantics
            self.axis_semantics_status = axis_status
            self.axis_semantics_warning = axis_warning
            self._home_pose = None
            self._home_sample_seq = None
            self._cached_source_token = None
            self._cached_channel = None
            self._cached_pose = None
        return {
            "status": "coordinate_profile_applied",
            "coordinate_profile_path": str(path),
            "display_reference_frame": reference_frame,
            "viewer_home_reset": True,
        }

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._cached_source_token = None
            self._cached_channel = None
            self._cached_pose = None
        try:
            self.source.start()
        except BaseException:
            with self._lock:
                self._started = False
            raise

    def close(self) -> None:
        with self._lock:
            started = self._started
            self._started = False
        if started:
            self.source.close()

    def _latest(
        self,
        *,
        now: float | None = None,
        source_stats: Mapping[str, Any] | None = None,
    ) -> tuple[Any | None, Any | None, float | None]:
        token: tuple[Any, Any] | None = None
        if source_stats is not None and source_stats.get("packet_count") is not None:
            token = (
                source_stats.get("packet_count"),
                source_stats.get("source_mapping_generation"),
            )
        if token is not None:
            with self._lock:
                cache_hit = token == self._cached_source_token
                if cache_hit:
                    channel = self._cached_channel
                    pose = (
                        None
                        if self._cached_pose is None
                        else self._cached_pose.copy()
                    )
                else:
                    channel = None
                    pose = None
            if cache_hit:
                checked_at = float(self.clock()) if now is None else float(now)
                return channel, pose, _channel_age(channel, now=checked_at)
        channels = self.source.get_latest_channels()
        channel = channels.get("wrist_pose")
        pose = _finite_pose(channel)
        if pose is not None and self.coordinate_calibration is not None:
            pose = projectable_pose_matrix(
                self.coordinate_calibration.world_to_vive_retarget(
                    pose,
                    tracker_role=self.config.source_role,
                ),
                name="vive_retarget_pose",
            )
        if token is not None:
            with self._lock:
                self._cached_source_token = token
                self._cached_channel = channel
                self._cached_pose = None if pose is None else pose.copy()
        checked_at = float(self.clock()) if now is None else float(now)
        return (
            channel,
            pose,
            _channel_age(channel, now=checked_at),
        )

    def _source_stats(self) -> dict[str, Any]:
        try:
            return dict(self.source.debug_stats())
        except Exception:
            return {}

    def recenter(self) -> dict[str, Any]:
        with self._lock:
            return self._recenter_locked()

    def _recenter_locked(self) -> dict[str, Any]:
        channel, pose, age = self._latest(
            now=float(self.clock()),
            source_stats=self._source_stats(),
        )
        failures: list[str] = []
        if channel is None:
            failures.append("channel_missing")
        elif not bool(getattr(channel, "valid", False)):
            failures.append("channel_invalid")
        if pose is None:
            failures.append("pose_unprojectable")
        if age is None:
            failures.append("age_unavailable")
        elif age > self.config.freshness_sec:
            failures.append(
                f"pose_stale:{age:.3f}s>{self.config.freshness_sec:.3f}s"
            )
        if failures:
            raise RuntimeError(
                "recenter unavailable: " + ",".join(failures)
            )
        metadata = getattr(channel, "metadata", {}) or {}
        with self._lock:
            self._home_pose = pose.copy()
            self._home_generation += 1
            self._home_sample_seq = (
                metadata.get("seq") if isinstance(metadata, Mapping) else None
            )
            generation = self._home_generation
        return {"status": "recentered", "home_generation": generation}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        snapshot_started_monotonic = float(self.monotonic_clock())
        now = float(self.clock())
        stats = self._source_stats()
        try:
            channel, pose, age = self._latest(
                now=now,
                source_stats=stats,
            )
        except Exception as error:
            channel, pose, age = None, None, None
            read_error = f"{type(error).__name__}:{error}"
        else:
            read_error = None
        valid = bool(
            channel is not None
            and pose is not None
            and getattr(channel, "valid", False)
            and age is not None
            and age <= self.config.freshness_sec
        )
        source_channel_valid = bool(
            channel is not None and getattr(channel, "valid", False)
        )
        pose_projectable = pose is not None
        fresh = bool(
            age is not None and age <= self.config.freshness_sec
        )
        if valid:
            with self._lock:
                needs_home = self._home_pose is None
            if needs_home:
                metadata = getattr(channel, "metadata", {}) or {}
                with self._lock:
                    if self._home_pose is None:
                        self._home_pose = pose.copy()
                        self._home_generation += 1
                        self._home_sample_seq = (
                            metadata.get("seq")
                            if isinstance(metadata, Mapping)
                            else None
                        )
        with self._lock:
            started = self._started
            home = None if self._home_pose is None else self._home_pose.copy()
            generation = self._home_generation
            home_seq = self._home_sample_seq
        components: list[dict[str, Any]] = []
        status = "waiting"
        message = "等待新鲜有效的 VIVE pose。"
        if valid and home is not None and pose is not None:
            try:
                component = project_pose_matrix_deflection(
                    home,
                    pose,
                    component_id=self.config.source_role,
                    translation_reference_frame=self.display_reference_frame,
                    rotation_controlled_frame=self.config.source_role,
                )
                if self.coordinate_calibration is not None:
                    component["translation_axis_semantics"] = dict(
                        self.translation_axis_semantics
                    )
                components = [component]
                status = "available"
                message = (
                    "显示已标定坐标中的 VIVE pose 相对 Viewer Home 偏转；"
                    + (
                        self.axis_semantics_warning
                        or "+X 前、+Y 左、+Z 上。"
                    )
                    + "旋转仅为只读显示投影。"
                    if self.coordinate_calibration is not None
                    else "调试模式：显示原始 vive_world 偏转，不声明前后左右语义。"
                )
            except ValueError as error:
                status = "invalid"
                message = str(error)
        elif read_error is not None:
            status = "error"
            message = read_error
        elif channel is not None and age is not None and age > self.config.freshness_sec:
            status = "stale"
            message = "VIVE pose 已过期；保留 Home，但不显示旧偏转。"
        elif channel is not None and not source_channel_valid:
            status = "invalid"
            message = "VIVE channel 当前标记为 invalid；等待新的有效 pose。"
        elif channel is not None and not pose_projectable:
            status = "invalid"
            message = "VIVE pose 不是可显示的有限刚体变换。"
        metadata = getattr(channel, "metadata", {}) or {}
        snapshot_finished_monotonic = float(self.monotonic_clock())

        def monotonic_age(field_name: str) -> float | None:
            value = stats.get(field_name)
            try:
                timestamp = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(timestamp):
                return None
            return max(0.0, snapshot_finished_monotonic - timestamp)

        process_duration = stats.get("last_process_duration_sec")
        try:
            process_duration_sec = float(process_duration)
        except (TypeError, ValueError):
            process_duration_sec = None
        if process_duration_sec is not None and not math.isfinite(
            process_duration_sec
        ):
            process_duration_sec = None
        return {
            "schema": VIVE_DEFLECTION_STATUS_SCHEMA_V1,
            "application": "control_station_vive_deflection_viewer",
            "started": started,
            "motion_authorized": False,
            "simulator_created": False,
            "policy_created": False,
            "composition_created": False,
            "source": {
                "status": "receiving" if valid else status,
                "role": self.config.source_role,
                "tracker_name": self.config.tracker_name,
                "display_reference_frame": self.display_reference_frame,
                "coordinate_profile_path": (
                    None
                    if self.coordinate_profile_path is None
                    else str(self.coordinate_profile_path)
                ),
                "coordinate_calibrated": self.coordinate_calibration is not None,
                "axis_semantics_status": self.axis_semantics_status,
                "axis_semantics_warning": self.axis_semantics_warning,
                "age_sec": age,
                "seq": metadata.get("seq") if isinstance(metadata, Mapping) else None,
                "receiver_timestamp": (
                    metadata.get("receiver_timestamp")
                    if isinstance(metadata, Mapping)
                    else None
                ),
                "packet_count": stats.get("packet_count"),
                "datagram_count": stats.get("datagram_count"),
                "transport_datagram_age_sec": monotonic_age(
                    "last_datagram_received_monotonic_time"
                ),
                "processed_packet_age_sec": monotonic_age(
                    "last_packet_received_monotonic_time"
                ),
                "last_process_age_sec": monotonic_age(
                    "last_process_finished_monotonic_time"
                ),
                "last_process_duration_sec": process_duration_sec,
                "last_process_batch_size": stats.get("last_process_batch_size"),
                "last_drain_count": stats.get("last_drain_count"),
                "worker_alive": stats.get("worker_alive"),
                "error": read_error,
                "channel_present": channel is not None,
                "channel_valid": source_channel_valid,
                "pose_projectable": pose_projectable,
                "fresh": fresh,
                "freshness_sec": self.config.freshness_sec,
                "recenter_ready": valid,
                "message": message,
            },
            "deflection": {
                "status": status,
                "message": message,
                "age_sec": age,
                "anchor_request_generation": generation or None,
                "home_sample_seq": home_seq,
                "components": components,
            },
            "sampled_at": now,
            "snapshot_duration_sec": max(
                0.0,
                snapshot_finished_monotonic - snapshot_started_monotonic,
            ),
        }


__all__ = [
    "VIVE_DEFLECTION_STATUS_SCHEMA_V1",
    "VIVE_RETARGET_AXIS_SEMANTICS",
    "ViveDeflectionConfig",
    "ViveDeflectionController",
]
