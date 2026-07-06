from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_motion.rm65 import (
    load_rm65_tool_profile,
    tool_tcp_pose_flange_m_rad,
)
from rm65_inspire_tac.pipeline import DEFAULT_LEFT_ARM_QPOS, DEFAULT_RIGHT_ARM_QPOS

from teleop_runtime.arm_frames import TOOL_TCP_FRAME
from teleop_runtime.calibration import TeleopCalibration
from teleop_runtime.configs import build_bimanual_pose_smoother, load_default_pose_smoothing_config
from teleop_runtime.replay.dataset import ReplayDatasetIndex
from teleop_runtime.retargeters.dual_arm_collision_aware import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
)
from teleop_runtime.schemas import EffectorCommand, ObservationChannel, TeleopCommand, TeleopObservation
from teleop_runtime.transforms import as_pose_matrix, rotation_angle_rad


SIDES = ("left", "right")
DEFAULT_ARM_MAX_ANGVEL_RAD_S = (0.35, 0.35, 0.35, 0.5, 0.5, 0.5)


@dataclass(frozen=True)
class OfflineTrackerReplayConfig:
    dataset_root: Path
    episode: str | int = 0
    start_frame: int = 0
    max_frames: int = 0
    active_sides: tuple[str, ...] = SIDES
    use_target_smoothing: bool = True
    enforce_realistic_joint_step: bool = True
    initial_qpos_source: str = "first_observation"
    output_command_type: str = "cartesian_pose"
    solve_max_iterations: int = 1
    seed_each_frame_from_observation: bool = False
    scene_profile: str | None = None
    contact_mode: str | None = None
    solver_frequency: float | None = None
    ik_tuning_json_path: str | None = None

    def __post_init__(self) -> None:
        sides = tuple(str(side).strip().lower() for side in self.active_sides)
        unknown = sorted(set(sides) - set(SIDES))
        if unknown:
            raise ValueError(f"active_sides must be within {SIDES}, got {unknown}.")
        if not sides:
            raise ValueError("active_sides cannot be empty.")
        if self.initial_qpos_source not in {"first_observation", "calibration_home", "default"}:
            raise ValueError("initial_qpos_source must be first_observation, calibration_home, or default.")
        if self.output_command_type not in {"cartesian_pose", "joint_vector"}:
            raise ValueError("output_command_type must be cartesian_pose or joint_vector.")
        if int(self.solve_max_iterations) <= 0:
            raise ValueError("solve_max_iterations must be positive.")
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser().resolve())
        object.__setattr__(self, "active_sides", sides)
        object.__setattr__(self, "solve_max_iterations", int(self.solve_max_iterations))


@dataclass(frozen=True)
class SideReplayError:
    translation_error_m: float | None = None
    rotation_error_deg: float | None = None
    qpos_max_abs_error_deg: float | None = None
    qpos_rms_error_deg: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "translation_error_m": self.translation_error_m,
            "rotation_error_deg": self.rotation_error_deg,
            "qpos_max_abs_error_deg": self.qpos_max_abs_error_deg,
            "qpos_rms_error_deg": self.qpos_rms_error_deg,
        }


@dataclass(frozen=True)
class OfflineTrackerReplayFrame:
    row_index: int
    frame_index: int
    timestamp: float | None
    frame_time_sec: float
    status: str
    fallback_reason: str | None
    source_pose_desk: dict[str, list[float] | None]
    original_action_pose_base: dict[str, list[float] | None]
    replay_action_pose_base: dict[str, list[float] | None]
    original_qpos: dict[str, list[float] | None]
    replay_qpos: dict[str, list[float] | None]
    errors: dict[str, SideReplayError]
    source_vs_action_errors: dict[str, SideReplayError]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "frame_time_sec": self.frame_time_sec,
            "status": self.status,
            "fallback_reason": self.fallback_reason,
            "source_pose_desk": self.source_pose_desk,
            "original_action_pose_base": self.original_action_pose_base,
            "replay_action_pose_base": self.replay_action_pose_base,
            "original_qpos": self.original_qpos,
            "replay_qpos": self.replay_qpos,
            "errors": {side: error.to_dict() for side, error in self.errors.items()},
            "source_vs_action_errors": {
                side: error.to_dict() for side, error in self.source_vs_action_errors.items()
            },
            "metadata": _json_safe(self.metadata),
        }


@dataclass(frozen=True)
class OfflineTrackerReplayReport:
    dataset_root: Path
    episode_id: str
    episode_index: int | None
    parquet_path: Path
    active_sides: tuple[str, ...]
    frame_count: int
    source_frame_count: int
    start_frame: int
    max_frames: int
    nominal_hz: float
    timing_source: str
    summary: dict[str, Any]
    frames: list[OfflineTrackerReplayFrame]
    video_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_frames: bool = True) -> dict[str, Any]:
        payload = {
            "dataset_root": str(self.dataset_root),
            "episode_id": self.episode_id,
            "episode_index": self.episode_index,
            "parquet_path": str(self.parquet_path),
            "active_sides": list(self.active_sides),
            "frame_count": self.frame_count,
            "source_frame_count": self.source_frame_count,
            "start_frame": self.start_frame,
            "max_frames": self.max_frames,
            "nominal_hz": self.nominal_hz,
            "timing_source": self.timing_source,
            "summary": _json_safe(self.summary),
            "video_paths": dict(self.video_paths),
            "metadata": _json_safe(self.metadata),
        }
        if include_frames:
            payload["frames"] = [frame.to_dict() for frame in self.frames]
        return payload


class RecordedTargetPoseRetargeter:
    """Retargeter shim that reuses recorded pose-retargeter outputs as IK targets."""

    def __init__(self, *, side: str, input_channel_name: str, effector_name: str):
        self.side = side
        self.input_channel_name = input_channel_name
        self.effector_name = effector_name

    @property
    def name(self) -> str:
        return f"recorded_target_pose_{self.side}"

    def retarget(
        self,
        observation: TeleopObservation,
        calibration: TeleopCalibration | None = None,
    ) -> TeleopCommand:
        command = TeleopCommand(timestamp=observation.timestamp)
        channel = observation.get_channel(self.input_channel_name)
        if channel is None or not channel.valid:
            command.metadata["skipped"] = self.input_channel_name
            return command
        if calibration is None:
            raise RuntimeError(f"{self.name} requires a TeleopCalibration instance.")

        target_pose_desk = as_pose_matrix(channel.value)
        target_pose_base = calibration.desk_to_base(target_pose_desk, side=self.side)
        target_pose_operator = channel.metadata.get("source_target_pose_operator")
        command.add_command(
            EffectorCommand(
                effector_name=self.effector_name,
                side=self.side,
                command_type="cartesian_pose",
                value=target_pose_base,
                frame_name="arm_base",
                metadata={
                    "arm_controlled_frame": TOOL_TCP_FRAME,
                    "target_pose_controlled_frame": TOOL_TCP_FRAME,
                    "arm_reference_frame": "arm_base",
                    "target_pose_desk": target_pose_desk.copy(),
                    "target_pose_tcp_desk": target_pose_desk.copy(),
                    "target_pose_base": target_pose_base.copy(),
                    "target_pose_tcp_base": target_pose_base.copy(),
                    "target_pose_operator": None
                    if target_pose_operator is None
                    else as_pose_matrix(target_pose_operator),
                    "input_channel": self.input_channel_name,
                    "tracker_role": channel.metadata.get("tracker_role"),
                    "tracker_name": channel.metadata.get("tracker_name"),
                    "offline_replay_source": "teleop.action_source.*_wrist_pose_desk",
                },
            )
        )
        return command


def run_offline_tracker_replay(config: OfflineTrackerReplayConfig) -> OfflineTrackerReplayReport:
    index = ReplayDatasetIndex(config.dataset_root)
    episode = index.load_episode(config.episode)
    rows = _slice_rows(
        episode.rows,
        start_frame=int(config.start_frame),
        max_frames=int(config.max_frames),
    )
    metadata = dict(episode.summary.metadata)
    calibration = _calibration_from_metadata(metadata)
    initial_qpos = _initial_qpos_by_side(
        rows=rows,
        calibration=calibration,
        source=config.initial_qpos_source,
    )
    retargeter = _build_retargeter(
        config=config,
        metadata=metadata,
        initial_qpos=initial_qpos,
    )
    frames: list[OfflineTrackerReplayFrame] = []
    for row_index, row in enumerate(rows):
        if config.seed_each_frame_from_observation:
            _seed_retargeter_from_row(retargeter, row)
        observation = _observation_from_row(row, config.active_sides)
        context = retargeter.build_target_context(observation, calibration)
        command = retargeter.build_command_for_context(context)
        replay_qpos = _current_qpos_by_side(retargeter)
        frames.append(
            _frame_result_from_command(
                row=row,
                row_index=int(config.start_frame) + row_index,
                frame_time_sec=episode.frame_times_sec[int(config.start_frame) + row_index]
                if int(config.start_frame) + row_index < len(episode.frame_times_sec)
                else row_index / max(float(episode.nominal_hz), 1e-6),
                command=command,
                replay_qpos=replay_qpos,
                active_sides=config.active_sides,
                calibration=calibration,
            )
        )

    summary = summarize_offline_tracker_frames(frames, active_sides=config.active_sides)
    summary.update(
        {
            "initial_qpos_source": config.initial_qpos_source,
            "target_smoothing_enabled": bool(config.use_target_smoothing),
            "output_command_type": config.output_command_type,
            "solve_max_iterations": int(config.solve_max_iterations),
            "seed_each_frame_from_observation": bool(config.seed_each_frame_from_observation),
            "scene_profile": _scene_profile_from_config(config, metadata),
        }
    )
    return OfflineTrackerReplayReport(
        dataset_root=episode.dataset_root,
        episode_id=episode.summary.episode_id,
        episode_index=episode.summary.episode_index,
        parquet_path=episode.summary.parquet_path,
        active_sides=config.active_sides,
        frame_count=len(frames),
        source_frame_count=episode.frame_count,
        start_frame=int(config.start_frame),
        max_frames=int(config.max_frames),
        nominal_hz=float(episode.nominal_hz),
        timing_source=str(episode.timing_source),
        summary=summary,
        frames=frames,
        video_paths=dict(episode.summary.video_paths),
        metadata={
            "dataset_info": index.info,
            "episode_metadata": metadata,
        },
    )


def summarize_offline_tracker_frames(
    frames: Sequence[OfflineTrackerReplayFrame],
    *,
    active_sides: Sequence[str] = SIDES,
) -> dict[str, Any]:
    status_counts = Counter(frame.status for frame in frames)
    fallback_counts = Counter(
        str(frame.fallback_reason)
        for frame in frames
        if frame.fallback_reason not in {None, ""}
    )
    by_side: dict[str, Any] = {}
    for side in active_sides:
        side_errors = [frame.errors.get(side) for frame in frames]
        by_side[str(side)] = {
            "ik_vs_action_translation_error_m": _series_stats(
                error.translation_error_m for error in side_errors if error is not None
            ),
            "ik_vs_action_rotation_error_deg": _series_stats(
                error.rotation_error_deg for error in side_errors if error is not None
            ),
            "qpos_max_abs_error_deg": _series_stats(
                error.qpos_max_abs_error_deg for error in side_errors if error is not None
            ),
            "qpos_rms_error_deg": _series_stats(
                error.qpos_rms_error_deg for error in side_errors if error is not None
            ),
            "source_vs_action_translation_error_m": _series_stats(
                frame.source_vs_action_errors.get(side).translation_error_m
                for frame in frames
                if frame.source_vs_action_errors.get(side) is not None
            ),
            "source_vs_action_rotation_error_deg": _series_stats(
                frame.source_vs_action_errors.get(side).rotation_error_deg
                for frame in frames
                if frame.source_vs_action_errors.get(side) is not None
            ),
        }
    return {
        "frame_count": len(frames),
        "status_counts": dict(status_counts),
        "fallback_reason_counts": dict(fallback_counts),
        "by_side": by_side,
    }


def report_exceeds_thresholds(
    report: OfflineTrackerReplayReport,
    *,
    max_translation_error_m: float | None = None,
    max_rotation_error_deg: float | None = None,
    max_qpos_error_deg: float | None = None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    by_side = report.summary.get("by_side")
    if not isinstance(by_side, Mapping):
        return False, failures
    for side, metrics in by_side.items():
        if not isinstance(metrics, Mapping):
            continue
        _check_metric(
            failures,
            side=str(side),
            metric="translation_error_m",
            stats=metrics.get("ik_vs_action_translation_error_m"),
            threshold=max_translation_error_m,
        )
        _check_metric(
            failures,
            side=str(side),
            metric="rotation_error_deg",
            stats=metrics.get("ik_vs_action_rotation_error_deg"),
            threshold=max_rotation_error_deg,
        )
        _check_metric(
            failures,
            side=str(side),
            metric="qpos_max_abs_error_deg",
            stats=metrics.get("qpos_max_abs_error_deg"),
            threshold=max_qpos_error_deg,
        )
    return bool(failures), failures


def save_offline_tracker_report(
    report: OfflineTrackerReplayReport,
    path: str | Path,
    *,
    include_frames: bool = True,
) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(include_frames=include_frames), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def _build_retargeter(
    *,
    config: OfflineTrackerReplayConfig,
    metadata: Mapping[str, Any],
    initial_qpos: Mapping[str, np.ndarray],
) -> DualArmCollisionAwareRetargeter:
    scene_profile = _scene_profile_from_config(config, metadata)
    contact_mode = str(config.contact_mode or metadata.get("collision_avoidance_mode") or "boundary_slide")
    solver_frequency = float(
        config.solver_frequency
        if config.solver_frequency is not None
        else metadata.get("hz") or metadata.get("dual_arm_ik_hz") or 15.0
    )
    action_dt = _float_or_default(metadata.get("action_dt"), 1.0 / max(solver_frequency, 1e-6))
    arm_max_angvel = np.asarray(
        metadata.get("arm_max_angvel") or DEFAULT_ARM_MAX_ANGVEL_RAD_S,
        dtype=float,
    ).reshape(6)
    realistic_joint_step = tuple((arm_max_angvel * action_dt).tolist())
    smoother = (
        build_bimanual_pose_smoother(load_default_pose_smoothing_config())
        if config.use_target_smoothing
        else None
    )
    left_retargeter = RecordedTargetPoseRetargeter(
        side="left",
        input_channel_name="left_wrist_pose",
        effector_name="left_arm",
    )
    right_retargeter = RecordedTargetPoseRetargeter(
        side="right",
        input_channel_name="right_wrist_pose",
        effector_name="right_arm",
    )
    return DualArmCollisionAwareRetargeter(
        DualArmCollisionAwareRetargeterConfig(
            control_mode=str(metadata.get("control_mode") or "absolute"),
            translation_scale=_float_or_default(metadata.get("translation_scale"), 1.0),
            scene_profile=scene_profile,
            contact_mode=contact_mode,
            solver_frequency=solver_frequency,
            ik_framework="robot_motion",
            enforce_realistic_joint_step=bool(config.enforce_realistic_joint_step),
            left_execution_max_joint_step_rad=realistic_joint_step,
            right_execution_max_joint_step_rad=realistic_joint_step,
            left_tool_tcp_pose_flange_m_rad=_tool_tcp_pose_from_metadata(metadata, "left"),
            right_tool_tcp_pose_flange_m_rad=_tool_tcp_pose_from_metadata(metadata, "right"),
            left_initial_qpos=initial_qpos["left"],
            right_initial_qpos=initial_qpos["right"],
            ik_tuning_json_path=config.ik_tuning_json_path,
            target_pose_smoother=smoother,
            output_command_type=config.output_command_type,
            solve_max_iterations=int(config.solve_max_iterations),
            debug_profile=False,
        ),
        left_pose_retargeter=left_retargeter,  # type: ignore[arg-type]
        right_pose_retargeter=right_retargeter,  # type: ignore[arg-type]
    )


def _calibration_from_metadata(metadata: Mapping[str, Any]) -> TeleopCalibration:
    payload = metadata.get("calibration_snapshot")
    if not isinstance(payload, Mapping):
        raise ValueError("Episode metadata does not contain calibration_snapshot.")
    return TeleopCalibration.from_dict(dict(payload))


def _initial_qpos_by_side(
    *,
    rows: Sequence[Mapping[str, Any]],
    calibration: TeleopCalibration,
    source: str,
) -> dict[str, np.ndarray]:
    if source == "first_observation":
        first = rows[0] if rows else {}
        return {
            "left": _first_qpos(
                _row_qpos(first, "left"),
                _calibration_qpos(calibration, "left"),
                DEFAULT_LEFT_ARM_QPOS,
            ),
            "right": _first_qpos(
                _row_qpos(first, "right"),
                _calibration_qpos(calibration, "right"),
                DEFAULT_RIGHT_ARM_QPOS,
            ),
        }
    if source == "calibration_home":
        return {
            "left": _first_qpos(_calibration_qpos(calibration, "left"), DEFAULT_LEFT_ARM_QPOS),
            "right": _first_qpos(_calibration_qpos(calibration, "right"), DEFAULT_RIGHT_ARM_QPOS),
        }
    return {
        "left": DEFAULT_LEFT_ARM_QPOS.copy(),
        "right": DEFAULT_RIGHT_ARM_QPOS.copy(),
    }


def _first_qpos(*candidates: np.ndarray | None) -> np.ndarray:
    for candidate in candidates:
        if candidate is None:
            continue
        return np.asarray(candidate, dtype=float).reshape(6).copy()
    raise ValueError("No qpos candidate was provided.")


def _calibration_qpos(calibration: TeleopCalibration, side: str) -> np.ndarray | None:
    for key in (f"{side}_arm", side):
        value = calibration.captured_home_qpos.get(key)
        if value is not None:
            return np.asarray(value, dtype=float).reshape(6).copy()
    try:
        arm = calibration.get_arm_for_side(side)
    except KeyError:
        return None
    if arm.home_qpos is None:
        return None
    return np.asarray(arm.home_qpos, dtype=float).reshape(6).copy()


def _row_qpos(row: Mapping[str, Any], side: str) -> np.ndarray | None:
    for key in (
        f"observation.arm.{side}.current_qpos",
        f"observation.{side}_arm.qpos",
        f"observation.{side}_arm.current_qpos",
    ):
        value = row.get(key)
        if value is None:
            continue
        qpos = _finite_vector(value, length=6)
        if qpos is not None:
            return qpos
    return None


def _observation_from_row(row: Mapping[str, Any], active_sides: Sequence[str]) -> TeleopObservation:
    timestamp = _float_or_none(row.get("timestamp"))
    channels: dict[str, ObservationChannel] = {}
    channel_metadata = _json_mapping(row.get("channels_json"))
    for side in active_sides:
        pose = _pose_from_row(row, f"teleop.action_source.{side}_wrist_pose_desk")
        valid = pose is not None and bool(
            _nested_get(channel_metadata, f"{side}_wrist_pose", "valid", default=True)
        )
        channels[f"{side}_wrist_pose"] = ObservationChannel(
            channel_name=f"{side}_wrist_pose",
            capability="pose6d",
            side=side,
            value=np.eye(4, dtype=float) if pose is None else pose,
            frame_name="desk",
            timestamp=_float_or_none(
                _nested_get(channel_metadata, f"{side}_wrist_pose", "timestamp")
            )
            or timestamp
            or 0.0,
            source_name="offline_dataset_action_source",
            valid=bool(valid),
            critical=True,
            metadata={
                "tracker_role": f"{side}_wrist",
                "tracker_name": _nested_get(channel_metadata, f"{side}_wrist_pose", "tracker_name")
                or _nested_get(channel_metadata, f"{side}_wrist_pose", "source_name"),
                "source_target_pose_operator": _pose_from_row(
                    row,
                    f"teleop.action_source.{side}_wrist_pose_operator",
                ),
            },
        )
        channels[f"{side}_clutch_button"] = ObservationChannel(
            channel_name=f"{side}_clutch_button",
            capability="bool",
            side=side,
            value=False,
            frame_name=None,
            timestamp=timestamp or 0.0,
            source_name="offline_dataset",
            valid=True,
            critical=False,
        )
    return TeleopObservation(channels=channels, timestamp=timestamp, metadata={"offline_replay": True})


def _frame_result_from_command(
    *,
    row: Mapping[str, Any],
    row_index: int,
    frame_time_sec: float,
    command: TeleopCommand,
    replay_qpos: Mapping[str, np.ndarray],
    active_sides: Sequence[str],
    calibration: TeleopCalibration,
) -> OfflineTrackerReplayFrame:
    frame_index = _int_or_default(row.get("frame_index"), row_index)
    timestamp = _float_or_none(row.get("timestamp"))
    status = str(command.metadata.get("collision_avoidance_status") or command.metadata.get("status") or "unknown")
    fallback_reason = command.metadata.get("collision_avoidance_fallback_reason")
    errors: dict[str, SideReplayError] = {}
    source_pose_desk: dict[str, list[float] | None] = {}
    original_pose_base: dict[str, list[float] | None] = {}
    replay_pose_base: dict[str, list[float] | None] = {}
    original_qpos: dict[str, list[float] | None] = {}
    replay_qpos_out: dict[str, list[float] | None] = {}
    source_vs_action_errors: dict[str, SideReplayError] = {}

    for side in active_sides:
        source_pose = _pose_from_row(row, f"teleop.action_source.{side}_wrist_pose_desk")
        source_pose_base = None if source_pose is None else calibration.desk_to_base(source_pose, side=side)
        original_pose = _pose_from_row(row, f"action.{side}_arm.pose")
        effector_command = command.commands.get(f"{side}_arm")
        replay_pose = (
            as_pose_matrix(effector_command.value)
            if effector_command is not None and effector_command.command_type == "cartesian_pose"
            else None
        )
        original_side_qpos = _row_qpos(row, side)
        replay_side_qpos = np.asarray(replay_qpos[side], dtype=float).reshape(6)
        source_pose_desk[side] = _pose_to_xyz_quat_xyzw_or_none(source_pose)
        original_pose_base[side] = _pose_to_xyz_quat_xyzw_or_none(original_pose)
        replay_pose_base[side] = _pose_to_xyz_quat_xyzw_or_none(replay_pose)
        original_qpos[side] = None if original_side_qpos is None else original_side_qpos.astype(float).tolist()
        replay_qpos_out[side] = replay_side_qpos.astype(float).tolist()
        errors[side] = _side_errors(
            original_pose=original_pose,
            replay_pose=replay_pose,
            original_qpos=original_side_qpos,
            replay_qpos=replay_side_qpos,
        )
        source_vs_action_errors[side] = _side_errors(
            original_pose=original_pose,
            replay_pose=source_pose_base,
            original_qpos=None,
            replay_qpos=None,
        )

    return OfflineTrackerReplayFrame(
        row_index=row_index,
        frame_index=frame_index,
        timestamp=timestamp,
        frame_time_sec=float(frame_time_sec),
        status=status,
        fallback_reason=None if fallback_reason is None else str(fallback_reason),
        source_pose_desk=source_pose_desk,
        original_action_pose_base=original_pose_base,
        replay_action_pose_base=replay_pose_base,
        original_qpos=original_qpos,
        replay_qpos=replay_qpos_out,
        errors=errors,
        source_vs_action_errors=source_vs_action_errors,
        metadata={
            "command_metadata": _json_safe(command.metadata),
            "faults": _json_or_value(row.get("faults_json"), default=[]),
        },
    )


def _side_errors(
    *,
    original_pose: np.ndarray | None,
    replay_pose: np.ndarray | None,
    original_qpos: np.ndarray | None,
    replay_qpos: np.ndarray | None,
) -> SideReplayError:
    translation_error = None
    rotation_error = None
    if original_pose is not None and replay_pose is not None:
        translation_error = float(np.linalg.norm(replay_pose[:3, 3] - original_pose[:3, 3]))
        rotation_error = float(np.rad2deg(rotation_angle_rad(original_pose, replay_pose)))
    qpos_max = None
    qpos_rms = None
    if original_qpos is not None and replay_qpos is not None:
        delta_deg = np.rad2deg(np.asarray(replay_qpos, dtype=float).reshape(6) - original_qpos.reshape(6))
        qpos_max = float(np.max(np.abs(delta_deg)))
        qpos_rms = float(np.sqrt(np.mean(delta_deg * delta_deg)))
    return SideReplayError(
        translation_error_m=translation_error,
        rotation_error_deg=rotation_error,
        qpos_max_abs_error_deg=qpos_max,
        qpos_rms_error_deg=qpos_rms,
    )


def _current_qpos_by_side(retargeter: DualArmCollisionAwareRetargeter) -> dict[str, np.ndarray]:
    left, right = retargeter._split_solution(retargeter.ik_solver.current_q)  # noqa: SLF001
    return {
        "left": np.asarray(left, dtype=float).reshape(6).copy(),
        "right": np.asarray(right, dtype=float).reshape(6).copy(),
    }


def _seed_retargeter_from_row(
    retargeter: DualArmCollisionAwareRetargeter,
    row: Mapping[str, Any],
) -> None:
    left = _row_qpos(row, "left")
    right = _row_qpos(row, "right")
    if left is None and right is None:
        return
    current = _current_qpos_by_side(retargeter)
    retargeter.set_current_qpos_pair(
        left_qpos=current["left"] if left is None else left,
        right_qpos=current["right"] if right is None else right,
    )


def _tool_tcp_pose_from_metadata(metadata: Mapping[str, Any], side: str) -> tuple[float, ...] | None:
    profile_name = metadata.get(f"{side}_tool_profile")
    if not profile_name:
        return None
    try:
        profile = load_rm65_tool_profile(str(profile_name), side=side)
        pose = tool_tcp_pose_flange_m_rad(profile)
    except Exception:
        return None
    return tuple(float(value) for value in pose.reshape(6))


def _scene_profile_from_config(config: OfflineTrackerReplayConfig, metadata: Mapping[str, Any]) -> str:
    return str(config.scene_profile or metadata.get("collision_scene_profile") or "dual_arm_hand_proxy_open")


def _pose_from_row(row: Mapping[str, Any], key: str) -> np.ndarray | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return as_pose_matrix(value)
    except Exception:
        return None


def _finite_vector(value: Any, *, length: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float).reshape(length)
    except Exception:
        return None
    if not np.all(np.isfinite(array)):
        return None
    return array.copy()


def _pose_to_xyz_quat_xyzw_or_none(pose: np.ndarray | None) -> list[float] | None:
    if pose is None:
        return None
    matrix = as_pose_matrix(pose)
    quat = R.from_matrix(matrix[:3, :3]).as_quat()
    return [
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]),
    ]


def _series_stats(values: Any) -> dict[str, float | int | None]:
    array = np.asarray([float(value) for value in values if value is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, "rms": None, "max": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(array * array))),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _check_metric(
    failures: list[str],
    *,
    side: str,
    metric: str,
    stats: Any,
    threshold: float | None,
) -> None:
    if threshold is None or float(threshold) <= 0.0 or not isinstance(stats, Mapping):
        return
    value = stats.get("max")
    if value is None:
        return
    if float(value) > float(threshold):
        failures.append(f"{side}.{metric}.max={float(value):.6g} > {float(threshold):.6g}")


def _slice_rows(
    rows: Sequence[dict[str, Any]],
    *,
    start_frame: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    start = int(start_frame)
    if start < 0:
        raise ValueError("start_frame must be non-negative.")
    end = None if int(max_frames) <= 0 else start + int(max_frames)
    selected = list(rows[start:end])
    if not selected:
        raise ValueError(
            f"No rows selected: start_frame={start_frame}, max_frames={max_frames}, total_rows={len(rows)}."
        )
    return selected


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not np.isfinite(number):
        return None
    return float(number)


def _float_or_default(value: Any, default: float) -> float:
    number = _float_or_none(value)
    return float(default) if number is None else float(number)


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _json_mapping(value: Any) -> dict[str, Any]:
    parsed = _json_or_value(value, default={})
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _json_or_value(value: Any, *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return value


def _nested_get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


__all__ = [
    "OfflineTrackerReplayConfig",
    "OfflineTrackerReplayFrame",
    "OfflineTrackerReplayReport",
    "RecordedTargetPoseRetargeter",
    "report_exceeds_thresholds",
    "run_offline_tracker_replay",
    "save_offline_tracker_report",
    "summarize_offline_tracker_frames",
]
