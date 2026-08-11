"""Debug trajectory episode paths, metadata, and lifecycle adapters."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from project_paths import teleop_records_root

from ..episodes import EpisodeRecorder
from ..hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER
from ..log_utils import log_event as _log
from ..special_tasks.fix_arm import FixArmPoseSnapshotRecorder
from .arm_tool_config import (
    _force_tare_sec_arg,
    _impedance_mode_arg,
    _strict_tool_profile_arg,
)
from .components import RealTeleopComponents
from .control_profile import (
    arm_control_mode as _arm_control_mode_arg,
    effective_arm_control_hz as _effective_arm_control_hz,
    effective_outer_control_hz as _effective_outer_control_hz,
    effective_position_reference_hz as _effective_position_reference_hz,
)
from collection_runtime.dataset_layout import expand_cli_path as _expand_cli_path
from .debug_timing import debug_timing_metadata as _debug_timing_metadata
from .gripper_policy import (
    gripper_force_dataset_metadata as _gripper_force_dataset_metadata,
    resolved_gripper_policies_by_side as _resolved_gripper_policies_by_side,
    resolved_gripper_policy as _resolved_gripper_policy,
)
from .hand_config import (
    dmtac_sides as _dmtac_sides,
    hand_backend_mode as _hand_backend_mode,
    hand_backend_mode_for_side as _hand_backend_mode_for_side,
)
from .hand_runtime import _parse_hand_tactile_regions
from .movement import movement_multiple_value as _movement_multiple_value
from .position_force_guard import (
    position_force_guard_config as _position_force_guard_config,
)
from .runtime_metadata import (
    arm_execution_profile_metadata as _arm_execution_profile_metadata,
    position_ik_iteration_metadata as _position_ik_iteration_metadata,
)
from .session_recording import (
    _session_record_root_dir,
    _session_recording_enabled,
)
from .side_mapping import (
    tracker_name_for_source_side as _tracker_name_for_source_side,
    tracker_source_side_for_logical_side as _tracker_source_side_for_logical_side,
)
from .wrist_depth_freshness import (
    DEFAULT_WRIST_DEPTH_FPS,
    DEFAULT_WRIST_DEPTH_MAX_M,
    DEFAULT_WRIST_DEPTH_MIN_M,
    DEFAULT_WRIST_DEPTH_SAVE_FPS,
    DEFAULT_WRIST_DEPTH_STREAM_FPS,
    WRIST_DEPTH_CAMERA_KEYS,
)


def _debug_episode_recording_enabled(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "no_episode_recording", False)):
        return False
    if bool(getattr(args, "no_collection", False)):
        return True
    if bool(getattr(args, "debug_trajectory_recording", False)):
        return True
    return bool(str(getattr(args, "episode_root", "") or "").strip())


def _episode_root_dir(
    args: argparse.Namespace,
    components: RealTeleopComponents,
) -> Path | None:
    episode_root = str(getattr(args, "episode_root", "") or "").strip()
    if episode_root:
        return _expand_cli_path(episode_root)
    canonical_trajectory_root = str(
        getattr(args, "_runtime_artifact_trajectory_root", "") or ""
    ).strip()
    if canonical_trajectory_root:
        return Path(canonical_trajectory_root).expanduser()
    if components.session.recorder is not None:
        return Path(components.session.recorder.root_dir) / "trajectories"
    session_record_root = _session_record_root_dir(args)
    if session_record_root is not None:
        return session_record_root / "trajectories"
    return None


def _fix_arm_pose_snapshot_recorder(
    components: RealTeleopComponents,
    *,
    root_dir: str | Path | None = None,
) -> FixArmPoseSnapshotRecorder | None:
    if getattr(components, "fix_arm_controller", None) is None:
        return None
    if root_dir is None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = teleop_records_root() / "fix_arm_pose_snapshots" / session_name
    return FixArmPoseSnapshotRecorder(
        Path(root_dir).expanduser()
    )


def _build_episode_metadata(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    start_reason: str,
) -> dict[str, object]:
    calibration = components.session.calibration
    assert calibration is not None
    session_record_enabled = _session_recording_enabled(args)
    session_record_root = (
        None if components.session.recorder is None else str(components.session.recorder.root_dir)
    )
    debug_episode_enabled = _debug_episode_recording_enabled(args)
    episode_root = _episode_root_dir(args, components) if debug_episode_enabled else None
    position_force_guard_enabled = bool(
        _arm_control_mode_arg(args) == "position"
        and getattr(args, "position_force_guard", True)
    )
    return {
        "source_entrypoint": "teleop_runtime.debug.run_vive_rm65_real_teleop",
        "start_reason": start_reason,
        "control_mode": args.control_mode,
        "arm_control_mode": _arm_control_mode_arg(args),
        **_arm_execution_profile_metadata(args),
        **_position_ik_iteration_metadata(args),
        "impedance_mode": _impedance_mode_arg(args),
        "strict_tool_profile": _strict_tool_profile_arg(args),
        "force_tare_sec": _force_tare_sec_arg(args),
        "position_force_guard": {
            "enabled": position_force_guard_enabled,
            **_position_force_guard_config(args).to_dict(),
            "fail_closed": position_force_guard_enabled,
            "latched": position_force_guard_enabled,
            "stop_scope": (
                "bimanual" if position_force_guard_enabled else None
            ),
        },
        **_debug_timing_metadata(args),
        "arm_mode": "real",
        "active_sides": list(components.active_sides),
        "runtime_arm_state_source": (
            "causal_measured_hardware_preferred"
            if _arm_control_mode_arg(args) == "position"
            else "last_commanded_action"
        ),
        "runtime_arm_state_fallback": (
            "last_commanded_action"
            if _arm_control_mode_arg(args) == "position"
            else None
        ),
        "record_actual_wrist_camera_pose": bool(
            getattr(args, "record_actual_wrist_camera_pose", True)
        ),
        "actual_wrist_camera_pose_source": "rm65_feedback_qpos_fk",
        "actual_wrist_camera_pose_hz": float(
            getattr(args, "actual_wrist_camera_pose_hz", 30.0)
        ),
        "actual_wrist_camera_pose_max_age_sec": float(
            getattr(args, "actual_wrist_camera_pose_max_age_sec", 0.2)
        ),
        "hz": float(args.hz),
        "sample_hz": float(args.hz),
        "collection_worker_mode_requested": str(
            getattr(args, "collection_worker_mode", "auto")
        ),
        "collection_worker_mode": str(
            getattr(args, "collection_worker_mode_resolved", "thread")
        ),
        "collection_process_queue_size": int(
            getattr(args, "collection_process_queue_size", 4)
        ),
        "collection_process_shared_slots": int(
            getattr(args, "collection_process_shared_slots", 4)
        ),
        "collection_process_extra_array_mb": int(
            getattr(args, "collection_process_extra_array_mb", 8)
        ),
        "collection_worker_cpus": list(
            getattr(args, "collection_worker_cpus", ()) or ()
        ),
        "collection_worker_nice": int(
            getattr(args, "collection_worker_nice", 5)
        ),
        "position_reference_hz": (
            _effective_position_reference_hz(args)
            if _arm_control_mode_arg(args) == "position"
            else None
        ),
        "outer_control_hz": _effective_outer_control_hz(args),
        "arm_control_hz": _effective_arm_control_hz(args),
        "position_reference_max_age_sec": (
            float(args.position_reference_max_age_sec)
            if _arm_control_mode_arg(args) == "position"
            else None
        ),
        "movement_multiple": _movement_multiple_value(args),
        "translation_scale": float(args.translation_scale),
        "left_right_base_distance_m": float(args.left_right_base_distance),
        "collision_scene_profile": args.collision_scene_profile,
        "collision_avoidance_mode": args.collision_avoidance_mode,
        "collision_guard_requested": bool(components.collision_guard_requested),
        "collision_guard_enabled": bool(components.dual_arm_retargeter is not None),
        "dataset_root": str(getattr(args, "dataset_root", "") or "") or None,
        "wrist_depth_mask": bool(getattr(args, "wrist_depth_mask", False)),
        "wrist_depth_raw": bool(getattr(args, "wrist_depth_raw", False)),
        "wrist_depth_camera_keys": list(WRIST_DEPTH_CAMERA_KEYS),
        "wrist_depth_fps": DEFAULT_WRIST_DEPTH_FPS,
        "wrist_depth_save_fps": DEFAULT_WRIST_DEPTH_SAVE_FPS,
        "wrist_depth_stream_fps": int(getattr(args, "wrist_depth_stream_fps", DEFAULT_WRIST_DEPTH_STREAM_FPS)),
        "wrist_cam_stream_width": getattr(args, "wrist_cam_width", None),
        "wrist_cam_stream_height": getattr(args, "wrist_cam_height", None),
        "wrist_depth_align_to_color": bool(getattr(args, "wrist_depth_align_to_color", True)),
        "wrist_depth_min_m": float(getattr(args, "wrist_depth_min_m", DEFAULT_WRIST_DEPTH_MIN_M)),
        "wrist_depth_max_m": float(getattr(args, "wrist_depth_max_m", DEFAULT_WRIST_DEPTH_MAX_M)),
        "session_record_enabled": session_record_enabled,
        "session_record_root": session_record_root,
        "session_record_commands": (
            session_record_enabled and not bool(getattr(args, "no_session_record_commands", False))
        ),
        "session_record_flush_every": int(getattr(args, "session_record_flush_every", 1) or 1),
        "session_record_compact_json": bool(getattr(args, "session_record_compact_json", False)),
        "debug_episode_recording_enabled": debug_episode_enabled,
        "episode_root": None if episode_root is None else str(episode_root),
        "load_profile": args.load_profile or None,
        "load_initial_alignment": args.load_initial_alignment or None,
        "logical_arm_ip_mapping": dict(components.logical_arm_ip_mapping),
        "left_hand_ip": args.left_hand_ip or None,
        "right_hand_ip": args.right_hand_ip or None,
        "left_hand_port": getattr(args, "left_hand_port", "") or None,
        "right_hand_port": getattr(args, "right_hand_port", "") or None,
        "hand_backend": _hand_backend_mode(args),
        "hand_backend_by_side": {
            side: _hand_backend_mode_for_side(args, side) for side in components.active_sides
        },
        "q6_config": str(getattr(args, "q6_config", "latest")),
        "q6_config_loaded_ports": dict(getattr(args, "q6_config_loaded_ports", {}) or {}),
        "dmtac_config": str(getattr(args, "dmtac_config", "latest")),
        "dmtac_default_config_enabled": bool(getattr(args, "dmtac_default_config", True)),
        "dmtac_config_loaded_sides": list(getattr(args, "dmtac_config_loaded_sides", []) or []),
        "dmtac_tactile_fields": list(getattr(args, "dmtac_tactile_fields", None) or []),
        "dmtac_tactile_compute_profile": str(
            getattr(args, "dmtac_tactile_compute_profile", None) or "dist-only"
        ),
        "dmtac_tactile_parallel_read": getattr(args, "dmtac_tactile_parallel_read", None),
        "dmtac_tactile_global_read_lock": getattr(args, "dmtac_tactile_global_read_lock", None),
        "dmtac_tactile_read_hz": getattr(args, "dmtac_tactile_read_hz", None),
        "dmtac_tactile_process_read": getattr(args, "dmtac_tactile_process_read", None),
        "dmtac_tactile_worker_cpus": list(getattr(args, "dmtac_tactile_worker_cpus", None) or []),
        "dmtac_tactile_worker_nice": getattr(args, "dmtac_tactile_worker_nice", None),
        "dmtac_tactile_worker_queue_size": getattr(args, "dmtac_tactile_worker_queue_size", None),
        "dmtac_tactile_max_fps": getattr(args, "dmtac_tactile_max_fps", None),
        "dmtac_tactile_warmup_frames": getattr(args, "dmtac_tactile_warmup_frames", None),
        "dmtac_tactile_denoise": getattr(args, "dmtac_tactile_denoise", None),
        "dmtac_tactile_flat_vector": getattr(args, "dmtac_tactile_flat_vector", None),
        "gripper_task_profile": str(getattr(args, "gripper_task_profile", "") or "") or None,
        "gripper_task": str(getattr(args, "gripper_task", "") or "") or None,
        "collection_task_id": str(getattr(args, "collection_task_id", "") or "") or None,
        "collection_task_name": str(getattr(args, "collection_task_name", "") or "") or None,
        "collection_task_name_zh": str(getattr(args, "collection_task_name_zh", "") or "") or None,
        "collection_task_description": str(getattr(args, "collection_task_description", "") or "") or None,
        "gripper_policy": (
            dict(getattr(args, "_resolved_gripper_policy", {}) or _resolved_gripper_policy(args))
            if _dmtac_sides(args, components.active_sides)
            else {}
        ),
        "gripper_policy_by_side": (
            _resolved_gripper_policies_by_side(args, components.active_sides)
            if _dmtac_sides(args, components.active_sides)
            else {}
        ),
        **_gripper_force_dataset_metadata(args, components),
        "left_logical_tracker_name": _tracker_name_for_source_side(
            args,
            _tracker_source_side_for_logical_side(args, "left"),
        ),
        "right_logical_tracker_name": _tracker_name_for_source_side(
            args,
            _tracker_source_side_for_logical_side(args, "right"),
        ),
        "hand_reader_baudrate": int(getattr(args, "hand_reader_baudrate", 115200)),
        "exo_hand_calibration": bool(getattr(args, "exo_hand_calibration", False)),
        "exo_hand_calibration_path": getattr(args, "exo_hand_calibration_path", "") or None,
        "hand_joint_order": list(getattr(args, "hand_joint_order", DEFAULT_INSPIRE_HAND_JOINT_ORDER)),
        "hand_invert": bool(getattr(args, "hand_invert", True)),
        "hand_command_alpha": float(getattr(args, "hand_command_alpha", 1.0)),
        "hand_send_threshold": float(getattr(args, "hand_send_threshold", 0.002)),
        "hand_telemetry_hz": float(getattr(args, "hand_telemetry_hz", 30.0)),
        "hand_tactile_regions": list(_parse_hand_tactile_regions(getattr(args, "hand_tactile_regions", "all"))),
        "teleop_hardware_ip": args.teleop_hardware_ip or None,
        "teleop_hardware_port": int(args.teleop_hardware_port) if args.teleop_hardware_port else None,
        "swap_arm_sides": bool(args.swap_arm_sides),
        "swap_operator_sides": bool(getattr(args, "swap_operator_sides", False)),
        "tracker_input_mapping": {
            side: _tracker_source_side_for_logical_side(args, side)
            for side in components.active_sides
        },
        "calibration_snapshot": calibration.to_dict(),
        "initial_alignment_snapshot": (
            None
            if components.loaded_initial_alignment is None
            else components.loaded_initial_alignment.to_dict()
        ),
    }


def _start_episode_recording(
    recorder: EpisodeRecorder,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    now: float,
    start_reason: str,
) -> Path | None:
    episode_dir = recorder.start_episode(
        metadata=_build_episode_metadata(
            args,
            components,
            start_reason=start_reason,
        ),
        timestamp=now,
    )
    if episode_dir is not None:
        _log("Episode", f"recording started -> {episode_dir}", now=now)
    return episode_dir


def _stop_episode_recording(
    recorder: EpisodeRecorder,
    *,
    now: float,
    end_reason: str,
    summary: dict[str, object] | None = None,
) -> Path | None:
    episode_dir = recorder.stop_episode(
        end_reason=end_reason,
        summary=summary,
        timestamp=now,
    )
    if episode_dir is not None:
        _log("Episode", f"recording stopped reason={end_reason} -> {episode_dir}", now=now)
    return episode_dir

__all__ = [
    "_build_episode_metadata",
    "_debug_episode_recording_enabled",
    "_episode_root_dir",
    "_fix_arm_pose_snapshot_recorder",
    "_start_episode_recording",
    "_stop_episode_recording",
]
