from __future__ import annotations

import inspect
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from dataset_contracts.hitl import (
    EXCLUSIVE_HITL_COMMAND_METADATA_KEYS,
    EXCLUSIVE_HITL_SAMPLE_METADATA_KEY,
    ExclusiveHITLSampleLineage,
)
from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    ARM_JOINT_DOF,
    DEPLOY_ACTION_DIM,
    DEPLOY_OBSERVATION_STATE_DIM,
    HAND_DOF,
    STATE_ARM_POSE_SLICES,
    STATE_ARM_QPOS_SLICES,
    STATE_HAND_SLICES,
)

from ..recording import _to_serializable
from ..schemas import EffectorCommand, ObservationChannel, TeleopCommand
from ..session import SessionStepResult
from .action_record import collection_action_record_metadata
from .arm_mode_labels import InitArmAuditTracker, build_arm_mode_metadata
from .build_collect_components import CollectorComponents
from .video_frame import NearestFrameBuffer, VideoFrame

_COMPACT_COMMAND_METADATA_KEYS = frozenset(
    {
        "arm_action_spec",
        "arm_mode_stop_reason",
        "arm_mode_stopped",
        "arm_mode_stopped_sides",
        "arm_controlled_frame",
        "arm_reference_frame",
        "controlled_frame",
        "frame_name",
        "hand_action_spec",
        "fix_arm_active",
        "fix_arm_active_side",
        "fix_arm_effectors",
        "fix_arm_mode",
        "fix_arm_scope",
        "fix_arm_side",
        "fix_arm_warning",
        "fix_arm_original_timestamp",
        "invert",
        "init_arm_active",
        "init_arm_active_side",
        "init_arm_mode",
        "init_arm_label_active",
        "init_arm_qpos_active",
        "init_arm_qpos_operation_id",
        "init_arm_qpos_phase",
        "init_arm_qpos_selected_side",
        "init_arm_qpos_step_count",
        "init_arm_qpos_step_index",
        "init_arm_side",
        "init_arm_target_pose_base",
        "init_arm_target_qpos",
        "output_dim",
        "source_command_type",
        "source_dim",
        "source_q6",
        "source_target_pose_base",
        "source_target_pose_desk",
        "source_target_pose_operator",
        "source_value",
        "target_pose_base",
        "target_pose_controlled_frame",
        "target_pose_desk",
        "target_pose_operator",
        "target_pose_tcp_base",
        "action_pose_source",
        "execution_step_metadata",
        "solver_output_qpos",
        "tcp_frame",
        *EXCLUSIVE_HITL_COMMAND_METADATA_KEYS,
    }
)

_POSE_QUALITY_SCHEMA = "dexteleop_pose_quality_v1"
_POSE_QUALITY_SPECIAL_SINGULARITY = "singularity"
DEFAULT_CAMERA_FRAME_MAX_AGE_SEC = 0.1


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _nan_vector(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=np.float32)


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, EffectorCommand):
        return value.copy()
    if isinstance(value, ObservationChannel):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return value


def _get_state_arg_names(get_state: Any) -> set[str]:
    code = getattr(get_state, "__code__", None)
    if code is None:
        return set()
    arg_count = int(getattr(code, "co_argcount", 0) or 0) + int(
        getattr(code, "co_kwonlyargcount", 0) or 0
    )
    return {str(name) for name in code.co_varnames[:arg_count]}


def _hand_backend_state_for_sample(
    backend: Any,
    *,
    include_tactile_arrays: bool | None = None,
    copy_tactile_arrays: bool | None = None,
) -> dict[str, Any]:
    get_state = getattr(backend, "get_state")
    arg_names = _get_state_arg_names(get_state)
    kwargs: dict[str, Any] = {}
    if include_tactile_arrays is not None and "include_tactile_arrays" in arg_names:
        kwargs["include_tactile_arrays"] = bool(include_tactile_arrays)
    if copy_tactile_arrays is not None and "copy_tactile_arrays" in arg_names:
        kwargs["copy_tactile_arrays"] = bool(copy_tactile_arrays)
    if "include_tactile_vector" in arg_names:
        kwargs["include_tactile_vector"] = False
    return dict(get_state(**kwargs))


def _backend_supports_tactile_array_filter(backend: Any) -> bool:
    get_state = getattr(backend, "get_state", None)
    if get_state is None:
        return False
    return "include_tactile_arrays" in _get_state_arg_names(get_state)


def _tactile_snapshot_key(state: Mapping[str, Any]) -> tuple[str, Any] | None:
    version = state.get("tactile_version")
    if version is not None:
        return ("version", version)
    frame_ids = state.get("tactile_frame_ids")
    if isinstance(frame_ids, Mapping) and frame_ids:
        return (
            "frame_ids",
            tuple(sorted((str(key), value) for key, value in frame_ids.items())),
        )
    timestamp = state.get("tactile_timestamp")
    if timestamp is not None:
        return ("timestamp", timestamp)
    return None


def _state_has_tactile_metadata(state: Mapping[str, Any]) -> bool:
    return (
        state.get("tactile_timestamp") is not None
        or bool(state.get("tactile_devices"))
        or bool(state.get("tactile_frame_ids"))
        or bool(state.get("tactile_shapes"))
    )


def _state_has_tactile_arrays(state: Mapping[str, Any]) -> bool:
    tactile_dict = state.get("tactile_dict")
    if isinstance(tactile_dict, Mapping) and bool(tactile_dict):
        return True
    tactile = state.get("tactile")
    if tactile is None:
        return False
    try:
        return np.asarray(tactile).size > 0
    except Exception:
        return True


def _camera_debug_snapshot(
    *,
    sample_timestamp: float,
    camera_frames: Mapping[str, VideoFrame | None] | None,
) -> dict[str, Any]:
    if not camera_frames:
        return {}
    debug: dict[str, Any] = {}
    for camera_key, frame in camera_frames.items():
        name = str(camera_key)
        if frame is None:
            debug[name] = {"missing": True}
            continue
        source_timestamp = float(frame.source_timestamp)
        receive_timestamp = (
            None if frame.receive_timestamp is None else float(frame.receive_timestamp)
        )
        available_timestamp = (
            None if frame.available_timestamp is None else float(frame.available_timestamp)
        )
        debug[name] = {
            "missing": False,
            "frame_index": frame.frame_index,
            "source_timestamp": source_timestamp,
            "receive_timestamp": receive_timestamp,
            "available_timestamp": available_timestamp,
            "source_age_sec": float(sample_timestamp - source_timestamp),
            "receive_age_sec": (
                None
                if receive_timestamp is None
                else float(sample_timestamp - receive_timestamp)
            ),
            "available_age_sec": (
                None
                if available_timestamp is None
                else float(sample_timestamp - available_timestamp)
            ),
        }
    return debug


def _timestamp_delta(sample_timestamp: float, timestamp: Any) -> dict[str, float | None]:
    event_timestamp = _finite_float(timestamp)
    if event_timestamp is None:
        return {
            "last_command_timestamp": None,
            "last_command_age_sec_at_sample": None,
            "sample_to_last_command_dt_sec": None,
        }
    age = float(sample_timestamp - event_timestamp)
    return {
        "last_command_timestamp": event_timestamp,
        "last_command_age_sec_at_sample": age,
        "sample_to_last_command_dt_sec": float(-age),
    }


def _hand_debug_snapshot(
    *,
    sample_timestamp: float,
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    debug: dict[str, Any] = {}
    for side, state in hand_states.items():
        hand_state = state or {}
        frame_ids = hand_state.get("tactile_frame_ids", {})
        timings = hand_state.get("tactile_timings_ms", {})
        cpu_affinity_value = hand_state.get("tactile_process_cpu_affinity")
        if cpu_affinity_value is None:
            cpu_affinity = []
        else:
            try:
                cpu_affinity = list(cpu_affinity_value)
            except TypeError:
                cpu_affinity = [cpu_affinity_value]

        def age(timestamp: Any) -> float | None:
            if timestamp is None:
                return None
            try:
                return float(sample_timestamp - float(timestamp))
            except (TypeError, ValueError):
                return None

        debug[str(side)] = {
            "joint_age_sec": age(hand_state.get("joint_timestamp")),
            "tactile_age_sec": age(hand_state.get("tactile_timestamp")),
            "telemetry_age_sec": age(hand_state.get("telemetry_timestamp")),
            "gripper_read_elapsed_s": hand_state.get("gripper_read_elapsed_s"),
            "tactile_read_elapsed_s": hand_state.get("tactile_read_elapsed_s"),
            "tactile_gate_wait_s": hand_state.get("tactile_gate_wait_s"),
            "telemetry_elapsed_s": hand_state.get("telemetry_elapsed_s"),
            "tactile_version": hand_state.get("tactile_version"),
            "tactile_devices": list(hand_state.get("tactile_devices", ())),
            "tactile_frame_ids": dict(frame_ids) if isinstance(frame_ids, Mapping) else {},
            "tactile_timings_ms": dict(timings) if isinstance(timings, Mapping) else {},
            "tactile_read_hz": hand_state.get("tactile_read_hz"),
            "tactile_read_skipped": hand_state.get("tactile_read_skipped"),
            "tactile_process_read": hand_state.get("tactile_process_read"),
            "tactile_process_nice": hand_state.get("tactile_process_nice"),
            "tactile_process_cpu_affinity": cpu_affinity,
            "tactile_process_queue_size": hand_state.get("tactile_process_queue_size"),
            "tactile_worker_alive": hand_state.get("tactile_worker_alive"),
            "tactile_worker_receiver_alive": hand_state.get("tactile_worker_receiver_alive"),
            "tactile_parallel_read": hand_state.get("tactile_parallel_read"),
            "tactile_global_read_lock": hand_state.get("tactile_global_read_lock"),
            "telemetry_error": hand_state.get("telemetry_error"),
        }
    return debug


def _compact_command_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        str(key): _to_serializable(value)
        for key, value in metadata.items()
        if str(key) in _COMPACT_COMMAND_METADATA_KEYS
    }


def _command_metadata_values(command: TeleopCommand | None, key: str) -> list[Any]:
    if command is None:
        return []
    values: list[Any] = []
    if key in (command.metadata or {}):
        values.append(command.metadata.get(key))
    for effector_command in command.commands.values():
        metadata = effector_command.metadata or {}
        if key in metadata:
            values.append(metadata.get(key))
    return values


def _first_command_metadata(command: TeleopCommand | None, key: str) -> Any:
    for value in _command_metadata_values(command, key):
        if value is not None:
            return value
    return None


def _pose_error_summary(value: Any) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        return {"max_translation_m": None, "max_rotation_deg": None}
    translations: list[float] = []
    rotations: list[float] = []
    for metrics in value.values():
        if not isinstance(metrics, Mapping):
            continue
        translation = _finite_float(metrics.get("translation_m"))
        rotation = _finite_float(metrics.get("rotation_deg"))
        if translation is not None:
            translations.append(translation)
        if rotation is not None:
            rotations.append(rotation)
    return {
        "max_translation_m": max(translations) if translations else None,
        "max_rotation_deg": max(rotations) if rotations else None,
    }


def _joint_margin_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"nearest_margin_deg": None, "near_limit": False}
    margins: list[float] = []
    near_limit = False
    by_side: dict[str, Any] = {}
    for side, payload in value.items():
        if not isinstance(payload, Mapping):
            continue
        margin = _finite_float(payload.get("nearest_margin_deg"))
        if margin is not None:
            margins.append(margin)
        side_near = _boolish(payload.get("near_limit", False))
        near_limit = near_limit or side_near
        by_side[str(side)] = {
            "nearest_margin_deg": margin,
            "nearest_joint": payload.get("nearest_joint"),
            "near_limit": side_near,
        }
    return {
        "nearest_margin_deg": min(margins) if margins else None,
        "near_limit": near_limit,
        "by_side": by_side,
    }


def _singularity_quality_signal(command: TeleopCommand | None) -> dict[str, Any]:
    """Extract an explicit singularity signal from the accepted-qref metadata.

    The solver's candidate score is an internal ordering value and must not be
    reused as a dataset quality score.  Position collection instead consumes
    the final-qref certificate emitted by the singularity-aware path.  The
    parser also accepts the older preference payloads so that a transition
    frame cannot silently lose its label while the runtime metadata is being
    migrated.
    """

    by_side: dict[str, dict[str, Any]] = {}
    reasons: list[str] = []
    certificate_present = False
    fallback_claimed = False
    traversal_claimed = False
    active = False

    def add_reason(reason: str) -> None:
        text = str(reason)
        if text and text not in reasons:
            reasons.append(text)

    def side_payload(side: str) -> dict[str, Any]:
        normalized = str(side).strip() or "unknown"
        return by_side.setdefault(
            normalized,
            {
                "kinds": [],
                "signals": [],
                "final_qref_singular": False,
            },
        )

    def mark_side(side: str, kind: str, signal: str) -> None:
        nonlocal active
        payload = side_payload(side)
        kind_text = str(kind).strip()
        signal_text = str(signal).strip()
        if kind_text and kind_text not in payload["kinds"]:
            payload["kinds"].append(kind_text)
        if signal_text and signal_text not in payload["signals"]:
            payload["signals"].append(signal_text)
        payload["final_qref_singular"] = bool(
            payload.get("final_qref_singular", False)
            or signal_text.startswith("final_qref_")
        )
        active = True

    def ingest_final_singularity(value: Any, *, source: str) -> None:
        if not isinstance(value, Mapping):
            return
        sides = value.get("sides", value)
        if not isinstance(sides, Mapping):
            return
        for raw_side, raw_payload in sides.items():
            if not isinstance(raw_payload, Mapping):
                continue
            side = str(raw_side)
            for kind in ("elbow", "wrist", "shoulder"):
                if _boolish(raw_payload.get(f"is_{kind}_singular", False)):
                    mark_side(side, kind, f"{source}_{kind}_singular")

    def ingest_traversal(value: Any, *, kind: str) -> None:
        nonlocal traversal_claimed
        if not isinstance(value, Mapping):
            return
        sides = value.get("by_side", value)
        if not isinstance(sides, Mapping):
            return
        for raw_side, raw_active in sides.items():
            if _boolish(raw_active):
                traversal_claimed = True
                mark_side(str(raw_side), kind, f"{kind}_traversal")

    def ingest_preference(value: Any, *, kind: str) -> None:
        nonlocal fallback_claimed
        if not isinstance(value, Mapping):
            return
        if not _boolish(value.get("fallback_to_singular", False)):
            return
        fallback_claimed = True
        selected = value.get("selected", {})
        selected_sides = selected.get("sides", {}) if isinstance(selected, Mapping) else {}
        if isinstance(selected_sides, Mapping):
            for raw_side, raw_payload in selected_sides.items():
                if not isinstance(raw_payload, Mapping):
                    continue
                for kind_name in ("elbow", "wrist", "shoulder"):
                    if _boolish(raw_payload.get(f"is_{kind_name}_singular", False)):
                        mark_side(str(raw_side), kind_name, f"{kind}_fallback")

    def ingest_selection(selection: Mapping[str, Any]) -> None:
        nonlocal certificate_present, fallback_claimed
        certificate = selection.get("candidate_certificate")
        if isinstance(certificate, Mapping):
            certificate_present = True
            if _boolish(certificate.get("singular_fallback_claimed", False)):
                fallback_claimed = True
                add_reason("singularity_fallback_claimed")
            if _boolish(
                certificate.get("candidate_selected_singular_fallback_claimed", False)
            ):
                fallback_claimed = True
                add_reason("candidate_selected_singular_fallback_claimed")
            ingest_traversal(
                certificate.get("singularity_aware_elbow_traversal_by_side"),
                kind="elbow",
            )
            ingest_traversal(
                certificate.get("singularity_aware_wrist_traversal_by_side"),
                kind="wrist",
            )
            ingest_final_singularity(
                certificate.get("final_qref_position_singularity"),
                source="final_qref_position",
            )
            ingest_final_singularity(
                certificate.get("final_qref_wrist_singularity"),
                source="final_qref_wrist",
            )

        # These preference fields are retained for compatibility with command
        # metadata emitted before the candidate certificate became mandatory.
        ingest_preference(selection.get("position_singularity_preference"), kind="position")
        ingest_preference(selection.get("wrist_singularity_preference"), kind="wrist")
        if _boolish(selection.get("singular_fallback_claimed", False)):
            fallback_claimed = True
            add_reason("singularity_fallback_claimed")
        ingest_final_singularity(
            selection.get("final_qref_position_singularity"),
            source="final_qref_position",
        )
        ingest_final_singularity(
            selection.get("final_qref_wrist_singularity"),
            source="final_qref_wrist",
        )

    if command is None:
        return {
            "active": False,
            "special_code": None,
            "reasons": [],
            "by_side": {},
            "certificate_present": False,
            "fallback_claimed": False,
            "traversal_claimed": False,
        }

    for raw_event in _command_metadata_values(command, "singularity_event"):
        if not isinstance(raw_event, Mapping):
            continue
        if _boolish(raw_event.get("active", False)):
            active = True
            add_reason(str(raw_event.get("code") or "singularity_event_active"))
        if _boolish(raw_event.get("fallback_claimed", False)):
            fallback_claimed = True
        if _boolish(raw_event.get("traversal_claimed", False)):
            traversal_claimed = True
        ingest_final_singularity(
            raw_event.get("final_qref_position_singularity"),
            source="final_qref_position",
        )
        ingest_final_singularity(
            raw_event.get("final_qref_wrist_singularity"),
            source="final_qref_wrist",
        )
        raw_by_side = raw_event.get("by_side", {})
        if isinstance(raw_by_side, Mapping):
            for raw_side, raw_payload in raw_by_side.items():
                if not isinstance(raw_payload, Mapping):
                    continue
                for kind in ("elbow", "wrist", "shoulder"):
                    if _boolish(raw_payload.get(f"{kind}_singular", False)):
                        mark_side(str(raw_side), kind, "singularity_event")

    for raw_selection in _command_metadata_values(command, "ik_candidate_selection"):
        if isinstance(raw_selection, Mapping):
            ingest_selection(raw_selection)

    if fallback_claimed:
        active = True
        if "singularity_fallback_claimed" not in reasons:
            add_reason("singularity_fallback_claimed")
    if traversal_claimed:
        active = True
        add_reason("singularity_traversal")
    if by_side:
        for side, payload in by_side.items():
            for kind in payload.get("kinds", ()):
                add_reason(f"{side}_{kind}_singular")

    return {
        "active": bool(active),
        "special_code": _POSE_QUALITY_SPECIAL_SINGULARITY if active else None,
        "reasons": reasons,
        "by_side": by_side,
        "certificate_present": bool(certificate_present),
        "fallback_claimed": bool(fallback_claimed),
        "traversal_claimed": bool(traversal_claimed),
    }


def _build_pose_quality_metadata(command: TeleopCommand | None) -> dict[str, Any]:
    if command is None:
        return {
            "schema": _POSE_QUALITY_SCHEMA,
            "available": False,
            "status": "unknown",
            "score": None,
            "special_code": None,
            "source": "no_command",
            "bad": False,
            "reasons": ["no_command"],
            "metrics": {},
        }

    singularity_signal = _singularity_quality_signal(command)
    collision_status = _first_command_metadata(command, "collision_avoidance_status")
    collision_distance_m = _finite_float(_first_command_metadata(command, "collision_distance_m"))
    collision_min_distance_m = _finite_float(
        _first_command_metadata(command, "collision_min_distance_m")
    )
    fallback_reason = _first_command_metadata(command, "collision_avoidance_fallback_reason")
    branch_candidate = _first_command_metadata(command, "ik_branch_candidate")
    jump_gate_status = _first_command_metadata(command, "ik_jump_gate_status")
    time_budget_exceeded = _boolish(
        _first_command_metadata(command, "ik_time_budget_exceeded")
    )
    execution_step = _first_command_metadata(command, "execution_step_metadata")
    execution_step_limited = (
        isinstance(execution_step, Mapping)
        and _boolish(execution_step.get("step_limited"))
    )
    pose_error = _pose_error_summary(_first_command_metadata(command, "ik_solve_pose_error"))
    joint_margin = _joint_margin_summary(_first_command_metadata(command, "joint_limit_margin"))

    has_signal = any(
        value is not None
        for value in (
            collision_status,
            collision_distance_m,
            collision_min_distance_m,
            fallback_reason,
            branch_candidate,
            jump_gate_status,
            pose_error["max_translation_m"],
            joint_margin["nearest_margin_deg"],
        )
    ) or bool(singularity_signal.get("active", False))
    if not has_signal:
        return {
            "schema": _POSE_QUALITY_SCHEMA,
            "available": False,
            "status": "unknown",
            "score": None,
            "special_code": None,
            "source": "command_without_quality_metadata",
            "bad": False,
            "reasons": ["quality_metadata_unavailable"],
            "metrics": {},
        }

    bad_reasons: list[str] = []
    warning_reasons: list[str] = []
    status_text = "" if collision_status is None else str(collision_status)
    if status_text in {"hold_last_safe", "unsafe_candidate", "jump_gate_hold"}:
        bad_reasons.append(f"collision_status:{status_text}")
    elif status_text == "boundary_active":
        warning_reasons.append("collision_boundary_active")
    if fallback_reason:
        bad_reasons.append(f"fallback:{fallback_reason}")
    if jump_gate_status not in {None, "pass"}:
        warning_reasons.append(f"jump_gate:{jump_gate_status}")
    if collision_distance_m is not None and collision_min_distance_m is not None:
        margin = collision_distance_m - collision_min_distance_m
        if margin < 0.0:
            bad_reasons.append("collision_distance_below_min")
        elif margin <= 0.02:
            warning_reasons.append("collision_distance_near_min")
    nearest_margin = _finite_float(joint_margin.get("nearest_margin_deg"))
    if nearest_margin is not None:
        if nearest_margin < 3.0:
            bad_reasons.append("joint_limit_margin_lt_3deg")
        elif nearest_margin < 10.0 or _boolish(joint_margin.get("near_limit")):
            warning_reasons.append("joint_limit_margin_low")
    max_translation = _finite_float(pose_error.get("max_translation_m"))
    max_rotation = _finite_float(pose_error.get("max_rotation_deg"))
    if max_translation is not None:
        if max_translation > 0.02:
            bad_reasons.append("ik_translation_error_gt_2cm")
        elif max_translation > 0.005:
            warning_reasons.append("ik_translation_error_gt_5mm")
    if max_rotation is not None:
        if max_rotation > 5.0:
            bad_reasons.append("ik_rotation_error_gt_5deg")
        elif max_rotation > 2.0:
            warning_reasons.append("ik_rotation_error_gt_2deg")
    if execution_step_limited:
        warning_reasons.append("execution_step_limited")
    if time_budget_exceeded:
        warning_reasons.append("ik_time_budget_exceeded")

    singularity_active = bool(singularity_signal.get("active", False))
    singularity_reasons = [
        "singularity",
        *(
            str(reason)
            for reason in singularity_signal.get("reasons", ())
            if str(reason) != "singularity"
        ),
    ]
    if singularity_active:
        # A singular accepted-qref is an explicit exclusion marker for
        # training/curation, not a low-but-valid quality score.  Keep the
        # ordinary diagnostics as context while making the sentinel stable
        # across v2.1/v2.2 row adapters.
        bad_reasons = singularity_reasons + bad_reasons
        return {
            "schema": _POSE_QUALITY_SCHEMA,
            "available": True,
            "status": "singularity",
            "score": -1.0,
            "special_code": _POSE_QUALITY_SPECIAL_SINGULARITY,
            "bad": True,
            "source": "accepted_qref_singularity_certificate",
            "reasons": bad_reasons + warning_reasons,
            "bad_reasons": bad_reasons,
            "warning_reasons": warning_reasons,
            "metrics": {
                "collision_avoidance_status": collision_status,
                "collision_distance_m": collision_distance_m,
                "collision_min_distance_m": collision_min_distance_m,
                "collision_margin_m": (
                    None
                    if collision_distance_m is None or collision_min_distance_m is None
                    else float(collision_distance_m - collision_min_distance_m)
                ),
                "collision_avoidance_fallback_reason": fallback_reason,
                "ik_branch_candidate": branch_candidate,
                "ik_jump_gate_status": jump_gate_status,
                "ik_solve_pose_error": pose_error,
                "joint_limit_margin": joint_margin,
                "execution_step_limited": execution_step_limited,
                "ik_time_budget_exceeded": time_budget_exceeded,
                "singularity": singularity_signal,
            },
        }

    score = 1.0
    score -= 0.35 * len(bad_reasons)
    score -= 0.12 * len(warning_reasons)
    score = max(0.0, min(1.0, float(score)))
    if bad_reasons:
        quality_status = "bad"
    elif warning_reasons:
        quality_status = "warning"
    else:
        quality_status = "good"
    return {
        "schema": _POSE_QUALITY_SCHEMA,
        "available": True,
        "status": quality_status,
        "score": score,
        "special_code": None,
        "bad": bool(bad_reasons),
        "source": "existing_command_metadata",
        "reasons": bad_reasons + warning_reasons,
        "bad_reasons": bad_reasons,
        "warning_reasons": warning_reasons,
        "metrics": {
            "collision_avoidance_status": collision_status,
            "collision_distance_m": collision_distance_m,
            "collision_min_distance_m": collision_min_distance_m,
            "collision_margin_m": (
                None
                if collision_distance_m is None or collision_min_distance_m is None
                else float(collision_distance_m - collision_min_distance_m)
            ),
            "collision_avoidance_fallback_reason": fallback_reason,
            "ik_branch_candidate": branch_candidate,
            "ik_jump_gate_status": jump_gate_status,
            "ik_solve_pose_error": pose_error,
            "joint_limit_margin": joint_margin,
            "execution_step_limited": execution_step_limited,
            "ik_time_budget_exceeded": time_budget_exceeded,
        },
    }


def _vector_or_nan(value: Any, *, length: int) -> np.ndarray:
    if value is None:
        return _nan_vector(length)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (int(length),):
        return _nan_vector(length)
    if not np.all(np.isfinite(array)):
        return _nan_vector(length)
    return array.astype(np.float32, copy=True)


def _pose_xyz_quat_or_nan(value: Any) -> np.ndarray:
    if value is None:
        return _nan_vector(7)
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return _nan_vector(7)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return _nan_vector(7)
    quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)
    xyz = pose[:3, 3].astype(np.float32, copy=True)
    return np.concatenate((xyz, quat_xyzw), dtype=np.float32)


def _pose_matrix_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose.copy()


def _pose_delta_summary(current_pose: Any, target_pose: Any) -> dict[str, float | None]:
    current = _pose_matrix_or_none(current_pose)
    target = _pose_matrix_or_none(target_pose)
    if current is None or target is None:
        return {
            "target_current_translation_m": None,
            "target_current_rotation_deg": None,
        }
    translation = float(np.linalg.norm(target[:3, 3] - current[:3, 3]))
    try:
        relative_rot = current[:3, :3].T @ target[:3, :3]
        rotation_deg = float(np.rad2deg(R.from_matrix(relative_rot).magnitude()))
    except Exception:
        rotation_deg = None
    return {
        "target_current_translation_m": translation,
        "target_current_rotation_deg": rotation_deg,
    }


def _action_pose_source_for_command(command: EffectorCommand | None) -> str | None:
    if command is None:
        return None
    metadata_source = command.metadata.get("action_pose_source")
    if metadata_source:
        return str(metadata_source)
    if command.command_type == "cartesian_pose":
        return "command_cartesian_pose"
    if command.command_type == "joint_vector":
        for key in (
            "target_pose_base",
            "target_pose_tcp_base",
            "solver_output_tcp_pose_base",
            "solver_output_endpoint_pose_base",
            "source_target_pose_base",
        ):
            if _finite_pose_or_none(command.metadata.get(key)) is not None:
                return key
    return None


def _arm_action_debug_snapshot(
    *,
    sample_timestamp: float,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    command: TeleopCommand | None,
) -> dict[str, Any]:
    debug: dict[str, Any] = {}
    commands = {} if command is None else command.commands
    for side, state in arm_states.items():
        arm_state = state or {}
        effector_command = commands.get(f"{side}_arm")
        timing = _timestamp_delta(sample_timestamp, arm_state.get("last_command_timestamp"))
        debug[str(side)] = {
            **timing,
            "pose_source": _action_pose_source_for_command(effector_command),
            **_pose_delta_summary(
                arm_state.get("current_tcp_pose"),
                arm_state.get("last_target_tcp_pose"),
            ),
        }
    return debug


def _command_joint_or_nan(command: EffectorCommand | None, *, length: int = 6) -> np.ndarray:
    if command is None or command.command_type != "joint_vector":
        return _nan_vector(length)
    array = _vector_or_nan(command.value, length=length)
    if np.all(np.isfinite(array)):
        return array
    raw = np.asarray(command.value, dtype=np.float32).reshape(-1)
    if raw.shape != (1,) or not np.all(np.isfinite(raw)):
        return _nan_vector(length)
    spec_payload = (command.metadata or {}).get("hand_action_spec")
    if not isinstance(spec_payload, Mapping):
        return _nan_vector(length)
    active_mask = spec_payload.get("active_mask")
    if not isinstance(active_mask, Sequence) or len(active_mask) != int(length):
        return _nan_vector(length)
    active_indices = [index for index, enabled in enumerate(active_mask) if bool(enabled)]
    if len(active_indices) != 1:
        return _nan_vector(length)
    padded = np.zeros(int(length), dtype=np.float32)
    padded[int(active_indices[0])] = float(raw[0])
    return padded


def _finite_pose_or_none(value: Any) -> np.ndarray | None:
    pose = _pose_xyz_quat_or_nan(value)
    if np.all(np.isfinite(pose)):
        return pose
    return None


def _command_pose_or_nan(
    command: EffectorCommand | None,
    *,
    arm_state: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if command is None:
        return _nan_vector(7)
    for key in ("target_pose_base", "target_pose_tcp_base"):
        pose = _finite_pose_or_none(command.metadata.get(key))
        if pose is not None:
            return pose
    if command.command_type == "cartesian_pose":
        return _pose_xyz_quat_or_nan(command.value)
    if command.command_type == "joint_vector":
        if arm_state:
            for key in ("last_target_tcp_pose", "last_command_tcp_pose"):
                pose = _finite_pose_or_none(arm_state.get(key))
                if pose is not None:
                    return pose
        for key in (
            "target_pose_base",
            "target_pose_tcp_base",
            "solver_output_tcp_pose_base",
            "solver_output_endpoint_pose_base",
            "source_target_pose_base",
        ):
            pose = _finite_pose_or_none(command.metadata.get(key))
            if pose is not None:
                return pose
    return _nan_vector(7)


def _side_from_arm_command(effector_name: str, command: EffectorCommand) -> str | None:
    if command.side in {"left", "right"}:
        return str(command.side)
    text = str(effector_name or command.effector_name or "")
    if text.startswith("left_"):
        return "left"
    if text.startswith("right_"):
        return "right"
    return None


def _command_with_action_target_pose(
    command: TeleopCommand | None,
    *,
    arm_states: Mapping[str, Mapping[str, Any] | None],
    use_backend_target_pose: bool,
) -> TeleopCommand | None:
    if command is None:
        return command
    adjusted = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
    changed = False
    for effector_name, effector_command in command.commands.items():
        out = effector_command.copy()
        if str(effector_name).endswith("_arm"):
            side = _side_from_arm_command(effector_name, out)
            target_pose = _pose_matrix_or_none(out.metadata.get("target_pose_base"))
            if target_pose is None:
                target_pose = _pose_matrix_or_none(out.metadata.get("target_pose_tcp_base"))
            action_pose_source = out.metadata.get("action_pose_source")
            if target_pose is None and use_backend_target_pose and out.command_type == "joint_vector":
                arm_state = arm_states.get(side or "") or {}
                target_pose = _pose_matrix_or_none(arm_state.get("last_target_tcp_pose"))
                action_pose_source = "backend_last_target_tcp_pose"
            if target_pose is not None:
                out.metadata = {
                    **dict(out.metadata),
                    "target_pose_base": target_pose.copy(),
                    "target_pose_tcp_base": target_pose.copy(),
                    "action_pose_source": str(action_pose_source or "target_pose_metadata"),
                }
                if out.command_type == "cartesian_pose":
                    out.value = target_pose.copy()
                changed = True
        adjusted.add_command(out)
    return adjusted if changed else command


def _state_tcp_pose_or_nan(state: Mapping[str, Any] | None) -> np.ndarray:
    if not state:
        return _nan_vector(7)
    current_tcp_pose = state.get("current_tcp_pose")
    if current_tcp_pose is not None:
        return _pose_xyz_quat_or_nan(current_tcp_pose)
    last_command = state.get("last_command")
    if isinstance(last_command, EffectorCommand):
        return _command_pose_or_nan(last_command, arm_state=state)
    return _nan_vector(7)


@dataclass(frozen=True)
class CameraFrameMatch:
    camera_key: str
    frame: VideoFrame | None
    selected_timestamp: float | None
    age_sec: float | None


@dataclass(frozen=True)
class CollectorSample:
    index: int
    episode_index: int
    frame_index: int
    timestamp: float
    session_state: str
    control_hz: float
    action: np.ndarray
    observation_state: np.ndarray
    observation_last_action: np.ndarray
    arm_state_source: str
    active_sides: tuple[str, ...]
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    arm_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    hand_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)
    camera_frames: dict[str, CameraFrameMatch] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CollectorSampleSnapshot:
    """Causal live-state snapshot consumed by detached sample assembly.

    Hardware and camera-buffer reads stay in the frontend process.  Everything
    after this boundary is pure data work and may safely run in a spawned
    collection process without inheriting vendor handles or runtime locks.
    """

    step_result: SessionStepResult
    arm_states: dict[str, dict[str, Any]]
    hand_states: dict[str, dict[str, Any]]
    camera_frames: dict[str, VideoFrame | None]
    position_reference: dict[str, Any] | None
    active_sides: tuple[str, ...]
    component_metadata: dict[str, Any]
    snapshot_timings_s: dict[str, float]
    include_debug_metadata: bool = False
    include_action_debug_metadata: bool = False


@dataclass(frozen=True)
class _SnapshotComponentView:
    active_sides: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass
class SampleAssemblerConfig:
    active_sides: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        normalized = tuple(str(side) for side in self.active_sides)
        invalid = set(normalized) - {"left", "right"}
        if invalid:
            raise ValueError(f"Unsupported sides in active_sides: {sorted(invalid)}")
        self.active_sides = normalized


MetadataBuilder = Callable[
    [SessionStepResult, CollectorComponents, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any] | None,
]
ArmStateMetadataProvider = Callable[
    [CollectorComponents, float],
    Mapping[str, Mapping[str, Any]] | None,
]
HandStateProvider = Callable[
    [CollectorComponents, float],
    Mapping[str, Mapping[str, Any]] | None,
]
PositionReferenceProvider = Callable[[float], Any | None]


def _reference_value(reference: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(reference, Mapping) and name in reference:
            return reference[name]
        if hasattr(reference, name):
            return getattr(reference, name)
    return default


def _reference_side_payload(reference: Any, side: str) -> Any:
    references = _reference_value(
        reference,
        "references",
        "arm_references",
        "references_by_side",
        default=None,
    )
    if isinstance(references, Mapping):
        return references.get(str(side))
    return None


def _reference_side_value(reference: Any, side: str, *names: str) -> Any:
    for name in names:
        by_side = _reference_value(reference, f"{name}_by_side", default=None)
        if isinstance(by_side, Mapping) and str(side) in by_side:
            return by_side[str(side)]
        value = _reference_value(reference, name, default=None)
        if isinstance(value, Mapping) and str(side) in value:
            return value[str(side)]
        direct = _reference_value(reference, f"{side}_{name}", default=None)
        if direct is not None:
            return direct
    payload = _reference_side_payload(reference, side)
    if payload is not None:
        return _reference_value(payload, *names, default=None)
    return None


def _reference_qpos_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        qpos = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if qpos.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(qpos)):
        return None
    return qpos.copy()


def _causal_measured_arm_state_eligibility(
    metadata: Mapping[str, Any],
    *,
    sample_timestamp: float,
) -> tuple[bool, str]:
    """Validate a measured-state snapshot before replacing commanded state.

    Poller snapshots use the strict contract below. Older custom providers that
    do not publish contract flags remain compatible, but an explicit failure,
    invalidity, stale marker, bad sequence, or future measurement always wins
    over the mere presence of cached qpos.
    """

    if _reference_qpos_or_none(metadata.get("actual_qpos")) is None:
        return False, "missing_or_invalid_qpos"
    strict_contract = "actual_state_valid" in metadata
    if strict_contract:
        if metadata.get("actual_state_read_success") is not True:
            return False, "hardware_read_failed"
        if metadata.get("actual_state_valid") is not True:
            return False, "measurement_invalid"
        if metadata.get("actual_state_stale") is not False:
            return False, "freshness_unconfirmed_or_stale"
        sequence = metadata.get("actual_state_sequence")
        if isinstance(sequence, bool):
            return False, "invalid_measurement_sequence"
        try:
            sequence_int = int(sequence)
        except (TypeError, ValueError):
            return False, "missing_measurement_sequence"
        if sequence_int <= 0 or sequence_int != sequence:
            return False, "invalid_measurement_sequence"
    elif metadata.get("actual_state_read_success") is False:
        return False, "hardware_read_failed"
    if metadata.get("actual_state_stale") is True:
        return False, "measurement_stale"
    measurement_timestamp = _finite_float(metadata.get("actual_state_timestamp"))
    if measurement_timestamp is None:
        return False, "missing_measurement_timestamp"
    if measurement_timestamp > float(sample_timestamp) + 1e-6:
        return False, "measurement_is_from_future"
    # Providers that publish the explicit poller contract must satisfy the
    # availability bound.  Keep older custom providers source-compatible; the
    # built-in arm poller always emits ``available_timestamp``.
    if "available_timestamp" in metadata or "actual_state_read_end_time" in metadata:
        available_timestamp = _finite_float(
            metadata.get("available_timestamp", metadata.get("actual_state_read_end_time"))
        )
        if available_timestamp is None:
            return False, "missing_snapshot_available_timestamp"
        if available_timestamp > float(sample_timestamp) + 1e-6:
            return False, "snapshot_available_after_sample"
    return True, "accepted"


def _reference_pose_matrix_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.shape == (4, 4):
        return array.copy() if np.all(np.isfinite(array)) else None
    vector = array.reshape(-1)
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        return None
    quaternion = vector[3:7]
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-12:
        return None
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = R.from_quat(quaternion / quaternion_norm).as_matrix()
    pose[:3, 3] = vector[:3]
    return pose


def _reference_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        reasons: list[str] = []
        for key, item in value.items():
            for reason in _reference_reasons(item):
                reasons.append(f"{key}:{reason}")
        return reasons
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _reference_pipeline_version(reference: Any) -> str | None:
    value = _reference_value(reference, "pipeline_version", "schema_version", default=None)
    if value is None:
        frame_metadata = _reference_value(reference, "frame_metadata", "metadata", default=None)
        if isinstance(frame_metadata, Mapping):
            value = frame_metadata.get("pipeline_version", frame_metadata.get("schema_version"))
    return None if value is None else str(value)


def _normalize_position_reference(
    reference: Any,
    *,
    sample_timestamp: float,
    active_sides: Sequence[str],
    max_age_sec: float | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if reference is None:
        return None, "no_reference_at_or_before_sample"

    published_timestamp = _finite_float(
        _reference_value(
            reference,
            "published_wall_time",
            "published_timestamp",
            "published_time",
            "reference_time",
            "timestamp",
            default=None,
        )
    )
    if published_timestamp is None:
        return None, "missing_published_timestamp"
    if published_timestamp > float(sample_timestamp):
        return None, "future_reference"
    wall_age_sec = float(sample_timestamp) - float(published_timestamp)
    selected_age_sec = _finite_float(
        _reference_value(reference, "selection_age_sec", "age_sec", default=None)
    )
    age_sec = wall_age_sec if selected_age_sec is None else max(0.0, selected_age_sec)
    if max_age_sec is not None and age_sec > float(max_age_sec):
        return None, "stale_reference"

    valid = bool(_reference_value(reference, "valid", "is_valid", default=True))
    if not valid:
        return None, "invalid_reference"
    epoch = _to_serializable(_reference_value(reference, "epoch", default=None))
    if epoch is None:
        return None, "missing_reference_epoch"
    sequence = _to_serializable(_reference_value(reference, "sequence", default=None))
    if sequence is None:
        return None, "missing_reference_sequence"
    raw_frame_metadata = _reference_value(
        reference,
        "frame_metadata",
        "metadata",
        default={},
    )
    if raw_frame_metadata is None:
        raw_frame_metadata = {}
    if not isinstance(raw_frame_metadata, Mapping):
        return None, "invalid_reference_frame_metadata"
    frame_metadata = dict(_to_serializable(raw_frame_metadata))
    hitl_payload = frame_metadata.get(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY)
    if hitl_payload is not None:
        if not isinstance(hitl_payload, Mapping):
            return None, "invalid_reference_hitl_lineage"
        try:
            hitl_lineage = ExclusiveHITLSampleLineage.from_payload(hitl_payload)
        except (TypeError, ValueError):
            return None, "invalid_reference_hitl_lineage"
        frame_metadata[EXCLUSIVE_HITL_SAMPLE_METADATA_KEY] = (
            hitl_lineage.to_payload()
        )

    q_ref_by_side: dict[str, np.ndarray] = {}
    q_ik_raw_by_side: dict[str, np.ndarray] = {}
    pose_ref_by_side: dict[str, np.ndarray] = {}
    side_valid_by_side: dict[str, bool] = {}
    side_hold_by_side: dict[str, bool] = {}
    reasons_by_side: dict[str, list[str]] = {}
    normalized_active_sides = tuple(str(side) for side in active_sides)
    for side in ALL_SIDES:
        payload = _reference_side_payload(reference, side)
        side_valid = bool(
            _reference_value(payload, "valid", "is_valid", default=True)
            if payload is not None
            else True
        )
        side_hold = bool(
            _reference_value(payload, "hold", "is_hold", default=False)
            if payload is not None
            else False
        )
        q_ref = _reference_qpos_or_none(
            _reference_side_value(reference, side, "q_ref", "joint_reference")
        )
        q_ik_raw = _reference_qpos_or_none(
            _reference_side_value(reference, side, "q_ik_raw", "raw_ik_qpos")
        )
        pose_ref = _reference_pose_matrix_or_none(
            _reference_side_value(reference, side, "pose_ref", "action_pose_ref")
        )
        reasons = _reference_reasons(
            _reference_side_value(reference, side, "modification_reasons", "modification_reason")
        )
        if q_ref is not None:
            q_ref_by_side[side] = q_ref
        if q_ik_raw is not None:
            q_ik_raw_by_side[side] = q_ik_raw
        if pose_ref is not None:
            pose_ref_by_side[side] = pose_ref
        if payload is not None or q_ref is not None or pose_ref is not None:
            side_valid_by_side[side] = side_valid
            side_hold_by_side[side] = side_hold
            reasons_by_side[side] = reasons

    for side in normalized_active_sides:
        if side not in q_ref_by_side:
            return None, f"missing_q_ref:{side}"
        if side not in pose_ref_by_side:
            return None, f"missing_pose_ref:{side}"
        if not side_valid_by_side.get(side, True):
            return None, f"invalid_side_reference:{side}"

    frame_reasons = _reference_reasons(
        _reference_value(reference, "modification_reasons", "modification_reason", default=None)
    )
    all_reasons = list(frame_reasons)
    for side in normalized_active_sides:
        all_reasons.extend(f"{side}:{reason}" for reason in reasons_by_side.get(side, ()))

    source_value = _reference_value(reference, "source", "source_label", default=None)
    source_timestamp = _finite_float(
        _reference_value(reference, "source_wall_time", "source_timestamp", "source_time")
    )
    generated_timestamp = _finite_float(
        _reference_value(
            reference,
            "generated_wall_time",
            "generated_timestamp",
            "generated_time",
        )
    )
    generated_monotonic_timestamp = _finite_float(
        _reference_value(
            reference,
            "generated_monotonic_time",
            "generated_monotonic_timestamp",
        )
    )
    published_monotonic_timestamp = _finite_float(
        _reference_value(
            reference,
            "published_monotonic_time",
            "published_monotonic_timestamp",
        )
    )
    return {
        "epoch": epoch,
        "sequence": sequence,
        "source": None if source_value is None else str(source_value),
        "source_timestamp": source_timestamp,
        "generated_timestamp": generated_timestamp,
        "published_timestamp": float(published_timestamp),
        "generated_monotonic_timestamp": generated_monotonic_timestamp,
        "published_monotonic_timestamp": published_monotonic_timestamp,
        "age_sec": float(age_sec),
        "wall_age_sec": float(wall_age_sec),
        "age_clock": _reference_value(
            reference,
            "selection_age_clock",
            default="wall",
        ),
        "valid": True,
        "hold": bool(
            _reference_value(reference, "hold", "is_hold", default=False)
            or any(side_hold_by_side.get(side, False) for side in normalized_active_sides)
        ),
        "modification_reasons": all_reasons,
        "modification_reasons_by_side": reasons_by_side,
        "pipeline_version": _reference_pipeline_version(reference),
        "q_ref_by_side": {
            side: qpos.copy() for side, qpos in q_ref_by_side.items()
        },
        "q_ik_raw_by_side": {
            side: qpos.copy() for side, qpos in q_ik_raw_by_side.items()
        },
        "pose_ref_by_side": {
            side: pose.copy() for side, pose in pose_ref_by_side.items()
        },
        "side_valid_by_side": side_valid_by_side,
        "side_hold_by_side": side_hold_by_side,
        "frame_metadata": frame_metadata,
    }, None


def _position_reference_command(
    command: TeleopCommand | None,
    reference: Mapping[str, Any],
    *,
    active_sides: Sequence[str],
) -> TeleopCommand:
    published_timestamp = float(reference["published_timestamp"])
    current_hitl = ExclusiveHITLSampleLineage.from_command_metadata(
        None if command is None else command.metadata
    )
    frame_metadata = reference.get("frame_metadata", {})
    if not isinstance(frame_metadata, Mapping):
        raise ValueError("position reference frame_metadata must be a mapping.")
    reference_hitl_payload = frame_metadata.get(EXCLUSIVE_HITL_SAMPLE_METADATA_KEY)
    reference_hitl = None
    if reference_hitl_payload is not None:
        if not isinstance(reference_hitl_payload, Mapping):
            raise ValueError("position reference HITL lineage must be a mapping.")
        reference_hitl = ExclusiveHITLSampleLineage.from_payload(
            reference_hitl_payload
        )
    if current_hitl is not None and reference_hitl is None:
        raise ValueError(
            "HITL collection cannot reuse a position reference without causal lineage."
        )
    base_metadata = (
        {}
        if command is None
        else {
            key: value
            for key, value in command.metadata.items()
            if key not in EXCLUSIVE_HITL_COMMAND_METADATA_KEYS
        }
    )
    out = TeleopCommand(
        timestamp=published_timestamp,
        metadata=base_metadata,
    )
    if command is not None:
        for effector_name, effector_command in command.commands.items():
            if not str(effector_name).endswith("_arm"):
                out.add_command(effector_command.copy())

    q_ref_by_side = reference.get("q_ref_by_side", {})
    pose_ref_by_side = reference.get("pose_ref_by_side", {})
    for side in tuple(str(value) for value in active_sides):
        q_ref = _reference_qpos_or_none(q_ref_by_side.get(side))
        pose_ref = _reference_pose_matrix_or_none(pose_ref_by_side.get(side))
        if q_ref is None or pose_ref is None:
            continue
        existing = None if command is None else command.commands.get(f"{side}_arm")
        metadata = {} if existing is None else dict(existing.metadata)
        metadata.update(
            {
                "target_pose_base": pose_ref.copy(),
                "target_pose_tcp_base": pose_ref.copy(),
                "solver_output_qpos": q_ref.copy(),
                "position_reference_epoch": reference.get("epoch"),
                "position_reference_sequence": reference.get("sequence"),
                "position_reference_published_timestamp": published_timestamp,
                "action_pose_source": "position_reference_frame_fk",
            }
        )
        out.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=q_ref.copy(),
                frame_name=(
                    existing.frame_name
                    if existing is not None and existing.frame_name is not None
                    else f"{side}_arm_base"
                ),
                metadata=metadata,
            )
        )
    out.metadata.update(
        {
            "position_reference_epoch": reference.get("epoch"),
            "position_reference_sequence": reference.get("sequence"),
            "position_reference_published_timestamp": published_timestamp,
        }
    )
    if reference_hitl is not None:
        out.metadata.update(reference_hitl.to_command_metadata())
    return out


def _step_with_command(step_result: SessionStepResult, command: TeleopCommand) -> SessionStepResult:
    return SessionStepResult(
        timestamp=float(step_result.timestamp),
        state=step_result.state,
        control_hz=float(step_result.control_hz),
        channels=dict(step_result.channels),
        raw_channels=dict(step_result.raw_channels),
        observation=step_result.observation,
        command=command,
        command_dispatched=bool(step_result.command_dispatched),
        action_record=None,
        faults=list(step_result.faults),
    )


@dataclass
class CollectorSampleBuilderConfig:
    active_sides: tuple[str, ...] = ("left", "right")
    arm_backend_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_arm", "right": "right_arm"}
    )
    hand_backend_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_hand", "right": "right_hand"}
    )
    camera_buffers: dict[str, NearestFrameBuffer] = field(default_factory=dict)
    require_camera_frames: bool = True
    camera_frame_max_age_sec: float | None = DEFAULT_CAMERA_FRAME_MAX_AGE_SEC
    episode_index: int = 0
    frame_index_start: int = 0
    sample_index_start: int = 0
    include_debug_metadata: bool = False
    include_action_debug_metadata: bool = False
    tactile_array_snapshot_policy: str = "on_change"
    position_reference_max_age_sec: float | None = 0.2
    position_reference_epoch: Any = None
    prefer_causal_measured_arm_state: bool = False

    def __post_init__(self) -> None:
        self.active_sides = tuple(str(side) for side in self.active_sides)
        self.arm_backend_names = {
            str(side): str(name) for side, name in dict(self.arm_backend_names).items()
        }
        self.hand_backend_names = {
            str(side): str(name) for side, name in dict(self.hand_backend_names).items()
        }
        self.camera_buffers = {
            str(camera_key): buffer for camera_key, buffer in dict(self.camera_buffers).items()
        }
        self.require_camera_frames = bool(self.require_camera_frames)
        if self.camera_frame_max_age_sec is not None:
            self.camera_frame_max_age_sec = float(self.camera_frame_max_age_sec)
            if self.camera_frame_max_age_sec <= 0.0:
                raise ValueError("camera_frame_max_age_sec must be positive or None.")
        self.include_debug_metadata = bool(self.include_debug_metadata)
        self.include_action_debug_metadata = bool(self.include_action_debug_metadata)
        self.prefer_causal_measured_arm_state = bool(
            self.prefer_causal_measured_arm_state
        )
        if self.position_reference_max_age_sec is not None:
            self.position_reference_max_age_sec = float(self.position_reference_max_age_sec)
            if self.position_reference_max_age_sec <= 0.0:
                raise ValueError("position_reference_max_age_sec must be positive or None.")
        policy = str(self.tactile_array_snapshot_policy).strip().lower().replace("-", "_")
        if policy not in {"always", "on_change", "never"}:
            raise ValueError("tactile_array_snapshot_policy must be always, on_change, or never.")
        self.tactile_array_snapshot_policy = policy


class SampleAssembler:
    def __init__(self, config: SampleAssemblerConfig | None = None):
        self.config = config if config is not None else SampleAssemblerConfig()
        self._last_action = _nan_vector(DEPLOY_ACTION_DIM)

    def reset_last_action(self) -> None:
        self._last_action = _nan_vector(DEPLOY_ACTION_DIM)

    def _side_enabled(self, side: str) -> bool:
        return str(side) in self.config.active_sides

    def _build_action_vector(
        self,
        command: TeleopCommand | None,
        *,
        arm_states: Mapping[str, Mapping[str, Any] | None] | None = None,
        use_backend_target_pose: bool = False,
    ) -> np.ndarray:
        if command is None:
            return _nan_vector(DEPLOY_ACTION_DIM)
        commands = command.commands
        states = arm_states or {}
        action = _nan_vector(DEPLOY_ACTION_DIM)
        for side in ALL_SIDES:
            arm_state = states.get(side) if use_backend_target_pose else None
            action[ACTION_ARM_POSE_SLICES[side]] = _command_pose_or_nan(
                commands.get(f"{side}_arm"),
                arm_state=arm_state,
            )
            action[ACTION_HAND_SLICES[side]] = _command_joint_or_nan(
                commands.get(f"{side}_hand"),
                length=HAND_DOF,
            )
        return action.astype(np.float32, copy=False)

    def _build_observation_state(
        self,
        *,
        arm_states: Mapping[str, Mapping[str, Any] | None],
        hand_states: Mapping[str, Mapping[str, Any] | None],
    ) -> np.ndarray:
        state = _nan_vector(DEPLOY_OBSERVATION_STATE_DIM)
        for side in ALL_SIDES:
            if not self._side_enabled(side):
                continue
            arm_state = arm_states.get(side) or {}
            hand_state = hand_states.get(side) or {}
            state[STATE_ARM_QPOS_SLICES[side]] = _vector_or_nan(
                arm_state.get("current_qpos"),
                length=ARM_JOINT_DOF,
            )
            state[STATE_ARM_POSE_SLICES[side]] = _state_tcp_pose_or_nan(arm_state)
            state[STATE_HAND_SLICES[side]] = _vector_or_nan(
                hand_state.get("joint_pos"),
                length=HAND_DOF,
            )
        return state.astype(np.float32, copy=False)

    def _camera_matches(
        self,
        *,
        sample_timestamp: float,
        camera_frames: Mapping[str, VideoFrame | None] | None,
    ) -> dict[str, CameraFrameMatch]:
        if not camera_frames:
            return {}
        matches: dict[str, CameraFrameMatch] = {}
        for camera_key, frame in camera_frames.items():
            camera_name = str(camera_key)
            if frame is None:
                matches[camera_name] = CameraFrameMatch(
                    camera_key=camera_name,
                    frame=None,
                    selected_timestamp=None,
                    age_sec=None,
                )
                continue
            age_sec = float(sample_timestamp - float(frame.source_timestamp))
            matches[camera_name] = CameraFrameMatch(
                camera_key=camera_name,
                frame=frame,
                selected_timestamp=float(frame.source_timestamp),
                age_sec=age_sec,
            )
        return matches

    def build_sample(
        self,
        *,
        step_result: SessionStepResult,
        arm_states: Mapping[str, Mapping[str, Any] | None],
        hand_states: Mapping[str, Mapping[str, Any] | None],
        camera_frames: Mapping[str, VideoFrame | None] | None,
        index: int,
        episode_index: int,
        frame_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> CollectorSample:
        action = self._build_action_vector(
            step_result.command,
            arm_states=arm_states,
            use_backend_target_pose=bool(getattr(step_result, "command_dispatched", False)),
        )
        observation_state = self._build_observation_state(
            arm_states=arm_states,
            hand_states=hand_states,
        )
        active_arm_state_sources = {
            str((arm_states.get(side) or {}).get("runtime_arm_state_source", "last_commanded_action"))
            for side in self.config.active_sides
        }
        if active_arm_state_sources == {"causal_measured_hardware"}:
            arm_state_source = "causal_measured_hardware"
        elif "causal_measured_hardware" in active_arm_state_sources:
            arm_state_source = "mixed_causal_measured_and_commanded"
        else:
            arm_state_source = "last_commanded_action"
        sample = CollectorSample(
            index=int(index),
            episode_index=int(episode_index),
            frame_index=int(frame_index),
            timestamp=float(step_result.timestamp),
            session_state=str(step_result.state.value),
            control_hz=float(step_result.control_hz),
            action=action,
            observation_state=observation_state,
            observation_last_action=self._last_action.copy(),
            arm_state_source=arm_state_source,
            active_sides=tuple(self.config.active_sides),
            channels={
                name: channel.copy()
                for name, channel in step_result.channels.items()
            },
            arm_states={
                str(side): _copy_value(state) if state is not None else {}
                for side, state in arm_states.items()
            },
            hand_states={
                str(side): _copy_value(state) if state is not None else {}
                for side, state in hand_states.items()
            },
            faults=[
                _to_serializable(fault)
                for fault in step_result.faults
            ],
            camera_frames=self._camera_matches(
                sample_timestamp=float(step_result.timestamp),
                camera_frames=camera_frames,
            ),
            metadata={} if metadata is None else dict(_to_serializable(metadata)),
        )
        self._last_action = action.copy()
        return sample


class CollectorSampleBuilder:
    def __init__(
        self,
        config: CollectorSampleBuilderConfig | None = None,
        *,
        assembler: SampleAssembler | None = None,
        metadata_builder: MetadataBuilder | None = None,
        arm_state_metadata_provider: ArmStateMetadataProvider | None = None,
        hand_state_provider: HandStateProvider | None = None,
        hand_state_provider_returns_isolated: bool = False,
        position_reference_provider: PositionReferenceProvider | None = None,
        position_reference_timeline: Any | None = None,
    ):
        if position_reference_provider is not None and position_reference_timeline is not None:
            raise ValueError(
                "Provide either position_reference_provider or position_reference_timeline, not both."
            )
        self.config = config if config is not None else CollectorSampleBuilderConfig()
        self.assembler = assembler if assembler is not None else SampleAssembler(
            SampleAssemblerConfig(active_sides=self.config.active_sides)
        )
        self.metadata_builder = metadata_builder
        self.arm_state_metadata_provider = arm_state_metadata_provider
        self.hand_state_provider = hand_state_provider
        self.hand_state_provider_returns_isolated = bool(
            hand_state_provider_returns_isolated
        )
        self.position_reference_provider = position_reference_provider
        self.position_reference_timeline = position_reference_timeline
        self.last_position_reference_rejection: str | None = None
        self.last_camera_rejection: str | None = None
        self.last_camera_frame_age_sec_by_key: dict[str, float | None] = {}
        self.include_debug_metadata = bool(self.config.include_debug_metadata)
        self.include_action_debug_metadata = bool(self.config.include_action_debug_metadata)
        self._episode_index = int(self.config.episode_index)
        self._frame_index = int(self.config.frame_index_start)
        self._sample_index = int(self.config.sample_index_start)
        self._last_tactile_snapshot_key_by_side: dict[str, tuple[str, Any] | None] = {}
        self._init_arm_audit_tracker = InitArmAuditTracker()

    @property
    def position_reference_enabled(self) -> bool:
        return bool(
            self.position_reference_provider is not None
            or self.position_reference_timeline is not None
        )

    def _lookup_position_reference(self, sample_timestamp: float) -> Any | None:
        provider = self.position_reference_provider
        if provider is not None:
            return provider(float(sample_timestamp))

        timeline = self.position_reference_timeline
        if timeline is None:
            return None
        lookup = getattr(timeline, "latest_at_or_before", None)
        if not callable(lookup):
            lookup = getattr(timeline, "get_latest_before_or_at", None)
        if not callable(lookup):
            raise TypeError(
                "position_reference_timeline must provide latest_at_or_before(sample_time, ...)."
            )

        try:
            parameters = inspect.signature(lookup).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs: dict[str, Any] = {}
        if "epoch" in parameters:
            if self.config.position_reference_epoch is None:
                raise ValueError(
                    "position_reference_epoch is required when injecting a ReferenceTimeline "
                    "directly; use a provider to bind a dynamic episode epoch."
                )
            kwargs["epoch"] = self.config.position_reference_epoch
        if "max_age" in parameters:
            kwargs["max_age"] = self.config.position_reference_max_age_sec
        elif "max_age_sec" in parameters:
            kwargs["max_age_sec"] = self.config.position_reference_max_age_sec
        return lookup(float(sample_timestamp), **kwargs)

    def _position_reference_for_sample(
        self,
        sample_timestamp: float,
    ) -> dict[str, Any] | None:
        if not self.position_reference_enabled:
            self.last_position_reference_rejection = None
            return None
        reference = self._lookup_position_reference(float(sample_timestamp))
        normalized, rejection = _normalize_position_reference(
            reference,
            sample_timestamp=float(sample_timestamp),
            active_sides=self.config.active_sides,
            max_age_sec=self.config.position_reference_max_age_sec,
        )
        if (
            normalized is not None
            and self.config.position_reference_epoch is not None
            and normalized.get("epoch") != self.config.position_reference_epoch
        ):
            normalized = None
            rejection = "reference_epoch_mismatch"
        self.last_position_reference_rejection = rejection
        return normalized

    @property
    def episode_index(self) -> int:
        return int(self._episode_index)

    @property
    def frame_index(self) -> int:
        return int(self._frame_index)

    @property
    def sample_index(self) -> int:
        return int(self._sample_index)

    def next_episode(self) -> int:
        self._episode_index += 1
        self._frame_index = 0
        self._last_tactile_snapshot_key_by_side.clear()
        self._init_arm_audit_tracker.reset()
        return int(self._episode_index)

    def set_episode(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)
        self._frame_index = 0
        self.assembler.reset_last_action()
        self._last_tactile_snapshot_key_by_side.clear()
        self._init_arm_audit_tracker.reset()

    def set_dataset_position(self, *, episode_index: int, sample_index: int) -> None:
        """Reset cached writer cursors after idle out-of-band dataset changes."""

        self.set_episode(int(episode_index))
        self._sample_index = int(sample_index)

    def reset_temporal_history(self) -> None:
        """Start a new model-continuity segment inside the current episode."""

        self.assembler.reset_last_action()
        self._init_arm_audit_tracker.reset()

    def _snapshot_arm_state_metadata(
        self,
        components: CollectorComponents,
        *,
        sample_timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        provider = self.arm_state_metadata_provider
        if provider is None:
            return {}
        try:
            metadata = provider(components, float(sample_timestamp))
        except Exception as exc:  # noqa: BLE001 - annotation failures should not block collection.
            return {
                side: {
                    "actual_state_provider_error": f"{type(exc).__name__}: {exc}",
                }
                for side in ("left", "right")
            }
        if not isinstance(metadata, Mapping):
            return {}
        return {
            str(side): dict(_to_serializable(value))
            for side, value in metadata.items()
            if isinstance(value, Mapping)
        }

    def _snapshot_arm_states(
        self,
        components: CollectorComponents,
        *,
        sample_timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        actual_metadata = self._snapshot_arm_state_metadata(
            components,
            sample_timestamp=sample_timestamp,
        )
        for side in ("left", "right"):
            backend_name = self.config.arm_backend_names.get(side)
            if backend_name is None:
                states[side] = {}
                continue
            backend = components.backends.get(backend_name)
            if backend is None or not hasattr(backend, "get_state"):
                states[side] = {}
                continue
            state = dict(getattr(backend, "get_state")())
            compute_tcp_pose = getattr(backend, "compute_tcp_pose", None)
            if callable(compute_tcp_pose):
                try:
                    state["current_tcp_pose"] = compute_tcp_pose()
                except Exception:
                    state["current_tcp_pose"] = None
                last_target_qpos = state.get("last_target_qpos")
                if last_target_qpos is not None and state.get("last_target_tcp_pose") is None:
                    try:
                        state["last_target_tcp_pose"] = compute_tcp_pose(qpos=last_target_qpos)
                    except Exception:
                        state["last_target_tcp_pose"] = None
            extra = actual_metadata.get(side)
            if self.config.prefer_causal_measured_arm_state and extra is None:
                # The live backend dictionary is mutable and may have been
                # refreshed after T0.  Keep only its commanded anchor as a
                # compatibility fallback; measured fields remain explicitly
                # absent until the causal poller supplies a valid snapshot.
                commanded_qpos = _reference_qpos_or_none(state.get("current_qpos"))
                commanded_tcp_pose = _pose_matrix_or_none(state.get("current_tcp_pose"))
                if commanded_qpos is not None:
                    state["commanded_qpos"] = commanded_qpos
                    state["current_qpos"] = commanded_qpos.copy()
                else:
                    state["current_qpos"] = None
                if commanded_tcp_pose is not None:
                    state["commanded_tcp_pose"] = commanded_tcp_pose
                    state["current_tcp_pose"] = commanded_tcp_pose.copy()
                else:
                    state["current_tcp_pose"] = None
                state.update(
                    {
                        "actual_qpos": None,
                        "actual_tcp_pose": None,
                        "actual_hardware_qpos": None,
                        "actual_hardware_pose_6d": None,
                        "wrist_camera_pose_base": None,
                        "causal_measured_state_eligible": False,
                        "causal_measured_state_rejection_reason": "no_causal_feedback",
                        "causal_snapshot_contract": (
                            "source_timestamp<=T0_and_available_timestamp<=T0"
                        ),
                    }
                )
            if extra:
                if self.config.prefer_causal_measured_arm_state:
                    commanded_qpos = _reference_qpos_or_none(state.get("current_qpos"))
                    commanded_tcp_pose = _pose_matrix_or_none(state.get("current_tcp_pose"))
                    actual_qpos = _reference_qpos_or_none(extra.get("actual_qpos"))
                    actual_tcp_pose = _pose_matrix_or_none(extra.get("actual_tcp_pose"))
                    actual_state_eligible, actual_state_reason = (
                        _causal_measured_arm_state_eligibility(
                            extra,
                            sample_timestamp=sample_timestamp,
                        )
                    )
                    if commanded_qpos is not None:
                        state["commanded_qpos"] = commanded_qpos
                    if commanded_tcp_pose is not None:
                        state["commanded_tcp_pose"] = commanded_tcp_pose
                state.update(_copy_value(extra))
                if self.config.prefer_causal_measured_arm_state:
                    state["causal_measured_state_eligible"] = bool(actual_state_eligible)
                    state["causal_measured_state_rejection_reason"] = actual_state_reason
                    if not actual_state_eligible:
                        # Never serialize rejected cached feedback in fields
                        # whose schema says "actual/measured".
                        state["actual_qpos"] = None
                        state["actual_tcp_pose"] = None
                        state["actual_hardware_qpos"] = None
                        state["actual_hardware_pose_6d"] = None
                        state["wrist_camera_pose_base"] = None
                if (
                    self.config.prefer_causal_measured_arm_state
                    and actual_qpos is not None
                    and actual_state_eligible
                ):
                    state["current_qpos"] = actual_qpos
                    if actual_tcp_pose is not None:
                        state["current_tcp_pose"] = actual_tcp_pose
                    state["runtime_arm_state_source"] = "causal_measured_hardware"
                else:
                    state.setdefault("runtime_arm_state_source", "last_commanded_action")
            else:
                state.setdefault("runtime_arm_state_source", "last_commanded_action")
            states[side] = state
        return states

    def _snapshot_hand_states(
        self,
        components: CollectorComponents,
        *,
        sample_timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        if self.hand_state_provider is not None:
            provided = self.hand_state_provider(components, float(sample_timestamp))
            if not isinstance(provided, Mapping):
                return {}
            if self.hand_state_provider_returns_isolated:
                return {
                    str(side): dict(state)
                    for side, state in provided.items()
                    if isinstance(state, Mapping)
                }
            return {
                str(side): _copy_value(state)
                for side, state in provided.items()
                if isinstance(state, Mapping)
            }
        states: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            backend_name = self.config.hand_backend_names.get(side)
            if backend_name is None:
                states[side] = {}
                continue
            backend = components.backends.get(backend_name)
            if backend is None or not hasattr(backend, "get_state"):
                states[side] = {}
                continue
            states[side] = self._snapshot_hand_state(side, backend)
        return states

    def _snapshot_hand_state(self, side: str, backend: Any) -> dict[str, Any]:
        policy = str(self.config.tactile_array_snapshot_policy)
        if policy == "always" or not _backend_supports_tactile_array_filter(backend):
            return _hand_backend_state_for_sample(backend, copy_tactile_arrays=False)
        if policy == "never":
            return _hand_backend_state_for_sample(backend, include_tactile_arrays=False)

        light_state = _hand_backend_state_for_sample(
            backend,
            include_tactile_arrays=False,
            copy_tactile_arrays=False,
        )
        snapshot_key = _tactile_snapshot_key(light_state)
        if snapshot_key is None or not _state_has_tactile_metadata(light_state):
            return light_state
        if self._last_tactile_snapshot_key_by_side.get(side) == snapshot_key:
            return light_state

        full_state = _hand_backend_state_for_sample(
            backend,
            include_tactile_arrays=True,
            copy_tactile_arrays=False,
        )
        if _state_has_tactile_arrays(full_state):
            self._last_tactile_snapshot_key_by_side[side] = snapshot_key
            return full_state
        return light_state

    def _snapshot_camera_frames(self, timestamp: float) -> dict[str, VideoFrame | None]:
        frames: dict[str, VideoFrame | None] = {}
        for camera_key, buffer in self.config.camera_buffers.items():
            frames[str(camera_key)] = buffer.get_latest_before_or_at(timestamp)
        return frames

    def _camera_rejection(
        self,
        *,
        sample_timestamp: float,
        camera_frames: Mapping[str, VideoFrame | None],
    ) -> str | None:
        if not self.config.camera_buffers or not self.config.require_camera_frames:
            self.last_camera_frame_age_sec_by_key = {}
            return None

        max_age = self.config.camera_frame_max_age_sec
        age_by_key: dict[str, float | None] = {}
        failures: list[str] = []
        for camera_key in sorted(self.config.camera_buffers):
            frame = camera_frames.get(camera_key)
            if frame is None:
                age_by_key[camera_key] = None
                failures.append(f"unavailable:{camera_key}")
                continue

            source_timestamp = _finite_float(frame.source_timestamp)
            available_timestamp = _finite_float(frame.available_timestamp)
            if source_timestamp is None or available_timestamp is None:
                age_by_key[camera_key] = None
                failures.append(f"invalid_timestamp:{camera_key}")
                continue

            source_age_sec = float(sample_timestamp - source_timestamp)
            available_age_sec = float(sample_timestamp - available_timestamp)
            effective_age_sec = max(source_age_sec, available_age_sec)
            age_by_key[camera_key] = effective_age_sec
            if source_age_sec < 0.0 or available_age_sec < 0.0:
                failures.append(
                    f"future:{camera_key}:{min(source_age_sec, available_age_sec):.6f}"
                )
                continue
            if max_age is not None and effective_age_sec > max_age:
                failures.append(f"stale:{camera_key}:{effective_age_sec:.6f}")

        self.last_camera_frame_age_sec_by_key = age_by_key
        if not failures:
            return None
        return "camera_frame_" + ",".join(failures)

    @staticmethod
    def _default_metadata(
        step_result: SessionStepResult,
        _components: CollectorComponents,
        _arm_states: Mapping[str, Any],
        _hand_states: Mapping[str, Any],
        camera_frames: Mapping[str, VideoFrame | None],
        *,
        include_debug_metadata: bool = False,
        include_action_debug_metadata: bool = False,
    ) -> dict[str, Any]:
        command = step_result.command
        canonical_command = _command_with_action_target_pose(
            command,
            arm_states=_arm_states,
            use_backend_target_pose=bool(getattr(step_result, "command_dispatched", False)),
        )
        canonical_step_result = step_result
        if canonical_command is not command:
            canonical_step_result = SessionStepResult(
                timestamp=float(step_result.timestamp),
                state=step_result.state,
                control_hz=float(step_result.control_hz),
                channels=dict(step_result.channels),
                raw_channels=dict(step_result.raw_channels),
                observation=step_result.observation,
                command=canonical_command,
                command_dispatched=bool(step_result.command_dispatched),
                action_record=None,
                faults=list(step_result.faults),
            )
        command_metadata = (
            {}
            if command is None
            else (
                _to_serializable(command.metadata)
                if include_debug_metadata
                else _compact_command_metadata(command.metadata)
            )
        )
        command_metadata_by_effector = (
            {}
            if command is None
            else {
                effector_name: (
                    _to_serializable(effector_command.metadata)
                    if include_debug_metadata
                    else _compact_command_metadata(effector_command.metadata)
                )
                for effector_name, effector_command in command.commands.items()
            }
        )
        hitl_lineage = ExclusiveHITLSampleLineage.from_command_metadata(
            None if command is None else command.metadata
        )
        channel_metadata = {
            channel_name: {
                "timestamp": float(channel.timestamp),
                "source_name": channel.source_name,
                "frame_name": channel.frame_name,
                "side": channel.side,
                "capability": channel.capability,
                "valid": bool(channel.valid),
                "sender_timestamp": channel.metadata.get("sender_timestamp"),
                "receiver_timestamp": channel.metadata.get("receiver_timestamp"),
                "sender_read_elapsed_ms": channel.metadata.get("sender_read_elapsed_ms"),
            }
            for channel_name, channel in step_result.channels.items()
        }
        metadata = {
            "canonical_action_record": collection_action_record_metadata(
                canonical_step_result,
                include_debug_metadata=include_debug_metadata,
            ),
            "pose_quality": _build_pose_quality_metadata(command),
            # Compatibility/default payload only.  CollectorSampleBuilder
            # replaces this with its actual-feedback audit before writing.
            "arm_mode": build_arm_mode_metadata(
                command=command,
                session_state=str(step_result.state.value),
                active_sides=tuple(getattr(_components, "active_sides", ("left", "right"))),
            ),
            "command_metadata": command_metadata,
            "command_metadata_by_effector": command_metadata_by_effector,
            "channel_metadata": channel_metadata,
            "camera_keys": sorted(str(camera_key) for camera_key in camera_frames),
        }
        if hitl_lineage is not None:
            metadata[EXCLUSIVE_HITL_SAMPLE_METADATA_KEY] = (
                hitl_lineage.to_payload()
            )
        if include_action_debug_metadata:
            metadata["arm_action_debug_by_side"] = _arm_action_debug_snapshot(
                sample_timestamp=float(step_result.timestamp),
                arm_states=_arm_states,
                command=canonical_command,
            )
        return metadata

    def _build_arm_mode_metadata(
        self,
        *,
        command: TeleopCommand | None,
        session_state: str,
        active_sides: Sequence[str],
        arm_states: Mapping[str, Mapping[str, Any] | None],
        timestamp: float,
    ) -> dict[str, Any]:
        return build_arm_mode_metadata(
            command=command,
            session_state=session_state,
            active_sides=active_sides,
            init_audit_by_side=self._init_arm_audit_tracker.observe(
                command=command,
                arm_states=arm_states,
                timestamp=timestamp,
            ),
        )

    @staticmethod
    def _freeze_step_result(step_result: SessionStepResult) -> SessionStepResult:
        command = step_result.command
        frozen_command = None
        if command is not None:
            frozen_effectors: dict[str, EffectorCommand] = {}
            for name, effector in command.commands.items():
                frozen_effector = effector.copy()
                frozen_effector.metadata = _copy_value(dict(effector.metadata))
                frozen_effectors[str(name)] = frozen_effector
            frozen_command = TeleopCommand(
                commands=frozen_effectors,
                timestamp=command.timestamp,
                metadata=_copy_value(dict(command.metadata)),
            )
        return SessionStepResult(
            timestamp=float(step_result.timestamp),
            state=step_result.state,
            control_hz=float(step_result.control_hz),
            channels={
                str(name): channel.copy()
                for name, channel in step_result.channels.items()
            },
            raw_channels={},
            observation=None,
            command=frozen_command,
            command_dispatched=bool(step_result.command_dispatched),
            action_record=None,
            faults=list(step_result.faults),
            motion_hold_mode=str(step_result.motion_hold_mode),
            motion_hold_reason=str(step_result.motion_hold_reason),
            tracker_loss_duration_sec=float(step_result.tracker_loss_duration_sec),
            tracker_loss_channels=tuple(step_result.tracker_loss_channels),
            home_recalibration_required=bool(step_result.home_recalibration_required),
        )

    def snapshot_inputs(
        self,
        step_result: SessionStepResult,
        components: CollectorComponents,
    ) -> CollectorSampleSnapshot | None:
        """Freeze only causal live inputs; do no schema or row construction."""

        position_reference = self._position_reference_for_sample(
            float(step_result.timestamp)
        )
        if self.position_reference_enabled and position_reference is None:
            return None
        effective_step_result = step_result
        if position_reference is not None:
            effective_step_result = _step_with_command(
                step_result,
                _position_reference_command(
                    step_result.command,
                    position_reference,
                    active_sides=self.config.active_sides,
                ),
            )

        timing_started = time.perf_counter()
        arm_states = self._snapshot_arm_states(
            components,
            sample_timestamp=float(effective_step_result.timestamp),
        )
        arm_snapshot_s = time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        hand_states = self._snapshot_hand_states(
            components,
            sample_timestamp=float(effective_step_result.timestamp),
        )
        hand_snapshot_s = time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        camera_frames = self._snapshot_camera_frames(float(effective_step_result.timestamp))
        camera_snapshot_s = time.perf_counter() - timing_started
        self.last_camera_rejection = self._camera_rejection(
            sample_timestamp=float(effective_step_result.timestamp),
            camera_frames=camera_frames,
        )
        if self.last_camera_rejection is not None:
            return None
        return CollectorSampleSnapshot(
            step_result=self._freeze_step_result(effective_step_result),
            arm_states={
                str(side): _copy_value(state)
                for side, state in arm_states.items()
            },
            hand_states=(
                {
                    str(side): state
                    for side, state in hand_states.items()
                }
                if self.hand_state_provider_returns_isolated
                else {
                    str(side): _copy_value(state)
                    for side, state in hand_states.items()
                }
            ),
            camera_frames=dict(camera_frames),
            position_reference=(
                None
                if position_reference is None
                else dict(_to_serializable(position_reference))
            ),
            active_sides=tuple(
                getattr(components, "active_sides", self.config.active_sides)
            ),
            component_metadata=dict(_to_serializable(components.metadata)),
            snapshot_timings_s={
                "arm_snapshot_s": float(arm_snapshot_s),
                "hand_snapshot_s": float(hand_snapshot_s),
                "camera_snapshot_s": float(camera_snapshot_s),
            },
            include_debug_metadata=bool(self.include_debug_metadata),
            include_action_debug_metadata=bool(
                self.include_action_debug_metadata
            ),
        )

    def build_from_snapshot(
        self,
        snapshot: CollectorSampleSnapshot,
        *,
        components: CollectorComponents | None = None,
    ) -> CollectorSample:
        """Perform pure sample assembly from an already frozen snapshot."""

        if self.metadata_builder is not None and components is None:
            raise RuntimeError(
                "Detached sample assembly does not support a live custom metadata_builder."
            )
        effective_components: Any = components
        if effective_components is None:
            effective_components = _SnapshotComponentView(
                active_sides=tuple(snapshot.active_sides),
                metadata=dict(snapshot.component_metadata),
            )
        step_result = snapshot.step_result
        arm_states = snapshot.arm_states
        hand_states = snapshot.hand_states
        camera_frames = snapshot.camera_frames
        timing_started = time.perf_counter()
        if self.metadata_builder is None:
            metadata = self._default_metadata(
                step_result,
                effective_components,
                arm_states,
                hand_states,
                camera_frames,
                include_debug_metadata=bool(snapshot.include_debug_metadata),
                include_action_debug_metadata=bool(
                    snapshot.include_action_debug_metadata
                ),
            )
        else:
            metadata = self.metadata_builder(
                step_result,
                effective_components,
                arm_states,
                hand_states,
                camera_frames,
            )
        metadata = {} if metadata is None else dict(metadata)
        if snapshot.position_reference is not None:
            metadata["position_reference"] = _to_serializable(
                snapshot.position_reference
            )
        metadata["arm_mode"] = self._build_arm_mode_metadata(
            command=step_result.command,
            session_state=str(step_result.state.value),
            active_sides=tuple(snapshot.active_sides),
            arm_states=arm_states,
            timestamp=float(step_result.timestamp),
        )
        sample_builder_debug = {
            **dict(snapshot.snapshot_timings_s),
            "metadata_build_s": float(time.perf_counter() - timing_started),
            "camera_frames": _camera_debug_snapshot(
                sample_timestamp=float(step_result.timestamp),
                camera_frames=camera_frames,
            ),
            "hand_states": _hand_debug_snapshot(
                sample_timestamp=float(step_result.timestamp),
                hand_states=hand_states,
            ),
        }
        if snapshot.include_action_debug_metadata:
            sample_builder_debug["arm_actions"] = _arm_action_debug_snapshot(
                sample_timestamp=float(step_result.timestamp),
                arm_states=arm_states,
                command=_command_with_action_target_pose(
                    step_result.command,
                    arm_states=arm_states,
                    use_backend_target_pose=bool(step_result.command_dispatched),
                ),
            )
        metadata["sample_builder_debug"] = sample_builder_debug
        timing_started = time.perf_counter()
        sample = self.assembler.build_sample(
            step_result=step_result,
            arm_states=arm_states,
            hand_states=hand_states,
            camera_frames=camera_frames,
            index=self._sample_index,
            episode_index=self._episode_index,
            frame_index=self._frame_index,
            metadata=metadata,
        )
        sample.metadata.setdefault("sample_builder_debug", {})[
            "build_sample_s"
        ] = float(time.perf_counter() - timing_started)
        self._sample_index += 1
        self._frame_index += 1
        return sample

    def __call__(
        self,
        step_result: SessionStepResult,
        components: CollectorComponents,
    ) -> CollectorSample | None:
        snapshot = self.snapshot_inputs(step_result, components)
        if snapshot is None:
            return None
        return self.build_from_snapshot(snapshot, components=components)
