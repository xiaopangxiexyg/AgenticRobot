"""Bounded, redacted payload projection for deployment diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.deploy.schema import (
    ACTION_ARM_POSE_SLICES,
    ACTION_DIM,
    ACTION_HAND_SLICES,
    ALL_SIDES,
    pose_matrix_to_xyzquat,
)

from ..interfaces import EffectorBackend


_DEBUG_REDACT_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "secret",
}

DEPLOY_DEBUG_PAYLOAD_COMPACT = "compact"
DEPLOY_DEBUG_PAYLOAD_FULL = "full"
DEPLOY_DEBUG_PAYLOAD_MODES = (
    DEPLOY_DEBUG_PAYLOAD_COMPACT,
    DEPLOY_DEBUG_PAYLOAD_FULL,
)

_DEBUG_EFFECTOR_METADATA_KEYS = (
    "source",
    "action_source",
    "deploy_source",
    "policy_name",
    "action_space",
    "action_timestamp",
    "action_pose_source",
    "position_reference_sequence",
    "position_reference_scheduled_monotonic_time",
    "position_reference_query_monotonic_time",
    "position_reference_emitted_monotonic_time",
    "position_reference_emitted_timestamp",
    "position_reference_wall_time",
    "position_reference_hz",
    "position_command_hz",
    "position_transport_target_sequence",
    "position_transport_target_source",
    "position_transport_target_monotonic_time",
    "position_transport_target_wall_time",
    "position_transport_planner",
    "position_collision_status",
    "position_collision_distance_m",
    "position_collision_hold",
    "position_collision_certification_status",
    "position_collision_time_budget_exceeded",
    "position_collision_modification_reasons",
    "position_collision_pair_count",
    "position_collision_planner",
    "position_collision_planner_sequence",
    "position_collision_scene_profile",
    "position_collision_geometry_model",
    "position_endpoint_collision_checked",
    "position_endpoint_collision_passed",
    "position_collision_frame_total_ms",
    "position_collision_horizon_plan_ms",
    "position_collision_precompute_submit_ms",
    "position_collision_selection_ms",
    "position_collision_solve_ms",
    "position_collision_validation_ms",
    "position_collision_endpoint_fk_ms",
    "position_collision_shadow_metadata_ms",
    "position_trajectory_policy_mode",
    "position_trajectory_policy_reasons",
    "position_trajectory_policy_speed_scale",
    "position_trajectory_policy_quick_stop",
    "position_lookahead_available_at_emit",
    "position_lookahead_used",
    "position_causal",
    "position_causal_delay_periods",
    "position_causal_output_lag_sec",
    "position_interpolation_alpha",
    "position_interpolation_alpha_by_modality",
    "position_parent_source_indices",
    "position_parent_source_indices_by_modality",
    "position_parent_source_timestamps_by_modality",
    "backend_final_action_source",
    "backend_final_target_qpos",
    "solver_output_qpos",
    "rm65_background_target_admission_timing",
    "rm65_background_command_phase_timing",
    "target_pose_available",
    "target_pose_base",
    "target_pose_controlled_frame",
    "target_pose_tcp_base",
    "singularity_metadata",
)

_DEBUG_LAST_COMMAND_METADATA_KEYS = (
    "source",
    "action_source",
    "deploy_source",
    "position_reference_sequence",
    "position_reference_scheduled_monotonic_time",
    "position_reference_emitted_monotonic_time",
    "position_reference_hz",
    "position_command_hz",
    "position_transport_target_sequence",
    "position_transport_target_source",
    "position_transport_planner",
    "position_collision_status",
    "position_collision_distance_m",
    "position_collision_hold",
    "position_collision_certification_status",
    "position_collision_time_budget_exceeded",
    "position_collision_frame_total_ms",
    "position_collision_horizon_plan_ms",
    "position_collision_solve_ms",
    "position_collision_validation_ms",
    "position_trajectory_policy_mode",
    "position_trajectory_policy_reasons",
    "position_trajectory_policy_speed_scale",
    "position_trajectory_policy_quick_stop",
    "backend_final_action_source",
    "backend_final_target_qpos",
    "rm65_background_target_admission_timing",
    "rm65_background_command_phase_timing",
)


def _debug_redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _DEBUG_REDACT_KEYS:
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _debug_redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_debug_redact(item) for item in value]
    return value


def _debug_compact_value(
    value: Any,
    *,
    depth: int = 0,
    max_items: int = 32,
) -> Any:
    """Bound passive debug values before they reach JSON serialization.

    Deployment debug runs beside a deadline-sensitive Python reference worker.
    Keeping bounds here avoids repeatedly walking artifact-sized policy metadata
    while retaining the scalar timing and identity fields used for diagnosis.
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 64:
            return value.copy()
        return {
            "omitted": True,
            "reason": "compact_deploy_debug_array",
            "shape": tuple(int(item) for item in value.shape),
            "dtype": str(value.dtype),
            "size": int(value.size),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) <= 512:
            return value
        return value[:512] + f"<... {len(value) - 512} chars omitted>"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_deploy_debug_depth",
                "type": "mapping",
                "size": len(value),
            }
        items = list(value.items())
        payload = {
            str(key): _debug_compact_value(
                item,
                depth=depth + 1,
                max_items=max_items,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            payload["_debug_omitted_item_count"] = int(len(items) - max_items)
        return payload
    if isinstance(value, (list, tuple)):
        if depth >= 3:
            return {
                "omitted": True,
                "reason": "compact_deploy_debug_depth",
                "type": "sequence",
                "size": len(value),
            }
        payload = [
            _debug_compact_value(
                item,
                depth=depth + 1,
                max_items=max_items,
            )
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            payload.append(
                {"_debug_omitted_item_count": int(len(value) - max_items)}
            )
        return payload
    return str(value)


def _debug_remote_policy_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    server = value.get("server")
    server_identity: dict[str, Any] = {}
    if isinstance(server, Mapping):
        for key in (
            "protocol_version",
            "deploy_schema_version",
            "model_id",
            "normalization_hash",
            "policy_io_layout_hash",
            "pose_action_chunk_contract_hash",
            "pose_frame_contract_hashes",
        ):
            if key in server:
                server_identity[key] = _debug_compact_value(server.get(key))
    payload = {
        key: _debug_compact_value(value.get(key), max_items=32)
        for key in (
            "action_key",
            "action_discrete_key",
            "server_summary",
            "validation_warnings",
            "response_keys",
            "server_timing",
            "policy_timing",
        )
        if key in value
    }
    if server_identity:
        payload["server_identity"] = server_identity
    payload["_debug_omitted_server_detail"] = bool(isinstance(server, Mapping))
    return payload


def _debug_metadata_summary(
    metadata: Any,
    *,
    full_payload: bool,
    selected_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    if full_payload:
        return dict(metadata)

    keys = (
        tuple(str(key) for key in metadata)
        if selected_keys is None
        else selected_keys
    )
    payload: dict[str, Any] = {}
    for key in keys:
        if key not in metadata:
            continue
        if key == "remote_policy":
            payload[key] = _debug_remote_policy_summary(metadata.get(key))
        else:
            payload[key] = _debug_compact_value(metadata.get(key), max_items=32)

    omitted = [str(key) for key in metadata if str(key) not in payload]
    if omitted:
        payload["_debug_omitted_metadata"] = {
            "count": len(omitted),
            "keys": omitted[:32],
        }
    return payload


def _debug_background_streaming_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "enabled",
        "healthy",
        "health_reasons",
        "thread_alive",
        "target_active",
        "target_sequence",
        "target_age_sec",
        "target_stale",
        "fault",
        "fault_count",
        "actual_hz",
        "last_call_ms",
        "last_deadline_lag_ms",
        "last_send_dt_sec",
        "deadline_miss_count",
        "low_rate_count",
        "target_admission_mode",
        "last_target_admission_timing",
        "inflight_target_preemption_count",
        "last_inflight_target_preemption",
    )
    return {
        key: _debug_compact_value(value.get(key), max_items=16)
        for key in keys
        if key in value
    }


def _debug_backend_dispatch_by_effector_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    payload: dict[str, Any] = {}
    for effector_name, raw_item in value.items():
        if not isinstance(raw_item, Mapping):
            continue
        keys = (
            "accepted",
            "rewritten",
            "backend_ready",
            "operation",
            "status",
            "result_code",
            "singularity_gate_status",
            "dispatch_timing",
            "target_admission_timing",
            "command_phase_timing",
            "expected_qpos",
            "requested_qpos",
            "final_qpos",
            "backend_final_target_qpos",
        )
        item = {
            key: _debug_compact_value(raw_item.get(key), max_items=16)
            for key in keys
            if key in raw_item
        }
        if "background_streaming" in raw_item:
            item["background_streaming"] = _debug_background_streaming_summary(
                raw_item.get("background_streaming")
            )
        payload[str(effector_name)] = item
    return payload


def _debug_backend_group_health_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    payload = {
        key: _debug_compact_value(value.get(key), max_items=16)
        for key in ("ready", "require_background_target", "required_sides", "timing")
        if key in value
    }
    raw_sides = value.get("sides")
    if isinstance(raw_sides, Mapping):
        payload["sides"] = {}
        for side, raw_item in raw_sides.items():
            if not isinstance(raw_item, Mapping):
                continue
            payload["sides"][str(side)] = {
                key: _debug_compact_value(raw_item.get(key), max_items=16)
                for key in (
                    "ready",
                    "reasons",
                    "backend_ready",
                    "state_error",
                    "hardware_ready",
                    "emergency_stopped",
                    "send_result_code",
                    "warnings",
                    "timing",
                )
                if key in raw_item
            }
    return payload


def _debug_position_executor_event_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    selection = compact.get("ik_candidate_selection")
    if isinstance(selection, Mapping):
        # Coupled IK carries nested candidate/branch diagnostics. Bound that
        # passive tree before the synchronous JSON sink walks it at 60 Hz;
        # keep the admitted candidate certificate verbatim for qref diagnosis.
        summary = _debug_compact_value(selection, max_items=64)
        if "candidate_certificate" in selection:
            summary["candidate_certificate"] = selection["candidate_certificate"]
        compact["ik_candidate_selection"] = summary
    collision_debug = compact.get("position_collision_debug")
    if isinstance(collision_debug, Mapping):
        compact["position_collision_debug"] = {
            "omitted": True,
            "reason": "compact_deploy_debug_redundant_detail",
            "available_keys": tuple(str(key) for key in collision_debug),
            "q_ik_raw_by_side": _debug_compact_value(
                collision_debug.get("q_ik_raw_by_side"),
                max_items=16,
            ),
            "q_ref_by_side": _debug_compact_value(
                collision_debug.get("q_ref_by_side"),
                max_items=16,
            ),
        }
    if "position_arm_backend_dispatch_by_effector" in compact:
        compact["position_arm_backend_dispatch_by_effector"] = (
            _debug_backend_dispatch_by_effector_summary(
                compact.get("position_arm_backend_dispatch_by_effector")
            )
        )
    if "position_arm_backend_group_health" in compact:
        compact["position_arm_backend_group_health"] = (
            _debug_backend_group_health_summary(
                compact.get("position_arm_backend_group_health")
            )
        )
    return compact


def _debug_args_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(vars(args))
    policy_kwargs_text = payload.get("policy_kwargs_json")
    if isinstance(policy_kwargs_text, str):
        try:
            policy_kwargs = json.loads(policy_kwargs_text or "{}")
        except json.JSONDecodeError:
            policy_kwargs = policy_kwargs_text
        payload["policy_kwargs_json"] = _debug_redact(policy_kwargs)
    return _debug_redact(payload)


def _debug_vector(
    value: Any,
    *,
    length: int | None = None,
    require_finite: bool = True,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if length is not None and vector.shape != (int(length),):
        return None
    if require_finite and not np.all(np.isfinite(vector)):
        return None
    return vector.astype(np.float32, copy=True)


def _debug_xyzquat(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape == (4, 4):
        pose = pose_matrix_to_xyzquat(array)
        return None if pose is None else pose.astype(np.float32, copy=True)
    vector = _debug_vector(array, length=7)
    if vector is None:
        return None
    quat = np.asarray(vector[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return None
    vector = vector.astype(np.float32, copy=True)
    vector[3:7] = (quat / norm).astype(np.float32)
    return vector


def _debug_action_vector(action: Any) -> np.ndarray | None:
    vector = getattr(action, "vector", action)
    return _debug_vector(vector, length=ACTION_DIM, require_finite=False)


def _debug_pose_by_side_from_vector(
    vector: np.ndarray | None,
    active_sides: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if vector is None:
        return {}
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        target_slice = ACTION_ARM_POSE_SLICES[side]
        pose = _debug_xyzquat(vector[target_slice])
        if pose is not None:
            poses[side] = pose
    return poses


def _debug_hand_by_side_from_vector(
    vector: np.ndarray | None,
    active_sides: tuple[str, ...],
) -> dict[str, np.ndarray]:
    if vector is None:
        return {}
    hands: dict[str, np.ndarray] = {}
    for side in active_sides:
        target_slice = ACTION_HAND_SLICES[side]
        target = _debug_vector(vector[target_slice])
        if target is not None:
            hands[side] = target
    return hands


def _debug_arm_joint_targets(
    action: Any, active_sides: tuple[str, ...]
) -> dict[str, np.ndarray]:
    targets = getattr(action, "arm_joint_targets", {}) or {}
    if not isinstance(targets, Mapping):
        return {}
    resolved: dict[str, np.ndarray] = {}
    for side in active_sides:
        target = _debug_vector(targets.get(side), length=6)
        if target is not None:
            resolved[side] = target
    return resolved


def _debug_action_summary(
    action: Any,
    active_sides: tuple[str, ...],
    *,
    full_payload: bool = False,
) -> dict[str, Any] | None:
    vector = _debug_action_vector(action)
    if vector is None:
        return None
    metadata = getattr(action, "metadata", {})
    return {
        "source": (
            str(metadata.get("source", "")) if isinstance(metadata, Mapping) else ""
        ),
        "metadata": _debug_metadata_summary(
            metadata,
            full_payload=full_payload,
        ),
        "vector": vector,
        "arm_pose_by_side": _debug_pose_by_side_from_vector(vector, active_sides),
        "arm_joint_targets_by_side": _debug_arm_joint_targets(action, active_sides),
        "hand_by_side": _debug_hand_by_side_from_vector(vector, active_sides),
    }


def _debug_quat_delta_deg(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    ref_quat = np.asarray(reference[3:7], dtype=float)
    candidate_quat = np.asarray(candidate[3:7], dtype=float)
    ref_norm = float(np.linalg.norm(ref_quat))
    candidate_norm = float(np.linalg.norm(candidate_quat))
    if ref_norm <= 1e-12 or candidate_norm <= 1e-12:
        return None
    ref_quat = ref_quat / ref_norm
    candidate_quat = candidate_quat / candidate_norm
    dot = float(np.clip(abs(np.dot(ref_quat, candidate_quat)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _debug_pose_delta(reference: Any, candidate: Any) -> dict[str, Any] | None:
    ref_pose = _debug_xyzquat(reference)
    candidate_pose = _debug_xyzquat(candidate)
    if ref_pose is None or candidate_pose is None:
        return None
    xyz_delta = np.asarray(candidate_pose[:3] - ref_pose[:3], dtype=np.float32)
    return {
        "xyz_delta_m": xyz_delta,
        "xyz_norm_m": float(np.linalg.norm(xyz_delta)),
        "rot_delta_deg": _debug_quat_delta_deg(ref_pose, candidate_pose),
    }


def _debug_pose_delta_by_side(
    reference_by_side: Mapping[str, Any],
    candidate_by_side: Mapping[str, Any],
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for side in ALL_SIDES:
        if side not in reference_by_side or side not in candidate_by_side:
            continue
        delta = _debug_pose_delta(reference_by_side[side], candidate_by_side[side])
        if delta is not None:
            deltas[side] = delta
    return deltas


def _debug_command_arm_pose_by_side(
    command: Any, active_sides: tuple[str, ...]
) -> dict[str, np.ndarray]:
    if command is None:
        return {}
    commands = getattr(command, "commands", {}) or {}
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        effector_command = commands.get(f"{side}_arm")
        if (
            effector_command is None
            or getattr(effector_command, "command_type", "") != "cartesian_pose"
        ):
            continue
        metadata = getattr(effector_command, "metadata", {}) or {}
        pose = None
        if isinstance(metadata, Mapping):
            pose = _debug_xyzquat(metadata.get("target_xyzquat"))
            if pose is None:
                pose = _debug_xyzquat(metadata.get("target_pose_base"))
        if pose is None:
            pose = _debug_xyzquat(getattr(effector_command, "value", None))
        if pose is not None:
            poses[side] = pose
    return poses


def _debug_command_summary(
    command: Any,
    active_sides: tuple[str, ...],
    *,
    full_payload: bool = False,
) -> dict[str, Any] | None:
    if command is None:
        return None
    commands = getattr(command, "commands", {}) or {}
    command_payload: dict[str, Any] = {}
    for effector_name, effector_command in commands.items():
        metadata = getattr(effector_command, "metadata", {}) or {}
        item: dict[str, Any] = {
            "command_type": getattr(effector_command, "command_type", ""),
            "side": getattr(effector_command, "side", None),
            "frame_name": getattr(effector_command, "frame_name", None),
            "value": getattr(effector_command, "value", None),
            "metadata": _debug_metadata_summary(
                metadata,
                full_payload=full_payload,
                selected_keys=(
                    None if full_payload else _DEBUG_EFFECTOR_METADATA_KEYS
                ),
            ),
        }
        if item["command_type"] == "cartesian_pose":
            item["target_xyzquat"] = _debug_xyzquat(
                item["metadata"].get("target_xyzquat")
                if isinstance(item["metadata"], Mapping)
                else None
            )
            if item["target_xyzquat"] is None:
                item["target_xyzquat"] = _debug_xyzquat(
                    getattr(effector_command, "value", None)
                )
        command_payload[str(effector_name)] = item
    return {
        "timestamp": getattr(command, "timestamp", None),
        "metadata": _debug_metadata_summary(
            getattr(command, "metadata", {}) or {},
            full_payload=full_payload,
        ),
        "commands": command_payload,
        "arm_pose_by_side": _debug_command_arm_pose_by_side(command, active_sides),
    }


def _debug_arm_state_summary(
    states: Mapping[str, Mapping[str, Any]] | None,
    *,
    full_payload: bool = False,
) -> dict[str, Any]:
    if not states:
        return {}
    keys = (
        "current_qpos",
        "current_tcp_pose",
        "hardware_qpos",
        "last_target_qpos",
        "last_command",
        "last_command_timestamp",
        "hardware_pose_6d",
        "hardware_refresh_time",
        "hardware_read_phase_timing",
        "hardware_result_code",
        "hardware_result_text",
        "hardware_arm_err",
        "hardware_sys_err",
        "send_result_code",
        "send_result_text",
        "emergency_stopped",
        "effective_control_dt_sec",
        "effective_step_scale",
        "ik_metadata",
        "singularity_metadata",
        "tcp_pose_source",
        "state_error",
        "pose_error",
        "active_tool_name",
        "configured_tcp_enabled",
        "tool_tcp_source",
        "tool_tcp_endpoint_delta_m",
        "tool_tcp_endpoint_delta_deg",
    )
    payload: dict[str, Any] = {}
    for side, state in states.items():
        side_payload = {key: state.get(key) for key in keys if key in state}
        if not full_payload:
            last_command = side_payload.get("last_command")
            if isinstance(last_command, Mapping):
                last_command_payload = dict(last_command)
                last_command_payload["metadata"] = _debug_metadata_summary(
                    last_command.get("metadata", {}),
                    full_payload=False,
                    selected_keys=_DEBUG_LAST_COMMAND_METADATA_KEYS,
                )
                last_command_payload.pop("execution_work_item", None)
                side_payload["last_command"] = last_command_payload
            for key in ("ik_metadata", "singularity_metadata"):
                if key in side_payload:
                    side_payload[key] = _debug_compact_value(
                        side_payload[key],
                        max_items=32,
                    )
        pose = _debug_xyzquat(side_payload.get("current_tcp_pose"))
        if pose is not None:
            side_payload["current_tcp_xyzquat"] = pose
        payload[str(side)] = side_payload
    return payload


def _debug_state_pose_by_side(
    states: Mapping[str, Mapping[str, Any]] | None
) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    if not states:
        return poses
    for side, state in states.items():
        pose = _debug_xyzquat(state.get("current_tcp_pose"))
        if pose is not None:
            poses[str(side)] = pose
    return poses


def _debug_backend_arm_states(
    backends: Mapping[str, EffectorBackend]
) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for side, backend in backends.items():
        try:
            states[side] = dict(backend.get_state())
        except Exception as exc:  # noqa: BLE001
            states[side] = {"state_error": f"{type(exc).__name__}: {exc}"}
    return states


def _debug_backend_limits(
    backends: Mapping[str, EffectorBackend],
    *,
    now: float,
) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    for side, backend in backends.items():
        get_effective_step = getattr(backend, "get_effective_max_joint_step", None)
        try:
            effective_step = (
                get_effective_step(now=now) if callable(get_effective_step) else None
            )
        except Exception as exc:  # noqa: BLE001
            effective_step = None
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""
        config = backend.config
        limits[side] = {
            "control_dt_sec": float(config.control_dt),
            "max_angvel_rad_s": tuple(float(value) for value in config.max_angvel),
            "effective_max_joint_step_rad": effective_step,
            "background_streaming_enabled": bool(
                getattr(config, "background_streaming_enabled", False)
            ),
            "background_streaming_hz": (
                None
                if getattr(config, "background_streaming_hz", None) is None
                else float(config.background_streaming_hz)
            ),
            "background_streaming_target_admission_mode": getattr(
                config,
                "background_streaming_target_admission_mode",
                None,
            ),
            "canfd_follow": getattr(config, "canfd_follow", None),
            "observation_only": bool(getattr(config, "observation_only", False)),
            "send_commands": getattr(config, "send_commands", None),
            "error": error,
        }
    return limits


__all__ = [
    "DEPLOY_DEBUG_PAYLOAD_COMPACT",
    "DEPLOY_DEBUG_PAYLOAD_FULL",
    "DEPLOY_DEBUG_PAYLOAD_MODES",
    "_DEBUG_EFFECTOR_METADATA_KEYS",
    "_DEBUG_LAST_COMMAND_METADATA_KEYS",
    "_debug_action_summary",
    "_debug_action_vector",
    "_debug_arm_joint_targets",
    "_debug_arm_state_summary",
    "_debug_args_payload",
    "_debug_backend_arm_states",
    "_debug_backend_dispatch_by_effector_summary",
    "_debug_backend_group_health_summary",
    "_debug_backend_limits",
    "_debug_background_streaming_summary",
    "_debug_compact_value",
    "_debug_command_arm_pose_by_side",
    "_debug_command_summary",
    "_debug_hand_by_side_from_vector",
    "_debug_metadata_summary",
    "_debug_pose_by_side_from_vector",
    "_debug_pose_delta",
    "_debug_pose_delta_by_side",
    "_debug_position_executor_event_summary",
    "_debug_quat_delta_deg",
    "_debug_redact",
    "_debug_remote_policy_summary",
    "_debug_state_pose_by_side",
    "_debug_vector",
    "_debug_xyzquat",
]
