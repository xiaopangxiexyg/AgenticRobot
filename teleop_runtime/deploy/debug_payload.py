"""Bounded, redacted payload projection for deployment diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


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


__all__ = [
    "DEPLOY_DEBUG_PAYLOAD_COMPACT",
    "DEPLOY_DEBUG_PAYLOAD_FULL",
    "DEPLOY_DEBUG_PAYLOAD_MODES",
    "_DEBUG_EFFECTOR_METADATA_KEYS",
    "_DEBUG_LAST_COMMAND_METADATA_KEYS",
    "_debug_args_payload",
    "_debug_backend_dispatch_by_effector_summary",
    "_debug_backend_group_health_summary",
    "_debug_background_streaming_summary",
    "_debug_compact_value",
    "_debug_metadata_summary",
    "_debug_position_executor_event_summary",
    "_debug_redact",
    "_debug_remote_policy_summary",
]
