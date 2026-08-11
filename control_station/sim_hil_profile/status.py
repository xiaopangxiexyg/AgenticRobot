"""Pure GUI-safe status projection for the Sim-HIL application profile.

The functions in this module only rebuild presentation dictionaries from
already-owned immutable facts or passive diagnostic snapshots. They do not
start resources, advance a source, mutate authority, request HOLD, step a
simulator, or write capture data.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping


SIM_HIL_STATUS_SCHEMA_V1 = "dexteleop.control_station.sim_hil_status.v1"


def _emission_status(emission: Any, *, now: float) -> dict[str, Any] | None:
    if emission is None:
        return None
    emitted_at = getattr(emission, "emitted_at", None)
    result: dict[str, Any] = {
        "emission_id": getattr(emission, "emission_id", None),
        "semantic_content_hash": getattr(emission, "semantic_content_hash", None),
        "action_payload_kind": getattr(emission, "action_payload_kind", None),
        "representation_trace_id": getattr(emission, "representation_trace_id", None),
        "source_binding_hash": getattr(emission, "source_binding_hash", None),
        "canonical_policy_action_hash": getattr(
            getattr(emission, "canonical_action", None),
            "semantic_content_hash",
            None,
        ),
        "canonical_action_hash": getattr(
            getattr(
                getattr(emission, "canonical_action", None),
                "canonicalization_receipt",
                None,
            ),
            "canonical_action_hash",
            None,
        ),
        "canonicalization_receipt_hash": getattr(
            getattr(
                getattr(emission, "canonical_action", None),
                "canonicalization_receipt",
                None,
            ),
            "semantic_content_hash",
            None,
        ),
        "emitted_at": emitted_at,
        "age_sec": None if emitted_at is None else float(now) - float(emitted_at),
        "target_effective_at": getattr(emission, "target_effective_at", None),
        "effective_from": getattr(emission, "effective_from", None),
        "effective_until": getattr(emission, "effective_until", None),
        "chunk_id": getattr(emission, "chunk_id", None),
        "chunk_keyframe_index": getattr(emission, "chunk_keyframe_index", None),
        "inference_id": getattr(emission, "inference_id", None),
    }
    return result


def _input_submission_status(submission: Any) -> dict[str, Any] | None:
    if submission is None:
        return None
    receipt = getattr(submission, "input_receipt", None)
    input_cycle = getattr(submission, "input_cycle", None)
    batch_publication = (
        None if input_cycle is None else getattr(input_cycle, "publication", None)
    )
    batch_receipt = (
        None
        if batch_publication is None
        else getattr(batch_publication, "receipt", None)
    )
    return {
        "role": getattr(submission, "role", None),
        "roles": list(getattr(submission, "roles", ()) or ()),
        "source_emission_id": getattr(submission, "source_emission_id", None),
        "accepted": bool(getattr(submission, "accepted", False)),
        "error": getattr(submission, "error", None),
        "semantic_content_hash": getattr(submission, "semantic_content_hash", None),
        "input_cycle_status": getattr(input_cycle, "status", None),
        "batch_receipt": (
            None
            if batch_receipt is None
            else {
                "accepted": bool(getattr(batch_receipt, "accepted", False)),
                "rejection_code": getattr(batch_receipt, "rejection_code", None),
                "message": getattr(batch_receipt, "message", ""),
                "batch_id": getattr(batch_receipt, "batch_id", None),
                "available_at": getattr(batch_receipt, "available_at", None),
                "available_clock_domain_id": getattr(
                    batch_receipt,
                    "available_clock_domain_id",
                    None,
                ),
                "last_control_admission_sequence": getattr(
                    batch_receipt,
                    "last_control_admission_sequence",
                    None,
                ),
                "semantic_content_hash": getattr(
                    batch_receipt,
                    "semantic_content_hash",
                    None,
                ),
            }
        ),
        "receipt": (
            None
            if receipt is None
            else {
                "accepted": bool(getattr(receipt, "accepted", False)),
                "rejection_code": getattr(receipt, "rejection_code", None),
                "message": getattr(receipt, "message", ""),
                "composition_input_id": getattr(receipt, "composition_input_id", None),
                "available_at": getattr(receipt, "available_at", None),
                "available_clock_domain_id": getattr(
                    receipt, "available_clock_domain_id", None
                ),
                "control_admission_sequence": getattr(
                    receipt, "control_admission_sequence", None
                ),
                "semantic_content_hash": getattr(
                    receipt, "semantic_content_hash", None
                ),
            }
        ),
    }


def _composition_step_status(step: Any) -> dict[str, Any] | None:
    if step is None:
        return None
    evaluation = getattr(step, "evaluation", None)
    if evaluation is None:
        # Allocation-aware cycles expose the exact formal submission through
        # their proposal-publication envelope rather than the compatibility
        # step's ``evaluation`` field.
        evaluation = getattr(step, "submission", None)
    composition_evaluation = (
        None
        if evaluation is None
        else getattr(evaluation, "composition_evaluation", None)
    )
    disposition = getattr(step, "disposition", None)
    proposal = getattr(step, "proposal_admission", None)
    if proposal is None and evaluation is not None:
        proposal = getattr(evaluation, "proposal_admission", None)
    hold = getattr(step, "hold_transition", None)
    hold_event = None if hold is None else getattr(hold, "event", None)
    difference = (
        None
        if evaluation is None
        else getattr(evaluation, "source_difference_observation", None)
    )

    def difference_payload(item: Any) -> dict[str, Any]:
        return {
            "compared_role": item.compared_role,
            "reference_role": item.reference_role,
            "available_at_delta_sec": item.available_at_delta_sec,
            "effective_from_delta_sec": item.effective_from_delta_sec,
            "valid_for_sec": item.valid_for_sec,
            "components": [
                {
                    "component_id": metric.component_id,
                    "representation": metric.representation,
                    "translation_delta_xyz_m": list(metric.translation_delta_xyz_m),
                    "translation_norm_m": metric.translation_norm_m,
                    "rotation_delta_rotvec_rad": list(metric.rotation_delta_rotvec_rad),
                    "rotation_angle_rad": metric.rotation_angle_rad,
                    "qpos_active_indices": list(metric.qpos_active_indices),
                    "qpos_delta": list(metric.qpos_delta),
                    "qpos_l2_norm": metric.qpos_l2_norm,
                    "qpos_max_abs": metric.qpos_max_abs,
                }
                for metric in item.component_metrics
            ],
        }

    return {
        "publication_status": getattr(step, "status", None),
        "decision_key": getattr(step, "decision_key", None),
        "target_effective_at": getattr(step, "target_effective_at", None),
        "semantic_content_hash": getattr(step, "semantic_content_hash", None),
        "readiness_hash": (
            None
            if getattr(step, "readiness", None) is None
            else getattr(step.readiness, "semantic_content_hash", None)
        ),
        "composition_evaluation": (
            None
            if composition_evaluation is None
            else {
                "evaluation_id": getattr(
                    composition_evaluation, "composition_evaluation_id", None
                ),
                "semantic_content_hash": getattr(
                    composition_evaluation, "semantic_content_hash", None
                ),
                "composition_supported_until": getattr(
                    composition_evaluation, "composition_supported_until", None
                ),
            }
        ),
        "immediate_difference": (
            None
            if difference is None
            else {
                "observation_id": difference.observation_id,
                "semantic_content_hash": difference.semantic_content_hash,
                "valid_until": difference.valid_until,
                "pairwise": [
                    difference_payload(item) for item in difference.pairwise_differences
                ],
                "output_residuals": [
                    difference_payload(item) for item in difference.output_residuals
                ],
            }
        ),
        "disposition": (
            None
            if disposition is None
            else {
                "disposition_id": getattr(disposition, "disposition_id", None),
                "downstream_request": getattr(disposition, "downstream_request", None),
                "reason": getattr(disposition, "reason", None),
                "semantic_content_hash": getattr(
                    disposition, "semantic_content_hash", None
                ),
            }
        ),
        "allocation": (
            {
                "selected_context_id": getattr(
                    step,
                    "selected_context_id",
                    None,
                ),
                "selected_context_hash": getattr(
                    step,
                    "selected_context_hash",
                    None,
                ),
                "selected_mode_by_component": dict(
                    getattr(step, "selected_mode_by_component", {}) or {}
                ),
                "promotion_blocked_mode_by_component": dict(
                    getattr(
                        step,
                        "promotion_blocked_mode_by_component",
                        {},
                    )
                    or {}
                ),
            }
            if getattr(step, "status", None) is not None
            else None
        ),
        "proposal": (
            None
            if proposal is None
            else {
                "accepted": bool(getattr(proposal, "accepted", False)),
                "proposal_id": getattr(proposal, "proposal_id", None),
                "rejection_code": getattr(proposal, "rejection_code", None),
                "message": getattr(proposal, "message", ""),
                "semantic_content_hash": getattr(
                    proposal, "semantic_content_hash", None
                ),
                "effective_from": getattr(
                    getattr(proposal, "proposal", None), "effective_from", None
                ),
                "effective_until": getattr(
                    getattr(proposal, "proposal", None), "effective_until", None
                ),
                "target_effective_at": getattr(
                    getattr(proposal, "proposal", None), "target_effective_at", None
                ),
            }
        ),
        "hold": (
            None
            if hold is None
            else {
                "event_id": (
                    None
                    if hold_event is None
                    else getattr(hold_event, "event_id", None)
                ),
                "status": getattr(getattr(hold, "snapshot", None), "status", None),
                "semantic_content_hash": getattr(
                    getattr(hold, "snapshot", None), "semantic_content_hash", None
                ),
            }
        ),
        "blocked_reason": getattr(step, "blocked_reason", None),
        "error": getattr(step, "error", None),
    }


def _human_channel_status(step: Any, *, now: float) -> dict[str, Any]:
    session_step = None if step is None else getattr(step, "session_step", None)
    channels = {} if session_step is None else getattr(session_step, "channels", {})
    result: dict[str, Any] = {}
    for name, channel in dict(channels or {}).items():
        timestamp = getattr(channel, "timestamp", None)
        status = {
            "valid": bool(getattr(channel, "valid", False)),
            "capability": getattr(channel, "capability", None),
            "timestamp": timestamp,
            "age_sec": None if timestamp is None else float(now) - float(timestamp),
        }
        metadata = dict(getattr(channel, "metadata", {}) or {})
        read_error = metadata.get("read_error")
        if read_error:
            status["read_error"] = str(read_error)
        if metadata.get("reader_error_count") is not None:
            status["reader_error_count"] = int(
                metadata["reader_error_count"]
            )
        value = getattr(channel, "value", None)
        try:
            serialized = (
                value.tolist() if callable(getattr(value, "tolist", None)) else value
            )
            numeric = (
                [float(serialized)]
                if isinstance(serialized, (int, float))
                else [float(item) for item in serialized]
            )
            if len(numeric) <= 16 and all(math.isfinite(item) for item in numeric):
                status["value"] = numeric[0] if len(numeric) == 1 else numeric
        except (TypeError, ValueError):
            pass
        result[str(name)] = status
    return result


def _human_gripper_target_status(emission: Any) -> dict[str, Any] | None:
    if emission is None:
        return None
    for command in tuple(getattr(emission, "merged_commands", ()) or ()):
        if (
            getattr(command, "effector_name", None) != "right_hand"
            or getattr(command, "command_type", None) != "joint_vector"
        ):
            continue
        metadata = dict(getattr(command, "metadata", {}) or {})
        target = metadata.get("gripper_joint_target")
        if target is None:
            values = getattr(command, "value", ())
            try:
                target = float(values[0])
            except (IndexError, TypeError, ValueError):
                return None
        return {
            "effector_name": "right_hand",
            "component_id": "right_gripper",
            "normalized_close": float(target),
            "input_channel": metadata.get("input_channel"),
            "source_value": metadata.get("source_value"),
        }
    return None


def _mujoco_gripper_execution_status(
    rig: Any,
    *,
    gripper_name: str,
) -> dict[str, Any] | None:
    snapshot = getattr(rig, "snapshot", None)
    if not callable(snapshot):
        return None
    try:
        execution = dict(snapshot().get("gripper_execution") or {})
    except Exception:  # noqa: BLE001 - status must not affect control.
        return None
    if not execution:
        return None

    def selected(field: str) -> Any:
        return dict(execution.get(field) or {}).get(gripper_name)

    return {
        "gripper_name": gripper_name,
        "accepted_close": selected("accepted_close_by_gripper"),
        "servo_goal_close": selected("servo_goal_close_by_gripper"),
        "actuator_target_close": selected("actuator_target_close_by_gripper"),
        "measured_close": selected("measured_close_by_gripper"),
        "status": selected("last_status_by_gripper"),
        "pending_delayed_targets": execution.get("pending_delayed_targets"),
    }


def _vive_receiver_status(
    session: Any,
    *,
    wall_now: float | None = None,
) -> dict[str, Any]:
    """Project the passive VIVE receiver into GUI-safe diagnostic facts.

    Upstream role channels are intentionally read before the active
    source-role mapping.  They are display-only and cannot enter the teleop or
    composition paths.
    """

    from teleop_runtime.real_teleop.gui_tracker_status import (
        TRACKER_GUI_STALE_AGE_SEC,
        raw_tracker_pose_state_from_channels,
    )
    from teleop_runtime.sources.pose.vive import VivePoseSource

    sources = [
        source
        for source in tuple(getattr(session, "sources", ()) or ())
        if isinstance(source, VivePoseSource)
    ]
    if len(sources) != 1:
        return {
            "status": "unavailable",
            "message": (
                "Expected exactly one VivePoseSource; " f"found {len(sources)}."
            ),
            "trackers": raw_tracker_pose_state_from_channels(
                {},
                active_sides=("left", "right"),
                now=time.time() if wall_now is None else float(wall_now),
            ),
        }

    source = sources[0]
    now = time.time() if wall_now is None else float(wall_now)
    stats = dict(source.debug_stats())
    upstream = source.get_latest_upstream_role_channels()
    tracker_status = raw_tracker_pose_state_from_channels(
        {
            "left_wrist_pose": upstream.get("left_wrist"),
            "right_wrist_pose": upstream.get("right_wrist"),
        },
        active_sides=("left", "right"),
        now=now,
    )
    role_by_output = source.source_role_by_output_channel
    for side, tracker_role in (
        ("left", "left_wrist"),
        ("right", "right_wrist"),
    ):
        tracker_status[side]["selected_for_output_channels"] = sorted(
            channel_name
            for channel_name, selected_role in role_by_output.items()
            if selected_role == tracker_role
        )

    # Transport freshness must be measured from socket arrival, not from the
    # later parse/publication completion.  The latter can be delayed by a
    # malformed or unusually large batch and is kept as a separate diagnostic.
    last_packet_at = stats.get("last_datagram_received_at")
    if last_packet_at is None:
        last_packet_at = stats.get("last_packet_received_at")
    packet_age_wall = (
        None if last_packet_at is None else max(0.0, now - float(last_packet_at))
    )
    last_packet_monotonic = stats.get("last_datagram_received_monotonic_time")
    if last_packet_monotonic is None:
        last_packet_monotonic = stats.get("last_packet_received_monotonic_time")
    packet_age_monotonic = (
        None
        if last_packet_monotonic is None
        else max(0.0, time.perf_counter() - float(last_packet_monotonic))
    )
    # Tests/diagnostic callers may pass an explicit wall-time cut.  The live
    # GUI path does not, so it uses the monotonic age and cannot be confused by
    # NTP/manual wall-clock changes.
    packet_age = (
        packet_age_wall
        if wall_now is not None or packet_age_monotonic is None
        else packet_age_monotonic
    )
    running = bool(stats.get("running", False))
    worker_alive = bool(stats.get("worker_alive", running))
    worker_started_once = bool(stats.get("worker_started_once", False))
    if not running:
        receiver_status = "stopped"
    elif worker_started_once and not worker_alive:
        receiver_status = "worker_dead"
    elif packet_age is None:
        receiver_status = "waiting_for_packet"
    elif packet_age > TRACKER_GUI_STALE_AGE_SEC:
        receiver_status = "packet_stale"
    else:
        receiver_status = "receiving"

    return {
        "status": receiver_status,
        "bind_host": stats.get("bind_host"),
        "port": stats.get("port"),
        "packet_count": int(
            stats.get("datagram_count")
            if int(stats.get("datagram_count") or 0) > 0
            else (stats.get("packet_count") or 0)
        ),
        "processed_packet_count": int(stats.get("packet_count") or 0),
        "worker_alive": worker_alive,
        "worker_started_once": worker_started_once,
        "last_packet_received_at": last_packet_at,
        "packet_age_sec": (None if packet_age is None else round(packet_age, 3)),
        "packet_age_wall_sec": (
            None if packet_age_wall is None else round(packet_age_wall, 3)
        ),
        "packet_age_monotonic_sec": (
            None if packet_age_monotonic is None else round(packet_age_monotonic, 3)
        ),
        "recv_error_count": int(stats.get("recv_error_count") or 0),
        "last_recv_error": stats.get("last_recv_error"),
        "process_error_count": int(stats.get("process_error_count") or 0),
        "last_process_error": stats.get("last_process_error"),
        "last_process_duration_sec": stats.get("last_process_duration_sec"),
        "last_process_finished_monotonic_time": stats.get(
            "last_process_finished_monotonic_time"
        ),
        "last_process_batch_size": int(stats.get("last_process_batch_size") or 0),
        "last_process_parse_error_count": int(
            stats.get("last_process_parse_error_count") or 0
        ),
        "dropped_stale_packet_count": int(stats.get("dropped_stale_packet_count") or 0),
        "source_role_by_output_channel": dict(role_by_output),
        "source_mapping_generation": int(stats.get("source_mapping_generation") or 0),
        "channel_update_counts": dict(stats.get("channel_update_counts") or {}),
        "last_channel_update_at": dict(stats.get("last_channel_update_at") or {}),
        "upstream_role_update_counts": dict(
            stats.get("upstream_role_update_counts") or {}
        ),
        "last_upstream_role_update_at": dict(
            stats.get("last_upstream_role_update_at") or {}
        ),
        "trackers": tracker_status,
    }


def _runtime_contract_status(runtime: Any, composition: Any) -> dict[str, Any]:
    contracts = getattr(composition, "contracts", None)
    composition_spec = getattr(contracts, "composition_spec", None)
    strategy_spec = getattr(contracts, "strategy_spec", None)
    authority_profile = getattr(contracts, "authority_profile", None)
    runtime_config = getattr(runtime, "config", None)
    target_contract = getattr(runtime, "target_runtime_action_contract", None)
    target_capability = getattr(runtime, "component_target_capability", None)
    target_port = getattr(runtime, "target_execution_port", None)
    return {
        "runtime_session_id": getattr(runtime_config, "runtime_session_id", None),
        "control_domain_id": getattr(runtime_config, "control_domain_id", None),
        "clock_domain_id": getattr(runtime_config, "clock_domain_id", None),
        "action_schema_hash": getattr(
            getattr(runtime, "action_schema", None), "semantic_content_hash", None
        ),
        "target": {
            "handoff_enabled": bool(
                getattr(runtime_config, "enable_component_execution_handoff", False)
            ),
            "target_handler_id": getattr(target_port, "target_handler_id", None),
            "activation_mode": getattr(target_port, "activation_mode", None),
            "target_runtime_action_contract_hash": getattr(
                target_contract, "semantic_content_hash", None
            ),
            "component_target_capability_id": getattr(
                target_capability, "capability_id", None
            ),
            "component_target_capability_hash": getattr(
                target_capability, "semantic_content_hash", None
            ),
            "component_target_action_schema_hash": getattr(
                target_capability, "action_schema_hash", None
            ),
            "execution_capability_profile_hash": getattr(
                runtime, "execution_capability_profile_hash", None
            ),
        },
        "composition": {
            "spec_id": getattr(composition_spec, "spec_id", None),
            "spec_hash": getattr(composition_spec, "semantic_content_hash", None),
            "required_roles": list(getattr(composition_spec, "required_roles", ())),
        },
        "strategy": {
            "spec_id": getattr(strategy_spec, "strategy_id", None),
            "spec_hash": getattr(strategy_spec, "semantic_content_hash", None),
        },
        "authority": {
            "profile_id": getattr(authority_profile, "profile_id", None),
            "profile_hash": getattr(authority_profile, "semantic_content_hash", None),
            "single_execution_owner": (
                "MujocoDeployRuntime/composer"
                if getattr(runtime_config, "enable_composer_authority", False)
                and not getattr(runtime_config, "enable_primary_policy_authority", True)
                and not getattr(runtime_config, "enable_replay_authority", True)
                else "invalid"
            ),
        },
    }


def _target_execution_status(snapshot: Any) -> dict[str, Any] | None:
    """Render a lower-owned immutable snapshot without deriving control state."""

    if snapshot is None:
        return None
    stages: dict[str, dict[str, Any]] = {}
    for name in ("request", "submission", "completion", "admission", "activation"):
        stage = getattr(snapshot, name)
        stages[name] = {
            "status": stage.status,
            "record_id": stage.record_id,
            "record_hash": stage.record_hash,
            "available_at": stage.available_at,
        }
    return {
        "snapshot_id": snapshot.snapshot_id,
        "semantic_content_hash": snapshot.semantic_content_hash,
        "captured_at": snapshot.captured_at,
        "control_domain_id": snapshot.control_domain_id,
        "target_handler_id": snapshot.target_handler_id,
        "clock_domain_id": snapshot.clock_domain_id,
        "activation_mode": snapshot.activation_mode,
        "request_id": snapshot.request_id,
        "target_id": snapshot.target_id,
        "target_hash": snapshot.target_hash,
        "stages": stages,
    }


def _execution_difference_status(observer: Any | None) -> dict[str, Any]:
    if observer is None:
        return {"enabled": False}
    snapshot = observer.snapshot()
    association = snapshot.latest_association
    return {
        "enabled": True,
        "reference_count": snapshot.reference_count,
        "measurement_count": snapshot.measurement_count,
        "unmatched_measurement_count": snapshot.unmatched_measurement_count,
        "pending_reference_count": snapshot.pending_reference_count,
        "watermarks": {
            stream: {
                "status": watermark.status,
                "reason": watermark.reason,
                "processed_through": watermark.processed_through,
                "emitted_count": watermark.emitted_count,
                "dropped_count": watermark.dropped_count,
                "semantic_content_hash": watermark.semantic_content_hash,
            }
            for stream, watermark in snapshot.watermark_by_stream.items()
        },
        "latest": (
            None
            if association is None
            else {
                "association_id": association.association_id,
                "semantic_content_hash": association.semantic_content_hash,
                "status": association.status,
                "reason": association.reason,
                "measurement_id": association.measurement_id,
                "reference_frame_id": association.reference_frame_id,
                "measured_at": association.measured_at,
                "execution_fence_sequence": (association.execution_fence_sequence),
                "reference_revision": association.reference_revision,
                "component_difference": {
                    component_id: dict(value)
                    for component_id, value in (
                        association.component_difference.items()
                    )
                },
            }
        ),
    }


def _mix_status(
    alpha_by_component: Mapping[str, float],
) -> dict[str, Any]:
    values = {
        str(component_id): float(value)
        for component_id, value in dict(alpha_by_component).items()
    }
    if values and all(value <= 1e-12 for value in values.values()):
        mode = "policy_only"
    elif values and all(value >= 1.0 - 1e-12 for value in values.values()):
        mode = "human_only"
    else:
        mode = "proportional_blend"
    return {
        "mode": mode,
        "human_alpha_by_component": values,
    }


__all__ = [
    "SIM_HIL_STATUS_SCHEMA_V1",
]
