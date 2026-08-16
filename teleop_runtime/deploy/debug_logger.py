"""Deployment debug JSONL path resolution and writer lifecycle."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from project_paths import repo_path
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.schema import to_serializable
from runtime_artifact_paths import (
    ArtifactWriteResolution,
    RuntimeArtifactLayout,
    resolve_artifact_write_path,
)

from ..interfaces import EffectorBackend
from .debug_payload import (
    DEPLOY_DEBUG_PAYLOAD_COMPACT,
    DEPLOY_DEBUG_PAYLOAD_FULL,
    DEPLOY_DEBUG_PAYLOAD_MODES,
    _debug_action_summary,
    _debug_args_payload,
    _debug_arm_state_summary,
    _debug_backend_limits,
    _debug_command_summary,
    _debug_compact_value,
    _debug_metadata_summary,
    _debug_pose_delta_by_side,
    _debug_position_executor_event_summary,
    _debug_redact,
    _debug_state_pose_by_side,
)


class _PreparedActionExecutionView(Protocol):
    dispatch_attempted: bool
    dispatch_completed: bool
    asynchronous: bool
    mode: str
    position_source_index: int | None


def canonical_deployment_debug_root(*, now: float | None = None) -> Path:
    timestamp_time = time.time() if now is None else float(now)
    local_time = time.localtime(timestamp_time)
    date_partition = time.strftime("%Y-%m-%d", local_time)
    stamp = time.strftime("%Y%m%d_%H%M%S", local_time)
    millis = int((timestamp_time % 1.0) * 1000.0)
    return RuntimeArtifactLayout.configured().run_dir(
        kind="policy_deploy",
        date_partition=date_partition,
        run_id=f"deploy_{stamp}_{millis:03d}",
    )


def resolve_deployment_debug_root(
    args: argparse.Namespace,
    *,
    now: float | None = None,
) -> ArtifactWriteResolution:
    return resolve_artifact_write_path(
        explicit_path=getattr(args, "debug_dir", None),
        artifact_layout=getattr(args, "artifact_layout", None),
        legacy_path=repo_path("teleop_records", "deployment_debug"),
        canonical_path=canonical_deployment_debug_root(now=now),
    )


class DeploymentDebugLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        root_dir: Path | None,
        active_sides: tuple[str, ...],
        payload_mode: str = DEPLOY_DEBUG_PAYLOAD_COMPACT,
        flush_every_events: int = 1,
        flush_interval_sec: float = 0.0,
    ):
        self.enabled = bool(enabled)
        self.active_sides = tuple(active_sides)
        self.payload_mode = str(payload_mode).strip().lower()
        if self.payload_mode not in DEPLOY_DEBUG_PAYLOAD_MODES:
            raise ValueError(
                "payload_mode must be one of "
                f"{DEPLOY_DEBUG_PAYLOAD_MODES}, got {payload_mode!r}."
            )
        self.full_payload = self.payload_mode == DEPLOY_DEBUG_PAYLOAD_FULL
        self.path: Path | None = None
        self._handle: Any = None
        self._lock = threading.RLock()
        self._write_failed = False
        self._flush_every_events = int(flush_every_events)
        self._flush_interval_sec = float(flush_interval_sec)
        if self._flush_every_events <= 0:
            raise ValueError("flush_every_events must be positive.")
        if (
            not math.isfinite(self._flush_interval_sec)
            or self._flush_interval_sec < 0.0
        ):
            raise ValueError("flush_interval_sec must be finite and non-negative.")
        self._pending_flush_events = 0
        self._last_flush_at = time.monotonic()
        if not self.enabled:
            return
        debug_root = (
            root_dir
            if root_dir is not None
            else repo_path("teleop_records", "deployment_debug")
        )
        debug_root = Path(debug_root).expanduser()
        if not debug_root.is_absolute():
            debug_root = repo_path(debug_root)
        debug_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        millis = int((time.time() % 1.0) * 1000.0)
        self.path = debug_root / f"deploy_debug_{stamp}_{millis:03d}.jsonl"
        self._handle = self.path.open("a", encoding="utf-8")
        print(
            f"[DeployDebug] writing action/pose debug JSONL to {self.path}", flush=True
        )

    def close(self) -> None:
        with self._lock:
            if self._handle is None:
                return
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            if not self.enabled or self._handle is None:
                return
            event_name = str(event)
            event_payload = dict(payload)
            if (
                not self.full_payload
                and event_name.startswith("position_executor_")
            ):
                event_payload = _debug_position_executor_event_summary(event_payload)
            record = {
                "event": event_name,
                "wall_time": time.time(),
                "debug_payload_mode": self.payload_mode,
                **event_payload,
            }
            try:
                try:
                    encoded = json.dumps(
                        to_serializable(record),
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                except Exception as serialization_exc:  # noqa: BLE001
                    # One newly added passive field must not disable every
                    # subsequent record.  Preserve a compact failure event and
                    # continue; actual file I/O failures below remain terminal
                    # for this observational sink.
                    encoded = json.dumps(
                        {
                            "event": "debug_serialization_error",
                            "wall_time": time.time(),
                            "source_event": str(event),
                            "error_type": type(serialization_exc).__name__,
                            "error": str(serialization_exc),
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    )
                self._handle.write(encoded + "\n")
                self._pending_flush_events += 1
                now = time.monotonic()
                if self._pending_flush_events >= self._flush_every_events or (
                    self._flush_interval_sec > 0.0
                    and now - self._last_flush_at >= self._flush_interval_sec
                ):
                    self._handle.flush()
                    self._pending_flush_events = 0
                    self._last_flush_at = now
            except Exception as exc:  # noqa: BLE001
                if not self._write_failed:
                    print(
                        f"[DeployDebug] disabled after write failure: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                self._write_failed = True
                self.enabled = False

    def log_startup(
        self,
        *,
        args: argparse.Namespace,
        policy: Any,
        active_sides: tuple[str, ...],
        hands_enabled: bool,
        effective_action_hz: float,
        action_hz_info: Mapping[str, Any],
        effective_policy_hz: float,
        policy_hz_source: str,
        backend_control_hz: float,
        action_queue_policy: str,
        send_actions: bool,
        arm_backends: Mapping[str, EffectorBackend],
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "startup",
            {
                "args": _debug_args_payload(args),
                "policy_name": getattr(policy, "name", ""),
                "debug_payload_mode": self.payload_mode,
                "policy_metadata": _debug_redact(getattr(policy, "metadata", {})),
                "active_sides": tuple(active_sides),
                "arms_enabled": bool(args.arms_enabled),
                "hands_enabled": bool(hands_enabled),
                "send_actions": bool(send_actions),
                "effective_action_hz": float(effective_action_hz),
                "effective_policy_hz": float(effective_policy_hz),
                "action_hz_info": dict(action_hz_info),
                "policy_hz_source": str(policy_hz_source),
                "backend_control_hz": float(backend_control_hz),
                "action_queue_policy": str(action_queue_policy),
                "arm_execution": (
                    args._real_arm_execution_config.to_metadata()
                    if getattr(args, "_real_arm_execution_config", None) is not None
                    else None
                ),
                "safety_limits": {
                    "arm_workspace_min": tuple(
                        float(value) for value in args.arm_workspace_min
                    ),
                    "arm_workspace_max": tuple(
                        float(value) for value in args.arm_workspace_max
                    ),
                    "max_arm_xyz_step_m": float(args.max_arm_xyz_step_m),
                    "max_arm_rot_step_deg": float(args.max_arm_rot_step_deg),
                    "max_arm_joint_step_rad": float(args.max_arm_joint_step_rad),
                    "max_hand_step": float(args.max_hand_step),
                },
                "backend_limits_by_side": _debug_backend_limits(
                    arm_backends, now=time.time()
                ),
            },
        )

    def log_policy_chunk(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        policy_output: Any,
        actions: list[NormalizedAction],
        accepted: bool,
        discarded_by_ik: bool,
        queue_policy: str,
        queue_len_before: int,
        queue_len_after: int,
        faults: list[str],
        ik_recovery_statuses: Mapping[str, str] | None = None,
        ik_recovery_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "policy_chunk",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "policy_output_type": type(policy_output).__name__,
                "chunk_size": len(actions),
                "accepted": bool(accepted),
                "discarded_by_ik": bool(discarded_by_ik),
                "queue_policy": str(queue_policy),
                "queue_len_before": int(queue_len_before),
                "queue_len_after": int(queue_len_after),
                "faults": list(faults),
                "ik_recovery_statuses": (
                    {} if ik_recovery_statuses is None else dict(ik_recovery_statuses)
                ),
                "ik_recovery_diagnostics": (
                    {}
                    if ik_recovery_diagnostics is None
                    else {
                        str(side): dict(value)
                        for side, value in ik_recovery_diagnostics.items()
                    }
                ),
                "actions": [
                    _debug_action_summary(
                        action,
                        self.active_sides,
                        full_payload=self.full_payload,
                    )
                    for action in actions
                ],
            },
        )

    def log_policy_error(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        exc: BaseException,
        queue_len_before: int,
    ) -> None:
        if not self.enabled:
            return
        self.log_event(
            "policy_error",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "queue_len_before": int(queue_len_before),
            },
        )

    def log_action_step(
        self,
        *,
        timestamp: float,
        step_index: int,
        policy_name: str,
        raw_action: NormalizedAction,
        canonical_action: Any,
        pre_arm_states: Mapping[str, Mapping[str, Any]],
        post_arm_states: Mapping[str, Mapping[str, Any]] | None,
        arm_backends: Mapping[str, EffectorBackend],
        send_actions: bool,
        queue_len_before_pop: int,
        queue_len_after_pop: int,
        faults: list[str],
        dispatch_error: BaseException | None,
        execution: _PreparedActionExecutionView | None = None,
        phase_timing: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        raw_summary = _debug_action_summary(
            raw_action,
            self.active_sides,
            full_payload=self.full_payload,
        )
        filtered_summary = _debug_action_summary(
            getattr(canonical_action, "action", None),
            self.active_sides,
            full_payload=self.full_payload,
        )
        command_summary = _debug_command_summary(
            getattr(canonical_action, "command", None),
            self.active_sides,
            full_payload=self.full_payload,
        )
        raw_pose = (
            {} if raw_summary is None else raw_summary.get("arm_pose_by_side", {})
        )
        filtered_pose = (
            {}
            if filtered_summary is None
            else filtered_summary.get("arm_pose_by_side", {})
        )
        command_pose = (
            {}
            if command_summary is None
            else command_summary.get("arm_pose_by_side", {})
        )
        pre_tcp_pose = _debug_state_pose_by_side(pre_arm_states)
        post_tcp_pose = _debug_state_pose_by_side(post_arm_states)
        command = getattr(canonical_action, "command", None)
        command_count = (
            len(getattr(command, "commands", {}) or {}) if command is not None else 0
        )
        dispatch_attempted = (
            bool(execution.dispatch_attempted)
            if execution is not None
            else bool(command_count and send_actions)
        )
        self.log_event(
            "action_step",
            {
                "timestamp": float(timestamp),
                "step_index": int(step_index),
                "policy_name": str(policy_name),
                "send_actions": bool(send_actions),
                "dispatch_attempted": bool(dispatch_attempted),
                "dispatch_completed": bool(
                    execution.dispatch_completed
                    if execution is not None
                    else dispatch_attempted and dispatch_error is None
                ),
                "dispatch_asynchronous": bool(
                    execution.asynchronous if execution is not None else False
                ),
                "dispatch_mode": (
                    execution.mode
                    if execution is not None
                    else ("send_actions" if send_actions else "dry_run")
                ),
                "position_source_index": (
                    None if execution is None else execution.position_source_index
                ),
                "dispatch_error": (
                    None
                    if dispatch_error is None
                    else {
                        "type": type(dispatch_error).__name__,
                        "message": str(dispatch_error),
                    }
                ),
                "queue_len_before_pop": int(queue_len_before_pop),
                "queue_len_after_pop": int(queue_len_after_pop),
                "faults": list(faults),
                "phase_timing": _debug_compact_value(
                    {} if phase_timing is None else phase_timing,
                    max_items=32,
                ),
                "raw_action": raw_summary,
                "filtered_action": filtered_summary,
                "command": command_summary,
                "canonical_metadata": _debug_metadata_summary(
                    getattr(canonical_action, "metadata", {}),
                    full_payload=self.full_payload,
                ),
                "raw_to_filtered_pose_delta": _debug_pose_delta_by_side(
                    raw_pose, filtered_pose
                ),
                "filtered_to_command_pose_delta": _debug_pose_delta_by_side(
                    filtered_pose, command_pose
                ),
                "command_to_pre_tcp_delta": _debug_pose_delta_by_side(
                    command_pose, pre_tcp_pose
                ),
                "command_to_post_tcp_delta": _debug_pose_delta_by_side(
                    command_pose, post_tcp_pose
                ),
                "pre_arm_state_by_side": _debug_arm_state_summary(
                    pre_arm_states,
                    full_payload=self.full_payload,
                ),
                "post_arm_state_by_side": _debug_arm_state_summary(
                    post_arm_states,
                    full_payload=self.full_payload,
                ),
                "backend_limits_by_side": _debug_backend_limits(
                    arm_backends, now=time.time()
                ),
            },
        )


__all__ = [
    "DeploymentDebugLogger",
    "RuntimeArtifactLayout",
    "canonical_deployment_debug_root",
    "resolve_deployment_debug_root",
]
