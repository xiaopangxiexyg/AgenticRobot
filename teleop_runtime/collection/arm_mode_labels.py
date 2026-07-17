from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robot_motion.pose import pose_delta_metrics

from ..schemas import TeleopCommand

ARM_MODE_SCHEMA = "dexteleop_arm_mode_v2"
INIT_AUDIT_SCHEMA = "dexteleop_init_audit_v1"

# Match the existing hand-eye auto-record stillness detector.  The velocity
# limits are the corresponding 1.5 mm / 0.25 deg micro-motion allowance over
# a 100 ms audit interval; they annotate collection data only and never alter
# the init controller.
INIT_TARGET_POSITION_TOLERANCE_M = 0.0015
INIT_TARGET_ROTATION_TOLERANCE_DEG = 0.25
INIT_STILL_TRANSLATION_SPEED_M_S = 0.015
INIT_STILL_ROTATION_SPEED_DEG_S = 2.5
INIT_STILL_DWELL_SEC = 3.0
INIT_AUDIT_MAX_SAMPLE_GAP_SEC = 0.5

ARM_MODE_IDS: dict[str, int] = {
    "unknown": 0,
    "teleop": 1,
    "fixed_hold": 2,
    "init_pose": 3,
    "stopped": 4,
}

ARM_MODE_ID_TO_LABEL = {mode_id: label for label, mode_id in ARM_MODE_IDS.items()}


def _normalize_side(side: Any) -> str:
    normalized = str(side or "").strip().lower()
    return normalized if normalized in {"left", "right"} else ""


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _pose_or_none(value: Any) -> np.ndarray | None:
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    return pose.copy()


def _metadata_values(command: TeleopCommand | None, key: str) -> list[Any]:
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


def _first_metadata(command: TeleopCommand | None, key: str) -> Any:
    for value in _metadata_values(command, key):
        if value is not None:
            return value
    return None


def _active_sides(command: TeleopCommand | None, *, active_key: str, side_key: str) -> set[str]:
    sides: set[str] = set()
    side = _normalize_side(_first_metadata(command, side_key))
    if side:
        sides.add(side)
    if command is None:
        return sides
    for effector_command in command.commands.values():
        metadata = effector_command.metadata or {}
        if not _boolish(metadata.get(active_key)):
            continue
        side = _normalize_side(metadata.get(side_key) or effector_command.side)
        if side:
            sides.add(side)
    return sides


def _init_sides(command: TeleopCommand | None) -> set[str]:
    sides = _active_sides(command, active_key="init_arm_active", side_key="init_arm_side")
    side = _normalize_side(_first_metadata(command, "init_arm_active_side"))
    if side:
        sides.add(side)
    return sides


def _fix_sides(command: TeleopCommand | None) -> set[str]:
    sides = _active_sides(command, active_key="fix_arm_active", side_key="fix_arm_side")
    side = _normalize_side(_first_metadata(command, "fix_arm_active_side"))
    if side:
        sides.add(side)
    return sides


def _default_init_audit(*, phase: str = "not_active", valid: bool = False, reason: str = "") -> dict[str, Any]:
    return {
        "schema": INIT_AUDIT_SCHEMA,
        "init_phase": phase,
        "init_phase_valid": bool(valid),
        "reason": str(reason),
        "target_position_error_m": None,
        "target_rotation_error_deg": None,
        "translation_speed_m_s": None,
        "rotation_speed_deg_s": None,
        "dwell_sec": 0.0,
    }


def _normalized_init_audit(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _default_init_audit()
    phase = str(value.get("init_phase", "unknown") or "unknown")
    if phase not in {"not_active", "init_move", "init_hold", "unknown"}:
        phase = "unknown"
    normalized = _default_init_audit(
        phase=phase,
        valid=bool(value.get("init_phase_valid", False)),
        reason=str(value.get("reason", "") or ""),
    )
    normalized["schema"] = str(value.get("schema") or INIT_AUDIT_SCHEMA)
    for field_name in (
        "target_position_error_m",
        "target_rotation_error_deg",
        "translation_speed_m_s",
        "rotation_speed_deg_s",
        "dwell_sec",
    ):
        normalized[field_name] = _finite_float(value.get(field_name))
    if normalized["dwell_sec"] is None:
        normalized["dwell_sec"] = 0.0
    return normalized


@dataclass
class _InitAuditSideState:
    last_pose: np.ndarray | None = None
    last_timestamp: float | None = None
    last_target: np.ndarray | None = None
    stable_since: float | None = None


class InitArmAuditTracker:
    """Derive audit-only init phases from actual TCP feedback.

    The tracker deliberately has no control authority.  It is held by the
    collection sample builder so phase history never appears in policy inputs
    or deployment observations.
    """

    def __init__(self) -> None:
        self._state_by_side = {side: _InitAuditSideState() for side in ("left", "right")}

    def reset(self) -> None:
        self._state_by_side = {side: _InitAuditSideState() for side in ("left", "right")}

    def observe(
        self,
        *,
        command: TeleopCommand | None,
        arm_states: Mapping[str, Mapping[str, Any] | None],
        timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        sample_timestamp = _finite_float(timestamp)
        init_sides = _init_sides(command)
        audits: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            if side not in init_sides:
                self._state_by_side[side] = _InitAuditSideState()
                audits[side] = _default_init_audit()
                continue
            audits[side] = self._observe_side(
                side=side,
                command=command,
                arm_state=arm_states.get(side),
                timestamp=sample_timestamp,
            )
        return audits

    def _observe_side(
        self,
        *,
        side: str,
        command: TeleopCommand | None,
        arm_state: Mapping[str, Any] | None,
        timestamp: float | None,
    ) -> dict[str, Any]:
        state = self._state_by_side[side]
        target = _init_target_pose(command, side)
        actual = _actual_tcp_pose(arm_state)
        if timestamp is None:
            self._state_by_side[side] = _InitAuditSideState()
            return _default_init_audit(phase="unknown", reason="invalid_sample_timestamp")
        if target is None:
            self._state_by_side[side] = _InitAuditSideState()
            return _default_init_audit(phase="unknown", reason="missing_init_target")
        if actual is None:
            self._state_by_side[side] = _InitAuditSideState()
            return _default_init_audit(phase="unknown", reason="missing_or_invalid_actual_tcp_feedback")

        target_position_error_m, target_rotation_error_deg = pose_delta_metrics(actual, target)
        target_changed = (
            state.last_target is not None
            and any(value > 1e-9 for value in pose_delta_metrics(state.last_target, target))
        )
        dt_sec: float | None = None
        translation_speed_m_s: float | None = None
        rotation_speed_deg_s: float | None = None
        continuity_valid = False
        if state.last_pose is not None and state.last_timestamp is not None:
            dt_sec = float(timestamp - state.last_timestamp)
            if 0.0 < dt_sec <= INIT_AUDIT_MAX_SAMPLE_GAP_SEC:
                moved_m, moved_deg = pose_delta_metrics(state.last_pose, actual)
                translation_speed_m_s = moved_m / dt_sec
                rotation_speed_deg_s = moved_deg / dt_sec
                continuity_valid = True

        within_target = (
            target_position_error_m <= INIT_TARGET_POSITION_TOLERANCE_M
            and target_rotation_error_deg <= INIT_TARGET_ROTATION_TOLERANCE_DEG
        )
        below_speed_limit = bool(
            continuity_valid
            and translation_speed_m_s is not None
            and rotation_speed_deg_s is not None
            and translation_speed_m_s <= INIT_STILL_TRANSLATION_SPEED_M_S
            and rotation_speed_deg_s <= INIT_STILL_ROTATION_SPEED_DEG_S
        )
        eligible_for_dwell = bool(within_target and below_speed_limit and not target_changed)
        if not eligible_for_dwell:
            state.stable_since = None
        elif state.stable_since is None:
            state.stable_since = float(timestamp)

        dwell_sec = (
            max(0.0, float(timestamp - state.stable_since))
            if state.stable_since is not None
            else 0.0
        )
        if eligible_for_dwell and dwell_sec >= INIT_STILL_DWELL_SEC:
            phase = "init_hold"
            reason = "target_reached_and_still"
        elif not continuity_valid:
            phase = "init_move"
            reason = "waiting_for_continuous_actual_feedback"
        elif not within_target:
            phase = "init_move"
            reason = "target_error_above_tolerance"
        elif not below_speed_limit:
            phase = "init_move"
            reason = "actual_speed_above_stillness_threshold"
        elif target_changed:
            phase = "init_move"
            reason = "init_target_changed"
        else:
            phase = "init_move"
            reason = "dwell_in_progress"

        state.last_pose = actual.copy()
        state.last_timestamp = float(timestamp)
        state.last_target = target.copy()
        return {
            "schema": INIT_AUDIT_SCHEMA,
            "init_phase": phase,
            "init_phase_valid": True,
            "reason": reason,
            "target_position_error_m": float(target_position_error_m),
            "target_rotation_error_deg": float(target_rotation_error_deg),
            "translation_speed_m_s": translation_speed_m_s,
            "rotation_speed_deg_s": rotation_speed_deg_s,
            "dwell_sec": float(dwell_sec),
        }


def _init_target_pose(command: TeleopCommand | None, side: str) -> np.ndarray | None:
    if command is None:
        return None
    arm_command = command.commands.get(f"{side}_arm")
    if arm_command is None:
        return None
    metadata = arm_command.metadata or {}
    return _pose_or_none(metadata.get("init_arm_target_pose_base"))


def _actual_tcp_pose(arm_state: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(arm_state, Mapping):
        return None
    if arm_state.get("causal_measured_state_eligible") is False:
        return None
    if arm_state.get("actual_state_valid") is False or _boolish(arm_state.get("actual_state_stale")):
        return None
    return _pose_or_none(arm_state.get("actual_tcp_pose"))


def _side_payload(
    *,
    mode: str,
    source: str,
    tags: Sequence[str] = (),
    detail: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    label = str(mode or "unknown")
    if label not in ARM_MODE_IDS:
        label = "unknown"
    mode_id = int(ARM_MODE_IDS[label])
    tag_list = [str(tag) for tag in tags if str(tag)]
    return {
        "mode": label,
        "label": label,
        "mode_id": mode_id,
        "id": mode_id,
        "source": str(source or ""),
        "is_special": bool(label in {"fixed_hold", "init_pose"}),
        "tags": tag_list,
        "detail": dict(detail or {}),
        "audit": _normalized_init_audit(audit),
    }


def build_arm_mode_metadata(
    *,
    command: TeleopCommand | None,
    session_state: str,
    active_sides: Sequence[str] = ("left", "right"),
    recording_paused: bool = False,
    init_audit_by_side: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build model mode labels plus non-model audit metadata.

    `init_audit_by_side` is emitted by :class:`InitArmAuditTracker`.  Its
    fields remain Parquet/JSON audit data and never change action or state
    vectors.
    """

    normalized_active_sides = {
        side for side in (_normalize_side(side) for side in active_sides) if side
    }
    state_text = str(session_state or "").strip()
    state_upper = state_text.upper()
    stopped_by_session = bool(recording_paused or state_upper not in {"RUNNING"})
    init_sides = _init_sides(command)
    fix_sides = _fix_sides(command)
    audit_by_side = init_audit_by_side if isinstance(init_audit_by_side, Mapping) else {}

    sides: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        enabled = side in normalized_active_sides
        detail = {
            "active_side": enabled,
            "session_state": state_text,
            "recording_paused": bool(recording_paused),
        }
        init_audit = audit_by_side.get(side)
        if stopped_by_session:
            source = "recording_pause" if recording_paused else "session_state"
            sides[side] = _side_payload(
                mode="stopped" if enabled else "unknown",
                source=source,
                detail=detail,
            )
        elif side in init_sides:
            sides[side] = _side_payload(
                mode="init_pose",
                source="init_arm",
                tags=("init_arm",),
                detail={**detail, "init_arm_mode": _first_metadata(command, "init_arm_mode")},
                audit=init_audit or _default_init_audit(phase="unknown", reason="audit_not_collected"),
            )
        elif side in fix_sides:
            scope = str(_first_metadata(command, "fix_arm_scope") or "arm_and_end_effector")
            warning = str(_first_metadata(command, "fix_arm_warning") or "")
            sides[side] = _side_payload(
                mode="fixed_hold",
                source="fix_arm",
                tags=("fix_arm",),
                detail={
                    **detail,
                    "fix_arm_mode": _first_metadata(command, "fix_arm_mode"),
                    "effector_scope": scope,
                    "fix_arm_warning": warning,
                    "fix_arm_effectors": _first_metadata(command, "fix_arm_effectors"),
                },
            )
        elif enabled:
            sides[side] = _side_payload(mode="teleop", source="normal", detail=detail)
        else:
            sides[side] = _side_payload(mode="unknown", source="inactive_side", detail=detail)

    active_special_tasks = sorted(
        {
            tag
            for side_payload in sides.values()
            for tag in side_payload.get("tags", ())
            if tag in {"fix_arm", "init_arm"}
        }
    )
    return {
        "schema": ARM_MODE_SCHEMA,
        "available": True,
        "sides": sides,
        "active_special_tasks": active_special_tasks,
    }


def build_operator_annotation_arm_mode_metadata(
    *,
    labels_by_side: Mapping[str, str],
    source_by_side: Mapping[str, str] | None = None,
    changed_at_by_side: Mapping[str, float] | None = None,
    sequence: int = 0,
) -> dict[str, Any]:
    """Build existing arm-mode columns for an arm-free annotation source.

    The wire values remain compatible with v2.3 readers, while the JSON detail
    makes it explicit that no arm controller or arrival audit was involved.
    """

    sources = {} if source_by_side is None else dict(source_by_side)
    changed = {} if changed_at_by_side is None else dict(changed_at_by_side)
    sides: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        mode = str(labels_by_side.get(side, "teleop"))
        if mode not in ARM_MODE_IDS:
            raise ValueError(f"unsupported operator arm-mode annotation {mode!r}")
        tags = ()
        if mode == "fixed_hold":
            tags = ("fix_arm",)
        elif mode == "init_pose":
            tags = ("init_arm",)
        sides[side] = _side_payload(
            mode=mode,
            source="umi_operator_annotation",
            tags=tags,
            detail={
                "active_side": True,
                "annotation_only": True,
                "control_effect": False,
                "embodiment_has_arm": False,
                "input_source": str(sources.get(side, "default")),
                "effector_scope": "tool_tcp_and_gripper",
                "changed_at": _finite_float(changed.get(side)),
                "event_sequence": int(sequence),
            },
            audit=_default_init_audit(
                phase="not_active",
                valid=False,
                reason="annotation_only_no_arm",
            ),
        )
    active_special_tasks = sorted(
        {
            tag
            for side_payload in sides.values()
            for tag in side_payload.get("tags", ())
        }
    )
    return {
        "schema": ARM_MODE_SCHEMA,
        "available": True,
        "sides": sides,
        "active_special_tasks": active_special_tasks,
    }


def normalize_arm_mode_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        raw_sides = value.get("sides")
        if isinstance(raw_sides, Mapping):
            sides = {}
            for side in ("left", "right"):
                raw_side = raw_sides.get(side)
                if isinstance(raw_side, Mapping):
                    raw_mode = raw_side.get("mode", raw_side.get("label", "unknown"))
                    mode = str(raw_mode or "unknown")
                    raw_tags = raw_side.get("tags", ())
                    tags = (
                        raw_tags
                        if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, (str, bytes))
                        else ()
                    )
                    sides[side] = _side_payload(
                        mode=mode,
                        source=str(raw_side.get("source", "")),
                        tags=tags,
                        detail=raw_side.get("detail", {}) if isinstance(raw_side.get("detail"), Mapping) else {},
                        audit=raw_side.get("audit"),
                    )
                else:
                    sides[side] = _side_payload(mode="unknown", source="missing_side")
            return {
                "schema": str(value.get("schema") or ARM_MODE_SCHEMA),
                "available": bool(value.get("available", True)),
                "sides": sides,
                "active_special_tasks": list(value.get("active_special_tasks", []))
                if isinstance(value.get("active_special_tasks"), Sequence)
                and not isinstance(value.get("active_special_tasks"), (str, bytes))
                else [],
            }
    return {
        "schema": ARM_MODE_SCHEMA,
        "available": False,
        "sides": {
            "left": _side_payload(mode="unknown", source="missing_metadata"),
            "right": _side_payload(mode="unknown", source="missing_metadata"),
        },
        "active_special_tasks": [],
    }


def flatten_arm_mode_columns(payload: Any) -> dict[str, Any]:
    normalized = normalize_arm_mode_metadata(payload)
    columns: dict[str, Any] = {
        "teleop.arm_mode.schema": normalized["schema"],
        "teleop.arm_mode.available": int(bool(normalized.get("available", False))),
    }
    sides = normalized.get("sides", {})
    for side in ("left", "right"):
        side_payload = sides.get(side, {}) if isinstance(sides, Mapping) else {}
        mode = str(side_payload.get("mode") or side_payload.get("label") or "unknown")
        if mode not in ARM_MODE_IDS:
            mode = "unknown"
        audit = _normalized_init_audit(side_payload.get("audit"))
        detail = side_payload.get("detail", {})
        detail = detail if isinstance(detail, Mapping) else {}
        columns[f"teleop.arm_mode.{side}.id"] = int(ARM_MODE_IDS[mode])
        columns[f"teleop.arm_mode.{side}.label"] = mode
        columns[f"teleop.arm_mode.{side}.is_special"] = int(
            bool(side_payload.get("is_special", mode in {"fixed_hold", "init_pose"}))
        )
        columns[f"teleop.arm_mode.{side}.source"] = str(side_payload.get("source", ""))
        tags = side_payload.get("tags", ())
        columns[f"teleop.arm_mode.{side}.tags"] = [
            str(tag) for tag in tags
        ] if isinstance(tags, Sequence) and not isinstance(tags, (str, bytes)) else []
        columns[f"teleop.arm_mode.{side}.scope"] = str(detail.get("effector_scope", "") or "")
        columns[f"teleop.arm_mode.{side}.init_phase"] = str(audit["init_phase"])
        columns[f"teleop.arm_mode.{side}.init_phase_valid"] = int(
            bool(audit["init_phase_valid"])
        )
        columns[f"teleop.arm_mode.{side}.init_target_position_error_m"] = audit[
            "target_position_error_m"
        ]
        columns[f"teleop.arm_mode.{side}.init_target_rotation_error_deg"] = audit[
            "target_rotation_error_deg"
        ]
        columns[f"teleop.arm_mode.{side}.init_translation_speed_m_s"] = audit[
            "translation_speed_m_s"
        ]
        columns[f"teleop.arm_mode.{side}.init_rotation_speed_deg_s"] = audit[
            "rotation_speed_deg_s"
        ]
        columns[f"teleop.arm_mode.{side}.init_dwell_sec"] = audit["dwell_sec"]
    return columns
