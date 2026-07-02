from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RM65JointSafetyReport:
    ok: bool
    joint_deg: tuple[float, ...]
    position_limit_code: int | None = None
    self_collision_code: int | None = None
    position_limit_available: bool = False
    self_collision_available: bool = False
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def raise_if_unsafe(self) -> None:
        if not self.ok:
            raise RuntimeError(self.message)

    @property
    def status_text(self) -> str:
        if self.ok:
            parts = []
            if self.position_limit_available:
                parts.append("position_limit=ok")
            if self.self_collision_available:
                parts.append("self_collision=ok")
            return " ".join(parts) if parts else "unavailable"
        return "unsafe"


def check_rm65_joint_safety(
    robot: Any,
    joint_deg: Sequence[float],
    *,
    side: str = "",
    label: object = "",
    previous_joint_deg: Sequence[float] | None = None,
    start_joint_deg: Sequence[float] | None = None,
    check_position_limit: bool = True,
    check_self_collision: bool = True,
    context: Mapping[str, Any] | None = None,
) -> RM65JointSafetyReport:
    joint = _as_joint_tuple(joint_deg)
    metadata: dict[str, Any] = {
        "side": str(side) if side else "",
        "label": str(label) if label != "" else "",
        "joint_deg": list(joint),
    }
    if context:
        metadata.update(dict(context))

    previous = _optional_joint(previous_joint_deg)
    if previous is not None:
        delta = np.asarray(joint, dtype=float) - np.asarray(previous, dtype=float)
        metadata["previous_joint_deg"] = list(previous)
        metadata["joint_step_delta_deg"] = delta.tolist()
        metadata["joint_step_abs_delta_deg"] = np.abs(delta).tolist()
        metadata["joint_step_max_abs_delta_deg"] = float(np.max(np.abs(delta)))

    start = _optional_joint(start_joint_deg)
    if start is not None:
        total = np.asarray(joint, dtype=float) - np.asarray(start, dtype=float)
        metadata["start_joint_deg"] = list(start)
        metadata["joint_total_delta_deg"] = total.tolist()
        metadata["joint_total_abs_delta_deg"] = np.abs(total).tolist()
        metadata["joint_total_max_abs_delta_deg"] = float(np.max(np.abs(total)))

    position_code: int | None = None
    position_available = False
    if check_position_limit:
        position_code, position_available = _call_position_limit_checker(robot, joint)
        metadata["position_limit_code"] = position_code
        metadata["position_limit_available"] = position_available

    self_collision_code: int | None = None
    self_collision_available = False
    if check_self_collision:
        self_collision_code, self_collision_available = _call_self_collision_checker(robot, joint)
        metadata["self_collision_code"] = self_collision_code
        metadata["self_collision_available"] = self_collision_available

    unsafe_reasons: list[str] = []
    if position_available and position_code not in (0, None):
        if position_code == -1:
            unsafe_reasons.append("position_limit_unsupported")
        else:
            unsafe_reasons.append(f"joint_position_limit_code={position_code}")
    if self_collision_available and self_collision_code not in (0, None):
        unsafe_reasons.append(f"self_collision_code={self_collision_code}")

    ok = not unsafe_reasons
    message = "" if ok else _format_joint_safety_failure(metadata, unsafe_reasons)
    return RM65JointSafetyReport(
        ok=ok,
        joint_deg=joint,
        position_limit_code=position_code,
        self_collision_code=self_collision_code,
        position_limit_available=position_available,
        self_collision_available=self_collision_available,
        message=message,
        metadata=metadata,
    )


def _call_position_limit_checker(robot: Any, joint: tuple[float, ...]) -> tuple[int | None, bool]:
    checker = getattr(robot, "Algo_IKine_Check_Joint_Position_Limit", None)
    if checker is None:
        core = getattr(robot, "_robot", None)
        checker = getattr(core, "rm_algo_ikine_check_joint_position_limit", None)
    if checker is None:
        return None, False
    try:
        return int(checker(list(joint))), True
    except Exception as exc:  # noqa: BLE001 - diagnostics should not hide caller context.
        return None, False


def _call_self_collision_checker(robot: Any, joint: tuple[float, ...]) -> tuple[int | None, bool]:
    checker = getattr(robot, "Algo_Safety_Robot_Self_Collision_Detection", None)
    if checker is None:
        core = getattr(robot, "_robot", None)
        checker = getattr(core, "rm_algo_safety_robot_self_collision_detection", None)
    if checker is None:
        return None, False
    try:
        return int(checker(list(joint))), True
    except Exception:  # noqa: BLE001 - diagnostics should report unavailable if SDK fails.
        return None, False


def _format_joint_safety_failure(metadata: Mapping[str, Any], unsafe_reasons: Sequence[str]) -> str:
    prefix = "RM65 joint safety check failed"
    side = str(metadata.get("side") or "")
    label = str(metadata.get("label") or metadata.get("step") or "")
    phase = str(metadata.get("phase") or "")
    if side:
        prefix = f"{side}: {prefix}"
    if label:
        prefix += f" at step={label}"
    if phase:
        prefix += f" phase={phase}"

    lines = [
        prefix,
        f"reasons={','.join(str(reason) for reason in unsafe_reasons)}",
        "rm_self_collision_code_note=0 means clear; 1 means collision or joint-position-limit according to RM SDK.",
    ]
    ordered_keys = [
        "position_limit_code",
        "self_collision_code",
        "joint_deg",
        "previous_joint_deg",
        "joint_step_delta_deg",
        "joint_step_abs_delta_deg",
        "joint_step_max_abs_delta_deg",
        "start_joint_deg",
        "joint_total_delta_deg",
        "joint_total_abs_delta_deg",
        "joint_total_max_abs_delta_deg",
        "requested_pose",
        "ik_target_pose",
        "ik_target_joint_deg",
        "ik_target_delta_from_previous_deg",
        "joint_rate_limited",
        "joint_rate_limit_index",
        "joint_rate_limit_count",
        "joint_velocity_limit_profile",
        "joint_velocity_limit_observed_dt_sec",
        "joint_velocity_limit_effective_dt_sec",
        "joint_velocity_limit_max_dt_sec",
        "joint_velocity_limit_delta_deg",
        "joint_velocity_limit_max_step_deg",
        "joint_velocity_limit_ratio",
        "dual_arm_collision_distance_m",
        "dual_arm_collision_min_distance_m",
    ]
    seen = set()
    for key in ordered_keys:
        if key in metadata:
            lines.append(f"{key}={_format_value(metadata[key])}")
            seen.add(key)
    for key in sorted(metadata):
        if key in seen or key in {"side", "label", "step", "phase"}:
            continue
        lines.append(f"{key}={_format_value(metadata[key])}")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _as_joint_tuple(values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        raise ValueError(f"joint_deg must contain 6 finite values, got shape={array.shape}")
    return tuple(float(value) for value in array)


def _optional_joint(values: Sequence[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    return _as_joint_tuple(values)
