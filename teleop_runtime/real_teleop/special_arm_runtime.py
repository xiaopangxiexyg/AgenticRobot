"""Safety-sensitive GUI special-arm runtime operations.

These helpers retain the exact state-read, planning-admission, Hold, and
backend-submode release order from the canonical real-teleop frontend.  They
only act when called by the owning coordinator.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from robot_motion.joint_limits import load_joint_velocity_limit_profiles

from ..backends.arms import RM65ImpedanceArmBackend
from ..log_utils import log_event as _log
from ..special_tasks.init_arm import InitArmTransition
from .components import RealTeleopComponents
from .defaults import (
    DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
    INIT_ARM_QPOS_SPEED_BASELINE_PROFILE,
)


MIN_INIT_ARM_QPOS_SPEED_SCALE = 0.1
MAX_INIT_ARM_QPOS_SPEED_SCALE = 10.0


def _normalize_init_arm_qpos_speed_scale(value: object, *, side: str) -> float:
    scale = float(value)
    if (
        not np.isfinite(scale)
        or scale < MIN_INIT_ARM_QPOS_SPEED_SCALE
        or scale > MAX_INIT_ARM_QPOS_SPEED_SCALE
    ):
        raise ValueError(
            f"{side} Init qpos speed scale must be within "
            f"[{MIN_INIT_ARM_QPOS_SPEED_SCALE:g}, {MAX_INIT_ARM_QPOS_SPEED_SCALE:g}], "
            f"got {value!r}."
        )
    return scale


def _init_arm_qpos_speed_scale_for_side(
    args: argparse.Namespace,
    side: str,
) -> float:
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"unsupported Init qpos side {side!r}.")
    return _normalize_init_arm_qpos_speed_scale(
        getattr(
            args,
            f"init_arm_{normalized_side}_qpos_speed_scale",
            DEFAULT_INIT_ARM_QPOS_SPEED_SCALE,
        ),
        side=normalized_side,
    )

def _fresh_gui_init_qpos_by_side(
    components: RealTeleopComponents,
    *,
    max_age_sec: float = 0.5,
) -> dict[str, np.ndarray]:
    """Read a fresh dual-arm hardware snapshot without issuing motion commands."""

    backends = {
        "left": components.left_state_backend,
        "right": components.right_state_backend,
    }
    preview_backends = {
        "left": getattr(components, "left_preview_backend", None),
        "right": getattr(components, "right_preview_backend", None),
    }
    if set(components.active_sides) != {"left", "right"}:
        raise RuntimeError("GUI Init qpos requires both arms to be active.")
    if any(backends[side] is None for side in ("left", "right")):
        raise RuntimeError("GUI Init qpos requires both real-arm state backends.")

    states: dict[str, Mapping[str, object]] = {}
    executor = components.arm_state_read_executor
    if executor is not None:
        futures = {
            side: executor.submit(backends[side].refresh_state_from_hardware)
            for side in ("left", "right")
        }
        for side, future in futures.items():
            states[side] = future.result()
    else:
        for side in ("left", "right"):
            states[side] = backends[side].refresh_state_from_hardware()

    now = time.time()
    qpos_by_side: dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        state = states[side]
        if not bool(state.get("hardware_read_success")):
            raise RuntimeError(
                f"{side} Init qpos state read failed: "
                f"{state.get('hardware_result_text', 'unknown')}"
            )
        if not bool(state.get("hardware_measurement_valid")):
            raise RuntimeError(f"{side} Init qpos hardware measurement is invalid.")
        arm_err = state.get("hardware_arm_err")
        sys_err = state.get("hardware_sys_err")
        if arm_err not in {None, 0} or sys_err not in {None, 0}:
            raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}.")
        timestamp = float(
            state.get("hardware_measurement_timestamp", state.get("hardware_refresh_time"))
        )
        age_sec = max(0.0, now - timestamp)
        if not np.isfinite(timestamp) or age_sec > float(max_age_sec):
            raise RuntimeError(
                f"{side} Init qpos feedback is stale: age={age_sec:.3f}s "
                f"> {float(max_age_sec):.3f}s."
            )
        qpos = np.asarray(state.get("hardware_qpos"), dtype=float).reshape(6)
        if not np.all(np.isfinite(qpos)):
            raise RuntimeError(f"{side} Init qpos feedback is non-finite.")
        preview_backend = preview_backends[side]
        normalizer = getattr(
            preview_backend,
            "normalize_local_qpos_observation",
            None,
        )
        if not callable(normalizer):
            raise RuntimeError(
                f"{side} Init qpos requires a preview backend with "
                "command-safe observation normalization."
            )
        try:
            qpos, _normalization = normalizer(
                qpos,
                label=f"{side} Init qpos hardware feedback",
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{side} Init qpos feedback violates the command-safe "
                f"observation envelope: {exc}"
            ) from exc
        qpos_by_side[side] = qpos.copy()
    return qpos_by_side


def start_gui_qpos_init(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    now: float,
    fresh_qpos_reader: Callable[..., dict[str, np.ndarray]],
    impedance_force_control_not_ready_sides: Callable[..., tuple[str, ...]],
    impedance_backend_type: type[object],
) -> InitArmTransition:
    """Start background planning for the selected exact-qpos GUI operation."""

    controller = getattr(components, "init_arm_controller", None)
    if controller is None:
        return InitArmTransition(False, "Init qpos controller is unavailable.")
    side = str(getattr(controller, "selected_side", "left") or "left")
    arm_control_mode = str(getattr(args, "arm_control_mode", ""))
    if arm_control_mode not in {"position", "impedance"}:
        return InitArmTransition(
            False,
            f"精确 Init qpos 不支持控制模式 {arm_control_mode!r}。",
            side=side,
        )
    if (
        not bool(getattr(components, "collision_guard_requested", False))
        or getattr(components, "dual_arm_retargeter", None) is None
    ):
        return InitArmTransition(
            False,
            "精确 Init qpos 要求双臂碰撞保护已启用；当前拒绝规划和执行。",
            side=side,
        )
    fix_controller = getattr(components, "fix_arm_controller", None)
    if fix_controller is not None and bool(getattr(fix_controller, "fixed", False)):
        return InitArmTransition(
            False,
            "Init qpos 需要锁定非目标臂为静态碰撞体；请先释放当前 Fix-arm。",
            side=side,
        )
    target = controller.targets_by_side.get(side)
    if target is None or target.arm_qpos is None:
        return InitArmTransition(
            False,
            f"{side} Init profile 没有 arm_qpos，不能执行精确关节归位。",
            side=side,
        )
    if arm_control_mode == "impedance":
        not_ready_sides = impedance_force_control_not_ready_sides(args, components)
        if not_ready_sides:
            side_text = "/".join(not_ready_sides)
            return InitArmTransition(
                False,
                "力控尚未初始化，未启动路径规划。"
                f"请先点击“开始遥操”初始化 {side_text} 力控；"
                "界面提示“力控初始化完成”后，再执行机械臂精确归位。",
                side=side,
            )

    start_qpos = fresh_qpos_reader(components)
    target_qpos = {name: value.copy() for name, value in start_qpos.items()}
    target_qpos[side] = np.asarray(target.arm_qpos, dtype=float).reshape(6).copy()
    speed_scale = _init_arm_qpos_speed_scale_for_side(args, side)
    base_velocity_profile = load_joint_velocity_limit_profiles()[
        INIT_ARM_QPOS_SPEED_BASELINE_PROFILE
    ]
    scaled_max_angvel_rad_s = tuple(
        float(np.deg2rad(value * speed_scale))
        for value in base_velocity_profile.max_velocity_deg_s
    )

    def feedback_provider() -> Mapping[str, Any]:
        return fresh_qpos_reader(components)

    def plan_builder():
        # Keep replay's hardware execution package out of multiprocessing
        # spawn bootstrap.  This import is needed only after an operator asks
        # for exact GUI Init planning.
        from ..replay.arm_init.qpos_execute import (  # noqa: PLC0415
            DEFAULT_QPOS_INIT_COLLISION_SCENE,
            build_rm65_qpos_init_plan,
        )

        return build_rm65_qpos_init_plan(
            start_qpos_by_side=start_qpos,
            target_qpos_by_side=target_qpos,
            active_sides=("left", "right"),
            qpos_dt_sec=0.05,
            joint_velocity_profile=INIT_ARM_QPOS_SPEED_BASELINE_PROFILE,
            max_steps_per_side=10000,
            collision_check=True,
            collision_scene_profile=DEFAULT_QPOS_INIT_COLLISION_SCENE,
            # Exact GUI Init is a single-target-arm operation.  The peer arm
            # remains a static collision body for the entire trajectory.
            branch_search=False,
            branch_search_max_branches=8,
            branch_neutral_qpos_by_side=None,
            qpos_init_max_angvel_rad_s=scaled_max_angvel_rad_s,
        )

    transition = controller.start_qpos_planning(
        start_qpos_by_side=start_qpos,
        target_qpos_by_side=target_qpos,
        feedback_provider=feedback_provider,
        plan_builder=plan_builder,
        now=float(now),
        max_tracking_error_deg=15.0,
        start_drift_tolerance_deg=1.0,
        arrival_tolerance_deg=0.75,
        arrival_dwell_sec=0.5,
        settle_timeout_sec=12.0,
        tracking_check_every=10,
    )
    if not transition.ok or arm_control_mode != "impedance":
        return transition

    operation_id = str(getattr(controller, "qpos_operation_id", "") or "")
    armed_backends: list[RM65ImpedanceArmBackend] = []
    try:
        for side_name, backend in (
            ("left", components.left_command_backend),
            ("right", components.right_command_backend),
        ):
            if not isinstance(backend, impedance_backend_type):
                raise RuntimeError(
                    f"{side_name} command backend does not support impedance exact-qpos transit"
                )
            backend.begin_exact_qpos_transit(operation_id)
            armed_backends.append(backend)
    except Exception:
        for backend in reversed(armed_backends):
            try:
                backend.end_exact_qpos_transit(operation_id)
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve admission failure.
                _log(
                    "InitArmQpos",
                    "WARNING: failed to roll back impedance exact-qpos transit: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    now=now,
                )
        controller.cancel_active(now=now, reason="impedance exact-qpos admission failed")
        raise
    return transition


def _cancel_special_arm_modes_for_hold(
    components: RealTeleopComponents,
    *,
    now: float,
    reason: str,
) -> list[str]:
    messages: list[str] = []
    fix_controller = getattr(components, "fix_arm_controller", None)
    if fix_controller is not None and bool(getattr(fix_controller, "fixed", False)):
        transition = fix_controller.cancel_active(
            now=now,
            reason=reason,
            clear_snapshots=True,
        )
        if transition.ok:
            side_label = "左臂" if transition.side == "left" else "右臂"
            messages.append(f"{side_label} fix-arm 已停止")
    init_controller = getattr(components, "init_arm_controller", None)
    if init_controller is not None and bool(getattr(init_controller, "active", False)):
        transition = init_controller.cancel_active(now=now, reason=reason)
        if transition.ok:
            side_label = "左臂" if transition.side == "left" else "右臂"
            messages.append(f"{side_label} init-arm 已停止")
    for side, backend in (
        ("left", getattr(components, "left_command_backend", None)),
        ("right", getattr(components, "right_command_backend", None)),
    ):
        release = getattr(backend, "end_exact_qpos_transit", None)
        if not callable(release):
            continue
        try:
            if release():
                messages.append(f"{side} impedance exact-qpos 子模式已释放")
        except Exception as exc:  # noqa: BLE001 - Hold must continue for the peer arm.
            messages.append(
                f"{side} impedance exact-qpos 释放失败: {type(exc).__name__}: {exc}"
            )
    if messages:
        reset_smoothing = getattr(
            getattr(components, "dual_arm_retargeter", None),
            "reset_target_pose_smoothing",
            None,
        )
        if callable(reset_smoothing):
            try:
                reset_smoothing()
            except Exception as exc:
                _log(
                    "Hold",
                    "WARNING: retargeter target smoothing reset failed after special-mode cancel: "
                    f"{type(exc).__name__}: {exc}",
                    now=now,
                )
    return messages


def _release_gui_qpos_init_transit(
    components: RealTeleopComponents,
    *,
    operation_id: str,
    now: float,
    sides: Sequence[str] = ("left", "right"),
) -> tuple[list[str], list[str]]:
    """Release an impedance-only exact-qpos lease at its ownership boundary."""

    messages: list[str] = []
    errors: list[str] = []
    backends = {
        "left": getattr(components, "left_command_backend", None),
        "right": getattr(components, "right_command_backend", None),
    }
    for side in tuple(dict.fromkeys(str(side) for side in sides)):
        backend = backends.get(side)
        if backend is None:
            continue
        release = getattr(backend, "end_exact_qpos_transit", None)
        if not callable(release):
            continue
        try:
            released = release(operation_id or None)
        except Exception as exc:  # noqa: BLE001 - caller converts this to safe Hold.
            errors.append(f"{side}: {type(exc).__name__}: {exc}")
        else:
            if released:
                messages.append(f"{side} impedance exact-qpos 子模式已释放")
    for message in messages:
        _log("InitArmQpos", message, now=now)
    for error in errors:
        _log("InitArmQpos", f"ERROR: exact-qpos 子模式释放失败: {error}", now=now)
    return messages, errors


def _disengage_position_backends_for_manual_interaction(
    components: RealTeleopComponents,
    *,
    now: float,
    reason: str,
) -> None:
    """Withdraw position qpos streams after entering an operator Hold state."""

    failures: list[str] = []
    for side in components.active_sides:
        backend = getattr(components, f"{side}_command_backend", None)
        disengage = getattr(backend, "disengage_position_command_stream", None)
        if not callable(disengage):
            continue
        try:
            disengage(reason)
        except Exception as exc:  # noqa: BLE001 - report both arms before failing.
            failures.append(f"{side}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            "position command stream disengage failed: " + "; ".join(failures)
        )
    _log(
        "Hold",
        "position command streams disengaged; no persistent qpos command remains "
        "active during manual interaction",
        now=now,
    )

__all__ = [
    "MAX_INIT_ARM_QPOS_SPEED_SCALE",
    "MIN_INIT_ARM_QPOS_SPEED_SCALE",
    "_cancel_special_arm_modes_for_hold",
    "_disengage_position_backends_for_manual_interaction",
    "_fresh_gui_init_qpos_by_side",
    "_init_arm_qpos_speed_scale_for_side",
    "_normalize_init_arm_qpos_speed_scale",
    "_release_gui_qpos_init_transit",
    "start_gui_qpos_init",
]
