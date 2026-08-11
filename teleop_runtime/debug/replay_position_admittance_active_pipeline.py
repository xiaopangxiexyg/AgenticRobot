"""Replay raw tool wrench through the production Active pre-IK pipeline.

This diagnostic installs the default-off Active target filter on an offline
shared RM65 retargeter, replays recorded raw tool-frame wrench at 60 Hz, and
feeds the admitted solver-frame qref FK back in the same order as the real
control loop.  It never constructs hardware, dispatch, or device resources.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.position_control.cartesian_admittance import (
    CartesianAdmittanceConfig,
)
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC,
    PositionAdmittanceShadowTargetFilter,
    admitted_solver_world_poses_from_joint_command,
)

from .replay_position_admittance_ik_trace import (
    DEFAULT_LEFT_TOOL_PROFILE,
    DEFAULT_REPLAY_HZ,
    DEFAULT_RIGHT_TOOL_PROFILE,
    SIDES,
    _build_retargeter,
    _command_qpos,
    _config_payload,
    _context,
    _frame_rows,
    _mapping,
    _qpos,
    _read_rows,
    _resolve_log,
    _rotation_error_deg,
    _stats,
    _target_at_tick,
    _tool_pose,
)
from .replay_position_admittance_force_trace import (
    _position_admittance_receipt,
)


RELEASE_TRANSLATION_LIMIT_M = 1e-3
RELEASE_ROTATION_LIMIT_DEG = 0.5
OFFLINE_ACTIVE_DEFAULT_ENVELOPE = "cautious"
OFFLINE_ACTIVE_ENVELOPES = {
    "cautious": {
        "max_offset_m": 0.002,
        "max_velocity_m_s": 0.005,
        "max_return_velocity_m_s": 0.005,
        "max_output_slew_m_s": 0.005,
    },
    "half_normal": {
        "max_offset_m": 0.04,
        "max_velocity_m_s": 0.03,
        "max_return_velocity_m_s": 0.015,
        "max_output_slew_m_s": 0.03,
    },
    "normal_parity": {
        "max_offset_m": 0.08,
        "max_velocity_m_s": 0.06,
        "max_return_velocity_m_s": 0.03,
        "max_output_slew_m_s": 0.06,
    },
}


def _active_config(
    recorded: CartesianAdmittanceConfig,
    *,
    active_envelope: str,
) -> CartesianAdmittanceConfig:
    """Retain recorded response dynamics under one named Active envelope."""

    def capped(
        values: Sequence[float],
        upper_bound: float,
    ) -> tuple[float, float, float]:
        if len(values) != 3:
            raise ValueError("active Cartesian limits must contain three values")
        return (
            min(float(values[0]), float(upper_bound)),
            min(float(values[1]), float(upper_bound)),
            min(float(values[2]), float(upper_bound)),
        )

    try:
        envelope = OFFLINE_ACTIVE_ENVELOPES[str(active_envelope)]
    except KeyError as exc:
        raise ValueError(
            "unknown active position admittance envelope: "
            + str(active_envelope)
        ) from exc
    return replace(
        recorded,
        max_offset_m=capped(
            recorded.max_offset_m,
            envelope["max_offset_m"],
        ),
        max_velocity_m_s=capped(
            recorded.max_velocity_m_s,
            envelope["max_velocity_m_s"],
        ),
        max_return_velocity_m_s=capped(
            recorded.max_return_velocity_m_s,
            envelope["max_return_velocity_m_s"],
        ),
        max_output_slew_m_s=capped(
            recorded.max_output_slew_m_s,
            envelope["max_output_slew_m_s"],
        ),
        stale_after_sec=min(
            recorded.stale_after_sec,
            POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC,
        ),
        max_force_dt_sec=min(
            recorded.max_force_dt_sec,
            POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC,
        ),
    )


def _cautious_active_config(
    recorded: CartesianAdmittanceConfig,
) -> CartesianAdmittanceConfig:
    """Compatibility helper for diagnostics that retain the cautious gate."""

    return _active_config(
        recorded,
        active_envelope=OFFLINE_ACTIVE_DEFAULT_ENVELOPE,
    )


def _install_offline_active_envelope(
    *,
    retargeter: Any,
    monitor: Any,
    active_sides: Sequence[str],
    config: CartesianAdmittanceConfig,
    side_activity: Any,
    apply_sides: Sequence[str],
    now_sec: float,
) -> PositionAdmittanceShadowTargetFilter:
    """Install a non-hardware Active counterfactual without widening runtime gates."""

    tool_tcp_from_solver = getattr(retargeter, "_tool_tcp_from_solver", {})
    rotations = {
        side: np.asarray(
            tool_tcp_from_solver.get(side, np.eye(4)),
            dtype=float,
        )[:3, :3]
        for side in active_sides
    }
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=active_sides,
        config=config,
        previous_filter=getattr(retargeter, "target_override_filter", None),
        side_activity=side_activity,
        tool_rotation_from_solver_by_side=rotations,
        apply_to_target=True,
        apply_sides=apply_sides,
    )
    observer.reset_from_monitor(now_sec=now_sec)
    retargeter.target_override_filter = observer
    return observer


def _vector(value: Any, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _pose(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 4x4 pose")
    return result.copy()


class _RecordedWrenchMonitor:
    def __init__(
        self,
        *,
        tare_by_side: Mapping[str, np.ndarray],
        initial_source_time_by_side: Mapping[str, float],
    ) -> None:
        self.values_by_side = {
            side: _vector(tare_by_side[side], size=6, name=f"{side}.tare")
            .astype(float)
            .tolist()
            for side in SIDES
        }
        self.source_time_by_side = {
            side: float(initial_source_time_by_side[side]) for side in SIDES
        }
        self.fault = None

    def update_from_frame(self, frame: Mapping[str, Any]) -> None:
        sides = _mapping(_mapping(frame.get("position_admittance")).get("sides"))
        for side in SIDES:
            payload = _mapping(sides.get(side))
            self.values_by_side[side] = _vector(
                payload.get("raw_wrench_n_nm"),
                size=6,
                name=f"{side}.raw_wrench_n_nm",
            ).tolist()
            source_time = float(payload.get("source_time_sec"))
            if not math.isfinite(source_time):
                raise ValueError(f"{side}.source_time_sec must be finite")
            self.source_time_by_side[side] = source_time

    def update_zero_tail(
        self,
        *,
        tare_by_side: Mapping[str, np.ndarray],
        source_time_by_side: Mapping[str, float],
    ) -> None:
        for side in SIDES:
            self.values_by_side[side] = np.asarray(
                tare_by_side[side], dtype=float
            ).tolist()
            self.source_time_by_side[side] = float(source_time_by_side[side])

    def snapshot(self) -> dict[str, object]:
        return {
            "fault": self.fault,
            "last_reading_by_side": {
                side: {
                    "side": side,
                    "source": "tool",
                    "values": list(self.values_by_side[side]),
                    "monotonic_time_sec": self.source_time_by_side[side],
                }
                for side in SIDES
            },
        }


def _frame_payload(
    frame: Mapping[str, Any],
    *,
    side: str,
) -> Mapping[str, Any]:
    return _mapping(
        _mapping(_mapping(frame.get("position_admittance")).get("sides")).get(
            side
        )
    )


def _recorded_inputs(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[
    list[float],
    list[dict[str, np.ndarray]],
    dict[str, np.ndarray],
    dict[str, float],
]:
    frame_times: list[float] = []
    targets: list[dict[str, np.ndarray]] = []
    for frame_index, frame in enumerate(frames):
        left_payload = _frame_payload(frame, side="left")
        observed_at = float(left_payload.get("observed_at_monotonic_sec"))
        if not math.isfinite(observed_at):
            raise ValueError(f"frame {frame_index} observed time must be finite")
        if frame_times and observed_at <= frame_times[-1]:
            raise ValueError("frame observed times must increase strictly")
        frame_times.append(observed_at)
        targets.append(
            {
                side: _pose(
                    _frame_payload(frame, side=side).get(
                        "raw_target_pose_solver_world"
                    ),
                    name=f"frame[{frame_index}].{side}.raw_target",
                )
                for side in SIDES
            }
        )

    first = frames[0]
    tare_by_side: dict[str, np.ndarray] = {}
    initial_source_time_by_side: dict[str, float] = {}
    for side in SIDES:
        payload = _frame_payload(first, side=side)
        raw = _vector(
            payload.get("raw_wrench_n_nm"),
            size=6,
            name=f"first.{side}.raw_wrench",
        )
        tared = _vector(
            payload.get("tared_wrench_n_nm"),
            size=6,
            name=f"first.{side}.tared_wrench",
        )
        tare_by_side[side] = raw - tared
        initial_source_time_by_side[side] = (
            float(payload.get("source_time_sec")) - 0.05
        )
    return frame_times, targets, tare_by_side, initial_source_time_by_side


def _active_rollout(
    *,
    mode: str,
    frames: Sequence[Mapping[str, Any]],
    frame_times: Sequence[float],
    raw_targets: Sequence[Mapping[str, np.ndarray]],
    tare_by_side: Mapping[str, np.ndarray],
    initial_source_time_by_side: Mapping[str, float],
    selected_side: str,
    config: CartesianAdmittanceConfig,
    active_envelope: str,
    config_payload: Mapping[str, Any],
    replay_hz: float,
    release_tail_sec: float,
    left_tool_pose: tuple[float, ...],
    right_tool_pose: tuple[float, ...],
) -> dict[str, Any]:
    initial_qpos = {
        side: _qpos(_mapping(frames[0].get("command_by_side")), side=side)
        for side in SIDES
    }
    retargeter, step_limit = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=left_tool_pose,
        right_tool_pose=right_tool_pose,
    )
    monitor = _RecordedWrenchMonitor(
        tare_by_side=tare_by_side,
        initial_source_time_by_side=initial_source_time_by_side,
    )
    tick_period = 1.0 / float(replay_hz)
    observer = _install_offline_active_envelope(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=SIDES,
        config=config,
        side_activity=lambda side: (
            (True, "offline_active_side")
            if side == selected_side
            else (False, "offline_fixed_side")
        ),
        apply_sides=(selected_side,),
        now_sec=frame_times[0] - tick_period,
    )
    opposite_side = "right" if selected_side == "left" else "left"
    end_time = frame_times[-1] + float(release_tail_sec)
    tick_count = int(
        math.ceil((end_time - frame_times[0]) * float(replay_hz))
    ) + 1
    tail_update_stride = max(1, int(round(float(replay_hz) / 20.0)))
    event_index = 0
    frame_index = 0
    current_qpos = {
        side: initial_qpos[side].copy() for side in SIDES
    }
    offsets: list[float] = []
    offset_slew_ratios: list[float] = []
    joint_step_ratios: list[float] = []
    target_translation_residuals: list[float] = []
    target_rotation_residuals: list[float] = []
    nominal_residuals: list[float] = []
    modifier_feedback_residuals: list[float] = []
    branch_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    feedback_frame_counts: Counter[str] = Counter()
    state_update_count = 0
    active_override_count = 0
    active_motion_tick_count = 0
    nonzero_modifier_ticks = 0
    max_anchor_norm_m = 0.0
    previous_offset = np.zeros(3, dtype=float)
    model_errors: list[dict[str, Any]] = []

    for tick_index in range(tick_count):
        now = frame_times[0] + tick_index * tick_period
        while (
            event_index < len(frames)
            and frame_times[event_index] <= now + 1e-9
        ):
            monitor.update_from_frame(frames[event_index])
            event_index += 1
        if (
            event_index == len(frames)
            and now > frame_times[-1]
            and tick_index % tail_update_stride == 0
        ):
            tail_delta = now - frame_times[-1]
            monitor.update_zero_tail(
                tare_by_side=tare_by_side,
                source_time_by_side={
                    side: float(
                        _frame_payload(frames[-1], side=side).get(
                            "source_time_sec"
                        )
                    )
                    + tail_delta
                    for side in SIDES
                },
            )
        raw, frame_index = _target_at_tick(
            tick_time=now,
            frame_times=frame_times,
            targets=raw_targets,
            frame_index=frame_index,
            mode=mode,
        )
        frame = frames[frame_index]
        try:
            context = _context(frame=frame, targets=raw)
            applied_targets = observer(
                raw,
                object(),
                object(),
                now_sec=now,
            )
            if applied_targets is None:
                raise RuntimeError("Active observer returned no target pair")
            retargeter._apply_target_override_activity_metadata(context)
            command = retargeter.build_command_for_context(
                context,
                targets=dict(applied_targets),
            )
            next_qpos: dict[str, np.ndarray] = {}
            metadata: dict[str, Mapping[str, Any]] = {}
            for side in SIDES:
                next_qpos[side], metadata[side] = _command_qpos(
                    command, side=side
                )
            selected_qpos_step = float(
                np.max(
                    np.abs(
                        next_qpos[selected_side]
                        - current_qpos[selected_side]
                    )
                )
            )
            joint_step_ratios.append(
                max(
                    float(
                        np.max(
                            np.abs(next_qpos[side] - current_qpos[side])
                            / step_limit
                        )
                    )
                    for side in SIDES
                )
            )
            retargeter.set_current_qpos_pair(
                left_qpos=next_qpos["left"],
                right_qpos=next_qpos["right"],
                reset_tracking=False,
            )
            admitted = admitted_solver_world_poses_from_joint_command(
                retargeter=retargeter,
                command=command,
            )
            observer.observe_admitted_poses(admitted, now_sec=now)
            current_qpos = next_qpos
        except Exception as exc:  # noqa: BLE001 - diagnostic report path.
            model_errors.append(
                {
                    "tick_index": tick_index,
                    "frame_index": frame_index,
                    "step_index": frame.get("step_index"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break

        snapshot = _mapping(
            _mapping(observer.snapshot(now_sec=now).get("sides")).get(
                selected_side
            )
        )
        offset = _vector(
            snapshot.get("applied_total_offset_m"),
            size=3,
            name="applied_total_offset_m",
        )
        offset_norm = float(np.linalg.norm(offset))
        offsets.append(offset_norm)
        nonzero_modifier_ticks += int(offset_norm > 1e-12)
        active_motion_tick_count += int(
            offset_norm > 1e-12 and selected_qpos_step > 1e-12
        )
        slew_limit = float(max(config.max_output_slew_m_s)) * tick_period
        offset_slew_ratios.append(
            0.0
            if slew_limit <= 0.0
            else float(np.linalg.norm(offset - previous_offset)) / slew_limit
        )
        previous_offset = offset
        anchor = _vector(
            snapshot.get("anchor_offset_m"),
            size=3,
            name="anchor_offset_m",
        )
        max_anchor_norm_m = max(max_anchor_norm_m, float(np.linalg.norm(anchor)))
        feedback = _mapping(snapshot.get("admission_feedback"))
        feedback_frame_counts[str(feedback.get("frame"))] += 1
        state_update_count += int(bool(feedback.get("state_update_applied")))
        modifier_feedback_residuals.append(
            float(
                np.linalg.norm(
                    _vector(
                        feedback.get("modifier_tracking_residual_m"),
                        size=3,
                        name="modifier_tracking_residual_m",
                    )
                )
            )
        )
        nominal_residuals.append(
            float(
                np.linalg.norm(
                    admitted[selected_side][:3, 3]
                    - raw[selected_side][:3, 3]
                )
            )
        )
        target_translation_residuals.append(
            float(
                np.linalg.norm(
                    admitted[selected_side][:3, 3]
                    - np.asarray(applied_targets[selected_side])[:3, 3]
                )
            )
        )
        target_rotation_residuals.append(
            _rotation_error_deg(
                admitted[selected_side][:3, :3],
                np.asarray(applied_targets[selected_side])[:3, :3],
            )
        )
        selected_command = (
            context.left_cartesian
            if selected_side == "left"
            else context.right_cartesian
        )
        active_override_count += int(
            selected_command is not None
            and selected_command.metadata.get("pose_action_motion_intent")
            is not None
        )
        for side in SIDES:
            branch_counts[str(metadata[side].get("ik_branch_candidate"))] += 1
            status_counts[
                str(metadata[side].get("collision_avoidance_status"))
            ] += 1

    completed = len(offsets) == tick_count and not model_errors
    final_translation = (
        target_translation_residuals[-1]
        if target_translation_residuals
        else None
    )
    final_nominal_translation = (
        nominal_residuals[-1] if nominal_residuals else None
    )
    final_rotation = (
        target_rotation_residuals[-1]
        if target_rotation_residuals
        else None
    )
    final_offset = offsets[-1] if offsets else None
    max_offset = float(
        max(
            limit
            for limit, enabled in zip(
                config.max_offset_m,
                config.enabled_axes,
                strict=True,
            )
            if enabled
        )
    )
    checks = [
        {
            "name": "active_pipeline_completed",
            "passed": completed,
            "detail": {
                "expected_ticks": tick_count,
                "completed_ticks": len(offsets),
                "errors": model_errors[:5],
            },
        },
        {
            "name": "admission_feedback_uses_solver_world",
            "passed": bool(
                feedback_frame_counts
                and set(feedback_frame_counts) == {"solver_world"}
            ),
            "detail": dict(feedback_frame_counts),
        },
        {
            "name": "final_composed_modifier_envelope",
            "passed": bool(
                offsets
                and max(offsets) <= max_offset + 1e-9
                and max(offset_slew_ratios, default=0.0) <= 1.0 + 1e-6
                and max_anchor_norm_m <= 1e-12
                and state_update_count == 0
            ),
            "detail": {
                "max_offset_m": max(offsets) if offsets else None,
                "offset_limit_m": max_offset,
                "max_slew_ratio": max(offset_slew_ratios, default=None),
                "max_anchor_norm_m": max_anchor_norm_m,
                "state_update_count": state_update_count,
            },
        },
        {
            "name": "formal_joint_step_preserved",
            "passed": bool(
                joint_step_ratios
                and max(joint_step_ratios) <= 1.0 + 1e-6
            ),
            "detail": {
                "max_ratio": max(joint_step_ratios, default=None),
            },
        },
        {
            "name": "nonzero_active_ik_coverage",
            "passed": bool(
                nonzero_modifier_ticks > 0
                and active_override_count > 0
                and active_motion_tick_count > 0
            ),
            "detail": {
                "nonzero_modifier_ticks": nonzero_modifier_ticks,
                "active_override_ticks": active_override_count,
                "active_qref_motion_ticks": active_motion_tick_count,
                "branch_counts": dict(branch_counts),
            },
        },
        {
            "name": "zero_force_release_rejoins_target",
            "passed": bool(
                completed
                and final_offset is not None
                and final_offset <= 1e-5
                and final_translation is not None
                and final_translation <= RELEASE_TRANSLATION_LIMIT_M
                and final_nominal_translation is not None
                and final_nominal_translation
                <= RELEASE_TRANSLATION_LIMIT_M
                and final_rotation is not None
                and final_rotation <= RELEASE_ROTATION_LIMIT_DEG
            ),
            "detail": {
                "final_offset_m": final_offset,
                "final_target_translation_m": final_translation,
                "final_nominal_translation_m": final_nominal_translation,
                "final_target_rotation_deg": final_rotation,
                "translation_limit_m": RELEASE_TRANSLATION_LIMIT_M,
                "rotation_limit_deg": RELEASE_ROTATION_LIMIT_DEG,
            },
        },
    ]
    return {
        "mode": mode,
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "tick_count": tick_count,
        "completed_tick_count": len(offsets),
        "model_errors": model_errors,
        "offset_norm_m": _stats(offsets),
        "offset_slew_ratio": _stats(offset_slew_ratios),
        "joint_step_ratio": _stats(joint_step_ratios),
        "nominal_translation_residual_m": _stats(nominal_residuals),
        "modifier_tracking_residual_m": _stats(
            modifier_feedback_residuals
        ),
        "target_translation_residual_m": _stats(
            target_translation_residuals
        ),
        "target_rotation_residual_deg": _stats(
            target_rotation_residuals
        ),
        "branch_counts": dict(branch_counts),
        "status_counts": dict(status_counts),
        "opposite_final_qpos_delta_deg": float(
            np.rad2deg(
                np.max(
                    np.abs(
                        current_qpos[opposite_side]
                        - initial_qpos[opposite_side]
                    )
                )
            )
        ),
    }


def replay_recorded_wrench_through_active_pipeline(
    path: Path,
    *,
    side: str = "left",
    active_envelope: str = OFFLINE_ACTIVE_DEFAULT_ENVELOPE,
    replay_hz: float = DEFAULT_REPLAY_HZ,
    release_tail_sec: float = 10.0,
    left_tool_profile: Path = Path(DEFAULT_LEFT_TOOL_PROFILE),
    right_tool_profile: Path = Path(DEFAULT_RIGHT_TOOL_PROFILE),
) -> dict[str, Any]:
    """Replay one shadow log through the production Active-only dataflow."""

    selected_side = str(side).strip().lower()
    if selected_side not in SIDES:
        raise ValueError("side must be left or right")
    if not math.isfinite(replay_hz) or replay_hz <= 0.0:
        raise ValueError("replay_hz must be positive and finite")
    if not math.isfinite(release_tail_sec) or release_tail_sec <= 0.0:
        raise ValueError("release_tail_sec must be positive and finite")
    log_path = _resolve_log(path)
    rows = _read_rows(log_path)
    frames = _frame_rows(rows)
    if not frames:
        raise ValueError("Active pipeline replay requires action-debug frames")
    receipt = _position_admittance_receipt(rows)
    recorded_config = _mapping(receipt.get("config"))
    envelope_name = str(active_envelope)
    config = _active_config(
        CartesianAdmittanceConfig(**dict(recorded_config)),
        active_envelope=envelope_name,
    )
    frame_times, raw_targets, tare_by_side, initial_source_times = (
        _recorded_inputs(frames)
    )
    config_payload = _config_payload(rows)
    left_tool_pose = _tool_pose(Path(left_tool_profile), side="left")
    right_tool_pose = _tool_pose(Path(right_tool_profile), side="right")
    rollouts = {
        mode: _active_rollout(
            mode=mode,
            frames=frames,
            frame_times=frame_times,
            raw_targets=raw_targets,
            tare_by_side=tare_by_side,
            initial_source_time_by_side=initial_source_times,
            selected_side=selected_side,
            config=config,
            active_envelope=envelope_name,
            config_payload=config_payload,
            replay_hz=float(replay_hz),
            release_tail_sec=float(release_tail_sec),
            left_tool_pose=left_tool_pose,
            right_tool_pose=right_tool_pose,
        )
        for mode in ("zoh", "linear")
    }
    checks = [
        {
            "name": "recorded_shadow_receipt",
            "passed": bool(
                receipt
                and receipt.get("enabled") is True
                and receipt.get("requested_mode") == "shadow"
            ),
            "detail": {
                "requested_mode": receipt.get("requested_mode"),
                "schema_version": receipt.get("schema_version"),
            },
        },
        {
            "name": "zoh_and_linear_active_pipeline",
            "passed": all(rollout["passed"] for rollout in rollouts.values()),
            "detail": {
                mode: rollout["passed"] for mode, rollout in rollouts.items()
            },
        },
    ]
    return {
        "schema": "dexteleop.position_admittance_active_pipeline_replay/v2",
        "passed": all(bool(check["passed"]) for check in checks),
        "evidence_level": "offline_raw_wrench_active_observer_shared_rm65_ik",
        "path": str(log_path),
        "side": selected_side,
        "active_envelope": envelope_name,
        "checks": checks,
        "recorded_config": recorded_config,
        "active_config": asdict(config),
        "rollouts": rollouts,
        "limitations": [
            "recorded wrench and target arrive at the 15 Hz action-debug rate",
            "recorded response dynamics are replayed under the explicitly "
            f"selected {envelope_name} Active caps",
            "the release tail injects fresh tare-valued wrench at 20 Hz",
            "admitted qref FK is a command receipt, not measured hardware pose",
            "position force guard and Arm_Stop are not executed by this replay",
            "offline replay does not authorize or execute hardware motion",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument(
        "--active-envelope",
        choices=tuple(OFFLINE_ACTIVE_ENVELOPES),
        default=OFFLINE_ACTIVE_DEFAULT_ENVELOPE,
    )
    parser.add_argument("--replay-hz", type=float, default=DEFAULT_REPLAY_HZ)
    parser.add_argument("--release-tail-sec", type=float, default=10.0)
    parser.add_argument(
        "--left-tool-profile",
        type=Path,
        default=Path(DEFAULT_LEFT_TOOL_PROFILE),
    )
    parser.add_argument(
        "--right-tool-profile",
        type=Path,
        default=Path(DEFAULT_RIGHT_TOOL_PROFILE),
    )
    args = parser.parse_args()
    report = replay_recorded_wrench_through_active_pipeline(
        args.log,
        side=args.side,
        active_envelope=args.active_envelope,
        replay_hz=args.replay_hz,
        release_tail_sec=args.release_tail_sec,
        left_tool_profile=args.left_tool_profile,
        right_tool_profile=args.right_tool_profile,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
