"""Offline J3 reference/brake/capture fixture on the existing B601 effort plant."""

from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601PinocchioModel,
    B601PolicyAssistCommand,
    B601SharedControlConfig,
    B601StagedAssistConfig,
    B601StagedSingleJointAssistRuntime,
)
from robot_motion.rebot_b601_rs.joint_motion import (
    B601JointMotionConfig,
    B601WindowStopEvidence,
    B601WindowVelocityEstimator,
)
from robot_motion.rebot_b601_rs.torque_experiment import B601OnsetProbeConfig
from sim.contracts import TimingSpec
from sim.mujoco.scene import MujocoScene
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .mit_driver import (
    B601EffortTick,
    B601MujocoEffortDriver,
    default_b601_mit_profile,
)
from .real2sim import B601Real2SimControllerProfile, B601Real2SimPlantProfile
from .scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_GRIPPER_OPEN_QPOS,
    B601_RS_PHYSICS_HZ,
    build_b601_mujoco_scene,
)
from .shared_control_demo import _arm_state

JOINT_MOTION_FIXTURE = "joint-motion"
JOINT_MOTION_SCENARIOS = ("lifecycle", "target-arrival")
_PHYSICS_WINDOW_SEC = 0.02


def _synthetic_friction_limit(*, static_nm: float, kinetic_nm: float, velocity_rad_s: float, transition_rad_s: float) -> float:
    """Declared Stribeck-like constraint bound, never controller compensation."""
    return kinetic_nm + (static_nm - kinetic_nm) * math.exp(-(abs(velocity_rad_s) / transition_rad_s) ** 2)


def _physics_dynamics_snapshot(
    scene: MujocoScene, model: B601PinocchioModel, tick: B601EffortTick,
) -> dict[str, object]:
    """Read the completed step's pre-integration force balance, never forward/step.

    mj_step leaves forces, inertia and qacc at the pre-integration state while
    qpos/qvel/time have advanced. Keep both clocks explicit. Project forces
    through the full inertia matrix, including constrained auxiliary joints;
    the projections are an algebraic decomposition, not independent rollouts.
    """
    if not tick.plant_advanced or tick.torque_breakdown is None:
        raise ValueError("dynamics snapshot requires a completed effort step")
    indices = [scene.joint_dof_addr(name) for name in model.arm_joint_names]
    matrix = np.empty((scene.model.nv, scene.model.nv))
    scene.mujoco.mj_fullM(scene.model, scene.data, matrix)
    fresh_gravity = model.arm_generalized_gravity(
        tick.control_qpos, auxiliary_joint_positions=B601_RS_GRIPPER_OPEN_QPOS,
    )

    def full(arm: tuple[float, ...] | np.ndarray) -> np.ndarray:
        vector = np.zeros(scene.model.nv)
        vector[indices] = arm
        return vector

    overlay = full(tick.torque_breakdown.overlay_effective_nm)
    held_gravity = full(tick.torque_breakdown.gravity_torque_nm)
    local_damping = full(tick.driver_damping_torque_nm)
    local_position = full(tick.driver_position_torque_nm)
    terms = {
        "software_overlay": overlay,
        "motor_local_damping": local_damping,
        "motor_local_position": local_position,
        "gravity_cache_error": held_gravity - full(fresh_gravity),
        "actuator_conditioning": (
            scene.data.qfrc_actuator - held_gravity - overlay - local_damping - local_position
        ),
        "fresh_arm_gravity_minus_full_bias": full(fresh_gravity) - scene.data.qfrc_bias,
        "passive": scene.data.qfrc_passive.copy(),
        "external_applied": scene.data.qfrc_applied.copy(),
        "constraints": scene.data.qfrc_constraint.copy(),
    }
    forces = np.column_stack(list(terms.values()))
    if (
        not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(forces))
        or not np.all(np.isfinite(scene.data.qacc))
    ):
        raise ValueError("non-finite simulation dynamics")
    acceleration = np.linalg.solve(matrix, forces)
    balance = matrix @ scene.data.qacc - np.sum(forces, axis=1)
    return {
        "parameter_source": "simulation_ground_truth_not_hardware_force",
        "force_state_time_sec": tick.physics_time_sec,
        "integrated_state_time_sec": tick.measured_time_sec,
        "force_state_qpos_rad": tick.control_qpos,
        "force_state_qvel_rad_s": tick.control_qvel,
        "post_step_qvel_rad_s": tick.measured_qvel,
        "pre_integration_qacc_rad_s2": tuple(float(v) for v in scene.data.qacc[indices]),
        "integrated_qacc_rad_s2": tuple(
            (v - before) / scene.model.opt.timestep
            for v, before in zip(tick.measured_qvel, tick.control_qvel)
        ),
        "fresh_model_gravity_nm": tuple(float(v) for v in fresh_gravity),
        "gravity_cache_error_nm": tuple(float(v) for v in (held_gravity - full(fresh_gravity))[indices]),
        "force_terms_nm": {
            name: tuple(float(v) for v in value[indices]) for name, value in terms.items()
        },
        "acceleration_terms_rad_s2": {
            name: tuple(float(v) for v in acceleration[indices, j])
            for j, name in enumerate(terms)
        },
        "force_balance_max_residual_nm": float(np.max(np.abs(balance))),
    }


def _target_arrival_time(
    rows: list[dict[str, object]],
    *,
    tolerance_rad: float,
    velocity_rad_s: float,
    dwell_sec: float,
    withdraw_sec: float,
) -> float | None:
    """Require continuous stationary arrival maintained until withdrawal."""
    started, confirmed = None, None
    for row in rows:
        now = row["time_sec"]
        if now >= withdraw_sec:
            break
        at_target = (
            row["phase"] in {"track", "brake", "soft_hold"}
            and abs(row["target_error_rad"]) <= tolerance_rad
            and max(abs(row["raw_v3_rad_s"]), abs(row["control_qvel_rad_s"][2]))
            <= velocity_rad_s
        )
        if at_target:
            if started is None:
                started = now
            if confirmed is None and now - started >= dwell_sec:
                confirmed = now
        else:
            started, confirmed = None, None
    return confirmed


def _withdrawal_evaluation(
    rows: list[dict[str, object]], *, withdraw_sec: float,
    staged: B601StagedAssistConfig, base_driver_kd: tuple[float, ...],
    torque_limits: tuple[float, ...], torque_rates: tuple[float, ...],
) -> dict[str, object]:
    """Observe the existing protected release, never advance or repair control.

    A software receipt is not output-force sensing. Completion needs actual
    conditioned zero, restored native gains, and the existing stop dwell, not
    the instant at which the operator requested withdrawal.
    """
    if not isinstance(staged, B601StagedAssistConfig) or staged.motion_control is None:
        raise ValueError("withdrawal evaluation requires explicit motion control")
    if isinstance(withdraw_sec, bool) or not math.isfinite(withdraw_sec) or withdraw_sec < 0:
        raise ValueError("withdrawal clock must be finite and non-negative")
    motion = staged.motion_control
    deadline = withdraw_sec + motion.braking_timeout_sec + motion.max_control_gap_sec
    checks = {
        "withdrawal_receipts_valid": True,
        "withdrawal_requests_only_bounded_braking": True,
        "withdrawal_conditioned_torque_bounds": True,
        "withdrawal_damping_retained_during_tail": True,
        "withdrawal_completed_within_deadline": False,
        "withdrawal_zero_and_base_gains_retained": True,
    }
    previous = None
    stationary_start, stationary_q, completed = None, None, None
    observed = 0
    invalid = None
    completion_receipt_seen = False
    eps = 1e-9  # Same software-zero tolerance as the confirmed runtime.

    def six(values):
        if len(values) != 6 or any(isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError("six finite trace values required")
        return tuple(float(v) for v in values)

    base_driver_kd, torque_limits, torque_rates = map(six, (base_driver_kd, torque_limits, torque_rates))
    if min(base_driver_kd) < 0 or min(torque_limits) <= 0 or min(torque_rates) <= 0:
        raise ValueError("invalid withdrawal baseline gains or torque envelope")

    for row in rows:
        try:
            now = row["time_sec"]
            dispatch = row["dispatch"]
            sent = six(dispatch["host_command_torque_nm"])
            delta = six(dispatch["conditioned_overlay_nm"])
            gravity = six(dispatch["torque_breakdown"]["gravity_torque_nm"])
            kd = six(dispatch["driver_kd_nm_s_per_rad"])
            raw = six(row["raw_qvel_rad_s"])
            window = six(row["control_qvel_rad_s"])
            q = six(row["qpos_rad"])
            sequence = dispatch["sequence"]
            clock = dispatch["feedback_time_sec"]
            overlay = row["overlay_nm"]
            frame = row["motion"]
            phase = row["phase"]
            if not isinstance(phase, str):
                raise TypeError("invalid trace phase")
            if frame is not None:
                if not isinstance(frame["phase"], str):
                    raise TypeError("invalid motion phase")
                terms = tuple(frame[k] for k in ("position_torque_nm", "qvel_reference_rad_s", "requested_torque_nm", "control_qvel_rad_s"))
                if any(isinstance(v, bool) or not math.isfinite(v) for v in terms):
                    raise ValueError("invalid motion trace")
            if (
                isinstance(now, bool) or not math.isfinite(now)
                or isinstance(clock, bool) or not math.isfinite(clock)
                or not now <= clock <= now + motion.max_control_gap_sec
                or isinstance(sequence, bool) or not isinstance(sequence, int)
                or not math.isfinite(overlay) or isinstance(overlay, bool)
                or dispatch["command_applied"] is not True
                or dispatch["plant_advanced"] is not True
                or min(kd) < 0
                or any(abs(t-g-d) > eps for t, g, d in zip(sent, gravity, delta))
                or previous is not None and (
                    clock <= previous[0] or sequence <= previous[1]
                    or now <= previous[3] or now-previous[3] > motion.max_control_gap_sec + eps
                )
            ):
                raise ValueError("invalid trace receipt or clock")
        except (KeyError, TypeError, ValueError) as error:
            checks["withdrawal_receipts_valid"] = False
            invalid = str(error)
            break
        zero = max(abs(x) for x in delta) <= eps
        base = all(abs(k - b) <= eps for k, b in zip(kd, base_driver_kd))
        phase_ok = phase in {"releasing", "brake", "soft_hold", "human_override", "passive", "authority_exhausted"}
        phase_ok &= frame is None or frame["phase"] in {"brake", "soft_hold", "authority_exhausted"}
        # An exhausted/unconfirmed trial may already have exited before the
        # final scripted clutch. Retain its verified completion only while
        # every subsequent packet remains inactive, zero and at base gains.
        if not (zero and base and phase_ok and abs(overlay) <= eps):
            completion_receipt_seen = False
        elif row.get("release_completed") is True:
            completion_receipt_seen = True
        if now >= withdraw_sec:
            if observed == 0 and (
                now > withdraw_sec + motion.max_control_gap_sec + eps
                or not (previous is not None and previous[3] <= withdraw_sec + eps
                        or withdraw_sec == 0.0 and now <= eps)
            ):
                checks["withdrawal_receipts_valid"] = False
                invalid = "withdrawal boundary is not covered by consecutive host receipts"
            observed += 1
            braking = abs(overlay) <= eps or (
                frame is not None and frame["phase"] == "brake"
                and abs(frame["position_torque_nm"]) <= eps
                and abs(frame["qvel_reference_rad_s"]) <= eps
                and abs(frame["control_qvel_rad_s"] - window[2]) <= eps
                and abs(frame["requested_torque_nm"] + motion.braking_kd_nm_s_per_rad * frame["control_qvel_rad_s"]) <= eps
                and abs(overlay - max(-staged.tracking_overlay_cap_nm,
                                     min(staged.tracking_overlay_cap_nm, frame["requested_torque_nm"]))) <= eps
                and abs(overlay) <= staged.tracking_overlay_cap_nm + eps
            )
            checks["withdrawal_requests_only_bounded_braking"] &= phase_ok and braking
            bounded = all(abs(t) <= limit + eps for t, limit in zip(sent, torque_limits))
            if previous is not None:
                bounded &= all(
                    abs(t - p) <= rate * (now - previous[3]) + eps
                    for t, p, rate in zip(sent, previous[2], torque_rates)
                )
            checks["withdrawal_conditioned_torque_bounds"] &= bounded
            if not zero:
                active_kd = tuple(max(b, d) for b, d in zip(base_driver_kd, motion.drive_driver_kd_nm_s_per_rad))
                checks["withdrawal_damping_retained_during_tail"] &= all(k + eps >= required for k, required in zip(kd, active_kd))
            stopped = max(*(abs(v) for v in raw), *(abs(v) for v in window)) <= staged.settle_velocity_rad_s
            if not stopped:
                stationary_start, stationary_q = None, None
            elif stationary_start is None or max(abs(a-b) for a, b in zip(q, stationary_q)) > staged.settle_velocity_rad_s * staged.settle_dwell_sec:
                stationary_start, stationary_q = now, q
            dwell = stationary_start is not None and now - stationary_start >= staged.settle_dwell_sec - eps
            if completed is None and zero and base and dwell:
                confirmed = staged.motion_onset_confirm_displacement_rad is None or completion_receipt_seen
                if confirmed:
                    completed = clock
                    checks["withdrawal_completed_within_deadline"] = clock <= deadline + eps
            if completed is not None:
                checks["withdrawal_zero_and_base_gains_retained"] &= zero and base and phase_ok and abs(overlay) <= eps
        previous = (clock, sequence, sent, now)
    if not observed:
        checks["withdrawal_receipts_valid"] = False
    return {
        "kind": "protected_withdrawal_software_evidence",
        "checks": checks, "passed": all(checks.values()),
        "withdrawal_requested_sec": withdraw_sec, "completion_sec": completed,
        "deadline_sec": deadline, "observed_host_samples": observed,
        "invalid_trace_reason": invalid,
        "semantics": "Braking can temporarily retain stage damping and conditioned tail; no old-target drive, final bounds, stop dwell, complete zero/base-gain receipt and deadline are required. Not measured output force or automatic human-contact detection.",
    }


def _motion_diagnostics(
    rows: list[dict[str, object]], *, staged: B601StagedAssistConfig,
    withdraw_sec: float, position_quantum_rad: float,
) -> dict[str, object]:
    """Summarize valid fixture receipts without changing admission or dwell.

    Demand, stage request, final software command and physics velocity have
    different meanings. Ground truth never authorizes an online transition.
    """
    tracked = [r for r in rows if r["phase"] == "track" and r["time_sec"] < withdraw_sec]
    cap = staged.tracking_overlay_cap_nm
    peak = lambda values: max(values, default=None)
    cap_clipped = sum(abs(r["motion"]["requested_torque_nm"]) > cap + 1e-9 for r in tracked)
    stage_differs = sum(abs(r["overlay_nm"] - max(-cap, min(cap, r["motion"]["requested_torque_nm"]))) > 1e-9 for r in tracked)
    final_differs = sum(abs(r["dispatch"]["conditioned_overlay_nm"][2] - r["overlay_nm"]) > 1e-9 for r in tracked)
    at_position = []
    raw_only = []
    window_blocks = 0
    gate_reasons = Counter()
    previous = None
    first_raw_only = None
    for row in rows:
        if row["time_sec"] >= withdraw_sec:
            break
        if row["phase"] in {"track", "brake", "soft_hold"} and abs(row["target_error_rad"]) <= staged.settle_error_rad:
            at_position.append(row)
            raw = row["raw_qvel_rad_s"][2]
            window = row["control_qvel_rad_s"][2]
            if abs(window) > staged.settle_velocity_rad_s:
                window_blocks += 1
            elif abs(raw) > staged.settle_velocity_rad_s:
                raw_only.append(row)
                if first_raw_only is None:
                    step = None if previous is None else row["q3_rad"] - previous["q3_rad"]
                    first_raw_only = {
                        "time_sec": row["time_sec"], "raw_velocity_rad_s": raw,
                        "window_velocity_rad_s": window,
                        "physics_velocity_rad_s": row["physics_qvel_at_host_rad_s"][2],
                        "position_step_rad": step,
                        "position_step_counts": None if not position_quantum_rad or step is None else step / position_quantum_rad,
                    }
            frame = row["motion"]
            if frame is not None and frame["stop_capture_block_reason"] is not None:
                gate_reasons[frame["stop_capture_block_reason"]] += 1
        previous = row
    return {
        "available": True,
        "semantics": "Observer-only fixture summary. Software demand/request/final command are not measured output force; physics velocity is simulation truth, not an alternative online gate.",
        "tracking": {
            "host_samples": len(tracked), "cap_nm": cap,
            "demand_peak_nm": peak(abs(r["motion"]["requested_torque_nm"]) for r in tracked),
            "stage_request_peak_nm": peak(abs(r["overlay_nm"]) for r in tracked),
            "final_command_delta_peak_nm": peak(abs(r["dispatch"]["conditioned_overlay_nm"][2]) for r in tracked),
            "demand_cap_exceeded_samples": cap_clipped,
            "demand_cap_exceeded_fraction": None if not tracked else cap_clipped / len(tracked),
            "stage_request_differs_from_capped_demand_samples": stage_differs,
            "final_command_differs_from_stage_request_samples": final_differs,
        },
        "arrival_stop": {
            "first_position_tolerance_sec": None if not at_position else at_position[0]["time_sec"],
            "position_tolerance_host_samples": len(at_position),
            "selected_raw_only_velocity_block_samples": len(raw_only),
            "selected_window_velocity_block_samples": window_blocks,
            "first_raw_only_velocity_block": first_raw_only,
            "last_raw_only_velocity_block_sec": None if not raw_only else raw_only[-1]["time_sec"],
            "maximum_physics_velocity_at_raw_only_block_rad_s": peak(abs(r["physics_qvel_at_host_rad_s"][2]) for r in raw_only),
            "runtime_stop_capture_gate_reason_counts": dict(gate_reasons),
        },
    }


def run_b601_joint_motion_demo(
    *,
    source_root: str | Path,
    duration_sec: float = 6.0,
    viewer: bool = False,
    host_hz: float = 50.0,
    synthetic_friction_nm: float = 0.06,
    target_offset_rad: float = 0.02,
    scenario: str = "lifecycle",
    motion_config: B601JointMotionConfig | None = None,
    breakaway_rise_slew_nm_s: float = 2.0,
    driver_kd_nm_s_per_rad: tuple[float, ...] | None = None,
    feedback_position_quantum_rad: float = 0.0,
    motion_onset_displacement_rad: float = 0.0007,
    synthetic_static_friction_nm: float | None = None,
    friction_transition_velocity_rad_s: float = 0.02,
    breakaway_torque_nm: float = 0.2,
    breakaway_max_duration_sec: float = 0.2,
    breakaway_max_impulse_nm_sec: float = 0.04,
    total_overlay_cap_nm: float = 0.3,
    initial_arm_qpos_rad: tuple[float, ...] | None = None,
    motion_onset_confirm_displacement_rad: float | None = None,
    motion_onset_confirm_samples: int = 1,
    continuous_motion_handoff: bool = False,
    tracking_overlay_cap_nm: float = 0.1,
    capture_overlay_cap_nm: float = 0.0,
    scripted_withdrawal_sec: float = 5.0,
    tracking_recovery_gain_nm_per_rad: float | None = None,
    breakaway_probe: B601OnsetProbeConfig | None = None,
) -> tuple[int, dict[str, object]]:
    if scenario not in JOINT_MOTION_SCENARIOS:
        raise ValueError(f"unknown joint-motion scenario: {scenario}")
    if isinstance(feedback_position_quantum_rad, bool) or not math.isfinite(feedback_position_quantum_rad) or feedback_position_quantum_rad < 0:
        raise ValueError("feedback position quantum must be finite and non-negative")
    if isinstance(motion_onset_displacement_rad, bool) or not math.isfinite(motion_onset_displacement_rad) or motion_onset_displacement_rad <= 0:
        raise ValueError("motion onset displacement must be finite and positive")
    if not math.isfinite(duration_sec) or duration_sec < 6.0:
        raise ValueError("joint-motion duration must be at least 6 seconds")
    if (isinstance(scripted_withdrawal_sec, bool) or not math.isfinite(scripted_withdrawal_sec)
            or not 0.6 < scripted_withdrawal_sec < duration_sec):
        raise ValueError("scripted withdrawal must be after authorization and before fixture end")
    if scenario != "target-arrival" and scripted_withdrawal_sec != 5.0:
        raise ValueError("non-default observation timing is target-arrival simulation only")
    if not math.isfinite(host_hz) or not 20 <= host_hz <= 100:
        raise ValueError("joint-motion fixture supports 20..100 Hz")
    if not math.isfinite(target_offset_rad) or not 0 < abs(target_offset_rad) <= 0.05:
        raise ValueError("fixture target offset must be nonzero and within .05 rad")
    if (
        isinstance(synthetic_friction_nm, bool)
        or not math.isfinite(synthetic_friction_nm)
        or synthetic_friction_nm < 0
    ):
        raise ValueError("synthetic friction must be finite and non-negative")
    static_friction = synthetic_friction_nm if synthetic_static_friction_nm is None else synthetic_static_friction_nm
    if isinstance(static_friction, bool) or not math.isfinite(static_friction) or static_friction < synthetic_friction_nm:
        raise ValueError("synthetic static friction must be finite and at least kinetic friction")
    if isinstance(friction_transition_velocity_rad_s, bool) or not math.isfinite(friction_transition_velocity_rad_s) or friction_transition_velocity_rad_s <= 0:
        raise ValueError("friction transition velocity must be finite and positive")
    if (
        isinstance(breakaway_rise_slew_nm_s, bool)
        or not math.isfinite(breakaway_rise_slew_nm_s)
        or breakaway_rise_slew_nm_s <= 0
    ):
        raise ValueError("breakaway rise slew must be finite and positive")
    motion = B601JointMotionConfig() if motion_config is None else motion_config
    if not isinstance(motion, B601JointMotionConfig):
        raise TypeError("motion_config must be B601JointMotionConfig")
    if motion.soft_velocity_rad_s >= 0.05:
        raise ValueError(
            "fixture soft velocity must remain below .05 rad/s hard acceptance"
        )
    if any(motion.drive_driver_kd_nm_s_per_rad) and host_hz < motion.minimum_owner_rate_hz:
        raise ValueError("host rate is below the motion profile minimum")
    shared = B601SharedControlConfig(
        position_kp_nm_per_rad=(0, 0, motion.tracking_kp_nm_per_rad, 0, 0, 0),
        velocity_kd_nm_s_per_rad=(0, 0, motion.tracking_kd_nm_s_per_rad, 0, 0, 0),
        max_target_error_rad=(0.05,) * 6,
        max_abs_assist_torque_nm=(total_overlay_cap_nm,) * 6,
        max_assist_torque_rate_nm_s=(2.0,) * 6,
        authority_rise_per_sec=3.0,
        policy_stale_timeout_sec=0.15,
    )
    staged = B601StagedAssistConfig(
        joint_name="joint3",
        cue_torque_nm=0.04,
        cue_duration_sec=0.3,
        cue_rise_slew_nm_s=0.15,
        breakaway_torque_nm=breakaway_torque_nm,
        breakaway_rise_slew_nm_s=breakaway_rise_slew_nm_s,
        breakaway_max_duration_sec=breakaway_max_duration_sec,
        breakaway_max_impulse_nm_sec=breakaway_max_impulse_nm_sec,
        minimum_breakaway_error_rad=0.002,
        stuck_max_velocity_rad_s=0.02,
        motion_onset_displacement_rad=motion_onset_displacement_rad,
        motion_onset_velocity_rad_s=0.025,
        motion_onset_confirm_samples=motion_onset_confirm_samples,
        motion_onset_confirm_displacement_rad=motion_onset_confirm_displacement_rad,
        continuous_motion_handoff=continuous_motion_handoff,
        capture_overlay_cap_nm=capture_overlay_cap_nm,
        new_intent_delta_rad=0.01,
        tracking_overlay_cap_nm=tracking_overlay_cap_nm,
        tracking_slew_nm_s=0.15,
        settle_error_rad=0.003,
        settle_capture_error_rad=0.015,
        settle_velocity_rad_s=0.005,
        settle_dwell_sec=0.15,
        settle_integral_gain_nm_per_rad_sec=0.5,
        settle_integral_leak_per_sec=1,
        settle_integral_cap_nm=0.02,
        settle_overlay_cap_nm=0.05,
        max_total_overlay_nm=total_overlay_cap_nm,
        motion_control=motion,
        tracking_recovery_gain_nm_per_rad=tracking_recovery_gain_nm_per_rad,
        breakaway_probe=breakaway_probe,
    )
    runtime = B601StagedSingleJointAssistRuntime(shared, staged)
    controller = B601Real2SimControllerProfile.raw_model_baseline()
    if driver_kd_nm_s_per_rad is not None:
        controller = replace(controller, driver_kd_nm_s_per_rad=driver_kd_nm_s_per_rad)
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    plant = B601Real2SimPlantProfile(
        name="j3_motion_synthetic_friction",
        parameter_source="simulation_only_not_hardware_identified",
        frictionloss_nm=(0, 0, static_friction, 0, 0, 0),
    )
    initial_q = B601_RS_DEFAULT_ARM_QPOS if initial_arm_qpos_rad is None else tuple(initial_arm_qpos_rad)
    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=initial_q,
        gravity_mps2=(0.0, 0.0, -9.81),
        actuation_mode="effort",
        plant_profile=plant,
    )
    profile = replace(
        default_b601_mit_profile(scene),
        damping_kd_nm_s_per_rad=(0.0,) * 6,
        max_abs_overlay_torque_nm=(total_overlay_cap_nm,) * 6,
    )
    driver = B601MujocoEffortDriver(
        scene,
        model,
        timing=TimingSpec(
            command_hz=host_hz,
            physics_hz=B601_RS_PHYSICS_HZ,
            parameter_source="j3_motion_offline_fixture",
        ),
        profile=profile,
        enable_gravity_compensation=True,
        plant_profile=plant,
        real2sim_profile=controller,
    )
    driver.reset(initial_q)
    estimators = tuple(B601WindowVelocityEstimator(motion.velocity_window_sec) for _ in range(6))
    arm_addresses = tuple(scene.joint_dof_addr(name) for name in model.arm_joint_names)
    viewer_session = MujocoPassiveViewerSession(scene) if viewer else None
    sequence, granted, stopped = 0, False, False
    def feedback_position(q):
        return q if feedback_position_quantum_rad == 0 else tuple(
            round(value / feedback_position_quantum_rad) * feedback_position_quantum_rad for value in q
        )
    initial = feedback_position(_arm_state(scene)[0])
    target = tuple(
        q + (target_offset_rad if i == 2 else 0) for i, q in enumerate(initial)
    )
    previous_q, previous_t = None, None
    intent_finished = False
    rows, phases = [], set()
    peak_overlay, peak_velocity, hold_drift, hold_point = 0.0, 0.0, 0.0, None
    fault, driver_fault, final = None, None, None
    peak_all_velocity, support_drift = 0.0, 0.0
    peak_physics_velocity, first_physics_violation = 0.0, None
    physics_velocity_samples = 0
    window_ticks = round(_PHYSICS_WINDOW_SEC * B601_RS_PHYSICS_HZ)
    physics_history: deque[dict[str, object]] = deque(maxlen=window_ticks + 1)
    physics_window: list[dict[str, object]] = []
    capture_end_sec = None
    max_force_balance_error = 0.0
    interrupted, viewer_teardown_error = False, None
    try:
        if viewer_session is not None:
            viewer_session.start()
        for step in range(math.ceil(duration_sec * B601_RS_PHYSICS_HZ)):
            reference, frame = None, None
            if driver.command_due:
                now = driver.physics_time_sec
                physics_q, actual_v = _arm_state(scene)
                q = feedback_position(physics_q)
                dt = 1 / host_hz if previous_t is None else now - previous_t
                raw_v = (
                    actual_v
                    if previous_q is None
                    else tuple((x - p) / dt for x, p in zip(q, previous_q))
                )
                control_v = list(raw_v)
                for i in range(6) if motion.windowed_soft_velocity else (2,):
                    control_v[i] = estimators[i].update(
                        timestamp_sec=now, qpos_rad=q[i], fallback_rad_s=raw_v[i]
                    )
                previous_q, previous_t = q, now
                peak_all_velocity = max(peak_all_velocity, *(abs(v) for v in raw_v))
                support_drift = max(
                    support_drift, *(abs(q[i] - initial[i]) for i in (0, 1, 3, 4, 5))
                )
                sequence += 1
                # Scripted intent is simulation-only. The hardware GUI still
                # requires explicit preparation and one-shot authorization.
                if now >= 0.1 and now < scripted_withdrawal_sec and not intent_finished:
                    runtime.publish_policy(
                        B601PolicyAssistCommand(
                            sequence=sequence,
                            source_timestamp_sec=now,
                            receive_timestamp_sec=now,
                            qpos_target_rad=target,
                        )
                    )
                if now >= 0.6 and not granted:
                    granted = runtime.authorize_breakaway(sequence=sequence)
                if scenario == "lifecycle" and now >= 3.0 and not stopped:
                    runtime.request_stop_and_hold()
                    stopped = True
                if now >= scripted_withdrawal_sec:
                    runtime.set_policy_clutch(True)
                frame = runtime.step(
                    control_sequence=sequence,
                    control_timestamp_sec=now,
                    measured_qpos_rad=q,
                    measured_qvel_rad_s=raw_v,
                    control_qvel_rad_s=control_v,
                    control_qvel_window_span_sec=min(e.span_sec for e in estimators) if motion.windowed_soft_velocity else None,
                    dt_sec=dt,
                    tracking_recovery_permitted=tracking_recovery_gain_nm_per_rad is not None,
                    stop_evidence=None if motion.position_resolution_rad is None else B601WindowStopEvidence(
                        timestamp_sec=now, qpos_rad=q, raw_qvel_rad_s=tuple(raw_v),
                        window_qvel_rad_s=tuple(control_v),
                        position_span_rad=tuple(e.position_span_rad for e in estimators),
                        window_span_sec=min(e.span_sec for e in estimators),
                        raw_velocity_limit_rad_s=.05, joint_index=2),
                )
                if frame.release_failed:
                    fault = fault or f"staged_exit_failed:{frame.reason}"
                reference = frame.reference
                if any(motion.drive_driver_kd_nm_s_per_rad) and not frame.release_restoring_base_damping:
                    # Change only the same drive's MIT-local damping setting.
                    # Keep gravity and passive/native baseline separately auditable.
                    driver.real2sim_profile = replace(
                        controller,
                        driver_kd_nm_s_per_rad=tuple(
                            max(base, active) for base, active in zip(
                                controller.driver_kd_nm_s_per_rad,
                                frame.drive_driver_kd_nm_s_per_rad,
                            )
                        ),
                    )
                phases.add(frame.phase.value)
                peak_overlay = max(
                    peak_overlay, max(abs(x) for x in frame.overlay_torque_nm)
                )
                peak_velocity = max(peak_velocity, abs(raw_v[2]))
                if (
                    frame.motion is not None
                    and frame.motion.hold_position_rad is not None
                ):
                    hold_point = frame.motion.hold_position_rad
                    hold_drift = max(hold_drift, abs(q[2] - hold_point))
                rows.append(
                    {
                        "time_sec": now,
                        "phase": frame.phase.value,
                        "reason": frame.reason,
                        "q3_rad": q[2],
                        # Same pre-step clock, observer only. Never feed plant
                        # truth back into the quantized controller path.
                        "physics_qpos_at_host_rad": physics_q,
                        "physics_qvel_at_host_rad_s": actual_v,
                        "raw_v3_rad_s": raw_v[2],
                        "overlay_nm": frame.overlay_torque_nm[2],
                        "qpos_rad": q,
                        "raw_qvel_rad_s": raw_v,
                        "control_qvel_rad_s": tuple(control_v),
                        "drive_driver_kd_nm_s_per_rad": frame.drive_driver_kd_nm_s_per_rad,
                        "support_max_velocity_rad_s": frame.support_max_velocity_rad_s,
                        "support_raw_max_velocity_rad_s": frame.support_raw_max_velocity_rad_s,
                        "target_error_rad": target[2] - q[2],
                        "release_terminal": frame.release_terminal,
                        "release_failed": frame.release_failed,
                        "release_restoring_base_damping": frame.release_restoring_base_damping,
                        "motion_onset_confirm_count": frame.motion_onset_confirm_count,
                        "motion_onset_required_samples": frame.motion_onset_required_samples,
                        "breakaway_impulse_nm_sec": frame.breakaway_impulse_nm_sec,
                        "conditioned_command_impulse_nm_sec": frame.conditioned_command_impulse_nm_sec,
                        "motion": None
                        if frame.motion is None
                        else asdict(frame.motion),
                        "tracking_recovery": frame.tracking_recovery,
                    }
                )
            friction_limit = _synthetic_friction_limit(
                static_nm=static_friction, kinetic_nm=synthetic_friction_nm,
                velocity_rad_s=float(scene.data.qvel[arm_addresses[2]]),
                transition_rad_s=friction_transition_velocity_rad_s,
            )
            scene.model.dof_frictionloss[arm_addresses[2]] = friction_limit
            final = driver.step_physics(
                reference,
                restore_driver_kd_nm_s_per_rad=(
                    controller.driver_kd_nm_s_per_rad
                    if frame is not None and frame.release_restoring_base_damping else None
                ),
            )
            if final.plant_advanced:
                # Physics ground truth must not be inferred from a host-period
                # finite difference, which averages away short velocity peaks.
                physics_peak = max(abs(v) for v in final.measured_qvel)
                physics_velocity_samples += 1
                peak_physics_velocity = max(peak_physics_velocity, physics_peak)
                # Bounded observer only: do not advance physics or evaluate a
                # new control frame while exposing the between-host-tick event.
                physics_row = {
                    "force_state_time_sec": final.physics_time_sec,
                    "measured_time_sec": final.measured_time_sec,
                    "host_time_sec": rows[-1]["time_sec"],
                    "host_phase": rows[-1]["phase"],
                    "host_raw_qvel_rad_s": rows[-1]["raw_qvel_rad_s"],
                    "host_control_qvel_rad_s": rows[-1]["control_qvel_rad_s"],
                    "physics_qpos_rad": final.measured_qpos,
                    "physics_qvel_rad_s": final.measured_qvel,
                    "software_overlay_nm": final.torque_breakdown.overlay_effective_nm,
                    "motor_local_damping_nm": final.driver_damping_torque_nm,
                    "synthetic_friction_limit_nm": friction_limit,
                }
                physics_history.append(physics_row)
                if first_physics_violation is None and physics_peak >= 0.05:
                    index = max(range(6), key=lambda i: abs(final.measured_qvel[i]))
                    first_physics_violation = {
                        "time_sec": final.measured_time_sec,
                        "phase": frame.phase.value if frame is not None else rows[-1]["phase"],
                        "joint_name": model.arm_joint_names[index],
                        "velocity_rad_s": final.measured_qvel[index],
                        "limit_rad_s": 0.05,
                        "dynamics": _physics_dynamics_snapshot(scene, model, final),
                    }
                    physics_window = list(physics_history)
                    capture_end_sec = final.measured_time_sec + _PHYSICS_WINDOW_SEC
                    max_force_balance_error = max(
                        max_force_balance_error,
                        first_physics_violation["dynamics"]["force_balance_max_residual_nm"],
                    )
                elif capture_end_sec is not None and final.measured_time_sec <= capture_end_sec + 1e-9:
                    physics_window.append(physics_row)
            if frame is not None:
                dynamics = _physics_dynamics_snapshot(scene, model, final) if final.plant_advanced else None
                rows[-1]["dynamics"] = dynamics
                if dynamics is not None:
                    max_force_balance_error = max(max_force_balance_error, dynamics["force_balance_max_residual_nm"])
                rows[-1]["dispatch"] = {
                    "synthetic_friction_limit_nm": friction_limit,
                    "sequence": sequence,
                    "command_applied": final.command_applied,
                    "rejection_reason": final.rejection_reason,
                    "plant_advanced": final.plant_advanced,
                    "physics_time_sec": final.physics_time_sec,
                    "feedback_time_sec": final.measured_time_sec,
                    "measured_qpos_rad": final.measured_qpos,
                    "torque_breakdown": None
                    if final.torque_breakdown is None
                    else asdict(final.torque_breakdown),
                    "applied_sim_torque_nm": final.applied_torque_nm,
                    "torque_limited": final.torque_limited,
                    "slew_limited": final.slew_limited,
                    "driver_kd_nm_s_per_rad": final.driver_kd_nm_s_per_rad,
                    "host_command_torque_nm": final.host_command_torque_nm,
                    "conditioned_overlay_nm": None if final.torque_breakdown is None else tuple(
                        sent - gravity for sent, gravity in zip(
                            final.host_command_torque_nm, final.torque_breakdown.gravity_torque_nm,
                        )
                    ),
                    # Simulator ground truth only: constraint forces include
                    # friction and other constraints, never a human-force label.
                    "sim_constraint_torque_nm": tuple(
                        float(scene.data.qfrc_constraint[i]) for i in arm_addresses
                    ),
                }
            if frame is not None and final.command_applied:
                runtime.record_dispatch(
                    sequence,
                    conditioned_torque_nm=final.host_command_torque_nm,
                    passive_torque_nm=final.torque_breakdown.gravity_torque_nm,
                    driver_kd_nm_s_per_rad=final.driver_kd_nm_s_per_rad,
                    torque_rate_nm_s=driver.profile.max_torque_rate_nm_s,
                    completed_at_sec=final.measured_time_sec,
                )
                # A confirmed release may commit DROP or PASSIVE only after
                # this final packet; preserve that transition, not just the
                # pre-dispatch requested phase. It is still software evidence.
                rows[-1]["phase_after_dispatch"] = runtime.phase.value
                rows[-1]["release_completed"] = runtime.release_completed
                phases.add(runtime.phase.value)
                if motion_onset_confirm_displacement_rad is not None and runtime.release_completed:
                    # One scripted trial, not a new policy proposal each time
                    # the previous exhausted/unconfirmed trial finishes.
                    intent_finished = True
            if final.driver_faulted:
                driver_fault = final.fault_reason
                fault = driver_fault
                break
            if viewer_session is not None:
                if step % 16 == 0 and not viewer_session.sync():
                    fault = "viewer_closed_before_acceptance"
                    break
                time.sleep(1 / B601_RS_PHYSICS_HZ)
    except KeyboardInterrupt:
        interrupted = True
        fault = "operator_interrupted"
    finally:
        runtime.clear_policy()
        driver.request_protective_hold("fixture_complete")
        if viewer_session is not None:
            try:
                viewer_session.close()
            except TimeoutError as error:
                viewer_teardown_error = str(error)
                fault = fault or "viewer_teardown_unconfirmed"
    displacement = max((abs(row["q3_rad"] - initial[2]) for row in rows), default=0.0)
    required = {"cue", "brake", "soft_hold", "human_override"}
    if static_friction > staged.cue_torque_nm:
        required |= {"capture", "track"} if staged.capture_overlay_cap_nm else {"drop", "track"}
    withdrawal = _withdrawal_evaluation(
        rows, withdraw_sec=scripted_withdrawal_sec, staged=staged,
        base_driver_kd=controller.driver_kd_nm_s_per_rad,
        torque_limits=driver.profile.max_abs_torque_nm, torque_rates=driver.profile.max_torque_rate_nm_s,
    )
    checks = {
        "no_fixture_fault": fault is None,
        "required_phases_present": required <= phases,
        "minimum_realized_motion": displacement >= 0.001,
        "overlay_peak_within_limit": peak_overlay <= staged.max_total_overlay_nm + 1e-9,
        "selected_joint_velocity_within_limit": peak_velocity < 0.05,
        "all_joint_velocity_within_limit": peak_all_velocity < 0.05,
        "physics_all_joint_velocity_within_limit": peak_physics_velocity < 0.05,
        "dynamics_force_balance_consistent": bool(rows)
        and all(row["dynamics"] is not None for row in rows)
        and max_force_balance_error < 1e-6,
        "support_drift_within_limit": support_drift <= math.radians(0.5),
        "stop_hold_drift_within_limit": hold_drift <= 0.003,
        **withdrawal["checks"],
        "complete_dispatch_and_physics": bool(rows)
        and physics_velocity_samples == math.ceil(duration_sec * B601_RS_PHYSICS_HZ)
        and all(
            row["dispatch"]["command_applied"] and row["dispatch"]["plant_advanced"]
            for row in rows
        ),
    }
    lifecycle_passed = all(checks.values())
    safety_passed = all(checks[k] for k in (
        "no_fixture_fault", "overlay_peak_within_limit", "selected_joint_velocity_within_limit",
        "all_joint_velocity_within_limit", "physics_all_joint_velocity_within_limit",
        "dynamics_force_balance_consistent", "support_drift_within_limit", "complete_dispatch_and_physics",
        "withdrawal_receipts_valid", "withdrawal_conditioned_torque_bounds", "withdrawal_damping_retained_during_tail",
    ))
    execution_passed = all(checks[k] for k in (
        "required_phases_present", "minimum_realized_motion", "stop_hold_drift_within_limit",
    ))
    # Arrival needs continuous stationary dwell near the original target,
    # before clutch withdrawal. Merely capturing another stop point is not arrival.
    first_brake = next((row for row in rows if row["phase"] == "brake"), None)
    arrival_confirmed = _target_arrival_time(
        rows,
        tolerance_rad=staged.settle_error_rad,
        velocity_rad_s=staged.settle_velocity_rad_s,
        dwell_sec=staged.settle_dwell_sec,
        withdraw_sec=scripted_withdrawal_sec,
    )
    target_arrived = arrival_confirmed is not None
    target_rows = [row for row in rows if 0.1 <= row["time_sec"] < scripted_withdrawal_sec]
    target_error = None if not target_rows else target_rows[-1]["target_error_rad"]
    first_velocity_violation = next(
        (
            {
                "time_sec": row["time_sec"],
                "phase": row["phase"],
                "reason": row["reason"],
                "joint_name": model.arm_joint_names[index],
                "raw_velocity_rad_s": velocity,
                "limit_rad_s": 0.05,
            }
            for row in rows
            for index, velocity in enumerate(row["raw_qvel_rad_s"])
            if abs(velocity) >= 0.05
        ),
        None,
    )
    acceptance_kind = "stop_lifecycle" if scenario == "lifecycle" else "target_arrival"
    if scenario == "target-arrival":
        checks["stationary_at_original_target"] = target_arrived
    passed = lifecycle_passed and (scenario == "lifecycle" or target_arrived)
    if not passed and fault is None:
        fault = (
            "target_not_reached_within_existing_envelope"
            if lifecycle_passed and not target_arrived
            else "lifecycle_acceptance_failed"
        )
    return (130 if interrupted else 0 if passed else 3), {
        "status": "INTERRUPTED" if interrupted else "PASS" if passed else "FAIL",
        "fixture": JOINT_MOTION_FIXTURE,
        "scenario": scenario,
        "simulation_acceptance": {
            "kind": acceptance_kind,
            "checks": checks,
            "failed_checks": [name for name, value in checks.items() if not value],
            "passed": passed,
            "stop_lifecycle_passed": lifecycle_passed,
            "safety_passed": safety_passed,
            "withdrawal_passed": withdrawal["passed"],
            "motion_execution_passed": execution_passed,
            "failure_dimensions": [name for name, ok in (
                ("safety", safety_passed), ("protected_withdrawal", withdrawal["passed"]),
                ("motion_execution", execution_passed),
                ("target_arrival", scenario != "target-arrival" or target_arrived),
            ) if not ok],
            "unassessed_dimensions": ["human_interaction"],
            "hardware_claim": False,
            "driver_fault": driver_fault,
            "failure_reason": fault,
            "phases": sorted(phases),
            "required_phases": sorted(required),
            "peak_overlay_nm": peak_overlay,
            "peak_raw_velocity_rad_s": peak_velocity,
            "peak_all_joint_raw_velocity_rad_s": peak_all_velocity,
            "first_velocity_violation": first_velocity_violation,
            "peak_physics_all_joint_velocity_rad_s": peak_physics_velocity,
            "physics_velocity_sample_count": physics_velocity_samples,
            "physics_velocity_sample_hz": B601_RS_PHYSICS_HZ,
            "first_physics_velocity_violation": first_physics_violation,
            "maximum_force_balance_residual_nm": max_force_balance_error,
            "maximum_support_drift_rad": support_drift,
            "peak_displacement_rad": displacement,
            "hold_position_rad": hold_point,
            "maximum_hold_drift_rad": hold_drift,
        },
        "synthetic_joint3_friction_nm": synthetic_friction_nm,
        "feedback_position_quantum_rad": feedback_position_quantum_rad,
        "initial_arm_qpos_rad": initial_q,
        "physics_velocity_event_window": {
            "sampling_hz": B601_RS_PHYSICS_HZ,
            "pre_window_sec": _PHYSICS_WINDOW_SEC,
            "post_window_sec": _PHYSICS_WINDOW_SEC,
            "maximum_samples": 2 * window_ticks + 1,
            "event_time_sec": None if first_physics_violation is None else first_physics_violation["time_sec"],
            "host_violation_delay_sec": None if first_physics_violation is None or first_velocity_violation is None else first_velocity_violation["time_sec"] - first_physics_violation["time_sec"],
            "host_violation_delay_semantics": "Signed first raw-host threshold time minus first physics threshold time; not hardware transport latency. Quantized raw estimates can cross earlier or never cross.",
            "post_window_complete": None if capture_end_sec is None else bool(physics_window) and physics_window[-1]["measured_time_sec"] >= capture_end_sec - 1e-9,
            "parameter_source": "simulation_ground_truth_not_hardware_latency_or_force",
            "samples": physics_window,
        },
        "target_offset_rad": target_offset_rad,
        "target_evaluation": {
            "arrived": target_arrived,
            "arrival_confirmed_time_sec": arrival_confirmed,
            "position_tolerance_rad": staged.settle_error_rad,
            "stationary_tolerance_rad_s": staged.settle_velocity_rad_s,
            "stationary_dwell_sec": staged.settle_dwell_sec,
            "final_error_before_withdraw_rad": target_error,
            "first_brake": first_brake,
        },
        "withdrawal_evaluation": withdrawal,
        "scripted_withdrawal_time_sec": scripted_withdrawal_sec,
        "motion_diagnostics": _motion_diagnostics(
            rows, staged=staged, withdraw_sec=scripted_withdrawal_sec,
            position_quantum_rad=feedback_position_quantum_rad,
        ) if withdrawal["checks"]["withdrawal_receipts_valid"] else {
            "available": False, "reason": "invalid_or_incomplete_dispatch_trace",
        },
        "human_interaction_evaluation": {
            "assessed": False,
            "explicit_clutch_withdrawal_passed": withdrawal["passed"],
            "human_force_measured": False,
            "semantics": "Only scripted policy-clutch cancellation is exercised. No gripping, human force, perceived cue or automatic override detection acceptance.",
        },
        "declared_model_diagnostics": {
            "parameter_source": "simulation_only_not_hardware_identified",
            "motion_config_source": "default"
            if motion_config is None
            else "explicit_offline_counterfactual",
            "driver_kd_nm_s_per_rad": controller.driver_kd_nm_s_per_rad,
            "driver_damping_source": "raw_model_baseline"
            if driver_kd_nm_s_per_rad is None
            else "explicit_offline_counterfactual",
            "controller_profile": asdict(controller),
            "cue_peak_nm": staged.cue_torque_nm,
            "breakaway_peak_nm": staged.breakaway_torque_nm,
            "motion_onset_displacement_rad": staged.motion_onset_displacement_rad,
            "continuous_motion_handoff": staged.continuous_motion_handoff,
            "capture_overlay_cap_nm": staged.capture_overlay_cap_nm,
            "scripted_trial_republishes_after_terminal_release": motion_onset_confirm_displacement_rad is None,
            "breakaway_rise_slew_nm_s": staged.breakaway_rise_slew_nm_s,
            "breakaway_max_duration_sec": staged.breakaway_max_duration_sec,
            "breakaway_max_impulse_nm_sec": staged.breakaway_max_impulse_nm_sec,
            "breakaway_command_envelope": staged.breakaway_envelope(control_period_sec=1 / host_hz),
            "tracking_overlay_cap_nm": staged.tracking_overlay_cap_nm,
            "total_overlay_cap_nm": staged.max_total_overlay_nm,
            "final_host_torque_limits_nm": driver.profile.max_abs_torque_nm,
            "final_host_torque_rates_nm_s": driver.profile.max_torque_rate_nm_s,
            "proportional_torque_at_lag_limit_nm": (
                motion.tracking_kp_nm_per_rad * motion.max_reference_lag_rad
            ),
            "proportional_torque_at_arrival_tolerance_nm": (
                motion.tracking_kp_nm_per_rad * staged.settle_error_rad
            ),
            "synthetic_frictionloss_nm": synthetic_friction_nm,
            "synthetic_static_friction_nm": static_friction,
            "friction_transition_velocity_rad_s": friction_transition_velocity_rad_s,
            "friction_model": "constant_dry_friction" if static_friction == synthetic_friction_nm else "declared_stribeck_like_limit",
            "hard_velocity_acceptance_rad_s": 0.05,
            "support_drift_acceptance_rad": math.radians(0.5),
            "human_force_measured": False,
        },
        "target_arrival_claim": scenario == "target-arrival" and passed,
        "viewer_teardown_error": viewer_teardown_error,
        "motion_profile": asdict(motion),
        "samples": rows,
        "evidence_limit": "Selected simulation-only acceptance on URDF plant with declared friction constraints and optional velocity-dependent bounds. Solver regularization can allow creep; frictionloss is not a measured breakaway threshold. Not identified gearbox stiction, human-force measurement or hardware acceptance. A stopped pose is not necessarily the original target.",
    }
