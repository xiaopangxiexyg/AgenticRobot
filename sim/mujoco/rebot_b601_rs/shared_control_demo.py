"""Deterministic shared-control lifecycle fixture for the B601 MuJoCo plant."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601PinocchioModel,
    B601PolicyAssistCommand,
    B601SharedControlConfig,
    B601SharedControlMode,
    B601SharedControlRuntime,
)
from sim.contracts import TimingSpec
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .mit_driver import B601MujocoEffortDriver, default_b601_mit_profile
from .real2sim import B601Real2SimControllerProfile
from .scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_PHYSICS_HZ,
    build_b601_mujoco_scene,
)

SHARED_CONTROL_FIXTURE = "policy-assist"
DEFAULT_SHARED_CONTROL_DURATION_SEC = 1.5
DEFAULT_SHARED_CONTROL_HOST_HZ = 100.0


def default_b601_shared_control_config() -> B601SharedControlConfig:
    """Return conservative simulation-only policy-assist parameters."""

    return B601SharedControlConfig(
        position_kp_nm_per_rad=(2.0,) * 6,
        velocity_kd_nm_s_per_rad=(0.0,) * 6,
        max_target_error_rad=(0.04,) * 6,
        max_abs_assist_torque_nm=(0.10,) * 6,
        max_assist_torque_rate_nm_s=(0.50,) * 6,
        authority_rise_per_sec=5.0,
        policy_stale_timeout_sec=0.08,
        maximum_future_skew_sec=0.005,
    )


def _arm_state(scene) -> tuple[tuple[float, ...], tuple[float, ...]]:
    qpos = tuple(
        float(scene.data.qpos[scene.joint_qpos_addr(name)])
        for name in B601_RS_ARM_JOINT_NAMES
    )
    qvel = tuple(
        float(scene.data.qvel[scene.joint_dof_addr(name)])
        for name in B601_RS_ARM_JOINT_NAMES
    )
    return qpos, qvel


def _phase(elapsed_sec: float) -> str:
    if elapsed_sec < 0.20:
        return "passive"
    if elapsed_sec < 0.65:
        return "policy_assist"
    if elapsed_sec < 0.85:
        return "human_override"
    if elapsed_sec < 1.00:
        return "await_rebased_policy"
    if elapsed_sec < 1.35:
        return "recovery"
    return "policy_stale"


def _policy_target(
    qpos: Sequence[float],
    *,
    phase: str,
    elapsed_sec: float,
) -> tuple[float, ...] | None:
    if phase == "policy_assist":
        target = list(qpos)
        target[5] += 0.04
        return tuple(target)
    if phase == "recovery":
        # The first post-clutch sample is exactly current state.  Subsequent
        # references introduce only a small local increment, never the old
        # pre-clutch target.
        progress = min(1.0, max(0.0, (elapsed_sec - 1.00) / 0.20))
        target = list(qpos)
        target[5] -= 0.025 * progress
        return tuple(target)
    return None


def run_b601_shared_control_demo(
    *,
    source_root: str | Path,
    duration_sec: float = DEFAULT_SHARED_CONTROL_DURATION_SEC,
    viewer: bool = False,
    host_hz: float = DEFAULT_SHARED_CONTROL_HOST_HZ,
) -> tuple[int, dict[str, object]]:
    """Exercise assist, clutch, rebase and stale fallback on the existing plant."""

    duration = float(duration_sec)
    rate = float(host_hz)
    if not math.isfinite(duration) or duration < DEFAULT_SHARED_CONTROL_DURATION_SEC:
        raise ValueError(
            f"duration_sec must be at least {DEFAULT_SHARED_CONTROL_DURATION_SEC}"
        )
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("host_hz must be finite and positive")

    initial_qpos = tuple(B601_RS_DEFAULT_ARM_QPOS)
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=initial_qpos,
        gravity_mps2=(0.0, 0.0, -9.81),
        actuation_mode="effort",
    )
    base_profile = default_b601_mit_profile(scene)
    assist_config = default_b601_shared_control_config()
    profile = replace(
        base_profile,
        damping_kd_nm_s_per_rad=(0.0,) * 6,
        max_abs_overlay_torque_nm=assist_config.max_abs_assist_torque_nm,
        max_abs_state_acceleration_rad_s2=(250.0,) * 6,
        metadata={
            **base_profile.metadata,
            "fixture_demo": SHARED_CONTROL_FIXTURE,
            "hardware_safety_claim": False,
        },
    )
    driver = B601MujocoEffortDriver(
        scene,
        model,
        timing=TimingSpec(
            command_hz=rate,
            physics_hz=B601_RS_PHYSICS_HZ,
            parameter_source="shared_control_simulation_fixture",
        ),
        profile=profile,
        enable_gravity_compensation=True,
        real2sim_profile=B601Real2SimControllerProfile(),
    )
    driver.reset(initial_qpos)
    runtime = B601SharedControlRuntime(assist_config)
    requested_steps = math.ceil(duration * B601_RS_PHYSICS_HZ)
    viewer_session = MujocoPassiveViewerSession(scene) if viewer else None
    if viewer_session is not None:
        viewer_session.start()

    sequence = 0
    policy_sequence = 0
    modes: list[str] = []
    phases: list[str] = []
    transition_log: list[dict[str, object]] = []
    maximum_assist_nm = 0.0
    maximum_override_assist_nm = 0.0
    maximum_await_assist_nm = 0.0
    applied_references = 0
    advanced_ticks = 0
    viewer_closed_early = False
    driver_fault_reason: str | None = None
    final_tick = None
    try:
        for step in range(requested_steps):
            reference = None
            if driver.command_due:
                now = driver.physics_time_sec
                phase = _phase(now)
                qpos, qvel = _arm_state(scene)
                runtime.set_policy_clutch(phase == "human_override")
                target = _policy_target(qpos, phase=phase, elapsed_sec=now)
                if target is not None:
                    policy_sequence += 1
                    runtime.publish_policy(
                        B601PolicyAssistCommand(
                            sequence=policy_sequence,
                            source_timestamp_sec=now,
                            receive_timestamp_sec=now,
                            qpos_target_rad=target,
                        )
                    )
                frame = runtime.step(
                    control_sequence=sequence,
                    control_timestamp_sec=now,
                    measured_qpos_rad=qpos,
                    measured_qvel_rad_s=qvel,
                    dt_sec=1.0 / rate,
                )
                reference = frame.reference
                sequence += 1
                assist = max(abs(value) for value in frame.decision.assist_torque_nm)
                maximum_assist_nm = max(maximum_assist_nm, assist)
                if phase == "human_override":
                    maximum_override_assist_nm = max(maximum_override_assist_nm, assist)
                if phase == "await_rebased_policy":
                    maximum_await_assist_nm = max(maximum_await_assist_nm, assist)
                mode = frame.decision.state.mode.value
                if not modes or modes[-1] != mode or phases[-1] != phase:
                    transition_log.append(
                        {
                            "time_sec": now,
                            "fixture_phase": phase,
                            "shared_control_mode": mode,
                            "reason": frame.decision.reason,
                        }
                    )
                modes.append(mode)
                phases.append(phase)

            final_tick = driver.step_physics(reference)
            applied_references += int(final_tick.command_applied)
            advanced_ticks += int(final_tick.plant_advanced)
            if final_tick.driver_faulted:
                driver_fault_reason = final_tick.fault_reason
                break
            if viewer_session is not None:
                if step % 16 == 0 and not viewer_session.sync():
                    viewer_closed_early = True
                    break
                time.sleep(1.0 / B601_RS_PHYSICS_HZ)
    finally:
        if viewer_session is not None:
            viewer_session.close()

    if final_tick is None:
        raise RuntimeError("shared-control fixture did not execute a physics tick")
    final_qpos = np.asarray(final_tick.measured_qpos, dtype=float)
    displacement = final_qpos - np.asarray(initial_qpos, dtype=float)
    required_modes = {
        B601SharedControlMode.PASSIVE.value,
        B601SharedControlMode.POLICY_ASSIST.value,
        B601SharedControlMode.HUMAN_OVERRIDE.value,
        B601SharedControlMode.RECOVERY.value,
    }
    accepted = bool(
        driver_fault_reason is None
        and advanced_ticks == requested_steps
        and not viewer_closed_early
        and required_modes.issubset(set(modes))
        and maximum_assist_nm <= max(assist_config.max_abs_assist_torque_nm) + 1e-12
        and maximum_override_assist_nm == 0.0
        and maximum_await_assist_nm == 0.0
        and modes
        and modes[-1] == B601SharedControlMode.PASSIVE.value
        and abs(displacement[5]) > 1e-4
    )
    payload = {
        "fixture": SHARED_CONTROL_FIXTURE,
        "simulation_acceptance": {
            "passed": accepted,
            "hardware_claim": False,
            "driver_fault_reason": driver_fault_reason,
            "requested_physics_ticks": requested_steps,
            "advanced_physics_ticks": advanced_ticks,
            "viewer_closed_early": viewer_closed_early,
            "applied_reference_count": applied_references,
            "required_modes": sorted(required_modes),
            "required_modes_observed": required_modes.issubset(set(modes)),
            "final_mode": modes[-1] if modes else None,
            "maximum_assist_torque_nm": maximum_assist_nm,
            "maximum_override_assist_torque_nm": maximum_override_assist_nm,
            "maximum_await_rebase_assist_torque_nm": maximum_await_assist_nm,
            "joint6_displacement_rad": float(displacement[5]),
        },
        "shared_control_config": {
            "position_kp_nm_per_rad": assist_config.position_kp_nm_per_rad,
            "velocity_kd_nm_s_per_rad": assist_config.velocity_kd_nm_s_per_rad,
            "max_target_error_rad": assist_config.max_target_error_rad,
            "max_abs_assist_torque_nm": assist_config.max_abs_assist_torque_nm,
            "max_assist_torque_rate_nm_s": assist_config.max_assist_torque_rate_nm_s,
            "authority_rise_per_sec": assist_config.authority_rise_per_sec,
            "policy_stale_timeout_sec": assist_config.policy_stale_timeout_sec,
        },
        "transitions": transition_log,
        "initial_qpos_rad": initial_qpos,
        "final_qpos_rad": tuple(float(value) for value in final_qpos),
        "driver": driver.execution_metadata,
        "fidelity": (
            "Deterministic simulation-only shared-control lifecycle fixture on the "
            "existing unidentified ideal-effort B601 plant. It validates command "
            "freshness, authority, clutch/rebase semantics, bounded overlay and "
            "qualitative motion only; it does not validate hardware force, timing, "
            "human feel, passivity or safety."
        ),
    }
    return (0 if accepted else 3), payload


__all__ = [
    "DEFAULT_SHARED_CONTROL_DURATION_SEC",
    "DEFAULT_SHARED_CONTROL_HOST_HZ",
    "SHARED_CONTROL_FIXTURE",
    "default_b601_shared_control_config",
    "run_b601_shared_control_demo",
]
