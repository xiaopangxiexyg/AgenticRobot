"""Headless-first B601 virtual-spring and virtual-wall acceptance demo."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601_RS_ARM_JOINT_NAMES,
    B601MitReference,
    B601PinocchioModel,
    B601TranslationalVirtualSpring,
    B601VirtualWall,
    map_world_aligned_ee_wrench_to_joint_torque,
)
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .mit_driver import B601MujocoEffortDriver, default_b601_mit_profile
from .scene import B601_RS_PHYSICS_HZ, build_b601_mujoco_scene

DEFAULT_HAPTIC_ANCHOR_QPOS = (0.15, 1.0, 1.1, 0.1, -0.15, 0.2)
DEFAULT_HAPTIC_INITIAL_QPOS = (0.23, 0.94, 1.15, 0.1, -0.15, 0.2)
DEFAULT_HAPTIC_DURATION_SEC = 1.5
DEFAULT_SPRING_STIFFNESS_N_PER_M = (80.0, 80.0, 80.0)
DEFAULT_SPRING_DAMPING_N_S_PER_M = (8.0, 8.0, 8.0)
DEFAULT_MAX_FORCE_N = 8.0
DEFAULT_MAX_OVERLAY_TORQUE_NM = 6.0
DEFAULT_WALL_PENETRATION_M = 0.025


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic B601 virtual spring or wall through the "
            "ideal-effort MuJoCo backend. Headless is the default."
        )
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        required=True,
        help="Explicit root of the pinned reBotArm_control_py source snapshot.",
    )
    parser.add_argument(
        "--fixture",
        choices=("spring", "wall"),
        default="spring",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=DEFAULT_HAPTIC_DURATION_SEC,
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the passive MuJoCo viewer and pace physics in real time.",
    )
    return parser


def _arm_qvel(scene) -> np.ndarray:
    return np.asarray(
        [
            scene.data.qvel[scene.joint_dof_addr(name)]
            for name in B601_RS_ARM_JOINT_NAMES
        ],
        dtype=float,
    )


def _fixture_metrics(
    *,
    fixture_name: str,
    model: B601PinocchioModel,
    fixture,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> tuple[object, float]:
    pose = model.forward_kinematics(qpos)
    jacobian = model.arm_jacobian(qpos)
    linear_velocity = jacobian[:3] @ qvel
    result = fixture.evaluate(pose[:3, 3], linear_velocity)
    if fixture_name == "spring":
        metric = float(np.linalg.norm(result.displacement_m))
    else:
        metric = float(result.penetration_m)
    return result, metric


def run_haptic_fixture_demo(
    *,
    source_root: str | Path,
    fixture_name: str,
    duration_sec: float = DEFAULT_HAPTIC_DURATION_SEC,
    viewer: bool = False,
) -> tuple[int, dict[str, object]]:
    """Run one closed-loop fixture and return an exit code plus JSON payload."""

    if fixture_name not in {"spring", "wall"}:
        raise ValueError("fixture_name must be 'spring' or 'wall'")
    if not isinstance(viewer, bool):
        raise TypeError("viewer must be a bool")
    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")

    model = B601PinocchioModel.from_rebot_source_root(source_root)
    initial_qpos = np.asarray(DEFAULT_HAPTIC_INITIAL_QPOS, dtype=float)
    anchor_qpos = np.asarray(DEFAULT_HAPTIC_ANCHOR_QPOS, dtype=float)
    initial_position = model.forward_kinematics(initial_qpos)[:3, 3]
    anchor_position = model.forward_kinematics(anchor_qpos)[:3, 3]
    if fixture_name == "spring":
        fixture = B601TranslationalVirtualSpring(
            anchor_position_m=tuple(float(value) for value in anchor_position),
            stiffness_n_per_m=DEFAULT_SPRING_STIFFNESS_N_PER_M,
            damping_n_s_per_m=DEFAULT_SPRING_DAMPING_N_S_PER_M,
            max_abs_force_n=(DEFAULT_MAX_FORCE_N,) * 3,
        )
        fixture_contract: dict[str, object] = {
            "anchor_position_m": [float(value) for value in anchor_position],
            "stiffness_n_per_m": list(DEFAULT_SPRING_STIFFNESS_N_PER_M),
            "damping_n_s_per_m": list(DEFAULT_SPRING_DAMPING_N_S_PER_M),
            "max_abs_force_n": [DEFAULT_MAX_FORCE_N] * 3,
        }
    else:
        wall_normal = np.asarray((1.0, 0.0, 0.0), dtype=float)
        wall_offset = float(wall_normal @ initial_position) - DEFAULT_WALL_PENETRATION_M
        fixture = B601VirtualWall(
            normal=tuple(float(value) for value in wall_normal),
            offset_m=wall_offset,
            stiffness_n_per_m=180.0,
            damping_n_s_per_m=12.0,
            max_force_n=DEFAULT_MAX_FORCE_N,
        )
        fixture_contract = {
            "normal": list(fixture.normal),
            "offset_m": fixture.offset_m,
            "forbidden_half_space": "normal_dot_position_greater_than_offset",
            "stiffness_n_per_m": fixture.stiffness_n_per_m,
            "damping_n_s_per_m": fixture.damping_n_s_per_m,
            "max_force_n": fixture.max_force_n,
        }

    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=initial_qpos,
        gravity_mps2=(0.0, 0.0, -9.81),
        actuation_mode="effort",
    )
    base_profile = default_b601_mit_profile(scene)
    profile = replace(
        base_profile,
        max_abs_overlay_torque_nm=(DEFAULT_MAX_OVERLAY_TORQUE_NM,) * 6,
        metadata={
            **base_profile.metadata,
            "fixture_demo": fixture_name,
            "fixture_parameter_source": "pre_hardware_effect_acceptance_seed",
        },
    )
    driver = B601MujocoEffortDriver(
        scene,
        model,
        profile=profile,
        enable_gravity_compensation=True,
    )
    driver.reset(initial_qpos)

    _initial_result, initial_metric = _fixture_metrics(
        fixture_name=fixture_name,
        model=model,
        fixture=fixture,
        qpos=initial_qpos,
        qvel=np.zeros(6, dtype=float),
    )
    requested_steps = math.ceil(duration * B601_RS_PHYSICS_HZ)
    viewer_session = MujocoPassiveViewerSession(scene) if viewer else None
    if viewer_session is not None:
        viewer_session.start()

    next_sequence = 0
    command_count = 0
    advanced_ticks = 0
    fixture_limited_ticks = 0
    overlay_limited_ticks = 0
    torque_limited_ticks = 0
    slew_limited_ticks = 0
    peak_force_n = 0.0
    peak_requested_overlay_torque_nm = 0.0
    peak_applied_torque_nm = 0.0
    maximum_metric = initial_metric
    minimum_metric = initial_metric
    final_tick = None
    viewer_closed_early = False
    try:
        for step in range(requested_steps):
            reference = None
            if driver.command_due:
                qpos = scene.get_arm_qpos("b601_rs")
                qvel = _arm_qvel(scene)
                fixture_result, metric = _fixture_metrics(
                    fixture_name=fixture_name,
                    model=model,
                    fixture=fixture,
                    qpos=qpos,
                    qvel=qvel,
                )
                mapping = map_world_aligned_ee_wrench_to_joint_torque(
                    model,
                    qpos,
                    fixture_result.wrench,
                    qvel_rad_s=qvel,
                )
                timestamp = driver.physics_time_sec
                reference = B601MitReference(
                    sequence=next_sequence,
                    source_timestamp_sec=timestamp,
                    receive_timestamp_sec=timestamp,
                    qpos_rad=tuple(float(value) for value in qpos),
                    qvel_rad_s=(0.0,) * 6,
                    joint_torque_overlay_nm=mapping.joint_torque_nm,
                )
                next_sequence += 1
                command_count += 1
                maximum_metric = max(maximum_metric, metric)
                minimum_metric = min(minimum_metric, metric)
                peak_force_n = max(
                    peak_force_n,
                    float(np.linalg.norm(fixture_result.wrench.force_n)),
                )
                peak_requested_overlay_torque_nm = max(
                    peak_requested_overlay_torque_nm,
                    float(np.max(np.abs(mapping.joint_torque_nm))),
                )
                if fixture_result.force_limited:
                    fixture_limited_ticks += 1

            final_tick = driver.step_physics(reference)
            if final_tick.plant_advanced:
                advanced_ticks += 1
            if (
                final_tick.torque_breakdown is not None
                and final_tick.torque_breakdown.overlay_limited
            ):
                overlay_limited_ticks += 1
            torque_limited_ticks += int(final_tick.torque_limited)
            slew_limited_ticks += int(final_tick.slew_limited)
            peak_applied_torque_nm = max(
                peak_applied_torque_nm,
                float(np.max(np.abs(final_tick.applied_torque_nm))),
            )
            if final_tick.driver_faulted:
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
        raise RuntimeError("haptic fixture demo did not execute a physics tick")
    final_qpos = np.asarray(final_tick.measured_qpos, dtype=float)
    final_qvel = np.asarray(final_tick.measured_qvel, dtype=float)
    final_result, final_metric = _fixture_metrics(
        fixture_name=fixture_name,
        model=model,
        fixture=fixture,
        qpos=final_qpos,
        qvel=final_qvel,
    )
    maximum_metric = max(maximum_metric, final_metric)
    minimum_metric = min(minimum_metric, final_metric)
    run_completed = advanced_ticks == requested_steps and not viewer_closed_early
    common_pass = bool(
        run_completed
        and not final_tick.driver_faulted
        and final_tick.state_valid
        and final_tick.state_within_limits
        and torque_limited_ticks == 0
        and peak_applied_torque_nm <= max(profile.max_abs_torque_nm) + 1e-12
    )
    if fixture_name == "spring":
        effect_pass = bool(
            initial_metric > 0.02
            and final_metric <= 0.45 * initial_metric
            and maximum_metric <= 1.05 * initial_metric
        )
        effect_contract = {
            "metric": "cartesian_distance_to_anchor_m",
            "initial_minimum_m": 0.02,
            "final_over_initial_max": 0.45,
            "maximum_over_initial_max": 1.05,
        }
    else:
        effect_pass = bool(
            initial_metric >= 0.02
            and final_metric <= 0.002
            and maximum_metric <= initial_metric + 0.002
        )
        effect_contract = {
            "metric": "unilateral_wall_penetration_m",
            "initial_minimum_m": 0.02,
            "final_maximum_m": 0.002,
            "maximum_growth_tolerance_m": 0.002,
        }
    accepted = common_pass and effect_pass
    payload: dict[str, object] = {
        "fixture": fixture_name,
        "fixture_contract": fixture_contract,
        "effect_acceptance": {
            **effect_contract,
            "passed": effect_pass,
            "initial_metric": initial_metric,
            "final_metric": final_metric,
            "minimum_metric": minimum_metric,
            "maximum_metric": maximum_metric,
        },
        "simulation_acceptance": {
            "passed": accepted,
            "run_completed": run_completed,
            "requested_physics_ticks": requested_steps,
            "advanced_physics_ticks": advanced_ticks,
            "command_count": command_count,
            "expected_command_count": math.ceil(
                duration * float(driver.timing.command_hz)
            ),
            "viewer_closed_early": viewer_closed_early,
            "driver_faulted": final_tick.driver_faulted,
            "fault_reason": final_tick.fault_reason,
            "state_valid": final_tick.state_valid,
            "state_within_limits": final_tick.state_within_limits,
            "fixture_force_limited_commands": fixture_limited_ticks,
            "overlay_limited_physics_ticks": overlay_limited_ticks,
            "torque_limited_physics_ticks": torque_limited_ticks,
            "slew_limited_physics_ticks": slew_limited_ticks,
            "peak_force_norm_n": peak_force_n,
            "peak_requested_overlay_torque_nm": (peak_requested_overlay_torque_nm),
            "peak_applied_torque_nm": peak_applied_torque_nm,
            "final_force_n": list(final_result.wrench.force_n),
            "final_qpos_rad": list(final_tick.measured_qpos),
            "final_qvel_rad_s": list(final_tick.measured_qvel),
        },
        "driver": driver.execution_metadata,
        "fidelity": (
            "offline deterministic ideal-generalized-effort MuJoCo acceptance; "
            "validates frame math, J-transpose direction, bounded software "
            "overlay, gravity/damping composition and closed-loop fixture effect; "
            "does not validate a motor, CAN, friction, torque calibration, force "
            "fidelity, haptic feel, contact sensor, hard real-time timing or "
            "physical safety"
        ),
    }
    return (0 if accepted else 3), payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    code, payload = run_haptic_fixture_demo(
        source_root=args.rebot_source_root,
        fixture_name=args.fixture,
        duration_sec=args.duration_sec,
        viewer=bool(args.viewer),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_HAPTIC_ANCHOR_QPOS",
    "DEFAULT_HAPTIC_DURATION_SEC",
    "DEFAULT_HAPTIC_INITIAL_QPOS",
    "run_haptic_fixture_demo",
]
