"""Headless-first B601-RS IK-to-MuJoCo position-servo demonstration."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robot_motion.rebot_b601_rs import (
    B601DampedLeastSquaresIK,
    B601PinocchioModel,
)
from robot_motion.transforms import interpolate_pose
from sim.mujoco.viewer import MujocoPassiveViewerSession

from .driver import B601JointPositionCommand, B601PositionDriver
from .scene import (
    B601_RS_DEFAULT_ARM_QPOS,
    B601_RS_PHYSICS_HZ,
    build_b601_mujoco_scene,
)

DEFAULT_TARGET_QPOS = (0.15, 1.1, 1.15, 0.1, -0.1, 0.2)
DEFAULT_ONE_SHOT_DURATION_SEC = 2.0
DEFAULT_CONTINUOUS_IK_DURATION_SEC = 5.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve a B601 pose target and track it with the provisional 500 Hz "
            "MuJoCo position servo, optionally streaming a continuous-seed IK path."
        )
    )
    parser.add_argument(
        "--rebot-source-root",
        type=Path,
        required=True,
        help="Explicit root of the pinned reBotArm_control_py source snapshot.",
    )
    parser.add_argument(
        "--target-qpos",
        type=float,
        nargs=6,
        default=DEFAULT_TARGET_QPOS,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Reachable q used only to generate the demonstration target pose.",
    )
    parser.add_argument(
        "--seed-qpos",
        type=float,
        nargs=6,
        default=B601_RS_DEFAULT_ARM_QPOS,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
        help="Explicit IK seed and initial simulated arm position.",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help=(
            "Total simulated duration. Defaults to 2 s for one-shot mode and "
            "5 s for continuous IK mode."
        ),
    )
    parser.add_argument(
        "--continuous-ik",
        action="store_true",
        help=(
            "Hold for 20%%, stream a minimum-jerk SE(3) target with previous-"
            "solution IK seeds for 60%%, then hold for 20%%."
        ),
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open the existing passive MuJoCo viewer and run in real time.",
    )
    return parser


def run_demo(
    *,
    source_root: str | Path,
    target_qpos: Sequence[float],
    seed_qpos: Sequence[float],
    duration_sec: float,
    viewer: bool,
    continuous_ik: bool = False,
) -> tuple[int, dict[str, object]]:
    """Run one deterministic demo and return an exit code plus JSON payload."""

    duration = float(duration_sec)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_sec must be finite and positive")
    seed = np.asarray(seed_qpos, dtype=float)
    target_configuration = np.asarray(target_qpos, dtype=float)
    model = B601PinocchioModel.from_rebot_source_root(source_root)
    solver = B601DampedLeastSquaresIK(model)
    start_pose = model.forward_kinematics(seed)
    target_pose = model.forward_kinematics(target_configuration)
    solution = solver.solve(
        target_pose,
        seed_qpos=seed,
    )
    payload: dict[str, object] = {
        "ik": {
            "success": solution.success,
            "status": solution.status,
            "iterations": solution.iterations,
            "position_error_m": solution.position_error_m,
            "orientation_error_rad": solution.orientation_error_rad,
            "seed_distance_rad": solution.seed_distance_rad,
            "sigma_min": solution.sigma_min,
            "condition_number": solution.condition_number,
            "qpos": list(solution.qpos),
        },
        "fidelity": (
            "visualization_only_provisional; no hardware, gravity, collision, "
            "gripper-actuation, hardware-safety, force, or haptic validation; "
            "target slew and measured-state guards are simulation-only"
        ),
    }
    if not solution.success:
        return 2, payload

    scene = build_b601_mujoco_scene(
        source_root,
        initial_arm_qpos=seed,
    )
    driver = B601PositionDriver(scene)
    driver.reset(seed)
    one_shot_command = B601JointPositionCommand(
        sequence=0,
        source_timestamp_sec=0.0,
        receive_timestamp_sec=0.0,
        qpos=solution.qpos,
    )
    viewer_session = MujocoPassiveViewerSession(scene) if viewer else None
    if viewer_session is not None:
        viewer_session.start()
    final_tick = None
    final_target_qpos = np.asarray(solution.qpos, dtype=float)
    continuous_seed = seed.copy()
    next_sequence = 0
    next_ik_time_sec = 0.2 * duration
    ik_period_sec = 1.0 / float(driver.timing.ik_hz)
    pending_qpos: np.ndarray | None = None
    pending_source_timestamp_sec: float | None = None
    pending_is_final = False
    ik_attempts = 0
    ik_successes = 0
    ik_failures = 0
    max_ik_position_error_m = 0.0
    max_ik_orientation_error_rad = 0.0
    min_ik_sigma = float("inf")
    max_ik_condition_number = 0.0
    max_ik_command_step_rad = 0.0
    max_actuator_target_step_rad = 0.0
    peak_actuator_tracking_error_rad = 0.0
    peak_measured_velocity_rad_s = 0.0
    previous_actuator_target = seed.copy()
    motion_start_sec = 0.2 * duration
    motion_end_sec = 0.8 * duration
    continuous_ik_failed = False
    final_waypoint_solved = False
    final_waypoint_accepted = False
    viewer_closed_early = False
    advanced_physics_ticks = 0
    steps = math.ceil(duration * B601_RS_PHYSICS_HZ)
    try:
        for step in range(steps):
            command = None
            if continuous_ik:
                physics_time = driver.physics_time_sec
                while (
                    not continuous_ik_failed
                    and not final_waypoint_solved
                    and next_ik_time_sec <= physics_time + 1e-12
                ):
                    ik_deadline_sec = next_ik_time_sec
                    pose_sample_time_sec = min(
                        ik_deadline_sec,
                        motion_end_sec,
                    )
                    phase = (pose_sample_time_sec - motion_start_sec) / (
                        motion_end_sec - motion_start_sec
                    )
                    phase = float(np.clip(phase, 0.0, 1.0))
                    blend = phase**3 * (10.0 - 15.0 * phase + 6.0 * phase**2)
                    waypoint_pose = interpolate_pose(start_pose, target_pose, blend)
                    waypoint_solution = solver.solve(
                        waypoint_pose,
                        seed_qpos=continuous_seed,
                    )
                    ik_attempts += 1
                    max_ik_position_error_m = max(
                        max_ik_position_error_m,
                        float(waypoint_solution.position_error_m),
                    )
                    max_ik_orientation_error_rad = max(
                        max_ik_orientation_error_rad,
                        float(waypoint_solution.orientation_error_rad),
                    )
                    min_ik_sigma = min(
                        min_ik_sigma,
                        float(waypoint_solution.sigma_min),
                    )
                    max_ik_condition_number = max(
                        max_ik_condition_number,
                        float(waypoint_solution.condition_number),
                    )
                    if waypoint_solution.success:
                        qpos = np.asarray(waypoint_solution.qpos, dtype=float)
                        max_ik_command_step_rad = max(
                            max_ik_command_step_rad,
                            float(np.max(np.abs(qpos - continuous_seed))),
                        )
                        continuous_seed = qpos
                        pending_qpos = qpos
                        pending_source_timestamp_sec = ik_deadline_sec
                        pending_is_final = phase >= 1.0
                        ik_successes += 1
                        if pending_is_final:
                            final_waypoint_solved = True
                            final_target_qpos = qpos
                    else:
                        ik_failures += 1
                        continuous_ik_failed = True
                    next_ik_time_sec = ik_deadline_sec + ik_period_sec
                    if ik_deadline_sec < motion_end_sec:
                        next_ik_time_sec = min(
                            next_ik_time_sec,
                            motion_end_sec,
                        )

                if pending_qpos is not None and driver.command_due:
                    assert pending_source_timestamp_sec is not None
                    command = B601JointPositionCommand(
                        sequence=next_sequence,
                        source_timestamp_sec=pending_source_timestamp_sec,
                        receive_timestamp_sec=physics_time,
                        qpos=tuple(float(value) for value in pending_qpos),
                    )
            elif step == 0:
                command = one_shot_command

            final_tick = driver.step_physics(command)
            if final_tick.plant_advanced:
                advanced_physics_ticks += 1
            if continuous_ik and command is not None and final_tick.command_applied:
                next_sequence += 1
                if pending_is_final:
                    final_waypoint_accepted = True
                pending_qpos = None
                pending_source_timestamp_sec = None
                pending_is_final = False
            actuator_target = np.asarray(
                final_tick.actuator_target_qpos,
                dtype=float,
            )
            max_actuator_target_step_rad = max(
                max_actuator_target_step_rad,
                float(np.max(np.abs(actuator_target - previous_actuator_target))),
            )
            previous_actuator_target = actuator_target
            peak_actuator_tracking_error_rad = max(
                peak_actuator_tracking_error_rad,
                float(
                    np.max(
                        np.abs(
                            np.asarray(final_tick.measured_qpos, dtype=float)
                            - actuator_target
                        )
                    )
                ),
            )
            peak_measured_velocity_rad_s = max(
                peak_measured_velocity_rad_s,
                float(np.max(np.abs(final_tick.measured_qvel))),
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

    assert final_tick is not None
    final_tracking_error = float(
        np.max(
            np.abs(
                np.asarray(final_tick.measured_qpos, dtype=float) - final_target_qpos
            )
        )
    )
    run_completed = advanced_physics_ticks == steps and not viewer_closed_early
    motion_completed = bool(
        continuous_ik
        and final_waypoint_accepted
        and run_completed
        and not final_tick.driver_faulted
    )
    if continuous_ik:
        payload["continuous_ik"] = {
            "phase_contract": "hold_20pct_motion_60pct_hold_20pct",
            "seed_policy": "previous_successful_solution",
            "clock_policy": "independent_timestamp_deadlines",
            "terminal_waypoint_policy": "exact_motion_boundary",
            "ik_hz": float(driver.timing.ik_hz),
            "command_hz": float(driver.timing.command_hz),
            "ik_attempts": ik_attempts,
            "ik_successes": ik_successes,
            "ik_failures": ik_failures,
            "terminated_on_ik_failure": continuous_ik_failed,
            "max_position_error_m": max_ik_position_error_m,
            "max_orientation_error_rad": max_ik_orientation_error_rad,
            "minimum_sigma": None if math.isinf(min_ik_sigma) else min_ik_sigma,
            "max_condition_number": max_ik_condition_number,
            "max_joint_command_step_rad": max_ik_command_step_rad,
            "final_waypoint_solved": final_waypoint_solved,
            "final_waypoint_accepted": final_waypoint_accepted,
            "motion_completed": motion_completed,
        }
    payload["simulation"] = {
        **driver.execution_metadata,
        "physics_ticks": advanced_physics_ticks,
        "requested_physics_ticks": steps,
        "run_completed": run_completed,
        "viewer_closed_early": viewer_closed_early,
        "accepted_sequence": final_tick.accepted_sequence,
        "state_valid": final_tick.state_valid,
        "state_within_limits": final_tick.state_within_limits,
        "driver_faulted": final_tick.driver_faulted,
        "fault_reason": final_tick.fault_reason,
        "max_joint_tracking_error_rad": final_tracking_error,
        "max_joint_tracking_error_semantics": ("legacy_name_reports_final_goal_error"),
        "final_joint_tracking_error_rad": final_tracking_error,
        "peak_actuator_tracking_error_rad": peak_actuator_tracking_error_rad,
        "max_actuator_target_step_rad": max_actuator_target_step_rad,
        "peak_measured_velocity_rad_s": peak_measured_velocity_rad_s,
        "measured_qpos": list(final_tick.measured_qpos),
    }
    if (
        not final_tick.state_valid
        or not final_tick.state_within_limits
        or final_tick.driver_faulted
    ):
        return 3, payload
    if continuous_ik and (ik_attempts == 0 or ik_failures > 0 or not motion_completed):
        return 5, payload
    if final_tracking_error > 2e-3:
        return 4, payload
    return 0, payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    continuous_ik = bool(args.continuous_ik)
    duration_sec = args.duration_sec
    if duration_sec is None:
        duration_sec = (
            DEFAULT_CONTINUOUS_IK_DURATION_SEC
            if continuous_ik
            else DEFAULT_ONE_SHOT_DURATION_SEC
        )
    code, payload = run_demo(
        source_root=args.rebot_source_root,
        target_qpos=args.target_qpos,
        seed_qpos=args.seed_qpos,
        duration_sec=duration_sec,
        viewer=bool(args.viewer),
        continuous_ik=continuous_ik,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
