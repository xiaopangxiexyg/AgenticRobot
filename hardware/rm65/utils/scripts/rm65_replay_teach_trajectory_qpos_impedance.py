from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.force_compensation import (  # noqa: E402
    PoseAdmittance,
    WrenchProcessor,
    angle_delta,
    apply_pose_offset,
    finite_list,
    mean_wrench,
    norm,
    normalize_pose,
    parse_axes,
    parse_signs,
)
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65  # noqa: E402


CONFIRM_TEXT = "TEST_RM65_QPOS_TRAJECTORY_IMPEDANCE"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    return ("left", args.left_ip) if args.arm == "left" else ("right", args.right_ip)


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


def _read_state(robot: Arm, side: str) -> dict[str, list[float]]:
    code, joint_deg, pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side} reports arm_err={arm_err}, sys_err={sys_err}")
    return {
        "joint_deg": finite_list(joint_deg, expected_len=6, label="joint_deg"),
        "pose": finite_list(pose, expected_len=6, label="pose"),
    }


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": finite_list(raw, expected_len=6, label="raw"),
        "zero": finite_list(zero, expected_len=6, label="zero"),
        "work": finite_list(work, expected_len=6, label="work"),
        "tool": finite_list(tool, expected_len=6, label="tool"),
    }


def _current_tool_name(frame_result: object) -> str:
    if isinstance(frame_result, tuple) and len(frame_result) >= 2:
        frame = frame_result[1]
    else:
        frame = frame_result
    if isinstance(frame, dict):
        value = frame.get("name") or frame.get("frame_name") or frame.get("tool_name")
        if value is not None:
            return str(value).strip("\x00")
    return str(frame_result).strip("\x00")


def _check_expected_tool(robot: Arm, expected_tool: str) -> str:
    if not expected_tool:
        return ""
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if expected_tool not in current_name:
        raise RuntimeError(f"expected active tool containing '{expected_tool}', got '{current_name}'")
    return current_name


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def _pose_delta(a: list[float], b: list[float]) -> tuple[float, float]:
    linear = norm(a[index] - b[index] for index in range(3))
    angular = norm(angle_delta(a[index], b[index]) for index in range(3, 6))
    return linear, angular


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _stats_ms(values_sec: list[float]) -> dict[str, float]:
    values_ms = [value * 1000.0 for value in values_sec]
    return {
        "mean_ms": statistics.fmean(values_ms) if values_ms else math.nan,
        "p50_ms": _percentile(values_ms, 0.50),
        "p95_ms": _percentile(values_ms, 0.95),
        "p99_ms": _percentile(values_ms, 0.99),
        "max_ms": max(values_ms) if values_ms else math.nan,
    }


def _load_trajectory(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("data_type") != "rm65_teach_trajectory":
        raise RuntimeError(f"{path} is not an rm65_teach_trajectory JSON")
    samples = data.get("samples") or []
    if len(samples) < 2:
        raise RuntimeError("trajectory must contain at least two samples")
    cleaned = []
    for sample in samples:
        cleaned.append(
            {
                "t_sec": float(sample["t_sec"]),
                "pose": finite_list(sample["pose"], expected_len=6, label="sample pose"),
                "joint_deg": finite_list(sample["joint_deg"], expected_len=6, label="sample joint_deg"),
            }
        )
    cleaned.sort(key=lambda item: item["t_sec"])
    first_time = cleaned[0]["t_sec"]
    for sample in cleaned:
        sample["t_sec"] -= first_time
    return {"metadata": data, "samples": cleaned, "duration_sec": cleaned[-1]["t_sec"]}


def _unwrap_poses(samples: list[dict[str, object]]) -> list[list[float]]:
    unwrapped = [list(samples[0]["pose"])]
    for index in range(1, len(samples)):
        previous = unwrapped[-1]
        raw = list(samples[index]["pose"])
        pose = raw[:3] + [
            previous[axis] + angle_delta(raw[axis], samples[index - 1]["pose"][axis])
            for axis in range(3, 6)
        ]
        unwrapped.append(pose)
    return unwrapped


def _trajectory_range(rows: list[list[float]]) -> list[float]:
    return [max(row[index] for row in rows) - min(row[index] for row in rows) for index in range(6)]


def _interpolate_rows(samples: list[dict[str, object]], rows: list[list[float]], t_sec: float, *, pose: bool) -> list[float]:
    if t_sec <= 0.0:
        return normalize_pose(rows[0]) if pose else list(rows[0])
    if t_sec >= samples[-1]["t_sec"]:
        return normalize_pose(rows[-1]) if pose else list(rows[-1])
    lo = 0
    hi = len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid]["t_sec"] <= t_sec:
            lo = mid
        else:
            hi = mid
    t0 = samples[lo]["t_sec"]
    t1 = samples[hi]["t_sec"]
    alpha = 0.0 if t1 <= t0 else (t_sec - t0) / (t1 - t0)
    out = [rows[lo][index] + (rows[hi][index] - rows[lo][index]) * alpha for index in range(6)]
    return normalize_pose(out) if pose else out


def _sample_tare(robot: Arm, side: str, args) -> list[float]:
    samples = max(1, int(round(args.tare_sec * args.control_hz)))
    interval = 1.0 / args.control_hz
    print(f"{side}: sampling tare source={args.force_source} samples={samples}")
    wrenches = []
    for index in range(samples):
        readings = _read_force(robot, side)
        wrenches.append(readings[args.force_source])
        if index + 1 < samples:
            time.sleep(interval)
    tare = mean_wrench(wrenches)
    print(f"{side}: tare={_format(tare)} |F|={norm(tare[:3]):.4f}N |T|={norm(tare[3:]):.4f}Nm")
    return tare


def _smoothstep(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _interpolate_joint(start_joint: list[float], target_joint: list[float], alpha: float) -> list[float]:
    smoothed = _smoothstep(alpha)
    return [start_joint[index] + (target_joint[index] - start_joint[index]) * smoothed for index in range(6)]


def _move_to_start_joint(robot: Arm, side: str, current_state: dict[str, list[float]], target_joint: list[float], args) -> dict[str, object]:
    joint_distance = _joint_delta_deg(current_state["joint_deg"], target_joint)
    if joint_distance > args.start_move_max_joint_delta_deg:
        raise RuntimeError(f"start joint distance {joint_distance:.4f}deg > --start-move-max-joint-delta-deg")
    if joint_distance <= args.max_start_joint_error_deg:
        print(f"{side}: already close to first trajectory joint; skipping start move")
        return {"skipped": True, "initial_joint_error_deg": joint_distance}

    duration = max(args.start_move_sec, joint_distance / args.start_move_max_joint_velocity_deg_s)
    period = 1.0 / args.start_move_hz
    steps = max(2, int(math.ceil(duration * args.start_move_hz)))
    hold_steps = max(0, int(math.ceil(args.start_move_hold_sec * args.start_move_hz)))
    print(f"{side}: moving smoothly to first trajectory joint distance={joint_distance:.4f}deg duration={duration:.3f}s steps={steps} hold={hold_steps}")

    records = []
    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_joint_tracking = 0.0
    missed = 0
    last_state = current_state
    next_deadline = time.perf_counter()
    motion_started = False
    try:
        for index in range(steps + hold_steps):
            loop_start = time.perf_counter()
            if index < steps:
                alpha = (index + 1) / steps
                command_joint = _interpolate_joint(current_state["joint_deg"], target_joint, alpha)
                phase = "move"
            else:
                alpha = 1.0
                command_joint = list(target_joint)
                phase = "hold"
            _require_zero(
                robot.Movej_CANFD(command_joint, follow=False, trajectory_mode=args.trajectory_mode, radio=args.radio),
                f"{side} start Movej_CANFD",
            )
            motion_started = True

            if index % args.start_move_state_read_every == 0 or index + 1 == steps + hold_steps:
                last_state = _read_state(robot, side)
            state = last_state
            force_norm = 0.0
            torque_norm = 0.0
            if index % args.start_move_force_read_every == 0 or index + 1 == steps + hold_steps:
                readings = _read_force(robot, side)
                source = readings[args.force_source]
                force_norm = norm(source[:3])
                torque_norm = norm(source[3:])
                if force_norm > args.hard_stop_force_n:
                    raise RuntimeError(f"start move force {force_norm:.4f}N > --hard-stop-force-n")
                if torque_norm > args.hard_stop_torque_nm:
                    raise RuntimeError(f"start move torque {torque_norm:.4f}Nm > --hard-stop-torque-nm")
            joint_tracking = _joint_delta_deg(state["joint_deg"], command_joint)
            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_joint_tracking = max(max_joint_tracking, joint_tracking)
            loop_elapsed = time.perf_counter() - loop_start
            missed_deadline = loop_elapsed > period
            if missed_deadline:
                missed += 1
            if index % args.print_every == 0 or index + 1 == steps + hold_steps:
                print(
                    f"{side}:qpos_start {phase} step={index + 1}/{steps + hold_steps} "
                    f"alpha={alpha:.3f} joint_track={joint_tracking:.3f}deg "
                    f"|F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm"
                )
            records.append(
                {
                    "step": index + 1,
                    "phase": phase,
                    "alpha": alpha,
                    "command_joint_deg": command_joint,
                    "observed_joint_deg": state["joint_deg"],
                    "joint_tracking_deg": joint_tracking,
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "missed_deadline": missed_deadline,
                }
            )
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if motion_started:
            code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop(start_move) -> {code}")
            time.sleep(args.start_move_settle_sec)

    final_state = _read_state(robot, side)
    final_error = _joint_delta_deg(final_state["joint_deg"], target_joint)
    print(f"{side}: start joint final_error={final_error:.4f}deg")
    return {
        "skipped": False,
        "initial_joint_error_deg": joint_distance,
        "final_joint_error_deg": final_error,
        "duration_sec": duration,
        "steps": steps,
        "hold_steps": hold_steps,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "max_joint_tracking_deg": max_joint_tracking,
        "missed_deadlines": missed,
        "records": records,
    }


def _solve_ik(robot: Arm, seed_joint: list[float], target_pose: list[float], side: str) -> list[float]:
    code, joint = robot.Algo_Inverse_Kinematics(seed_joint, target_pose, flag=1)
    if int(code) != 0:
        raise RuntimeError(f"{side}: IK failed with return code {code}")
    return finite_list(joint[:6], expected_len=6, label="ik_joint_deg")


def _run_replay(
    robot: Arm,
    side: str,
    start_state: dict[str, list[float]],
    traj: dict[str, object],
    unwrapped_poses: list[list[float]],
    joint_rows: list[list[float]],
    tare: list[float],
    args,
) -> dict[str, object]:
    duration = float(traj["duration_sec"]) / args.speed_scale
    period = 1.0 / args.control_hz
    loops = max(1, int(math.ceil(duration * args.control_hz)))
    processor = WrenchProcessor(
        tare=tare,
        lowpass_alpha=args.lowpass_alpha,
        force_deadband_n=args.force_deadband_n,
        torque_deadband_nm=args.torque_deadband_nm,
    )
    admittance = PoseAdmittance(
        enabled_axes=args.enabled_axes,
        force_sign=args.force_sign,
        angular_force_sign=args.angular_force_sign,
        angular_axis_signs=args.angular_axis_signs,
        linear_gain_m_s_per_n=args.linear_gain_m_s_per_n,
        angular_gain_rad_s_per_nm=args.angular_gain_rad_s_per_nm,
        linear_compliance_m_per_n=args.linear_compliance_m_per_n,
        angular_compliance_rad_per_nm=args.angular_compliance_rad_per_nm,
        max_linear_velocity_m_s=args.max_linear_velocity_m_s,
        max_angular_velocity_rad_s=args.max_angular_velocity_rad_s,
        max_linear_offset_m=args.max_linear_offset_m,
        max_angular_offset_rad=args.max_angular_offset_rad,
        return_rate_s=args.return_rate_s,
    )

    records = []
    timing = {"force_read": [], "state_read": [], "compute": [], "ik": [], "command": [], "loop_elapsed": [], "loop_interval": []}
    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_linear_offset = 0.0
    max_angular_offset = 0.0
    max_joint_tracking = 0.0
    max_pose_tracking_linear = 0.0
    max_pose_tracking_angular = 0.0
    max_admittance_joint_offset = 0.0
    max_command_step = 0.0
    missed = 0
    last_state = start_state
    previous_command_joint = list(start_state["joint_deg"])
    ik_seed = list(start_state["joint_deg"])
    previous_loop_start = None
    run_start = time.perf_counter()
    next_deadline = run_start
    motion_started = False
    print(
        f"{side}: QPOS trajectory impedance duration={duration:.3f}s hz={args.control_hz:.2f} "
        f"speed_scale={args.speed_scale:.3f} axes={args.axes}"
    )
    try:
        for index in range(loops):
            loop_start = time.perf_counter()
            if previous_loop_start is not None:
                timing["loop_interval"].append(loop_start - previous_loop_start)
            previous_loop_start = loop_start
            elapsed = loop_start - run_start
            trajectory_t = min(elapsed * args.speed_scale, float(traj["duration_sec"]))
            ref_pose = _interpolate_rows(traj["samples"], unwrapped_poses, trajectory_t, pose=True)
            ref_joint = _interpolate_rows(traj["samples"], joint_rows, trajectory_t, pose=False)

            force_start = time.perf_counter()
            readings = _read_force(robot, side)
            timing["force_read"].append(time.perf_counter() - force_start)

            compute_start = time.perf_counter()
            processed = processor.process(readings[args.force_source])
            update = admittance.update(processed["deadbanded"], period)
            target_pose = normalize_pose(apply_pose_offset(ref_pose, update["offset"]))
            timing["compute"].append(time.perf_counter() - compute_start)

            ik_start = time.perf_counter()
            seed = ik_seed if args.ik_seed_mode == "previous-command" else ref_joint
            command_joint = _solve_ik(robot, seed, target_pose, side)
            timing["ik"].append(time.perf_counter() - ik_start)
            ik_seed = list(command_joint)

            command_step = _joint_delta_deg(previous_command_joint, command_joint)
            admittance_joint_offset = _joint_delta_deg(ref_joint, command_joint)
            if command_step > args.max_command_joint_step_deg:
                raise RuntimeError(f"command joint step {command_step:.4f}deg > --max-command-joint-step-deg")
            if admittance_joint_offset > args.max_admittance_joint_offset_deg:
                raise RuntimeError(f"admittance joint offset {admittance_joint_offset:.4f}deg > --max-admittance-joint-offset-deg")

            command_start = time.perf_counter()
            _require_zero(
                robot.Movej_CANFD(command_joint, follow=False, trajectory_mode=args.trajectory_mode, radio=args.radio),
                f"{side} Movej_CANFD",
            )
            timing["command"].append(time.perf_counter() - command_start)
            motion_started = True
            previous_command_joint = list(command_joint)

            state_sampled = index % args.state_read_every == 0 or index + 1 == loops
            state_read_sec = 0.0
            if state_sampled:
                state_start = time.perf_counter()
                last_state = _read_state(robot, side)
                state_read_sec = time.perf_counter() - state_start
                timing["state_read"].append(state_read_sec)
            state = last_state

            force_norm = norm(processed["tare_corrected"][:3])
            torque_norm = norm(processed["tare_corrected"][3:])
            joint_tracking = _joint_delta_deg(state["joint_deg"], ref_joint)
            pose_tracking_linear, pose_tracking_angular = _pose_delta(state["pose"], ref_pose)
            linear_offset = float(update["linear_offset_norm_m"])
            angular_offset = float(update["angular_offset_norm_rad"])
            if force_norm > args.hard_stop_force_n:
                raise RuntimeError(f"force {force_norm:.4f}N > --hard-stop-force-n")
            if torque_norm > args.hard_stop_torque_nm:
                raise RuntimeError(f"torque {torque_norm:.4f}Nm > --hard-stop-torque-nm")
            if joint_tracking > args.max_tracking_joint_deg:
                raise RuntimeError(f"joint tracking {joint_tracking:.4f}deg > --max-tracking-joint-deg")
            if pose_tracking_linear > args.max_tracking_linear_m:
                raise RuntimeError(f"pose tracking linear {pose_tracking_linear:.5f}m > --max-tracking-linear-m")
            if pose_tracking_angular > args.max_tracking_angular_rad:
                raise RuntimeError(f"pose tracking angular {pose_tracking_angular:.5f}rad > --max-tracking-angular-rad")

            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_linear_offset = max(max_linear_offset, linear_offset)
            max_angular_offset = max(max_angular_offset, angular_offset)
            max_joint_tracking = max(max_joint_tracking, joint_tracking)
            max_pose_tracking_linear = max(max_pose_tracking_linear, pose_tracking_linear)
            max_pose_tracking_angular = max(max_pose_tracking_angular, pose_tracking_angular)
            max_admittance_joint_offset = max(max_admittance_joint_offset, admittance_joint_offset)
            max_command_step = max(max_command_step, command_step)

            loop_elapsed = time.perf_counter() - loop_start
            timing["loop_elapsed"].append(loop_elapsed)
            missed_deadline = loop_elapsed > period
            if missed_deadline:
                missed += 1
            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:qpos_traj step={index + 1}/{loops} t={trajectory_t:.3f}s "
                    f"offset={_format(update['offset'], precision=4)} "
                    f"joint_track={joint_tracking:.3f}deg adm_q={admittance_joint_offset:.3f}deg "
                    f"pose_track={pose_tracking_linear:.4f}m/{pose_tracking_angular:.4f}rad "
                    f"|F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm loop_ms={loop_elapsed * 1000.0:.2f}"
                )
            records.append(
                {
                    "step": index + 1,
                    "time_sec": elapsed,
                    "trajectory_t_sec": trajectory_t,
                    "reference_joint_deg": ref_joint,
                    "reference_pose": ref_pose,
                    "target_pose": target_pose,
                    "command_joint_deg": command_joint,
                    "observed_joint_deg": state["joint_deg"],
                    "observed_pose": state["pose"],
                    "tare_corrected_wrench": processed["tare_corrected"],
                    "filtered_wrench": processed["filtered"],
                    "active_wrench": update["active_wrench"],
                    "target_offset": update["target_offset"],
                    "offset": update["offset"],
                    "admittance_joint_offset_deg": admittance_joint_offset,
                    "command_joint_step_deg": command_step,
                    "joint_tracking_deg": joint_tracking,
                    "pose_tracking_linear_m": pose_tracking_linear,
                    "pose_tracking_angular_rad": pose_tracking_angular,
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "state_sampled": state_sampled,
                    "timing": {
                        "state_read_sec": state_read_sec,
                        "loop_elapsed_sec": loop_elapsed,
                        "missed_deadline": missed_deadline,
                    },
                }
            )
            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)
    finally:
        if motion_started:
            code = robot.Arm_Slow_Stop()
            print(f"{side}: Arm_Slow_Stop -> {code}")
            time.sleep(args.stop_settle_sec)

    summary = {
        "loops": loops,
        "missed_deadlines": missed,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "max_linear_offset_m": max_linear_offset,
        "max_angular_offset_rad": max_angular_offset,
        "max_joint_tracking_deg": max_joint_tracking,
        "max_pose_tracking_linear_m": max_pose_tracking_linear,
        "max_pose_tracking_angular_rad": max_pose_tracking_angular,
        "max_admittance_joint_offset_deg": max_admittance_joint_offset,
        "max_command_joint_step_deg": max_command_step,
        "timing": {key: _stats_ms(values) for key, values in timing.items() if values},
    }
    print(f"{side}: QPOS replay summary={summary}")
    return {"records": records, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay an RM65 teach trajectory through QPOS/CANFD with Cartesian admittance.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="inspire")
    parser.add_argument("--input-json", default="hardware/rm65/utils/data/trajectories/rm65_left_teach_trajectory.json")
    parser.add_argument("--output-json", default="hardware/rm65/utils/results/rm65_left_teach_trajectory_qpos_impedance_replay.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--move-to-start", action="store_true")
    parser.add_argument("--start-move-hz", type=float, default=20.0)
    parser.add_argument("--start-move-sec", type=float, default=6.0)
    parser.add_argument("--start-move-hold-sec", type=float, default=1.5)
    parser.add_argument("--start-move-max-joint-delta-deg", type=float, default=120.0)
    parser.add_argument("--start-move-max-joint-velocity-deg-s", type=float, default=20.0)
    parser.add_argument("--start-move-state-read-every", type=int, default=1)
    parser.add_argument("--start-move-force-read-every", type=int, default=2)
    parser.add_argument("--start-move-settle-sec", type=float, default=0.8)
    parser.add_argument("--max-start-joint-error-deg", type=float, default=5.0)
    parser.add_argument("--speed-scale", type=float, default=0.25)
    parser.add_argument("--control-hz", type=float, default=40.0)
    parser.add_argument("--force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--axes", default="all")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--angular-force-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument("--angular-axis-signs", default="1,1,-1")
    parser.add_argument("--tare-sec", type=float, default=1.0)
    parser.add_argument("--lowpass-alpha", type=float, default=0.85)
    parser.add_argument("--force-deadband-n", type=float, default=0.30)
    parser.add_argument("--torque-deadband-nm", type=float, default=0.03)
    parser.add_argument("--linear-gain-m-s-per-n", type=float, default=0.0)
    parser.add_argument("--angular-gain-rad-s-per-nm", type=float, default=0.0)
    parser.add_argument("--linear-compliance-m-per-n", type=float, default=0.004)
    parser.add_argument("--angular-compliance-rad-per-nm", type=float, default=0.04)
    parser.add_argument("--max-linear-velocity-m-s", type=float, default=0.04)
    parser.add_argument("--max-angular-velocity-rad-s", type=float, default=0.15)
    parser.add_argument("--max-linear-offset-m", type=float, default=0.05)
    parser.add_argument("--max-angular-offset-rad", type=float, default=0.4)
    parser.add_argument("--return-rate-s", type=float, default=0.35)
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--state-read-every", type=int, default=1)
    parser.add_argument("--ik-seed-mode", choices=("previous-command", "reference"), default="previous-command")
    parser.add_argument("--hard-stop-force-n", type=float, default=50.0)
    parser.add_argument("--hard-stop-torque-nm", type=float, default=10.0)
    parser.add_argument("--max-trajectory-joint-range-deg", type=float, default=140.0)
    parser.add_argument("--max-trajectory-linear-range-m", type=float, default=0.60)
    parser.add_argument("--max-trajectory-angular-range-rad", type=float, default=2.50)
    parser.add_argument("--max-tracking-joint-deg", type=float, default=40.0)
    parser.add_argument("--max-tracking-linear-m", type=float, default=0.16)
    parser.add_argument("--max-tracking-angular-rad", type=float, default=0.70)
    parser.add_argument("--max-admittance-joint-offset-deg", type=float, default=25.0)
    parser.add_argument("--max-command-joint-step-deg", type=float, default=5.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    return parser


def _validate_args(args) -> None:
    args.enabled_axes = parse_axes(args.axes, allow_rotation=True)
    args.angular_axis_signs = parse_signs(args.angular_axis_signs, expected_len=3, label="angular-axis-signs")
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"QPOS trajectory replay requires --confirm {CONFIRM_TEXT}")
    if args.speed_scale <= 0.0:
        raise RuntimeError("--speed-scale must be positive")
    if args.control_hz <= 0.0 or args.control_hz > 40.0:
        raise RuntimeError("--control-hz must be in (0, 40]")
    if args.start_move_hz <= 0.0 or args.start_move_hz > 40.0:
        raise RuntimeError("--start-move-hz must be in (0, 40]")
    for name in (
        "start_move_sec",
        "start_move_hold_sec",
        "start_move_max_joint_delta_deg",
        "start_move_max_joint_velocity_deg_s",
        "start_move_settle_sec",
        "max_start_joint_error_deg",
        "tare_sec",
        "force_deadband_n",
        "torque_deadband_nm",
        "linear_gain_m_s_per_n",
        "angular_gain_rad_s_per_nm",
        "linear_compliance_m_per_n",
        "angular_compliance_rad_per_nm",
        "max_linear_velocity_m_s",
        "max_angular_velocity_rad_s",
        "max_linear_offset_m",
        "max_angular_offset_rad",
        "return_rate_s",
        "hard_stop_force_n",
        "hard_stop_torque_nm",
        "max_trajectory_joint_range_deg",
        "max_trajectory_linear_range_m",
        "max_trajectory_angular_range_rad",
        "max_tracking_joint_deg",
        "max_tracking_linear_m",
        "max_tracking_angular_rad",
        "max_admittance_joint_offset_deg",
        "max_command_joint_step_deg",
        "stop_settle_sec",
    ):
        if getattr(args, name) < 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative")
    if args.start_move_state_read_every < 1 or args.start_move_force_read_every < 1:
        raise RuntimeError("--start-move-*-every must be >= 1")
    if args.state_read_every < 1 or args.print_every < 1:
        raise RuntimeError("--state-read-every and --print-every must be >= 1")
    if not 0.0 < args.lowpass_alpha <= 1.0:
        raise RuntimeError("--lowpass-alpha must be in (0, 1]")


def _write_report(path_text: str, report: dict[str, object]) -> None:
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote report: {path}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        traj = _load_trajectory(Path(args.input_json))
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    samples = traj["samples"]
    unwrapped_poses = _unwrap_poses(samples)
    joint_rows = [list(sample["joint_deg"]) for sample in samples]
    joint_range = _trajectory_range(joint_rows)
    pose_range = _trajectory_range(unwrapped_poses)
    linear_range = norm(pose_range[:3])
    angular_range = norm(pose_range[3:])
    max_joint_range = max(joint_range)
    print(
        f"trajectory samples={len(samples)} duration={traj['duration_sec']:.3f}s "
        f"joint_range={_format(joint_range, precision=3)} pose_range={_format(pose_range, precision=4)}"
    )
    if max_joint_range > args.max_trajectory_joint_range_deg:
        print(f"ERROR: trajectory joint range {max_joint_range:.4f}deg > --max-trajectory-joint-range-deg", file=sys.stderr)
        return 2
    if linear_range > args.max_trajectory_linear_range_m:
        print(f"ERROR: trajectory linear range {linear_range:.4f}m > --max-trajectory-linear-range-m", file=sys.stderr)
        return 2
    if angular_range > args.max_trajectory_angular_range_rad:
        print(f"ERROR: trajectory angular range {angular_range:.4f}rad > --max-trajectory-angular-range-rad", file=sys.stderr)
        return 2

    side, ip = _selected_ip(args)
    robot = None
    try:
        robot = _connect_robot(side, ip)
        active_tool = _check_expected_tool(robot, args.expected_tool)
        initial_state = _read_state(robot, side)
        first_joint = joint_rows[0]
        first_joint_error = _joint_delta_deg(initial_state["joint_deg"], first_joint)
        print(f"{side}: current_joint={_format(initial_state['joint_deg'])}")
        print(f"start_joint_error={first_joint_error:.4f}deg")
        if not args.move_to_start and first_joint_error > args.max_start_joint_error_deg:
            raise RuntimeError("current joint is far from first trajectory joint; use --move-to-start")
        if first_joint_error > args.start_move_max_joint_delta_deg:
            raise RuntimeError(f"start joint distance {first_joint_error:.4f}deg > --start-move-max-joint-delta-deg")

        start_move_report = {"requested": bool(args.move_to_start), "executed": False, "initial_joint_error_deg": first_joint_error}
        start_state = initial_state
        if args.execute and args.move_to_start:
            start_move_report = _move_to_start_joint(robot, side, initial_state, first_joint, args)
            start_move_report["requested"] = True
            start_move_report["executed"] = True
            start_state = _read_state(robot, side)
            final_start_error = _joint_delta_deg(start_state["joint_deg"], first_joint)
            if final_start_error > args.max_start_joint_error_deg:
                raise RuntimeError(f"post-move start joint error {final_start_error:.4f}deg > --max-start-joint-error-deg")
        elif not args.execute and args.move_to_start:
            print("DRY-RUN: --move-to-start is requested; start move will be sent only with --execute.")

        tare = _sample_tare(robot, side, args)
        report: dict[str, object] = {
            "schema_version": 1,
            "generated_by": "hardware/rm65/utils/scripts/rm65_replay_teach_trajectory_qpos_impedance.py",
            "arm": side,
            "ip": ip,
            "active_tool": active_tool,
            "input_json": args.input_json,
            "executed": bool(args.execute),
            "move_to_start": args.move_to_start,
            "speed_scale": args.speed_scale,
            "control_hz": args.control_hz,
            "axes": args.axes,
            "enabled_axes": args.enabled_axes,
            "trajectory": {
                "samples": len(samples),
                "duration_sec": traj["duration_sec"],
                "joint_range_deg": joint_range,
                "pose_range": pose_range,
                "linear_range_m": linear_range,
                "angular_range_rad": angular_range,
            },
            "initial_state": initial_state,
            "start_state": start_state,
            "start_move": start_move_report,
            "tare": tare,
            "parameters": vars(args),
        }
        if not args.execute:
            print("DRY-RUN: QPOS trajectory commands will not be sent.")
            _write_report(args.output_json, report)
            return 0
        result = _run_replay(robot, side, start_state, traj, unwrapped_poses, joint_rows, tare, args)
        report.update(result)
        report["final_state"] = _read_state(robot, side)
        report["final_force"] = _read_force(robot, side)
        _write_report(args.output_json, report)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.Arm_Socket_Close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
