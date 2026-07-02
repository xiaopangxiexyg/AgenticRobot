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
    admittance_wrench_for_arm,
    apply_pose_offset,
    finite_list,
    mean_wrench,
    norm,
    parse_axes,
    parse_signs,
)
from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65  # noqa: E402


CONFIRM_TEXT = "TEST_RM65_TRAJECTORY_IMPEDANCE"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _selected_ip(args) -> tuple[str, str]:
    if args.arm == "left":
        return "left", args.left_ip
    return "right", args.right_ip


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


def _check_expected_tool(robot: Arm, expected_tool: str) -> None:
    if not expected_tool:
        return
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    current_name = _current_tool_name((code, frame))
    print(f"current_tool={current_name}")
    if expected_tool not in current_name:
        raise RuntimeError(f"expected active tool containing '{expected_tool}', got '{current_name}'")


def _normalize_angle(value: float) -> float:
    while value > math.pi:
        value -= 2.0 * math.pi
    while value < -math.pi:
        value += 2.0 * math.pi
    return value


def _angle_delta(current: float, previous: float) -> float:
    return _normalize_angle(current - previous)


def _normalize_pose(pose: Iterable[float]) -> list[float]:
    values = finite_list(pose, expected_len=6, label="pose")
    return values[:3] + [_normalize_angle(value) for value in values[3:]]


def _pose_delta(a: list[float], b: list[float]) -> tuple[float, float]:
    linear = norm(a[index] - b[index] for index in range(3))
    angular = norm(_angle_delta(a[index], b[index]) for index in range(3, 6))
    return linear, angular


def _joint_delta_deg(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


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
    return {
        "metadata": data,
        "samples": cleaned,
        "duration_sec": cleaned[-1]["t_sec"],
    }


def _unwrap_poses(samples: list[dict[str, object]]) -> list[list[float]]:
    unwrapped = [list(samples[0]["pose"])]
    for index in range(1, len(samples)):
        previous = unwrapped[-1]
        raw = list(samples[index]["pose"])
        pose = raw[:3] + [
            previous[axis] + _angle_delta(raw[axis], samples[index - 1]["pose"][axis])
            for axis in range(3, 6)
        ]
        unwrapped.append(pose)
    return unwrapped


def _trajectory_range(poses: list[list[float]]) -> tuple[list[float], float, float]:
    ranges = [max(pose[index] for pose in poses) - min(pose[index] for pose in poses) for index in range(6)]
    linear_range = norm(ranges[:3])
    angular_range = norm(ranges[3:])
    return ranges, linear_range, angular_range


def _interpolate_pose(samples: list[dict[str, object]], poses: list[list[float]], t_sec: float) -> list[float]:
    if t_sec <= 0.0:
        return _normalize_pose(poses[0])
    if t_sec >= samples[-1]["t_sec"]:
        return _normalize_pose(poses[-1])
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
    pose = [poses[lo][index] + (poses[hi][index] - poses[lo][index]) * alpha for index in range(6)]
    return _normalize_pose(pose)


def _apply_pose_base_offset(pose: list[float], base_offset: list[float]) -> list[float]:
    out = [pose[index] + base_offset[index] for index in range(6)]
    return _normalize_pose(out)


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


def _check_start_pose(current_pose: list[float], first_pose: list[float], args) -> tuple[float, float]:
    linear, angular = _pose_delta(current_pose, first_pose)
    print(f"start_pose_error={linear:.5f}m/{angular:.5f}rad")
    if not args.relative_to_current and not args.move_to_start:
        if linear > args.max_start_pose_linear_m:
            raise RuntimeError(f"current pose is {linear:.5f}m from first trajectory pose; move to start or use --relative-to-current")
        if angular > args.max_start_pose_angular_rad:
            raise RuntimeError(f"current pose is {angular:.5f}rad from first trajectory pose; move to start or use --relative-to-current")
    return linear, angular


def _enforce_start_pose(current_pose: list[float], first_pose: list[float], args) -> tuple[float, float]:
    linear, angular = _pose_delta(current_pose, first_pose)
    print(f"post_start_move_error={linear:.5f}m/{angular:.5f}rad")
    if linear > args.max_start_pose_linear_m:
        raise RuntimeError(f"post-move start pose is {linear:.5f}m from first trajectory pose")
    if angular > args.max_start_pose_angular_rad:
        raise RuntimeError(f"post-move start pose is {angular:.5f}rad from first trajectory pose")
    return linear, angular


def _smoothstep(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _interpolate_between_poses(start_pose: list[float], target_pose: list[float], alpha: float) -> list[float]:
    smoothed = _smoothstep(alpha)
    out = [
        start_pose[index] + (target_pose[index] - start_pose[index]) * smoothed
        for index in range(3)
    ]
    out.extend(
        _normalize_angle(start_pose[index] + _angle_delta(target_pose[index], start_pose[index]) * smoothed)
        for index in range(3, 6)
    )
    return out


def _move_to_start_pose(
    robot: Arm,
    side: str,
    current_state: dict[str, list[float]],
    target_pose: list[float],
    args,
) -> dict[str, object]:
    linear, angular = _pose_delta(current_state["pose"], target_pose)
    if linear > args.start_move_max_linear_m:
        raise RuntimeError(f"start move linear distance {linear:.5f}m > --start-move-max-linear-m")
    if angular > args.start_move_max_angular_rad:
        raise RuntimeError(f"start move angular distance {angular:.5f}rad > --start-move-max-angular-rad")
    if linear <= args.max_start_pose_linear_m and angular <= args.max_start_pose_angular_rad:
        print(f"{side}: already close to first trajectory pose; skipping start move")
        return {
            "skipped": True,
            "initial_error_linear_m": linear,
            "initial_error_angular_rad": angular,
        }

    duration = args.start_move_sec
    if args.start_move_max_linear_velocity_m_s > 0.0:
        duration = max(duration, linear / args.start_move_max_linear_velocity_m_s)
    if args.start_move_max_angular_velocity_rad_s > 0.0:
        duration = max(duration, angular / args.start_move_max_angular_velocity_rad_s)
    period = 1.0 / args.start_move_hz
    steps = max(2, int(math.ceil(duration * args.start_move_hz)))
    hold_steps = max(0, int(math.ceil(args.start_move_hold_sec * args.start_move_hz)))
    print(
        f"{side}: moving smoothly to first trajectory pose "
        f"distance={linear:.5f}m/{angular:.5f}rad duration={duration:.3f}s "
        f"steps={steps} hold_steps={hold_steps}"
    )

    records = []
    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_joint_delta = 0.0
    max_tracking_linear = 0.0
    max_tracking_angular = 0.0
    missed = 0
    next_deadline = time.perf_counter()
    motion_started = False
    last_state = current_state
    try:
        for index in range(steps):
            loop_start = time.perf_counter()
            alpha = (index + 1) / steps
            command_pose = _interpolate_between_poses(current_state["pose"], target_pose, alpha)
            _require_zero(
                robot.Movep_CANFD(
                    command_pose,
                    follow=False,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                ),
                f"{side} start Movep_CANFD",
            )
            motion_started = True

            state_sampled = (
                index % args.start_move_state_read_every == 0
                or index + 1 == steps
                or index % args.start_move_force_read_every == 0
            )
            if state_sampled:
                last_state = _read_state(robot, side)
            state = last_state
            source = [0.0] * 6
            force_norm = 0.0
            torque_norm = 0.0
            if index % args.start_move_force_read_every == 0 or index + 1 == steps:
                readings = _read_force(robot, side)
                source = readings[args.force_source]
                force_norm = norm(source[:3])
                torque_norm = norm(source[3:])
                if force_norm > args.hard_stop_force_n:
                    raise RuntimeError(f"start move force {force_norm:.4f}N > --hard-stop-force-n")
                if torque_norm > args.hard_stop_torque_nm:
                    raise RuntimeError(f"start move torque {torque_norm:.4f}Nm > --hard-stop-torque-nm")

            tracking_linear, tracking_angular = _pose_delta(state["pose"], command_pose)
            joint_delta = _joint_delta_deg(current_state["joint_deg"], state["joint_deg"])
            if joint_delta > args.max_joint_delta_deg:
                raise RuntimeError(f"start move joint delta {joint_delta:.4f}deg > --max-joint-delta-deg")
            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_tracking_linear = max(max_tracking_linear, tracking_linear)
            max_tracking_angular = max(max_tracking_angular, tracking_angular)

            loop_elapsed = time.perf_counter() - loop_start
            missed_deadline = loop_elapsed > period
            if missed_deadline:
                missed += 1
            if index % args.print_every == 0 or index + 1 == steps:
                print(
                    f"{side}:start_move step={index + 1}/{steps} "
                    f"alpha={alpha:.3f} track={tracking_linear:.4f}m/{tracking_angular:.4f}rad "
                    f"|F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm "
                    f"joint_delta={joint_delta:.3f}deg"
                )
            records.append(
                {
                    "step": index + 1,
                    "phase": "move",
                    "alpha": alpha,
                    "command_pose": command_pose,
                    "observed_pose": state["pose"],
                    "observed_joint_deg": state["joint_deg"],
                    "force_source_wrench": source,
                    "tracking_linear_m": tracking_linear,
                    "tracking_angular_rad": tracking_angular,
                    "joint_delta_deg": joint_delta,
                    "loop_elapsed_sec": loop_elapsed,
                    "missed_deadline": missed_deadline,
                }
            )

            next_deadline += period
            sleep_sec = next_deadline - time.perf_counter()
            if sleep_sec > 0.0:
                time.sleep(sleep_sec)

        for hold_index in range(hold_steps):
            loop_start = time.perf_counter()
            _require_zero(
                robot.Movep_CANFD(
                    target_pose,
                    follow=False,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                ),
                f"{side} start hold Movep_CANFD",
            )
            state_sampled = (
                hold_index % args.start_move_state_read_every == 0
                or hold_index + 1 == hold_steps
                or hold_index % args.start_move_force_read_every == 0
            )
            if state_sampled:
                last_state = _read_state(robot, side)
            state = last_state
            source = [0.0] * 6
            force_norm = 0.0
            torque_norm = 0.0
            if hold_index % args.start_move_force_read_every == 0 or hold_index + 1 == hold_steps:
                readings = _read_force(robot, side)
                source = readings[args.force_source]
                force_norm = norm(source[:3])
                torque_norm = norm(source[3:])
                if force_norm > args.hard_stop_force_n:
                    raise RuntimeError(f"start move hold force {force_norm:.4f}N > --hard-stop-force-n")
                if torque_norm > args.hard_stop_torque_nm:
                    raise RuntimeError(f"start move hold torque {torque_norm:.4f}Nm > --hard-stop-torque-nm")
            tracking_linear, tracking_angular = _pose_delta(state["pose"], target_pose)
            joint_delta = _joint_delta_deg(current_state["joint_deg"], state["joint_deg"])
            if joint_delta > args.max_joint_delta_deg:
                raise RuntimeError(f"start move hold joint delta {joint_delta:.4f}deg > --max-joint-delta-deg")
            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_tracking_linear = max(max_tracking_linear, tracking_linear)
            max_tracking_angular = max(max_tracking_angular, tracking_angular)
            loop_elapsed = time.perf_counter() - loop_start
            missed_deadline = loop_elapsed > period
            if missed_deadline:
                missed += 1
            if hold_index % args.print_every == 0 or hold_index + 1 == hold_steps:
                print(
                    f"{side}:start_hold step={hold_index + 1}/{hold_steps} "
                    f"track={tracking_linear:.4f}m/{tracking_angular:.4f}rad "
                    f"|F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm "
                    f"joint_delta={joint_delta:.3f}deg"
                )
            records.append(
                {
                    "step": steps + hold_index + 1,
                    "phase": "hold",
                    "alpha": 1.0,
                    "command_pose": target_pose,
                    "observed_pose": state["pose"],
                    "observed_joint_deg": state["joint_deg"],
                    "force_source_wrench": source,
                    "tracking_linear_m": tracking_linear,
                    "tracking_angular_rad": tracking_angular,
                    "joint_delta_deg": joint_delta,
                    "loop_elapsed_sec": loop_elapsed,
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
    final_linear, final_angular = _pose_delta(final_state["pose"], target_pose)
    print(f"{side}: start move final error={final_linear:.5f}m/{final_angular:.5f}rad")
    return {
        "skipped": False,
        "duration_sec": duration,
        "steps": steps,
        "hold_steps": hold_steps,
        "hold_sec": args.start_move_hold_sec,
        "initial_error_linear_m": linear,
        "initial_error_angular_rad": angular,
        "final_error_linear_m": final_linear,
        "final_error_angular_rad": final_angular,
        "max_force_norm_n": max_force_norm,
        "max_torque_norm_nm": max_torque_norm,
        "max_joint_delta_deg": max_joint_delta,
        "max_tracking_linear_m": max_tracking_linear,
        "max_tracking_angular_rad": max_tracking_angular,
        "missed_deadlines": missed,
        "records": records,
    }


def _run_replay(
    robot: Arm,
    side: str,
    start_state: dict[str, list[float]],
    traj: dict[str, object],
    unwrapped_poses: list[list[float]],
    tare: list[float],
    args,
) -> dict[str, object]:
    samples = traj["samples"]
    duration = float(traj["duration_sec"]) / args.speed_scale
    period = 1.0 / args.control_hz
    loops = max(1, int(math.ceil(duration * args.control_hz)))
    first_pose = _normalize_pose(samples[0]["pose"])
    base_offset = [0.0] * 6
    if args.relative_to_current:
        base_offset = [
            start_state["pose"][index] - first_pose[index] if index < 3 else _angle_delta(start_state["pose"][index], first_pose[index])
            for index in range(6)
        ]
        print(f"{side}: relative_to_current base_offset={_format(base_offset, precision=5)}")

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
    timing = {
        "force_read": [],
        "state_read": [],
        "compute": [],
        "command": [],
        "loop_elapsed": [],
        "loop_interval": [],
    }
    max_force_norm = 0.0
    max_torque_norm = 0.0
    max_linear_offset = 0.0
    max_angular_offset = 0.0
    max_tracking_linear = 0.0
    max_tracking_angular = 0.0
    max_joint_delta = 0.0
    missed = 0
    last_state = start_state
    previous_loop_start = None
    run_start = time.perf_counter()
    next_deadline = run_start
    motion_started = False
    print(
        f"{side}: replay impedance duration={duration:.3f}s hz={args.control_hz:.2f} "
        f"speed_scale={args.speed_scale:.3f} axes={args.axes} "
        f"angular_sign={args.angular_force_sign:+.1f} angular_axis_signs={_format(args.angular_axis_signs, precision=0)}"
    )
    try:
        for index in range(loops):
            loop_start = time.perf_counter()
            if previous_loop_start is not None:
                timing["loop_interval"].append(loop_start - previous_loop_start)
            previous_loop_start = loop_start
            elapsed = loop_start - run_start
            trajectory_t = min(elapsed * args.speed_scale, float(traj["duration_sec"]))
            ref_pose = _interpolate_pose(samples, unwrapped_poses, trajectory_t)
            ref_pose = _apply_pose_base_offset(ref_pose, base_offset)

            force_start = time.perf_counter()
            readings = _read_force(robot, side)
            force_read_sec = time.perf_counter() - force_start
            timing["force_read"].append(force_read_sec)

            compute_start = time.perf_counter()
            processed = processor.process(readings[args.force_source])
            raw_deadbanded = list(processed["deadbanded"])
            admittance_wrench = admittance_wrench_for_arm(raw_deadbanded, side)
            update = admittance.update(admittance_wrench, period)
            command_pose = _normalize_pose(apply_pose_offset(ref_pose, update["offset"]))
            compute_sec = time.perf_counter() - compute_start
            timing["compute"].append(compute_sec)

            command_start = time.perf_counter()
            _require_zero(
                robot.Movep_CANFD(
                    command_pose,
                    follow=False,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                ),
                f"{side} Movep_CANFD",
            )
            command_sec = time.perf_counter() - command_start
            timing["command"].append(command_sec)
            motion_started = True

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
            linear_offset = float(update["linear_offset_norm_m"])
            angular_offset = float(update["angular_offset_norm_rad"])
            tracking_linear, tracking_angular = _pose_delta(state["pose"], ref_pose)
            joint_delta = _joint_delta_deg(start_state["joint_deg"], state["joint_deg"])
            max_force_norm = max(max_force_norm, force_norm)
            max_torque_norm = max(max_torque_norm, torque_norm)
            max_linear_offset = max(max_linear_offset, linear_offset)
            max_angular_offset = max(max_angular_offset, angular_offset)
            max_tracking_linear = max(max_tracking_linear, tracking_linear)
            max_tracking_angular = max(max_tracking_angular, tracking_angular)
            max_joint_delta = max(max_joint_delta, joint_delta)
            if force_norm > args.hard_stop_force_n:
                raise RuntimeError(f"force {force_norm:.4f}N > --hard-stop-force-n")
            if torque_norm > args.hard_stop_torque_nm:
                raise RuntimeError(f"torque {torque_norm:.4f}Nm > --hard-stop-torque-nm")
            if joint_delta > args.max_joint_delta_deg:
                raise RuntimeError(f"joint delta {joint_delta:.4f}deg > --max-joint-delta-deg")
            if tracking_linear > args.max_tracking_linear_m:
                raise RuntimeError(f"tracking linear error {tracking_linear:.5f}m > --max-tracking-linear-m")
            if tracking_angular > args.max_tracking_angular_rad:
                raise RuntimeError(f"tracking angular error {tracking_angular:.5f}rad > --max-tracking-angular-rad")

            loop_elapsed = time.perf_counter() - loop_start
            timing["loop_elapsed"].append(loop_elapsed)
            missed_deadline = loop_elapsed > period
            if missed_deadline:
                missed += 1
            if index % args.print_every == 0 or index + 1 == loops:
                print(
                    f"{side}:traj step={index + 1}/{loops} t={trajectory_t:.3f}s "
                    f"offset={_format(update['offset'], precision=4)} "
                    f"track={tracking_linear:.4f}m/{tracking_angular:.4f}rad "
                    f"|F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm "
                    f"loop_ms={loop_elapsed * 1000.0:.2f}"
                )
            records.append(
                {
                    "step": index + 1,
                    "time_sec": elapsed,
                    "trajectory_t_sec": trajectory_t,
                    "reference_pose": ref_pose,
                    "command_pose": command_pose,
                    "observed_pose": state["pose"],
                    "observed_joint_deg": state["joint_deg"],
                    "tare_corrected_wrench": processed["tare_corrected"],
                    "filtered_wrench": processed["filtered"],
                    "raw_deadbanded_wrench": raw_deadbanded,
                    "admittance_wrench": admittance_wrench,
                    "active_wrench": update["active_wrench"],
                    "target_offset": update["target_offset"],
                    "offset": update["offset"],
                    "tracking_linear_m": tracking_linear,
                    "tracking_angular_rad": tracking_angular,
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "joint_delta_deg": joint_delta,
                    "state_sampled": state_sampled,
                    "timing": {
                        "force_read_sec": force_read_sec,
                        "compute_sec": compute_sec,
                        "command_sec": command_sec,
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
        "max_tracking_linear_m": max_tracking_linear,
        "max_tracking_angular_rad": max_tracking_angular,
        "max_joint_delta_deg": max_joint_delta,
        "timing": {key: _stats_ms(values) for key, values in timing.items() if values},
    }
    print(f"{side}: replay summary={summary}")
    return {
        "records": records,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a recorded RM65 teach trajectory with outer-loop admittance offset.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--expected-tool", default="inspire")
    parser.add_argument("--input-json", default="hardware/rm65/utils/data/trajectories/rm65_left_teach_trajectory.json")
    parser.add_argument("--output-json", default="hardware/rm65/utils/results/rm65_left_teach_trajectory_impedance_replay.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must be exactly {CONFIRM_TEXT} for motion.")
    parser.add_argument("--relative-to-current", action="store_true")
    parser.add_argument("--move-to-start", action="store_true")
    parser.add_argument("--start-move-hz", type=float, default=20.0)
    parser.add_argument("--start-move-sec", type=float, default=6.0)
    parser.add_argument("--start-move-hold-sec", type=float, default=1.5)
    parser.add_argument("--start-move-max-linear-m", type=float, default=0.60)
    parser.add_argument("--start-move-max-angular-rad", type=float, default=2.50)
    parser.add_argument("--start-move-max-linear-velocity-m-s", type=float, default=0.05)
    parser.add_argument("--start-move-max-angular-velocity-rad-s", type=float, default=0.40)
    parser.add_argument("--start-move-state-read-every", type=int, default=1)
    parser.add_argument("--start-move-force-read-every", type=int, default=2)
    parser.add_argument("--start-move-settle-sec", type=float, default=0.8)
    parser.add_argument("--speed-scale", type=float, default=0.50)
    parser.add_argument("--control-hz", type=float, default=40.0)
    parser.add_argument("--force-source", choices=("zero", "work", "tool"), default="work")
    parser.add_argument("--axes", default="xyz")
    parser.add_argument("--force-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--angular-force-sign", type=float, choices=(-1.0, 1.0), default=-1.0)
    parser.add_argument(
        "--angular-axis-signs",
        default="1,1,-1",
        help="Per-axis rotational signs for rx,ry,rz, applied after --angular-force-sign.",
    )
    parser.add_argument("--tare-sec", type=float, default=1.0)
    parser.add_argument("--lowpass-alpha", type=float, default=0.85)
    parser.add_argument("--force-deadband-n", type=float, default=0.30)
    parser.add_argument("--torque-deadband-nm", type=float, default=0.03)
    parser.add_argument("--linear-gain-m-s-per-n", type=float, default=0.005)
    parser.add_argument("--angular-gain-rad-s-per-nm", type=float, default=0.0)
    parser.add_argument("--linear-compliance-m-per-n", type=float, default=0.004)
    parser.add_argument("--angular-compliance-rad-per-nm", type=float, default=0.0)
    parser.add_argument("--max-linear-velocity-m-s", type=float, default=0.04)
    parser.add_argument("--max-angular-velocity-rad-s", type=float, default=0.15)
    parser.add_argument("--max-linear-offset-m", type=float, default=0.05)
    parser.add_argument("--max-angular-offset-rad", type=float, default=0.4)
    parser.add_argument("--return-rate-s", type=float, default=0.35)
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--state-read-every", type=int, default=1)
    parser.add_argument("--hard-stop-force-n", type=float, default=50.0)
    parser.add_argument("--hard-stop-torque-nm", type=float, default=3.0)
    parser.add_argument("--max-start-pose-linear-m", type=float, default=0.04)
    parser.add_argument("--max-start-pose-angular-rad", type=float, default=0.20)
    parser.add_argument("--max-trajectory-linear-range-m", type=float, default=0.45)
    parser.add_argument("--max-trajectory-angular-range-rad", type=float, default=7.0)
    parser.add_argument("--max-tracking-linear-m", type=float, default=0.12)
    parser.add_argument("--max-tracking-angular-rad", type=float, default=0.50)
    parser.add_argument("--max-joint-delta-deg", type=float, default=70.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--stop-settle-sec", type=float, default=0.5)
    return parser


def _validate_args(args) -> None:
    args.enabled_axes = parse_axes(args.axes, allow_rotation=True)
    args.angular_axis_signs = parse_signs(args.angular_axis_signs, expected_len=3, label="angular-axis-signs")
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"trajectory impedance replay requires --confirm {CONFIRM_TEXT}")
    if args.relative_to_current and args.move_to_start:
        raise RuntimeError("--relative-to-current and --move-to-start are mutually exclusive")
    if args.speed_scale <= 0.0:
        raise RuntimeError("--speed-scale must be positive")
    if args.control_hz <= 0.0 or args.control_hz > 40.0:
        raise RuntimeError("--control-hz must be in (0, 40]")
    if args.start_move_hz <= 0.0 or args.start_move_hz > 40.0:
        raise RuntimeError("--start-move-hz must be in (0, 40]")
    if args.start_move_state_read_every < 1:
        raise RuntimeError("--start-move-state-read-every must be >= 1")
    if args.start_move_force_read_every < 1:
        raise RuntimeError("--start-move-force-read-every must be >= 1")
    if args.tare_sec <= 0.0:
        raise RuntimeError("--tare-sec must be positive")
    if args.state_read_every < 1:
        raise RuntimeError("--state-read-every must be >= 1")
    if args.print_every < 1:
        raise RuntimeError("--print-every must be >= 1")
    if not 0.0 < args.lowpass_alpha <= 1.0:
        raise RuntimeError("--lowpass-alpha must be in (0, 1]")
    for name in (
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
        "max_start_pose_linear_m",
        "max_start_pose_angular_rad",
        "max_trajectory_linear_range_m",
        "max_trajectory_angular_range_rad",
        "max_tracking_linear_m",
        "max_tracking_angular_rad",
        "max_joint_delta_deg",
        "start_move_sec",
        "start_move_hold_sec",
        "start_move_max_linear_m",
        "start_move_max_angular_rad",
        "start_move_max_linear_velocity_m_s",
        "start_move_max_angular_velocity_rad_s",
        "start_move_settle_sec",
    ):
        if getattr(args, name) < 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be non-negative")


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
    ranges, linear_range, angular_range = _trajectory_range(unwrapped_poses)
    print(
        f"trajectory samples={len(samples)} duration={traj['duration_sec']:.3f}s "
        f"range={_format(ranges, precision=4)} linear={linear_range:.4f}m angular={angular_range:.4f}rad"
    )
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
        _check_expected_tool(robot, args.expected_tool)
        initial_state = _read_state(robot, side)
        print(f"{side}: current_pose={_format(initial_state['pose'])}")
        first_pose = _normalize_pose(samples[0]["pose"])
        initial_start_linear, initial_start_angular = _check_start_pose(initial_state["pose"], first_pose, args)
        if args.move_to_start:
            if initial_start_linear > args.start_move_max_linear_m:
                raise RuntimeError(
                    f"start move linear distance {initial_start_linear:.5f}m > --start-move-max-linear-m"
                )
            if initial_start_angular > args.start_move_max_angular_rad:
                raise RuntimeError(
                    f"start move angular distance {initial_start_angular:.5f}rad > --start-move-max-angular-rad"
                )
        start_move_report = {
            "requested": bool(args.move_to_start),
            "executed": False,
            "initial_error_linear_m": initial_start_linear,
            "initial_error_angular_rad": initial_start_angular,
        }
        start_state = initial_state
        if args.execute and args.move_to_start:
            start_move_report = _move_to_start_pose(robot, side, initial_state, first_pose, args)
            start_move_report["requested"] = True
            start_move_report["executed"] = True
            start_state = _read_state(robot, side)
            _enforce_start_pose(start_state["pose"], first_pose, args)
        elif not args.execute and args.move_to_start:
            print("DRY-RUN: --move-to-start is requested; start move will be sent only with --execute.")
        elif not args.relative_to_current:
            _enforce_start_pose(start_state["pose"], first_pose, args)

        tare = _sample_tare(robot, side, args)
        report: dict[str, object] = {
            "schema_version": 1,
            "generated_by": "hardware/rm65/utils/scripts/rm65_replay_teach_trajectory_impedance.py",
            "arm": side,
            "ip": ip,
            "input_json": args.input_json,
            "executed": bool(args.execute),
            "relative_to_current": args.relative_to_current,
            "move_to_start": args.move_to_start,
            "speed_scale": args.speed_scale,
            "control_hz": args.control_hz,
            "axes": args.axes,
            "enabled_axes": args.enabled_axes,
            "trajectory": {
                "samples": len(samples),
                "duration_sec": traj["duration_sec"],
                "range": ranges,
                "linear_range_m": linear_range,
                "angular_range_rad": angular_range,
            },
            "initial_state": initial_state,
            "start_state": start_state,
            "start_move": start_move_report,
            "tare": tare,
            "parameters": {
                "force_source": args.force_source,
                "force_sign": args.force_sign,
                "angular_force_sign": args.angular_force_sign,
                "angular_axis_signs": args.angular_axis_signs,
                "move_to_start": args.move_to_start,
                "start_move_hz": args.start_move_hz,
                "start_move_sec": args.start_move_sec,
                "start_move_hold_sec": args.start_move_hold_sec,
                "start_move_max_linear_m": args.start_move_max_linear_m,
                "start_move_max_angular_rad": args.start_move_max_angular_rad,
                "start_move_max_linear_velocity_m_s": args.start_move_max_linear_velocity_m_s,
                "start_move_max_angular_velocity_rad_s": args.start_move_max_angular_velocity_rad_s,
                "lowpass_alpha": args.lowpass_alpha,
                "force_deadband_n": args.force_deadband_n,
                "torque_deadband_nm": args.torque_deadband_nm,
                "linear_gain_m_s_per_n": args.linear_gain_m_s_per_n,
                "angular_gain_rad_s_per_nm": args.angular_gain_rad_s_per_nm,
                "linear_compliance_m_per_n": args.linear_compliance_m_per_n,
                "angular_compliance_rad_per_nm": args.angular_compliance_rad_per_nm,
                "max_linear_velocity_m_s": args.max_linear_velocity_m_s,
                "max_angular_velocity_rad_s": args.max_angular_velocity_rad_s,
                "max_linear_offset_m": args.max_linear_offset_m,
                "max_angular_offset_rad": args.max_angular_offset_rad,
                "trajectory_mode": args.trajectory_mode,
                "radio": args.radio,
                "state_read_every": args.state_read_every,
            },
        }
        if not args.execute:
            print("DRY-RUN: trajectory impedance commands will not be sent.")
            _write_report(args.output_json, report)
            return 0
        result = _run_replay(robot, side, start_state, traj, unwrapped_poses, tare, args)
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
