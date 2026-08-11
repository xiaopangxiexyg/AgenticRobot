from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "TEST_RM65_NATIVE_FORCE_POSITION_PASSTHROUGH"
FORCE_TRACK_CONFIRM_TEXT = "TEST_RM65_NATIVE_FORCE_TRACK"
SPRING_AXIS_CONFIRM_TEXT = "TEST_RM65_NATIVE_SPRING_AXIS"
SPRING_AXIS_PRELOAD_CONFIRM_TEXT = "TEST_RM65_NATIVE_SPRING_AXIS_WITH_STATIC_CABLE_PRELOAD"
FLOAT_AXIS_CONFIRM_TEXT = "TEST_RM65_NATIVE_FLOAT_AXIS"
FLOAT_AXIS_PRELOAD_CONFIRM_TEXT = "TEST_RM65_NATIVE_FLOAT_AXIS_WITH_STATIC_CABLE_PRELOAD"
AXES = ("x", "y", "z", "rx", "ry", "rz")
FORCE_CAPABLE_TYPES = {"6F", "6FB", "6FB-V"}
COMPLIANT_PROBE_MODES = {"force-track", "spring-axis", "float-axis"}


def _finite_list(values: Iterable[float], *, length: int, label: str) -> list[float]:
    result = [float(value) for value in values]
    if len(result) != length:
        raise RuntimeError(f"{label}: expected {length} values, got {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"{label}: non-finite values {result}")
    return result


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _wrench_delta_metrics(
    current: Iterable[float],
    baseline: Iterable[float],
    *,
    axis_index: int,
) -> dict[str, float]:
    current_values = _finite_list(current, length=6, label="current wrench")
    baseline_values = _finite_list(baseline, length=6, label="baseline wrench")
    delta = [
        current_value - baseline_value
        for current_value, baseline_value in zip(current_values, baseline_values)
    ]
    force_delta_norm = _norm(delta[:3])
    torque_delta_norm = _norm(delta[3:])
    selected_delta = float(delta[axis_index])
    selected_group_norm = force_delta_norm if axis_index < 3 else torque_delta_norm
    selected_axis_ratio = (
        abs(selected_delta) / selected_group_norm
        if selected_group_norm > 1e-9
        else 1.0
    )
    return {
        "force_delta_norm_n": force_delta_norm,
        "torque_delta_norm_nm": torque_delta_norm,
        "selected_axis_delta": selected_delta,
        "selected_axis_ratio": selected_axis_ratio,
    }


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _connect(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect to {ip}")
    return robot


def _read_state(robot: Arm, side: str) -> dict[str, list[float]]:
    code, joint, pose, arm_error, system_error = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_error) != 0 or int(system_error) != 0:
        raise RuntimeError(
            f"{side}: controller reports arm_error={arm_error}, system_error={system_error}"
        )
    return {
        "joint_deg": _finite_list(joint, length=6, label="joint_deg"),
        "pose_xyz_rpy": _finite_list(pose, length=6, label="pose_xyz_rpy"),
    }


def _read_force(robot: Arm, side: str) -> dict[str, list[float]]:
    code, raw, zero, work, tool = robot.Get_Force_Data()
    _require_zero(code, f"{side} Get_Force_Data")
    return {
        "raw": _finite_list(raw, length=6, label="raw force"),
        "zero": _finite_list(zero, length=6, label="zero force"),
        "work": _finite_list(work, length=6, label="work force"),
        "tool": _finite_list(tool, length=6, label="tool force"),
    }


def _sample_force_baseline(
    robot: Arm,
    side: str,
    *,
    frame: str,
    duration_sec: float,
    hz: float,
) -> dict[str, object]:
    period_sec = 1.0 / hz
    count = max(3, int(math.ceil(duration_sec * hz)))
    samples: list[list[float]] = []
    for index in range(count):
        started = time.monotonic()
        samples.append(_read_force(robot, side)[frame])
        remaining = period_sec - (time.monotonic() - started)
        if index + 1 < count and remaining > 0.0:
            time.sleep(remaining)
    component_median = [statistics.median(sample[axis] for sample in samples) for axis in range(6)]
    component_std = [statistics.pstdev(sample[axis] for sample in samples) for axis in range(6)]
    return {
        "frame": frame,
        "duration_sec": duration_sec,
        "requested_hz": hz,
        "sample_count": len(samples),
        "samples": samples,
        "component_median": component_median,
        "component_std": component_std,
    }


def _robot_capability(robot: Arm) -> dict[str, object]:
    code, info = robot._robot.rm_get_robot_info()
    _require_zero(code, "rm_get_robot_info")
    software_code, plan, controller, _, _, product = robot.Get_Arm_Software_Version()
    _require_zero(software_code, "Get_Arm_Software_Version")
    return {
        "api_version": robot.API_Version(),
        "arm_dof": int(info.get("arm_dof", 0)),
        "arm_model": str(info.get("arm_model", "")),
        "force_type": str(info.get("force_type", "")),
        "robot_controller_version": int(info.get("robot_controller_version", 0)),
        "planning_software_version": str(plan),
        "controller_software_version": str(controller),
        "product_version": str(product),
    }


def _validate_capability(capability: dict[str, object]) -> None:
    if capability["arm_model"] != "RM_65" or capability["arm_dof"] != 6:
        raise RuntimeError(f"probe requires a 6-DoF RM65, got {capability}")
    if capability["force_type"] not in FORCE_CAPABLE_TYPES:
        raise RuntimeError(
            "probe requires an RM65 six-axis force sensor; "
            f"force_type={capability['force_type']!r}"
        )


def _load_calibration_profile(path: Path, *, side: str) -> dict[str, object]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load calibration profile {path}: {exc}") from exc
    if profile.get("data_type") != "rm65_calibration_profile":
        raise RuntimeError(f"{path} is not an RM65 calibration profile")
    if profile.get("arm") != side:
        raise RuntimeError(
            f"calibration profile arm={profile.get('arm')!r} does not match --arm {side}"
        )
    payload = profile.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError(f"calibration profile {path} has no payload object")
    mass_kg = float(payload.get("mass_kg", math.nan))
    com_m = _finite_list(payload.get("com_tool_m", []), length=3, label="profile CoM")
    if not math.isfinite(mass_kg) or mass_kg <= 0.0:
        raise RuntimeError(f"calibration profile {path} has invalid mass_kg={mass_kg}")
    return {
        "path": str(path),
        "tool_name": str(profile.get("tool_name") or profile.get("name") or ""),
        "mass_kg": mass_kg,
        "com_tool_m": com_m,
    }


def _active_tool_preflight(
    robot: Arm,
    profile: dict[str, object],
    *,
    max_payload_error_kg: float,
    max_com_error_m: float,
) -> dict[str, object]:
    code, frame = robot.Get_Current_Tool_Frame()
    _require_zero(code, "Get_Current_Tool_Frame")
    if not isinstance(frame, dict):
        raise RuntimeError(f"unexpected current tool frame: {frame}")
    current_name = str(
        frame.get("name") or frame.get("frame_name") or frame.get("tool_name") or ""
    ).strip("\x00")
    expected_name = str(profile["tool_name"])
    if not expected_name or expected_name not in current_name:
        raise RuntimeError(
            f"active controller tool {current_name!r} does not match profile tool {expected_name!r}"
        )
    current_payload = float(frame.get("payload", math.nan))
    current_com_raw = [float(frame.get(axis, math.nan)) for axis in ("x", "y", "z")]
    if not math.isfinite(current_payload) or not all(math.isfinite(value) for value in current_com_raw):
        raise RuntimeError(f"active tool lacks finite payload/CoM: {frame}")
    expected_payload = float(profile["mass_kg"])
    expected_com = list(profile["com_tool_m"])
    payload_error = abs(current_payload - expected_payload)
    candidates = {
        "m": current_com_raw,
        "mm_readback": [value / 1000.0 for value in current_com_raw],
    }
    com_unit, current_com_m = min(
        candidates.items(),
        key=lambda item: max(abs(a - b) for a, b in zip(item[1], expected_com)),
    )
    com_error = [abs(a - b) for a, b in zip(current_com_m, expected_com)]
    if payload_error > max_payload_error_kg:
        raise RuntimeError(
            f"active tool payload mismatch: current={current_payload:.6f}kg, "
            f"profile={expected_payload:.6f}kg, error={payload_error:.6f}kg"
        )
    if max(com_error) > max_com_error_m:
        raise RuntimeError(
            f"active tool CoM mismatch: current={current_com_m}, profile={expected_com}, "
            f"max_error={max(com_error):.6f}m"
        )
    return {
        "active_tool_name": current_name,
        "controller_payload_kg": current_payload,
        "profile_payload_kg": expected_payload,
        "payload_error_kg": payload_error,
        "controller_com_raw": current_com_raw,
        "controller_com_inferred_unit": com_unit,
        "controller_com_m": current_com_m,
        "profile_com_m": expected_com,
        "com_error_m": com_error,
    }
    if capability["robot_controller_version"] != 3:
        raise RuntimeError(
            "this probe is reviewed for the RM65 third-generation controller only; "
            f"got controller generation {capability['robot_controller_version']}"
        )


def _max_joint_delta_deg(reference: list[float], current: list[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(reference, current))


def _rpy_rotation_matrix(rpy: Iterable[float]) -> tuple[tuple[float, float, float], ...]:
    roll, pitch, yaw = _finite_list(rpy, length=3, label="rpy")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # RM65 pose RPY follows the conventional Rz(yaw) Ry(pitch) Rx(roll)
    # composition. Comparing the resulting rotations avoids false guard trips
    # from component-wise Euler deltas near pitch = +/- pi/2.
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _rotation_distance_deg(reference_rpy: Iterable[float], current_rpy: Iterable[float]) -> float:
    reference = _rpy_rotation_matrix(reference_rpy)
    current = _rpy_rotation_matrix(current_rpy)
    relative_trace = sum(
        reference[row][column] * current[row][column]
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, 0.5 * (relative_trace - 1.0)))
    return math.degrees(math.acos(cosine))


def _pose_delta(reference: list[float], current: list[float]) -> tuple[float, float]:
    translation = _norm(float(current[i]) - float(reference[i]) for i in range(3))
    rotation = _rotation_distance_deg(reference[3:6], current[3:6])
    return translation, rotation


def _measured_pose_rate(
    previous: list[float],
    current: list[float],
    *,
    elapsed_sec: float,
) -> tuple[float, float]:
    if not math.isfinite(elapsed_sec) or elapsed_sec <= 0.0:
        raise RuntimeError(f"measured pose rate requires positive elapsed_sec, got {elapsed_sec}")
    translation, rotation = _pose_delta(previous, current)
    return translation / elapsed_sec, rotation / elapsed_sec


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _emergency_stop_worker(
    *,
    side: str,
    ip: str,
    parent_pid: int,
    ready_event,
    armed_event,
    stop_request_event,
    normal_shutdown_event,
    receipt_queue,
) -> None:
    signal_stop_requested = False

    def _request_stop_from_process_signal(_signum, _frame) -> None:
        nonlocal signal_stop_requested
        # Python signal handlers must not acquire multiprocessing locks. In
        # particular, Event.set() can deadlock when SIGINT interrupts this
        # process inside Event.wait(). The worker observes this local latch on
        # its next bounded poll and calls the SDK from normal execution context.
        signal_stop_requested = True

    # Ctrl-C is normally delivered to the whole foreground process group. Let
    # this independent process request Arm_Stop even if the parent is blocked
    # inside an SDK call and has not run its Python signal handler yet.
    signal.signal(signal.SIGINT, _request_stop_from_process_signal)
    signal.signal(signal.SIGTERM, _request_stop_from_process_signal)
    robot: Arm | None = None
    try:
        robot = _connect(side, ip)
        receipt_queue.put({"kind": "ready", "pid": os.getpid()})
        ready_event.set()
        while True:
            if normal_shutdown_event.is_set():
                return
            parent_lost = os.getppid() != parent_pid
            shared_stop_requested = stop_request_event.wait(0.02)
            if (
                signal_stop_requested
                or shared_stop_requested
                or (parent_lost and armed_event.is_set())
            ):
                if not armed_event.is_set():
                    receipt_queue.put(
                        {"kind": "stop_skipped", "reason": "force-position mode was not armed"}
                    )
                    return
                started_ns = time.monotonic_ns()
                try:
                    code = int(robot.Arm_Stop())
                    receipt_queue.put(
                        {
                            "kind": "stop_result",
                            "source": "independent_guard",
                            "code": code,
                            "elapsed_ms": (time.monotonic_ns() - started_ns) / 1e6,
                            "reason": "parent_lost" if parent_lost else "signal_or_fault",
                        }
                    )
                    cleanup_started_ns = time.monotonic_ns()
                    cleanup_code = int(robot.Stop_Force_Position_Move())
                    receipt_queue.put(
                        {
                            "kind": "force_position_cleanup_result",
                            "source": "independent_guard",
                            "code": cleanup_code,
                            "elapsed_ms": (time.monotonic_ns() - cleanup_started_ns) / 1e6,
                        }
                    )
                except BaseException as exc:
                    receipt_queue.put(
                        {
                            "kind": "stop_error",
                            "source": "independent_guard",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                return
    except BaseException as exc:
        receipt_queue.put(
            {
                "kind": "setup_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        ready_event.set()
    finally:
        if robot is not None:
            try:
                robot.Arm_Socket_Close()
            except BaseException:
                pass


def _one_shot_emergency_stop_worker(
    *,
    side: str,
    ip: str,
    receipt_queue,
) -> None:
    # This worker is spawned only after an interrupt. Ignore additional Ctrl-C
    # presses so they cannot interrupt the SDK stop call in this isolated
    # process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    robot: Arm | None = None
    try:
        robot = _connect(side, ip)
        started_ns = time.monotonic_ns()
        code = int(robot.Arm_Stop())
        receipt_queue.put(
            {
                "kind": "stop_result",
                "source": "redundant_guard",
                "code": code,
                "elapsed_ms": (time.monotonic_ns() - started_ns) / 1e6,
            }
        )
        cleanup_started_ns = time.monotonic_ns()
        cleanup_code = int(robot.Stop_Force_Position_Move())
        receipt_queue.put(
            {
                "kind": "force_position_cleanup_result",
                "source": "redundant_guard",
                "code": cleanup_code,
                "elapsed_ms": (time.monotonic_ns() - cleanup_started_ns) / 1e6,
            }
        )
    except BaseException as exc:
        receipt_queue.put(
            {
                "kind": "stop_error",
                "source": "redundant_guard",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if robot is not None:
            try:
                robot.Arm_Socket_Close()
            except BaseException:
                pass


def _queue_receipt(receipt_queue, *, timeout_sec: float) -> dict[str, object] | None:
    try:
        value = receipt_queue.get(timeout=max(0.0, timeout_sec))
    except queue.Empty:
        return None
    return dict(value)


def _drain_receipts(receipt_queue) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    while True:
        try:
            receipts.append(dict(receipt_queue.get_nowait()))
        except queue.Empty:
            return receipts


def _reap_process(process, *, grace_sec: float) -> dict[str, object]:
    process.join(timeout=max(0.0, grace_sec))
    terminated = False
    killed = False
    if process.is_alive():
        terminated = True
        process.terminate()
        process.join(timeout=0.20)
    if process.is_alive():
        killed = True
        process.kill()
        process.join(timeout=0.20)
    return {
        "pid": process.pid,
        "exitcode": process.exitcode,
        "terminated_after_timeout": terminated,
        "killed_after_timeout": killed,
        "still_alive": process.is_alive(),
    }


def _start_emergency_guard(side: str, ip: str, *, timeout_sec: float):
    context = mp.get_context("spawn")
    ready_event = context.Event()
    armed_event = context.Event()
    stop_request_event = context.Event()
    normal_shutdown_event = context.Event()
    receipt_queue = context.Queue()
    process = context.Process(
        target=_emergency_stop_worker,
        kwargs={
            "side": side,
            "ip": ip,
            "parent_pid": os.getpid(),
            "ready_event": ready_event,
            "armed_event": armed_event,
            "stop_request_event": stop_request_event,
            "normal_shutdown_event": normal_shutdown_event,
            "receipt_queue": receipt_queue,
        },
        name=f"rm65-{side}-emergency-stop-guard",
        daemon=True,
    )
    process.start()
    if not ready_event.wait(timeout_sec):
        normal_shutdown_event.set()
        raise RuntimeError("independent emergency-stop guard did not become ready")
    receipt = _queue_receipt(receipt_queue, timeout_sec=0.5)
    if receipt is None or receipt.get("kind") != "ready":
        normal_shutdown_event.set()
        raise RuntimeError(f"independent emergency-stop guard setup failed: {receipt}")
    return {
        "process": process,
        "armed": armed_event,
        "stop_request": stop_request_event,
        "normal_shutdown": normal_shutdown_event,
        "receipts": receipt_queue,
        "ready_receipt": receipt,
    }


def _request_emergency_stop(
    side: str,
    ip: str,
    guard: dict[str, object],
    *,
    timeout_sec: float,
) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    guard["stop_request"].set()
    receipt = _queue_receipt(guard["receipts"], timeout_sec=timeout_sec)
    if receipt is not None:
        receipts.append(receipt)
    # Always issue a second stop after the mode-start call has returned. Run it
    # in another process so an SDK timeout cannot block Ctrl-C handling in the
    # parent process.
    context = mp.get_context("spawn")
    redundant_receipts = context.Queue()
    redundant_process = context.Process(
        target=_one_shot_emergency_stop_worker,
        kwargs={"side": side, "ip": ip, "receipt_queue": redundant_receipts},
        name=f"rm65-{side}-redundant-emergency-stop-guard",
        daemon=True,
    )
    redundant_process.start()
    redundant_receipt = _queue_receipt(redundant_receipts, timeout_sec=timeout_sec)
    if redundant_receipt is not None:
        receipts.append(redundant_receipt)
    else:
        receipts.append(
            {
                "kind": "stop_timeout",
                "source": "redundant_guard",
                "timeout_sec": timeout_sec,
            }
        )
    redundant_reap = _reap_process(redundant_process, grace_sec=timeout_sec)
    receipts.extend(_drain_receipts(redundant_receipts))
    receipts.append(
        {
            "kind": "worker_reap",
            "source": "redundant_guard",
            **redundant_reap,
        }
    )
    return receipts


def _stop_acknowledged(receipts: list[dict[str, object]]) -> bool:
    return any(item.get("kind") == "stop_result" and item.get("code") == 0 for item in receipts)


def _control_vectors(
    args,
) -> tuple[list[int], list[float], list[float]]:
    control_mode = [3] * 6
    desired_force = [0.0] * 6
    limit_velocity = [0.0] * 6
    if args.probe_mode == "transport-only":
        return control_mode, desired_force, limit_velocity

    axis_index = AXES.index(args.force_axis)
    if args.probe_mode == "force-track":
        control_mode[axis_index] = 4
        # Validation requires an explicit contact-force target. A free-space
        # sensor residual is not a physical equilibrium and must never be
        # promoted to a constant-force command implicitly.
        desired_force[axis_index] = float(args.desired_force)
    elif args.probe_mode == "spring-axis":
        control_mode[axis_index] = 2
    elif args.probe_mode == "float-axis":
        # Diagnostic-only comparison against spring-axis. Float mode is not a
        # production recommendation: after external force is removed it does
        # not return to the pre-force pose. Tight drift guards are mandatory.
        control_mode[axis_index] = 1
    else:  # pragma: no cover - parser and validation own the public choices.
        raise RuntimeError(f"unsupported probe mode: {args.probe_mode}")
    # RealMan's official API example passes 0.2 for a documented 0.2 m/s
    # translational limit. The documented 0.001 m/s and 0.001 deg/s values are
    # parameter resolutions, not integer scale factors. Keep SI values intact
    # at this boundary. The CLI validation above also caps this isolated probe
    # at 0.020 m/s and 10 deg/s.
    limit_velocity[axis_index] = (
        float(args.max_force_axis_linear_speed_m_s)
        if axis_index < 3
        else float(args.max_force_axis_angular_speed_deg_s)
    )
    return control_mode, desired_force, limit_velocity


def _validate_spring_baseline(
    args,
    *,
    axis_index: int,
    baseline_component: float,
    baseline_force_norm: float,
    baseline_torque_norm: float,
) -> None:
    mode_label = str(args.probe_mode)
    if args.static_cable_preload_confirmed:
        active_axis_limit = (
            args.max_spring_active_axis_preload_n
            if axis_index < 3
            else args.max_spring_active_axis_preload_nm
        )
        active_axis_unit = "N" if axis_index < 3 else "Nm"
        if (
            baseline_force_norm > args.max_spring_static_preload_force_n
            or baseline_torque_norm > args.max_spring_static_preload_torque_nm
            or abs(baseline_component) > active_axis_limit
        ):
            raise RuntimeError(
                f"{mode_label} static cable preload exceeds the declared "
                "inactive-axis envelope: "
                f"|F|={baseline_force_norm:.3f}N/"
                f"{args.max_spring_static_preload_force_n:.3f}N, "
                f"|T|={baseline_torque_norm:.3f}Nm/"
                f"{args.max_spring_static_preload_torque_nm:.3f}Nm, "
                f"active_{args.force_axis}={baseline_component:.3f}"
                f"{active_axis_unit}/{active_axis_limit:.3f}{active_axis_unit}"
            )
        return

    if (
        baseline_force_norm > args.max_spring_free_space_force_n
        or baseline_torque_norm > args.max_spring_free_space_torque_nm
    ):
        raise RuntimeError(
            f"{mode_label} requires a healthy unloaded force baseline: "
            f"|F|={baseline_force_norm:.3f}N "
            f"> {args.max_spring_free_space_force_n:.3f}N or "
            f"|T|={baseline_torque_norm:.3f}Nm "
            f"> {args.max_spring_free_space_torque_nm:.3f}Nm; "
            "inspect cable preload/tool assembly and recalibrate instead "
            "of clearing force data automatically"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RM65 controller-native pass-through force-position probe. The default "
            "transport-only mode keeps the current qpos and enables no force axis. "
            "Ctrl+C/SIGTERM requests Arm_Stop through an independent SDK connection."
        )
    )
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--calibration-profile", type=Path, default=None)
    parser.add_argument("--max-payload-error-kg", type=float, default=0.010)
    parser.add_argument("--max-com-error-m", type=float, default=0.005)
    parser.add_argument(
        "--probe-mode",
        choices=("transport-only", "force-track", "spring-axis", "float-axis"),
        default="transport-only",
    )
    parser.add_argument("--frame", choices=("work", "tool"), default="tool")
    parser.add_argument("--force-axis", choices=AXES, default="z")
    parser.add_argument(
        "--desired-force",
        type=float,
        default=None,
        help=(
            "Explicit contact force (N) or torque (Nm) for force-track. "
            "It is forbidden for transport-only and spring-axis."
        ),
    )
    parser.add_argument(
        "--contact-fixture-confirmed",
        action="store_true",
        help=(
            "Declare that force-track starts against a known compliant contact "
            "fixture. Free-space force tracking is rejected."
        ),
    )
    parser.add_argument(
        "--static-cable-preload-confirmed",
        action="store_true",
        help=(
            "Allow spring-axis or float-axis testing with a stable cable preload "
            "on inactive axes. The selected compliant axis must still have low preload."
        ),
    )
    parser.add_argument("--max-force-axis-linear-speed-m-s", type=float, default=0.003)
    parser.add_argument("--max-force-axis-angular-speed-deg-s", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--state-hz", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=5.0)
    parser.add_argument("--baseline-duration-sec", type=float, default=0.6)
    parser.add_argument("--baseline-hz", type=float, default=20.0)
    parser.add_argument("--max-baseline-force-std-n", type=float, default=0.15)
    parser.add_argument("--max-baseline-torque-std-nm", type=float, default=0.02)
    parser.add_argument("--max-spring-free-space-force-n", type=float, default=0.80)
    parser.add_argument("--max-spring-free-space-torque-nm", type=float, default=0.08)
    parser.add_argument("--max-spring-static-preload-force-n", type=float, default=2.50)
    parser.add_argument("--max-spring-static-preload-torque-nm", type=float, default=0.10)
    parser.add_argument("--max-spring-active-axis-preload-n", type=float, default=0.35)
    parser.add_argument("--max-spring-active-axis-preload-nm", type=float, default=0.05)
    parser.add_argument("--max-desired-force-offset-n", type=float, default=0.50)
    parser.add_argument("--max-desired-torque-offset-nm", type=float, default=0.05)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--trajectory-mode", choices=(0, 1, 2), type=int, default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--hard-max-force-n", type=float, default=15.0)
    parser.add_argument("--hard-max-torque-nm", type=float, default=2.0)
    parser.add_argument(
        "--max-force-delta-from-baseline-n",
        type=float,
        default=None,
        help=(
            "Optional norm limit for force change relative to the stable baseline. "
            "This supplements, and never replaces, --hard-max-force-n."
        ),
    )
    parser.add_argument(
        "--max-torque-delta-from-baseline-nm",
        type=float,
        default=None,
        help=(
            "Optional norm limit for torque change relative to the stable baseline. "
            "This supplements, and never replaces, --hard-max-torque-nm."
        ),
    )
    parser.add_argument(
        "--min-selected-axis-force-ratio",
        type=float,
        default=None,
        help=(
            "Optional minimum fraction of baseline-relative force/torque change "
            "that must lie on the selected compliant axis."
        ),
    )
    parser.add_argument(
        "--direction-check-min-delta",
        type=float,
        default=0.40,
        help=(
            "Only enforce --min-selected-axis-force-ratio above this "
            "baseline-relative force (N) or torque (Nm) magnitude."
        ),
    )
    parser.add_argument("--max-joint-drift-deg", type=float, default=None)
    parser.add_argument("--max-linear-drift-m", type=float, default=None)
    parser.add_argument("--max-angular-drift-deg", type=float, default=None)
    parser.add_argument("--max-measured-linear-speed-m-s", type=float, default=None)
    parser.add_argument("--max-measured-angular-speed-deg-s", type=float, default=None)
    parser.add_argument("--guard-ready-timeout-sec", type=float, default=5.0)
    parser.add_argument("--stop-ack-timeout-sec", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirm",
        default="",
        help=(
            f"transport-only: {CONFIRM_TEXT}; "
            f"force-track: {FORCE_TRACK_CONFIRM_TEXT}; "
            f"spring-axis: {SPRING_AXIS_CONFIRM_TEXT}; "
            "spring-axis with declared static cable preload: "
            f"{SPRING_AXIS_PRELOAD_CONFIRM_TEXT}; "
            f"float-axis: {FLOAT_AXIS_CONFIRM_TEXT}; "
            "float-axis with declared static cable preload: "
            f"{FLOAT_AXIS_PRELOAD_CONFIRM_TEXT}."
        ),
    )
    return parser


def _validate_args(args) -> None:
    required_confirmation = {
        "transport-only": CONFIRM_TEXT,
        "force-track": FORCE_TRACK_CONFIRM_TEXT,
        "spring-axis": SPRING_AXIS_CONFIRM_TEXT,
        "float-axis": FLOAT_AXIS_CONFIRM_TEXT,
    }[args.probe_mode]
    if args.probe_mode == "spring-axis" and args.static_cable_preload_confirmed:
        required_confirmation = SPRING_AXIS_PRELOAD_CONFIRM_TEXT
    if args.probe_mode == "float-axis" and args.static_cable_preload_confirmed:
        required_confirmation = FLOAT_AXIS_PRELOAD_CONFIRM_TEXT
    if not args.execute or args.confirm != required_confirmation:
        raise RuntimeError(f"execution requires --execute --confirm {required_confirmation}")
    if args.hz <= 0.0 or args.hz > 120.0:
        raise RuntimeError("--hz must be in (0, 120]")
    if args.state_hz <= 0.0 or args.state_hz > args.hz:
        raise RuntimeError("--state-hz must be in (0, --hz]")
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.baseline_duration_sec <= 0.0:
        raise RuntimeError("--baseline-duration-sec must be positive")
    if args.baseline_hz <= 0.0 or args.baseline_hz > 100.0:
        raise RuntimeError("--baseline-hz must be in (0, 100]")
    if args.radio < 0 or args.radio > 100:
        raise RuntimeError("--radio must be in [0, 100]")
    if args.max_force_axis_linear_speed_m_s > 0.020:
        raise RuntimeError("--max-force-axis-linear-speed-m-s must be <= 0.020 for this probe")
    if args.max_force_axis_angular_speed_deg_s > 10.0:
        raise RuntimeError("--max-force-axis-angular-speed-deg-s must be <= 10 for this probe")
    for name in (
        "max_force_axis_linear_speed_m_s",
        "max_force_axis_angular_speed_deg_s",
        "hard_max_force_n",
        "hard_max_torque_nm",
        "guard_ready_timeout_sec",
        "stop_ack_timeout_sec",
        "max_baseline_force_std_n",
        "max_baseline_torque_std_nm",
        "max_spring_free_space_force_n",
        "max_spring_free_space_torque_nm",
        "max_spring_static_preload_force_n",
        "max_spring_static_preload_torque_nm",
        "max_spring_active_axis_preload_n",
        "max_spring_active_axis_preload_nm",
        "max_desired_force_offset_n",
        "max_desired_torque_offset_nm",
        "max_payload_error_kg",
        "max_com_error_m",
        "direction_check_min_delta",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "max_force_delta_from_baseline_n",
        "max_torque_delta_from_baseline_nm",
    ):
        value = getattr(args, name)
        if value is not None and float(value) <= 0.0:
            raise RuntimeError(f"--{name.replace('_', '-')} must be positive")
    if args.min_selected_axis_force_ratio is not None and not (
        0.0 < float(args.min_selected_axis_force_ratio) <= 1.0
    ):
        raise RuntimeError("--min-selected-axis-force-ratio must be in (0, 1]")
    if (
        args.probe_mode not in COMPLIANT_PROBE_MODES
        and any(
            value is not None
            for value in (
                args.max_force_delta_from_baseline_n,
                args.max_torque_delta_from_baseline_nm,
                args.min_selected_axis_force_ratio,
            )
        )
    ):
        raise RuntimeError(
            "baseline-relative wrench guards require a compliant probe mode"
        )
    if args.probe_mode != "force-track" and args.desired_force is not None:
        raise RuntimeError("--desired-force is only valid with --probe-mode force-track")
    if args.probe_mode in COMPLIANT_PROBE_MODES and args.calibration_profile is None:
        raise RuntimeError(f"{args.probe_mode} requires --calibration-profile")
    if args.probe_mode == "force-track":
        if args.desired_force is None:
            raise RuntimeError(
                "force-track requires an explicit --desired-force; a measured "
                "free-space baseline is not a valid constant-force target"
            )
        if not args.contact_fixture_confirmed:
            raise RuntimeError(
                "force-track is contact-only and requires --contact-fixture-confirmed"
            )
    elif args.contact_fixture_confirmed:
        raise RuntimeError("--contact-fixture-confirmed is only valid for force-track")
    if (
        args.probe_mode not in {"spring-axis", "float-axis"}
        and args.static_cable_preload_confirmed
    ):
        raise RuntimeError(
            "--static-cable-preload-confirmed is only valid for spring-axis or float-axis"
        )
    if args.max_spring_static_preload_force_n > args.hard_max_force_n:
        raise RuntimeError(
            "--max-spring-static-preload-force-n must not exceed --hard-max-force-n"
        )
    if args.max_spring_static_preload_torque_nm > args.hard_max_torque_nm:
        raise RuntimeError(
            "--max-spring-static-preload-torque-nm must not exceed --hard-max-torque-nm"
        )
    if args.desired_force is not None:
        selected_limit = (
            args.hard_max_force_n if AXES.index(args.force_axis) < 3 else args.hard_max_torque_nm
        )
        if abs(args.desired_force) > selected_limit:
            unit = "N" if AXES.index(args.force_axis) < 3 else "Nm"
            raise RuntimeError(
                f"--desired-force magnitude must be <= {selected_limit:g}{unit} for {args.force_axis}"
            )
    optional_drift_limits = {
        "max_joint_drift_deg": 10.0,
        "max_linear_drift_m": 0.050,
        "max_angular_drift_deg": 15.0,
    }
    for name, safe_upper_bound in optional_drift_limits.items():
        value = getattr(args, name)
        if value is not None and not (0.0 < float(value) <= safe_upper_bound):
            raise RuntimeError(
                f"--{name.replace('_', '-')} must be in (0, {safe_upper_bound:g}]"
            )
    optional_speed_limits = {
        "max_measured_linear_speed_m_s": 0.100,
        "max_measured_angular_speed_deg_s": 30.0,
    }
    for name, safe_upper_bound in optional_speed_limits.items():
        value = getattr(args, name)
        if value is not None and not (0.0 < float(value) <= safe_upper_bound):
            raise RuntimeError(
                f"--{name.replace('_', '-')} must be in (0, {safe_upper_bound:g}]"
            )


def _selected_side_ip(args) -> tuple[str, str]:
    return (args.arm, args.left_ip if args.arm == "left" else args.right_ip)


def _default_output(side: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rm65_{side}_native_force_position_passthrough_{timestamp}.json"
    return REPO_ROOT / "hardware" / "rm65" / "utils" / "results" / filename


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    side, ip = _selected_side_ip(args)
    output_path = args.output_json or _default_output(side)
    result: dict[str, object] = {
        "schema": "rm65_native_force_position_passthrough_probe/v2",
        "side": side,
        "ip": ip,
        "probe_mode": args.probe_mode,
        "safety_contract": {
            "force_track_contact_only": True,
            "force_track_requires_explicit_target": True,
            "spring_axis_requires_unloaded_baseline": True,
            "spring_axis_max_free_space_force_n": args.max_spring_free_space_force_n,
            "spring_axis_max_free_space_torque_nm": (
                args.max_spring_free_space_torque_nm
            ),
            "static_cable_preload_confirmed": bool(
                args.static_cable_preload_confirmed
            ),
            "spring_axis_max_static_preload_force_n": (
                args.max_spring_static_preload_force_n
            ),
            "spring_axis_max_static_preload_torque_nm": (
                args.max_spring_static_preload_torque_nm
            ),
            "spring_axis_max_active_preload_n": (
                args.max_spring_active_axis_preload_n
            ),
            "spring_axis_max_active_preload_nm": (
                args.max_spring_active_axis_preload_nm
            ),
            "automatic_force_clear": False,
            "hard_force_guard_is_absolute": True,
            "max_force_delta_from_baseline_n": (
                args.max_force_delta_from_baseline_n
            ),
            "max_torque_delta_from_baseline_nm": (
                args.max_torque_delta_from_baseline_nm
            ),
            "min_selected_axis_force_ratio": (
                args.min_selected_axis_force_ratio
            ),
            "direction_check_min_delta": args.direction_check_min_delta,
            "max_measured_linear_speed_m_s": (
                args.max_measured_linear_speed_m_s
            ),
            "max_measured_angular_speed_deg_s": (
                args.max_measured_angular_speed_deg_s
            ),
        },
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "initializing",
        "samples": [],
        "emergency_stop_receipts": [],
    }
    robot: Arm | None = None
    guard: dict[str, object] | None = None
    force_position_active = False
    emergency_requested = False
    signal_number: int | None = None
    previous_handlers: dict[int, object] = {}
    exit_code = 1
    baseline_wrench: list[float] | None = None

    try:
        calibration_profile = (
            _load_calibration_profile(args.calibration_profile, side=side)
            if args.calibration_profile is not None
            else None
        )
        if calibration_profile is not None:
            result["calibration_profile"] = calibration_profile
        robot = _connect(side, ip)
        capability = _robot_capability(robot)
        _validate_capability(capability)
        result["capability"] = capability
        print(f"[{side}] capability={json.dumps(capability, ensure_ascii=False)}")

        if args.probe_mode in COMPLIANT_PROBE_MODES:
            tool_preflight = _active_tool_preflight(
                robot,
                calibration_profile,
                max_payload_error_kg=args.max_payload_error_kg,
                max_com_error_m=args.max_com_error_m,
            )
            result["active_tool_preflight"] = tool_preflight
            print(
                f"[{side}] active tool preflight passed name={tool_preflight['active_tool_name']} "
                f"payload={tool_preflight['controller_payload_kg']:.4f}kg"
            )

        initial_state = _read_state(robot, side)
        initial_force = _read_force(robot, side)
        result["initial_state"] = initial_state
        result["initial_force"] = initial_force
        force_source = initial_force[args.frame]
        initial_force_norm = _norm(force_source[:3])
        initial_torque_norm = _norm(force_source[3:])
        if (
            initial_force_norm > args.hard_max_force_n
            or initial_torque_norm > args.hard_max_torque_nm
        ):
            raise RuntimeError(
                f"starting {args.frame} wrench exceeds guard: "
                f"|F|={initial_force_norm:.3f}N, |T|={initial_torque_norm:.3f}Nm"
            )

        if args.probe_mode in COMPLIANT_PROBE_MODES:
            baseline = _sample_force_baseline(
                robot,
                side,
                frame=args.frame,
                duration_sec=args.baseline_duration_sec,
                hz=args.baseline_hz,
            )
            result["force_baseline"] = baseline
            axis_index = AXES.index(args.force_axis)
            axis_std = float(baseline["component_std"][axis_index])
            max_axis_std = (
                args.max_baseline_force_std_n
                if axis_index < 3
                else args.max_baseline_torque_std_nm
            )
            if axis_std > max_axis_std:
                unit = "N" if axis_index < 3 else "Nm"
                raise RuntimeError(
                    f"unstable {args.frame} {args.force_axis} baseline: "
                    f"std={axis_std:.4f}{unit} > {max_axis_std:.4f}{unit}"
                )
            baseline_wrench = list(baseline["component_median"])
            baseline_force_norm = _norm(baseline_wrench[:3])
            baseline_torque_norm = _norm(baseline_wrench[3:])
            if (
                baseline_force_norm > args.hard_max_force_n
                or baseline_torque_norm > args.hard_max_torque_nm
            ):
                raise RuntimeError(
                    f"stable baseline exceeds guard: |F|={baseline_force_norm:.3f}N, "
                    f"|T|={baseline_torque_norm:.3f}Nm"
                )
            baseline_component = float(baseline["component_median"][axis_index])
            if args.probe_mode == "force-track":
                desired_offset = abs(float(args.desired_force) - baseline_component)
                max_desired_offset = (
                    args.max_desired_force_offset_n
                    if axis_index < 3
                    else args.max_desired_torque_offset_nm
                )
                if desired_offset > max_desired_offset:
                    unit = "N" if axis_index < 3 else "Nm"
                    raise RuntimeError(
                        f"desired {args.force_axis} differs from stable baseline by "
                        f"{desired_offset:.4f}{unit} > {max_desired_offset:.4f}{unit}"
                    )
            else:
                _validate_spring_baseline(
                    args,
                    axis_index=axis_index,
                    baseline_component=baseline_component,
                    baseline_force_norm=baseline_force_norm,
                    baseline_torque_norm=baseline_torque_norm,
                )
            print(
                f"[{side}] stable {args.frame} {args.force_axis} baseline="
                f"{baseline_component:.4f} std={axis_std:.4f}"
            )

        max_joint_drift = args.max_joint_drift_deg
        max_linear_drift = args.max_linear_drift_m
        max_angular_drift = args.max_angular_drift_deg
        if max_joint_drift is None:
            max_joint_drift = 1.0 if args.probe_mode == "transport-only" else 0.3
        if max_linear_drift is None:
            max_linear_drift = 0.005 if args.probe_mode == "transport-only" else 0.002
        if max_angular_drift is None:
            max_angular_drift = 2.0 if args.probe_mode == "transport-only" else 0.5
        control_mode, desired_force, limit_velocity = _control_vectors(args)
        result["resolved_control"] = {
            "frame": args.frame,
            "force_axis": (
                args.force_axis if args.probe_mode in COMPLIANT_PROBE_MODES else None
            ),
            "control_mode": control_mode,
            "desired_force": desired_force,
            "limit_velocity_sdk_value": limit_velocity,
            "limit_velocity_physical_units": {
                "linear_m_s": args.max_force_axis_linear_speed_m_s,
                "angular_deg_s": args.max_force_axis_angular_speed_deg_s,
            },
            "limit_velocity_sdk_contract": {
                "linear_value_unit": "m/s",
                "linear_resolution_m_s": 0.001,
                "angular_value_unit": "deg/s",
                "angular_resolution_deg_s": 0.001,
            },
            "follow": bool(args.follow),
            "trajectory_mode": int(args.trajectory_mode),
            "radio": int(args.radio),
            "contact_fixture_confirmed": bool(args.contact_fixture_confirmed),
            "static_cable_preload_confirmed": bool(
                args.static_cable_preload_confirmed
            ),
            "max_joint_drift_deg": max_joint_drift,
            "max_linear_drift_m": max_linear_drift,
            "max_angular_drift_deg": max_angular_drift,
        }

        guard = _start_emergency_guard(side, ip, timeout_sec=args.guard_ready_timeout_sec)
        result["emergency_guard"] = guard["ready_receipt"]
        print(
            f"[{side}] independent emergency-stop guard ready pid={guard['ready_receipt']['pid']}; "
            "Ctrl+C/SIGTERM => Arm_Stop"
        )

        def _handle_signal(signum, _frame) -> None:
            nonlocal signal_number, emergency_requested
            # Keep the handler lock-free. Calling multiprocessing.Event.set()
            # here can deadlock if SIGINT arrives during Event.wait(). The
            # normal control-loop context propagates the request to the guard.
            signal_number = int(signum)
            emergency_requested = True

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)

        if emergency_requested or guard["stop_request"].is_set():
            raise RuntimeError("interrupted before force-position mode was armed")
        guard["armed"].set()
        if emergency_requested or guard["stop_request"].is_set():
            guard["armed"].clear()
            raise RuntimeError("interrupted while force-position mode was being armed")
        start_code = int(robot.Start_Force_Position_Move())
        result["start_force_position_code"] = start_code
        _require_zero(start_code, f"{side} Start_Force_Position_Move")
        force_position_active = True
        print(
            f"[{side}] native mode active probe_mode={args.probe_mode} hz={args.hz:.1f}; "
            "press Ctrl+C for emergency stop"
        )

        period_sec = 1.0 / args.hz
        state_period_sec = 1.0 / args.state_hz
        start_monotonic = time.monotonic()
        next_command = start_monotonic
        next_state = start_monotonic
        command_count = 0
        max_send_ms = 0.0
        previous_measured_pose: list[float] | None = None
        previous_measured_time: float | None = None
        max_measured_linear_speed_m_s = 0.0
        max_measured_angular_speed_deg_s = 0.0
        while time.monotonic() - start_monotonic < args.duration_sec:
            if emergency_requested or guard["stop_request"].is_set():
                emergency_requested = True
                break
            now = time.monotonic()
            if now < next_command:
                guard["stop_request"].wait(min(next_command - now, 0.02))
                continue
            send_started = time.monotonic_ns()
            send_code = int(
                robot.Force_Position_Move_New(
                    flag=0,
                    joint=initial_state["joint_deg"],
                    sensor=1,
                    mode=1 if args.frame == "tool" else 0,
                    follow=args.follow,
                    control_mode=control_mode,
                    desired_force=desired_force,
                    limit_vel=limit_velocity,
                    trajectory_mode=args.trajectory_mode,
                    radio=args.radio,
                )
            )
            send_ms = (time.monotonic_ns() - send_started) / 1e6
            max_send_ms = max(max_send_ms, send_ms)
            command_count += 1
            result["command_count"] = command_count
            result["max_send_ms"] = max_send_ms
            if send_code != 0:
                raise RuntimeError(f"Force_Position_Move_New failed with return code {send_code}")
            next_command += period_sec
            if next_command < time.monotonic() - period_sec:
                next_command = time.monotonic()

            if time.monotonic() >= next_state:
                current_state = _read_state(robot, side)
                current_force = _read_force(robot, side)
                measured_time = time.monotonic()
                monitored = current_force[args.frame]
                force_norm = _norm(monitored[:3])
                torque_norm = _norm(monitored[3:])
                joint_drift = _max_joint_delta_deg(
                    initial_state["joint_deg"], current_state["joint_deg"]
                )
                linear_drift, angular_drift = _pose_delta(
                    initial_state["pose_xyz_rpy"], current_state["pose_xyz_rpy"]
                )
                measured_linear_speed = 0.0
                measured_angular_speed = 0.0
                measured_state_dt = None
                if previous_measured_pose is not None and previous_measured_time is not None:
                    measured_state_dt = measured_time - previous_measured_time
                    measured_linear_speed, measured_angular_speed = _measured_pose_rate(
                        previous_measured_pose,
                        current_state["pose_xyz_rpy"],
                        elapsed_sec=measured_state_dt,
                    )
                    max_measured_linear_speed_m_s = max(
                        max_measured_linear_speed_m_s,
                        measured_linear_speed,
                    )
                    max_measured_angular_speed_deg_s = max(
                        max_measured_angular_speed_deg_s,
                        measured_angular_speed,
                    )
                previous_measured_pose = list(current_state["pose_xyz_rpy"])
                previous_measured_time = measured_time
                result["max_measured_linear_speed_m_s"] = max_measured_linear_speed_m_s
                result["max_measured_angular_speed_deg_s"] = (
                    max_measured_angular_speed_deg_s
                )
                sample = {
                    "elapsed_sec": time.monotonic() - start_monotonic,
                    "joint_deg": current_state["joint_deg"],
                    "pose_xyz_rpy": current_state["pose_xyz_rpy"],
                    "wrench": monitored,
                    "force_norm_n": force_norm,
                    "torque_norm_nm": torque_norm,
                    "joint_drift_deg": joint_drift,
                    "linear_drift_m": linear_drift,
                    "angular_drift_deg": angular_drift,
                    "measured_state_dt_sec": measured_state_dt,
                    "measured_linear_speed_m_s": measured_linear_speed,
                    "measured_angular_speed_deg_s": measured_angular_speed,
                    "send_ms": send_ms,
                }
                if args.probe_mode in COMPLIANT_PROBE_MODES:
                    force_axis_index = AXES.index(args.force_axis)
                    sample["force_axis"] = args.force_axis
                    sample["force_axis_value"] = monitored[force_axis_index]
                    if baseline_wrench is None:  # pragma: no cover - defensive contract.
                        raise RuntimeError("compliant probe is missing its force baseline")
                    delta_metrics = _wrench_delta_metrics(
                        monitored,
                        baseline_wrench,
                        axis_index=force_axis_index,
                    )
                    sample.update(delta_metrics)
                    if args.probe_mode == "force-track":
                        sample["force_axis_target"] = desired_force[force_axis_index]
                        sample["force_axis_error"] = (
                            monitored[force_axis_index] - desired_force[force_axis_index]
                        )
                result["samples"].append(sample)
                force_axis_text = ""
                if args.probe_mode in COMPLIANT_PROBE_MODES:
                    force_axis_unit = "N" if force_axis_index < 3 else "Nm"
                    force_axis_text = (
                        f" {args.frame}_{args.force_axis}="
                        f"{monitored[force_axis_index]:+.3f}{force_axis_unit}"
                        f" delta={sample['selected_axis_delta']:+.3f}{force_axis_unit}"
                        f" ratio={sample['selected_axis_ratio']:.2f}"
                    )
                    if args.probe_mode == "force-track":
                        force_axis_text += f" err={sample['force_axis_error']:+.3f}"
                print(
                    f"[{side}] t={sample['elapsed_sec']:.2f}s command={command_count} "
                    f"drift={joint_drift:.3f}deg/{linear_drift:.4f}m/{angular_drift:.2f}deg "
                    f"speed={measured_linear_speed * 1000.0:.2f}mm/s/"
                    f"{measured_angular_speed:.2f}deg/s "
                    f"|F|={force_norm:.2f}N |T|={torque_norm:.2f}Nm"
                    f"{force_axis_text} send={send_ms:.2f}ms"
                )
                if force_norm > args.hard_max_force_n:
                    raise RuntimeError(
                        f"force guard tripped: {force_norm:.3f}N > {args.hard_max_force_n:.3f}N"
                    )
                if torque_norm > args.hard_max_torque_nm:
                    raise RuntimeError(
                        f"torque guard tripped: {torque_norm:.3f}Nm > {args.hard_max_torque_nm:.3f}Nm"
                    )
                if args.probe_mode in COMPLIANT_PROBE_MODES:
                    force_delta_norm = float(sample["force_delta_norm_n"])
                    torque_delta_norm = float(sample["torque_delta_norm_nm"])
                    if (
                        args.max_force_delta_from_baseline_n is not None
                        and force_delta_norm > args.max_force_delta_from_baseline_n
                    ):
                        raise RuntimeError(
                            "baseline-relative force guard tripped: "
                            f"{force_delta_norm:.3f}N > "
                            f"{args.max_force_delta_from_baseline_n:.3f}N"
                        )
                    if (
                        args.max_torque_delta_from_baseline_nm is not None
                        and torque_delta_norm > args.max_torque_delta_from_baseline_nm
                    ):
                        raise RuntimeError(
                            "baseline-relative torque guard tripped: "
                            f"{torque_delta_norm:.3f}Nm > "
                            f"{args.max_torque_delta_from_baseline_nm:.3f}Nm"
                        )
                    selected_group_delta = (
                        force_delta_norm if force_axis_index < 3 else torque_delta_norm
                    )
                    if (
                        args.min_selected_axis_force_ratio is not None
                        and selected_group_delta >= args.direction_check_min_delta
                        and float(sample["selected_axis_ratio"])
                        < args.min_selected_axis_force_ratio
                    ):
                        raise RuntimeError(
                            "selected-axis direction guard tripped: "
                            f"ratio={sample['selected_axis_ratio']:.3f} < "
                            f"{args.min_selected_axis_force_ratio:.3f} at "
                            f"delta={selected_group_delta:.3f}"
                        )
                if joint_drift > max_joint_drift:
                    raise RuntimeError(
                        f"joint drift guard tripped: {joint_drift:.3f}deg > {max_joint_drift:.3f}deg"
                    )
                if linear_drift > max_linear_drift:
                    raise RuntimeError(
                        f"linear drift guard tripped: {linear_drift:.4f}m > {max_linear_drift:.4f}m"
                    )
                if angular_drift > max_angular_drift:
                    raise RuntimeError(
                        f"angular drift guard tripped: {angular_drift:.3f}deg > {max_angular_drift:.3f}deg"
                    )
                if (
                    args.max_measured_linear_speed_m_s is not None
                    and measured_linear_speed > args.max_measured_linear_speed_m_s
                ):
                    raise RuntimeError(
                        "measured linear speed guard tripped: "
                        f"{measured_linear_speed:.4f}m/s > "
                        f"{args.max_measured_linear_speed_m_s:.4f}m/s"
                    )
                if (
                    args.max_measured_angular_speed_deg_s is not None
                    and measured_angular_speed > args.max_measured_angular_speed_deg_s
                ):
                    raise RuntimeError(
                        "measured angular speed guard tripped: "
                        f"{measured_angular_speed:.3f}deg/s > "
                        f"{args.max_measured_angular_speed_deg_s:.3f}deg/s"
                    )
                next_state += state_period_sec

        result["command_count"] = command_count
        result["max_send_ms"] = max_send_ms
        if emergency_requested:
            receipts = _request_emergency_stop(
                side, ip, guard, timeout_sec=args.stop_ack_timeout_sec
            )
            result["emergency_stop_receipts"] = receipts
            acknowledged = _stop_acknowledged(receipts)
            result["status"] = (
                "interrupted_emergency_stop_acknowledged"
                if acknowledged
                else "critical_emergency_stop_unacknowledged"
            )
            print(
                f"[{side}] Ctrl+C/SIGTERM emergency-stop receipts={receipts}",
                file=sys.stderr,
            )
            if not acknowledged:
                print(f"[{side}] CRITICAL: Arm_Stop was not acknowledged", file=sys.stderr)
                exit_code = 1
            else:
                exit_code = 130
        else:
            stop_code = int(robot.Stop_Force_Position_Move())
            result["stop_force_position_code"] = stop_code
            _require_zero(stop_code, f"{side} Stop_Force_Position_Move")
            force_position_active = False
            guard["armed"].clear()
            guard["normal_shutdown"].set()
            result["status"] = "completed"
            exit_code = 0

    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "failed"
        print(f"ERROR: {result['error']}", file=sys.stderr)
        if robot is not None and guard is not None and guard["armed"].is_set():
            emergency_requested = True
            receipts = _request_emergency_stop(
                side, ip, guard, timeout_sec=args.stop_ack_timeout_sec
            )
            result["emergency_stop_receipts"] = receipts
            acknowledged = _stop_acknowledged(receipts)
            result["status"] = (
                "failed_emergency_stop_acknowledged"
                if acknowledged
                else "critical_failed_emergency_stop_unacknowledged"
            )
            print(f"[{side}] fault emergency-stop receipts={receipts}", file=sys.stderr)
        exit_code = 1
    finally:
        if force_position_active and robot is not None and not emergency_requested:
            try:
                result["cleanup_stop_force_position_code"] = int(robot.Stop_Force_Position_Move())
            except BaseException as exc:
                result["cleanup_stop_force_position_error"] = f"{type(exc).__name__}: {exc}"
        if guard is not None:
            if not emergency_requested or not guard["armed"].is_set():
                guard["armed"].clear()
                guard["normal_shutdown"].set()
            guard_reap = _reap_process(guard["process"], grace_sec=2.0)
            result["emergency_guard_reap"] = guard_reap
            result["emergency_guard_exitcode"] = guard_reap["exitcode"]
            trailing_receipts = _drain_receipts(guard["receipts"])
            if trailing_receipts:
                result["emergency_stop_receipts"].extend(trailing_receipts)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if robot is not None and not emergency_requested:
            try:
                robot.Arm_Socket_Close()
            except BaseException as exc:
                result["socket_close_error"] = f"{type(exc).__name__}: {exc}"
        elif robot is not None:
            result["socket_close"] = "skipped_after_emergency_to_avoid_blocking_parent"
        result["signal"] = signal_number
        result["finished_at"] = datetime.now().astimezone().isoformat()
        try:
            _atomic_write_json(output_path, result)
            print(f"result_json={output_path}")
        except BaseException as exc:
            print(f"ERROR: failed to write result JSON: {type(exc).__name__}: {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
