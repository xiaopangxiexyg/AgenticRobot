from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import Arm, LEFT_ARM_IP, RIGHT_ARM_IP, RM65


CONFIRM_TEXT = "SEND_CURRENT_QPOS_CANFD"


def _format(values: Iterable[float], *, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _require_zero(code: object, label: str) -> None:
    if int(code) != 0:
        raise RuntimeError(f"{label} failed with return code {code}")


def _parse_hz_list(text: str) -> list[float]:
    values: list[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or value < 0.0:
            raise argparse.ArgumentTypeError("--hz-list values must be finite and >= 0")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("--hz-list must contain at least one value")
    return values


def _bool_arg(text: str) -> bool:
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of true/false/1/0/yes/no/on/off")


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
    values_ms = [float(value) * 1000.0 for value in values_sec]
    return {
        "mean_ms": statistics.fmean(values_ms) if values_ms else math.nan,
        "p50_ms": _percentile(values_ms, 0.50),
        "p95_ms": _percentile(values_ms, 0.95),
        "p99_ms": _percentile(values_ms, 0.99),
        "max_ms": max(values_ms) if values_ms else math.nan,
    }


def _connect_robot(side: str, ip: str) -> Arm:
    robot = Arm(RM65, ip)
    if int(robot.Arm_Socket_State()) != 0:
        robot.Arm_Socket_Close()
        raise RuntimeError(f"{side}: failed to connect ip={ip}")
    print(f"{side}: connected ip={ip} api={robot.API_Version()}")
    return robot


def _try_arm_stop(robot: Arm | None, side: str, reason: str) -> object | None:
    if robot is None:
        return None
    try:
        code = robot.Arm_Stop()
        print(f"{side}: Arm_Stop({reason}) -> {code}", file=sys.stderr)
        return code
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        print(f"{side}: Arm_Stop({reason}) failed: {message}", file=sys.stderr)
        return message


def _read_joint_deg(robot: Arm, side: str) -> list[float]:
    code, joint_deg, _pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=2)
    _require_zero(code, f"{side} Get_Current_Arm_State")
    if int(arm_err) != 0 or int(sys_err) != 0:
        raise RuntimeError(f"{side}: arm reports arm_err={arm_err}, sys_err={sys_err}")
    joint = [float(value) for value in joint_deg[:6]]
    if len(joint) != 6 or not all(math.isfinite(value) for value in joint):
        raise RuntimeError(f"{side}: invalid current joint_deg={joint_deg!r}")
    return joint


def _max_abs_delta_deg(a: Iterable[float], b: Iterable[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


@dataclass(frozen=True)
class ArmSpec:
    side: str
    ip: str


@dataclass
class ArmRunResult:
    side: str
    ip: str
    hz: float
    requested_mode: str
    start_joint_deg: list[float]
    end_joint_deg: list[float] | None
    max_end_delta_deg: float | None
    send_count: int
    error_count: int
    first_error: str | None
    total_sec: float
    actual_send_hz: float
    missed_deadlines: int
    send_call_ms: dict[str, float]
    loop_interval_ms: dict[str, float]
    safety_monitor: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "side": self.side,
            "ip": self.ip,
            "hz": self.hz,
            "requested_mode": self.requested_mode,
            "start_joint_deg": self.start_joint_deg,
            "end_joint_deg": self.end_joint_deg,
            "max_end_delta_deg": self.max_end_delta_deg,
            "send_count": self.send_count,
            "error_count": self.error_count,
            "first_error": self.first_error,
            "total_sec": self.total_sec,
            "actual_send_hz": self.actual_send_hz,
            "missed_deadlines": self.missed_deadlines,
            "send_call_ms": self.send_call_ms,
            "loop_interval_ms": self.loop_interval_ms,
            "safety_monitor": self.safety_monitor,
        }


class SafetyMonitor(threading.Thread):
    def __init__(
        self,
        *,
        spec: ArmSpec,
        start_joint_deg: list[float],
        stop_event: threading.Event,
        read_hz: float,
        max_delta_deg: float,
    ) -> None:
        super().__init__(name=f"rm65_canfd_safety_{spec.side}", daemon=True)
        self.spec = spec
        self.start_joint_deg = list(start_joint_deg)
        self.stop_event = stop_event
        self.read_hz = float(read_hz)
        self.max_delta_deg = float(max_delta_deg)
        self.read_count = 0
        self.max_observed_delta_deg = 0.0
        self.triggered = False
        self.stop_result: object | None = None
        self.error: BaseException | None = None
        self.reason: str | None = None

    def run(self) -> None:
        robot: Arm | None = None
        period = 1.0 / max(1e-6, self.read_hz)
        try:
            robot = _connect_robot(f"{self.spec.side}-safety", self.spec.ip)
            next_deadline = time.perf_counter()
            while not self.stop_event.is_set():
                joint = _read_joint_deg(robot, self.spec.side)
                self.read_count += 1
                delta = _max_abs_delta_deg(self.start_joint_deg, joint)
                self.max_observed_delta_deg = max(self.max_observed_delta_deg, delta)
                if delta > self.max_delta_deg:
                    self.triggered = True
                    self.reason = (
                        f"{self.spec.side}: safety delta {delta:.4f}deg > "
                        f"{self.max_delta_deg:.4f}deg"
                    )
                    self.stop_event.set()
                    self.stop_result = _try_arm_stop(
                        robot,
                        self.spec.side,
                        "safety_delta",
                    )
                    break
                next_deadline += period
                sleep_sec = next_deadline - time.perf_counter()
                if sleep_sec > 0.0:
                    self.stop_event.wait(sleep_sec)
        except BaseException as exc:  # noqa: BLE001
            self.error = exc
            self.stop_event.set()
            self.stop_result = _try_arm_stop(
                robot,
                self.spec.side,
                "safety_monitor_error",
            )
        finally:
            if robot is not None:
                robot.Arm_Socket_Close()

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "read_hz": self.read_hz,
            "max_delta_deg": self.max_delta_deg,
            "read_count": self.read_count,
            "max_observed_delta_deg": self.max_observed_delta_deg,
            "triggered": self.triggered,
            "reason": self.reason,
            "stop_result": self.stop_result,
            "error": None if self.error is None else f"{type(self.error).__name__}: {self.error}",
        }


class ArmHoldSender(threading.Thread):
    def __init__(
        self,
        *,
        spec: ArmSpec,
        hz: float,
        duration_sec: float,
        follow: bool,
        trajectory_mode: int,
        radio: int,
        stop_on_error: bool,
        max_end_delta_deg: float,
        safety_monitor_enabled: bool,
        safety_read_hz: float,
        safety_max_delta_deg: float,
        max_rate_yield_sec: float,
    ) -> None:
        super().__init__(name=f"rm65_canfd_hold_{spec.side}", daemon=True)
        self.spec = spec
        self.hz = float(hz)
        self.duration_sec = float(duration_sec)
        self.follow = bool(follow)
        self.trajectory_mode = int(trajectory_mode)
        self.radio = int(radio)
        self.stop_on_error = bool(stop_on_error)
        self.max_end_delta_deg_limit = float(max_end_delta_deg)
        self.safety_monitor_enabled = bool(safety_monitor_enabled)
        self.safety_read_hz = float(safety_read_hz)
        self.safety_max_delta_deg = float(safety_max_delta_deg)
        self.max_rate_yield_sec = float(max_rate_yield_sec)
        self.result: ArmRunResult | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        robot: Arm | None = None
        side = self.spec.side
        send_durations: list[float] = []
        intervals: list[float] = []
        send_count = 0
        error_count = 0
        first_error: str | None = None
        missed = 0
        end_joint: list[float] | None = None
        max_end_delta: float | None = None
        start_joint: list[float] | None = None
        clean_completion = False
        emergency_stop_issued = False
        started = time.perf_counter()
        stop_event = threading.Event()
        safety_monitor: SafetyMonitor | None = None
        try:
            robot = _connect_robot(side, self.spec.ip)
            start_joint = _read_joint_deg(robot, side)
            print(f"{side}: hold_joint_deg={_format(start_joint, precision=3)}")
            if self.safety_monitor_enabled:
                safety_monitor = SafetyMonitor(
                    spec=self.spec,
                    start_joint_deg=start_joint,
                    stop_event=stop_event,
                    read_hz=self.safety_read_hz,
                    max_delta_deg=self.safety_max_delta_deg,
                )
                safety_monitor.start()
            period = 0.0 if self.hz <= 0.0 else 1.0 / self.hz
            loops = None if self.hz <= 0.0 else max(1, int(round(self.duration_sec * self.hz)))
            deadline = time.perf_counter()
            previous_loop_start: float | None = None
            index = 0
            while True:
                if stop_event.is_set():
                    break
                if loops is not None and index >= loops:
                    break
                now = time.perf_counter()
                if now - started >= self.duration_sec:
                    break
                if previous_loop_start is not None:
                    intervals.append(now - previous_loop_start)
                previous_loop_start = now
                send_started = time.perf_counter()
                code = robot.Movej_CANFD(
                    start_joint,
                    follow=self.follow,
                    trajectory_mode=self.trajectory_mode,
                    radio=self.radio,
                )
                send_elapsed = time.perf_counter() - send_started
                send_durations.append(send_elapsed)
                if int(code) == 0:
                    send_count += 1
                else:
                    error_count += 1
                    if first_error is None:
                        first_error = f"Movej_CANFD returned {code}"
                    if self.stop_on_error:
                        _try_arm_stop(robot, side, "send_error")
                        emergency_stop_issued = True
                        stop_event.set()
                        break
                if period > 0.0:
                    if send_elapsed > period:
                        missed += 1
                    deadline += period
                    sleep_sec = deadline - time.perf_counter()
                    if sleep_sec > 0.0:
                        stop_event.wait(sleep_sec)
                elif self.max_rate_yield_sec > 0.0:
                    stop_event.wait(self.max_rate_yield_sec)
                index += 1
            total_sec = time.perf_counter() - started
            stop_event.set()
            if safety_monitor is not None:
                safety_monitor.join(timeout=max(0.5, 2.0 / max(1e-6, self.safety_read_hz)))
            try:
                end_joint = _read_joint_deg(robot, side)
            except BaseException:
                _try_arm_stop(robot, side, "final_state_read_failed")
                emergency_stop_issued = True
                raise
            max_end_delta = _max_abs_delta_deg(start_joint, end_joint)
            safety_payload = (
                {"enabled": False}
                if safety_monitor is None
                else safety_monitor.to_dict()
            )
            if safety_monitor is not None and safety_monitor.error is not None:
                emergency_stop_issued = True
                raise RuntimeError(
                    f"{side}: safety monitor failed: "
                    f"{type(safety_monitor.error).__name__}: {safety_monitor.error}"
                )
            if safety_monitor is not None and safety_monitor.triggered:
                emergency_stop_issued = True
                raise RuntimeError(str(safety_monitor.reason or f"{side}: safety monitor triggered"))
            if error_count > 0 and self.stop_on_error:
                emergency_stop_issued = True
                raise RuntimeError(first_error or f"{side}: Movej_CANFD returned an error")
            if max_end_delta > self.max_end_delta_deg_limit:
                _try_arm_stop(robot, side, "final_joint_drift")
                emergency_stop_issued = True
                raise RuntimeError(
                    f"{side}: final joint drift {max_end_delta:.4f}deg > "
                    f"{self.max_end_delta_deg_limit:.4f}deg"
                )
            requested_mode = "max_rate" if self.hz <= 0.0 else "fixed_rate"
            self.result = ArmRunResult(
                side=side,
                ip=self.spec.ip,
                hz=self.hz,
                requested_mode=requested_mode,
                start_joint_deg=start_joint,
                end_joint_deg=end_joint,
                max_end_delta_deg=max_end_delta,
                send_count=send_count,
                error_count=error_count,
                first_error=first_error,
                total_sec=total_sec,
                actual_send_hz=(send_count / total_sec if total_sec > 0.0 else math.nan),
                missed_deadlines=missed,
                send_call_ms=_stats_ms(send_durations),
                loop_interval_ms=_stats_ms(intervals),
                safety_monitor=safety_payload,
            )
            clean_completion = True
        except BaseException as exc:  # noqa: BLE001
            stop_event.set()
            if robot is not None and not emergency_stop_issued:
                _try_arm_stop(robot, side, "sender_exception")
                emergency_stop_issued = True
            self.error = exc
        finally:
            if safety_monitor is not None and safety_monitor.is_alive():
                safety_monitor.join(timeout=max(0.5, 2.0 / max(1e-6, self.safety_read_hz)))
            if robot is not None:
                if clean_completion:
                    try:
                        code = robot.Arm_Slow_Stop()
                        print(f"{side}: Arm_Slow_Stop -> {code}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"{side}: Arm_Slow_Stop failed: {exc}")
                else:
                    print(f"{side}: skip Arm_Slow_Stop after stop/error condition")
                robot.Arm_Socket_Close()


def _arm_specs(args: argparse.Namespace) -> list[ArmSpec]:
    if args.arms == "right":
        return [ArmSpec("right", args.right_ip)]
    if args.arms == "left":
        return [ArmSpec("left", args.left_ip)]
    return [ArmSpec("left", args.left_ip), ArmSpec("right", args.right_ip)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send the current RM65 joint position repeatedly with Movej_CANFD and "
            "measure the achievable command submission frequency. This is intended "
            "for transparent-transfer diagnostics with no commanded position offset."
        )
    )
    parser.add_argument("--arms", choices=("right", "left", "both"), default="right")
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument(
        "--hz-list",
        default="0",
        help="Comma-separated target rates. 0 means no sleep / maximum achievable rate.",
    )
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--settle-sec", type=float, default=0.5)
    parser.add_argument(
        "--follow",
        nargs="?",
        const=True,
        default=False,
        type=_bool_arg,
        help="Pass follow=True to Movej_CANFD. Accepts --follow or --follow true/false.",
    )
    parser.add_argument(
        "--no-follow",
        dest="follow",
        action="store_false",
        help="Pass follow=False to Movej_CANFD.",
    )
    parser.add_argument("--trajectory-mode", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--radio", type=int, default=0)
    parser.add_argument("--max-end-delta-deg", type=float, default=0.5)
    parser.add_argument(
        "--no-safety-monitor",
        dest="safety_monitor",
        action="store_false",
        help="Disable the independent current-qpos safety monitor.",
    )
    parser.set_defaults(safety_monitor=True)
    parser.add_argument(
        "--safety-read-hz",
        type=float,
        default=50.0,
        help="Frequency for the independent qpos safety monitor.",
    )
    parser.add_argument(
        "--safety-max-delta-deg",
        type=float,
        default=1.0,
        help="Arm_Stop immediately if any joint moves this far from the initial qpos.",
    )
    parser.add_argument(
        "--max-rate-yield-sec",
        type=float,
        default=0.0,
        help=(
            "Optional sleep/yield after each send only when a hz-list value is 0. "
            "Use a small value such as 0.001 if SDK max-rate sending starves state reads."
        ),
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="", help=f"Must equal {CONFIRM_TEXT}.")
    parser.add_argument("--output-json", default="")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    args.hz_values = _parse_hz_list(args.hz_list)
    if args.duration_sec <= 0.0:
        raise RuntimeError("--duration-sec must be positive")
    if args.settle_sec < 0.0:
        raise RuntimeError("--settle-sec must be >= 0")
    if args.max_end_delta_deg <= 0.0:
        raise RuntimeError("--max-end-delta-deg must be positive")
    if args.safety_read_hz <= 0.0:
        raise RuntimeError("--safety-read-hz must be positive")
    if args.safety_max_delta_deg <= 0.0:
        raise RuntimeError("--safety-max-delta-deg must be positive")
    if args.max_rate_yield_sec < 0.0:
        raise RuntimeError("--max-rate-yield-sec must be >= 0")
    if not 0 <= int(args.radio) <= 100:
        raise RuntimeError("--radio must be in [0, 100]")
    if not args.execute or args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"hardware command requires --execute --confirm {CONFIRM_TEXT}")


def _run_rate(args: argparse.Namespace, hz: float) -> list[ArmRunResult]:
    senders = [
        ArmHoldSender(
            spec=spec,
            hz=hz,
            duration_sec=args.duration_sec,
            follow=args.follow,
            trajectory_mode=args.trajectory_mode,
            radio=args.radio,
            stop_on_error=not bool(args.continue_on_error),
            max_end_delta_deg=args.max_end_delta_deg,
            safety_monitor_enabled=bool(args.safety_monitor),
            safety_read_hz=float(args.safety_read_hz),
            safety_max_delta_deg=float(args.safety_max_delta_deg),
            max_rate_yield_sec=float(args.max_rate_yield_sec),
        )
        for spec in _arm_specs(args)
    ]
    for sender in senders:
        sender.start()
    for sender in senders:
        sender.join()
    errors = [sender.error for sender in senders if sender.error is not None]
    if errors:
        raise RuntimeError("; ".join(f"{type(exc).__name__}: {exc}" for exc in errors))
    results: list[ArmRunResult] = []
    for sender in senders:
        if sender.result is None:
            raise RuntimeError(f"{sender.spec.side}: no result produced")
        results.append(sender.result)
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        "Movej_CANFD current-qpos hold frequency probe: "
        f"arms={args.arms} hz_values={args.hz_values} duration={args.duration_sec:.3f}s "
        f"follow={bool(args.follow)} trajectory_mode={args.trajectory_mode} radio={args.radio} "
        f"safety_monitor={bool(args.safety_monitor)} "
        f"max_rate_yield_sec={float(args.max_rate_yield_sec):.6f}"
    )
    if any(float(hz) <= 0.0 for hz in args.hz_values) and bool(args.safety_monitor):
        print(
            "NOTE: hz=0 submits as fast as possible. If the SDK prints repeated "
            "'Send would block' or state reads fail, sweep fixed rates first or add "
            "--max-rate-yield-sec 0.001."
        )
    all_results: list[ArmRunResult] = []
    try:
        for index, hz in enumerate(args.hz_values):
            if index > 0 and args.settle_sec > 0.0:
                time.sleep(args.settle_sec)
            print(f"\n=== rate {'max' if hz <= 0.0 else f'{hz:g} Hz'} ===")
            results = _run_rate(args, hz)
            all_results.extend(results)
            for result in results:
                send_ms = result.send_call_ms
                interval_ms = result.loop_interval_ms
                print(
                    f"{result.side}: mode={result.requested_mode} "
                    f"send_count={result.send_count} actual={result.actual_send_hz:.2f}Hz "
                    f"send_ms mean/p95/p99/max="
                    f"{send_ms['mean_ms']:.3f}/{send_ms['p95_ms']:.3f}/"
                    f"{send_ms['p99_ms']:.3f}/{send_ms['max_ms']:.3f} "
                    f"interval_ms mean/p95/p99="
                    f"{interval_ms['mean_ms']:.3f}/{interval_ms['p95_ms']:.3f}/"
                    f"{interval_ms['p99_ms']:.3f} "
                    f"errors={result.error_count} end_delta={result.max_end_delta_deg:.4f}deg"
                )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    payload = {
        "schema_version": 1,
        "generated_by": "hardware/rm65/debug_high_transfer/hold_current_qpos_canfd_rate.py",
        "arms": args.arms,
        "duration_sec": float(args.duration_sec),
        "hz_values": [float(value) for value in args.hz_values],
        "follow": bool(args.follow),
        "trajectory_mode": int(args.trajectory_mode),
        "radio": int(args.radio),
        "max_rate_yield_sec": float(args.max_rate_yield_sec),
        "safety_monitor": {
            "enabled": bool(args.safety_monitor),
            "read_hz": float(args.safety_read_hz),
            "max_delta_deg": float(args.safety_max_delta_deg),
        },
        "results": [result.to_dict() for result in all_results],
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
