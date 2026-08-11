"""Read-only RM65 force/state contention probe for two real arms.

This is deliberately a non-production probe.  It creates one SDK connection
per arm and calls only ``Get_Force_Data``, ``Get_Current_Arm_State`` and
``Arm_Socket_Close``.  It never enables an arm, changes a frame, clears a force
sensor, sends a hold, or dispatches a motion command.

Force reads are kept at or below 20 Hz because the production backend records
the vendor requirement of at least 50 ms between actual ``Get_Force_Data``
queries.  State and force calls share one lock per arm, matching the serialized
SDK boundary in the real-teleop backend.  The state worker is then swept up to
the requested rates so contention, call latency and force-read degradation can
be measured without motion.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.rm65.rm_api2_adapter import (  # noqa: E402
    Arm,
    LEFT_ARM_IP,
    RIGHT_ARM_IP,
    RM65,
)


FORCE_PERIOD_SEC = 0.05
ALLOWED_SDK_CALLS = (
    "Get_Force_Data",
    "Get_Current_Arm_State",
    "Arm_Socket_Close",
)


def _parse_rates(text: str) -> list[float]:
    try:
        rates = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid state rate list: {text!r}") from exc
    if not rates or any(not math.isfinite(rate) or rate < 0.0 for rate in rates):
        raise argparse.ArgumentTypeError("state rates must be finite and >= 0")
    return rates


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _latency_summary_ms(values_sec: list[float]) -> dict[str, float | None]:
    if not values_sec:
        return {key: None for key in ("mean", "p50", "p95", "p99", "max")}
    values_ms = [1000.0 * float(value) for value in values_sec]
    return {
        "mean": statistics.fmean(values_ms),
        "p50": _percentile(values_ms, 0.50),
        "p95": _percentile(values_ms, 0.95),
        "p99": _percentile(values_ms, 0.99),
        "max": max(values_ms),
    }


def _rate_from_starts(starts: list[float], duration_sec: float) -> dict[str, float]:
    count = len(starts)
    wall_hz = count / duration_sec if duration_sec > 0.0 else 0.0
    span_hz = 0.0
    if count >= 2 and starts[-1] > starts[0]:
        span_hz = (count - 1) / (starts[-1] - starts[0])
    return {"wall_hz": wall_hz, "span_hz": span_hz}


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


@dataclass
class CallStats:
    requested: int = 0
    succeeded: int = 0
    errors: list[str] = field(default_factory=list)
    request_times: list[float] = field(default_factory=list)
    call_start_times: list[float] = field(default_factory=list)
    lock_wait_sec: list[float] = field(default_factory=list)
    call_elapsed_sec: list[float] = field(default_factory=list)
    schedule_lateness_sec: list[float] = field(default_factory=list)
    force_change_count: int = 0
    force_max_norm_n: float = 0.0
    torque_max_norm_nm: float = 0.0
    max_joint_drift_deg: float = 0.0
    _last_force: tuple[float, ...] | None = None

    def payload(self, duration_sec: float) -> dict[str, object]:
        return {
            "requested": int(self.requested),
            "succeeded": int(self.succeeded),
            "error_count": len(self.errors),
            "errors": list(self.errors[:8]),
            "requested_rate": _rate_from_starts(self.request_times, duration_sec),
            "actual_call_rate": _rate_from_starts(self.call_start_times, duration_sec),
            "lock_wait_ms": _latency_summary_ms(self.lock_wait_sec),
            "sdk_call_ms": _latency_summary_ms(self.call_elapsed_sec),
            "schedule_lateness_ms": _latency_summary_ms(self.schedule_lateness_sec),
            "force_change_count": int(self.force_change_count),
            "force_max_norm_n": float(self.force_max_norm_n),
            "torque_max_norm_nm": float(self.torque_max_norm_nm),
            "max_joint_drift_deg": float(self.max_joint_drift_deg),
        }


class ArmReadCoordinator:
    def __init__(
        self,
        *,
        side: str,
        ip: str,
        max_force_n: float,
        max_torque_nm: float,
        max_joint_drift_deg: float,
    ) -> None:
        self.side = side
        self.ip = ip
        self.max_force_n = float(max_force_n)
        self.max_torque_nm = float(max_torque_nm)
        self.max_joint_drift_deg = float(max_joint_drift_deg)
        self.robot: Arm | None = None
        self.lock = threading.Lock()
        self.last_force_call_start: float | None = None
        self.initial_joint_deg: tuple[float, ...] | None = None

    def connect(self) -> None:
        self.robot = Arm(RM65, self.ip)
        if int(self.robot.Arm_Socket_State()) != 0:
            self.robot.Arm_Socket_Close()
            self.robot = None
            raise RuntimeError(f"{self.side}: failed to connect ip={self.ip}")
        code, joint, _pose, arm_err, sys_err = self.robot.Get_Current_Arm_State(retry=0)
        if int(code) != 0:
            raise RuntimeError(f"{self.side}: initial state read failed code={code}")
        if int(arm_err) != 0 or int(sys_err) != 0:
            raise RuntimeError(
                f"{self.side}: initial state reports arm_err={arm_err} sys_err={sys_err}"
            )
        joint_values = tuple(float(value) for value in joint)
        if len(joint_values) != 6 or not all(math.isfinite(value) for value in joint_values):
            raise RuntimeError(f"{self.side}: invalid initial joint state {joint_values}")
        self.initial_joint_deg = joint_values
        print(
            f"[{self.side}] connected ip={self.ip} api={self.robot.API_Version()} "
            f"joint_deg={[round(value, 4) for value in joint_values]}"
        )

    def close(self) -> None:
        if self.robot is not None:
            self.robot.Arm_Socket_Close()
            self.robot = None
            print(f"[{self.side}] socket closed")

    def read_state(self, stats: CallStats) -> None:
        robot = self.robot
        if robot is None or self.initial_joint_deg is None:
            raise RuntimeError(f"{self.side}: state read without connection")
        requested = time.perf_counter()
        stats.requested += 1
        stats.request_times.append(requested)
        with self.lock:
            call_start = time.perf_counter()
            stats.lock_wait_sec.append(call_start - requested)
            code, joint, _pose, arm_err, sys_err = robot.Get_Current_Arm_State(retry=0)
            call_end = time.perf_counter()
        stats.call_start_times.append(call_start)
        stats.call_elapsed_sec.append(call_end - call_start)
        if int(code) != 0 or int(arm_err) != 0 or int(sys_err) != 0:
            stats.errors.append(
                f"code={code} arm_err={arm_err} sys_err={sys_err}"
            )
            return
        joint_values = tuple(float(value) for value in joint)
        if len(joint_values) != 6 or not all(math.isfinite(value) for value in joint_values):
            stats.errors.append(f"invalid_joint={joint_values}")
            return
        drift = max(
            abs(value - initial)
            for value, initial in zip(joint_values, self.initial_joint_deg, strict=True)
        )
        stats.max_joint_drift_deg = max(stats.max_joint_drift_deg, drift)
        stats.succeeded += 1
        if drift > self.max_joint_drift_deg:
            raise RuntimeError(
                f"{self.side}: observed joint drift {drift:.4f}deg exceeds "
                f"{self.max_joint_drift_deg:.4f}deg; aborting read-only probe"
            )

    def read_force(self, stats: CallStats, stop: threading.Event) -> None:
        robot = self.robot
        if robot is None:
            raise RuntimeError(f"{self.side}: force read without connection")
        requested = time.perf_counter()
        stats.requested += 1
        stats.request_times.append(requested)
        while not stop.is_set():
            with self.lock:
                lock_acquired = time.perf_counter()
                remaining = (
                    0.0
                    if self.last_force_call_start is None
                    else FORCE_PERIOD_SEC
                    - (lock_acquired - self.last_force_call_start)
                )
                if remaining <= 0.0:
                    stats.lock_wait_sec.append(lock_acquired - requested)
                    self.last_force_call_start = lock_acquired
                    code, _raw, _zero, _work, tool = robot.Get_Force_Data()
                    call_end = time.perf_counter()
                    break
            stop.wait(min(max(remaining, 0.0005), 0.005))
        else:
            return
        stats.call_start_times.append(lock_acquired)
        stats.call_elapsed_sec.append(call_end - lock_acquired)
        if int(code) != 0:
            stats.errors.append(f"code={code}")
            return
        tool_values = tuple(float(value) for value in tool)
        if len(tool_values) != 6 or not all(math.isfinite(value) for value in tool_values):
            stats.errors.append(f"invalid_tool_force={tool_values}")
            return
        force_norm = _norm(tool_values[:3])
        torque_norm = _norm(tool_values[3:])
        stats.force_max_norm_n = max(stats.force_max_norm_n, force_norm)
        stats.torque_max_norm_nm = max(stats.torque_max_norm_nm, torque_norm)
        if stats._last_force is not None and any(
            abs(value - previous) > 1e-9
            for value, previous in zip(tool_values, stats._last_force, strict=True)
        ):
            stats.force_change_count += 1
        stats._last_force = tool_values
        stats.succeeded += 1
        if force_norm > self.max_force_n or torque_norm > self.max_torque_nm:
            raise RuntimeError(
                f"{self.side}: observed |F|={force_norm:.3f}N |T|={torque_norm:.3f}Nm "
                "above read-only probe threshold; aborting"
            )


def _periodic_worker(
    *,
    label: str,
    rate_hz: float,
    duration_sec: float,
    stop: threading.Event,
    stats: CallStats,
    callback: Callable[[CallStats], None],
    failures: list[str],
) -> None:
    period = 1.0 / float(rate_hz)
    started = time.perf_counter()
    deadline = started
    try:
        while not stop.is_set():
            now = time.perf_counter()
            if now >= started + duration_sec:
                return
            if now < deadline:
                stop.wait(deadline - now)
                continue
            stats.schedule_lateness_sec.append(max(0.0, now - deadline))
            callback(stats)
            deadline += period
            # When the requested rate is above transport capacity, do not
            # manufacture an unbounded backlog of historical deadlines.
            now = time.perf_counter()
            if deadline < now - period:
                deadline = now
    except Exception as exc:  # noqa: BLE001 - preserve hardware error text
        failures.append(f"{label}: {type(exc).__name__}: {exc}")
        stop.set()


def _run_stage(
    coordinators: list[ArmReadCoordinator],
    *,
    state_hz: float,
    force_hz: float,
    duration_sec: float,
) -> dict[str, object]:
    stop = threading.Event()
    failures: list[str] = []
    stats_by_side: dict[str, dict[str, CallStats]] = {
        coordinator.side: {"force": CallStats(), "state": CallStats()}
        for coordinator in coordinators
    }
    threads: list[threading.Thread] = []
    for coordinator in coordinators:
        if force_hz > 0.0:
            force_stats = stats_by_side[coordinator.side]["force"]
            threads.append(
                threading.Thread(
                    target=_periodic_worker,
                    kwargs={
                        "label": f"{coordinator.side}.force",
                        "rate_hz": force_hz,
                        "duration_sec": duration_sec,
                        "stop": stop,
                        "stats": force_stats,
                        "callback": lambda stats, c=coordinator: c.read_force(stats, stop),
                        "failures": failures,
                    },
                    name=f"rm65-{coordinator.side}-force",
                )
            )
        if state_hz > 0.0:
            state_stats = stats_by_side[coordinator.side]["state"]
            threads.append(
                threading.Thread(
                    target=_periodic_worker,
                    kwargs={
                        "label": f"{coordinator.side}.state",
                        "rate_hz": state_hz,
                        "duration_sec": duration_sec,
                        "stop": stop,
                        "stats": state_stats,
                        "callback": coordinator.read_state,
                        "failures": failures,
                    },
                    name=f"rm65-{coordinator.side}-state",
                )
            )
    stage_started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - stage_started
    payload = {
        "requested_state_hz_per_arm": float(state_hz),
        "requested_force_hz_per_arm": float(force_hz),
        "duration_sec": float(elapsed),
        "failures": list(failures),
        "sides": {
            side: {
                kind: stats.payload(elapsed)
                for kind, stats in side_stats.items()
            }
            for side, side_stats in stats_by_side.items()
        },
    }
    print(
        f"[stage] state={state_hz:.1f}Hz/arm force={force_hz:.1f}Hz/arm "
        f"elapsed={elapsed:.3f}s failures={len(failures)}"
    )
    for side, side_payload in payload["sides"].items():
        force_payload = side_payload["force"]
        state_payload = side_payload["state"]
        print(
            f"  {side}: force={force_payload['actual_call_rate']['span_hz']:.2f}Hz "
            f"force_sdk_p95={force_payload['sdk_call_ms']['p95']}ms "
            f"state={state_payload['actual_call_rate']['span_hz']:.2f}Hz "
            f"state_sdk_p95={state_payload['sdk_call_ms']['p95']}ms "
            f"errors={force_payload['error_count'] + state_payload['error_count']}"
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only dual-RM65 force/state SDK contention sweep."
    )
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="Required acknowledgement; the probe contains no motion calls.",
    )
    parser.add_argument("--left-ip", default=LEFT_ARM_IP)
    parser.add_argument("--right-ip", default=RIGHT_ARM_IP)
    parser.add_argument("--force-hz", type=float, default=20.0)
    parser.add_argument(
        "--state-hz-list",
        type=_parse_rates,
        default=_parse_rates("0,20,40,60,100,150,200"),
    )
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--max-force-n", type=float, default=100.0)
    parser.add_argument("--max-torque-nm", type=float, default=5.0)
    parser.add_argument("--max-joint-drift-deg", type=float, default=2.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.no_motion:
        print("ERROR: --no-motion is required", file=sys.stderr)
        return 2
    if not (0.0 <= args.force_hz <= 20.0):
        print("ERROR: --force-hz must be in [0, 20]", file=sys.stderr)
        return 2
    if args.duration_sec <= 0.0 or args.settle_sec < 0.0:
        print("ERROR: invalid duration/settle", file=sys.stderr)
        return 2
    if any(value <= 0.0 for value in (
        args.max_force_n,
        args.max_torque_nm,
        args.max_joint_drift_deg,
    )):
        print("ERROR: safety thresholds must be positive", file=sys.stderr)
        return 2

    print("[safety] READ-ONLY RM65 PROBE")
    print(f"[safety] allowed SDK calls={ALLOWED_SDK_CALLS}")
    print(
        "[safety] forbidden: enable/disable, motion, servo, hold, stop, force mode, "
        "force clear/tare, frame changes and payload changes"
    )
    coordinators = [
        ArmReadCoordinator(
            side="left",
            ip=args.left_ip,
            max_force_n=args.max_force_n,
            max_torque_nm=args.max_torque_nm,
            max_joint_drift_deg=args.max_joint_drift_deg,
        ),
        ArmReadCoordinator(
            side="right",
            ip=args.right_ip,
            max_force_n=args.max_force_n,
            max_torque_nm=args.max_torque_nm,
            max_joint_drift_deg=args.max_joint_drift_deg,
        ),
    ]
    report: dict[str, object] = {
        "schema_version": 1,
        "probe": "rm65_force_state_read_rate",
        "allowed_sdk_calls": list(ALLOWED_SDK_CALLS),
        "force_min_interval_sec": FORCE_PERIOD_SEC,
        "state_hz_list": list(args.state_hz_list),
        "stages": [],
    }
    try:
        for coordinator in coordinators:
            coordinator.connect()
        for index, state_hz in enumerate(args.state_hz_list):
            if index and args.settle_sec > 0.0:
                time.sleep(args.settle_sec)
            report["stages"].append(
                _run_stage(
                    coordinators,
                    state_hz=state_hz,
                    force_hz=args.force_hz,
                    duration_sec=args.duration_sec,
                )
            )
    except Exception as exc:  # noqa: BLE001 - report SDK failure verbatim
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR: {report['fatal_error']}", file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        for coordinator in reversed(coordinators):
            try:
                coordinator.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[{coordinator.side}] close failed: {exc}", file=sys.stderr)
    print("RESULT_JSON_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("RESULT_JSON_END")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
