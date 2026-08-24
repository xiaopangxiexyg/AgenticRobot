"""Episode-admission gripper warmup probe coordination.

The factory intentionally keeps the historical closure state model: one lock,
one generation counter, one daemon worker reference, and one status mapping
are shared by the returned callbacks.  Backend lookup, probing, scalar
normalization, and logging remain injected compatibility seams.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import argparse
import threading
import time

from ..backends.grippers import DMTacGripperBackend
from .components import RealTeleopComponents


@dataclass(frozen=True)
class GripperWarmupProbeRuntime:
    enabled: Callable[[], bool]
    reset: Callable[[str], None]
    running: Callable[[], bool]
    failure_message: Callable[[Mapping[str, object]], str]
    start: Callable[..., tuple[bool, str]]
    status: Callable[..., tuple[bool, str]]


def build_gripper_warmup_probe_runtime(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    gripper_backend_for_side: Callable[[RealTeleopComponents, str], object | None],
    run_probe_for_side: Callable[..., Mapping[str, object]],
    safe_float_value: Callable[[object], float | None],
    log: Callable[..., None],
) -> GripperWarmupProbeRuntime:
    """Build the six historical callbacks over one private probe state."""

    _gripper_backend_for_side = gripper_backend_for_side
    _run_gripper_warmup_probe_for_side = run_probe_for_side
    _safe_float_value = safe_float_value
    _log = log

    gripper_warmup_probe_lock = threading.Lock()
    gripper_warmup_probe_thread: threading.Thread | None = None
    gripper_warmup_probe_generation = 0
    gripper_warmup_probe_state: dict[str, object] = {
        "status": "idle",
        "message": "",
        "reason": "",
        "started_at": None,
        "completed_at": None,
        "results": {},
    }

    def partial_progress_entry_requires_passive_grippers() -> bool:
        if bool(getattr(args, "_gripper_warmup_probe_explicit", False)):
            return False
        profile = getattr(args, "collection_progress_profile", None)
        initial_step_id = str(getattr(profile, "initial_step_id", "") or "").strip()
        entry_step_id = str(
            getattr(args, "collection_progress_entry_step", "") or ""
        ).strip()
        return bool(
            initial_step_id
            and entry_step_id
            and entry_step_id != initial_step_id
        )

    def configured_gripper_backends() -> list[tuple[str, object]]:
        if not bool(getattr(args, "gripper_warmup_probe", True)):
            return []
        if partial_progress_entry_requires_passive_grippers():
            return []
        return [
            (side, backend)
            for side in components.active_sides
            if (backend := _gripper_backend_for_side(components, side)) is not None
            and bool(
                getattr(
                    getattr(backend, "config", None),
                    "connect_hardware",
                    True,
                )
            )
        ]

    def profile_fixed_sides() -> set[str]:
        controller = getattr(components, "fix_arm_controller", None)
        explicit = tuple(getattr(controller, "fixed_sides", ()) or ())
        if explicit:
            return {str(side) for side in explicit}
        if bool(getattr(controller, "fixed", False)):
            fallback = str(getattr(controller, "fixed_side", "") or "")
            return {fallback} if fallback else set()
        return set()

    def eligible_gripper_backends() -> list[tuple[str, object]]:
        fixed_sides = profile_fixed_sides()
        return [
            (side, backend)
            for side, backend in configured_gripper_backends()
            if side not in fixed_sides
        ]

    def gripper_warmup_probe_enabled() -> bool:
        return bool(eligible_gripper_backends())

    def reset_gripper_warmup_probe(reason: str) -> None:
        nonlocal gripper_warmup_probe_generation, gripper_warmup_probe_state
        if not configured_gripper_backends():
            return
        with gripper_warmup_probe_lock:
            gripper_warmup_probe_generation += 1
            gripper_warmup_probe_state = {
                "status": "idle",
                "message": "",
                "reason": str(reason),
                "started_at": None,
                "completed_at": None,
                "results": {},
            }

    def gripper_warmup_probe_running() -> bool:
        if not gripper_warmup_probe_enabled():
            return False
        with gripper_warmup_probe_lock:
            return str(gripper_warmup_probe_state.get("status") or "") == "running"

    def gripper_warmup_probe_failure_message(results: Mapping[str, object]) -> str:
        details: list[str] = []
        for side in components.active_sides:
            result = results.get(side)
            if not isinstance(result, Mapping) or bool(result.get("ok")):
                continue
            attempts = result.get("attempts")
            last_attempt = None
            if isinstance(attempts, Sequence) and not isinstance(attempts, (str, bytes)) and attempts:
                maybe_attempt = attempts[-1]
                if isinstance(maybe_attempt, Mapping):
                    last_attempt = maybe_attempt
            error = (
                str(last_attempt.get("error") or "")
                if isinstance(last_attempt, Mapping)
                else ""
            )
            target = (
                last_attempt.get("target")
                if isinstance(last_attempt, Mapping)
                else None
            )
            measured = (
                last_attempt.get("measured")
                if isinstance(last_attempt, Mapping)
                else None
            )
            target_f = _safe_float_value(target)
            measured_f = _safe_float_value(measured)
            if target_f is not None and measured_f is not None:
                details.append(
                    f"{side}: {error or 'state_mismatch'} "
                    f"target={target_f:.2f} measured={measured_f:.2f}"
                )
            else:
                details.append(f"{side}: {error or 'state_mismatch'}")
        suffix = "；".join(details) if details else "unknown failure"
        return f"gripper warmup probe 失败，已尝试重连后仍未通过：{suffix}。"

    def start_gripper_warmup_probe(now: float, *, reason: str) -> tuple[bool, str]:
        nonlocal gripper_warmup_probe_generation, gripper_warmup_probe_state
        nonlocal gripper_warmup_probe_thread
        if not gripper_warmup_probe_enabled():
            return True, "gripper warmup probe 未启用。"
        side_backends: list[tuple[str, DMTacGripperBackend]] = [
            (side, backend)
            for side, backend in eligible_gripper_backends()
        ]
        if not side_backends:
            return True, "固定侧 end-effector 保持中，无可安全主动探测的 gripper。"
        with gripper_warmup_probe_lock:
            status = str(gripper_warmup_probe_state.get("status") or "idle")
            if status == "ready":
                return True, str(
                    gripper_warmup_probe_state.get("message") or "gripper warmup probe 已通过。"
                )
            if status == "running":
                return False, str(
                    gripper_warmup_probe_state.get("message") or "等待 gripper warmup probe 完成。"
                )
            if status == "failed":
                return False, str(
                    gripper_warmup_probe_state.get("message") or "gripper warmup probe 失败。"
                )
            gripper_warmup_probe_generation += 1
            generation = int(gripper_warmup_probe_generation)
            sides_text = ",".join(side for side, _backend in side_backends)
            gripper_warmup_probe_state = {
                "status": "running",
                "message": f"gripper warmup probe 运行中：{sides_text} open->half->open。",
                "reason": str(reason),
                "started_at": float(now),
                "completed_at": None,
                "results": {},
            }

        def cancelled() -> bool:
            with gripper_warmup_probe_lock:
                stale_generation = (
                    int(gripper_warmup_probe_generation) != generation
                )
            return stale_generation or not gripper_warmup_probe_enabled()

        def worker() -> None:
            nonlocal gripper_warmup_probe_state
            results: dict[str, object] = {}
            targets = (0.0, 0.5, 0.0)
            try:
                for side, backend in side_backends:
                    if cancelled():
                        return
                    results[side] = _run_gripper_warmup_probe_for_side(
                        side,
                        backend,
                        targets=targets,
                        tolerance=max(
                            0.0,
                            float(getattr(args, "gripper_warmup_probe_tolerance", 0.18)),
                        ),
                        step_timeout_s=max(
                            0.01,
                            float(getattr(args, "gripper_warmup_probe_step_timeout_sec", 2.0)),
                        ),
                        settle_s=max(
                            0.0,
                            float(getattr(args, "gripper_warmup_probe_settle_sec", 0.20)),
                        ),
                        poll_s=max(
                            0.01,
                            float(getattr(args, "gripper_warmup_probe_poll_sec", 0.05)),
                        ),
                        reconnect_on_failure=True,
                        cancelled=cancelled,
                    )
                if cancelled():
                    return
                passed = bool(results) and all(
                    isinstance(result, Mapping) and bool(result.get("ok"))
                    for result in results.values()
                )
                if passed:
                    reconnected_sides = [
                        side
                        for side, result in results.items()
                        if isinstance(result, Mapping) and bool(result.get("reconnected"))
                    ]
                    if reconnected_sides:
                        message = (
                            "gripper warmup probe 通过；已自动重连 "
                            f"{','.join(reconnected_sides)}。"
                        )
                    else:
                        message = "gripper warmup probe 通过：open->half->open state matched。"
                    status = "ready"
                else:
                    message = gripper_warmup_probe_failure_message(results)
                    status = "failed"
            except Exception as exc:  # noqa: BLE001 - warmup must fail closed, not hang forever.
                if cancelled():
                    return
                message = f"gripper warmup probe 内部异常：{type(exc).__name__}: {exc}。"
                status = "failed"
            completed_at = time.time()
            with gripper_warmup_probe_lock:
                if int(gripper_warmup_probe_generation) != generation:
                    return
                gripper_warmup_probe_state = {
                    "status": status,
                    "message": message,
                    "reason": str(reason),
                    "started_at": float(now),
                    "completed_at": completed_at,
                    "results": results,
                }
            _log("GripperWarmup", message, now=completed_at)

        gripper_warmup_probe_thread = threading.Thread(
            target=worker,
            name="gripper-warmup-probe",
            daemon=True,
        )
        gripper_warmup_probe_thread.start()
        _log(
            "GripperWarmup",
            f"started reason={reason} sides={','.join(side for side, _backend in side_backends)}",
            now=now,
        )
        return False, "等待 gripper warmup probe 完成。"

    def gripper_warmup_probe_status(now: float, *, reason: str) -> tuple[bool, str]:
        if not gripper_warmup_probe_enabled():
            return True, ""
        with gripper_warmup_probe_lock:
            status = str(gripper_warmup_probe_state.get("status") or "idle")
            message = str(gripper_warmup_probe_state.get("message") or "")
        if status == "ready":
            return True, message or "gripper warmup probe 已通过。"
        if status == "running":
            return False, message or "等待 gripper warmup probe 完成。"
        if status == "failed":
            return False, message or "gripper warmup probe 失败。"
        return start_gripper_warmup_probe(now, reason=reason)

    return GripperWarmupProbeRuntime(
        enabled=gripper_warmup_probe_enabled,
        reset=reset_gripper_warmup_probe,
        running=gripper_warmup_probe_running,
        failure_message=gripper_warmup_probe_failure_message,
        start=start_gripper_warmup_probe,
        status=gripper_warmup_probe_status,
    )


__all__ = [
    "GripperWarmupProbeRuntime",
    "build_gripper_warmup_probe_runtime",
]
