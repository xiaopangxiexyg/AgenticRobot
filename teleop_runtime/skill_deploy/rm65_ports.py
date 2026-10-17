"""RM65 application ports for the independent real Skill control owner.

This module binds already-started dual RM65 backends to the generic Skill
control cycle.  It reuses the policy-deploy state, dispatch, and HOLD owners;
it does not construct RealTeleop, a Gateway, a device, or a control thread.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from hardware.rm65.realtime_feedback import RM65UDPFeedbackReceiver
from utils.owned_thread import OwnedThread
from robot_motion.runtime import TeleopCommand, TeleopObservation
from teleop_runtime.deploy.arm_safety_runtime import (
    bimanual_emergency_stop,
    emergency_stop_deploy_arm_backends,
    hold_deploy_arm_backends,
)
from teleop_runtime.deploy.arm_state_reader import ArmStateReader
from teleop_runtime.deploy.dispatch import (
    POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES,
    POSITION_REFERENCE_DISPATCH_AUDIT_POST,
    dispatch_position_reference_command,
    require_arm_backends_ready_for_position_dispatch,
)
from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.position_force_guard import (
    PositionForceGuardConfig,
    PositionForceGuardFault,
    PositionForceGuardMonitor,
)

from .control_runtime import (
    RealSkillMotionDispatchReceipt,
    RealSkillMotionEmergencyStopReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
    RealSkillMotionObservationPaused,
    RealSkillMotionObservationUnavailable,
)
from .motion_port import REAL_AGENT_MOTION_COMMAND_METADATA_KEY

# The fourth consecutive coherent-cut availability loss becomes a permanent
# sampler fault. Active or pending motion still fails closed on the first loss.
RM65_MAX_CONSECUTIVE_RECOVERABLE_MEASUREMENT_FAILURES = 3


def _next_periodic_measurement_deadline(
    *,
    previous_deadline: float,
    completed_at: float,
    period_sec: float,
) -> float:
    """Advance one phase-locked sampler deadline without catch-up bursts."""

    deadline = float(previous_deadline) + float(period_sec)
    completed = float(completed_at)
    period = float(period_sec)
    if not all(math.isfinite(value) for value in (deadline, completed, period)):
        raise ValueError("measurement sampler timing values must be finite.")
    if period <= 0.0:
        raise ValueError("measurement sampler period must be positive.")
    if deadline <= completed:
        missed = math.floor((completed - deadline) / period) + 1
        deadline += missed * period
    return deadline


class RM65SkillTargetBuilder(Protocol):
    """Coupled target solver synchronization used by this concrete binding."""

    def set_current_qpos_pair(
        self,
        *,
        left_qpos: np.ndarray,
        right_qpos: np.ndarray,
        reset_tracking: bool = True,
    ) -> None: ...


class RM65SkillPositionForceGuard(Protocol):
    """Lower force guard required by the formal real-position profile."""

    @property
    def motion_ready(self) -> bool: ...

    @property
    def fault(self) -> object | None: ...

    def start(
        self,
        *,
        arm_stale_watchdog: bool = True,
    ) -> Mapping[str, object]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ExclusiveRM65SkillPortProfile:
    """Trusted lower profile; none of these fields are Agent arguments."""

    state_refresh_hz: float = 30.0
    max_measurement_age_sec: float = 0.2
    background_measurement_sampling: bool = True
    dispatch_audit_mode: str = POSITION_REFERENCE_DISPATCH_AUDIT_POST
    hold_confirmation_timeout_sec: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "state_refresh_hz", "max_measurement_age_sec", "hold_confirmation_timeout_sec",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite.")
            object.__setattr__(self, field_name, value)
        mode = str(self.dispatch_audit_mode).strip().lower()
        if mode not in POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES:
            raise ValueError(
                "dispatch_audit_mode must be one of "
                f"{POSITION_REFERENCE_DISPATCH_AUDIT_CHOICES}."
            )
        object.__setattr__(self, "dispatch_audit_mode", mode)
        if not isinstance(self.background_measurement_sampling, bool):
            raise TypeError("background_measurement_sampling must be a boolean.")


def _bounded_backend_state(state: Mapping[str, Any]) -> dict[str, Any]:
    background = state.get("background_streaming")
    background_state = (
        dict(background) if isinstance(background, Mapping) else {}
    )
    return {
        "hardware_ready": state.get("hardware_ready"),
        "emergency_stopped": bool(state.get("emergency_stopped", False)),
        "send_result_code": state.get("send_result_code"),
        "background_enabled": bool(background_state.get("enabled", False)),
        "background_healthy": background_state.get("healthy"),
        "background_target_active": background_state.get("target_active"),
        "background_fault": background_state.get("fault"),
    }


def _bounded_backend_diagnostics_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    background = state.get("background_streaming")
    background_state = (
        dict(background) if isinstance(background, Mapping) else {}
    )
    raw_attempts = background_state.get("fault_stop_attempts")
    bounded_attempts: list[dict[str, Any]] = []
    if isinstance(raw_attempts, (list, tuple)):
        for raw_attempt in raw_attempts[:4]:
            if not isinstance(raw_attempt, Mapping):
                continue
            bounded_attempts.append(
                {
                    "method": raw_attempt.get("method"),
                    "result_code": raw_attempt.get("result_code"),
                    "result_text": _bounded_diagnostic_text(
                        raw_attempt.get("result_text")
                    ),
                    "exception": _bounded_diagnostic_text(
                        raw_attempt.get("exception")
                    ),
                }
            )
    raw_health_reasons = background_state.get("health_reasons")
    health_reasons = (
        list(raw_health_reasons)
        if isinstance(raw_health_reasons, (list, tuple))
        else []
    )
    result = {
        "hardware_ready": state.get("hardware_ready"),
        "emergency_stopped": bool(state.get("emergency_stopped", False)),
        "send_result_code": state.get("send_result_code"),
        "send_result_text": _bounded_diagnostic_text(
            state.get("send_result_text")
        ),
        "background_enabled": bool(background_state.get("enabled", False)),
        "background_healthy": background_state.get("healthy"),
        "background_health_reasons": [
            str(reason)[:128]
            for reason in health_reasons[:16]
        ],
        "background_thread_alive": background_state.get("thread_alive"),
        "background_target_active": background_state.get("target_active"),
        "background_target_stale": background_state.get("target_stale"),
        "background_fault": _bounded_diagnostic_text(
            background_state.get("fault")
        ),
        "background_fault_count": background_state.get("fault_count"),
        "background_fault_timestamp": background_state.get("fault_timestamp"),
        "background_fault_stop_attempted": background_state.get(
            "fault_stop_attempted"
        ),
        "background_fault_stop_success": background_state.get(
            "fault_stop_success"
        ),
        "background_fault_stop_attempts": bounded_attempts,
        "background_fault_stop_timestamp": background_state.get(
            "fault_stop_timestamp"
        ),
        "background_shutdown_requested": background_state.get(
            "shutdown_requested"
        ),
        "background_shutdown_complete": background_state.get(
            "shutdown_complete"
        ),
        "background_join_timed_out": background_state.get("join_timed_out"),
    }
    if bool(background_state.get("debug_timing_enabled", False)):
        raw_percentiles = background_state.get("timing_percentiles")
        bounded_percentiles: dict[str, dict[str, int | float | None]] = {}
        if isinstance(raw_percentiles, Mapping):
            for phase, raw_metrics in list(raw_percentiles.items())[:16]:
                if not isinstance(raw_metrics, Mapping):
                    continue
                bounded_percentiles[str(phase)[:64]] = {
                    key: raw_metrics.get(key)
                    for key in ("count", "p50", "p95", "p99", "max")
                }
        raw_last_phase = background_state.get("last_phase_timing_ms")
        last_phase = (
            {
                str(name)[:64]: float(value)
                for name, value in list(raw_last_phase.items())[:16]
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
            if isinstance(raw_last_phase, Mapping)
            else {}
        )
        result["background_stream_timing"] = {
            "target_hz": background_state.get("hz"),
            "warning_threshold_hz": background_state.get(
                "warning_threshold_hz"
            ),
            "actual_hz": background_state.get("actual_hz"),
            "send_count": background_state.get("send_count"),
            "scheduled_tick_count": background_state.get(
                "scheduled_tick_count"
            ),
            "executed_tick_count": background_state.get(
                "executed_tick_count"
            ),
            "skipped_tick_count": background_state.get("skipped_tick_count"),
            "deadline_miss_count": background_state.get(
                "deadline_miss_count"
            ),
            "low_rate_count": background_state.get("low_rate_count"),
            "last_phase_timing_ms": last_phase,
            "timing_percentiles": bounded_percentiles,
        }
    return result


def _bounded_diagnostic_text(value: object, *, limit: int = 512) -> str | None:
    if value is None:
        return None
    return " ".join(str(value).split())[:limit]


def build_exclusive_rm65_skill_position_force_guard(
    *,
    arm_backends: Mapping[str, EffectorBackend],
    config: PositionForceGuardConfig | None = None,
) -> PositionForceGuardMonitor:
    """Build the shared latched guard without importing a RealTeleop host."""

    backends = dict(arm_backends)
    if set(backends) != {"left", "right"}:
        raise ValueError(
            "exclusive RM65 Skill force guard requires exact left/right backends."
        )
    readers: dict[str, Callable[[], Mapping[str, Any]]] = {}
    for side, backend in backends.items():
        reader = getattr(backend, "read_force_data", None)
        if not callable(reader):
            raise TypeError(
                f"{side} RM65 Skill backend does not expose read_force_data."
            )
        readers[side] = reader

    def emergency_stop(fault: PositionForceGuardFault) -> None:
        emergency_stop_deploy_arm_backends(
            backends,
            reason=(
                f"position_force_guard:{fault.reason}:{fault.side or 'all'}"
            ),
        )

    return PositionForceGuardMonitor(
        readers_by_side=readers,
        emergency_stop=emergency_stop,
        config=config,
    )


class ExclusiveRM65SkillDeployPorts:
    """Typed production callbacks over one exact dual-RM65 backend inventory."""

    def __init__(
        self,
        *,
        arm_backends: Mapping[str, EffectorBackend],
        target_builder: RM65SkillTargetBuilder,
        position_force_guard: RM65SkillPositionForceGuard,
        profile: ExclusiveRM65SkillPortProfile | None = None,
        clock: Callable[[], float] = time.time,
        udp_feedback: RM65UDPFeedbackReceiver | None = None,
    ) -> None:
        backends = dict(arm_backends)
        if set(backends) != {"left", "right"}:
            raise ValueError(
                "exclusive RM65 Skill ports require exact left/right backends."
            )
        if not callable(getattr(target_builder, "set_current_qpos_pair", None)):
            raise TypeError(
                "target_builder must expose set_current_qpos_pair."
            )
        if not callable(clock):
            raise TypeError("clock must be callable.")
        for name in ("start", "close"):
            if not callable(getattr(position_force_guard, name, None)):
                raise TypeError(f"position_force_guard must expose {name}.")
        if not hasattr(position_force_guard, "motion_ready"):
            raise TypeError("position_force_guard must expose motion_ready.")
        if not hasattr(position_force_guard, "fault"):
            raise TypeError("position_force_guard must expose fault.")
        self.arm_backends = backends
        self.target_builder = target_builder
        self.position_force_guard = position_force_guard
        self.profile = profile or ExclusiveRM65SkillPortProfile()
        self.clock = clock
        self.udp_feedback = udp_feedback
        self._udp_feedback_started = False
        self._backend_start_attempted = False
        measurement_refresh_executor = (
            ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="exclusive-rm65-state-refresh",
            )
            if self.profile.background_measurement_sampling
            else None
        )
        self.arm_state_reader = ArmStateReader(
            backends,
            refresh_hz=self.profile.state_refresh_hz,
            # The control owner needs authoritative qpos only. Coupled FK is
            # performed after a new source sequence is consumed; computing
            # backend TCP poses here would age the hardware cut before the
            # freshness gate can inspect it.
            compute_tcp_pose=False,
            clock=clock,
            concurrent_refresh_executor=measurement_refresh_executor,
        )
        self._measurement_refresh_executor = measurement_refresh_executor
        self._measurement_refresh_executor_shutdown = (
            measurement_refresh_executor is None
        )
        self._sync_lock = threading.Lock()
        self._target_builder_synchronized = False
        self._lifecycle_lock = threading.Lock()
        self._resource_lock = threading.Lock()
        self._resources_started = False
        self._resources_closed = False
        self._start_attempted = False
        self._closing = False
        self._force_guard_started = False
        self._backend_close_pending = {"left", "right"}
        self._shutdown_hold_attempted = False
        self._shutdown_hold_confirmed = False
        self._shutdown_hold_error: str | None = None
        self._terminal_close_error: str | None = None
        self._startup_hold_receipt: RealSkillMotionHoldReceipt | None = None
        self._last_emergency_stop_receipt: (
            RealSkillMotionEmergencyStopReceipt | None
        ) = None
        self._measurement_lock = threading.Lock()
        self._measurement_read_lock = threading.Lock()
        self._measurement_stop = threading.Event()
        self._measurement_thread: OwnedThread | None = None
        self._latest_observation_cut: RealSkillMotionObservationCut | None = None
        self._measurement_fault: str | None = None
        self._measurement_sample_count = 0
        self._measurement_transient_failure_count = 0
        self._measurement_consecutive_transient_failures = 0
        self._last_measurement_transient_failure: str | None = None
        self._non_motion_read_window_active = False

    @property
    def resources_started(self) -> bool:
        with self._resource_lock:
            return self._resources_started

    @property
    def resources_closed(self) -> bool:
        with self._resource_lock:
            return self._resources_closed

    @contextmanager
    def non_motion_read_window(self) -> Iterator[None]:
        """Pause stale trips and background qpos refresh for read-only work.

        The position-force guard keeps force/torque and reader-error checks
        active in this window.  It also clears ``motion_ready``; therefore the
        existing command admission boundary cannot accept a motion command.
        When background measurement sampling is enabled, the sampler's read
        lock is held for the whole window.  This gives the force pollers sole
        access to the shared RM65 SDK I/O path instead of allowing a slow qpos
        refresh to block a force read.  Exiting the window requires one fresh
        dual-arm qpos cut before motion readiness is restored.
        """

        if not self.resources_started:
            raise RuntimeError(
                "exclusive RM65 Skill resources are not started."
            )
        # Never suspend stale protection or qpos publication during motion.
        # Reuse the backend's acknowledgement, without issuing another HOLD.
        self._confirm_arm_holds()
        suspend = getattr(
            self.position_force_guard,
            "suspend_stale_watchdog_for_non_motion",
            None,
        )
        resume = getattr(
            self.position_force_guard,
            "resume_stale_watchdog_after_non_motion",
            None,
        )
        if not callable(suspend) or not callable(resume):
            raise RuntimeError(
                "position force guard does not support non-motion read windows"
            )
        token = suspend()
        window_marked_active = False
        measurement_lock_acquired = False
        previous_sample_count: int | None = None
        primary_error: BaseException | None = None
        cleanup_errors: list[BaseException] = []
        try:
            with self._measurement_lock:
                if self._non_motion_read_window_active:
                    raise RuntimeError(
                        "RM65 non-motion read windows cannot overlap."
                    )
                self._non_motion_read_window_active = True
                window_marked_active = True
            if self.profile.background_measurement_sampling:
                # If a sampler read is already in flight, wait for that one
                # read to finish after stale admission has been suspended.
                # Thereafter no new qpos SDK call can start while the
                # read-only operation is running.
                self._measurement_read_lock.acquire()
                measurement_lock_acquired = True
                with self._measurement_lock:
                    previous_sample_count = self._measurement_sample_count
            try:
                yield
            except BaseException as exc:  # noqa: BLE001 - preserve the primary failure.
                primary_error = exc
        finally:
            if measurement_lock_acquired:
                self._measurement_read_lock.release()
                try:
                    assert previous_sample_count is not None
                    self._wait_for_fresh_measurement_sample(
                        previous_sample_count
                    )
                except BaseException as exc:  # noqa: BLE001 - fail closed.
                    cleanup_errors.append(exc)
                    with self._measurement_lock:
                        if self._measurement_fault is None:
                            self._measurement_fault = (
                                f"post-window measurement freshness failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
            try:
                # Re-arm only after the post-window qpos cut is available.
                # Force/torque and reader-error protection stayed active
                # throughout the window and during this wait.
                resume(token)
            except BaseException as exc:  # noqa: BLE001 - retain evidence.
                cleanup_errors.append(exc)

            if window_marked_active:
                with self._measurement_lock:
                    self._non_motion_read_window_active = False

            if primary_error is not None:
                for cleanup_error in cleanup_errors:
                    add_note = getattr(primary_error, "add_note", None)
                    note = (
                        "non-motion read window cleanup failure: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    if callable(add_note):
                        add_note(note)
                    else:
                        notes = getattr(primary_error, "__notes__", None)
                        if notes is None:
                            notes = []
                            setattr(primary_error, "__notes__", notes)
                        notes.append(note)
                raise primary_error
            if cleanup_errors:
                raise RuntimeError(
                    "RM65 non-motion read window cleanup failed: "
                    + "; ".join(
                        f"{type(error).__name__}: {error}"
                        for error in cleanup_errors
                    )
                ) from cleanup_errors[0]

    def _wait_for_fresh_measurement_sample(
        self,
        previous_sample_count: int,
    ) -> None:
        """Require one sampler cut published after a read-only window."""

        timeout_sec = max(1.0, 4.0 / self.profile.state_refresh_hz)
        deadline = time.monotonic() + timeout_sec
        while True:
            with self._measurement_lock:
                fault = self._measurement_fault
                sample_count = self._measurement_sample_count
            if fault is not None:
                raise RuntimeError(
                    "RM65 measurement sampler failed after non-motion read: "
                    + fault
                )
            if sample_count > previous_sample_count:
                return
            thread = self._measurement_thread
            if thread is None or not thread.is_alive():
                raise RuntimeError(
                    "RM65 measurement sampler stopped before publishing a "
                    "fresh post-window qpos cut."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    "RM65 measurement sampler did not publish a fresh "
                    f"post-window qpos cut within {timeout_sec:.3f}s."
                )
            self._measurement_stop.wait(min(0.01, remaining))

    @property
    def startup_hold_receipt(self) -> RealSkillMotionHoldReceipt | None:
        return self._startup_hold_receipt

    @property
    def last_emergency_stop_receipt(
        self,
    ) -> RealSkillMotionEmergencyStopReceipt | None:
        with self._resource_lock:
            return self._last_emergency_stop_receipt

    def start(self) -> None:
        """Start both lower backends and reject any partial inventory."""

        with self._lifecycle_lock:
            with self._resource_lock:
                if self._resources_started:
                    return
                if (
                    self._resources_closed
                    or self._start_attempted
                    or self._closing
                ):
                    raise RuntimeError("closed RM65 Skill resources cannot restart.")
                self._start_attempted = True
            try:
                if self.udp_feedback is not None:
                    # Bind/wait before SDK attachment: a occupied port, stale
                    # stream or wrong force frame must never start motion/HOLD.
                    self._udp_feedback_started = True
                    self.udp_feedback.start()
                    self.udp_feedback.wait_ready(timeout_sec=2.,
                        max_age_sec=self.profile.max_measurement_age_sec)
                for side in ("left", "right"):
                    backend = self.arm_backends[side]
                    self._backend_start_attempted = True
                    backend.start()
                    if not backend.is_ready():
                        raise RuntimeError(f"{side} RM65 backend is not ready.")
                require_arm_backends_ready_for_position_dispatch(
                    self.arm_backends,
                    require_background_target=False,
                    stage="exclusive_real_skill_resource_start",
                    required_sides=("left", "right"),
                )
                startup_hold = self.hold("exclusive real Skill startup fence")
                if not startup_hold.succeeded:
                    raise RuntimeError(
                        "exclusive RM65 Skill startup HOLD was not confirmed."
                    )
                self._startup_hold_receipt = startup_hold
                # A start() implementation may acquire resources before raising,
                # so cleanup ownership begins before the call rather than after
                # a successful return.
                self._force_guard_started = True
                self.position_force_guard.start(arm_stale_watchdog=True)
                self._require_force_guard_motion_ready()
                with self._resource_lock:
                    self._resources_started = True
                if self.profile.background_measurement_sampling:
                    self._start_measurement_sampler()
            except Exception as startup_exc:
                with self._resource_lock:
                    self._resources_started = False
                    self._closing = True
                cleanup_errors = self._attempt_resource_shutdown(
                    reason="exclusive real Skill resource startup rollback",
                    request_hold=self._backend_start_attempted,
                )
                self._finish_resource_shutdown()
                if cleanup_errors:
                    raise RuntimeError(
                        f"{type(startup_exc).__name__}: {startup_exc}; "
                        "startup rollback incomplete: " + "; ".join(cleanup_errors)
                    ) from startup_exc
                raise
            with self._resource_lock:
                self._resources_started = True

    def close(self) -> None:
        """Fence motion and retry only unfinished lower cleanup stages."""

        with self._lifecycle_lock:
            with self._resource_lock:
                if self._resources_closed:
                    terminal_error = self._terminal_close_error
                    if terminal_error is None:
                        return
                    raise RuntimeError(
                        "exclusive RM65 Skill resource close failed: "
                        + terminal_error
                    )
                # This is the admission fence.  It is permanent even when a
                # lower close stage fails and must be retried.
                self._resources_started = False
                self._closing = True
            errors = self._attempt_resource_shutdown(
                reason="exclusive real Skill resource shutdown",
                request_hold=self._backend_start_attempted,
            )
            self._finish_resource_shutdown()
            if errors:
                raise RuntimeError(
                    "exclusive RM65 Skill resource close failed: "
                    + "; ".join(errors)
                )

    def _attempt_resource_shutdown(self, *, reason: str, request_hold: bool = True) -> list[str]:
        """Attempt each owned stage once, retaining only failed stages."""

        errors: list[str] = []
        if request_hold and not self._shutdown_hold_attempted:
            self._shutdown_hold_attempted = True
            hold_receipt = self.hold(reason)
            self._shutdown_hold_confirmed = bool(hold_receipt.succeeded)
            if not self._shutdown_hold_confirmed:
                self._shutdown_hold_error = "lower HOLD was not confirmed"
        if self._shutdown_hold_error is not None:
            errors.append(self._shutdown_hold_error)

        measurement_sampler_stopped = self._stop_measurement_sampler()
        if not measurement_sampler_stopped:
            errors.append("measurement_sampler_join_timeout")

        measurement_refresh_executor_stopped = False
        if measurement_sampler_stopped:
            try:
                self._shutdown_measurement_refresh_executor()
            except Exception as exc:  # noqa: BLE001 - retain lower resources.
                errors.append(
                    "measurement_refresh_executor: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                measurement_refresh_executor_stopped = True

        if measurement_refresh_executor_stopped and self._force_guard_started:
            try:
                self.position_force_guard.close()
            except Exception as exc:  # noqa: BLE001 - retain live monitor dependencies.
                errors.append(f"force_guard: {type(exc).__name__}: {exc}")
            else:
                self._force_guard_started = False

        # A failed monitor close can leave SDK readers/watchdog callbacks alive.
        # Preserve both backend owners until that exact monitor confirms stop.
        for side in (
            ("right", "left")
            if measurement_refresh_executor_stopped and not self._force_guard_started
            else ()
        ):
            if side not in self._backend_close_pending:
                continue
            try:
                self.arm_backends[side].close()
            except Exception as exc:  # noqa: BLE001 - close the other side too.
                errors.append(f"{side}: {type(exc).__name__}: {exc}")
            else:
                self._backend_close_pending.remove(side)
        if (measurement_refresh_executor_stopped and not self._force_guard_started
                and not self._backend_close_pending and self._udp_feedback_started):
            try:
                self.udp_feedback.close()
            except Exception as exc:  # noqa: BLE001 - retain unfinished owner for retry.
                errors.append(f"udp_feedback: {type(exc).__name__}: {exc}")
            else:
                self._udp_feedback_started = False
        return errors

    def _finish_resource_shutdown(self) -> None:
        """Publish terminal ownership only after every resource is released."""

        cleanup_complete = (
            self._measurement_refresh_executor_shutdown
            and not self._force_guard_started
            and not self._backend_close_pending
            and not self._udp_feedback_started
        )
        with self._resource_lock:
            self._resources_started = False
            self._resources_closed = cleanup_complete
            if cleanup_complete:
                self._terminal_close_error = self._shutdown_hold_error

    def _require_force_guard_motion_ready(self) -> None:
        fault = self.position_force_guard.fault
        if fault is not None:
            reason = getattr(fault, "reason", type(fault).__name__)
            raise RuntimeError(
                f"RM65 Skill position force guard is faulted: {reason}."
            )
        if not bool(self.position_force_guard.motion_ready):
            raise RuntimeError(
                "RM65 Skill position force guard is not motion-ready."
            )

    def _start_measurement_sampler(self) -> None:
        """Publish one initial cut, then keep one lower-owned sampler alive."""

        initial = self._read_hardware_cut()
        self._publish_measurement_sample(initial)
        thread = OwnedThread(
            target=self._run_measurement_sampler,
            name="exclusive-rm65-measurement-sampler",
            daemon=False,
        )
        self._measurement_thread = thread
        thread.start()

    def _run_measurement_sampler(self) -> None:
        period = 1.0 / self.profile.state_refresh_hz
        deadline = time.monotonic() + period
        while not self._measurement_stop.wait(
            max(0.0, deadline - time.monotonic())
        ):
            try:
                cut = self._read_hardware_cut()
                self._publish_measurement_sample(cut)
            except RealSkillMotionObservationUnavailable as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"[:4096]
                with self._measurement_lock:
                    self._measurement_transient_failure_count += 1
                    self._measurement_consecutive_transient_failures += 1
                    consecutive = self._measurement_consecutive_transient_failures
                    self._last_measurement_transient_failure = diagnostic
                    if (
                        consecutive
                        > RM65_MAX_CONSECUTIVE_RECOVERABLE_MEASUREMENT_FAILURES
                    ):
                        self._measurement_fault = (
                            diagnostic
                            + "; max_recoverable_transient_failures="
                            + str(
                                RM65_MAX_CONSECUTIVE_RECOVERABLE_MEASUREMENT_FAILURES
                            )
                            + "; consecutive_transient_failures="
                            + str(consecutive)
                        )
                if (
                    consecutive
                    > RM65_MAX_CONSECUTIVE_RECOVERABLE_MEASUREMENT_FAILURES
                ):
                    self._measurement_stop.set()
                    return
            except Exception as exc:  # noqa: BLE001 - consumed fail-closed by owner.
                with self._measurement_lock:
                    self._measurement_fault = (
                        f"{type(exc).__name__}: {exc}"
                    )
                self._measurement_stop.set()
                return
            else:
                with self._measurement_lock:
                    self._measurement_consecutive_transient_failures = 0
            # Do not burst through missed deadlines after a slow SDK read.
            deadline = _next_periodic_measurement_deadline(
                previous_deadline=deadline,
                completed_at=time.monotonic(),
                period_sec=period,
            )

    def _publish_measurement_sample(
        self,
        cut: RealSkillMotionObservationCut,
    ) -> None:
        with self._measurement_lock:
            previous = self._latest_observation_cut
            current_sequences = dict(cut.source_sequence_by_side)
            if previous is not None and current_sequences:
                previous_sequences = dict(previous.source_sequence_by_side)
                regressed = [
                    side
                    for side in ("left", "right")
                    if current_sequences[side] < previous_sequences[side]
                ]
                if regressed:
                    raise RuntimeError(
                        "hardware measurement source sequence regressed for "
                        + ",".join(regressed)
                        + "."
                    )
                if current_sequences == previous_sequences:
                    same_measurement = bool(
                        cut.measured_at == previous.measured_at
                        and all(
                            np.array_equal(
                                cut.qpos_by_side[side],
                                previous.qpos_by_side[side],
                            )
                            for side in ("left", "right")
                        )
                    )
                    if not same_measurement:
                        raise RuntimeError(
                            "hardware measurement changed without a new source "
                            "sequence."
                        )
                    return
            self._latest_observation_cut = cut
            self._measurement_sample_count += 1

    def _stop_measurement_sampler(self) -> bool:
        self._measurement_stop.set()
        thread = self._measurement_thread
        if thread is None or thread is threading.current_thread():
            return thread is None
        return thread.join_and_confirm(
            timeout=max(1.0, 4.0 / self.profile.state_refresh_hz)
        )

    def _shutdown_measurement_refresh_executor(self) -> None:
        if self._measurement_refresh_executor_shutdown:
            return
        executor = self._measurement_refresh_executor
        if executor is None:
            self._measurement_refresh_executor_shutdown = True
            return
        executor.shutdown(wait=True, cancel_futures=True)
        self._measurement_refresh_executor_shutdown = True

    def measurement_sampler_snapshot(self) -> dict[str, object]:
        """Return bounded timing/fault evidence without backend I/O."""

        with self._measurement_lock:
            cut = self._latest_observation_cut
            fault = self._measurement_fault
            count = self._measurement_sample_count
            transient_count = self._measurement_transient_failure_count
            consecutive_transient = (
                self._measurement_consecutive_transient_failures
            )
            last_transient = self._last_measurement_transient_failure
            non_motion_read_window_active = (
                self._non_motion_read_window_active
            )
        now = float(self.clock())
        return {
            "enabled": self.profile.background_measurement_sampling,
            **({"udp_feedback": self.udp_feedback.diagnostics()}
               if self.udp_feedback is not None else {}),
            "thread_alive": bool(
                self._measurement_thread is not None
                and self._measurement_thread.is_alive()
            ),
            "sample_count": count,
            "fault": fault,
            "transient_failure_count": transient_count,
            "consecutive_transient_failures": consecutive_transient,
            "max_consecutive_recoverable_transient_failures": (
                RM65_MAX_CONSECUTIVE_RECOVERABLE_MEASUREMENT_FAILURES
            ),
            "last_transient_failure": last_transient,
            "non_motion_read_window_active": (
                non_motion_read_window_active
            ),
            "latest_measurement_age_sec": (
                None if cut is None else max(0.0, now - cut.measured_at)
            ),
            "latest_available_age_sec": (
                None if cut is None else max(0.0, now - cut.available_at)
            ),
            "source_sequence_by_side": (
                {} if cut is None else dict(cut.source_sequence_by_side)
            ),
        }

    def arm_backend_diagnostics_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return cached, bounded terminal evidence without hardware reads."""

        snapshot: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            backend = self.arm_backends[side]
            try:
                snapshot[side] = _bounded_backend_diagnostics_state(
                    backend.get_state()
                )
            except Exception as exc:  # noqa: BLE001 - preserve unreadable side.
                snapshot[side] = {
                    "state_error_type": type(exc).__name__,
                    "state_error": _bounded_diagnostic_text(exc),
                }
        return snapshot

    def observe(self) -> RealSkillMotionObservationCut:
        """Consume the latest fresh cut; background mode performs no device I/O."""

        if not self.profile.background_measurement_sampling:
            return self._read_hardware_cut()
        if not self.resources_started:
            raise RuntimeError("exclusive RM65 Skill resources are not started.")
        with self._measurement_lock:
            cut = self._latest_observation_cut
            fault = self._measurement_fault
            non_motion_read_window_active = (
                self._non_motion_read_window_active
            )
        if non_motion_read_window_active:
            raise RealSkillMotionObservationPaused(
                "RM65 qpos publication is intentionally paused during a "
                "non-motion read window."
            )
        if fault is not None:
            raise RuntimeError("RM65 measurement sampler failed: " + fault)
        if cut is None:
            raise RuntimeError("RM65 measurement sampler has not published a cut.")
        age_sec = max(0.0, float(self.clock()) - cut.measured_at)
        if age_sec > self.profile.max_measurement_age_sec:
            diagnostic = self.measurement_sampler_snapshot()
            raise RealSkillMotionObservationUnavailable(
                f"RM65 sampled hardware qpos cut is stale: {age_sec:.6f}s; "
                "sampler_diagnostic="
                + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
            )
        return cut

    def _read_hardware_cut(self) -> RealSkillMotionObservationCut:
        with self._measurement_read_lock:
            return self._read_hardware_cut_unlocked()

    def _read_hardware_cut_unlocked(self) -> RealSkillMotionObservationCut:
        """Read one fresh dual hardware cut; command/preview qpos is forbidden."""

        if not self.resources_started:
            raise RuntimeError("exclusive RM65 Skill resources are not started.")
        requested_at = float(self.clock())
        # This is an explicit live observation request, not a historical
        # replay cut.  The reader refreshes first and then captures one shared
        # post-refresh causal cut so a just-measured hardware state is not
        # rejected merely because the SDK call took time.
        states = self.arm_state_reader.snapshot(
            requested_at,
            cut_after_refresh=True,
            concurrent_refresh_sides=("left", "right"),
        )
        available_at = max(requested_at, float(self.clock()))
        qpos_by_side: dict[str, np.ndarray] = {}
        measured_at_by_side: dict[str, float] = {}
        hardware_read_phase_timing_by_side: dict[str, dict[str, float]] = {}
        hardware_observation_phase_timing_by_side: dict[str, dict[str, float]] = {}
        evidence_payload: dict[str, Any] = {}
        source_sequence_by_side: dict[str, int] = {}
        for side in ("left", "right"):
            state = states.get(side)
            if not isinstance(state, Mapping):
                # This is a live observation fault, not caller type validation.
                raise RuntimeError(  # noqa: TRY004
                    f"{side} RM65 hardware state is missing."
                )
            if state.get("causal_snapshot_missing") is True:
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is causally unavailable "
                    f"at {state.get('causal_snapshot_cutoff_timestamp')!r}."
                )
            try:
                qpos = np.asarray(state.get("hardware_qpos"), dtype=float).reshape(6)
                measured_at = float(state.get("hardware_measurement_timestamp"))
                sequence = int(state.get("hardware_measurement_sequence"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is invalid."
                ) from exc
            if (
                not np.all(np.isfinite(qpos))
                or not math.isfinite(measured_at)
                or sequence <= 0
                or state.get("hardware_measurement_valid") is not True
                or state.get("hardware_read_success") is False
            ):
                raise RuntimeError(
                    f"{side} RM65 hardware qpos cut is not authoritative."
                )
            age_sec = max(0.0, available_at - measured_at)
            if age_sec > self.profile.max_measurement_age_sec:
                timing_by_side = {}
                observation_timing_by_side = {}
                age_by_side = {}
                measured_by_side = {}
                for diagnostic_side in ("left", "right"):
                    diagnostic_state = states.get(diagnostic_side)
                    if not isinstance(diagnostic_state, Mapping):
                        continue
                    try:
                        diagnostic_measured_at = float(
                            diagnostic_state.get("hardware_measurement_timestamp")
                        )
                    except (TypeError, ValueError, OverflowError):
                        continue
                    measured_by_side[diagnostic_side] = diagnostic_measured_at
                    age_by_side[diagnostic_side] = max(
                        0.0,
                        available_at - diagnostic_measured_at,
                    )
                    raw_timing = diagnostic_state.get("hardware_read_phase_timing")
                    if isinstance(raw_timing, Mapping):
                        timing_by_side[diagnostic_side] = {
                            str(key): float(value)
                            for key, value in raw_timing.items()
                            if isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                        }
                    raw_observation_timing = diagnostic_state.get(
                        "hardware_observation_phase_timing"
                    )
                    if isinstance(raw_observation_timing, Mapping):
                        observation_timing_by_side[diagnostic_side] = {
                            str(key): float(value)
                            for key, value in raw_observation_timing.items()
                            if isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and math.isfinite(float(value))
                        }
                diagnostic = {
                    "available_at": available_at,
                    "measurement_age_sec_by_side": age_by_side,
                    "measurement_span_sec": (
                        max(measured_by_side.values())
                        - min(measured_by_side.values())
                        if len(measured_by_side) == 2
                        else None
                    ),
                    "concurrently_refreshed_sides": ["left", "right"],
                    "hardware_read_phase_timing_by_side": timing_by_side,
                    "hardware_observation_phase_timing_by_side": (
                        observation_timing_by_side
                    ),
                }
                raise RealSkillMotionObservationUnavailable(
                    f"{side} RM65 hardware qpos cut is stale: {age_sec:.6f}s; "
                    "coherence_diagnostic="
                    + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
                )
            qpos_by_side[side] = qpos.copy()
            measured_at_by_side[side] = measured_at
            source_sequence_by_side[side] = sequence
            raw_timing = state.get("hardware_read_phase_timing")
            if isinstance(raw_timing, Mapping):
                hardware_read_phase_timing_by_side[side] = {
                    str(key): float(value)
                    for key, value in raw_timing.items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                }
            raw_observation_timing = state.get(
                "hardware_observation_phase_timing"
            )
            if isinstance(raw_observation_timing, Mapping):
                hardware_observation_phase_timing_by_side[side] = {
                    str(key): float(value)
                    for key, value in raw_observation_timing.items()
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                }
            evidence_payload[side] = {
                "measurement_timestamp": measured_at,
                "measurement_sequence": sequence,
                "qpos": tuple(float(value) for value in qpos),
            }
        measurement_span_sec = max(measured_at_by_side.values()) - min(
            measured_at_by_side.values()
        )
        evidence_hash = canonical_content_hash(
            {
                "available_at": available_at,
                "concurrently_refreshed_sides": ["left", "right"],
                "measurement_span_sec": measurement_span_sec,
                "sides": evidence_payload,
            },
            domain="dexteleop.real_skill.rm65_hardware_qpos_cut.v1",
        )
        return RealSkillMotionObservationCut(
            observation=TeleopObservation(
                timestamp=available_at,
                metadata={
                    "source": "exclusive_rm65_skill_deploy",
                    "hardware_qpos_cut_hash": evidence_hash,
                    "concurrently_refreshed_sides": ("left", "right"),
                    "hardware_measurement_span_sec": measurement_span_sec,
                    "hardware_read_phase_timing_by_side": (
                        hardware_read_phase_timing_by_side
                    ),
                    "hardware_observation_phase_timing_by_side": (
                        hardware_observation_phase_timing_by_side
                    ),
                },
            ),
            qpos_by_side=qpos_by_side,
            measured_at=min(measured_at_by_side.values()),
            available_at=available_at,
            evidence_refs=("rm65-hardware-qpos-cut:" + evidence_hash,),
            source_sequence_by_side=source_sequence_by_side,
        )

    def synchronize_target_builder(
        self,
        qpos_by_side: Mapping[str, np.ndarray],
    ) -> None:
        """Seed once, then follow the exact hardware cut without reset churn."""

        if set(qpos_by_side) != {"left", "right"}:
            raise ValueError("target synchronization requires left and right qpos.")
        with self._sync_lock:
            reset_tracking = not self._target_builder_synchronized
            self.target_builder.set_current_qpos_pair(
                left_qpos=np.asarray(qpos_by_side["left"], dtype=float).reshape(6),
                right_qpos=np.asarray(qpos_by_side["right"], dtype=float).reshape(6),
                reset_tracking=reset_tracking,
            )
            self._target_builder_synchronized = True

    def admit_command(
        self,
        command: TeleopCommand,
        observation: TeleopObservation,
        now: float,
    ) -> bool:
        """Require exact coupled joint targets and a ready lower inventory."""

        del observation, now
        if not self.resources_started:
            raise RuntimeError("exclusive RM65 Skill resources are not started.")
        self._require_force_guard_motion_ready()
        if set(command.commands) != {"left_arm", "right_arm"}:
            return False
        if not isinstance(
            command.metadata.get(REAL_AGENT_MOTION_COMMAND_METADATA_KEY),
            Mapping,
        ):
            return False
        for side in ("left", "right"):
            effector = command.commands[f"{side}_arm"]
            try:
                value = np.asarray(effector.value, dtype=float).reshape(6)
            except (TypeError, ValueError):
                return False
            if (
                effector.side != side
                or effector.command_type != "joint_vector"
                or not np.all(np.isfinite(value))
            ):
                return False
        require_arm_backends_ready_for_position_dispatch(
            self.arm_backends,
            require_background_target=False,
            stage="exclusive_real_skill_pre_dispatch",
            required_sides=("left", "right"),
        )
        return True

    def dispatch(self, command: TeleopCommand) -> RealSkillMotionDispatchReceipt:
        """Dispatch through the existing checked real position-reference path."""

        try:
            if not self.resources_started:
                raise RuntimeError(
                    "exclusive RM65 Skill resources are not started."
                )
            self._require_force_guard_motion_ready()
            batch = dispatch_position_reference_command(
                command,
                arm_backends=self.arm_backends,
                required_sides=("left", "right"),
                audit_mode=self.profile.dispatch_audit_mode,
            )
            succeeded = bool(batch.get("accepted", False))
            summary = {
                "accepted": succeeded,
                "audit_mode": batch.get("position_dispatch_audit_mode"),
                "effectors": {
                    name: {
                        "accepted": receipt.get("accepted"),
                        "command_type": receipt.get("command_type"),
                        "rewritten": receipt.get("rewritten"),
                        "result_code": receipt.get("result_code"),
                        "status": receipt.get("status"),
                    }
                    for name, receipt in dict(batch.get("effectors") or {}).items()
                },
            }
            post_health = dict(batch.get("arm_backend_health") or {})
            diagnostics = {
                "position_dispatch_timing": dict(
                    batch.get("position_dispatch_timing") or {}
                ),
                "backend_streaming_by_side": {
                    side: dict(
                        dict(post_health.get("sides") or {})
                        .get(side, {})
                        .get("background_streaming")
                        or {}
                    )
                    for side in ("left", "right")
                },
                "singularity_by_side": {
                    side: dict(
                        dict(post_health.get("sides") or {})
                        .get(side, {})
                        .get("singularity_metadata")
                        or {}
                    )
                    for side in ("left", "right")
                },
            }
        except Exception as exc:  # noqa: BLE001 - convert to fail-closed receipt.
            succeeded = False
            summary = {
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": _bounded_diagnostic_text(exc, limit=2048),
            }
            diagnostics = {
                "error_type": type(exc).__name__,
                "error": _bounded_diagnostic_text(exc, limit=2048),
            }
            logging.getLogger(__name__).exception("Real Skill lower dispatch failed")
        receipt_hash = canonical_content_hash(
            summary,
            domain="dexteleop.real_skill.rm65_dispatch_receipt.v1",
        )
        return RealSkillMotionDispatchReceipt(
            succeeded=succeeded,
            evidence_refs=("rm65-skill-dispatch:" + receipt_hash,),
            diagnostics=diagnostics,
        )

    def _confirm_arm_holds(self) -> None:
        deadline = time.monotonic() + self.profile.hold_confirmation_timeout_sec
        for side, backend in self.arm_backends.items():
            confirm = getattr(backend, "confirm_hold", None)
            if callable(confirm):
                confirm(timeout_sec=max(0.001, deadline - time.monotonic()))
            else:
                # Synchronous backend adapters must have withdrawn their
                # target before returning from hold_motion(). Native RM65 uses
                # confirm_hold() above and requires an actual worker SDK ACK.
                state = backend.get_state()
                background = state.get("background_streaming") or {}
                if background.get("target_active") or state.get("send_result_code") not in (None, 0):
                    raise RuntimeError(f"{side} backend has not confirmed HOLD.")

    def hold(self, reason: str) -> RealSkillMotionHoldReceipt:
        """Ask both lower backends to HOLD and compile their bounded result."""

        error_type: str | None = None
        try:
            hold_deploy_arm_backends(self.arm_backends, reason=str(reason))
            self._confirm_arm_holds()
            succeeded = True
        except Exception as exc:  # noqa: BLE001 - preserve a typed failed receipt.
            succeeded = False
            error_type = type(exc).__name__
            logging.getLogger(__name__).exception("Real Skill lower HOLD not confirmed")
        state_by_side: dict[str, Any] = {}
        for side, backend in self.arm_backends.items():
            try:
                state_by_side[side] = _bounded_backend_state(backend.get_state())
            except Exception as exc:  # noqa: BLE001 - receipt records unreadable state.
                succeeded = False
                state_by_side[side] = {
                    "state_error_type": type(exc).__name__,
                }
        receipt_hash = canonical_content_hash(
            {
                "succeeded": succeeded,
                "reason": str(reason),
                "confirmed_at": self.clock() if succeeded else None,
                "error_type": error_type,
                "state_by_side": state_by_side,
            },
            domain="dexteleop.real_skill.rm65_hold_receipt.v1",
        )
        return RealSkillMotionHoldReceipt(
            succeeded=succeeded,
            evidence_refs=("rm65-skill-hold:" + receipt_hash,),
        )

    def emergency_stop(self, reason: str) -> RealSkillMotionEmergencyStopReceipt:
        """Immediately fan out latched RM65 emergency stop to both arms.

        This path deliberately bypasses measurement, IK, motion admission, the
        command mailbox, and the control-cycle lock.  Each side must both
        return without error and expose its latched emergency-stop state before
        the combined receipt can succeed.
        """

        normalized_reason = str(reason).strip()
        if not normalized_reason:
            raise ValueError("emergency-stop reason must be non-empty.")
        errors = bimanual_emergency_stop(
            self.arm_backends,
            reason=normalized_reason,
        )
        status_by_side: dict[str, str] = {}
        evidence_refs: list[str] = []
        for side in ("left", "right"):
            backend = self.arm_backends[side]
            state: Mapping[str, Any]
            try:
                state = _bounded_backend_state(backend.get_state())
            except Exception as exc:  # noqa: BLE001 - retain side evidence.
                state = {"state_error_type": type(exc).__name__}
            stop_state_reader = getattr(backend, "is_emergency_stopped", None)
            try:
                latched = bool(
                    stop_state_reader()
                    if callable(stop_state_reader)
                    else state.get("emergency_stopped") is True
                )
            except Exception as exc:  # noqa: BLE001 - retain side evidence.
                latched = False
                state = {
                    **dict(state),
                    "emergency_stop_state_error_type": type(exc).__name__,
                }
            confirmed = bool(
                side not in errors and latched
            )
            status = "confirmed" if confirmed else "unconfirmed"
            status_by_side[side] = status
            side_hash = canonical_content_hash(
                {
                    "side": side,
                    "reason": normalized_reason,
                    "status": status,
                    "error": errors.get(side),
                    "state": dict(state),
                },
                domain="dexteleop.real_skill.rm65_emergency_stop_side.v1",
            )
            evidence_refs.append(
                f"rm65-skill-emergency-stop:{side}:{status}:{side_hash}"
            )
        receipt = RealSkillMotionEmergencyStopReceipt(
            succeeded=set(status_by_side.values()) == {"confirmed"},
            evidence_refs=tuple(evidence_refs),
            status_by_side=status_by_side,
        )
        with self._resource_lock:
            self._last_emergency_stop_receipt = receipt
        return receipt


__all__ = [
    "ExclusiveRM65SkillDeployPorts",
    "ExclusiveRM65SkillPortProfile",
    "RM65SkillPositionForceGuard",
    "RM65SkillTargetBuilder",
    "build_exclusive_rm65_skill_position_force_guard",
]
