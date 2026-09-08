"""RM65 application ports for the independent real Skill control owner.

This module binds already-started dual RM65 backends to the generic Skill
control cycle.  It reuses the policy-deploy state, dispatch, and HOLD owners;
it does not construct RealTeleop, a Gateway, a device, or a control thread.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
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
)
from .motion_port import REAL_AGENT_MOTION_COMMAND_METADATA_KEY


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

    def __post_init__(self) -> None:
        for field_name in ("state_refresh_hz", "max_measurement_age_sec"):
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
        self._measurement_stop = threading.Event()
        self._measurement_thread: threading.Thread | None = None
        self._latest_observation_cut: RealSkillMotionObservationCut | None = None
        self._measurement_fault: str | None = None
        self._measurement_sample_count = 0

    @property
    def resources_started(self) -> bool:
        with self._resource_lock:
            return self._resources_started

    @property
    def resources_closed(self) -> bool:
        with self._resource_lock:
            return self._resources_closed

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
                for side in ("left", "right"):
                    backend = self.arm_backends[side]
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
                    reason="exclusive real Skill resource startup rollback"
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
                reason="exclusive real Skill resource shutdown"
            )
            self._finish_resource_shutdown()
            if errors:
                raise RuntimeError(
                    "exclusive RM65 Skill resource close failed: "
                    + "; ".join(errors)
                )

    def _attempt_resource_shutdown(self, *, reason: str) -> list[str]:
        """Attempt each owned stage once, retaining only failed stages."""

        errors: list[str] = []
        if not self._shutdown_hold_attempted:
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
            except Exception as exc:  # noqa: BLE001 - still close both arms.
                errors.append(f"force_guard: {type(exc).__name__}: {exc}")
            else:
                self._force_guard_started = False

        for side in (
            ("right", "left") if measurement_refresh_executor_stopped else ()
        ):
            if side not in self._backend_close_pending:
                continue
            try:
                self.arm_backends[side].close()
            except Exception as exc:  # noqa: BLE001 - close the other side too.
                errors.append(f"{side}: {type(exc).__name__}: {exc}")
            else:
                self._backend_close_pending.remove(side)
        return errors

    def _finish_resource_shutdown(self) -> None:
        """Publish terminal ownership only after every resource is released."""

        cleanup_complete = (
            self._measurement_refresh_executor_shutdown
            and not self._force_guard_started
            and not self._backend_close_pending
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
        thread = threading.Thread(
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
            except Exception as exc:  # noqa: BLE001 - consumed fail-closed by owner.
                with self._measurement_lock:
                    self._measurement_fault = (
                        f"{type(exc).__name__}: {exc}"
                    )
                self._measurement_stop.set()
                return
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
        thread.join(timeout=max(1.0, 4.0 / self.profile.state_refresh_hz))
        return not thread.is_alive()

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
        now = float(self.clock())
        return {
            "enabled": self.profile.background_measurement_sampling,
            "thread_alive": bool(
                self._measurement_thread is not None
                and self._measurement_thread.is_alive()
            ),
            "sample_count": count,
            "fault": fault,
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

    def observe(self) -> RealSkillMotionObservationCut:
        """Consume the latest fresh cut; background mode performs no device I/O."""

        if not self.profile.background_measurement_sampling:
            return self._read_hardware_cut()
        if not self.resources_started:
            raise RuntimeError("exclusive RM65 Skill resources are not started.")
        with self._measurement_lock:
            cut = self._latest_observation_cut
            fault = self._measurement_fault
        if fault is not None:
            raise RuntimeError("RM65 measurement sampler failed: " + fault)
        if cut is None:
            raise RuntimeError("RM65 measurement sampler has not published a cut.")
        age_sec = max(0.0, float(self.clock()) - cut.measured_at)
        if age_sec > self.profile.max_measurement_age_sec:
            diagnostic = self.measurement_sampler_snapshot()
            raise RuntimeError(
                f"RM65 sampled hardware qpos cut is stale: {age_sec:.6f}s; "
                "sampler_diagnostic="
                + json.dumps(diagnostic, sort_keys=True, separators=(",", ":"))
            )
        return cut

    def _read_hardware_cut(self) -> RealSkillMotionObservationCut:
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
                }
                raise RuntimeError(
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
            }
        except Exception as exc:  # noqa: BLE001 - convert to fail-closed receipt.
            succeeded = False
            summary = {
                "accepted": False,
                "error_type": type(exc).__name__,
            }
            diagnostics = {
                "error_type": type(exc).__name__,
            }
        receipt_hash = canonical_content_hash(
            summary,
            domain="dexteleop.real_skill.rm65_dispatch_receipt.v1",
        )
        return RealSkillMotionDispatchReceipt(
            succeeded=succeeded,
            evidence_refs=("rm65-skill-dispatch:" + receipt_hash,),
            diagnostics=diagnostics,
        )

    def hold(self, reason: str) -> RealSkillMotionHoldReceipt:
        """Ask both lower backends to HOLD and compile their bounded result."""

        error_type: str | None = None
        try:
            hold_deploy_arm_backends(self.arm_backends, reason=str(reason))
            succeeded = True
        except Exception as exc:  # noqa: BLE001 - preserve a typed failed receipt.
            succeeded = False
            error_type = type(exc).__name__
        state_by_side: dict[str, Any] = {}
        for side, backend in self.arm_backends.items():
            try:
                state_by_side[side] = _bounded_backend_state(backend.get_state())
            except Exception as exc:  # noqa: BLE001 - receipt records unreadable state.
                state_by_side[side] = {
                    "state_error_type": type(exc).__name__,
                }
        receipt_hash = canonical_content_hash(
            {
                "succeeded": succeeded,
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
