"""Process-level owner for the independent exclusive real Skill deployment."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from teleop_runtime.deploy.dispatch import PositionDispatchNotReady
from teleop_runtime.skill_deploy.control_runtime import RealSkillControlLifecycle
from utils.owned_thread import OwnedThread

from .exclusive_factory import ExclusiveRealAgentPrimitiveDeployment

_LOGGER = logging.getLogger(__name__)


class ExclusiveRealSkillResources(Protocol):
    """Lower resource transaction; the Gateway never receives this object."""

    def start(self) -> None: ...

    def close(self) -> None: ...


class ExclusiveRealSkillServer(Protocol):
    """Bounded server lifecycle needed by the application owner."""

    @property
    def control_fault_reader(self) -> Callable[[], str | None] | None: ...

    @control_fault_reader.setter
    def control_fault_reader(self, reader: Callable[[], str | None]) -> None: ...

    def serve_forever(self, poll_interval: float = 0.5) -> None: ...

    def shutdown(self) -> None: ...

    def server_close(self) -> None: ...


class ExclusiveRealSkillStartupAdmission(Protocol):
    """Trusted lower startup gate invoked before transport is exposed."""

    def __call__(self) -> object: ...


class ExclusiveRealSkillApplicationState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    FAULT = "fault"
    CLOSED = "closed"


@dataclass(frozen=True)
class ExclusiveRealSkillApplicationConfig:
    control_hz: float = 30.0
    server_poll_interval_sec: float = 0.05
    control_join_timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        for field_name in (
            "control_hz",
            "server_poll_interval_sec",
            "control_join_timeout_sec",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be positive and finite.")
            object.__setattr__(self, field_name, value)


class ExclusiveRealSkillApplication:
    """Own resources, one lower control thread, and one Gateway server."""

    def __init__(
        self,
        *,
        deployment: ExclusiveRealAgentPrimitiveDeployment,
        resources: ExclusiveRealSkillResources,
        startup_admission: ExclusiveRealSkillStartupAdmission,
        server_factory: Callable[[], ExclusiveRealSkillServer],
        config: ExclusiveRealSkillApplicationConfig | None = None,
    ) -> None:
        if not isinstance(deployment, ExclusiveRealAgentPrimitiveDeployment):
            raise TypeError("deployment must be typed.")
        for name, value in (
            ("resources.start", getattr(resources, "start", None)),
            ("resources.close", getattr(resources, "close", None)),
            ("startup_admission", startup_admission),
            ("server_factory", server_factory),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable.")
        self.deployment = deployment
        self.resources = resources
        self.startup_admission = startup_admission
        self.server_factory = server_factory
        self.config = config or ExclusiveRealSkillApplicationConfig()
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._state = ExclusiveRealSkillApplicationState.CREATED
        self._server: ExclusiveRealSkillServer | None = None
        self._control_thread: OwnedThread | None = None
        self._control_stop = threading.Event()
        self._serving = threading.Event()
        self._control_fault: BaseException | None = None
        self._startup_admission_receipt: object | None = None
        self._control_runtime_closed = False
        self._server_closed = False
        self._resources_closed = False
        self._resources_start_attempted = False
        self._control_runtime_start_attempted = False

    @property
    def state(self) -> ExclusiveRealSkillApplicationState:
        with self._lock:
            return self._state

    @property
    def control_fault(self) -> BaseException | None:
        with self._lock:
            return self._control_fault

    @property
    def control_fault_reason(self) -> str | None:
        """Read the application's existing fault/lifecycle state; no device I/O."""
        with self._lock:
            fault, state = self._control_fault, self._state
        if fault is not None:
            name = type(fault).__name__
            detail = " ".join(str(fault).split())
            return (f"{name}: {detail}" if detail else name)[:512].strip()
        if state != ExclusiveRealSkillApplicationState.RUNNING:
            return f"application_not_running:{state.value}"
        return None

    @property
    def startup_admission_receipt(self) -> object | None:
        with self._lock:
            return self._startup_admission_receipt

    def start(self) -> None:
        """Serialize startup and shutdown while preserving partial acquisition."""

        with self._close_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        """Pass lower startup admission before publishing or exposing transport."""

        with self._lock:
            if (self._state != ExclusiveRealSkillApplicationState.CREATED
                    or self._resources_start_attempted):
                raise RuntimeError("exclusive real Skill application starts once.")
            self._resources_start_attempted = True
        server: ExclusiveRealSkillServer | None = None
        try:
            self.resources.start()
            self._control_runtime_start_attempted = True
            self.deployment.control_runtime.start()
            startup_receipt = self.startup_admission()
            if startup_receipt is None:
                raise RuntimeError(
                    "exclusive real Skill startup admission returned no receipt."
                )
            with self._lock:
                self._startup_admission_receipt = startup_receipt
            if (
                self.deployment.gateway_session.contract_bundle.gripper_contract
                is not None
            ):
                # Keep the exact lower startup admission. Its already-required
                # HOLD is projected by the same owner before publishing Routes.
                self.deployment.control_runtime.request_startup_hold()
            self.deployment.control_runtime.step()
            server = self.server_factory()
            server.control_fault_reader = lambda: self.control_fault_reason
            for name in ("serve_forever", "shutdown", "server_close"):
                if not callable(getattr(server, name, None)):
                    raise TypeError(f"server must expose {name}.")
            thread = OwnedThread(
                target=self._run_control_loop,
                name="exclusive-real-skill-control",
                daemon=False,
            )
            with self._lock:
                self._server = server
                self._control_thread = thread
                self._state = ExclusiveRealSkillApplicationState.RUNNING
            thread.start()
        except Exception as start_error:
            cleanup_errors: list[str] = []
            if self._control_runtime_start_attempted:
                try:
                    self.deployment.control_runtime.fail_closed(
                        "exclusive_skill_application_start_failed"
                    )
                except Exception as error:  # noqa: BLE001 - retain failed safety evidence.
                    cleanup_errors.append(
                        f"startup_hold:{type(error).__name__}:{error}"
                    )
                # Reuse the normal composed shutdown owner. Failed arm or hand
                # stop proof keeps device resources available for a later close.
                with self._lock:
                    self._server = server
                try:
                    self.close()
                except Exception as error:  # noqa: BLE001 - preserve startup and cleanup faults.
                    cleanup_errors.append(
                        f"startup_close:{type(error).__name__}:{error}"
                    )
            elif self._resources_start_attempted:
                try:
                    self.close()
                except Exception as error:  # noqa: BLE001 - preserve cleanup for retry.
                    cleanup_errors.append(
                        f"startup_close:{type(error).__name__}:{error}"
                    )
            with self._lock:
                self._state = ExclusiveRealSkillApplicationState.FAULT
            if cleanup_errors:
                raise RuntimeError(
                    f"exclusive real Skill startup failed: {start_error}; "
                    + "; ".join(cleanup_errors)
                ) from start_error
            raise

    def _run_control_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        deadline = time.monotonic()
        while not self._control_stop.is_set():
            try:
                self.deployment.gateway_session.reconcile_gripper_connections(
                    self.deployment.control_runtime.fail_closed)
                self.deployment.control_runtime.step()
            except BaseException as exc:  # noqa: BLE001 - fail closed then stop.
                reason = (
                    "lower_backend_not_ready"
                    if isinstance(exc, PositionDispatchNotReady)
                    else "exclusive_skill_control_cycle_failed"
                )
                self.deployment.control_runtime.motion_port.fence_activations(reason)
                with self._lock:
                    server = self._server
                retain_fault = getattr(server, "retain_control_fault", None)
                diagnostics_retained = False
                if (callable(retain_fault)
                        and self.deployment.gateway_session.contract_bundle.gripper_contract is None):
                    try:
                        # Fence all submissions before HOLD. Only the existing
                        # transport may retain queries; it never resumes control.
                        diagnostics_retained = retain_fault(reason) is True
                    except Exception:  # noqa: BLE001 - old shutdown remains fail-closed.
                        _LOGGER.exception("Unable to fence diagnostic transport after control fault")
                        diagnostics_retained = False
                # Composed hand owners still require their existing close path
                # to stop active gripper commands; arm HOLD alone is not enough.
                try:
                    diagnostic = " ".join(str(exc).split())[:512]
                    self.deployment.control_runtime.fail_closed(
                        reason + (f": {diagnostic}" if diagnostic else "")
                    )
                except BaseException as hold_exc:  # noqa: BLE001
                    exc = RuntimeError(
                        "control cycle and lower HOLD both failed: "
                        f"{type(exc).__name__}; {type(hold_exc).__name__}"
                    )
                with self._lock:
                    self._control_fault = exc
                    self._state = ExclusiveRealSkillApplicationState.FAULT
                    server = self._server
                self._control_stop.set()
                if diagnostics_retained:
                    # The transport keeps serving, so make the latch visible to
                    # the operator instead of looking like a silent stall.
                    _LOGGER.error(
                        "Real Skill control loop fault latched: %s: %s. Lower HOLD was "
                        "requested; new Skill submissions are rejected and the Gateway "
                        "stays up only for queries. Press Ctrl+C to close it.",
                        type(exc).__name__, " ".join(str(exc).split())[:512],
                    )
                if self._serving.is_set() and server is not None and not diagnostics_retained:
                    server.shutdown()
                return
            deadline += period
            self._control_stop.wait(max(0.0, deadline - time.monotonic()))

    def serve_forever(
        self,
        *,
        ready_callback: Callable[[], None] | None = None,
    ) -> None:
        """Serve until explicit shutdown; faults retain fenced diagnostics.

        The callback is a process-host seam: it runs only after lower startup
        admission and server construction, immediately before transport
        serving.  It grants no motion or shutdown authority.
        """

        if ready_callback is not None and not callable(ready_callback):
            raise TypeError("ready_callback must be callable or None.")

        self.start()
        with self._lock:
            server = self._server
        assert server is not None
        self._serving.set()
        try:
            if ready_callback is not None:
                ready_callback()
            server.serve_forever(
                poll_interval=self.config.server_poll_interval_sec,
            )
        finally:
            self._serving.clear()
            self.close()
        fault = self.control_fault
        if fault is not None:
            raise RuntimeError(
                "exclusive real Skill control loop failed."
            ) from fault

    def request_shutdown(self) -> bool:
        """Ask the owned transport loop to stop; normal close still owns HOLD."""

        with self._lock:
            server = self._server
            serving = self._serving.is_set()
        if server is None or not serving:
            return False
        server.shutdown()
        return True

    def close(self) -> None:
        """Fence once, but retain every unconfirmed close stage for retry."""

        with self._close_lock:
            with self._lock:
                if self._state == ExclusiveRealSkillApplicationState.CLOSED:
                    return
                server = self._server
                thread = self._control_thread
            errors: list[str] = []
            gateway_closed = False
            try:
                if not self._control_runtime_start_attempted:
                    if (self.deployment.control_runtime.lifecycle != RealSkillControlLifecycle.CREATED
                            or server is not None or thread is not None
                            or self._serving.is_set()
                            or self._startup_admission_receipt is not None):
                        raise RuntimeError("Unstarted application acquisition facts disagree.")
                    self.deployment.gateway_session.close_unused()
                    gateway_closed = True
                else:
                    gateway_outcome = self.deployment.gateway_session.close()
                    gateway_closed = gateway_outcome.accepted
                    if not gateway_closed:
                        errors.append(
                            "gateway_close_not_confirmed:"
                            + gateway_outcome.reason_code
                        )
            except Exception as exc:  # noqa: BLE001 - keep retryable owner state.
                errors.append(f"gateway_close:{type(exc).__name__}:{exc}")

            # Prevent another cycle from entering after the serialized HOLD.
            # An already-running cycle may finish before the runtime cycle lock
            # establishes the barrier.
            self._control_stop.set()
            if self._control_runtime_start_attempted and not self._control_runtime_closed:
                try:
                    self.deployment.control_runtime.close()
                except Exception as exc:  # noqa: BLE001 - retry on next close.
                    errors.append(f"control_close:{type(exc).__name__}:{exc}")
                else:
                    self._control_runtime_closed = True

            if self._serving.is_set() and server is not None:
                try:
                    server.shutdown()
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"server_shutdown:{type(exc).__name__}:{exc}")

            control_thread_stopped = True
            if thread is threading.current_thread():
                control_thread_stopped = False
                errors.append("control_thread_cannot_join_itself")
            elif thread is not None:
                if not thread.join_and_confirm(timeout=self.config.control_join_timeout_sec):
                    control_thread_stopped = False
                    errors.append("control_thread_join_timeout")

            if server is not None and not self._server_closed:
                try:
                    server.server_close()
                except Exception as exc:  # noqa: BLE001 - retry transport close.
                    errors.append(f"server_close:{type(exc).__name__}:{exc}")
                else:
                    self._server_closed = True
            server_closed = server is None or self._server_closed

            prerequisites_closed = (
                gateway_closed
                # An unacquired control stage has no attachment or HOLD
                # receipt. Resource rollback remains its own shutdown owner.
                and (not self._control_runtime_start_attempted or self._control_runtime_closed)
                and control_thread_stopped
                and server_closed
            )
            if prerequisites_closed and not self._resources_closed:
                try:
                    self.resources.close()
                except Exception as exc:  # noqa: BLE001 - preserve for retry.
                    errors.append(f"resources_close:{type(exc).__name__}:{exc}")
                else:
                    self._resources_closed = True

            complete = prerequisites_closed and self._resources_closed
            with self._lock:
                self._state = (
                    ExclusiveRealSkillApplicationState.CLOSED
                    if complete and not errors
                    else ExclusiveRealSkillApplicationState.FAULT
                )
            if not complete and not errors:
                errors.append("close_transaction_incomplete")
            if errors:
                raise RuntimeError(
                    "exclusive real Skill application close failed: "
                    + "; ".join(errors)
                )


__all__ = [
    "ExclusiveRealSkillApplication",
    "ExclusiveRealSkillApplicationConfig",
    "ExclusiveRealSkillApplicationState",
    "ExclusiveRealSkillResources",
    "ExclusiveRealSkillServer",
    "ExclusiveRealSkillStartupAdmission",
]
