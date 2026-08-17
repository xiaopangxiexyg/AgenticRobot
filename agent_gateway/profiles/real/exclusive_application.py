"""Process-level owner for the independent exclusive real Skill deployment."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .exclusive_factory import ExclusiveRealAgentPrimitiveDeployment


class ExclusiveRealSkillResources(Protocol):
    """Lower resource transaction; the Gateway never receives this object."""

    def start(self) -> None: ...

    def close(self) -> None: ...


class ExclusiveRealSkillServer(Protocol):
    """Bounded server lifecycle needed by the application owner."""

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
        self._state = ExclusiveRealSkillApplicationState.CREATED
        self._server: ExclusiveRealSkillServer | None = None
        self._control_thread: threading.Thread | None = None
        self._control_stop = threading.Event()
        self._serving = threading.Event()
        self._control_fault: BaseException | None = None
        self._startup_admission_receipt: object | None = None

    @property
    def state(self) -> ExclusiveRealSkillApplicationState:
        with self._lock:
            return self._state

    @property
    def control_fault(self) -> BaseException | None:
        with self._lock:
            return self._control_fault

    @property
    def startup_admission_receipt(self) -> object | None:
        with self._lock:
            return self._startup_admission_receipt

    def start(self) -> None:
        """Pass lower startup admission before publishing or exposing transport."""

        with self._lock:
            if self._state != ExclusiveRealSkillApplicationState.CREATED:
                raise RuntimeError("exclusive real Skill application starts once.")
        resources_started = False
        runtime_started = False
        server: ExclusiveRealSkillServer | None = None
        try:
            self.resources.start()
            resources_started = True
            self.deployment.control_runtime.start()
            runtime_started = True
            startup_receipt = self.startup_admission()
            if startup_receipt is None:
                raise RuntimeError(
                    "exclusive real Skill startup admission returned no receipt."
                )
            with self._lock:
                self._startup_admission_receipt = startup_receipt
            self.deployment.control_runtime.step()
            server = self.server_factory()
            for name in ("serve_forever", "shutdown", "server_close"):
                if not callable(getattr(server, name, None)):
                    raise TypeError(f"server must expose {name}.")
            thread = threading.Thread(
                target=self._run_control_loop,
                name="exclusive-real-skill-control",
                daemon=False,
            )
            with self._lock:
                self._server = server
                self._control_thread = thread
                self._state = ExclusiveRealSkillApplicationState.RUNNING
            thread.start()
        except Exception:
            if runtime_started:
                try:
                    self.deployment.control_runtime.fail_closed(
                        "exclusive_skill_application_start_failed"
                    )
                except Exception:
                    pass
            if server is not None:
                try:
                    server.server_close()
                except Exception:
                    pass
            if resources_started:
                try:
                    self.resources.close()
                except Exception:
                    pass
            with self._lock:
                self._state = ExclusiveRealSkillApplicationState.FAULT
            raise

    def _run_control_loop(self) -> None:
        period = 1.0 / self.config.control_hz
        deadline = time.monotonic()
        while not self._control_stop.is_set():
            try:
                self.deployment.control_runtime.step()
            except BaseException as exc:  # noqa: BLE001 - fail closed then stop.
                try:
                    self.deployment.control_runtime.fail_closed(
                        "exclusive_skill_control_cycle_failed"
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
                if self._serving.is_set() and server is not None:
                    server.shutdown()
                return
            deadline += period
            self._control_stop.wait(max(0.0, deadline - time.monotonic()))

    def serve_forever(
        self,
        *,
        ready_callback: Callable[[], None] | None = None,
    ) -> None:
        """Run the sole Gateway server until shutdown or a lower control fault.

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
        """Fence Gateway and future cycles, then HOLD and close resources."""

        with self._lock:
            if self._state == ExclusiveRealSkillApplicationState.CLOSED:
                return
            server = self._server
            thread = self._control_thread
        errors: list[str] = []
        try:
            self.deployment.gateway_session.close()
        except Exception as exc:  # noqa: BLE001 - continue lower shutdown.
            errors.append(f"gateway_close:{type(exc).__name__}:{exc}")
        # Prevent another cycle from entering after the serialized HOLD below.
        # An already-running cycle is allowed to finish before HOLD becomes the
        # barrier; the runtime's cycle lock makes that ordering explicit.
        self._control_stop.set()
        try:
            self.deployment.control_runtime.close()
        except Exception as exc:  # noqa: BLE001 - continue resource shutdown.
            errors.append(f"control_close:{type(exc).__name__}:{exc}")
        if self._serving.is_set() and server is not None:
            try:
                server.shutdown()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"server_shutdown:{type(exc).__name__}:{exc}")
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.config.control_join_timeout_sec)
            if thread.is_alive():
                errors.append("control_thread_join_timeout")
        if server is not None:
            try:
                server.server_close()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"server_close:{type(exc).__name__}:{exc}")
        try:
            self.resources.close()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"resources_close:{type(exc).__name__}:{exc}")
        with self._lock:
            self._state = ExclusiveRealSkillApplicationState.CLOSED
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
