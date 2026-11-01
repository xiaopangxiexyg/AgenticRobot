"""Application-owned hardware generations with resident cameras and models.

This is an operator control surface, never a Skill or an SDK execution owner.
Each generation reuses the canonical application start/close transaction.
The same operator may also open or close either gripper of the connected
generation while no Skill operation runs (``operator_gripper``).
"""

from __future__ import annotations

import threading
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from agent_gateway.skills.gripper.runtime import GripperConvergenceProfile
from utils.owned_thread import OwnedThread

_LOG = logging.getLogger(__name__)

OPERATOR_GRIPPER_SCHEMA = "agent_gateway.operator_gripper.v1"
# Full DMTAC travel takes about 1-2 s; a command that has not settled by then is reported, never retried.
OPERATOR_GRIPPER_TIMEOUT_SEC = 5.0
_OPERATOR_GRIPPER_COMPONENTS = ("left_gripper", "right_gripper")


def run_operator_gripper_command(port: Any, target_state: str, *,
                                 timeout_sec: float = OPERATOR_GRIPPER_TIMEOUT_SEC,
                                 convergence: GripperConvergenceProfile | None = None,
                                 clock: Callable[[], float] = time.monotonic,
                                 sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Open or close one gripper through its Skill port and wait until it settles.

    Uses the port's own command path (arm HOLD guard, command-attempt record,
    so a held object's retained grip is no longer trusted) and the Skill's
    convergence rule: the target within tolerance, or a close that stopped on
    contact, for consecutive non-moving samples. Then the gripper is held.
    """
    convergence = convergence or GripperConvergenceProfile()
    target = 0.0 if target_state == "open" else 1.0
    started = clock()
    receipt = port.command_close(target)
    evidence = list(receipt.evidence_refs)
    result: dict[str, Any] = {"schema": OPERATOR_GRIPPER_SCHEMA, "component_id": port.component_id,
                              "target_state": target_state, "command_accepted": bool(receipt.accepted)}
    if not receipt.accepted:
        return {**result, "status": "command_rejected", "elapsed_sec": clock() - started,
                "evidence_refs": evidence[-8:]}
    sample, kind, stable = None, None, 0
    try:
        while clock() - started < timeout_sec:
            sample = port.poll_state()
            evidence.append(sample.evidence_ref)
            if sample.fault_code is not None:
                kind = "fault"
                break
            reached = sample.moving is False and abs(sample.measured_close - target) <= convergence.target_tolerance
            contact = (target_state == "closed" and sample.moving is False
                       and clock() - started >= convergence.minimum_contact_settle_sec
                       and sample.measured_close >= convergence.minimum_closed_position)
            current = "target_reached" if reached else "settled_before_full_close" if contact else None
            stable = stable + 1 if current is not None and current == kind else (1 if current else 0)
            kind = current
            if stable >= convergence.stable_sample_count:
                break
            sleep(convergence.poll_interval_sec)
    except BaseException:
        # A failed read after a sent command still stops the hand before reporting.
        try:
            port.hold(reason="operator_gripper_read_failed")
        except Exception:
            _LOG.exception("Operator gripper HOLD after a failed read also failed: %s", port.component_id)
        raise
    settled = kind in {"target_reached", "settled_before_full_close"} and stable >= convergence.stable_sample_count
    status = kind if settled else "fault" if kind == "fault" else "timeout"
    hold = port.hold(reason="operator_gripper_" + ("settled" if settled else status))
    evidence.extend(hold.evidence_refs)
    return {**result, "status": status, "hold_confirmed": bool(hold.succeeded),
            "measured_close": None if sample is None else sample.measured_close,
            "moving": None if sample is None else sample.moving,
            "fault_code": None if sample is None else sample.fault_code,
            "elapsed_sec": clock() - started, "evidence_refs": evidence[-8:]}


class ResidentPerceptionResources:
    """Keep exactly one camera/model owner across hardware generations."""

    def __init__(self) -> None:
        self.source_binding = None
        self._camera_started = False
        self._camera_closed = False
        self._camera_error: BaseException | None = None
        self._models: list[ResidentModelLease] = []

    def model(self, model: Any) -> Any:
        if model is None:
            return None
        lease = ResidentModelLease(model)
        self._models.append(lease)
        return lease

    def start_cameras(self, sources: Mapping[str, Any]) -> None:
        from .realsense_perception import start_realsense_sources

        if not self._camera_started:
            # Ownership begins before SDK entry; a failed start is not retried.
            self._camera_started = True
            try:
                start_realsense_sources(sources)
            except BaseException as error:
                self._camera_error = error
                raise
        if self._camera_error is not None:
            raise RuntimeError("Resident camera startup did not complete; no retry was attempted") from self._camera_error

    def retain_cameras(self, _sources: Mapping[str, Any]) -> None:
        """A generation must not release session-owned cameras."""

    def close(self) -> None:
        from .realsense_perception import _formal_close_camera_sources

        errors = []
        for model in reversed(self._models):
            try:
                model.release()
            except Exception as error:
                errors.append(f"model:{type(error).__name__}:{error}")
        if self._camera_started and not self._camera_closed and self.source_binding is not None:
            try:
                _formal_close_camera_sources(self.source_binding.camera_sources)
            except Exception as error:
                errors.append(f"cameras:{type(error).__name__}:{error}")
            else:
                self._camera_closed = True
        if errors:
            raise RuntimeError("Resident perception close incomplete: " + "; ".join(errors))


class ResidentModelLease:
    """Delegate inference while the session retains model lifecycle ownership."""

    def __init__(self, model: Any) -> None:
        self._model = model
        self._started = False
        self._attempted = False
        self._start_error: BaseException | None = None
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Resident model has been closed")
        if self._start_error is not None:
            raise RuntimeError("Resident model startup did not complete") from self._start_error
        if not self._started:
            self._attempted = True
            try:
                self._model.start()
            except BaseException as error:
                self._start_error = error
                raise
            self._started = True

    def close(self) -> None:
        """Generation provider release; the model remains resident."""

    def release(self) -> None:
        if self._attempted and not self._closed:
            self._model.close()
            self._closed = True


class GenerationServer:
    """The supervisor owns transport; a generation owns only its control loop."""

    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass

    def serve_forever(self, poll_interval: float = 0.05) -> None:
        raise RuntimeError("Hardware generations must not start a second server")


class RealHardwareConnectionOwner:
    """Serialize explicit release/rebuild without reviving an old application."""

    def __init__(self, *, initial: Any, rebuild: Callable[[], Any],
                 resident: ResidentPerceptionResources, operator_client_ids: frozenset[str],
                 server_factory: Callable[[Any], Any],
                 prepare_generation: Callable[[Any], None] | None = None,
                 event_sink: Callable[[Mapping[str, Any]], None] | None = None) -> None:
        self._validate_component_scope(initial)
        self.current = initial
        self._rebuild = rebuild
        self._resident = resident
        self._operators = operator_client_ids
        self._server_factory = server_factory
        self._prepare_generation = prepare_generation
        self._event_sink = event_sink
        self._lock = threading.RLock()
        self._transition: OwnedThread | None = None
        self._state = "connected"
        self._generation = 0
        self._error: str | None = None
        self._emergency_latched = False
        self._closing = False
        self._closed = False
        self._server = None
        self._serving = threading.Event()
        # Component id of the operator gripper command in progress (None when idle).
        self._gripper_command: str | None = None

    @staticmethod
    def _validate_component_scope(generation: Any) -> None:
        if set(generation.deployment.gateway_session.gripper_ports) != {"left_gripper", "right_gripper"}:
            raise ValueError("Hardware connection control requires both canonical gripper ports")

    @property
    def state(self):
        from .exclusive_application import ExclusiveRealSkillApplicationState

        return (ExclusiveRealSkillApplicationState.CLOSED if self._closed
                else ExclusiveRealSkillApplicationState.FAULT
                if self._state == "connected" and self.current.application.control_fault is not None
                else ExclusiveRealSkillApplicationState.RUNNING)

    @property
    def transport(self):
        if self._server is None or not self._serving.is_set():
            raise RuntimeError("Hardware reconnect requires the retained live transport")
        return self._server

    @property
    def control_fault_reason(self) -> str | None:
        """Project the current canonical owner, never retain a previous generation's fault."""
        with self._lock:
            fault = self.current.application.control_fault
        return None if fault is None else f"{type(fault).__name__}:{fault}"[:512].strip()

    def _authorize(self, client_id: str) -> None:
        if client_id not in self._operators:
            raise PermissionError("Hardware connection control requires the operator peer")

    def _emit_state(self) -> None:
        if self._event_sink is not None:
            try:
                self._event_sink({"event": "hardware_connection_state",
                                  **self.snapshot(next(iter(self._operators)))})
            except Exception:
                # Optional diagnostics never change the established stop/release result.
                _LOG.exception("Hardware connection state logging failed")

    def snapshot(self, client_id: str) -> dict[str, Any]:
        self._authorize(client_id)
        with self._lock:
            fault = self.current.application.control_fault
            if self._state == "connected" and fault is not None:
                self._state = "faulted"
                self._error = f"control:{type(fault).__name__}:{fault}"[:512]
                _LOG.error("Hardware generation control failed: generation=%d runtime=%s",
                           self._generation, self.current.deployment.gateway_session.contract_bundle.route_manifest.runtime_session_id,
                           exc_info=(type(fault), fault, fault.__traceback__))
            manifest = self.current.deployment.gateway_session.contract_bundle.route_manifest
            return {
                "schema": "agent_gateway.hardware_connection.v1",
                "state": self._state, "generation": self._generation,
                "runtime_session_id": manifest.runtime_session_id,
                "route_manifest_hash": manifest.semantic_content_hash,
                "disconnect_components": ["left_arm", "right_arm", "left_gripper", "right_gripper"],
                "retained_resources": ["cameras", "perception_models"],
                "error": self._error,
                "reconnect_allowed": self._state == "disconnected" and not self._closing and not self._emergency_latched,
                "emergency_latched": self._emergency_latched,
                "operator_gripper_components": list(_OPERATOR_GRIPPER_COMPONENTS),
                "operator_gripper_busy": self._gripper_command,
            }

    def request(self, client_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._authorize(client_id)
        if set(payload) != {"action", "expected_generation"} or payload["action"] not in {"disconnect", "reconnect"}:
            raise ValueError("Invalid hardware connection action")
        expected = payload["expected_generation"]
        if type(expected) is not int or expected < 0:
            raise ValueError("Invalid hardware connection generation")
        with self._lock:
            if expected != self._generation or self._closing:
                raise RuntimeError("Hardware connection binding changed")
            if self._transition is not None and self._transition.is_alive():
                raise RuntimeError("Hardware connection transition is already pending")
            if payload["action"] == "reconnect":
                if self._state != "disconnected" or self._emergency_latched:
                    raise RuntimeError("Reconnect requires complete release and an unlatched control domain")
                self._state = "reconnecting"
                self._generation += 1
            else:
                if self._state == "disconnected":
                    return self.snapshot(client_id)
                self._state = "disconnecting"
            self._error = None
            _LOG.info("Hardware connection request: action=%s generation=%d runtime_session_id=%s",
                      payload["action"], self._generation,
                      self.current.deployment.gateway_session.contract_bundle.route_manifest.runtime_session_id)
            thread = OwnedThread(target=lambda: self._change(payload["action"]),
                                 name="real-hardware-connection", daemon=False)
            self._transition = thread
            try:
                thread.start()
            except BaseException as error:
                self._state = "faulted"
                self._error = f"{type(error).__name__}:{error}"[:512]
                raise
        self._emit_state()
        return self.snapshot(client_id)

    def operator_gripper(self, client_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Open or close one gripper of the connected generation for the operator.

        Not a Skill and not limited to the Agent's authorized gripper sides. Refused
        unless the hardware is connected, the generation matches, no stop is
        latched, the control loop has no fault and no Skill operation is active;
        new Skill submissions are refused while it runs.
        """
        self._authorize(client_id)
        if (set(payload) != {"component_id", "target_state", "expected_generation"}
                or payload["component_id"] not in _OPERATOR_GRIPPER_COMPONENTS
                or payload["target_state"] not in {"open", "closed"}):
            raise ValueError("Invalid operator gripper request")
        expected = payload["expected_generation"]
        if type(expected) is not int or expected < 0:
            raise ValueError("Invalid hardware connection generation")
        component = payload["component_id"]
        with self._lock:
            if expected != self._generation or self._closing:
                raise RuntimeError("Hardware connection binding changed")
            if self._state != "connected":
                raise RuntimeError("Hardware is disconnected or transitioning; reconnect before using a gripper")
            if self._emergency_latched:
                raise RuntimeError("Emergency stop is latched; the gripper is not commanded")
            if self._gripper_command is not None:
                raise RuntimeError("Another operator gripper command is still running")
            composition = self.current
            if composition.application.control_fault is not None:
                raise RuntimeError("Control fault latched; the gripper is not commanded")
            session = composition.deployment.gateway_session
            if session.service.operation_store.active_operation() is not None:
                raise RuntimeError("A Skill operation is active; wait until it finishes")
            port = session.gripper_ports[component]
            self._gripper_command = component
        _LOG.info("Operator gripper request: component=%s target=%s generation=%d",
                  component, payload["target_state"], expected)
        try:
            result = run_operator_gripper_command(port, payload["target_state"])
        finally:
            with self._lock:
                self._gripper_command = None
        _LOG.info("Operator gripper result: component=%s status=%s measured_close=%s hold=%s",
                  component, result["status"], result.get("measured_close"), result.get("hold_confirmed"))
        return {**result, "generation": expected,
                "runtime_session_id": session.contract_bundle.route_manifest.runtime_session_id}

    def _remember_latch(self) -> None:
        receipt = getattr(self.current.lower.ports, "last_emergency_stop_receipt", None)
        if receipt is not None:
            with self._lock:
                self._emergency_latched = True

    def _change(self, action: str) -> None:
        try:
            if action == "disconnect":
                self._remember_latch()
                self.current.application.close()
                self._remember_latch()
                with self._lock:
                    self._state = "disconnected"
            else:
                candidate = self._rebuild()
                # Retain failed startup acquisitions for the same close retry.
                with self._lock:
                    self.current = candidate
                    cancelled = self._emergency_latched or self._closing
                self._validate_component_scope(candidate)
                if cancelled:
                    candidate.application.close()
                    raise RuntimeError("Reconnect cancelled by stop or shutdown")
                if self._prepare_generation is not None:
                    self._prepare_generation(candidate)
                candidate.application.server_factory = GenerationServer
                candidate.application.start()
                with self._lock:
                    cancelled = self._emergency_latched or self._closing
                    if not cancelled:
                        self._state = "connected"
                if cancelled:
                    candidate.application.close()
                    raise RuntimeError("Reconnect cancelled by stop or shutdown")
            _LOG.info("Hardware connection transition complete: action=%s generation=%d state=%s",
                      action, self._generation, self._state)
        except BaseException as error:
            with self._lock:
                self._state = "faulted"
                self._error = f"{type(error).__name__}:{error}"[:512]
            _LOG.exception("Hardware connection transition failed: action=%s generation=%d state=%s",
                           action, self._generation, self._state)
        finally:
            self._emit_state()

    def facade(self, client_id: str, method: str | None = None, bound_facade=None):
        with self._lock:
            if method == "request_emergency_stop_own_domain":
                self._authorize(client_id)
                if bound_facade is None or bound_facade.service is self.current.deployment.gateway_session.service:
                    self._emergency_latched = True
            # A draining task may still need its exact old terminal operation
            # and stop receipt. Never route that reconciliation into a new owner.
            if bound_facade is not None and method in {
                "healthz", "get_own_operation", "cancel_own_operation",
                "cancel_own_read_only_operation", "request_hold_own_domain",
                "request_emergency_stop_own_domain",
            }:
                return bound_facade
            metadata = {"get_skill_definition_catalog", "get_agent_skill_schema_catalog",
                        "get_skill_route_manifest", "get_skill_io_catalog", "get_gateway_session_capabilities"}
            if self._gripper_command is not None and method == "submit_skill_request":
                raise RuntimeError("An operator gripper command is running; retry after it completes")
            if method is not None and self._state != "connected" and method not in metadata | {"healthz", "request_emergency_stop_own_domain"}:
                raise RuntimeError("Hardware is disconnected or transitioning; start a new task after reconnect")
            composition = self.current
            if bound_facade is not None and method not in metadata | {"healthz"} and bound_facade.service is not composition.deployment.gateway_session.service:
                raise RuntimeError("Hardware generation changed; rediscover before starting a new task")
        return composition.deployment.gateway_session.wire_adapter(
            authenticated_client_id=client_id,
            grant_resolver=composition.authorization.grant_resolver,
            read_only_access=client_id not in self._operators,
        )

    def serve_forever(self, ready_callback=None) -> None:
        self.current.application.server_factory = GenerationServer
        try:
            self.current.application.start()
            self._server = self._server_factory(self)
            self._serving.set()
            if ready_callback is not None:
                ready_callback()
            self._server.serve_forever(poll_interval=0.05)
        finally:
            self._serving.clear()
            self.close()

    def request_shutdown(self) -> None:
        if self._serving.is_set() and self._server is not None:
            self._server.shutdown()

    def close(self) -> None:
        with self._lock:
            self._closing = True
            thread = self._transition
        if thread is not None and not thread.join_and_confirm(timeout=3.0):
            raise RuntimeError("Hardware transition still owns resources; close is incomplete")
        self.current.application.close()
        if self._server is not None:
            self._server.server_close()
        self._resident.close()
        self._closed = True
