"""Reusable Gateway lifecycle and session for the atomic gripper Skill."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_gateway.protocol.contracts import (
    AgentObservationSnapshot,
    AgentSkillOperation,
    AgentSkillRequest,
)
from agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from agent_gateway.runtime.service import (
    AgentGatewayObservationOutcome,
    AgentGatewayOperationOutcome,
    AgentGatewayService,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleExecutionCompletion,
    StationOperationWorkerBinding,
)
from agent_gateway.transport.unix_transport import (
    AgentGatewayUnixJsonServer,
    AgentGatewayUnixPeerClientMap,
)
from agent_gateway.transport.wire_adapter import BoundAgentGatewayWireAdapter

from .definition import (
    SET_GRIPPER_STATE_RESULT_SCHEMA_V1,
    SET_GRIPPER_STATE_RESULT_SCHEMA_V2,
)
from .deployment import GripperSkillContractBundle


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _close_scalar(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be finite and in [0, 1].")
    return normalized


def _refs(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    refs = tuple(
        dict.fromkeys(_required(value, field_name=field_name) for value in values)
    )
    if not refs:
        raise ValueError(f"{field_name} must not be empty.")
    return refs


@dataclass(frozen=True)
class GripperReadiness:
    ready: bool
    lower_operation_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ready, bool):
            raise TypeError("ready must be boolean.")
        object.__setattr__(
            self,
            "lower_operation_id",
            _required(self.lower_operation_id, field_name="lower_operation_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _refs(self.evidence_refs, field_name="readiness evidence"),
        )


@dataclass(frozen=True)
class GripperCommandReceipt:
    accepted: bool
    lower_operation_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be boolean.")
        object.__setattr__(
            self,
            "lower_operation_id",
            _required(self.lower_operation_id, field_name="lower_operation_id"),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _refs(self.evidence_refs, field_name="command evidence"),
        )


@dataclass(frozen=True)
class GripperStateSample:
    measured_close: float
    moving: bool | None
    fault_code: str | None
    observed_at: float
    evidence_ref: str
    target_contact_confirmed: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measured_close",
            _close_scalar(self.measured_close, field_name="measured_close"),
        )
        if self.moving is not None and not isinstance(self.moving, bool):
            raise TypeError("moving must be boolean or None.")
        if self.target_contact_confirmed is not None and not isinstance(
            self.target_contact_confirmed, bool
        ):
            raise TypeError("target_contact_confirmed must be boolean or None.")
        if self.fault_code is not None:
            object.__setattr__(
                self,
                "fault_code",
                _required(self.fault_code, field_name="fault_code"),
            )
        observed = float(self.observed_at)
        if not math.isfinite(observed):
            raise ValueError("observed_at must be finite.")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(
            self,
            "evidence_ref",
            _required(self.evidence_ref, field_name="evidence_ref"),
        )


@dataclass(frozen=True)
class GripperHoldReceipt:
    succeeded: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be boolean.")
        object.__setattr__(
            self,
            "evidence_refs",
            _refs(self.evidence_refs, field_name="hold evidence"),
        )


def hold_gripper_ports(
    *, ports: Mapping[str, GripperControlPort], reason: str
) -> GripperHoldReceipt:
    """Attempt every configured stop, preserving the first process interrupt."""
    if not ports:
        raise ValueError("Gripper stop requires configured ports.")
    receipts, errors = [], []
    interrupted = None
    for component, port in ports.items():
        try:
            receipts.append(port.hold(reason=reason))
        except (KeyboardInterrupt, SystemExit) as error:
            if interrupted is None:
                interrupted = error
        except Exception as error:  # noqa: BLE001 - all independent stops must run.
            errors.append(f"{component}:{type(error).__name__}:{error}")
    if interrupted is not None:
        raise interrupted
    if errors:
        raise RuntimeError("Gripper stop failed: " + "; ".join(errors))
    return GripperHoldReceipt(
        all(item.succeeded for item in receipts),
        tuple(ref for item in receipts for ref in item.evidence_refs),
    )

@runtime_checkable
class GripperControlPort(Protocol):
    component_id: str
    target_kind: str
    port_binding_hash: str

    def readiness(self) -> GripperReadiness: ...

    def command_close(
        self, target_close: float, *, cancel_requested: Callable[[], bool] | None = None,
    ) -> GripperCommandReceipt: ...

    def observe_state(self) -> GripperStateSample: ...

    def poll_state(self) -> GripperStateSample: ...

    def hold(self, *, reason: str) -> GripperHoldReceipt: ...


@dataclass(frozen=True)
class GripperConvergenceProfile:
    target_tolerance: float = 0.04
    stable_sample_count: int = 3
    poll_interval_sec: float = 0.01
    minimum_contact_settle_sec: float = 0.10
    minimum_closed_position: float = 0.05

    def __post_init__(self) -> None:
        for field_name in (
            "target_tolerance",
            "poll_interval_sec",
            "minimum_contact_settle_sec",
            "minimum_closed_position",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative.")
            object.__setattr__(self, field_name, value)
        if not 0.0 < self.target_tolerance < 0.5:
            raise ValueError("target_tolerance must be in (0, 0.5).")
        if not 0.0 <= self.minimum_closed_position <= 1.0:
            raise ValueError("minimum_closed_position must be in [0, 1].")
        if (
            isinstance(self.stable_sample_count, bool)
            or not isinstance(self.stable_sample_count, int)
            or self.stable_sample_count <= 0
        ):
            raise ValueError("stable_sample_count must be a positive integer.")


class GripperSkillLifecycle:
    """One operation over one deployment-bound gripper control port."""

    def __init__(
        self,
        *,
        port: GripperControlPort,
        contract_bundle: GripperSkillContractBundle,
        convergence: GripperConvergenceProfile,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(port, GripperControlPort):
            raise TypeError("port must implement GripperControlPort.")
        if not isinstance(contract_bundle, GripperSkillContractBundle):
            raise TypeError("contract_bundle must be typed.")
        if not isinstance(convergence, GripperConvergenceProfile):
            raise TypeError("convergence must be typed.")
        binding = contract_bundle.deployment_binding
        if (
            port.target_kind != binding.target_kind
            or port.port_binding_hash != binding.port_hash_for(port.component_id)
        ):
            raise ValueError("port identity does not match deployment binding.")
        if (binding.skill_version == "3" and port.component_id == "right_gripper"
                and not all(callable(getattr(port, name, None)) for name in (
                    "command_open_to_support", "observe_named_release"))):
            raise TypeError("v3 right gripper lacks named release support binding.")
        if not callable(clock) or not callable(sleep):
            raise TypeError("clock and sleep must be callable.")
        self.port = port
        self.contract_bundle = contract_bundle
        self.deployment_binding = binding
        self.route = contract_bundle.route_manifest.skill_routes[0].route
        self.executor_id = binding.executor_id
        self.convergence = convergence
        self.clock = clock
        self.sleep = sleep
        self._lock = threading.RLock()
        self._activated_operation_id: str | None = None
        self._lower_operation_id: str | None = None
        self._hold_receipt: GripperHoldReceipt | None = None
        self._closed = False

    def _validate_call(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        expected_status: str,
    ) -> None:
        if operation.status != expected_status:
            raise ValueError(f"operation must be {expected_status}.")
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("operation/request hash mismatch.")
        checks = (
            request.runtime_session_id == self.deployment_binding.runtime_session_id,
            request.skill_id == self.route.skill_id,
            request.skill_version == self.route.skill_version,
            request.skill_definition_hash == self.route.skill_definition_hash,
            request.route_id == self.route.route_id,
            request.route_version == self.route.route_version,
            request.route_hash == self.route.semantic_content_hash,
            request.deployment_binding_hash
            == self.deployment_binding.semantic_content_hash,
        )
        if not all(checks):
            raise ValueError("request does not bind this exact gripper lifecycle.")
        if (
            self.deployment_binding.skill_version in {"2", "3"}
            and request.parameters["component_id"] != self.port.component_id
        ):
            raise ValueError("request selected a different gripper.")

    def _ensure_hold(self, reason: str) -> GripperHoldReceipt:
        with self._lock:
            if self._hold_receipt is None:
                self._hold_receipt = self.port.hold(
                    reason=_required(reason, field_name="reason")
                )
            return self._hold_receipt

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleActivationEvidence:
        self._validate_call(
            request=request, operation=operation, expected_status="activated"
        )
        readiness = self.port.readiness()
        with self._lock:
            if self._activated_operation_id is not None:
                raise RuntimeError("gripper lifecycle is single-use.")
            self._activated_operation_id = operation.operation_id
            self._lower_operation_id = readiness.lower_operation_id
        if not readiness.ready:
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="gripper_not_ready",
                lower_operation_id=readiness.lower_operation_id,
                evidence_refs=readiness.evidence_refs,
            )
        if bool(cancel_requested()):
            hold = self._ensure_hold("activation_preempted")
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="gripper_activation_preempted_under_hold",
                lower_operation_id=readiness.lower_operation_id,
                evidence_refs=(*readiness.evidence_refs, *hold.evidence_refs),
                lower_hold_confirmed=hold.succeeded,
            )
        return StationLifecycleActivationEvidence(
            succeeded=True,
            status="gripper_ready",
            lower_operation_id=readiness.lower_operation_id,
            evidence_refs=readiness.evidence_refs,
        )

    def _failed_completion(
        self,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> StationLifecycleExecutionCompletion:
        hold = self._ensure_hold(reason_code)
        return StationLifecycleExecutionCompletion(
            succeeded=False,
            reason_code=reason_code,
            execution_receipt_refs=tuple(
                dict.fromkeys((*evidence_refs, *hold.evidence_refs))
            ),
            terminal_hold_confirmed=hold.succeeded,
        )

    def run(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleExecutionCompletion:
        self._validate_call(
            request=request, operation=operation, expected_status="running"
        )
        if self._activated_operation_id != operation.operation_id:
            raise ValueError("run does not match activation.")
        target_state = str(request.parameters["target_state"])
        target_close = 0.0 if target_state == "open" else 1.0
        preempted = lambda: bool(cancel_requested()) or float(self.clock()) >= operation.deadline_at
        release_support = request.parameters.get("release_support_object_id")
        if release_support is not None:
            if (self.deployment_binding.skill_version != "3" or target_state != "open"
                    or self.port.component_id != "right_gripper"):
                raise ValueError("Named release differs from the admitted gripper Route.")
            command = self.port.command_open_to_support(
                release_support, cancel_requested=preempted)
        else:
            command = self.port.command_close(target_close, cancel_requested=preempted)
        # Cancellation and HOLD stay bound to the activation's lower owner.
        cancelled = bool(cancel_requested())
        expired = float(self.clock()) >= operation.deadline_at
        if not command.accepted or cancelled or expired:
            return self._failed_completion(
                reason_code=(
                    "gripper_execution_preempted" if cancelled
                    else "gripper_convergence_timeout" if expired
                    else "gripper_command_rejected"
                ),
                evidence_refs=command.evidence_refs,
            )
        started_at = float(self.clock())
        stable_count = 0
        final_sample: GripperStateSample | None = None
        completion_kind: str | None = None
        evidence = list(command.evidence_refs)
        command_evidence_count = len(evidence)
        while float(self.clock()) < operation.deadline_at:
            if bool(cancel_requested()):
                return self._failed_completion(
                    reason_code="gripper_execution_preempted",
                    evidence_refs=tuple(evidence),
                )
            sample = self.port.poll_state()
            final_sample = sample
            evidence.append(sample.evidence_ref)
            if len(evidence) > 120:
                evidence = [
                    *evidence[:command_evidence_count],
                    *evidence[-(120 - command_evidence_count) :],
                ]
            # A synchronous read may complete after cancellation or expiry.
            cancelled = bool(cancel_requested())
            expired = float(self.clock()) >= operation.deadline_at
            if cancelled or expired:
                return self._failed_completion(
                    reason_code=("gripper_execution_preempted" if cancelled
                        else "gripper_convergence_timeout"),
                    evidence_refs=tuple(evidence),
                )
            if sample.fault_code is not None:
                return self._failed_completion(
                    reason_code="gripper_fault_during_execution",
                    evidence_refs=tuple(evidence),
                )
            target_reached = sample.moving is False and (
                abs(sample.measured_close - target_close) <= self.convergence.target_tolerance
            )
            contact_settled = bool(
                target_state == "closed"
                and sample.moving is False
                and float(self.clock()) - started_at
                >= self.convergence.minimum_contact_settle_sec
                and (
                    sample.measured_close >= self.convergence.minimum_closed_position
                    or (
                        sample.target_contact_confirmed is True
                        and sample.measured_close > 0.0
                    )
                )
            )
            current_kind = (
                "target_reached"
                if target_reached
                else "settled_before_full_close" if contact_settled else None
            )
            if current_kind is None:
                stable_count = 0
                completion_kind = None
            elif current_kind == completion_kind:
                stable_count += 1
            else:
                completion_kind = current_kind
                stable_count = 1
            if stable_count >= self.convergence.stable_sample_count:
                break
            self.sleep(self.convergence.poll_interval_sec)
        if (
            final_sample is None
            or completion_kind is None
            or stable_count < (self.convergence.stable_sample_count)
        ):
            return self._failed_completion(
                reason_code="gripper_convergence_timeout",
                evidence_refs=tuple(evidence),
            )
        if release_support is not None:
            # The named-support open is complete; the release window needs
            # simulated time to pass with the hand open. Its owner keeps the
            # command and advances its own physics; status is read separately.
            self.port.observe_named_release(cancel_requested=preempted)
        hold = self._ensure_hold("gripper_target_settled")
        if not hold.succeeded:
            return StationLifecycleExecutionCompletion(
                succeeded=False,
                reason_code="gripper_terminal_hold_failed",
                execution_receipt_refs=tuple(
                    dict.fromkeys((*evidence, *hold.evidence_refs))
                ),
                terminal_hold_confirmed=False,
            )
        cancelled = bool(cancel_requested())
        expired = float(self.clock()) >= operation.deadline_at
        if cancelled or expired:
            return self._failed_completion(
                reason_code=("gripper_execution_preempted" if cancelled
                    else "gripper_convergence_timeout"),
                evidence_refs=tuple(evidence),
            )
        result = {
            "schema_version": SET_GRIPPER_STATE_RESULT_SCHEMA_V1
            if self.deployment_binding.skill_version == "1"
            else SET_GRIPPER_STATE_RESULT_SCHEMA_V2,
            "component_id": self.port.component_id,
            "commanded_state": target_state,
            "measured_close": final_sample.measured_close,
            "completion_kind": completion_kind,
            "terminal_hold_confirmed": True,
        }
        return StationLifecycleExecutionCompletion(
            succeeded=True,
            reason_code="gripper_state_command_completed_under_hold",
            execution_receipt_refs=tuple(
                dict.fromkeys((*evidence, *hold.evidence_refs))
            ),
            terminal_hold_confirmed=True,
            execution_result=result,
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("cancel operation/request mismatch.")
        hold = self._ensure_hold(f"cancel:{_required(reason, field_name='reason')}")
        return StationLifecycleControlEvidence(
            succeeded=hold.succeeded,
            status=(
                "gripper_cancel_hold_confirmed"
                if hold.succeeded
                else "gripper_cancel_hold_failed"
            ),
            lower_operation_id=self._lower_operation_id,
            evidence_refs=hold.evidence_refs,
            lower_hold_confirmed=hold.succeeded,
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        if operation is not None and self._activated_operation_id not in {
            None,
            operation.operation_id,
        }:
            raise ValueError("HOLD operation identity mismatch.")
        hold = self._ensure_hold(f"hold:{_required(reason, field_name='reason')}")
        return StationLifecycleControlEvidence(
            succeeded=hold.succeeded,
            status=(
                "gripper_hold_confirmed" if hold.succeeded else "gripper_hold_failed"
            ),
            lower_operation_id=self._lower_operation_id,
            evidence_refs=hold.evidence_refs,
            lower_hold_confirmed=hold.succeeded,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._ensure_hold("lifecycle_close")
            self._closed = True


@dataclass
class GripperAgentGatewaySession:
    contract_bundle: GripperSkillContractBundle
    port: GripperControlPort | None
    executor: ReusableStationAgentSkillOperationExecutor
    service: AgentGatewayService
    current_station_revision: Callable[[], int]
    clock: Callable[[], float]
    clock_domain_id: str
    ports: Mapping[str, GripperControlPort] = field(default_factory=dict)
    _snapshot_sequence: int = field(default=0, init=False, repr=False)
    _snapshot_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _close_outcome: AgentGatewayOperationOutcome | None = field(
        default=None, init=False, repr=False
    )

    def _publish_observation_outcome(self) -> AgentGatewayObservationOutcome:
        binding = self.contract_bundle.deployment_binding
        if binding.skill_version == "1":
            sample = self.port.observe_state()
            summary = {
                "component_id": binding.component_id,
                "measured_close": sample.measured_close,
                "moving": sample.moving,
                "fault_code": sample.fault_code,
                "evidence_ref": sample.evidence_ref,
            }
            captured_at, fault_code = sample.observed_at, sample.fault_code
        else:
            samples = {key: port.observe_state() for key, port in self.ports.items()}
            summary = {
                "gripper_states_by_component": {
                    key: {
                        "measured_close": item.measured_close,
                        "moving": item.moving,
                        "fault_code": item.fault_code,
                        "evidence_ref": item.evidence_ref,
                    }
                    for key, item in samples.items()
                }
            }
            captured_at = min(item.observed_at for item in samples.values())
            fault_code = next(
                (item.fault_code for item in samples.values() if item.fault_code), None
            )
        revision = int(self.current_station_revision())
        with self._snapshot_lock:
            self._snapshot_sequence += 1
            sequence = self._snapshot_sequence
        snapshot = AgentObservationSnapshot(
            snapshot_id=f"gripper:{binding.runtime_session_id}:{revision}:{sequence}",
            runtime_session_id=binding.runtime_session_id,
            skill_route_manifest_hash=self.contract_bundle.route_manifest.semantic_content_hash,
            station_revision=revision,
            captured_at=captured_at,
            available_at=max(captured_at, float(self.clock())),
            clock_domain_id=self.clock_domain_id,
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=0,
            lifecycle_state="active" if fault_code is None else "fault",
            execution_gate="run" if fault_code is None else "hold",
            active_operation_id=self.service.operation_store.active_operation_id,
            source_binding_id=None,
            producer_epoch=None,
            target_binding_id=binding.target_binding_id,
            execution_manifest_hash=binding.execution_manifest_hash,
            frame_graph_hash="not-applicable:gripper-only",
            tool_config_hash=binding.port_binding_hash,
            calibration_hash=binding.port_binding_hash,
            measurement_summary=summary,
            measurement_freshness_ms=0,
            measurement_valid=fault_code is None,
            fault_class=fault_code,
            fault_rearmable=False,
        )
        return self.service.publish_observation(
            snapshot=snapshot,
            now=snapshot.available_at,
        )

    def publish_observation(self) -> AgentObservationSnapshot:
        outcome = self._publish_observation_outcome()
        if not outcome.accepted:
            raise RuntimeError(f"gripper observation rejected: {outcome.reason_code}")
        if outcome.snapshot is None:
            raise RuntimeError("gripper observation accepted without a snapshot")
        return outcome.snapshot

    def wire_adapter(
        self, *, authenticated_client_id: str
    ) -> BoundAgentGatewayWireAdapter:
        return BoundAgentGatewayWireAdapter(
            authenticated_client_id=authenticated_client_id,
            definition_catalog=self.contract_bundle.definition_catalog,
            schema_catalog=self.contract_bundle.schema_catalog,
            service=self.service,
            publish_observation=self._publish_observation_outcome,
            current_station_revision=self.current_station_revision,
            operation_launcher=self.service.launch_operation,
        )

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        return self.executor.wait(timeout_sec=timeout_sec)

    def build_unix_json_server(
        self,
        *,
        socket_path: str,
        client_id_by_uid: Mapping[int, str],
        socket_mode: int = 0o600,
    ) -> AgentGatewayUnixJsonServer:
        peer_map = AgentGatewayUnixPeerClientMap(client_id_by_uid)
        if set(peer_map.client_id_by_uid.values()) != set(
            self.service.access_policy.allowed_client_ids
        ):
            raise ValueError(
                "Unix peer map must exactly cover the session access policy."
            )
        return AgentGatewayUnixJsonServer(
            socket_path=socket_path,
            peer_client_map=peer_map,
            facade_factory=lambda client_id: self.wire_adapter(
                authenticated_client_id=client_id
            ),
            close_gateway=self.close,
            socket_mode=socket_mode,
        )

    def close(self) -> AgentGatewayOperationOutcome:
        with self._close_lock:
            if self._close_outcome is None:
                outcome = self.service.close(
                    current_station_revision=int(self.current_station_revision())
                )
                if not outcome.accepted:
                    return outcome
                self._close_outcome = outcome
            return self._close_outcome


def build_gripper_skill_executor(
    *,
    contract_bundle: GripperSkillContractBundle,
    port: GripperControlPort | None = None,
    ports: Mapping[str, GripperControlPort] | None = None,
    current_station_revision: Callable[[], int],
    convergence: GripperConvergenceProfile | None = None,
    domain_hold: Callable[..., StationLifecycleControlEvidence] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ReusableStationAgentSkillOperationExecutor:
    """Bind the existing lifecycle to a standalone or composed Gateway owner."""
    if not isinstance(contract_bundle, GripperSkillContractBundle):
        raise TypeError("contract_bundle must be typed.")
    convergence = GripperConvergenceProfile() if convergence is None else convergence
    if not isinstance(convergence, GripperConvergenceProfile):
        raise TypeError("convergence must be a GripperConvergenceProfile.")
    binding = contract_bundle.deployment_binding
    route = contract_bundle.route_manifest.skill_routes[0].route

    if binding.skill_version == "1":
        if ports is not None or not isinstance(port, GripperControlPort):
            raise TypeError("v1 requires exactly one gripper port.")
        selected_ports = {port.component_id: port}
    else:
        if port is not None or set(ports or {}) != set(
            binding.port_binding_hash_by_component
        ):
            raise ValueError("v2/v3 require the exact configured left and right ports.")
        selected_ports = dict(ports)
    for component, item in selected_ports.items():
        if (
            not isinstance(item, GripperControlPort)
            or item.component_id != component
            or item.target_kind != binding.target_kind
            or item.port_binding_hash != binding.port_hash_for(component)
        ):
            raise ValueError("port identity does not match deployment binding.")
    if not callable(clock) or not callable(sleep):
        raise TypeError("clock and sleep must be callable.")

    def lifecycle_factory(
        *, request: AgentSkillRequest, **_kwargs: Any
    ) -> GripperSkillLifecycle:
        return GripperSkillLifecycle(
            port=selected_ports[request.parameters["component_id"]]
            if binding.skill_version in {"2", "3"}
            else port,
            contract_bundle=contract_bundle,
            convergence=convergence,
            clock=clock,
            sleep=sleep,
        )

    def gripper_hold(**kwargs: Any) -> StationLifecycleControlEvidence:
        hold = hold_gripper_ports(
            ports=selected_ports,
            reason=_required(kwargs["reason"], field_name="reason"),
        )
        return StationLifecycleControlEvidence(
            succeeded=hold.succeeded,
            status="gripper_domain_hold_confirmed"
            if hold.succeeded
            else "gripper_domain_hold_failed",
            lower_operation_id=None,
            evidence_refs=hold.evidence_refs,
            lower_hold_confirmed=hold.succeeded,
        )

    return ReusableStationAgentSkillOperationExecutor(
        port_id=f"agent-gateway:gripper:{binding.runtime_session_id}",
        binding=StationOperationWorkerBinding(
            worker_id=f"gripper-worker:{binding.runtime_session_id}",
            executor_id=binding.executor_id,
            runtime_session_id=binding.runtime_session_id,
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            route_id=route.route_id,
            route_version=route.route_version,
            deployment_binding_hash=binding.semantic_content_hash,
        ),
        lifecycle_factory=lifecycle_factory,
        domain_hold=gripper_hold if domain_hold is None else domain_hold,
        current_station_revision=current_station_revision,
        clock=clock,
    )


def assemble_gripper_agent_gateway_session(
    *,
    contract_bundle: GripperSkillContractBundle,
    port: GripperControlPort | None = None,
    ports: Mapping[str, GripperControlPort] | None = None,
    allowed_client_ids: tuple[str, ...],
    current_station_revision: Callable[[], int],
    clock_domain_id: str,
    convergence: GripperConvergenceProfile | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> GripperAgentGatewaySession:
    if not isinstance(contract_bundle, GripperSkillContractBundle):
        raise TypeError("contract_bundle must be typed.")
    if convergence is None:
        convergence = GripperConvergenceProfile()
    elif not isinstance(convergence, GripperConvergenceProfile):
        raise TypeError("convergence must be a GripperConvergenceProfile.")
    if not allowed_client_ids or len(set(allowed_client_ids)) != len(
        allowed_client_ids
    ):
        raise ValueError("allowed_client_ids must be non-empty and unique.")
    binding = contract_bundle.deployment_binding
    route = contract_bundle.route_manifest.skill_routes[0].route

    executor = build_gripper_skill_executor(
        contract_bundle=contract_bundle,
        port=port,
        ports=ports,
        convergence=convergence,
        current_station_revision=current_station_revision,
        clock=clock,
        sleep=sleep,
    )
    registry = SkillExecutorRegistry(
        manifest=contract_bundle.route_manifest,
        bindings=(
            SkillExecutorBinding.from_manifest_entry(
                manifest=contract_bundle.route_manifest,
                entry=contract_bundle.route_manifest.skill_routes[0],
                execution_port=executor,
            ),
        ),
    )
    service = AgentGatewayService(
        manifest=contract_bundle.route_manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id=f"agent-access:gripper:{binding.runtime_session_id}",
            allowed_client_ids=allowed_client_ids,
            allowed_modes=(binding.current_mode,),
            max_request_timeout_ms=route.max_operation_duration_ms,
            max_grant_lifetime_sec=30.0,
        ),
        executor_registry=registry,
        domain_control_port=executor,
        payload_validators=contract_bundle.payload_validators,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=16,
        ),
        clock=clock,
        clock_domain_id=_required(clock_domain_id, field_name="clock_domain_id"),
    )
    return GripperAgentGatewaySession(
        ports=dict(ports or {}),
        contract_bundle=contract_bundle,
        port=port,
        executor=executor,
        service=service,
        current_station_revision=current_station_revision,
        clock=clock,
        clock_domain_id=clock_domain_id,
    )


__all__ = [
    "GripperAgentGatewaySession",
    "GripperCommandReceipt",
    "GripperControlPort",
    "GripperConvergenceProfile",
    "GripperHoldReceipt",
    "GripperReadiness",
    "GripperSkillLifecycle",
    "GripperStateSample",
    "assemble_gripper_agent_gateway_session",
    "build_gripper_skill_executor",
    "hold_gripper_ports",
]
