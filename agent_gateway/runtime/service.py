"""Bounded application/CAS facade for the external Agent Gateway contract.

Submission performs access assessment and bounded application admission only;
it never runs an operation body.  An explicit worker invokes the typed
operation port while status and safety-control calls stay on independent short
critical sections.  ``NoIoAgentGatewayService`` retains the stricter B2a
fake/null-only compatibility surface.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..protocol.artifact_contracts import AgentArtifactBinding
from ..protocol.contracts import (
    AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1,
    AgentObservationSnapshot,
    AgentOwnDomainHoldRequest,
    AgentOwnOperationCancelRequest,
    AgentOwnOperationQuery,
    AgentSkillOperation,
    AgentSkillRequest,
)
from ..protocol.skill_catalog import SkillRouteManifest
from .access_policy import (
    AgentSkillAccessPolicy,
    SkillAccessDecision,
    SkillExecutionGrant,
    assess_skill_request_access,
)
from .artifact_validation import validate_skill_artifact_bindings
from .executor_registry import SkillExecutorRegistry
from .operation_port import (
    AgentDomainControlPort,
    AgentGatewayAuthoritativePortReceipt,
    AgentGatewayNoIoPortReceipt,
    AgentSkillExecutionPort,
    AgentSkillOperationPort,
    AgentSkillOperationPortReceipt,
    NoIoAgentSkillOperationPort,
)
from .operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
    OperationStoreOutcome,
)
from .payload_validation import AgentSkillPayloadValidatorRegistry
from .protocol_runtime import AgentGatewayProtocolRuntime


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _finite_time(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("current_station_revision must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class AgentGatewaySubmission:
    """Application admission result; acceptance never means execution."""

    accepted: bool
    reason_code: str
    operation: AgentSkillOperation | None = None
    access_decision: SkillAccessDecision | None = None
    idempotent_replay: bool = False


@dataclass(frozen=True)
class AgentGatewayOperationOutcome:
    """One worker or control call result with optional fake/null receipt."""

    accepted: bool
    reason_code: str
    operation: AgentSkillOperation | None = None
    port_receipt: AgentSkillOperationPortReceipt | None = None


@dataclass(frozen=True)
class AgentGatewayObservationOutcome:
    """Receiver-admission result for one already-produced immutable snapshot."""

    accepted: bool
    reason_code: str
    snapshot: AgentObservationSnapshot | None = None


@dataclass(frozen=True)
class AgentGatewayControlPlaneOutcome:
    """Scoped control result with a feature-gated authoritative projection."""

    accepted: bool
    reason_code: str
    request_hash: str
    operation: AgentSkillOperation | None = None
    idempotent_replay: bool = False
    port_receipt: AgentSkillOperationPortReceipt | None = None
    publish_authoritative_port_receipt: bool = False

    def to_wire(self) -> dict[str, Any]:
        public_receipt = (
            self.port_receipt.to_wire()
            if self.publish_authoritative_port_receipt
            and isinstance(
                self.port_receipt,
                AgentGatewayAuthoritativePortReceipt,
            )
            else None
        )
        return {
            "accepted": self.accepted,
            "reason_code": self.reason_code,
            "request_hash": self.request_hash,
            "idempotent_replay": self.idempotent_replay,
            "operation": (
                None if self.operation is None else self.operation.to_wire()
            ),
            "port_receipt": public_receipt,
        }


class AgentGatewayService:
    """Bounded admission/CAS facade with no transport or operation thread."""

    def __init__(
        self,
        *,
        manifest: SkillRouteManifest,
        access_policy: AgentSkillAccessPolicy,
        operation_port: AgentSkillOperationPort | None = None,
        executor_registry: SkillExecutorRegistry | None = None,
        domain_control_port: AgentDomainControlPort | None = None,
        payload_validators: AgentSkillPayloadValidatorRegistry | None = None,
        operation_store: BoundedInMemoryAgentSkillOperationStore | None = None,
        protocol_runtime: AgentGatewayProtocolRuntime | None = None,
        clock: Callable[[], float] = time.monotonic,
        clock_domain_id: str = "agent-gateway-monotonic",
    ) -> None:
        if not isinstance(manifest, SkillRouteManifest):
            raise TypeError("manifest must be a SkillRouteManifest.")
        if not isinstance(access_policy, AgentSkillAccessPolicy):
            raise TypeError("access_policy must be an AgentSkillAccessPolicy.")
        if operation_store is None:
            operation_store = BoundedInMemoryAgentSkillOperationStore()
        if not isinstance(
            operation_store,
            BoundedInMemoryAgentSkillOperationStore,
        ):
            raise TypeError(
                "operation_store must be a BoundedInMemoryAgentSkillOperationStore."
            )
        if executor_registry is None:
            if operation_port is None:
                raise TypeError(
                    "Either operation_port or executor_registry is required."
                )
            executor_registry = SkillExecutorRegistry.bind_all_routes_to_port(
                manifest=manifest,
                execution_port=operation_port,
            )
        elif not isinstance(executor_registry, SkillExecutorRegistry):
            raise TypeError("executor_registry must be a SkillExecutorRegistry.")
        elif executor_registry.manifest != manifest:
            raise ValueError("executor_registry manifest does not match service.")
        elif operation_port is not None and (
            executor_registry.single_execution_port is not operation_port
        ):
            raise ValueError(
                "Legacy operation_port must be the registry's single executor."
            )

        if domain_control_port is None:
            if operation_port is None:
                raise TypeError(
                    "domain_control_port is required with an explicit executor "
                    "registry."
                )
            domain_control_port = operation_port
        domain_no_io = getattr(domain_control_port, "no_io", None)
        if not isinstance(domain_no_io, bool):
            raise TypeError("domain_control_port must declare boolean no_io.")
        _required_text(
            getattr(domain_control_port, "port_id", None),
            field_name="domain_control_port.port_id",
        )
        if not callable(getattr(domain_control_port, "request_hold", None)):
            raise TypeError(
                "domain_control_port does not implement request_hold()."
            )
        if payload_validators is not None and not isinstance(
            payload_validators,
            AgentSkillPayloadValidatorRegistry,
        ):
            raise TypeError(
                "payload_validators must be an AgentSkillPayloadValidatorRegistry "
                "or None."
            )
        if (
            executor_registry.has_authoritative_execution or not domain_no_io
        ) and payload_validators is None:
            raise TypeError(
                "An authoritative execution or domain-control port requires "
                "server-side payload validators."
            )
        if payload_validators is not None:
            payload_validators.validate_manifest(manifest)
        if not callable(clock):
            raise TypeError("clock must be callable.")
        normalized_clock_domain = _required_text(
            clock_domain_id,
            field_name="clock_domain_id",
        )
        if protocol_runtime is None:
            protocol_runtime = AgentGatewayProtocolRuntime.control_plane_only(
                manifest=manifest,
                clock=clock,
            )
        elif not isinstance(protocol_runtime, AgentGatewayProtocolRuntime):
            raise TypeError(
                "protocol_runtime must be AgentGatewayProtocolRuntime or None."
            )
        elif protocol_runtime.manifest != manifest:
            raise ValueError("protocol_runtime manifest does not match service.")

        self.manifest = manifest
        self.access_policy = access_policy
        self.executor_registry = executor_registry
        self.domain_control_port = domain_control_port
        # Compatibility projection for existing one-port callers.  New
        # multi-executor code must use executor_registry/domain_control_port.
        self.operation_port = (
            operation_port or executor_registry.single_execution_port
        )
        self._no_io_port = (
            executor_registry.all_no_io and domain_no_io
        )
        self._domain_no_io = domain_no_io
        self.payload_validators = payload_validators
        self.operation_store = operation_store
        self.protocol_runtime = protocol_runtime
        self._clock = clock
        self.clock_domain_id = normalized_clock_domain
        self._admission_lock = threading.RLock()
        self._observation_lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_outcome: AgentGatewayOperationOutcome | None = None
        self._grant_invocations_by_hash: dict[str, int] = {}
        self._latest_observation: AgentObservationSnapshot | None = None
        self._control_request_lock = threading.Lock()
        self._control_request_outcomes: OrderedDict[
            tuple[str, str, str],
            tuple[str, AgentGatewayControlPlaneOutcome | None],
        ] = OrderedDict()

    @property
    def is_closed(self) -> bool:
        with self._admission_lock:
            return self._closed

    def get_routes(self) -> SkillRouteManifest:
        return self.manifest

    def execution_port_for_request(
        self,
        request: AgentSkillRequest,
    ) -> AgentSkillExecutionPort:
        """Resolve the immutable request to its exact Route executor."""

        return self.executor_registry.execution_port_for_request(request)

    def execution_port_for_operation(
        self,
        operation_id: str,
    ) -> AgentSkillExecutionPort:
        """Resolve an admitted operation without trusting caller Route fields."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        request = self.operation_store.request_for_operation(normalized_id)
        if request is None:
            raise KeyError("Operation has no immutable request.")
        return self.execution_port_for_request(request)

    def launch_operation(self, operation_id: str) -> Any:
        """Launch through the executor selected by the immutable request."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        execution_port = self.execution_port_for_operation(normalized_id)
        launcher = getattr(execution_port, "launch", None)
        if not callable(launcher):
            raise TypeError(
                "The exact Route executor does not implement launch()."
            )
        outcome = launcher(service=self, operation_id=normalized_id)
        if not isinstance(getattr(outcome, "accepted", None), bool):
            raise TypeError("Executor launch outcome must declare accepted.")
        _required_text(
            getattr(outcome, "status", None),
            field_name="executor launch outcome status",
        )
        return outcome

    def get_observation(self) -> AgentObservationSnapshot | None:
        with self._observation_lock:
            return self._latest_observation

    def publish_observation(
        self,
        *,
        snapshot: AgentObservationSnapshot,
        now: float | None = None,
    ) -> AgentGatewayObservationOutcome:
        """Receiver-admit a snapshot without reading Station or any device."""

        if not isinstance(snapshot, AgentObservationSnapshot):
            raise TypeError("snapshot must be an AgentObservationSnapshot.")
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        with self._admission_lock:
            if self._closed:
                return AgentGatewayObservationOutcome(
                    False,
                    "agent_gateway_service_closed",
                    self.get_observation(),
                )
            with self._observation_lock:
                current = self._latest_observation
                checks = (
                    (
                        snapshot.runtime_session_id == self.manifest.runtime_session_id,
                        "observation_runtime_session_mismatch",
                    ),
                    (
                        snapshot.skill_route_manifest_hash
                        == self.manifest.semantic_content_hash,
                        "observation_skill_route_manifest_mismatch",
                    ),
                    (
                        snapshot.target_binding_id == self.manifest.target_binding_id,
                        "observation_target_binding_mismatch",
                    ),
                    (
                        snapshot.execution_manifest_hash
                        == self.manifest.execution_manifest_hash,
                        "observation_execution_manifest_mismatch",
                    ),
                    (
                        snapshot.clock_domain_id == self.clock_domain_id,
                        "observation_clock_domain_mismatch",
                    ),
                    (
                        snapshot.available_at <= timestamp,
                        "observation_available_in_future",
                    ),
                    (
                        snapshot.active_operation_id
                        == self.operation_store.active_operation_id,
                        "observation_active_operation_mismatch",
                    ),
                )
                for matches, reason_code in checks:
                    if not matches:
                        return AgentGatewayObservationOutcome(
                            False,
                            reason_code,
                            current,
                        )
                store_revision = self.operation_store.latest_station_revision
                if (
                    store_revision is not None
                    and snapshot.station_revision < store_revision
                ):
                    return AgentGatewayObservationOutcome(
                        False,
                        "observation_station_revision_regressed",
                        current,
                    )
                if current is not None:
                    if snapshot.snapshot_id == current.snapshot_id:
                        if (
                            snapshot.semantic_content_hash
                            == current.semantic_content_hash
                        ):
                            return AgentGatewayObservationOutcome(
                                True,
                                "observation_idempotent_lookup",
                                current,
                            )
                        return AgentGatewayObservationOutcome(
                            False,
                            "observation_snapshot_id_conflict",
                            current,
                        )
                    if snapshot.station_revision < current.station_revision:
                        return AgentGatewayObservationOutcome(
                            False,
                            "observation_station_revision_regressed",
                            current,
                        )
                    if snapshot.available_at < current.available_at:
                        return AgentGatewayObservationOutcome(
                            False,
                            "observation_available_at_regressed",
                            current,
                        )
                self._latest_observation = snapshot
                return AgentGatewayObservationOutcome(
                    True,
                    "observation_published",
                    snapshot,
                )

    def get_operation(self, operation_id: str) -> AgentSkillOperation | None:
        return self.operation_store.get(operation_id)

    def get_active_operation(self) -> AgentSkillOperation | None:
        return self.operation_store.active_operation()

    def get_own_operation(
        self,
        *,
        query: AgentOwnOperationQuery,
    ) -> AgentGatewayControlPlaneOutcome:
        """Return only an operation created by this exact client/session."""

        if not isinstance(query, AgentOwnOperationQuery):
            raise TypeError("query must be an AgentOwnOperationQuery.")
        rejection = self._control_scope_rejection(
            client_id=query.client_id,
            runtime_session_id=query.runtime_session_id,
            skill_route_manifest_hash=query.skill_route_manifest_hash,
        )
        if rejection is not None:
            return AgentGatewayControlPlaneOutcome(
                False,
                rejection,
                query.semantic_content_hash,
            )
        operation = self._owned_operation(
            client_id=query.client_id,
            runtime_session_id=query.runtime_session_id,
            skill_route_manifest_hash=query.skill_route_manifest_hash,
            operation_id=query.operation_id,
        )
        if operation is None:
            return AgentGatewayControlPlaneOutcome(
                False,
                "own_operation_not_found",
                query.semantic_content_hash,
            )
        return AgentGatewayControlPlaneOutcome(
            True,
            "own_operation_found",
            query.semantic_content_hash,
            operation,
        )

    def cancel_own_operation(
        self,
        *,
        request: AgentOwnOperationCancelRequest,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayControlPlaneOutcome:
        """Cancel only the caller's operation; no arbitrary client/domain input."""

        if not isinstance(request, AgentOwnOperationCancelRequest):
            raise TypeError(
                "request must be an AgentOwnOperationCancelRequest."
            )
        revision = _revision(current_station_revision)
        rejection = self._control_scope_rejection(
            client_id=request.client_id,
            runtime_session_id=request.runtime_session_id,
            skill_route_manifest_hash=request.skill_route_manifest_hash,
        )
        if rejection is None and revision < request.expected_station_revision:
            rejection = "station_revision_precedes_observed_revision"
        if rejection is not None:
            return AgentGatewayControlPlaneOutcome(
                False,
                rejection,
                request.semantic_content_hash,
            )
        operation = self._owned_operation(
            client_id=request.client_id,
            runtime_session_id=request.runtime_session_id,
            skill_route_manifest_hash=request.skill_route_manifest_hash,
            operation_id=request.operation_id,
        )
        if operation is None:
            return AgentGatewayControlPlaneOutcome(
                False,
                "own_operation_not_found",
                request.semantic_content_hash,
            )
        replay = self._begin_control_request(
            action="cancel_own_operation",
            client_id=request.client_id,
            idempotency_key=request.idempotency_key,
            request_hash=request.semantic_content_hash,
        )
        if replay is not None:
            return replay
        try:
            outcome = self.cancel_operation(
                operation_id=request.operation_id,
                reason=request.reason_code,
                current_station_revision=revision,
                now=now,
            )
        except Exception:
            self._abandon_control_request(
                action="cancel_own_operation",
                client_id=request.client_id,
                idempotency_key=request.idempotency_key,
                request_hash=request.semantic_content_hash,
            )
            raise
        public_operation = outcome.operation
        if public_operation is not None and self._owned_operation(
            client_id=request.client_id,
            runtime_session_id=request.runtime_session_id,
            skill_route_manifest_hash=request.skill_route_manifest_hash,
            operation_id=public_operation.operation_id,
        ) is None:
            public_operation = None
        result = AgentGatewayControlPlaneOutcome(
            outcome.accepted,
            outcome.reason_code,
            request.semantic_content_hash,
            public_operation,
            port_receipt=outcome.port_receipt,
            publish_authoritative_port_receipt=(
                AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1
                in self.manifest.supported_transport_features
            ),
        )
        self._complete_control_request(
            action="cancel_own_operation",
            client_id=request.client_id,
            idempotency_key=request.idempotency_key,
            request_hash=request.semantic_content_hash,
            outcome=result,
        )
        return result

    def request_hold_own_domain(
        self,
        *,
        request: AgentOwnDomainHoldRequest,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayControlPlaneOutcome:
        """Request one-way HOLD for this service's exact domain and target."""

        if not isinstance(request, AgentOwnDomainHoldRequest):
            raise TypeError("request must be an AgentOwnDomainHoldRequest.")
        revision = _revision(current_station_revision)
        rejection = self._control_scope_rejection(
            client_id=request.client_id,
            runtime_session_id=request.runtime_session_id,
            skill_route_manifest_hash=request.skill_route_manifest_hash,
            control_domain_id=request.control_domain_id,
            target_binding_id=request.target_binding_id,
        )
        if rejection is None and revision < request.expected_station_revision:
            rejection = "station_revision_precedes_observed_revision"
        if rejection is not None:
            return AgentGatewayControlPlaneOutcome(
                False,
                rejection,
                request.semantic_content_hash,
            )
        replay = self._begin_control_request(
            action="request_hold_own_domain",
            client_id=request.client_id,
            idempotency_key=request.idempotency_key,
            request_hash=request.semantic_content_hash,
        )
        if replay is not None:
            return replay
        try:
            outcome = self.request_hold(
                reason=request.reason_code,
                current_station_revision=revision,
                now=now,
            )
        except Exception:
            self._abandon_control_request(
                action="request_hold_own_domain",
                client_id=request.client_id,
                idempotency_key=request.idempotency_key,
                request_hash=request.semantic_content_hash,
            )
            raise
        public_operation = outcome.operation
        if public_operation is not None and self._owned_operation(
            client_id=request.client_id,
            runtime_session_id=request.runtime_session_id,
            skill_route_manifest_hash=request.skill_route_manifest_hash,
            operation_id=public_operation.operation_id,
        ) is None:
            public_operation = None
        result = AgentGatewayControlPlaneOutcome(
            outcome.accepted,
            outcome.reason_code,
            request.semantic_content_hash,
            public_operation,
            port_receipt=outcome.port_receipt,
            publish_authoritative_port_receipt=(
                AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1
                in self.manifest.supported_transport_features
            ),
        )
        self._complete_control_request(
            action="request_hold_own_domain",
            client_id=request.client_id,
            idempotency_key=request.idempotency_key,
            request_hash=request.semantic_content_hash,
            outcome=result,
        )
        return result

    def grant_invocations_used(self, grant: SkillExecutionGrant) -> int:
        if not isinstance(grant, SkillExecutionGrant):
            raise TypeError("grant must be a SkillExecutionGrant.")
        with self._admission_lock:
            return self._grant_invocations_by_hash.get(
                grant.semantic_content_hash,
                0,
            )

    def submit_request(
        self,
        *,
        request: AgentSkillRequest,
        current_station_revision: int,
        grant: SkillExecutionGrant | None = None,
        now: float | None = None,
    ) -> AgentGatewaySubmission:
        """Assess and admit only; never call the operation port inline."""

        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if grant is not None and not isinstance(grant, SkillExecutionGrant):
            raise TypeError("grant must be a SkillExecutionGrant or None.")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")

        with self._admission_lock:
            if self._closed:
                return AgentGatewaySubmission(False, "agent_gateway_service_closed")
            idempotency = self.operation_store.lookup_idempotency(request)
            if idempotency is not None:
                return self._submission_from_store(idempotency)

            observation_rejection = self._validate_request_observation(
                request=request,
                current_station_revision=revision,
                now=timestamp,
            )
            if observation_rejection is not None:
                return AgentGatewaySubmission(False, observation_rejection)

            grant_invocations_used = 0
            if grant is not None:
                grant_invocations_used = self._grant_invocations_by_hash.get(
                    grant.semantic_content_hash,
                    0,
                )
            decision = assess_skill_request_access(
                policy=self.access_policy,
                manifest=self.manifest,
                request=request,
                current_station_revision=revision,
                now=timestamp,
                grant=grant,
                grant_invocations_used=grant_invocations_used,
            )
            if not decision.accepted:
                return AgentGatewaySubmission(
                    accepted=False,
                    reason_code=decision.reason_code,
                    access_decision=decision,
                )
            route_entry = self.manifest.route(
                request.skill_id,
                request.skill_version,
                request.route_id,
                request.route_version,
            )
            route = route_entry.route
            if self.payload_validators is not None:
                try:
                    self.payload_validators.validate_parameters(
                        route=route,
                        payload=request.parameters,
                    )
                except (TypeError, ValueError, KeyError):
                    return AgentGatewaySubmission(
                        accepted=False,
                        reason_code="parameter_payload_rejected",
                        access_decision=decision,
                    )
            observation = self.get_observation()
            if observation is None:
                raise RuntimeError("Validated request lost its observation snapshot.")
            if (
                route.privilege_tier != "t0_read_only"
                and not observation.measurement_valid
            ):
                return AgentGatewaySubmission(
                    accepted=False,
                    reason_code="observation_measurement_invalid",
                    access_decision=decision,
                )
            try:
                self.protocol_runtime.validate_and_pin_request_inputs(
                    request=request,
                    now=timestamp,
                )
            except (KeyError, PermissionError, RuntimeError, TypeError, ValueError):
                return AgentGatewaySubmission(
                    accepted=False,
                    reason_code="input_artifact_bindings_rejected",
                    access_decision=decision,
                )
            store_outcome = self.operation_store.admit(
                request=request,
                access_decision=decision,
                current_station_revision=revision,
                admitted_at=timestamp,
            )
            if not store_outcome.accepted:
                self.protocol_runtime.release_request_input_pins(request)
            if (
                store_outcome.accepted
                and not store_outcome.idempotent_replay
                and decision.grant_hash is not None
            ):
                self._grant_invocations_by_hash[decision.grant_hash] = (
                    grant_invocations_used + 1
                )
            return self._submission_from_store(
                store_outcome,
                access_decision=decision,
            )

    def activate_operation(
        self,
        *,
        operation_id: str,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        """Explicit worker seam; never called by ``submit_request``."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        with self._admission_lock:
            if self._closed:
                return AgentGatewayOperationOutcome(
                    False,
                    "agent_gateway_service_closed",
                    self.operation_store.get(normalized_id),
                )
        current = self.operation_store.get(normalized_id)
        request = self.operation_store.request_for_operation(normalized_id)
        if current is None or request is None:
            return AgentGatewayOperationOutcome(False, "operation_not_found")
        try:
            execution_port = self.execution_port_for_request(request)
        except (KeyError, TypeError, ValueError):
            return AgentGatewayOperationOutcome(
                False,
                "exact_skill_executor_not_found",
                current,
            )
        executor_no_io = execution_port.no_io
        activation = self.operation_store.transition_worker_projection(
            operation_id=normalized_id,
            expected_operation_hash=current.semantic_content_hash,
            next_status="activated",
            reason_code=(
                "fake_worker_activation_started"
                if executor_no_io
                else "station_worker_activation_started"
            ),
            current_station_revision=revision,
            updated_at=timestamp,
            progress_summary={"phase": "activated"},
        )
        if not activation.accepted or activation.operation is None:
            return self._operation_outcome_from_store(activation)
        activated = activation.operation

        try:
            receipt = execution_port.activate(
                request=request,
                operation=activated,
            )
            self._require_port_receipt(
                receipt,
                expected_port=execution_port,
                operation_kind="activate",
                operation_id=normalized_id,
            )
        except Exception as error:
            failure = self.operation_store.transition_worker_projection(
                operation_id=normalized_id,
                expected_operation_hash=activated.semantic_content_hash,
                next_status="failed",
                reason_code=f"activation_port_exception:{type(error).__name__}",
                current_station_revision=revision,
                updated_at=self._time_at_least(activated.updated_at),
                progress_summary={"phase": "failed"},
            )
            if not failure.accepted:
                return AgentGatewayOperationOutcome(
                    False,
                    "operation_superseded_after_activation_exception",
                    failure.operation,
                )
            return AgentGatewayOperationOutcome(
                False,
                "activation_port_exception",
                failure.operation,
            )

        finished_at = self._time_at_least(activated.updated_at)
        execution_receipt_refs = (receipt.receipt_id,)
        if receipt.succeeded:
            projection = self.operation_store.transition_worker_projection(
                operation_id=normalized_id,
                expected_operation_hash=activated.semantic_content_hash,
                next_status="running",
                reason_code=(
                    "fake_activation_accepted_not_completed"
                    if executor_no_io
                    else "station_activation_accepted_not_completed"
                ),
                current_station_revision=revision,
                updated_at=finished_at,
                lower_operation_id=receipt.lower_operation_id,
                execution_receipt_refs=execution_receipt_refs,
                progress_summary={"phase": "running"},
            )
        else:
            projection = self.operation_store.transition_worker_projection(
                operation_id=normalized_id,
                expected_operation_hash=activated.semantic_content_hash,
                next_status="failed",
                reason_code=(
                    "fake_or_null_activation_rejected"
                    if executor_no_io
                    else "station_activation_rejected"
                ),
                current_station_revision=revision,
                updated_at=finished_at,
                execution_receipt_refs=execution_receipt_refs,
                progress_summary={"phase": "failed"},
            )
        if not projection.accepted:
            return AgentGatewayOperationOutcome(
                False,
                "operation_superseded_after_port_call",
                projection.operation,
                receipt,
            )
        return AgentGatewayOperationOutcome(
            receipt.succeeded,
            (
                (
                    "operation_running_fake_only"
                    if executor_no_io
                    else "operation_running_station_owned"
                )
                if receipt.succeeded
                else "operation_activation_failed_closed"
            ),
            projection.operation,
            receipt,
        )

    def complete_operation(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        succeeded: bool,
        reason_code: str,
        current_station_revision: int,
        execution_result: Mapping[str, Any] | None = None,
        execution_receipt_refs: tuple[str, ...] = (),
        output_artifact_bindings: tuple[AgentArtifactBinding, ...] = (),
        progress_summary: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        """CAS an already-produced worker result onto the live projection.

        ``progress_summary`` is an additive, bounded diagnostic seam for a
        worker that has already failed closed.  It does not authorize motion
        or alter the terminal status selected by ``succeeded``.
        """

        normalized_id = _required_text(operation_id, field_name="operation_id")
        expected_hash = _required_text(
            expected_operation_hash,
            field_name="expected_operation_hash",
        )
        if not isinstance(succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        normalized_reason = _required_text(reason_code, field_name="reason_code")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        with self._admission_lock:
            if self._closed:
                return AgentGatewayOperationOutcome(
                    False,
                    "agent_gateway_service_closed",
                    self.operation_store.get(normalized_id),
                )
        request = self.operation_store.request_for_operation(normalized_id)
        if request is None:
            return AgentGatewayOperationOutcome(False, "operation_not_found")
        route = self.manifest.route(
            request.skill_id,
            request.skill_version,
            request.route_id,
            request.route_version,
        ).route
        io_contract = self.protocol_runtime.io_catalog.contract(
            request.skill_id,
            request.skill_version,
        )
        try:
            # Output-slot minimum cardinality is a success postcondition.  A
            # failed worker must still be able to commit a terminal failure
            # without fabricating the Route's required success Artifacts.
            if succeeded or output_artifact_bindings:
                validate_skill_artifact_bindings(
                    io_contract=io_contract,
                    direction="output",
                    bindings=output_artifact_bindings,
                    session_capabilities=self.protocol_runtime.session_capabilities,
                )
            if output_artifact_bindings:
                store, _broker = self.protocol_runtime.require_artifact_plane()
                for binding in output_artifact_bindings:
                    record = store.artifact(binding.artifact.artifact_id)
                    if not record.available or record.artifact != binding.artifact:
                        raise ValueError(
                            "Output Artifact is unavailable or does not match."
                        )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return AgentGatewayOperationOutcome(
                False,
                "output_artifact_bindings_rejected",
                self.operation_store.get(normalized_id),
            )
        if execution_result is not None and self.payload_validators is not None:
            try:
                self.payload_validators.validate_execution_result(
                    route=route,
                    payload=execution_result,
                    request_parameters=request.parameters,
                )
            except (TypeError, ValueError, KeyError):
                return AgentGatewayOperationOutcome(
                    False,
                    "execution_result_payload_rejected",
                    self.operation_store.get(normalized_id),
                )
        next_status = (
            "awaiting_evaluation"
            if succeeded and route.requires_task_evaluation
            else ("completed" if succeeded else "failed")
        )
        projection = self.operation_store.transition_worker_projection(
            operation_id=normalized_id,
            expected_operation_hash=expected_hash,
            next_status=next_status,
            reason_code=normalized_reason,
            current_station_revision=revision,
            updated_at=timestamp,
            execution_receipt_refs=execution_receipt_refs,
            progress_summary=(
                {"phase": next_status}
                if progress_summary is None
                else progress_summary
            ),
            execution_result=execution_result,
            output_artifact_bindings=output_artifact_bindings,
        )
        return AgentGatewayOperationOutcome(
            projection.accepted,
            (
                (
                    "operation_execution_committed_awaiting_evaluation"
                    if next_status == "awaiting_evaluation"
                    else "operation_result_committed"
                )
                if projection.accepted
                else projection.reason_code
            ),
            projection.operation,
        )

    def complete_evaluation(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        evaluation_status: str,
        task_outcome: str,
        reason_code: str,
        current_station_revision: int,
        evaluator_receipt_refs: tuple[str, ...],
        aggregate_evaluation_receipt_ref: str,
        task_evaluation: Mapping[str, Any],
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        """CAS already-produced evaluator evidence; run no evaluator here."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        expected_hash = _required_text(
            expected_operation_hash,
            field_name="expected_operation_hash",
        )
        normalized_reason = _required_text(reason_code, field_name="reason_code")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        with self._admission_lock:
            if self._closed:
                return AgentGatewayOperationOutcome(
                    False,
                    "agent_gateway_service_closed",
                    self.operation_store.get(normalized_id),
                )
        request = self.operation_store.request_for_operation(normalized_id)
        if request is None:
            return AgentGatewayOperationOutcome(False, "operation_not_found")
        route = self.manifest.route(
            request.skill_id,
            request.skill_version,
            request.route_id,
            request.route_version,
        ).route
        if self.payload_validators is not None:
            try:
                self.payload_validators.validate_task_evaluation(
                    route=route,
                    payload=task_evaluation,
                )
            except (TypeError, ValueError, KeyError):
                return AgentGatewayOperationOutcome(
                    False,
                    "task_evaluation_payload_rejected",
                    self.operation_store.get(normalized_id),
                )
        projection = self.operation_store.transition_evaluation_projection(
            operation_id=normalized_id,
            expected_operation_hash=expected_hash,
            evaluation_status=evaluation_status,
            task_outcome=task_outcome,
            reason_code=normalized_reason,
            current_station_revision=revision,
            updated_at=timestamp,
            evaluator_receipt_refs=evaluator_receipt_refs,
            aggregate_evaluation_receipt_ref=(
                aggregate_evaluation_receipt_ref
            ),
            task_evaluation=task_evaluation,
        )
        return AgentGatewayOperationOutcome(
            projection.accepted,
            (
                "operation_evaluation_committed"
                if projection.accepted
                else projection.reason_code
            ),
            projection.operation,
        )

    def cancel_operation(
        self,
        *,
        operation_id: str,
        reason: str,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        normalized_id = _required_text(operation_id, field_name="operation_id")
        normalized_reason = _required_text(reason, field_name="reason")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        operation = self.operation_store.get(normalized_id)
        request = self.operation_store.request_for_operation(normalized_id)
        if operation is None or request is None:
            return AgentGatewayOperationOutcome(False, "operation_not_found")
        if operation.terminal:
            return AgentGatewayOperationOutcome(
                operation.status == "cancelled",
                (
                    "operation_already_cancelled"
                    if operation.status == "cancelled"
                    else "operation_terminal"
                ),
                operation,
            )
        route = self.manifest.route(
            request.skill_id,
            request.skill_version,
            request.route_id,
            request.route_version,
        ).route
        use_hold = route.cancellation_policy in {
            "hold_on_cancel",
            "hold_on_cancel_disconnect_or_timeout",
        }
        control_no_io = self._control_port_no_io(
            request=request,
            use_hold=use_hold,
        )
        return self._control_operation(
            operation=operation,
            request=request,
            control_kind="hold" if use_hold else "cancel",
            terminal_status="cancelled",
            reason=normalized_reason,
            reservation_reason=(
                "hold_on_cancel_reserved" if use_hold else "cancel_reserved"
            ),
            success_reason=(
                "operation_cancelled_after_fake_no_io_hold"
                if control_no_io and use_hold
                else "operation_cancelled_by_fake_no_io_port"
                if control_no_io
                else "operation_cancelled_after_station_hold"
                if use_hold
                else "operation_cancelled_by_station_port"
            ),
            failure_reason=(
                "fake_no_io_hold_not_confirmed"
                if control_no_io and use_hold
                else "fake_no_io_cancel_not_confirmed"
                if control_no_io
                else "station_hold_not_confirmed"
                if use_hold
                else "station_cancel_not_confirmed"
            ),
            current_station_revision=revision,
            requested_at=timestamp,
            use_hold=use_hold,
        )

    def request_hold(
        self,
        *,
        reason: str,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        normalized_reason = _required_text(reason, field_name="reason")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        return self._request_hold(
            reason=normalized_reason,
            control_kind="hold",
            current_station_revision=revision,
            requested_at=timestamp,
        )

    def expire_due_operation(
        self,
        *,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        operation = self.operation_store.due_active_operation(now=timestamp)
        if operation is None:
            return AgentGatewayOperationOutcome(False, "no_due_active_operation")
        request = self.operation_store.request_for_operation(operation.operation_id)
        if request is None:
            raise RuntimeError("Active operation lost its immutable request.")
        route = self.manifest.route(
            request.skill_id,
            request.skill_version,
            request.route_id,
            request.route_version,
        ).route
        use_hold = route.cancellation_policy in {
            "hold_on_cancel_disconnect_or_timeout",
            "hold_on_disconnect_or_timeout",
        }
        control_no_io = self._control_port_no_io(
            request=request,
            use_hold=use_hold,
        )
        return self._control_operation(
            operation=operation,
            request=request,
            control_kind="timeout",
            terminal_status="timed_out",
            reason="operation_deadline_exceeded",
            reservation_reason="timeout_control_reserved",
            success_reason=(
                "operation_timed_out_after_fake_no_io_control"
                if control_no_io
                else "operation_timed_out_after_station_control"
            ),
            failure_reason="timeout_control_not_confirmed",
            current_station_revision=revision,
            requested_at=timestamp,
            use_hold=use_hold,
        )

    def handle_client_disconnect(
        self,
        *,
        client_id: str,
        reason: str,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        normalized_client = _required_text(client_id, field_name="client_id")
        normalized_reason = _required_text(reason, field_name="reason")
        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        operation = self.operation_store.active_operation()
        if operation is None:
            return AgentGatewayOperationOutcome(
                True,
                "no_active_operation_for_disconnect",
            )
        request = self.operation_store.request_for_operation(operation.operation_id)
        if request is None:
            raise RuntimeError("Active operation lost its immutable request.")
        if request.client_id != normalized_client:
            return AgentGatewayOperationOutcome(
                False,
                "disconnect_client_does_not_own_active_operation",
                operation,
            )
        route = self.manifest.route(
            request.skill_id,
            request.skill_version,
            request.route_id,
            request.route_version,
        ).route
        use_hold = route.cancellation_policy in {
            "hold_on_cancel_disconnect_or_timeout",
            "hold_on_disconnect_or_timeout",
        }
        control_no_io = self._control_port_no_io(
            request=request,
            use_hold=use_hold,
        )
        return self._control_operation(
            operation=operation,
            request=request,
            control_kind="disconnect",
            terminal_status="cancelled",
            reason=normalized_reason,
            reservation_reason="disconnect_control_reserved",
            success_reason=(
                "operation_cancelled_after_fake_no_io_disconnect_control"
                if control_no_io
                else "operation_cancelled_after_station_disconnect_control"
            ),
            failure_reason="disconnect_control_not_confirmed",
            current_station_revision=revision,
            requested_at=timestamp,
            use_hold=use_hold,
        )

    def close(
        self,
        *,
        current_station_revision: int,
        now: float | None = None,
    ) -> AgentGatewayOperationOutcome:
        """Fence admission once and retry HOLD until it is confirmed."""

        revision = _revision(current_station_revision)
        timestamp = self._time() if now is None else _finite_time(now, field_name="now")
        with self._close_lock:
            if self._close_outcome is not None:
                return self._close_outcome
            with self._admission_lock:
                self._closed = True
            outcome = self._request_hold(
                reason="agent_gateway_service_close",
                control_kind="service_close",
                current_station_revision=revision,
                requested_at=timestamp,
            )
            if outcome.accepted:
                self._close_outcome = outcome
            return outcome

    def _request_hold(
        self,
        *,
        reason: str,
        control_kind: str,
        current_station_revision: int,
        requested_at: float,
    ) -> AgentGatewayOperationOutcome:
        operation = self.operation_store.active_operation()
        if operation is None:
            try:
                receipt = self.domain_control_port.request_hold(
                    operation=None,
                    reason=reason,
                )
                self._require_port_receipt(
                    receipt,
                    expected_port=self.domain_control_port,
                    operation_kind="hold",
                    operation_id=None,
                )
            except Exception:
                return AgentGatewayOperationOutcome(
                    False,
                    "hold_port_exception_no_active_operation",
                )
            return AgentGatewayOperationOutcome(
                receipt.succeeded,
                (
                    "fake_no_io_hold_confirmed_no_active_operation"
                    if self._domain_no_io and receipt.succeeded
                    else "fake_or_null_hold_not_confirmed_no_active_operation"
                    if self._domain_no_io
                    else "station_hold_confirmed_no_active_operation"
                    if receipt.succeeded
                    else "station_hold_not_confirmed_no_active_operation"
                ),
                port_receipt=receipt,
            )
        request = self.operation_store.request_for_operation(operation.operation_id)
        if request is None:
            raise RuntimeError("Active operation lost its immutable request.")
        return self._control_operation(
            operation=operation,
            request=request,
            control_kind=control_kind,
            terminal_status="cancelled",
            reason=reason,
            reservation_reason=f"{control_kind}_hold_reserved",
            success_reason=(
                "operation_cancelled_after_fake_no_io_hold"
                if self._domain_no_io
                else "operation_cancelled_after_station_hold"
            ),
            failure_reason=(
                "fake_or_null_hold_not_confirmed"
                if self._domain_no_io
                else "station_hold_not_confirmed"
            ),
            current_station_revision=current_station_revision,
            requested_at=requested_at,
            use_hold=True,
            allow_existing_control=True,
        )

    def _control_port_no_io(
        self,
        *,
        request: AgentSkillRequest,
        use_hold: bool,
    ) -> bool:
        if use_hold:
            return self._domain_no_io
        return self.execution_port_for_request(request).no_io

    def _control_operation(
        self,
        *,
        operation: AgentSkillOperation,
        request: AgentSkillRequest,
        control_kind: str,
        terminal_status: str,
        reason: str,
        reservation_reason: str,
        success_reason: str,
        failure_reason: str,
        current_station_revision: int,
        requested_at: float,
        use_hold: bool,
        allow_existing_control: bool = False,
    ) -> AgentGatewayOperationOutcome:
        reservation = self.operation_store.reserve_control(
            operation_id=operation.operation_id,
            control_kind=control_kind,
            terminal_status=terminal_status,
            reason_code=reservation_reason,
            current_station_revision=current_station_revision,
            updated_at=requested_at,
        )
        if not reservation.accepted or reservation.operation is None:
            if not (
                allow_existing_control
                and reservation.reason_code
                == "operation_control_reservation_conflict"
                and reservation.operation is not None
                and not reservation.operation.terminal
            ):
                return self._operation_outcome_from_store(reservation)
        reserved = reservation.operation
        if reserved is None:
            return AgentGatewayOperationOutcome(False, "operation_not_found")
        if reserved.terminal:
            return AgentGatewayOperationOutcome(
                True,
                "operation_already_terminal",
                reserved,
            )

        try:
            if use_hold:
                control_port: AgentDomainControlPort | AgentSkillExecutionPort = (
                    self.domain_control_port
                )
                selected_executor = self.execution_port_for_request(request)
                hold_operation = (
                    reserved
                    if selected_executor is self.domain_control_port
                    else None
                )
                receipt = self.domain_control_port.request_hold(
                    operation=hold_operation,
                    reason=reason,
                )
                receipt_kind = "hold"
                receipt_operation_id = (
                    None
                    if hold_operation is None
                    else hold_operation.operation_id
                )
            else:
                control_port = self.execution_port_for_request(request)
                receipt = control_port.cancel(
                    request=request,
                    operation=reserved,
                    reason=reason,
                )
                receipt_kind = "cancel"
                receipt_operation_id = reserved.operation_id
            self._require_port_receipt(
                receipt,
                expected_port=control_port,
                operation_kind=receipt_kind,
                operation_id=receipt_operation_id,
            )
        except Exception as error:
            failed = self.operation_store.record_control_failure(
                operation_id=reserved.operation_id,
                expected_operation_hash=reserved.semantic_content_hash,
                reason_code=f"control_port_exception:{type(error).__name__}",
                current_station_revision=current_station_revision,
                updated_at=self._time_at_least(reserved.updated_at),
            )
            return AgentGatewayOperationOutcome(
                False,
                (
                    "control_port_exception"
                    if failed.accepted
                    else "operation_superseded_after_control_exception"
                ),
                failed.operation,
            )

        finished_at = self._time_at_least(reserved.updated_at)
        if receipt.succeeded:
            finalized = self.operation_store.finalize_control(
                operation_id=reserved.operation_id,
                expected_operation_hash=reserved.semantic_content_hash,
                reason_code=success_reason,
                current_station_revision=current_station_revision,
                updated_at=finished_at,
                execution_receipt_refs=(receipt.receipt_id,),
            )
            return AgentGatewayOperationOutcome(
                finalized.accepted,
                success_reason if finalized.accepted else finalized.reason_code,
                finalized.operation,
                receipt,
            )
        failed = self.operation_store.record_control_failure(
            operation_id=reserved.operation_id,
            expected_operation_hash=reserved.semantic_content_hash,
            reason_code=failure_reason,
            current_station_revision=current_station_revision,
            updated_at=finished_at,
            execution_receipt_refs=(receipt.receipt_id,),
        )
        return AgentGatewayOperationOutcome(
            False,
            failure_reason if failed.accepted else failed.reason_code,
            failed.operation,
            receipt,
        )

    def _time(self) -> float:
        return _finite_time(self._clock(), field_name="clock result")

    def _time_at_least(self, floor: float) -> float:
        return max(self._time(), floor)

    def _control_scope_rejection(
        self,
        *,
        client_id: str,
        runtime_session_id: str,
        skill_route_manifest_hash: str,
        control_domain_id: str | None = None,
        target_binding_id: str | None = None,
    ) -> str | None:
        if client_id not in self.access_policy.allowed_client_ids:
            return "control_client_not_allowed"
        if runtime_session_id != self.manifest.runtime_session_id:
            return "control_runtime_session_mismatch"
        if skill_route_manifest_hash != self.manifest.semantic_content_hash:
            return "control_route_manifest_mismatch"
        if (
            control_domain_id is not None
            and control_domain_id != self.manifest.control_domain_id
        ):
            return "control_domain_mismatch"
        if (
            target_binding_id is not None
            and target_binding_id != self.manifest.target_binding_id
        ):
            return "control_target_binding_mismatch"
        return None

    def _owned_operation(
        self,
        *,
        client_id: str,
        runtime_session_id: str,
        skill_route_manifest_hash: str,
        operation_id: str,
    ) -> AgentSkillOperation | None:
        operation = self.operation_store.get(operation_id)
        request = self.operation_store.request_for_operation(operation_id)
        if operation is None or request is None:
            return None
        if (
            request.client_id != client_id
            or request.runtime_session_id != runtime_session_id
            or request.skill_route_manifest_hash != skill_route_manifest_hash
            or operation.runtime_session_id != runtime_session_id
        ):
            return None
        return operation

    def _begin_control_request(
        self,
        *,
        action: str,
        client_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> AgentGatewayControlPlaneOutcome | None:
        key = (client_id, action, idempotency_key)
        with self._control_request_lock:
            existing = self._control_request_outcomes.get(key)
            if existing is None:
                self._control_request_outcomes[key] = (request_hash, None)
                return None
            known_hash, known_outcome = existing
            self._control_request_outcomes.move_to_end(key)
            if known_hash != request_hash:
                return AgentGatewayControlPlaneOutcome(
                    False,
                    "control_idempotency_key_conflict",
                    request_hash,
                )
            if known_outcome is None:
                return AgentGatewayControlPlaneOutcome(
                    False,
                    "control_request_in_progress",
                    request_hash,
                )
            return AgentGatewayControlPlaneOutcome(
                known_outcome.accepted,
                known_outcome.reason_code,
                known_outcome.request_hash,
                known_outcome.operation,
                idempotent_replay=True,
                port_receipt=known_outcome.port_receipt,
                publish_authoritative_port_receipt=(
                    known_outcome.publish_authoritative_port_receipt
                ),
            )

    def _complete_control_request(
        self,
        *,
        action: str,
        client_id: str,
        idempotency_key: str,
        request_hash: str,
        outcome: AgentGatewayControlPlaneOutcome,
    ) -> None:
        key = (client_id, action, idempotency_key)
        with self._control_request_lock:
            existing = self._control_request_outcomes.get(key)
            if existing is None or existing[0] != request_hash:
                raise RuntimeError("Control request reservation was lost.")
            self._control_request_outcomes[key] = (request_hash, outcome)
            self._control_request_outcomes.move_to_end(key)
            while len(self._control_request_outcomes) > 256:
                evictable = next(
                    (
                        candidate
                        for candidate, (_, cached) in (
                            self._control_request_outcomes.items()
                        )
                        if cached is not None and candidate != key
                    ),
                    None,
                )
                if evictable is None:
                    break
                self._control_request_outcomes.pop(evictable)

    def _abandon_control_request(
        self,
        *,
        action: str,
        client_id: str,
        idempotency_key: str,
        request_hash: str,
    ) -> None:
        key = (client_id, action, idempotency_key)
        with self._control_request_lock:
            if self._control_request_outcomes.get(key) == (request_hash, None):
                self._control_request_outcomes.pop(key)

    def _validate_request_observation(
        self,
        *,
        request: AgentSkillRequest,
        current_station_revision: int,
        now: float,
    ) -> str | None:
        observation = self.get_observation()
        if observation is None:
            return "observation_not_available"
        if request.observation_snapshot_id != observation.snapshot_id:
            return "observation_snapshot_id_mismatch"
        if observation.station_revision != current_station_revision:
            return "observation_station_revision_mismatch"
        if observation.available_at > now:
            return "observation_available_in_future"
        return None

    def _require_port_receipt(
        self,
        receipt: object,
        *,
        expected_port: AgentDomainControlPort | AgentSkillExecutionPort,
        operation_kind: str,
        operation_id: str | None,
    ) -> AgentSkillOperationPortReceipt:
        if not isinstance(
            receipt,
            (
                AgentGatewayNoIoPortReceipt,
                AgentGatewayAuthoritativePortReceipt,
            ),
        ):
            raise TypeError("Operation port returned an invalid receipt type.")
        if receipt.port_id != expected_port.port_id:
            raise ValueError("Operation port receipt port_id mismatch.")
        if receipt.operation_kind != operation_kind:
            raise ValueError("Operation port receipt kind mismatch.")
        if receipt.operation_id != operation_id:
            raise ValueError("Operation port receipt operation mismatch.")
        if expected_port.no_io:
            if not isinstance(receipt, AgentGatewayNoIoPortReceipt):
                raise ValueError(
                    "A no-I/O port cannot return an authoritative receipt."
                )
            if receipt.authoritative_lower_receipt or receipt.proves_motion_or_hold:
                raise ValueError(
                    "B2a receipt cannot claim lower authority or motion."
                )
        else:
            if not isinstance(receipt, AgentGatewayAuthoritativePortReceipt):
                raise ValueError(
                    "An authoritative port must return an authoritative receipt."
                )
            if receipt.succeeded and operation_kind == "hold" and not (
                receipt.proves_motion_or_hold
            ):
                raise ValueError(
                    "Successful HOLD must prove the lower boundary."
                )
        return receipt

    @staticmethod
    def _submission_from_store(
        outcome: OperationStoreOutcome,
        *,
        access_decision: SkillAccessDecision | None = None,
    ) -> AgentGatewaySubmission:
        return AgentGatewaySubmission(
            accepted=outcome.accepted,
            reason_code=outcome.reason_code,
            operation=outcome.operation,
            access_decision=access_decision,
            idempotent_replay=outcome.idempotent_replay,
        )

    @staticmethod
    def _operation_outcome_from_store(
        outcome: OperationStoreOutcome,
    ) -> AgentGatewayOperationOutcome:
        return AgentGatewayOperationOutcome(
            outcome.accepted,
            outcome.reason_code,
            outcome.operation,
        )


class NoIoAgentGatewayService(AgentGatewayService):
    """Compatibility facade that accepts only fake/null operation ports."""

    def __init__(
        self,
        *,
        manifest: SkillRouteManifest,
        access_policy: AgentSkillAccessPolicy,
        operation_port: NoIoAgentSkillOperationPort,
        payload_validators: AgentSkillPayloadValidatorRegistry | None = None,
        operation_store: BoundedInMemoryAgentSkillOperationStore | None = None,
        protocol_runtime: AgentGatewayProtocolRuntime | None = None,
        clock: Callable[[], float] = time.monotonic,
        clock_domain_id: str = "agent-gateway-monotonic",
    ) -> None:
        if getattr(operation_port, "no_io", None) is not True:
            raise TypeError("B2a service accepts only a no_io operation port.")
        super().__init__(
            manifest=manifest,
            access_policy=access_policy,
            operation_port=operation_port,
            payload_validators=payload_validators,
            operation_store=operation_store,
            protocol_runtime=protocol_runtime,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )


__all__ = [
    "AgentGatewayControlPlaneOutcome",
    "AgentGatewayObservationOutcome",
    "AgentGatewayOperationOutcome",
    "AgentGatewayService",
    "AgentGatewaySubmission",
    "NoIoAgentGatewayService",
]
