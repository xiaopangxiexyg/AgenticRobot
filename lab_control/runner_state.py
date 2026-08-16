from __future__ import annotations

import fcntl
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .cancellation import (
    build_cancel_teardown_receipt,
    validate_cancel_service_teardown_evidence,
    validate_cancel_teardown_receipt,
)
from .clean_slot import validate_clean_slot_receipt
from .endpoint_lease import validate_endpoint_lease_receipt
from .execution_sandbox import (
    validate_execution_sandbox_for_launch,
    validate_execution_sandbox_receipt,
    validate_service_sandbox_receipt,
)
from .gpu_lease import (
    GpuLeaseStore,
    build_gpu_cleanup_receipt,
    require_gpu_compute_processes_absent,
    select_single_gpu,
    validate_gpu_release_receipt,
)
from .network_namespace import (
    NetworkNamespaceProbePolicy,
    validate_network_namespace_anchor_observation,
    validate_network_namespace_plan,
    validate_network_namespace_probe_receipt,
)
from .remote_execution import validate_attempt_result, validate_attempt_start
from .remote_preflight import HostProfile, validate_preflight_receipt
from .remote_request import validate_remote_request_identity
from .resource_reservation import (
    build_gpu_resource_reservation,
    validate_gpu_resource_reservation,
)
from .service_identity import (
    validate_service_launch_spec_identity,
    validate_service_receipt,
)
from .service_launch_plan import (
    TrustedServiceAdapter,
    validate_service_launch_plan_bindings,
    validate_service_launch_plan_receipt,
)
from .slot_readiness import validate_slot_readiness_receipt
from .workspace import LabError, _atomic_write_text, validate_task_id

REQUEST_STATE_SCHEMA_ID = "lab.request-state.v8"
ACTIVE_STATES = frozenset(
    {
        "RESERVED",
        "PREPARING",
        "RUNNING",
        "COLLECTING",
        "CANCEL_REQUESTED",
    }
)
TERMINAL_STATES = frozenset({"COMPLETED", "FAILED", "CANCELLED", "QUARANTINED"})
FAILURE_CLASSIFICATIONS = frozenset(
    {
        "FAILED_ASSERTION",
        "BLOCKED_PRECONDITION",
        "INVALID_SOURCE",
        "INVALID_ENVIRONMENT",
        "INFRASTRUCTURE_FAILURE",
    }
)
TRANSITIONS = {
    "SUBMITTED": frozenset({"RESERVED", "CANCELLED", "FAILED"}),
    "RESERVED": frozenset({"PREPARING", "CANCEL_REQUESTED", "FAILED"}),
    "PREPARING": frozenset({"RUNNING", "CANCEL_REQUESTED", "FAILED", "QUARANTINED"}),
    "RUNNING": frozenset({"COLLECTING", "CANCEL_REQUESTED", "FAILED", "QUARANTINED"}),
    "COLLECTING": frozenset({"COMPLETED", "CANCEL_REQUESTED", "FAILED", "QUARANTINED"}),
    "CANCEL_REQUESTED": frozenset({"CANCELLED", "QUARANTINED"}),
}


def validate_request_state_identity(
    state: object,
    *,
    request: object,
) -> dict[str, object]:
    """Validate an externally observed request state against its exact request.

    The state store is trusted to create receipts, but callers still need a strict
    parser before persisting stdout returned over a transport.  This validator is
    intentionally side-effect free and rejects history forks, impossible
    transitions, and receipts that are not bound to the request/slot.
    """

    validated_request = validate_remote_request_identity(request)
    if not isinstance(state, dict):
        raise LabError("request state must be an object")
    base_keys = {
        "schema",
        "request_id",
        "idempotency_key",
        "task_id",
        "profile",
        "state",
        "slot_id",
        "resource_reservation_receipt",
        "resource_release_receipt",
        "preflight_receipt",
        "source_receipt",
        "readiness_receipt",
        "service_launch_spec",
        "endpoint_lease_receipt",
        "network_namespace_plan",
        "network_namespace_anchor_observation",
        "network_namespace_probe_receipt",
        "service_sandbox_receipt",
        "service_launch_plan",
        "service_receipt",
        "execution_sandbox_receipt",
        "active_attempt",
        "attempts",
        "outcome",
        "cancel_teardown_receipt",
        "history",
    }
    expected_keys = base_keys | (
        {"failure"} if state.get("state") == "FAILED" else set()
    )
    if set(state) != expected_keys:
        raise LabError("request state has an unexpected shape")
    for field in ("request_id", "idempotency_key", "task_id", "profile"):
        if state.get(field) != validated_request.get(field):
            raise LabError(f"request state identity mismatch: {field}")
    if state.get("schema") != REQUEST_STATE_SCHEMA_ID:
        raise LabError("unsupported request state schema")

    current_state = state.get("state")
    allowed_states = {"SUBMITTED"} | ACTIVE_STATES | TERMINAL_STATES
    if current_state not in allowed_states:
        raise LabError(f"request state is invalid: {current_state}")
    history = state.get("history")
    if not isinstance(history, list) or not history:
        raise LabError("request state history is empty")
    previous: str | None = None
    for sequence, entry in enumerate(history):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"sequence", "state"}
            or entry.get("sequence") != sequence
            or entry.get("state") not in allowed_states
        ):
            raise LabError("request state history has an invalid entry")
        observed = str(entry["state"])
        if sequence == 0:
            if observed != "SUBMITTED":
                raise LabError("request state history must start at SUBMITTED")
        elif previous is None or observed not in TRANSITIONS.get(previous, frozenset()):
            raise LabError(
                f"request state history has invalid transition: {previous} -> {observed}"
            )
        previous = observed
    if previous != current_state:
        raise LabError("request state does not match its final history entry")

    slot_id = state.get("slot_id")
    if slot_id is not None:
        if not isinstance(slot_id, str):
            raise LabError("request state slot_id must be a string or null")
        validate_task_id(slot_id)

    resource_reservation = state.get("resource_reservation_receipt")
    if resource_reservation is not None:
        if slot_id is None or current_state == "SUBMITTED":
            raise LabError("resource reservation requires an active request slot")
        validate_gpu_resource_reservation(
            resource_reservation,
            request=validated_request,
            slot_id=slot_id,
            require_active=current_state not in TERMINAL_STATES,
        )
    resource_release = state.get("resource_release_receipt")
    if resource_release is not None:
        if not isinstance(resource_release, dict):
            raise LabError("resource release receipt must be an object")
        released = validate_gpu_release_receipt(
            resource_release,
            host_id=str(resource_release.get("host_id", "")),
            host_profile_sha256=str(
                resource_release.get("host_profile_sha256", "")
            ),
            gpu_uuid=str(resource_release.get("gpu_uuid", "")),
        )
        if released.get("request_id") != validated_request.get("request_id"):
            raise LabError("resource release belongs to a different request")
        if isinstance(resource_reservation, dict):
            lease = resource_reservation.get("gpu_lease_receipt")
            if (
                not isinstance(lease, dict)
                or released.get("lease_receipt_sha256")
                != lease.get("receipt_sha256")
            ):
                raise LabError("resource release does not match its reservation")

    request_id = str(validated_request["request_id"])
    idempotency_key = str(validated_request["idempotency_key"])
    preflight = state.get("preflight_receipt")
    if preflight is not None:
        validate_preflight_receipt(
            preflight,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
    source = state.get("source_receipt")
    if source is not None:
        if slot_id is None:
            raise LabError("source receipt requires a request slot")
        validate_clean_slot_receipt(
            source,
            request_id=request_id,
            idempotency_key=idempotency_key,
            slot_id=slot_id,
        )
    readiness = state.get("readiness_receipt")
    if readiness is not None:
        if slot_id is None:
            raise LabError("readiness receipt requires a request slot")
        validate_slot_readiness_receipt(
            readiness,
            request_id=request_id,
            idempotency_key=idempotency_key,
            slot_id=slot_id,
        )
    service_spec = state.get("service_launch_spec")
    validated_spec: dict[str, object] | None = None
    if service_spec is not None:
        validated_spec = validate_service_launch_spec_identity(service_spec)
        if (
            validated_spec.get("request_id") != request_id
            or validated_spec.get("idempotency_key") != idempotency_key
            or validated_spec.get("slot_id") != slot_id
        ):
            raise LabError("service launch spec does not match request state")
    endpoint_lease = state.get("endpoint_lease_receipt")
    validated_endpoint_lease: dict[str, object] | None = None
    if endpoint_lease is not None:
        if validated_spec is None or not isinstance(preflight, dict):
            raise LabError("endpoint lease requires a launch spec and preflight")
        validated_endpoint_lease = validate_endpoint_lease_receipt(
            endpoint_lease,
            host_id=str(preflight.get("host_id", "")),
            host_profile_sha256=str(preflight.get("host_profile_sha256", "")),
            request_id=request_id,
            slot_id=str(slot_id or ""),
            require_active=current_state not in TERMINAL_STATES,
        )
        endpoint_policy = validated_spec.get("endpoint_policy")
        if (
            not isinstance(endpoint_policy, dict)
            or validated_endpoint_lease.get("scheme")
            != endpoint_policy.get("scheme")
            or validated_endpoint_lease.get("host") != endpoint_policy.get("host")
        ):
            raise LabError("endpoint lease differs from request state policy")
    network_namespace_plan = state.get("network_namespace_plan")
    validated_network_plan: dict[str, object] | None = None
    if network_namespace_plan is not None:
        if validated_endpoint_lease is None or not isinstance(preflight, dict):
            raise LabError("network namespace plan requires endpoint and preflight")
        validated_network_plan = validate_network_namespace_plan(
            network_namespace_plan,
            endpoint_lease_receipt=validated_endpoint_lease,
            require_active=current_state not in TERMINAL_STATES,
        )
        if (
            validated_network_plan.get("host_id") != preflight.get("host_id")
            or validated_network_plan.get("host_profile_sha256")
            != preflight.get("host_profile_sha256")
        ):
            raise LabError("network namespace plan differs from request state host")
    network_anchor = state.get("network_namespace_anchor_observation")
    validated_network_anchor: dict[str, object] | None = None
    if network_anchor is not None:
        if validated_network_plan is None or not isinstance(preflight, dict):
            raise LabError("network namespace anchor requires its plan and preflight")
        validated_network_anchor = validate_network_namespace_anchor_observation(
            network_anchor,
            network_namespace_plan=validated_network_plan,
        )
        if (
            validated_network_anchor.get("host_id") != preflight.get("host_id")
            or validated_network_anchor.get("host_profile_sha256")
            != preflight.get("host_profile_sha256")
        ):
            raise LabError("network namespace anchor differs from request state host")
    network_probe = state.get("network_namespace_probe_receipt")
    validated_network_probe: dict[str, object] | None = None
    if network_probe is not None:
        if validated_network_plan is None or validated_network_anchor is None:
            raise LabError("network namespace probe requires its plan and anchor")
        validated_network_probe = validate_network_namespace_probe_receipt(
            network_probe,
            network_namespace_plan=validated_network_plan,
            anchor_observation=validated_network_anchor,
        )
    service_sandbox = state.get("service_sandbox_receipt")
    validated_service_sandbox: dict[str, object] | None = None
    if service_sandbox is not None:
        if validated_spec is None or not isinstance(preflight, dict):
            raise LabError("service sandbox requires a launch spec and preflight")
        validated_service_sandbox = validate_service_sandbox_receipt(
            service_sandbox
        )
        sandbox_bindings = {
            "service_id": validated_spec.get("service_id"),
            "request_id": request_id,
            "slot_id": slot_id,
            "host_id": preflight.get("host_id"),
            "host_profile_sha256": preflight.get("host_profile_sha256"),
            "remote_profile_sha256": validated_spec.get("remote_profile_sha256"),
            "control_plane_version": preflight.get("control_plane_version"),
            "launch_spec_sha256": validated_spec.get("launch_spec_sha256"),
            "readiness_receipt_sha256": validated_spec.get(
                "readiness_receipt_sha256"
            ),
            "gpu": validated_spec.get("gpu"),
        }
        for field, expected in sandbox_bindings.items():
            if validated_service_sandbox.get(field) != expected:
                raise LabError(
                    f"service sandbox does not match request state: {field}"
                )
    service_plan = state.get("service_launch_plan")
    validated_service_plan: dict[str, object] | None = None
    if service_plan is not None:
        if (
            validated_spec is None
            or validated_endpoint_lease is None
            or validated_network_plan is None
            or validated_service_sandbox is None
        ):
            raise LabError(
                "service launch plan requires its spec, endpoint, network, and sandbox"
            )
        validated_service_plan = validate_service_launch_plan_receipt(service_plan)
        raw_adapter = validated_service_plan.get("adapter")
        raw_endpoint = validated_service_plan.get("endpoint")
        endpoint_policy = validated_spec.get("endpoint_policy")
        if (
            not isinstance(raw_adapter, dict)
            or not isinstance(raw_endpoint, dict)
            or not isinstance(endpoint_policy, dict)
        ):
            raise LabError("service launch plan identity is incomplete")
        plan_bindings = {
            "service_id": validated_spec.get("service_id"),
            "request_id": request_id,
            "slot_id": slot_id,
            "host_id": validated_service_sandbox.get("host_id"),
            "host_profile_sha256": validated_service_sandbox.get(
                "host_profile_sha256"
            ),
            "launch_spec_sha256": validated_spec.get("launch_spec_sha256"),
            "service_sandbox_receipt_sha256": validated_service_sandbox.get(
                "receipt_sha256"
            ),
            "endpoint_lease_receipt_sha256": validated_endpoint_lease.get(
                "receipt_sha256"
            ),
            "network_namespace_plan_sha256": validated_network_plan.get(
                "receipt_sha256"
            ),
        }
        for field, expected in plan_bindings.items():
            if validated_service_plan.get(field) != expected:
                raise LabError(
                    f"service launch plan does not match request state: {field}"
                )
        if (
            raw_adapter.get("launch_command_id")
            != validated_spec.get("launch_command_id")
            or raw_endpoint.get("scheme") != endpoint_policy.get("scheme")
            or raw_endpoint.get("host") != endpoint_policy.get("host")
        ):
            raise LabError("service launch plan policy differs from request state")
    service_receipt = state.get("service_receipt")
    validated_service: dict[str, object] | None = None
    if service_receipt is not None:
        validated_service = validate_service_receipt(service_receipt)
        if (
            validated_spec is None
            or validated_service_plan is None
            or validated_network_anchor is None
            or validated_network_probe is None
        ):
            raise LabError(
                "service receipt requires launch plan and network evidence"
            )
        service_bindings = {
            "request_id": request_id,
            "task_id": state.get("task_id"),
            "slot_id": slot_id,
            "launch_spec_sha256": service_spec.get("launch_spec_sha256"),
            "remote_profile_sha256": service_spec.get("remote_profile_sha256"),
            "readiness_receipt_sha256": service_spec.get("readiness_receipt_sha256"),
        }
        for field, expected in service_bindings.items():
            if validated_service.get(field) != expected:
                raise LabError(f"service receipt does not match request state: {field}")
        if validated_service.get("endpoint") != validated_service_plan.get(
            "endpoint"
        ):
            raise LabError("service receipt endpoint differs from launch plan")
    sandbox_receipt = state.get("execution_sandbox_receipt")
    validated_sandbox: dict[str, object] | None = None
    if sandbox_receipt is not None:
        validated_sandbox = validate_execution_sandbox_receipt(sandbox_receipt)
        if not isinstance(service_spec, dict) or validated_service is None:
            raise LabError("execution sandbox requires a managed service receipt")
        sandbox_bindings = {
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "task_id": state.get("task_id"),
            "slot_id": slot_id,
            "remote_profile_sha256": service_spec.get("remote_profile_sha256"),
            "launch_spec_sha256": service_spec.get("launch_spec_sha256"),
            "service_receipt_sha256": validated_service.get("receipt_sha256"),
            "gpu": validated_service.get("gpu"),
        }
        for field, expected in sandbox_bindings.items():
            if validated_sandbox.get(field) != expected:
                raise LabError(f"execution sandbox does not match request state: {field}")
    active_attempt = state.get("active_attempt")
    if active_attempt is not None:
        start = validate_attempt_start(active_attempt)
        if (
            validated_service is None
            or start.get("request_id") != request_id
            or start.get("idempotency_key") != idempotency_key
            or start.get("task_id") != state.get("task_id")
            or start.get("slot_id") != slot_id
            or validated_sandbox is None
            or start.get("execution_sandbox_receipt_sha256")
            != validated_sandbox.get("receipt_sha256")
            or start.get("service_receipt_sha256")
            != validated_service.get("receipt_sha256")
        ):
            raise LabError("active attempt does not match request state")
    attempts = state.get("attempts")
    if not isinstance(attempts, list):
        raise LabError("request state attempts must be a list")
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict) or set(attempt) != {"start", "result"}:
            raise LabError("request state has an invalid attempt entry")
        start = validate_attempt_start(attempt.get("start"))
        result = validate_attempt_result(attempt.get("result"), start_receipt=start)
        if (
            start.get("request_id") != request_id
            or start.get("attempt_number") != index
            or result.get("attempt_number") != index
        ):
            raise LabError("request state attempt sequence mismatch")
    outcome = state.get("outcome")
    if outcome is not None:
        if (
            not attempts
            or not isinstance(outcome, dict)
            or set(outcome)
            != {
                "attempt_id",
                "classification",
                "task_success",
                "evidence_complete",
            }
        ):
            raise LabError("request state outcome is invalid")
        latest_result = attempts[-1]["result"]
        if not isinstance(latest_result, dict) or outcome != {
            "attempt_id": latest_result.get("attempt_id"),
            "classification": latest_result.get("classification"),
            "task_success": latest_result.get("task_success"),
            "evidence_complete": latest_result.get("evidence_complete"),
        }:
            raise LabError("request state outcome does not match latest attempt")
    cancel_receipt = state.get("cancel_teardown_receipt")
    if cancel_receipt is not None:
        if slot_id is None or current_state != "CANCELLED":
            raise LabError(
                "cancel teardown receipt requires a cancelled active request"
            )
        validate_cancel_teardown_receipt(
            cancel_receipt,
            request=validated_request,
            slot_id=slot_id,
            resource_reservation_receipt=resource_reservation,
            resource_release_receipt=resource_release,
            service_launch_spec=service_spec,
            service_receipt=service_receipt,
        )
    if current_state == "FAILED":
        failure = state.get("failure")
        if (
            not isinstance(failure, dict)
            or set(failure) != {"classification", "detail"}
            or failure.get("classification") not in FAILURE_CLASSIFICATIONS
            or not isinstance(failure.get("detail"), str)
            or not str(failure["detail"]).strip()
        ):
            raise LabError("failed request state has an invalid failure receipt")
    return state


class RequestStateStore:
    """Persistent request/slot state with no execution or remote side effects."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.requests_root = self.root / "requests"
        self.locks_root = self.root / "locks"

    @contextmanager
    def _lock(self, name: str) -> Iterator[None]:
        validate_task_id(name)
        self.locks_root.mkdir(parents=True, exist_ok=True)
        path = self.locks_root / f"{name}.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LabError(f"request state lock is held: {name}") from exc
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _path(self, request_id: str) -> Path:
        validate_task_id(request_id)
        return self.requests_root / request_id / "state.json"

    def load(self, request_id: str) -> dict[str, object]:
        path = self._path(request_id)
        if not path.is_file():
            raise LabError(f"request state does not exist: {request_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read request state {path}: {exc}") from exc
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != REQUEST_STATE_SCHEMA_ID
            or payload.get("request_id") != request_id
        ):
            raise LabError(f"invalid request state identity: {path}")
        return payload

    def watch(
        self,
        request_id: str,
        *,
        after_sequence: int = -1,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 0.25,
    ) -> dict[str, object]:
        """Wait for append-only state history to advance; never mutate request state."""

        if after_sequence < -1:
            raise LabError("watch after_sequence must be at least -1")
        if not 0 <= timeout_seconds <= 60:
            raise LabError("watch timeout must be between 0 and 60 seconds")
        if not 0.01 <= poll_interval_seconds <= 5:
            raise LabError("watch poll interval must be between 0.01 and 5 seconds")
        deadline = time.monotonic() + timeout_seconds
        while True:
            current = self.load(request_id)
            history = current.get("history")
            if not isinstance(history, list) or not history:
                raise LabError(f"request state has invalid history: {request_id}")
            last = history[-1]
            if not isinstance(last, dict) or not isinstance(last.get("sequence"), int):
                raise LabError(f"request state has invalid history entry: {request_id}")
            sequence = int(last["sequence"])
            terminal = current.get("state") in TERMINAL_STATES
            changed = sequence > after_sequence
            if changed or terminal or time.monotonic() >= deadline:
                return {
                    "schema": "lab.request-watch.v1",
                    "request_id": request_id,
                    "after_sequence": after_sequence,
                    "observed_sequence": sequence,
                    "changed": changed,
                    "terminal": terminal,
                    "request": current,
                }
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval_seconds, max(remaining, 0)))

    def submit(self, request: dict[str, object]) -> dict[str, object]:
        request = validate_remote_request_identity(request)
        request_id = str(request.get("request_id", ""))
        idempotency_key = str(request.get("idempotency_key", ""))
        validate_task_id(request_id)
        with self._lock("request-store"):
            path = self._path(request_id)
            if path.exists():
                existing = self.load(request_id)
                if existing.get("idempotency_key") != idempotency_key:
                    raise LabError("request ID collision with different content")
                return existing
            payload: dict[str, object] = {
                "schema": REQUEST_STATE_SCHEMA_ID,
                "request_id": request_id,
                "idempotency_key": idempotency_key,
                "task_id": request.get("task_id"),
                "profile": request.get("profile"),
                "state": "SUBMITTED",
                "slot_id": None,
                "resource_reservation_receipt": None,
                "resource_release_receipt": None,
                "preflight_receipt": None,
                "source_receipt": None,
                "readiness_receipt": None,
                "service_launch_spec": None,
                "endpoint_lease_receipt": None,
                "network_namespace_plan": None,
                "network_namespace_anchor_observation": None,
                "network_namespace_probe_receipt": None,
                "service_sandbox_receipt": None,
                "service_launch_plan": None,
                "service_receipt": None,
                "execution_sandbox_receipt": None,
                "active_attempt": None,
                "attempts": [],
                "outcome": None,
                "cancel_teardown_receipt": None,
                "history": [{"sequence": 0, "state": "SUBMITTED"}],
            }
            _atomic_write_text(
                path, json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            return payload

    def reserve(
        self,
        request_id: str,
        *,
        slot_id: str,
        allowed_slots: tuple[str, ...],
    ) -> dict[str, object]:
        validate_task_id(slot_id)
        if slot_id not in allowed_slots:
            raise LabError(f"slot is not authorized by host Profile: {slot_id}")
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") == "RESERVED" and current.get("slot_id") == slot_id:
                return current
            if current.get("state") != "SUBMITTED":
                raise LabError(
                    f"request cannot reserve from state {current.get('state')}: {request_id}"
                )
            for path in self.requests_root.glob("*/state.json"):
                other = self.load(path.parent.name)
                if (
                    other.get("request_id") != request_id
                    and other.get("slot_id") == slot_id
                    and other.get("state") in ACTIVE_STATES
                ):
                    raise LabError(
                        f"slot {slot_id} is already reserved by {other.get('request_id')}"
                    )
            return self._transition_unlocked(current, "RESERVED", slot_id=slot_id)

    def reserve_gpu(
        self,
        request: dict[str, object],
        *,
        slot_id: str,
        allowed_slots: tuple[str, ...],
        host_id: str,
        host_profile_sha256: str,
        remote_profile_sha256: str,
        inventory_receipt: object,
        process_inventory_receipt: object,
        gpu_store: GpuLeaseStore,
    ) -> dict[str, object]:
        """Reserve the request slot and GPU while holding the request-store lock.

        A host crash after the persistent GPU write but before the request-state
        write leaves an orphan lease.  The lease intentionally remains fail-closed
        and requires trusted reconciliation; it is never silently reused.
        """

        validated_request = validate_remote_request_identity(request)
        request_id = str(validated_request["request_id"])
        validate_task_id(slot_id)
        if slot_id not in allowed_slots:
            raise LabError(f"slot is not authorized by host Profile: {slot_id}")
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("idempotency_key") != validated_request.get(
                "idempotency_key"
            ):
                raise LabError("resource reservation request identity mismatch")
            existing = current.get("resource_reservation_receipt")
            if current.get("state") == "RESERVED" and isinstance(existing, dict):
                validated = validate_gpu_resource_reservation(
                    existing,
                    request=validated_request,
                    slot_id=slot_id,
                    host_id=host_id,
                    host_profile_sha256=host_profile_sha256,
                    remote_profile_sha256=remote_profile_sha256,
                    require_active=True,
                )
                lease = validated["gpu_lease_receipt"]
                assert isinstance(lease, dict)
                gpu_store.observation(lease, gpu_uuid=str(lease["gpu_uuid"]))
                return current
            if current.get("state") != "SUBMITTED":
                raise LabError(
                    "request cannot reserve GPU resources from state "
                    f"{current.get('state')}: {request_id}"
                )
            for path in self.requests_root.glob("*/state.json"):
                other = self.load(path.parent.name)
                if (
                    other.get("request_id") != request_id
                    and other.get("slot_id") == slot_id
                    and other.get("state") in ACTIVE_STATES
                ):
                    raise LabError(
                        f"slot {slot_id} is already reserved by {other.get('request_id')}"
                    )
            gpu_id, gpu_uuid = select_single_gpu(inventory_receipt)
            validated_process_inventory = require_gpu_compute_processes_absent(
                process_inventory_receipt,
                gpu_uuid=gpu_uuid,
            )
            lease = gpu_store.acquire(
                gpu_id=gpu_id,
                gpu_uuid=gpu_uuid,
                request_id=request_id,
                slot_id=slot_id,
            )
            reservation = build_gpu_resource_reservation(
                validated_request,
                slot_id=slot_id,
                host_id=host_id,
                host_profile_sha256=host_profile_sha256,
                remote_profile_sha256=remote_profile_sha256,
                inventory_receipt=inventory_receipt,
                process_inventory_receipt=validated_process_inventory,
                lease_receipt=lease,
            )
            return self._transition_unlocked(
                current,
                "RESERVED",
                slot_id=slot_id,
                resource_reservation_receipt=reservation,
            )

    def reconcile_nonstarted_gpu(
        self,
        request: dict[str, object],
        *,
        gpu_uuid: str,
        process_inventory_receipt: object,
        gpu_store: GpuLeaseStore,
    ) -> dict[str, object]:
        """Release an orphan/failed lease only when service startup is impossible."""

        validated_request = validate_remote_request_identity(request)
        request_id = str(validated_request["request_id"])
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("idempotency_key") != validated_request.get(
                "idempotency_key"
            ):
                raise LabError("GPU reconciliation request identity mismatch")
            existing_release = current.get("resource_release_receipt")
            if isinstance(existing_release, dict):
                validate_gpu_release_receipt(
                    existing_release,
                    host_id=gpu_store.host_id,
                    host_profile_sha256=gpu_store.host_profile_sha256,
                    gpu_uuid=gpu_uuid,
                )
                return current
            if current.get("state") not in {
                "SUBMITTED",
                "FAILED",
                "CANCEL_REQUESTED",
                "QUARANTINED",
            }:
                raise LabError(
                    "nonstarted GPU reconciliation requires a stopped request state"
                )
            if (
                current.get("service_launch_spec") is not None
                or current.get("service_receipt") is not None
                or current.get("active_attempt") is not None
                or current.get("attempts") != []
            ):
                raise LabError("GPU reconciliation cannot prove service never started")
            reservation = current.get("resource_reservation_receipt")
            lease: dict[str, object] | None = None
            if isinstance(reservation, dict):
                raw_lease = reservation.get("gpu_lease_receipt")
                if isinstance(raw_lease, dict):
                    lease = raw_lease
            if lease is None:
                lease = gpu_store.active_lease(gpu_uuid=gpu_uuid)
            if (
                not isinstance(lease, dict)
                or lease.get("request_id") != request_id
                or lease.get("gpu_uuid") != gpu_uuid
            ):
                raise LabError("GPU reconciliation cannot identify the request lease")
            require_gpu_compute_processes_absent(
                process_inventory_receipt,
                gpu_uuid=gpu_uuid,
                max_age_seconds=30,
            )
            cleanup = build_gpu_cleanup_receipt(
                lease,
                host_id=gpu_store.host_id,
                gpu_uuid=gpu_uuid,
                service_started=False,
                service_stopped=False,
                process_inventory_receipt=process_inventory_receipt,
                host_profile_sha256=gpu_store.host_profile_sha256,
            )
            released = gpu_store.release(
                lease,
                gpu_uuid=gpu_uuid,
                cleanup_receipt=cleanup,
            )
            if current.get("state") == "SUBMITTED":
                return self._transition_unlocked(
                    current,
                    "FAILED",
                    resource_release_receipt=released,
                    failure={
                        "classification": "INFRASTRUCTURE_FAILURE",
                        "detail": "reconciled orphan GPU lease before service startup",
                    },
                )
            updated = {**current, "resource_release_receipt": released}
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def attach_preflight(
        self,
        request_id: str,
        receipt: dict[str, object],
    ) -> dict[str, object]:
        with self._lock("request-store"):
            current = self.load(request_id)
            validated_receipt = validate_preflight_receipt(
                receipt,
                request_id=request_id,
                idempotency_key=str(current.get("idempotency_key", "")),
            )
            if current.get("state") == "PREPARING":
                if current.get("preflight_receipt") == validated_receipt:
                    return current
                raise LabError("request already has a different preflight receipt")
            if current.get("state") != "RESERVED":
                raise LabError(
                    f"request cannot prepare from state {current.get('state')}: {request_id}"
                )
            reservation = current.get("resource_reservation_receipt")
            if isinstance(reservation, dict):
                lease = reservation.get("gpu_lease_receipt")
                if not isinstance(lease, dict) or (
                    validated_receipt.get("gpu_lease_id") != lease.get("lease_id")
                    or validated_receipt.get("gpu_lease_receipt_sha256")
                    != lease.get("receipt_sha256")
                    or validated_receipt.get("gpu_uuid") != lease.get("gpu_uuid")
                ):
                    raise LabError("preflight does not match the reserved GPU lease")
            return self._transition_unlocked(
                current,
                "PREPARING",
                preflight_receipt=validated_receipt,
            )

    def attach_source_preparation(
        self,
        request_id: str,
        receipt: dict[str, object],
    ) -> dict[str, object]:
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    f"source preparation can only attach while PREPARING: {request_id}"
                )
            preflight = current.get("preflight_receipt")
            if not isinstance(preflight, dict):
                raise LabError("source preparation requires a preflight receipt")
            slot_id = str(current.get("slot_id", ""))
            validated = validate_clean_slot_receipt(
                receipt,
                request_id=request_id,
                idempotency_key=str(current.get("idempotency_key", "")),
                slot_id=slot_id,
            )
            for field in (
                "host_profile_sha256",
                "remote_profile_sha256",
                "control_plane_version",
            ):
                if validated.get(field) != preflight.get(field):
                    raise LabError(f"source receipt does not match preflight {field}")
            existing = current.get("source_receipt")
            if existing is not None:
                if existing == validated:
                    return current
                raise LabError("request already has a different source receipt")
            updated = {**current, "source_receipt": validated}
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def transition(self, request_id: str, new_state: str) -> dict[str, object]:
        if new_state not in ACTIVE_STATES | TERMINAL_STATES:
            raise LabError(f"unknown request state: {new_state}")
        if new_state == "FAILED":
            raise LabError("FAILED transitions require an explicit failure receipt")
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") == new_state:
                return current
            if new_state == "RUNNING" and current.get("state") == "PREPARING":
                raise LabError(
                    "RUNNING is disabled until the trusted execution edge is implemented"
                )
            if new_state == "CANCELLED" and current.get("state") == "CANCEL_REQUESTED":
                raise LabError(
                    "active cancellation requires an explicit teardown receipt"
                )
            return self._transition_unlocked(current, new_state)

    def acknowledge_cancel(
        self,
        request: dict[str, object],
        receipt: dict[str, object],
    ) -> dict[str, object]:
        """Complete active cancellation only after trusted executor teardown."""

        validated_request = validate_remote_request_identity(request)
        request_id = str(validated_request["request_id"])
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("idempotency_key") != validated_request.get(
                "idempotency_key"
            ):
                raise LabError("cancel acknowledgement request identity mismatch")
            if current.get("state") == "CANCELLED":
                existing = current.get("cancel_teardown_receipt")
                if existing == receipt:
                    return current
                raise LabError("request was cancelled with a different receipt")
            if current.get("state") != "CANCEL_REQUESTED":
                raise LabError(
                    f"cancel acknowledgement requires CANCEL_REQUESTED: {request_id}"
                )
            validated = validate_cancel_teardown_receipt(
                receipt,
                request=validated_request,
                slot_id=str(current.get("slot_id", "")),
                resource_reservation_receipt=current.get(
                    "resource_reservation_receipt"
                ),
                resource_release_receipt=current.get("resource_release_receipt"),
                service_launch_spec=current.get("service_launch_spec"),
                service_receipt=current.get("service_receipt"),
            )
            return self._transition_unlocked(
                current,
                "CANCELLED",
                cancel_teardown_receipt=validated,
            )

    def complete_cancel(
        self,
        request: dict[str, object],
        *,
        gpu_uuid: str | None,
        gpu_process_inventory_receipt: object | None,
        gpu_store: GpuLeaseStore | None,
        service_sandbox_receipt: object | None = None,
        managed_process_receipt: object | None = None,
        managed_service_teardown_receipt: object | None = None,
    ) -> dict[str, object]:
        """Release resources and atomically record strict cancellation evidence.

        This method never sends a signal. The protected launcher must terminate
        and reap all registered processes before supplying teardown evidence.
        """

        validated_request = validate_remote_request_identity(request)
        request_id = str(validated_request["request_id"])
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("idempotency_key") != validated_request.get(
                "idempotency_key"
            ):
                raise LabError("cancel completion request identity mismatch")
            if current.get("state") == "CANCELLED":
                if current.get("cancel_teardown_receipt") is None:
                    raise LabError("cancelled request lacks active teardown evidence")
                validate_request_state_identity(current, request=validated_request)
                return current
            if current.get("state") != "CANCEL_REQUESTED":
                raise LabError(
                    f"cancel completion requires CANCEL_REQUESTED: {request_id}"
                )
            if current.get("active_attempt") is not None:
                raise LabError(
                    "cancel completion requires separate active-attempt teardown"
                )
            slot_id = str(current.get("slot_id", ""))
            reservation = current.get("resource_reservation_receipt")
            release = current.get("resource_release_receipt")
            service_spec = current.get("service_launch_spec")
            service = current.get("service_receipt")
            service_started = service is not None
            evidence = (
                service_sandbox_receipt,
                managed_process_receipt,
                managed_service_teardown_receipt,
            )
            if service_started:
                if not isinstance(reservation, dict) or not isinstance(
                    service_spec, dict
                ):
                    raise LabError("managed service cancellation lacks state identity")
                if any(value is None for value in evidence):
                    raise LabError(
                        "managed service cancellation requires full teardown evidence"
                    )
                validate_cancel_service_teardown_evidence(
                    request=validated_request,
                    slot_id=slot_id,
                    resource_reservation_receipt=reservation,
                    service_launch_spec=service_spec,
                    service_receipt=service,
                    service_sandbox_receipt=service_sandbox_receipt,
                    managed_process_receipt=managed_process_receipt,
                    managed_service_teardown_receipt=managed_service_teardown_receipt,
                )
            elif service_spec is not None or any(value is not None for value in evidence):
                raise LabError(
                    "partially started service requires quarantine, not cancellation"
                )
            if isinstance(reservation, dict):
                if gpu_store is None or gpu_uuid is None:
                    raise LabError("cancel completion requires its GPU lease store")
                raw_lease = reservation.get("gpu_lease_receipt")
                if (
                    not isinstance(raw_lease, dict)
                    or raw_lease.get("gpu_uuid") != gpu_uuid
                    or raw_lease.get("request_id") != request_id
                ):
                    raise LabError("cancel completion GPU lease identity mismatch")
                if release is None:
                    if gpu_process_inventory_receipt is None:
                        raise LabError(
                            "cancel completion requires fresh GPU process evidence"
                        )
                    require_gpu_compute_processes_absent(
                        gpu_process_inventory_receipt,
                        gpu_uuid=gpu_uuid,
                        max_age_seconds=30,
                    )
                    cleanup = build_gpu_cleanup_receipt(
                        raw_lease,
                        host_id=gpu_store.host_id,
                        gpu_uuid=gpu_uuid,
                        service_started=service_started,
                        service_stopped=service_started,
                        process_inventory_receipt=gpu_process_inventory_receipt,
                        host_profile_sha256=gpu_store.host_profile_sha256,
                    )
                    release = gpu_store.release(
                        raw_lease,
                        gpu_uuid=gpu_uuid,
                        cleanup_receipt=cleanup,
                    )
            elif any(
                value is not None
                for value in (release, gpu_uuid, gpu_process_inventory_receipt, gpu_store)
            ):
                raise LabError("cancel completion received unexpected GPU evidence")
            cancel_receipt = build_cancel_teardown_receipt(
                validated_request,
                slot_id=slot_id,
                resource_reservation_receipt=reservation,
                resource_release_receipt=release,
                service_launch_spec=service_spec,
                service_receipt=service,
                service_sandbox_receipt=service_sandbox_receipt,
                managed_process_receipt=managed_process_receipt,
                managed_service_teardown_receipt=managed_service_teardown_receipt,
            )
            return self._transition_unlocked(
                current,
                "CANCELLED",
                resource_release_receipt=release,
                cancel_teardown_receipt=cancel_receipt,
            )

    def attach_readiness(
        self,
        request_id: str,
        receipt: dict[str, object],
    ) -> dict[str, object]:
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    f"readiness can only attach while PREPARING: {request_id}"
                )
            preflight = current.get("preflight_receipt")
            source = current.get("source_receipt")
            if not isinstance(preflight, dict) or not isinstance(source, dict):
                raise LabError("readiness requires preflight and source receipts")
            slot_id = str(current.get("slot_id", ""))
            validated = validate_slot_readiness_receipt(
                receipt,
                request_id=request_id,
                idempotency_key=str(current.get("idempotency_key", "")),
                slot_id=slot_id,
            )
            bindings = {
                "host_profile_sha256": preflight.get("host_profile_sha256"),
                "remote_profile_sha256": preflight.get("remote_profile_sha256"),
                "control_plane_version": preflight.get("control_plane_version"),
                "preflight_receipt_sha256": preflight.get("receipt_sha256"),
                "source_receipt_sha256": source.get("receipt_sha256"),
                "observation_sha256": preflight.get("observation_sha256"),
            }
            for field, expected in bindings.items():
                if validated.get(field) != expected:
                    raise LabError(f"readiness receipt binding mismatch: {field}")
            existing = current.get("readiness_receipt")
            if existing is not None:
                if existing == validated:
                    return current
                raise LabError("request already has a different readiness receipt")
            updated = {**current, "readiness_receipt": validated}
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def attach_service_launch_spec(
        self,
        request_id: str,
        spec: dict[str, object],
    ) -> dict[str, object]:
        """Attach an inert service identity without claiming a process was started."""

        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    f"service launch spec can only attach while PREPARING: {request_id}"
                )
            readiness = current.get("readiness_receipt")
            if not isinstance(readiness, dict):
                raise LabError("service launch spec requires a readiness receipt")
            validated = validate_service_launch_spec_identity(spec)
            bindings = {
                "request_id": request_id,
                "idempotency_key": current.get("idempotency_key"),
                "task_id": current.get("task_id"),
                "slot_id": current.get("slot_id"),
                "remote_profile_sha256": readiness.get("remote_profile_sha256"),
                "readiness_receipt_sha256": readiness.get("receipt_sha256"),
            }
            for field, expected in bindings.items():
                if validated.get(field) != expected:
                    raise LabError(f"service launch spec binding mismatch: {field}")
            existing = current.get("service_launch_spec")
            if existing is not None:
                if existing == validated:
                    return current
                raise LabError("request already has a different service launch spec")
            updated = {**current, "service_launch_spec": validated}
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def attach_service_receipt(
        self,
        request_id: str,
        receipt: dict[str, object],
    ) -> dict[str, object]:
        """Bind a managed service identity; this alone grants no execution."""

        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    f"service receipt can only attach while PREPARING: {request_id}"
                )
            spec = current.get("service_launch_spec")
            plan = current.get("service_launch_plan")
            network_anchor = current.get("network_namespace_anchor_observation")
            network_probe = current.get("network_namespace_probe_receipt")
            if (
                not isinstance(spec, dict)
                or not isinstance(plan, dict)
                or not isinstance(network_anchor, dict)
                or not isinstance(network_probe, dict)
            ):
                raise LabError(
                    "service receipt requires launch plan and network evidence"
                )
            validated = validate_service_receipt(receipt)
            bindings = {
                "request_id": request_id,
                "task_id": current.get("task_id"),
                "slot_id": current.get("slot_id"),
                "launch_spec_sha256": spec.get("launch_spec_sha256"),
                "remote_profile_sha256": spec.get("remote_profile_sha256"),
                "readiness_receipt_sha256": spec.get("readiness_receipt_sha256"),
                "service_id": spec.get("service_id"),
            }
            for field, expected in bindings.items():
                if validated.get(field) != expected:
                    raise LabError(f"service receipt binding mismatch: {field}")
            if validated.get("endpoint") != plan.get("endpoint"):
                raise LabError("service receipt endpoint differs from launch plan")
            existing = current.get("service_receipt")
            if existing is not None:
                if existing == validated:
                    return current
                raise LabError("request already has a different service receipt")
            updated = {**current, "service_receipt": validated}
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def attach_service_launch_plan(
        self,
        request_id: str,
        *,
        service_sandbox_receipt: dict[str, object],
        launch_plan: dict[str, object],
        endpoint_lease_receipt: dict[str, object],
        network_namespace_plan: dict[str, object],
        host_profile: HostProfile,
        adapter: TrustedServiceAdapter,
    ) -> dict[str, object]:
        """Atomically persist one protected, inert service plan and its sandbox."""

        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    f"service launch plan can only attach while PREPARING: {request_id}"
                )
            spec = current.get("service_launch_spec")
            readiness = current.get("readiness_receipt")
            preflight = current.get("preflight_receipt")
            if (
                not isinstance(spec, dict)
                or not isinstance(readiness, dict)
                or not isinstance(preflight, dict)
            ):
                raise LabError(
                    "service launch plan requires preflight, readiness, and launch spec"
                )
            if (
                host_profile.host_id != preflight.get("host_id")
                or host_profile.sha256 != preflight.get("host_profile_sha256")
            ):
                raise LabError("service launch plan host Profile differs from preflight")
            sandbox = validate_service_sandbox_receipt(service_sandbox_receipt)
            endpoint_lease = validate_endpoint_lease_receipt(
                endpoint_lease_receipt,
                host_id=host_profile.host_id,
                host_profile_sha256=host_profile.sha256,
                request_id=request_id,
                slot_id=str(current.get("slot_id", "")),
            )
            network_plan = validate_network_namespace_plan(
                network_namespace_plan,
                endpoint_lease_receipt=endpoint_lease,
                host_profile=host_profile,
            )
            bindings = {
                "service_id": spec.get("service_id"),
                "request_id": request_id,
                "slot_id": current.get("slot_id"),
                "host_id": host_profile.host_id,
                "host_profile_sha256": host_profile.sha256,
                "remote_profile_sha256": spec.get("remote_profile_sha256"),
                "control_plane_version": preflight.get("control_plane_version"),
                "launch_spec_sha256": spec.get("launch_spec_sha256"),
                "readiness_receipt_sha256": readiness.get("receipt_sha256"),
                "gpu": spec.get("gpu"),
            }
            for field, expected in bindings.items():
                if sandbox.get(field) != expected:
                    raise LabError(f"service sandbox binding mismatch: {field}")
            validated_plan = validate_service_launch_plan_bindings(
                launch_plan,
                launch_spec=spec,
                service_sandbox_receipt=sandbox,
                host_profile=host_profile,
                adapter=adapter,
                endpoint_lease_receipt=endpoint_lease,
                network_namespace_plan=network_plan,
            )
            existing_endpoint = current.get("endpoint_lease_receipt")
            existing_network = current.get("network_namespace_plan")
            existing_sandbox = current.get("service_sandbox_receipt")
            existing_plan = current.get("service_launch_plan")
            if (
                existing_endpoint is not None
                or existing_network is not None
                or existing_sandbox is not None
                or existing_plan is not None
            ):
                if (
                    existing_endpoint == endpoint_lease
                    and existing_network == network_plan
                    and existing_sandbox == sandbox
                    and existing_plan == validated_plan
                ):
                    return current
                raise LabError("request already has a different service launch plan")
            updated = {
                **current,
                "endpoint_lease_receipt": endpoint_lease,
                "network_namespace_plan": network_plan,
                "service_sandbox_receipt": sandbox,
                "service_launch_plan": validated_plan,
            }
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def attach_execution_sandbox(
        self,
        request_id: str,
        receipt: dict[str, object],
    ) -> dict[str, object]:
        """Persist the enforced sandbox identity before entering RUNNING."""

        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    f"execution sandbox can only attach while PREPARING: {request_id}"
                )
            service_spec = current.get("service_launch_spec")
            service_plan = current.get("service_launch_plan")
            service = current.get("service_receipt")
            preflight = current.get("preflight_receipt")
            network_anchor = current.get("network_namespace_anchor_observation")
            network_probe = current.get("network_namespace_probe_receipt")
            if (
                not isinstance(service_spec, dict)
                or not isinstance(service_plan, dict)
                or not isinstance(service, dict)
                or not isinstance(preflight, dict)
                or not isinstance(network_anchor, dict)
                or not isinstance(network_probe, dict)
            ):
                raise LabError(
                    "execution sandbox requires managed service and network evidence"
                )
            if service.get("endpoint") != service_plan.get("endpoint"):
                raise LabError("managed service endpoint differs from launch plan")
            validated = validate_execution_sandbox_for_launch(receipt)
            bindings = {
                "request_id": request_id,
                "idempotency_key": current.get("idempotency_key"),
                "task_id": current.get("task_id"),
                "slot_id": current.get("slot_id"),
                "host_profile_sha256": preflight.get("host_profile_sha256"),
                "remote_profile_sha256": service_spec.get(
                    "remote_profile_sha256"
                ),
                "launch_spec_sha256": service_spec.get("launch_spec_sha256"),
                "service_receipt_sha256": service.get("receipt_sha256"),
                "gpu": service.get("gpu"),
            }
            for field, expected in bindings.items():
                if validated.get(field) != expected:
                    raise LabError(f"execution sandbox binding mismatch: {field}")
            existing = current.get("execution_sandbox_receipt")
            if existing is not None:
                if existing == validated:
                    return current
                raise LabError("request already has a different execution sandbox")
            updated = {**current, "execution_sandbox_receipt": validated}
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def attach_network_namespace_evidence(
        self,
        request_id: str,
        *,
        anchor_observation: dict[str, object],
        probe_receipt: dict[str, object],
        host_profile: HostProfile,
        policy: NetworkNamespaceProbePolicy,
    ) -> dict[str, object]:
        """Persist one plan-bound anchor/probe pair without granting launch."""

        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "PREPARING":
                raise LabError(
                    "network namespace evidence can only attach while PREPARING: "
                    f"{request_id}"
                )
            plan = current.get("network_namespace_plan")
            preflight = current.get("preflight_receipt")
            service_plan = current.get("service_launch_plan")
            if (
                not isinstance(plan, dict)
                or not isinstance(preflight, dict)
                or not isinstance(service_plan, dict)
            ):
                raise LabError(
                    "network namespace evidence requires preflight and service plan"
                )
            if (
                host_profile.host_id != preflight.get("host_id")
                or host_profile.sha256 != preflight.get("host_profile_sha256")
                or service_plan.get("network_namespace_plan_sha256")
                != plan.get("receipt_sha256")
            ):
                raise LabError(
                    "network namespace evidence host or service plan differs"
                )
            validated_anchor = validate_network_namespace_anchor_observation(
                anchor_observation,
                network_namespace_plan=plan,
                host_profile=host_profile,
            )
            validated_probe = validate_network_namespace_probe_receipt(
                probe_receipt,
                network_namespace_plan=plan,
                anchor_observation=validated_anchor,
                policy=policy,
            )
            existing_anchor = current.get(
                "network_namespace_anchor_observation"
            )
            existing_probe = current.get("network_namespace_probe_receipt")
            if existing_anchor is not None or existing_probe is not None:
                if (
                    existing_anchor == validated_anchor
                    and existing_probe == validated_probe
                ):
                    return current
                raise LabError(
                    "request already has different network namespace evidence"
                )
            updated = {
                **current,
                "network_namespace_anchor_observation": validated_anchor,
                "network_namespace_probe_receipt": validated_probe,
            }
            _atomic_write_text(
                self._path(request_id),
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
            )
            return updated

    def begin_attempt(
        self,
        request_id: str,
        start_receipt: dict[str, object],
    ) -> dict[str, object]:
        """Enter RUNNING only through a fully bound trusted attempt receipt."""

        with self._lock("request-store"):
            current = self.load(request_id)
            validated = validate_attempt_start(start_receipt)
            if current.get("state") == "RUNNING":
                if current.get("active_attempt") == validated:
                    return current
                raise LabError("request is running a different attempt")
            if current.get("state") != "PREPARING":
                raise LabError(f"attempt can only begin while PREPARING: {request_id}")
            service = current.get("service_receipt")
            if not isinstance(service, dict):
                raise LabError("attempt requires a verified service receipt")
            sandbox = current.get("execution_sandbox_receipt")
            if not isinstance(sandbox, dict):
                raise LabError("attempt requires an enforced execution sandbox")
            if (
                not isinstance(
                    current.get("network_namespace_anchor_observation"), dict
                )
                or not isinstance(
                    current.get("network_namespace_probe_receipt"), dict
                )
            ):
                raise LabError("attempt requires network namespace evidence")
            attempts = current.get("attempts")
            if not isinstance(attempts, list):
                raise LabError("request attempts state is invalid")
            bindings = {
                "request_id": request_id,
                "idempotency_key": current.get("idempotency_key"),
                "task_id": current.get("task_id"),
                "slot_id": current.get("slot_id"),
                "remote_profile_sha256": service.get("remote_profile_sha256"),
                "service_receipt_sha256": service.get("receipt_sha256"),
                "execution_sandbox_receipt_sha256": sandbox.get("receipt_sha256"),
                "gpu": service.get("gpu"),
                "attempt_number": len(attempts) + 1,
            }
            for field, expected in bindings.items():
                if validated.get(field) != expected:
                    raise LabError(f"attempt start binding mismatch: {field}")
            return self._transition_unlocked(
                current,
                "RUNNING",
                active_attempt=validated,
            )

    def finish_attempt(
        self,
        request_id: str,
        result_receipt: dict[str, object],
    ) -> dict[str, object]:
        """Record immutable attempt evidence and reach a truthful terminal state."""

        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") != "RUNNING":
                raise LabError(f"attempt can only finish while RUNNING: {request_id}")
            start = current.get("active_attempt")
            if not isinstance(start, dict):
                raise LabError("running request has no active attempt receipt")
            result = validate_attempt_result(result_receipt, start_receipt=start)
            attempts = current.get("attempts")
            if not isinstance(attempts, list):
                raise LabError("request attempts state is invalid")
            collecting = self._transition_unlocked(current, "COLLECTING")
            outcome = {
                "attempt_id": result.get("attempt_id"),
                "classification": result.get("classification"),
                "task_success": result.get("task_success"),
                "evidence_complete": result.get("evidence_complete"),
            }
            updates: dict[str, object] = {
                "active_attempt": None,
                "attempts": [*attempts, {"start": start, "result": result}],
                "outcome": outcome,
            }
            if result.get("classification") == "PASSED":
                return self._transition_unlocked(
                    collecting,
                    "COMPLETED",
                    **updates,
                )
            classification = str(result.get("classification"))
            return self._transition_unlocked(
                collecting,
                "FAILED",
                **updates,
                failure={
                    "classification": classification,
                    "detail": (
                        "trusted command service was not acknowledged"
                        if classification == "INFRASTRUCTURE_FAILURE"
                        else "trusted command assertions did not pass"
                    ),
                },
            )

    def fail(
        self,
        request_id: str,
        *,
        classification: str,
        detail: str,
    ) -> dict[str, object]:
        if classification not in FAILURE_CLASSIFICATIONS:
            raise LabError(f"invalid request failure classification: {classification}")
        normalized_detail = " ".join(detail.split())[:1000]
        if not normalized_detail:
            raise LabError("request failure receipt requires a detail")
        with self._lock("request-store"):
            current = self.load(request_id)
            if current.get("state") == "FAILED":
                failure = current.get("failure")
                if failure == {
                    "classification": classification,
                    "detail": normalized_detail,
                }:
                    return current
                raise LabError("request already failed with a different receipt")
            return self._transition_unlocked(
                current,
                "FAILED",
                failure={
                    "classification": classification,
                    "detail": normalized_detail,
                },
            )

    def request_cancel(self, request_id: str) -> dict[str, object]:
        """Request cancellation without claiming teardown or hardware safety."""

        with self._lock("request-store"):
            current = self.load(request_id)
            state = str(current.get("state"))
            if state in TERMINAL_STATES or state == "CANCEL_REQUESTED":
                return current
            if state == "SUBMITTED":
                return self._transition_unlocked(current, "CANCELLED")
            if state in ACTIVE_STATES:
                return self._transition_unlocked(current, "CANCEL_REQUESTED")
            raise LabError(
                f"request cannot be cancelled from state {state}: {request_id}"
            )

    def _transition_unlocked(
        self,
        current: dict[str, object],
        new_state: str,
        **updates: object,
    ) -> dict[str, object]:
        old_state = str(current.get("state"))
        if new_state not in TRANSITIONS.get(old_state, frozenset()):
            raise LabError(f"invalid request transition: {old_state} -> {new_state}")
        history = current.get("history")
        if not isinstance(history, list):
            raise LabError("request state has invalid history")
        updated = {
            **current,
            **updates,
            "state": new_state,
            "history": [
                *history,
                {"sequence": len(history), "state": new_state},
            ],
        }
        _atomic_write_text(
            self._path(str(current["request_id"])),
            json.dumps(updated, indent=2, sort_keys=True) + "\n",
        )
        return updated
