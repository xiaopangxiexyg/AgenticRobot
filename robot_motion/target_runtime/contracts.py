"""Backend-neutral contracts for one resolved embodiment target runtime.

The target runtime is the lower-owned unit behind teleoperation, collection,
policy deployment, or another action source.  It binds an exact embodiment
instance and provider implementation to the existing component action,
execution, and observation contracts without exposing a vendor SDK or a
backend object to those applications.

These values do not create a device connection, grant authority, authorize
motion, or claim that a target has started.  Concrete real and simulated
providers remain in their owning packages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_handoff import (
    TargetExecutionPort,
    TargetExecutionStatusProvider,
)
from robot_motion.action.execution_observation import (
    ExecutionObservationCapabilitySpec,
    ExecutionObservationProvider,
)
from robot_motion.action.target_capability import ComponentTargetCapabilitySpec
from robot_motion.runtime.dispatch_owner import (
    LIFECYCLE_CONTROL_OPERATIONS,
    SAFETY_CONTROL_OPERATIONS,
    DispatchPermit,
)
from robot_motion.runtime.immutable import deep_freeze

TARGET_RUNTIME_PROVIDER_API_V1 = "dexteleop.target_runtime_provider.v1"
TARGET_RUNTIME_PROVIDER_IDENTITY_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_provider_identity.v1"
)
TARGET_RUNTIME_ASSET_REFERENCE_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_asset_reference.v1"
)
TARGET_RUNTIME_OPERATION_CAPABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_operation_capability.v1"
)
RESOLVED_TARGET_RUNTIME_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.resolved_target_runtime_manifest.v1"
)
TARGET_RUNTIME_STATUS_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_status_snapshot.v1"
)
TARGET_RUNTIME_OPERATION_REQUEST_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_operation_request.v1"
)
TARGET_RUNTIME_OPERATION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_operation_receipt.v1"
)

TARGET_RUNTIME_LIFECYCLE_STATES_V1 = frozenset(
    {
        "created",
        "starting",
        "ready",
        "holding",
        "faulted",
        "closing",
        "closed",
    }
)
TARGET_RUNTIME_OPERATION_MOTION_EFFECTS_V1 = frozenset(
    {"none", "possible", "required"}
)
TARGET_RUNTIME_OPERATION_PERMIT_KINDS_V1 = frozenset(
    {"none", "safety_control", "lifecycle_control"}
)
TARGET_RUNTIME_OPERATION_RECEIPT_STATUSES_V1 = frozenset(
    {"completed", "rejected", "failed"}
)
TARGET_RUNTIME_PROVIDER_HANDLE_MODES_V1 = frozenset(
    {"preflight_only", "inert_handle"}
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _optional(value: object | None, *, name: str) -> str | None:
    if value is None:
        return None
    return _required(value, name=name)


def _unique_identifiers(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return normalized


def _non_negative(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class TargetRuntimeProviderIdentity:
    """Portable identity of the code/environment provider for one target.

    ``provider_id`` is resolved through an allowlisted registry by an
    application package.  It is not a Python import string.  Dependency and
    implementation hashes let a resolved session distinguish two providers
    with the same public API without placing SDK details in the shared core.
    """

    provider_id: str
    provider_version: str
    provider_api: str
    implementation_contract_hash: str
    package_name: str
    package_version: str
    runtime_handle_mode: str
    dependency_lock_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "provider_id",
                "provider_version",
                "provider_api",
                "implementation_contract_hash",
                "package_name",
                "package_version",
                "runtime_handle_mode",
            )
        }
        if (
            payload["runtime_handle_mode"]
            not in TARGET_RUNTIME_PROVIDER_HANDLE_MODES_V1
        ):
            raise ValueError("Unsupported target-runtime provider handle mode.")
        payload["dependency_lock_hash"] = _optional(
            self.dependency_lock_hash,
            name="dependency_lock_hash",
        )
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_PROVIDER_IDENTITY_HASH_DOMAIN_V1,
            ),
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "provider_api": self.provider_api,
            "implementation_contract_hash": self.implementation_contract_hash,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "runtime_handle_mode": self.runtime_handle_mode,
            "dependency_lock_hash": self.dependency_lock_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TargetRuntimeAssetReference:
    """Content identity for an asset required by one resolved target.

    Paths and download locations are intentionally absent.  A provider may
    resolve storage locally, while the manifest continues to bind the exact
    URDF, mesh, tool, controller, calibration schema, or other artifact used.
    """

    asset_id: str
    asset_kind: str
    content_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            name: _required(getattr(self, name), name=name)
            for name in ("asset_id", "asset_kind", "content_hash")
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_ASSET_REFERENCE_HASH_DOMAIN_V1,
            ),
        )

    def semantic_payload(self) -> dict[str, str]:
        return {
            "asset_id": self.asset_id,
            "asset_kind": self.asset_kind,
            "content_hash": self.content_hash,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TargetRuntimeOperationCapability:
    """One provider-owned lifecycle or device operation.

    The shared core does not assign semantics to operation IDs such as home,
    clear-fault, tool configuration, or quick-stop.  Exact request/receipt
    schemas and lifecycle admission remain provider capabilities.  Motion and
    confirmation declarations make dangerous operations visible before a
    device is opened.
    """

    operation_id: str
    operation_version: str
    request_schema_hash: str
    receipt_schema_hash: str
    allowed_lifecycle_states: tuple[str, ...]
    motion_effect: str
    requires_operator_confirmation: bool
    required_permit_kind: str
    required_permit_operation: str | None
    resource_claims: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        operation_id = _required(self.operation_id, name="operation_id")
        operation_version = _required(
            self.operation_version,
            name="operation_version",
        )
        request_schema_hash = _required(
            self.request_schema_hash,
            name="request_schema_hash",
        )
        receipt_schema_hash = _required(
            self.receipt_schema_hash,
            name="receipt_schema_hash",
        )
        states = tuple(
            sorted(
                _unique_identifiers(
                    tuple(self.allowed_lifecycle_states),
                    name="allowed_lifecycle_state",
                    allow_empty=False,
                )
            )
        )
        unsupported_states = sorted(
            set(states) - TARGET_RUNTIME_LIFECYCLE_STATES_V1
        )
        if unsupported_states:
            raise ValueError(
                "Operation capability contains unsupported lifecycle states: "
                f"{unsupported_states!r}."
            )
        motion_effect = _required(self.motion_effect, name="motion_effect")
        if motion_effect not in TARGET_RUNTIME_OPERATION_MOTION_EFFECTS_V1:
            raise ValueError("Unsupported target-runtime motion effect.")
        permit_kind = _required(
            self.required_permit_kind,
            name="required_permit_kind",
        )
        if permit_kind not in TARGET_RUNTIME_OPERATION_PERMIT_KINDS_V1:
            raise ValueError("Unsupported target-runtime operation permit kind.")
        permit_operation = _optional(
            self.required_permit_operation,
            name="required_permit_operation",
        )
        if permit_kind == "none":
            if permit_operation is not None:
                raise ValueError(
                    "An operation without a permit cannot name a permit operation."
                )
            if motion_effect != "none":
                raise ValueError(
                    "An operation with a possible motion effect requires a permit."
                )
        else:
            if permit_operation is None:
                raise ValueError(
                    "A permitted operation must name the canonical permit operation."
                )
            supported_permit_operations = (
                SAFETY_CONTROL_OPERATIONS
                if permit_kind == "safety_control"
                else LIFECYCLE_CONTROL_OPERATIONS
            )
            if permit_operation not in supported_permit_operations:
                raise ValueError(
                    "Operation capability uses an unsupported canonical permit "
                    "operation."
                )
        resources = tuple(
            sorted(
                _unique_identifiers(
                    tuple(self.resource_claims),
                    name="resource_claim",
                )
            )
        )
        payload = {
            "operation_id": operation_id,
            "operation_version": operation_version,
            "request_schema_hash": request_schema_hash,
            "receipt_schema_hash": receipt_schema_hash,
            "allowed_lifecycle_states": states,
            "motion_effect": motion_effect,
            "requires_operator_confirmation": bool(
                self.requires_operator_confirmation
            ),
            "required_permit_kind": permit_kind,
            "required_permit_operation": permit_operation,
            "resource_claims": resources,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_OPERATION_CAPABILITY_HASH_DOMAIN_V1,
            ),
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "operation_version": self.operation_version,
            "request_schema_hash": self.request_schema_hash,
            "receipt_schema_hash": self.receipt_schema_hash,
            "allowed_lifecycle_states": self.allowed_lifecycle_states,
            "motion_effect": self.motion_effect,
            "requires_operator_confirmation": (
                self.requires_operator_confirmation
            ),
            "required_permit_kind": self.required_permit_kind,
            "required_permit_operation": self.required_permit_operation,
            "resource_claims": self.resource_claims,
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class ResolvedTargetRuntimeManifest:
    """Exact build-time identity for one target runtime session.

    This is the composition boundary between an application and a provider.
    It references the existing component target and optional execution
    observation capabilities instead of redefining their semantics.
    """

    manifest_id: str
    manifest_version: str
    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    embodiment_definition_id: str
    embodiment_definition_hash: str
    embodiment_instance_id: str
    embodiment_instance_hash: str
    target_handler_id: str
    provider: TargetRuntimeProviderIdentity
    component_target_capability: ComponentTargetCapabilitySpec
    execution_observation_capability: (
        ExecutionObservationCapabilitySpec | None
    ) = None
    operation_capabilities: tuple[TargetRuntimeOperationCapability, ...] = ()
    asset_references: tuple[TargetRuntimeAssetReference, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "manifest_id",
                "manifest_version",
                "runtime_session_id",
                "control_domain_id",
                "clock_domain_id",
                "embodiment_definition_id",
                "embodiment_definition_hash",
                "embodiment_instance_id",
                "embodiment_instance_hash",
                "target_handler_id",
            )
        }
        if not isinstance(self.provider, TargetRuntimeProviderIdentity):
            raise TypeError("provider must be TargetRuntimeProviderIdentity.")
        if not isinstance(
            self.component_target_capability,
            ComponentTargetCapabilitySpec,
        ):
            raise TypeError(
                "component_target_capability must be "
                "ComponentTargetCapabilitySpec."
            )
        observation = self.execution_observation_capability
        if observation is not None:
            if not isinstance(observation, ExecutionObservationCapabilitySpec):
                raise TypeError(
                    "execution_observation_capability must be "
                    "ExecutionObservationCapabilitySpec or None."
                )
            if observation.target_handler_id != identifiers["target_handler_id"]:
                raise ValueError(
                    "Execution observation capability belongs to another "
                    "target handler."
                )
            if observation.clock_domain_id != identifiers["clock_domain_id"]:
                raise ValueError(
                    "Execution observation capability uses another clock domain."
                )

        operations = tuple(self.operation_capabilities)
        if any(
            not isinstance(item, TargetRuntimeOperationCapability)
            for item in operations
        ):
            raise TypeError(
                "operation_capabilities must contain "
                "TargetRuntimeOperationCapability values."
            )
        operation_ids = tuple(item.operation_id for item in operations)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("Target runtime operation IDs must be unique.")
        ordered_operations = tuple(
            sorted(operations, key=lambda item: item.operation_id)
        )

        assets = tuple(self.asset_references)
        if any(not isinstance(item, TargetRuntimeAssetReference) for item in assets):
            raise TypeError(
                "asset_references must contain TargetRuntimeAssetReference values."
            )
        asset_ids = tuple(item.asset_id for item in assets)
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("Target runtime asset IDs must be unique.")
        ordered_assets = tuple(sorted(assets, key=lambda item: item.asset_id))

        payload = {
            **identifiers,
            "provider_identity_hash": self.provider.semantic_content_hash,
            "component_target_capability_hash": (
                self.component_target_capability.semantic_content_hash
            ),
            "execution_observation_capability_hash": (
                None
                if observation is None
                else observation.semantic_content_hash
            ),
            "operation_capability_hashes": tuple(
                item.semantic_content_hash for item in ordered_operations
            ),
            "asset_reference_hashes": tuple(
                item.semantic_content_hash for item in ordered_assets
            ),
        }
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "operation_capabilities", ordered_operations)
        object.__setattr__(self, "asset_references", ordered_assets)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=RESOLVED_TARGET_RUNTIME_MANIFEST_HASH_DOMAIN_V1,
            ),
        )

    @property
    def action_schema_hash(self) -> str:
        return self.component_target_capability.action_schema_hash

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.component_target_capability.action_schema.component_ids

    @property
    def operation_by_id(
        self,
    ) -> Mapping[str, TargetRuntimeOperationCapability]:
        return MappingProxyType(
            {item.operation_id: item for item in self.operation_capabilities}
        )

    @property
    def asset_by_id(self) -> Mapping[str, TargetRuntimeAssetReference]:
        return MappingProxyType(
            {item.asset_id: item for item in self.asset_references}
        )

    def semantic_payload(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_version": self.manifest_version,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "clock_domain_id": self.clock_domain_id,
            "embodiment_definition_id": self.embodiment_definition_id,
            "embodiment_definition_hash": self.embodiment_definition_hash,
            "embodiment_instance_id": self.embodiment_instance_id,
            "embodiment_instance_hash": self.embodiment_instance_hash,
            "target_handler_id": self.target_handler_id,
            "provider": self.provider.semantic_payload(),
            "component_target_capability_hash": (
                self.component_target_capability.semantic_content_hash
            ),
            "execution_observation_capability_hash": (
                None
                if self.execution_observation_capability is None
                else self.execution_observation_capability.semantic_content_hash
            ),
            "operation_capabilities": tuple(
                item.semantic_payload() for item in self.operation_capabilities
            ),
            "asset_references": tuple(
                item.semantic_payload() for item in self.asset_references
            ),
            "semantic_content_hash": self.semantic_content_hash,
        }


@dataclass(frozen=True)
class TargetRuntimeStatusSnapshot:
    """Read-only target lifecycle/health observation.

    The execution epoch is observed from the authoritative execution owner;
    this snapshot does not mint or advance it.
    """

    manifest_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_handler_id: str
    lifecycle_state: str
    execution_epoch: int
    execution_epoch_source_id: str
    ready: bool
    fault_latched: bool
    status_sequence: int
    captured_at: float
    clock_domain_id: str
    reason: str
    source_event_ids: tuple[str, ...] = ()
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        lifecycle_state = _required(
            self.lifecycle_state,
            name="lifecycle_state",
        )
        if lifecycle_state not in TARGET_RUNTIME_LIFECYCLE_STATES_V1:
            raise ValueError("Unsupported target-runtime lifecycle state.")
        ready = bool(self.ready)
        if ready != (lifecycle_state == "ready"):
            raise ValueError(
                "ready must be true exactly for the ready lifecycle state."
            )
        fault_latched = bool(self.fault_latched)
        if lifecycle_state == "faulted" and not fault_latched:
            raise ValueError("The faulted lifecycle state must latch a fault.")
        payload = {
            "manifest_hash": _required(
                self.manifest_hash,
                name="manifest_hash",
            ),
            "runtime_session_id": _required(
                self.runtime_session_id,
                name="runtime_session_id",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "target_handler_id": _required(
                self.target_handler_id,
                name="target_handler_id",
            ),
            "lifecycle_state": lifecycle_state,
            "execution_epoch": _non_negative(
                self.execution_epoch,
                name="execution_epoch",
            ),
            "execution_epoch_source_id": _required(
                self.execution_epoch_source_id,
                name="execution_epoch_source_id",
            ),
            "ready": ready,
            "fault_latched": fault_latched,
            "status_sequence": _non_negative(
                self.status_sequence,
                name="status_sequence",
            ),
            "captured_at": _finite(self.captured_at, name="captured_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "reason": _required(self.reason, name="reason"),
            "source_event_ids": _unique_identifiers(
                tuple(self.source_event_ids),
                name="source_event_id",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_STATUS_SNAPSHOT_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class TargetRuntimeOperationRequest:
    """Schema-bound request for one declared provider operation."""

    request_id: str
    manifest_hash: str
    operation_id: str
    request_schema_hash: str
    payload: Mapping[str, Any]
    target_execution_epoch: int
    requested_at: float
    clock_domain_id: str
    source_event_ids: tuple[str, ...]
    operator_confirmation_event_id: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload_value = deep_freeze(dict(self.payload))
        payload = {
            "request_id": _required(self.request_id, name="request_id"),
            "manifest_hash": _required(
                self.manifest_hash,
                name="manifest_hash",
            ),
            "operation_id": _required(
                self.operation_id,
                name="operation_id",
            ),
            "request_schema_hash": _required(
                self.request_schema_hash,
                name="request_schema_hash",
            ),
            "payload": payload_value,
            "target_execution_epoch": _non_negative(
                self.target_execution_epoch,
                name="target_execution_epoch",
            ),
            "requested_at": _finite(self.requested_at, name="requested_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "source_event_ids": _unique_identifiers(
                tuple(self.source_event_ids),
                name="source_event_id",
            ),
            "operator_confirmation_event_id": _optional(
                self.operator_confirmation_event_id,
                name="operator_confirmation_event_id",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_OPERATION_REQUEST_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class TargetRuntimeOperationReceipt:
    """Terminal result of one provider operation.

    This receipt does not replace a motion, safety-control, or lifecycle
    dispatch permit.  Provider-specific evidence remains referenced by hash.
    """

    request_id: str
    request_hash: str
    manifest_hash: str
    operation_id: str
    receipt_schema_hash: str
    target_execution_epoch: int
    status: str
    completed_at: float
    clock_domain_id: str
    evidence_refs: tuple[str, ...]
    result_payload: Mapping[str, Any] = field(default_factory=dict)
    receipt_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = _required(self.status, name="status")
        if status not in TARGET_RUNTIME_OPERATION_RECEIPT_STATUSES_V1:
            raise ValueError("Unsupported target-runtime operation status.")
        payload = {
            "request_id": _required(self.request_id, name="request_id"),
            "request_hash": _required(self.request_hash, name="request_hash"),
            "manifest_hash": _required(
                self.manifest_hash,
                name="manifest_hash",
            ),
            "operation_id": _required(
                self.operation_id,
                name="operation_id",
            ),
            "receipt_schema_hash": _required(
                self.receipt_schema_hash,
                name="receipt_schema_hash",
            ),
            "target_execution_epoch": _non_negative(
                self.target_execution_epoch,
                name="target_execution_epoch",
            ),
            "status": status,
            "completed_at": _finite(self.completed_at, name="completed_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "evidence_refs": _unique_identifiers(
                tuple(self.evidence_refs),
                name="evidence_ref",
            ),
            "result_payload": deep_freeze(dict(self.result_payload)),
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=TARGET_RUNTIME_OPERATION_RECEIPT_HASH_DOMAIN_V1,
        )
        for name, value in (
            *payload.items(),
            ("receipt_id", f"target-runtime-operation-receipt:{semantic_hash}"),
            ("semantic_content_hash", semantic_hash),
        ):
            object.__setattr__(self, name, value)


class TargetRuntimeStatusProvider(Protocol):
    """Read-only lifecycle and health surface for one resolved runtime."""

    @property
    def resolved_target_runtime_manifest(self) -> ResolvedTargetRuntimeManifest:
        ...

    def capture_target_runtime_status(self) -> TargetRuntimeStatusSnapshot:
        ...


class TargetRuntimeOperationPort(Protocol):
    """Provider-owned lifecycle/device operation surface.

    A required permit is issued and validated by the existing authoritative
    dispatch owner.  The operation port must not mint or advance epochs.
    """

    @property
    def resolved_target_runtime_manifest(self) -> ResolvedTargetRuntimeManifest:
        ...

    def execute_target_runtime_operation(
        self,
        request: TargetRuntimeOperationRequest,
        *,
        dispatch_permit: DispatchPermit | None,
    ) -> TargetRuntimeOperationReceipt:
        ...


class TargetRuntimeHandle(Protocol):
    """Application-facing aggregate of lower-owned target ports.

    The handle is a set of ports, not a second execution owner.  Every port is
    expected to refer to the same resolved manifest and concrete target.
    """

    @property
    def resolved_target_runtime_manifest(self) -> ResolvedTargetRuntimeManifest:
        ...

    @property
    def execution_port(self) -> TargetExecutionPort:
        ...

    @property
    def execution_status_provider(self) -> TargetExecutionStatusProvider:
        ...

    @property
    def execution_observation_provider(
        self,
    ) -> ExecutionObservationProvider | None:
        ...

    @property
    def target_runtime_status_provider(self) -> TargetRuntimeStatusProvider:
        ...

    @property
    def target_runtime_operation_port(self) -> TargetRuntimeOperationPort:
        ...


__all__ = [
    "RESOLVED_TARGET_RUNTIME_MANIFEST_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_ASSET_REFERENCE_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_LIFECYCLE_STATES_V1",
    "TARGET_RUNTIME_OPERATION_CAPABILITY_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_OPERATION_MOTION_EFFECTS_V1",
    "TARGET_RUNTIME_OPERATION_PERMIT_KINDS_V1",
    "TARGET_RUNTIME_OPERATION_RECEIPT_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_OPERATION_RECEIPT_STATUSES_V1",
    "TARGET_RUNTIME_OPERATION_REQUEST_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROVIDER_API_V1",
    "TARGET_RUNTIME_PROVIDER_HANDLE_MODES_V1",
    "TARGET_RUNTIME_PROVIDER_IDENTITY_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_STATUS_SNAPSHOT_HASH_DOMAIN_V1",
    "ResolvedTargetRuntimeManifest",
    "TargetRuntimeAssetReference",
    "TargetRuntimeHandle",
    "TargetRuntimeOperationCapability",
    "TargetRuntimeOperationPort",
    "TargetRuntimeOperationReceipt",
    "TargetRuntimeOperationRequest",
    "TargetRuntimeProviderIdentity",
    "TargetRuntimeStatusProvider",
    "TargetRuntimeStatusSnapshot",
]
