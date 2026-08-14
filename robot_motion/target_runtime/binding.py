"""Offline binding of application requirements to a target runtime manifest."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionObservationBinding,
    ExecutionObservationPlan,
    bind_execution_observation_plan,
)
from robot_motion.runtime.dispatch_owner import (
    DispatchPermit,
    LifecycleControlDispatchPermit,
    SafetyControlDispatchPermit,
)

from .contracts import (
    TARGET_RUNTIME_PROVIDER_API_V1,
    ResolvedTargetRuntimeManifest,
    TargetRuntimeOperationCapability,
    TargetRuntimeOperationReceipt,
    TargetRuntimeOperationRequest,
    TargetRuntimeStatusSnapshot,
)

TARGET_RUNTIME_REQUIREMENT_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_requirement.v1"
)
TARGET_RUNTIME_BINDING_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_binding_receipt.v1"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


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


@dataclass(frozen=True)
class TargetRuntimeRequirement:
    """Application demand resolved before a provider opens any device."""

    requirement_id: str
    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    embodiment_instance_id: str
    target_handler_id: str
    action_schema_hash: str
    required_component_ids: tuple[str, ...]
    required_operation_ids: tuple[str, ...] = ()
    required_provider_api: str = TARGET_RUNTIME_PROVIDER_API_V1
    observation_plan: ExecutionObservationPlan | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload: dict[str, object] = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "requirement_id",
                "runtime_session_id",
                "control_domain_id",
                "clock_domain_id",
                "embodiment_instance_id",
                "target_handler_id",
                "action_schema_hash",
                "required_provider_api",
            )
        }
        payload["required_component_ids"] = _unique_identifiers(
            tuple(self.required_component_ids),
            name="required_component_id",
            allow_empty=False,
        )
        payload["required_operation_ids"] = tuple(
            sorted(
                _unique_identifiers(
                    tuple(self.required_operation_ids),
                    name="required_operation_id",
                )
            )
        )
        observation_plan = self.observation_plan
        if observation_plan is not None and not isinstance(
            observation_plan,
            ExecutionObservationPlan,
        ):
            raise TypeError(
                "observation_plan must be ExecutionObservationPlan or None."
            )
        payload["observation_plan_hash"] = (
            None
            if observation_plan is None
            else observation_plan.semantic_content_hash
        )
        for name, value in payload.items():
            if name != "observation_plan_hash":
                object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_REQUIREMENT_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class TargetRuntimeBindingReceipt:
    """Proof that one exact manifest satisfies one application requirement.

    This is a build/preflight receipt only.  It does not prove that assets are
    locally present, a device opened, state is fresh, or motion is authorized.
    """

    requirement_id: str
    requirement_hash: str
    manifest_id: str
    manifest_hash: str
    provider_identity_hash: str
    component_target_capability_hash: str
    target_handler_id: str
    embodiment_instance_id: str
    bound_component_ids: tuple[str, ...]
    bound_operation_ids: tuple[str, ...]
    execution_observation_binding: ExecutionObservationBinding | None
    bound_at: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        component_ids = _unique_identifiers(
            tuple(self.bound_component_ids),
            name="bound_component_id",
            allow_empty=False,
        )
        operation_ids = tuple(
            sorted(
                _unique_identifiers(
                    tuple(self.bound_operation_ids),
                    name="bound_operation_id",
                )
            )
        )
        observation = self.execution_observation_binding
        if observation is not None and not isinstance(
            observation,
            ExecutionObservationBinding,
        ):
            raise TypeError(
                "execution_observation_binding must be "
                "ExecutionObservationBinding or None."
            )
        bound_at = float(self.bound_at)
        if not math.isfinite(bound_at):
            raise ValueError("bound_at must be finite.")
        payload = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "requirement_id",
                "requirement_hash",
                "manifest_id",
                "manifest_hash",
                "provider_identity_hash",
                "component_target_capability_hash",
                "target_handler_id",
                "embodiment_instance_id",
            )
        }
        payload["bound_component_ids"] = component_ids
        payload["bound_operation_ids"] = operation_ids
        payload["execution_observation_binding_hash"] = (
            None if observation is None else observation.semantic_content_hash
        )
        payload["bound_at"] = bound_at
        for name, value in payload.items():
            if name != "execution_observation_binding_hash":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "bound_component_ids", component_ids)
        object.__setattr__(self, "bound_operation_ids", operation_ids)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_BINDING_RECEIPT_HASH_DOMAIN_V1,
            ),
        )


def bind_target_runtime_requirement(
    requirement: TargetRuntimeRequirement,
    manifest: ResolvedTargetRuntimeManifest,
    *,
    bound_at: float,
) -> TargetRuntimeBindingReceipt:
    """Fail closed unless one resolved target satisfies the entire demand."""

    if not isinstance(requirement, TargetRuntimeRequirement):
        raise TypeError("requirement must be TargetRuntimeRequirement.")
    if not isinstance(manifest, ResolvedTargetRuntimeManifest):
        raise TypeError("manifest must be ResolvedTargetRuntimeManifest.")

    exact_fields = (
        "runtime_session_id",
        "control_domain_id",
        "clock_domain_id",
        "embodiment_instance_id",
        "target_handler_id",
    )
    mismatches = tuple(
        name
        for name in exact_fields
        if getattr(requirement, name) != getattr(manifest, name)
    )
    if mismatches:
        raise ValueError(
            "Target runtime requirement/manifest identity mismatch: "
            + ",".join(mismatches)
            + "."
        )
    if requirement.required_provider_api != manifest.provider.provider_api:
        raise ValueError("Target runtime provider API is incompatible.")
    if requirement.action_schema_hash != manifest.action_schema_hash:
        raise ValueError("Target runtime action schema is incompatible.")

    unknown_components = sorted(
        set(requirement.required_component_ids) - set(manifest.component_ids)
    )
    if unknown_components:
        raise ValueError(
            "Target runtime does not provide required components: "
            f"{unknown_components!r}."
        )
    unknown_operations = sorted(
        set(requirement.required_operation_ids) - set(manifest.operation_by_id)
    )
    if unknown_operations:
        raise ValueError(
            "Target runtime does not provide required operations: "
            f"{unknown_operations!r}."
        )

    observation_binding: ExecutionObservationBinding | None = None
    if (
        requirement.observation_plan is not None
        and requirement.observation_plan.enabled
    ):
        capability = manifest.execution_observation_capability
        if capability is None:
            raise ValueError(
                "Target runtime does not provide execution observations."
            )
        observation_binding = bind_execution_observation_plan(
            requirement.observation_plan,
            capability,
            bound_at=bound_at,
        )

    return TargetRuntimeBindingReceipt(
        requirement_id=requirement.requirement_id,
        requirement_hash=requirement.semantic_content_hash,
        manifest_id=manifest.manifest_id,
        manifest_hash=manifest.semantic_content_hash,
        provider_identity_hash=manifest.provider.semantic_content_hash,
        component_target_capability_hash=(
            manifest.component_target_capability.semantic_content_hash
        ),
        target_handler_id=manifest.target_handler_id,
        embodiment_instance_id=manifest.embodiment_instance_id,
        bound_component_ids=requirement.required_component_ids,
        bound_operation_ids=requirement.required_operation_ids,
        execution_observation_binding=observation_binding,
        bound_at=bound_at,
    )


def validate_target_runtime_operation_invocation(
    manifest: ResolvedTargetRuntimeManifest,
    status: TargetRuntimeStatusSnapshot,
    request: TargetRuntimeOperationRequest,
    *,
    dispatch_permit: DispatchPermit | None,
) -> TargetRuntimeOperationCapability:
    """Validate structural operation binding before provider-private work.

    This function validates identity, schema, lifecycle, epoch, confirmation,
    and permit shape.  It deliberately does not validate or consume a permit;
    the authoritative :class:`DispatchOwnerRegistry` remains the only owner of
    that operation.
    """

    if not isinstance(manifest, ResolvedTargetRuntimeManifest):
        raise TypeError("manifest must be ResolvedTargetRuntimeManifest.")
    if not isinstance(status, TargetRuntimeStatusSnapshot):
        raise TypeError("status must be TargetRuntimeStatusSnapshot.")
    if not isinstance(request, TargetRuntimeOperationRequest):
        raise TypeError("request must be TargetRuntimeOperationRequest.")
    if (
        status.manifest_hash != manifest.semantic_content_hash
        or status.runtime_session_id != manifest.runtime_session_id
        or status.control_domain_id != manifest.control_domain_id
        or status.target_handler_id != manifest.target_handler_id
        or status.clock_domain_id != manifest.clock_domain_id
    ):
        raise ValueError("Target runtime status belongs to another manifest.")
    if (
        request.manifest_hash != manifest.semantic_content_hash
        or request.clock_domain_id != manifest.clock_domain_id
    ):
        raise ValueError("Target runtime operation belongs to another manifest.")
    capability = manifest.operation_by_id.get(request.operation_id)
    if capability is None:
        raise ValueError("Target runtime operation is not declared by the manifest.")
    if request.request_schema_hash != capability.request_schema_hash:
        raise ValueError("Target runtime operation request schema is incompatible.")
    if status.lifecycle_state not in capability.allowed_lifecycle_states:
        raise ValueError(
            "Target runtime operation is not allowed in the current lifecycle state."
        )
    if request.target_execution_epoch != status.execution_epoch:
        raise ValueError("Target runtime operation uses a stale execution epoch.")
    if (
        capability.requires_operator_confirmation
        and request.operator_confirmation_event_id is None
    ):
        raise ValueError("Target runtime operation requires operator confirmation.")

    required_kind = capability.required_permit_kind
    if required_kind == "none":
        if dispatch_permit is not None:
            raise ValueError(
                "A permit was supplied for an operation that declares none."
            )
        return capability
    if dispatch_permit is None:
        raise ValueError("Target runtime operation requires a dispatch permit.")
    expected_type = (
        SafetyControlDispatchPermit
        if required_kind == "safety_control"
        else LifecycleControlDispatchPermit
    )
    if not isinstance(dispatch_permit, expected_type):
        raise TypeError("Target runtime operation uses the wrong permit kind.")
    if (
        dispatch_permit.runtime_session_id != manifest.runtime_session_id
        or dispatch_permit.control_domain_id != manifest.control_domain_id
        or dispatch_permit.operation != capability.required_permit_operation
        or dispatch_permit.target_execution_epoch
        != request.target_execution_epoch
    ):
        raise ValueError("Target runtime operation permit identity is incompatible.")
    if (
        capability.requires_operator_confirmation
        and isinstance(dispatch_permit, LifecycleControlDispatchPermit)
        and dispatch_permit.authorization_event_id
        != request.operator_confirmation_event_id
    ):
        raise ValueError(
            "Target runtime lifecycle permit uses another operator confirmation."
        )
    return capability


def validate_target_runtime_operation_receipt(
    manifest: ResolvedTargetRuntimeManifest,
    request: TargetRuntimeOperationRequest,
    receipt: TargetRuntimeOperationReceipt,
) -> None:
    """Reject a terminal operation receipt from another request or schema."""

    if not isinstance(manifest, ResolvedTargetRuntimeManifest):
        raise TypeError("manifest must be ResolvedTargetRuntimeManifest.")
    if not isinstance(request, TargetRuntimeOperationRequest):
        raise TypeError("request must be TargetRuntimeOperationRequest.")
    if not isinstance(receipt, TargetRuntimeOperationReceipt):
        raise TypeError("receipt must be TargetRuntimeOperationReceipt.")
    capability = manifest.operation_by_id.get(request.operation_id)
    if capability is None:
        raise ValueError("Target runtime operation is not declared by the manifest.")
    if (
        request.manifest_hash != manifest.semantic_content_hash
        or receipt.manifest_hash != manifest.semantic_content_hash
        or receipt.request_id != request.request_id
        or receipt.request_hash != request.semantic_content_hash
        or receipt.operation_id != request.operation_id
        or receipt.receipt_schema_hash != capability.receipt_schema_hash
        or receipt.target_execution_epoch != request.target_execution_epoch
        or receipt.clock_domain_id != request.clock_domain_id
    ):
        raise ValueError("Target runtime operation receipt binding is incompatible.")
    if receipt.completed_at < request.requested_at:
        raise ValueError("Target runtime operation completed before its request.")


__all__ = [
    "TARGET_RUNTIME_BINDING_RECEIPT_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_REQUIREMENT_HASH_DOMAIN_V1",
    "TargetRuntimeBindingReceipt",
    "TargetRuntimeRequirement",
    "bind_target_runtime_requirement",
    "validate_target_runtime_operation_invocation",
    "validate_target_runtime_operation_receipt",
]
