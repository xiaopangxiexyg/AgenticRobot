"""Pure manifests for auditing final control-dispatch call sites.

An inventory describes where a configured execution profile can reach a
device-facing call and which guard must eventually mediate that call.  It does
not install a wrapper, create an owner registry, mint a permit, or authorize
motion.  Those distinctions are deliberately represented in the assessment so
that M5.8-a code review evidence cannot be mistaken for M5.8-b integration.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash


DISPATCH_SEAM_INVENTORY_ENTRY_HASH_DOMAIN_V1 = (
    "dexteleop.dispatch_seam_inventory_entry.v1"
)
DISPATCH_SEAM_INVENTORY_MANIFEST_HASH_DOMAIN_V1 = (
    "dexteleop.dispatch_seam_inventory_manifest.v1"
)
DISPATCH_SEAM_INVENTORY_ASSESSMENT_HASH_DOMAIN_V1 = (
    "dexteleop.dispatch_seam_inventory_assessment.v1"
)

DISPATCH_INVENTORY_CALL_STAGES_V1 = frozenset(
    {
        "logical_target_admission",
        "worker_dequeue",
        "operation_admission",
        "pre_sdk_call",
        "disconnect_call",
    }
)
DISPATCH_INVENTORY_PATH_KINDS_V1 = frozenset(
    {
        "direct",
        "background_worker",
        "safety_control",
        "lifecycle_control",
        "cleanup",
    }
)
DISPATCH_INVENTORY_PERMIT_KINDS_V1 = frozenset(
    {"motion", "safety_control", "lifecycle_control", "not_applicable"}
)
DISPATCH_INVENTORY_AUTHORIZATION_REQUIREMENTS_V1 = frozenset(
    {
        "typed_context",
        "epoch_guard",
        "one_shot_dispatch_permit",
        "bounded_operation_then_one_shot",
        "lifecycle_state_machine_then_one_shot",
        "lifecycle_state_guard",
    }
)
DISPATCH_INVENTORY_INTEGRATION_STATUSES_V1 = frozenset(
    {
        "inventoried_unwrapped",
        "pure_wrapper_validated",
        "production_mediated",
    }
)
DISPATCH_INVENTORY_CHECKS_V1 = frozenset(
    {
        "typed_control_context",
        "owner_token",
        "owner_generation",
        "authority_epoch",
        "execution_epoch",
        "group_generation",
        "active_safety_operation",
        "one_shot_per_call",
        "lifecycle_authorization",
        "worker_drain",
    }
)


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _unique_identifiers(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} entries must be unique.")
    return normalized


@dataclass(frozen=True)
class DispatchSeamInventoryEntry:
    """One semantic use of one concrete implementation call site."""

    seam_id: str
    seam_role: str
    implementation_path: str
    call_stage: str
    path_kind: str
    permit_kind: str
    typed_operation: str | None
    authorization_requirement: str
    required_checks: tuple[str, ...]
    retry_behavior: str
    activation_condition: str
    enabled_profile_ids: tuple[str, ...]
    required_for_legacy_mediation: bool
    integration_status: str
    evidence_ids: tuple[str, ...]
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        seam_id: str,
        seam_role: str,
        implementation_path: str,
        call_stage: str,
        path_kind: str,
        permit_kind: str,
        typed_operation: str | None,
        authorization_requirement: str,
        required_checks: tuple[str, ...],
        retry_behavior: str,
        activation_condition: str,
        enabled_profile_ids: tuple[str, ...],
        required_for_legacy_mediation: bool = True,
        integration_status: str = "inventoried_unwrapped",
        evidence_ids: tuple[str, ...],
    ) -> "DispatchSeamInventoryEntry":
        normalized_stage = _identifier(call_stage, name="call_stage")
        normalized_path_kind = _identifier(path_kind, name="path_kind")
        normalized_permit_kind = _identifier(permit_kind, name="permit_kind")
        normalized_requirement = _identifier(
            authorization_requirement,
            name="authorization_requirement",
        )
        normalized_status = _identifier(
            integration_status,
            name="integration_status",
        )
        if normalized_stage not in DISPATCH_INVENTORY_CALL_STAGES_V1:
            raise ValueError("Unsupported dispatch inventory call stage.")
        if normalized_path_kind not in DISPATCH_INVENTORY_PATH_KINDS_V1:
            raise ValueError("Unsupported dispatch inventory path kind.")
        if normalized_permit_kind not in DISPATCH_INVENTORY_PERMIT_KINDS_V1:
            raise ValueError("Unsupported dispatch inventory permit kind.")
        if (
            normalized_requirement
            not in DISPATCH_INVENTORY_AUTHORIZATION_REQUIREMENTS_V1
        ):
            raise ValueError("Unsupported dispatch inventory authorization requirement.")
        if normalized_status not in DISPATCH_INVENTORY_INTEGRATION_STATUSES_V1:
            raise ValueError("Unsupported dispatch inventory integration status.")

        normalized_operation = (
            None
            if typed_operation is None
            else _identifier(typed_operation, name="typed_operation")
        )
        if normalized_permit_kind in {"safety_control", "lifecycle_control"}:
            if normalized_operation is None:
                raise ValueError("Safety/lifecycle inventory entries require an operation.")
        elif normalized_operation is not None:
            raise ValueError("Motion/not-applicable inventory entries cannot carry an operation.")

        normalized_checks = _unique_identifiers(
            required_checks,
            name="required_checks",
        )
        unsupported_checks = set(normalized_checks) - DISPATCH_INVENTORY_CHECKS_V1
        if unsupported_checks:
            raise ValueError(
                "Unsupported dispatch inventory checks: "
                + ", ".join(sorted(unsupported_checks))
            )
        payload = {
            "seam_id": _identifier(seam_id, name="seam_id"),
            "seam_role": _identifier(seam_role, name="seam_role"),
            "implementation_path": _identifier(
                implementation_path,
                name="implementation_path",
            ),
            "call_stage": normalized_stage,
            "path_kind": normalized_path_kind,
            "permit_kind": normalized_permit_kind,
            "typed_operation": normalized_operation,
            "authorization_requirement": normalized_requirement,
            "required_checks": normalized_checks,
            "retry_behavior": _identifier(retry_behavior, name="retry_behavior"),
            "activation_condition": _identifier(
                activation_condition,
                name="activation_condition",
            ),
            "enabled_profile_ids": _unique_identifiers(
                enabled_profile_ids,
                name="enabled_profile_ids",
            ),
            "required_for_legacy_mediation": bool(required_for_legacy_mediation),
            "integration_status": normalized_status,
            "evidence_ids": _unique_identifiers(evidence_ids, name="evidence_ids"),
        }
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=DISPATCH_SEAM_INVENTORY_ENTRY_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class DispatchSeamInventoryManifest:
    inventory_id: str
    inventory_version: str
    execution_manifest_id: str
    execution_manifest_hash: str
    entries: tuple[DispatchSeamInventoryEntry, ...]
    semantic_content_hash: str

    @classmethod
    def create(
        cls,
        *,
        inventory_id: str,
        inventory_version: str,
        execution_manifest_id: str,
        execution_manifest_hash: str,
        entries: tuple[DispatchSeamInventoryEntry, ...],
    ) -> "DispatchSeamInventoryManifest":
        normalized_entries = tuple(entries)
        if not normalized_entries:
            raise ValueError("Dispatch seam inventory must contain entries.")
        if any(not isinstance(item, DispatchSeamInventoryEntry) for item in normalized_entries):
            raise TypeError("entries must contain DispatchSeamInventoryEntry values.")
        seam_ids = tuple(item.seam_id for item in normalized_entries)
        if len(set(seam_ids)) != len(seam_ids):
            raise ValueError("Dispatch seam IDs must be unique.")
        sorted_entries = tuple(sorted(normalized_entries, key=lambda item: item.seam_id))
        normalized_inventory_id = _identifier(inventory_id, name="inventory_id")
        normalized_inventory_version = _identifier(
            inventory_version,
            name="inventory_version",
        )
        normalized_execution_manifest_id = _identifier(
            execution_manifest_id,
            name="execution_manifest_id",
        )
        normalized_execution_manifest_hash = _identifier(
            execution_manifest_hash,
            name="execution_manifest_hash",
        )
        hash_payload = {
            "inventory_id": normalized_inventory_id,
            "inventory_version": normalized_inventory_version,
            "execution_manifest_id": normalized_execution_manifest_id,
            "execution_manifest_hash": normalized_execution_manifest_hash,
            "entry_refs": tuple(
                {
                    "seam_id": entry.seam_id,
                    "semantic_content_hash": entry.semantic_content_hash,
                }
                for entry in sorted_entries
            ),
        }
        return cls(
            inventory_id=normalized_inventory_id,
            inventory_version=normalized_inventory_version,
            execution_manifest_id=normalized_execution_manifest_id,
            execution_manifest_hash=normalized_execution_manifest_hash,
            entries=sorted_entries,
            semantic_content_hash=canonical_content_hash(
                hash_payload,
                domain=DISPATCH_SEAM_INVENTORY_MANIFEST_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class DispatchSeamInventoryAssessment:
    inventory_id: str
    inventory_hash: str
    required_roles: tuple[str, ...]
    missing_roles: tuple[str, ...]
    unmediated_seam_ids: tuple[str, ...]
    inventory_structurally_complete: bool
    legacy_mediation_ready: bool
    confers_execution_authorization: bool
    semantic_content_hash: str


def assess_dispatch_seam_inventory(
    manifest: DispatchSeamInventoryManifest,
    *,
    required_roles: tuple[str, ...],
) -> DispatchSeamInventoryAssessment:
    """Assess inventory coverage without granting hardware authorization."""

    if not isinstance(manifest, DispatchSeamInventoryManifest):
        raise TypeError("manifest must be a DispatchSeamInventoryManifest.")
    normalized_roles = tuple(
        sorted(_unique_identifiers(required_roles, name="required_roles"))
    )
    present_roles = {entry.seam_role for entry in manifest.entries}
    missing_roles = tuple(role for role in normalized_roles if role not in present_roles)
    unmediated = tuple(
        sorted(
            entry.seam_id
            for entry in manifest.entries
            if entry.required_for_legacy_mediation
            and entry.integration_status != "production_mediated"
        )
    )
    structurally_complete = not missing_roles
    legacy_ready = structurally_complete and not unmediated
    payload = {
        "inventory_id": manifest.inventory_id,
        "inventory_hash": manifest.semantic_content_hash,
        "required_roles": normalized_roles,
        "missing_roles": missing_roles,
        "unmediated_seam_ids": unmediated,
        "inventory_structurally_complete": structurally_complete,
        "legacy_mediation_ready": legacy_ready,
        # Dispatch ownership and operator/hardware authorization are separate
        # runtime facts.  An inventory assessment can never confer them.
        "confers_execution_authorization": False,
    }
    return DispatchSeamInventoryAssessment(
        **payload,
        semantic_content_hash=canonical_content_hash(
            payload,
            domain=DISPATCH_SEAM_INVENTORY_ASSESSMENT_HASH_DOMAIN_V1,
        ),
    )
