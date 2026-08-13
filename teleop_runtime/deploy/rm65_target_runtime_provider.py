"""Preflight-only target-runtime provider for the frozen RM65 qpos candidate.

This adapter projects the existing ``m6_candidate_policy_arm_qpos_v1``
manifest and its evidence bundle into the shared target-runtime contracts.  It
does not create an RM65 backend, import the vendor SDK, open a device, create a
worker, or confer motion/lifecycle authority.  The existing deployment,
teleoperation, and collection entrypoints remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NoReturn

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ControlExecutionManifest
from robot_motion.target_runtime import (
    TARGET_RUNTIME_PROVIDER_API_V1,
    ResolvedTargetRuntimeManifest,
    TargetRuntimeAssetReference,
    TargetRuntimeBindingReceipt,
    TargetRuntimeProviderIdentity,
    TargetRuntimeProviderRegistry,
    TargetRuntimeProviderRegistryEntry,
    TargetRuntimeRequirement,
    bind_target_runtime_requirement,
)

from .rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
    build_rm65_policy_arm_qpos_candidate_target_capability,
)
from .rm65_execution_evidence import (
    RM65ExecutionEvidenceBundle,
    build_rm65_execution_evidence_bundle,
)


RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID = (
    "dexteleop.rm65_policy_arm_qpos_candidate.preflight"
)
RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_HANDLER_ID = (
    "dexteleop.rm65_policy_arm_qpos_candidate.target.v1"
)
RM65_POLICY_ARM_QPOS_CANDIDATE_EMBODIMENT_DEFINITION_ID = (
    "dexteleop.rm65_bimanual_policy_arm_qpos_candidate.v1"
)
RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_policy_arm_qpos_candidate.provider_contract.v1"
)
RM65_POLICY_ARM_QPOS_CANDIDATE_DEFINITION_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_policy_arm_qpos_candidate.definition.v1"
)
RM65_POLICY_ARM_QPOS_CANDIDATE_MANIFEST_ID_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_policy_arm_qpos_candidate.manifest_id.v1"
)
RM65_POLICY_ARM_QPOS_CANDIDATE_PREFLIGHT_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_policy_arm_qpos_candidate.preflight.v1"
)


def _required(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


_PROVIDER_CONTRACT_HASH = canonical_content_hash(
    {
        "source_execution_manifest": "m6_candidate_policy_arm_qpos_v1",
        "projection": "exact_hash_bound_preflight_only",
        "target_handler_id": (
            RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_HANDLER_ID
        ),
        "runtime_handle_mode": "preflight_only",
        "device_initialization": "forbidden",
        "motion_authorization": "none",
        "observation_capability": "not_declared",
        "operation_capabilities": (),
    },
    domain=(
        RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_CONTRACT_HASH_DOMAIN_V1
    ),
)

RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_IDENTITY = (
    TargetRuntimeProviderIdentity(
        provider_id=RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID,
        provider_version="1.0.0",
        provider_api=TARGET_RUNTIME_PROVIDER_API_V1,
        implementation_contract_hash=_PROVIDER_CONTRACT_HASH,
        package_name="dexteleop",
        package_version="0.1.0",
        runtime_handle_mode="preflight_only",
    )
)


@dataclass(frozen=True)
class RM65PolicyArmQposCandidateTargetConfig:
    """Explicit input to the preflight-only RM65 target projection."""

    control_execution_manifest: ControlExecutionManifest
    embodiment_instance_id: str
    embodiment_instance_hash: str
    clock_domain_id: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.control_execution_manifest,
            ControlExecutionManifest,
        ):
            raise TypeError(
                "control_execution_manifest must be a ControlExecutionManifest."
            )
        for name in (
            "embodiment_instance_id",
            "embodiment_instance_hash",
            "clock_domain_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )


@dataclass(frozen=True)
class RM65PolicyArmQposCandidatePreflight:
    """Offline binding result; deliberately carries no runtime handle."""

    resolved_manifest: ResolvedTargetRuntimeManifest
    binding_receipt: TargetRuntimeBindingReceipt
    legacy_execution_manifest_hash: str
    execution_evidence_bundle_hash: str
    integration_status: str
    runtime_handle_created: bool = False
    confers_execution_authorization: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.resolved_manifest,
            ResolvedTargetRuntimeManifest,
        ):
            raise TypeError(
                "resolved_manifest must be ResolvedTargetRuntimeManifest."
            )
        if not isinstance(self.binding_receipt, TargetRuntimeBindingReceipt):
            raise TypeError(
                "binding_receipt must be TargetRuntimeBindingReceipt."
            )
        if (
            self.binding_receipt.manifest_hash
            != self.resolved_manifest.semantic_content_hash
        ):
            raise ValueError("Preflight binding belongs to another manifest.")
        if self.runtime_handle_created or self.confers_execution_authorization:
            raise ValueError(
                "RM65 candidate preflight cannot create a handle or authorize "
                "execution."
            )
        payload = {
            "resolved_manifest_hash": (
                self.resolved_manifest.semantic_content_hash
            ),
            "binding_receipt_hash": self.binding_receipt.semantic_content_hash,
            "legacy_execution_manifest_hash": _required(
                self.legacy_execution_manifest_hash,
                name="legacy_execution_manifest_hash",
            ),
            "execution_evidence_bundle_hash": _required(
                self.execution_evidence_bundle_hash,
                name="execution_evidence_bundle_hash",
            ),
            "integration_status": _required(
                self.integration_status,
                name="integration_status",
            ),
            "runtime_handle_created": False,
            "confers_execution_authorization": False,
        }
        object.__setattr__(
            self,
            "legacy_execution_manifest_hash",
            payload["legacy_execution_manifest_hash"],
        )
        object.__setattr__(
            self,
            "execution_evidence_bundle_hash",
            payload["execution_evidence_bundle_hash"],
        )
        object.__setattr__(
            self,
            "integration_status",
            payload["integration_status"],
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    RM65_POLICY_ARM_QPOS_CANDIDATE_PREFLIGHT_HASH_DOMAIN_V1
                ),
            ),
        )


def _candidate_and_evidence(
    config: RM65PolicyArmQposCandidateTargetConfig,
) -> tuple[ControlExecutionManifest, RM65ExecutionEvidenceBundle]:
    candidate = config.control_execution_manifest
    canonical_candidate = build_rm65_policy_arm_qpos_candidate_manifest()
    if candidate != canonical_candidate:
        raise ValueError(
            "RM65 target provider requires the exact frozen policy-arm-qpos "
            "candidate manifest."
        )
    evidence = build_rm65_execution_evidence_bundle(candidate)
    if (
        evidence.production_binding_enabled
        or evidence.confers_execution_authorization
        or evidence.integration_status != "evidence_only_unbound"
    ):
        raise ValueError(
            "RM65 candidate provider requires evidence-only unbound semantics."
        )
    return candidate, evidence


def _definition_hash(
    candidate: ControlExecutionManifest,
    evidence: RM65ExecutionEvidenceBundle,
) -> str:
    return canonical_content_hash(
        {
            "definition_id": (
                RM65_POLICY_ARM_QPOS_CANDIDATE_EMBODIMENT_DEFINITION_ID
            ),
            "execution_manifest_hash": candidate.semantic_content_hash,
            "action_schema_hash": candidate.action_schema_hash,
            "target_capability_hash": (
                candidate.component_target_capability_hash
            ),
            "dispatch_inventory_hash": (
                evidence.dispatch_inventory.semantic_content_hash
            ),
            "worker_drain_plan_hash": (
                evidence.worker_drain_plan.semantic_content_hash
            ),
            "safety_migration_matrix_hash": (
                evidence.safety_migration_matrix.semantic_content_hash
            ),
            "excluded_capability_hashes": tuple(
                item.semantic_content_hash for item in candidate.exclusions
            ),
            "integration_status": evidence.integration_status,
        },
        domain=RM65_POLICY_ARM_QPOS_CANDIDATE_DEFINITION_HASH_DOMAIN_V1,
    )


class RM65PolicyArmQposCandidateTargetProvider:
    """Project the existing candidate into a shared, preflight-only manifest."""

    @property
    def provider_identity(self) -> TargetRuntimeProviderIdentity:
        return RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_IDENTITY

    def resolve_target_runtime_manifest(
        self,
        config: Any,
        *,
        runtime_session_id: str,
        control_domain_id: str,
    ) -> ResolvedTargetRuntimeManifest:
        if not isinstance(config, RM65PolicyArmQposCandidateTargetConfig):
            raise TypeError(
                "config must be RM65PolicyArmQposCandidateTargetConfig."
            )
        session_id = _required(
            runtime_session_id,
            name="runtime_session_id",
        )
        domain_id = _required(control_domain_id, name="control_domain_id")
        candidate, evidence = _candidate_and_evidence(config)
        target_capability = (
            build_rm65_policy_arm_qpos_candidate_target_capability()
        )
        if (
            target_capability.semantic_content_hash
            != candidate.component_target_capability_hash
            or target_capability.action_schema_hash
            != candidate.action_schema_hash
        ):
            raise ValueError(
                "RM65 target capability drifted from the frozen candidate."
            )
        definition_hash = _definition_hash(candidate, evidence)
        manifest_identity_hash = canonical_content_hash(
            {
                "runtime_session_id": session_id,
                "control_domain_id": domain_id,
                "clock_domain_id": config.clock_domain_id,
                "embodiment_instance_id": config.embodiment_instance_id,
                "embodiment_instance_hash": config.embodiment_instance_hash,
                "embodiment_definition_hash": definition_hash,
                "provider_identity_hash": (
                    self.provider_identity.semantic_content_hash
                ),
            },
            domain=(
                RM65_POLICY_ARM_QPOS_CANDIDATE_MANIFEST_ID_HASH_DOMAIN_V1
            ),
        )
        return ResolvedTargetRuntimeManifest(
            manifest_id=(
                "resolved-rm65-policy-arm-qpos-candidate:"
                f"{manifest_identity_hash}"
            ),
            manifest_version="1",
            runtime_session_id=session_id,
            control_domain_id=domain_id,
            clock_domain_id=config.clock_domain_id,
            embodiment_definition_id=(
                RM65_POLICY_ARM_QPOS_CANDIDATE_EMBODIMENT_DEFINITION_ID
            ),
            embodiment_definition_hash=definition_hash,
            embodiment_instance_id=config.embodiment_instance_id,
            embodiment_instance_hash=config.embodiment_instance_hash,
            target_handler_id=(
                RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_HANDLER_ID
            ),
            provider=self.provider_identity,
            component_target_capability=target_capability,
            execution_observation_capability=None,
            operation_capabilities=(),
            asset_references=(
                TargetRuntimeAssetReference(
                    asset_id="legacy-control-execution-manifest",
                    asset_kind="control_execution_manifest",
                    content_hash=candidate.semantic_content_hash,
                ),
                TargetRuntimeAssetReference(
                    asset_id="legacy-dispatch-inventory",
                    asset_kind="dispatch_inventory",
                    content_hash=(
                        evidence.dispatch_inventory.semantic_content_hash
                    ),
                ),
                TargetRuntimeAssetReference(
                    asset_id="legacy-safety-migration-matrix",
                    asset_kind="safety_migration_matrix",
                    content_hash=(
                        evidence.safety_migration_matrix.semantic_content_hash
                    ),
                ),
                TargetRuntimeAssetReference(
                    asset_id="legacy-worker-drain-plan",
                    asset_kind="worker_drain_plan",
                    content_hash=(
                        evidence.worker_drain_plan.semantic_content_hash
                    ),
                ),
            ),
        )

    def create_target_runtime_handle(
        self,
        config: Any,
        *,
        manifest: ResolvedTargetRuntimeManifest,
        binding_receipt: TargetRuntimeBindingReceipt,
    ) -> NoReturn:
        raise RuntimeError(
            "The RM65 policy-arm-qpos candidate provider is preflight-only; "
            "the existing execution path remains unbound."
        )


def build_rm65_policy_arm_qpos_candidate_provider_entry(
) -> TargetRuntimeProviderRegistryEntry:
    """Return an allowlisted lazy registry entry for this exact provider."""

    return TargetRuntimeProviderRegistryEntry(
        provider_identity=(
            RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_IDENTITY
        ),
        loader=RM65PolicyArmQposCandidateTargetProvider,
    )


def preflight_rm65_policy_arm_qpos_candidate_target(
    config: RM65PolicyArmQposCandidateTargetConfig,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    bound_at: float,
) -> RM65PolicyArmQposCandidatePreflight:
    """Resolve and bind the current candidate without constructing a target."""

    registry = TargetRuntimeProviderRegistry()
    registry.register(
        build_rm65_policy_arm_qpos_candidate_provider_entry()
    )
    manifest = registry.resolve_manifest(
        RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID,
        config,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
    )
    requirement = TargetRuntimeRequirement(
        requirement_id=(
            "rm65-policy-arm-qpos-candidate-preflight:"
            f"{manifest.runtime_session_id}:{manifest.control_domain_id}"
        ),
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        clock_domain_id=manifest.clock_domain_id,
        embodiment_instance_id=manifest.embodiment_instance_id,
        target_handler_id=manifest.target_handler_id,
        action_schema_hash=manifest.action_schema_hash,
        required_component_ids=manifest.component_ids,
    )
    binding = bind_target_runtime_requirement(
        requirement,
        manifest,
        bound_at=bound_at,
    )
    evidence = build_rm65_execution_evidence_bundle(
        config.control_execution_manifest
    )
    return RM65PolicyArmQposCandidatePreflight(
        resolved_manifest=manifest,
        binding_receipt=binding,
        legacy_execution_manifest_hash=(
            config.control_execution_manifest.semantic_content_hash
        ),
        execution_evidence_bundle_hash=evidence.semantic_content_hash,
        integration_status=evidence.integration_status,
    )


__all__ = [
    "RM65_POLICY_ARM_QPOS_CANDIDATE_EMBODIMENT_DEFINITION_ID",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_ID",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_PROVIDER_IDENTITY",
    "RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_HANDLER_ID",
    "RM65PolicyArmQposCandidatePreflight",
    "RM65PolicyArmQposCandidateTargetConfig",
    "RM65PolicyArmQposCandidateTargetProvider",
    "build_rm65_policy_arm_qpos_candidate_provider_entry",
    "preflight_rm65_policy_arm_qpos_candidate_target",
]
