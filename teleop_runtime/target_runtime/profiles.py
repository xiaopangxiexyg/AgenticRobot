"""Application-side selection of exact target-runtime profiles.

The provider registry answers *how* an allowlisted target implementation is
loaded.  This module answers a separate application question: which exact
provider, topology, capability, operation, observation, and asset set a named
profile permits.  Selection remains preflight-only and never creates a runtime
handle, opens a device, or grants motion authority.
Asset references are hash identities only; local presence and integrity remain
a later provider/application preflight responsibility.

Portable profile specs contain no Python loader and can be stored, audited, or
compared without importing an integration package.  A provider package binds
one of those specs to an explicit allowlisted loader.  The catalog is assembled
by an application, so importing this module does not discover or import
hardware integrations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.target_runtime import (
    ResolvedTargetRuntimeManifest,
    TargetRuntimeBindingReceipt,
    TargetRuntimeProviderRegistry,
    TargetRuntimeProviderRegistryEntry,
    TargetRuntimeRequirement,
    bind_target_runtime_requirement,
)

TARGET_RUNTIME_PROFILE_SPEC_SCHEMA_V1 = "dexteleop.target_runtime_profile_spec.v1"
# The first profile-selection slice already exposed this hash identity.  A
# portable spec is the same semantic object with the loader removed, so retain
# the domain and all existing profile/preflight hashes.
TARGET_RUNTIME_PROFILE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_profile_binding.v1"
)
TARGET_RUNTIME_PROFILE_SPEC_HASH_DOMAIN_V1 = (
    TARGET_RUNTIME_PROFILE_BINDING_HASH_DOMAIN_V1
)
TARGET_RUNTIME_PROFILE_PREFLIGHT_HASH_DOMAIN_V1 = (
    "dexteleop.target_runtime_profile_preflight.v1"
)


def _required(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _unique_identifiers(
    values: tuple[str, ...],
    *,
    name: str,
    allow_empty: bool = True,
    sort_values: bool = False,
) -> tuple[str, ...]:
    normalized = tuple(_required(value, name=name) for value in values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must be non-empty.")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must be unique.")
    return tuple(sorted(normalized)) if sort_values else normalized


def _hashes_by_id(
    values: Mapping[str, str],
    *,
    name: str,
) -> Mapping[str, str]:
    normalized_items = tuple(
        (
            _required(identifier, name=f"{name}_id"),
            _required(content_hash, name=f"{name}_hash"),
        )
        for identifier, content_hash in dict(values).items()
    )
    normalized = dict(normalized_items)
    if len(normalized) != len(normalized_items):
        raise ValueError(f"{name}_id must be unique after normalization.")
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class TargetRuntimeProfileSpec:
    """Portable exact-profile allowlist with no loader or vendor code.

    A spec pins every resolved runtime surface that an application is willing
    to use.  It intentionally contains only immutable data.  The provider
    identity hash binds the future executable integration without embedding a
    module path or importing its package.
    """

    profile_id: str
    profile_version: str
    requirement_id_prefix: str
    provider_identity_hash: str
    supported_clock_domain_ids: tuple[str, ...]
    expected_manifest_version: str
    expected_embodiment_definition_id: str
    expected_embodiment_definition_hash: str
    expected_target_handler_id: str
    expected_action_schema_hash: str
    expected_component_target_capability_hash: str
    expected_component_ids: tuple[str, ...]
    expected_execution_observation_capability_hash: str | None
    expected_operation_capability_hash_by_id: Mapping[str, str] = field(
        default_factory=dict
    )
    expected_asset_reference_hash_by_id: Mapping[str, str] = field(
        default_factory=dict
    )
    semantic_content_hash: str = field(init=False)
    schema: str = field(
        default=TARGET_RUNTIME_PROFILE_SPEC_SCHEMA_V1,
        init=False,
    )

    def __post_init__(self) -> None:
        identifiers = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "profile_id",
                "profile_version",
                "requirement_id_prefix",
                "provider_identity_hash",
                "expected_manifest_version",
                "expected_embodiment_definition_id",
                "expected_embodiment_definition_hash",
                "expected_target_handler_id",
                "expected_action_schema_hash",
                "expected_component_target_capability_hash",
            )
        }
        clock_domains = _unique_identifiers(
            tuple(self.supported_clock_domain_ids),
            name="supported_clock_domain_id",
            allow_empty=False,
            sort_values=True,
        )
        component_ids = _unique_identifiers(
            tuple(self.expected_component_ids),
            name="expected_component_id",
            allow_empty=False,
        )
        observation_hash = (
            None
            if self.expected_execution_observation_capability_hash is None
            else _required(
                self.expected_execution_observation_capability_hash,
                name="expected_execution_observation_capability_hash",
            )
        )
        operation_hashes = _hashes_by_id(
            self.expected_operation_capability_hash_by_id,
            name="expected_operation_capability",
        )
        asset_hashes = _hashes_by_id(
            self.expected_asset_reference_hash_by_id,
            name="expected_asset_reference",
        )
        for name, value in identifiers.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "supported_clock_domain_ids", clock_domains)
        object.__setattr__(self, "expected_component_ids", component_ids)
        object.__setattr__(
            self,
            "expected_execution_observation_capability_hash",
            observation_hash,
        )
        object.__setattr__(
            self,
            "expected_operation_capability_hash_by_id",
            operation_hashes,
        )
        object.__setattr__(
            self,
            "expected_asset_reference_hash_by_id",
            asset_hashes,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                self._semantic_content_payload(),
                domain=TARGET_RUNTIME_PROFILE_SPEC_HASH_DOMAIN_V1,
            ),
        )

    def _semantic_content_payload(self) -> dict[str, object]:
        payload = self.to_payload(include_hash=False)
        # The schema labels the portable envelope.  It is omitted from the
        # content body to preserve the exact v1 binding hash already emitted
        # before the loader-free representation was introduced.
        payload.pop("schema")
        return payload

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "requirement_id_prefix": self.requirement_id_prefix,
            "provider_identity_hash": self.provider_identity_hash,
            "supported_clock_domain_ids": self.supported_clock_domain_ids,
            "expected_manifest_version": self.expected_manifest_version,
            "expected_embodiment_definition_id": (
                self.expected_embodiment_definition_id
            ),
            "expected_embodiment_definition_hash": (
                self.expected_embodiment_definition_hash
            ),
            "expected_target_handler_id": self.expected_target_handler_id,
            "expected_action_schema_hash": self.expected_action_schema_hash,
            "expected_component_target_capability_hash": (
                self.expected_component_target_capability_hash
            ),
            "expected_component_ids": self.expected_component_ids,
            "expected_execution_observation_capability_hash": (
                self.expected_execution_observation_capability_hash
            ),
            "expected_operation_capability_hash_by_id": dict(
                self.expected_operation_capability_hash_by_id
            ),
            "expected_asset_reference_hash_by_id": dict(
                self.expected_asset_reference_hash_by_id
            ),
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload

    def semantic_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        return self.to_payload(include_hash=include_hash)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> TargetRuntimeProfileSpec:
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping.")
        if payload.get("schema") != TARGET_RUNTIME_PROFILE_SPEC_SCHEMA_V1:
            raise ValueError("Unsupported target runtime profile spec schema.")
        clock_domains = payload.get("supported_clock_domain_ids", ())
        component_ids = payload.get("expected_component_ids", ())
        operation_hashes = payload.get(
            "expected_operation_capability_hash_by_id",
            {},
        )
        asset_hashes = payload.get(
            "expected_asset_reference_hash_by_id",
            {},
        )
        if isinstance(clock_domains, (str, bytes)):
            raise TypeError("supported_clock_domain_ids must be a sequence.")
        if isinstance(component_ids, (str, bytes)):
            raise TypeError("expected_component_ids must be a sequence.")
        if not isinstance(operation_hashes, Mapping):
            raise TypeError(
                "expected_operation_capability_hash_by_id must be a mapping."
            )
        if not isinstance(asset_hashes, Mapping):
            raise TypeError(
                "expected_asset_reference_hash_by_id must be a mapping."
            )
        result = cls(
            profile_id=payload.get("profile_id", ""),
            profile_version=payload.get("profile_version", ""),
            requirement_id_prefix=payload.get("requirement_id_prefix", ""),
            provider_identity_hash=payload.get("provider_identity_hash", ""),
            supported_clock_domain_ids=tuple(clock_domains),
            expected_manifest_version=payload.get(
                "expected_manifest_version",
                "",
            ),
            expected_embodiment_definition_id=payload.get(
                "expected_embodiment_definition_id",
                "",
            ),
            expected_embodiment_definition_hash=payload.get(
                "expected_embodiment_definition_hash",
                "",
            ),
            expected_target_handler_id=payload.get(
                "expected_target_handler_id",
                "",
            ),
            expected_action_schema_hash=payload.get(
                "expected_action_schema_hash",
                "",
            ),
            expected_component_target_capability_hash=payload.get(
                "expected_component_target_capability_hash",
                "",
            ),
            expected_component_ids=tuple(component_ids),
            expected_execution_observation_capability_hash=payload.get(
                "expected_execution_observation_capability_hash"
            ),
            expected_operation_capability_hash_by_id=dict(operation_hashes),
            expected_asset_reference_hash_by_id=dict(asset_hashes),
        )
        declared_hash = payload.get("semantic_content_hash")
        if (
            declared_hash is not None
            and str(declared_hash) != result.semantic_content_hash
        ):
            raise ValueError("Target runtime profile spec semantic hash mismatch.")
        return result

    def validate_manifest(
        self,
        manifest: ResolvedTargetRuntimeManifest,
    ) -> None:
        """Fail closed if a provider resolves outside this profile."""

        if not isinstance(manifest, ResolvedTargetRuntimeManifest):
            raise TypeError("manifest must be ResolvedTargetRuntimeManifest.")
        observation = manifest.execution_observation_capability
        actual_observation_hash = (
            None if observation is None else observation.semantic_content_hash
        )
        actual_operation_hashes = {
            operation_id: capability.semantic_content_hash
            for operation_id, capability in manifest.operation_by_id.items()
        }
        actual_asset_hashes = {
            asset_id: reference.semantic_content_hash
            for asset_id, reference in manifest.asset_by_id.items()
        }
        mismatches: list[str] = []
        exact_values = (
            (
                "manifest_version",
                manifest.manifest_version,
                self.expected_manifest_version,
            ),
            (
                "provider_identity_hash",
                manifest.provider.semantic_content_hash,
                self.provider_identity_hash,
            ),
            (
                "embodiment_definition_id",
                manifest.embodiment_definition_id,
                self.expected_embodiment_definition_id,
            ),
            (
                "embodiment_definition_hash",
                manifest.embodiment_definition_hash,
                self.expected_embodiment_definition_hash,
            ),
            (
                "target_handler_id",
                manifest.target_handler_id,
                self.expected_target_handler_id,
            ),
            (
                "action_schema_hash",
                manifest.action_schema_hash,
                self.expected_action_schema_hash,
            ),
            (
                "component_target_capability_hash",
                manifest.component_target_capability.semantic_content_hash,
                self.expected_component_target_capability_hash,
            ),
            ("component_ids", manifest.component_ids, self.expected_component_ids),
            (
                "execution_observation_capability_hash",
                actual_observation_hash,
                self.expected_execution_observation_capability_hash,
            ),
            (
                "operation_capability_hash_by_id",
                actual_operation_hashes,
                dict(self.expected_operation_capability_hash_by_id),
            ),
            (
                "asset_reference_hash_by_id",
                actual_asset_hashes,
                dict(self.expected_asset_reference_hash_by_id),
            ),
        )
        for name, actual, expected in exact_values:
            if actual != expected:
                mismatches.append(name)
        if manifest.clock_domain_id not in self.supported_clock_domain_ids:
            mismatches.append("clock_domain_id")
        if mismatches:
            raise ValueError(
                "Resolved target runtime manifest differs from selected "
                f"profile {self.profile_id!r}: {','.join(mismatches)}."
            )


@dataclass(frozen=True)
class TargetRuntimeProfileBinding:
    """One portable profile spec bound to an allowlisted provider loader.

    The constructor retains the original flat field surface for compatibility.
    ``profile_spec`` is the authoritative, loader-free representation and the
    binding's semantic hash is exactly the spec hash.
    """

    profile_id: str
    profile_version: str
    requirement_id_prefix: str
    provider_entry: TargetRuntimeProviderRegistryEntry
    supported_clock_domain_ids: tuple[str, ...]
    expected_manifest_version: str
    expected_embodiment_definition_id: str
    expected_embodiment_definition_hash: str
    expected_target_handler_id: str
    expected_action_schema_hash: str
    expected_component_target_capability_hash: str
    expected_component_ids: tuple[str, ...]
    expected_execution_observation_capability_hash: str | None
    expected_operation_capability_hash_by_id: Mapping[str, str] = field(
        default_factory=dict
    )
    expected_asset_reference_hash_by_id: Mapping[str, str] = field(
        default_factory=dict
    )
    profile_spec: TargetRuntimeProfileSpec = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.provider_entry,
            TargetRuntimeProviderRegistryEntry,
        ):
            raise TypeError(
                "provider_entry must be TargetRuntimeProviderRegistryEntry."
            )
        spec = TargetRuntimeProfileSpec(
            profile_id=self.profile_id,
            profile_version=self.profile_version,
            requirement_id_prefix=self.requirement_id_prefix,
            provider_identity_hash=(
                self.provider_entry.provider_identity.semantic_content_hash
            ),
            supported_clock_domain_ids=tuple(self.supported_clock_domain_ids),
            expected_manifest_version=self.expected_manifest_version,
            expected_embodiment_definition_id=(
                self.expected_embodiment_definition_id
            ),
            expected_embodiment_definition_hash=(
                self.expected_embodiment_definition_hash
            ),
            expected_target_handler_id=self.expected_target_handler_id,
            expected_action_schema_hash=self.expected_action_schema_hash,
            expected_component_target_capability_hash=(
                self.expected_component_target_capability_hash
            ),
            expected_component_ids=tuple(self.expected_component_ids),
            expected_execution_observation_capability_hash=(
                self.expected_execution_observation_capability_hash
            ),
            expected_operation_capability_hash_by_id=(
                self.expected_operation_capability_hash_by_id
            ),
            expected_asset_reference_hash_by_id=(
                self.expected_asset_reference_hash_by_id
            ),
        )
        for name in (
            "profile_id",
            "profile_version",
            "requirement_id_prefix",
            "supported_clock_domain_ids",
            "expected_manifest_version",
            "expected_embodiment_definition_id",
            "expected_embodiment_definition_hash",
            "expected_target_handler_id",
            "expected_action_schema_hash",
            "expected_component_target_capability_hash",
            "expected_component_ids",
            "expected_execution_observation_capability_hash",
            "expected_operation_capability_hash_by_id",
            "expected_asset_reference_hash_by_id",
        ):
            object.__setattr__(self, name, getattr(spec, name))
        object.__setattr__(self, "profile_spec", spec)
        object.__setattr__(
            self,
            "semantic_content_hash",
            spec.semantic_content_hash,
        )

    @classmethod
    def from_spec(
        cls,
        spec: TargetRuntimeProfileSpec,
        *,
        provider_entry: TargetRuntimeProviderRegistryEntry,
    ) -> TargetRuntimeProfileBinding:
        if not isinstance(spec, TargetRuntimeProfileSpec):
            raise TypeError("spec must be TargetRuntimeProfileSpec.")
        if not isinstance(provider_entry, TargetRuntimeProviderRegistryEntry):
            raise TypeError(
                "provider_entry must be TargetRuntimeProviderRegistryEntry."
            )
        if (
            provider_entry.provider_identity.semantic_content_hash
            != spec.provider_identity_hash
        ):
            raise ValueError(
                "Target runtime profile spec belongs to another provider "
                "identity."
            )
        return cls(
            profile_id=spec.profile_id,
            profile_version=spec.profile_version,
            requirement_id_prefix=spec.requirement_id_prefix,
            provider_entry=provider_entry,
            supported_clock_domain_ids=spec.supported_clock_domain_ids,
            expected_manifest_version=spec.expected_manifest_version,
            expected_embodiment_definition_id=(
                spec.expected_embodiment_definition_id
            ),
            expected_embodiment_definition_hash=(
                spec.expected_embodiment_definition_hash
            ),
            expected_target_handler_id=spec.expected_target_handler_id,
            expected_action_schema_hash=spec.expected_action_schema_hash,
            expected_component_target_capability_hash=(
                spec.expected_component_target_capability_hash
            ),
            expected_component_ids=spec.expected_component_ids,
            expected_execution_observation_capability_hash=(
                spec.expected_execution_observation_capability_hash
            ),
            expected_operation_capability_hash_by_id=(
                spec.expected_operation_capability_hash_by_id
            ),
            expected_asset_reference_hash_by_id=(
                spec.expected_asset_reference_hash_by_id
            ),
        )

    def semantic_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload = self.profile_spec.to_payload(include_hash=include_hash)
        # Preserve the original binding payload shape.  New portable callers
        # should serialize ``profile_spec`` and therefore receive its schema.
        payload.pop("schema")
        return payload

    def validate_manifest(
        self,
        manifest: ResolvedTargetRuntimeManifest,
    ) -> None:
        if (
            self.provider_entry.provider_identity.semantic_content_hash
            != self.profile_spec.provider_identity_hash
        ):
            raise ValueError(
                "Target runtime profile binding provider identity drifted."
            )
        self.profile_spec.validate_manifest(manifest)


@dataclass(frozen=True)
class TargetRuntimeProfilePreflight:
    """Offline profile selection result with no handle or authority."""

    profile_binding: TargetRuntimeProfileBinding
    resolved_manifest: ResolvedTargetRuntimeManifest
    requirement: TargetRuntimeRequirement
    binding_receipt: TargetRuntimeBindingReceipt
    runtime_handle_created: bool = False
    confers_execution_authorization: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.profile_binding, TargetRuntimeProfileBinding):
            raise TypeError(
                "profile_binding must be TargetRuntimeProfileBinding."
            )
        if not isinstance(
            self.resolved_manifest,
            ResolvedTargetRuntimeManifest,
        ):
            raise TypeError(
                "resolved_manifest must be ResolvedTargetRuntimeManifest."
            )
        if not isinstance(self.requirement, TargetRuntimeRequirement):
            raise TypeError("requirement must be TargetRuntimeRequirement.")
        if not isinstance(self.binding_receipt, TargetRuntimeBindingReceipt):
            raise TypeError(
                "binding_receipt must be TargetRuntimeBindingReceipt."
            )
        self.profile_binding.validate_manifest(self.resolved_manifest)
        expected_receipt = bind_target_runtime_requirement(
            self.requirement,
            self.resolved_manifest,
            bound_at=self.binding_receipt.bound_at,
        )
        if (
            self.binding_receipt.semantic_content_hash
            != expected_receipt.semantic_content_hash
        ):
            raise ValueError(
                "Target profile preflight receipt belongs to another binding."
            )
        if self.runtime_handle_created or self.confers_execution_authorization:
            raise ValueError(
                "Target profile preflight cannot create a handle or authorize "
                "execution."
            )
        payload = {
            "profile_binding_hash": self.profile_binding.semantic_content_hash,
            "resolved_manifest_hash": (
                self.resolved_manifest.semantic_content_hash
            ),
            "requirement_hash": self.requirement.semantic_content_hash,
            "binding_receipt_hash": self.binding_receipt.semantic_content_hash,
            "runtime_handle_created": False,
            "confers_execution_authorization": False,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=TARGET_RUNTIME_PROFILE_PREFLIGHT_HASH_DOMAIN_V1,
            ),
        )


class TargetRuntimeProfileCatalog:
    """Explicit profile allowlist that loads only the selected provider."""

    def __init__(
        self,
        bindings: Iterable[TargetRuntimeProfileBinding] = (),
    ) -> None:
        self._binding_by_id: dict[str, TargetRuntimeProfileBinding] = {}
        for binding in bindings:
            self.register(binding)

    def register(self, binding: TargetRuntimeProfileBinding) -> None:
        if not isinstance(binding, TargetRuntimeProfileBinding):
            raise TypeError("binding must be TargetRuntimeProfileBinding.")
        if binding.profile_id in self._binding_by_id:
            raise KeyError(
                f"Target runtime profile {binding.profile_id!r} is already "
                "registered."
            )
        self._binding_by_id[binding.profile_id] = binding

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._binding_by_id))

    def profile_bindings(self) -> tuple[TargetRuntimeProfileBinding, ...]:
        return tuple(
            self._binding_by_id[profile_id]
            for profile_id in self.profile_ids()
        )

    def profile_specs(self) -> tuple[TargetRuntimeProfileSpec, ...]:
        """List portable profiles without invoking any provider loader."""

        return tuple(binding.profile_spec for binding in self.profile_bindings())

    def preflight(
        self,
        profile_id: str,
        config: Any,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        bound_at: float,
    ) -> TargetRuntimeProfilePreflight:
        selected_id = _required(profile_id, name="profile_id")
        binding = self._binding_by_id.get(selected_id)
        if binding is None:
            raise KeyError(
                f"Unknown target runtime profile {selected_id!r}. Registered: "
                f"{list(self.profile_ids())!r}."
            )
        provider_registry = TargetRuntimeProviderRegistry()
        provider_registry.register(binding.provider_entry)
        provider_id = binding.provider_entry.provider_identity.provider_id
        manifest = provider_registry.resolve_manifest(
            provider_id,
            config,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
        )
        binding.validate_manifest(manifest)
        requirement = TargetRuntimeRequirement(
            requirement_id=(
                f"{binding.requirement_id_prefix}:"
                f"{manifest.runtime_session_id}:{manifest.control_domain_id}"
            ),
            runtime_session_id=manifest.runtime_session_id,
            control_domain_id=manifest.control_domain_id,
            clock_domain_id=manifest.clock_domain_id,
            embodiment_instance_id=manifest.embodiment_instance_id,
            target_handler_id=manifest.target_handler_id,
            action_schema_hash=manifest.action_schema_hash,
            required_component_ids=binding.expected_component_ids,
            required_operation_ids=tuple(
                binding.expected_operation_capability_hash_by_id
            ),
            required_provider_api=(
                binding.provider_entry.provider_identity.provider_api
            ),
        )
        receipt = bind_target_runtime_requirement(
            requirement,
            manifest,
            bound_at=bound_at,
        )
        return TargetRuntimeProfilePreflight(
            profile_binding=binding,
            resolved_manifest=manifest,
            requirement=requirement,
            binding_receipt=receipt,
        )


__all__ = [
    "TARGET_RUNTIME_PROFILE_BINDING_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROFILE_PREFLIGHT_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROFILE_SPEC_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROFILE_SPEC_SCHEMA_V1",
    "TargetRuntimeProfileBinding",
    "TargetRuntimeProfileCatalog",
    "TargetRuntimeProfilePreflight",
    "TargetRuntimeProfileSpec",
]
