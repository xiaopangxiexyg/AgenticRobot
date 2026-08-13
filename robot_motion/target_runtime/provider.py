"""Target provider SPI built on the shared target-runtime contracts.

Providers are registered explicitly by stable ID.  Registry discovery never
imports a module path from configuration, starts a device, or owns the
lifecycle of a returned handle. Concrete providers must declare whether they
are preflight-only or can construct an inert handle. Handle-capable providers
keep vendor imports behind their loader; an authorized lower operation starts
or configures a real or simulated target later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .binding import TargetRuntimeBindingReceipt
from .contracts import (
    ResolvedTargetRuntimeManifest,
    TargetRuntimeHandle,
    TargetRuntimeProviderIdentity,
    TARGET_RUNTIME_PROVIDER_API_V1,
)


class TargetRuntimeProvider(Protocol):
    """One allowlisted target implementation provider."""

    @property
    def provider_identity(self) -> TargetRuntimeProviderIdentity:
        ...

    def resolve_target_runtime_manifest(
        self,
        config: Any,
        *,
        runtime_session_id: str,
        control_domain_id: str,
    ) -> ResolvedTargetRuntimeManifest:
        """Resolve an immutable manifest without opening a device."""
        ...

    def create_target_runtime_handle(
        self,
        config: Any,
        *,
        manifest: ResolvedTargetRuntimeManifest,
        binding_receipt: TargetRuntimeBindingReceipt,
    ) -> TargetRuntimeHandle:
        """Create an inert handle when the provider identity permits it."""
        ...


TargetRuntimeProviderLoader = Callable[[], TargetRuntimeProvider]


@dataclass(frozen=True)
class TargetRuntimeProviderRegistryEntry:
    provider_identity: TargetRuntimeProviderIdentity
    loader: TargetRuntimeProviderLoader

    def __post_init__(self) -> None:
        if not isinstance(self.provider_identity, TargetRuntimeProviderIdentity):
            raise TypeError(
                "provider_identity must be TargetRuntimeProviderIdentity."
            )
        if self.provider_identity.provider_api != TARGET_RUNTIME_PROVIDER_API_V1:
            raise ValueError("Target runtime provider uses an unsupported API.")
        if not callable(self.loader):
            raise TypeError("Target runtime provider loader must be callable.")


class TargetRuntimeProviderRegistry:
    """Allowlisted provider catalog with lazy implementation loading.

    The registry may cache provider factory objects, but never a runtime
    handle, connection, worker, cursor, authority token, or execution state.
    """

    def __init__(self) -> None:
        self._entry_by_id: dict[str, TargetRuntimeProviderRegistryEntry] = {}
        self._provider_by_id: dict[str, TargetRuntimeProvider] = {}

    def register(
        self,
        entry: TargetRuntimeProviderRegistryEntry,
    ) -> None:
        if not isinstance(entry, TargetRuntimeProviderRegistryEntry):
            raise TypeError("entry must be TargetRuntimeProviderRegistryEntry.")
        provider_id = entry.provider_identity.provider_id
        if provider_id in self._entry_by_id:
            raise KeyError(
                f"Target runtime provider {provider_id!r} is already registered."
            )
        self._entry_by_id[provider_id] = entry

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entry_by_id))

    def provider_identities(self) -> tuple[TargetRuntimeProviderIdentity, ...]:
        return tuple(
            self._entry_by_id[provider_id].provider_identity
            for provider_id in self.provider_ids()
        )

    def resolve_manifest(
        self,
        provider_id: str,
        config: Any,
        *,
        runtime_session_id: str,
        control_domain_id: str,
    ) -> ResolvedTargetRuntimeManifest:
        provider = self._load_provider(provider_id)
        manifest = provider.resolve_target_runtime_manifest(
            config,
            runtime_session_id=runtime_session_id,
            control_domain_id=control_domain_id,
        )
        if not isinstance(manifest, ResolvedTargetRuntimeManifest):
            raise TypeError(
                "Target runtime provider must resolve a "
                "ResolvedTargetRuntimeManifest."
            )
        expected_identity = self._entry(provider_id).provider_identity
        if (
            manifest.provider.semantic_content_hash
            != expected_identity.semantic_content_hash
        ):
            raise ValueError(
                "Resolved target manifest uses another provider identity."
            )
        if (
            manifest.runtime_session_id != str(runtime_session_id).strip()
            or manifest.control_domain_id != str(control_domain_id).strip()
        ):
            raise ValueError(
                "Resolved target manifest uses another session or control domain."
            )
        return manifest

    def create_handle(
        self,
        provider_id: str,
        config: Any,
        *,
        manifest: ResolvedTargetRuntimeManifest,
        binding_receipt: TargetRuntimeBindingReceipt,
    ) -> TargetRuntimeHandle:
        if not isinstance(manifest, ResolvedTargetRuntimeManifest):
            raise TypeError("manifest must be ResolvedTargetRuntimeManifest.")
        if not isinstance(binding_receipt, TargetRuntimeBindingReceipt):
            raise TypeError(
                "binding_receipt must be TargetRuntimeBindingReceipt."
            )
        expected_identity = self._entry(provider_id).provider_identity
        if expected_identity.runtime_handle_mode != "inert_handle":
            raise RuntimeError(
                "Target runtime provider is preflight-only and cannot create "
                "a runtime handle."
            )
        provider = self._load_provider(provider_id)
        if (
            manifest.provider.semantic_content_hash
            != expected_identity.semantic_content_hash
        ):
            raise ValueError("Manifest belongs to another target provider.")
        if (
            binding_receipt.manifest_id != manifest.manifest_id
            or binding_receipt.manifest_hash != manifest.semantic_content_hash
            or binding_receipt.provider_identity_hash
            != manifest.provider.semantic_content_hash
            or binding_receipt.component_target_capability_hash
            != manifest.component_target_capability.semantic_content_hash
            or binding_receipt.target_handler_id != manifest.target_handler_id
            or binding_receipt.embodiment_instance_id
            != manifest.embodiment_instance_id
        ):
            raise ValueError("Binding receipt belongs to another target manifest.")
        handle = provider.create_target_runtime_handle(
            config,
            manifest=manifest,
            binding_receipt=binding_receipt,
        )
        self._validate_handle(handle, manifest)
        return handle

    def _entry(self, provider_id: str) -> TargetRuntimeProviderRegistryEntry:
        normalized = str(provider_id).strip()
        if normalized not in self._entry_by_id:
            raise KeyError(
                f"Unknown target runtime provider {normalized!r}. Registered: "
                f"{list(self.provider_ids())!r}."
            )
        return self._entry_by_id[normalized]

    def _load_provider(self, provider_id: str) -> TargetRuntimeProvider:
        entry = self._entry(provider_id)
        provider = self._provider_by_id.get(entry.provider_identity.provider_id)
        if provider is not None:
            return provider
        provider = entry.loader()
        actual_identity = getattr(provider, "provider_identity", None)
        if not isinstance(actual_identity, TargetRuntimeProviderIdentity):
            raise TypeError(
                "Target runtime provider must expose provider_identity."
            )
        if (
            actual_identity.semantic_content_hash
            != entry.provider_identity.semantic_content_hash
        ):
            raise ValueError(
                "Loaded target runtime provider identity differs from its "
                "registry entry."
            )
        self._provider_by_id[entry.provider_identity.provider_id] = provider
        return provider

    @staticmethod
    def _validate_handle(
        handle: TargetRuntimeHandle,
        manifest: ResolvedTargetRuntimeManifest,
    ) -> None:
        handle_manifest = getattr(
            handle,
            "resolved_target_runtime_manifest",
            None,
        )
        if not isinstance(handle_manifest, ResolvedTargetRuntimeManifest) or (
            handle_manifest.semantic_content_hash != manifest.semantic_content_hash
        ):
            raise ValueError("Target runtime handle exposes another manifest.")

        execution_port = handle.execution_port
        if (
            execution_port.target_handler_id != manifest.target_handler_id
            or execution_port.component_target_capability.semantic_content_hash
            != manifest.component_target_capability.semantic_content_hash
            or not callable(getattr(execution_port, "submit", None))
        ):
            raise ValueError(
                "Target runtime execution port does not match the manifest."
            )

        execution_status_provider = handle.execution_status_provider
        if not callable(
            getattr(
                execution_status_provider,
                "capture_target_execution_status_snapshot",
                None,
            )
        ):
            raise TypeError(
                "Target runtime handle lacks an execution status provider."
            )

        observation_provider = handle.execution_observation_provider
        expected_observation = manifest.execution_observation_capability
        actual_observation = (
            None
            if observation_provider is None
            else observation_provider.execution_observation_capability
        )
        if expected_observation is None:
            if actual_observation is not None:
                raise ValueError(
                    "Target runtime handle exposes undeclared observations."
                )
        elif actual_observation is None or (
            actual_observation.semantic_content_hash
            != expected_observation.semantic_content_hash
        ):
            raise ValueError(
                "Target runtime observation port does not match the manifest."
            )

        status_provider = handle.target_runtime_status_provider
        operation_port = handle.target_runtime_operation_port
        if not callable(
            getattr(status_provider, "capture_target_runtime_status", None)
        ):
            raise TypeError("Target runtime handle lacks a status provider.")
        if not callable(
            getattr(operation_port, "execute_target_runtime_operation", None)
        ):
            raise TypeError("Target runtime handle lacks an operation port.")
        status_manifest = status_provider.resolved_target_runtime_manifest
        operation_manifest = operation_port.resolved_target_runtime_manifest
        if (
            status_manifest.semantic_content_hash != manifest.semantic_content_hash
            or operation_manifest.semantic_content_hash
            != manifest.semantic_content_hash
        ):
            raise ValueError(
                "Target runtime lifecycle ports do not match the manifest."
            )


__all__ = [
    "TargetRuntimeProvider",
    "TargetRuntimeProviderLoader",
    "TargetRuntimeProviderRegistry",
    "TargetRuntimeProviderRegistryEntry",
]
