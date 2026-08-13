"""Application-side preparation of one selected target-runtime handle.

This is the narrow handoff between profile preflight and a concrete provider.
It may construct only a provider-declared inert handle.  It does not invoke a
target operation, submit an action, start or home a device, mint a permit, or
grant execution authority.  The returned handle remains a set of lower-owned
ports and its subsequent lifecycle belongs to the calling application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robot_motion.target_runtime import (
    ResolvedTargetRuntimeManifest,
    TargetRuntimeHandle,
    TargetRuntimeProviderRegistry,
    TargetRuntimeStatusSnapshot,
)

from .target_profiles import TargetRuntimeProfilePreflight


@dataclass(frozen=True)
class TargetRuntimeHandlePreparation:
    """Validated inert handle plus its exact preflight and initial status.

    This is an in-process resource handoff, not a serializable receipt.  The
    handle object is therefore deliberately excluded from value equality.
    """

    profile_preflight: TargetRuntimeProfilePreflight
    initial_status: TargetRuntimeStatusSnapshot
    runtime_handle: TargetRuntimeHandle = field(repr=False, compare=False)
    runtime_handle_created: bool = field(default=True, init=False)
    target_operation_invoked: bool = field(default=False, init=False)
    execution_submitted: bool = field(default=False, init=False)
    confers_execution_authorization: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.profile_preflight,
            TargetRuntimeProfilePreflight,
        ):
            raise TypeError(
                "profile_preflight must be TargetRuntimeProfilePreflight."
            )
        if not isinstance(self.initial_status, TargetRuntimeStatusSnapshot):
            raise TypeError("initial_status must be TargetRuntimeStatusSnapshot.")
        manifest = self.profile_preflight.resolved_manifest
        handle_manifest = getattr(
            self.runtime_handle,
            "resolved_target_runtime_manifest",
            None,
        )
        if (
            not isinstance(handle_manifest, ResolvedTargetRuntimeManifest)
            or handle_manifest.semantic_content_hash
            != manifest.semantic_content_hash
        ):
            raise ValueError(
                "Prepared target runtime handle belongs to another manifest."
            )
        if manifest.provider.runtime_handle_mode != "inert_handle":
            raise ValueError(
                "Prepared target runtime handle provider is not inert-handle "
                "capable."
            )
        exact_status_values = (
            (
                "manifest_hash",
                self.initial_status.manifest_hash,
                manifest.semantic_content_hash,
            ),
            (
                "runtime_session_id",
                self.initial_status.runtime_session_id,
                manifest.runtime_session_id,
            ),
            (
                "control_domain_id",
                self.initial_status.control_domain_id,
                manifest.control_domain_id,
            ),
            (
                "target_handler_id",
                self.initial_status.target_handler_id,
                manifest.target_handler_id,
            ),
            (
                "clock_domain_id",
                self.initial_status.clock_domain_id,
                manifest.clock_domain_id,
            ),
        )
        mismatches = [
            name for name, actual, expected in exact_status_values if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "Prepared target runtime initial status differs from its "
                f"manifest: {','.join(mismatches)}."
            )
        if self.initial_status.lifecycle_state != "created":
            raise ValueError(
                "An inert target runtime handle must begin in created state."
            )
        if self.initial_status.ready:
            raise ValueError("An inert target runtime handle cannot begin ready.")


def prepare_target_runtime_handle(
    profile_preflight: TargetRuntimeProfilePreflight,
    config: Any,
) -> TargetRuntimeHandlePreparation:
    """Create one selected inert handle without advancing its lifecycle.

    The provider resolves ``config`` again in this preparation boundary.  Its
    manifest must exactly match the prior preflight, which prevents a caller
    from preflighting one target/configuration and constructing another.  A
    provider must bind every handle-affecting configuration choice into that
    manifest as required by the provider SPI.
    """

    if not isinstance(profile_preflight, TargetRuntimeProfilePreflight):
        raise TypeError(
            "profile_preflight must be TargetRuntimeProfilePreflight."
        )
    binding = profile_preflight.profile_binding
    provider_identity = binding.provider_entry.provider_identity
    # Reject before invoking the loader: a preflight-only integration may be
    # intentionally incapable of importing or constructing runtime resources.
    if provider_identity.runtime_handle_mode != "inert_handle":
        raise RuntimeError(
            "Selected target runtime profile is preflight-only and cannot "
            "prepare a runtime handle."
        )

    provider_registry = TargetRuntimeProviderRegistry()
    provider_registry.register(binding.provider_entry)
    manifest = profile_preflight.resolved_manifest
    resolved_again = provider_registry.resolve_manifest(
        provider_identity.provider_id,
        config,
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
    )
    binding.validate_manifest(resolved_again)
    if resolved_again.semantic_content_hash != manifest.semantic_content_hash:
        raise ValueError(
            "Target runtime preparation config resolves another manifest than "
            "the selected profile preflight."
        )

    handle = provider_registry.create_handle(
        provider_identity.provider_id,
        config,
        manifest=resolved_again,
        binding_receipt=profile_preflight.binding_receipt,
    )
    initial_status = (
        handle.target_runtime_status_provider.capture_target_runtime_status()
    )
    return TargetRuntimeHandlePreparation(
        profile_preflight=profile_preflight,
        initial_status=initial_status,
        runtime_handle=handle,
    )


__all__ = [
    "TargetRuntimeHandlePreparation",
    "prepare_target_runtime_handle",
]
