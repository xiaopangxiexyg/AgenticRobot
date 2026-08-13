"""Compatibility exports for target-runtime profile selection.

The canonical implementation lives in :mod:`teleop_runtime.target_runtime`.
Keep this module importable while application callers migrate; it owns no
profile registry, provider loader, or runtime state.
"""

from .target_runtime.profiles import (
    TARGET_RUNTIME_PROFILE_BINDING_HASH_DOMAIN_V1,
    TARGET_RUNTIME_PROFILE_PREFLIGHT_HASH_DOMAIN_V1,
    TARGET_RUNTIME_PROFILE_SPEC_HASH_DOMAIN_V1,
    TARGET_RUNTIME_PROFILE_SPEC_SCHEMA_V1,
    TargetRuntimeProfileBinding,
    TargetRuntimeProfileCatalog,
    TargetRuntimeProfilePreflight,
    TargetRuntimeProfileSpec,
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
