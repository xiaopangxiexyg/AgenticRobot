"""Application-side target selection and inert-handle preparation.

This package composes the backend-neutral contracts from
``robot_motion.target_runtime`` for teleoperation applications.  It contains
no concrete hardware integration and importing it does not discover or load a
provider.
"""

from .preparation import (
    TargetRuntimeHandlePreparation,
    prepare_target_runtime_handle,
)
from .profiles import (
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
    "TargetRuntimeHandlePreparation",
    "TargetRuntimeProfileBinding",
    "TargetRuntimeProfileCatalog",
    "TargetRuntimeProfilePreflight",
    "TargetRuntimeProfileSpec",
    "prepare_target_runtime_handle",
]
