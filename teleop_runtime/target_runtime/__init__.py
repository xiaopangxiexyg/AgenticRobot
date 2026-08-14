"""Application-side target selection and inert-handle preparation.

This package composes the backend-neutral contracts from
``robot_motion.target_runtime`` for teleoperation applications.  It contains
no concrete hardware integration and importing it does not discover or load a
provider.
"""

from .application_suites import (
    EMBODIMENT_APPLICATION_SUITE_SPEC_HASH_DOMAIN_V1,
    EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1,
    EMBODIMENT_APPLICATION_SURFACE_SPEC_HASH_DOMAIN_V1,
    EMBODIMENT_APPLICATION_SURFACE_SPEC_SCHEMA_V1,
    EmbodimentApplicationSuiteBinding,
    EmbodimentApplicationSuiteCatalog,
    EmbodimentApplicationSuiteSpec,
    EmbodimentApplicationSurfaceLoader,
    EmbodimentApplicationSurfaceSpec,
    LoadedEmbodimentApplicationSurface,
)
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
    "EMBODIMENT_APPLICATION_SUITE_SPEC_HASH_DOMAIN_V1",
    "EMBODIMENT_APPLICATION_SUITE_SPEC_SCHEMA_V1",
    "EMBODIMENT_APPLICATION_SURFACE_SPEC_HASH_DOMAIN_V1",
    "EMBODIMENT_APPLICATION_SURFACE_SPEC_SCHEMA_V1",
    "TARGET_RUNTIME_PROFILE_BINDING_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROFILE_PREFLIGHT_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROFILE_SPEC_HASH_DOMAIN_V1",
    "TARGET_RUNTIME_PROFILE_SPEC_SCHEMA_V1",
    "EmbodimentApplicationSuiteBinding",
    "EmbodimentApplicationSuiteCatalog",
    "EmbodimentApplicationSuiteSpec",
    "EmbodimentApplicationSurfaceLoader",
    "EmbodimentApplicationSurfaceSpec",
    "LoadedEmbodimentApplicationSurface",
    "TargetRuntimeHandlePreparation",
    "TargetRuntimeProfileBinding",
    "TargetRuntimeProfileCatalog",
    "TargetRuntimeProfilePreflight",
    "TargetRuntimeProfileSpec",
    "prepare_target_runtime_handle",
]
