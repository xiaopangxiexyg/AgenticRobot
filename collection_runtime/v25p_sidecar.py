"""Compatibility imports for the frozen version-named sidecar module."""

from .compatibility.episode_sidecar import (
    V25P_TASK_EVENT_FILE_DOMAIN_V1,
    CompatibilityEpisodeSidecarFinalizeReport,
    CompatibilityEpisodeSidecarStore,
)


V25pEpisodeSidecarFinalizeReport = CompatibilityEpisodeSidecarFinalizeReport
V25pEpisodeSidecarStore = CompatibilityEpisodeSidecarStore
V25pV23CompatibilitySidecarStore = CompatibilityEpisodeSidecarStore


__all__ = [
    "V25P_TASK_EVENT_FILE_DOMAIN_V1",
    "V25pEpisodeSidecarFinalizeReport",
    "V25pEpisodeSidecarStore",
    "V25pV23CompatibilitySidecarStore",
]
