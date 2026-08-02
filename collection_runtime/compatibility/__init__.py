"""Frozen compatibility persistence isolated from mainline capture runtime."""

from .episode_sidecar import (
    CompatibilityEpisodeSidecarFinalizeReport,
    CompatibilityEpisodeSidecarStore,
)


__all__ = [
    "CompatibilityEpisodeSidecarFinalizeReport",
    "CompatibilityEpisodeSidecarStore",
]
