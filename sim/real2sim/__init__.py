"""Compatibility shim; RM65 real2sim evidence lives in ``sim.rm65``."""

from sim.rm65.real2sim import (
    REAL2SIM_ARTIFACT_VERSION,
    ArtifactRef,
    FidelityClaim,
    Real2SimArtifactManifest,
    ResolvedTiming,
    ValidationEvidence,
)

__all__ = [
    "REAL2SIM_ARTIFACT_VERSION",
    "ArtifactRef",
    "FidelityClaim",
    "Real2SimArtifactManifest",
    "ResolvedTiming",
    "ValidationEvidence",
]
