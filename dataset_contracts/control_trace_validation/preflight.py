"""Assessment-profile and manifest preflight validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..control_trace import ControlTraceFact
from .context import ControlTraceValidationContext
from .issue_sink import ControlTraceLineageIssueCollector


class TraceRequirementProfileView(Protocol):
    profile_id: str
    implementation_status: str


@dataclass(frozen=True, slots=True)
class ControlTraceManifestSelection:
    execution_manifests: tuple[ControlTraceFact, ...]
    station_manifests: tuple[ControlTraceFact, ...]

    @property
    def execution_manifest(self) -> ControlTraceFact | None:
        return self.execution_manifests[0] if self.execution_manifests else None

    @property
    def station_manifest(self) -> ControlTraceFact | None:
        return self.station_manifests[0] if self.station_manifests else None


def select_control_trace_manifests(
    context: ControlTraceValidationContext,
) -> ControlTraceManifestSelection:
    return ControlTraceManifestSelection(
        execution_manifests=context.facts_of_type("execution_manifest"),
        station_manifests=context.facts_of_type("station_run_manifest"),
    )


def validate_profile_implementation(
    context: ControlTraceValidationContext,
    profile: TraceRequirementProfileView,
    manifests: ControlTraceManifestSelection,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    del context, manifests
    if profile.implementation_status == "planned":
        issues.add(
            "profile_not_implemented",
            profile.profile_id,
            "The profile is classified but has no accepted runtime fixture yet.",
        )


def validate_manifest_multiplicity(
    context: ControlTraceValidationContext,
    profile: TraceRequirementProfileView,
    manifests: ControlTraceManifestSelection,
    issues: ControlTraceLineageIssueCollector,
) -> None:
    del context, profile
    if len(manifests.execution_manifests) > 1:
        for duplicate in manifests.execution_manifests[1:]:
            issues.add(
                "multiple_execution_manifests",
                duplicate.fact_id,
                "One assessed control domain may bind only one execution manifest.",
                manifests.execution_manifests[0].fact_id,
            )
    if len(manifests.station_manifests) > 1:
        for duplicate in manifests.station_manifests[1:]:
            issues.add(
                "multiple_station_run_manifests",
                duplicate.fact_id,
                "One assessed Station runtime may bind only one run manifest.",
                manifests.station_manifests[0].fact_id,
            )


__all__ = [
    "ControlTraceManifestSelection",
    "TraceRequirementProfileView",
    "select_control_trace_manifests",
    "validate_manifest_multiplicity",
    "validate_profile_implementation",
]
