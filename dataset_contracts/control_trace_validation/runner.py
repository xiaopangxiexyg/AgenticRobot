"""Explicit validator ordering for migrated ControlTrace rule families."""

from __future__ import annotations

from collections.abc import Callable

from .context import ControlTraceValidationContext
from .context_lineage import validate_context_lineage
from .issue_sink import ControlTraceLineageIssueCollector
from .membership import validate_episode_membership
from .preflight import (
    ControlTraceManifestSelection,
    TraceRequirementProfileView,
    select_control_trace_manifests,
    validate_manifest_multiplicity,
    validate_profile_implementation,
)
from .station import validate_station_lineage
from .source_eligibility import validate_composition_source_eligibility


FoundationValidator = Callable[
    [
        ControlTraceValidationContext,
        TraceRequirementProfileView,
        ControlTraceManifestSelection,
        ControlTraceLineageIssueCollector,
    ],
    None,
]

FOUNDATION_VALIDATORS: tuple[FoundationValidator, ...] = (
    validate_profile_implementation,
    validate_manifest_multiplicity,
    validate_station_lineage,
    validate_episode_membership,
    validate_context_lineage,
    validate_composition_source_eligibility,
)


def run_control_trace_foundation(
    context: ControlTraceValidationContext,
    profile: TraceRequirementProfileView,
    issues: ControlTraceLineageIssueCollector,
) -> ControlTraceManifestSelection:
    manifests = select_control_trace_manifests(context)
    for validator in FOUNDATION_VALIDATORS:
        validator(context, profile, manifests, issues)
    return manifests


__all__ = ["FOUNDATION_VALIDATORS", "run_control_trace_foundation"]
