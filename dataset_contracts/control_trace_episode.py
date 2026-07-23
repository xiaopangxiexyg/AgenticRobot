"""Per-episode ControlTrace seal and additive v2.4 capture commit contracts.

The legacy dataset writer remains the episode lifecycle owner.  These records
only describe the immutable ControlTrace range attached to that external
episode and whether a v2.4 reader may expose it as a committed capability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping

from .canonical_encoding import canonical_content_hash


EPISODE_TRACE_RANGE_SCHEMA_V1 = "dexteleop_episode_trace_range_v1"
EPISODE_TRACE_SEAL_SCHEMA_V1 = "dexteleop_episode_trace_seal_v1"
V24_CAPTURE_COMMIT_SCHEMA_V1 = "dexteleop_v24_capture_commit_v1"

# Stable public capture layout. The episode metadata points at immutable
# sidecars, while the trace session identity selects the single session WAL.
CONTROL_TRACE_EPISODE_RELATIVE_ROOT_V1 = "meta/v24/control_trace/episodes"
CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1 = "meta/v24/control_trace/sessions"
EPISODE_SEAL_FILE_DOMAIN_V1 = "dexteleop.episode_trace_seal.file.v1"
EPISODE_COMMIT_FILE_DOMAIN_V1 = "dexteleop.v24_capture_commit.file.v1"

EPISODE_TRACE_RANGE_HASH_DOMAIN_V1 = "dexteleop.episode_trace_range.v1"
EPISODE_TRACE_SEAL_HASH_DOMAIN_V1 = "dexteleop.episode_trace_seal.v1"
V24_CAPTURE_COMMIT_HASH_DOMAIN_V1 = "dexteleop.v24_capture_commit.v1"
EPISODE_RANGE_RECORD_HASH_DOMAIN_V1 = "dexteleop.episode_trace_range.records.v1"
EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1 = "dexteleop.episode_trace.dependencies.v1"


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _timestamp(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _sha256(value: object, *, field_name: str) -> str:
    normalized = _required(value, field_name=field_name).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return normalized


def _string_tuple(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({_required(value, field_name="tuple item") for value in values}))


def _gap_tuple(values: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    previous_end = 0
    for raw_start, raw_end in sorted((int(start), int(end)) for start, end in values):
        if raw_start <= 0 or raw_end < raw_start:
            raise ValueError("Gap ranges must be positive inclusive intervals.")
        if raw_start <= previous_end:
            raise ValueError("Gap ranges must be disjoint and ordered.")
        result.append((raw_start, raw_end))
        previous_end = raw_end
    return tuple(result)


@dataclass(frozen=True)
class EpisodeTraceDependencyRef:
    fact_id: str
    journal_sequence: int
    record_integrity_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _required(self.fact_id, field_name="fact_id"))
        sequence = int(self.journal_sequence)
        if sequence <= 0:
            raise ValueError("Dependency journal_sequence must be positive.")
        object.__setattr__(self, "journal_sequence", sequence)
        object.__setattr__(
            self,
            "record_integrity_hash",
            _sha256(self.record_integrity_hash, field_name="record_integrity_hash"),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "journal_sequence": self.journal_sequence,
            "record_integrity_hash": self.record_integrity_hash,
        }


@dataclass(frozen=True)
class EpisodeTraceRange:
    trace_session_id: str
    runtime_session_id: str
    control_domain_id: str
    clock_domain_id: str
    episode_uuid: str
    lifecycle_owner_id: str
    binding_event_id: str
    unbinding_event_id: str
    binding_membership_epoch: int
    unbinding_membership_epoch: int
    first_journal_sequence: int
    last_journal_sequence: int
    bound_at: float
    unbound_at: float
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EPISODE_TRACE_RANGE_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "trace_session_id",
            "runtime_session_id",
            "control_domain_id",
            "clock_domain_id",
            "episode_uuid",
            "lifecycle_owner_id",
            "binding_event_id",
            "unbinding_event_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), field_name=name))
        bind_epoch = int(self.binding_membership_epoch)
        unbind_epoch = int(self.unbinding_membership_epoch)
        first = int(self.first_journal_sequence)
        last = int(self.last_journal_sequence)
        if bind_epoch < 0 or unbind_epoch <= bind_epoch:
            raise ValueError("Episode membership epochs must strictly advance.")
        if first <= 0 or last < first:
            raise ValueError("Episode journal range must be a positive inclusive interval.")
        bound_at = _timestamp(self.bound_at, field_name="bound_at")
        unbound_at = _timestamp(self.unbound_at, field_name="unbound_at")
        if unbound_at < bound_at:
            raise ValueError("Episode unbound_at cannot precede bound_at.")
        object.__setattr__(self, "binding_membership_epoch", bind_epoch)
        object.__setattr__(self, "unbinding_membership_epoch", unbind_epoch)
        object.__setattr__(self, "first_journal_sequence", first)
        object.__setattr__(self, "last_journal_sequence", last)
        object.__setattr__(self, "bound_at", bound_at)
        object.__setattr__(self, "unbound_at", unbound_at)
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=EPISODE_TRACE_RANGE_HASH_DOMAIN_V1),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "trace_session_id": self.trace_session_id,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "clock_domain_id": self.clock_domain_id,
            "episode_uuid": self.episode_uuid,
            "lifecycle_owner_id": self.lifecycle_owner_id,
            "binding_event_id": self.binding_event_id,
            "unbinding_event_id": self.unbinding_event_id,
            "binding_membership_epoch": self.binding_membership_epoch,
            "unbinding_membership_epoch": self.unbinding_membership_epoch,
            "first_journal_sequence": self.first_journal_sequence,
            "last_journal_sequence": self.last_journal_sequence,
            "bound_at": self.bound_at,
            "unbound_at": self.unbound_at,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class EpisodeTraceSealAssessment:
    assessment_id: str
    trace_range: EpisodeTraceRange
    profile_id: str
    profile_scope: str
    assessed_at: float
    observed_event_types: tuple[str, ...]
    missing_required_event_types: tuple[str, ...]
    observed_not_applicable_event_types: tuple[str, ...]
    admission_gap_ranges: tuple[tuple[int, int], ...]
    durability_gap_ranges: tuple[tuple[int, int], ...]
    lineage_issue_codes: tuple[str, ...]
    range_record_hash: str
    dependency_refs: tuple[EpisodeTraceDependencyRef, ...]
    dependency_bundle_hash: str
    trace_sealed: bool
    capture_commit_eligible: bool
    reason_codes: tuple[str, ...]
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=EPISODE_TRACE_SEAL_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "assessment_id", _required(self.assessment_id, field_name="assessment_id"))
        if not isinstance(self.trace_range, EpisodeTraceRange):
            raise TypeError("trace_range must be EpisodeTraceRange.")
        object.__setattr__(self, "profile_id", _required(self.profile_id, field_name="profile_id"))
        scope = _required(self.profile_scope, field_name="profile_scope")
        if scope not in {"runtime_diagnostic", "formal_capture"}:
            raise ValueError("Unsupported profile_scope.")
        object.__setattr__(self, "profile_scope", scope)
        object.__setattr__(self, "assessed_at", _timestamp(self.assessed_at, field_name="assessed_at"))
        for name in (
            "observed_event_types",
            "missing_required_event_types",
            "observed_not_applicable_event_types",
            "lineage_issue_codes",
            "reason_codes",
        ):
            object.__setattr__(self, name, _string_tuple(getattr(self, name)))
        object.__setattr__(self, "admission_gap_ranges", _gap_tuple(self.admission_gap_ranges))
        object.__setattr__(self, "durability_gap_ranges", _gap_tuple(self.durability_gap_ranges))
        object.__setattr__(self, "range_record_hash", _sha256(self.range_record_hash, field_name="range_record_hash"))
        raw_dependencies = tuple(self.dependency_refs)
        if any(not isinstance(item, EpisodeTraceDependencyRef) for item in raw_dependencies):
            raise TypeError("dependency_refs must contain EpisodeTraceDependencyRef values.")
        dependencies = tuple(
            sorted(
                raw_dependencies,
                key=lambda item: (item.journal_sequence, item.fact_id),
            )
        )
        if len({item.fact_id for item in dependencies}) != len(dependencies):
            raise ValueError("dependency_refs must have unique fact IDs.")
        object.__setattr__(self, "dependency_refs", dependencies)
        object.__setattr__(
            self,
            "dependency_bundle_hash",
            _sha256(self.dependency_bundle_hash, field_name="dependency_bundle_hash"),
        )
        sealed = bool(self.trace_sealed)
        eligible = bool(self.capture_commit_eligible)
        reasons = self.reason_codes
        if sealed != (not reasons):
            raise ValueError("trace_sealed must be true exactly when reason_codes is empty.")
        if eligible and (not sealed or scope != "formal_capture"):
            raise ValueError("Capture commit eligibility requires a sealed formal capture range.")
        object.__setattr__(self, "trace_sealed", sealed)
        object.__setattr__(self, "capture_commit_eligible", eligible)
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=EPISODE_TRACE_SEAL_HASH_DOMAIN_V1),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "assessment_id": self.assessment_id,
            "trace_range": self.trace_range.to_payload(),
            "profile_id": self.profile_id,
            "profile_scope": self.profile_scope,
            "assessed_at": self.assessed_at,
            "observed_event_types": self.observed_event_types,
            "missing_required_event_types": self.missing_required_event_types,
            "observed_not_applicable_event_types": self.observed_not_applicable_event_types,
            "admission_gap_ranges": self.admission_gap_ranges,
            "durability_gap_ranges": self.durability_gap_ranges,
            "lineage_issue_codes": self.lineage_issue_codes,
            "range_record_hash": self.range_record_hash,
            "dependency_refs": tuple(item.to_payload() for item in self.dependency_refs),
            "dependency_bundle_hash": self.dependency_bundle_hash,
            "trace_sealed": self.trace_sealed,
            "capture_commit_eligible": self.capture_commit_eligible,
            "reason_codes": self.reason_codes,
        }
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


@dataclass(frozen=True)
class V24EpisodeCaptureCommit:
    commit_id: str
    episode_uuid: str
    legacy_episode_index: int
    legacy_schema: str
    legacy_finalize_status: str
    trace_seal_assessment_id: str
    trace_seal_hash: str
    control_trace_seal_status: str
    capabilities: tuple[str, ...]
    committed_at: float
    committed_clock_domain_id: str
    capability_artifact_hash_by_name: Mapping[str, str] = field(
        default_factory=dict
    )
    semantic_content_hash: str = field(init=False)
    schema: str = field(default=V24_CAPTURE_COMMIT_SCHEMA_V1, init=False)

    def __post_init__(self) -> None:
        for name in (
            "commit_id",
            "episode_uuid",
            "legacy_schema",
            "trace_seal_assessment_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), field_name=name))
        if self.legacy_finalize_status != "finalized":
            raise ValueError("A v2.4 capture commit requires legacy finalize status finalized.")
        if self.control_trace_seal_status != "sealed":
            raise ValueError("A v2.4 capture commit requires ControlTrace seal status sealed.")
        index = int(self.legacy_episode_index)
        if index < 0:
            raise ValueError("legacy_episode_index must be non-negative.")
        object.__setattr__(self, "legacy_episode_index", index)
        object.__setattr__(self, "trace_seal_hash", _sha256(self.trace_seal_hash, field_name="trace_seal_hash"))
        capabilities = _string_tuple(self.capabilities)
        if not capabilities:
            raise ValueError("A v2.4 capture commit requires at least one capability.")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "committed_at", _timestamp(self.committed_at, field_name="committed_at"))
        object.__setattr__(
            self,
            "committed_clock_domain_id",
            _required(self.committed_clock_domain_id, field_name="committed_clock_domain_id"),
        )
        artifact_hashes = {
            _required(name, field_name="capability artifact name"): _sha256(
                artifact_hash,
                field_name="capability artifact hash",
            )
            for name, artifact_hash in dict(
                self.capability_artifact_hash_by_name
            ).items()
        }
        if not set(artifact_hashes).issubset(capabilities):
            raise ValueError(
                "Capability artifact hashes must reference declared capabilities."
            )
        if "native_streams" in capabilities and "native_streams" not in artifact_hashes:
            raise ValueError(
                "The native_streams capability requires an episode manifest hash."
            )
        object.__setattr__(
            self,
            "capability_artifact_hash_by_name",
            MappingProxyType(dict(sorted(artifact_hashes.items()))),
        )
        payload = self.to_payload(include_hash=False)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(payload, domain=V24_CAPTURE_COMMIT_HASH_DOMAIN_V1),
        )

    @classmethod
    def create(
        cls,
        *,
        commit_id: str,
        legacy_episode_index: int,
        legacy_schema: str,
        seal: EpisodeTraceSealAssessment,
        capabilities: Iterable[str],
        committed_at: float,
        committed_clock_domain_id: str = "local_wall_clock",
        capability_artifact_hash_by_name: Mapping[str, str] | None = None,
    ) -> "V24EpisodeCaptureCommit":
        if not seal.capture_commit_eligible:
            raise ValueError("ControlTrace seal is not eligible for a v2.4 capture commit.")
        return cls(
            commit_id=commit_id,
            episode_uuid=seal.trace_range.episode_uuid,
            legacy_episode_index=legacy_episode_index,
            legacy_schema=legacy_schema,
            legacy_finalize_status="finalized",
            trace_seal_assessment_id=seal.assessment_id,
            trace_seal_hash=seal.semantic_content_hash,
            control_trace_seal_status="sealed",
            capabilities=tuple(capabilities),
            committed_at=committed_at,
            committed_clock_domain_id=committed_clock_domain_id,
            capability_artifact_hash_by_name=dict(
                capability_artifact_hash_by_name or {}
            ),
        )

    def to_payload(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "commit_id": self.commit_id,
            "episode_uuid": self.episode_uuid,
            "legacy_episode_index": self.legacy_episode_index,
            "legacy_schema": self.legacy_schema,
            "legacy_finalize_status": self.legacy_finalize_status,
            "trace_seal_assessment_id": self.trace_seal_assessment_id,
            "trace_seal_hash": self.trace_seal_hash,
            "control_trace_seal_status": self.control_trace_seal_status,
            "capabilities": self.capabilities,
            "committed_at": self.committed_at,
            "committed_clock_domain_id": self.committed_clock_domain_id,
        }
        if self.capability_artifact_hash_by_name:
            payload["capability_artifact_hash_by_name"] = (
                self.capability_artifact_hash_by_name
            )
        if include_hash:
            payload["semantic_content_hash"] = self.semantic_content_hash
        return payload


__all__ = [
    "CONTROL_TRACE_EPISODE_RELATIVE_ROOT_V1",
    "CONTROL_TRACE_SESSION_RELATIVE_ROOT_V1",
    "EPISODE_COMMIT_FILE_DOMAIN_V1",
    "EPISODE_DEPENDENCY_BUNDLE_HASH_DOMAIN_V1",
    "EPISODE_RANGE_RECORD_HASH_DOMAIN_V1",
    "EPISODE_SEAL_FILE_DOMAIN_V1",
    "EPISODE_TRACE_RANGE_SCHEMA_V1",
    "EPISODE_TRACE_SEAL_SCHEMA_V1",
    "V24_CAPTURE_COMMIT_SCHEMA_V1",
    "EpisodeTraceDependencyRef",
    "EpisodeTraceRange",
    "EpisodeTraceSealAssessment",
    "V24EpisodeCaptureCommit",
]
