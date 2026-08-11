"""Orthogonal freshness assessment for source-intent trajectory previews.

Producer lock/speculation answers whether a source may rewrite a keyframe.
Freshness answers how old the observation basis is.  These are independent:
this module never changes preview trust classes and never infers observation
age from ``ComponentActionTrajectory.created_at``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from dataset_contracts.canonical_encoding import canonical_content_hash

from .preview import TrajectoryIntentPreviewSnapshot


TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN = "unknown"
TRAJECTORY_PREVIEW_FRESHNESS_FRESH = "fresh"
TRAJECTORY_PREVIEW_FRESHNESS_AGING = "aging"
TRAJECTORY_PREVIEW_FRESHNESS_STALE_DISPLAYABLE = "stale_displayable"
TRAJECTORY_PREVIEW_FRESHNESS_EXPIRED = "expired"
TRAJECTORY_PREVIEW_FRESHNESS_CLASSES_DEVELOPMENT = frozenset(
    {
        TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN,
        TRAJECTORY_PREVIEW_FRESHNESS_FRESH,
        TRAJECTORY_PREVIEW_FRESHNESS_AGING,
        TRAJECTORY_PREVIEW_FRESHNESS_STALE_DISPLAYABLE,
        TRAJECTORY_PREVIEW_FRESHNESS_EXPIRED,
    }
)
TRAJECTORY_PREVIEW_FRESHNESS_SPEC_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_preview_freshness_spec.development"
)
TRAJECTORY_PREVIEW_FRESHNESS_EVIDENCE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_preview_freshness_evidence.development"
)
TRAJECTORY_PREVIEW_FRESHNESS_ASSESSMENT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.trajectory_preview_freshness_assessment.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class TrajectoryPreviewFreshnessSpec:
    """Profile-selected display-age thresholds on a mapped clock."""

    spec_id: str
    fresh_max_age_sec: float
    aging_max_age_sec: float
    stale_displayable_max_age_sec: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        spec_id = _required(self.spec_id, name="spec_id")
        fresh = _finite(self.fresh_max_age_sec, name="fresh_max_age_sec")
        aging = _finite(self.aging_max_age_sec, name="aging_max_age_sec")
        stale = _finite(
            self.stale_displayable_max_age_sec,
            name="stale_displayable_max_age_sec",
        )
        if fresh < 0.0 or not fresh < aging < stale:
            raise ValueError(
                "Freshness thresholds must satisfy 0 <= fresh < aging < stale."
            )
        payload = {
            "spec_id": spec_id,
            "age_basis": "explicit_source_observation_time",
            "fresh_max_age_sec": fresh,
            "aging_max_age_sec": aging,
            "stale_displayable_max_age_sec": stale,
        }
        object.__setattr__(self, "spec_id", spec_id)
        object.__setattr__(self, "fresh_max_age_sec", fresh)
        object.__setattr__(self, "aging_max_age_sec", aging)
        object.__setattr__(self, "stale_displayable_max_age_sec", stale)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_PREVIEW_FRESHNESS_SPEC_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class TrajectoryPreviewFreshnessEvidence:
    """Explicit observation/availability facts for one exact trajectory."""

    trajectory_id: str
    trajectory_hash: str
    observation_fact_id: str
    observation_fact_hash: str
    observed_at: float
    observed_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    valid_until: float | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        trajectory_id = _required(self.trajectory_id, name="trajectory_id")
        trajectory_hash = _required(
            self.trajectory_hash,
            name="trajectory_hash",
        )
        fact_id = _required(
            self.observation_fact_id,
            name="observation_fact_id",
        )
        fact_hash = _required(
            self.observation_fact_hash,
            name="observation_fact_hash",
        )
        observed_at = _finite(self.observed_at, name="observed_at")
        observed_clock = _required(
            self.observed_clock_domain_id,
            name="observed_clock_domain_id",
        )
        available_at = _finite(self.available_at, name="available_at")
        available_clock = _required(
            self.available_clock_domain_id,
            name="available_clock_domain_id",
        )
        if observed_clock == available_clock and available_at < observed_at:
            raise ValueError("Trajectory evidence cannot arrive before observation.")
        valid_until = self.valid_until
        if valid_until is not None:
            valid_until = _finite(valid_until, name="valid_until")
            if valid_until <= available_at and available_clock == observed_clock:
                raise ValueError("Trajectory evidence expires before availability.")
        payload = {
            "trajectory_id": trajectory_id,
            "trajectory_hash": trajectory_hash,
            "observation_fact_id": fact_id,
            "observation_fact_hash": fact_hash,
            "observed_at": observed_at,
            "observed_clock_domain_id": observed_clock,
            "available_at": available_at,
            "available_clock_domain_id": available_clock,
            "valid_until": valid_until,
        }
        object.__setattr__(self, "trajectory_id", trajectory_id)
        object.__setattr__(self, "trajectory_hash", trajectory_hash)
        object.__setattr__(self, "observation_fact_id", fact_id)
        object.__setattr__(self, "observation_fact_hash", fact_hash)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "observed_clock_domain_id", observed_clock)
        object.__setattr__(self, "available_at", available_at)
        object.__setattr__(self, "available_clock_domain_id", available_clock)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_PREVIEW_FRESHNESS_EVIDENCE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def evidence_id(self) -> str:
        return f"trajectory-preview-freshness-evidence:{self.semantic_content_hash}"


@dataclass(frozen=True)
class TrajectoryPreviewFreshnessAssessment:
    """One display classification that does not alter preview identity."""

    preview_id: str
    preview_hash: str
    freshness_spec_hash: str
    evaluated_at: float
    evaluated_clock_domain_id: str
    freshness_class: str
    reason: str
    evidence_id: str | None = None
    evidence_hash: str | None = None
    observation_age_sec: float | None = None
    remaining_validity_sec: float | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        preview_id = _required(self.preview_id, name="preview_id")
        preview_hash = _required(self.preview_hash, name="preview_hash")
        spec_hash = _required(
            self.freshness_spec_hash,
            name="freshness_spec_hash",
        )
        evaluated_at = _finite(self.evaluated_at, name="evaluated_at")
        clock = _required(
            self.evaluated_clock_domain_id,
            name="evaluated_clock_domain_id",
        )
        freshness_class = _required(
            self.freshness_class,
            name="freshness_class",
        )
        if freshness_class not in TRAJECTORY_PREVIEW_FRESHNESS_CLASSES_DEVELOPMENT:
            raise ValueError("Unsupported trajectory preview freshness class.")
        reason = _required(self.reason, name="reason")
        evidence_id = self.evidence_id
        evidence_hash = self.evidence_hash
        if (evidence_id is None) != (evidence_hash is None):
            raise ValueError("Freshness evidence identity must be complete.")
        if evidence_id is not None:
            evidence_id = _required(evidence_id, name="evidence_id")
            evidence_hash = _required(evidence_hash, name="evidence_hash")
        age = self.observation_age_sec
        if age is not None:
            age = _finite(age, name="observation_age_sec")
            if age < 0.0:
                raise ValueError("observation_age_sec must be non-negative.")
        remaining = self.remaining_validity_sec
        if remaining is not None:
            remaining = _finite(remaining, name="remaining_validity_sec")
        if freshness_class == TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN and age is not None:
            raise ValueError("Unknown freshness cannot claim observation age.")
        payload = {
            "preview_id": preview_id,
            "preview_hash": preview_hash,
            "freshness_spec_hash": spec_hash,
            "evaluated_at": evaluated_at,
            "evaluated_clock_domain_id": clock,
            "freshness_class": freshness_class,
            "reason": reason,
            "evidence_id": evidence_id,
            "evidence_hash": evidence_hash,
            "observation_age_sec": age,
            "remaining_validity_sec": remaining,
        }
        object.__setattr__(self, "preview_id", preview_id)
        object.__setattr__(self, "preview_hash", preview_hash)
        object.__setattr__(self, "freshness_spec_hash", spec_hash)
        object.__setattr__(self, "evaluated_at", evaluated_at)
        object.__setattr__(self, "evaluated_clock_domain_id", clock)
        object.__setattr__(self, "freshness_class", freshness_class)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "evidence_hash", evidence_hash)
        object.__setattr__(self, "observation_age_sec", age)
        object.__setattr__(self, "remaining_validity_sec", remaining)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    TRAJECTORY_PREVIEW_FRESHNESS_ASSESSMENT_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def assessment_id(self) -> str:
        return f"trajectory-preview-freshness:{self.semantic_content_hash}"

    @property
    def displayable(self) -> bool:
        return self.freshness_class != TRAJECTORY_PREVIEW_FRESHNESS_EXPIRED


def assess_trajectory_intent_preview_freshness(
    preview: TrajectoryIntentPreviewSnapshot,
    *,
    spec: TrajectoryPreviewFreshnessSpec,
    evaluated_at: float,
    evaluated_clock_domain_id: str,
    evidence: TrajectoryPreviewFreshnessEvidence | None = None,
) -> TrajectoryPreviewFreshnessAssessment:
    """Classify explicit source observation age without mutating the preview."""

    if not isinstance(preview, TrajectoryIntentPreviewSnapshot):
        raise TypeError("preview must be TrajectoryIntentPreviewSnapshot.")
    if not isinstance(spec, TrajectoryPreviewFreshnessSpec):
        raise TypeError("spec must be TrajectoryPreviewFreshnessSpec.")
    evaluated = _finite(evaluated_at, name="evaluated_at")
    clock = _required(
        evaluated_clock_domain_id,
        name="evaluated_clock_domain_id",
    )
    common = {
        "preview_id": preview.preview_id,
        "preview_hash": preview.semantic_content_hash,
        "freshness_spec_hash": spec.semantic_content_hash,
        "evaluated_at": evaluated,
        "evaluated_clock_domain_id": clock,
    }
    if evidence is None:
        return TrajectoryPreviewFreshnessAssessment(
            **common,
            freshness_class=TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN,
            reason="explicit_observation_evidence_unavailable",
        )
    if not isinstance(evidence, TrajectoryPreviewFreshnessEvidence):
        raise TypeError("evidence must be typed or None.")
    if (
        evidence.trajectory_id != preview.trajectory.trajectory_id
        or evidence.trajectory_hash != preview.trajectory.semantic_content_hash
    ):
        raise ValueError("Freshness evidence belongs to another trajectory.")
    evidence_fields = {
        "evidence_id": evidence.evidence_id,
        "evidence_hash": evidence.semantic_content_hash,
    }
    if (
        evidence.observed_clock_domain_id != clock
        or evidence.available_clock_domain_id != clock
    ):
        return TrajectoryPreviewFreshnessAssessment(
            **common,
            **evidence_fields,
            freshness_class=TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN,
            reason="freshness_clock_unmapped",
        )
    if evaluated < evidence.available_at:
        return TrajectoryPreviewFreshnessAssessment(
            **common,
            **evidence_fields,
            freshness_class=TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN,
            reason="evidence_not_available_at_assessment",
        )
    age = evaluated - evidence.observed_at
    if age < 0.0:
        return TrajectoryPreviewFreshnessAssessment(
            **common,
            **evidence_fields,
            freshness_class=TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN,
            reason="observation_after_assessment",
        )
    remaining = (
        None
        if evidence.valid_until is None
        else evidence.valid_until - evaluated
    )
    if remaining is not None and remaining <= 0.0:
        freshness_class = TRAJECTORY_PREVIEW_FRESHNESS_EXPIRED
        reason = "explicit_validity_expired"
    elif age <= spec.fresh_max_age_sec:
        freshness_class = TRAJECTORY_PREVIEW_FRESHNESS_FRESH
        reason = "observation_within_fresh_window"
    elif age <= spec.aging_max_age_sec:
        freshness_class = TRAJECTORY_PREVIEW_FRESHNESS_AGING
        reason = "observation_within_aging_window"
    elif age <= spec.stale_displayable_max_age_sec:
        freshness_class = TRAJECTORY_PREVIEW_FRESHNESS_STALE_DISPLAYABLE
        reason = "observation_stale_but_displayable"
    else:
        freshness_class = TRAJECTORY_PREVIEW_FRESHNESS_EXPIRED
        reason = "observation_age_expired"
    return TrajectoryPreviewFreshnessAssessment(
        **common,
        **evidence_fields,
        freshness_class=freshness_class,
        reason=reason,
        observation_age_sec=age,
        remaining_validity_sec=remaining,
    )


__all__ = [
    "TRAJECTORY_PREVIEW_FRESHNESS_AGING",
    "TRAJECTORY_PREVIEW_FRESHNESS_CLASSES_DEVELOPMENT",
    "TRAJECTORY_PREVIEW_FRESHNESS_EXPIRED",
    "TRAJECTORY_PREVIEW_FRESHNESS_FRESH",
    "TRAJECTORY_PREVIEW_FRESHNESS_STALE_DISPLAYABLE",
    "TRAJECTORY_PREVIEW_FRESHNESS_UNKNOWN",
    "TrajectoryPreviewFreshnessAssessment",
    "TrajectoryPreviewFreshnessEvidence",
    "TrajectoryPreviewFreshnessSpec",
    "assess_trajectory_intent_preview_freshness",
]
