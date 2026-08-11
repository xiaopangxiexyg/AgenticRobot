"""Default-off Station display port for corrected trajectory previews.

The lower-owned supplier must capture and evaluate its coherent frozen cut.
Station only requests an already-typed, read-only snapshot when the selected
profile needs it. This module cannot admit, promote, or send an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from control_composition.trajectory import (
    TrajectoryCorrectedPreviewSnapshot,
    TrajectoryPreviewFreshnessAssessment,
    TrajectoryPreviewFreshnessEvidence,
    TrajectoryPreviewFreshnessSpec,
    assess_trajectory_intent_preview_freshness,
)


TRAJECTORY_CORRECTED_PREVIEW_PROFILE_OFF = "off"


@dataclass(frozen=True)
class TrajectoryCorrectedPreviewProfile:
    """Demand plan for one non-authoritative Station view."""

    profile_id: str = TRAJECTORY_CORRECTED_PREVIEW_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False
    freshness_spec: TrajectoryPreviewFreshnessSpec | None = None

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise ValueError("profile_id must be non-empty.")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("Corrected trajectory preview cannot send actions.")
        if self.enabled and profile_id == TRAJECTORY_CORRECTED_PREVIEW_PROFILE_OFF:
            raise ValueError("An enabled corrected-preview profile needs an id.")
        if self.freshness_spec is not None and not isinstance(
            self.freshness_spec,
            TrajectoryPreviewFreshnessSpec,
        ):
            raise TypeError("freshness_spec must be typed or None.")
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class TrajectoryCorrectedPreviewCycle:
    profile_id: str
    status: str
    preview: TrajectoryCorrectedPreviewSnapshot | None = None
    freshness: TrajectoryPreviewFreshnessAssessment | None = None

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        status = str(self.status).strip()
        if not profile_id or status not in {"disabled", "projected"}:
            raise ValueError("Corrected trajectory preview cycle is invalid.")
        if status == "disabled":
            if self.preview is not None or self.freshness is not None:
                raise ValueError("A disabled corrected-preview cycle has no output.")
        elif not isinstance(self.preview, TrajectoryCorrectedPreviewSnapshot):
            raise TypeError("A projected cycle requires a corrected preview.")
        if self.freshness is not None:
            if not isinstance(
                self.freshness,
                TrajectoryPreviewFreshnessAssessment,
            ):
                raise TypeError("freshness must be typed or None.")
            assert self.preview is not None
            intent = self.preview.intent_preview
            if (
                self.freshness.preview_id != intent.preview_id
                or self.freshness.preview_hash != intent.semantic_content_hash
            ):
                raise ValueError("Freshness assessment belongs to another preview.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)


class TrajectoryCorrectedPreviewController:
    """Lazily expose a lower-owned corrected snapshot to Station consumers."""

    def __init__(
        self,
        *,
        profile: TrajectoryCorrectedPreviewProfile,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        if not isinstance(profile, TrajectoryCorrectedPreviewProfile):
            raise TypeError("profile must be TrajectoryCorrectedPreviewProfile.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        normalized_clock = str(clock_domain_id).strip()
        if not normalized_clock:
            raise ValueError("clock_domain_id must be non-empty.")
        self.profile = profile
        self.clock = clock
        self.clock_domain_id = normalized_clock

    def run_preview_cycle(
        self,
        preview_factory: Callable[[], TrajectoryCorrectedPreviewSnapshot],
        *,
        freshness_evidence_factory: (
            Callable[[], TrajectoryPreviewFreshnessEvidence | None] | None
        ) = None,
    ) -> TrajectoryCorrectedPreviewCycle:
        """Do not evaluate either supplier while the profile is disabled."""

        if not callable(preview_factory):
            raise TypeError("preview_factory must be callable.")
        if not self.profile.enabled:
            return TrajectoryCorrectedPreviewCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        preview = preview_factory()
        if not isinstance(preview, TrajectoryCorrectedPreviewSnapshot):
            raise TypeError(
                "preview_factory must return TrajectoryCorrectedPreviewSnapshot."
            )
        freshness = None
        if self.profile.freshness_spec is not None:
            evidence = (
                None
                if freshness_evidence_factory is None
                else freshness_evidence_factory()
            )
            if evidence is not None and not isinstance(
                evidence,
                TrajectoryPreviewFreshnessEvidence,
            ):
                raise TypeError(
                    "freshness_evidence_factory must return typed evidence or None."
                )
            freshness = assess_trajectory_intent_preview_freshness(
                preview.intent_preview,
                spec=self.profile.freshness_spec,
                evaluated_at=float(self.clock()),
                evaluated_clock_domain_id=self.clock_domain_id,
                evidence=evidence,
            )
        return TrajectoryCorrectedPreviewCycle(
            profile_id=self.profile.profile_id,
            status="projected",
            preview=preview,
            freshness=freshness,
        )


__all__ = [
    "TRAJECTORY_CORRECTED_PREVIEW_PROFILE_OFF",
    "TrajectoryCorrectedPreviewController",
    "TrajectoryCorrectedPreviewCycle",
    "TrajectoryCorrectedPreviewProfile",
]
