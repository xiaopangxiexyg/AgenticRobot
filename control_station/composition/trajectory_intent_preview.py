"""Default-off Station port for generic future source-intent visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from control_composition.trajectory import (
    TrajectoryIntentPreviewSnapshot,
    TrajectoryPreviewFreshnessAssessment,
    TrajectoryPreviewFreshnessEvidence,
    TrajectoryPreviewFreshnessSpec,
    assess_trajectory_intent_preview_freshness,
    build_trajectory_intent_preview,
)
from robot_motion.action.trajectory import ComponentActionTrajectory


TRAJECTORY_INTENT_PREVIEW_PROFILE_OFF = "off"


@dataclass(frozen=True)
class TrajectoryIntentPreviewProfile:
    """Application switch; this port can never send an action."""

    profile_id: str = TRAJECTORY_INTENT_PREVIEW_PROFILE_OFF
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
            raise ValueError("Trajectory intent preview cannot send actions.")
        if self.freshness_spec is not None and not isinstance(
            self.freshness_spec,
            TrajectoryPreviewFreshnessSpec,
        ):
            raise TypeError("freshness_spec must be typed or None.")
        if self.enabled and profile_id == TRAJECTORY_INTENT_PREVIEW_PROFILE_OFF:
            raise ValueError("An enabled intent-preview profile needs an id.")
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class TrajectoryIntentPreviewCycle:
    profile_id: str
    status: str
    preview: TrajectoryIntentPreviewSnapshot | None = None
    freshness: TrajectoryPreviewFreshnessAssessment | None = None

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        status = str(self.status).strip()
        if not profile_id or status not in {"disabled", "projected"}:
            raise ValueError("Trajectory intent preview cycle is invalid.")
        if status == "disabled" and (
            self.preview is not None or self.freshness is not None
        ):
            raise ValueError("A disabled preview cycle cannot carry output.")
        if status == "projected" and not isinstance(
            self.preview,
            TrajectoryIntentPreviewSnapshot,
        ):
            raise TypeError("A projected cycle requires its preview snapshot.")
        if self.freshness is not None:
            if not isinstance(
                self.freshness,
                TrajectoryPreviewFreshnessAssessment,
            ):
                raise TypeError("freshness must be typed or None.")
            assert self.preview is not None
            if (
                self.freshness.preview_id != self.preview.preview_id
                or self.freshness.preview_hash
                != self.preview.semantic_content_hash
            ):
                raise ValueError("Freshness assessment belongs to another preview.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)


class TrajectoryIntentPreviewController:
    """Lazily project an already-produced trajectory for a Station consumer."""

    def __init__(
        self,
        *,
        profile: TrajectoryIntentPreviewProfile,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        if not isinstance(profile, TrajectoryIntentPreviewProfile):
            raise TypeError("profile must be TrajectoryIntentPreviewProfile.")
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
        trajectory_factory: Callable[[], ComponentActionTrajectory],
        *,
        freshness_evidence_factory: (
            Callable[[], TrajectoryPreviewFreshnessEvidence | None] | None
        ) = None,
    ) -> TrajectoryIntentPreviewCycle:
        """Fetch only information explicitly requested by the active profile."""

        if not callable(trajectory_factory):
            raise TypeError("trajectory_factory must be callable.")
        if not self.profile.enabled:
            return TrajectoryIntentPreviewCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        trajectory = trajectory_factory()
        if not isinstance(trajectory, ComponentActionTrajectory):
            raise TypeError("trajectory_factory must return ComponentActionTrajectory.")
        evaluated_at = float(self.clock())
        preview = build_trajectory_intent_preview(
            trajectory,
            generated_at=evaluated_at,
            generated_clock_domain_id=self.clock_domain_id,
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
                preview,
                spec=self.profile.freshness_spec,
                evaluated_at=evaluated_at,
                evaluated_clock_domain_id=self.clock_domain_id,
                evidence=evidence,
            )
        return TrajectoryIntentPreviewCycle(
            profile_id=self.profile.profile_id,
            status="projected",
            preview=preview,
            freshness=freshness,
        )


__all__ = [
    "TRAJECTORY_INTENT_PREVIEW_PROFILE_OFF",
    "TrajectoryIntentPreviewController",
    "TrajectoryIntentPreviewCycle",
    "TrajectoryIntentPreviewProfile",
]
