"""Default-off Station gate for generic trajectory composition shadow cycles.

The Station layer owns only the application profile switch and lazy invocation.
It does not own contributor production, trajectory/replay cursors, composition
math, authority, execution targets, references, safety, or dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from control_composition.events import CompositionDomainEvent
from control_composition.trajectory import (
    TrajectoryCompositionShadowResult,
    trajectory_composition_shadow_domain_event,
)


TRAJECTORY_COMPOSITION_SHADOW_PROFILE_OFF = "off"


@dataclass(frozen=True)
class TrajectoryCompositionShadowProfile:
    """Application switch for a non-authoritative preview-only path."""

    profile_id: str = TRAJECTORY_COMPOSITION_SHADOW_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        if not profile_id:
            raise ValueError("profile_id must be non-empty.")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        enabled = bool(self.enabled)
        send_actions = bool(self.send_actions)
        if send_actions:
            raise ValueError(
                "Trajectory composition shadow does not support send_actions=true."
            )
        if enabled and profile_id == TRAJECTORY_COMPOSITION_SHADOW_PROFILE_OFF:
            raise ValueError("An enabled trajectory shadow profile needs an id.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "send_actions", send_actions)


@dataclass(frozen=True)
class TrajectoryCompositionShadowCycle:
    """One Station-level lazy-cycle receipt."""

    profile_id: str
    status: str
    evaluation: TrajectoryCompositionShadowResult | None = None
    domain_event: CompositionDomainEvent | None = None

    def __post_init__(self) -> None:
        profile_id = str(self.profile_id).strip()
        status = str(self.status).strip()
        if not profile_id or status not in {"disabled", "evaluated"}:
            raise ValueError("Trajectory shadow cycle identity/status is invalid.")
        if status == "disabled":
            if self.evaluation is not None or self.domain_event is not None:
                raise ValueError("Disabled shadow cycle cannot carry an evaluation.")
        else:
            if not isinstance(self.evaluation, TrajectoryCompositionShadowResult):
                raise TypeError("Evaluated cycle requires a trajectory shadow result.")
            if not isinstance(self.domain_event, CompositionDomainEvent):
                raise TypeError("Evaluated cycle requires its summary event.")
            if self.domain_event.subject is not self.evaluation:
                raise ValueError("Shadow cycle event/evaluation mismatch.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)


class TrajectoryCompositionShadowController:
    """Invoke an externally prepared shadow evaluation only when enabled."""

    def __init__(
        self,
        *,
        profile: TrajectoryCompositionShadowProfile,
        clock: Callable[[], float],
        event_emitter: Callable[[tuple[CompositionDomainEvent, ...]], object],
    ) -> None:
        if not isinstance(profile, TrajectoryCompositionShadowProfile):
            raise TypeError("profile must be TrajectoryCompositionShadowProfile.")
        if not callable(clock) or not callable(event_emitter):
            raise TypeError("clock and event_emitter must be callable.")
        self.profile = profile
        self.clock = clock
        self.event_emitter = event_emitter

    def run_shadow_cycle(
        self,
        evaluation_factory: Callable[[], TrajectoryCompositionShadowResult],
    ) -> TrajectoryCompositionShadowCycle:
        """Run lazily so default-off does not even fetch source payloads."""

        if not callable(evaluation_factory):
            raise TypeError("evaluation_factory must be callable.")
        if not self.profile.enabled:
            return TrajectoryCompositionShadowCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        evaluation = evaluation_factory()
        if not isinstance(evaluation, TrajectoryCompositionShadowResult):
            raise TypeError(
                "evaluation_factory must return TrajectoryCompositionShadowResult."
            )
        if evaluation.authoritative or evaluation.proposal_admitted:
            raise ValueError("Station shadow cycle received an authoritative result.")
        if evaluation.executable_action is not None:
            raise ValueError("Station shadow cycle cannot carry executable action.")
        event = trajectory_composition_shadow_domain_event(
            evaluation,
            occurred_at=float(self.clock()),
        )
        self.event_emitter((event,))
        return TrajectoryCompositionShadowCycle(
            profile_id=self.profile.profile_id,
            status="evaluated",
            evaluation=evaluation,
            domain_event=event,
        )


__all__ = [
    "TRAJECTORY_COMPOSITION_SHADOW_PROFILE_OFF",
    "TrajectoryCompositionShadowController",
    "TrajectoryCompositionShadowCycle",
    "TrajectoryCompositionShadowProfile",
]
