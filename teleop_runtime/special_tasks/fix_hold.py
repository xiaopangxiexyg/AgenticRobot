from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FixHoldTransition:
    changed: bool
    entered: bool
    exited: bool
    active: bool
    locked_sides: tuple[str, ...]
    source_by_side: dict[str, str]
    reason: str = ""


class SpecialArmFixHoldCoordinator:
    """Derive the auto-releasable dual-side hold from per-side special states.

    This coordinator has no command or hardware authority.  Fix/Init
    controllers continue holding their own sides; the runtime uses this state
    solely to pause collection and distinguish an automatic release from the
    explicit regular Hold workflow.
    """

    def __init__(self) -> None:
        self._active = False
        self._locked_sides: tuple[str, ...] = ()
        self._source_by_side: dict[str, str] = {}

    @property
    def active(self) -> bool:
        return bool(self._active)

    @property
    def locked_sides(self) -> tuple[str, ...]:
        return tuple(self._locked_sides)

    @property
    def source_by_side(self) -> dict[str, str]:
        return dict(self._source_by_side)

    @staticmethod
    def _fixed_sides(controller: Any, *, fallback_property: str) -> tuple[str, ...]:
        if controller is None:
            return ()
        hold_sides = getattr(controller, "fix_hold_sides", None)
        if hold_sides is not None:
            explicit_hold_sides = tuple(hold_sides or ())
            return tuple(
                side for side in ("left", "right") if side in explicit_hold_sides
            )
        explicit = tuple(getattr(controller, "fixed_sides", ()) or ())
        if explicit:
            return tuple(side for side in ("left", "right") if side in explicit)
        if bool(getattr(controller, "fixed", False)):
            side = str(getattr(controller, fallback_property, "") or "")
            return (side,) if side in {"left", "right"} else ()
        return ()

    def update(self, *, fix_controller: Any, init_controller: Any) -> FixHoldTransition:
        source_by_side: dict[str, str] = {}
        for side in self._fixed_sides(fix_controller, fallback_property="fixed_side"):
            source_by_side[side] = "fix_arm"
        for side in self._fixed_sides(init_controller, fallback_property="active_side"):
            source_by_side[side] = "init_fixed"
        locked_sides = tuple(side for side in ("left", "right") if side in source_by_side)
        next_active = len(locked_sides) == 2
        entered = bool(next_active and not self._active)
        exited = bool(self._active and not next_active)
        changed = bool(
            entered
            or exited
            or locked_sides != self._locked_sides
            or source_by_side != self._source_by_side
        )
        self._active = next_active
        self._locked_sides = locked_sides
        self._source_by_side = source_by_side
        return FixHoldTransition(
            changed=changed,
            entered=entered,
            exited=exited,
            active=next_active,
            locked_sides=locked_sides,
            source_by_side=dict(source_by_side),
            reason=(
                "both_sides_fixed_like"
                if next_active
                else "fixed_like_coverage_released"
                if exited
                else ""
            ),
        )

    def escalate_to_regular_hold(self) -> FixHoldTransition:
        previous_sides = self._locked_sides
        previous_sources = dict(self._source_by_side)
        was_active = self._active
        self._active = False
        self._locked_sides = ()
        self._source_by_side = {}
        return FixHoldTransition(
            changed=was_active,
            entered=False,
            exited=was_active,
            active=False,
            locked_sides=previous_sides,
            source_by_side=previous_sources,
            reason="escalated_to_regular_hold" if was_active else "",
        )


__all__ = ["FixHoldTransition", "SpecialArmFixHoldCoordinator"]
