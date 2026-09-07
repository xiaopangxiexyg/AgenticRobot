"""Pure RealTeleop decisions for profile-bound per-subtask arm control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .progress_contracts import (
    PROGRESS_ARM_CONTROL_MODE_IGNORE,
    PROGRESS_ARM_CONTROL_MODE_STRICT,
    ProgressMutationCommand,
    normalize_progress_arm_control_mode,
)
from dataset_contracts.progress_profile import (
    ProgressStepArmControl,
    TaskProgressProfileV1,
)

from ..collection.arm_mode_labels import build_arm_mode_metadata
from ..session import SessionStepResult


@dataclass(frozen=True)
class ProgressArmTransitionPlan:
    """One profile transition computed from canonical and actual arm state."""

    target_progress_step_id: str
    requirement: ProgressStepArmControl
    current_fixed_sides: tuple[str, ...]
    target_fixed_sides: tuple[str, ...]
    newly_teleop_sides: tuple[str, ...]

    @property
    def changes_arm_control(self) -> bool:
        return self.current_fixed_sides != self.target_fixed_sides

    @property
    def requires_home(self) -> bool:
        return bool(self.newly_teleop_sides)


@dataclass(frozen=True)
class PendingProgressArmTransition:
    """An entry/Advance/Undo waiting behind Hold and runtime Home."""

    intent: ProgressMutationCommand | None
    plan: ProgressArmTransitionPlan
    capture_sides: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.intent is not None and self.intent.kind not in {
            "ADVANCE_PROGRESS",
            "UNDO_LAST_ADVANCE",
        }:
            raise ValueError("Only Advance/Undo may wait on a profile arm transition.")
        capture_sides = tuple(self.capture_sides)
        invalid = set(capture_sides) - {"left", "right"}
        if invalid or not capture_sides:
            raise ValueError(
                "A pending profile arm transition requires non-empty capture_sides."
            )
        expected_sides = tuple(self.plan.newly_teleop_sides)
        if not self.plan.requires_home or capture_sides != expected_sides:
            raise ValueError(
                "A pending profile arm transition must capture exactly its "
                "newly_teleop_sides: "
                f"expected={expected_sides!r} actual={capture_sides!r}."
            )
        object.__setattr__(self, "capture_sides", capture_sides)


def validate_progress_arm_plan_fixed_scopes(
    plan: ProgressArmTransitionPlan,
    *,
    scope_for_side: Callable[[str], str],
) -> None:
    """Require the runtime FixArm topology to match the profile exactly."""

    for side in plan.target_fixed_sides:
        actual_scope = str(scope_for_side(side))
        if actual_scope != plan.requirement.fixed_scope:
            raise RuntimeError(
                "Profile fixed_hold scope does not match runtime FixArm: "
                f"expected={plan.requirement.fixed_scope!r} "
                f"actual={actual_scope!r} side={side}."
            )


def validate_progress_profile_fixed_scopes(
    profile: TaskProgressProfileV1,
    *,
    scope_for_side: Callable[[str], str],
) -> None:
    """Admit a complete profile against the live FixArm topology once.

    A transition-level check is still required at the final command boundary,
    but checking only the currently selected step made an incompatible profile
    fail repeatedly from the realtime loop as soon as the operator changed
    progress entry.  Validate every declared fixed side up front and report
    all mismatches together so the application can latch one actionable UI
    blocker.  Profiles with no ``arm_control`` declaration remain compatible
    in the default (non-strict) mode; strict-mode declaration validation is
    owned by :func:`build_progress_arm_transition_plan`.
    """

    if not isinstance(profile, TaskProgressProfileV1):
        raise TypeError("profile must be TaskProgressProfileV1.")
    if not callable(scope_for_side):
        raise TypeError("scope_for_side must be callable.")
    mismatches: list[str] = []
    for step in profile.steps:
        requirement = step.arm_control
        if requirement is None:
            continue
        for side in requirement.fixed_sides:
            try:
                actual_scope = str(scope_for_side(side))
            except Exception as exc:  # noqa: BLE001 - admission remains fail-closed.
                mismatches.append(
                    f"step={step.progress_step_id}:side={side}:"
                    f"scope lookup failed ({type(exc).__name__}: {exc})"
                )
                continue
            if actual_scope != requirement.fixed_scope:
                mismatches.append(
                    f"step={step.progress_step_id}:side={side}:"
                    f"expected={requirement.fixed_scope}:actual={actual_scope or 'missing'}"
                )
    if mismatches:
        raise RuntimeError(
            "Progress profile fixed_hold scope is incompatible with the live "
            "FixArm topology: " + "; ".join(mismatches)
        )


def progress_step_arm_requirement(
    profile: TaskProgressProfileV1,
    progress_step_id: str,
) -> ProgressStepArmControl | None:
    step = profile.step_by_id.get(str(progress_step_id))
    if step is None:
        raise ValueError(f"Unknown progress step {progress_step_id!r}.")
    return step.arm_control


def progress_step_arm_requirements_differ(
    profile: TaskProgressProfileV1,
    from_progress_step_id: str,
    to_progress_step_id: str,
) -> bool:
    """Compare profile declarations, independent of incidental runtime state."""

    return progress_step_arm_requirement(
        profile,
        from_progress_step_id,
    ) != progress_step_arm_requirement(
        profile,
        to_progress_step_id,
    )


def build_progress_arm_transition_plan(
    *,
    profile: TaskProgressProfileV1,
    target_progress_step_id: str,
    current_fixed_sides: tuple[str, ...],
    enforcement_mode: str,
) -> ProgressArmTransitionPlan | None:
    mode = normalize_progress_arm_control_mode(enforcement_mode)
    if mode == PROGRESS_ARM_CONTROL_MODE_IGNORE:
        return None
    requirement = progress_step_arm_requirement(profile, target_progress_step_id)
    if requirement is None:
        if mode == PROGRESS_ARM_CONTROL_MODE_STRICT:
            raise RuntimeError(
                "Strict profile arm-control mode requires every active progress "
                f"step to declare arm_control; missing={target_progress_step_id!r}."
            )
        return None
    normalized_current = tuple(
        side for side in ("left", "right") if side in set(current_fixed_sides)
    )
    target_fixed = requirement.fixed_sides
    newly_teleop = tuple(
        side for side in normalized_current if side not in set(target_fixed)
    )
    return ProgressArmTransitionPlan(
        target_progress_step_id=str(target_progress_step_id),
        requirement=requirement,
        current_fixed_sides=normalized_current,
        target_fixed_sides=target_fixed,
        newly_teleop_sides=newly_teleop,
    )


def build_progress_entry_arm_transition_plan(
    *,
    profile: TaskProgressProfileV1,
    selected_entry_step_id: object,
    current_fixed_sides: tuple[str, ...],
    enforcement_mode: str,
) -> ProgressArmTransitionPlan | None:
    """Bind the next-Episode entry selection to its exact arm restriction."""

    target_step_id = str(selected_entry_step_id or "").strip()
    if not target_step_id:
        target_step_id = profile.initial_step_id
    return build_progress_arm_transition_plan(
        profile=profile,
        target_progress_step_id=target_step_id,
        current_fixed_sides=current_fixed_sides,
        enforcement_mode=enforcement_mode,
    )


def strict_progress_arm_sample_rejection(
    *,
    profile: TaskProgressProfileV1,
    progress_step_id: str,
    step_result: SessionStepResult,
    active_sides: tuple[str, ...],
) -> str | None:
    """Return a stable writer-admission reason, or ``None`` when exact."""

    requirement = progress_step_arm_requirement(profile, progress_step_id)
    if requirement is None:
        return f"progress_step_arm_control_missing:{progress_step_id}"
    arm_mode = build_arm_mode_metadata(
        command=step_result.command,
        session_state=str(step_result.state.value),
        active_sides=active_sides,
        recording_paused=False,
    )
    raw_sides = arm_mode.get("sides")
    sides: Mapping[str, Any] = raw_sides if isinstance(raw_sides, Mapping) else {}
    for side in ("left", "right"):
        side_payload = sides.get(side)
        actual = (
            str(side_payload.get("mode", "unknown"))
            if isinstance(side_payload, Mapping)
            else "unknown"
        )
        expected = str(getattr(requirement, side))
        if actual != expected:
            return (
                "progress_step_arm_mode_mismatch:"
                f"step={progress_step_id}:side={side}:"
                f"expected={expected}:actual={actual}"
            )
        if expected == "fixed_hold":
            detail = (
                side_payload.get("detail")
                if isinstance(side_payload, Mapping)
                else None
            )
            scope = (
                str(detail.get("effector_scope", ""))
                if isinstance(detail, Mapping)
                else ""
            )
            if scope != requirement.fixed_scope:
                return (
                    "progress_step_fixed_scope_mismatch:"
                    f"step={progress_step_id}:side={side}:"
                    f"expected={requirement.fixed_scope}:actual={scope or 'missing'}"
                )
    return None


__all__ = [
    "PendingProgressArmTransition",
    "ProgressArmTransitionPlan",
    "build_progress_entry_arm_transition_plan",
    "build_progress_arm_transition_plan",
    "progress_step_arm_requirement",
    "progress_step_arm_requirements_differ",
    "strict_progress_arm_sample_rejection",
    "validate_progress_arm_plan_fixed_scopes",
    "validate_progress_profile_fixed_scopes",
]
