"""Pure RealTeleop decisions for profile-bound per-subtask arm control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .progress_contracts import (
    PROGRESS_ARM_CONTROL_MODE_IGNORE,
    PROGRESS_ARM_CONTROL_MODE_PROFILE,
    PROGRESS_ARM_CONTROL_MODE_STRICT,
    ProgressMutationCommand,
    normalize_progress_arm_control_mode,
)
from dataset_contracts.progress_profile import (
    ProgressStepArmControlV1,
    TaskProgressProfileV1,
)

from ..collection.arm_mode_labels import build_arm_mode_metadata
from ..session import SessionStepResult


@dataclass(frozen=True)
class ProgressArmTransitionPlan:
    """One profile transition computed from canonical and actual arm state."""

    target_progress_step_id: str
    requirement: ProgressStepArmControlV1
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
        invalid = set(self.capture_sides) - {"left", "right"}
        if invalid or not self.capture_sides:
            raise ValueError(
                "A pending profile arm transition requires non-empty capture_sides."
            )


def progress_step_arm_requirement(
    profile: TaskProgressProfileV1,
    progress_step_id: str,
) -> ProgressStepArmControlV1 | None:
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
    "build_progress_arm_transition_plan",
    "progress_step_arm_requirement",
    "progress_step_arm_requirements_differ",
    "strict_progress_arm_sample_rejection",
]
