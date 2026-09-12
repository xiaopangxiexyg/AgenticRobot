"""Pure Control Station policy for RealTeleop progress presentation.

This module owns UI-facing event vocabulary and availability only.  It does
not import collection runtime objects, mutate a progress reducer, pause a
writer, or request robot Hold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dataset_contracts.progress_profile import TaskProgressProfileV1

PROGRESS_ADVANCE_EVENT = "progress_advance"
PROGRESS_UNDO_EVENT = "progress_undo"
PROGRESS_FAILURE_EVENT = "progress_failure"
PROGRESS_PROFILE_SELECT_EVENT = "progress_profile_select"
PROGRESS_ENTRY_STEP_SELECT_EVENT = "progress_entry_step_select"
PROGRESS_ARM_CONTROL_MODE_SELECT_EVENT = "progress_arm_control_mode_select"
PROGRESS_ARM_CONTROL_MODE_STRICT = "strict_profile"
PROGRESS_ARM_CONTROL_MODE_PROFILE = "profile"
PROGRESS_ARM_CONTROL_MODE_IGNORE = "ignore_profile"
PROGRESS_ARM_CONTROL_MODES = (
    PROGRESS_ARM_CONTROL_MODE_STRICT,
    PROGRESS_ARM_CONTROL_MODE_PROFILE,
    PROGRESS_ARM_CONTROL_MODE_IGNORE,
)


def normalize_progress_arm_control_mode(value: object) -> str:
    mode = str(value or PROGRESS_ARM_CONTROL_MODE_PROFILE).strip().lower()
    if mode not in PROGRESS_ARM_CONTROL_MODES:
        raise ValueError(f"Unsupported progress arm-control mode {mode!r}.")
    return mode


@dataclass(frozen=True)
class ProgressProfileOption:
    """One application-supplied managed profile shown by Control Station."""

    profile_id: str
    profile_version: str
    task_id: str
    display_name: str
    step_count: int

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "profile_version",
            "task_id",
            "display_name",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.step_count, int)
            or isinstance(self.step_count, bool)
            or self.step_count < 1
        ):
            raise ValueError("step_count must be a positive integer.")

    def to_payload(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "task_id": self.task_id,
            "display_name": self.display_name,
            "step_count": self.step_count,
        }


@dataclass(frozen=True)
class ProgressControlAvailability:
    """Rebuildable UI-facing availability derived from canonical state."""

    enabled: bool = False
    recording_active: bool = False
    phase: str = "ABSENT"
    allowed_next_step_ids: tuple[str, ...] = ()
    can_undo_last_advance: bool = False
    application_held: bool = False
    failure_promotes_existing_hold: bool = False

    @property
    def active_attempt(self) -> bool:
        return bool(
            self.enabled
            and self.recording_active
            and self.phase == "ACTIVE_ATTEMPT"
        )


class ProgressUiStateView(Protocol):
    """Minimal canonical state surface required by the presentation layer."""

    phase: str
    current_progress_step_id: str | None
    active_attempt_id: str | None


class ProgressUiEpisodeView(Protocol):
    """Structural view that keeps the Station independent of runtime classes."""

    @property
    def state(self) -> ProgressUiStateView: ...


def available_progress_controls(
    state: ProgressControlAvailability,
) -> tuple[str, ...]:
    """Return canonical event names; rendering remains a separate concern."""

    if not state.active_attempt:
        return ()
    if state.application_held:
        return (
            (PROGRESS_FAILURE_EVENT,)
            if state.failure_promotes_existing_hold
            else ()
        )
    controls = [PROGRESS_FAILURE_EVENT]
    if state.allowed_next_step_ids:
        controls.append(PROGRESS_ADVANCE_EVENT)
    if state.can_undo_last_advance:
        controls.append(PROGRESS_UNDO_EVENT)
    return tuple(controls)


def build_progress_ui_payload(
    *,
    gui_enabled: bool,
    configured: bool,
    availability: ProgressControlAvailability,
    profile_id: str = "",
    topology_kind: str = "",
    current_progress_step_id: str = "",
    current_progress_step_display_name: str = "",
    current_attempt_id: str = "",
    current_step_ordinal: int | None = None,
    total_step_count: int = 0,
    next_steps: Sequence[tuple[str, str]] = (),
    entry_steps: Sequence[tuple[str, str]] = (),
    selected_entry_step_id: str = "",
    entry_step_selection_enabled: bool = False,
    available_profiles: Sequence[ProgressProfileOption] = (),
    profile_selection_enabled: bool = False,
    profile_selection_locked: bool = False,
    profile_binding_source: str = "",
    arm_control_mode: str = PROGRESS_ARM_CONTROL_MODE_PROFILE,
    arm_control_mode_selection_enabled: bool = False,
    current_step_left_arm_control: str = "",
    current_step_right_arm_control: str = "",
    current_step_fixed_scope: str = "",
    message: str = "",
) -> dict[str, object]:
    """Build an ephemeral, rebuildable browser projection.

    This is deliberately not a dataset contract.  It contains display labels
    and control availability only; canonical progress facts remain owned by
    the collection progress store.
    """

    if not isinstance(availability, ProgressControlAvailability):
        raise TypeError("availability must be ProgressControlAvailability.")
    enabled = bool(gui_enabled)
    normalized_next_steps: list[dict[str, str]] = []
    for step_id, display_name in next_steps:
        normalized_id = str(step_id).strip()
        normalized_display = str(display_name).strip()
        if not normalized_id or not normalized_display:
            raise ValueError("Progress UI next-step entries must be non-empty.")
        normalized_next_steps.append(
            {
                "progress_step_id": normalized_id,
                "display_name": normalized_display,
            }
        )
    normalized_profiles: list[dict[str, object]] = []
    seen_profile_ids: set[str] = set()
    for option in available_profiles:
        if not isinstance(option, ProgressProfileOption):
            raise TypeError(
                "available_profiles must contain ProgressProfileOption values."
            )
        if option.profile_id in seen_profile_ids:
            raise ValueError("Progress profile options must have unique IDs.")
        seen_profile_ids.add(option.profile_id)
        normalized_profiles.append(option.to_payload())
    normalized_entry_steps: list[dict[str, str]] = []
    seen_entry_step_ids: set[str] = set()
    for step_id, display_name in entry_steps:
        normalized_id = str(step_id).strip()
        normalized_display = str(display_name).strip()
        if not normalized_id or not normalized_display:
            raise ValueError("Progress UI entry-step entries must be non-empty.")
        if normalized_id in seen_entry_step_ids:
            raise ValueError("Progress UI entry-step IDs must be unique.")
        seen_entry_step_ids.add(normalized_id)
        normalized_entry_steps.append(
            {
                "progress_step_id": normalized_id,
                "display_name": normalized_display,
            }
        )
    normalized_selected_entry = str(selected_entry_step_id).strip()
    if (
        normalized_selected_entry
        and normalized_selected_entry not in seen_entry_step_ids
    ):
        raise ValueError("Selected progress entry step is not available.")
    selection_enabled = bool(
        enabled
        and profile_selection_enabled
        and not profile_selection_locked
        and normalized_profiles
    )
    normalized_arm_control_mode = normalize_progress_arm_control_mode(
        arm_control_mode
    )
    arm_control_selection_enabled = bool(
        enabled and arm_control_mode_selection_enabled
    )
    entry_selection_enabled = bool(
        enabled
        and configured
        and entry_step_selection_enabled
        and normalized_entry_steps
    )
    active_controls = (
        list(available_progress_controls(availability))
        if enabled and configured
        else []
    )
    if selection_enabled:
        active_controls.append(PROGRESS_PROFILE_SELECT_EVENT)
    if entry_selection_enabled:
        active_controls.append(PROGRESS_ENTRY_STEP_SELECT_EVENT)
    if arm_control_selection_enabled:
        active_controls.append(PROGRESS_ARM_CONTROL_MODE_SELECT_EVENT)
    return {
        "enabled": enabled,
        "configured": bool(configured),
        "recording_active": bool(availability.recording_active),
        "phase": str(availability.phase),
        "profile_id": str(profile_id),
        "topology_kind": str(topology_kind),
        "current_progress_step_id": str(current_progress_step_id),
        "current_progress_step_display_name": str(
            current_progress_step_display_name
        ),
        "current_attempt_id": str(current_attempt_id),
        "current_step_ordinal": current_step_ordinal,
        "total_step_count": int(total_step_count),
        "next_steps": normalized_next_steps,
        "entry_steps": normalized_entry_steps,
        "selected_entry_step_id": normalized_selected_entry,
        "entry_step_selection_enabled": entry_selection_enabled,
        "available_profiles": normalized_profiles,
        "profile_selection_enabled": selection_enabled,
        "profile_selection_locked": bool(profile_selection_locked),
        "profile_binding_source": str(profile_binding_source),
        "arm_control_mode": normalized_arm_control_mode,
        "arm_control_modes": list(PROGRESS_ARM_CONTROL_MODES),
        "arm_control_mode_selection_enabled": arm_control_selection_enabled,
        "current_step_arm_control": {
            "left": str(current_step_left_arm_control),
            "right": str(current_step_right_arm_control),
            "fixed_scope": str(current_step_fixed_scope),
        },
        "can_undo_last_advance": bool(
            availability.can_undo_last_advance
        ),
        "application_held": bool(availability.application_held),
        "failure_promotes_existing_hold": bool(
            availability.failure_promotes_existing_hold
        ),
        "active_controls": active_controls,
        "message": str(message),
    }


def build_real_teleop_progress_ui_payload(
    *,
    profile: TaskProgressProfileV1 | None,
    episode: ProgressUiEpisodeView | None,
    availability: ProgressControlAvailability,
    gui_enabled: bool,
    available_profiles: Sequence[ProgressProfileOption] = (),
    profile_selection_enabled: bool = False,
    profile_selection_locked: bool = False,
    profile_binding_source: str = "",
    arm_control_mode: str = PROGRESS_ARM_CONTROL_MODE_PROFILE,
    arm_control_mode_selection_enabled: bool = False,
    selected_entry_step_id: str = "",
    entry_step_selection_enabled: bool = False,
) -> dict[str, object]:
    """Project canonical progress state into ephemeral RealTeleop UI state."""

    if profile is None:
        if not gui_enabled:
            profile_message = ""
        elif profile_selection_locked:
            if profile_binding_source == "populated_profileless_dataset":
                profile_message = (
                    "当前数据集已经写入普通 Episode，长程 progress 未启用；"
                    "普通数采不受阻塞。若要启用 profile，请切换到新的空数据集。"
                )
            else:
                profile_message = (
                    "当前数据集保持无长程 progress；普通数采不受阻塞。"
                    "若要启用 profile，请切换到新的空数据集。"
                )
        elif not profile_selection_enabled:
            profile_message = (
                "当前 Episode 正在准备、录制、暂停或保存；"
                "回到空闲后即可选择 profile。未选择 profile 也可以开始普通数采。"
            )
        else:
            profile_message = (
                "长程任务进度 profile 未启用（可选）；"
                "现在可选择任意 managed LINEAR profile，选择本身不会开始数采。"
                "未选择 profile 也可以开始普通数采。"
            )
        return build_progress_ui_payload(
            gui_enabled=gui_enabled,
            configured=False,
            availability=ProgressControlAvailability(),
            available_profiles=available_profiles,
            profile_selection_enabled=profile_selection_enabled,
            profile_selection_locked=profile_selection_locked,
            profile_binding_source=profile_binding_source,
            arm_control_mode=arm_control_mode,
            arm_control_mode_selection_enabled=(
                arm_control_mode_selection_enabled
            ),
            message=profile_message,
        )
    if not isinstance(profile, TaskProgressProfileV1):
        raise TypeError("profile must be TaskProgressProfileV1 or None.")
    if not isinstance(availability, ProgressControlAvailability):
        raise TypeError("availability must be ProgressControlAvailability.")

    steps_by_id = {
        step.progress_step_id: step
        for step in profile.steps
    }
    state = None if episode is None else episode.state
    normalized_entry_step_id = (
        str(selected_entry_step_id).strip() or profile.initial_step_id
    )
    if normalized_entry_step_id not in steps_by_id:
        raise ValueError("Selected progress entry step is not in the profile.")
    current_step_id = (
        normalized_entry_step_id
        if state is None or state.current_progress_step_id is None
        else state.current_progress_step_id
    )
    current_step = steps_by_id[current_step_id]
    next_steps = tuple(
        (
            step_id,
            steps_by_id[step_id].display_name,
        )
        for step_id in availability.allowed_next_step_ids
    )
    phase = "IDLE" if state is None else state.phase
    message_by_phase = {
        "IDLE": "开始正式数采后将从所选起始子任务进入；之后的推进、失败、撤销和标注流程保持不变。",
        "ACTIVE_ATTEMPT": "当前阶段正在采集；推进只结束当前阶段，不停止 episode。",
        "AWAITING_HOLD_RECEIPT": "失败已记录，正在等待机械臂 Hold 与正式采集暂停。",
        "PAUSED_NO_ACTIVE_ATTEMPT": (
            "失败阶段已关闭；人工调整后恢复遥操，首个有效写入样本会开始同阶段 Retry。"
        ),
        "RESUME_RECEIVED": "遥操与采集已恢复，正在等待 writer 接纳 Retry 第一帧。",
        "TERMINATED": "本 episode 的进度事实已终止。",
    }
    projected_availability = ProgressControlAvailability(
        enabled=bool(gui_enabled and availability.enabled),
        recording_active=availability.recording_active,
        phase=phase,
        allowed_next_step_ids=availability.allowed_next_step_ids,
        can_undo_last_advance=availability.can_undo_last_advance,
        application_held=availability.application_held,
        failure_promotes_existing_hold=(
            availability.failure_promotes_existing_hold
        ),
    )
    if projected_availability.failure_promotes_existing_hold:
        message = (
            "当前为普通 Hold；可将这次未恢复的原暂停边界记为当前阶段失败。"
            "不会重新暂停，也不会把人工调整时间写入失败 Attempt。"
        )
    elif projected_availability.application_held:
        message = (
            "当前 Hold 不具备可验证的原始正式暂停边界；"
            "阶段推进、撤销和失败记录均保持禁用。"
        )
    else:
        message = message_by_phase.get(
            phase,
            f"当前 progress phase: {phase}",
        )
        if (
            gui_enabled
            and not profile_selection_enabled
            and not profile_selection_locked
        ):
            message = (
                "当前 Episode 正在准备、录制、暂停或保存；"
                "task/profile 将在回到空闲后重新可选。"
            )
    return build_progress_ui_payload(
        gui_enabled=gui_enabled,
        configured=True,
        availability=projected_availability,
        profile_id=profile.profile_id,
        topology_kind=profile.topology_kind,
        current_progress_step_id=current_step_id,
        current_progress_step_display_name=current_step.display_name,
        current_attempt_id=(
            ""
            if state is None or state.active_attempt_id is None
            else state.active_attempt_id
        ),
        current_step_ordinal=current_step.ordinal,
        total_step_count=len(profile.steps),
        next_steps=next_steps,
        entry_steps=tuple(
            (step.progress_step_id, step.display_name)
            for step in profile.steps
        ),
        selected_entry_step_id=normalized_entry_step_id,
        entry_step_selection_enabled=entry_step_selection_enabled,
        available_profiles=available_profiles,
        profile_selection_enabled=profile_selection_enabled,
        profile_selection_locked=profile_selection_locked,
        profile_binding_source=profile_binding_source,
        arm_control_mode=arm_control_mode,
        arm_control_mode_selection_enabled=arm_control_mode_selection_enabled,
        current_step_left_arm_control=(
            "" if current_step.arm_control is None else current_step.arm_control.left
        ),
        current_step_right_arm_control=(
            "" if current_step.arm_control is None else current_step.arm_control.right
        ),
        current_step_fixed_scope=(
            ""
            if current_step.arm_control is None
            else current_step.arm_control.fixed_scope
        ),
        message=message,
    )


__all__ = [
    "PROGRESS_ADVANCE_EVENT",
    "PROGRESS_ARM_CONTROL_MODES",
    "PROGRESS_ARM_CONTROL_MODE_IGNORE",
    "PROGRESS_ARM_CONTROL_MODE_PROFILE",
    "PROGRESS_ARM_CONTROL_MODE_SELECT_EVENT",
    "PROGRESS_ARM_CONTROL_MODE_STRICT",
    "PROGRESS_ENTRY_STEP_SELECT_EVENT",
    "PROGRESS_FAILURE_EVENT",
    "PROGRESS_PROFILE_SELECT_EVENT",
    "PROGRESS_UNDO_EVENT",
    "ProgressControlAvailability",
    "ProgressProfileOption",
    "ProgressUiEpisodeView",
    "ProgressUiStateView",
    "available_progress_controls",
    "build_progress_ui_payload",
    "build_real_teleop_progress_ui_payload",
    "normalize_progress_arm_control_mode",
]
