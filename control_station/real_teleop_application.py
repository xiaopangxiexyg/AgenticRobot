"""Concrete Control Station binding for the lower RealTeleop progress core.

The pure UI policy stays in :mod:`control_station.real_teleop`. This concrete
application leaf is the only Station layer that binds those intents and
projections to real-teleop profile, recorder, and workflow ports.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

from collection_runtime.progress.profiles import (
    DatasetProgressProfileAdmission,
    inspect_dataset_progress_profile,
)
from teleop_runtime.real_teleop.progress_adapter import (
    list_real_teleop_managed_progress_profiles,
    select_real_teleop_managed_progress_profile,
    select_real_teleop_progress_entry_step,
)
from teleop_runtime.real_teleop.progress_contracts import (
    ProgressApplicationProjection,
    ProgressArmControlModeSelectionCommand,
    ProgressEntryStepSelectionCommand,
    ProgressEventBindings,
    ProgressMutationCommand,
    ProgressProfileSelectionCommand,
    PromotableHoldPause,
    RealTeleopProgressWorkflowPort,
)
from teleop_runtime.real_teleop.hitl_terminal import (
    HITLTerminalClutchInput,
    validate_hitl_terminal_keys,
)

from .real_teleop.progress import (
    PROGRESS_ADVANCE_EVENT,
    PROGRESS_FAILURE_EVENT,
    PROGRESS_UNDO_EVENT,
    ProgressControlAvailability,
    ProgressOperatorIntent,
    ProgressProfileOption,
    available_progress_controls,
    build_real_teleop_progress_ui_payload,
    progress_arm_control_mode_selection_intent_from_event,
    progress_entry_step_selection_intent_from_event,
    progress_intent_from_event,
    progress_profile_selection_intent_from_event,
)
from .real_teleop.progress_workflow import RealTeleopProgressWorkflow
from .real_teleop.hitl_application import (
    build_exclusive_hitl_real_teleop_extension,
)


class StationProgressWorkflowAdapter:
    """Expose a lower command port around the Station-owned coordinator."""

    def __init__(self, recorder: object) -> None:
        self._workflow = RealTeleopProgressWorkflow(recorder=recorder)

    def handle_command(
        self,
        command: ProgressMutationCommand,
        **kwargs: object,
    ):
        if not isinstance(command, ProgressMutationCommand):
            raise TypeError("command must be ProgressMutationCommand.")
        return self._workflow.handle_operator_intent(
            ProgressOperatorIntent(
                intent_id=command.command_id,
                kind=command.kind,
                target_progress_step_id=command.target_progress_step_id,
            ),
            **kwargs,
        )

    def __getattr__(self, attribute: str) -> object:
        return getattr(self._workflow, attribute)


def progress_command_from_intent(
    intent: ProgressOperatorIntent,
) -> ProgressMutationCommand:
    """Adapt a Station-owned intent to the lower real-teleop command."""

    if not isinstance(intent, ProgressOperatorIntent):
        raise TypeError("intent must be ProgressOperatorIntent.")
    return ProgressMutationCommand(
        command_id=intent.intent_id,
        kind=intent.kind,
        target_progress_step_id=intent.target_progress_step_id,
    )


@lru_cache(maxsize=32)
def list_real_teleop_managed_progress_profile_options(
    *,
    collection_task_id: str | None = None,
) -> tuple[ProgressProfileOption, ...]:
    """Build immutable Station options for managed LINEAR profiles."""

    return tuple(
        ProgressProfileOption(
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            task_id=profile.task_id,
            display_name=profile.profile_id,
            step_count=len(profile.steps),
        )
        for profile in list_real_teleop_managed_progress_profiles(
            collection_task_id=collection_task_id,
        )
    )


def _dataset_recorder_matches_progress_selection(
    *,
    args: argparse.Namespace,
    dataset_recorder: object | None,
) -> bool:
    if dataset_recorder is None:
        return False
    recorder_root = getattr(dataset_recorder, "dataset_root", None)
    storage_config = getattr(dataset_recorder, "writer_storage_config", None)
    recorder_task_id = getattr(storage_config, "task_id", None)
    if recorder_root is None or recorder_task_id is None:
        return False
    return bool(
        Path(recorder_root).expanduser().resolve()
        == Path(args.dataset_root).expanduser().resolve()
        and str(recorder_task_id) == str(args.collection_task_id)
    )


def build_real_teleop_startup_progress_ui_payload(
    *,
    args: argparse.Namespace,
    dataset_recorder: object | None,
    profile_selection_enabled: bool = True,
) -> dict[str, object]:
    """Build the pre-Home Station projection before the control loop exists."""

    recorder_matches = _dataset_recorder_matches_progress_selection(
        args=args,
        dataset_recorder=dataset_recorder,
    )
    if recorder_matches:
        profile = getattr(dataset_recorder, "progress_profile", None)
        profile_selection_locked = bool(
            getattr(
                dataset_recorder,
                "progress_profile_selection_locked",
                False,
            )
        )
        profile_binding_source = str(
            getattr(
                dataset_recorder,
                "progress_profile_binding_source",
                "",
            )
        )
    else:
        admission = inspect_dataset_progress_profile(
            Path(args.dataset_root),
            requested_profile=getattr(
                args,
                "collection_progress_profile",
                None,
            ),
        )
        profile = admission.profile
        profile_selection_locked = admission.locked
        profile_binding_source = admission.source
    available_profiles = list_real_teleop_managed_progress_profile_options(
        collection_task_id=str(args.collection_task_id),
    )
    payload = build_real_teleop_progress_ui_payload(
        profile=profile,
        episode=None,
        availability=ProgressControlAvailability(),
        gui_enabled=bool(getattr(args, "collection_progress_gui", False)),
        available_profiles=available_profiles,
        profile_selection_enabled=profile_selection_enabled,
        profile_selection_locked=profile_selection_locked,
        profile_binding_source=profile_binding_source,
        arm_control_mode=str(
            getattr(
                args,
                "collection_progress_arm_control_mode",
                "profile",
            )
        ),
        arm_control_mode_selection_enabled=profile_selection_enabled,
        selected_entry_step_id=str(
            getattr(args, "collection_progress_entry_step", "") or ""
        ),
        entry_step_selection_enabled=profile_selection_enabled,
    )
    if (
        bool(getattr(args, "collection_progress_gui", False))
        and profile is None
        and not profile_selection_locked
        and not available_profiles
    ):
        payload["message"] = (
            "当前没有可选的 managed LINEAR progress profile；"
            "请检查 progress profile catalog。"
        )
    if (
        bool(getattr(args, "collection_progress_gui", False))
        and not profile_selection_enabled
        and not profile_selection_locked
        and available_profiles
    ):
        payload["message"] = (
            "设备仍在启动；完成连接检查后即可在 runtime Home 前选择进度 profile。"
        )
    return payload


class StationProgressApplication:
    """Concrete Station leaf injected into the lower RealTeleop runtime."""

    _EVENT_BINDINGS = ProgressEventBindings(
        advance=PROGRESS_ADVANCE_EVENT,
        undo=PROGRESS_UNDO_EVENT,
        failure=PROGRESS_FAILURE_EVENT,
    )

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._selected_admission_key: tuple[Path, str | None] | None = None
        self._selected_admission: DatasetProgressProfileAdmission | None = None

    def _clear_selected_admission(self) -> None:
        self._selected_admission_key = None
        self._selected_admission = None

    def _inspect_selected_dataset_progress_profile(
        self,
    ) -> DatasetProgressProfileAdmission:
        """Inspect one stable pending selection at most once.

        The dataset/task selection transaction already performs authoritative
        admission before committing its ``args`` mutation.  This cache keeps
        the display-only control projection from repeating the legacy Episode
        scan on every frame while the recorder owns the previous selection.
        """

        root = Path(self._args.dataset_root).expanduser().resolve()
        requested_profile = getattr(
            self._args,
            "collection_progress_profile",
            None,
        )
        requested_profile_hash = (
            None
            if requested_profile is None
            else str(getattr(requested_profile, "semantic_content_hash", ""))
        )
        key = (root, requested_profile_hash)
        if self._selected_admission_key != key:
            admission = inspect_dataset_progress_profile(
                root,
                requested_profile=requested_profile,
            )
            self._selected_admission_key = key
            self._selected_admission = admission
        if self._selected_admission is None:
            raise RuntimeError(
                "selected dataset progress admission is unavailable"
            )
        return self._selected_admission

    @property
    def event_bindings(self) -> ProgressEventBindings:
        return self._EVENT_BINDINGS

    def build_progress_workflow(
        self,
        recorder: object,
    ) -> RealTeleopProgressWorkflowPort:
        return StationProgressWorkflowAdapter(recorder)

    def command_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressMutationCommand | None:
        intent = progress_intent_from_event(event, intent_id=command_id)
        return None if intent is None else progress_command_from_intent(intent)

    def profile_selection_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressProfileSelectionCommand | None:
        intent = progress_profile_selection_intent_from_event(
            event,
            intent_id=command_id,
        )
        if intent is None:
            return None
        return ProgressProfileSelectionCommand(
            command_id=intent.intent_id,
            profile_id=intent.profile_id,
        )

    def arm_control_mode_selection_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressArmControlModeSelectionCommand | None:
        intent = progress_arm_control_mode_selection_intent_from_event(
            event,
            intent_id=command_id,
        )
        if intent is None:
            return None
        return ProgressArmControlModeSelectionCommand(
            command_id=intent.intent_id,
            mode=intent.mode,
        )

    def entry_step_selection_from_event(
        self,
        *,
        event: str,
        command_id: str,
    ) -> ProgressEntryStepSelectionCommand | None:
        intent = progress_entry_step_selection_intent_from_event(
            event,
            intent_id=command_id,
        )
        if intent is None:
            return None
        return ProgressEntryStepSelectionCommand(
            command_id=intent.intent_id,
            progress_step_id=intent.progress_step_id,
        )

    def build_startup_progress_ui_payload(
        self,
        *,
        dataset_recorder: object | None,
        profile_selection_enabled: bool,
    ) -> dict[str, object]:
        return build_real_teleop_startup_progress_ui_payload(
            args=self._args,
            dataset_recorder=dataset_recorder,
            profile_selection_enabled=profile_selection_enabled,
        )

    def select_managed_progress_profile(
        self,
        *,
        dataset_recorder: object | None,
        profile_id: str,
    ):
        return select_real_teleop_managed_progress_profile(
            args=self._args,
            dataset_recorder=dataset_recorder,
            profile_id=profile_id,
        )

    def select_progress_entry_step(
        self,
        *,
        dataset_recorder: object | None,
        progress_step_id: str,
    ) -> str:
        return select_real_teleop_progress_entry_step(
            args=self._args,
            dataset_recorder=dataset_recorder,
            progress_step_id=progress_step_id,
        )

    def project_progress_application_state(
        self,
        *,
        dataset_recorder: object | None,
        progress_workflow: RealTeleopProgressWorkflowPort | None,
        promotable_hold_pause: PromotableHoldPause | None,
        application_held: bool,
        arm_transition_pending: bool,
        profile_selection_enabled: bool,
    ) -> ProgressApplicationProjection:
        recorder_matches = _dataset_recorder_matches_progress_selection(
            args=self._args,
            dataset_recorder=dataset_recorder,
        )
        if recorder_matches:
            # A matching recorder may bind a profile or seal Episodes.  Force
            # one fresh inspection if this application later returns to an
            # unmatched pending-selection state.
            self._clear_selected_admission()
            profile = getattr(dataset_recorder, "progress_profile", None)
            profile_selection_locked = bool(
                getattr(
                    dataset_recorder,
                    "progress_profile_selection_locked",
                    False,
                )
            )
            profile_binding_source = str(
                getattr(
                    dataset_recorder,
                    "progress_profile_binding_source",
                    "",
                )
            )
        else:
            admission = self._inspect_selected_dataset_progress_profile()
            profile = admission.profile
            profile_selection_locked = admission.locked
            profile_binding_source = admission.source

        episode = (
            None
            if dataset_recorder is None
            else getattr(dataset_recorder, "active_progress_episode", None)
        )
        availability = (
            progress_workflow.availability(
                application_held=application_held,
                promotable_hold_pause=promotable_hold_pause,
            )
            if progress_workflow is not None
            else ProgressControlAvailability()
        )
        active_controls = (
            available_progress_controls(availability)
            if profile is not None
            else ()
        )
        blocked_reason = ""
        if arm_transition_pending:
            active_controls = ()
            blocked_reason = "arm_transition_home_pending"

        payload = build_real_teleop_progress_ui_payload(
            profile=profile,
            episode=episode,
            availability=availability,
            gui_enabled=bool(
                getattr(self._args, "collection_progress_gui", False)
            ),
            available_profiles=(
                list_real_teleop_managed_progress_profile_options(
                    collection_task_id=str(
                        getattr(self._args, "collection_task_id", "") or ""
                    ),
                )
            ),
            profile_selection_enabled=profile_selection_enabled,
            profile_selection_locked=profile_selection_locked,
            profile_binding_source=profile_binding_source,
            arm_control_mode=str(
                getattr(
                    self._args,
                    "collection_progress_arm_control_mode",
                    "profile",
                )
            ),
            arm_control_mode_selection_enabled=profile_selection_enabled,
            selected_entry_step_id=str(
                getattr(
                    self._args,
                    "collection_progress_entry_step",
                    "",
                )
                or ""
            ),
            entry_step_selection_enabled=profile_selection_enabled,
        )
        return ProgressApplicationProjection(
            configured=profile is not None,
            active_controls=tuple(active_controls),
            recording_active=bool(availability.recording_active),
            phase=("IDLE" if episode is None else str(episode.state.phase)),
            application_held=bool(availability.application_held),
            allowed_next_step_ids=tuple(availability.allowed_next_step_ids),
            blocked_reason=blocked_reason,
            ui_payload=payload,
        )


def build_real_teleop_progress_application(
    args: argparse.Namespace,
) -> StationProgressApplication:
    """Build the canonical application leaf at the RealTeleop entry edge."""

    hitl_extension = None
    if bool(getattr(args, "hitl_policy_control", False)):
        # Keep the default RealTeleop import domain unchanged. The concrete
        # policy/composition consumer is loaded only for the explicit HITL
        # application profile.
        from teleop_runtime.real_teleop.hitl_policy_runtime import (
            ExclusiveHITLPolicyCommandSelector,
            build_normal_hitl_policy_source,
            preflight_normal_hitl_policy_source,
        )

        normal_source_config = (
            preflight_normal_hitl_policy_source(args)
            if str(getattr(args, "hitl_policy_mode", "normal")) == "normal"
            else None
        )
        hitl_extension = build_exclusive_hitl_real_teleop_extension(
            args,
            selector_factory=ExclusiveHITLPolicyCommandSelector,
            normal_policy_source_builder=build_normal_hitl_policy_source,
            normal_source_config=normal_source_config,
        )
    if hitl_extension is not None:
        existing_extension = getattr(
            args,
            "_real_teleop_runtime_extension",
            None,
        )
        if existing_extension is not None:
            raise RuntimeError(
                "Exclusive HITL and the Agent Primitive runtime cannot share "
                "the single RealTeleop application-extension slot."
            )
        takeover_key, resume_key = validate_hitl_terminal_keys(
            getattr(args, "hitl_takeover_key", "u"),
            getattr(args, "hitl_policy_resume_key", "i"),
        )

        def build_hitl_terminal(keyboard_source: object) -> HITLTerminalClutchInput:
            return HITLTerminalClutchInput(
                keyboard_source,
                takeover_key=takeover_key,
                policy_resume_key=resume_key,
            )

        args._real_teleop_runtime_extension = hitl_extension
        args._real_teleop_terminal_factory = build_hitl_terminal

    return StationProgressApplication(args)


__all__ = [
    "StationProgressApplication",
    "StationProgressWorkflowAdapter",
    "build_real_teleop_progress_application",
    "build_real_teleop_startup_progress_ui_payload",
    "list_real_teleop_managed_progress_profile_options",
    "progress_command_from_intent",
]
