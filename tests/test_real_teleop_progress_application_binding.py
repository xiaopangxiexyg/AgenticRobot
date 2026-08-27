from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import control_station.real_teleop_application as real_teleop_application
from control_station.real_teleop.progress import (
    PROGRESS_ARM_CONTROL_MODES as STATION_ARM_CONTROL_MODES,
    ProgressOperatorIntent,
    normalize_progress_arm_control_mode as normalize_station_arm_control_mode,
)
from control_station.real_teleop_application import (
    build_real_teleop_progress_application,
    progress_command_from_intent,
)
from teleop_runtime.real_teleop.progress_contracts import (
    ProgressArmControlModeSelectionCommand,
    ProgressEntryStepSelectionCommand,
    PROGRESS_ARM_CONTROL_MODES as RUNTIME_ARM_CONTROL_MODES,
    ProgressMutationCommand,
    ProgressProfileSelectionCommand,
    normalize_progress_arm_control_mode as normalize_runtime_arm_control_mode,
)
from teleop_runtime.real_teleop.startup_progress_runtime import (
    RealTeleopStartupProgressRuntime,
)


def test_station_intent_is_adapted_to_one_ui_neutral_runtime_command() -> None:
    intent = ProgressOperatorIntent(
        intent_id="station:advance:7",
        kind="ADVANCE_PROGRESS",
        target_progress_step_id="place",
    )

    assert progress_command_from_intent(intent) == ProgressMutationCommand(
        command_id="station:advance:7",
        kind="ADVANCE_PROGRESS",
        target_progress_step_id="place",
    )


def test_station_progress_application_returns_only_lower_commands() -> None:
    application = build_real_teleop_progress_application(SimpleNamespace())

    assert application.command_from_event(
        event="progress_failure",
        command_id="station:failure:9",
    ) == ProgressMutationCommand(
        command_id="station:failure:9",
        kind="REPORT_FAILURE",
    )
    assert application.profile_selection_from_event(
        event="progress_profile_select:writing-v1",
        command_id="station:profile:3",
    ) == ProgressProfileSelectionCommand(
        command_id="station:profile:3",
        profile_id="writing-v1",
    )
    assert application.entry_step_selection_from_event(
        event="progress_entry_step_select:right_write",
        command_id="station:entry:4",
    ) == ProgressEntryStepSelectionCommand(
        command_id="station:entry:4",
        progress_step_id="right_write",
    )


def test_station_progress_application_projects_rebuildable_absent_state(
    tmp_path: Path,
) -> None:
    application = build_real_teleop_progress_application(
        SimpleNamespace(
            dataset_root=tmp_path,
            collection_progress_profile=None,
            collection_task_id="fixture-task-without-managed-profile",
            collection_progress_gui=True,
            collection_progress_arm_control_mode="profile",
        )
    )

    projection = application.project_progress_application_state(
        dataset_recorder=None,
        progress_workflow=None,
        promotable_hold_pause=None,
        application_held=False,
        arm_transition_pending=False,
        profile_selection_enabled=True,
    )

    assert not projection.configured
    assert projection.active_controls == ()
    assert projection.phase == "IDLE"
    assert projection.ui_payload["configured"] is False


def test_station_progress_application_caches_stable_unmatched_dataset_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    inspect_calls: list[tuple[Path, object | None]] = []

    def inspect_dataset_progress_profile(
        dataset_root: str | Path,
        *,
        requested_profile: object | None = None,
    ) -> object:
        inspect_calls.append((Path(dataset_root).resolve(), requested_profile))
        return SimpleNamespace(
            profile=None,
            locked=False,
            source="blank_dataset",
        )

    monkeypatch.setattr(
        real_teleop_application,
        "inspect_dataset_progress_profile",
        inspect_dataset_progress_profile,
    )
    args = SimpleNamespace(
        dataset_root=first_root,
        collection_progress_profile=None,
        collection_task_id="fixture-task-without-managed-profile",
        collection_progress_gui=True,
        collection_progress_arm_control_mode="profile",
    )
    application = build_real_teleop_progress_application(args)
    projection_kwargs = {
        "dataset_recorder": None,
        "progress_workflow": None,
        "promotable_hold_pause": None,
        "application_held": False,
        "arm_transition_pending": False,
        "profile_selection_enabled": True,
    }

    application.project_progress_application_state(**projection_kwargs)
    application.project_progress_application_state(**projection_kwargs)

    assert inspect_calls == [(first_root.resolve(), None)]

    args.dataset_root = second_root
    application.project_progress_application_state(**projection_kwargs)

    assert inspect_calls == [
        (first_root.resolve(), None),
        (second_root.resolve(), None),
    ]

    selected_profile = SimpleNamespace(semantic_content_hash="profile-hash-2")
    args.collection_progress_profile = selected_profile
    application.project_progress_application_state(**projection_kwargs)
    args.collection_progress_profile = SimpleNamespace(
        semantic_content_hash="profile-hash-2"
    )
    application.project_progress_application_state(**projection_kwargs)

    assert inspect_calls == [
        (first_root.resolve(), None),
        (second_root.resolve(), None),
        (second_root.resolve(), selected_profile),
    ]

    matching_recorder = SimpleNamespace(
        dataset_root=second_root,
        writer_storage_config=SimpleNamespace(
            task_id="fixture-task-without-managed-profile",
        ),
        progress_profile=None,
        progress_profile_selection_locked=False,
        progress_profile_binding_source="blank_dataset",
    )
    application.project_progress_application_state(
        **{**projection_kwargs, "dataset_recorder": matching_recorder}
    )
    application.project_progress_application_state(**projection_kwargs)

    assert inspect_calls == [
        (first_root.resolve(), None),
        (second_root.resolve(), None),
        (second_root.resolve(), selected_profile),
        (second_root.resolve(), args.collection_progress_profile),
    ]


def test_station_startup_projection_reuses_unmatched_dataset_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_calls: list[Path] = []

    def inspect_dataset_progress_profile(
        dataset_root: str | Path,
        *,
        requested_profile: object | None = None,
    ) -> object:
        del requested_profile
        inspect_calls.append(Path(dataset_root).resolve())
        return SimpleNamespace(
            profile=None,
            locked=False,
            source="blank_dataset",
        )

    monkeypatch.setattr(
        real_teleop_application,
        "inspect_dataset_progress_profile",
        inspect_dataset_progress_profile,
    )
    args = SimpleNamespace(
        dataset_root=tmp_path,
        collection_progress_profile=None,
        collection_task_id="fixture-task-without-managed-profile",
        collection_progress_gui=True,
        collection_progress_arm_control_mode="profile",
        collection_progress_entry_step="",
    )
    application = build_real_teleop_progress_application(args)

    application.build_startup_progress_ui_payload(
        dataset_recorder=None,
        profile_selection_enabled=True,
    )
    application.build_startup_progress_ui_payload(
        dataset_recorder=None,
        profile_selection_enabled=True,
    )

    assert inspect_calls == [tmp_path.resolve()]


@pytest.mark.parametrize("mode", RUNTIME_ARM_CONTROL_MODES)
def test_station_and_runtime_arm_control_vocabulary_stays_compatible(
    mode: str,
) -> None:
    assert STATION_ARM_CONTROL_MODES == RUNTIME_ARM_CONTROL_MODES
    assert normalize_station_arm_control_mode(mode) == mode
    assert normalize_runtime_arm_control_mode(mode) == mode


def test_runtime_progress_command_rejects_non_advance_target() -> None:
    with pytest.raises(ValueError, match="Only ADVANCE_PROGRESS"):
        ProgressMutationCommand(
            command_id="station:failure:1",
            kind="REPORT_FAILURE",
            target_progress_step_id="unexpected",
        )


class _StartupProgressControl:
    def __init__(self) -> None:
        self.build_calls: list[tuple[object | None, bool]] = []
        self.published: list[dict[str, object]] = []
        self.profile_selections: list[ProgressProfileSelectionCommand] = []
        self.arm_mode_selections: list[ProgressArmControlModeSelectionCommand] = []
        self.entry_step_selections: list[ProgressEntryStepSelectionCommand] = []
        self.selected_profiles: list[tuple[object | None, str]] = []
        self.selected_entry_steps: list[tuple[object | None, str]] = []

    def build_startup_progress_ui_payload(
        self,
        *,
        dataset_recorder: object | None,
        profile_selection_enabled: bool,
    ) -> dict[str, object]:
        self.build_calls.append((dataset_recorder, profile_selection_enabled))
        return {"enabled": True, "active_controls": []}

    def publish_progress_ui_state(self, payload: dict[str, object]) -> None:
        self.published.append(dict(payload))

    def consume_progress_profile_selection_intents(
        self,
    ) -> tuple[ProgressProfileSelectionCommand, ...]:
        result = tuple(self.profile_selections)
        self.profile_selections.clear()
        return result

    def consume_progress_arm_control_mode_selection_intents(
        self,
    ) -> tuple[ProgressArmControlModeSelectionCommand, ...]:
        result = tuple(self.arm_mode_selections)
        self.arm_mode_selections.clear()
        return result

    def consume_progress_entry_step_selection_intents(
        self,
    ) -> tuple[ProgressEntryStepSelectionCommand, ...]:
        result = tuple(self.entry_step_selections)
        self.entry_step_selections.clear()
        return result

    def select_managed_progress_profile(
        self,
        *,
        dataset_recorder: object | None,
        profile_id: str,
    ) -> object:
        self.selected_profiles.append((dataset_recorder, profile_id))
        return SimpleNamespace(profile_id=profile_id, profile_version="1")

    def select_progress_entry_step(
        self,
        *,
        dataset_recorder: object | None,
        progress_step_id: str,
    ) -> str:
        self.selected_entry_steps.append((dataset_recorder, progress_step_id))
        return progress_step_id


def test_startup_progress_runtime_deduplicates_unchanged_projection(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        dataset_root=tmp_path,
        collection_progress_gui=True,
        collection_task_id="writing",
        collection_progress_profile=None,
        collection_progress_arm_control_mode="profile",
        collection_progress_entry_step="",
    )
    control = _StartupProgressControl()
    messages: list[tuple[str, str]] = []
    runtime = RealTeleopStartupProgressRuntime(
        args=args,
        control_input=control,
        control_ui=SimpleNamespace(append_message=lambda *item: messages.append(item)),
        dataset_recorder=lambda: None,
        log=lambda *_args, **_kwargs: None,
    )

    runtime.publish(force=True)
    runtime.publish()

    assert len(control.build_calls) == 1
    assert len(control.published) == 1
    assert messages == []


def test_startup_progress_runtime_consumes_idle_configuration_selections(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        dataset_root=tmp_path,
        collection_progress_gui=True,
        collection_task_id="writing",
        collection_progress_profile=None,
        collection_progress_arm_control_mode="profile",
        collection_progress_entry_step="",
    )
    recorder = object()
    control = _StartupProgressControl()
    control.arm_mode_selections.append(
        ProgressArmControlModeSelectionCommand(
            command_id="arm-mode:1",
            mode="strict_profile",
        )
    )
    control.profile_selections.append(
        ProgressProfileSelectionCommand(
            command_id="profile:1",
            profile_id="writing-v1",
        )
    )
    control.entry_step_selections.append(
        ProgressEntryStepSelectionCommand(
            command_id="entry:1",
            progress_step_id="right_write",
        )
    )
    messages: list[tuple[str, str]] = []
    runtime = RealTeleopStartupProgressRuntime(
        args=args,
        control_input=control,
        control_ui=SimpleNamespace(append_message=lambda *item: messages.append(item)),
        dataset_recorder=lambda: recorder,
        log=lambda *_args, **_kwargs: None,
    )

    runtime.handle_pending_selections(now=12.5)

    assert args.collection_progress_arm_control_mode == "strict_profile"
    assert control.selected_profiles == [(recorder, "writing-v1")]
    assert control.selected_entry_steps == [(recorder, "right_write")]
    assert any("strict_profile" in message for _, message in messages)
    assert any("writing-v1@1" in message for _, message in messages)
    assert any("right_write" in message for _, message in messages)
