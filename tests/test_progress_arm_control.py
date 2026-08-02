from __future__ import annotations

from types import SimpleNamespace

import pytest

from dataset_contracts.progress_profile import (
    ProgressStepArmControlV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.progress_arm_control import (
    build_progress_arm_transition_plan,
    progress_step_arm_requirements_differ,
    strict_progress_arm_sample_rejection,
)
from teleop_runtime.real_teleop.dataset_recorder import RealTeleopDatasetRecorder
from teleop_runtime.schemas import EffectorCommand, TeleopCommand
from teleop_runtime.session import SessionStepResult


def _profile(*, declared: bool = True) -> TaskProgressProfileV1:
    requirement = (
        ProgressStepArmControlV1(left="fixed_hold", right="teleop")
        if declared
        else None
    )
    return TaskProgressProfileV1(
        profile_id="arm-profile",
        profile_version="1",
        task_id="task",
        steps=(
            ProgressStepDefinitionV1(
                "right_move",
                "MOVE",
                "Right move",
                1,
                arm_control=requirement,
            ),
        ),
        initial_step_id="right_move",
        terminal_step_ids=("right_move",),
    )


def _step(*, fixed_side: str | None) -> SessionStepResult:
    metadata = {}
    effector_metadata = {}
    if fixed_side is not None:
        metadata = {
            "fix_arm_active_side": fixed_side,
            "fix_arm_scope": "arm_and_end_effector",
        }
        effector_metadata = {
            "fix_arm_active": True,
            "fix_arm_side": fixed_side,
            "fix_arm_scope": "arm_and_end_effector",
        }
    command = TeleopCommand(timestamp=1.0, metadata=metadata)
    for side in ("left", "right"):
        command.commands[f"{side}_arm"] = EffectorCommand(
            effector_name=f"{side}_arm",
            side=side,
            command_type="joint_position",
            value=[0.0] * 6,
            metadata=(effector_metadata if side == fixed_side else {}),
        )
    return SessionStepResult(
        timestamp=1.0,
        state=TeleopRunState.RUNNING,
        control_hz=60.0,
        command=command,
    )


def test_profile_plan_only_requires_home_for_a_newly_live_arm() -> None:
    direct = build_progress_arm_transition_plan(
        profile=_profile(),
        target_progress_step_id="right_move",
        current_fixed_sides=(),
        enforcement_mode="profile",
    )
    assert direct is not None
    assert direct.target_fixed_sides == ("left",)
    assert direct.newly_teleop_sides == ()

    release = build_progress_arm_transition_plan(
        profile=_profile(),
        target_progress_step_id="right_move",
        current_fixed_sides=("right",),
        enforcement_mode="profile",
    )
    assert release is not None
    assert release.newly_teleop_sides == ("right",)


def test_undeclared_profile_is_compatible_except_in_strict_mode() -> None:
    assert build_progress_arm_transition_plan(
        profile=_profile(declared=False),
        target_progress_step_id="right_move",
        current_fixed_sides=(),
        enforcement_mode="profile",
    ) is None
    with pytest.raises(RuntimeError, match="requires every active progress step"):
        build_progress_arm_transition_plan(
            profile=_profile(declared=False),
            target_progress_step_id="right_move",
            current_fixed_sides=(),
            enforcement_mode="strict_profile",
        )


def test_strict_writer_gate_checks_step_id_mode_and_fixed_scope() -> None:
    profile = _profile()
    assert strict_progress_arm_sample_rejection(
        profile=profile,
        progress_step_id="right_move",
        step_result=_step(fixed_side="left"),
        active_sides=("left", "right"),
    ) is None
    assert "side=left" in str(
        strict_progress_arm_sample_rejection(
            profile=profile,
            progress_step_id="right_move",
            step_result=_step(fixed_side=None),
            active_sides=("left", "right"),
        )
    )


def test_async_strict_gate_uses_progress_step_captured_at_sample_cut() -> None:
    profile = TaskProgressProfileV1(
        profile_id="two-step-arm-profile",
        profile_version="1",
        task_id="task",
        steps=(
            ProgressStepDefinitionV1(
                "right_move",
                "MOVE",
                "Right move",
                1,
                allowed_next_step_ids=("left_move",),
                arm_control=ProgressStepArmControlV1(
                    left="fixed_hold",
                    right="teleop",
                ),
            ),
            ProgressStepDefinitionV1(
                "left_move",
                "MOVE",
                "Left move",
                2,
                arm_control=ProgressStepArmControlV1(
                    left="teleop",
                    right="fixed_hold",
                ),
            ),
        ),
        initial_step_id="right_move",
        terminal_step_ids=("left_move",),
    )
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.args = SimpleNamespace(
        collection_progress_arm_control_mode="strict_profile"
    )
    recorder.collector_components = SimpleNamespace(
        active_sides=("left", "right")
    )
    recorder._active_progress_episode = SimpleNamespace(
        profile=profile,
        state=SimpleNamespace(
            phase="ACTIVE_ATTEMPT",
            current_progress_step_id="left_move",
        ),
    )
    old_step_command = _step(fixed_side="left")

    assert progress_step_arm_requirements_differ(
        profile,
        "right_move",
        "left_move",
    )
    assert not progress_step_arm_requirements_differ(
        profile,
        "right_move",
        "right_move",
    )

    assert recorder._progress_arm_sample_rejection(
        old_step_command,
        progress_step_id="right_move",
    ) is None
    assert "side=left" in str(
        recorder._progress_arm_sample_rejection(old_step_command)
    )

    submitted: dict[str, object] = {}

    class _Executor:
        def submit(self, function, *args, **kwargs):
            submitted.update(kwargs)
            return SimpleNamespace(done=lambda: False)

    recorder.active_episode_index = 1
    recorder._pause_started_at = None
    recorder._async_sample_enabled = True
    recorder._sample_future = None
    recorder._sample_executor = _Executor()
    recorder._active_progress_episode.state.current_progress_step_id = (
        "right_move"
    )

    assert recorder.record_step_async(old_step_command)
    assert submitted["strict_progress_step_id"] == "right_move"
