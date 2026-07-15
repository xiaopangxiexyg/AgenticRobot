from __future__ import annotations

from types import SimpleNamespace

from robot_motion.runtime import EffectorCommand, TeleopCommand

from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.collection_bridge import (
    collection_backends_from_components,
    collection_sample_step_result,
    collection_sources_from_components,
    collection_step_result,
    copy_teleop_command,
)
from teleop_runtime.session import TRACKER_HOLD_AUTOMATIC, SessionStepResult
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def test_legacy_collection_adapter_helpers_are_direct_aliases() -> None:
    assert (
        legacy_entrypoint._collection_backends_from_components
        is collection_backends_from_components
    )
    assert (
        legacy_entrypoint._collection_sources_from_components
        is collection_sources_from_components
    )
    assert legacy_entrypoint._collection_step_result is collection_step_result


def test_extracted_collection_bridge_keeps_real_component_names_and_hold_dispatch_rule() -> None:
    left_state = object()
    left_command = object()
    right_command = object()
    left_hand = object()
    keyboard = SimpleNamespace(name="keyboard")
    components = SimpleNamespace(
        left_state_backend=left_state,
        left_command_backend=left_command,
        right_state_backend=None,
        right_command_backend=right_command,
        left_hand_backend=left_hand,
        right_hand_backend=None,
        keyboard_source=keyboard,
    )

    assert collection_backends_from_components(components) == {
        "left_arm": left_state,
        "right_arm": right_command,
        "left_hand": left_hand,
    }
    camera = object()
    assert collection_sources_from_components(components, {"left_wrist": camera}) == {
        "keyboard": keyboard,
        "camera.left_wrist": camera,
    }

    held = SessionStepResult(
        timestamp=1.0,
        state=TeleopRunState.RUNNING,
        control_hz=15.0,
        command_dispatched=True,
        motion_hold_mode=TRACKER_HOLD_AUTOMATIC,
    )
    copied = collection_step_result(
        held,
        active_command=TeleopCommand(timestamp=1.0),
        gripper_command=None,
    )
    assert copied.command is not None
    assert copied.command_dispatched is False
    assert copied.motion_hold_mode == TRACKER_HOLD_AUTOMATIC


def test_collection_sample_uses_a_deep_copy_of_the_latest_dispatched_command() -> None:
    dispatched = TeleopCommand(timestamp=1.0)
    dispatched.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=[0.1] * 6,
        )
    )
    current = SessionStepResult(
        timestamp=2.0,
        state=TeleopRunState.RUNNING,
        control_hz=30.0,
        command=TeleopCommand(timestamp=2.0),
        command_dispatched=False,
    )

    sampled = collection_sample_step_result(current, dispatched_command=dispatched)
    copied = copy_teleop_command(dispatched)

    assert sampled.timestamp == 2.0
    assert sampled.command_dispatched is True
    assert sampled.command is not dispatched
    assert sampled.command is not None
    assert sampled.command.timestamp == 1.0
    assert list(sampled.command.commands["left_arm"].value) == [0.1] * 6
    assert copied is not dispatched
