from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime import (
    EffectorCommand,
    TeleopCommand,
    command_effector_kind_subset,
)
from teleop_runtime.deploy import action_dispatch, run_policy


def _mixed_command() -> TeleopCommand:
    command = TeleopCommand(timestamp=12.5, metadata={"source": "unit"})
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.arange(6, dtype=np.float32),
            metadata={"owner": "arm"},
        )
    )
    command.add_command(
        EffectorCommand(
            effector_name="left_hand",
            side="left",
            command_type="joint_vector",
            value=np.full(6, 0.25, dtype=np.float32),
            metadata={"owner": "hand"},
        )
    )
    return command


class _Backend:
    def __init__(
        self,
        *,
        result: object = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.commands: list[EffectorCommand] = []

    def send_command(self, command: EffectorCommand) -> object:
        self.commands.append(command)
        if self.error is not None:
            raise self.error
        return self.result

    def get_state(self) -> dict[str, object]:
        return {"telemetry_error": None}

    def is_ready(self) -> bool:
        return True


def test_run_policy_preserves_action_dispatch_compatibility_exports() -> None:
    assert action_dispatch.command_effector_kind_subset is command_effector_kind_subset
    assert (
        run_policy.command_effector_kind_subset
        is action_dispatch.command_effector_kind_subset
    )
    assert (
        run_policy._dispatch_position_discrete_command_checked
        is action_dispatch.dispatch_position_discrete_command_checked
    )


def test_command_effector_kind_subset_copies_only_requested_commands() -> None:
    command = _mixed_command()
    execution_work_item = object()
    command.execution_work_item = execution_work_item

    subset = action_dispatch.command_effector_kind_subset(command, kind="hand")

    assert subset is not None
    assert subset.timestamp == 12.5
    assert subset.metadata == {"source": "unit"}
    assert subset.execution_work_item is execution_work_item
    assert set(subset.commands) == {"left_hand"}
    assert subset.commands["left_hand"] is not command.commands["left_hand"]
    subset.commands["left_hand"].metadata["owner"] = "changed"
    subset.commands["left_hand"].value[0] = 0.75
    assert command.commands["left_hand"].metadata == {"owner": "hand"}
    assert float(command.commands["left_hand"].value[0]) == pytest.approx(0.25)
    assert action_dispatch.command_effector_kind_subset(command, kind="missing") is None
    assert action_dispatch.command_effector_kind_subset(None, kind="hand") is None


def test_discrete_dispatch_records_generic_backend_receipt() -> None:
    command = action_dispatch.command_effector_kind_subset(
        _mixed_command(),
        kind="hand",
    )
    assert command is not None
    backend = _Backend()

    receipt = action_dispatch.dispatch_position_discrete_command_checked(
        command,
        hand_backends={"left": backend},
    )

    assert receipt["accepted"] is True
    assert receipt["skipped"] is False
    assert receipt["fatal"] is False
    effector_receipt = receipt["effectors"]["left_hand"]
    assert effector_receipt["status"] == "accepted"
    assert effector_receipt["backend_ready"] is True
    assert command.metadata["position_discrete_backend_receipts"]["left_hand"] == (
        effector_receipt
    )
    assert (
        command.commands["left_hand"].metadata["position_discrete_backend_receipt"]
        == effector_receipt
    )


def test_discrete_dispatch_exception_is_attached_before_propagation() -> None:
    command = action_dispatch.command_effector_kind_subset(
        _mixed_command(),
        kind="hand",
    )
    assert command is not None

    with pytest.raises(RuntimeError, match="left_hand discrete dispatch failed"):
        action_dispatch.dispatch_position_discrete_command_checked(
            command,
            hand_backends={"left": _Backend(error=OSError("transport failed"))},
        )

    receipt = command.metadata["position_discrete_backend_receipts"]["left_hand"]
    assert receipt["accepted"] is False
    assert receipt["fatal"] is True
    assert receipt["status"] == "exception"
    assert receipt["error_type"] == "OSError"
    assert receipt["error"] == "transport failed"


def test_discrete_dispatch_missing_backend_fails_before_send() -> None:
    command = action_dispatch.command_effector_kind_subset(
        _mixed_command(),
        kind="hand",
    )
    assert command is not None

    with pytest.raises(KeyError, match="left_hand"):
        action_dispatch.dispatch_position_discrete_command_checked(
            command,
            hand_backends={},
        )

    assert "position_discrete_backend_receipts" not in command.metadata
