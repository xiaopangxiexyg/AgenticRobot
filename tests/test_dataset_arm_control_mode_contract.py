from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.deploy.replay import bind_debug_replay_arm_control_mode
from teleop_runtime.deploy.run_policy import _validate_debug_replay_arm_control_mode
from teleop_runtime.execution_mode_contract import (
    DatasetArmControlModeError,
    validate_dataset_arm_control_mode,
)
from teleop_runtime.replay.executor import ReplayExecutionController
from teleop_runtime.replay.hardware import ReplayHardwareConfig
from teleop_runtime.replay.session import ReplayGuiSession


def test_recorded_mode_must_match_selected_mode() -> None:
    contract = validate_dataset_arm_control_mode(
        {"arm_control_mode": "impedance"},
        selected_mode="impedance",
        hardware_output=True,
    )
    assert contract.compatible
    assert contract.recorded_mode == "impedance"
    assert not contract.legacy_default_applied

    with pytest.raises(DatasetArmControlModeError, match="mismatch"):
        validate_dataset_arm_control_mode(
            {"arm_control_mode": "impedance"},
            selected_mode="position",
            hardware_output=True,
        )


def test_conflicting_recorded_mode_fields_are_rejected() -> None:
    with pytest.raises(DatasetArmControlModeError, match="contradictory"):
        validate_dataset_arm_control_mode(
            {
                "arm_control_mode": "impedance",
                "arm_execution_profile": {"mode": "position"},
            },
            selected_mode="impedance",
            hardware_output=True,
        )


def test_legacy_hardware_episode_defaults_to_impedance_contract() -> None:
    contract = validate_dataset_arm_control_mode(
        {},
        selected_mode="impedance",
        hardware_output=True,
    )
    assert contract.recorded_mode == "impedance"
    assert contract.legacy_default_applied
    assert contract.warning

    with pytest.raises(DatasetArmControlModeError, match="legacy missing-mode default"):
        validate_dataset_arm_control_mode(
            {},
            selected_mode="position",
            hardware_output=True,
        )


def test_legacy_dry_run_remains_inspectable_with_warning() -> None:
    contract = validate_dataset_arm_control_mode(
        {},
        selected_mode="position",
        hardware_output=False,
    )
    assert contract.recorded_mode is None
    assert contract.compatible
    assert contract.warning


def test_replay_controller_applies_legacy_impedance_default_before_connect() -> None:
    episode = SimpleNamespace(summary=SimpleNamespace(metadata={}))
    position = ReplayExecutionController(
        hardware_config=ReplayHardwareConfig(
            arm_control_mode="position",
            connect_hardware=True,
        )
    )
    with pytest.raises(DatasetArmControlModeError, match="mismatch"):
        position.validate_episode_arm_control_mode(episode)  # type: ignore[arg-type]

    impedance = ReplayExecutionController(
        hardware_config=ReplayHardwareConfig(
            arm_control_mode="impedance",
            connect_hardware=True,
        )
    )
    contract = impedance.validate_episode_arm_control_mode(episode)  # type: ignore[arg-type]
    assert contract["recorded_mode"] == "impedance"
    assert contract["legacy_default_applied"]


def test_replay_session_blocks_mismatch_before_connect() -> None:
    session = ReplayGuiSession(
        hardware_config=ReplayHardwareConfig(
            arm_control_mode="position",
            connect_hardware=True,
        )
    )
    session.episode = SimpleNamespace(
        summary=SimpleNamespace(metadata={"arm_control_mode": "impedance"})
    )
    try:
        with pytest.raises(DatasetArmControlModeError, match="mismatch"):
            session.dispatch({"type": "connect_hardware"})
        assert session.executor.status()["connected"] is False
    finally:
        session.close()


def test_deploy_debug_replay_validates_before_backend_creation() -> None:
    policy = SimpleNamespace(
        metadata={"episode_metadata": {"arm_control_mode": "position"}}
    )
    args = SimpleNamespace(arm_control_mode="position", send_actions=True)
    contract = _validate_debug_replay_arm_control_mode(policy, args)
    assert contract["compatible"]
    assert policy.metadata["dataset_arm_control_mode_contract"] == contract

    args.arm_control_mode = "impedance"
    with pytest.raises(DatasetArmControlModeError, match="mismatch"):
        _validate_debug_replay_arm_control_mode(policy, args)


def test_replay_owner_binds_receipt_and_application_prints_warning(
    capsys: pytest.CaptureFixture[str],
) -> None:
    policy = SimpleNamespace(metadata={})

    payload, warning = bind_debug_replay_arm_control_mode(
        policy,
        selected_mode="position",
        hardware_output=False,
    )

    assert warning
    assert payload["recorded_mode"] is None
    assert policy.metadata["dataset_arm_control_mode_contract"] == payload

    args = SimpleNamespace(arm_control_mode="position", send_actions=False)
    assert _validate_debug_replay_arm_control_mode(policy, args) == payload
    assert capsys.readouterr().out == f"[ReplayMode] WARNING: {warning}\n"
