from __future__ import annotations

import pytest

import control_station.sim_hil as sim_hil_module
from control_station.sim_hil_gui import _HTML
from control_station.sim_hil_profile.config import MujocoSimHILApplicationConfig


def test_state_bc_cli_binds_artifact_and_cadence():
    args = sim_hil_module.build_arg_parser().parse_args(
        [
            "--policy-backend",
            "state_bc",
            "--state-bc-artifact",
            "/tmp/state_bc_fixture.npz",
            "--state-bc-action-hz",
            "20",
            "--state-bc-policy-hz",
            "10",
            "--state-bc-prediction-queue-threshold",
            "2",
        ]
    )

    config = sim_hil_module._config_from_args(args)

    assert config.policy_backend == "state_bc"
    assert str(config.state_bc_artifact_path) == "/tmp/state_bc_fixture.npz"
    assert config.state_bc_action_hz == pytest.approx(20.0)
    assert config.state_bc_policy_hz == pytest.approx(10.0)
    assert config.state_bc_prediction_queue_threshold == 2


def test_state_bc_backend_requires_an_artifact():
    with pytest.raises(ValueError, match="state_bc.*artifact"):
        MujocoSimHILApplicationConfig(policy_backend="state_bc")


def test_keyboard_controls_submit_existing_exclusive_intents_only():
    assert "keydown" in _HTML
    assert "selectExclusiveSource('human')" in _HTML
    assert "selectExclusiveSource('policy')" in _HTML
    assert "submitIntent('hold')" in _HTML
    assert "exclusive_truncated_hitl" in _HTML
    assert "target.tagName === 'INPUT'" in _HTML
