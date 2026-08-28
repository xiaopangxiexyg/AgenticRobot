from __future__ import annotations

from pathlib import Path

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


def test_state_bc_bootstrap_status_exposes_gui_policy_contract():
    config = MujocoSimHILApplicationConfig(
        enabled=True,
        policy_backend="state_bc",
        state_bc_artifact_path=Path("/tmp/state_bc_fixture.npz"),
    )

    status = sim_hil_module._sim_hil_bootstrap_status(
        config,
        setup={
            "phase": "vive_setup",
            "input_ready": False,
            "profile_ready": False,
        },
        vive_status={},
    )

    policy = status["policy"]
    assert policy["backend"] == "state_bc"
    assert policy["policy_artifact"] == "/tmp/state_bc_fixture.npz"
    assert policy["artifact_path"] == "/tmp/state_bc_fixture.npz"
    assert policy["artifact_status"] == "not_found"
    assert policy["policy_dimensions"] == [16, 16]
    assert policy["model_observation_sides"] == ["left", "right"]
    assert policy["output_active_sides"] == ["right"]
    assert policy["state_only_capability"] is True
    assert policy["full_bimanual_execution"] is False
    assert policy["artifact_schema"] == "dexteleop.state_bc_model.v1"
    assert policy["layout_hash"]


def test_state_bc_gui_exposes_model_runtime_and_handoff_contract():
    for element_id in (
        "policy-runtime-panel",
        "policy-runtime-state",
        "policy-runtime-backend",
        "policy-runtime-contract",
        "policy-runtime-output",
        "policy-runtime-artifact",
        "policy-runtime-help",
        "policy-runtime-note",
        "policy-source-title",
        "exclusive-policy-title",
        "exclusive-policy-description",
    ):
        assert f'id="{element_id}"' in _HTML
    assert "function policyPresentation(policy)" in _HTML
    assert "38D MuJoCo state → 16D state-BC → 16D action → 26D runtime" in _HTML
    assert "state-BC 只读取 MuJoCo 状态" in _HTML
    assert "model_state_dim" in _HTML
    assert "output_active_sides" in _HTML
    assert "human-timeout-policy-option" in _HTML


def test_keyboard_controls_submit_existing_exclusive_intents_only():
    assert "keydown" in _HTML
    assert "selectExclusiveSource('human')" in _HTML
    assert "selectExclusiveSource('policy')" in _HTML
    assert "submitIntent('hold')" in _HTML
    assert "exclusive_truncated_hitl" in _HTML
    assert "target.tagName === 'INPUT'" in _HTML
