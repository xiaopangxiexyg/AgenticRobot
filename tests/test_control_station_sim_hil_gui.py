from __future__ import annotations

import http.client
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from control_station.composition import MujocoSimHILReadinessEvidence
from control_station.sim_hil import (
    MujocoSimHILApplication,
    MujocoSimHILApplicationConfig,
    SIM_HIL_STATUS_SCHEMA_V1,
    _bind_vive_coordinate_profile,
    _config_from_args,
    build_arg_parser,
)
from control_station.sim_hil_gui import SimHILStatusServer
from control_station.vive_calibration_profiles import (
    new_coordinate_calibration,
    save_coordinate_profile,
)
from robot_motion.action import (
    TargetExecutionStageStatus,
    TargetExecutionStatusSnapshot,
)
from teleop_runtime.calibration import ArmCalibration, TeleopCalibration


def test_sim_hil_rate_config_keeps_compose_rate_explicit_and_legacy_alias() -> None:
    default_config = MujocoSimHILApplicationConfig()
    assert default_config.position_control_profile_name == "position_60_120hz"
    assert default_config.compose_target_hz is None
    assert default_config.control_rate_hz is None

    explicit = MujocoSimHILApplicationConfig(compose_target_hz=45.0)
    assert explicit.compose_target_hz == 45.0
    assert explicit.control_rate_hz is None

    # Station configuration is no longer the owner of the historical 60 Hz
    # ceiling.  The active deploy/authority capability may still reject a
    # rate it cannot execute; that gate is checked when the target is built.
    future_rate = MujocoSimHILApplicationConfig(compose_target_hz=120.0)
    assert future_rate.compose_target_hz == 120.0

    legacy = MujocoSimHILApplicationConfig(control_rate_hz=45.0)
    assert legacy.compose_target_hz == 45.0
    assert legacy.control_rate_hz == 45.0

    with pytest.raises(ValueError, match="compatibility alias"):
        MujocoSimHILApplicationConfig(
            compose_target_hz=45.0,
            control_rate_hz=60.0,
        )


def test_sim_hil_rate_cli_exposes_profile_and_compose_cadence() -> None:
    args = build_arg_parser().parse_args(
        [
            "--position-control-profile",
            "position",
            "--compose-hz",
            "45",
        ]
    )

    assert args.position_control_profile == "position"
    assert args.compose_hz == 45.0
    assert args.rate is None

    legacy_args = build_arg_parser().parse_args(["--rate", "45"])
    legacy_config = _config_from_args(legacy_args)
    assert legacy_config.compose_target_hz == 45.0
    assert legacy_config.control_rate_hz == 45.0


def _request(
    address: tuple[str, int],
    method: str,
    path: str,
) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(address[0], address[1], timeout=2.0)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_sim_hil_status_server_is_loopback_only() -> None:
    with pytest.raises(ValueError, match="loopback"):
        SimHILStatusServer(lambda: {}, host="0.0.0.0")

def test_sim_hil_status_server_serves_structured_read_only_surface() -> None:
    value = {
        "schema": "dexteleop.control_station.sim_hil_status.v1",
        "started": False,
        "readiness": {"ready": False},
    }
    server = SimHILStatusServer(lambda: value, port=0)
    address = server.start()
    try:
        status, headers, payload = _request(address, "GET", "/api/status")
        assert status == 200
        assert headers["Content-Type"].startswith("application/json")
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert json.loads(payload) == value

        status, headers, payload = _request(address, "GET", "/")
        assert status == 200
        assert headers["X-Frame-Options"] == "DENY"
        assert b"Readiness gate" in payload
        assert b"no raw action values" in payload

        status, headers, payload = _request(address, "POST", "/api/status")
        assert status == 405
        assert headers["Allow"] == "GET"
        assert payload == b"read-only status server\n"
    finally:
        server.close()


def test_sim_hil_status_server_reports_provider_or_json_contract_errors() -> None:
    server = SimHILStatusServer(lambda: {"invalid": object()}, port=0)
    address = server.start()
    try:
        status, _headers, payload = _request(address, "GET", "/api/status")
        assert status == 500
        result = json.loads(payload)
        assert result["schema"] == "dexteleop.control_station.sim_hil_status.v1"
        assert "status_error" in result
    finally:
        server.close()


def test_sim_hil_application_status_is_versioned_and_json_strict() -> None:
    readiness = MujocoSimHILReadinessEvidence.all_ready()
    runtime_config = SimpleNamespace(
        runtime_session_id="session",
        control_domain_id="domain",
        clock_domain_id="clock",
        enable_composer_authority=True,
        enable_primary_policy_authority=False,
        enable_replay_authority=False,
    )
    contracts = SimpleNamespace(
        composition_spec=SimpleNamespace(
            spec_id="composition",
            semantic_content_hash="composition-hash",
            required_roles=("human", "policy"),
        ),
        strategy_spec=SimpleNamespace(
            strategy_id="strategy",
            semantic_content_hash="strategy-hash",
        ),
        authority_profile=SimpleNamespace(
            profile_id="authority",
            semantic_content_hash="authority-hash",
        ),
    )
    policy_source = SimpleNamespace(
        last_emission=None,
        status=lambda *, now: {"source_id": "ppo", "started": False},
    )
    human_source = SimpleNamespace(
        session_state="debug_only",
        started=False,
        last_emission=None,
        binding_receipt=SimpleNamespace(
            source_contract_hash="human-source-hash",
            observer_binding_id="observer",
            semantic_content_hash="binding-hash",
            rig_absent=True,
            observer_exclusive=True,
            authoritative=False,
        ),
        session=SimpleNamespace(
            home_recalibration_required=False,
            motion_hold_mode="",
            motion_hold_reason="",
            tracker_loss_duration_sec=0.0,
            tracker_loss_channels=(),
        ),
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        action_schema=SimpleNamespace(semantic_content_hash="action-schema-hash"),
    )
    composition = SimpleNamespace(
        contracts=contracts,
        started=False,
        activation_scope=None,
    )
    app = MujocoSimHILApplication(
        config=MujocoSimHILApplicationConfig(),
        task_config=SimpleNamespace(),
        scene=SimpleNamespace(),
        runtime=runtime,
        composition=composition,
        policy_source=policy_source,
        human_source=human_source,
        readiness_provider=SimpleNamespace(snapshot=lambda *, now: readiness),
        clock=lambda: 10.0,
    )

    status = app.status()
    json.dumps(status, allow_nan=False)
    assert status["schema"] == SIM_HIL_STATUS_SCHEMA_V1
    assert status["contracts"]["composition"]["spec_hash"] == "composition-hash"
    assert status["human"]["binding"]["authoritative"] is False
    assert status["execution_gate"]["status"] == "not_started"

    snapshot = TargetExecutionStatusSnapshot(
        control_domain_id="domain",
        target_handler_id="target-handler",
        clock_domain_id="clock",
        activation_mode="required",
        captured_at=10.0,
        request=TargetExecutionStageStatus(
            stage="request",
            status="submitted",
            record_id="request-1",
            record_hash="request-hash-1",
            available_at=9.90,
        ),
        submission=TargetExecutionStageStatus(
            stage="submission",
            status="accepted_for_processing",
            record_id="submission-1",
            record_hash="submission-hash-1",
            available_at=9.91,
        ),
        completion=TargetExecutionStageStatus(
            stage="completion",
            status="resolved",
            record_id="completion-1",
            record_hash="completion-hash-1",
            available_at=9.92,
        ),
        admission=TargetExecutionStageStatus(
            stage="admission",
            status="accepted",
            record_id="admission-1",
            record_hash="admission-hash-1",
            available_at=9.93,
        ),
        activation=TargetExecutionStageStatus(
            stage="activation",
            status="pending",
        ),
        target_id="target-1",
        target_hash="target-hash-1",
    )
    runtime.capture_execution_gate_snapshot = lambda *, now: SimpleNamespace(
        status="active",
        reason=None,
        gate_epoch=1,
        source_event_ids=(),
        semantic_content_hash="gate-hash",
    )
    runtime.capture_target_execution_status_snapshot = (
        lambda *, now: snapshot
    )
    app._started = True

    running_status = app.status()
    target_status = running_status["target_execution"]
    assert target_status["request_id"] == "request-1"
    assert target_status["stages"]["submission"]["status"] == (
        "accepted_for_processing"
    )
    assert target_status["stages"]["admission"]["record_id"] == "admission-1"
    assert target_status["stages"]["activation"]["status"] == "pending"
    json.dumps(running_status, allow_nan=False)


def test_station_coordinate_binding_does_not_mutate_home_profile(tmp_path) -> None:
    coordinate = new_coordinate_calibration(active_sides=("left", "right"))
    coordinate.set_world_to_desk(np.asarray(np.eye(4), dtype=float))
    coordinate.set_tracker_offset("left_wrist", np.asarray(np.eye(4), dtype=float))
    right_offset = np.asarray(np.eye(4), dtype=float)
    right_offset[0, 3] = 0.12
    coordinate.set_tracker_offset("right_wrist", right_offset)
    saved_path = save_coordinate_profile(
        coordinate,
        root=tmp_path,
        profile_name="bimanual",
    )
    home = TeleopCalibration(
        arms={
            side: ArmCalibration(name=f"{side}_arm", side=side, T_D_B=np.eye(4))
            for side in ("left", "right")
        },
        metadata={"home_ready": True},
    )
    with patch(
        "control_station.vive_calibration_profiles.control_station_vive_coordinate_profiles_dir",
        return_value=tmp_path,
    ):
        merged, binding = _bind_vive_coordinate_profile(
            home,
            coordinate_profile_path=saved_path,
        )

    assert home.world_to_desk_pose() is None
    assert home.tracker_offsets == {}
    np.testing.assert_allclose(merged.world_to_desk_pose(), np.eye(4))
    np.testing.assert_allclose(merged.tracker_offsets["right_wrist"], right_offset)
    assert binding["mode"] == "independent_vive_coordinate_plus_target_home_v1"
    assert binding["vive_coordinate_profile_path"] == str(saved_path)
