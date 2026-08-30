from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sim.rm65.actuation import (
    DmtacPositionActuationProfile,
    DmtacPositionActuator,
    dmtac_bimanual_position_profile,
)
from sim.rm65.real2sim.gripper_position_fit import FIT_SCHEMA, MODEL_TYPE


def _write_artifact(path: Path) -> None:
    payload = {
        "schema": FIT_SCHEMA,
        "model_type": MODEL_TYPE,
        "scope": "free_motion_normalized_position_only",
        "fit_by_side": {
            "left": {
                "close": {
                    "delay_sec": 0.08,
                    "max_rate_per_sec": 1.2,
                    "acceleration_per_sec2": 20.0,
                },
                "open": {
                    "delay_sec": 0.04,
                    "max_rate_per_sec": 1.5,
                    "acceleration_per_sec2": 25.0,
                },
            }
        },
        "source_conditions_observed": {
            "speed_setting": [1.0],
            "force_setting": [0.2],
            "continuous_grasp": [True],
            "command_mode": ["runtime-equivalent"],
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.integration
def test_profile_loads_hashed_directional_position_fit(tmp_path: Path) -> None:
    artifact_path = tmp_path / "fit.json"
    _write_artifact(artifact_path)

    profile = DmtacPositionActuationProfile.from_fit_artifact(artifact_path)

    assert tuple(profile.parameters_by_side) == ("left",)
    assert profile.command_hz == 20.0
    assert profile.send_threshold == 0.002
    assert profile.parameters("left", direction="close").delay_sec == 0.08
    assert profile.parameters("left", direction="open").max_rate_per_sec == 1.5
    assert profile.source_conditions["force_setting"] == [0.2]


@pytest.mark.integration
def test_profile_rejects_modified_fit_payload(tmp_path: Path) -> None:
    artifact_path = tmp_path / "fit.json"
    _write_artifact(artifact_path)
    payload = json.loads(artifact_path.read_text())
    payload["fit_by_side"]["left"]["close"]["delay_sec"] = 0.2
    artifact_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        DmtacPositionActuationProfile.from_fit_artifact(artifact_path)


def test_canonical_bilateral_stage_profile_is_content_addressed() -> None:
    profile = dmtac_bimanual_position_profile()

    assert profile.name == "dmtac_bimanual_position_20260722.v1"
    assert profile.command_hz == 20.0
    assert profile.send_threshold == 0.002
    assert profile.source_payload_sha256 == (
        "3cbb510090d008d1c58ffc5c5e721ad3a2e3b45cda1285dcacb29ef121860691"
    )
    assert profile.parameters("left", direction="close").delay_sec == pytest.approx(
        0.0592514185
    )
    assert profile.parameters("right", direction="open").max_rate_per_sec == pytest.approx(
        1.3864800875513763
    )


@pytest.mark.integration
def test_actuator_applies_delay_rate_limit_threshold_and_reverse_fit(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "fit.json"
    _write_artifact(artifact_path)
    profile = DmtacPositionActuationProfile.from_fit_artifact(artifact_path)
    actuator = DmtacPositionActuator(profile)
    actuator.reset({"left": 0.0})
    actuator.queue_targets({"left": 1.0})

    tick = actuator.step(physics_time_sec=0.0, timestep_sec=0.001)
    assert tick.accepted_by_side["left"] is True
    assert tick.servo_goal_by_side["left"] == 0.0
    for index in range(1, 80):
        tick = actuator.step(
            physics_time_sec=index / 1000.0,
            timestep_sec=0.001,
        )
    assert tick.actuator_position_by_side["left"] == 0.0

    tick = actuator.step(physics_time_sec=0.080, timestep_sec=0.001)
    assert tick.servo_goal_by_side["left"] == 1.0
    assert tick.actuator_rate_by_side["left"] == pytest.approx(0.02)
    assert tick.actuator_position_by_side["left"] == pytest.approx(0.00002)

    for index in range(81, 1201):
        tick = actuator.step(
            physics_time_sec=index / 1000.0,
            timestep_sec=0.001,
        )
    assert tick.actuator_position_by_side["left"] == pytest.approx(1.0)

    actuator.queue_targets({"left": 0.999})
    tick = actuator.step(physics_time_sec=1.201, timestep_sec=0.001)
    assert tick.accepted_by_side["left"] is False
    assert tick.accepted_target_by_side["left"] == 1.0

    actuator.queue_targets({"left": 0.0})
    tick = actuator.step(physics_time_sec=1.250, timestep_sec=0.001)
    assert tick.accepted_by_side["left"] is True
    assert tick.servo_goal_by_side["left"] == 1.0
    for index in range(1251, 1290):
        tick = actuator.step(
            physics_time_sec=index / 1000.0,
            timestep_sec=0.001,
        )
    assert tick.actuator_position_by_side["left"] == 1.0
    tick = actuator.step(physics_time_sec=1.290, timestep_sec=0.001)
    assert tick.servo_goal_by_side["left"] == 0.0
    assert tick.actuator_rate_by_side["left"] == pytest.approx(-0.025)


def test_gripper_snapshot_bootstrap_preserves_absolute_command_clock(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "fit.json"
    _write_artifact(artifact_path)
    profile = DmtacPositionActuationProfile.from_fit_artifact(artifact_path)
    actuator = DmtacPositionActuator(profile)
    actuator.reset({"left": 0.0}, initial_physics_time_sec=0.5)
    actuator.queue_targets({"left": 1.0})

    tick = actuator.step(physics_time_sec=0.5, timestep_sec=0.001)
    assert tick.accepted_by_side["left"] is False
    tick = actuator.step(physics_time_sec=0.55, timestep_sec=0.001)
    assert tick.accepted_by_side["left"] is True

    # Keep the established cold-start behavior where a target queued after a
    # t=0 reset is accepted on the first physics tick.
    actuator.reset({"left": 0.0})
    actuator.queue_targets({"left": 1.0})
    tick = actuator.step(physics_time_sec=0.0, timestep_sec=0.001)
    assert tick.accepted_by_side["left"] is True
def test_gripper_snapshot_rejects_negative_command_deadline(tmp_path: Path) -> None:
    artifact_path = tmp_path / "fit.json"
    _write_artifact(artifact_path)
    profile = DmtacPositionActuationProfile.from_fit_artifact(artifact_path)
    source = DmtacPositionActuator(profile)
    source.reset({"left": 0.0})
    snapshot = source.snapshot()
    snapshot["next_command_time_by_side"]["left"] = -1.0

    restored = DmtacPositionActuator(profile)
    with pytest.raises(ValueError, match="next_command_time_by_side.*invalid"):
        restored.reset({"left": 0.0}, initial_state=snapshot)
