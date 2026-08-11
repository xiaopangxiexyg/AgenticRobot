from __future__ import annotations

import json
import math

import numpy as np
import pytest

from teleop_runtime.debug.replay_position_admittance_mujoco import (
    _PhysicalRollout,
    _read_mujoco_rows,
    _compare_rollouts,
    _recorded_input_contract,
    _torque_release_transients,
)
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
)
from teleop_runtime.debug.replay_position_rotational_admittance_mujoco import (
    ROTATIONAL_ADMITTANCE_ENVELOPES,
    replay_recorded_rotational_admittance_through_mujoco,
    rotational_admittance_config_for_envelope,
)


def _z_rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return pose


def _rotation_rollout(
    name: str,
    *,
    physical_effect_deg: tuple[float, ...],
    filtered_torque_nm: tuple[float, ...],
) -> _PhysicalRollout:
    count = len(physical_effect_deg)
    assert len(filtered_torque_nm) == count
    identity = np.eye(4, dtype=float)
    zero_qpos = np.zeros(6, dtype=float)
    return _PhysicalRollout(
        name=name,
        report={},
        sample_times_sec=[0.05 * index for index in range(count)],
        measured_qpos={
            side: [zero_qpos.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        measured_pose={
            "left": [_z_rotation(value) for value in physical_effect_deg],
            "right": [identity.copy() for _ in range(count)],
        },
        qref={
            side: [zero_qpos.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        qcmd={
            side: [zero_qpos.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        raw_target_pose={
            side: [identity.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        applied_target_pose={
            side: [identity.copy() for _ in range(count)]
            for side in ("left", "right")
        },
        filtered_force_control_n=[np.zeros(3) for _ in range(count)],
        active_offset_m=[np.zeros(3) for _ in range(count)],
        filtered_torque_control_nm=[
            np.array([0.0, 0.0, value], dtype=float)
            for value in filtered_torque_nm
        ],
        active_rotation_rotvec_rad=[
            np.array([0.0, 0.0, math.radians(2.0)], dtype=float)
            for _ in range(count)
        ],
    )


def test_compare_rollouts_reports_component_release_clock_mismatch() -> None:
    baseline = _rotation_rollout(
        "baseline",
        physical_effect_deg=(0.0, 0.0),
        filtered_torque_nm=(0.0, 0.0),
    )
    candidate = _rotation_rollout(
        "candidate",
        physical_effect_deg=(0.0, 0.1),
        filtered_torque_nm=(0.1, 0.0),
    )
    short_release_baseline = _rotation_rollout(
        "translation_only",
        physical_effect_deg=(0.0,),
        filtered_torque_nm=(0.0,),
    )

    result = _compare_rollouts(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        incremental_translation_limit_m=0.01,
        incremental_rotation_limit_deg=1.0,
        release_force_deadband_n=(0.1, 0.1, 0.1),
        release_torque_deadband_nm=(0.1, 0.1, 0.1),
        torque_release_baseline=short_release_baseline,
    )

    assert result == {
        "passed": False,
        "safety_passed": False,
        "error": (
            "component release baseline and Active MuJoCo sample clocks differ"
        ),
        "sample_clock_mismatches": [
            {
                "baseline": "torque_release_baseline",
                "baseline_samples": 1,
                "candidate_samples": 2,
            }
        ],
    }


def test_mujoco_log_reader_compacts_runtime_only_frame_payload(tmp_path) -> None:
    raw_target = np.eye(4, dtype=float).tolist()
    frame = {
        "event": "frame",
        "teleop_command_timestamp": 3.0,
        "sample_timestamp": 3.1,
        "position_admittance": {
            "sides": {
                side: {
                    "observed_at_monotonic_sec": 2.0,
                    "raw_target_pose_solver_world": raw_target,
                    "raw_wrench_n_nm": [0.0] * 6,
                    "tared_wrench_n_nm": [0.0] * 6,
                    "source_time_sec": 1.9,
                    "tare_generation": 3,
                    "applied_target_pose_solver_world": "large-unused-field",
                }
                for side in ("left", "right")
            }
        },
        "command_by_side": {
            side: {
                "value_qpos": {"deg": [0.0] * 6},
                "metadata": {
                    "target_activity": {"state": "ACTIVE"},
                    "output_qpos_smoothing": {"large": [1] * 100},
                },
            }
            for side in ("left", "right")
        },
        "tracker_channels": {
            f"{side}_wrist_pose": {
                "motion_window": {"quality_valid": True},
                "pose": {"large": [1] * 100},
            }
            for side in ("left", "right")
        },
        "command_backend_by_side": {"large": [1] * 100},
    }
    config = {"event": "config", "payload": {"hz": 15.0}}
    path = tmp_path / "action_debug.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (config, frame)) + "\n",
        encoding="utf-8",
    )

    rows = _read_mujoco_rows(path)

    assert rows[0] == config
    compact = rows[1]
    assert "command_backend_by_side" not in compact
    left = compact["position_admittance"]["sides"]["left"]
    assert left["raw_target_pose_solver_world"] == raw_target
    assert left["tare_generation"] == 3
    assert "applied_target_pose_solver_world" not in left
    assert compact["command_by_side"]["left"] == {
        "value_qpos": {"deg": [0.0] * 6},
        "metadata": {"target_activity": {"state": "ACTIVE"}},
    }
    assert compact["tracker_channels"]["left_wrist_pose"] == {
        "motion_window": {"quality_valid": True}
    }


def test_fixed_rotational_candidate_envelopes_are_bounded_and_ordered() -> None:
    assert ROTATIONAL_ADMITTANCE_ENVELOPES == (
        "cautious-3deg",
        "balanced-5deg",
        "rotation-emphasis-5deg",
        "moderated-8deg",
        "legacy-normal",
    )
    configs = {
        name: rotational_admittance_config_for_envelope(name)
        for name in ROTATIONAL_ADMITTANCE_ENVELOPES
    }

    assert math.degrees(max(configs["cautious-3deg"].max_offset_rad)) == pytest.approx(3.0)
    assert math.degrees(max(configs["balanced-5deg"].max_offset_rad)) == pytest.approx(5.0)
    emphasis = configs["rotation-emphasis-5deg"]
    assert math.degrees(max(emphasis.max_offset_rad)) == pytest.approx(5.0)
    assert math.degrees(max(emphasis.max_velocity_rad_s)) == pytest.approx(9.0)
    assert math.degrees(max(emphasis.max_output_slew_rad_s)) == pytest.approx(9.0)
    assert math.degrees(max(emphasis.max_return_velocity_rad_s)) == pytest.approx(4.5)
    assert math.degrees(max(configs["moderated-8deg"].max_offset_rad)) == pytest.approx(8.0)
    assert max(configs["legacy-normal"].max_offset_rad) == pytest.approx(0.4)
    for config in configs.values():
        assert config.enabled_axes == (True, True, True)
        assert config.control_frame == "solver_world"
        assert config.torque_deadband_nm == (0.04, 0.04, 0.04)
        assert config.torque_sign == (1.0, 1.0, 1.0)


def test_offline_combined_normal_parity_isolated_from_production_installer(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluation(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return ({"passed": True}, {})

    monkeypatch.setattr(
        "teleop_runtime.debug.replay_position_rotational_admittance_mujoco."
        "_replay_recorded_wrench_through_mujoco_evaluation",
        fake_evaluation,
    )

    report = replay_recorded_rotational_admittance_through_mujoco(
        "/tmp/recorded.jsonl",  # type: ignore[arg-type]
        envelope="balanced-5deg",
        offline_combined_normal_parity=True,
    )

    assert captured["admittance_envelope"] == "normal-parity-offline"
    assert captured["max_tool_excursion_m"] == pytest.approx(0.08)
    assert captured["combined_translation_rotation_active"] is False
    assert report["offline_combined_normal_parity"] is True


def test_offline_and_production_combined_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        replay_recorded_rotational_admittance_through_mujoco(
            "/tmp/recorded.jsonl",  # type: ignore[arg-type]
            combined_translation_rotation_active=True,
            offline_combined_normal_parity=True,
        )


def test_production_combined_normal_parity_selects_production_installer_path(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluation(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return (
            {
                "passed": True,
                "production_rotational_active_installer_exercised": True,
            },
            {},
        )

    monkeypatch.setattr(
        "teleop_runtime.debug.replay_position_rotational_admittance_mujoco."
        "_replay_recorded_wrench_through_mujoco_evaluation",
        fake_evaluation,
    )

    report = replay_recorded_rotational_admittance_through_mujoco(
        "/tmp/recorded.jsonl",  # type: ignore[arg-type]
        envelope="balanced-5deg",
        production_combined_normal_parity=True,
    )

    assert captured["admittance_envelope"] == "normal-parity-active"
    assert captured["max_tool_excursion_m"] == pytest.approx(0.08)
    assert captured["max_tool_rotation_excursion_deg"] == pytest.approx(5.5)
    assert captured["combined_translation_rotation_active"] is True
    assert report["production_combined_normal_parity"] is True


def test_formal_teleop_replay_selects_same_production_installer_without_home_guard(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluation(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return ({"passed": True}, {})

    monkeypatch.setattr(
        "teleop_runtime.debug.replay_position_rotational_admittance_mujoco."
        "_replay_recorded_wrench_through_mujoco_evaluation",
        fake_evaluation,
    )

    report = replay_recorded_rotational_admittance_through_mujoco(
        "/tmp/recorded.jsonl",  # type: ignore[arg-type]
        envelope="rotation-emphasis-5deg",
        production_combined_normal_parity=True,
        endpoint_policy="formal_teleop",
    )

    assert captured["admittance_envelope"] == "normal-parity-active"
    assert captured["combined_translation_rotation_active"] is True
    assert captured["endpoint_policy"] == "formal_teleop"
    assert report["active_endpoint_policy"] == "formal_teleop"


@pytest.mark.parametrize(
    "requested_mode",
    ("shadow", "rotation_active", "combined_active"),
)
def test_rotational_replay_accepts_saved_raw_inputs_from_safe_modes(
    requested_mode: str,
) -> None:
    passed, accepted_modes = _recorded_input_contract(
        receipt={
            "enabled": True,
            "requested_mode": requested_mode,
        },
        config_payload={
            "position_control_profile": {
                "name": RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
            }
        },
        rotational_replay=True,
    )

    assert passed is True
    assert accepted_modes == ("shadow", "rotation_active", "combined_active")


def test_translation_replay_still_rejects_active_record_as_counterfactual_input(
) -> None:
    passed, accepted_modes = _recorded_input_contract(
        receipt={
            "enabled": True,
            "requested_mode": "rotation_active",
        },
        config_payload={
            "position_control_profile": {
                "name": RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
            }
        },
        rotational_replay=False,
    )

    assert passed is False
    assert accepted_modes == ("shadow",)


def test_torque_release_gate_accepts_short_small_angular_settling() -> None:
    baseline = _rotation_rollout(
        "baseline",
        physical_effect_deg=(0.0, 0.0, 0.0, 0.0),
        filtered_torque_nm=(0.0, 0.0, 0.0, 0.0),
    )
    candidate = _rotation_rollout(
        "candidate",
        physical_effect_deg=(2.0, 2.0, 2.1, 1.9),
        filtered_torque_nm=(1.0, 0.0, 0.0, 0.0),
    )

    result = _torque_release_transients(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        release_torque_deadband_nm=(0.04, 0.04, 0.04),
    )

    assert result["passed"] is True
    assert result["event_count"] == 1
    assert result["max_outward_catchup_deg"] == pytest.approx(0.1)
    assert result["max_outward_catchup_duration_sec"] == pytest.approx(0.05)


def test_torque_release_gate_treats_picosecond_timestamp_error_as_boundary() -> None:
    baseline = _rotation_rollout(
        "baseline",
        physical_effect_deg=(0.0, 0.0, 0.0, 0.0),
        filtered_torque_nm=(0.0, 0.0, 0.0, 0.0),
    )
    candidate = _rotation_rollout(
        "candidate",
        physical_effect_deg=(2.0, 2.0, 2.0, 2.1),
        filtered_torque_nm=(1.0, 0.0, 0.0, 0.0),
    )
    timestamps = [0.0, 0.05, 0.10, 0.150000000005]
    baseline.sample_times_sec = list(timestamps)
    candidate.sample_times_sec = list(timestamps)

    result = _torque_release_transients(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        release_torque_deadband_nm=(0.04, 0.04, 0.04),
    )

    assert result["passed"] is True
    assert result["max_outward_catchup_duration_sec"] > 0.1
    assert result["max_outward_catchup_duration_sec"] == pytest.approx(0.1)


def test_torque_release_gate_rejects_large_delayed_angular_catchup() -> None:
    baseline = _rotation_rollout(
        "baseline",
        physical_effect_deg=(0.0, 0.0, 0.0, 0.0),
        filtered_torque_nm=(0.0, 0.0, 0.0, 0.0),
    )
    candidate = _rotation_rollout(
        "candidate",
        physical_effect_deg=(2.0, 2.0, 2.3, 2.6),
        filtered_torque_nm=(1.0, 0.0, 0.0, 0.0),
    )

    result = _torque_release_transients(
        baseline=baseline,
        candidate=candidate,
        selected_side="left",
        release_torque_deadband_nm=(0.04, 0.04, 0.04),
    )

    assert result["passed"] is False
    assert result["max_outward_catchup_deg"] == pytest.approx(0.6)


def test_combined_release_uses_translation_only_torque_counterfactual() -> None:
    baseline = _rotation_rollout(
        "baseline",
        physical_effect_deg=(0.0, 0.0, 0.0, 0.0),
        filtered_torque_nm=(0.0, 0.0, 0.0, 0.0),
    )
    combined = _rotation_rollout(
        "combined",
        physical_effect_deg=(2.0, 2.0, 2.3, 2.6),
        filtered_torque_nm=(1.0, 0.0, 0.0, 0.0),
    )
    translation_only = _rotation_rollout(
        "translation_only",
        physical_effect_deg=(2.0, 2.0, 2.3, 2.6),
        filtered_torque_nm=(0.0, 0.0, 0.0, 0.0),
    )
    for rollout in (baseline, combined, translation_only):
        rollout.report.update(
            completed=True,
            nonzero_modifier_ticks=(0 if rollout is baseline else 1),
            active_qref_motion_ticks=(0 if rollout is baseline else 1),
        )

    direct = _torque_release_transients(
        baseline=baseline,
        candidate=combined,
        selected_side="left",
        release_torque_deadband_nm=(0.04, 0.04, 0.04),
    )
    isolated = _compare_rollouts(
        baseline=baseline,
        candidate=combined,
        selected_side="left",
        incremental_translation_limit_m=0.08,
        incremental_rotation_limit_deg=5.5,
        release_force_deadband_n=(0.45, 0.45, 0.45),
        release_torque_deadband_nm=(0.04, 0.04, 0.04),
        torque_release_baseline=translation_only,
    )

    assert direct["passed"] is False
    assert isolated["checks"][
        "torque_release_no_delayed_outward_catchup"
    ] is True
    assert isolated["torque_release_transients"]["passed"] is True
    assert isolated["release_counterfactual_baselines"]["torque"] == (
        "translation_only"
    )
