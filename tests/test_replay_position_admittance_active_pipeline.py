from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.position_control.rotational_admittance import (
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    rotational_admittance_config_for_profile,
)
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED,
    RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1,
    RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1_NAME,
    RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V2,
    RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V2_NAME,
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
)
from robot_motion.position_control.sample_aligned_trajectory import (
    SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
    SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
)
from sim.contracts import TimingSpec
from sim.mujoco.actuation import (
    identified_bimanual_dmtac_gripper_profile,
    rm65_bimanual_udp_feedback_profile,
)
from teleop_runtime.debug.replay_position_admittance_active_pipeline import (
    replay_recorded_wrench_through_active_pipeline,
)
from teleop_runtime.debug.replay_position_admittance_ik_trace import (
    DEFAULT_LEFT_TOOL_PROFILE,
    DEFAULT_RIGHT_TOOL_PROFILE,
    _build_retargeter,
    _tool_pose,
)
from teleop_runtime.debug.replay_position_admittance_mujoco import (
    _profile_execution_config,
    replay_recorded_wrench_through_mujoco,
)
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
    POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA,
    POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG,
)


SIDES = ("left", "right")


def _initial_qpos() -> dict[str, np.ndarray]:
    return {
        "left": np.deg2rad(
            [87.794, 52.686, 116.974, 4.320, -81.283, -1.525]
        ),
        "right": np.deg2rad(
            [-86.568, 48.118, 114.321, -1.812, -74.008, 4.020]
        ),
    }


def _write_raw_wrench_log(
    path: Path,
    *,
    force_n: float,
    force_side: str = "left",
    torque_nm: float = 0.0,
    max_offset_m: float = 0.002,
    max_rate_m_s: float = 0.005,
    max_return_rate_m_s: float | None = None,
) -> Path:
    if force_side not in SIDES:
        raise ValueError("force_side must be left or right")
    initial_qpos = _initial_qpos()
    config_payload = {
        "position_control_profile": {
            "name": RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
            "scene_profile": "dual_arm_hand_proxy_open",
        },
        "arm_max_angvel_effective_rad_s": [
            5.6,
            5.6,
            7.2,
            12.0,
            12.0,
            16.0,
        ],
    }
    retargeter, _ = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=_tool_pose(
            Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"
        ),
        right_tool_pose=_tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE), side="right"
        ),
    )
    targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    rotation_solver_from_tool = (
        targets[force_side][:3, :3]
        @ retargeter._tool_tcp_from_solver[force_side][:3, :3]
    )
    tool_force = rotation_solver_from_tool.T @ np.array(
        [-float(force_n), 0.0, 0.0]
    )
    tool_torque = rotation_solver_from_tool.T @ np.array(
        [0.0, 0.0, float(torque_nm)]
    )
    config = {
        "enabled_axes": [True, True, True],
        "control_frame": "solver_world",
        "response_model": "profile_compliance",
        "virtual_mass_kg": [2.0, 2.0, 2.0],
        "damping_n_s_m": [50.0, 50.0, 50.0],
        "stiffness_n_m": [250.0, 250.0, 250.0],
        "force_deadband_n": [0.45, 0.45, 0.45],
        "force_filter_tau_sec": 0.010857362,
        "force_sign": [1.0, 1.0, 1.0],
        "max_offset_m": [max_offset_m] * 3,
        "max_velocity_m_s": [max_rate_m_s] * 3,
        "max_return_velocity_m_s": [
            max_rate_m_s
            if max_return_rate_m_s is None
            else max_return_rate_m_s
        ]
        * 3,
        "max_output_slew_m_s": [max_rate_m_s] * 3,
        "return_force_deadband_n": [0.45, 0.45, 0.45],
        "return_rate_s": 1.0,
        "response_reference_hz": 40.0,
        "stale_after_sec": 0.15,
        "max_force_dt_sec": 0.15,
    }
    rows: list[dict[str, object]] = [
        {"event": "config", "payload": config_payload},
        {
            "event": "position_admittance_config",
            "payload": {
                "enabled": True,
                "requested_mode": "shadow",
                "schema_version": "position-admittance-v5",
                "active_sides": list(SIDES),
                "config": config,
            },
        },
    ]
    qpos_deg = {
        side: np.rad2deg(initial_qpos[side]).astype(float).tolist()
        for side in SIDES
    }
    for frame_index in range(40):
        observed_at = 1.0 + frame_index / 15.0
        active_force = 5 <= frame_index < 25
        rows.append(
            {
                "event": "frame",
                "step_index": frame_index,
                "teleop_command_timestamp": observed_at,
                "command_by_side": {
                    side: {
                        "value_qpos": {"deg": qpos_deg[side]},
                        "metadata": {
                            "target_activity": {
                                "state": "INACTIVE",
                                "inactive": True,
                                "hold_weight": 1.0,
                            }
                        },
                    }
                    for side in SIDES
                },
                "position_admittance": {
                    "sides": {
                        side: {
                            "observed_at_monotonic_sec": observed_at,
                            "source_time_sec": observed_at - 0.02,
                            "raw_wrench_n_nm": (
                                [
                                    *tool_force.astype(float).tolist(),
                                    *tool_torque.astype(float).tolist(),
                                ]
                                if side == force_side and active_force
                                else [0.0] * 6
                            ),
                            "tared_wrench_n_nm": (
                                [
                                    *tool_force.astype(float).tolist(),
                                    *tool_torque.astype(float).tolist(),
                                ]
                                if side == force_side and active_force
                                else [0.0] * 6
                            ),
                            "raw_target_pose_solver_world": targets[side]
                            .astype(float)
                            .tolist(),
                        }
                        for side in SIDES
                    }
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_raw_wrench_active_pipeline_exercises_shared_ik(
    tmp_path: Path,
) -> None:
    log = _write_raw_wrench_log(
        tmp_path / "action_debug_active_pipeline.jsonl",
        force_n=5.0,
    )

    report = replay_recorded_wrench_through_active_pipeline(
        log,
        release_tail_sec=6.0,
    )

    assert report["passed"] is True
    assert report["active_config"]["max_offset_m"] == (0.002,) * 3
    assert report["active_config"]["max_velocity_m_s"] == (0.005,) * 3
    assert report["active_config"]["max_output_slew_m_s"] == (0.005,) * 3
    for rollout in report["rollouts"].values():
        assert rollout["passed"] is True
        coverage = next(
            check
            for check in rollout["checks"]
            if check["name"] == "nonzero_active_ik_coverage"
        )
        assert coverage["detail"]["active_qref_motion_ticks"] > 0
        assert rollout["opposite_final_qpos_delta_deg"] < 0.01


def test_sub_hold_wrench_is_not_promoted_as_active_ik_coverage(
    tmp_path: Path,
) -> None:
    log = _write_raw_wrench_log(
        tmp_path / "action_debug_sub_hold_pipeline.jsonl",
        force_n=0.5,
    )

    report = replay_recorded_wrench_through_active_pipeline(
        log,
        release_tail_sec=6.0,
    )

    assert report["passed"] is False
    for rollout in report["rollouts"].values():
        coverage = next(
            check
            for check in rollout["checks"]
            if check["name"] == "nonzero_active_ik_coverage"
        )
        assert coverage["passed"] is False
        assert coverage["detail"]["nonzero_modifier_ticks"] > 0
        assert coverage["detail"]["active_qref_motion_ticks"] == 0


def test_normal_shadow_envelope_is_replayed_under_cautious_active_caps(
    tmp_path: Path,
) -> None:
    log = _write_raw_wrench_log(
        tmp_path / "action_debug_normal_shadow_pipeline.jsonl",
        force_n=5.0,
        max_offset_m=0.08,
        max_rate_m_s=0.06,
        max_return_rate_m_s=0.03,
    )

    report = replay_recorded_wrench_through_active_pipeline(
        log,
        release_tail_sec=6.0,
    )

    assert report["recorded_config"]["max_offset_m"] == [0.08] * 3
    assert report["active_config"]["max_offset_m"] == (0.002,) * 3
    assert report["active_config"]["max_velocity_m_s"] == (0.005,) * 3
    assert report["active_config"]["max_return_velocity_m_s"] == (0.005,) * 3
    assert report["active_config"]["max_output_slew_m_s"] == (0.005,) * 3
    assert report["passed"] is True
    for rollout in report["rollouts"].values():
        assert rollout["offset_norm_m"]["max"] <= 0.002 + 1e-9


@pytest.mark.parametrize(
    (
        "active_envelope",
        "force_n",
        "max_offset_m",
        "max_rate_m_s",
        "max_return_rate_m_s",
    ),
    (
        ("half_normal", 14.0, 0.04, 0.03, 0.015),
        ("normal_parity", 29.0, 0.08, 0.06, 0.03),
    ),
)
def test_named_larger_active_envelopes_close_through_shared_ik_and_release(
    tmp_path: Path,
    active_envelope: str,
    force_n: float,
    max_offset_m: float,
    max_rate_m_s: float,
    max_return_rate_m_s: float,
) -> None:
    log = _write_raw_wrench_log(
        tmp_path / f"action_debug_{active_envelope}_pipeline.jsonl",
        force_n=force_n,
        max_offset_m=max_offset_m,
        max_rate_m_s=max_rate_m_s,
        max_return_rate_m_s=max_return_rate_m_s,
    )

    report = replay_recorded_wrench_through_active_pipeline(
        log,
        active_envelope=active_envelope,
        release_tail_sec=10.0,
    )

    assert report["passed"] is True
    assert report["active_envelope"] == active_envelope
    assert report["active_config"]["max_offset_m"] == (
        max_offset_m,
    ) * 3
    assert report["active_config"]["max_velocity_m_s"] == (
        max_rate_m_s,
    ) * 3
    assert report["active_config"]["max_return_velocity_m_s"] == (
        max_return_rate_m_s,
    ) * 3
    for rollout in report["rollouts"].values():
        assert rollout["passed"] is True
        assert rollout["offset_norm_m"]["max"] <= max_offset_m + 1e-9
        assert rollout["offset_norm_m"]["max"] >= max_offset_m - 1e-6
        assert rollout["offset_slew_ratio"]["max"] <= 1.0 + 1e-6
        final = next(
            check
            for check in rollout["checks"]
            if check["name"] == "zero_force_release_rejoins_target"
        )
        assert final["passed"] is True


def test_mujoco_replay_execution_config_is_exact_v2_balanced_profile() -> None:
    timing = TimingSpec(
        sample_hz=15.0,
        policy_hz=15.0,
        reference_hz=60.0,
        ik_hz=60.0,
        command_hz=120.0,
        physics_hz=1000.0,
        parameter_source=RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    )
    config = _profile_execution_config(
        timing=timing,
        arm_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_profile=identified_bimanual_dmtac_gripper_profile(),
    )
    formal = RM65_BIMANUAL_POSITION_V2_BALANCED.command_execution

    assert config.realtime_worker is False
    assert config.position_command_planner == "acceleration_limited"
    assert config.sample_aligned_source_hz == pytest.approx(15.0)
    assert config.sample_aligned_horizon_source_steps == 2
    assert config.max_velocity_rad_s == tuple(formal.max_velocity_rad_s)
    assert config.max_acceleration_rad_s2 == tuple(
        formal.max_acceleration_rad_s2
    )
    assert config.max_command_step_deg == formal.max_command_step_deg
    assert config.target_velocity_filter_tau_sec == (
        formal.target_velocity_filter_tau_sec
    )
    assert config.velocity_ramp_sec == formal.velocity_ramp_sec
    assert config.follow_qpos_alpha == formal.follow_qpos_alpha
    assert config.degraded_speed_scale == formal.degraded_speed_scale
    assert config.actuation_profile is not None
    assert config.gripper_actuation_profile is not None


def test_mujoco_replay_execution_config_is_exact_60hz_jerk_profile() -> None:
    profile = RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1
    timing = TimingSpec(
        sample_hz=profile.rates.sample_hz,
        policy_hz=profile.rates.sample_hz,
        reference_hz=profile.rates.reference_hz,
        ik_hz=profile.rates.ik_hz,
        command_hz=profile.rates.command_hz,
        physics_hz=1000.0,
        parameter_source=profile.name,
    )
    config = _profile_execution_config(
        timing=timing,
        arm_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_profile=identified_bimanual_dmtac_gripper_profile(),
        position_execution_profile_name=(
            RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1_NAME
        ),
    )
    execution = profile.command_execution

    assert config.position_command_planner == "sample_aligned_jerk_limited"
    assert config.sample_aligned_source_hz == pytest.approx(60.0)
    assert (
        config.sample_aligned_source_latch_mode
        == SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET
    )
    assert config.sample_aligned_horizon_source_steps == 2
    assert config.sample_aligned_safety_horizon_source_steps == 20
    assert config.velocity_ramp_sec == pytest.approx(0.05)
    assert config.follow_qpos_alpha == pytest.approx(1.0)
    assert config.sample_aligned_scale_degraded_target is False
    assert config.sample_aligned_jerk_regularization == pytest.approx(1e-6)
    assert config.sample_aligned_jerk_delta_regularization == pytest.approx(1e-4)
    assert config.max_velocity_rad_s == tuple(execution.max_velocity_rad_s)
    assert config.max_acceleration_rad_s2 == tuple(
        execution.max_acceleration_rad_s2
    )
    assert config.max_command_step_deg == pytest.approx(
        execution.max_command_step_deg
    )
    assert config.actuation_profile is not None
    assert config.gripper_actuation_profile is not None


def test_mujoco_replay_execution_config_v2_changes_only_source_latch() -> None:
    profile = RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V2
    timing = TimingSpec(
        sample_hz=profile.rates.sample_hz,
        policy_hz=profile.rates.sample_hz,
        reference_hz=profile.rates.reference_hz,
        ik_hz=profile.rates.ik_hz,
        command_hz=profile.rates.command_hz,
        physics_hz=1000.0,
        parameter_source=profile.name,
    )
    v1 = _profile_execution_config(
        timing=timing,
        arm_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_profile=identified_bimanual_dmtac_gripper_profile(),
        position_execution_profile_name=(
            RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V1_NAME
        ),
    )
    v2 = _profile_execution_config(
        timing=timing,
        arm_profile=rm65_bimanual_udp_feedback_profile(),
        gripper_profile=identified_bimanual_dmtac_gripper_profile(),
        position_execution_profile_name=(
            RM65_BIMANUAL_POSITION_V2_BALANCED_60HZ_JERK_V2_NAME
        ),
    )

    assert (
        v2.sample_aligned_source_latch_mode
        == SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
    )
    assert replace(
        v2,
        sample_aligned_source_latch_mode=(
            SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET
        ),
    ) == v1


def test_raw_wrench_active_pipeline_runs_through_physical_mujoco(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    log = _write_raw_wrench_log(
        tmp_path / "action_debug_active_mujoco.jsonl",
        force_n=5.0,
    )

    report = replay_recorded_wrench_through_mujoco(
        log,
        # Match the Active-pipeline release proof: three seconds still leaves
        # ~35 um of the bounded response state, while six seconds reaches the
        # predeclared 10 um final modifier gate.
        release_tail_sec=6.0,
        interpolation_modes=("zoh",),
    )

    assert report["passed"] is True
    assert report["safety_passed"] is True
    assert report["coverage_passed"] is True
    assert report["input"]["recorded_position_profile"] == (
        RM65_BIMANUAL_POSITION_V2_BALANCED_NAME
    )
    for rollout in report["rollouts"].values():
        assert rollout["completed"] is True
        assert rollout["execution"]["execution_mode"] == (
            "actuated_position_servo"
        )
        assert rollout["frame_check"]["passed"] is True
        assert all(rollout["checks"].values())
    comparison = report["comparisons"]["zoh"]
    assert comparison["passed"] is True
    assert comparison["nonzero_active_physical_ticks"] > 0
    assert comparison["selected_measured_translation_m"]["max"] > 0.0


def test_right_combined_formal_active_runs_through_physical_mujoco(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    log = _write_raw_wrench_log(
        tmp_path / "action_debug_right_combined_formal_mujoco.jsonl",
        force_side="right",
        force_n=5.0,
        torque_nm=1.0,
    )

    report = replay_recorded_wrench_through_mujoco(
        log,
        side="right",
        admittance_envelope="normal-parity-active",
        release_tail_sec=10.0,
        interpolation_modes=("zoh",),
        max_tool_excursion_m=0.08,
        rotational_config=rotational_admittance_config_for_profile(
            ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
        ),
        max_tool_rotation_excursion_deg=5.5,
        combined_translation_rotation_active=True,
        endpoint_policy=(
            POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        ),
    )

    assert report["passed"] is True
    assert report["side"] == "right"
    assert report["production_normal_parity_active"] is True
    assert report["production_rotational_active_installer_exercised"] is True
    active = report["rollouts"]["zoh_active"]
    assert active["completed"] is True
    assert active["endpoint_excursion_guard"] is None
    assert active["active_offset_norm_m"]["max"] > 0.0
    assert active["active_rotation_offset_norm_rad"]["max"] > 0.0
    comparison = report["comparisons"]["zoh"]
    assert comparison["nonzero_active_physical_ticks"] > 0
    assert comparison["selected_measured_translation_m"]["max"] > 0.0
    assert comparison["selected_measured_rotation_deg"]["max"] > 0.0


@pytest.mark.parametrize(
    (
        "envelope_name",
        "force_n",
        "max_offset_m",
        "max_rate_m_s",
        "max_return_rate_m_s",
        "expected_qpos_alpha",
        "expected_release_passed",
    ),
    (
        ("half_normal", 14.0, 0.04, 0.03, 0.015, 0.85, True),
        ("normal_parity", 29.0, 0.08, 0.06, 0.03, 0.80, False),
    ),
)
def test_larger_envelope_mujoco_abrupt_release_retains_frozen_catchup_gate(
    tmp_path: Path,
    envelope_name: str,
    force_n: float,
    max_offset_m: float,
    max_rate_m_s: float,
    max_return_rate_m_s: float,
    expected_qpos_alpha: float,
    expected_release_passed: bool,
) -> None:
    pytest.importorskip("mujoco")
    log = _write_raw_wrench_log(
        tmp_path / f"action_debug_{envelope_name}_mujoco.jsonl",
        force_n=force_n,
        max_offset_m=max_offset_m,
        max_rate_m_s=max_rate_m_s,
        max_return_rate_m_s=max_return_rate_m_s,
    )

    report = replay_recorded_wrench_through_mujoco(
        log,
        admittance_envelope="recorded-shadow",
        release_tail_sec=10.0,
        interpolation_modes=("zoh",),
        max_tool_excursion_m=0.08,
    )

    assert report["passed"] is expected_release_passed
    assert report["safety_passed"] is expected_release_passed
    comparison = report["comparisons"]["zoh"]
    assert comparison["checks"][
        "force_release_no_delayed_outward_catchup"
    ] is expected_release_passed
    active_rollout = report["rollouts"]["zoh_active"]
    assert active_rollout["checks"]["active_tool_endpoint_excursion"] is True
    assert active_rollout["tool_endpoint_excursion_m"]["max"] <= 0.08 + 1e-12
    assert active_rollout["endpoint_excursion_guard"]["hold_latched"] is False
    release = comparison["force_release_transients"]
    assert release["passed"] is expected_release_passed
    if expected_release_passed:
        assert release["max_outward_catchup_m"] <= 0.001
    else:
        assert envelope_name == "normal_parity"
        assert release["max_outward_catchup_m"] > 0.001
        assert report["hardware_authorization"] is False
    assert release["max_outward_catchup_duration_sec"] <= 0.1
    fast_path = report["rollouts"]["zoh_active"]["execution"][
        "active_modifier_qpos_fast_path"
    ]
    assert fast_path["enabled"] is True
    assert fast_path["requested_alpha"] == pytest.approx(expected_qpos_alpha)
    assert fast_path["configured_alpha_cap"] == pytest.approx(
        POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA
    )
    assert fast_path["micro_reversal_hold_deg"] == pytest.approx(
        POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG
    )


def test_half_normal_mujoco_endpoint_guard_holds_before_larger_qref_dispatch(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    log = _write_raw_wrench_log(
        tmp_path / "action_debug_half_normal_endpoint_stop.jsonl",
        force_n=14.0,
        max_offset_m=0.04,
        max_rate_m_s=0.03,
        max_return_rate_m_s=0.015,
    )

    report = replay_recorded_wrench_through_mujoco(
        log,
        admittance_envelope="recorded-shadow",
        release_tail_sec=10.0,
        interpolation_modes=("zoh",),
        max_tool_excursion_m=0.02,
    )

    active = report["rollouts"]["zoh_active"]
    assert active["completed"] is True
    assert active["model_errors"] == []
    assert active["endpoint_excursion_guard"]["hold_latched"] is True
    assert active["endpoint_excursion_guard"]["last_excursion_m"] > 0.02
    assert (
        active["endpoint_excursion_guard"]["last_accepted_excursion_m"]
        <= 0.02 + 1e-12
    )
    assert (
        active["dispatched_tool_endpoint_excursion_m"]["max"]
        <= 0.02 + 1e-12
    )
