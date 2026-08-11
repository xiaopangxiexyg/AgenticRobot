from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_motion.rm65.defaults import (
    DEFAULT_LEFT_ARM_QPOS,
    DEFAULT_RIGHT_ARM_QPOS,
)
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
    resolve_rm65_bimanual_position_profile,
)
from teleop_runtime.debug.replay_position_admittance_ik_trace import (
    DEFAULT_LEFT_TOOL_PROFILE,
    DEFAULT_RIGHT_TOOL_PROFILE,
    _build_retargeter,
    _command_qpos,
    _context,
    _source_command,
    _synchronize_counterfactual_state,
    _tool_pose,
    replay_recorded_force_through_shared_ik,
)


def test_source_command_restores_recorded_motion_window_for_force_modifier() -> None:
    target = np.eye(4, dtype=float)
    frame = {
        "command_by_side": {
            "left": {
                "metadata": {
                    "target_activity": {
                        "state": "MOVING",
                        "inactive": False,
                        "hold_weight": 0.0,
                        "evidence": {"linear_speed_m_s": 0.05},
                    }
                }
            }
        },
        "tracker_channels": {
            "left_wrist_pose": {
                "motion_window": {
                    "quality_valid": True,
                    "linear_speed_m_s": 0.05,
                    "angular_speed_deg_s": 2.0,
                }
            }
        },
    }

    command = _source_command(
        frame=frame,
        side="left",
        target=target,
        modifier_active=True,
    )

    tracker = command.metadata["tracker_motion_intent"]
    modifier = command.metadata["pose_action_motion_intent"]
    assert tracker["motion_window"]["quality_valid"] is True
    assert modifier["motion_window"] == tracker["motion_window"]
    assert modifier["motion_window"] is not tracker["motion_window"]
    assert modifier["evidence"] == tracker["evidence"]


def test_offline_retargeter_uses_v2_balanced_acceleration_contract() -> None:
    profile = resolve_rm65_bimanual_position_profile(
        RM65_BIMANUAL_POSITION_V2_BALANCED_NAME
    )
    retargeter, _ = _build_retargeter(
        initial_qpos={
            "left": DEFAULT_LEFT_ARM_QPOS.copy(),
            "right": DEFAULT_RIGHT_ARM_QPOS.copy(),
        },
        config_payload={
            "position_control_profile": {
                "name": RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
                "scene_profile": "dual_arm_hand_proxy_open",
            }
        },
        left_tool_pose=_tool_pose(Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"),
        right_tool_pose=_tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE),
            side="right",
        ),
    )

    assert np.asarray(
        retargeter.config.joint_acceleration_limit_rad_s2,
        dtype=float,
    ) == pytest.approx(profile.command_execution.max_acceleration_rad_s2)


def _write_exact_pose_log(path: Path) -> Path:
    initial_qpos = {
        "left": DEFAULT_LEFT_ARM_QPOS.copy(),
        "right": DEFAULT_RIGHT_ARM_QPOS.copy(),
    }
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
        left_tool_pose=_tool_pose(Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"),
        right_tool_pose=_tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE),
            side="right",
        ),
    )
    targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    rows: list[dict[str, object]] = [
        {"event": "config", "payload": config_payload},
        {
            "event": "position_admittance_config",
            "payload": {
                "enabled": True,
                "requested_mode": "shadow",
                "schema_version": "position-admittance-v3",
                "active_sides": ["left", "right"],
                "config": {
                    "enabled_axes": [True, True, True],
                    "control_frame": "solver_world",
                    "response_model": "profile_compliance",
                    "virtual_mass_kg": [2.0, 2.0, 2.0],
                    "damping_n_s_m": [50.0, 50.0, 50.0],
                    "stiffness_n_m": [250.0, 250.0, 250.0],
                    "force_deadband_n": [0.45, 0.45, 0.45],
                    "force_filter_tau_sec": 0.010857362,
                    "force_sign": [1.0, 1.0, 1.0],
                    "max_offset_m": [0.005, 0.005, 0.005],
                    "max_velocity_m_s": [0.01, 0.01, 0.01],
                    "max_return_velocity_m_s": [0.01, 0.01, 0.01],
                    "max_output_slew_m_s": [0.01, 0.01, 0.01],
                    "return_force_deadband_n": [0.45, 0.45, 0.45],
                    "return_rate_s": 1.0,
                    "response_reference_hz": 40.0,
                    "stale_after_sec": 0.15,
                    "max_force_dt_sec": 0.15,
                },
            },
        },
    ]
    qpos_deg = {
        side: np.rad2deg(qpos).astype(float).tolist()
        for side, qpos in initial_qpos.items()
    }
    for index in range(16):
        observed = 1.0 + index / 15.0
        force_x = 2.0 if 2 <= index < 9 else 0.0
        rows.append(
            {
                "event": "frame",
                "step_index": index,
                "teleop_command_timestamp": observed,
                "command_by_side": {
                    side: {
                        "value_qpos": {"deg": qpos_deg[side]},
                        "metadata": {
                            "target_activity": {
                                "state": "ACTIVE",
                                "inactive": False,
                                "hold_weight": 0.0,
                            }
                        },
                    }
                    for side in ("left", "right")
                },
                "position_admittance": {
                    "sides": {
                        side: {
                            "observed_at_monotonic_sec": observed,
                            "source_time_sec": observed - 0.02,
                            "filtered_force_control_n": (
                                [force_x, 0.0, 0.0]
                                if side == "left"
                                else [0.0, 0.0, 0.0]
                            ),
                            "raw_target_pose_solver_world": targets[side]
                            .astype(float)
                            .tolist(),
                            "applied_target_translation": targets[side][
                                :3, 3
                            ]
                            .astype(float)
                            .tolist(),
                            "applied_total_offset_m": [0.0, 0.0, 0.0],
                        }
                        for side in ("left", "right")
                    }
                },
            }
        )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_exact_pose_force_trace_passes_shared_ik_closed_loop(
    tmp_path: Path,
) -> None:
    log = _write_exact_pose_log(tmp_path / "action_debug_exact_pose.jsonl")

    report = replay_recorded_force_through_shared_ik(
        log,
        release_tail_sec=0.25,
    )

    assert report["passed"] is True
    assert report["nonzero_candidate_count"] > 0
    assert report["translation_gate_eligible_count"] == report[
        "nonzero_candidate_count"
    ]
    assert report["shared_ik_completed_count"] == report[
        "translation_gate_eligible_count"
    ]
    assert report["exact_recorded_rotation_count"] == report["frame_count"]
    # The exact default pose can legitimately select local_step_hold under the
    # frozen V2 Balanced IK envelope.  Reaching a nonzero raw IK candidate is
    # the invariant here; final admission remains safety-owned and may hold.
    assert report["selected_raw_candidate_step_max_deg"]["max"] > 0.0
    assert report["opposite_incremental_fk_translation_m"]["max"] < 1e-4
    assert all(
        rollout["passed"]
        for rollout in report["continuous_rollouts"].values()
    )
    for rollout in report["continuous_rollouts"].values():
        assert rollout["counterfactual_baseline"] == (
            "same_tick_same_admitted_state_nominal_preview"
        )
        assert rollout["common_seed_q_error_rad"]["max"] == 0.0
    config = report["candidate_config"]
    assert config["max_offset_m"] == (0.002,) * 3
    assert config["max_velocity_m_s"] == (0.005,) * 3
    assert config["max_return_velocity_m_s"] == (0.005,) * 3
    assert config["max_output_slew_m_s"] == (0.005,) * 3


def test_exact_pose_force_trace_passes_normal_profile_shared_ik_closed_loop(
    tmp_path: Path,
) -> None:
    log = _write_exact_pose_log(
        tmp_path / "action_debug_exact_pose_normal_profile.jsonl"
    )

    report = replay_recorded_force_through_shared_ik(
        log,
        max_offset_m=0.08,
        max_speed_m_s=0.06,
        max_return_speed_m_s=0.03,
        max_output_slew_m_s=0.06,
        # Observe convergence beyond the position path's 1 mm hold band;
        # a short C0 tail only proves that the modifier itself returned.
        release_tail_sec=10.0,
    )

    assert report["passed"] is True
    assert report["nonzero_candidate_count"] > 0
    assert report["translation_gate_eligible_count"] == report[
        "nonzero_candidate_count"
    ]
    assert report["shared_ik_completed_count"] == report[
        "translation_gate_eligible_count"
    ]
    assert all(
        rollout["passed"]
        for rollout in report["continuous_rollouts"].values()
    )
    config = report["candidate_config"]
    assert config["max_offset_m"] == (0.08,) * 3
    assert config["max_velocity_m_s"] == (0.06,) * 3
    assert config["max_return_velocity_m_s"] == (0.03,) * 3
    assert config["max_output_slew_m_s"] == (0.06,) * 3


def test_counterfactual_release_forks_from_authoritative_ik_state() -> None:
    initial_qpos = {
        "left": DEFAULT_LEFT_ARM_QPOS.copy(),
        "right": DEFAULT_RIGHT_ARM_QPOS.copy(),
    }
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
    tool_args = {
        "left_tool_pose": _tool_pose(
            Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"
        ),
        "right_tool_pose": _tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE), side="right"
        ),
    }
    nominal, _ = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        **tool_args,
    )
    modified, _ = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        **tool_args,
    )
    raw_targets = modified.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    frame = {
        "teleop_command_timestamp": 1.0,
        "command_by_side": {
            side: {
                "metadata": {
                    "target_activity": {
                        "state": "ACTIVE",
                        "inactive": False,
                        "hold_weight": 0.0,
                    }
                }
            }
            for side in ("left", "right")
        },
    }
    modified_q = {side: initial_qpos[side].copy() for side in ("left", "right")}
    modified_q["left"][0] += np.deg2rad(0.5)
    modified.set_current_qpos_pair(
        left_qpos=modified_q["left"],
        right_qpos=modified_q["right"],
        reset_tracking=False,
    )
    modified_targets = modified.solver_world_targets_for_qpos_pair(
        left_qpos=modified_q["left"],
        right_qpos=modified_q["right"],
    )
    # Commit one no-motion tick so the authoritative branch also has its own
    # target/activity/smoothing history, not merely a different q seed.
    command = modified.build_command_for_context(
        _context(
            frame=frame,
            targets=modified_targets,
            modifier_active_side="left",
        )
    )
    for side in ("left", "right"):
        modified_q[side], _ = _command_qpos(command, side=side)
    modified.set_current_qpos_pair(
        left_qpos=modified_q["left"],
        right_qpos=modified_q["right"],
        reset_tracking=False,
    )

    nominal_before = nominal.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    modified_before = modified.solver_world_targets_for_qpos_pair(
        left_qpos=modified_q["left"],
        right_qpos=modified_q["right"],
    )
    assert np.linalg.norm(
        modified_before["left"][:3, 3] - nominal_before["left"][:3, 3]
    ) > 1e-5

    seed_error = _synchronize_counterfactual_state(
        authoritative=modified,
        counterfactual=nominal,
        authoritative_qpos=modified_q,
    )
    assert seed_error == 0.0

    nominal_release = nominal.build_command_for_context(
        _context(frame=frame, targets=raw_targets)
    )
    modified_release = modified.build_command_for_context(
        _context(frame=frame, targets=raw_targets)
    )
    for side in ("left", "right"):
        nominal_q, nominal_metadata = _command_qpos(
            nominal_release, side=side
        )
        modified_release_q, modified_metadata = _command_qpos(
            modified_release, side=side
        )
        np.testing.assert_allclose(
            modified_release_q,
            nominal_q,
            atol=1e-12,
            rtol=0.0,
        )
        assert modified_metadata["ik_branch_candidate"] == (
            nominal_metadata["ik_branch_candidate"]
        )
        assert modified_metadata["collision_avoidance_status"] == (
            nominal_metadata["collision_avoidance_status"]
        )
