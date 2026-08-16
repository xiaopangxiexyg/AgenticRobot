from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_motion.position_control.cartesian_admittance import (
    CartesianAdmittanceConfig,
)
from robot_motion.position_control.rotational_admittance import (
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    rotation_matrix_from_rotvec,
    rotational_admittance_config_for_profile,
)
from robot_motion.rm65.position_profile import (
    RM65_BIMANUAL_POSITION_V2_BALANCED_NAME,
)
from teleop_runtime.debug.replay_position_admittance_ik_trace import (
    DEFAULT_LEFT_TOOL_PROFILE,
    DEFAULT_RIGHT_TOOL_PROFILE,
    _build_retargeter,
    _command_qpos,
    _context,
    _tool_pose,
)
from teleop_runtime.real_teleop.position_admittance_shadow import (
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
    admitted_solver_world_poses_from_joint_command,
    install_position_admittance_active,
)


SIDES = ("left", "right")


class _Monitor:
    def __init__(self) -> None:
        self.source_time = 0.0
        self.values_by_side = {side: [0.0] * 6 for side in SIDES}
        self.fault = None

    def snapshot(self) -> dict[str, object]:
        return {
            "fault": self.fault,
            "last_reading_by_side": {
                side: {
                    "side": side,
                    "source": "tool",
                    "values": list(values),
                    "monotonic_time_sec": self.source_time,
                }
                for side, values in self.values_by_side.items()
            },
        }


def _initial_qpos() -> dict[str, np.ndarray]:
    # A regular, non-singular pose captured by the normal shadow run.  The
    # cautious profile's 2 mm overlay is reachable here and exercises shared
    # IK rather than being entirely swallowed by current_pose_hold.
    return {
        "left": np.deg2rad(
            [87.794, 52.686, 116.974, 4.320, -81.283, -1.525]
        ),
        "right": np.deg2rad(
            [-86.568, 48.118, 114.321, -1.812, -74.008, 4.020]
        ),
    }


def _cautious_config() -> CartesianAdmittanceConfig:
    return CartesianAdmittanceConfig(
        enabled_axes=(True, True, True),
        control_frame="solver_world",
        response_model="profile_compliance",
        virtual_mass_kg=(2.0, 2.0, 2.0),
        damping_n_s_m=(50.0, 50.0, 50.0),
        stiffness_n_m=(250.0, 250.0, 250.0),
        force_deadband_n=(0.45, 0.45, 0.45),
        force_filter_tau_sec=0.010857362,
        max_offset_m=(0.002, 0.002, 0.002),
        max_velocity_m_s=(0.005, 0.005, 0.005),
        max_return_velocity_m_s=(0.005, 0.005, 0.005),
        max_output_slew_m_s=(0.005, 0.005, 0.005),
        return_force_deadband_n=(0.45, 0.45, 0.45),
        return_rate_s=1.0,
        response_reference_hz=40.0,
        stale_after_sec=0.15,
        max_force_dt_sec=0.15,
    )


def test_exact_runtime_home_target_is_zero_step_in_shared_60hz_ik() -> None:
    initial_qpos = _initial_qpos()
    retargeter, _ = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload={
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
        },
        left_tool_pose=_tool_pose(
            Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"
        ),
        right_tool_pose=_tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE), side="right"
        ),
    )
    exact_home_targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    command = retargeter.build_command_for_context(
        _context(frame={}, targets=exact_home_targets),
        targets=exact_home_targets,
    )

    for side in SIDES:
        qpos, metadata = _command_qpos(command, side=side)
        np.testing.assert_allclose(
            qpos,
            initial_qpos[side],
            atol=1e-10,
            rtol=0.0,
        )
        assert metadata["ik_branch_candidate"] == "current_pose_hold"
        assert metadata["ik_solve_iterations"] == 0


def test_active_observer_and_shared_ik_close_in_one_solver_frame(
) -> None:
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
    retargeter, step_limit = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=_tool_pose(
            Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"
        ),
        right_tool_pose=_tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE), side="right"
        ),
    )
    raw_targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    tool_tcp_targets = retargeter.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    assert np.linalg.norm(
        tool_tcp_targets["left"][:3, 3] - raw_targets["left"][:3, 3]
    ) > 0.1
    monitor = _Monitor()
    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=SIDES,
        config=_cautious_config(),
        side_activity=lambda side: (
            (True, "running_live_side")
            if side == "left"
            else (False, "fixed_side")
        ),
        apply_sides=("left",),
        now_sec=0.0,
    )
    frame = {
        "teleop_command_timestamp": 0.0,
        "command_by_side": {
            side: {
                "metadata": {
                    "target_activity": {
                        "state": "INACTIVE",
                        "inactive": True,
                        "hold_weight": 1.0,
                        "fixed_by_authority": side == "right",
                    }
                }
            }
            for side in SIDES
        },
    }
    rotation_solver_from_tool = (
        raw_targets["left"][:3, :3]
        @ observer._tool_rotation_from_solver["left"]
    )
    tool_force = rotation_solver_from_tool.T @ np.array([-5.0, 0.0, 0.0])

    previous_qpos = {
        side: initial_qpos[side].copy() for side in SIDES
    }
    offset_norms: list[float] = []
    q_step_ratios: list[float] = []
    active_override_count = 0
    non_hold_branch_count = 0

    for tick in range(1, 721):
        now = tick / 60.0
        frame["teleop_command_timestamp"] = now
        if tick % 3 == 0:
            monitor.source_time = now
            monitor.values_by_side["left"][:3] = (
                tool_force.tolist()
                if 30 <= tick < 150
                else [0.0, 0.0, 0.0]
            )

        context = _context(frame=frame, targets=raw_targets)
        applied_targets = observer(
            raw_targets,
            object(),
            object(),
            now_sec=now,
        )
        assert applied_targets is not None
        retargeter._apply_target_override_activity_metadata(context)
        command = retargeter.build_command_for_context(
            context,
            targets=dict(applied_targets),
        )
        qpos: dict[str, np.ndarray] = {}
        metadata = {}
        for side in SIDES:
            qpos[side], metadata[side] = _command_qpos(command, side=side)
        q_step_ratios.append(
            max(
                float(
                    np.max(
                        np.abs(qpos[side] - previous_qpos[side])
                        / step_limit
                    )
                )
                for side in SIDES
            )
        )
        retargeter.set_current_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
            reset_tracking=False,
        )
        admitted = admitted_solver_world_poses_from_joint_command(
            retargeter=retargeter,
            command=command,
        )
        observer.observe_admitted_poses(admitted, now_sec=now)

        snapshot = observer.snapshot(now_sec=now)["sides"]["left"]
        offset = np.asarray(snapshot["applied_total_offset_m"], dtype=float)
        offset_norms.append(float(np.linalg.norm(offset)))
        assert np.linalg.norm(offset) <= 0.002 + 1e-9
        assert snapshot["anchor_offset_m"] == [0.0, 0.0, 0.0]
        assert snapshot["admission_feedback"]["frame"] == "solver_world"
        assert snapshot["admission_feedback"]["state_update_applied"] is False
        active_override_count += int(
            context.left_cartesian.metadata.get("pose_action_motion_intent")
            is not None
        )
        non_hold_branch_count += int(
            metadata["left"].get("ik_branch_candidate")
            != "current_pose_hold"
        )
        previous_qpos = qpos

    final_poses = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=previous_qpos["left"],
        right_qpos=previous_qpos["right"],
    )
    assert max(offset_norms) > 0.001
    assert offset_norms[-1] < 1e-5
    assert max(q_step_ratios) <= 1.0 + 1e-6
    assert active_override_count > 0
    assert non_hold_branch_count > 0
    assert np.linalg.norm(
        final_poses["left"][:3, 3] - raw_targets["left"][:3, 3]
    ) <= 1e-3
    assert np.rad2deg(
        np.max(
            np.abs(previous_qpos["right"] - initial_qpos["right"])
        )
    ) <= 0.01


@pytest.mark.slow_sim
def test_dual_side_formal_active_closes_through_one_shared_ik_budget() -> None:
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
    retargeter, step_limit = _build_retargeter(
        initial_qpos=initial_qpos,
        config_payload=config_payload,
        left_tool_pose=_tool_pose(
            Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"
        ),
        right_tool_pose=_tool_pose(
            Path(DEFAULT_RIGHT_TOOL_PROFILE), side="right"
        ),
    )
    raw_targets = retargeter.solver_world_targets_for_qpos_pair(
        left_qpos=initial_qpos["left"],
        right_qpos=initial_qpos["right"],
    )
    monitor = _Monitor()
    translation_config = replace(
        _cautious_config(),
        max_offset_m=(0.08, 0.08, 0.08),
        max_velocity_m_s=(0.06, 0.06, 0.06),
        max_return_velocity_m_s=(0.42, 0.42, 0.42),
        max_output_slew_m_s=(0.06, 0.06, 0.06),
        return_rate_s=4.0,
    )
    observer = install_position_admittance_active(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=SIDES,
        config=translation_config,
        rotational_config=rotational_admittance_config_for_profile(
            ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
        ),
        side_activity=lambda _side: (True, "running_live_side"),
        apply_sides=SIDES,
        active_envelope=POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY,
        combined_translation_rotation_active=True,
        endpoint_policy=POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
        now_sec=0.0,
    )
    frame = {
        "teleop_command_timestamp": 0.0,
        "command_by_side": {
            side: {
                "metadata": {
                    "target_activity": {
                        "state": "ACTIVE",
                        "inactive": False,
                        "hold_weight": 0.0,
                        "fixed_by_authority": False,
                    }
                }
            }
            for side in SIDES
        },
    }

    # Establish the exact admitted-qref baseline before either modifier becomes
    # nonzero. Both controllers then consume distinct 20 Hz tool wrenches while
    # the coupled retargeter keeps one 60 Hz physical-step budget per side.
    initial_targets = observer(raw_targets, object(), object(), now_sec=0.0)
    assert initial_targets is not None
    initial_context = _context(frame=frame, targets=raw_targets)
    retargeter._apply_target_override_activity_metadata(initial_context)
    initial_command = retargeter.build_command_for_context(
        initial_context,
        targets=dict(initial_targets),
    )
    observer.observe_admitted_poses(
        admitted_solver_world_poses_from_joint_command(
            retargeter=retargeter,
            command=initial_command,
        ),
        now_sec=0.0,
    )

    previous_qpos = {
        side: initial_qpos[side].copy() for side in SIDES
    }
    max_offset = {side: 0.0 for side in SIDES}
    max_rotation_deg = {side: 0.0 for side in SIDES}
    non_hold_branch_count = {side: 0 for side in SIDES}
    q_step_ratios: list[float] = []
    solve_iteration_max = {side: 0 for side in SIDES}

    for tick in range(1, 901):
        now = tick / 60.0
        frame["teleop_command_timestamp"] = now
        if tick % 3 == 0:
            monitor.source_time = now
            if 30 <= tick < 150:
                monitor.values_by_side["left"] = [
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                monitor.values_by_side["right"] = [
                    -5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                ]
            else:
                monitor.values_by_side["left"] = [0.0] * 6
                monitor.values_by_side["right"] = [0.0] * 6

        context = _context(frame=frame, targets=raw_targets)
        applied_targets = observer(
            raw_targets,
            object(),
            object(),
            now_sec=now,
        )
        assert applied_targets is not None
        retargeter._apply_target_override_activity_metadata(context)
        command = retargeter.build_command_for_context(
            context,
            targets=dict(applied_targets),
        )
        qpos: dict[str, np.ndarray] = {}
        metadata: dict[str, dict[str, object]] = {}
        for side in SIDES:
            qpos[side], metadata[side] = _command_qpos(command, side=side)
            iterations = int(metadata[side]["ik_solve_iterations"])
            solve_iteration_max[side] = max(
                solve_iteration_max[side], iterations
            )
            assert iterations <= 1
            assert metadata[side]["ik_solve_iteration_mode"] == (
                "single_physical_step"
            )
            non_hold_branch_count[side] += int(
                metadata[side].get("ik_branch_candidate")
                != "current_pose_hold"
            )
        q_step_ratios.append(
            max(
                float(
                    np.max(
                        np.abs(qpos[side] - previous_qpos[side])
                        / step_limit
                    )
                )
                for side in SIDES
            )
        )
        retargeter.set_current_qpos_pair(
            left_qpos=qpos["left"],
            right_qpos=qpos["right"],
            reset_tracking=False,
        )
        observer.observe_admitted_poses(
            admitted_solver_world_poses_from_joint_command(
                retargeter=retargeter,
                command=command,
            ),
            now_sec=now,
        )
        snapshot = observer.snapshot(now_sec=now)
        for side in SIDES:
            side_snapshot = snapshot["sides"][side]
            offset = np.asarray(
                side_snapshot["applied_total_offset_m"], dtype=float
            )
            rotation = np.asarray(
                side_snapshot["applied_rotation_rotvec_rad"], dtype=float
            )
            max_offset[side] = max(
                max_offset[side], float(np.linalg.norm(offset))
            )
            max_rotation_deg[side] = max(
                max_rotation_deg[side],
                float(np.degrees(np.linalg.norm(rotation))),
            )
            assert np.linalg.norm(offset) <= 0.08 + 1e-9
            assert np.degrees(np.linalg.norm(rotation)) <= 5.0 + 1e-9
            assert side_snapshot["admission_feedback"]["frame"] == (
                "solver_world"
            )
            assert side_snapshot["admission_feedback"][
                "state_update_applied"
            ] is False
        previous_qpos = qpos

    final_snapshot = observer.snapshot(now_sec=15.0)
    assert final_snapshot["apply_sides"] == ["left", "right"]
    assert final_snapshot["rotation_authority"].endswith("dual_side")
    assert max(q_step_ratios) <= 1.0 + 1e-6
    for side in SIDES:
        assert max_offset[side] > 0.001
        assert max_rotation_deg[side] > 0.05
        assert non_hold_branch_count[side] > 0
        assert solve_iteration_max[side] == 1
        side_snapshot = final_snapshot["sides"][side]
        assert np.linalg.norm(
            np.asarray(side_snapshot["applied_total_offset_m"], dtype=float)
        ) <= 1e-5
        assert np.degrees(
            np.linalg.norm(
                np.asarray(
                    side_snapshot["applied_rotation_rotvec_rad"],
                    dtype=float,
                )
            )
        ) <= 0.05


@pytest.mark.slow_sim
def test_rotation_active_preserves_shared_ik_hold_and_single_step_budget(
) -> None:
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
    left_tool_pose = _tool_pose(
        Path(DEFAULT_LEFT_TOOL_PROFILE), side="left"
    )
    right_tool_pose = _tool_pose(
        Path(DEFAULT_RIGHT_TOOL_PROFILE), side="right"
    )

    def fresh_retargeter():
        return _build_retargeter(
            initial_qpos=initial_qpos,
            config_payload=config_payload,
            left_tool_pose=left_tool_pose,
            right_tool_pose=right_tool_pose,
        )[0]

    retargeter = fresh_retargeter()
    raw_targets = retargeter.solver_world_targets_for_qpos_pair(
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
            for side in SIDES
        },
    }

    small_rotation_deg = 0.1
    small_targets = {
        side: target.copy() for side, target in raw_targets.items()
    }
    for side in SIDES:
        small_targets[side][:3, :3] = (
            rotation_matrix_from_rotvec(
                [0.0, 0.0, math.radians(small_rotation_deg)]
            )
            @ raw_targets[side][:3, :3]
        )

    normal = retargeter.build_command_for_context(
        _context(frame=frame, targets=small_targets),
        targets=small_targets,
    )
    _, normal_metadata = _command_qpos(normal, side="left")
    assert normal_metadata["ik_branch_candidate"] == "current_pose_hold"
    assert normal_metadata["ik_solve_iterations"] == 0
    assert normal_metadata["ik_solve_iteration_mode"] == (
        "single_physical_step"
    )
    assert normal_metadata["ik_current_target_hold"] == {
        "enabled": True,
        "policy": "shared_ik_unchanged",
        "translation_threshold_m": 0.001,
        "rotation_threshold_deg": 0.5,
        "target_already_reached": True,
    }

    translation_active_context = _context(
        frame=frame,
        targets=small_targets,
        modifier_active_side="left",
    )
    translation_active = retargeter.build_command_for_context(
        translation_active_context,
        targets=small_targets,
    )
    _, translation_active_metadata = _command_qpos(
        translation_active,
        side="left",
    )
    assert translation_active_metadata["ik_branch_candidate"] == (
        "current_pose_hold"
    )
    assert translation_active_metadata["ik_solve_iterations"] == 0
    assert translation_active_metadata["ik_current_target_hold"] == (
        normal_metadata["ik_current_target_hold"]
    )

    rotation_active_context = _context(
        frame=frame,
        targets=small_targets,
        modifier_active_side="left",
    )
    rotation_active_context.left_cartesian.metadata[
        "pose_action_motion_intent"
    ]["modifier_rotation_offset_norm_rad"] = math.radians(
        small_rotation_deg
    )
    rotation_active = retargeter.build_command_for_context(
        rotation_active_context,
        targets=small_targets,
    )
    rotation_active_qpos, rotation_active_metadata = _command_qpos(
        rotation_active,
        side="left",
    )
    assert rotation_active_metadata["ik_branch_candidate"] == (
        "current_pose_hold"
    )
    assert rotation_active_metadata["ik_solve_iterations"] == 0
    assert rotation_active_metadata["ik_solve_iteration_mode"] == (
        "single_physical_step"
    )
    assert rotation_active_metadata["ik_current_target_hold"] == (
        normal_metadata["ik_current_target_hold"]
    )
    np.testing.assert_allclose(
        rotation_active_qpos,
        initial_qpos["left"],
        atol=1e-10,
        rtol=0.0,
    )

    # Above the unchanged 0.5-degree Hold threshold, ordinary and force-tagged
    # versions of the same composed target must use the same one-step raw IK.
    # Force metadata may only alter the downstream qref smoothing stage.
    large_rotation_deg = 0.6
    large_targets = {
        side: target.copy() for side, target in raw_targets.items()
    }
    large_targets["left"][:3, :3] = (
        rotation_matrix_from_rotvec(
            [0.0, 0.0, math.radians(large_rotation_deg)]
        )
        @ raw_targets["left"][:3, :3]
    )
    normal_large_retargeter = fresh_retargeter()
    normal_large = normal_large_retargeter.build_command_for_context(
        _context(frame=frame, targets=large_targets),
        targets=large_targets,
    )
    _, normal_large_metadata = _command_qpos(normal_large, side="left")

    active_large_retargeter = fresh_retargeter()
    active_large_context = _context(
        frame=frame,
        targets=large_targets,
        modifier_active_side="left",
    )
    active_large_context.left_cartesian.metadata[
        "pose_action_motion_intent"
    ]["modifier_rotation_offset_norm_rad"] = math.radians(
        large_rotation_deg
    )
    active_large = active_large_retargeter.build_command_for_context(
        active_large_context,
        targets=large_targets,
    )
    _, active_large_metadata = _command_qpos(active_large, side="left")

    for metadata in (normal_large_metadata, active_large_metadata):
        assert metadata["ik_branch_candidate"] != "current_pose_hold"
        assert metadata["ik_solve_iterations"] == 1
        assert metadata["ik_solve_iteration_mode"] == "single_physical_step"
        assert metadata["ik_current_target_hold"] == {
            "enabled": True,
            "policy": "shared_ik_unchanged",
            "translation_threshold_m": 0.001,
            "rotation_threshold_deg": 0.5,
            "target_already_reached": False,
        }
    assert active_large_metadata["ik_branch_candidate"] == (
        normal_large_metadata["ik_branch_candidate"]
    )
    for key in (
        "ik_time_budget_ms",
        "ik_solve_iteration_budget_ms",
        "ik_solve_iteration_min",
        "ik_solve_iteration_max",
        "ik_solve_iteration_reserve_ms",
        "ik_candidate_certification_reserve_ms",
        "ik_damping",
    ):
        assert active_large_metadata[key] == normal_large_metadata[key]
    np.testing.assert_allclose(
        active_large_metadata["q_ik_raw"],
        normal_large_metadata["q_ik_raw"],
        atol=1e-10,
        rtol=0.0,
    )

    # A target carrying both bounded translation and rotation modifiers still
    # reaches the exact same shared IK call.  The modifier metadata may select
    # downstream qref smoothing only; it must not add an IK solve or alter the
    # raw candidate/budget contract.
    combined_targets = {
        side: target.copy() for side, target in raw_targets.items()
    }
    combined_targets["left"][0, 3] += 0.0015
    combined_targets["left"][:3, :3] = (
        rotation_matrix_from_rotvec(
            [0.0, 0.0, math.radians(large_rotation_deg)]
        )
        @ raw_targets["left"][:3, :3]
    )
    normal_combined_retargeter = fresh_retargeter()
    normal_combined = normal_combined_retargeter.build_command_for_context(
        _context(frame=frame, targets=combined_targets),
        targets=combined_targets,
    )
    _, normal_combined_metadata = _command_qpos(
        normal_combined,
        side="left",
    )

    active_combined_retargeter = fresh_retargeter()
    active_combined_context = _context(
        frame=frame,
        targets=combined_targets,
        modifier_active_side="left",
    )
    combined_intent = active_combined_context.left_cartesian.metadata[
        "pose_action_motion_intent"
    ]
    combined_intent["modifier_offset_norm_m"] = 0.0015
    combined_intent["modifier_rotation_offset_norm_rad"] = math.radians(
        large_rotation_deg
    )
    active_combined = active_combined_retargeter.build_command_for_context(
        active_combined_context,
        targets=combined_targets,
    )
    _, active_combined_metadata = _command_qpos(
        active_combined,
        side="left",
    )

    for metadata in (normal_combined_metadata, active_combined_metadata):
        assert metadata["ik_solve_iterations"] == 1
        assert metadata["ik_solve_iteration_mode"] == "single_physical_step"
    for key in (
        "ik_time_budget_ms",
        "ik_solve_iteration_budget_ms",
        "ik_solve_iteration_min",
        "ik_solve_iteration_max",
        "ik_solve_iteration_reserve_ms",
        "ik_candidate_certification_reserve_ms",
        "ik_damping",
    ):
        assert active_combined_metadata[key] == normal_combined_metadata[key]
    np.testing.assert_allclose(
        active_combined_metadata["q_ik_raw"],
        normal_combined_metadata["q_ik_raw"],
        atol=1e-10,
        rtol=0.0,
    )
