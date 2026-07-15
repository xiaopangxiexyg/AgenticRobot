from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import robot_motion.replay.action_frame as robot_action_frame
import robot_motion.replay.control_schedule as robot_control_schedule
import robot_motion.replay.trajectory as robot_trajectory
import teleop_runtime.replay.action_frame as legacy_action_frame
import teleop_runtime.replay.trajectory as legacy_trajectory
from robot_motion.action import BIMANUAL_EE_POSE_V1, HandActionSpec, RobotActionResolver
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ACTION_HAND_SLICES
from robot_motion.position_control import (
    JointCommandInterpolator,
    PositionReferenceFrame,
    PositionSideReference,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.relative_action import ACTION_POSE_MODE_ABSOLUTE, ACTION_POSE_MODE_RELATIVE


def _repo_source(relative_path: str) -> str:
    return (Path(__file__).resolve().parents[1] / relative_path).read_text(encoding="utf-8")


def _group(*, hand_spec: bool = False) -> robot_trajectory.ReplayTrajectoryGroup:
    actions = np.full((2, ACTION_DIM), np.nan, dtype=np.float32)
    actions[:, ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
        [
            [0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            [0.21, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    actions[:, ACTION_HAND_SLICES["right"]] = np.asarray(
        [
            [0.4, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.5, np.nan, np.nan, np.nan, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    dataset_info = {}
    if hand_spec:
        dataset_info["hand_action_specs_by_side"] = {
            "right": HandActionSpec.dmtac_gripper("right").to_metadata()
        }
    return robot_trajectory.ReplayTrajectoryGroup(
        key="g",
        label="test",
        source_path="episode_000000.parquet",
        source_episode_index=0,
        start_index=0,
        end_index=2,
        frame_count=2,
        action_hz=10.0,
        actions=actions,
        arm_joint_targets={"right": np.zeros((2, 6), dtype=np.float32)},
        arm_observation_current_qpos={"right": np.ones((2, 6), dtype=np.float32)},
        metadata={"source_metadata": {"dataset_info": dataset_info}},
    )


def test_replay_action_chunk_relative_mode_uses_first_actual_tcp_anchor() -> None:
    group = _group()
    group.arm_observation_actual_tcp_pose["right"] = np.asarray(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    relative = robot_trajectory.replay_action_chunk_at_index(
        group,
        ("right",),
        start_index=0,
        chunk_size=2,
        source_mode=ACTION_POSE_MODE_ABSOLUTE,
        target_mode=ACTION_POSE_MODE_RELATIVE,
    )
    assert relative[0, ACTION_ARM_POSE_SLICES["right"]][:3] == pytest.approx(
        [0.1, 0.0, 0.3]
    )
    assert relative[1, ACTION_ARM_POSE_SLICES["right"]][:3] == pytest.approx(
        [0.11, 0.0, 0.3]
    )
    relative_group = replace(group, actions=relative)
    reconstructed = robot_trajectory.replay_action_chunk_at_index(
        relative_group,
        ("right",),
        start_index=0,
        chunk_size=2,
        source_mode=ACTION_POSE_MODE_RELATIVE,
        target_mode=ACTION_POSE_MODE_ABSOLUTE,
    )
    finite = np.isfinite(group.actions)
    assert np.allclose(reconstructed[finite], group.actions[finite], atol=1e-6)
    assert np.array_equal(np.isfinite(reconstructed), finite)


def test_replay_action_chunk_round_trip_reports_zero_pose_error() -> None:
    group = _group()
    group.arm_observation_actual_tcp_pose["right"] = np.asarray(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    report = robot_trajectory.replay_action_chunk_round_trip_metrics(
        group,
        ("right",),
        start_index=0,
        chunk_size=2,
    )
    metrics = report["metrics_by_side"]["right"]
    assert metrics.max_translation_error_m < 1e-6
    assert metrics.max_rotation_error_rad < 1e-6


def test_replay_action_chunk_round_trip_slices_anchor_history_for_nonzero_start() -> None:
    group = _group()
    group.arm_observation_actual_tcp_pose["right"] = np.asarray(
        [
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    report = robot_trajectory.replay_action_chunk_round_trip_metrics(
        group,
        ("right",),
        start_index=1,
        chunk_size=1,
    )
    metrics = report["metrics_by_side"]["right"]
    assert metrics.max_translation_error_m < 1e-6
    assert metrics.max_rotation_error_rad < 1e-6


def test_replay_group_prefers_strict_source_frame_times_with_nominal_fallback() -> None:
    fallback = _group()
    assert fallback.relative_time_sec(1) == pytest.approx(0.1)

    timed = robot_trajectory.ReplayTrajectoryGroup(
        key="timed",
        label="timed",
        source_path="episode.parquet",
        source_episode_index=0,
        start_index=4,
        end_index=6,
        frame_count=2,
        action_hz=10.0,
        actions=fallback.actions.copy(),
        frame_times_sec=(100.0, 100.073),
    )
    assert timed.frame_times_sec[0] == 0.0
    assert timed.relative_time_sec(1) == pytest.approx(0.073)

    with pytest.raises(ValueError, match="strictly increasing"):
        robot_trajectory.ReplayTrajectoryGroup(
            key="bad",
            label="bad",
            source_path="episode.parquet",
            source_episode_index=0,
            start_index=0,
            end_index=2,
            frame_count=2,
            action_hz=10.0,
            actions=fallback.actions.copy(),
            frame_times_sec=(1.0, 1.0),
        )


def test_legacy_replay_action_frame_reexports_robot_motion_objects() -> None:
    assert legacy_action_frame.replay_action_metadata is robot_action_frame.replay_action_metadata
    assert legacy_action_frame.replay_robot_action_at_index is (
        robot_action_frame.replay_robot_action_at_index
    )
    assert legacy_action_frame.replay_teleop_command_at_index is (
        robot_action_frame.replay_teleop_command_at_index
    )


def test_legacy_replay_trajectory_defaults_to_robot_motion_contracts() -> None:
    assert legacy_trajectory.ReplayTrajectoryGroup is robot_trajectory.ReplayTrajectoryGroup
    assert legacy_trajectory.ReplayTrajectoryPool is robot_trajectory.ReplayTrajectoryPool
    assert legacy_trajectory.parse_replay_group_spec is robot_trajectory.parse_replay_group_spec
    assert legacy_trajectory.replay_action_poses_at_index is (
        robot_trajectory.replay_action_poses_at_index
    )
    assert legacy_trajectory.select_replay_qpos_init_targets is (
        robot_trajectory.select_replay_qpos_init_targets
    )


def test_legacy_replay_trajectory_file_keeps_only_dataset_loading_and_wrappers() -> None:
    source = inspect.getsource(legacy_trajectory)

    assert "from robot_motion.replay.trajectory import" in source
    assert "def build_replay_group_for_episode" in source
    assert "class ReplayInitEpisodePlan" in source
    assert "class ReplayTrajectoryGroup" not in source
    assert "class ReplayTrajectoryPool" not in source
    assert "def parse_replay_group_spec" not in source
    assert "def replay_action_poses_at_index" not in source


def test_teleop_replay_execution_defaults_use_robot_motion_common_contracts() -> None:
    executor_source = _repo_source("teleop_runtime/replay/executor.py")
    assert "from robot_motion.replay.action_frame import replay_teleop_command_at_index" in executor_source
    assert "from robot_motion.replay.trajectory import" in executor_source
    assert "from .trajectory import build_replay_group_for_episode" in executor_source
    assert "from .action_frame import" not in executor_source
    assert "from ..deploy.schema import" not in executor_source

    sim_runner_source = _repo_source("sim/mujoco/replay/runner.py")
    assert "from robot_motion.replay.control_schedule import" in sim_runner_source

    checks_source = _repo_source("teleop_runtime/replay/checks.py")
    assert "from robot_motion.deploy.schema import" in checks_source
    assert "from robot_motion.replay.action_frame import" in checks_source
    assert "from robot_motion.replay.trajectory import" in checks_source
    assert "from .action_frame import" not in checks_source
    assert "from .trajectory import" not in checks_source

    hardware_source = _repo_source("teleop_runtime/replay/hardware.py")
    assert "from robot_motion.action import RobotAction, RobotActionResolver" in hardware_source
    assert "from robot_motion.deploy.schema import ARM_JOINT_DOF" in hardware_source
    assert "from robot_motion.runtime import EffectorCommand, TeleopCommand" in hardware_source
    assert "from ..action." not in hardware_source
    assert "from ..deploy.schema import" not in hardware_source
    assert "from ..schemas import" not in hardware_source


def test_replay_trajectory_and_action_frame_reference_sample() -> None:
    group = _group(hand_spec=True)

    poses = robot_trajectory.replay_action_poses_at_index(group, ("right",), index=1)
    np.testing.assert_allclose(poses["right"], [0.21, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])

    qpos_targets, qpos_sources = robot_trajectory.select_replay_qpos_init_targets(
        group,
        ("right",),
        index=0,
    )
    np.testing.assert_allclose(qpos_targets["right"], np.zeros(6))
    assert qpos_sources == {"right": robot_trajectory.QPOS_INIT_REPLAY_QPOS}

    hand_targets = robot_action_frame.replay_end_effector_targets_at_index(
        group,
        ("right",),
        index=0,
    )
    np.testing.assert_allclose(hand_targets["right"], [0.4, 0.0, 0.0, 0.0, 0.0, 0.0])

    action = robot_action_frame.replay_robot_action_at_index(
        group,
        ("right",),
        index=0,
        include_arms=True,
        include_end_effectors=True,
    )
    assert action.active_effectors == ("right_arm", "right_hand")
    assert action.metadata["replay_frame_index"] == 0

    command = robot_action_frame.replay_teleop_command_at_index(
        group,
        ("right",),
        index=0,
        include_arms=True,
        include_end_effectors=True,
    )
    assert set(command.commands) == {"right_arm", "right_hand"}
    assert command.metadata["source"] == "replay_action"


def test_position_qpos_control_schedule_expands_15hz_actions_to_150hz_ticks() -> None:
    actions = [
        robot_control_schedule.ReplayControlAction(
            action_index=0,
            frame_index=100,
            relative_time_sec=0.0,
            qpos_by_side={"right": np.zeros(6, dtype=float)},
        ),
        robot_control_schedule.ReplayControlAction(
            action_index=1,
            frame_index=101,
            relative_time_sec=1.0 / 15.0,
            qpos_by_side={"right": np.ones(6, dtype=float)},
        ),
    ]

    references = robot_control_schedule.resample_position_qpos_references(
        actions,
        reference_hz=45.0,
    )
    ticks = robot_control_schedule.iter_position_qpos_control_ticks(
        actions,
        reference_hz=45.0,
        control_hz=150.0,
        include_final_settle=True,
    )

    assert len(references) == 4
    assert [reference.relative_time_sec for reference in references] == pytest.approx(
        [0.0, 1.0 / 45.0, 2.0 / 45.0, 1.0 / 15.0]
    )
    assert [reference.action_interpolation_alpha for reference in references] == pytest.approx(
        [0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0]
    )
    np.testing.assert_allclose(references[1].qpos_by_side["right"], np.full(6, 1.0 / 3.0))
    np.testing.assert_allclose(references[2].qpos_by_side["right"], np.full(6, 2.0 / 3.0))

    assert len(ticks) == 15
    assert ticks[0].control_index == 0
    assert ticks[0].action_index == 0
    assert ticks[4].reference_index == 1
    assert ticks[7].reference_index == 2
    assert ticks[10].action_index == 1
    assert ticks[10].reference_index == 3
    assert ticks[10].interpolation_alpha == pytest.approx(0.0)
    assert ticks[-1].action_index == 1
    assert ticks[-1].interpolation_alpha == pytest.approx(1.0)
    settle_deadline = (1.0 / 15.0) + (1.0 / 45.0)
    assert ticks[-1].relative_time_sec == pytest.approx(
        np.ceil(settle_deadline * 150.0) / 150.0
    )
    assert ticks[-1].relative_time_sec >= settle_deadline
    assert np.diff([tick.relative_time_sec for tick in ticks]) == pytest.approx(
        np.full(len(ticks) - 1, 1.0 / 150.0)
    )
    np.testing.assert_allclose(ticks[0].qpos_by_side["right"], np.zeros(6))
    np.testing.assert_allclose(
        ticks[10].qpos_by_side["right"],
        ticks[9].qpos_by_side["right"],
    )
    np.testing.assert_allclose(ticks[-1].qpos_by_side["right"], np.ones(6))
    assert ticks[10].metadata["command_segment_anchor"] == "last_emitted_command"
    assert ticks[10].metadata["command_planner"] == (
        "robot_motion.position_control.JointCommandInterpolator"
    )


def test_position_qpos_control_schedule_final_settle_covers_unaligned_action_time() -> None:
    actions = [
        robot_control_schedule.ReplayControlAction(
            action_index=0,
            frame_index=0,
            relative_time_sec=0.0,
            qpos_by_side={"right": np.zeros(6, dtype=float)},
        ),
        robot_control_schedule.ReplayControlAction(
            action_index=1,
            frame_index=1,
            relative_time_sec=(1.0 / 15.0) + 0.001,
            qpos_by_side={"right": np.ones(6, dtype=float)},
        ),
    ]

    ticks = robot_control_schedule.iter_position_qpos_control_ticks(
        actions,
        control_hz=150.0,
        include_final_settle=True,
    )

    assert ticks[-1].action_index == 1
    settle_deadline = (
        actions[-1].relative_time_sec
        + (1.0 / robot_control_schedule.DEFAULT_REPLAY_REFERENCE_HZ)
    )
    assert ticks[-1].relative_time_sec == pytest.approx(
        np.ceil(settle_deadline * 150.0) / 150.0
    )
    assert ticks[-1].interpolation_alpha == pytest.approx(1.0)
    np.testing.assert_allclose(ticks[-1].qpos_by_side["right"], np.ones(6))


def test_position_qpos_control_schedule_without_final_settle_still_reaches_last_action_tick() -> None:
    actions = [
        robot_control_schedule.ReplayControlAction(
            action_index=0,
            frame_index=0,
            relative_time_sec=0.0,
            qpos_by_side={"right": np.zeros(6, dtype=float)},
        ),
        robot_control_schedule.ReplayControlAction(
            action_index=1,
            frame_index=1,
            relative_time_sec=(1.0 / 15.0) + 0.001,
            qpos_by_side={"right": np.ones(6, dtype=float)},
        ),
    ]

    ticks = robot_control_schedule.iter_position_qpos_control_ticks(
        actions,
        control_hz=150.0,
        include_final_settle=False,
    )

    assert ticks[-1].action_index == 1
    assert ticks[-1].relative_time_sec == pytest.approx(
        np.ceil(actions[-1].relative_time_sec * 150.0) / 150.0
    )
    assert 0.0 <= ticks[-1].interpolation_alpha < 1.0


def test_position_qpos_segment_duration_defaults_to_reference_period() -> None:
    assert robot_control_schedule.position_qpos_segment_duration(
        control_hz=150.0,
        reference_hz=45.0,
    ) == pytest.approx(1.0 / 45.0)
    assert robot_control_schedule.position_qpos_segment_duration(
        control_hz=137.0,
        reference_hz=43.0,
        segment_duration_sec=0.01,
    ) == pytest.approx(0.01)


def test_position_qpos_schedule_supports_noninteger_custom_stage_rates() -> None:
    actions = [
        robot_control_schedule.ReplayControlAction(
            action_index=0,
            frame_index=20,
            relative_time_sec=0.0,
            qpos_by_side={"right": np.zeros(6, dtype=float)},
        ),
        robot_control_schedule.ReplayControlAction(
            action_index=1,
            frame_index=21,
            relative_time_sec=0.073,
            qpos_by_side={"right": np.ones(6, dtype=float)},
        ),
    ]

    references = robot_control_schedule.resample_position_qpos_references(
        actions,
        reference_hz=43.0,
    )
    ticks = robot_control_schedule.iter_position_qpos_control_ticks(
        actions,
        reference_hz=43.0,
        control_hz=137.0,
    )

    assert [reference.relative_time_sec for reference in references] == pytest.approx(
        [0.0, 1.0 / 43.0, 2.0 / 43.0, 3.0 / 43.0, 0.073]
    )
    assert np.diff([tick.relative_time_sec for tick in ticks]) == pytest.approx(
        np.full(len(ticks) - 1, 1.0 / 137.0)
    )
    settle_deadline = 0.073 + (1.0 / 43.0)
    assert ticks[-1].relative_time_sec == pytest.approx(
        np.ceil(settle_deadline * 137.0) / 137.0
    )
    assert ticks[-1].reference_index == len(references) - 1
    assert ticks[-1].interpolation_alpha == pytest.approx(1.0)
    np.testing.assert_allclose(ticks[-1].qpos_by_side["right"], np.ones(6))


def test_replay_command_stage_matches_common_joint_command_interpolator() -> None:
    actions = [
        robot_control_schedule.ReplayControlAction(
            action_index=0,
            frame_index=30,
            relative_time_sec=0.0,
            qpos_by_side={"right": np.zeros(6, dtype=float)},
        ),
        robot_control_schedule.ReplayControlAction(
            action_index=1,
            frame_index=31,
            relative_time_sec=0.073,
            qpos_by_side={"right": np.ones(6, dtype=float)},
        ),
    ]
    references = robot_control_schedule.resample_position_qpos_references(
        actions,
        reference_hz=43.0,
    )
    replay_ticks = robot_control_schedule.iter_position_qpos_control_ticks(
        actions,
        reference_hz=43.0,
        control_hz=137.0,
    )
    interpolator = JointCommandInterpolator(
        segment_duration_sec=1.0 / 43.0,
        stale_after_sec=None,
    )
    interpolator.reset(
        references[0].qpos_by_side,
        epoch=5,
        monotonic_time=0.0,
    )

    next_reference = 0
    common_ticks = []
    for replay_tick in replay_ticks:
        while (
            next_reference < len(references)
            and references[next_reference].relative_time_sec
            <= replay_tick.relative_time_sec + 1e-12
        ):
            reference = references[next_reference]
            q_ref = reference.qpos_by_side["right"]
            interpolator.accept_target(
                PositionReferenceFrame(
                    epoch=5,
                    sequence=reference.reference_index,
                    source_wall_time=reference.relative_time_sec,
                    generated_wall_time=reference.relative_time_sec,
                    published_wall_time=reference.relative_time_sec,
                    generated_monotonic_time=reference.relative_time_sec,
                    published_monotonic_time=reference.relative_time_sec,
                    references={
                        "right": PositionSideReference(
                            side="right",
                            q_ik_raw=q_ref,
                            q_ref=q_ref,
                            pose_ref=np.eye(4, dtype=float),
                        )
                    },
                    source="replay_test",
                ),
                monotonic_time=reference.relative_time_sec,
            )
            next_reference += 1
        common_ticks.append(
            interpolator.step(monotonic_time=replay_tick.relative_time_sec)
        )

    assert len(common_ticks) == len(replay_ticks)
    for replay_tick, common_tick in zip(replay_ticks, common_ticks):
        assert replay_tick.reference_index == common_tick.reference_sequence
        assert replay_tick.interpolation_alpha == pytest.approx(
            common_tick.interpolation_alpha
        )
        np.testing.assert_allclose(
            replay_tick.qpos_by_side["right"],
            common_tick.q_cmd_by_side["right"],
        )


def _assert_effector_command_equal(actual: EffectorCommand, expected: EffectorCommand) -> None:
    assert actual.effector_name == expected.effector_name
    assert actual.side == expected.side
    assert actual.command_type == expected.command_type
    assert actual.frame_name == expected.frame_name
    np.testing.assert_allclose(actual.value, expected.value)
    assert set(actual.metadata) == set(expected.metadata)
    for key, expected_value in expected.metadata.items():
        actual_value = actual.metadata[key]
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_allclose(actual_value, expected_value)
        else:
            assert actual_value == expected_value


def _assert_command_equal(actual: TeleopCommand, expected: TeleopCommand) -> None:
    assert actual.timestamp == expected.timestamp
    assert actual.metadata == expected.metadata
    assert set(actual.commands) == set(expected.commands)
    for effector_name, expected_effector in expected.commands.items():
        _assert_effector_command_equal(actual.commands[effector_name], expected_effector)


def test_replay_action_robot_action_and_command_are_consistent(monkeypatch) -> None:
    group = _group(hand_spec=True)
    source = "unit_replay"
    monkeypatch.setattr(robot_action_frame.time, "time", lambda: 123.0)

    action = robot_action_frame.replay_robot_action_at_index(
        group,
        ("right",),
        index=1,
        include_arms=True,
        include_end_effectors=True,
        source=source,
    )
    assert action.action_space == BIMANUAL_EE_POSE_V1
    assert action.active_effectors == ("right_arm", "right_hand")
    np.testing.assert_allclose(
        action.arm_pose_by_side["right"],
        group.actions[1, ACTION_ARM_POSE_SLICES["right"]],
    )
    np.testing.assert_allclose(
        action.hand_qpos_by_side["right"],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    resolver = RobotActionResolver()
    expected_command = robot_action_frame.annotate_replay_command_metadata(
        resolver.resolve(action),
        source=source,
    )
    helper_command = robot_action_frame.replay_teleop_command_at_index(
        group,
        ("right",),
        index=1,
        include_arms=True,
        include_end_effectors=True,
        source=source,
        resolver=resolver,
    )
    _assert_command_equal(helper_command, expected_command)

    arm_command = helper_command.commands["right_arm"]
    hand_command = helper_command.commands["right_hand"]
    assert arm_command.command_type == "cartesian_pose"
    assert hand_command.command_type == "joint_vector"
    np.testing.assert_allclose(arm_command.metadata["target_xyzquat"], action.arm_pose_by_side["right"])
    np.testing.assert_allclose(arm_command.metadata["source_target_pose_base"], arm_command.value)
    np.testing.assert_allclose(arm_command.metadata["target_pose_tcp_base"], arm_command.value)
    np.testing.assert_allclose(hand_command.value, [0.5])
    np.testing.assert_allclose(hand_command.metadata["hand_qpos_full"], action.hand_qpos_by_side["right"])
    assert helper_command.metadata["replay_source"] == source
    assert helper_command.metadata["replay_frame_index"] == 1
    assert arm_command.metadata["replay_source"] == source
    assert hand_command.metadata["replay_source"] == source


def test_robot_motion_replay_modules_do_not_import_teleop_runtime() -> None:
    for module in (robot_trajectory, robot_action_frame, robot_control_schedule):
        assert "teleop_runtime" not in inspect.getsource(module)
