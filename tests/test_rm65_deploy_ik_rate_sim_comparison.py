from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from todos.probes.compare_rm65_official_ik_rates import PoseTrack, RolloutArtifact
from todos.probes.view_rm65_deploy_ik_rate_comparison_mujoco import (
    _auto_select_chunk,
    _advance_gripper_source_state,
    _gripper_targets_by_source_index,
    _recorded_chunk_arrays,
    _trajectory_from_recorded_full,
    _trajectory_from_rollout,
    _with_viewer_startup_hold,
    build_parser,
    cli_main,
    recorded_chunks_from_events,
)


def _dispatch(timestamp: float, left: float, right: float) -> dict:
    return {
        "event": "position_executor_arm_reference_dispatch",
        "wall_time": 1000.0 + timestamp,
        "position_reference_emitted_monotonic_time": timestamp,
        "position_arm_commands": {
            "left_arm": {"value": [left] * 6},
            "right_arm": {"value": [right] * 6},
        },
    }


def _discrete(
    timestamp: float,
    source_index: int,
    left: float,
    right: float,
) -> dict:
    return {
        "event": "position_executor_discrete_dispatch",
        "position_discrete_logical_tick_monotonic_time": timestamp,
        "position_discrete_source_index": source_index,
        "position_discrete_commands": {
            "left_hand": {
                "command_type": "joint_vector",
                "value": [left],
            },
            "right_hand": {
                "command_type": "joint_vector",
                "value": [right],
            },
        },
    }


def _track(chunk: int) -> PoseTrack:
    pose = np.repeat(np.eye(4)[None, :, :], 3, axis=0)
    return PoseTrack(
        name=f"debug:chunk={chunk}",
        pose_15hz_by_side={"left": pose.copy(), "right": pose.copy()},
        initial_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        endpoint_to_tool_by_side={"left": np.eye(4), "right": np.eye(4)},
        source="/tmp/debug.jsonl",
    )


def test_recorded_chunks_preserve_policy_chunk_boundaries_and_timing() -> None:
    events = (
        {"event": "position_executor_stream_cleared"},
        {"event": "policy_chunk", "wall_time": 1000.0},
        _dispatch(10.000, 0.0, 0.0),
        _discrete(10.005, 1, 0.1, 0.2),
        _dispatch(10.010, 0.01, -0.01),
        {"event": "position_chunk_gap_hold", "wall_time": 1010.050},
        {"event": "policy_chunk", "wall_time": 1011.0},
        _dispatch(11.000, 0.0, 0.0),
        _discrete(11.005, 1, 0.3, 0.4),
        _dispatch(11.030, 0.10, -0.10),
    )

    chunks = recorded_chunks_from_events(events)

    assert [chunk.chunk_id for chunk in chunks] == [0, 1]
    assert [len(chunk.dispatch_events) for chunk in chunks] == [2, 2]
    assert [len(chunk.discrete_dispatch_events) for chunk in chunks] == [1, 1]
    times, qpos, metrics = _recorded_chunk_arrays(chunks[0])
    assert np.allclose(times, [1.0 / 120.0, 1.0 / 120.0 + 0.01])
    assert qpos.shape == (2, 12)
    assert np.isclose(metrics["max_gap_ms"], 10.0)
    assert np.isclose(metrics["recorded_tail_hold_sec"], 0.04)


def test_auto_selection_prefers_larger_recorded_joint_step() -> None:
    events = (
        {"event": "policy_chunk"},
        _dispatch(10.000, 0.0, 0.0),
        _dispatch(10.010, 0.01, -0.01),
        {"event": "policy_chunk"},
        _dispatch(11.000, 0.0, 0.0),
        _dispatch(11.010, 0.10, -0.10),
    )
    chunks = recorded_chunks_from_events(events)

    selected, metrics = _auto_select_chunk(
        chunks,
        {0: _track(0), 1: _track(1)},
    )

    assert selected.chunk_id == 1
    assert metrics["max_joint_step_deg"] > 5.0


def test_full_recorded_trajectory_preserves_all_chunks_and_wall_timing() -> None:
    events = (
        {"event": "policy_chunk"},
        _dispatch(10.000, 0.0, 0.0),
        _discrete(10.005, 1, 0.25, 0.5),
        _dispatch(10.010, 0.01, -0.01),
        {"event": "policy_chunk"},
        _dispatch(10.500, 0.02, -0.02),
        _discrete(10.505, 1, 0.75, 1.0),
        _dispatch(10.510, 0.03, -0.03),
    )
    chunks = recorded_chunks_from_events(events)

    trajectory, metrics = _trajectory_from_recorded_full(
        {0: _track(0), 1: _track(1)},
        chunks,
    )

    assert trajectory.source_episode_index == -1
    assert trajectory.source_frame_count == 6
    assert len(trajectory.frames) == 4
    assert np.isclose(
        trajectory.frames[-1].timestamp - trajectory.frames[0].timestamp,
        0.510,
    )
    assert metrics["chunk_count"] == 2
    assert metrics["dispatch_count"] == 4
    assert metrics["gripper"]["recorded_discrete_dispatch_count"] == 2
    assert trajectory.frames[0].gripper_close_by_name == {
        "left_gripper": 0.0,
        "right_gripper": 0.0,
    }
    assert trajectory.frames[-1].gripper_close_by_name == {
        "left_gripper": 0.75,
        "right_gripper": 1.0,
    }


def test_gripper_source_targets_shift_with_stream_start_prelude() -> None:
    chunk = recorded_chunks_from_events(
        (
            {"event": "policy_chunk"},
            _dispatch(10.000, 0.0, 0.0),
            _discrete(10.005, 1, 0.2, 0.8),
            _dispatch(10.010, 0.01, -0.01),
        )
    )[0]
    targets = _gripper_targets_by_source_index(
        chunk,
        source_index_shift=9,
    )
    state = {"left_gripper": 0.0, "right_gripper": 0.0}

    before = _advance_gripper_source_state(
        state,
        targets,
        source_index=9,
    )
    after = _advance_gripper_source_state(
        state,
        targets,
        source_index=10,
    )

    assert before == {"left_gripper": 0.0, "right_gripper": 0.0}
    assert after == {"left_gripper": 0.2, "right_gripper": 0.8}


def test_rollout_trajectory_is_fixed_120hz_and_keeps_initial_qpos() -> None:
    track = _track(3)
    qpath = np.vstack((np.full(12, 0.01), np.full(12, 0.02)))
    artifact = RolloutArtifact(
        summary={"error": None, "command_hz": 120.0},
        raw_metrics={},
        q_command_rad=qpath,
        q_ik_raw_rad=qpath.copy(),
        source_qpos_rad_by_index={
            0: {"left": np.zeros(6), "right": np.zeros(6)},
        },
    )

    trajectory = _trajectory_from_rollout(
        track,
        artifact,
        profile_name="candidate",
        ik_hz=30.0,
    )

    assert trajectory.source_episode_index == 3
    assert trajectory.reference_hz == 120.0
    assert np.array_equal(trajectory.initial_qpos_by_side["left"], np.zeros(6))
    assert [frame.timestamp for frame in trajectory.frames] == [
        1.0 / 120.0,
        2.0 / 120.0,
    ]
    assert np.array_equal(trajectory.frames[-1].qpos_by_side["right"], qpath[-1, 6:])
    assert trajectory.initial_gripper_close_by_name == {
        "left_gripper": 0.0,
        "right_gripper": 0.0,
    }


def test_viewer_namespace_has_fixed_120hz_reference_rate() -> None:
    args = build_parser().parse_args(
        ["--debug-jsonl", "/tmp/debug.jsonl", "--play", "recorded"]
    )

    assert args.reference_hz == 120.0
    assert args.planner_command_hz == 120.0
    assert args.candidate_trajectory_mode == "formal"
    assert args.candidate_qref_acceleration_profile == "hard_v1"


def test_viewer_accepts_deploy_qref_acceleration_profile() -> None:
    args = build_parser().parse_args(
        [
            "--debug-jsonl",
            "/tmp/debug.jsonl",
            "--play",
            "candidate",
            "--candidate-qref-acceleration-profile",
            "comfort_50_v1",
        ]
    )

    assert args.candidate_qref_acceleration_profile == "comfort_50_v1"


def test_rollout_trajectory_supports_current_60hz_reference_layer() -> None:
    track = _track(5)
    qpath = np.vstack((np.full(12, 0.01), np.full(12, 0.02)))
    artifact = RolloutArtifact(
        summary={"error": None, "command_hz": 60.0},
        raw_metrics={},
        q_command_rad=qpath,
        q_ik_raw_rad=qpath.copy(),
        source_qpos_rad_by_index={},
    )

    trajectory = _trajectory_from_rollout(
        track,
        artifact,
        profile_name="candidate_c2",
        ik_hz=60.0,
    )

    assert trajectory.reference_hz == 60.0
    assert [frame.timestamp for frame in trajectory.frames] == [
        1.0 / 60.0,
        2.0 / 60.0,
    ]
    assert [frame.interpolation_alpha for frame in trajectory.frames] == [
        0.25,
        0.5,
    ]


def test_viewer_startup_hold_is_visual_only_and_shifts_timestamps() -> None:
    track = _track(4)
    qpath = np.full((1, 12), 0.02)
    artifact = RolloutArtifact(
        summary={"error": None},
        raw_metrics={},
        q_command_rad=qpath,
        q_ik_raw_rad=qpath.copy(),
        source_qpos_rad_by_index={},
    )
    trajectory = _trajectory_from_rollout(
        track,
        artifact,
        profile_name="candidate",
        ik_hz=30.0,
    )

    viewed = _with_viewer_startup_hold(trajectory, hold_sec=0.75)

    assert len(viewed.frames) == 2
    assert viewed.frames[0].status == "viewer_startup_hold"
    assert viewed.frames[0].timestamp == 0.75
    assert viewed.frames[1].timestamp == 0.75 + 1.0 / 120.0
    assert trajectory.frames[0].timestamp == 1.0 / 120.0


def test_cli_treats_keyboard_interrupt_as_clean_viewer_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "todos.probes.view_rm65_deploy_ik_rate_comparison_mujoco.main",
        lambda _argv=None: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    assert cli_main([]) == 0
    assert "interrupted; viewer closed cleanly" in capsys.readouterr().out


@pytest.mark.parametrize(
    "filename",
    (
        "20260808_142726_sim_baseline.md",
        "20260808_142726_sim_candidate.md",
    ),
)
def test_sim_handoff_commands_play_full_once_without_shell_control_logic(
    filename: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    command = (root / "scripts" / "temp_cmd" / filename).read_text(
        encoding="utf-8"
    )

    assert "--full-deployment" in command
    assert "--no-loop" in command
    assert "--repeat 1" in command
    assert "--hold-viewer-at-end" in command
    assert command.startswith(
        "conda activate /home/zc/anaconda3/envs/dexteleop310\npython "
    )
    for forbidden in (
        "#!",
        "trap ",
        "set ",
        "exec ",
        "| tee",
        "return ",
        "exit ",
        "cd ",
        "(",
        ")",
    ):
        assert forbidden not in command
