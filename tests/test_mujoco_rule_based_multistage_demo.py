from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.pose_frames import load_pose_frame_contract
from robot_motion.pose_frame_profiles import (
    DEFAULT_RM65_DUAL_ARM_COLLECTION_POSE_FRAME_CONTRACT_PATH,
    DEFAULT_RM65_DUAL_ARM_RUNTIME_POSE_FRAME_CONTRACT_PATH,
)
from sim.mujoco.replay.rule_based_multistage_demo import (
    _circle_targets,
    _require_pose_frame_alignment,
    _trajectory_tcp_speed_summary,
    parse_args,
    parse_stage_span,
)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def test_parse_stage_span_is_half_open_and_bounded() -> None:
    span = parse_stage_span("3:8", total_frames=10)

    assert span.start == 3
    assert span.end == 8
    assert span.frame_count == 5

    with pytest.raises(ValueError, match="exceeds source frame count"):
        parse_stage_span("3:11", total_frames=10)


def test_circle_is_continuous_and_parallel_to_desk_xy() -> None:
    boundary = {"left": _pose(-0.1, 0.2, 0.3), "right": _pose(0.1, -0.2, 0.4)}
    targets = _circle_targets(boundary, radius_m=0.02, steps=9, hz=30.0)

    right = np.stack([np.asarray(target.pose_by_side["right"]) for target in targets])
    left = np.stack([np.asarray(target.pose_by_side["left"]) for target in targets])
    center = np.asarray([0.08, -0.2, 0.4])
    radius = np.linalg.norm(right[:, :3, 3] - center, axis=1)

    np.testing.assert_allclose(right[0], boundary["right"])
    np.testing.assert_allclose(right[-1], boundary["right"])
    np.testing.assert_allclose(left, np.repeat(boundary["left"][None, ...], len(targets), axis=0))
    np.testing.assert_allclose(radius, 0.02, atol=1e-9)
    np.testing.assert_allclose(right[:, 2, 3], 0.4, atol=1e-9)
    np.testing.assert_allclose(
        right[:, :3, :3],
        np.repeat(boundary["right"][None, :3, :3], len(targets), axis=0),
    )


def test_circle_linear_speed_is_defined_by_target_path_duration() -> None:
    boundary = {"left": _pose(-0.1, 0.2, 0.3), "right": _pose(0.1, -0.2, 0.4)}
    targets = _circle_targets(
        boundary,
        radius_m=0.02,
        steps=61,
        linear_speed_m_s=0.05,
    )

    positions = np.stack(
        [np.asarray(target.pose_by_side["right"], dtype=float)[:3, 3] for target in targets]
    )
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    duration = float(targets[-1].timestamp - targets[0].timestamp)

    assert path_length / duration == pytest.approx(0.05)
    assert targets[0].metadata["pose_frame"] == "canonical_desk"
    assert targets[0].metadata["speed_source"] == "explicit_linear_speed_m_s"


def test_selected_source_tcp_speed_ignores_gaps_between_editable_spans() -> None:
    frames_a = tuple(
        SimpleNamespace(timestamp=float(index), pose_desk_by_side={"right": _pose(0.1 * index, 0.0, 0.4)})
        for index in (0, 1)
    )
    frames_b = tuple(
        SimpleNamespace(timestamp=float(index + 10), pose_desk_by_side={"right": _pose(0.1 * index, 0.0, 0.4)})
        for index in (0, 1)
    )

    summary = _trajectory_tcp_speed_summary((frames_a, frames_b), side="right")

    assert summary["path_length_m"] == pytest.approx(0.2)
    assert summary["duration_sec"] == pytest.approx(2.0)
    assert summary["average_linear_speed_m_s"] == pytest.approx(0.1)


def test_default_circle_speed_is_source_derived_and_station_contract_is_required() -> None:
    args = parse_args(["--output", "/tmp/rule-based-test"])
    assert args.circle_hz is None
    assert args.circle_speed_m_s is None

    contract = load_pose_frame_contract(
        DEFAULT_RM65_DUAL_ARM_COLLECTION_POSE_FRAME_CONTRACT_PATH
    )
    episode = SimpleNamespace(
        pose_frame_contract=contract,
        pose_frame_contract_hash=contract.contract_hash,
    )
    control = SimpleNamespace(
        station_pose_contract=contract,
        world_from_desk=np.eye(4),
    )

    alignment = _require_pose_frame_alignment(episode, control)

    assert alignment["contract_match"] is True
    assert alignment["canonical_absolute_frame"] == "desk"
    assert alignment["action_pose_frame"] == "canonical_desk"

    runtime_contract = load_pose_frame_contract(
        DEFAULT_RM65_DUAL_ARM_RUNTIME_POSE_FRAME_CONTRACT_PATH
    )
    with pytest.raises(ValueError, match="does not match"):
        _require_pose_frame_alignment(
            episode,
            SimpleNamespace(
                station_pose_contract=runtime_contract,
                world_from_desk=np.eye(4),
            ),
        )
