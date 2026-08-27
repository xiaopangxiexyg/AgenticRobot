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
    DETAIL_PHASE_IDS,
    _grippers_for_reference,
    _load_detail_writing_selection,
    _require_left_hold_state,
    _stage_targets,
    _circle_targets,
    _require_pose_frame_alignment,
    _trajectory_tcp_speed_summary,
    StageSpan,
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
    assert args.fixed_trajectory_scale_m == pytest.approx(0.08)
    assert args.fixed_trajectory_direction == "up"

    digit_args = parse_args(
        [
            "--output",
            "/tmp/rule-based-digit-test",
            "--fixed-trajectory-id",
            "digit_1_v1",
            "--fixed-trajectory-size-m",
            "0.12",
            "--fixed-trajectory-direction",
            "left",
        ]
    )
    assert digit_args.fixed_trajectory_id == "digit_1_v1"
    assert digit_args.fixed_trajectory_speed_m_s is None
    assert digit_args.fixed_trajectory_scale_m == pytest.approx(0.12)
    assert digit_args.fixed_trajectory_direction == "left"

    scale_alias_args = parse_args(
        [
            "--output",
            "/tmp/rule-based-digit-scale-alias-test",
            "--fixed-trajectory-id",
            "digit_1_v1",
            "--fixed-trajectory-scale-m",
            "0.06",
        ]
    )
    assert scale_alias_args.fixed_trajectory_scale_m == pytest.approx(0.06)

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


def test_detail_annotation_maps_three_phases_to_half_open_source_spans(
    tmp_path, monkeypatch
) -> None:
    module = __import__(
        "sim.mujoco.replay.rule_based_multistage_demo",
        fromlist=["load_detail_stage_annotation_set"],
    )
    episode_uuid = "episode-detail-uuid"
    base_id = "stage-base"
    identity_hash = "a" * 64
    manifest_hash = "b" * 64
    base_manifest_hash = "c" * 64
    base_profile_hash = "d" * 64
    detail_profile_hash = "e" * 64
    attempt_id = "attempt-1"
    boundaries = (1.0, 2.0, 3.0, 4.0)
    occurrences = tuple(
        SimpleNamespace(
            progress_step_id="right_write",
            stage_id=stage_id,
            coarse_attempt_id=attempt_id,
            occurrence_index_in_attempt=index,
            start_at=boundaries[index - 1],
            end_at=boundaries[index],
            outcome="SUCCESS",
            clock_domain_id="local_wall_time",
            annotation_id=f"annotation-{index}",
        )
        for index, stage_id in enumerate(DETAIL_PHASE_IDS, start=1)
    )
    detail = SimpleNamespace(
        binding=SimpleNamespace(
            episode_capture_identity_hash=identity_hash,
            source_progress_manifest_hash=manifest_hash,
            bound_clock_domain_id="local_wall_time",
            target_progress_step_id="right_write",
            base_annotation_set_id=base_id,
            base_task_stage_profile_hash=base_profile_hash,
            detail_task_stage_profile_hash=detail_profile_hash,
            annotation_set_id="detail-set",
            base_annotation_set_manifest_hash=base_manifest_hash,
        ),
        bundle=SimpleNamespace(
            occurrences=occurrences,
            semantic_content_hash="f" * 64,
        ),
        manifest=SimpleNamespace(semantic_content_hash="1" * 64),
    )
    sealed = SimpleNamespace(
        binding=SimpleNamespace(
            episode_capture_identity_hash=identity_hash,
            bound_clock_domain_id="local_wall_time",
        ),
        manifest=SimpleNamespace(semantic_content_hash=manifest_hash),
    )
    base = SimpleNamespace(
        binding=SimpleNamespace(
            annotation_set_id=base_id,
            episode_capture_identity_hash=identity_hash,
            source_progress_manifest_hash=manifest_hash,
            task_stage_profile_hash=base_profile_hash,
            bound_clock_domain_id="local_wall_time",
        ),
        manifest=SimpleNamespace(semantic_content_hash=base_manifest_hash),
    )
    monkeypatch.setattr(module, "load_detail_stage_annotation_set", lambda _: detail)
    monkeypatch.setattr(module, "load_sealed_progress_episode", lambda _: sealed)
    monkeypatch.setattr(module, "load_stage_annotation_set", lambda _: base)
    episode = SimpleNamespace(
        episode=SimpleNamespace(
            dataset_root=tmp_path,
            record={
                "metadata": {
                    "progress_episode": {"episode_uuid": episode_uuid}
                }
            },
        ),
        frames=tuple(
            SimpleNamespace(timestamp=float(index)) for index in range(6)
        ),
    )

    selection = _load_detail_writing_selection(
        episode,
        rows=[{"timestamp": float(index)} for index in range(6)],
        annotation_dir=tmp_path / "detail",
    )

    assert selection.coarse_attempt_id == attempt_id
    assert selection.phases["right_write.pen_down"].span == StageSpan(1, 2)
    assert selection.phases["right_write.write"].span.frame_count == 1
    assert selection.phases["right_write.pen_up"].span == StageSpan(3, 4)


def test_detail_left_hold_is_constant_and_stage_targets_keep_it_fixed() -> None:
    hold = _pose(-0.1, 0.2, 0.3)
    right_a = _pose(0.1, 0.0, 0.4)
    right_b = _pose(0.12, 0.0, 0.4)
    frames = tuple(
        SimpleNamespace(
            timestamp=float(index),
            frame_index=index,
            pose_desk_by_side={"left": hold.copy(), "right": right.copy()},
            gripper_close_by_side={"left": 0.25, "right": 0.5},
        )
        for index, right in enumerate((right_a, right_b))
    )
    rows = tuple({"teleop.arm_mode.left.label": "fixed_hold"} for _ in frames)

    hold_by_side, hold_gripper, source_index = _require_left_hold_state(
        rows,
        frames,
        StageSpan(0, 2),
    )
    assert source_index == 0
    assert hold_gripper == pytest.approx(0.25)
    np.testing.assert_allclose(hold_by_side["left"], hold)

    references = _stage_targets(
        frames,
        phase="right_write.pen_down",
        hold_pose_by_side=hold_by_side,
    )
    np.testing.assert_allclose(references[0].pose_by_side["left"], hold)
    np.testing.assert_allclose(references[1].pose_by_side["left"], hold)
    grippers = _grippers_for_reference(
        frames,
        SimpleNamespace(
            left_source_index=0,
            right_source_index=1,
            interpolation_alpha=0.5,
        ),
        hold_by_side={"left": hold_gripper},
    )
    assert grippers["left"] == pytest.approx(0.25)
