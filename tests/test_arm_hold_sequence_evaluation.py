from __future__ import annotations

import numpy as np
import pytest

from experiments.policy_learning.arm_hold import (
    arm_hold_chunk_sequence_evaluation,
    arm_hold_fixed_lead_sequence_evaluation,
)
from experiments.policy_learning.arm_hold.sequence_evaluation import _match_events


def _case(
    target_values: list[int],
    prediction_values: list[int],
    *,
    starts: list[int] | None = None,
    transition_values: list[bool] | None = None,
) -> dict[str, object]:
    count = len(target_values)
    assert len(prediction_values) == count
    target = np.repeat(
        np.asarray(target_values, dtype=np.float32)[:, None, None],
        2,
        axis=2,
    )
    logits = np.where(
        np.repeat(
            np.asarray(prediction_values, dtype=bool)[:, None, None],
            2,
            axis=2,
        ),
        8.0,
        -8.0,
    ).astype(np.float32)
    if transition_values is None:
        transitions = [False] + [
            current != previous
            for previous, current in zip(target_values[:-1], target_values[1:])
        ]
    else:
        transitions = transition_values
    transition_mask = np.repeat(
        np.asarray(transitions, dtype=bool)[:, None, None],
        2,
        axis=2,
    )
    return {
        "logits": logits,
        "target": target,
        "valid_mask": np.ones_like(target, dtype=bool),
        "loss_mask": np.ones_like(target, dtype=bool),
        "transition_mask": transition_mask,
        "metadata": tuple(
            {
                "source_dataset_id": "source",
                "episode_capture_identity_hash": "episode",
                "continuity_segment_index": 0,
                "start_index": start,
            }
            for start in (range(count) if starts is None else starts)
        ),
        "lead_offset": 0,
    }


def _chunk_case(
    target_values: list[int],
    prediction_values: list[int],
    *,
    start: int = 10,
    transition_values: list[bool] | None = None,
) -> dict[str, object]:
    assert len(target_values) == len(prediction_values)
    target = np.repeat(
        np.asarray(target_values, dtype=np.float32)[None, :, None],
        2,
        axis=2,
    )
    logits = np.where(
        np.repeat(
            np.asarray(prediction_values, dtype=bool)[None, :, None],
            2,
            axis=2,
        ),
        8.0,
        -8.0,
    ).astype(np.float32)
    if transition_values is None:
        transitions = [False] + [
            current != previous
            for previous, current in zip(target_values[:-1], target_values[1:])
        ]
    else:
        transitions = transition_values
    return {
        "logits": logits,
        "target": target,
        "valid_mask": np.ones_like(target, dtype=bool),
        "loss_mask": np.ones_like(target, dtype=bool),
        "transition_mask": np.repeat(
            np.asarray(transitions, dtype=bool)[None, :, None],
            2,
            axis=2,
        ),
        "metadata": (
            {
                "source_dataset_id": "source",
                "episode_capture_identity_hash": "episode",
                "continuity_segment_index": 0,
                "start_index": start,
            },
        ),
    }


def test_perfect_fixed_lead_sequence_has_exact_events_and_no_bounce() -> None:
    report = arm_hold_fixed_lead_sequence_evaluation(
        **_case([0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0])
    )

    left = report["per_side"]["left"]
    assert left["transition"]["enter_hold"]["recall_exact"] == 1.0
    assert left["transition"]["exit_hold"]["recall_exact"] == 1.0
    assert left["stable_boundary_count"] == 3
    assert left["stable_false_switch_count"] == 0
    assert left["unmatched_predicted_transition_count"] == 0
    assert left["observed_predicted_run_steps"] == {
        "count": 3,
        "minimum": 2,
        "maximum": 2,
        "mean": 2.0,
        "median": 2.0,
    }
    assert left["interior_predicted_dwell_steps"]["count"] == 1
    assert left["single_frame_interior_predicted_segment_count"] == 0
    assert "segments" not in left


def test_tolerance_matching_reports_signed_timing_and_unmatched_switches() -> None:
    report = arm_hold_fixed_lead_sequence_evaluation(
        **_case(
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
        )
    )

    left = report["per_side"]["left"]
    enter = left["transition"]["enter_hold"]
    exit_hold = left["transition"]["exit_hold"]
    assert enter["recall_exact"] == 0.0
    assert enter["recall_within_tolerance"] == 1.0
    assert enter["signed_timing_offset_steps"]["mean"] == 1.0
    assert exit_hold["recall_within_tolerance"] == 1.0
    assert exit_hold["signed_timing_offset_steps"]["mean"] == -1.0
    assert left["predicted_transition_count"] == 4
    assert left["unmatched_predicted_transition_count"] == 2
    assert left["stable_false_switch_count"] == 3
    assert left["single_frame_interior_predicted_segment_count"] == 3


def test_gap_breaks_timeline_and_boundary_transition_is_not_scored() -> None:
    report = arm_hold_fixed_lead_sequence_evaluation(
        **_case(
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            starts=[0, 1, 3, 4],
            transition_values=[False, False, True, False],
        )
    )

    left = report["per_side"]["left"]
    assert left["continuity_coverage_segment_count"] == 2
    assert left["unsupported_boundary_transition_count"] == 1
    assert left["transition"]["enter_hold"]["support"] == 0
    assert left["interior_predicted_segment_count"] == 0


def test_fixed_lead_uses_one_explicit_horizon_column() -> None:
    case = _case([0, 0, 0], [0, 0, 0])
    case["logits"] = np.asarray(
        [
            [[-8, -8], [-8, -8]],
            [[-8, -8], [8, 8]],
            [[-8, -8], [8, 8]],
        ],
        dtype=np.float32,
    )
    case["target"] = np.asarray(
        [
            [[0, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ],
        dtype=np.float32,
    )
    case["valid_mask"] = np.ones((3, 2, 2), dtype=bool)
    case["loss_mask"] = np.ones((3, 2, 2), dtype=bool)
    case["transition_mask"] = np.asarray(
        [
            [[0, 0], [0, 0]],
            [[0, 0], [1, 1]],
            [[0, 0], [0, 0]],
        ],
        dtype=bool,
    )
    case["lead_offset"] = 1

    report = arm_hold_fixed_lead_sequence_evaluation(**case)

    assert report["lead_offset"] == 1
    assert report["per_side"]["left"]["transition"]["enter_hold"][
        "recall_exact"
    ] == 1.0


def test_duplicate_fixed_lead_target_identity_fails_closed() -> None:
    case = _case([0, 0], [0, 0], starts=[0, 0])

    with pytest.raises(ValueError, match="Duplicate"):
        arm_hold_fixed_lead_sequence_evaluation(**case)


def test_canonical_transition_conflict_fails_closed() -> None:
    case = _case(
        [0, 0, 1],
        [0, 0, 1],
        transition_values=[False, False, False],
    )

    with pytest.raises(ValueError, match="conflicts"):
        arm_hold_fixed_lead_sequence_evaluation(**case)


def test_masked_point_breaks_sequence_without_inventing_transition() -> None:
    case = _case([0, 0, 1, 1, 1], [0, 0, 1, 1, 1])
    case["valid_mask"][2] = False
    case["loss_mask"][2] = False
    case["transition_mask"][2] = False

    report = arm_hold_fixed_lead_sequence_evaluation(**case)

    left = report["per_side"]["left"]
    assert left["continuity_coverage_segment_count"] == 2
    assert left["transition"]["enter_hold"]["support"] == 0
    assert left["stable_false_switch_count"] == 0


def test_debug_only_adds_segment_details() -> None:
    report = arm_hold_fixed_lead_sequence_evaluation(
        **_case([0, 0], [0, 0]), debug=True
    )

    assert report["debug"] is True
    assert report["per_side"]["left"]["segments"][0]["point_count"] == 2


def test_perfect_chunk_sequence_scores_only_inside_chunk_boundaries() -> None:
    report = arm_hold_chunk_sequence_evaluation(
        **_chunk_case(
            [1, 1, 0, 0, 1],
            [1, 1, 0, 0, 1],
            transition_values=[True, False, True, False, True],
        )
    )

    left = report["per_side"]["left"]
    assert report["scope"] == "within_output_chunk_timelines_v1"
    assert report["transition_occurrence_basis"] == (
        "per_overlapping_output_window_v1"
    )
    assert left["unsupported_boundary_transition_count"] == 1
    assert left["transition"]["enter_hold"]["support"] == 1
    assert left["transition"]["exit_hold"]["support"] == 1
    assert left["transition"]["enter_hold"]["recall_exact"] == 1.0
    assert left["transition"]["exit_hold"]["recall_exact"] == 1.0


def test_chunk_sequence_timing_is_matched_within_each_plan() -> None:
    report = arm_hold_chunk_sequence_evaluation(
        **_chunk_case(
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 0],
        )
    )

    left = report["per_side"]["left"]
    assert left["transition"]["enter_hold"]["recall_exact"] == 0.0
    assert left["transition"]["enter_hold"]["recall_within_tolerance"] == 1.0
    assert left["transition"]["enter_hold"]["signed_timing_offset_steps"][
        "mean"
    ] == 1.0
    assert left["transition"]["exit_hold"]["recall_exact"] == 1.0


def test_chunk_mask_gap_breaks_plan_without_bridging_a_transition() -> None:
    case = _chunk_case([0, 0, 1, 1, 0], [0, 0, 1, 1, 0])
    case["valid_mask"][:, 2] = False
    case["loss_mask"][:, 2] = False
    case["transition_mask"][:, 2] = False

    report = arm_hold_chunk_sequence_evaluation(**case)

    left = report["per_side"]["left"]
    assert left["chunk_coverage_segment_count"] == 2
    assert left["transition"]["enter_hold"]["support"] == 0
    assert left["transition"]["exit_hold"]["support"] == 1


def test_duplicate_chunk_identity_fails_closed() -> None:
    first = _chunk_case([0, 0], [0, 0])
    case = {
        key: (np.concatenate((value, value), axis=0) if isinstance(value, np.ndarray) else value)
        for key, value in first.items()
    }
    case["metadata"] = first["metadata"] + first["metadata"]

    with pytest.raises(ValueError, match="Duplicate"):
        arm_hold_chunk_sequence_evaluation(**case)


def test_chunk_debug_details_are_default_off() -> None:
    case = _chunk_case([0, 1], [0, 1])
    ordinary = arm_hold_chunk_sequence_evaluation(**case)
    debug = arm_hold_chunk_sequence_evaluation(**case, debug=True)

    assert "segments" not in ordinary["per_side"]["left"]
    segment = debug["per_side"]["left"]["segments"][0]
    assert segment["window_offset_start"] == 0
    assert segment["window_offset_stop_exclusive"] == 2


def test_event_matching_maximizes_cardinality_before_nearest_distance() -> None:
    offsets, matched = _match_events([2, 3], [1, 2], tolerance=1)

    assert offsets == [-1, -1]
    assert matched == {1, 2}
