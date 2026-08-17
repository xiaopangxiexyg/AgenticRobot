from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.policy_learning.arm_hold import arm_hold_window_evaluation


def _perfect_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(
        [
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 1],
            ]
        ],
        dtype=np.float32,
    )
    transition = np.asarray(
        [
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 0],
            ]
        ],
        dtype=bool,
    )
    logits = np.where(target == 1, 8.0, -8.0).astype(np.float32)
    valid = np.ones_like(target, dtype=bool)
    return logits, target, valid, transition


def test_perfect_report_covers_sides_directions_and_pairs() -> None:
    logits, target, valid, transition = _perfect_case()

    report = arm_hold_window_evaluation(
        logits=logits,
        target=target,
        valid_mask=valid,
        transition_mask=transition,
    )

    assert report["scope"] == "independent_admitted_windows_v1"
    assert report["overall"]["balanced_accuracy"] == 1.0
    assert report["macro_side"]["balanced_accuracy"] == 1.0
    assert report["overall"]["pr_auc_average_precision"] == 1.0
    assert report["per_side"]["left"]["f1"] == 1.0
    assert report["pair"]["exact_match_accuracy"] == 1.0
    assert report["pair"]["confusion"] == {
        "00": {"00": 1},
        "01": {"01": 1},
        "10": {"10": 1},
        "11": {"11": 1},
    }
    assert report["exact_transition"] == {
        "left_enter_hold": {"support": 1, "correct": 1, "recall": 1.0},
        "left_exit_hold": {"support": 1, "correct": 1, "recall": 1.0},
        "right_enter_hold": {"support": 1, "correct": 1, "recall": 1.0},
        "right_exit_hold": {"support": 0, "correct": 0, "recall": None},
    }
    assert "signed_transition_timing" in report["excluded_sequence_metrics"]


def test_masks_exclude_values_from_metrics_and_transition_support() -> None:
    logits, target, valid, transition = _perfect_case()
    logits[0, 0, 0] = np.nan
    target[0, 0, 0] = np.nan
    valid[0, 0, 0] = False
    transition[0, 1, 0] = False
    loss = np.ones_like(valid)
    loss[0, 3, 0] = False
    transition[0, 3, 0] = False

    report = arm_hold_window_evaluation(
        logits=torch.from_numpy(logits),
        target=torch.from_numpy(target),
        valid_mask=torch.from_numpy(valid),
        loss_mask=torch.from_numpy(loss),
        transition_mask=torch.from_numpy(transition),
    )

    assert report["per_side"]["left"]["support"] == 2
    assert report["per_side"]["right"]["support"] == 4
    assert report["pair"]["support"] == 2
    assert report["exact_transition"]["left_enter_hold"]["support"] == 0


def test_constant_positive_prediction_exposes_collapse() -> None:
    _, target, valid, transition = _perfect_case()
    logits = np.full_like(target, 8.0)

    report = arm_hold_window_evaluation(
        logits=logits,
        target=target,
        valid_mask=valid,
        transition_mask=transition,
    )

    assert report["overall"]["accuracy"] == 0.5
    assert report["overall"]["balanced_accuracy"] == 0.5
    assert report["macro_side"]["balanced_accuracy"] == 0.5
    assert report["overall"]["true_negative"] == 0
    assert report["exact_transition"]["left_exit_hold"]["recall"] == 0.0
    assert report["pair"]["exact_match_accuracy"] == 0.25


def test_uncovered_class_is_explicitly_undefined() -> None:
    target = np.zeros((1, 2, 2), dtype=np.float32)
    report = arm_hold_window_evaluation(
        logits=np.full_like(target, -1.0),
        target=target,
        valid_mask=np.ones_like(target, dtype=bool),
    )

    assert report["overall"]["balanced_accuracy"] is None
    assert report["overall"]["recall"] is None
    assert report["overall"]["pr_auc_average_precision"] is None
    assert report["overall"]["accuracy"] == 1.0


def test_macro_side_exposes_side_specific_constant_predictors() -> None:
    target = np.asarray(
        [[[1, 0], [1, 0], [1, 0], [0, 1]]], dtype=np.float32
    )
    logits = np.asarray(
        [[[8, -8], [8, -8], [8, -8], [8, -8]]], dtype=np.float32
    )

    report = arm_hold_window_evaluation(
        logits=logits,
        target=target,
        valid_mask=np.ones_like(target, dtype=bool),
    )

    assert report["overall"]["balanced_accuracy"] == 0.75
    assert report["macro_side"]["balanced_accuracy"] == 0.5


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold", 1.0, "threshold"),
        ("target", np.full((1, 1, 2), 0.5), "binary"),
        ("valid_mask", np.ones((1, 1, 2), dtype=np.int64), "bool"),
    ],
)
def test_invalid_evaluation_inputs_fail_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "logits": np.zeros((1, 1, 2), dtype=np.float32),
        "target": np.zeros((1, 1, 2), dtype=np.float32),
        "valid_mask": np.ones((1, 1, 2), dtype=bool),
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        arm_hold_window_evaluation(**kwargs)


def test_transition_cannot_point_to_excluded_element() -> None:
    transition = np.zeros((1, 1, 2), dtype=bool)
    transition[0, 0, 0] = True
    valid = np.ones_like(transition)
    valid[0, 0, 0] = False

    with pytest.raises(ValueError, match="cannot select"):
        arm_hold_window_evaluation(
            logits=np.zeros((1, 1, 2), dtype=np.float32),
            target=np.zeros((1, 1, 2), dtype=np.float32),
            valid_mask=valid,
            transition_mask=transition,
        )
