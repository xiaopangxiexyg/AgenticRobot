"""Learning metrics for independent arm-hold window predictions.

The report intentionally stays at the admitted-window level.  It does not
reconstruct one authoritative Episode timeline from overlapping windows, so
sequence metrics such as signed transition timing, bouncing, and dwell are
outside this module's scope.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


ARM_HOLD_EVALUATION_SCHEMA = "dexteleop.arm_hold_window_evaluation.v1"
ARM_HOLD_SIDE_NAMES = ("left", "right")


class ArmHoldEvaluationError(ValueError):
    """Raised when prediction arrays cannot be compared unambiguously."""


def _numpy(value: Any) -> np.ndarray:
    """Convert numpy/torch-like inputs without retaining an autograd graph."""

    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _average_precision(
    probability: np.ndarray,
    target: np.ndarray,
) -> float | None:
    positives = int(np.count_nonzero(target))
    if positives == 0:
        return None
    order = np.argsort(-probability, kind="stable")
    sorted_target = target[order].astype(np.int64, copy=False)
    true_positive = np.cumsum(sorted_target)
    rank = np.arange(1, sorted_target.size + 1)
    precision = true_positive / rank
    return float(np.sum(precision * sorted_target) / positives)


def _binary_report(
    *,
    probability: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    target_bool = target.astype(bool, copy=False)
    prediction_bool = prediction.astype(bool, copy=False)
    tp = int(np.count_nonzero(prediction_bool & target_bool))
    tn = int(np.count_nonzero(~prediction_bool & ~target_bool))
    fp = int(np.count_nonzero(prediction_bool & ~target_bool))
    fn = int(np.count_nonzero(~prediction_bool & target_bool))
    positive_recall = _safe_ratio(tp, tp + fn)
    negative_recall = _safe_ratio(tn, tn + fp)
    balanced_accuracy = (
        None
        if positive_recall is None or negative_recall is None
        else float((positive_recall + negative_recall) / 2.0)
    )
    precision = _safe_ratio(tp, tp + fp)
    f1 = (
        None
        if precision is None or positive_recall is None
        else (
            0.0
            if precision + positive_recall == 0.0
            else float(
                2.0 * precision * positive_recall / (precision + positive_recall)
            )
        )
    )
    return {
        "support": int(target.size),
        "positive_support": int(tp + fn),
        "negative_support": int(tn + fp),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": _safe_ratio(tp + tn, target.size),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": positive_recall,
        "f1": f1,
        "pr_auc_average_precision": _average_precision(
            probability, target_bool
        ),
    }


def _validate_arrays(
    *,
    logits: Any,
    target: Any,
    valid_mask: Any,
    loss_mask: Any | None,
    transition_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_logits = _numpy(logits).astype(np.float64, copy=False)
    resolved_target = _numpy(target).astype(np.float64, copy=False)
    resolved_valid = _numpy(valid_mask)
    if resolved_logits.ndim != 3 or resolved_logits.shape[-1] != 2:
        raise ArmHoldEvaluationError("logits must have shape [B,H,2].")
    if resolved_target.shape != resolved_logits.shape:
        raise ArmHoldEvaluationError("target shape must match logits.")
    if resolved_valid.shape != resolved_logits.shape or resolved_valid.dtype != np.bool_:
        raise ArmHoldEvaluationError("valid_mask must be bool and match logits.")
    resolved_loss = (
        np.ones_like(resolved_valid)
        if loss_mask is None
        else _numpy(loss_mask)
    )
    if resolved_loss.shape != resolved_logits.shape or resolved_loss.dtype != np.bool_:
        raise ArmHoldEvaluationError("loss_mask must be bool and match logits.")
    resolved_transition = (
        np.zeros_like(resolved_valid)
        if transition_mask is None
        else _numpy(transition_mask)
    )
    if (
        resolved_transition.shape != resolved_logits.shape
        or resolved_transition.dtype != np.bool_
    ):
        raise ArmHoldEvaluationError(
            "transition_mask must be bool and match logits."
        )
    effective = resolved_valid & resolved_loss
    if not np.any(effective):
        raise ArmHoldEvaluationError("Evaluation contains no valid target elements.")
    if not np.all(np.isfinite(resolved_logits[effective])):
        raise ArmHoldEvaluationError("Valid logits must be finite.")
    if not np.all(np.isin(resolved_target[effective], (0.0, 1.0))):
        raise ArmHoldEvaluationError("Valid targets must be binary.")
    if np.any(resolved_transition & ~effective):
        raise ArmHoldEvaluationError(
            "transition_mask cannot select an excluded target element."
        )
    return resolved_logits, resolved_target, effective, resolved_transition


def arm_hold_window_evaluation(
    *,
    logits: Any,
    target: Any,
    valid_mask: Any,
    loss_mask: Any | None = None,
    transition_mask: Any | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Report masked classification and exact transition-keyframe metrics.

    ``exact_transition`` means that the predicted mode at an annotated
    transition keyframe equals its new target mode.  It does not claim that a
    transition detector reconstructed a unique Episode-level switch time.
    """

    resolved_threshold = float(threshold)
    if not math.isfinite(resolved_threshold) or not 0.0 < resolved_threshold < 1.0:
        raise ArmHoldEvaluationError("threshold must be finite and between 0 and 1.")
    resolved_logits, resolved_target, effective, resolved_transition = (
        _validate_arrays(
            logits=logits,
            target=target,
            valid_mask=valid_mask,
            loss_mask=loss_mask,
            transition_mask=transition_mask,
        )
    )
    probability = np.empty_like(resolved_logits)
    positive = resolved_logits >= 0.0
    probability[positive] = 1.0 / (1.0 + np.exp(-resolved_logits[positive]))
    negative = ~positive
    exp_value = np.exp(resolved_logits[negative])
    probability[negative] = exp_value / (1.0 + exp_value)
    prediction = probability >= resolved_threshold

    per_side: dict[str, Any] = {}
    for side_index, side_name in enumerate(ARM_HOLD_SIDE_NAMES):
        mask = effective[:, :, side_index]
        per_side[side_name] = _binary_report(
            probability=probability[:, :, side_index][mask],
            target=resolved_target[:, :, side_index][mask],
            prediction=prediction[:, :, side_index][mask],
        )
    macro_side = {
        metric: (
            None
            if not (
                values := [
                    float(per_side[side][metric])
                    for side in ARM_HOLD_SIDE_NAMES
                    if per_side[side][metric] is not None
                ]
            )
            else float(np.mean(values))
        )
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "pr_auc_average_precision",
        )
    }
    overall = _binary_report(
        probability=probability[effective],
        target=resolved_target[effective],
        prediction=prediction[effective],
    )

    exact_transition: dict[str, Any] = {}
    for side_index, side_name in enumerate(ARM_HOLD_SIDE_NAMES):
        for target_value, direction in ((1, "enter_hold"), (0, "exit_hold")):
            mask = (
                resolved_transition[:, :, side_index]
                & (resolved_target[:, :, side_index] == target_value)
            )
            support = int(np.count_nonzero(mask))
            correct = int(
                np.count_nonzero(prediction[:, :, side_index][mask] == bool(target_value))
            )
            exact_transition[f"{side_name}_{direction}"] = {
                "support": support,
                "correct": correct,
                "recall": _safe_ratio(correct, support),
            }

    pair_mask = np.all(effective, axis=-1)
    pair_confusion: dict[str, dict[str, int]] = {}
    pair_correct = 0
    pair_support = int(np.count_nonzero(pair_mask))
    if pair_support:
        binary_target = resolved_target == 1.0
        actual_code = (
            binary_target[:, :, 0].astype(np.int64) * 2
            + binary_target[:, :, 1].astype(np.int64)
        )
        predicted_code = (
            prediction[:, :, 0].astype(np.int64) * 2
            + prediction[:, :, 1].astype(np.int64)
        )
        pair_correct = int(np.count_nonzero(actual_code[pair_mask] == predicted_code[pair_mask]))
        for actual in range(4):
            actual_key = f"{actual:02b}"
            actual_mask = pair_mask & (actual_code == actual)
            if not np.any(actual_mask):
                continue
            pair_confusion[actual_key] = {
                f"{predicted:02b}": int(
                    np.count_nonzero(actual_mask & (predicted_code == predicted))
                )
                for predicted in range(4)
                if np.any(actual_mask & (predicted_code == predicted))
            }

    return {
        "schema": ARM_HOLD_EVALUATION_SCHEMA,
        "scope": "independent_admitted_windows_v1",
        "threshold": resolved_threshold,
        "batch_size": int(resolved_logits.shape[0]),
        "horizon": int(resolved_logits.shape[1]),
        "per_side": per_side,
        "macro_side": macro_side,
        "overall": overall,
        "exact_transition": exact_transition,
        "pair": {
            "support": pair_support,
            "correct": pair_correct,
            "exact_match_accuracy": _safe_ratio(pair_correct, pair_support),
            "confusion": pair_confusion,
        },
        "excluded_sequence_metrics": [
            "plus_minus_one_transition_recall",
            "signed_transition_timing",
            "mode_bouncing",
            "predicted_dwell",
        ],
    }


__all__ = [
    "ARM_HOLD_EVALUATION_SCHEMA",
    "ARM_HOLD_SIDE_NAMES",
    "ArmHoldEvaluationError",
    "arm_hold_window_evaluation",
]
