"""Offline free-running evaluation for the lightweight arm-hold baseline.

The training reader can expose the previous canonical collected label for
teacher forcing.  This module measures the separate case where a model must
consume its own preceding offset-zero prediction.  It is deliberately an
offline learner evaluation: it owns no RTC plan, policy admission, safety,
resolver, or target execution semantics.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from dataset_interface.arm_hold_windows import (
    ArmHoldWindowLocation,
    ArmHoldWindowSample,
)

from .baseline import (
    ARM_HOLD_DWELL_BUCKET_THRESHOLDS,
    ArmHoldBaseline,
    arm_hold_tensor_batch,
)
from .evaluation import arm_hold_window_evaluation
from .sequence_evaluation import arm_hold_fixed_lead_sequence_evaluation


ARM_HOLD_FREE_RUNNING_EVALUATION_SCHEMA = (
    "dexteleop.arm_hold_free_running_evaluation.v1"
)
ARM_HOLD_FREE_RUNNING_CONTEXT_MODES = frozenset(
    {
        "none",
        "previous_mode",
        "previous_dwell_bucket",
        "previous_mode_dwell",
        "previous_two_dwell_bucket",
        "previous_two_mode_dwell",
    }
)
_ARM_HOLD_DWELL_AWARE_CONTEXT_MODES = frozenset(
    {
        "previous_dwell_bucket",
        "previous_mode_dwell",
        "previous_two_dwell_bucket",
        "previous_two_mode_dwell",
    }
)
_ARM_HOLD_TWO_STEP_CONTEXT_MODES = frozenset(
    {"previous_two_dwell_bucket", "previous_two_mode_dwell"}
)


class ArmHoldFreeRunningError(ValueError):
    """Raised when an offline rollout cannot preserve causal ordering."""


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _integer_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "minimum": int(np.min(array)),
        "maximum": int(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _location_key(location: ArmHoldWindowLocation) -> tuple[str, int, int, str]:
    return (
        location.source_dataset_id,
        location.source_episode_index,
        location.start_index,
        location.episode_capture_identity_hash,
    )


def _metadata_location_key(
    metadata: Mapping[str, Any],
) -> tuple[object, object, object, object]:
    return (
        metadata.get("source_dataset_id"),
        metadata.get("source_episode_index"),
        metadata.get("start_index"),
        metadata.get("episode_capture_identity_hash"),
    )


def _same_rollout_segment(
    previous: ArmHoldWindowLocation,
    current: ArmHoldWindowLocation,
) -> bool:
    return (
        previous.source_dataset_id == current.source_dataset_id
        and previous.source_episode_index == current.source_episode_index
        and previous.episode_capture_identity_hash
        == current.episode_capture_identity_hash
        and previous.continuity_segment_index == current.continuity_segment_index
        and current.start_index == previous.start_index + 1
    )


def _validate_sample_binding(
    sample: ArmHoldWindowSample,
    location: ArmHoldWindowLocation,
) -> None:
    _validate_metadata_binding(sample.metadata, location)


def _validate_metadata_binding(
    metadata: Mapping[str, Any],
    location: ArmHoldWindowLocation,
) -> None:
    expected = {
        "source_dataset_id": location.source_dataset_id,
        "episode_capture_identity_hash": location.episode_capture_identity_hash,
        "start_index": location.start_index,
        "continuity_segment_index": location.continuity_segment_index,
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ArmHoldFreeRunningError(
            "Sample metadata does not match its free-running location."
        )


def _numpy(value: Any) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate)


def _resolve_threshold(threshold: float) -> float:
    resolved = float(threshold)
    if not math.isfinite(resolved) or not 0.0 < resolved < 1.0:
        raise ArmHoldFreeRunningError(
            "threshold must be finite and strictly between zero and one."
        )
    return resolved


def _rollout_segments(
    locations: Sequence[ArmHoldWindowLocation],
) -> tuple[tuple[int, ...], ...]:
    resolved = tuple(locations)
    if not resolved:
        raise ArmHoldFreeRunningError("Free-running evaluation requires locations.")
    keys = tuple(_location_key(item) for item in resolved)
    if keys != tuple(sorted(keys)):
        raise ArmHoldFreeRunningError(
            "Free-running locations must retain canonical inventory order."
        )
    if len(keys) != len(set(keys)):
        raise ArmHoldFreeRunningError("Free-running locations contain duplicates.")
    segments: list[list[int]] = [[0]]
    for index in range(1, len(resolved)):
        if _same_rollout_segment(resolved[index - 1], resolved[index]):
            segments[-1].append(index)
        else:
            segments.append([index])
    return tuple(tuple(segment) for segment in segments)


def _error_dynamics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    segments: Sequence[Sequence[int]],
    locations: Sequence[ArmHoldWindowLocation],
) -> dict[str, Any]:
    per_side: dict[str, Any] = {}
    for side_index, side_name in enumerate(("left", "right")):
        first_error_offsets: list[int] = []
        error_run_steps: list[int] = []
        recovered_runs = 0
        constant_segments = 0
        nontrivial_constant_segments = 0
        for segment in segments:
            indices = np.asarray(segment, dtype=np.int64)
            side_prediction = prediction[indices, side_index]
            side_target = target[indices, side_index]
            error = side_prediction != side_target
            error_indices = np.flatnonzero(error)
            if error_indices.size:
                first_error_offsets.append(int(error_indices[0]))
            if np.all(side_prediction == side_prediction[0]):
                constant_segments += 1
                if len(segment) > 1:
                    nontrivial_constant_segments += 1
            start = 0
            while start < len(segment):
                if not bool(error[start]):
                    start += 1
                    continue
                stop = start + 1
                while stop < len(segment) and bool(error[stop]):
                    stop += 1
                error_run_steps.append(stop - start)
                if stop < len(segment):
                    recovered_runs += 1
                start = stop

        episode_predictions: dict[tuple[str, str], list[bool]] = {}
        for index, location in enumerate(locations):
            episode_predictions.setdefault(
                (
                    location.source_dataset_id,
                    location.episode_capture_identity_hash,
                ),
                [],
            ).append(bool(prediction[index, side_index]))
        constant_episodes = sum(
            bool(values) and all(item == values[0] for item in values)
            for values in episode_predictions.values()
        )
        error_run_count = len(error_run_steps)
        per_side[side_name] = {
            "point_count": int(prediction.shape[0]),
            "predicted_positive_rate": float(
                np.mean(prediction[:, side_index], dtype=np.float64)
            ),
            "target_positive_rate": float(
                np.mean(target[:, side_index], dtype=np.float64)
            ),
            "rollout_segment_count": len(segments),
            "segments_with_error_count": len(first_error_offsets),
            "first_error_offset_steps": _integer_summary(first_error_offsets),
            "error_run_steps": _integer_summary(error_run_steps),
            "recovered_error_run_count": recovered_runs,
            "unrecovered_error_run_count": error_run_count - recovered_runs,
            "error_run_recovery_rate": _safe_ratio(
                recovered_runs, error_run_count
            ),
            "constant_prediction_segment_count": constant_segments,
            "nontrivial_constant_prediction_segment_count": (
                nontrivial_constant_segments
            ),
            "constant_prediction_episode_count": int(constant_episodes),
            "episode_count": len(episode_predictions),
        }
    return per_side


_DWELL_BUCKET_LABELS = {
    1: "1",
    2: "2",
    3: "3",
    4: "4-7",
    5: "8-15",
    6: "16+",
}


def _dwell_bucket_indices(
    dwell_steps: np.ndarray,
    available: np.ndarray,
) -> np.ndarray:
    if (
        dwell_steps.shape != available.shape
        or dwell_steps.ndim != 2
        or dwell_steps.shape[1] != 2
        or dwell_steps.dtype != np.int64
        or available.dtype != np.bool_
        or bool(np.any(dwell_steps < 0))
        or not np.array_equal(dwell_steps > 0, available)
    ):
        raise ArmHoldFreeRunningError("Dwell diagnostic history is invalid.")
    bucket = available.astype(np.int64)
    for threshold in ARM_HOLD_DWELL_BUCKET_THRESHOLDS:
        bucket += (dwell_steps >= threshold).astype(np.int64)
    return bucket


def _causal_mode_dwell_history(
    values: np.ndarray,
    segments: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 2 or values.dtype != np.bool_:
        raise ArmHoldFreeRunningError(
            "Causal dwell reference values must be boolean [N,2]."
        )
    previous = np.zeros_like(values, dtype=bool)
    available = np.zeros_like(values, dtype=bool)
    dwell = np.zeros_like(values, dtype=np.int64)
    for segment in segments:
        segment_previous = np.zeros(2, dtype=bool)
        segment_available = np.zeros(2, dtype=bool)
        segment_dwell = np.zeros(2, dtype=np.int64)
        for index in segment:
            previous[index] = segment_previous
            available[index] = segment_available
            dwell[index] = segment_dwell
            current = values[index]
            unchanged = segment_available & (segment_previous == current)
            segment_dwell = np.where(unchanged, segment_dwell + 1, 1)
            segment_previous = current
            segment_available[:] = True
    return previous, available, dwell


def _mode_dwell_event_rows(
    *,
    previous: np.ndarray,
    available: np.ndarray,
    dwell: np.ndarray,
    events: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    buckets = _dwell_bucket_indices(dwell, available)
    result: dict[str, Any] = {}
    for side_index, side_name in enumerate(("left", "right")):
        rows: dict[str, dict[str, int | float | None]] = {}
        for mode_value, mode_name in ((False, "teleop"), (True, "fixed_hold")):
            for bucket_index, bucket_label in _DWELL_BUCKET_LABELS.items():
                selected = (
                    available[:, side_index]
                    & (previous[:, side_index] == mode_value)
                    & (buckets[:, side_index] == bucket_index)
                )
                support = int(np.count_nonzero(selected))
                row: dict[str, int | float | None] = {"support": support}
                for event_name, event in events.items():
                    if event.shape != previous.shape or event.dtype != np.bool_:
                        raise ArmHoldFreeRunningError(
                            f"Dwell diagnostic event {event_name!r} is invalid."
                        )
                    count = int(np.count_nonzero(event[:, side_index] & selected))
                    row[f"{event_name}_count"] = count
                    row[f"{event_name}_rate"] = _safe_ratio(count, support)
                rows[f"{mode_name}:{bucket_label}"] = row
        feedback_support = int(np.count_nonzero(available[:, side_index]))
        saturated_support = int(
            np.count_nonzero(available[:, side_index] & (buckets[:, side_index] == 6))
        )
        result[side_name] = {
            "support": feedback_support,
            "saturated_16_plus_support": saturated_support,
            "saturated_16_plus_rate": _safe_ratio(
                saturated_support, feedback_support
            ),
            "by_previous_mode_and_dwell_bucket": rows,
        }
    return result


def _joint_distribution_total_variation(
    left: Mapping[str, Mapping[str, int | float | None]],
    right: Mapping[str, Mapping[str, int | float | None]],
) -> float | None:
    left_support = sum(int(row["support"]) for row in left.values())
    right_support = sum(int(row["support"]) for row in right.values())
    if left_support == 0 or right_support == 0:
        return None
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(
            int(left.get(key, {}).get("support", 0)) / left_support
            - int(right.get(key, {}).get("support", 0)) / right_support
        )
        for key in keys
    )


def _dwell_dynamics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    segments: Sequence[Sequence[int]],
    feedback_previous: np.ndarray,
    feedback_available: np.ndarray,
    feedback_dwell: np.ndarray,
) -> dict[str, Any]:
    canonical_previous, canonical_available, canonical_dwell = (
        _causal_mode_dwell_history(target, segments)
    )
    predicted_feedback = _mode_dwell_event_rows(
        previous=feedback_previous,
        available=feedback_available,
        dwell=feedback_dwell,
        events={
            "predicted_switch": prediction != feedback_previous,
            "target_disagreement": target != feedback_previous,
        },
    )
    canonical_reference = _mode_dwell_event_rows(
        previous=canonical_previous,
        available=canonical_available,
        dwell=canonical_dwell,
        events={
            "target_switch": target != canonical_previous,
            "prediction_disagreement": prediction != canonical_previous,
        },
    )
    total_variation: dict[str, float | None] = {}
    for side_name in ("left", "right"):
        total_variation[side_name] = _joint_distribution_total_variation(
            predicted_feedback[side_name][
                "by_previous_mode_and_dwell_bucket"
            ],
            canonical_reference[side_name][
                "by_previous_mode_and_dwell_bucket"
            ],
        )
    return {
        "bucket_semantics": dict(_DWELL_BUCKET_LABELS),
        "predicted_feedback": {
            "history_source": (
                "previous_predicted_offset_zero_mode_and_consecutive_dwell_v1"
            ),
            "per_side": predicted_feedback,
        },
        "canonical_reference": {
            "history_source": (
                "previous_canonical_offset_zero_target_mode_and_consecutive_dwell_v1"
            ),
            "evaluation_only_not_model_input": True,
            "per_side": canonical_reference,
        },
        "joint_mode_dwell_distribution_total_variation": total_variation,
    }


def _build_evaluation_report(
    *,
    logits: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    loss: np.ndarray,
    transition: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    locations: Sequence[ArmHoldWindowLocation],
    segments: Sequence[Sequence[int]],
    context_mode: str,
    threshold: float,
    transition_tolerance_steps: int,
    feedback_previous: np.ndarray | None = None,
    feedback_available: np.ndarray | None = None,
    feedback_dwell: np.ndarray | None = None,
) -> dict[str, Any]:
    if logits.shape[0] != len(locations) or len(metadata) != len(locations):
        raise ArmHoldFreeRunningError(
            "Free-running arrays, metadata, and locations must have equal length."
        )
    window_report = arm_hold_window_evaluation(
        logits=logits,
        target=target,
        valid_mask=valid,
        loss_mask=loss,
        transition_mask=transition,
        threshold=threshold,
    )
    sequence_report = arm_hold_fixed_lead_sequence_evaluation(
        logits=logits,
        target=target,
        valid_mask=valid,
        loss_mask=loss,
        transition_mask=transition,
        metadata=metadata,
        lead_offset=0,
        threshold=threshold,
        transition_tolerance_steps=transition_tolerance_steps,
        debug=False,
    )
    logit_threshold = math.log(threshold / (1.0 - threshold))
    prediction = logits[:, 0] >= logit_threshold
    target_bool = target[:, 0] == 1.0
    history_enabled = context_mode != "none"
    feedback_semantics = {
        "none": "none",
        "previous_mode": "previous_predicted_offset_zero_mode_v1",
        "previous_dwell_bucket": (
            "previous_predicted_offset_zero_consecutive_dwell_bucket_v1"
        ),
        "previous_mode_dwell": (
            "previous_predicted_offset_zero_mode_and_consecutive_dwell_v1"
        ),
        "previous_two_dwell_bucket": (
            "previous_two_predicted_offset_zero_consecutive_dwell_buckets_v1"
        ),
        "previous_two_mode_dwell": (
            "previous_two_predicted_offset_zero_modes_and_consecutive_dwell_v1"
        ),
    }.get(context_mode)
    if feedback_semantics is None:
        raise ArmHoldFreeRunningError(
            f"Unsupported report context mode {context_mode!r}."
        )
    report = {
        "schema": ARM_HOLD_FREE_RUNNING_EVALUATION_SCHEMA,
        "scope": "offline_predicted_history_offset_zero_v1",
        "context_mode": context_mode,
        "feedback_semantics": feedback_semantics,
        "bootstrap_semantics": (
            "unavailable_on_episode_continuity_or_admitted_window_gap_v1"
        ),
        "threshold": threshold,
        "window_count": len(locations),
        "rollout_segment_count": len(segments),
        "bootstrap_context_window_count": (
            len(segments) if history_enabled else 0
        ),
        "feedback_context_window_count": (
            len(locations) - len(segments) if history_enabled else 0
        ),
        "window_evaluation": window_report,
        "sequence_evaluation": sequence_report,
        "error_dynamics": _error_dynamics(
            prediction=prediction,
            target=target_bool,
            segments=segments,
            locations=locations,
        ),
    }
    feedback_values = (
        feedback_previous,
        feedback_available,
        feedback_dwell,
    )
    if context_mode in _ARM_HOLD_DWELL_AWARE_CONTEXT_MODES:
        if any(value is None for value in feedback_values):
            raise ArmHoldFreeRunningError(
                "Dwell-aware evaluation requires the exact feedback history."
            )
        assert feedback_previous is not None
        assert feedback_available is not None
        assert feedback_dwell is not None
        report["dwell_dynamics"] = _dwell_dynamics(
            prediction=prediction,
            target=target_bool,
            segments=segments,
            feedback_previous=feedback_previous,
            feedback_available=feedback_available,
            feedback_dwell=feedback_dwell,
        )
    elif any(value is not None for value in feedback_values):
        raise ArmHoldFreeRunningError(
            "Feedback dwell history is only valid for dwell-aware context modes."
        )
    return report


def arm_hold_offset_zero_evaluation(
    *,
    logits: Any,
    target: Any,
    valid_mask: Any,
    loss_mask: Any,
    transition_mask: Any,
    metadata: Sequence[Mapping[str, Any]],
    locations: Sequence[ArmHoldWindowLocation],
    threshold: float = 0.5,
    transition_tolerance_steps: int = 1,
) -> dict[str, Any]:
    """Evaluate a context-free offset-zero timeline without another model pass.

    This is equivalent to free-running evaluation only when the model has no
    autoregressive label context.  It consumes logits already produced by the
    natural validation pass, avoiding a second image/video read.
    """

    resolved_threshold = _resolve_threshold(threshold)
    resolved_locations = tuple(locations)
    resolved_metadata = tuple(metadata)
    segments = _rollout_segments(resolved_locations)
    arrays = {
        "logits": _numpy(logits),
        "target": _numpy(target),
        "valid": _numpy(valid_mask),
        "loss": _numpy(loss_mask),
        "transition": _numpy(transition_mask),
    }
    expected_shape = arrays["logits"].shape
    if (
        len(expected_shape) != 3
        or expected_shape[1] < 1
        or expected_shape[2] != 2
        or any(value.shape != expected_shape for value in arrays.values())
    ):
        raise ArmHoldFreeRunningError(
            "Offset-zero source arrays must share shape [N,H,2] with H >= 1."
        )
    if arrays["valid"].dtype != np.bool_ or arrays["loss"].dtype != np.bool_:
        raise ArmHoldFreeRunningError(
            "Offset-zero valid and loss masks must be boolean."
        )
    if arrays["transition"].dtype != np.bool_:
        raise ArmHoldFreeRunningError(
            "Offset-zero transition mask must be boolean."
        )
    if not np.all(arrays["valid"][:, 0] & arrays["loss"][:, 0]):
        raise ArmHoldFreeRunningError(
            "Every admitted offset-zero row must be valid and selected for loss."
        )
    if expected_shape[0] != len(resolved_locations) or len(
        resolved_metadata
    ) != len(resolved_locations):
        raise ArmHoldFreeRunningError(
            "Offset-zero arrays, metadata, and locations must have equal length."
        )
    row_by_key: dict[tuple[object, object, object, object], int] = {}
    for row_index, item in enumerate(resolved_metadata):
        key = _metadata_location_key(item)
        if key in row_by_key:
            raise ArmHoldFreeRunningError(
                "Offset-zero metadata contains duplicate window identities."
            )
        row_by_key[key] = row_index
    location_keys = tuple(_location_key(item) for item in resolved_locations)
    if set(row_by_key) != set(location_keys):
        raise ArmHoldFreeRunningError(
            "Offset-zero metadata identities do not match the locations."
        )
    row_order = np.asarray(
        [row_by_key[key] for key in location_keys], dtype=np.int64
    )
    arrays = {name: value[row_order] for name, value in arrays.items()}
    resolved_metadata = tuple(resolved_metadata[index] for index in row_order)
    for item, location in zip(resolved_metadata, resolved_locations, strict=True):
        _validate_metadata_binding(item, location)
    return _build_evaluation_report(
        logits=arrays["logits"][:, :1],
        target=arrays["target"][:, :1],
        valid=arrays["valid"][:, :1],
        loss=arrays["loss"][:, :1],
        transition=arrays["transition"][:, :1],
        metadata=resolved_metadata,
        locations=resolved_locations,
        segments=segments,
        context_mode="none",
        threshold=resolved_threshold,
        transition_tolerance_steps=transition_tolerance_steps,
    )


def arm_hold_free_running_evaluation(
    *,
    model: ArmHoldBaseline,
    locations: Sequence[ArmHoldWindowLocation],
    sample_at: Callable[[int], ArmHoldWindowSample],
    device: torch.device | str,
    threshold: float = 0.5,
    transition_tolerance_steps: int = 1,
    inference_batch_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate offset-zero predictions with predicted mode/dwell feedback.

    Consecutive admitted windows in the same Episode and continuity segment
    share feedback state.  Episode changes, continuity changes, and missing
    window starts all begin a new unavailable-history bootstrap.  The model is
    never given a future or teacher-forced label by this function.
    """

    resolved_threshold = _resolve_threshold(threshold)
    resolved_inference_batch_size = (
        len(locations)
        if inference_batch_size is None
        else int(inference_batch_size)
    )
    if resolved_inference_batch_size <= 0:
        raise ArmHoldFreeRunningError(
            "inference_batch_size must be positive when provided."
        )
    context_mode = str(model.config.label_context_mode)
    if context_mode not in ARM_HOLD_FREE_RUNNING_CONTEXT_MODES:
        raise ArmHoldFreeRunningError(
            "Unsupported free-running label context mode "
            f"{context_mode!r}; expected one of "
            f"{sorted(ARM_HOLD_FREE_RUNNING_CONTEXT_MODES)!r}."
        )
    resolved_locations = tuple(locations)
    segments = _rollout_segments(resolved_locations)
    segment_previous = np.zeros((len(segments), 2), dtype=np.float32)
    segment_available = np.zeros((len(segments), 2), dtype=bool)
    segment_dwell = np.zeros((len(segments), 2), dtype=np.int64)
    segment_sequence_previous = np.zeros(
        (len(segments), 2, 2), dtype=np.float32
    )
    segment_sequence_available = np.zeros(
        (len(segments), 2, 2), dtype=bool
    )
    segment_sequence_dwell = np.zeros(
        (len(segments), 2, 2), dtype=np.int64
    )
    feedback_previous = np.zeros((len(resolved_locations), 2), dtype=bool)
    feedback_available = np.zeros((len(resolved_locations), 2), dtype=bool)
    feedback_dwell = np.zeros((len(resolved_locations), 2), dtype=np.int64)
    logits = np.empty((len(resolved_locations), 1, 2), dtype=np.float32)
    target = np.empty_like(logits)
    valid = np.ones_like(logits, dtype=bool)
    loss = np.ones_like(logits, dtype=bool)
    transition = np.empty_like(logits, dtype=bool)
    metadata: list[Mapping[str, Any] | None] = [None] * len(resolved_locations)
    resolved_device = torch.device(device)
    preloaded_samples: tuple[ArmHoldWindowSample, ...] | None = None
    if str(model.config.input_mode) == "state_only":
        loaded: list[ArmHoldWindowSample] = []
        for index, location in enumerate(resolved_locations):
            sample = replace(
                sample_at(index),
                arm_hold_history_context=None,
                arm_hold_history_sequence_context=None,
            )
            _validate_sample_binding(sample, location)
            loaded.append(sample)
        preloaded_samples = tuple(loaded)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for cohort_start in range(
                0, len(segments), resolved_inference_batch_size
            ):
                cohort_stop = min(
                    cohort_start + resolved_inference_batch_size,
                    len(segments),
                )
                cohort = tuple(range(cohort_start, cohort_stop))
                maximum_length = max(len(segments[index]) for index in cohort)
                for offset in range(maximum_length):
                    active = tuple(
                        segment_index
                        for segment_index in cohort
                        if offset < len(segments[segment_index])
                    )
                    inventory_indices = tuple(
                        segments[segment_index][offset]
                        for segment_index in active
                    )
                    if context_mode in _ARM_HOLD_DWELL_AWARE_CONTEXT_MODES:
                        active_array = np.asarray(active, dtype=np.int64)
                        inventory_array = np.asarray(
                            inventory_indices, dtype=np.int64
                        )
                        feedback_previous[inventory_array] = segment_previous[
                            active_array
                        ].astype(bool)
                        feedback_available[inventory_array] = segment_available[
                            active_array
                        ]
                        feedback_dwell[inventory_array] = segment_dwell[
                            active_array
                        ]
                    if preloaded_samples is not None:
                        samples = tuple(
                            preloaded_samples[index]
                            for index in inventory_indices
                        )
                    else:
                        samples = tuple(
                            replace(
                                sample_at(index),
                                arm_hold_history_context=None,
                                arm_hold_history_sequence_context=None,
                            )
                            for index in inventory_indices
                        )
                        for sample, inventory_index in zip(
                            samples, inventory_indices, strict=True
                        ):
                            _validate_sample_binding(
                                sample, resolved_locations[inventory_index]
                            )
                    batch = arm_hold_tensor_batch(samples).to(resolved_device)
                    if context_mode == "none":
                        batch_logits = model.forward_batch(batch)
                    elif context_mode in _ARM_HOLD_TWO_STEP_CONTEXT_MODES:
                        active_indices = np.asarray(active)
                        batch_logits = model(
                            front_rgb=batch.front_rgb,
                            observation_state=batch.observation_state,
                            history_sequence_previous_mode=torch.from_numpy(
                                segment_sequence_previous[active_indices]
                            ).to(resolved_device),
                            history_sequence_available_mask=torch.from_numpy(
                                segment_sequence_available[active_indices]
                            ).to(resolved_device),
                            history_sequence_dwell_steps=torch.from_numpy(
                                segment_sequence_dwell[active_indices]
                            ).to(resolved_device),
                        )
                    else:
                        batch_logits = model(
                            front_rgb=batch.front_rgb,
                            observation_state=batch.observation_state,
                            history_previous_mode=torch.from_numpy(
                                segment_previous[np.asarray(active)]
                            ).to(resolved_device),
                            history_available_mask=torch.from_numpy(
                                segment_available[np.asarray(active)]
                            ).to(resolved_device),
                            history_dwell_steps=torch.from_numpy(
                                segment_dwell[np.asarray(active)]
                            ).to(resolved_device),
                        )
                    if (
                        batch_logits.ndim != 3
                        or batch_logits.shape[0] != len(active)
                        or batch_logits.shape[1] < 1
                        or batch_logits.shape[2] != 2
                        or not bool(
                            torch.all(torch.isfinite(batch_logits[:, 0]))
                        )
                    ):
                        raise ArmHoldFreeRunningError(
                            "Model returned invalid offset-zero arm-hold logits."
                        )
                    offset_zero = (
                        batch_logits[:, 0].detach().cpu().to(torch.float32)
                    )
                    probabilities = torch.sigmoid(offset_zero)
                    predictions = (probabilities >= resolved_threshold).numpy()
                    for batch_index, (
                        segment_index,
                        inventory_index,
                    ) in enumerate(zip(active, inventory_indices, strict=True)):
                        sample = samples[batch_index]
                        logits[inventory_index, 0] = offset_zero[
                            batch_index
                        ].numpy()
                        target[inventory_index, 0] = sample.arm_hold_target[0]
                        transition[inventory_index, 0] = (
                            sample.arm_hold_transition_mask[0]
                        )
                        metadata[inventory_index] = dict(sample.metadata)
                        prediction = predictions[batch_index]
                        unchanged = segment_available[segment_index] & (
                            segment_previous[segment_index] == prediction
                        )
                        next_dwell = np.where(
                            unchanged,
                            segment_dwell[segment_index] + 1,
                            1,
                        )
                        segment_sequence_previous[segment_index, 0] = (
                            segment_sequence_previous[segment_index, 1]
                        )
                        segment_sequence_available[segment_index, 0] = (
                            segment_sequence_available[segment_index, 1]
                        )
                        segment_sequence_dwell[segment_index, 0] = (
                            segment_sequence_dwell[segment_index, 1]
                        )
                        segment_sequence_previous[segment_index, 1] = (
                            prediction.astype(np.float32, copy=False)
                        )
                        segment_sequence_available[segment_index, 1] = True
                        segment_sequence_dwell[segment_index, 1] = next_dwell
                        segment_dwell[segment_index] = next_dwell
                        segment_previous[segment_index] = prediction.astype(
                            np.float32, copy=False
                        )
                        segment_available[segment_index] = True
    finally:
        model.train(was_training)

    if any(item is None for item in metadata):
        raise ArmHoldFreeRunningError("Free-running rollout omitted a location.")
    resolved_metadata = tuple(item for item in metadata if item is not None)
    report = _build_evaluation_report(
        logits=logits,
        target=target,
        valid=valid,
        loss=loss,
        transition=transition,
        metadata=resolved_metadata,
        locations=resolved_locations,
        segments=segments,
        context_mode=context_mode,
        threshold=resolved_threshold,
        transition_tolerance_steps=transition_tolerance_steps,
        feedback_previous=(
            feedback_previous
            if context_mode in _ARM_HOLD_DWELL_AWARE_CONTEXT_MODES
            else None
        ),
        feedback_available=(
            feedback_available
            if context_mode in _ARM_HOLD_DWELL_AWARE_CONTEXT_MODES
            else None
        ),
        feedback_dwell=(
            feedback_dwell
            if context_mode in _ARM_HOLD_DWELL_AWARE_CONTEXT_MODES
            else None
        ),
    )
    if inference_batch_size is not None:
        report["inference_batch_size"] = resolved_inference_batch_size
        report["inference_segment_cohort_count"] = int(
            math.ceil(len(segments) / resolved_inference_batch_size)
        )
    return report


__all__ = [
    "ARM_HOLD_FREE_RUNNING_CONTEXT_MODES",
    "ARM_HOLD_FREE_RUNNING_EVALUATION_SCHEMA",
    "ArmHoldFreeRunningError",
    "arm_hold_free_running_evaluation",
    "arm_hold_offset_zero_evaluation",
]
