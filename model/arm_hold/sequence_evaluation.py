"""Arm-hold sequence metrics on two explicit, non-interchangeable timelines.

Fixed-lead evaluation follows one horizon column across rolling observations.
Chunk evaluation follows the horizon inside each independently produced plan.
The module never averages those timelines and never claims ownership of a
deployment scheduling policy.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np


ARM_HOLD_FIXED_LEAD_SEQUENCE_SCHEMA = (
    "dexteleop.arm_hold_fixed_lead_sequence_evaluation.v1"
)
ARM_HOLD_CHUNK_SEQUENCE_SCHEMA = "dexteleop.arm_hold_chunk_sequence_evaluation.v1"
ARM_HOLD_SEQUENCE_SIDES = ("left", "right")


class ArmHoldSequenceEvaluationError(ValueError):
    """Raised when windows cannot form an unambiguous sequence timeline."""


def _numpy(value: Any) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    return np.asarray(candidate)


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ArmHoldSequenceEvaluationError(f"{label} must be an integer.")
    result = int(value)
    if result < 0:
        raise ArmHoldSequenceEvaluationError(f"{label} must be nonnegative.")
    return result


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": int(np.min(array)),
        "maximum": int(np.max(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
    }


def _run_lengths(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    points = np.concatenate(([0], boundaries, [values.size]))
    return [int(stop - start) for start, stop in zip(points[:-1], points[1:])]


def _match_events(
    true_positions: Sequence[int],
    predicted_positions: Sequence[int],
    *,
    tolerance: int,
) -> tuple[list[int], set[int]]:
    truths = tuple(sorted(int(value) for value in true_positions))
    predictions = tuple(sorted(int(value) for value in predicted_positions))
    # Dynamic programming maximizes one-to-one cardinality first, minimizes
    # total absolute timing error second, then uses positions/offsets only as
    # deterministic tie breakers.  A local-nearest greedy rule can lose a
    # legal match when two nearby events compete for one prediction.
    table: list[list[tuple[tuple[int, ...], tuple[int, ...]]]] = [
        [((), ()) for _ in range(len(predictions) + 1)]
        for _ in range(len(truths) + 1)
    ]

    def ranking(
        candidate: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> tuple[Any, ...]:
        offsets, matched = candidate
        return (
            -len(offsets),
            sum(abs(value) for value in offsets),
            matched,
            offsets,
        )

    for truth_index in range(len(truths) - 1, -1, -1):
        for prediction_index in range(len(predictions) - 1, -1, -1):
            candidates = [
                table[truth_index + 1][prediction_index],
                table[truth_index][prediction_index + 1],
            ]
            offset = predictions[prediction_index] - truths[truth_index]
            if abs(offset) <= tolerance:
                tail_offsets, tail_predictions = table[truth_index + 1][
                    prediction_index + 1
                ]
                candidates.append(
                    (
                        (offset, *tail_offsets),
                        (predictions[prediction_index], *tail_predictions),
                    )
                )
            table[truth_index][prediction_index] = min(candidates, key=ranking)
    offsets, matched_predictions = table[0][0]
    return list(offsets), set(matched_predictions)


def _new_side_state() -> dict[str, Any]:
    return {
        "point_count": 0,
        "segment_count": 0,
        "stable_boundary_count": 0,
        "stable_false_switch_count": 0,
        "predicted_transition_count": 0,
        "matched_predicted_transition_count": 0,
        "unsupported_boundary_transition_count": 0,
        "dwell_steps": [],
        "interior_dwell_steps": [],
        "direction": {
            "enter_hold": {"support": 0, "predicted_count": 0, "offsets": []},
            "exit_hold": {"support": 0, "predicted_count": 0, "offsets": []},
        },
        "segment_debug": [],
    }


def _accumulate_sequence_segment(
    state: dict[str, Any],
    *,
    truth: np.ndarray,
    predicted: np.ndarray,
    canonical_transition: np.ndarray,
    positions: np.ndarray,
    tolerance: int,
    debug_payload: Mapping[str, Any] | None,
) -> None:
    length = int(truth.size)
    if (
        length == 0
        or predicted.shape != truth.shape
        or canonical_transition.shape != truth.shape
        or positions.shape != truth.shape
    ):
        raise ArmHoldSequenceEvaluationError("Sequence segment arrays are invalid.")
    state["point_count"] += length
    state["segment_count"] += 1
    if canonical_transition[0]:
        state["unsupported_boundary_transition_count"] += 1
    if length > 1:
        derived_transition = truth[1:] != truth[:-1]
        if not np.array_equal(canonical_transition[1:], derived_transition):
            raise ArmHoldSequenceEvaluationError(
                "Canonical transition mask conflicts with sequence targets."
            )
        predicted_transition = predicted[1:] != predicted[:-1]
        stable = ~derived_transition
        state["stable_boundary_count"] += int(np.count_nonzero(stable))
        state["stable_false_switch_count"] += int(
            np.count_nonzero(stable & predicted_transition)
        )
        state["predicted_transition_count"] += int(
            np.count_nonzero(predicted_transition)
        )
        true_events = {"enter_hold": [], "exit_hold": []}
        predicted_events = {"enter_hold": [], "exit_hold": []}
        for offset, changed in enumerate(derived_transition, start=1):
            if changed:
                direction = "enter_hold" if truth[offset] else "exit_hold"
                true_events[direction].append(int(positions[offset]))
        for offset, changed in enumerate(predicted_transition, start=1):
            if changed:
                direction = "enter_hold" if predicted[offset] else "exit_hold"
                predicted_events[direction].append(int(positions[offset]))
        for direction in ("enter_hold", "exit_hold"):
            offsets, matched_predictions = _match_events(
                true_events[direction],
                predicted_events[direction],
                tolerance=tolerance,
            )
            direction_state = state["direction"][direction]
            direction_state["support"] += len(true_events[direction])
            direction_state["predicted_count"] += len(predicted_events[direction])
            direction_state["offsets"].extend(offsets)
            state["matched_predicted_transition_count"] += len(
                matched_predictions
            )
    dwell = _run_lengths(predicted)
    state["dwell_steps"].extend(dwell)
    state["interior_dwell_steps"].extend(dwell[1:-1])
    if debug_payload is not None:
        state["segment_debug"].append(
            {
                **dict(debug_payload),
                "start_position": int(positions[0]),
                "stop_position": int(positions[-1]),
                "point_count": length,
                "predicted_dwell_steps": dwell,
            }
        )


def _finalize_side_state(
    state: Mapping[str, Any],
    *,
    segment_count_field: str,
    debug: bool,
) -> dict[str, Any]:
    directions: dict[str, Any] = {}
    for direction in ("enter_hold", "exit_hold"):
        direction_state = state["direction"][direction]
        offsets = direction_state["offsets"]
        support = direction_state["support"]
        exact = sum(offset == 0 for offset in offsets)
        directions[direction] = {
            "support": support,
            "predicted_count": direction_state["predicted_count"],
            "matched_within_tolerance": len(offsets),
            "recall_within_tolerance": _ratio(len(offsets), support),
            "matched_exact": exact,
            "recall_exact": _ratio(exact, support),
            "signed_timing_offset_steps": _summary(offsets),
        }
    dwell_steps = state["dwell_steps"]
    interior_dwell_steps = state["interior_dwell_steps"]
    interior_segment_count = len(interior_dwell_steps)
    single_frame_count = sum(value == 1 for value in interior_dwell_steps)
    payload: dict[str, Any] = {
        "point_count": state["point_count"],
        segment_count_field: state["segment_count"],
        "unsupported_boundary_transition_count": state[
            "unsupported_boundary_transition_count"
        ],
        "transition": directions,
        "predicted_transition_count": state["predicted_transition_count"],
        "unmatched_predicted_transition_count": (
            state["predicted_transition_count"]
            - state["matched_predicted_transition_count"]
        ),
        "unmatched_predicted_transition_rate": _ratio(
            state["predicted_transition_count"]
            - state["matched_predicted_transition_count"],
            state["predicted_transition_count"],
        ),
        "stable_boundary_count": state["stable_boundary_count"],
        "stable_false_switch_count": state["stable_false_switch_count"],
        "stable_false_switch_rate": _ratio(
            state["stable_false_switch_count"],
            state["stable_boundary_count"],
        ),
        "predicted_segment_count": len(dwell_steps),
        "interior_predicted_segment_count": interior_segment_count,
        "single_frame_interior_predicted_segment_count": single_frame_count,
        "single_frame_interior_predicted_segment_rate": _ratio(
            single_frame_count, interior_segment_count
        ),
        "observed_predicted_run_steps": _summary(dwell_steps),
        "interior_predicted_dwell_steps": _summary(interior_dwell_steps),
    }
    if debug:
        payload["segments"] = state["segment_debug"]
    return payload


def _sequence_arrays(
    *,
    logits: Any,
    target: Any,
    valid_mask: Any,
    loss_mask: Any,
    transition_mask: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_logits = _numpy(logits).astype(np.float64, copy=False)
    resolved_target = _numpy(target).astype(np.float64, copy=False)
    resolved_valid = _numpy(valid_mask)
    resolved_loss = _numpy(loss_mask)
    resolved_transition = _numpy(transition_mask)
    if resolved_logits.ndim != 3 or resolved_logits.shape[-1] != 2:
        raise ArmHoldSequenceEvaluationError("logits must have shape [B,H,2].")
    for name, value in (
        ("target", resolved_target),
        ("valid_mask", resolved_valid),
        ("loss_mask", resolved_loss),
        ("transition_mask", resolved_transition),
    ):
        if value.shape != resolved_logits.shape:
            raise ArmHoldSequenceEvaluationError(f"{name} shape must match logits.")
    for name, value in (
        ("valid_mask", resolved_valid),
        ("loss_mask", resolved_loss),
        ("transition_mask", resolved_transition),
    ):
        if value.dtype != np.bool_:
            raise ArmHoldSequenceEvaluationError(f"{name} must be bool.")
    return (
        resolved_logits,
        resolved_target,
        resolved_valid,
        resolved_loss,
        resolved_transition,
    )


def _threshold_and_tolerance(
    *, threshold: float, transition_tolerance_steps: int
) -> tuple[float, float, int]:
    resolved_threshold = float(threshold)
    if not math.isfinite(resolved_threshold) or not 0.0 < resolved_threshold < 1.0:
        raise ArmHoldSequenceEvaluationError(
            "threshold must be finite and between zero and one."
        )
    tolerance = _integer(
        transition_tolerance_steps, label="transition_tolerance_steps"
    )
    logit_threshold = math.log(resolved_threshold / (1.0 - resolved_threshold))
    return resolved_threshold, logit_threshold, tolerance


def arm_hold_fixed_lead_sequence_evaluation(
    *,
    logits: Any,
    target: Any,
    valid_mask: Any,
    loss_mask: Any,
    transition_mask: Any,
    metadata: Sequence[Mapping[str, Any]],
    lead_offset: int,
    threshold: float = 0.5,
    transition_tolerance_steps: int = 1,
    debug: bool = False,
) -> dict[str, Any]:
    """Evaluate one ``prediction[t, lead] -> target[t + lead]`` timeline."""

    (
        resolved_logits,
        resolved_target,
        resolved_valid,
        resolved_loss,
        resolved_transition,
    ) = _sequence_arrays(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        transition_mask=transition_mask,
    )
    batch_size, horizon, _ = resolved_logits.shape
    resolved_lead = _integer(lead_offset, label="lead_offset")
    if resolved_lead >= horizon:
        raise ArmHoldSequenceEvaluationError("lead_offset exceeds the horizon.")
    resolved_threshold, logit_threshold, tolerance = _threshold_and_tolerance(
        threshold=threshold,
        transition_tolerance_steps=transition_tolerance_steps,
    )
    rows = tuple(metadata)
    if len(rows) != batch_size:
        raise ArmHoldSequenceEvaluationError(
            "metadata count must match the prediction batch."
        )

    effective = resolved_valid[:, resolved_lead] & resolved_loss[:, resolved_lead]
    if not np.any(effective):
        raise ArmHoldSequenceEvaluationError("Fixed-lead evaluation has no support.")
    if not np.all(np.isfinite(resolved_logits[:, resolved_lead][effective])):
        raise ArmHoldSequenceEvaluationError("Effective logits must be finite.")
    if not np.all(np.isin(resolved_target[:, resolved_lead][effective], (0.0, 1.0))):
        raise ArmHoldSequenceEvaluationError("Effective targets must be binary.")
    if np.any(resolved_transition[:, resolved_lead] & ~effective):
        raise ArmHoldSequenceEvaluationError(
            "transition_mask cannot select an excluded fixed-lead target."
        )
    prediction = resolved_logits[:, resolved_lead] >= logit_threshold

    group_rows: dict[tuple[str, str, int], list[tuple[int, int]]] = defaultdict(list)
    seen: set[tuple[str, str, int, int]] = set()
    for batch_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ArmHoldSequenceEvaluationError("metadata rows must be mappings.")
        source = str(row.get("source_dataset_id", "")).strip()
        episode = str(row.get("episode_capture_identity_hash", "")).strip()
        if not source or not episode:
            raise ArmHoldSequenceEvaluationError(
                "metadata source/Episode identity must be non-empty."
            )
        continuity = _integer(
            row.get("continuity_segment_index"),
            label="continuity_segment_index",
        )
        start = _integer(row.get("start_index"), label="start_index")
        target_index = start + resolved_lead
        identity = (source, episode, continuity, target_index)
        if identity in seen:
            raise ArmHoldSequenceEvaluationError(
                "Duplicate fixed-lead target identity detected."
            )
        seen.add(identity)
        group_rows[(source, episode, continuity)].append(
            (target_index, batch_index)
        )

    side_state = {side: _new_side_state() for side in ARM_HOLD_SEQUENCE_SIDES}

    for group, indexed_rows in sorted(group_rows.items()):
        ordered = sorted(indexed_rows)
        for side_index, side in enumerate(ARM_HOLD_SEQUENCE_SIDES):
            segments: list[list[tuple[int, int]]] = []
            current: list[tuple[int, int]] = []
            previous_index: int | None = None
            for target_index, batch_index in ordered:
                if not effective[batch_index, side_index]:
                    if current:
                        segments.append(current)
                        current = []
                    previous_index = None
                    continue
                if previous_index is None or target_index != previous_index + 1:
                    if current:
                        segments.append(current)
                    current = []
                current.append((target_index, batch_index))
                previous_index = target_index
            if current:
                segments.append(current)

            for segment in segments:
                indices = np.asarray([item[1] for item in segment], dtype=np.int64)
                positions = np.asarray([item[0] for item in segment], dtype=np.int64)
                truth = resolved_target[indices, resolved_lead, side_index].astype(bool)
                predicted = prediction[indices, side_index].astype(bool)
                canonical_transition = resolved_transition[
                    indices, resolved_lead, side_index
                ]
                _accumulate_sequence_segment(
                    side_state[side],
                    truth=truth,
                    predicted=predicted,
                    canonical_transition=canonical_transition,
                    positions=positions,
                    tolerance=tolerance,
                    debug_payload=(
                        {
                            "source_dataset_id": group[0],
                            "episode_capture_identity_hash": group[1],
                            "continuity_segment_index": group[2],
                        }
                        if debug
                        else None
                    ),
                )

    per_side = {
        side: _finalize_side_state(
            side_state[side],
            segment_count_field="continuity_coverage_segment_count",
            debug=debug,
        )
        for side in ARM_HOLD_SEQUENCE_SIDES
    }

    return {
        "schema": ARM_HOLD_FIXED_LEAD_SEQUENCE_SCHEMA,
        "scope": "fixed_forecast_lead_episode_timelines_v1",
        "lead_offset": resolved_lead,
        "threshold": resolved_threshold,
        "transition_tolerance_steps": tolerance,
        "window_count": batch_size,
        "episode_count": len({(key[0], key[1]) for key in group_rows}),
        "continuity_group_count": len(group_rows),
        "per_side": per_side,
        "debug": bool(debug),
    }


def arm_hold_chunk_sequence_evaluation(
    *,
    logits: Any,
    target: Any,
    valid_mask: Any,
    loss_mask: Any,
    transition_mask: Any,
    metadata: Sequence[Mapping[str, Any]],
    threshold: float = 0.5,
    transition_tolerance_steps: int = 1,
    debug: bool = False,
) -> dict[str, Any]:
    """Evaluate label timing inside each independently produced H-step chunk.

    Transition support is intentionally counted per overlapping output window.
    Offset zero has no predicted predecessor inside the chunk, so a canonical
    transition there is boundary-unsupported rather than scored.
    """

    (
        resolved_logits,
        resolved_target,
        resolved_valid,
        resolved_loss,
        resolved_transition,
    ) = _sequence_arrays(
        logits=logits,
        target=target,
        valid_mask=valid_mask,
        loss_mask=loss_mask,
        transition_mask=transition_mask,
    )
    batch_size, horizon, _ = resolved_logits.shape
    resolved_threshold, logit_threshold, tolerance = _threshold_and_tolerance(
        threshold=threshold,
        transition_tolerance_steps=transition_tolerance_steps,
    )
    rows = tuple(metadata)
    if len(rows) != batch_size:
        raise ArmHoldSequenceEvaluationError(
            "metadata count must match the prediction batch."
        )
    effective = resolved_valid & resolved_loss
    if not np.any(effective):
        raise ArmHoldSequenceEvaluationError("Chunk evaluation has no support.")
    if not np.all(np.isfinite(resolved_logits[effective])):
        raise ArmHoldSequenceEvaluationError("Effective logits must be finite.")
    if not np.all(np.isin(resolved_target[effective], (0.0, 1.0))):
        raise ArmHoldSequenceEvaluationError("Effective targets must be binary.")
    if np.any(resolved_transition & ~effective):
        raise ArmHoldSequenceEvaluationError(
            "transition_mask cannot select an excluded chunk target."
        )
    prediction = resolved_logits >= logit_threshold
    side_state = {side: _new_side_state() for side in ARM_HOLD_SEQUENCE_SIDES}
    seen: set[tuple[str, str, int, int]] = set()
    episodes: set[tuple[str, str]] = set()

    for batch_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ArmHoldSequenceEvaluationError("metadata rows must be mappings.")
        source = str(row.get("source_dataset_id", "")).strip()
        episode = str(row.get("episode_capture_identity_hash", "")).strip()
        if not source or not episode:
            raise ArmHoldSequenceEvaluationError(
                "metadata source/Episode identity must be non-empty."
            )
        continuity = _integer(
            row.get("continuity_segment_index"),
            label="continuity_segment_index",
        )
        start = _integer(row.get("start_index"), label="start_index")
        identity = (source, episode, continuity, start)
        if identity in seen:
            raise ArmHoldSequenceEvaluationError(
                "Duplicate output-chunk window identity detected."
            )
        seen.add(identity)
        episodes.add((source, episode))
        for side_index, side in enumerate(ARM_HOLD_SEQUENCE_SIDES):
            selected = effective[batch_index, :, side_index]
            offset = 0
            while offset < horizon:
                while offset < horizon and not selected[offset]:
                    offset += 1
                if offset >= horizon:
                    break
                stop = offset + 1
                while stop < horizon and selected[stop]:
                    stop += 1
                positions = np.arange(start + offset, start + stop, dtype=np.int64)
                _accumulate_sequence_segment(
                    side_state[side],
                    truth=resolved_target[
                        batch_index, offset:stop, side_index
                    ].astype(bool),
                    predicted=prediction[batch_index, offset:stop, side_index],
                    canonical_transition=resolved_transition[
                        batch_index, offset:stop, side_index
                    ],
                    positions=positions,
                    tolerance=tolerance,
                    debug_payload=(
                        {
                            "source_dataset_id": source,
                            "episode_capture_identity_hash": episode,
                            "continuity_segment_index": continuity,
                            "window_start_index": start,
                            "window_offset_start": offset,
                            "window_offset_stop_exclusive": stop,
                        }
                        if debug
                        else None
                    ),
                )
                offset = stop

    return {
        "schema": ARM_HOLD_CHUNK_SEQUENCE_SCHEMA,
        "scope": "within_output_chunk_timelines_v1",
        "transition_occurrence_basis": "per_overlapping_output_window_v1",
        "offset_zero_transition_policy": "boundary_unsupported_v1",
        "threshold": resolved_threshold,
        "transition_tolerance_steps": tolerance,
        "window_count": batch_size,
        "horizon": horizon,
        "episode_count": len(episodes),
        "per_side": {
            side: _finalize_side_state(
                side_state[side],
                segment_count_field="chunk_coverage_segment_count",
                debug=debug,
            )
            for side in ARM_HOLD_SEQUENCE_SIDES
        },
        "debug": bool(debug),
    }


__all__ = [
    "ARM_HOLD_CHUNK_SEQUENCE_SCHEMA",
    "ARM_HOLD_FIXED_LEAD_SEQUENCE_SCHEMA",
    "ArmHoldSequenceEvaluationError",
    "arm_hold_chunk_sequence_evaluation",
    "arm_hold_fixed_lead_sequence_evaluation",
]
