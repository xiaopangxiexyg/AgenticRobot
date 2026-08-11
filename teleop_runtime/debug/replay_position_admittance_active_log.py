"""Replay the Active translation gate against a recorded action-debug log.

The action recorder samples at 15 Hz, so this is not a reconstruction of every
60 Hz controller tick.  It replays the exact recorded pre-IK target, bounded
transient, persistent anchor, and admitted-qref translation snapshots in their
causal order.  The report asks whether the current C0 translation-baseline gate
would reject before the recorded total modifier first exceeded its envelope.

This module is diagnostic only.  It owns no device, IK, safety, or dispatch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SIDES = ("left", "right")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(vector) != 3 or not all(math.isfinite(item) for item in vector):
        return None
    return vector


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(item) ** 2 for item in vector))


def _subtract(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))


def _resolve_log(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    candidates = sorted(resolved.rglob("action_debug_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no action_debug_*.jsonl under {resolved}")
    return candidates[-1]


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _first_active_receipt(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for row in rows:
        if row.get("event") != "position_admittance_config":
            continue
        payload = _mapping(row.get("payload"))
        if payload.get("requested_mode") == "active":
            return payload
    return {}


def _event_ref(frame_number: int, frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_number": int(frame_number),
        "step_index": frame.get("step_index"),
        "sample_timestamp": frame.get("sample_timestamp"),
    }


def replay_active_translation_gate(
    path: Path,
    *,
    side: str = "left",
) -> dict[str, Any]:
    """Evaluate the current translation gate on legacy Active snapshots."""

    selected_side = str(side).strip().lower()
    if selected_side not in SIDES:
        raise ValueError("side must be left or right")
    log_path = _resolve_log(path)
    rows = _read_rows(log_path)
    receipt = _first_active_receipt(rows)
    config = _mapping(receipt.get("config"))
    enabled_axes = tuple(bool(value) for value in config.get("enabled_axes", ()))
    max_offset = _vector3(config.get("max_offset_m"))
    if len(enabled_axes) == 3 and max_offset is not None:
        enabled_limits = [
            max_offset[index] for index, enabled in enumerate(enabled_axes) if enabled
        ]
        translation_limit_m = max(enabled_limits) if enabled_limits else 0.0
    else:
        translation_limit_m = None
    stale_after_sec = config.get("stale_after_sec")
    try:
        stale_limit_sec = float(stale_after_sec)
    except (TypeError, ValueError):
        stale_limit_sec = None

    frame_count = 0
    replayable_frame_count = 0
    nonzero_candidate_count = 0
    predicted_rejection_count = 0
    max_transient_m = 0.0
    max_anchor_m = 0.0
    max_total_modifier_m = 0.0
    first_nonzero_candidate: dict[str, Any] | None = None
    first_predicted_rejection: dict[str, Any] | None = None
    first_old_total_violation: dict[str, Any] | None = None
    last_admitted_translation: tuple[float, float, float] | None = None
    last_admitted_at: float | None = None

    for frame_number, row in enumerate(
        (item for item in rows if item.get("event") == "frame"),
        start=1,
    ):
        frame_count += 1
        snapshot = _mapping(row.get("position_admittance"))
        payload = _mapping(_mapping(snapshot.get("sides")).get(selected_side))
        transient = _vector3(payload.get("output_offset_m"))
        anchor = _vector3(payload.get("anchor_offset_m"))
        total = _vector3(payload.get("applied_total_offset_m"))
        applied_target = _vector3(payload.get("applied_target_translation"))
        observed_at_raw = payload.get("observed_at_monotonic_sec")
        try:
            observed_at = float(observed_at_raw)
        except (TypeError, ValueError):
            observed_at = None
        if (
            transient is not None
            and anchor is not None
            and total is not None
            and applied_target is not None
            and observed_at is not None
            and math.isfinite(observed_at)
        ):
            replayable_frame_count += 1
            transient_norm = _norm(transient)
            anchor_norm = _norm(anchor)
            total_norm = _norm(total)
            max_transient_m = max(max_transient_m, transient_norm)
            max_anchor_m = max(max_anchor_m, anchor_norm)
            max_total_modifier_m = max(max_total_modifier_m, total_norm)
            event = _event_ref(frame_number, row)
            if (
                translation_limit_m is not None
                and total_norm > translation_limit_m + 1e-9
                and first_old_total_violation is None
            ):
                first_old_total_violation = {
                    **event,
                    "total_modifier_m": total_norm,
                    "translation_limit_m": translation_limit_m,
                }
            if transient_norm > 1e-12:
                nonzero_candidate_count += 1
                raw_target = _subtract(applied_target, total)
                residual_m = (
                    None
                    if last_admitted_translation is None
                    else _norm(_subtract(last_admitted_translation, raw_target))
                )
                composed_target = tuple(
                    raw_target[index] + transient[index] for index in range(3)
                )
                composed_residual_m = (
                    None
                    if last_admitted_translation is None
                    else _norm(
                        _subtract(last_admitted_translation, composed_target)
                    )
                )
                baseline_age_sec = (
                    None
                    if last_admitted_at is None
                    else observed_at - last_admitted_at
                )
                rejection_reason = None
                if last_admitted_translation is None or last_admitted_at is None:
                    rejection_reason = "missing_admitted_qref_baseline"
                elif (
                    stale_limit_sec is None
                    or baseline_age_sec is None
                    or baseline_age_sec < 0.0
                    or baseline_age_sec > stale_limit_sec
                ):
                    rejection_reason = "stale_admitted_qref_baseline"
                elif (
                    translation_limit_m is None
                    or residual_m is None
                    or residual_m > translation_limit_m + 1e-9
                ):
                    rejection_reason = "nominal_translation_residual"
                elif (
                    composed_residual_m is None
                    or composed_residual_m
                    > translation_limit_m + transient_norm + 1e-9
                ):
                    rejection_reason = "composed_translation_residual"
                candidate = {
                    **event,
                    "transient_m": transient_norm,
                    "baseline_age_sec": baseline_age_sec,
                    "nominal_translation_residual_m": residual_m,
                    "composed_translation_residual_m": composed_residual_m,
                    "translation_limit_m": translation_limit_m,
                    "composed_translation_limit_m": (
                        None
                        if translation_limit_m is None
                        else translation_limit_m + transient_norm
                    ),
                    "rejection_reason": rejection_reason,
                }
                if first_nonzero_candidate is None:
                    first_nonzero_candidate = candidate
                if rejection_reason is not None:
                    predicted_rejection_count += 1
                    if first_predicted_rejection is None:
                        first_predicted_rejection = candidate

        feedback = _mapping(payload.get("admission_feedback"))
        admitted = _vector3(feedback.get("admitted_target_translation"))
        feedback_at_raw = feedback.get("at_monotonic_sec")
        try:
            feedback_at = float(feedback_at_raw)
        except (TypeError, ValueError):
            feedback_at = None
        if (
            admitted is not None
            and feedback_at is not None
            and math.isfinite(feedback_at)
        ):
            last_admitted_translation = admitted
            last_admitted_at = feedback_at

    rejection_precedes_violation = bool(
        first_predicted_rejection is not None
        and first_old_total_violation is not None
        and first_predicted_rejection["frame_number"]
        <= first_old_total_violation["frame_number"]
    )
    active_receipt_valid = bool(
        receipt
        and receipt.get("enabled") is True
        and receipt.get("requested_mode") == "active"
        and selected_side in tuple(receipt.get("apply_sides", ()))
    )
    checks = [
        {
            "name": "recorded_active_receipt",
            "passed": active_receipt_valid,
            "detail": {
                "schema_version": receipt.get("schema_version"),
                "apply_sides": receipt.get("apply_sides"),
            },
        },
        {
            "name": "replayable_active_frames",
            "passed": replayable_frame_count > 0,
            "detail": replayable_frame_count,
        },
        {
            "name": "recorded_nonzero_transient",
            "passed": first_nonzero_candidate is not None,
            "detail": first_nonzero_candidate,
        },
        {
            "name": "recorded_persistent_anchor_and_total_violation",
            "passed": max_anchor_m > 1e-12 and first_old_total_violation is not None,
            "detail": {
                "max_anchor_m": max_anchor_m,
                "first_old_total_violation": first_old_total_violation,
            },
        },
        {
            "name": "current_gate_rejects_before_recorded_violation",
            "passed": rejection_precedes_violation,
            "detail": first_predicted_rejection,
        },
    ]
    return {
        "schema": "dexteleop.position_admittance_active_log_replay/v1",
        "passed": all(check["passed"] for check in checks),
        "diagnostic_only": True,
        "sampled_replay_hz": 15.0,
        "path": str(log_path),
        "side": selected_side,
        "checks": checks,
        "frame_count": frame_count,
        "replayable_frame_count": replayable_frame_count,
        "nonzero_candidate_count": nonzero_candidate_count,
        "predicted_rejection_count": predicted_rejection_count,
        "translation_limit_m": translation_limit_m,
        "stale_limit_sec": stale_limit_sec,
        "max_transient_m": max_transient_m,
        "max_anchor_m": max_anchor_m,
        "max_total_modifier_m": max_total_modifier_m,
        "first_nonzero_candidate": first_nonzero_candidate,
        "first_predicted_rejection": first_predicted_rejection,
        "first_old_total_violation": first_old_total_violation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    args = parser.parse_args()
    report = replay_active_translation_gate(args.log, side=args.side)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
