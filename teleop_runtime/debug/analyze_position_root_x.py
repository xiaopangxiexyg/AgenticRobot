"""Summarize root-frame X tracking from a full position-control debug capture.

The runtime writes the three poses needed for this comparison without querying
hardware state: source target, final 60 Hz q_ref FK, and the qcmd FK only after
the RM65 SDK accepted that qcmd.  This tool joins them by the causal reference
epoch/sequence already present in the position audit sidecars.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_FILE = "position_reference.jsonl"
COMMAND_FILE = "position_command_150.jsonl"


@dataclass(frozen=True)
class RootXSample:
    side: str
    timestamp_wall: float
    epoch: int
    reference_sequence: int
    target_x_m: float
    qref_x_m: float
    qcmd_x_m: float
    target_minus_qref_x_m: float
    qref_minus_qcmd_x_m: float
    target_minus_qcmd_x_m: float
    tracker_state: str | None
    hold: bool | None
    collision_status: str | None
    inactive_velocity_release_scale: float | None


@dataclass
class RootXAnalysis:
    reference_rows: int = 0
    references_with_root_poses: int = 0
    accepted_qcmd_rows: int = 0
    accepted_qcmd_without_root_fk: int = 0
    unmatched_accepted_qcmd_rows: int = 0
    samples: list[RootXSample] = field(default_factory=list)


def _jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}.")
            yield row


def _matrix_x(value: object) -> float | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) != 4:
        return None
    row = value[0]
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 4:
        return None
    try:
        x = float(row[3])
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _finite_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer >= 0 and integer == value else None


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _target_activity_state(metadata: Mapping[str, Any]) -> str | None:
    activity = _as_mapping(metadata.get("target_activity"))
    state = activity.get("state")
    return None if state is None else str(state)


def _inactive_velocity_release_scale(metadata: Mapping[str, Any], side: str) -> float | None:
    execution = _as_mapping(metadata.get("execution_step_metadata"))
    inactive_cap = _as_mapping(execution.get("inactive_side_hold_execution_cap"))
    side_metadata = _as_mapping(_as_mapping(inactive_cap.get("sides")).get(side))
    release = _as_mapping(side_metadata.get("tracker_inactive_execution_release"))
    return _finite_float(release.get("velocity_release_scale"))


def analyze_root_x(
    control_debug_dir: Path,
    *,
    sides: set[str] | None = None,
) -> RootXAnalysis:
    control_debug_dir = Path(control_debug_dir)
    reference_path = control_debug_dir / REFERENCE_FILE
    command_path = control_debug_dir / COMMAND_FILE
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing full-debug reference sidecar: {reference_path}")
    if not command_path.is_file():
        raise FileNotFoundError(f"Missing full-debug command sidecar: {command_path}")

    analysis = RootXAnalysis()
    references: dict[
        tuple[str, int, int],
        tuple[float, float, float, str | None, bool | None, str | None, float | None],
    ] = {}
    for row in _jsonl_rows(reference_path):
        epoch = _finite_int(row.get("epoch"))
        reference_sequence = _finite_int(row.get("reference_sequence"))
        if epoch is None or reference_sequence is None:
            continue
        timestamp_wall = _finite_float(row.get("published_wall_time"))
        if timestamp_wall is None:
            timestamp_wall = _finite_float(row.get("timestamp_wall")) or 0.0
        for raw_side, raw_reference in _as_mapping(row.get("references")).items():
            side = str(raw_side)
            if sides is not None and side not in sides:
                continue
            analysis.reference_rows += 1
            reference = _as_mapping(raw_reference)
            metadata = _as_mapping(reference.get("frame_metadata"))
            target_x = _matrix_x(metadata.get("source_target_pose_urdf_root"))
            qref_x = _matrix_x(metadata.get("q_ref_pose_urdf_root"))
            if target_x is None or qref_x is None:
                continue
            analysis.references_with_root_poses += 1
            references[(side, epoch, reference_sequence)] = (
                timestamp_wall,
                target_x,
                qref_x,
                _target_activity_state(metadata),
                _optional_bool(reference.get("hold")),
                _optional_text(metadata.get("collision_avoidance_status")),
                _inactive_velocity_release_scale(metadata, side),
            )

    for row in _jsonl_rows(command_path):
        side = str(row.get("side", ""))
        if not side or (sides is not None and side not in sides):
            continue
        if row.get("sdk_accepted") is not True:
            continue
        analysis.accepted_qcmd_rows += 1
        metadata = _as_mapping(row.get("metadata"))
        qcmd_x = _matrix_x(metadata.get("sdk_accepted_qcmd_pose_urdf_root"))
        if qcmd_x is None:
            analysis.accepted_qcmd_without_root_fk += 1
            continue
        epoch = _finite_int(row.get("reference_epoch"))
        reference_sequence = _finite_int(row.get("reference_sequence"))
        if epoch is None or reference_sequence is None:
            analysis.unmatched_accepted_qcmd_rows += 1
            continue
        reference = references.get((side, epoch, reference_sequence))
        if reference is None:
            analysis.unmatched_accepted_qcmd_rows += 1
            continue
        (
            timestamp_wall,
            target_x,
            qref_x,
            tracker_state,
            hold,
            collision_status,
            inactive_velocity_release_scale,
        ) = reference
        analysis.samples.append(
            RootXSample(
                side=side,
                timestamp_wall=timestamp_wall,
                epoch=epoch,
                reference_sequence=reference_sequence,
                target_x_m=target_x,
                qref_x_m=qref_x,
                qcmd_x_m=qcmd_x,
                target_minus_qref_x_m=target_x - qref_x,
                qref_minus_qcmd_x_m=qref_x - qcmd_x,
                target_minus_qcmd_x_m=target_x - qcmd_x,
                tracker_state=tracker_state,
                hold=hold,
                collision_status=collision_status,
                inactive_velocity_release_scale=inactive_velocity_release_scale,
            )
        )
    return analysis


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _format_metric(name: str, values: Sequence[float]) -> str:
    absolute = [abs(value) for value in values]
    if not absolute:
        return f"{name}: no matched samples"
    p50 = _percentile(absolute, 0.50)
    p95 = _percentile(absolute, 0.95)
    return (
        f"{name}: n={len(absolute)} |abs| m "
        f"p50={p50:.4f} p95={p95:.4f} max={max(absolute):.4f}"
    )


def render_analysis(analysis: RootXAnalysis, *, top: int, threshold_m: float) -> str:
    samples = analysis.samples
    lines = [
        "root-X accepted-qcmd FK diagnostic (URDF root / desk frame)",
        (
            "reference rows="
            f"{analysis.reference_rows}, root-pose references={analysis.references_with_root_poses}, "
            f"accepted qcmd rows={analysis.accepted_qcmd_rows}, matched={len(samples)}, "
            f"accepted qcmd missing root FK={analysis.accepted_qcmd_without_root_fk}, "
            f"unmatched accepted qcmd={analysis.unmatched_accepted_qcmd_rows}"
        ),
        _format_metric(
            "target - qref (retargeter / inactive-cap path)",
            [sample.target_minus_qref_x_m for sample in samples],
        ),
        _format_metric(
            "qref - accepted qcmd (120 Hz planner / SDK path)",
            [sample.qref_minus_qcmd_x_m for sample in samples],
        ),
        _format_metric(
            "target - accepted qcmd (end-to-end command path)",
            [sample.target_minus_qcmd_x_m for sample in samples],
        ),
    ]
    flagged = [
        sample
        for sample in samples
        if abs(sample.target_minus_qcmd_x_m) >= threshold_m
    ]
    lines.append(
        f"end-to-end |X error| >= {threshold_m * 100.0:.1f} cm: {len(flagged)} samples"
    )
    if top > 0 and flagged:
        lines.append("largest end-to-end signed X residuals (target-qcmd):")
        for sample in sorted(
            flagged,
            key=lambda item: abs(item.target_minus_qcmd_x_m),
            reverse=True,
        )[:top]:
            lines.append(
                "  "
                f"side={sample.side} ref={sample.epoch}:{sample.reference_sequence} "
                f"t={sample.timestamp_wall:.3f} target={sample.target_x_m:.4f} "
                f"qref={sample.qref_x_m:.4f} qcmd={sample.qcmd_x_m:.4f} "
                f"target-qref={sample.target_minus_qref_x_m:+.4f} "
                f"qref-qcmd={sample.qref_minus_qcmd_x_m:+.4f} "
                f"target-qcmd={sample.target_minus_qcmd_x_m:+.4f} "
                f"tracker={sample.tracker_state} hold={sample.hold} "
                f"inactive_release={sample.inactive_velocity_release_scale} "
                f"collision={sample.collision_status}"
            )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--control-debug-dir",
        type=Path,
        required=True,
        help="Directory containing position_reference.jsonl and position_command_150.jsonl.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="both",
        help="Arm side to include (default: both).",
    )
    parser.add_argument(
        "--threshold-cm",
        type=float,
        default=1.0,
        help="Print samples whose end-to-end absolute root-X error exceeds this value (default: 1).",
    )
    parser.add_argument("--top", type=int, default=20, help="Maximum flagged rows to print.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not math.isfinite(args.threshold_cm) or args.threshold_cm < 0.0:
        raise SystemExit("--threshold-cm must be finite and >= 0.")
    if args.top < 0:
        raise SystemExit("--top must be >= 0.")
    sides = None if args.side == "both" else {str(args.side)}
    analysis = analyze_root_x(args.control_debug_dir, sides=sides)
    print(
        render_analysis(
            analysis,
            top=int(args.top),
            threshold_m=float(args.threshold_cm) / 100.0,
        )
    )


if __name__ == "__main__":
    main()
