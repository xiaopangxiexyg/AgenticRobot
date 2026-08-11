"""Read-only audit of RM65 action-reference spikes in collected episodes.

The probe reads finalized parquet episodes and optional actuator traces.  It
never imports hardware modules, initializes a device, or dispatches commands.
It reports joint-reference spikes together with the contemporaneous VIVE pose
step and the recorded action/pose-quality diagnostics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")


def _list_matrix(values: list[Any], columns: int) -> np.ndarray:
    rows = [
        np.asarray(value, dtype=float).reshape(columns)
        if value is not None
        else np.full(columns, np.nan, dtype=float)
        for value in values
    ]
    return np.asarray(rows, dtype=float)


def _pose_steps(values: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    poses = _list_matrix(values, 7)
    translation_mm = np.r_[
        0.0,
        1e3 * np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1),
    ]
    quaternions = poses[:, 3:]
    rotation_deg = np.zeros(len(poses), dtype=float)
    for index in range(1, len(poses)):
        if not (
            np.all(np.isfinite(quaternions[index - 1]))
            and np.all(np.isfinite(quaternions[index]))
        ):
            rotation_deg[index] = np.nan
            continue
        previous = Rotation.from_quat(quaternions[index - 1])
        current = Rotation.from_quat(quaternions[index])
        rotation_deg[index] = np.rad2deg((current * previous.inv()).magnitude())
    return translation_mm, rotation_deg


def _json(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def audit_episode(
    path: Path,
    *,
    event_limit: int,
    quiet_only: bool = False,
    min_qref_step_deg: float = 0.0,
) -> dict[str, Any]:
    columns = [
        "timestamp",
        "action.reference.sequence",
        "action.reference.modification_reason",
        "teleop.pose_quality_json",
    ]
    for side in SIDES:
        columns.extend(
            (
                f"action.reference.{side}_arm.q_ref",
                f"action.reference.{side}_arm.q_ik_raw",
                f"action.reference.{side}_arm.modification_reasons_json",
                f"teleop.action_source.{side}_wrist_pose_base",
            )
        )
    table = pq.read_table(path, columns=columns)
    data = table.to_pydict()
    timestamp = np.asarray(data["timestamp"], dtype=float)
    sequence = np.asarray(data["action.reference.sequence"], dtype=np.int64)
    dt = np.r_[np.nan, np.diff(timestamp)]
    pose_quality = [_json(value) for value in data["teleop.pose_quality_json"]]
    episode: dict[str, Any] = {
        "path": str(path),
        "rows": int(len(timestamp)),
        "duration_sec": float(timestamp[-1] - timestamp[0]) if len(timestamp) > 1 else 0.0,
        "sequence": {"first": int(sequence[0]), "last": int(sequence[-1])},
        "sides": {},
    }
    for side in SIDES:
        qref = _list_matrix(data[f"action.reference.{side}_arm.q_ref"], 6)
        raw = _list_matrix(data[f"action.reference.{side}_arm.q_ik_raw"], 6)
        qref_step = np.rad2deg(np.vstack((np.zeros((1, 6)), np.diff(qref, axis=0))))
        raw_step = np.rad2deg(np.vstack((np.zeros((1, 6)), np.diff(raw, axis=0))))
        output_delta = np.rad2deg(qref - raw)
        velocity = np.divide(
            qref_step,
            dt[:, None],
            out=np.full_like(qref_step, np.nan),
            where=np.isfinite(dt[:, None]) & (dt[:, None] > 0.0),
        )
        velocity_delta = np.vstack((np.zeros((1, 6)), np.diff(velocity, axis=0)))
        acceleration = np.divide(
            velocity_delta,
            dt[:, None],
            out=np.full_like(velocity_delta, np.nan),
            where=np.isfinite(dt[:, None]) & (dt[:, None] > 0.0),
        )
        translation_mm, rotation_deg = _pose_steps(
            data[f"teleop.action_source.{side}_wrist_pose_base"]
        )
        max_step = np.nanmax(np.abs(qref_step), axis=1)
        finite_acceleration = np.where(np.isfinite(acceleration), np.abs(acceleration), 0.0)
        max_acceleration = np.max(finite_acceleration, axis=1)
        quiet_pose = (translation_mm <= 1.0) & (rotation_deg <= 0.25)
        branch_counts = Counter(
            str(dict(item.get("metrics", {}) or {}).get("ik_branch_candidate") or "")
            for item in pose_quality
        )
        side_reason_counts = Counter(
            reason
            for value in data[f"action.reference.{side}_arm.modification_reasons_json"]
            for reason in (
                _json(value)
                if str(value or "").lstrip().startswith("{")
                else json.loads(str(value or "[]"))
            )
            if isinstance(reason, str)
        )
        # Rank high acceleration first, but retain a step contribution so a
        # one-frame branch pulse is not hidden by a long smooth acceleration.
        severity = np.nan_to_num(max_acceleration, nan=0.0) + 100.0 * np.nan_to_num(
            max_step,
            nan=0.0,
        )
        eligible = max_step >= max(0.0, float(min_qref_step_deg))
        if quiet_only:
            eligible &= quiet_pose
        ranked = np.flatnonzero(eligible)
        ranked = ranked[np.argsort(severity[ranked])[::-1]]
        indices = ranked[: max(0, int(event_limit))]
        events: list[dict[str, Any]] = []
        for index in indices:
            joint = int(np.nanargmax(np.abs(acceleration[index]))) if np.any(
                np.isfinite(acceleration[index])
            ) else int(np.nanargmax(np.abs(qref_step[index])))
            metrics = dict(pose_quality[index].get("metrics", {}) or {})
            events.append(
                {
                    "row": int(index),
                    "sequence": int(sequence[index]),
                    "timestamp": float(timestamp[index]),
                    "joint": int(joint + 1),
                    "qref_deg": float(np.rad2deg(qref[index, joint])),
                    "qref_step_deg": float(qref_step[index, joint]),
                    "raw_step_deg": float(raw_step[index, joint]),
                    "qref_minus_raw_deg": float(output_delta[index, joint]),
                    "velocity_deg_s": float(velocity[index, joint]),
                    "acceleration_deg_s2": float(acceleration[index, joint]),
                    "source_translation_step_mm": float(translation_mm[index]),
                    "source_rotation_step_deg": float(rotation_deg[index]),
                    "quiet_source": bool(quiet_pose[index]),
                    "modification_reason": _text(
                        data["action.reference.modification_reason"][index]
                    ),
                    "side_modification_reasons": _text(
                        data[
                            f"action.reference.{side}_arm.modification_reasons_json"
                        ][index]
                    ),
                    "ik_branch_candidate": metrics.get("ik_branch_candidate"),
                    "ik_jump_gate_status": metrics.get("ik_jump_gate_status"),
                    "collision_status": metrics.get(
                        "collision_avoidance_status"
                    ),
                    "joint_limit_margin": metrics.get("joint_limit_margin"),
                    "pose_quality_status": pose_quality[index].get("status"),
                    "pose_quality_reasons": pose_quality[index].get("reasons"),
                }
            )
        episode["sides"][side] = {
            "max_qref_step_deg": float(np.nanmax(max_step)),
            "qref_step_gt_1_deg_count": int(np.count_nonzero(max_step > 1.0)),
            "max_acceleration_deg_s2": float(np.nanmax(max_acceleration)),
            "quiet_source_step_gt_0p5_deg_count": int(
                np.count_nonzero(quiet_pose & (max_step > 0.5))
            ),
            "max_qref_minus_raw_deg": float(np.nanmax(np.abs(output_delta))),
            "ik_branch_counts": dict(branch_counts),
            "side_modification_reason_counts": dict(side_reason_counts),
            "events": events,
        }
    return episode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--event-limit", type=int, default=12)
    parser.add_argument(
        "--quiet-only",
        action="store_true",
        help="rank only events whose per-side VIVE input step is <=1 mm/0.25 deg",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="omit event details and print compact per-episode summaries",
    )
    parser.add_argument(
        "--min-qref-step-deg",
        type=float,
        default=0.0,
        help="only report events whose largest joint-reference step exceeds this value",
    )
    parser.add_argument("--no-motion", action="store_true")
    args = parser.parse_args()
    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("episode_*.parquet")))
        else:
            files.append(path)
    payload = [
        audit_episode(
            path,
            event_limit=args.event_limit,
            quiet_only=bool(args.quiet_only),
            min_qref_step_deg=float(args.min_qref_step_deg),
        )
        for path in files
    ]
    if args.summary_only:
        for episode in payload:
            for side in SIDES:
                episode["sides"][side].pop("events", None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
