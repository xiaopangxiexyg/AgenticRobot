"""Offline audit for rare RM65 IK branch and boundary events.

The script reads completed actuator ``command_trace.npz`` files only.  It does
not import hardware packages or initialize devices.  Its output is compact JSON
so several real-run traces can be compared with one deterministic definition of
"large step" and with the opt-in boundary-hold projection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


# Static mirror used by the runtime transport guard.  Keeping the read-only
# probe dependency-free lets it inspect traces in minimal analysis environments.
RM65_ABSOLUTE_Q_MIN_RAD = np.asarray(
    (-3.1, -1.57079632679, -2.355, -2.443460952792061, -2.233, -2.0943951023931953),
    dtype=float,
)
RM65_ABSOLUTE_Q_MAX_RAD = -RM65_ABSOLUTE_Q_MIN_RAD


SIDES = ("left", "right")
SIDE_SLICES = {"left": slice(0, 6), "right": slice(6, 12)}
STAGES = (
    "raw_solver_q_rad",
    "canonical_solver_q_rad",
    "admitted_q_rad",
)
QREF_MAX_STEP_DEG = np.asarray((170.0, 170.0, 190.0, 210.0, 240.0, 240.0)) / 60.0


def _rotation_step_deg(poses: np.ndarray) -> np.ndarray:
    relative = np.einsum("...ji,...jk->...ik", poses[:-1, :3, :3], poses[1:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else 0.0


def _stage_summary(values: np.ndarray) -> dict[str, Any]:
    step = np.rad2deg(np.diff(values, axis=0))
    abs_step = np.abs(step)
    finite_abs_step = np.where(np.isfinite(abs_step), abs_step, -np.inf)
    has_finite = bool(abs_step.size and np.any(np.isfinite(abs_step)))
    flat_index = int(np.argmax(finite_abs_step)) if has_finite else 0
    frame_offset, joint_index = np.unravel_index(flat_index, abs_step.shape) if abs_step.size else (0, 0)
    return {
        "max_step_deg": float(abs_step[frame_offset, joint_index]) if has_finite else 0.0,
        "max_step_frame": int(frame_offset + 1),
        "max_step_joint": int(joint_index + 1),
        "p99_step_deg": _safe_percentile(abs_step, 99.0),
        "step_gt_1p5_count": int(np.count_nonzero(abs_step > 1.5)),
        "step_gt_3_count": int(np.count_nonzero(abs_step > 3.0)),
        "per_joint_max_step_deg": (
            np.nanmax(abs_step, axis=0).astype(float).tolist() if has_finite else [0.0] * 6
        ),
    }


def _delta_summary(delta_rad: np.ndarray) -> dict[str, Any]:
    abs_delta = np.abs(np.rad2deg(np.asarray(delta_rad, dtype=float)))
    finite = np.where(np.isfinite(abs_delta), abs_delta, -np.inf)
    has_finite = bool(abs_delta.size and np.any(np.isfinite(abs_delta)))
    flat_index = int(np.argmax(finite)) if has_finite else 0
    frame, joint = np.unravel_index(flat_index, abs_delta.shape) if abs_delta.size else (0, 0)
    return {
        "max_deg": float(abs_delta[frame, joint]) if has_finite else 0.0,
        "max_frame": int(frame),
        "max_joint": int(joint + 1),
        "p99_deg": _safe_percentile(abs_delta, 99.0),
        "gt_1p5_count": int(np.count_nonzero(abs_delta > 1.5)),
        "gt_3_count": int(np.count_nonzero(abs_delta > 3.0)),
        "per_joint_max_deg": (
            np.nanmax(abs_delta, axis=0).astype(float).tolist() if has_finite else [0.0] * 6
        ),
    }


def _boundary_projection(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    margin_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    upper = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    margin = np.deg2rad(float(margin_deg))
    projected = candidate.copy()
    held = np.zeros(candidate.shape, dtype=bool)
    for frame in range(candidate.shape[0]):
        for joint in range(6):
            lower_band = min(reference[frame, joint], candidate[frame, joint]) <= lower[joint] + margin
            upper_band = max(reference[frame, joint], candidate[frame, joint]) >= upper[joint] - margin
            inward_lower = candidate[frame, joint] < reference[frame, joint]
            inward_upper = candidate[frame, joint] > reference[frame, joint]
            if not (
                np.isfinite(reference[frame, joint])
                and np.isfinite(candidate[frame, joint])
            ):
                continue
            if (lower_band and inward_lower) or (upper_band and inward_upper):
                projected[frame, joint] = reference[frame, joint]
                held[frame, joint] = True
    return projected, held


def _top_events(
    rows: np.ndarray,
    *,
    side: str,
    limit: int,
    boundary_margin_deg: float,
) -> list[dict[str, Any]]:
    side_slice = SIDE_SLICES[side]
    pre = np.asarray(rows["pre_q_rad"][:, side_slice], dtype=float)
    raw = np.asarray(rows["raw_solver_q_rad"][:, side_slice], dtype=float)
    canonical = np.asarray(rows["canonical_solver_q_rad"][:, side_slice], dtype=float)
    admitted = np.asarray(rows["admitted_q_rad"][:, side_slice], dtype=float)
    target = np.asarray(
        rows["target_world_pose"][:, 0 if side == "left" else 1],
        dtype=float,
    )
    pose_translation_step = np.r_[0.0, np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1)]
    pose_rotation_step = np.r_[0.0, _rotation_step_deg(target)]
    admitted_step = np.rad2deg(admitted - pre)
    canonical_step = np.rad2deg(canonical - pre)
    raw_step = np.rad2deg(raw - pre)
    canonical_extra = np.rad2deg(canonical - raw)
    severity = np.nanmax(
        np.stack(
            (
                np.abs(admitted_step),
                np.abs(canonical_step),
                np.abs(raw_step),
                np.abs(canonical_extra),
            ),
            axis=0,
        ),
        axis=(0, 2),
    )
    candidate_projected, held = _boundary_projection(
        pre,
        canonical,
        margin_deg=boundary_margin_deg,
    )
    projected_step = np.rad2deg(candidate_projected - pre)
    lower = np.rad2deg(np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float))
    upper = np.rad2deg(np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float))
    pre_deg = np.rad2deg(pre)
    margin = np.minimum(pre_deg - lower, upper - pre_deg)
    severity = np.where(np.isfinite(severity), severity, -np.inf)
    indices = np.argsort(severity)[::-1][: int(limit)]
    events: list[dict[str, Any]] = []
    for frame in indices:
        joint_severity = np.nanmax(np.stack((
            np.abs(admitted_step[frame]),
            np.abs(canonical_step[frame]),
            np.abs(raw_step[frame]),
            np.abs(canonical_extra[frame]),
        )), axis=0)
        joint = int(np.argmax(np.where(np.isfinite(joint_severity), joint_severity, -np.inf)))
        events.append(
            {
                "frame": int(frame),
                "joint": int(joint + 1),
                "pre_deg": float(pre_deg[frame, joint]),
                "limit_margin_deg": float(margin[frame, joint]),
                "raw_from_pre_deg": float(raw_step[frame, joint]),
                "canonical_from_pre_deg": float(canonical_step[frame, joint]),
                "canonical_minus_raw_deg": float(canonical_extra[frame, joint]),
                "admitted_from_pre_deg": float(admitted_step[frame, joint]),
                "boundary_hold_would_hold": bool(held[frame, joint]),
                "projected_from_pre_deg": float(projected_step[frame, joint]),
                "target_translation_step_m": float(pose_translation_step[frame]),
                "target_rotation_step_deg": float(pose_rotation_step[frame]),
                "status": bytes(rows["status"][frame]).rstrip(b"\0").decode("utf-8", "replace"),
            }
        )
    return events


def _legacy_escape_counterfactual(
    rows: np.ndarray,
    *,
    side: str,
    boundary_margin_deg: float,
) -> dict[str, Any]:
    """Approximate boundary mode from fields shared by old and new traces.

    Boundary mode removes the legacy synthetic escape source before candidate
    ranking.  ``raw_solver_q`` is therefore the closest trace-resident proxy
    for the ordinary target.  We conservatively apply the configured 60 Hz
    qref velocity bound before the same hard-band projection used online.
    """

    side_slice = SIDE_SLICES[side]
    pre = np.asarray(rows["pre_q_rad"][:, side_slice], dtype=float)
    raw = np.asarray(rows["raw_solver_q_rad"][:, side_slice], dtype=float)
    canonical = np.asarray(rows["canonical_solver_q_rad"][:, side_slice], dtype=float)
    raw_delta_deg = np.rad2deg(raw - pre)
    canonical_delta_deg = np.rad2deg(canonical - pre)
    canonical_extra_deg = np.rad2deg(canonical - raw)
    finite = np.all(np.isfinite(pre), axis=1) & np.all(np.isfinite(raw), axis=1)

    # The historical failure signature is an inward ordinary J3 request that
    # is replaced by a full outward limit-escape step.  Ignore startup frames.
    joint = 2
    escape_like = (
        finite
        & (np.arange(rows.size) >= 10)
        & (raw_delta_deg[:, joint] * canonical_delta_deg[:, joint] < 0.0)
        & (np.abs(canonical_extra_deg[:, joint]) > 1.5)
        & (np.abs(canonical_delta_deg[:, joint]) > 1.5)
    )
    lower = np.rad2deg(np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float))
    upper = np.rad2deg(np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float))
    pre_deg = np.rad2deg(pre)
    limit_margin_deg = np.minimum(pre_deg - lower, upper - pre_deg)
    escape_margins = limit_margin_deg[escape_like, joint]
    target = np.asarray(
        rows["target_world_pose"][:, 0 if side == "left" else 1],
        dtype=float,
    )
    target_translation_step = np.r_[
        0.0,
        np.linalg.norm(np.diff(target[:, :3, 3], axis=0), axis=1),
    ]
    target_rotation_step = np.r_[0.0, _rotation_step_deg(target)]

    clipped_delta_deg = np.clip(raw_delta_deg, -QREF_MAX_STEP_DEG, QREF_MAX_STEP_DEG)
    ordinary_proxy = pre + np.deg2rad(clipped_delta_deg)
    projected, held = _boundary_projection(
        pre,
        ordinary_proxy,
        margin_deg=boundary_margin_deg,
    )
    projected_deg = np.rad2deg(projected)
    projected_margin = np.minimum(projected_deg - lower, upper - projected_deg)
    return {
        "legacy_escape_like_count": int(np.count_nonzero(escape_like)),
        "legacy_escape_like_limit_margin_deg": {
            "min": float(np.min(escape_margins)) if escape_margins.size else None,
            "median": float(np.median(escape_margins)) if escape_margins.size else None,
            "p95": float(np.percentile(escape_margins, 95.0)) if escape_margins.size else None,
            "max": float(np.max(escape_margins)) if escape_margins.size else None,
        },
        "legacy_escape_like_max_canonical_extra_deg": (
            float(np.max(np.abs(canonical_extra_deg[escape_like, joint])))
            if np.any(escape_like)
            else 0.0
        ),
        "legacy_escape_like_target_step": {
            "translation_m_p95": (
                float(np.percentile(target_translation_step[escape_like], 95.0))
                if np.any(escape_like)
                else None
            ),
            "translation_m_max": (
                float(np.max(target_translation_step[escape_like]))
                if np.any(escape_like)
                else None
            ),
            "rotation_deg_p95": (
                float(np.percentile(target_rotation_step[escape_like], 95.0))
                if np.any(escape_like)
                else None
            ),
            "rotation_deg_max": (
                float(np.max(target_rotation_step[escape_like]))
                if np.any(escape_like)
                else None
            ),
        },
        "ordinary_proxy_boundary_held_frames": int(np.count_nonzero(np.any(held, axis=1))),
        "ordinary_proxy_per_joint_held_count": np.count_nonzero(held, axis=0).astype(int).tolist(),
        "ordinary_proxy_min_output_margin_deg": (
            np.nanmin(projected_margin, axis=0).astype(float).tolist()
        ),
        "ordinary_proxy_limit_violation_count": int(
            np.count_nonzero(projected_margin < -1e-9)
        ),
        "ordinary_proxy_from_pre": _delta_summary(projected - pre),
    }


def audit_trace(path: Path, *, event_limit: int, boundary_margin_deg: float) -> dict[str, Any]:
    npz_path = path / "command_trace.npz" if path.is_dir() else path
    with np.load(npz_path, allow_pickle=False) as archive:
        rows = np.asarray(archive["ik_shadow"])
        payload: dict[str, Any] = {
            "path": str(npz_path),
            "ik_shadow_rows": int(rows.size),
            "boundary_margin_deg": float(boundary_margin_deg),
            "sides": {},
        }
        for side in SIDES:
            side_slice = SIDE_SLICES[side]
            pre = np.asarray(rows["pre_q_rad"][:, side_slice], dtype=float)
            canonical = np.asarray(rows["canonical_solver_q_rad"][:, side_slice], dtype=float)
            projected, held = _boundary_projection(
                pre,
                canonical,
                margin_deg=boundary_margin_deg,
            )
            side_payload: dict[str, Any] = {
                "stages": {
                    stage: {
                        "successive": _stage_summary(
                            np.asarray(rows[stage][:, side_slice], dtype=float)
                        ),
                        "from_pre": _delta_summary(
                            np.asarray(rows[stage][:, side_slice], dtype=float) - pre
                        ),
                    }
                    for stage in STAGES
                },
                "canonical_minus_raw": _delta_summary(
                    np.asarray(rows["canonical_solver_q_rad"][:, side_slice], dtype=float)
                    - np.asarray(rows["raw_solver_q_rad"][:, side_slice], dtype=float)
                ),
                "boundary_projection": {
                    "held_component_count": int(np.count_nonzero(held)),
                    "held_frame_count": int(np.count_nonzero(np.any(held, axis=1))),
                    "per_joint_held_count": np.count_nonzero(held, axis=0).astype(int).tolist(),
                    "projected_step": _stage_summary(projected),
                },
                "top_events": _top_events(
                    rows,
                    side=side,
                    limit=event_limit,
                    boundary_margin_deg=boundary_margin_deg,
                ),
                "counterfactual": _legacy_escape_counterfactual(
                    rows,
                    side=side,
                    boundary_margin_deg=boundary_margin_deg,
                ),
            }
            payload["sides"][side] = side_payload
        return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--event-limit", type=int, default=8)
    parser.add_argument("--boundary-margin-deg", type=float, default=1.5)
    parser.add_argument(
        "--boundary-margin-sweep-deg",
        default="",
        help="Comma-separated margins; emit aggregate counterfactuals only.",
    )
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--counterfactual-only", action="store_true")
    parser.add_argument("--no-motion", action="store_true")
    args = parser.parse_args()
    if str(args.boundary_margin_sweep_deg).strip():
        sweep: list[dict[str, Any]] = []
        for margin in (
            float(value.strip())
            for value in str(args.boundary_margin_sweep_deg).split(",")
            if value.strip()
        ):
            margin_audits = [
                audit_trace(path, event_limit=0, boundary_margin_deg=margin)
                for path in args.paths
            ]
            per_side: dict[str, dict[str, Any]] = {}
            for side in SIDES:
                side_rows = [
                    audit["sides"][side]["counterfactual"]
                    for audit in margin_audits
                ]
                per_side[side] = {
                    "legacy_escape_like_count": int(
                        sum(int(row["legacy_escape_like_count"]) for row in side_rows)
                    ),
                    "ordinary_proxy_boundary_held_frames": int(
                        sum(
                            int(row["ordinary_proxy_boundary_held_frames"])
                            for row in side_rows
                        )
                    ),
                    "ordinary_proxy_limit_violation_count": int(
                        sum(
                            int(row["ordinary_proxy_limit_violation_count"])
                            for row in side_rows
                        )
                    ),
                    "affected_trace_count": int(
                        sum(
                            int(row["ordinary_proxy_boundary_held_frames"] > 0)
                            for row in side_rows
                        )
                    ),
                }
            sweep.append({"boundary_margin_deg": margin, "sides": per_side})
        print(json.dumps(sweep, ensure_ascii=False, indent=2))
        return
    audits = [
        audit_trace(
            path,
            event_limit=args.event_limit,
            boundary_margin_deg=args.boundary_margin_deg,
        )
        for path in args.paths
    ]
    if args.compact:
        audits = [
            {
                "path": audit["path"],
                "ik_shadow_rows": audit["ik_shadow_rows"],
                "sides": {
                    side: {
                        "raw_from_pre": audit["sides"][side]["stages"]["raw_solver_q_rad"]["from_pre"],
                        "canonical_from_pre": audit["sides"][side]["stages"]["canonical_solver_q_rad"]["from_pre"],
                        "admitted_from_pre": audit["sides"][side]["stages"]["admitted_q_rad"]["from_pre"],
                        "canonical_minus_raw": audit["sides"][side]["canonical_minus_raw"],
                        "boundary_projection": {
                            key: audit["sides"][side]["boundary_projection"][key]
                            for key in (
                                "held_component_count",
                                "held_frame_count",
                                "per_joint_held_count",
                            )
                        },
                        "top_events": audit["sides"][side]["top_events"],
                        "counterfactual": audit["sides"][side]["counterfactual"],
                    }
                    for side in SIDES
                },
            }
            for audit in audits
        ]
    if args.counterfactual_only:
        audits = [
            {
                "path": audit["path"],
                "ik_shadow_rows": audit["ik_shadow_rows"],
                "sides": {
                    side: audit["sides"][side]["counterfactual"]
                    for side in SIDES
                },
            }
            for audit in audits
        ]
    print(
        json.dumps(
            audits,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
