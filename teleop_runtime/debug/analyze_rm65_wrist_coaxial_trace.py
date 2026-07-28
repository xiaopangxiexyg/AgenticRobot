"""Offline shadow evaluation for RM65 wrist-coaxial continuity.

This tool reads an existing actuator ``command_trace.npz`` and applies the
single-step q4/q6 canonicalizer without touching hardware or rewriting the
recording.  It is intentionally independent of the real-teleop entrypoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.rm65 import canonicalize_rm65_wrist_coaxial_step


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def analyze_qpos_series(
    qpos_rad: np.ndarray,
    *,
    soft_threshold_deg: float = 4.0,
    hard_threshold_deg: float = 3.0,
    null_motion_soft_limit_deg: float = 5.0,
) -> dict[str, Any]:
    qpos = np.asarray(qpos_rad, dtype=float)
    if qpos.ndim != 2 or qpos.shape[1] != 6:
        raise ValueError(f"qpos_rad must have shape (N, 6), got {qpos.shape}.")
    if qpos.shape[0] == 0:
        return {"samples": 0}
    if not np.all(np.isfinite(qpos)):
        raise ValueError("qpos_rad contains non-finite values.")

    projected = np.empty_like(qpos)
    projected[0] = qpos[0]
    applied = np.zeros(qpos.shape[0], dtype=bool)
    null_scale = np.ones(qpos.shape[0], dtype=float)
    for index in range(1, qpos.shape[0]):
        # This recording contains historical qcmd only, not the raw Cartesian
        # target needed to re-solve the next frame from a modified state.
        # Evaluate each recorded increment independently, then accumulate the
        # transformed increment.  Feeding projected state back into the next
        # historical absolute qcmd would fabricate a closed-loop catch-up that
        # never existed in either controller.
        projected_raw_step, metadata = canonicalize_rm65_wrist_coaxial_step(
            qpos[index - 1],
            qpos[index],
            soft_threshold_deg=soft_threshold_deg,
            hard_threshold_deg=hard_threshold_deg,
            null_motion_soft_limit_deg=null_motion_soft_limit_deg,
        )
        projected[index] = projected[index - 1] + (
            projected_raw_step - qpos[index - 1]
        )
        applied[index] = bool(metadata["applied"])
        null_scale[index] = float(metadata["null_scale"])

    raw_step_deg = np.rad2deg(np.diff(qpos, axis=0))
    projected_step_deg = np.rad2deg(np.diff(projected, axis=0))
    raw_null_step_deg = raw_step_deg[:, 3] - raw_step_deg[:, 5]
    projected_null_step_deg = (
        projected_step_deg[:, 3] - projected_step_deg[:, 5]
    )
    raw_observable = qpos[:, 3] + qpos[:, 5]
    projected_observable = projected[:, 3] + projected[:, 5]
    observable_error_deg = np.rad2deg(projected_observable - raw_observable)
    near_soft = np.minimum(
        np.abs(np.rad2deg(qpos[:-1, 4])),
        np.abs(np.rad2deg(qpos[1:, 4])),
    ) <= float(soft_threshold_deg)

    def joint_step_metrics(step_deg: np.ndarray) -> dict[str, float]:
        wrist_abs = np.abs(step_deg[:, [3, 5]]).reshape(-1)
        return {
            "p99_abs_wrist_step_deg": _percentile(wrist_abs, 99.0),
            "max_abs_wrist_step_deg": (
                float(np.max(wrist_abs)) if wrist_abs.size else 0.0
            ),
        }

    return {
        "samples": int(qpos.shape[0]),
        "shadow_mode": "independent_recorded_increment_open_loop",
        "closed_loop_counterfactual": False,
        "soft_threshold_deg": float(soft_threshold_deg),
        "hard_threshold_deg": float(hard_threshold_deg),
        "null_motion_soft_limit_deg": float(null_motion_soft_limit_deg),
        "projected_samples": int(np.count_nonzero(applied)),
        "near_soft_transitions": int(np.count_nonzero(near_soft)),
        "raw": {
            **joint_step_metrics(raw_step_deg),
            "near_soft_p99_abs_null_step_deg": _percentile(
                np.abs(raw_null_step_deg[near_soft]),
                99.0,
            ),
            "near_soft_max_abs_null_step_deg": (
                float(np.max(np.abs(raw_null_step_deg[near_soft])))
                if np.any(near_soft)
                else 0.0
            ),
        },
        "projected": {
            **joint_step_metrics(projected_step_deg),
            "near_soft_p99_abs_null_step_deg": _percentile(
                np.abs(projected_null_step_deg[near_soft]),
                99.0,
            ),
            "near_soft_max_abs_null_step_deg": (
                float(np.max(np.abs(projected_null_step_deg[near_soft])))
                if np.any(near_soft)
                else 0.0
            ),
        },
        "observable_q4_plus_q6_max_error_deg": (
            float(np.max(np.abs(observable_error_deg)))
            if observable_error_deg.size
            else 0.0
        ),
        "min_null_scale": float(np.min(null_scale)),
    }


def analyze_trace(
    path: str | Path,
    *,
    soft_threshold_deg: float = 4.0,
    hard_threshold_deg: float = 3.0,
    null_motion_soft_limit_deg: float = 5.0,
) -> dict[str, Any]:
    trace_path = Path(path).expanduser()
    with np.load(trace_path, allow_pickle=False) as trace:
        result: dict[str, Any] = {
            "schema": "dexteleop.rm65_wrist_coaxial_shadow/v1",
            "source": str(trace_path),
            "sides": {},
        }
        for side in ("left", "right"):
            if side not in trace.files:
                continue
            records = trace[side]
            if "q_cmd_rad" not in records.dtype.names:
                raise ValueError(f"{side} trace has no q_cmd_rad field.")
            accepted = (
                np.asarray(records["sdk_accepted"], dtype=bool)
                if "sdk_accepted" in records.dtype.names
                else np.ones(records.shape[0], dtype=bool)
            )
            qpos = np.asarray(records["q_cmd_rad"][accepted], dtype=float)
            result["sides"][side] = analyze_qpos_series(
                qpos,
                soft_threshold_deg=soft_threshold_deg,
                hard_threshold_deg=hard_threshold_deg,
                null_motion_soft_limit_deg=null_motion_soft_limit_deg,
            )
        return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline-only shadow analysis of RM65 q4/q6 coaxial continuity "
            "against an existing actuator command trace."
        )
    )
    parser.add_argument(
        "--offline-no-motion",
        "--no-motion",
        dest="offline_no_motion",
        action="store_true",
        required=True,
        help=(
            "Required safety acknowledgement: this command only reads a saved NPZ "
            "and performs no device initialization or motion."
        ),
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--soft-threshold-deg", type=float, default=4.0)
    parser.add_argument("--hard-threshold-deg", type=float, default=3.0)
    parser.add_argument("--null-motion-soft-limit-deg", type=float, default=5.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = analyze_trace(
        args.trace,
        soft_threshold_deg=float(args.soft_threshold_deg),
        hard_threshold_deg=float(args.hard_threshold_deg),
        null_motion_soft_limit_deg=float(args.null_motion_soft_limit_deg),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
