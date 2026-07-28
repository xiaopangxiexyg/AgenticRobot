from __future__ import annotations

from typing import Any

import numpy as np


def canonicalize_rm65_wrist_coaxial_step(
    reference_qpos: np.ndarray,
    candidate_qpos: np.ndarray,
    *,
    soft_threshold_deg: float = 4.0,
    hard_threshold_deg: float = 3.0,
    null_motion_soft_limit_deg: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Suppress unobservable RM65 q4-q6 redistribution near q5 == 0.

    The spherical-wrist observable combination at this singularity is
    ``q4 + q6``.  The returned candidate preserves that combination exactly
    while continuously reducing the orthogonal ``q4 - q6`` increment.
    """

    reference = np.asarray(reference_qpos, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_qpos, dtype=float).reshape(-1)
    if reference.shape != (6,) or candidate.shape != (6,):
        raise ValueError("reference_qpos and candidate_qpos must both be 6D.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("reference_qpos and candidate_qpos must be finite.")
    soft = float(soft_threshold_deg)
    hard = float(hard_threshold_deg)
    if not np.isfinite(soft) or not np.isfinite(hard) or hard < 0.0 or soft < hard:
        raise ValueError("thresholds must satisfy finite 0 <= hard <= soft.")
    null_motion_limit = float(null_motion_soft_limit_deg)
    if not np.isfinite(null_motion_limit) or null_motion_limit <= 0.0:
        raise ValueError("null_motion_soft_limit_deg must be finite and positive.")

    soft_rad = float(np.deg2rad(soft))
    hard_rad = float(np.deg2rad(hard))
    distance = min(abs(float(reference[4])), abs(float(candidate[4])))
    if soft_rad <= hard_rad:
        null_scale = 1.0 if distance > soft_rad else 0.0
    elif distance <= hard_rad:
        null_scale = 0.0
    elif distance >= soft_rad:
        null_scale = 1.0
    else:
        ratio = (distance - hard_rad) / (soft_rad - hard_rad)
        null_scale = float(ratio * ratio * (3.0 - 2.0 * ratio))

    delta4 = float(candidate[3] - reference[3])
    delta6 = float(candidate[5] - reference[5])
    observable_half = 0.5 * (delta4 + delta6)
    null_half = 0.5 * (delta4 - delta6)
    # A small, deliberate null-direction movement is still useful slightly
    # outside exact coaxiality and is already bounded by the downstream joint
    # step/acceleration admission.  Attenuate it progressively by magnitude:
    # small operator inputs remain responsive, while a large equal-and-
    # opposite branch jump receives the full singularity suppression.
    null_motion_deg = abs(float(np.rad2deg(delta4 - delta6)))
    null_motion_factor = float(
        np.clip(null_motion_deg / null_motion_limit, 0.0, 1.0)
    )
    effective_null_scale = 1.0 - (1.0 - float(null_scale)) * null_motion_factor
    output = candidate.copy()
    output[3] = reference[3] + observable_half + effective_null_scale * null_half
    output[5] = reference[5] + observable_half - effective_null_scale * null_half
    applied = bool(
        abs(float(output[3] - candidate[3])) > 1e-12
        or abs(float(output[5] - candidate[5])) > 1e-12
    )
    return output, {
        "applied": applied,
        "min_abs_q5_deg": float(np.rad2deg(distance)),
        "null_scale": float(null_scale),
        "effective_null_scale": float(effective_null_scale),
        "null_motion_deg": float(null_motion_deg),
        "null_motion_soft_limit_deg": float(null_motion_limit),
        "raw_delta_q4_deg": float(np.rad2deg(delta4)),
        "raw_delta_q6_deg": float(np.rad2deg(delta6)),
        "raw_observable_delta_deg": float(np.rad2deg(delta4 + delta6)),
        "raw_null_delta_deg": float(np.rad2deg(delta4 - delta6)),
        "output_delta_q4_deg": float(np.rad2deg(output[3] - reference[3])),
        "output_delta_q6_deg": float(np.rad2deg(output[5] - reference[5])),
        "output_observable_delta_deg": float(
            np.rad2deg(
                (output[3] - reference[3]) + (output[5] - reference[5])
            )
        ),
        "output_null_delta_deg": float(
            np.rad2deg(
                (output[3] - reference[3]) - (output[5] - reference[5])
            )
        ),
    }


def stabilize_rm65_wrist_q5_step(
    reference_qpos: np.ndarray,
    candidate_qpos: np.ndarray,
    *,
    escape_threshold_deg: float = 3.25,
    max_step_rad: float | None = None,
    hysteresis_release_threshold_deg: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep a smoothed RM65 wrist step out of the q5 coaxial band.

    The q4/q6 projection above removes the unobservable wrist-null motion, but
    it deliberately leaves q5 unchanged.  A later exponential qpos smoother
    can nevertheless blend a non-singular candidate through q5 == 0 and put
    the emitted qref back inside the vendor SDK's wrist-singularity gate.  At
    that point the backend has no safe Cartesian direction to infer and holds.

    This helper is a small, position-mode-only task relaxation: while either
    endpoint is close to q5 == 0 (or a *nearby* one-cycle interpolation
    crosses zero), it keeps the previous q5 side and moves toward a finite
    escape boundary.  A far, ordinary sign-changing step remains untouched.
    It never changes q4/q6, and an optional per-cycle step cap prevents the
    escape from bypassing the normal joint-velocity admission.  When
    ``hysteresis_release_threshold_deg`` is set, an inward same-side step
    inside that release band keeps the previous escaped q5 until the target
    clearly leaves the band.
    """

    reference = np.asarray(reference_qpos, dtype=float).reshape(-1)
    candidate = np.asarray(candidate_qpos, dtype=float).reshape(-1)
    if reference.shape != (6,) or candidate.shape != (6,):
        raise ValueError("reference_qpos and candidate_qpos must both be 6D.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("reference_qpos and candidate_qpos must be finite.")

    escape_deg = float(escape_threshold_deg)
    if not np.isfinite(escape_deg) or escape_deg <= 0.0:
        raise ValueError("escape_threshold_deg must be finite and positive.")
    escape_rad = float(np.deg2rad(escape_deg))
    if max_step_rad is not None:
        max_step = float(max_step_rad)
        if not np.isfinite(max_step) or max_step <= 0.0:
            raise ValueError("max_step_rad must be finite and positive when provided.")
    else:
        max_step = None

    if hysteresis_release_threshold_deg is None:
        release_deg = None
    else:
        release_deg = float(hysteresis_release_threshold_deg)
        if not np.isfinite(release_deg) or release_deg < escape_deg:
            raise ValueError(
                "hysteresis_release_threshold_deg must be finite and >= "
                "escape_threshold_deg when provided."
            )

    reference_q5 = float(reference[4])
    candidate_q5 = float(candidate[4])
    reference_sign = float(np.sign(reference_q5))
    if abs(reference_q5) <= 1e-12:
        reference_sign = float(np.sign(candidate_q5))
    if abs(reference_sign) <= 1e-12:
        reference_sign = 1.0

    # A sign change by itself is not evidence that this *step* is near the
    # coaxial gate.  For example, a normal wrist motion from +20 deg to
    # -20 deg crosses zero in Cartesian intent but both admitted endpoints
    # are safely outside the local escape band; treating that as singular
    # would pull a perfectly ordinary target back to +/-3.25 deg and feel
    # permanently sticky.  Only preserve the conservative crossing behavior
    # when at least one endpoint is near the band (or can reach it within the
    # already-admitted one-cycle q5 step cap).
    crossing = bool(reference_q5 * candidate_q5 < 0.0)
    crossing_near_band = crossing and (
        abs(reference_q5) <= escape_rad + (max_step or 0.0)
        or abs(candidate_q5) <= escape_rad + (max_step or 0.0)
    )
    near_band = bool(
        abs(reference_q5) < escape_rad
        or abs(candidate_q5) < escape_rad
        or crossing_near_band
    )
    output = candidate.copy()
    metadata: dict[str, Any] = {
        "enabled": True,
        "applied": False,
        "reason": "outside_escape_band",
        "crossing": crossing,
        "crossing_near_band": crossing_near_band,
        "escape_threshold_deg": escape_deg,
        "reference_q5_deg": float(np.rad2deg(reference_q5)),
        "candidate_q5_deg": float(np.rad2deg(candidate_q5)),
        "output_q5_deg": float(np.rad2deg(candidate_q5)),
        "reference_sign": int(reference_sign),
        "max_step_deg": None if max_step is None else float(np.rad2deg(max_step)),
        "hysteresis_release_threshold_deg": release_deg,
        "hysteresis_hold": False,
    }
    # A small amount of hysteresis is useful after a deliberate singularity
    # escape.  Without it, a raw IK target that alternates around the 3 deg
    # vendor boundary makes the escape candidate alternate between the
    # previous-side boundary and the raw q5 every outer tick.  Only hold an
    # *inward* same-side step while both endpoints are inside the optional
    # release band; ordinary motion outside that band remains untouched.
    if (
        release_deg is not None
        and abs(float(np.rad2deg(reference_q5))) < release_deg
        and abs(float(np.rad2deg(candidate_q5))) < release_deg
        and reference_q5 * candidate_q5 > 0.0
        and abs(candidate_q5) < abs(reference_q5)
        and abs(reference_q5) >= escape_rad
    ):
        output[4] = reference_q5
        metadata.update(
            {
                "applied": bool(abs(reference_q5 - candidate_q5) > 1e-12),
                "reason": "hysteresis_hold_previous_side",
                "hysteresis_hold": True,
                "output_q5_deg": float(np.rad2deg(reference_q5)),
                "output_delta_q5_deg": 0.0,
                "candidate_to_output_delta_q5_deg": float(
                    np.rad2deg(reference_q5 - candidate_q5)
                ),
            }
        )
        return output, metadata

    if not near_band:
        return output, metadata

    # Do not cross the coaxial point from a near-zero reference.  If the
    # candidate already lies safely on that same side, preserve it; otherwise
    # steer to the smallest safe point on the previous side.
    candidate_sign = float(np.sign(candidate_q5))
    if (
        abs(candidate_q5) >= escape_rad
        and candidate_sign == reference_sign
    ):
        desired_q5 = candidate_q5
        reason = "candidate_same_side_safe"
    else:
        desired_q5 = reference_sign * escape_rad
        reason = "escape_previous_side"

    if max_step is not None:
        delta = desired_q5 - reference_q5
        desired_q5 = reference_q5 + float(np.clip(delta, -max_step, max_step))
        if abs(desired_q5 - reference_q5) < abs(delta) - 1e-12:
            reason = f"{reason}_step_limited"

    output[4] = desired_q5
    metadata.update(
        {
            "applied": bool(abs(desired_q5 - candidate_q5) > 1e-12),
            "reason": reason,
            "output_q5_deg": float(np.rad2deg(desired_q5)),
            "output_delta_q5_deg": float(np.rad2deg(desired_q5 - reference_q5)),
            "candidate_to_output_delta_q5_deg": float(
                np.rad2deg(desired_q5 - candidate_q5)
            ),
        }
    )
    return output, metadata


__all__ = [
    "canonicalize_rm65_wrist_coaxial_step",
    "stabilize_rm65_wrist_q5_step",
]
