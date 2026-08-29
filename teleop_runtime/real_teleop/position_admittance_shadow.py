"""Compose bounded position-mode admittance before the existing IK pipeline.

The default installation remains shadow-only.  The opt-in active installation
uses the exact same observer and lifecycle state, but returns copied targets
with a translation offset.  The production installer keeps cautious Active as
the default and admits larger normal-parity translation only with one explicit
bounded 5-degree rotation profile.  The observer never mutates its inputs,
bypasses InitArm, or owns IK, collision, qref admission, dispatch, or force-stop
authority.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from robot_motion.position_control.cartesian_admittance import (
    BoundedCartesianAdmittance,
    CartesianAdmittanceConfig,
    CartesianAdmittanceSnapshot,
    CartesianAdmittanceWrenchSample,
)
from robot_motion.position_control.rotational_admittance import (
    BoundedRotationalAdmittance,
    ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG,
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG,
    RotationalAdmittanceConfig,
    RotationalAdmittanceSnapshot,
    compose_world_rotation_offset,
    rotation_matrix_from_rotvec,
    rotational_admittance_config_for_profile,
)
from robot_motion.runtime import TeleopCommand, merge_teleop_commands


POSITION_ADMITTANCE_SHADOW_AXES = ("x", "y", "z")
POSITION_ADMITTANCE_SCHEMA_VERSION = "position-admittance-v6"
POSITION_ADMITTANCE_ROTATION_SCHEMA_VERSION = "position-admittance-v7-rotation"
POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG = 3.0
POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CAUTIOUS = "cautious"
POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL = "half_normal"
POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY = "normal_parity"
POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE = (
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CAUTIOUS
)
POSITION_ADMITTANCE_ACTIVE_ENVELOPES = {
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CAUTIOUS: {
        "max_offset_m": 0.002,
        "max_velocity_m_s": 0.005,
        "max_return_velocity_m_s": 0.005,
        "max_output_slew_m_s": 0.005,
    },
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL: {
        "max_offset_m": 0.04,
        "max_velocity_m_s": 0.03,
        "max_return_velocity_m_s": 0.06,
        "max_output_slew_m_s": 0.03,
    },
    POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY: {
        "max_offset_m": 0.08,
        "max_velocity_m_s": 0.06,
        # The larger rate is return-only: the final boundary admits it only
        # for a complete componentwise-and-norm step toward zero.  Existing
        # 0.03 m/s commands remain valid and behaviorally unchanged.
        "max_return_velocity_m_s": 0.42,
        "max_output_slew_m_s": 0.06,
    },
}
POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CHOICES = tuple(
    POSITION_ADMITTANCE_ACTIVE_ENVELOPES
)
POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M = 0.08
POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD = (
    "runtime_home_hold"
)
POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP = "formal_teleop"
POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES = (
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD,
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP,
)
POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY = (
    POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD
)
POSITION_ADMITTANCE_ACTIVE_SIDE_CHOICES = ("left", "right", "both")
POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE = (
    ROTATIONAL_ADMITTANCE_PROFILE_CAUTIOUS_3DEG
)
POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE = (
    ROTATIONAL_ADMITTANCE_PROFILE_BALANCED_5DEG
)
POSITION_ADMITTANCE_ROTATION_ACTIVE_EMPHASIS_PROFILE = (
    ROTATIONAL_ADMITTANCE_PROFILE_ROTATION_EMPHASIS_5DEG
)
POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES = (
    POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE,
    POSITION_ADMITTANCE_ROTATION_ACTIVE_EMPHASIS_PROFILE,
)
POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG = 3.5
POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_MAX_TOOL_EXCURSION_DEG = 5.5
POSITION_ADMITTANCE_ROTATION_ACTIVE_NOMINAL_TRANSLATION_LIMIT_M = 0.005
POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC = 0.15
POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX = 0.1
POSITION_ADMITTANCE_ACTIVE_RETURN_RATE_S_MAX = 4.0
POSITION_ADMITTANCE_ACTIVE_NORMAL_PARITY_RETURN_VELOCITY_M_S = 0.42
POSITION_ADMITTANCE_ACTIVE_NORMAL_PARITY_RETURN_RATE_S = 4.0
POSITION_ADMITTANCE_ACTIVE_HALF_NORMAL_RETURN_VELOCITY_M_S = 0.06
POSITION_ADMITTANCE_ACTIVE_HALF_NORMAL_RETURN_RATE_S = 2.0
# Public tuning is expressed relative to the real-robot profile that passed
# dual-arm acceptance.  The weights scale compliance (not the final envelope):
# a weight of 2.0 produces twice the unconstrained offset for the same
# deadbanded wrench, while the existing offset/velocity/slew bounds remain the
# last authoritative modifier limits.
POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT = 1.0
POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MIN = 0.25
POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MAX = 4.0
POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M = 250.0
POSITION_ADMITTANCE_ACCEPTED_FILTER_TAU_SEC = 0.010857362
# The accepted profile-compliance response has no physical M/B/K damping term;
# input low-pass time constants are its effective noise/step damping knobs.
# Active permits no filtering for controlled comparison, but rejects excessive
# lag that could keep a stale wrench response alive past the existing 150 ms
# freshness boundary.
POSITION_ADMITTANCE_ACTIVE_FILTER_TAU_SEC_MAX = 0.10
# Modifier-only post-IK endpoint.  The ordinary V2 Balanced 0.55 path is kept
# unchanged for no-force teleoperation; this endpoint is eligible only while
# a nonzero bounded Active modifier is admitted and all ordinary fast-path
# geometry/quality guards pass.
POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA = 0.85
POSITION_ADMITTANCE_ACTIVE_HIGH_SLEW_QPOS_FAST_PATH_ALPHA = 0.80
POSITION_ADMITTANCE_ACTIVE_HIGH_SLEW_THRESHOLD_M_S = 0.03
# Rotation reaches the shared 60 Hz IK as a bounded target orientation.  Keep
# its post-IK endpoint at the existing V2 Balanced responsive alpha so the
# shared 0.5 degree IK target-hold does not become a sparse 0.85-alpha qref
# staircase.  Translation Active retains its separately audited 0.85/0.80
# schedule above.
POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA = 0.55
# Return toward the unmodified VIVE target with the established Active
# endpoint.  This avoids leaving two independently held IK paths on opposite
# sides of the shared 0.5 degree target tolerance after torque release.
POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA = 0.85
# Combined translation+rotation uses the rotation contact endpoint above while
# torque is active.  Once force contact is gone, a still-nonzero translation
# modifier is returning to zero and must not inherit that slower endpoint;
# doing so turns command-tracking lag into delayed outward tool motion.
POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA = 0.85
# Do not switch the accepted-qcmd-to-qref endpoint discontinuously when torque
# contact ends.  A short ramp removes that execution-side phase step while
# still reaching the established responsive release endpoint before the
# 100 ms release-response acceptance boundary.
POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC = 0.075
# Suppress only sub-visible one-frame qref direction chatter.  A persistent
# reversal is released on the following 60 Hz reference tick, while larger
# direction changes and monotonic force response keep the fast endpoint.
POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG = 0.05
# The rerecorded Active trace showed the zero-step form firing 505 frames in
# 59.1 s.  Preserve its one-tick reversal suppression, but admit at most this
# sub-visible creep instead of forcing an exact per-joint stop.  Zero remains
# available in the retargeter config as the atomic rollback behavior.
POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_CREEP_DEG = 0.005


@dataclass(frozen=True)
class ResolvedPositionAdmittanceProfile:
    """One validated position-admittance authority used by every runtime layer."""

    mode: str
    translation_config: CartesianAdmittanceConfig | None
    rotational_config: RotationalAdmittanceConfig | None
    apply_sides: tuple[str, ...] = ()
    active_envelope: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY
    max_tool_excursion_m: float | None = None
    max_tool_rotation_excursion_deg: float | None = None
    maximum_allowed_tool_rotation_excursion_deg: float | None = None

    @property
    def requested(self) -> bool:
        return self.translation_config is not None

    @property
    def active(self) -> bool:
        return bool(self.apply_sides)

    @property
    def rotation_requested(self) -> bool:
        return self.rotational_config is not None

    @property
    def combined_active(self) -> bool:
        return self.mode == "combined_active"


class PositionAdmittanceActiveFault(RuntimeError):
    """Fail-closed pre-IK rejection for the default-off Active modifier."""


def position_admittance_active_apply_sides(
    selection: str,
) -> tuple[str, ...]:
    """Resolve one explicit CLI authority selection to concrete arm sides."""

    normalized = str(selection).strip().lower()
    if normalized == "both":
        return ("left", "right")
    if normalized in {"left", "right"}:
        return (normalized,)
    raise ValueError(
        "position admittance Active side must be one of "
        + ", ".join(POSITION_ADMITTANCE_ACTIVE_SIDE_CHOICES)
    )


def position_admittance_active_qpos_fast_path_alpha(
    config: CartesianAdmittanceConfig,
) -> float:
    """Select the audited modifier-only qpos endpoint from output slew.

    The half-normal 30 mm/s envelope needs the 0.85 endpoint to clear the
    0.1 s release-duration gate.  The 60 mm/s normal-parity envelope uses
    0.80 to keep actuator catch-up below 1 mm.  Cautious Active remains on the
    lower-slew endpoint.  This schedule does not alter modifier offset/slew or
    any downstream joint/collision limit.
    """

    maximum_slew = max(float(value) for value in config.max_output_slew_m_s)
    if maximum_slew <= POSITION_ADMITTANCE_ACTIVE_HIGH_SLEW_THRESHOLD_M_S + 1e-12:
        return POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA
    return POSITION_ADMITTANCE_ACTIVE_HIGH_SLEW_QPOS_FAST_PATH_ALPHA


def position_admittance_final_translation_slew_rate_m_s(
    config: CartesianAdmittanceConfig,
    *,
    previous_offset_m: Sequence[float],
    candidate_offset_m: Sequence[float],
    passive_return_active: bool,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, bool]:
    """Select the final-bound slew rate without weakening outward motion.

    The faster return rate is eligible only while the controller declares
    passive return and the complete candidate modifier moves monotonically
    toward zero componentwise and in total norm.  Production enforcement and
    offline release audits share this predicate so a diagnostic cannot label
    an outward or non-monotonic step as a valid return.
    """

    previous = np.asarray(previous_offset_m, dtype=float).reshape(3)
    candidate = np.asarray(candidate_offset_m, dtype=float).reshape(3)
    enabled = np.asarray(config.enabled_axes, dtype=bool)
    returning_to_zero = bool(
        passive_return_active
        and np.all(
            np.abs(candidate[enabled])
            <= np.abs(previous[enabled]) + float(tolerance)
        )
        and float(np.linalg.norm(candidate))
        <= float(np.linalg.norm(previous)) + float(tolerance)
    )
    return (
        np.asarray(
            (
                config.max_return_velocity_m_s
                if returning_to_zero
                else config.max_output_slew_m_s
            ),
            dtype=float,
        ),
        returning_to_zero,
    )


def position_admittance_final_translation_slew_ratio(
    config: CartesianAdmittanceConfig,
    *,
    previous_offset_m: Sequence[float],
    candidate_offset_m: Sequence[float],
    passive_return_active: bool,
    dt_sec: float,
) -> tuple[float, bool]:
    """Return the exact component-and-norm ratio used by the final bound."""

    dt = float(dt_sec)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError(
            "final translation slew dt must be finite and nonnegative"
        )
    previous = np.asarray(previous_offset_m, dtype=float).reshape(3)
    candidate = np.asarray(candidate_offset_m, dtype=float).reshape(3)
    delta = candidate - previous
    enabled = np.asarray(config.enabled_axes, dtype=bool)
    rates, returning_to_zero = (
        position_admittance_final_translation_slew_rate_m_s(
            config,
            previous_offset_m=previous,
            candidate_offset_m=candidate,
            passive_return_active=passive_return_active,
        )
    )
    limits = rates * dt
    ratios: list[float] = []
    for component_delta, component_limit in zip(
        np.abs(delta[enabled]),
        limits[enabled],
        strict=True,
    ):
        ratios.append(
            0.0
            if component_delta <= 0.0
            else (
                float("inf")
                if component_limit <= 0.0
                else float(component_delta / component_limit)
            )
        )
    norm_delta = float(np.linalg.norm(delta))
    norm_limit = float(np.max(limits[enabled])) if np.any(enabled) else 0.0
    ratios.append(
        0.0
        if norm_delta <= 0.0
        else (
            float("inf")
            if norm_limit <= 0.0
            else norm_delta / norm_limit
        )
    )
    return max(ratios, default=0.0), returning_to_zero


def _positive_response_weight(value: object, *, name: str) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def position_admittance_translation_response_weight(args: Any) -> float:
    """Return the declared compliance multiplier relative to acceptance."""

    return _positive_response_weight(
        getattr(
            args,
            "position_admittance_translation_response_weight",
            POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT,
        ),
        name="position admittance translation response weight",
    )


def position_admittance_rotation_response_weight(args: Any) -> float:
    """Return the declared rotational compliance multiplier."""

    return _positive_response_weight(
        getattr(
            args,
            "position_admittance_rotation_response_weight",
            POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT,
        ),
        name="position admittance rotation response weight",
    )


def position_rotational_admittance_config(
    args: Any,
    *,
    profile: str,
) -> RotationalAdmittanceConfig:
    """Apply public response tuning without changing a named hard envelope."""

    baseline = rotational_admittance_config_for_profile(profile)
    weight = position_admittance_rotation_response_weight(args)
    filter_tau_sec = float(
        getattr(
            args,
            "position_admittance_rotation_filter_tau_sec",
            baseline.torque_filter_tau_sec,
        )
    )
    if not np.isfinite(filter_tau_sec) or filter_tau_sec < 0.0:
        raise ValueError(
            "position admittance rotation filter tau must be finite and non-negative"
        )
    return replace(
        baseline,
        stiffness_nm_rad=tuple(
            float(value) / weight for value in baseline.stiffness_nm_rad
        ),
        torque_filter_tau_sec=filter_tau_sec,
        output_follow_tau_sec=float(
            getattr(
                args,
                "position_admittance_rotation_output_follow_tau_sec",
                baseline.output_follow_tau_sec,
            )
        ),
        max_output_angular_acceleration_rad_s2=(
            float(
                getattr(
                    args,
                    "position_admittance_rotation_max_output_angular_acceleration_rad_s2",
                    0.0,
                )
            ),
        )
        * 3,
    )


def _rotational_config_matches_profile_shape(
    config: RotationalAdmittanceConfig,
    profile: str,
) -> bool:
    expected = rotational_admittance_config_for_profile(profile)
    return (
        replace(
            config,
            stiffness_nm_rad=expected.stiffness_nm_rad,
            torque_filter_tau_sec=expected.torque_filter_tau_sec,
            output_follow_tau_sec=expected.output_follow_tau_sec,
            max_output_angular_acceleration_rad_s2=(0.0, 0.0, 0.0),
        )
        == expected
    )


def position_rotational_admittance_active_profile(
    config: RotationalAdmittanceConfig,
) -> str:
    """Identify the admitted profile while ignoring public response tuning."""

    matches = tuple(
        profile
        for profile in POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES
        if _rotational_config_matches_profile_shape(config, profile)
    )
    if len(matches) != 1:
        raise ValueError(
            "active rotational admittance requires an admitted profile shape"
        )
    return matches[0]


def validate_position_admittance_active_config(
    config: CartesianAdmittanceConfig,
    *,
    active_envelope: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
) -> CartesianAdmittanceConfig:
    """Reject configs outside one explicitly selected hardware envelope.

    Cautious remains the default.  Larger envelopes are available only through
    their explicit names.  The separate authority validator below restricts
    normal-parity to an admitted combined 5-degree profile.  Keep both checks at
    the canonical Active installer so raw CLI numbers cannot silently promote
    a larger hardware envelope.
    """

    if config.response_model != "profile_compliance":
        raise ValueError(
            "active position admittance requires profile_compliance"
        )
    stiffness = np.asarray(config.stiffness_n_m, dtype=float)
    if np.any(stiffness <= 0.0) or not np.allclose(
        stiffness,
        stiffness[0],
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "active position admittance requires one positive isotropic "
            "translation response weight"
        )
    response_weight = (
        POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M
        / float(stiffness[0])
    )
    if not (
        POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MIN
        <= response_weight
        <= POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MAX
    ):
        raise ValueError(
            "active position admittance translation response weight must be in "
            f"[{POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MIN:g}, "
            f"{POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MAX:g}]"
        )
    if (
        config.force_filter_tau_sec
        > POSITION_ADMITTANCE_ACTIVE_FILTER_TAU_SEC_MAX + 1e-12
    ):
        raise ValueError(
            "active position admittance translation filter tau must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_FILTER_TAU_SEC_MAX:g} s"
        )
    if (
        config.output_follow_tau_sec
        > POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX + 1e-12
    ):
        raise ValueError(
            "active position admittance translation output-follow tau must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX:g} s"
        )
    envelope_name = str(active_envelope).strip()
    try:
        envelope = POSITION_ADMITTANCE_ACTIVE_ENVELOPES[envelope_name]
    except KeyError as exc:
        raise ValueError(
            "unknown active position admittance envelope: " + envelope_name
        ) from exc
    limits = {
        field: (getattr(config, field), float(upper_bound))
        for field, upper_bound in envelope.items()
    }
    tolerance = 1e-12
    for field, (values, upper_bound) in limits.items():
        if any(float(value) > upper_bound + tolerance for value in values):
            raise ValueError(
                "active position admittance "
                f"{field} must be <= {upper_bound:g}"
            )
    exact_return = {
        POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL: (
            POSITION_ADMITTANCE_ACTIVE_HALF_NORMAL_RETURN_VELOCITY_M_S,
            POSITION_ADMITTANCE_ACTIVE_HALF_NORMAL_RETURN_RATE_S,
        ),
        POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY: (
            POSITION_ADMITTANCE_ACTIVE_NORMAL_PARITY_RETURN_VELOCITY_M_S,
            POSITION_ADMITTANCE_ACTIVE_NORMAL_PARITY_RETURN_RATE_S,
        ),
    }.get(envelope_name)
    if exact_return is not None:
        exact_return_velocity, exact_return_rate = exact_return
        if not np.allclose(
            config.max_return_velocity_m_s,
            exact_return_velocity,
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError(
                f"active {envelope_name} requires exact "
                "max_return_velocity_m_s="
                f"{exact_return_velocity:g}"
            )
        if not math.isclose(
            config.return_rate_s,
            exact_return_rate,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"active {envelope_name} requires exact return_rate_s="
                f"{exact_return_rate:g}"
            )
    if config.stale_after_sec > POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC:
        raise ValueError(
            "active position admittance stale_after_sec must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC:g}"
        )
    if config.max_force_dt_sec > POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC:
        raise ValueError(
            "active position admittance max_force_dt_sec must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC:g}"
        )
    if config.return_rate_s > POSITION_ADMITTANCE_ACTIVE_RETURN_RATE_S_MAX:
        raise ValueError(
            "active position admittance return_rate_s must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_RETURN_RATE_S_MAX:g}"
        )
    return config


def validate_position_rotational_admittance_active_config(
    config: RotationalAdmittanceConfig,
    *,
    active_profile: str = POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
) -> RotationalAdmittanceConfig:
    """Require one named envelope with bounded public response tuning."""

    profile = str(active_profile).strip()
    if profile not in POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES:
        raise ValueError(
            "active rotational admittance profile must be one of "
            + ", ".join(POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES)
        )
    expected = rotational_admittance_config_for_profile(profile)
    if not _rotational_config_matches_profile_shape(config, profile):
        raise ValueError(
            "active rotational admittance tuning may change only response "
            f"weight, input filter tau, and output-follow tau within the "
            f"{profile} profile"
        )
    stiffness = np.asarray(config.stiffness_nm_rad, dtype=float)
    expected_stiffness = np.asarray(expected.stiffness_nm_rad, dtype=float)
    response_weight_by_axis = expected_stiffness / stiffness
    response_weight = float(response_weight_by_axis[0])
    if not np.allclose(
        response_weight_by_axis,
        response_weight,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(
            "active rotational admittance requires one isotropic response weight"
        )
    if not (
        POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MIN
        <= response_weight
        <= POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MAX
    ):
        raise ValueError(
            "active rotational admittance response weight must be in "
            f"[{POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MIN:g}, "
            f"{POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MAX:g}]"
        )
    if (
        config.torque_filter_tau_sec
        > POSITION_ADMITTANCE_ACTIVE_FILTER_TAU_SEC_MAX + 1e-12
    ):
        raise ValueError(
            "active rotational admittance filter tau must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_FILTER_TAU_SEC_MAX:g} s"
        )
    if (
        config.output_follow_tau_sec
        > POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX + 1e-12
    ):
        raise ValueError(
            "active rotational admittance output-follow tau must be <= "
            f"{POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX:g} s"
        )
    return config


def validate_position_admittance_active_authority(
    config: CartesianAdmittanceConfig,
    *,
    active_envelope: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
    rotational_config: RotationalAdmittanceConfig | None = None,
    combined_translation_rotation_active: bool = False,
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    apply_sides: Sequence[str] = ("left",),
    max_tool_excursion_m: float | None = None,
    max_tool_rotation_excursion_deg: float | None = None,
) -> float | None:
    """Validate one complete production Active authority tuple.

    Returns the largest final tool-rotation envelope authorized by the tuple.
    The normal-parity translation envelope is intentionally inseparable from
    combined bounded 5-degree rotation; cautious and half-normal translation-only
    behavior remain unchanged.
    """

    validate_position_admittance_active_config(
        config,
        active_envelope=active_envelope,
    )
    envelope = str(active_envelope).strip()
    normalized_endpoint_policy = str(endpoint_policy).strip()
    if (
        normalized_endpoint_policy
        not in POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES
    ):
        raise ValueError(
            "active endpoint policy must be one of "
            + ", ".join(POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES)
        )
    translation_enabled = any(config.enabled_axes)
    normalized_apply_sides = tuple(
        str(side).strip().lower() for side in apply_sides
    )
    if (
        not normalized_apply_sides
        or len(set(normalized_apply_sides)) != len(normalized_apply_sides)
        or set(normalized_apply_sides) - {"left", "right"}
    ):
        raise ValueError(
            "active position admittance apply_sides must be a non-empty unique "
            "subset of left/right"
        )
    dual_side_authority = set(normalized_apply_sides) == {"left", "right"}

    if max_tool_excursion_m is not None:
        translation_limit = float(max_tool_excursion_m)
        if not np.isfinite(translation_limit) or translation_limit <= 0.0:
            raise ValueError(
                "active endpoint max tool excursion must be finite and positive"
            )
        if (
            translation_limit
            > POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M + 1e-12
        ):
            raise ValueError(
                "active endpoint max tool excursion must be <= "
                f"{POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M:g}"
            )

    if rotational_config is None:
        if dual_side_authority:
            raise ValueError(
                "dual-side Active requires formal_teleop with exact combined "
                "normal_parity translation and an admitted 5-degree rotation profile"
            )
        if (
            normalized_endpoint_policy
            == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        ):
            raise ValueError(
                "formal_teleop endpoint policy requires exact combined normal_parity "
                "translation and an admitted 5-degree rotation profile"
            )
        if combined_translation_rotation_active:
            raise ValueError(
                "combined translation and rotation Active requires a rotational config"
            )
        if envelope == POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY:
            raise ValueError(
                "normal_parity Active requires combined admitted 5-degree rotation"
            )
        return None

    rotation_profile = position_rotational_admittance_active_profile(
        rotational_config
    )
    validate_position_rotational_admittance_active_config(
        rotational_config,
        active_profile=rotation_profile,
    )
    normal_parity_rotation_profiles = {
        POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE,
        POSITION_ADMITTANCE_ROTATION_ACTIVE_EMPHASIS_PROFILE,
    }
    if rotation_profile == POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE:
        maximum_rotation_excursion_deg = (
            POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG
        )
        if envelope != POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CAUTIOUS:
            raise ValueError(
                "cautious-3deg rotation Active requires the cautious translation envelope"
            )
    elif rotation_profile in normal_parity_rotation_profiles:
        maximum_rotation_excursion_deg = (
            POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_MAX_TOOL_EXCURSION_DEG
        )
        if not combined_translation_rotation_active or not translation_enabled:
            raise ValueError(
                "normal-parity 5-degree rotation Active requires explicit combined "
                "translation and rotation authority"
            )
        if envelope != POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY:
            raise ValueError(
                "normal-parity 5-degree rotation Active requires the "
                "normal_parity translation envelope"
            )
    if translation_enabled and not combined_translation_rotation_active:
        raise ValueError(
            "active rotational admittance requires translation axes disabled"
        )
    if combined_translation_rotation_active and not translation_enabled:
        raise ValueError(
            "combined translation and rotation Active requires at least one "
            "enabled translation axis"
        )
    if (
        normalized_endpoint_policy
        == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        and (
            envelope != POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY
            or not combined_translation_rotation_active
            or rotation_profile not in normal_parity_rotation_profiles
        )
    ):
        raise ValueError(
            "formal_teleop endpoint policy requires exact combined normal_parity "
            "translation and an admitted 5-degree rotation profile"
        )
    if dual_side_authority and (
        normalized_endpoint_policy
        != POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        or envelope != POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY
        or not combined_translation_rotation_active
        or rotation_profile not in normal_parity_rotation_profiles
    ):
        raise ValueError(
            "dual-side Active requires formal_teleop with exact combined "
            "normal_parity translation and an admitted 5-degree rotation profile"
        )
    if max_tool_rotation_excursion_deg is not None:
        rotation_limit = float(max_tool_rotation_excursion_deg)
        if not np.isfinite(rotation_limit) or rotation_limit <= 0.0:
            raise ValueError(
                "active endpoint max tool rotation excursion must be finite and positive"
            )
        if rotation_limit > maximum_rotation_excursion_deg + 1e-12:
            raise ValueError(
                "active endpoint max tool rotation excursion must be <= "
                f"{maximum_rotation_excursion_deg:g}"
            )
    return maximum_rotation_excursion_deg


def position_admittance_active_uses_home_endpoint_guard(
    endpoint_policy: str,
) -> bool:
    """Return whether Active must enforce the temporary Home-relative envelope."""

    normalized = str(endpoint_policy).strip()
    if normalized not in POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES:
        raise ValueError(
            "active endpoint policy must be one of "
            + ", ".join(POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES)
        )
    return (
        normalized
        == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD
    )


def _enabled_axes(value: str) -> tuple[bool, bool, bool]:
    text = str(value).strip().lower()
    if not text or any(character not in POSITION_ADMITTANCE_SHADOW_AXES for character in text):
        raise ValueError("position admittance shadow axes must be a non-empty subset of xyz")
    if len(set(text)) != len(text):
        raise ValueError("position admittance shadow axes must not contain duplicates")
    return tuple(axis in text for axis in POSITION_ADMITTANCE_SHADOW_AXES)


def position_admittance_shadow_config(args: Any) -> CartesianAdmittanceConfig:
    """Resolve one isotropic exploratory config from CLI-like attributes."""

    def triplet(name: str) -> tuple[float, float, float]:
        value = float(getattr(args, name))
        return (value, value, value)

    response_model = str(
        getattr(
            args,
            "position_admittance_response_model",
            "second_order",
        )
    ).strip().lower()
    response_weight = position_admittance_translation_response_weight(args)
    if (
        response_model != "profile_compliance"
        and abs(response_weight - POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT)
        > 1e-12
    ):
        raise ValueError(
            "translation response weight requires profile_compliance; physical "
            "second_order tuning must use mass, damping, and stiffness"
        )
    base_stiffness_n_m = float(
        getattr(args, "position_admittance_shadow_stiffness_n_m")
    )
    return CartesianAdmittanceConfig(
        enabled_axes=_enabled_axes(
            getattr(args, "position_admittance_shadow_axes", "xyz")
        ),
        control_frame="solver_world",
        response_model=response_model,
        virtual_mass_kg=triplet("position_admittance_shadow_mass_kg"),
        damping_n_s_m=triplet("position_admittance_shadow_damping_n_s_m"),
        stiffness_n_m=(base_stiffness_n_m / response_weight,) * 3,
        force_deadband_n=triplet("position_admittance_shadow_deadband_n"),
        force_filter_tau_sec=float(
            getattr(args, "position_admittance_shadow_filter_tau_sec")
        ),
        output_follow_tau_sec=float(
            getattr(
                args,
                "position_admittance_translation_output_follow_tau_sec",
                0.0,
            )
        ),
        max_output_acceleration_m_s2=(
            float(
                getattr(
                    args,
                    "position_admittance_translation_max_output_acceleration_m_s2",
                    0.0,
                )
            ),
        )
        * 3,
        max_offset_m=triplet("position_admittance_shadow_max_offset_m"),
        max_velocity_m_s=triplet(
            "position_admittance_shadow_max_velocity_m_s"
        ),
        max_return_velocity_m_s=triplet(
            "position_admittance_shadow_max_return_velocity_m_s"
        ),
        max_output_slew_m_s=triplet(
            "position_admittance_shadow_max_slew_m_s"
        ),
        return_force_deadband_n=triplet(
            "position_admittance_shadow_return_deadband_n"
        ),
        return_rate_s=float(
            getattr(args, "position_admittance_shadow_return_rate_s")
        ),
        response_reference_hz=float(
            getattr(args, "position_admittance_response_reference_hz")
        ),
        stale_after_sec=float(
            getattr(args, "position_admittance_shadow_stale_sec")
        ),
        max_force_dt_sec=float(
            getattr(args, "position_admittance_shadow_max_force_dt_sec")
        ),
    )


def position_admittance_active_config(args: Any) -> CartesianAdmittanceConfig:
    """Resolve the effective pre-IK translation axes for production Active."""

    config = position_admittance_shadow_config(args)
    if bool(getattr(args, "position_admittance_rotation_active", False)) and not bool(
        getattr(args, "position_admittance_combined_active", False)
    ):
        return replace(config, enabled_axes=(False, False, False))
    return config


def resolve_position_admittance_profile(
    args: Any,
) -> ResolvedPositionAdmittanceProfile:
    """Resolve and validate the complete position-admittance request once."""

    shadow, active, rotation_shadow, rotation_active, combined_active = (
        bool(getattr(args, name, False))
        for name in (
            "position_admittance_shadow",
            "position_admittance_active",
            "position_admittance_rotation_shadow",
            "position_admittance_rotation_active",
            "position_admittance_combined_active",
        )
    )
    if shadow and active:
        raise ValueError(
            "--position-admittance-shadow and --position-admittance-active "
            "are mutually exclusive"
        )
    if rotation_shadow and not shadow:
        raise ValueError(
            "--position-admittance-rotation-shadow requires "
            "--position-admittance-shadow"
        )
    if rotation_active and not active:
        raise ValueError(
            "--position-admittance-rotation-active requires "
            "--position-admittance-active"
        )
    if combined_active and not (active and rotation_active):
        raise ValueError(
            "--position-admittance-combined-active requires both "
            "--position-admittance-active and "
            "--position-admittance-rotation-active"
        )

    active_side = getattr(args, "position_admittance_active_side", None)
    if active and active_side is None:
        raise ValueError(
            "--position-admittance-active requires the explicit authority "
            "selection --position-admittance-active-side {left,right,both}"
        )
    if not active and active_side is not None:
        raise ValueError(
            "--position-admittance-active-side requires "
            "--position-admittance-active"
        )

    def setting(name: str, default: str) -> str:
        return str(getattr(args, name, default))

    active_envelope = setting(
        "position_admittance_active_envelope",
        POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
    )
    endpoint_policy = setting(
        "position_admittance_active_endpoint_policy",
        POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    )
    if not active and active_envelope != POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE:
        raise ValueError(
            "--position-admittance-active-envelope other than cautious requires "
            "--position-admittance-active"
        )
    if (
        not active
        and endpoint_policy
        != POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY
    ):
        raise ValueError(
            "--position-admittance-active-endpoint-policy other than "
            "runtime_home_hold requires --position-admittance-active"
        )

    if not shadow and not active:
        return ResolvedPositionAdmittanceProfile(
            mode="disabled",
            translation_config=None,
            rotational_config=None,
        )

    translation_config = (
        position_admittance_active_config(args)
        if active
        else position_admittance_shadow_config(args)
    )
    rotation_profile = None
    if rotation_active:
        rotation_profile = setting(
            "position_admittance_rotation_active_profile",
            POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE,
        )
    elif rotation_shadow:
        rotation_profile = setting(
            "position_admittance_rotation_shadow_profile",
            POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE,
        )
    rotational_config = (
        position_rotational_admittance_config(args, profile=rotation_profile)
        if rotation_profile is not None
        else None
    )
    if rotation_shadow:
        translation_config = replace(
            translation_config,
            enabled_axes=(False, False, False),
        )

    if not active:
        return ResolvedPositionAdmittanceProfile(
            mode="rotation_shadow" if rotation_shadow else "shadow",
            translation_config=translation_config,
            rotational_config=rotational_config,
        )

    apply_sides = position_admittance_active_apply_sides(str(active_side))
    max_tool_excursion_m = float(
        getattr(
            args,
            "position_admittance_active_max_tool_excursion_m",
            POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M,
        )
    )
    max_tool_rotation_excursion_deg = (
        float(
            getattr(
                args,
                "position_admittance_active_max_tool_rotation_excursion_deg",
                POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG,
            )
        )
        if rotation_active
        else None
    )
    maximum_rotation_excursion_deg = validate_position_admittance_active_authority(
        translation_config,
        active_envelope=active_envelope,
        rotational_config=rotational_config,
        combined_translation_rotation_active=combined_active,
        endpoint_policy=endpoint_policy,
        apply_sides=apply_sides,
        max_tool_excursion_m=max_tool_excursion_m,
        max_tool_rotation_excursion_deg=max_tool_rotation_excursion_deg,
    )
    return ResolvedPositionAdmittanceProfile(
        mode=(
            "combined_active"
            if combined_active
            else "rotation_active"
            if rotation_active
            else "active"
        ),
        translation_config=translation_config,
        rotational_config=rotational_config,
        apply_sides=apply_sides,
        active_envelope=active_envelope,
        endpoint_policy=endpoint_policy,
        max_tool_excursion_m=max_tool_excursion_m,
        max_tool_rotation_excursion_deg=max_tool_rotation_excursion_deg,
        maximum_allowed_tool_rotation_excursion_deg=maximum_rotation_excursion_deg,
    )


def _qpos_by_side_from_joint_command(command: TeleopCommand) -> dict[str, np.ndarray]:
    qpos_by_side: dict[str, np.ndarray] = {}
    for effector in command.commands.values():
        side = str(effector.side or "").strip().lower()
        if side not in {"left", "right"}:
            continue
        if effector.command_type != "joint_vector":
            raise ValueError(
                "active position admittance feedback requires final joint_vector "
                f"commands, got {side}={effector.command_type!r}"
            )
        qpos = np.asarray(effector.value, dtype=float).reshape(-1)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"{side} admitted qref must contain six finite joints")
        qpos_by_side[side] = qpos.copy()
    missing = {"left", "right"}.difference(qpos_by_side)
    if missing:
        raise ValueError(
            "active position admittance feedback requires a complete dual-arm qref; "
            f"missing {', '.join(sorted(missing))}"
        )
    return qpos_by_side


def admitted_solver_world_poses_from_joint_command(
    *,
    retargeter: Any,
    command: TeleopCommand,
) -> dict[str, np.ndarray]:
    """Resolve the final dual-arm qref command to solver-world target poses.

    The command has already passed the session command filters, including the
    temporary dual-arm limiter and Fix.  Reading FK here therefore observes the
    admitted position reference in the exact frame modified before IK, without
    adding a second IK, dispatch, or robot-state owner.  Returning tool-TCP FK
    here would mix it with the solver-frame raw target and inject the configured
    tool transform (about 140 mm for the current DMTAC profile) into the nominal
    residual.
    """

    qpos_by_side = _qpos_by_side_from_joint_command(command)
    resolver = getattr(retargeter, "solver_world_targets_for_qpos_pair", None)
    if not callable(resolver):
        raise TypeError(
            "dual-arm retargeter has no admitted-qref solver-world FK resolver"
        )
    poses = resolver(
        left_qpos=qpos_by_side["left"],
        right_qpos=qpos_by_side["right"],
    )
    return {
        side: np.asarray(poses[side], dtype=float).reshape(4, 4).copy()
        for side in ("left", "right")
    }


class PositionAdmittanceEndpointExcursionGuard:
    """Convert an out-of-envelope final tool-TCP qref into a bounded Hold.

    The anchor is reset from the authoritative runtime-Home qpos pair.  Every
    later command is evaluated after the normal command filters, so the checked
    qpos pair is exactly the pair that would otherwise be dispatched.  The
    guard owns no IK, transport, or recovery.  On a limit crossing it returns
    the last accepted command, annotates that command with a Hold request, and
    remains latched until runtime Home resets the anchor.  The outer real-
    teleop loop then enters its ordinary GUI Hold/Home flow without turning the
    recoverable envelope event into a terminal Session FAULT.
    """

    def __init__(
        self,
        *,
        retargeter: Any,
        active_side: str,
        max_tool_excursion_m: float,
        max_tool_rotation_excursion_deg: float | None = None,
        maximum_allowed_tool_rotation_excursion_deg: float = (
            POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG
        ),
    ) -> None:
        side = str(active_side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("active endpoint excursion side must be left or right")
        limit = float(max_tool_excursion_m)
        if not np.isfinite(limit) or limit <= 0.0:
            raise ValueError(
                "active endpoint max tool excursion must be finite and positive"
            )
        if (
            limit
            > POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M + 1e-12
        ):
            raise ValueError(
                "active endpoint max tool excursion must be <= "
                f"{POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M:g}"
            )
        if not callable(
            getattr(retargeter, "tool_tcp_world_targets_for_qpos_pair", None)
        ):
            raise TypeError(
                "dual-arm retargeter has no final tool-TCP FK resolver"
            )
        rotation_limit = (
            None
            if max_tool_rotation_excursion_deg is None
            else float(max_tool_rotation_excursion_deg)
        )
        if rotation_limit is not None and (
            not np.isfinite(rotation_limit) or rotation_limit <= 0.0
        ):
            raise ValueError(
                "active endpoint max tool rotation excursion must be finite and positive"
            )
        maximum_rotation_limit = float(
            maximum_allowed_tool_rotation_excursion_deg
        )
        if not np.isfinite(maximum_rotation_limit) or maximum_rotation_limit <= 0.0:
            raise ValueError(
                "maximum allowed tool rotation excursion must be finite and positive"
            )
        if (
            maximum_rotation_limit
            > POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_MAX_TOOL_EXCURSION_DEG
            + 1e-12
        ):
            raise ValueError(
                "maximum allowed tool rotation excursion must be <= "
                f"{POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_MAX_TOOL_EXCURSION_DEG:g}"
            )
        if (
            rotation_limit is not None
            and rotation_limit
            > maximum_rotation_limit + 1e-12
        ):
            raise ValueError(
                "active endpoint max tool rotation excursion must be <= "
                f"{maximum_rotation_limit:g}"
            )
        self.retargeter = retargeter
        self.active_side = side
        self.max_tool_excursion_m = limit
        self.max_tool_rotation_excursion_deg = rotation_limit
        self.maximum_allowed_tool_rotation_excursion_deg = maximum_rotation_limit
        self._anchor_tool_translation_m: np.ndarray | None = None
        self._anchor_tool_rotation: np.ndarray | None = None
        self._last_tool_translation_m: np.ndarray | None = None
        self._last_tool_rotation: np.ndarray | None = None
        self._last_excursion_m: float | None = None
        self._last_rotation_excursion_deg: float | None = None
        self._last_accepted_tool_translation_m: np.ndarray | None = None
        self._last_accepted_tool_rotation: np.ndarray | None = None
        self._last_accepted_excursion_m: float | None = None
        self._last_accepted_rotation_excursion_deg: float | None = None
        self._anchor_qpos_by_side: dict[str, np.ndarray] | None = None
        self._last_accepted_command: TeleopCommand | None = None
        self._hold_latched = False
        self._hold_code: str | None = None
        self._hold_reason: str | None = None
        self._hold_count = 0
        self._last_hold_timestamp: float | None = None
        self._last_check_error: str | None = None
        self._reset_count = 0

    @staticmethod
    def _copy_command(command: TeleopCommand) -> TeleopCommand:
        return merge_teleop_commands((command,))

    def _tool_tcp_poses_for_qpos(
        self,
        qpos_by_side: Mapping[str, Sequence[float] | np.ndarray],
    ) -> dict[str, np.ndarray]:
        missing = {"left", "right"}.difference(qpos_by_side)
        if missing:
            raise ValueError(
                "endpoint excursion guard requires a complete dual-arm qpos; missing "
                + ", ".join(sorted(missing))
            )
        poses = self.retargeter.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=qpos_by_side["left"],
            right_qpos=qpos_by_side["right"],
        )
        result: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            pose = np.asarray(poses[side], dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError(
                    f"endpoint excursion {side} tool pose must be a finite 4x4 matrix"
                )
            result[side] = pose.copy()
        return result

    def reset_anchor_from_qpos(
        self,
        qpos_by_side: Mapping[str, Sequence[float] | np.ndarray] | None = None,
    ) -> dict[str, object]:
        resolved_qpos = (
            self.retargeter.current_qpos_by_side()
            if qpos_by_side is None
            else qpos_by_side
        )
        poses = self._tool_tcp_poses_for_qpos(resolved_qpos)
        anchor_qpos = {
            side: np.asarray(resolved_qpos[side], dtype=float).reshape(-1).copy()
            for side in ("left", "right")
        }
        anchor = poses[self.active_side][:3, 3].copy()
        anchor_rotation = poses[self.active_side][:3, :3].copy()
        self._anchor_tool_translation_m = anchor
        self._anchor_tool_rotation = anchor_rotation
        self._last_tool_translation_m = anchor.copy()
        self._last_tool_rotation = anchor_rotation.copy()
        self._last_excursion_m = 0.0
        self._last_rotation_excursion_deg = 0.0
        self._last_accepted_tool_translation_m = anchor.copy()
        self._last_accepted_tool_rotation = anchor_rotation.copy()
        self._last_accepted_excursion_m = 0.0
        self._last_accepted_rotation_excursion_deg = 0.0
        self._anchor_qpos_by_side = anchor_qpos
        self._last_accepted_command = None
        self._hold_latched = False
        self._hold_code = None
        self._hold_reason = None
        self._last_hold_timestamp = None
        self._last_check_error = None
        self._reset_count += 1
        return self.snapshot()

    def _held_command(
        self,
        command: TeleopCommand,
        *,
        now: float,
    ) -> TeleopCommand:
        if self._last_accepted_command is not None:
            held = self._copy_command(self._last_accepted_command)
        else:
            if self._anchor_qpos_by_side is None:
                raise PositionAdmittanceActiveFault(
                    "endpoint Hold has neither an accepted command nor a Home qpos anchor"
                )
            held = self._copy_command(command)
            for side in ("left", "right"):
                arm_command = held.commands.get(f"{side}_arm")
                if arm_command is None:
                    raise PositionAdmittanceActiveFault(
                        f"endpoint Hold cannot restore missing {side}_arm command"
                    )
                arm_command.value = self._anchor_qpos_by_side[side].copy()
        held.timestamp = float(now)
        held.metadata["position_admittance_endpoint_hold"] = {
            "schema": "teleop_runtime.position_admittance_endpoint_hold/v1",
            "code": self._hold_code,
            "reason": self._hold_reason,
            "active_side": self.active_side,
            "candidate_excursion_m": self._last_excursion_m,
            "candidate_rotation_excursion_deg": self._last_rotation_excursion_deg,
            "max_tool_excursion_m": self.max_tool_excursion_m,
            "max_tool_rotation_excursion_deg": self.max_tool_rotation_excursion_deg,
            "last_accepted_excursion_m": self._last_accepted_excursion_m,
            "last_accepted_rotation_excursion_deg": (
                self._last_accepted_rotation_excursion_deg
            ),
            "requires_runtime_home": True,
        }
        return held

    def filter_command(
        self,
        command: TeleopCommand,
        *,
        now: float,
    ) -> TeleopCommand:
        if (
            self._anchor_tool_translation_m is None
            or self._anchor_tool_rotation is None
        ):
            return command
        if self._hold_latched:
            return self._held_command(command, now=now)
        try:
            qpos_by_side = _qpos_by_side_from_joint_command(command)
            poses = self._tool_tcp_poses_for_qpos(qpos_by_side)
        except Exception as exc:  # noqa: BLE001 - pre-dispatch filter must fail closed.
            self._last_check_error = f"{type(exc).__name__}: {exc}"
            raise PositionAdmittanceActiveFault(
                "cannot resolve final Active tool endpoint before dispatch: "
                + self._last_check_error
            ) from exc
        tool_translation = poses[self.active_side][:3, 3].copy()
        tool_rotation = poses[self.active_side][:3, :3].copy()
        excursion_m = float(
            np.linalg.norm(tool_translation - self._anchor_tool_translation_m)
        )
        relative_rotation = self._anchor_tool_rotation.T @ tool_rotation
        rotation_cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        rotation_excursion_deg = float(np.degrees(np.arccos(rotation_cosine)))
        self._last_tool_translation_m = tool_translation
        self._last_tool_rotation = tool_rotation
        self._last_excursion_m = excursion_m
        self._last_rotation_excursion_deg = rotation_excursion_deg
        translation_safe = excursion_m <= self.max_tool_excursion_m + 1e-12
        rotation_safe = bool(
            self.max_tool_rotation_excursion_deg is None
            or rotation_excursion_deg
            <= self.max_tool_rotation_excursion_deg + 1e-12
        )
        if translation_safe and rotation_safe:
            self._last_accepted_tool_translation_m = tool_translation.copy()
            self._last_accepted_tool_rotation = tool_rotation.copy()
            self._last_accepted_excursion_m = excursion_m
            self._last_accepted_rotation_excursion_deg = rotation_excursion_deg
            self._last_accepted_command = self._copy_command(command)
            self._last_check_error = None
            return command
        self._hold_latched = True
        self._hold_count += 1
        self._last_hold_timestamp = float(now)
        if not rotation_safe:
            self._hold_code = "position_admittance_endpoint_rotation_excursion_hold"
            self._hold_reason = (
                f"{self.active_side} final tool rotation excursion "
                f"{rotation_excursion_deg:.6f}deg exceeds "
                f"{self.max_tool_rotation_excursion_deg:.6f}deg"
            )
        else:
            self._hold_code = "position_admittance_endpoint_excursion_hold"
            self._hold_reason = (
                f"{self.active_side} final tool endpoint excursion "
                f"{excursion_m:.6f}m exceeds {self.max_tool_excursion_m:.6f}m"
            )
        return self._held_command(command, now=now)

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": "teleop_runtime.position_admittance_endpoint_excursion/v3",
            "active_side": self.active_side,
            "frame": "solver_world_calibrated_tool_tcp",
            "reference": "runtime_home_final_qpos_fk",
            "max_tool_excursion_m": self.max_tool_excursion_m,
            "max_tool_rotation_excursion_deg": (
                self.max_tool_rotation_excursion_deg
            ),
            "maximum_allowed_tool_rotation_excursion_deg": (
                self.maximum_allowed_tool_rotation_excursion_deg
            ),
            "armed": self._anchor_tool_translation_m is not None,
            "hold_latched": self._hold_latched,
            "hold_code": self._hold_code,
            "hold_reason": self._hold_reason,
            "hold_count": self._hold_count,
            "last_hold_timestamp": self._last_hold_timestamp,
            "last_check_error": self._last_check_error,
            "reset_count": self._reset_count,
            "anchor_tool_translation_m": (
                None
                if self._anchor_tool_translation_m is None
                else self._anchor_tool_translation_m.astype(float).tolist()
            ),
            "anchor_tool_rotation": (
                None
                if self._anchor_tool_rotation is None
                else self._anchor_tool_rotation.astype(float).tolist()
            ),
            "last_tool_translation_m": (
                None
                if self._last_tool_translation_m is None
                else self._last_tool_translation_m.astype(float).tolist()
            ),
            "last_excursion_m": self._last_excursion_m,
            "last_rotation_excursion_deg": self._last_rotation_excursion_deg,
            "last_accepted_tool_translation_m": (
                None
                if self._last_accepted_tool_translation_m is None
                else self._last_accepted_tool_translation_m.astype(float).tolist()
            ),
            "last_accepted_excursion_m": self._last_accepted_excursion_m,
            "last_accepted_rotation_excursion_deg": (
                self._last_accepted_rotation_excursion_deg
            ),
        }


class PositionAdmittanceShadowTargetFilter:
    """Composable pre-IK admittance observer/optional target modifier.

    ``previous_filter`` remains authoritative.  This object observes either
    its override or the original targets.  Shadow mode returns the previous
    result object unchanged.  Active mode only modifies normal teleoperation
    targets; an InitArm override is returned by identity and is never overlaid.
    """

    def __init__(
        self,
        *,
        monitor: Any,
        active_sides: Sequence[str],
        config: CartesianAdmittanceConfig,
        rotational_config: RotationalAdmittanceConfig | None = None,
        previous_filter: Callable[..., Mapping[str, np.ndarray] | None] | None,
        side_activity: Callable[[str], tuple[bool, str]],
        tool_rotation_from_solver_by_side: Mapping[str, np.ndarray],
        apply_to_target: bool = False,
        apply_sides: Sequence[str] | None = None,
        qpos_fast_path_alpha: float | None = None,
        rotation_release_qpos_fast_path_alpha: float | None = None,
        rotation_release_qpos_ramp_sec: float | None = None,
        active_envelope: str | None = None,
        active_endpoint_policy: str | None = None,
    ) -> None:
        sides = tuple(str(side) for side in active_sides)
        if not sides:
            raise ValueError("position admittance shadow requires active sides")
        if monitor is None or not callable(getattr(monitor, "snapshot", None)):
            raise TypeError("position admittance shadow requires a snapshot monitor")
        self.monitor = monitor
        self.active_sides = sides
        self.config = config
        self.rotational_config = rotational_config
        self.previous_filter = previous_filter
        self.side_activity = side_activity
        self.apply_to_target = bool(apply_to_target)
        self.active_envelope = (
            None if active_envelope is None else str(active_envelope).strip()
        )
        self.active_endpoint_policy = (
            None
            if active_endpoint_policy is None
            else str(active_endpoint_policy).strip()
        )
        resolved_qpos_alpha = (
            position_admittance_active_qpos_fast_path_alpha(config)
            if qpos_fast_path_alpha is None
            else float(qpos_fast_path_alpha)
        )
        if not np.isfinite(resolved_qpos_alpha) or not 0.0 <= resolved_qpos_alpha <= 1.0:
            raise ValueError("position admittance qpos fast-path alpha must be in [0, 1]")
        self.qpos_fast_path_alpha = resolved_qpos_alpha
        resolved_rotation_release_alpha = (
            POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA
            if rotation_release_qpos_fast_path_alpha is None
            else float(rotation_release_qpos_fast_path_alpha)
        )
        if (
            not np.isfinite(resolved_rotation_release_alpha)
            or not 0.0 <= resolved_rotation_release_alpha <= 1.0
        ):
            raise ValueError(
                "position admittance rotation-release qpos fast-path alpha "
                "must be in [0, 1]"
            )
        self.rotation_release_qpos_fast_path_alpha = (
            resolved_rotation_release_alpha
        )
        resolved_rotation_release_ramp_sec = (
            POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC
            if rotation_release_qpos_ramp_sec is None
            else float(rotation_release_qpos_ramp_sec)
        )
        if (
            not np.isfinite(resolved_rotation_release_ramp_sec)
            or resolved_rotation_release_ramp_sec < 0.0
        ):
            raise ValueError(
                "position admittance rotation-release qpos ramp must be "
                "finite and nonnegative"
            )
        self.rotation_release_qpos_ramp_sec = (
            resolved_rotation_release_ramp_sec
        )
        requested_apply_sides = (
            sides
            if apply_sides is None and self.apply_to_target
            else tuple(apply_sides or ())
        )
        requested_apply_side_set = set(requested_apply_sides)
        unknown_apply_sides = requested_apply_side_set.difference(sides)
        if unknown_apply_sides:
            raise ValueError(
                "position admittance apply_sides are not active: "
                + ", ".join(sorted(unknown_apply_sides))
            )
        self.apply_sides = tuple(
            side for side in sides if side in requested_apply_side_set
        )
        self.controllers = {
            side: BoundedCartesianAdmittance(config) for side in sides
        }
        self.rotation_controllers = (
            {}
            if rotational_config is None
            else {
                side: BoundedRotationalAdmittance(rotational_config)
                for side in sides
            }
        )
        self._tool_rotation_from_solver = {
            side: self._validate_rotation(
                tool_rotation_from_solver_by_side.get(side, np.eye(3)),
                name=f"{side}.tool_rotation_from_solver",
            )
            for side in sides
        }
        self._mode_active_by_side = {side: False for side in sides}
        self._rotation_contact_seen_by_side = {side: False for side in sides}
        self._rotation_release_started_at_by_side: dict[str, float | None] = {
            side: None for side in sides
        }
        self._tare_generation_by_side = {side: 0 for side in sides}
        self._tare_values_by_side: dict[str, tuple[float, ...]] = {}
        self._latest_by_side: dict[str, dict[str, object]] = {}
        self._contact_axes_by_side = {
            side: np.zeros(3, dtype=bool) for side in sides
        }
        self._pending_raw_target_by_side: dict[str, np.ndarray] = {}
        self._admission_feedback_by_side: dict[str, dict[str, object]] = {}
        self._admission_feedback_fault_by_side: dict[str, str | None] = {
            side: None for side in sides
        }
        self._last_composed_offset_by_side: dict[str, np.ndarray] = {}
        self._last_composed_time_by_side: dict[str, float] = {}
        self._last_composed_rotation_by_side: dict[str, np.ndarray] = {}
        self._last_composed_rotation_time_by_side: dict[str, float] = {}
        self._last_admitted_pose_by_side: dict[str, np.ndarray] = {}
        self._last_admitted_observed_at_by_side: dict[str, float] = {}
        self._frame_count = 0
        self._last_error: str | None = None

    @staticmethod
    def _validate_rotation(value: np.ndarray, *, name: str) -> np.ndarray:
        rotation = np.asarray(value, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"{name} must be a finite 3x3 matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
            raise ValueError(f"{name} must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(f"{name} must be a proper rotation")
        return rotation.copy()

    @staticmethod
    def _reading_by_side(snapshot: Mapping[str, object]) -> Mapping[str, object]:
        readings = snapshot.get("last_reading_by_side", {})
        if not isinstance(readings, Mapping):
            raise ValueError("force monitor snapshot has no side readings")
        return readings

    def reset_from_monitor(
        self,
        *,
        sides: Sequence[str] | None = None,
        now_sec: float | None = None,
    ) -> dict[str, object]:
        """Create a fresh per-side tare generation after runtime Home."""

        now = time.monotonic() if now_sec is None else float(now_sec)
        monitor_snapshot = self.monitor.snapshot()
        readings = self._reading_by_side(monitor_snapshot)
        selected = self.active_sides if sides is None else tuple(str(side) for side in sides)
        for side in selected:
            if side not in self.controllers:
                raise ValueError(f"unknown position admittance shadow side {side!r}")
            reading = readings.get(side)
            if not isinstance(reading, Mapping):
                raise RuntimeError(f"no force reading is available for {side}")
            source = str(reading.get("source", "")).strip()
            if source != "tool":
                raise RuntimeError(
                    "position admittance shadow currently requires tool-frame wrench, "
                    f"got {side} source={source!r}"
                )
            values = tuple(float(value) for value in reading.get("values", ()))
            generation = self._tare_generation_by_side[side] + 1
            sample = CartesianAdmittanceWrenchSample(
                values_n_nm=values,
                frame="tool",
                source_time_sec=float(reading["monotonic_time_sec"]),
                receive_time_sec=now,
                tare_generation=generation,
            )
            self.controllers[side].reset(
                tare_sample=sample,
                output_time_sec=now,
            )
            rotation_controller = self.rotation_controllers.get(side)
            if rotation_controller is not None:
                rotation_controller.reset(
                    tare_sample=sample,
                    output_time_sec=now,
                )
            self._tare_generation_by_side[side] = generation
            self._tare_values_by_side[side] = values
            self._mode_active_by_side[side] = False
            self._rotation_contact_seen_by_side[side] = False
            self._rotation_release_started_at_by_side[side] = None
            self._contact_axes_by_side[side].fill(False)
            self._pending_raw_target_by_side.pop(side, None)
            self._admission_feedback_by_side.pop(side, None)
            self._admission_feedback_fault_by_side[side] = None
            self._last_composed_offset_by_side[side] = np.zeros(3, dtype=float)
            self._last_composed_time_by_side[side] = now
            self._last_composed_rotation_by_side[side] = np.zeros(3, dtype=float)
            self._last_composed_rotation_time_by_side[side] = now
            self._last_admitted_pose_by_side.pop(side, None)
            self._last_admitted_observed_at_by_side.pop(side, None)
            self._latest_by_side[side] = self._side_payload(
                side=side,
                reading=reading,
                state=self.controllers[side].snapshot(now_sec=now),
                active=False,
                activity_reason="runtime_home_tare",
                target=None,
                now_sec=now,
                rotation_state=(
                    None
                    if rotation_controller is None
                    else rotation_controller.snapshot(now_sec=now)
                ),
            )
        return self.snapshot(now_sec=now)

    def freeze_all(self, reason: str) -> None:
        for side, controller in self.controllers.items():
            controller.freeze(reason)
            rotation_controller = self.rotation_controllers.get(side)
            if rotation_controller is not None:
                rotation_controller.freeze(reason)
            self._mode_active_by_side[side] = False
            self._rotation_contact_seen_by_side[side] = False
            self._rotation_release_started_at_by_side[side] = None

    def rearm_after_control_plane_pause(
        self,
        *,
        sides: Sequence[str] | None = None,
        now_sec: float | None = None,
        pause_duration_sec: float = 0.0,
        reason: str = "known_control_plane_pause",
    ) -> dict[str, object]:
        """Re-arm force sample clocks after one explicit runtime pause.

        Formal collection startup may synchronously quiesce tactile workers and
        start/reconfigure the dataset writer from the control-loop thread.  If
        that known transition takes longer than ``max_force_dt_sec``, the next
        wrench sample must not be integrated across the pause.  This method
        clears only the causal sample/output clocks while preserving the
        already admitted force and rotation offsets; the next fresh sample is
        therefore a clock re-arm, not a force step.

        ``pause_duration_sec`` advances only the admitted-qref freshness clock
        by the measured blocked interval.  The pose and target remain exactly
        unchanged, while the normal nominal-target residual checks still see
        the old admitted pose and can reject real operator motion.

        Callers must use this only for a measured, named control-plane pause.
        Unexpected sensor gaps continue through ``force_dt_exceeded`` /
        ``torque_dt_exceeded`` and remain fail-closed.
        """

        now = time.monotonic() if now_sec is None else float(now_sec)
        if not np.isfinite(now):
            raise ValueError("control-plane rearm time must be finite")
        pause_duration = float(pause_duration_sec)
        if not np.isfinite(pause_duration) or pause_duration < 0.0:
            raise ValueError(
                "control-plane pause duration must be finite and nonnegative"
            )
        selected = (
            self.active_sides
            if sides is None
            else tuple(str(side) for side in sides)
        )
        rearmed: list[str] = []
        for side in selected:
            if side not in self.controllers:
                raise ValueError(f"unknown position admittance shadow side {side!r}")
            # A fixed/held side owns no live force target and must keep its
            # existing lifecycle state.  Only a side that was active before
            # the measured pause is eligible for this one-shot re-arm.
            if not self._mode_active_by_side.get(side, False):
                continue

            controller = self.controllers[side]
            controller_snapshot = controller.snapshot(now_sec=now)
            if controller_snapshot.frozen:
                raise RuntimeError(
                    f"{side} force controller is already frozen: "
                    f"{controller_snapshot.reason}"
                )
            previous_output = controller_snapshot.output_offset_m
            controller.resume()
            # ``resume`` intentionally preserves the force offset but leaves
            # the output clock at its pre-pause value.  Rebase that realized
            # output timestamp without changing the requested offset so the
            # first post-pause frame cannot consume the whole pause as a
            # synthetic output-slew interval.
            controller.rebase_realized_offset(
                previous_output,
                output_time_sec=now,
                preserve_requested=True,
            )

            rotation_controller = self.rotation_controllers.get(side)
            if rotation_controller is not None:
                rotation_snapshot = rotation_controller.snapshot(now_sec=now)
                if rotation_snapshot.frozen:
                    raise RuntimeError(
                        f"{side} rotation controller is already frozen: "
                        f"{rotation_snapshot.reason}"
                    )
                previous_rotation = rotation_snapshot.output_rotvec_rad
                rotation_controller.resume()
                rotation_controller.rebase_realized_rotation(
                    previous_rotation,
                    output_time_sec=now,
                    preserve_requested=True,
                )

            self._last_composed_time_by_side[side] = now
            self._last_composed_rotation_time_by_side[side] = now
            # The admitted-qref pose itself remains the last safe baseline.
            # Advance only its causal timestamp by the measured blocked
            # duration, so the freshness guard counts active control time and
            # never treats the writer/quiesce pause as a sensor-age interval.
            # No pose or target is invented by this timing-only operation.
            if side in self._last_admitted_observed_at_by_side:
                self._last_admitted_observed_at_by_side[side] = min(
                    now,
                    self._last_admitted_observed_at_by_side[side]
                    + pause_duration,
                )
            latest = self._latest_by_side.get(side)
            if latest is not None:
                latest["control_plane_rearm"] = {
                    "reason": str(reason),
                    "at_monotonic_sec": now,
                }
            rearmed.append(side)
        return {
            "reason": str(reason),
            "at_monotonic_sec": now,
            "pause_duration_sec": pause_duration,
            "sides": rearmed,
        }

    def latch_admission_feedback_fault(self, reason: str) -> None:
        """Keep active output frozen until a fresh runtime-Home reset."""

        message = str(reason).strip() or "admission_feedback_fault"
        for side in self.apply_sides:
            self._admission_feedback_fault_by_side[side] = message
            self.controllers[side].freeze("admission_feedback_fault")
            rotation_controller = self.rotation_controllers.get(side)
            if rotation_controller is not None:
                rotation_controller.freeze("admission_feedback_fault")
            self._mode_active_by_side[side] = False
            self._rotation_contact_seen_by_side[side] = False
            self._rotation_release_started_at_by_side[side] = None
            latest = self._latest_by_side.get(side)
            if latest is not None:
                latest["active"] = False
                latest["activity_reason"] = "admission_feedback_fault"
                latest["frozen"] = True
                latest["admission_feedback_fault"] = message

    def _checked_final_modifier(
        self,
        *,
        side: str,
        state: CartesianAdmittanceSnapshot,
        now_sec: float,
    ) -> np.ndarray:
        """Validate the final C0 modifier without silently clamping a defect."""

        if state.frozen:
            raise PositionAdmittanceActiveFault(
                f"{side} Active admittance state is frozen: {state.reason}"
            )
        offset = np.asarray(state.output_offset_m, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise PositionAdmittanceActiveFault(
                f"{side} final admittance modifier must contain three finite values"
            )
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        limits = np.asarray(self.config.max_offset_m, dtype=float)
        tolerance = 1e-9
        if np.any(np.abs(offset[~enabled]) > tolerance):
            raise PositionAdmittanceActiveFault(
                f"{side} final admittance modifier is nonzero on a disabled axis"
            )
        if np.any(np.abs(offset[enabled]) > limits[enabled] + tolerance):
            raise PositionAdmittanceActiveFault(
                f"{side} final admittance modifier exceeds a component limit"
            )
        norm_limit = float(np.max(limits[enabled])) if np.any(enabled) else 0.0
        if float(np.linalg.norm(offset)) > norm_limit + tolerance:
            raise PositionAdmittanceActiveFault(
                f"{side} final admittance modifier exceeds total translation limit"
            )

        previous = self._last_composed_offset_by_side.get(side)
        previous_time = self._last_composed_time_by_side.get(side)
        if previous is not None and previous_time is not None:
            dt = float(now_sec) - float(previous_time)
            if dt < 0.0:
                raise PositionAdmittanceActiveFault(
                    f"{side} final admittance modifier time moved backwards"
                )
            delta = offset - previous
            slew_rate, _ = position_admittance_final_translation_slew_rate_m_s(
                self.config,
                previous_offset_m=previous,
                candidate_offset_m=offset,
                passive_return_active=state.passive_return_active,
                tolerance=tolerance,
            )
            slew_limits = slew_rate * dt
            if np.any(np.abs(delta[enabled]) > slew_limits[enabled] + tolerance):
                raise PositionAdmittanceActiveFault(
                    f"{side} final admittance modifier exceeds a component slew limit"
                )
            slew_norm_limit = (
                float(np.max(slew_limits[enabled])) if np.any(enabled) else 0.0
            )
            if float(np.linalg.norm(delta)) > slew_norm_limit + tolerance:
                raise PositionAdmittanceActiveFault(
                    f"{side} final admittance modifier exceeds total translation slew"
                )

        self._last_composed_offset_by_side[side] = offset.copy()
        self._last_composed_time_by_side[side] = float(now_sec)
        return offset

    def _checked_final_rotation_modifier(
        self,
        *,
        side: str,
        state: RotationalAdmittanceSnapshot,
        now_sec: float,
    ) -> np.ndarray:
        """Validate the complete rotation-vector modifier before SO(3) composition."""

        config = self.rotational_config
        if config is None:
            return np.zeros(3, dtype=float)
        if state.frozen:
            raise PositionAdmittanceActiveFault(
                f"{side} rotational admittance state is frozen: {state.reason}"
            )
        rotvec = np.asarray(state.output_rotvec_rad, dtype=float)
        if rotvec.shape != (3,) or not np.all(np.isfinite(rotvec)):
            raise PositionAdmittanceActiveFault(
                f"{side} final rotation modifier must contain three finite values"
            )
        enabled = np.asarray(config.enabled_axes, dtype=bool)
        limits = np.asarray(config.max_offset_rad, dtype=float)
        tolerance = 1e-9
        if np.any(np.abs(rotvec[~enabled]) > tolerance):
            raise PositionAdmittanceActiveFault(
                f"{side} final rotation modifier is nonzero on a disabled axis"
            )
        if np.any(np.abs(rotvec[enabled]) > limits[enabled] + tolerance):
            raise PositionAdmittanceActiveFault(
                f"{side} final rotation modifier exceeds a component limit"
            )
        norm_limit = float(np.max(limits[enabled])) if np.any(enabled) else 0.0
        if float(np.linalg.norm(rotvec)) > norm_limit + tolerance:
            raise PositionAdmittanceActiveFault(
                f"{side} final rotation modifier exceeds total angle limit"
            )

        previous = self._last_composed_rotation_by_side.get(side)
        previous_time = self._last_composed_rotation_time_by_side.get(side)
        if previous is not None and previous_time is not None:
            dt = float(now_sec) - float(previous_time)
            if dt < 0.0:
                raise PositionAdmittanceActiveFault(
                    f"{side} final rotation modifier time moved backwards"
                )
            delta = rotvec - previous
            slew_limits = np.asarray(config.max_output_slew_rad_s) * dt
            if np.any(np.abs(delta[enabled]) > slew_limits[enabled] + tolerance):
                raise PositionAdmittanceActiveFault(
                    f"{side} final rotation modifier exceeds component angular slew"
                )
            norm_slew_limit = (
                float(np.max(slew_limits[enabled])) if np.any(enabled) else 0.0
            )
            if float(np.linalg.norm(delta)) > norm_slew_limit + tolerance:
                raise PositionAdmittanceActiveFault(
                    f"{side} final rotation modifier exceeds total angular slew"
                )
        self._last_composed_rotation_by_side[side] = rotvec.copy()
        self._last_composed_rotation_time_by_side[side] = float(now_sec)
        return rotvec

    def _check_nominal_target_alignment(
        self,
        *,
        side: str,
        raw_target: np.ndarray,
        candidate_offset_m: Sequence[float],
        previous_offset_m: Sequence[float] = (0.0, 0.0, 0.0),
        candidate_rotation_rotvec_rad: Sequence[float] = (0.0, 0.0, 0.0),
        previous_rotation_rotvec_rad: Sequence[float] = (0.0, 0.0, 0.0),
        now_sec: float,
    ) -> None:
        """Reject a nonzero overlay that could wake a distant base target."""

        candidate = np.asarray(candidate_offset_m, dtype=float).reshape(3)
        previous_offset = np.asarray(previous_offset_m, dtype=float).reshape(3)
        rotation_candidate = np.asarray(
            candidate_rotation_rotvec_rad,
            dtype=float,
        ).reshape(3)
        previous_rotation = np.asarray(
            previous_rotation_rotvec_rad,
            dtype=float,
        ).reshape(3)
        if not np.all(np.isfinite(previous_rotation)):
            raise PositionAdmittanceActiveFault(
                f"{side} previous rotation modifier is non-finite"
            )
        if not np.all(np.isfinite(previous_offset)):
            raise PositionAdmittanceActiveFault(
                f"{side} previous translation modifier is non-finite"
            )
        if (
            float(np.linalg.norm(candidate)) <= 1e-12
            and float(np.linalg.norm(rotation_candidate)) <= 1e-12
        ):
            return
        admitted = self._last_admitted_pose_by_side.get(side)
        observed_at = self._last_admitted_observed_at_by_side.get(side)
        if admitted is None or observed_at is None:
            raise PositionAdmittanceActiveFault(
                f"{side} nonzero Active modifier has no admitted-qref baseline"
            )
        age = float(now_sec) - float(observed_at)
        if age < 0.0 or age > float(self.config.stale_after_sec):
            raise PositionAdmittanceActiveFault(
                f"{side} admitted-qref baseline is stale: age={age:.6f}s"
            )

        # ``formal_teleop`` delegates ordinary nominal-target tracking lag to
        # the unchanged V2 Balanced IK, collision, joint, and qref/qcmd
        # admission chain.  The extra residual ranges below were introduced
        # only as a temporary Active bring-up guard; they are tighter than the
        # formal position-control recovery contract and can reject normal
        # motion while a bounded modifier is present.  Keep the admitted-qref
        # freshness requirement above and the authoritative final modifier
        # component/norm/slew checks before this point.
        if (
            self.active_endpoint_policy
            == POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        ):
            return

        target = np.asarray(raw_target, dtype=float)
        # The admitted qref belongs to the previous tick and already contains
        # the previous bounded translation modifier.  Remove only that known
        # same-frame incremental term before measuring ordinary nominal IK/
        # execution lag.  Never absorb the absolute target-to-admitted residual
        # into controller state; this value remains a fail-closed check only.
        nominalized_admitted_translation = (
            admitted[:3, 3] - previous_offset
        )
        translation_residual = float(
            np.linalg.norm(
                nominalized_admitted_translation - target[:3, 3]
            )
        )
        enabled = np.asarray(self.config.enabled_axes, dtype=bool)
        limits = np.asarray(self.config.max_offset_m, dtype=float)
        translation_limit = (
            POSITION_ADMITTANCE_ROTATION_ACTIVE_NOMINAL_TRANSLATION_LIMIT_M
            if self.rotational_config is not None and not np.any(enabled)
            else float(np.max(limits[enabled]))
            if np.any(enabled)
            else float(np.max(limits))
        )
        if translation_residual > translation_limit + 1e-9:
            raise PositionAdmittanceActiveFault(
                f"{side} nominal target-to-admitted translation residual "
                f"{translation_residual:.6f}m exceeds "
                f"{translation_limit:.6f}m"
            )
        composed_translation = target[:3, 3] + candidate
        composed_residual = float(
            np.linalg.norm(admitted[:3, 3] - composed_translation)
        )
        # Between adjacent ticks, only the bounded modifier delta is newly
        # requested.  The triangle bound therefore uses candidate-previous,
        # not the absolute candidate, and continues to reject any hidden
        # persistent or absolute-residual term.
        composed_limit = translation_limit + float(
            np.linalg.norm(candidate - previous_offset)
        )
        if composed_residual > composed_limit + 1e-9:
            raise PositionAdmittanceActiveFault(
                f"{side} composed target-to-admitted translation residual "
                f"{composed_residual:.6f}m exceeds "
                f"the nominal-plus-modifier bound {composed_limit:.6f}m"
            )

        # The admitted qref is the result of the previous tick and therefore
        # already contains the previous rotational modifier.  Remove that
        # known, bounded term before measuring nominal IK lag; otherwise every
        # valid modifier above this 3 degree lag tolerance is falsely treated
        # as a distant raw target.
        nominalized_admitted_rotation = compose_world_rotation_offset(
            admitted[:3, :3],
            -previous_rotation,
        )
        relative_rotation = (
            nominalized_admitted_rotation.T @ target[:3, :3]
        )
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        rotation_residual_deg = float(np.degrees(np.arccos(cosine)))
        if (
            rotation_residual_deg
            > POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG + 1e-9
        ):
            raise PositionAdmittanceActiveFault(
                f"{side} nominal target-to-admitted rotation residual "
                f"{rotation_residual_deg:.6f}deg exceeds "
                f"{POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG:.6f}deg"
            )
        composed_rotation = compose_world_rotation_offset(
            target[:3, :3],
            rotation_candidate,
        )
        relative_composed = admitted[:3, :3].T @ composed_rotation
        composed_cosine = float(
            np.clip((np.trace(relative_composed) - 1.0) * 0.5, -1.0, 1.0)
        )
        composed_rotation_residual_deg = float(
            np.degrees(np.arccos(composed_cosine))
        )
        modifier_step_rotation = (
            rotation_matrix_from_rotvec(previous_rotation).T
            @ rotation_matrix_from_rotvec(rotation_candidate)
        )
        modifier_step_cosine = float(
            np.clip(
                (np.trace(modifier_step_rotation) - 1.0) * 0.5,
                -1.0,
                1.0,
            )
        )
        modifier_step_deg = float(np.degrees(np.arccos(modifier_step_cosine)))
        composed_rotation_limit_deg = (
            POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG
            + modifier_step_deg
        )
        if composed_rotation_residual_deg > composed_rotation_limit_deg + 1e-9:
            raise PositionAdmittanceActiveFault(
                f"{side} composed target-to-admitted rotation residual "
                f"{composed_rotation_residual_deg:.6f}deg exceeds "
                f"the nominal-plus-modifier bound {composed_rotation_limit_deg:.6f}deg"
            )

    def _side_payload(
        self,
        *,
        side: str,
        reading: Mapping[str, object] | None,
        state: CartesianAdmittanceSnapshot,
        active: bool,
        activity_reason: str,
        target: np.ndarray | None,
        now_sec: float,
        integration_state: CartesianAdmittanceSnapshot | None = None,
        applied_offset_m: Sequence[float] | None = None,
        rotation_state: RotationalAdmittanceSnapshot | None = None,
        rotation_integration_state: RotationalAdmittanceSnapshot | None = None,
        applied_rotation_rotvec_rad: Sequence[float] | None = None,
    ) -> dict[str, object]:
        raw_values = (
            None
            if reading is None
            else [float(value) for value in reading.get("values", ())]
        )
        tare_values = self._tare_values_by_side.get(side)
        tared_values = None
        if raw_values is not None and tare_values is not None and len(raw_values) == 6:
            tared_values = [
                raw_values[index] - float(tare_values[index])
                for index in range(6)
            ]
        anchor_offset = np.zeros(3, dtype=float)
        would_apply_translation = None
        raw_target_pose_solver_world = None
        would_apply_target_pose_solver_world = None
        if target is not None:
            raw_target_pose_solver_world = np.asarray(
                target,
                dtype=float,
            ).copy()
            would_apply_target_pose_solver_world = (
                raw_target_pose_solver_world.copy()
            )
            would_apply_target_pose_solver_world[:3, 3] += (
                anchor_offset + np.asarray(state.output_offset_m, dtype=float)
            )
            if rotation_state is not None:
                would_apply_target_pose_solver_world[:3, :3] = (
                    compose_world_rotation_offset(
                        raw_target_pose_solver_world[:3, :3],
                        rotation_state.output_rotvec_rad,
                    )
                )
            would_apply_translation = (
                raw_target_pose_solver_world[:3, 3]
                + anchor_offset
                + np.asarray(state.output_offset_m, dtype=float)
            ).tolist()
        applied_offset = (
            [0.0, 0.0, 0.0]
            if applied_offset_m is None
            else [float(value) for value in applied_offset_m]
        )
        applied_rotation = (
            np.zeros(3, dtype=float)
            if applied_rotation_rotvec_rad is None
            else np.asarray(applied_rotation_rotvec_rad, dtype=float).reshape(3)
        )
        applied_target_translation = None
        applied_target_pose_solver_world = None
        if target is not None:
            applied_target_pose_solver_world = raw_target_pose_solver_world.copy()
            applied_target_pose_solver_world[:3, 3] += (
                anchor_offset + np.asarray(applied_offset, dtype=float)
            )
            applied_target_pose_solver_world[:3, :3] = (
                compose_world_rotation_offset(
                    raw_target_pose_solver_world[:3, :3],
                    applied_rotation,
                )
            )
            applied_target_translation = (
                raw_target_pose_solver_world[:3, 3]
                + anchor_offset
                + np.asarray(applied_offset, dtype=float)
            ).tolist()
        contact_axes = self._contact_axes_by_side[side]
        payload: dict[str, object] = {
            "active": bool(active),
            "activity_reason": str(activity_reason),
            "raw_wrench_n_nm": raw_values,
            "tared_wrench_n_nm": tared_values,
            "source": None if reading is None else reading.get("source"),
            "source_time_sec": (
                None if reading is None else reading.get("monotonic_time_sec")
            ),
            "sample_age_sec": state.last_sample_age_sec,
            "tare_generation": state.tare_generation,
            "requested_offset_m": list(state.requested_offset_m),
            "output_offset_m": list(state.output_offset_m),
            "velocity_m_s": list(state.velocity_m_s),
            "deadbanded_force_control_n": list(
                state.deadbanded_force_control_n
            ),
            "filtered_force_control_n": list(
                state.filtered_force_control_n
            ),
            "passive_return_active": bool(state.passive_return_active),
            "state_reason": state.reason,
            "sample_state_reason": (
                state.reason
                if integration_state is None
                else integration_state.reason
            ),
            "frozen": bool(state.frozen),
            "new_sample_integrated": bool(
                state.new_sample_integrated
                if integration_state is None
                else integration_state.new_sample_integrated
            ),
            "integration_dt_sec": (
                state.integration_dt_sec
                if integration_state is None
                else integration_state.integration_dt_sec
            ),
            "raw_target_pose_solver_world": (
                None
                if raw_target_pose_solver_world is None
                else raw_target_pose_solver_world.astype(float).tolist()
            ),
            "would_apply_target_translation": would_apply_translation,
            "would_apply_target_pose_solver_world": (
                None
                if would_apply_target_pose_solver_world is None
                else would_apply_target_pose_solver_world.astype(float).tolist()
            ),
            "anchor_offset_m": anchor_offset.astype(float).tolist(),
            "applied_offset_m": applied_offset,
            "applied_total_offset_m": (
                anchor_offset + np.asarray(applied_offset, dtype=float)
            ).astype(float).tolist(),
            "applied_target_translation": applied_target_translation,
            "applied_target_pose_solver_world": (
                None
                if applied_target_pose_solver_world is None
                else applied_target_pose_solver_world.astype(float).tolist()
            ),
            "contact_active": bool(np.any(contact_axes)),
            "contact_active_axes": [
                axis
                for axis, active_axis in zip(
                    POSITION_ADMITTANCE_SHADOW_AXES,
                    contact_axes,
                )
                if bool(active_axis)
            ],
            "contact_rebase_count": 0,
            "last_contact_rebase": None,
            "active_feedback_policy": "diagnostic_only_no_state_update",
            "active_qpos_fast_path_alpha": self.qpos_fast_path_alpha,
            "admission_feedback": self._admission_feedback_by_side.get(side),
            "admission_feedback_fault": self._admission_feedback_fault_by_side[side],
            "observed_at_monotonic_sec": float(now_sec),
        }
        if rotation_state is not None:
            sample_rotation_state = (
                rotation_state
                if rotation_integration_state is None
                else rotation_integration_state
            )
            filtered_torque = np.asarray(
                rotation_state.filtered_torque_control_nm,
                dtype=float,
            )
            deadbanded_torque = np.asarray(
                rotation_state.deadbanded_torque_control_nm,
                dtype=float,
            )
            rotation_config = self.rotational_config
            assert rotation_config is not None
            enabled_rotation_axes = np.asarray(
                rotation_config.enabled_axes,
                dtype=bool,
            )
            return_deadband = np.asarray(
                rotation_config.return_torque_deadband_nm,
                dtype=float,
            )
            rotation_contact_axes = enabled_rotation_axes & (
                (np.abs(deadbanded_torque) > 1e-12)
                | (np.abs(filtered_torque) > return_deadband + 1e-12)
            )
            if not active:
                rotation_contact_axes.fill(False)
            payload.update(
                {
                    "rotation_requested_rotvec_rad": list(
                        rotation_state.requested_rotvec_rad
                    ),
                    "rotation_output_rotvec_rad": list(
                        rotation_state.output_rotvec_rad
                    ),
                    "rotation_angular_velocity_rad_s": list(
                        rotation_state.angular_velocity_rad_s
                    ),
                    "deadbanded_torque_control_nm": list(
                        rotation_state.deadbanded_torque_control_nm
                    ),
                    "filtered_torque_control_nm": list(
                        rotation_state.filtered_torque_control_nm
                    ),
                    "rotation_state_reason": rotation_state.reason,
                    "rotation_sample_state_reason": sample_rotation_state.reason,
                    "rotation_frozen": bool(rotation_state.frozen),
                    "rotation_new_sample_integrated": bool(
                        sample_rotation_state.new_sample_integrated
                    ),
                    "rotation_integration_dt_sec": (
                        sample_rotation_state.integration_dt_sec
                    ),
                    "would_apply_rotation_rotvec_rad": list(
                        rotation_state.output_rotvec_rad
                    ),
                    "applied_rotation_rotvec_rad": applied_rotation
                    .astype(float)
                    .tolist(),
                    "rotation_contact_active": bool(
                        np.any(rotation_contact_axes)
                    ),
                    "rotation_contact_active_axes": [
                        axis
                        for axis, active_axis in zip(
                            POSITION_ADMITTANCE_SHADOW_AXES,
                            rotation_contact_axes,
                            strict=True,
                        )
                        if bool(active_axis)
                    ],
                }
            )
        return payload

    def _observe_side(
        self,
        *,
        side: str,
        target: np.ndarray,
        reading: Mapping[str, object] | None,
        now_sec: float,
        init_override_active: bool,
        force_guard_fault: bool,
    ) -> np.ndarray:
        controller = self.controllers[side]
        rotation_controller = self.rotation_controllers.get(side)
        active, activity_reason = self.side_activity(side)
        feedback_fault = self._admission_feedback_fault_by_side[side]
        if init_override_active:
            active, activity_reason = False, "init_arm_override"
        elif force_guard_fault:
            active, activity_reason = False, "position_force_guard_fault"
        elif feedback_fault is not None:
            active, activity_reason = False, "admission_feedback_fault"
        if not active:
            controller.freeze(activity_reason)
            if rotation_controller is not None:
                rotation_controller.freeze(activity_reason)
            self._mode_active_by_side[side] = False
        elif not self._mode_active_by_side[side]:
            # A Hold/fixed/Init interval owns its own target anchor.  Never
            # re-apply a pre-Hold force offset when the side becomes live.
            controller.rebase_realized_offset(
                (0.0, 0.0, 0.0),
                output_time_sec=now_sec,
            )
            if rotation_controller is not None:
                rotation_controller.rebase_realized_rotation(
                    (0.0, 0.0, 0.0),
                    output_time_sec=now_sec,
                )
            self._contact_axes_by_side[side].fill(False)
            self._pending_raw_target_by_side.pop(side, None)
            self._admission_feedback_by_side.pop(side, None)
            self._last_composed_offset_by_side[side] = np.zeros(3, dtype=float)
            self._last_composed_time_by_side[side] = now_sec
            self._last_composed_rotation_by_side[side] = np.zeros(3, dtype=float)
            self._last_composed_rotation_time_by_side[side] = now_sec
            self._last_admitted_pose_by_side.pop(side, None)
            self._last_admitted_observed_at_by_side.pop(side, None)
            controller.resume()
            if rotation_controller is not None:
                rotation_controller.resume()
            self._mode_active_by_side[side] = True
        elif (
            controller.snapshot(now_sec=now_sec).frozen
            or (
                rotation_controller is not None
                and rotation_controller.snapshot(now_sec=now_sec).frozen
            )
        ) and reading is not None:
            # A short force-sample gap may be tighter than the independent
            # emergency-stop watchdog.  Re-arm the sample clock while keeping
            # the last slew-limited offset continuous; the first fresh sample
            # is never integrated across the frozen interval.
            controller.resume()
            if (
                rotation_controller is not None
                and rotation_controller.snapshot(now_sec=now_sec).frozen
            ):
                rotation_controller.resume()

        if reading is None:
            controller.freeze("force_reading_missing")
            if rotation_controller is not None:
                rotation_controller.freeze("force_reading_missing")
            state = controller.snapshot(now_sec=now_sec)
            integration_state = None
            rotation_state = (
                None
                if rotation_controller is None
                else rotation_controller.snapshot(now_sec=now_sec)
            )
            rotation_integration_state = None
        elif active and not init_override_active:
            generation = self._tare_generation_by_side[side]
            sample = CartesianAdmittanceWrenchSample(
                values_n_nm=tuple(float(value) for value in reading.get("values", ())),
                frame="tool",
                source_time_sec=float(reading["monotonic_time_sec"]),
                receive_time_sec=now_sec,
                tare_generation=generation,
            )
            target_rotation = np.asarray(target, dtype=float)[:3, :3]
            rotation_control_from_tool = (
                target_rotation @ self._tool_rotation_from_solver[side]
            )
            integration_state = controller.step_new_sample(
                sample,
                now_sec=now_sec,
                rotation_control_from_sample=rotation_control_from_tool,
                active=True,
            )
            state = controller.sample_output(now_sec=now_sec, active=True)
            if rotation_controller is None:
                rotation_state = None
                rotation_integration_state = None
            else:
                rotation_integration_state = rotation_controller.step_new_sample(
                    sample,
                    now_sec=now_sec,
                    rotation_control_from_sample=rotation_control_from_tool,
                    active=True,
                )
                rotation_state = rotation_controller.sample_output(
                    now_sec=now_sec,
                    active=True,
                )
        else:
            state = controller.snapshot(now_sec=now_sec)
            integration_state = None
            rotation_state = (
                None
                if rotation_controller is None
                else rotation_controller.snapshot(now_sec=now_sec)
            )
            rotation_integration_state = None

        enabled_axes = np.asarray(self.config.enabled_axes, dtype=bool)
        return_deadband = np.asarray(
            self.config.return_force_deadband_n,
            dtype=float,
        )
        deadbanded_force = np.asarray(
            state.deadbanded_force_control_n,
            dtype=float,
        )
        filtered_force = np.asarray(
            state.filtered_force_control_n,
            dtype=float,
        )
        contact_axes = enabled_axes & (
            (np.abs(deadbanded_force) > 1e-12)
            | (np.abs(filtered_force) > return_deadband + 1e-12)
        )
        if not active:
            contact_axes.fill(False)
        self._contact_axes_by_side[side] = contact_axes.copy()

        applied_offset = np.zeros(3, dtype=float)
        applied_rotation = np.zeros(3, dtype=float)
        result = np.asarray(target, dtype=float).copy()
        if (
            self.apply_to_target
            and side in self.apply_sides
            and active
            and not init_override_active
        ):
            previous_applied_offset = self._last_composed_offset_by_side.get(
                side,
                np.zeros(3, dtype=float),
            ).copy()
            previous_applied_rotation = self._last_composed_rotation_by_side.get(
                side,
                np.zeros(3, dtype=float),
            ).copy()
            applied_offset = self._checked_final_modifier(
                side=side,
                state=state,
                now_sec=now_sec,
            )
            if rotation_state is not None:
                applied_rotation = self._checked_final_rotation_modifier(
                    side=side,
                    state=rotation_state,
                    now_sec=now_sec,
                )
            self._check_nominal_target_alignment(
                side=side,
                raw_target=target,
                candidate_offset_m=applied_offset,
                previous_offset_m=previous_applied_offset,
                candidate_rotation_rotvec_rad=applied_rotation,
                previous_rotation_rotvec_rad=previous_applied_rotation,
                now_sec=now_sec,
            )
            result[:3, 3] += applied_offset
            result[:3, :3] = compose_world_rotation_offset(
                result[:3, :3],
                applied_rotation,
            )
            self._pending_raw_target_by_side[side] = np.asarray(
                target,
                dtype=float,
            ).copy()
        else:
            self._pending_raw_target_by_side.pop(side, None)

        self._latest_by_side[side] = self._side_payload(
            side=side,
            reading=reading,
            state=state,
            active=active and not init_override_active,
            activity_reason=activity_reason,
            target=target,
            now_sec=now_sec,
            integration_state=integration_state,
            applied_offset_m=applied_offset,
            rotation_state=rotation_state,
            rotation_integration_state=rotation_integration_state,
            applied_rotation_rotvec_rad=applied_rotation,
        )
        return result

    def observe_admitted_poses(
        self,
        admitted_pose_by_side: Mapping[str, np.ndarray],
        *,
        now_sec: float | None = None,
    ) -> dict[str, object]:
        """Observe final admitted-qref FK without mutating Active state.

        C0 intentionally has no admission-driven anchor or anti-windup state.
        The admitted command may contain ordinary hold, smoothing, branch, and
        rate-limit residuals, so its absolute difference from the raw target is
        diagnostic only and can never become a modifier update.
        """

        now = time.monotonic() if now_sec is None else float(now_sec)
        if not self.apply_to_target:
            return self.snapshot(now_sec=now)
        for side in self.apply_sides:
            raw_target = self._pending_raw_target_by_side.get(side)
            if raw_target is None:
                raise RuntimeError(f"{side} has no active admittance target for feedback")
            if side not in admitted_pose_by_side:
                raise RuntimeError(f"{side} admitted-qref FK is missing")
            admitted_pose = np.asarray(
                admitted_pose_by_side[side],
                dtype=float,
            )
            if admitted_pose.shape != (4, 4) or not np.all(np.isfinite(admitted_pose)):
                raise ValueError(f"{side} admitted-qref FK must be a finite 4x4 pose")

            state_before = self.controllers[side].snapshot(now_sec=now)
            deadbanded = np.asarray(
                state_before.deadbanded_force_control_n,
                dtype=float,
            )
            active_axes = (
                np.asarray(self.config.enabled_axes, dtype=bool)
                & (np.abs(deadbanded) > 1e-12)
            )
            admitted_transient = np.asarray(
                state_before.output_offset_m,
                dtype=float,
            )
            raw_realized = admitted_pose[:3, 3] - raw_target[:3, 3]
            composed_translation = raw_target[:3, 3] + admitted_transient
            modifier_tracking_residual = (
                admitted_pose[:3, 3] - composed_translation
            )
            zero = np.zeros(3, dtype=float)

            feedback = {
                "observed": True,
                "reason": "diagnostic_only_no_state_update",
                "frame": "solver_world",
                "at_monotonic_sec": float(now),
                "active_axes": [
                    axis
                    for axis, active_axis in zip(
                        POSITION_ADMITTANCE_SHADOW_AXES,
                        active_axes,
                    )
                    if bool(active_axis)
                ],
                "guarded_axes": [],
                "antiwindup_applied": False,
                "state_update_applied": False,
                "feedback_baseline": "none_absolute_residual_diagnostic_only",
                "raw_realized_offset_m": raw_realized.astype(float).tolist(),
                "modifier_tracking_residual_m": modifier_tracking_residual
                .astype(float)
                .tolist(),
                "admitted_transient_offset_m": admitted_transient.astype(float).tolist(),
                "anchor_before_m": zero.tolist(),
                "anchor_correction_m": zero.tolist(),
                "anchor_after_m": zero.tolist(),
                "admitted_target_translation": admitted_pose[:3, 3].astype(float).tolist(),
            }
            rotation_controller = self.rotation_controllers.get(side)
            if rotation_controller is not None:
                rotation_state = rotation_controller.snapshot(now_sec=now)
                rotation_rotvec = np.asarray(
                    rotation_state.output_rotvec_rad,
                    dtype=float,
                )
                composed_rotation = compose_world_rotation_offset(
                    raw_target[:3, :3],
                    rotation_rotvec,
                )

                def rotation_error_deg(
                    reference: np.ndarray,
                    value: np.ndarray,
                ) -> float:
                    relative = reference.T @ value
                    cosine = float(
                        np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
                    )
                    return float(np.degrees(np.arccos(cosine)))

                feedback.update(
                    {
                        "admitted_rotation_modifier_rotvec_rad": rotation_rotvec
                        .astype(float)
                        .tolist(),
                        "raw_target_to_admitted_rotation_deg": rotation_error_deg(
                            raw_target[:3, :3],
                            admitted_pose[:3, :3],
                        ),
                        "rotation_modifier_tracking_residual_deg": rotation_error_deg(
                            composed_rotation,
                            admitted_pose[:3, :3],
                        ),
                    }
                )
            self._admission_feedback_by_side[side] = feedback
            self._last_admitted_pose_by_side[side] = admitted_pose.copy()
            self._last_admitted_observed_at_by_side[side] = now
            latest = self._latest_by_side.get(side)
            if latest is not None:
                latest.update(
                    {
                        "anchor_offset_m": zero.tolist(),
                        "applied_offset_m": admitted_transient.astype(float).tolist(),
                        "applied_total_offset_m": admitted_transient
                        .astype(float)
                        .tolist(),
                        "would_apply_target_translation": composed_translation
                        .astype(float)
                        .tolist(),
                        "applied_target_translation": composed_translation
                        .astype(float)
                        .tolist(),
                        "admission_feedback": feedback,
                    }
                )
        return self.snapshot(now_sec=now)

    def target_activity_override_by_side(self) -> dict[str, dict[str, object]]:
        """Declare only an actually applied nonzero Active modifier as motion.

        Position teleop's tracker inactivity hold must not erase an authorized
        force-driven target change merely because the operator's hand is
        still.  Shadow never publishes an override, and fixed/clutched sides
        remain inactive through ``side_activity`` and therefore cannot enter
        this result.  The retargeter still owns IK, collision, branch, and
        final qref admission.
        """

        if not self.apply_to_target:
            return {}
        overrides: dict[str, dict[str, object]] = {}
        for side in self.apply_sides:
            payload = self._latest_by_side.get(side)
            if not isinstance(payload, Mapping):
                continue
            offset = np.asarray(
                payload.get("applied_total_offset_m", (0.0, 0.0, 0.0)),
                dtype=float,
            ).reshape(3)
            rotation_offset = np.asarray(
                payload.get(
                    "applied_rotation_rotvec_rad",
                    (0.0, 0.0, 0.0),
                ),
                dtype=float,
            ).reshape(3)
            active = bool(payload.get("active", False))
            offset_norm = float(np.linalg.norm(offset))
            rotation_offset_norm = float(np.linalg.norm(rotation_offset))
            if (
                not active
                or not self._mode_active_by_side.get(side, False)
                or not np.all(np.isfinite(offset))
                or not np.all(np.isfinite(rotation_offset))
                or (
                    offset_norm <= 1e-12
                    and rotation_offset_norm <= 1e-12
                )
            ):
                self._rotation_contact_seen_by_side[side] = False
                self._rotation_release_started_at_by_side[side] = None
                continue
            modifier_qpos_alpha = self.qpos_fast_path_alpha
            translation_contact_active = bool(
                payload.get("contact_active", False)
            )
            translation_release_active = bool(
                self.rotational_config is not None
                and np.any(np.asarray(self.config.enabled_axes, dtype=bool))
                and not translation_contact_active
                and offset_norm > 1e-12
            )
            rotation_contact_active = bool(
                payload.get("rotation_contact_active", False)
            )
            if self.rotational_config is not None:
                if rotation_contact_active:
                    self._rotation_contact_seen_by_side[side] = True
                    self._rotation_release_started_at_by_side[side] = None
                elif self._rotation_contact_seen_by_side[side]:
                    observed_at = payload.get("observed_at_monotonic_sec")
                    try:
                        observed_at_sec = float(observed_at)
                    except (TypeError, ValueError):
                        observed_at_sec = float("nan")
                    if np.isfinite(observed_at_sec):
                        release_started_at = (
                            self._rotation_release_started_at_by_side[side]
                        )
                        if release_started_at is None:
                            release_started_at = observed_at_sec
                            self._rotation_release_started_at_by_side[side] = (
                                release_started_at
                            )
                        if self.rotation_release_qpos_ramp_sec <= 0.0:
                            release_fraction = 1.0
                        else:
                            release_fraction = float(
                                np.clip(
                                    (observed_at_sec - release_started_at)
                                    / self.rotation_release_qpos_ramp_sec,
                                    0.0,
                                    1.0,
                                )
                            )
                        modifier_qpos_alpha = self.qpos_fast_path_alpha + (
                            release_fraction
                            * (
                                self.rotation_release_qpos_fast_path_alpha
                                - self.qpos_fast_path_alpha
                            )
                        )
                        if release_fraction >= 1.0:
                            self._rotation_contact_seen_by_side[side] = False
                            self._rotation_release_started_at_by_side[side] = None
                    else:
                        # Activity metadata without a usable causal timestamp
                        # must not leave release response indefinitely slow.
                        modifier_qpos_alpha = (
                            self.rotation_release_qpos_fast_path_alpha
                        )
                        self._rotation_contact_seen_by_side[side] = False
                        self._rotation_release_started_at_by_side[side] = None
                else:
                    modifier_qpos_alpha = (
                        self.rotation_release_qpos_fast_path_alpha
                    )
            if translation_release_active:
                modifier_qpos_alpha = max(
                    modifier_qpos_alpha,
                    POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA,
                )
            overrides[side] = {
                "source": "position_admittance_active_modifier",
                "state": "ACTIVE",
                "inactive": False,
                "hold_weight": 0.0,
                "modifier_active": True,
                "modifier_offset_norm_m": offset_norm,
                "modifier_qpos_fast_path_alpha": modifier_qpos_alpha,
            }
            if self.rotational_config is not None:
                overrides[side]["modifier_rotation_offset_norm_rad"] = (
                    rotation_offset_norm
                )
                overrides[side]["modifier_phase"] = (
                    "contact"
                    if rotation_contact_active
                    else "return"
                )
        return overrides

    def __call__(
        self,
        targets: Mapping[str, np.ndarray],
        observation: Any,
        calibration: Any,
        *,
        now_sec: float | None = None,
    ) -> Mapping[str, np.ndarray] | None:
        prior_result = (
            None
            if self.previous_filter is None
            else self.previous_filter(targets, observation, calibration)
        )
        effective_targets = targets if prior_result is None else prior_result
        init_override_active = prior_result is not None
        now = time.monotonic() if now_sec is None else float(now_sec)
        if not np.isfinite(now):
            raise ValueError("position admittance target time must be finite")
        applied_targets: dict[str, np.ndarray] = {}
        try:
            monitor_snapshot = self.monitor.snapshot()
            readings = self._reading_by_side(monitor_snapshot)
            force_guard_fault = monitor_snapshot.get("fault") is not None
            if force_guard_fault:
                for controller in self.controllers.values():
                    controller.freeze("position_force_guard_fault")
                for controller in self.rotation_controllers.values():
                    controller.freeze("position_force_guard_fault")
            for side in self.active_sides:
                target = np.asarray(effective_targets[side], dtype=float).copy()
                reading = readings.get(side)
                applied_targets[side] = self._observe_side(
                    side=side,
                    target=target,
                    reading=reading if isinstance(reading, Mapping) else None,
                    now_sec=now,
                    init_override_active=init_override_active,
                    force_guard_fault=force_guard_fault,
                )
            self._last_error = None
        except Exception as exc:  # noqa: BLE001 - retain the existing target path.
            self._last_error = f"{type(exc).__name__}: {exc}"
            for controller in self.controllers.values():
                controller.freeze("admittance_observer_error")
            for controller in self.rotation_controllers.values():
                controller.freeze("admittance_observer_error")
            if self.apply_to_target:
                self.latch_admission_feedback_fault(self._last_error)
                if isinstance(exc, PositionAdmittanceActiveFault):
                    raise
                raise PositionAdmittanceActiveFault(self._last_error) from exc
        self._frame_count += 1
        if not self.apply_to_target or init_override_active:
            # Preserve both the object and historical None/override semantics
            # for shadow mode and every authoritative InitArm override.
            return prior_result
        if len(applied_targets) != len(self.active_sides):
            # An Active observer failure cannot invent a partial target or
            # silently fall back to a potentially distant base target.
            raise PositionAdmittanceActiveFault(
                "active admittance did not produce one complete dual-arm target pair"
            )
        return applied_targets

    def snapshot(self, *, now_sec: float | None = None) -> dict[str, object]:
        now = time.monotonic() if now_sec is None else float(now_sec)
        nominal_alignment_range_guard_enabled = (
            self.active_endpoint_policy
            != POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP
        )
        nominal_translation_limit_m = (
            POSITION_ADMITTANCE_ROTATION_ACTIVE_NOMINAL_TRANSLATION_LIMIT_M
            if self.rotational_config is not None
            and not any(self.config.enabled_axes)
            else max(
                limit
                for limit, enabled in zip(
                    self.config.max_offset_m,
                    self.config.enabled_axes,
                    strict=True,
                )
                if enabled
            )
            if any(self.config.enabled_axes)
            else 0.0
        )
        sides: dict[str, dict[str, object]] = {}
        for side in self.active_sides:
            latest = self._latest_by_side.get(side)
            if latest is None:
                latest = self._side_payload(
                    side=side,
                    reading=None,
                    state=self.controllers[side].snapshot(now_sec=now),
                    active=False,
                    activity_reason="not_observed",
                    target=None,
                    now_sec=now,
                    rotation_state=(
                        None
                        if side not in self.rotation_controllers
                        else self.rotation_controllers[side].snapshot(now_sec=now)
                    ),
                )
            sides[side] = dict(latest)
        payload: dict[str, object] = {
            "schema_version": POSITION_ADMITTANCE_SCHEMA_VERSION,
            "mode": "active" if self.apply_to_target else "shadow",
            "shadow_only": not self.apply_to_target,
            "applied_to_target": self.apply_to_target,
            "apply_sides": list(self.apply_sides),
            "active_envelope": self.active_envelope,
            "active_endpoint_policy": self.active_endpoint_policy,
            "active_nominal_alignment_range_guard_enabled": (
                nominal_alignment_range_guard_enabled
            ),
            "target_identity_preserved": not self.apply_to_target,
            "frame_count": int(self._frame_count),
            "last_error": self._last_error,
            "active_feedback_policy": "diagnostic_only_no_state_update",
            "active_qpos_fast_path_alpha": self.qpos_fast_path_alpha,
            "rotation_release_qpos_fast_path_alpha": (
                self.rotation_release_qpos_fast_path_alpha
            ),
            "translation_release_qpos_fast_path_alpha": (
                POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA
            ),
            "rotation_release_qpos_ramp_sec": (
                self.rotation_release_qpos_ramp_sec
            ),
            "composition": {
                "frame": "solver_world",
                "equation": "p_applied = p_raw + output_offset",
                "transient_terms": ["output_offset_m"],
                "persistent_terms": [],
                "final_modifier_field": "applied_total_offset_m",
                "final_envelope": (
                    "max_offset_m_and_max_output_slew_m_s"
                ),
            },
            "active_nominal_translation_limit_m": (
                nominal_translation_limit_m
                if nominal_alignment_range_guard_enabled
                else None
            ),
            "active_nominal_rotation_limit_deg": (
                POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG
                if nominal_alignment_range_guard_enabled
                else None
            ),
            "config": asdict(self.config),
            "sides": sides,
        }
        if self.rotational_config is not None:
            rotation_config = self.rotational_config
            payload["schema_version"] = POSITION_ADMITTANCE_ROTATION_SCHEMA_VERSION
            payload["composition"] = {
                "frame": "solver_world",
                "equations": [
                    "p_applied = p_raw + output_offset",
                    "R_applied = Exp(rotation_output_rotvec) @ R_raw",
                ],
                "transient_terms": [
                    "output_offset_m",
                    "rotation_output_rotvec_rad",
                ],
                "persistent_terms": [],
                "final_modifier_fields": [
                    "applied_total_offset_m",
                    "applied_rotation_rotvec_rad",
                ],
                "final_envelope": (
                    "translation_and_rotation_component_norm_and_slew_limits"
                ),
            }
            payload["rotational_config"] = asdict(rotation_config)
            try:
                active_rotation_profile = (
                    position_rotational_admittance_active_profile(rotation_config)
                )
            except ValueError:
                active_rotation_profile = None
            if self.apply_to_target and (
                self.active_envelope
                == POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY
                and active_rotation_profile in {
                    POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE,
                    POSITION_ADMITTANCE_ROTATION_ACTIVE_EMPHASIS_PROFILE,
                }
            ):
                profile_slug = (
                    "balanced_5deg"
                    if active_rotation_profile
                    == POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE
                    else "rotation_emphasis_5deg"
                )
                authority_scope = (
                    "dual_side"
                    if set(self.apply_sides) == {"left", "right"}
                    else "single_side"
                )
                payload["rotation_authority"] = (
                    "production_active_normal_parity_"
                    f"{profile_slug}_translation_rotation_{authority_scope}"
                )
            elif self.apply_to_target and any(self.config.enabled_axes):
                authority_scope = (
                    "dual_side"
                    if set(self.apply_sides) == {"left", "right"}
                    else "single_side"
                )
                payload["rotation_authority"] = (
                    "production_active_cautious_translation_rotation_"
                    + authority_scope
                )
            elif self.apply_to_target:
                authority_scope = (
                    "dual_side"
                    if set(self.apply_sides) == {"left", "right"}
                    else "single_side"
                )
                payload["rotation_authority"] = (
                    "production_active_cautious_3deg_" + authority_scope
                )
            else:
                payload["rotation_authority"] = "shadow_or_device_free_replay_only"
            enabled = np.asarray(rotation_config.enabled_axes, dtype=bool)
            payload["rotation_modifier_limit_deg"] = (
                float(
                    np.degrees(
                        max(
                            limit
                            for limit, is_enabled in zip(
                                rotation_config.max_offset_rad,
                                enabled,
                                strict=True,
                            )
                            if is_enabled
                        )
                    )
                )
                if np.any(enabled)
                else 0.0
            )
        return payload


def install_position_admittance_shadow(
    *,
    retargeter: Any,
    monitor: Any,
    active_sides: Sequence[str],
    config: CartesianAdmittanceConfig,
    rotational_config: RotationalAdmittanceConfig | None = None,
    side_activity: Callable[[str], tuple[bool, str]],
    now_sec: float | None = None,
) -> PositionAdmittanceShadowTargetFilter:
    """Atomically install the observer after its initial tare succeeds."""

    tool_tcp_from_solver = getattr(retargeter, "_tool_tcp_from_solver", {})
    rotations = {
        side: np.asarray(tool_tcp_from_solver.get(side, np.eye(4)), dtype=float)[:3, :3]
        for side in active_sides
    }
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=active_sides,
        config=config,
        rotational_config=rotational_config,
        previous_filter=getattr(retargeter, "target_override_filter", None),
        side_activity=side_activity,
        tool_rotation_from_solver_by_side=rotations,
        apply_to_target=False,
        apply_sides=(),
    )
    observer.reset_from_monitor(now_sec=now_sec)
    retargeter.target_override_filter = observer
    return observer


def install_position_admittance_active(
    *,
    retargeter: Any,
    monitor: Any,
    active_sides: Sequence[str],
    config: CartesianAdmittanceConfig,
    rotational_config: RotationalAdmittanceConfig | None = None,
    side_activity: Callable[[str], tuple[bool, str]],
    apply_sides: Sequence[str] | None = None,
    active_envelope: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE,
    combined_translation_rotation_active: bool = False,
    endpoint_policy: str = POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY,
    now_sec: float | None = None,
) -> PositionAdmittanceShadowTargetFilter:
    """Atomically install the explicit, default-off active target modifier."""

    resolved_apply_sides = (
        tuple(str(side) for side in active_sides)
        if apply_sides is None
        else tuple(str(side) for side in apply_sides)
    )
    validate_position_admittance_active_authority(
        config,
        active_envelope=active_envelope,
        rotational_config=rotational_config,
        combined_translation_rotation_active=combined_translation_rotation_active,
        endpoint_policy=endpoint_policy,
        apply_sides=resolved_apply_sides,
    )
    tool_tcp_from_solver = getattr(retargeter, "_tool_tcp_from_solver", {})
    rotations = {
        side: np.asarray(tool_tcp_from_solver.get(side, np.eye(4)), dtype=float)[:3, :3]
        for side in active_sides
    }
    observer = PositionAdmittanceShadowTargetFilter(
        monitor=monitor,
        active_sides=active_sides,
        config=config,
        rotational_config=rotational_config,
        previous_filter=getattr(retargeter, "target_override_filter", None),
        side_activity=side_activity,
        tool_rotation_from_solver_by_side=rotations,
        apply_to_target=True,
        apply_sides=resolved_apply_sides,
        qpos_fast_path_alpha=(
            POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA
            if rotational_config is not None
            else None
        ),
        active_envelope=active_envelope,
        active_endpoint_policy=endpoint_policy,
    )
    observer.reset_from_monitor(now_sec=now_sec)
    retargeter.target_override_filter = observer
    return observer


def install_resolved_position_admittance_profile(
    *,
    profile: ResolvedPositionAdmittanceProfile,
    retargeter: Any,
    monitor: Any,
    active_sides: Sequence[str],
    side_activity: Callable[[str], tuple[bool, str]],
    now_sec: float | None = None,
) -> PositionAdmittanceShadowTargetFilter:
    """Install one already-resolved profile without rebuilding it from args."""

    if not profile.requested or profile.translation_config is None:
        raise ValueError("cannot install a disabled position admittance profile")
    if profile.active:
        return install_position_admittance_active(
            retargeter=retargeter,
            monitor=monitor,
            active_sides=active_sides,
            config=profile.translation_config,
            rotational_config=profile.rotational_config,
            side_activity=side_activity,
            apply_sides=profile.apply_sides,
            active_envelope=profile.active_envelope,
            combined_translation_rotation_active=profile.combined_active,
            endpoint_policy=profile.endpoint_policy,
            now_sec=now_sec,
        )
    return install_position_admittance_shadow(
        retargeter=retargeter,
        monitor=monitor,
        active_sides=active_sides,
        config=profile.translation_config,
        rotational_config=profile.rotational_config,
        side_activity=side_activity,
        now_sec=now_sec,
    )


__all__ = [
    "POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENVELOPE",
    "POSITION_ADMITTANCE_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_M",
    "POSITION_ADMITTANCE_ACTIVE_DEFAULT_ENDPOINT_POLICY",
    "POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_CHOICES",
    "POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_FORMAL_TELEOP",
    "POSITION_ADMITTANCE_ACTIVE_ENDPOINT_POLICY_RUNTIME_HOME_HOLD",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_DEFAULT_MAX_TOOL_EXCURSION_DEG",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_NOMINAL_TRANSLATION_LIMIT_M",
    "POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CAUTIOUS",
    "POSITION_ADMITTANCE_ACTIVE_ENVELOPE_CHOICES",
    "POSITION_ADMITTANCE_ACTIVE_ENVELOPE_HALF_NORMAL",
    "POSITION_ADMITTANCE_ACTIVE_ENVELOPE_NORMAL_PARITY",
    "POSITION_ADMITTANCE_ACTIVE_ENVELOPES",
    "POSITION_ADMITTANCE_ACCEPTED_FILTER_TAU_SEC",
    "POSITION_ADMITTANCE_ACCEPTED_TRANSLATION_STIFFNESS_N_M",
    "POSITION_ADMITTANCE_ACTIVE_FILTER_TAU_SEC_MAX",
    "POSITION_ADMITTANCE_ACTIVE_MAX_STALE_SEC",
    "POSITION_ADMITTANCE_ACTIVE_QPOS_FAST_PATH_ALPHA",
    "POSITION_ADMITTANCE_ACTIVE_HIGH_SLEW_QPOS_FAST_PATH_ALPHA",
    "POSITION_ADMITTANCE_ACTIVE_HIGH_SLEW_THRESHOLD_M_S",
    "POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_HOLD_DEG",
    "POSITION_ADMITTANCE_ACTIVE_QPOS_MICRO_REVERSAL_CREEP_DEG",
    "POSITION_ADMITTANCE_ACTIVE_SIDE_CHOICES",
    "POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MAX",
    "POSITION_ADMITTANCE_ACTIVE_RESPONSE_WEIGHT_MIN",
    "POSITION_ADMITTANCE_ACTIVE_OUTPUT_FOLLOW_TAU_SEC_MAX",
    "POSITION_ADMITTANCE_ACTIVE_RETURN_RATE_S_MAX",
    "POSITION_ADMITTANCE_ACTIVE_NORMAL_PARITY_RETURN_RATE_S",
    "POSITION_ADMITTANCE_ACTIVE_NORMAL_PARITY_RETURN_VELOCITY_M_S",
    "POSITION_ADMITTANCE_ACTIVE_HALF_NORMAL_RETURN_RATE_S",
    "POSITION_ADMITTANCE_ACTIVE_HALF_NORMAL_RETURN_VELOCITY_M_S",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_QPOS_FAST_PATH_ALPHA",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_MAX_TOOL_EXCURSION_DEG",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_BALANCED_PROFILE",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_EMPHASIS_PROFILE",
    "POSITION_ADMITTANCE_ROTATION_ACTIVE_PROFILE_CHOICES",
    "POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_FAST_PATH_ALPHA",
    "POSITION_ADMITTANCE_TRANSLATION_RELEASE_QPOS_FAST_PATH_ALPHA",
    "POSITION_ADMITTANCE_ROTATION_RELEASE_QPOS_RAMP_SEC",
    "POSITION_ADMITTANCE_RESPONSE_WEIGHT_DEFAULT",
    "POSITION_ADMITTANCE_SCHEMA_VERSION",
    "POSITION_ADMITTANCE_ACTIVE_NOMINAL_ROTATION_LIMIT_DEG",
    "POSITION_ADMITTANCE_SHADOW_AXES",
    "PositionAdmittanceActiveFault",
    "PositionAdmittanceEndpointExcursionGuard",
    "PositionAdmittanceShadowTargetFilter",
    "admitted_solver_world_poses_from_joint_command",
    "install_position_admittance_active",
    "install_resolved_position_admittance_profile",
    "install_position_admittance_shadow",
    "ResolvedPositionAdmittanceProfile",
    "resolve_position_admittance_profile",
    "position_admittance_shadow_config",
    "position_admittance_active_config",
    "position_admittance_active_apply_sides",
    "position_admittance_final_translation_slew_rate_m_s",
    "position_admittance_final_translation_slew_ratio",
    "position_admittance_active_qpos_fast_path_alpha",
    "position_admittance_active_uses_home_endpoint_guard",
    "position_admittance_rotation_response_weight",
    "position_admittance_translation_response_weight",
    "position_rotational_admittance_active_profile",
    "position_rotational_admittance_config",
    "validate_position_admittance_active_config",
    "validate_position_admittance_active_authority",
    "validate_position_rotational_admittance_active_config",
]
