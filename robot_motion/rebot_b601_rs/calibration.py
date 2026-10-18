"""Offline, condition-specific command identification. No actuator I/O.

The fitted quantity is a software command residual, not measured friction or
human force. A local model is bound to one run, joint, gravity profile and
observed six-axis pose/load region. Nothing here installs compensation.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .joint_motion import B601WindowVelocityEstimator
from .model import B601_RS_ARM_JOINT_NAMES


def finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def six(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != 6:
        raise ValueError(f"{name} requires six values")
    return tuple(finite(v, name) for v in value)


def joint_index(name: str) -> int:
    if name not in B601_RS_ARM_JOINT_NAMES:
        raise ValueError("calibration supports joint1..joint6 only")
    return B601_RS_ARM_JOINT_NAMES.index(name)


@dataclass(frozen=True)
class CalibrationCondition:
    # Run-local scope prevents pooling an unobserved payload/mounting change.
    run_id: str
    config_sha256: str
    urdf_sha256: str
    gravity_scale: float
    gravity_joint_scales: tuple[float, ...]
    gravity_joint_offsets_rad: tuple[float, ...]
    gravity_joint_bias_nm: tuple[float, ...]
    gravity_coulomb_friction_nm: tuple[float, ...]
    gravity_coulomb_velocity_threshold_rad_s: float
    driver_kd_nm_s_per_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError(
                "run_id is required; unknown physical setups cannot be pooled"
            )
        for name in ("config_sha256", "urdf_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"{name} requires existing source provenance")
        for name in ("gravity_scale", "gravity_coulomb_velocity_threshold_rad_s"):
            value = finite(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "gravity_joint_scales",
            "gravity_joint_offsets_rad",
            "gravity_joint_bias_nm",
            "gravity_coulomb_friction_nm",
            "driver_kd_nm_s_per_rad",
        ):
            values = six(getattr(self, name), name)
            if (
                name
                in (
                    "gravity_joint_scales",
                    "gravity_coulomb_friction_nm",
                    "driver_kd_nm_s_per_rad",
                )
                and min(values) < 0
            ):
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, values)

    @classmethod
    def from_mapping(
        cls, profile: Mapping[str, object], *, run_id: str
    ) -> CalibrationCondition:
        if not isinstance(profile, Mapping):
            raise TypeError("calibration context must be an object")
        names = set(cls.__dataclass_fields__) - {"run_id"}
        missing = names - profile.keys()
        if missing:
            raise ValueError(f"missing calibration context: {sorted(missing)}")
        return cls(run_id=run_id, **{name: profile[name] for name in names})


@dataclass(frozen=True)
class CalibrationObservation:
    condition: CalibrationCondition
    joint_name: str
    trial_id: str
    qpos_rad: tuple[float, ...]
    gravity_command_nm: tuple[float, ...]
    velocity_rad_s: float
    command_residual_nm: float
    # Actual motor-local damping during this window, distinct from passive
    # gravity-profile damping. Never infer its physical torque from a cache.
    active_driver_kd_nm_s_per_rad: tuple[float, ...] = (0.0,) * 6

    def __post_init__(self) -> None:
        if not isinstance(self.condition, CalibrationCondition):
            raise TypeError("condition must be CalibrationCondition")
        joint_index(self.joint_name)
        if not isinstance(self.trial_id, str) or not self.trial_id:
            raise ValueError("trial_id is required for whole-trial validation")
        object.__setattr__(self, "qpos_rad", six(self.qpos_rad, "qpos_rad"))
        gains = six(self.active_driver_kd_nm_s_per_rad, "active_driver_kd_nm_s_per_rad")
        if min(gains) < 0:
            raise ValueError("active driver damping must be non-negative")
        object.__setattr__(self, "active_driver_kd_nm_s_per_rad", gains)
        object.__setattr__(
            self,
            "gravity_command_nm",
            six(self.gravity_command_nm, "gravity_command_nm"),
        )
        for name in ("velocity_rad_s", "command_residual_nm"):
            object.__setattr__(self, name, finite(getattr(self, name), name))
        if self.velocity_rad_s == 0:
            raise ValueError("a static sample does not identify running resistance")


def reduce_command_window(
    *,
    condition: CalibrationCondition,
    joint_name: str,
    trial_id: str,
    times_sec: Sequence[float],
    qpos_rad: Sequence[Sequence[float]],
    gravity_command_nm: Sequence[Sequence[float]],
    conditioned_command_nm: Sequence[Sequence[float]],
    driver_kd_nm_s_per_rad: Sequence[Sequence[float]],
    minimum_motion_rad: float,
    maximum_line_error_rad: float,
    maximum_residual_span_nm: float,
) -> tuple[CalibrationObservation | None, str | None]:
    """Reduce one steady interval without depending on a trace/UI schema."""
    j = joint_index(joint_name)
    for name, value in (
        ("minimum_motion_rad", minimum_motion_rad),
        ("maximum_line_error_rad", maximum_line_error_rad),
        ("maximum_residual_span_nm", maximum_residual_span_nm),
    ):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    count = len(times_sec)
    if count < 3 or any(
        len(values) != count
        for values in (
            qpos_rad,
            gravity_command_nm,
            conditioned_command_nm,
            driver_kd_nm_s_per_rad,
        )
    ):
        raise ValueError("window requires at least three aligned samples")
    times = np.asarray([finite(t, "timestamp") for t in times_sec])
    if np.any(np.diff(times) <= 0):
        raise ValueError("window timestamps must advance")
    q = np.asarray([six(v, "qpos_rad") for v in qpos_rad])
    gravity = np.asarray([six(v, "gravity_command_nm") for v in gravity_command_nm])
    command = np.asarray(
        [six(v, "conditioned_command_nm") for v in conditioned_command_nm]
    )
    driver_kd = [six(v, "driver_kd_nm_s_per_rad") for v in driver_kd_nm_s_per_rad]
    if any(min(gains) < 0 for gains in driver_kd):
        raise ValueError("driver damping must be non-negative")
    if any(gains != driver_kd[0] for gains in driver_kd):
        return None, "changing_motor_local_damping"
    if float(np.ptp(command[:, j] - gravity[:, j])) > maximum_residual_span_nm:
        return None, "changing_command_not_quasisteady_identification"
    displacement = float(q[-1, j] - q[0, j])
    if abs(displacement) < minimum_motion_rad:
        return None, "insufficient_motion_not_zero_friction"
    velocity = B601WindowVelocityEstimator(float(times[-1] - times[0]))
    speed = 0.0
    for t, position in zip(times, q[:, j]):
        speed = velocity.update(
            timestamp_sec=float(t), qpos_rad=float(position), fallback_rad_s=0.0
        )
    if not velocity.ready:
        return None, "velocity_window_not_complete"
    centered_time = times - times.mean()
    deviation = q[:, j] - q[:, j].mean() - speed * centered_time
    if (
        speed * displacement <= 0
        or float(np.max(np.abs(deviation))) > maximum_line_error_rad
        or float(np.max(np.ptp(np.delete(q, j, axis=1), axis=0)))
        > maximum_line_error_rad * 2
    ):
        return None, "nonsteady_or_coupled_motion"
    # One interval is one observation, not a collection of independent frames.
    return CalibrationObservation(
        condition=condition,
        joint_name=joint_name,
        trial_id=trial_id,
        qpos_rad=tuple(np.median(q, axis=0)),
        gravity_command_nm=tuple(np.median(gravity, axis=0)),
        velocity_rad_s=speed,
        command_residual_nm=float(np.median(command[:, j] - gravity[:, j])),
        active_driver_kd_nm_s_per_rad=driver_kd[0],
    ), None


def _features(velocity: float, gravity: float, *, load_varying: bool) -> list[float]:
    direction = math.copysign(1.0, velocity)
    row = [1.0, direction, velocity]
    if load_varying:
        row.extend((gravity, direction * abs(gravity)))
    return row


def fit_local_command_model(
    training: Sequence[CalibrationObservation],
    validation: Sequence[CalibrationObservation],
    *,
    max_pose_span_rad: float = 0.10,
    load_feature_span_nm: float = 0.05,
    max_validation_rmse_nm: float = 0.05,
) -> dict[str, object]:
    """Fit one local pose region; reject confounded, one-way or leaked data.

    Load dependence is identified only if the gravity command varies enough.
    Full-pose/load coverage is retained even when that feature is inactive.
    The coefficient on direction is an empirical term, not a friction sensor.
    """
    for name, value in (
        ("max_pose_span_rad", max_pose_span_rad),
        ("load_feature_span_nm", load_feature_span_nm),
        ("max_validation_rmse_nm", max_validation_rmse_nm),
    ):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    result: dict[str, object] = {
        "schema": "b601.local-command-calibration.v1",
        "status": "INSUFFICIENT_DATA",
        "auto_apply": False,
        "hardware_approved": False,
        "quantity": "conditioned_command_minus_recorded_gravity_baseline_not_physical_friction",
        "training_windows": len(training),
        "validation_windows": len(validation),
        "reasons": [],
    }

    def reject(reason: str) -> dict[str, object]:
        result["reasons"] = [reason]
        return result

    if not training or not validation:
        return reject("requires_independent_training_and_validation_trials")
    first = training[0]
    if any(
        o.condition != first.condition or o.joint_name != first.joint_name
        for o in (*training, *validation)
    ):
        return reject("joint_or_gravity_or_source_condition_mismatch")
    if any(
        o.active_driver_kd_nm_s_per_rad != first.active_driver_kd_nm_s_per_rad
        for o in (*training, *validation)
    ):
        return reject("active_driver_damping_mismatch")
    train_ids = {o.trial_id for o in training}
    validation_ids = {o.trial_id for o in validation}
    if train_ids & validation_ids:
        return reject("whole_trial_validation_leakage")
    if len(train_ids) < 4 or len(validation_ids) < 2:
        return reject("requires_four_training_and_two_validation_trials")
    for subset in (training, validation):
        if {math.copysign(1, o.velocity_rad_s) for o in subset} != {-1, 1}:
            return reject("both_directions_required_in_each_split")
    q = np.asarray([o.qpos_rad for o in training])
    g = np.asarray([o.gravity_command_nm for o in training])
    if float(np.max(np.ptp(q, axis=0))) > max_pose_span_rad:
        return reject("full_pose_region_too_wide_split_or_collect_local_sweeps")
    j = joint_index(first.joint_name)
    load_varying = float(np.ptp(g[:, j])) >= load_feature_span_nm
    feature_names = ["bias", "direction", "velocity_rad_s"]
    if load_varying:
        feature_names += [
            "gravity_command_nm",
            "direction_times_abs_gravity_command_nm",
        ]
    x = np.asarray(
        [
            _features(
                o.velocity_rad_s, o.gravity_command_nm[j], load_varying=load_varying
            )
            for o in training
        ]
    )
    y = np.asarray([o.command_residual_nm for o in training])
    if len(training) < 2 * x.shape[1]:
        return reject("too_few_independent_motion_windows")
    # Give each trial equal weight; long runs must not dominate the fit.
    weights = np.asarray(
        [1 / sum(other.trial_id == o.trial_id for other in training) for o in training]
    )
    scale = np.maximum(np.max(np.abs(x), axis=0), 1e-12)
    design = x / scale * np.sqrt(weights[:, None])
    coefficients, _, rank, singular = np.linalg.lstsq(
        design, y * np.sqrt(weights), rcond=None
    )
    conditioning = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    result.update(
        feature_names=feature_names,
        design_rank=int(rank),
        design_condition_number=conditioning if math.isfinite(conditioning) else None,
        load_dependence_identified=load_varying,
    )
    if rank != x.shape[1] or conditioning > 1e4:
        return reject("unidentifiable_features_add_speeds_or_load_conditions")
    coefficients = coefficients / scale
    coverage = {
        "semantics": "local_axis_aligned_observation_bounds_not_full_workspace_validation",
        "qpos_min_rad": q.min(axis=0).tolist(),
        "qpos_max_rad": q.max(axis=0).tolist(),
        "gravity_min_nm": g.min(axis=0).tolist(),
        "gravity_max_nm": g.max(axis=0).tolist(),
        "speed_min_rad_s": min(abs(o.velocity_rad_s) for o in training),
        "speed_max_rad_s": max(abs(o.velocity_rad_s) for o in training),
    }
    for observation in validation:
        if not _in_coverage(observation, coverage):
            return reject("validation_outside_training_pose_load_or_speed_coverage")
    xv = np.asarray(
        [
            _features(
                o.velocity_rad_s, o.gravity_command_nm[j], load_varying=load_varying
            )
            for o in validation
        ]
    )
    residual = xv @ coefficients - np.asarray(
        [o.command_residual_nm for o in validation]
    )
    # Report equal-trial held-out error rather than a random per-frame split.
    rmse = math.sqrt(
        float(
            np.mean(
                [
                    np.mean(
                        [
                            e * e
                            for e, o in zip(residual, validation)
                            if o.trial_id == trial
                        ]
                    )
                    for trial in sorted(validation_ids)
                ]
            )
        )
    )
    result.update(
        condition=asdict(first.condition),
        joint_name=first.joint_name,
        active_driver_kd_nm_s_per_rad=first.active_driver_kd_nm_s_per_rad,
        coefficients=coefficients.tolist(),
        coverage=coverage,
        training_trial_ids=sorted(train_ids),
        validation_trial_ids=sorted(validation_ids),
        validation_rmse_nm=rmse,
        max_validation_rmse_nm=max_validation_rmse_nm,
        unresolved="preload_temperature_and_continuous_control_improvement_not_validated",
    )
    if rmse > max_validation_rmse_nm:
        result["status"] = "VALIDATION_FAILED"
        return reject("held_out_command_prediction_error")
    result["status"] = "OFFLINE_CANDIDATE"
    return result


def _in_coverage(
    observation: CalibrationObservation, coverage: Mapping[str, object]
) -> bool:
    # Numeric tolerance is not an extrapolation allowance.
    for values, low_key, high_key in (
        (observation.qpos_rad, "qpos_min_rad", "qpos_max_rad"),
        (observation.gravity_command_nm, "gravity_min_nm", "gravity_max_nm"),
    ):
        lower, upper = (
            six(coverage[low_key], low_key),
            six(coverage[high_key], high_key),
        )
        if any(lo > hi for lo, hi in zip(lower, upper)):
            raise ValueError("invalid calibration coverage interval")
        if any(
            v < lo - 1e-9 or v > hi + 1e-9 for v, lo, hi in zip(values, lower, upper)
        ):
            return False
    speed = abs(observation.velocity_rad_s)
    lo = finite(coverage["speed_min_rad_s"], "speed_min_rad_s")
    hi = finite(coverage["speed_max_rad_s"], "speed_max_rad_s")
    if not 0 < lo <= hi:
        raise ValueError("invalid calibration speed interval")
    return lo - 1e-9 <= speed <= hi + 1e-9


def predict_command_residual(
    model: Mapping[str, object], observation: CalibrationObservation
) -> float:
    """Offline prediction only; context/range mismatch is an error, not a clamp."""
    if (
        model.get("schema") != "b601.local-command-calibration.v1"
        or model.get("status") != "OFFLINE_CANDIDATE"
    ):
        raise ValueError("prediction requires a validated offline candidate")
    if not isinstance(model.get("condition"), Mapping) or not isinstance(
        model.get("coverage"), Mapping
    ):
        raise TypeError("prediction model requires condition and coverage objects")
    condition = CalibrationCondition.from_mapping(
        model["condition"], run_id=model["condition"]["run_id"]
    )
    if (
        condition != observation.condition
        or model.get("joint_name") != observation.joint_name
    ):
        raise ValueError("prediction condition or joint differs from calibration")
    if (
        six(
            model.get("active_driver_kd_nm_s_per_rad", (0.0,) * 6),
            "active_driver_kd_nm_s_per_rad",
        )
        != observation.active_driver_kd_nm_s_per_rad
    ):
        raise ValueError("prediction active driver damping differs from calibration")
    if not _in_coverage(observation, model["coverage"]):
        raise ValueError("prediction is outside calibrated pose/load/speed coverage")
    if not isinstance(model.get("load_dependence_identified"), bool):
        raise TypeError("load_dependence_identified must be boolean")
    row = _features(
        observation.velocity_rad_s,
        observation.gravity_command_nm[joint_index(observation.joint_name)],
        load_varying=model["load_dependence_identified"],
    )
    coefficients = [finite(v, "coefficient") for v in model["coefficients"]]
    if len(coefficients) != len(row):
        raise ValueError("model feature/coefficients mismatch")
    return float(np.dot(row, coefficients))


def plan_local_sweeps(
    *,
    condition: CalibrationCondition,
    joint_name: str,
    anchor_qpos_rad: Sequence[float],
    position_limits_rad: Sequence[Sequence[float]],
    excursion_rad: float,
    speeds_rad_s: Sequence[float],
    max_reference_velocity_rad_s: float,
    training_repeats: int = 2,
    validation_repeats: int = 1,
    purpose: str = "model_sweeps",
) -> dict[str, object]:
    """Generate reversible low-speed experiment intents, never MIT commands.

    A future owner adapter must execute these through the existing reference,
    braking, receipt and safety contracts. This plan is not hardware approval.
    """
    if not isinstance(condition, CalibrationCondition):
        raise TypeError("condition must be CalibrationCondition")
    if purpose not in {"model_sweeps", "motion_check", "local_identification"}:
        raise ValueError("unknown local sweep purpose")
    j = joint_index(joint_name)
    anchor = six(anchor_qpos_rad, "anchor_qpos_rad")
    excursion = finite(excursion_rad, "excursion_rad")
    maximum_speed = finite(max_reference_velocity_rad_s, "max_reference_velocity_rad_s")
    speeds = sorted({finite(v, "speed") for v in speeds_rad_s})
    if (
        excursion <= 0
        or maximum_speed <= 0
        or len(speeds) < 2
        or min(speeds) <= 0
        or max(speeds) > maximum_speed
    ):
        raise ValueError(
            "need positive excursion and two distinct speeds inside the approved envelope"
        )
    for repeats in (training_repeats, validation_repeats):
        if (
            isinstance(repeats, bool)
            or not isinstance(repeats, int)
            or not 1 <= repeats <= 10
        ):
            raise ValueError("repeat counts must be integers in 1..10")
    if len(position_limits_rad) != 6:
        raise ValueError("approved position envelope must contain six intervals")
    limits = []
    for i, interval in enumerate(position_limits_rad):
        if len(interval) != 2:
            raise ValueError("position interval requires lower and upper bounds")
        lo, hi = (finite(v, "position bound") for v in interval)
        if not lo <= anchor[i] <= hi or lo >= hi:
            raise ValueError("anchor lies outside approved position envelope")
        limits.append((lo, hi))
    low, high = anchor[j] - excursion, anchor[j] + excursion
    if purpose == "local_identification":
        high = anchor[j]
    if low < limits[j][0] or high > limits[j][1]:
        raise ValueError("sweep would exceed approved position envelope")
    steps = []

    def append(target: float, speed: float, role: str) -> None:
        q = list(anchor)
        q[j] = target
        steps.append(
            {
                "trial_id": f"sweep-{len(steps) + 1}",
                "role": role,
                "target_qpos_rad": q,
                "max_reference_velocity_rad_s": speed,
            }
        )

    check_speed = speeds[-1] if purpose == "motion_check" else speeds[0]
    if purpose != "local_identification":
        append(low, check_speed, "positioning_not_for_fit")
    if purpose in {"model_sweeps", "local_identification"}:
        for role, repeats in (
            ("training", training_repeats),
            ("validation", validation_repeats),
        ):
            for _ in range(repeats):
                for speed in speeds:
                    # Local identification reuses the accepted motion-check
                    # excursion and return. Transients are filtered by the
                    # window reducer; fit roles are assigned before motion.
                    targets = (
                        (low, high) if purpose == "local_identification" else (high, low)
                    )
                    for target in targets:
                        append(target, speed, role)
    if purpose != "local_identification":
        append(anchor[j], check_speed, "return_not_for_fit")
    return {
        "schema": "b601.local-calibration-plan.v1",
        "purpose": purpose,
        "condition": asdict(condition),
        "anchor_qpos_rad": anchor,
        "joint_name": joint_name,
        "steps": steps,
        "execution_enabled": False,
        "requires": "existing_owner_reference_braking_receipts_and_all_existing_guards",
        "physical_load_observation": "not_measured_run_local_only",
    }


def summarize_automatic_acquisition(
    result: Mapping[str, object], *, pose_tolerance_rad: float = 0.03
) -> dict[str, object]:
    """Summarize encoder response and repeatability, without fitting static force."""
    if result.get("schema") != "b601.automatic-joint-calibration.v1":
        raise ValueError("unknown automatic calibration result schema")
    if finite(pose_tolerance_rad, "pose_tolerance_rad") <= 0:
        raise ValueError("pose tolerance must be positive")
    raw_trials = result.get("onset_trials")
    if not isinstance(raw_trials, list):
        raise TypeError("onset trials must be a list")
    trials = []
    groups: list[list[dict[str, object]]] = []
    for raw in raw_trials:
        if (
            not isinstance(raw, dict)
            or type(raw.get("direction")) is not int
            or raw.get("direction") not in {-1, 1}
        ):
            raise ValueError("onset trial requires a signed direction")
        baseline = six(raw.get("baseline_qpos_rad"), "baseline_qpos_rad")
        final = six(raw.get("final_qpos_rad"), "final_qpos_rad")
        lo, hi = (
            six(raw.get("qpos_min_rad"), "qpos_min_rad"),
            six(raw.get("qpos_max_rad"), "qpos_max_rad"),
        )
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError("invalid observed position range")
        active_kd = (
            None
            if raw.get("active_driver_kd_nm_s_per_rad") is None
            else six(raw["active_driver_kd_nm_s_per_rad"], "active driver damping")
        )
        if active_kd is not None and min(active_kd) < 0:
            raise ValueError("active driver damping must be non-negative")
        direction = raw["direction"]
        peak_progress = direction * ((hi[2] if direction > 0 else lo[2]) - baseline[2])
        final_progress = direction * (final[2] - baseline[2])
        confirmed = (
            raw.get("outcome") == "motion_onset_confirmed"
            and raw.get("human_intervention") is False
        )
        onset = raw.get("onset_dispatch")
        command = None
        if confirmed:
            if not isinstance(onset, dict):
                raise ValueError("confirmed encoder onset is missing its dispatch")
            command = finite(onset.get("conditioned_total_nm"), "onset total") - finite(
                onset.get("gravity_nm"), "onset gravity"
            )
            if command * direction <= 0:
                raise ValueError("onset command opposes the recorded direction")
        outcome = (
            "confirmed_encoder_progress"
            if confirmed
            else "unconfirmed_micro_displacement"
            if raw.get("outcome") == "probe_cumulative_displacement_withdrawn"
            else "cap_reached_without_confirmation"
            if raw.get("scan_end_reason") == "cap_dwell_without_confirmed_motion"
            else str(raw.get("outcome") or "incomplete")
        )
        trial = {
            "trial_sequence": raw.get("trial_sequence"),
            "direction": direction,
            "baseline_qpos_rad": baseline,
            "outcome": outcome,
            "confirmed_onset_command_nm": command,
            "peak_conditioned_command_nm": finite(
                raw.get("peak_conditioned_delta_nm"), "peak conditioned command"
            ),
            "peak_encoder_progress_rad": peak_progress,
            "final_encoder_progress_rad": final_progress,
            "returned_encoder_progress_rad": peak_progress - final_progress,
            "human_intervention": raw.get("human_intervention"),
            "completed_at_sec": raw.get("completed_at_sec"),
            "active_driver_kd_nm_s_per_rad": active_kd,
            "zero_assist_completed_at_sec": raw.get("zero_assist_completed_at_sec"),
            "withdrawal_reason": raw.get("withdrawal_reason"),
        }
        trials.append(trial)
        if command is not None:
            for group in groups:
                if (
                    group[0]["direction"] == direction
                    and group[0]["active_driver_kd_nm_s_per_rad"] == active_kd
                    and max(
                        abs(a - b)
                        for a, b in zip(group[0]["baseline_qpos_rad"], baseline)
                    )
                    <= pose_tolerance_rad
                ):
                    group.append(trial)
                    break
            else:
                groups.append([trial])
    repetition = []
    for group in groups:
        commands = [abs(t["confirmed_onset_command_nm"]) for t in group]
        repetition.append(
            {
                "direction": group[0]["direction"],
                "anchor_qpos_rad": group[0]["baseline_qpos_rad"],
                "count": len(group),
                "active_driver_kd_nm_s_per_rad": group[0][
                    "active_driver_kd_nm_s_per_rad"
                ],
                "command_median_abs_nm": statistics.median(commands),
                "command_min_abs_nm": min(commands),
                "command_max_abs_nm": max(commands),
                "command_spread_nm": max(commands) - min(commands),
                "repeatability_status": "OBSERVED_SPREAD"
                if len(group) >= 2
                else "INSUFFICIENT_REPEATS",
            }
        )
    motion = result.get("motion")
    legs = motion.get("completed_legs", []) if isinstance(motion, Mapping) else []
    assessed = [leg for leg in legs if type(leg.get("tracking_target_reached")) is bool]
    deviations = [leg for leg in assessed if not leg["tracking_target_reached"]]
    settings = result.get("settings")
    repeats = (
        settings.get("onset_repeats_per_direction")
        if isinstance(settings, Mapping)
        else None
    )
    expected_onset_trials = 2 * repeats if type(repeats) is int and repeats > 0 else None
    confirmed_onset_trials = [
        trial
        for trial in trials
        if trial["outcome"] == "confirmed_encoder_progress"
        and trial["confirmed_onset_command_nm"] is not None
    ]
    onset_damping = result.get("onset_damping")
    base_damping_restored = (
        isinstance(onset_damping, Mapping)
        and onset_damping.get("base_restore_completed_at_sec") is not None
    )
    onset_scan_set_complete = (
        expected_onset_trials is not None
        and len(trials) == expected_onset_trials
        and result.get("incomplete_onset_trial") is None
    )
    onset_stage_complete = onset_scan_set_complete and base_damping_restored
    onset_data_assessment = {
        "status": "EXPLORATORY_RESPONSE_DATA_AVAILABLE"
        if trials
        else "NO_COMPLETED_ONSET_TRIALS",
        "completed_trial_count": len(trials),
        "confirmed_encoder_progress_trial_count": len(confirmed_onset_trials),
        "other_completed_observation_count": len(trials) - len(confirmed_onset_trials),
        "expected_trial_count": expected_onset_trials,
        "scan_set_complete": onset_scan_set_complete,
        "base_damping_restore_confirmed": base_damping_restored,
        "stage_complete": onset_stage_complete,
        "incomplete_trial_present": result.get("incomplete_onset_trial") is not None,
        "later_motion_failure_invalidates_completed_trials": False,
        "usable_for": "exploratory_direction_conditioned_software_command_and_encoder_response_analysis",
        "not_established": "measured_output_torque_or_static_or_running_friction_in_Nm",
        "semantics": (
            "completed onset trials are assessed independently of later motion acquisition; "
            "confirmed onset command is the software command associated with confirmed encoder progress, "
            "not a measured breakaway torque"
        ),
    }
    return {
        "schema": "b601.automatic-calibration-report.v1",
        "status": "ACQUISITION_COMPLETE"
        if result.get("state") == "complete"
        else "ACQUISITION_PARTIAL"
        if trials or legs
        else "NO_COMPLETED_TRIALS",
        "batch_state": result.get("state"),
        "batch_reason": result.get("reason"),
        "acquisition_complete": result.get("state") == "complete",
        "tracking_arrival_quality": {
            "assessed_legs": len(assessed),
            "outside_tracking_tolerance_legs": len(deviations),
            "unknown_legacy_legs": len(legs) - len(assessed),
            "status": "TARGET_DEVIATION_RECORDED" if deviations else
            "ALL_COLLECTED_LEGS_WITHIN_TRACKING_TOLERANCE" if assessed and len(assessed) == len(legs) else
            "NOT_FULLY_ASSESSED",
            "semantics": "acquisition completion is not precise positioning or a validated friction model",
        },
        "condition": result.get("condition"),
        "trials": trials,
        "onset_data_assessment": onset_data_assessment,
        "repeatability": repetition,
        "motion": result.get("motion"),
        "motion_state": result.get("motion_state"),
        "incomplete_onset_trial": result.get("incomplete_onset_trial"),
        "onset_damping": result.get("onset_damping"),
        "quantity": "software_command_and_encoder_response_not_measured_output_friction",
        "gravity_residual_and_friction_separated": False,
        "auto_apply": False,
        "unobserved_conditions": [
            "physical_payload",
            "transmission_preload",
            "temperature",
            "independent_output_motion",
        ],
        "pose_tolerance_rad": pose_tolerance_rad,
    }
