"""Local encoder/command response prediction, without actuator authority.

The field predicts the recorded software residual at fixed source and damping
conditions. Its even/odd decomposition is empirical, not a physical gravity or
friction measurement. Static support, breakaway and extrapolation are excluded.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .calibration import CalibrationCondition, CalibrationObservation, finite, six
from .gravity_identification import pair_motion_observations


@dataclass(frozen=True)
class LocalCommandField:
    """One training-only local region; one nearest estimate per source trial."""

    training: tuple[CalibrationObservation, ...]

    def __post_init__(self):
        samples = tuple(self.training)
        if not samples or any(not isinstance(o, CalibrationObservation) for o in samples):
            raise ValueError("command field requires admitted training observations")
        first = samples[0]
        if first.joint_name not in {"joint2", "joint3"}:
            raise ValueError("command field supports J2/J3 only")
        if any(o.joint_name != first.joint_name or o.condition != first.condition
               or o.active_driver_kd_nm_s_per_rad != first.active_driver_kd_nm_s_per_rad for o in samples):
            raise ValueError("command field requires one joint, source condition and active damping")
        object.__setattr__(self, "training", tuple(dict.fromkeys(samples)))

    def predict(self, *, condition: CalibrationCondition, qpos_rad, velocity_rad_s,
                active_driver_kd_nm_s_per_rad):
        """Query without a target command label; abstain outside observed support.

        A missing prediction is None, never an inferred zero compensation.
        Source/damping mismatches are caller errors, not out-of-region queries.
        """
        first = self.training[0]
        if (condition != first.condition or six(active_driver_kd_nm_s_per_rad, "active damping")
                != first.active_driver_kd_nm_s_per_rad):
            raise ValueError("command field query source condition or active damping mismatch")
        q = six(qpos_rad, "command field query pose")
        velocity = finite(velocity_rad_s, "command field query velocity")
        nearest, direction_trials, pose_trials = {}, set(), set()
        for source in self.training:
            if source.velocity_rad_s*velocity <= 0:
                continue
            direction_trials.add(source.trial_id)
            distance = max(abs(a-b) for a, b in zip(q, source.qpos_rad))
            if distance > .0045:
                continue
            pose_trials.add(source.trial_id)
            ratio = max(abs(velocity), abs(source.velocity_rad_s))/min(abs(velocity), abs(source.velocity_rad_s))
            if ratio > 1.25:
                continue
            match = (distance, ratio)
            prior = nearest.get(source.trial_id)
            if prior is None or match < prior[0]:
                nearest[source.trial_id] = (match, [source.command_residual_nm])
            elif match == prior[0]:
                prior[1].append(source.command_residual_nm)
        values = [sum(commands)/len(commands) for _, commands in nearest.values()]
        predicted = len(values) >= 2
        return {"predicted": predicted,
            "reason": ("matched_independent_training_trials" if predicted else
                       "static_and_onset_not_identified" if velocity == 0 else
                       "direction_not_observed" if not direction_trials else
                       "full_pose_outside_training_support" if not pose_trials else
                       "actual_speed_outside_training_support" if not nearest else
                       "insufficient_independent_training_repeats"),
            "predicted_command_residual_nm": sum(values)/len(values) if predicted else None,
            "training_trial_ids": sorted(nearest),
            "training_command_residual_range_nm": [min(values), max(values)] if values else None}

    def components(self, *, condition, qpos_rad, velocity_rad_s, active_driver_kd_nm_s_per_rad):
        """Compare both directions at exactly the same query pose and speed."""
        speed = abs(finite(velocity_rad_s, "command field component speed"))
        estimates = [self.predict(condition=condition, qpos_rad=qpos_rad,
            velocity_rad_s=direction*speed, active_driver_kd_nm_s_per_rad=active_driver_kd_nm_s_per_rad)
            for direction in (1, -1)]
        available = all(p["predicted"] for p in estimates)
        p, n = [e["predicted_command_residual_nm"] for e in estimates]
        return {"available": available, "positive": estimates[0], "negative": estimates[1],
            "even_command_nm": (p+n)/2 if available else None,
            "odd_command_nm": (p-n)/2 if available else None,
            "identifies_physical_gravity_or_friction": False,
            "usable_as_static_support": False}


def _equal_trial_rmse(errors):
    return math.sqrt(sum(sum(v)/len(v) for v in errors.values())/len(errors)) if errors else None


def evaluate_local_command_field(training, validation):
    """Whole-trial shadow prediction and independent paired-component checks.

    No validation label participates in model construction or neighbour choice.
    Error reduction is algebraic command prediction, not a closed-loop rollout.
    """
    training, validation = tuple(training), tuple(validation)
    if any(not isinstance(o, CalibrationObservation) for o in (*training, *validation)):
        raise TypeError("command field evaluation requires admitted observations")
    if {o.trial_id for o in training} & {o.trial_id for o in validation}:
        raise ValueError("whole trial leakage in local repeat prediction")
    field = LocalCommandField(tuple(training)) if training else None
    predictions, errors, baseline = [], defaultdict(list), defaultdict(list)
    for target in validation:
        if field is not None and target.joint_name != field.training[0].joint_name:
            raise ValueError("command field validation joint mismatch")
        predicted = field.predict(condition=target.condition, qpos_rad=target.qpos_rad,
            velocity_rad_s=target.velocity_rad_s, active_driver_kd_nm_s_per_rad=target.active_driver_kd_nm_s_per_rad
        ) if field else {"predicted": False, "reason": "no_training_observations",
                        "predicted_command_residual_nm": None, "training_trial_ids": [],
                        "training_command_residual_range_nm": None}
        record = {**predicted, "validation_trial_id": target.trial_id,
            "qpos_rad": target.qpos_rad, "velocity_rad_s": target.velocity_rad_s,
            "observed_command_residual_nm": target.command_residual_nm}
        if predicted["predicted"]:
            error = predicted["predicted_command_residual_nm"]-target.command_residual_nm
            record["error_nm"] = error
            errors[target.trial_id].append(error*error)
            baseline[target.trial_id].append(target.command_residual_nm**2)
        predictions.append(record)
    predicted_count = sum(p["predicted"] for p in predictions)
    rmse, zero = _equal_trial_rmse(errors), _equal_trial_rmse(baseline)
    direct = {"status": "DIAGNOSTIC_ONLY", "auto_apply": False,
        "pose_tolerance_rad": .0045, "maximum_speed_ratio": 1.25,
        "minimum_training_trials_per_prediction": 2,
        "validation_windows": len(validation), "predicted_windows": predicted_count,
        "unpredicted_windows": len(validation)-predicted_count,
        "all_validation_windows_predicted": bool(validation) and predicted_count == len(validation),
        "validation_coverage_fraction": predicted_count/len(validation) if validation else None,
        "equal_trial_rmse_on_predicted_windows_nm": rmse,
        "zero_prediction_rmse_on_same_windows_nm": zero,
        "command_prediction_error_reduction_fraction": 1-rmse/zero if zero else None,
        "max_diagnostic_rmse_nm": .05, "covered_subset_error_passed": rmse is not None and rmse <= .05,
        "physical_compensation_ready": False, "predictions": predictions,
        "semantics": "partial_local_command_predictability_not_full_model_validation_or_physical_friction"}
    paired, even_errors, odd_errors = [], defaultdict(list), defaultdict(list)
    for pair in pair_motion_observations(validation):
        components = field.components(condition=validation[0].condition, qpos_rad=pair["qpos_rad"],
            velocity_rad_s=pair["mean_abs_velocity_rad_s"],
            active_driver_kd_nm_s_per_rad=validation[0].active_driver_kd_nm_s_per_rad) if field else None
        row = {"positive_trial_id": pair["positive_trial_id"], "negative_trial_id": pair["negative_trial_id"],
            "qpos_rad": pair["qpos_rad"], "mean_abs_velocity_rad_s": pair["mean_abs_velocity_rad_s"],
            "predicted": components is not None and components["available"],
            "observed_even_command_nm": pair["even_command_residual_nm"],
            "observed_odd_command_nm": pair["odd_command_residual_nm"]}
        if row["predicted"]:
            identity = (pair["positive_trial_id"], pair["negative_trial_id"])
            for name, target_errors in (("even", even_errors), ("odd", odd_errors)):
                error = components[f"{name}_command_nm"]-pair[f"{name}_command_residual_nm"]
                row.update({f"predicted_{name}_command_nm": components[f"{name}_command_nm"], f"{name}_error_nm": error})
                target_errors[identity].append(error*error)
        paired.append(row)
    trials = {direction: sorted({o.trial_id for o in training if o.velocity_rad_s*direction > 0})
              for direction in (-1, 1)}
    checks = {"independent_bidirectional_training": all(len(ids) >= 2 for ids in trials.values()),
        "independent_bidirectional_validation": all(any(o.velocity_rad_s*d > 0 for o in validation) for d in (-1, 1)),
        "coverage_passed": direct["all_validation_windows_predicted"],
        "prediction_error_passed": direct["covered_subset_error_passed"]}
    even_rmse, odd_rmse = _equal_trial_rmse(even_errors), _equal_trial_rmse(odd_errors)
    component_checks = {"coverage_passed": len(paired) >= 2 and all(p["predicted"] for p in paired),
        "even_prediction_error_passed": even_rmse is not None and even_rmse <= .05,
        "odd_prediction_error_passed": odd_rmse is not None and odd_rmse <= .05}
    gaps = defaultdict(list)
    for prediction in predictions:
        if not prediction["predicted"]:
            gaps[prediction["reason"], int(math.copysign(1, prediction["velocity_rad_s"]))].append(prediction)
    return {"status": "OFFLINE_CANDIDATE" if all(checks.values()) else "DIAGNOSTIC_ONLY",
        "auto_apply": False, "hardware_approved": False, "live_compensation_enabled": False,
        "candidate_scope": "moving_command_prediction_only; component_validation_is_separate",
        "quantity": "local_direction_and_speed_conditioned_software_command_residual",
        "training_windows": len(training), "training_trial_ids": sorted({o.trial_id for o in training}),
        "validation_checks": checks, "directional_validation": direct,
        "coverage_gaps": [{"reason": reason, "direction": direction, "validation_windows": len(rows),
            "validation_trial_ids": sorted({r["validation_trial_id"] for r in rows}),
            "actual_abs_speed_range_rad_s": [min(abs(r["velocity_rad_s"]) for r in rows),
                                             max(abs(r["velocity_rad_s"]) for r in rows)],
            "qpos_min_rad": [min(r["qpos_rad"][j] for r in rows) for j in range(6)],
            "qpos_max_rad": [max(r["qpos_rad"][j] for r in rows) for j in range(6)]}
            for (reason, direction), rows in sorted(gaps.items())],
        "paired_component_validation": {"validation_pairs": len(paired),
            "status": "OFFLINE_CANDIDATE" if all(component_checks.values()) and checks["independent_bidirectional_training"] else "DIAGNOSTIC_ONLY",
            "validation_checks": component_checks,
            "predicted_pairs": sum(p["predicted"] for p in paired),
            "equal_trial_even_rmse_on_predicted_pairs_nm": even_rmse,
            "equal_trial_odd_rmse_on_predicted_pairs_nm": odd_rmse,
            "predictions": paired, "identifies_physical_gravity_or_friction": False},
        "static_and_breakaway_identified": False,
        "semantics": "training_only_neighbours; held_out_command_error_not_realized_feedback_reduction; no_static_or_out_of_coverage_application"}
