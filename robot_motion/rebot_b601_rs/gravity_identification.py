"""Offline effective-gravity regression shared by capture and motion analysis.

Inputs remain software commands. No output-force calibration, installation,
device I/O or autonomous change to compensation is provided here.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict

import numpy as np

from .calibration import CalibrationObservation, finite, joint_index


def solve_gravity_design(design, observed, *, model, nominal_scale, bias_count=6):
    """Solve the existing scale/bias/payload parameterization and constraints."""
    x, y = np.asarray(design, dtype=float), np.asarray(observed, dtype=float)
    expected = {"global-scale": 1, "global-scale-joint-bias": 1+bias_count,
                "global-scale-payload": 5}.get(model)
    if (expected is None or x.ndim != 2 or x.shape[1] != expected or not len(x)
        or y.shape != (len(x),) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
        or finite(nominal_scale, "nominal scale") <= 0):
        raise ValueError("invalid effective gravity regression design")
    solution, _, rank, singular = np.linalg.lstsq(x, y, rcond=None)
    if model == "global-scale-payload" and solution[1] < 0:
        solution[1] = 0.0
        solution[[0, 2, 3, 4]] = np.linalg.lstsq(x[:, (0, 2, 3, 4)], y, rcond=None)[0]
    if solution[0] < -nominal_scale:
        solution[0] = -nominal_scale
        corrected = y-x[:, 0]*solution[0]
        if model == "global-scale-joint-bias":
            solution[1:] = np.linalg.lstsq(x[:, 1:], corrected, rcond=None)[0]
        elif model == "global-scale-payload":
            solution[2:] = np.linalg.lstsq(x[:, 2:], corrected, rcond=None)[0]
    return solution, int(rank), singular


def pair_motion_observations(observations, *, pose_tolerance_rad=.0045,
                             maximum_speed_ratio=1.25):
    """Match disjoint opposite-direction windows at actual full poses/speeds.

    No unmatched, censored, static or one-sided response is a gravity label.
    The pair mean only cancels approximately odd, repeatable resistance.
    """
    if finite(pose_tolerance_rad, "pair pose tolerance") <= 0 or finite(maximum_speed_ratio, "pair speed ratio") < 1:
        raise ValueError("invalid motion pairing tolerances")
    positive = [o for o in observations if o.velocity_rad_s > 0]
    remaining = [o for o in observations if o.velocity_rad_s < 0]
    pairs = []
    for p in positive:
        candidates = []
        for index, n in enumerate(remaining):
            if (p.condition != n.condition or p.joint_name != n.joint_name
                or p.active_driver_kd_nm_s_per_rad != n.active_driver_kd_nm_s_per_rad
                or p.trial_id == n.trial_id):
                continue
            distance = max(abs(a-b) for a, b in zip(p.qpos_rad, n.qpos_rad))
            ratio = max(abs(p.velocity_rad_s), abs(n.velocity_rad_s))/min(abs(p.velocity_rad_s), abs(n.velocity_rad_s))
            if distance <= pose_tolerance_rad and ratio <= maximum_speed_ratio:
                candidates.append((distance, ratio, index))
        if not candidates:
            continue
        distance, ratio, index = min(candidates)
        n = remaining.pop(index)
        j = joint_index(p.joint_name)
        pairs.append({"joint_name": p.joint_name, "positive_trial_id": p.trial_id,
            "negative_trial_id": n.trial_id,
            "qpos_rad": [(a+b)/2 for a, b in zip(p.qpos_rad, n.qpos_rad)],
            "maximum_pose_mismatch_rad": distance, "speed_magnitude_ratio": ratio,
            "gravity_command_nm": (p.gravity_command_nm[j]+n.gravity_command_nm[j])/2,
            "even_command_residual_nm": (p.command_residual_nm+n.command_residual_nm)/2,
            "odd_command_residual_nm": (p.command_residual_nm-n.command_residual_nm)/2,
            "mean_abs_velocity_rad_s": (abs(p.velocity_rad_s)+abs(n.velocity_rad_s))/2})
    return pairs


def fit_paired_gravity_command(training, validation, *, max_validation_rmse_nm=.05):
    """Fit a scale plus J2/J3 biases, reserving whole trials for validation."""
    finite(max_validation_rmse_nm, "validation error limit")
    if max_validation_rmse_nm <= 0:
        raise ValueError("validation error limit must be positive")
    result = {"status": "INSUFFICIENT_DATA", "auto_apply": False,
        "hardware_approved": False, "identifies_unique_physical_gravity": False,
        "quantity": "paired_even_software_command_residual",
        "model": "global-scale-joint-bias", "reasons": [],
        "assumptions": ["odd resistance approximately cancels at matched speed and full pose",
                        "asymmetric friction and elastic preload remain in the effective correction"],
        "training_pairs": [], "validation_pairs": []}

    def reject(reason):
        result["reasons"] = [reason]
        return result

    all_observations = [*training, *validation]
    if not training or not validation:
        return reject("requires_independent_paired_motion_trials")
    if any(not isinstance(o, CalibrationObservation) for o in all_observations):
        raise TypeError("gravity fit requires admitted motion observations")
    first = all_observations[0]
    if any(o.condition != first.condition or o.active_driver_kd_nm_s_per_rad != first.active_driver_kd_nm_s_per_rad for o in all_observations):
        return reject("mixed_gravity_source_or_active_damping")
    if any(first.condition.gravity_coulomb_friction_nm):
        return reject("paired_gravity_requires_separate_velocity_compensation_baseline")
    if {o.trial_id for o in training} & {o.trial_id for o in validation}:
        return reject("whole_trial_validation_leakage")
    joints = sorted({o.joint_name for o in training})
    if set(joints) != {"joint2", "joint3"}:
        return reject("requires_both_j2_and_j3")
    train_pairs, valid_pairs = map(pair_motion_observations, (training, validation))
    result.update(training_pairs=train_pairs, validation_pairs=valid_pairs,
        condition=asdict(first.condition), active_driver_kd_nm_s_per_rad=first.active_driver_kd_nm_s_per_rad)
    for pairs in (train_pairs, valid_pairs):
        if {p["joint_name"] for p in pairs} != set(joints):
            return reject("requires_matched_opposite_motion_in_each_split_and_joint")
    for joint in joints:
        for direction in (-1, 1):
            if len({o.trial_id for o in training if o.joint_name == joint and o.velocity_rad_s*direction > 0}) < 2:
                return reject("requires_independent_training_repeats_per_direction_and_joint")
    if len(train_pairs) < 2*(1+len(joints)):
        return reject("too_few_independent_paired_windows")

    def design(pairs):
        return np.asarray([[p["gravity_command_nm"]/first.condition.gravity_scale,
            *[float(p["joint_name"] == joint) for joint in joints]] for p in pairs])

    x = design(train_pairs)
    y = np.asarray([p["even_command_residual_nm"] for p in train_pairs])
    # Long trajectories must not masquerade as more independent experiments.
    identities = [(p["joint_name"], p["positive_trial_id"], p["negative_trial_id"]) for p in train_pairs]
    counts = Counter(identities)
    weights = np.asarray([1/math.sqrt(counts[key]) for key in identities])
    solution, rank, singular = solve_gravity_design(x*weights[:, None], y*weights, model="global-scale-joint-bias",
        nominal_scale=first.condition.gravity_scale, bias_count=len(joints))
    conditioning = float(singular[0]/singular[-1]) if singular[-1] > 0 else math.inf
    result.update(design_rank=rank, design_condition_number=conditioning if math.isfinite(conditioning) else None)
    if rank < x.shape[1] or conditioning > 1e4:
        return reject("gravity_features_not_identifiable_expand_pose_information")
    q = np.asarray([p["qpos_rad"] for p in train_pairs])
    # Validation must not silently extrapolate outside the training region.
    if any(np.any(np.asarray(p["qpos_rad"]) < q.min(axis=0)-.0045)
           or np.any(np.asarray(p["qpos_rad"]) > q.max(axis=0)+.0045) for p in valid_pairs):
        return reject("validation_outside_training_pose_coverage")
    error = design(valid_pairs)@solution-np.asarray([p["even_command_residual_nm"] for p in valid_pairs])
    trial_errors = {}
    for p, e in zip(valid_pairs, error):
        trial_errors.setdefault((p["joint_name"], p["positive_trial_id"], p["negative_trial_id"]), []).append(float(e*e))
    per_joint = {joint: math.sqrt(float(np.mean([np.mean(values) for key, values in trial_errors.items()
        if key[0] == joint]))) for joint in joints}
    rmse = math.sqrt(sum(v*v for v in per_joint.values())/len(per_joint))
    factor = 1+float(solution[0])/first.condition.gravity_scale
    scales = list(first.condition.gravity_joint_scales)
    biases = list(first.condition.gravity_joint_bias_nm)
    correction = {}
    for joint, intercept in zip(joints, solution[1:]):
        j = joint_index(joint)
        # The recorded baseline already contains its configured bias. Scale
        # that bias too when translating the regression to profile parameters.
        correction[joint] = float((factor-1)*biases[j]+intercept)
        scales[j] *= factor
        biases[j] += correction[joint]
    result.update(scale_correction=float(solution[0]),
        joint_intercept_nm={joint: float(v) for joint, v in zip(joints, solution[1:])},
        joint_bias_correction_nm=correction,
        parameterization="delta = scale_correction * recorded_gravity_command / old_gravity_scale + joint_intercept",
        application_scope=joints,
        candidate_profile_fields={"gravity_scale": first.condition.gravity_scale,
            "gravity_joint_scales": scales, "gravity_joint_bias_nm": biases},
        validation_rmse_nm=rmse, validation_rmse_by_joint_nm=per_joint,
        max_validation_rmse_nm=max_validation_rmse_nm,
        coverage={"qpos_min_rad": q.min(axis=0).tolist(), "qpos_max_rad": q.max(axis=0).tolist()})
    if rmse > max_validation_rmse_nm:
        result["status"] = "VALIDATION_FAILED"
        return reject("held_out_paired_command_prediction_error")
    result["status"] = "OFFLINE_CANDIDATE"
    return result
