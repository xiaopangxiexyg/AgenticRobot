"""Metrics for low-dimensional state-policy fits and simulation traces."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import (
    STATE_BC_ACTION_DIM,
    STATE_BC_GRIPPER_SLICES,
    STATE_BC_POSE_SLICES,
)


def _matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != STATE_BC_ACTION_DIM or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite [N, {STATE_BC_ACTION_DIM}], got {array.shape}.")
    return array


def _quaternion_angle(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = predicted[:, 3:7]
    truth = target[:, 3:7]
    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True)
    truth_norm = np.linalg.norm(truth, axis=1, keepdims=True)
    if np.any(pred_norm <= 1e-8) or np.any(truth_norm <= 1e-8):
        raise ValueError("pose quaternions must have non-zero norms.")
    dot = np.sum((pred / pred_norm) * (truth / truth_norm), axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return (2.0 * np.arccos(dot)).astype(np.float32)


def action_metrics(predicted: Any, target: Any) -> dict[str, float]:
    """Return aggregate and component metrics without claiming task success."""

    pred = _matrix(predicted, name="predicted")
    truth = _matrix(target, name="target")
    if pred.shape != truth.shape or pred.shape[0] == 0:
        raise ValueError("predicted and target must have the same non-empty shape.")
    error = pred - truth
    metrics: dict[str, float] = {
        "frame_count": float(pred.shape[0]),
        "action_mse": float(np.mean(np.square(error))),
        "action_mae": float(np.mean(np.abs(error))),
    }
    pose_position_errors: list[np.ndarray] = []
    pose_angles: list[np.ndarray] = []
    for side, pose_slice in STATE_BC_POSE_SLICES.items():
        pose_position_errors.append(np.linalg.norm(error[:, pose_slice.start : pose_slice.start + 3], axis=1))
        pose_angles.append(_quaternion_angle(pred[:, pose_slice], truth[:, pose_slice]))
        metrics[f"{side}_position_rmse_m"] = float(
            np.sqrt(np.mean(np.square(error[:, pose_slice.start : pose_slice.start + 3])))
        )
        metrics[f"{side}_orientation_mae_rad"] = float(np.mean(pose_angles[-1]))
    for side, gripper_slice in STATE_BC_GRIPPER_SLICES.items():
        metrics[f"{side}_gripper_mae"] = float(np.mean(np.abs(error[:, gripper_slice])))
    metrics["pose_position_mae_m"] = float(np.mean(np.concatenate(pose_position_errors)))
    metrics["pose_orientation_mae_rad"] = float(np.mean(np.concatenate(pose_angles)))
    return metrics


def finite_action_envelope(actions: Any) -> Mapping[str, float]:
    """Summarize finite output ranges before any target safety gate."""

    values = _matrix(actions, name="actions")
    envelope: dict[str, float] = {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "max_abs": float(np.max(np.abs(values))),
    }
    for side, pose_slice in STATE_BC_POSE_SLICES.items():
        envelope[f"{side}_position_max_abs_m"] = float(np.max(np.abs(values[:, pose_slice.start : pose_slice.start + 3])))
    for side, gripper_slice in STATE_BC_GRIPPER_SLICES.items():
        envelope[f"{side}_gripper_min"] = float(np.min(values[:, gripper_slice]))
        envelope[f"{side}_gripper_max"] = float(np.max(values[:, gripper_slice]))
    return envelope


__all__ = ["action_metrics", "finite_action_envelope"]
