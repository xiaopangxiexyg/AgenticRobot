"""Read-only simulator binding of the existing RM65 tool calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from project_paths import resolve_path
from robot_motion.pose import pose6_to_matrix


def load_endpoint_to_tool_transform(path: str | Path, *, side: str) -> np.ndarray:
    """Load TCP geometry without importing a vendor hardware backend."""
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"RM65 tool profile {resolved} must contain a JSON object.")  # noqa: TRY004 - calibration validation contract
    arm = str(payload.get("arm", "")).strip().lower()
    if arm and arm != str(side).strip().lower():
        raise ValueError(f"RM65 tool profile {resolved} is for arm {arm!r}, expected {side!r}.")
    tcp = payload.get("tcp")
    if not isinstance(tcp, dict):
        raise ValueError(f"RM65 tool profile {resolved} is missing tcp metadata.")  # noqa: TRY004 - calibration validation contract
    values = np.asarray(tcp.get("pose_flange_m_rad"), dtype=float).reshape(-1)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError(f"RM65 tool profile {resolved} tcp.pose_flange_m_rad must contain six finite values.")
    return pose6_to_matrix(values)


def resolve_task_grasp_frame(scene, config) -> tuple[int, np.ndarray]:
    """Resolve a task's explicitly chosen body frame or calibrated tool TCP.

    The returned offset is body-from-grasp. Call once per scene; no file reads
    or geometry reconstruction belong in the physics loop.
    """
    if config.grasp_frame_source == "body":
        return scene.body_id(config.right_grasp_body_name), np.eye(4)
    if config.grasp_frame_source != "tool_tcp":
        raise ValueError("Unknown task grasp-frame source.")
    side = config.controlled_side
    body = scene.arm_specs[side].controlled_body
    profile_path = getattr(config, f"{side}_tool_profile_path")
    return scene.body_id(body), load_endpoint_to_tool_transform(profile_path, side=side)
