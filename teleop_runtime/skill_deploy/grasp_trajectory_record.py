"""Immutable diagnostic record of the exact sampled RM65 grasp plan.

This is written during read-only planning, before any motion admission.  It is
not an execution authority and must never be used to bypass a fresh plan.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from .motion_port import RealAgentJointTrajectory


SCHEMA = "dexteleop.real_agent_motion.grasp_trajectory_record.v1"


def load_grasp_trajectory_record(
    path: str | Path,
) -> tuple[RealAgentJointTrajectory, dict[str, Any]]:
    """Read a diagnostic record and verify the lower trajectory identity."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].tobytes().decode("utf-8"))
        trajectory = RealAgentJointTrajectory(
            time_from_start_sec=data["time_from_start_sec"],
            qpos_by_side={
                side: data[f"{side}_qpos"] for side in ("left", "right")
            },
            planner_evidence_refs=tuple(metadata["planner_evidence_refs"]),
        )
    if metadata["schema"] != SCHEMA:
        raise ValueError("Unknown grasp trajectory record schema.")
    if (
        trajectory.semantic_content_hash != metadata["trajectory_hash"]
        or source.stem != trajectory.semantic_content_hash
        or trajectory.time_from_start_sec.size != metadata["sample_count"]
        or float(trajectory.time_from_start_sec[-1]) != metadata["duration_sec"]
    ):
        raise ValueError("Grasp trajectory record does not match its identity.")
    return trajectory, metadata


def write_grasp_trajectory_record(
    directory: str | Path,
    trajectory: RealAgentJointTrajectory,
    *,
    moving_side: str,
    scene_hash: str,
    collision_scope: str,
    collision_scene_profile: str,
    target_tool_tcp_world: np.ndarray,
    speed_scale: float,
    stage_boundaries: tuple[Mapping[str, object], ...],
) -> Path:
    """Publish a full plan once, without replacing an earlier record."""

    output_dir = Path(directory)
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("Grasp trajectory record directory must already exist.")
    if moving_side not in {"left", "right"}:
        raise ValueError("moving_side must be left or right.")
    if not isinstance(trajectory, RealAgentJointTrajectory):
        raise TypeError("trajectory must be RealAgentJointTrajectory.")
    pose = np.asarray(target_tool_tcp_world, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("target_tool_tcp_world must be a finite 4x4 matrix.")
    identity = trajectory.semantic_content_hash
    destination = output_dir / f"{identity}.npz"
    metadata = {
        "schema": SCHEMA,
        "trajectory_hash": identity,
        "planner_evidence_refs": list(trajectory.planner_evidence_refs),
        "sample_count": int(trajectory.time_from_start_sec.size),
        "duration_sec": float(trajectory.time_from_start_sec[-1]),
        "moving_side": moving_side,
        "scene_hash": str(scene_hash),
        "collision_scope": collision_scope,
        "collision_scene_profile": collision_scene_profile,
        "target_tool_tcp_world": pose.tolist(),
        "speed_scale": float(speed_scale),
        "stage_boundaries": [dict(item) for item in stage_boundaries],
    }
    payload = json.dumps(
        metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".grasp-plan-", suffix=".npz", dir=output_dir,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(
                stream,
                metadata_json=np.frombuffer(payload, dtype=np.uint8),
                time_from_start_sec=trajectory.time_from_start_sec,
                left_qpos=trajectory.qpos_by_side["left"],
                right_qpos=trajectory.qpos_by_side["right"],
            )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_path, destination)
        except FileExistsError:
            existing, existing_metadata = load_grasp_trajectory_record(destination)
            if (
                existing.semantic_content_hash != identity
                or existing_metadata != metadata
            ):
                raise ValueError("Existing grasp trajectory record differs.")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


__all__ = [
    "SCHEMA",
    "load_grasp_trajectory_record",
    "write_grasp_trajectory_record",
]
