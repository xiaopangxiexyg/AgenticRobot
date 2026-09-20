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

from dataset_contracts.canonical_encoding import canonical_content_hash

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
        if "source_waypoint_hash" in metadata:
            if not {"source_left_qpos", "source_right_qpos"} <= set(data.files):
                raise ValueError("Grasp trajectory record lost its source waypoints.")
            source_waypoints = {
                side: np.asarray(data[f"source_{side}_qpos"], dtype=float)
                for side in ("left", "right")
            }
            if (
                source_waypoints["left"].shape
                != source_waypoints["right"].shape
                or source_waypoints["left"].ndim != 2
                or source_waypoints["left"].shape[1] != 6
                or not all(np.all(np.isfinite(value)) for value in source_waypoints.values())
                or source_waypoints["left"].shape[0]
                != int(metadata["source_waypoint_count"])
                or canonical_content_hash(
                    {
                        side: source_waypoints[side]
                        for side in ("left", "right")
                    },
                    domain="dexteleop.real_agent_motion.grasp_source_waypoints.v1",
                )
                != metadata["source_waypoint_hash"]
            ):
                raise ValueError("Grasp trajectory source waypoints do not match metadata.")
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
    parameterization: Mapping[str, object] | None = None,
    source_waypoints_by_side: Mapping[str, np.ndarray] | None = None,
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
    source_waypoints: dict[str, np.ndarray] | None = None
    if source_waypoints_by_side is not None:
        if set(source_waypoints_by_side) != {"left", "right"}:
            raise ValueError("source_waypoints_by_side must contain left and right.")
        source_waypoints = {
            side: np.asarray(source_waypoints_by_side[side], dtype=float).copy()
            for side in ("left", "right")
        }
        if (
            source_waypoints["left"].shape != source_waypoints["right"].shape
            or source_waypoints["left"].ndim != 2
            or source_waypoints["left"].shape[0] < 1
            or source_waypoints["left"].shape[1] != 6
            or not all(np.all(np.isfinite(value)) for value in source_waypoints.values())
        ):
            raise ValueError(
                "source waypoints must be matching finite waypoint_count x 6 arrays."
            )
    parameterization_payload = (
        None
        if parameterization is None
        else json.loads(
            json.dumps(
                dict(parameterization),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    )
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
    if parameterization_payload is not None:
        metadata["parameterization"] = parameterization_payload
    if source_waypoints is not None:
        metadata.update(
            {
                "source_waypoint_count": int(source_waypoints["left"].shape[0]),
                "source_waypoint_hash": canonical_content_hash(
                    source_waypoints,
                    domain="dexteleop.real_agent_motion.grasp_source_waypoints.v1",
                ),
            }
        )
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
            arrays = {
                "metadata_json": np.frombuffer(payload, dtype=np.uint8),
                "time_from_start_sec": trajectory.time_from_start_sec,
                "left_qpos": trajectory.qpos_by_side["left"],
                "right_qpos": trajectory.qpos_by_side["right"],
            }
            if source_waypoints is not None:
                arrays.update(
                    {
                        "source_left_qpos": source_waypoints["left"],
                        "source_right_qpos": source_waypoints["right"],
                    }
                )
            np.savez_compressed(stream, **arrays)
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
