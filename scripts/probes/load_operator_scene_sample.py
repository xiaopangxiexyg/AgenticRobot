"""Load an operator sample offline; export captures for existing replay tools.

No devices, Gateway connection, model, planner, execution or lifecycle is started.
Original observation bytes use the existing public dataset_contracts codec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from dataset_contracts.grasp_multiview_snapshot import (
    decode_grasp_multiview_snapshot_arrays,
)
from deployments.perception.grasp_multiview import GraspRgbdView

_FILES = re.compile(
    r"(record\.json|artifact-[0-3]\.(zip|jpg)|context/[a-z_]+\.(json|urdf))"
)
_MAX = 48 * 1024 * 1024


@dataclass(frozen=True)
class RecordedOperatorScene:
    record: dict
    metadata: dict
    decoded: tuple
    context: dict

    @property
    def grasp_views(self):
        """Inputs for existing segmentation/fusion and grasp planners, no recapture."""
        return tuple(
            GraspRgbdView(
                role=m["role"],
                rgb=a["rgb"],
                depth_m=a["depth_m"],
                intrinsics=m["camera_intrinsics"],
                reference_from_camera=np.asarray(m["T_reference_from_camera"]).reshape(
                    4, 4
                ),
                acquired_at=m["acquired_at"],
            )
            for a, m in self.decoded
        )


def load_operator_scene_sample(path: Path) -> RecordedOperatorScene:
    """Verify the durable manifest and decode through the authoritative codec."""
    path = Path(path)
    if path.is_dir():

        def read(name):
            if _FILES.fullmatch(name) is None:
                raise ValueError("recorded_scene_path_invalid")
            p = path / name
            if p.is_symlink() or not p.is_file() or p.stat().st_size > _MAX:
                raise ValueError("recorded_scene_file_invalid")
            return p.read_bytes()
    else:
        if path.stat().st_size > _MAX:
            raise ValueError("recorded_scene_bundle_too_large")
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if (
                len(infos) > 20
                or len({i.filename for i in infos}) != len(infos)
                or any(_FILES.fullmatch(i.filename) is None for i in infos)
                or sum(i.file_size for i in infos) > _MAX
            ):
                raise ValueError("recorded_scene_bundle_members_invalid")
            payload = {i.filename: archive.read(i) for i in infos}
        read = payload.__getitem__
    record = json.loads(read("record.json"))
    if (
        record.get("schema") != "rpent.operator_scene_sample.v1"
        or record.get("status") not in ("complete", "partial", "failed")
        or len(record.get("artifacts", [])) > 4
        or len(record.get("context", [])) > 8
    ):
        raise ValueError("recorded_scene_not_a_completed_capture")
    blobs = []
    for item in record["artifacts"]:
        data = read(item["file"])
        if (
            len(data) != item["byte_length"]
            or hashlib.sha256(data).hexdigest() != item["sha256"]
        ):
            raise ValueError("recorded_scene_artifact_digest_mismatch")
        if item["binding"]["artifact"]["sha256"] != item["sha256"] or item["binding"][
            "artifact"
        ]["byte_length"] != len(data):
            raise ValueError("recorded_scene_artifact_identity_mismatch")
        blobs.append(data)
    index = record["observation_artifact_index"]
    if type(index) is not int or not 0 <= index < len(blobs):
        raise ValueError("recorded_scene_observation_missing")
    decoded, metadata, points, _ = decode_grasp_multiview_snapshot_arrays(blobs[index])
    if (
        points is not None
        or metadata.get("runtime_session_id") != record["runtime_session_id"]
    ):
        raise ValueError("recorded_scene_observation_provenance_invalid")
    if record["artifacts"][index]["binding"]["artifact"]["schema_id"] != metadata.get(
        "schema_id"
    ):
        raise ValueError("recorded_scene_artifact_schema_mismatch")
    for _, view_metadata in decoded:
        if any(
            view_metadata.get(k) != metadata.get(k)
            for k in ("frame_binding_hash", "station_revision")
        ):
            raise ValueError("recorded_scene_cross_view_binding_mismatch")
    context = {}
    for item in record.get("context", []):
        data = read(item["file"])
        if (
            hashlib.sha256(data).hexdigest() != item["sha256"]
            or len(data) != item["byte_length"]
        ):
            raise ValueError("recorded_scene_context_digest_mismatch")
        context[item["file"].split("/")[-1]] = data
    return RecordedOperatorScene(record, metadata, tuple(decoded), context)


def export_replay_capture(scene: RecordedOperatorScene, output: Path):
    """Export exact metric arrays and qualified measured-state associations.

    A static camera may have no native qpos in capture_rgbd@1. Associate it
    only with a nearby stationary wrist cut and preserve that distinction.
    No interpolation, invented state or present-day camera transform is used.
    """
    from teleop_runtime.skill_deploy.motion_port import (
        REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD,
    )
    from teleop_runtime.skill_deploy.rm65_profile import (
        RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
    )

    decoded = scene.decoded
    if {m["role"] for _, m in decoded} != {"main", "wrist_left", "wrist_right"}:
        raise ValueError("recorded_scene_requires_three_views_for_replay")
    times = [m["acquired_at"] for _, m in decoded]
    if np.ptp(times) > 0.1:
        raise ValueError("recorded_scene_inter_camera_skew_exceeded")
    states = [
        (a, m)
        for a, m in decoded
        if a["left_qpos"].shape == (6,)
        and m.get("stationary_only") is True
        and m["role"] in ("wrist_left", "wrist_right")
    ]
    if not states:
        raise ValueError("recorded_scene_stationary_bilateral_state_missing")
    used_native_states = []
    for a, m in decoded:
        if a["left_qpos"].shape == (6,):
            if m.get("stationary_only") is not True:
                raise ValueError("recorded_scene_native_state_not_stationary")
            used_native_states.append((a, m))
        elif m["role"] != "main":
            raise ValueError("recorded_scene_wrist_native_state_missing")
    for a, _ in used_native_states:
        for side in ("left", "right"):
            if (
                np.max(np.abs(a[side + "_qpos"] - states[0][0][side + "_qpos"]))
                > REAL_AGENT_TRAJECTORY_START_MAX_DRIFT_RAD
            ):
                raise ValueError("recorded_scene_qpos_drift_exceeded")
    views = []
    # Qualify all metadata before creating any output.
    for arrays, m in decoded:
        pair = (
            (arrays, m)
            if arrays["left_qpos"].shape == (6,)
            else min(states, key=lambda s: abs(s[1]["acquired_at"] - m["acquired_at"]))
        )
        state_arrays, state_meta = pair
        measured = state_meta.get("measured_qpos", state_meta.get("camera_pose", {}))
        measured_at = measured.get("measured_at")
        available_at = measured.get("available_at")
        if (
            not isinstance(measured_at, (float, int))
            or not isinstance(available_at, (float, int))
            or not np.isfinite([measured_at, available_at]).all()
            or available_at < measured_at
            or abs(measured_at - m["acquired_at"]) > 0.1
        ):
            raise ValueError("recorded_scene_state_frame_skew_exceeded")
        pose = np.asarray(m["T_source_from_camera"], dtype=float).reshape(4, 4)
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("recorded_scene_camera_transform_invalid")
        views.append(
            {
                "role": m["role"],
                "source_id": m["source_id"],
                "file": m["role"] + ".npz",
                "acquired_at": m["acquired_at"],
                "station_revision": m["station_revision"],
                "source_generation": m["source_generation"],
                "frame_index": m["frame_index"],
                "source_binding_hash": m["source_binding_hash"],
                "rgb_content_hash": m["rgb_content_hash"],
                "depth_content_hash": m["depth_content_hash"],
                "intrinsics": m["camera_intrinsics"],
                "T_world_from_camera": pose.tolist(),
                "capture_qpos_by_side": {
                    s: state_arrays[s + "_qpos"].tolist() for s in ("left", "right")
                },
                "pose_measured_at": measured_at,
                "pose_available_at": available_at,
                "pose_evidence_refs": measured["source_evidence_refs"],
                "qpos_association": {
                    "native_to_this_view": pair[1] is m,
                    "state_source_role": state_meta["role"],
                    "state_source_acquired_at": state_meta["acquired_at"],
                    "scope": "bounded_stationary_capture_association_not_interpolated",
                },
            }
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    by_role = {m["role"]: a for a, m in decoded}
    for view in views:
        arrays = by_role[view["role"]]
        np.savez_compressed(
            output / view["file"], rgb=arrays["rgb"], depth_m=arrays["depth_m"]
        )
        Image.fromarray(arrays["rgb"]).save(
            output / (view["role"] + ".jpg"), quality=95
        )
    report = {
        "schema": "agenticrobot.safe_move_inputs.v1",
        "reference_frame_id": RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
        "configuration": {
            "roles": ["main", "wrist_left", "wrist_right"],
            "maximum_depth_m": 1.5,
            "maximum_scene_age_sec": 30.0,
            "voxel_size_m": 0.004,
        },
        "recorded_at": scene.record["created_at"],
        "request": {},
        "views": sorted(
            views, key=lambda v: ["main", "wrist_left", "wrist_right"].index(v["role"])
        ),
        "operator_scene_sample_id": scene.record["sample_id"],
        "source_capture_status": scene.record["status"],
        "note": scene.record["note"],
        "scope": "offline_historical_input_not_live_authorization",
    }
    (output / "snapshot.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    if scene.context:
        (output / "context").mkdir()
        for name, data in scene.context.items():
            (output / "context" / name).write_bytes(data)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        required=True,
        type=Path,
        help="Sample directory or downloaded scene ZIP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for the existing frozen replay/inspection tools",
    )
    args = parser.parse_args()
    scene = load_operator_scene_sample(args.sample)
    result = {
        "sample_id": scene.record["sample_id"],
        "note": scene.record["note"],
        "roles": [v.role for v in scene.grasp_views],
        "qpos_roles": [m["role"] for a, m in scene.decoded if a["left_qpos"].size],
        "devices_opened": False,
        "models_started": False,
        "motion_executed": False,
        "context_files": list(scene.context),
    }
    if args.output:
        export_replay_capture(scene, args.output)
        result["exported_capture"] = str(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
