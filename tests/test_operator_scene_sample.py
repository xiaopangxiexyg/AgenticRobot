"""Dataset loading/export reuses native codecs and existing frozen capture readers."""

import hashlib
import json
import zipfile

import numpy as np
import pytest

from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_SCHEMA_ID,
    encode_grasp_multiview_arrays,
)
from scripts.probes.inspect_safe_move_snapshot import load_captures
from scripts.probes.load_operator_scene_sample import (
    export_replay_capture,
    load_operator_scene_sample,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def sample(
    tmp_path,
    *,
    roles=("main", "wrist_left", "wrist_right"),
    no_state=False,
    skew=0.0,
    drift=0.0,
    bad_wrist=False,
    main_state_only=False,
):
    views = []
    for i, role in enumerate(roles):
        q = {s: np.full(6, i * drift) for s in ("left", "right")}
        native = (
            role == "main" if main_state_only else role != "main"
        ) and not no_state
        metadata = {
            "schema_id": "dexteleop.perception.grasp_rgbd_snapshot.v1",
            "role": role,
            "source_id": role,
            "acquired_at": 10.0 + i * 0.01 + skew * i,
            "station_revision": 1,
            "frame_binding_hash": "binding",
            "source_binding_hash": "source",
            "rgb_content_hash": "rgb",
            "depth_content_hash": "depth",
            "source_generation": 1,
            "frame_index": i,
            "reference_frame_id": "canonical_station",
            "camera_intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
            "T_source_from_camera": np.eye(4).ravel().tolist(),
            "T_reference_from_camera": np.eye(4).tolist(),
            "stationary_only": native and not (bad_wrist and role == "wrist_right"),
            "camera_pose": {
                "measured_at": 10.0 + i * 0.01 + skew * i,
                "available_at": 10.0 + i * 0.01 + skew * i + 0.001,
                "source_evidence_refs": ["qpos:cut"],
            },
        }
        views.append(
            {
                "rgb": np.zeros((3, 4, 3), np.uint8),
                "depth_m": np.ones((3, 4), np.float32) * 0.5,
                "target_mask": np.zeros((3, 4), bool),
                "metadata": metadata,
                "qpos_by_side": q if native else None,
            }
        )
    payload = encode_grasp_multiview_arrays(
        views=views,
        metadata={
            "schema_id": GRASP_MULTIVIEW_SCHEMA_ID,
            "runtime_session_id": "real:test",
            "frame_binding_hash": "binding",
            "station_revision": 1,
        },
    )
    root = tmp_path / "sample"
    root.mkdir()
    (root / "artifact-0.zip").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    record = {
        "schema": "rpent.operator_scene_sample.v1",
        "sample_id": "1" * 32,
        "created_at": 10.0,
        "note": "scene",
        "runtime_session_id": "real:test",
        "status": "complete",
        "roles": list(roles),
        "artifacts": [
            {
                "file": "artifact-0.zip",
                "sha256": digest,
                "byte_length": len(payload),
                "binding": {
                    "artifact": {
                        "sha256": digest,
                        "byte_length": len(payload),
                        "schema_id": GRASP_MULTIVIEW_SCHEMA_ID,
                    }
                },
            }
        ],
        "observation_artifact_index": 0,
        "context": [],
    }
    (root / "record.json").write_text(json.dumps(record))
    return root


def test_native_load_and_qualified_export_feed_existing_reader(tmp_path):
    scene = load_operator_scene_sample(sample(tmp_path))
    assert len(scene.grasp_views) == 3
    assert np.all(scene.grasp_views[0].depth_m == 0.5)  # Meters, never rescaled.
    output = tmp_path / "export"
    report = export_replay_capture(scene, output)
    main = report["views"][0]
    assert not main["qpos_association"]["native_to_this_view"]
    assert main["qpos_association"]["state_source_role"] == "wrist_left"
    assert main["pose_measured_at"] == 10.01
    captures = load_captures(output, None, None, None)
    assert len(captures) == 3
    assert np.array_equal(captures[0].depth_m, scene.grasp_views[0].depth_m)
    with pytest.raises(FileExistsError):
        export_replay_capture(scene, output)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"roles": ("main",)}, "three_views"),
        ({"no_state": True}, "stationary_bilateral"),
        ({"bad_wrist": True, "drift": 1.0}, "not_stationary"),
        ({"main_state_only": True}, "stationary_bilateral"),
        ({"skew": 0.2}, "skew"),
        ({"drift": 1.0}, "drift"),
    ],
)
def test_replay_refuses_missing_state_skew_or_drift_before_writing(
    tmp_path, kwargs, reason
):
    scene = load_operator_scene_sample(sample(tmp_path, **kwargs))
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match=reason):
        export_replay_capture(scene, output)
    assert not output.exists()


def test_bundle_load_is_offline_and_digest_checked(tmp_path):
    root = sample(tmp_path)
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w") as z:
        for p in root.iterdir():
            z.write(p, p.name)
    assert load_operator_scene_sample(path).record["note"] == "scene"
    (root / "artifact-0.zip").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="digest"):
        load_operator_scene_sample(root)


def test_zip_traversal_and_pickle_are_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../record.json", "{}")
    with pytest.raises(ValueError, match="members"):
        load_operator_scene_sample(path)
