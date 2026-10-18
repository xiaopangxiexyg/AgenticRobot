from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.probes import replay_bimanual_grasp_snapshot as replay


@pytest.mark.parametrize("scene", (
    "dual_arm_hand_proxy_dmtac_strict", "dual_arm_hand_proxy_dmtac_grasp"))
def test_observation_camera_does_not_select_the_motion_arm(monkeypatch, scene):
    monkeypatch.setattr(replay, "ExclusiveRM65SkillDeploymentProfile", lambda **kwargs: kwargs)
    monkeypatch.setattr(replay, "resolve_exclusive_rm65_skill_deployment_profile", lambda value: value)
    args = SimpleNamespace(repo=Path("/repo"), canonical=Path("/canonical"),
        anchor="wrist_left", collision_scene_profile=scene)
    left, right = (replay.profile_for(args, side) for side in ("left", "right"))
    assert left["envelope_profile_path"].name == "agent_grasp_left_absolute_standard_v1.json"
    assert right["envelope_profile_path"].name == "agent_grasp_right_absolute_standard_v1.json"
    for profile in (left, right):
        assert set(profile["tool_profile_path_by_side"]) == {"left", "right"}
        assert profile["collision_scene_profile"] == scene


def test_retained_model_output_cannot_be_reused_for_another_cloud(tmp_path):
    points = np.arange(12, dtype=np.float32).reshape(4, 3)
    path = tmp_path / "inference.npz"
    np.savez(path, points=points, poses=np.eye(4)[None], scores=np.array([.8]))
    backend = replay.RetainedInference(path)
    poses, scores = backend.infer_object(points.copy())
    assert poses.shape == (1, 4, 4)
    assert scores.tolist() == [.8]
    with pytest.raises(ValueError, match="exact target cloud"):
        backend.infer_object(points+.01)
    with pytest.raises(ValueError, match="exact target cloud"):
        backend.infer_object(points[:2])


@pytest.mark.parametrize("status,clean", [("ready", False), ("cleanup_failed", False), ("closed", True)])
def test_persistent_snapshot_requires_eventual_session_cleanup(tmp_path, status, clean):
    import json
    capture = tmp_path / "state_0001/capture"
    capture.mkdir(parents=True)
    device = tmp_path / "devices"
    device.mkdir()
    report = {"schema": "agenticrobot.readonly_rgbd_qpos.v2", "status": "captured", "stationarity": {},
        "session_lifecycle": "../../devices/lifecycle.json"}
    (capture / "snapshot.json").write_text(json.dumps(report))
    (device / "lifecycle.json").write_text(json.dumps({"status": status, "cleanup_complete": clean}))
    if clean:
        assert replay.load_snapshot_report(capture) == report
        report["session_lifecycle"] = "../../outside.json"
        (capture / "snapshot.json").write_text(json.dumps(report))
        with pytest.raises(ValueError, match="lifecycle reference"):
            replay.load_snapshot_report(capture)
    else:
        with pytest.raises(ValueError, match="still active"):
            replay.load_snapshot_report(capture)


def test_auxiliary_mask_is_bound_to_both_image_and_association_seed(tmp_path):
    import json
    from PIL import Image
    rgb = np.full((10, 10, 3), 20, dtype=np.uint8)
    mask = np.ones((10, 10), dtype=bool)
    np.save(tmp_path / "rgb.npy", rgb)
    np.save(tmp_path / "mask.npy", mask)
    (tmp_path / "result.json").write_text(json.dumps({"mode": "segment", "pixel_uv": [5, 6]}))
    segment = replay.RetainedAuxiliarySegmenter([tmp_path])
    np.testing.assert_array_equal(segment(Image.fromarray(rgb), pixel_uv=(5, 6)), mask)
    with pytest.raises(RuntimeError, match="exact image/seed"):
        segment(Image.fromarray(rgb+1), pixel_uv=(5, 6))
    with pytest.raises(RuntimeError, match="exact image/seed"):
        segment(Image.fromarray(rgb), pixel_uv=(5, 5))
