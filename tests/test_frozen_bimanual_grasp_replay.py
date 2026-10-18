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
    assert backend.inference_calls == 1


def test_duplicate_execution_arm_is_rejected_before_observation(monkeypatch):
    def unexpected_capture(_):
        pytest.fail("A duplicate component must fail before loading or preparing a scene")
    monkeypatch.setattr(replay, "prepare", unexpected_capture)
    with pytest.raises(ValueError, match="only once"):
        replay.replay(SimpleNamespace(sides=("left", "left")))


@pytest.mark.parametrize("sides", [("left",), ("left", "right")])
def test_shared_replay_uses_each_builders_canonical_envelope(monkeypatch, tmp_path, sides):
    import json
    from deployments.perception.sam2_graspgenx_point_grasp import (
        GRASPGENX_REVIEWED_REVISION,
        GRASPGENX_CHECKPOINT_REVIEWED_REVISION,
    )

    class ReachedSharedPlan(Exception):
        pass

    output = tmp_path / "output"
    output.mkdir()
    inference = tmp_path / "inference.npz"
    np.savez(inference, points=np.zeros((1, 3)), poses=np.eye(4)[None], scores=np.ones(1))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model_profile_id": "test:retained",
        "source": {"revision": GRASPGENX_REVIEWED_REVISION},
        "checkpoint_repository": {"revision": GRASPGENX_CHECKPOINT_REVIEWED_REVISION},
        "files": {name: {"sha256": "a" * 64} for name in (
            "generator_config", "generator_checkpoint",
            "discriminator_config", "discriminator_checkpoint")},
    }))
    anchor = SimpleNamespace(rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2)), intrinsics={}, reference_from_camera=np.eye(4), acquired_at=1.)
    scene = SimpleNamespace(points=np.zeros((1, 3)), target_mask=np.ones(1, dtype=bool))
    monkeypatch.setattr(replay, "prepare", lambda _: (
        anchor, np.ones((2, 2), dtype=bool),
        {s: np.zeros(6) for s in ("left", "right")},
        SimpleNamespace(scene=scene), {"qpos_measured_at": 1.}))
    profiles = {side: SimpleNamespace(side=side, collision_scene_profile="test",
        semantic_content_hash="profile:" + side, scene_world_from_canonical_station=np.eye(4),
        scene_world_from_arm_base_by_side={s: np.eye(4) for s in ("left", "right")},
        tool_tcp_pose_flange_m_rad_by_side={s: np.zeros(6) for s in ("left", "right")})
        for side in sides}
    reads, filters, motions, selections, lifecycle, bindings = [], {}, [], [], [], []

    class Envelope:
        def __init__(self, side):
            self.side = side

        @property
        def profile_id(self):
            reads.append(self.side)
            return "canonical-envelope:" + self.side

    filters.update({side: SimpleNamespace(profile=Envelope(side)) for side in sides})
    runtimes = {side: SimpleNamespace(envelope_profile=filters[side].profile,
        envelope_filter=filters[side]) for side in sides}
    builders = {side: SimpleNamespace(pre_ik_target_safety_filter=None) for side in sides}
    monkeypatch.setattr(replay, "profile_for", lambda _, side: profiles[side])
    monkeypatch.setattr(replay, "build_exclusive_rm65_skill_target_builder",
        lambda p: builders[p.side])

    def bind_target_safety(*, profile, target_builder):
        assert target_builder is builders[profile.side]
        assert target_builder.pre_ik_target_safety_filter is None
        bindings.append(profile.side)
        target_builder.pre_ik_target_safety_filter = runtimes[profile.side].envelope_filter
        return runtimes[profile.side]

    monkeypatch.setattr(replay, "bind_exclusive_rm65_skill_target_safety", bind_target_safety)
    config = SimpleNamespace(semantic_content_hash="config", time_budget_ms=10., max_solve_cycles=1,
        standard_position_tolerance_m=.01, standard_orientation_tolerance_deg=1.,
        basic_position_tolerance_m=.02, basic_orientation_tolerance_deg=2.)

    def motion(**kwargs):
        motions.append(kwargs)
        return SimpleNamespace(reachability_config=config)

    monkeypatch.setattr(replay, "RM65GraspJointTrajectoryPlanner", motion)
    frames = SimpleNamespace(source_frame_id="source", source_frame_spec_hash="source-hash",
        reference_frame_id="station", frame_graph=SimpleNamespace(semantic_content_hash="graph"))
    monkeypatch.setattr(replay, "build_real_perception_station_frame_binding", lambda **_: frames)
    monkeypatch.setattr(replay, "load_rm65_wrist_camera_side_axis_tool", lambda **_: (1., 0., 0.))
    monkeypatch.setattr(replay, "StationPointGraspArmReachabilityProfile", lambda **kw: SimpleNamespace(**kw))

    class Adapter:
        def __init__(self, **kwargs):
            self.side = kwargs["profile"].arm_side

        def select_candidates(self, *args, **kwargs):
            selections.append(self.side)
            return SimpleNamespace()

    monkeypatch.setattr(replay, "RM65PointGraspReachabilityAdapter", Adapter)

    class Point:
        last_plan_timing_ms = {}
        last_filter_summary = {}

        def __init__(self, **kwargs):
            pass

        def start(self):
            lifecycle.append("start")

        def plan_for_components(self, *args, **kwargs):
            lifecycle.append("shared-plan")
            assert tuple(kwargs["candidate_selectors"]) == sides
            for selector in kwargs["candidate_selectors"].values():
                selector((), (0., 0., 0.), 1)
            raise ReachedSharedPlan()

        def close(self):
            lifecycle.append("close")

    monkeypatch.setattr(replay, "Sam2GraspGenXPointGraspPlanner", Point)
    args = SimpleNamespace(sides=sides, output=output, inference=inference,
        model_manifest=manifest, plan_budget=1., urdf=tmp_path / "unused.urdf",
        anchor="main", pixel=(0, 0))
    with pytest.raises(ReachedSharedPlan):
        replay.replay(args)
    assert reads == list(sides)
    assert bindings == list(sides)
    assert selections == list(sides)
    assert lifecycle == ["start", "shared-plan", "close"]
    assert all(m["target_envelope_filter"] is filters[m["profile"].side] for m in motions)
    assert all(builders[side].pre_ik_target_safety_filter is filters[side] for side in sides)


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
