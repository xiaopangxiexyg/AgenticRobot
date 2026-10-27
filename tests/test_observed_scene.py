"""Sensor-only geometry and retained-input reuse without a backend owner."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.observed_scene import (
    SafeMotionObservationConfig,
    obstacle_cloud_from_captures,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def sensor_cut(**changes):
    fields = dict(
        role="main", source_id="front", acquired_at=10., station_revision=1,
        source_binding_hash="calibration:1", source_generation=1, frame_index=5,
        width=4, height=4, depth_m=np.full((4, 4), .5),
        depth_content_hash="depth:1", source_from_camera=tuple(np.eye(4).ravel()),
        intrinsics={"fx": 100., "fy": 100., "cx": 2., "cy": 2.},
        capture_qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        pose_measured_at=10., pose_evidence_refs=("encoders:1",),
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


def unfiltered(**kwargs):
    assert not kwargs["target_mask"].any()
    return SimpleNamespace(depth_m=kwargs["depth_m"], evidence={"method": "unit_identity"})


def cloud(cut, **changes):
    kwargs = dict(captures=(cut,), config=SafeMotionObservationConfig(roles=("main",)),
        reference_frame_id="station", self_filter=unfiltered,
        planning_qpos_by_side={side: np.zeros(6) for side in ("left", "right")},
        max_inter_camera_skew_sec=.1, max_qpos_drift_rad=.01)
    kwargs.update(changes)
    return obstacle_cloud_from_captures(**kwargs)


def test_retained_cuts_reused_without_acquisition_or_segmentation_and_not_mutated():
    cut = sensor_cut()
    before = cut.depth_m.copy()
    first, second = cloud(cut), cloud(cut)
    assert first.semantic_content_hash == second.semantic_content_hash
    np.testing.assert_array_equal(first.points_m, second.points_m)
    np.testing.assert_array_equal(cut.depth_m, before)
    assert not first.points_m.flags.writeable
    assert first.evidence["views"][0]["frame_index"] == 5
    assert first.evidence["views"][0]["acquired_at"] == 10.
    assert first.evidence["views"][0]["depth_content_hash"] == "depth:1"


def test_observed_changes_not_hidden_scene_metadata_drive_geometry():
    cut = sensor_cut()
    first = cloud(cut)
    # Native scene facts have no place in the projection interface.
    cut.hidden_object_pose = (900., 800., 700.)
    cut.native_body_ids = (1, 2, 3)
    unchanged = cloud(cut)
    assert unchanged.semantic_content_hash == first.semantic_content_hash
    changed = cloud(sensor_cut(depth_m=np.full((4, 4), .6), depth_content_hash="depth:2"))
    assert changed.semantic_content_hash != first.semantic_content_hash
    assert changed.points_m[:, 2].mean() > first.points_m[:, 2].mean() + .09


def test_capture_time_camera_geometry_is_preserved_when_observations_are_reused():
    pose = np.eye(4)
    pose[0, 3] = .2
    projected = cloud(sensor_cut(source_from_camera=tuple(pose.ravel())))
    original = cloud(sensor_cut())
    assert projected.points_m[:, 0].mean() - original.points_m[:, 0].mean() == pytest.approx(.2)
    assert projected.evidence["views"][0]["T_world_from_camera"][0][3] == .2


@pytest.mark.parametrize("field", ["max_qpos_drift_rad", "max_inter_camera_skew_sec"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True,
                                  np.bool_(True), np.bool_(False)])
def test_new_independent_geometry_boundary_rejects_invalid_admission_bounds(field, value):
    with pytest.raises(ValueError, match=field):
        cloud(sensor_cut(), **{field: value})


@pytest.mark.parametrize("pair", [{"right": np.zeros(6)},
    {"left": np.zeros(5), "right": np.zeros(6)},
    {"left": np.full(6, np.nan), "right": np.zeros(6)}])
def test_current_joint_measurements_cannot_bypass_drift_admission(pair):
    with pytest.raises(ValueError, match="measured dual-arm"):
        cloud(sensor_cut(), planning_qpos_by_side=pair)


def test_reuse_does_not_relax_the_consuming_profile_joint_drift_bound():
    pair = {side: np.full(6, .02) for side in ("left", "right")}
    with pytest.raises(ValueError, match="differs from safe-move start"):
        cloud(sensor_cut(), planning_qpos_by_side=pair)


def test_geometry_import_has_no_backend_or_gateway_dependency():
    program = (
        "import json,sys; import robot_motion.observed_scene; "
        "print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] "
        "in {'agent_gateway','sim','teleop_runtime','hardware','pyrealsense2','mujoco'})))"
    )
    result = subprocess.run([sys.executable, "-c", program], check=True,
                            text=True, capture_output=True, timeout=30)
    assert json.loads(result.stdout) == []
