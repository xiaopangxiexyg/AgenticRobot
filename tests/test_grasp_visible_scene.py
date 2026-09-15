import numpy as np
import pytest

from robot_motion.grasp_visible_scene import ParallelJawSolidGeometry, VisibleGraspScene
from dataset_contracts.grasp_rgbd_snapshot import (
    encode_grasp_rgbd_snapshot, decode_grasp_rgbd_snapshot, GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
)


GEOMETRY = ParallelJawSolidGeometry(-0.043, 0.05, 0.01, 0.0255, 0.07)


def scene(points, labels):
    return VisibleGraspScene(points=np.asarray(points).reshape(-1, 3),
                             target_mask=np.asarray(labels, dtype=bool), geometry=GEOMETRY)


def test_opening_is_free_but_fingers_and_palm_are_solid():
    view = scene([[0, 0, -0.02], [0.04, 0, -0.02], [0, 0, -0.05]], [True, True, False])
    assert view.collision_counts(np.eye(4), opening_m=0.07) == (1, 1)


def test_endpoint_clear_does_not_hide_thin_obstacle_in_approach_sweep():
    view = scene([[0.04, 0, -0.08]], [False])
    assert view.collision_counts(np.eye(4), opening_m=0.07) == (0, 0)
    assert view.approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.06) == (0, 1)
    assert view.approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.02) == (0, 0)


def test_target_is_not_deleted_from_noncontact_approach():
    view = scene([[0.04, 0, -0.08]], [True])
    assert view.approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.06) == (1, 0)


def test_pose_and_reference_transform_do_not_change_collision_counts():
    view = scene([[0.04, 0, -0.08]], [False])
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [0.2, -0.4, 0.1]
    moved = scene(view.points @ transform[:3, :3].T+transform[:3, 3], [False])
    assert moved.approach_collision_counts(transform, opening_m=0.07, distance_m=0.06) == (0, 1)


@pytest.mark.parametrize("opening", [0, -0.1, 0.071, float("nan")])
def test_invalid_openings_fail_closed(opening):
    with pytest.raises(ValueError):
        scene([], []).collision_counts(np.eye(4), opening_m=opening)


def test_no_visible_points_is_not_an_exception_or_full_scene_proof():
    assert scene([], []).approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.06) == (0, 0)


def test_rgbd_snapshot_retains_original_arrays_and_both_measured_arms():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[2, 3] = [1, 2, 255]
    depth = np.full((4, 5), 0.4, dtype=np.float32)
    mask = np.zeros((4, 5), dtype=bool)
    mask[2, 3] = True
    qpos = {"left": np.arange(6), "right": np.arange(6)+0.5}
    data = encode_grasp_rgbd_snapshot(rgb=rgb, depth_m=depth, target_mask=mask,
        qpos_by_side=qpos, metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID, "capture": {"frame_index": 42}})
    arrays, metadata = decode_grasp_rgbd_snapshot(data)
    for name, value in [("rgb", rgb), ("depth_m", depth), ("target_mask", mask),
                        ("left_qpos", qpos["left"]), ("right_qpos", qpos["right"])]:
        np.testing.assert_array_equal(arrays[name], value)
    assert metadata["capture"]["frame_index"] == 42


def test_snapshot_invalid_depth_does_not_become_free_space():
    with pytest.raises(ValueError):
        encode_grasp_rgbd_snapshot(rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            depth_m=np.array([[float("nan")]], dtype=np.float32),
            target_mask=np.ones((1, 1), dtype=bool),
            metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID})


def test_oversized_npy_header_is_rejected_before_numpy_can_allocate(monkeypatch):
    import io
    import zipfile
    # Valid representation first, then replace one member header without
    # allocating its claimed gigabytes or touching any real file.
    valid = encode_grasp_rgbd_snapshot(rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        depth_m=np.ones((1, 1), dtype=np.float32), target_mask=np.ones((1, 1), dtype=bool),
        metadata={"schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID})
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        "descr": "<f4", "fortran_order": False, "shape": (1_000_000_000, 1),
    })
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(valid)) as original, zipfile.ZipFile(out, "w") as changed:
        for name in original.namelist():
            changed.writestr(name, header.getvalue() if name == "depth_m.npy" else original.read(name))
    monkeypatch.setattr(np, "load", lambda *_args, **_kwargs: pytest.fail("unsafe allocation attempted"))
    with pytest.raises(ValueError, match="bounded body"):
        decode_grasp_rgbd_snapshot(out.getvalue())
