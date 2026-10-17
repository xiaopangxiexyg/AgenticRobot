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


def test_offset_solid_frame_is_used_by_endpoint_sweep_and_capture_filter():
    from dataclasses import replace
    geometry = replace(GEOMETRY, tool_from_solid_xyz_m=(.0245, 0, .0082),
                       solid_yaw_rad=np.pi/2)
    solid_points = np.array([[0, 0, -.05], [.04, 0, -.02], [.04, 0, -.08]])
    pose = geometry.solid_pose(np.eye(4))
    view = VisibleGraspScene(points=solid_points @ pose[:3, :3].T + pose[:3, 3],
                             target_mask=np.array([False, True, False]), geometry=geometry)
    assert view.collision_counts(np.eye(4), opening_m=.07) == (1, 1)
    assert view.approach_collision_counts(np.eye(4), opening_m=.07, distance_m=.06) == (1, 2)
    filtered, removed = view.without_captured_gripper_observations({"right": np.eye(4)})
    assert removed == 1
    assert filtered.collision_counts(np.eye(4), opening_m=.07) == (1, 0)
    assert filtered.approach_collision_counts(np.eye(4), opening_m=.07, distance_m=.06) == (1, 1)
    assert geometry.solid_pose(np.eye(4))[:3, 2].tolist() == [0, 0, 1]


@pytest.mark.parametrize("fields", [{"tool_from_solid_xyz_m": (0, 0)},
                                    {"tool_from_solid_xyz_m": (0, np.nan, 0)},
                                    {"solid_yaw_rad": np.inf}])
def test_invalid_solid_frame_fails_at_geometry_boundary(fields):
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(GEOMETRY, **fields)


def test_endpoint_clear_does_not_hide_thin_obstacle_in_approach_sweep():
    view = scene([[0.04, 0, -0.08]], [False])
    assert view.collision_counts(np.eye(4), opening_m=0.07) == (0, 0)
    assert view.approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.06) == (0, 1)
    assert view.approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.02) == (0, 0)


def test_target_is_not_deleted_from_noncontact_approach():
    view = scene([[0.04, 0, -0.08]], [True])
    assert view.approach_collision_counts(np.eye(4), opening_m=0.07, distance_m=0.06) == (1, 0)


def test_terminal_contact_allows_only_target_on_inner_tactile_pad_skin():
    # With a 70 mm opening, the nominal inner finger faces are at +/-35 mm.
    # The first two target points enter those faces by 3 mm.  The remaining
    # target points are respectively too deep, on the outer face, and in the
    # palm.  A background point at an otherwise allowed pad location must stay
    # an obstacle.
    view = scene(
        [
            [0.038, 0.0, -0.020],
            [-0.038, 0.0, -0.020],
            [0.044, 0.0, -0.020],
            [0.044, 0.0, -0.020],
            [0.038, 0.0, -0.050],
            [0.038, 0.0, -0.020],
        ],
        [True, True, True, True, True, False],
    )

    target_hits, allowed_contacts, background_hits = (
        view.collision_counts_with_terminal_target_contact(
            np.eye(4), opening_m=0.07, contact_depth_m=0.006
        )
    )

    assert allowed_contacts == 2
    assert target_hits == 3
    assert background_hits == 1


def test_terminal_contact_depth_is_bounded_by_finger_thickness():
    with pytest.raises(ValueError, match="within finger thickness"):
        scene([], []).collision_counts_with_terminal_target_contact(
            np.eye(4), opening_m=0.07, contact_depth_m=0.0101
        )


def test_captured_gripper_filter_preserves_targets_gap_sweep_and_nearby_background():
    view = scene([[0, 0, -.05], [0, 0, -.05], [.04, 0, -.02],
                  [0, 0, -.054], [0, .013, -.05], [0, 0, -.02],
                  [-.04, 0, -.02], [.04, 0, -.02], [.04, 0, -.08]],
                 [False, True, False, False, False, False, False, True, False])
    filtered, removed = view.without_captured_gripper_observations({"right": np.eye(4)})
    assert removed == 3
    np.testing.assert_array_equal(filtered.points, view.points[[1, 3, 4, 5, 7, 8]])
    assert filtered.collision_counts(np.eye(4), opening_m=.07) == (2, 0)
    assert filtered.approach_collision_counts(np.eye(4), opening_m=.07, distance_m=.06) == (2, 2)
    assert len(view.points) == 9


def test_capture_surface_margin_removes_only_near_solid_background_and_targets_stay():
    points = np.asarray([
        [.049, 0, -.02],     # within the 5 mm finger surface band
        [.049, 0, -.02],     # target at the same location must stay
        [.0501, 0, -.02],    # just beyond the reviewed surface band
        [0, 0, -.02],        # open finger gap
        [.04, .0178, -.02],  # beyond the padded finger height
        [.04, 0, -.054],     # within the padded palm surface band
    ])
    view = scene(points, [False, True, False, False, False, False])

    exact, exact_removed = view.without_captured_gripper_observations(
        {"right": np.eye(4)}
    )
    filtered, removed = view.without_captured_gripper_observations(
        {"right": np.eye(4)}, surface_margin_m=.005
    )

    assert exact_removed == 0
    np.testing.assert_array_equal(exact.points, points)
    assert removed == 2
    np.testing.assert_array_equal(filtered.points, points[[1, 2, 3, 4]])
    assert filtered.target_mask.sum() == 1
    assert filtered.collision_counts(np.eye(4), opening_m=.07) == (0, 0)


@pytest.mark.parametrize("margin", [-.001, .0101, np.inf, True])
def test_capture_surface_margin_is_bounded(margin):
    with pytest.raises(ValueError, match="surface_margin_m"):
        scene([], []).without_captured_gripper_observations(
            {"right": np.eye(4)}, surface_margin_m=margin
        )


@pytest.mark.parametrize("local_point", [[0, 0, -.05], [.04, 0, -.02], [-.04, 0, -.02]])
def test_gripper_filter_uses_capture_pose_not_a_later_robot_pose(local_point):
    capture = np.eye(4)
    later = np.eye(4)
    later[0, 3] = .2
    point = np.array(local_point)
    view = scene([point, point + later[:3, 3]], [False, False])
    filtered, removed = view.without_captured_gripper_observations({"right": capture})
    assert removed == 1
    assert filtered.collision_counts(later, opening_m=.07) == (0, 1)


def test_gripper_filter_uses_both_capture_poses_and_preserves_finger_gaps():
    poses = {"right": np.eye(4), "left": np.eye(4)}
    poses["left"][:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses["left"][:3, 3] = [.2, .3, .1]
    local = np.array([[0, 0, -.05], [.04, 0, -.02], [-.04, 0, -.02],
                      [0, 0, -.02], [.04, 0, -.08]])
    points = np.concatenate([local @ pose[:3, :3].T + pose[:3, 3]
                             for pose in poses.values()])
    view = scene(points, np.zeros(len(points), dtype=bool))
    filtered, removed = view.without_captured_gripper_observations(poses)
    assert removed == 6
    np.testing.assert_array_equal(filtered.points, points[[3, 4, 8, 9]])
    for pose in poses.values():
        assert filtered.collision_counts(pose, opening_m=.07) == (0, 0)
        assert filtered.approach_collision_counts(pose, opening_m=.07, distance_m=.06) == (0, 1)


def test_capture_qpos_are_copied_immutable_and_retained_after_filter():
    qpos = {side: np.zeros(6) for side in ("left", "right")}
    view = VisibleGraspScene(points=np.empty((0, 3)), target_mask=np.empty(0, dtype=bool),
                             geometry=GEOMETRY, capture_qpos_by_side=qpos)
    qpos["right"][0] = 1
    assert view.capture_qpos_by_side["right"][0] == 0
    with pytest.raises(ValueError):
        view.capture_qpos_by_side["right"][0] = 2
    filtered, _ = view.without_captured_gripper_observations({"right": np.eye(4)})
    np.testing.assert_array_equal(filtered.capture_qpos_by_side["right"], np.zeros(6))


def test_capture_robot_filter_evidence_is_hashed_and_retained_after_analytic_filter():
    evidence = {
        "schema": "dexteleop.rm65.captured_robot_observation_filter.v1",
        "removed_background_points": 9,
        "removed_target_points": 0,
    }
    view = VisibleGraspScene(
        points=np.empty((0, 3)),
        target_mask=np.empty(0, dtype=bool),
        geometry=GEOMETRY,
        capture_self_observation_evidence=evidence,
    )
    evidence["removed_background_points"] = 10

    assert view.capture_self_observation_evidence["removed_background_points"] == 9
    filtered, _ = view.without_captured_gripper_observations(
        {"right": np.eye(4)}
    )
    assert filtered.capture_self_observation_evidence == (
        view.capture_self_observation_evidence
    )
    assert filtered.semantic_content_hash == view.semantic_content_hash


@pytest.mark.parametrize(
    "evidence",
    (
        {"removed_background_points": -1, "removed_target_points": 0},
        {"removed_background_points": 1, "removed_target_points": 1},
        {"removed_target_points": 0},
    ),
)
def test_invalid_capture_robot_filter_evidence_fails_closed(evidence):
    with pytest.raises(ValueError, match="capture self-observation evidence"):
        VisibleGraspScene(
            points=np.empty((0, 3)),
            target_mask=np.empty(0, dtype=bool),
            geometry=GEOMETRY,
            capture_self_observation_evidence=evidence,
        )


@pytest.mark.parametrize("qpos", [{"right": np.zeros(6)},
                                  {"right": np.zeros(6), "left": np.empty(0)},
                                  {"right": np.zeros(6), "left": np.full(6, np.nan)}])
def test_invalid_capture_qpos_cannot_filter_scene(qpos):
    with pytest.raises(ValueError, match="capture qpos"):
        VisibleGraspScene(points=np.empty((0, 3)), target_mask=np.empty(0, dtype=bool),
                          geometry=GEOMETRY, capture_qpos_by_side=qpos)


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


@pytest.mark.parametrize("count", [0, 1, 10000])
def test_native_query_index_reuse_matches_original_xyz_and_labels(count):
    import math

    rng = np.random.default_rng(42)
    view = scene(rng.uniform(-.2, .2, (count, 3)), rng.random(count) > .5)
    for opening, distance in ((.07, 0.), (.035, .06), (.07, .1)):
        for yaw in (0., .7, -1.2):
            pose = np.eye(4)
            pose[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            pose[:3, 3] = rng.uniform(-.05, .05, 3)
            g = view.geometry
            radius = math.sqrt((opening/2+g.finger_thickness_m)**2 +
                (g.pad_height_m/2)**2 +
                max(abs(g.palm_z_m-g.finger_thickness_m-distance), abs(g.palm_z_m+g.finger_length_m))**2)
            indices = view._tree.query_ball_point(pose[:3, 3], radius+1e-9)
            expected_points = (view.points[indices]-pose[:3, 3]) @ pose[:3, :3]
            actual_points, actual_labels = view._local_points(pose, opening_m=opening,
                distance_m=distance)
            np.testing.assert_array_equal(actual_points, expected_points)
            np.testing.assert_array_equal(actual_labels, view.target_mask[indices])
