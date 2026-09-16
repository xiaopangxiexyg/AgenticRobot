import numpy as np
import pytest

from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilterConfig,
    RM65CapturedRobotObservationFilterResult,
    captured_robot_observation_mask,
)


def test_capture_robot_mask_uses_bounded_silhouette_and_depth_consistency():
    depth = np.ones((5, 6), dtype=float)
    depth[2, 4] = 0.95
    target = np.zeros(depth.shape, dtype=bool)
    target[2, 1] = True
    rendered = np.full(depth.shape, np.nan, dtype=float)
    rendered[2, 2] = 1.005

    removed = captured_robot_observation_mask(
        depth_m=depth,
        target_mask=target,
        rendered_depth_maps_m=(rendered,),
        silhouette_radius_px=2.0,
        depth_tolerance_m=0.015,
    )

    assert removed[2, 2]
    assert removed[1, 2]
    assert not removed[2, 1]
    assert not removed[2, 4]
    assert not removed[0, 0]


def test_capture_robot_mask_unions_rendered_finger_states_without_removing_target():
    depth = np.ones((3, 5), dtype=float)
    target = np.zeros(depth.shape, dtype=bool)
    target[1, 3] = True
    first = np.full(depth.shape, np.nan, dtype=float)
    second = np.full(depth.shape, np.nan, dtype=float)
    first[1, 1] = 1.0
    second[1, 3] = 1.0

    removed = captured_robot_observation_mask(
        depth_m=depth,
        target_mask=target,
        rendered_depth_maps_m=(first, second),
        silhouette_radius_px=0.0,
        depth_tolerance_m=0.001,
    )

    np.testing.assert_array_equal(
        removed,
        np.asarray(
            [
                [False, False, False, False, False],
                [False, True, False, False, False],
                [False, False, False, False, False],
            ]
        ),
    )


@pytest.mark.parametrize(
    "fields",
    (
        {"silhouette_radius_px": -1},
        {"silhouette_radius_px": 16.1},
        {"depth_tolerance_m": 0},
        {"depth_tolerance_m": 0.0151},
        {"finger_sweep_q_rad": (0.0, 0.5)},
        {"finger_sweep_q_rad": (0.0, 0.864, 0.432)},
    ),
)
def test_capture_robot_filter_config_is_bounded(fields):
    with pytest.raises(ValueError):
        RM65CapturedRobotObservationFilterConfig(**fields)


def test_capture_robot_filter_result_copies_arrays_and_freezes_evidence():
    depth = np.ones((2, 2), dtype=float)
    removed = np.zeros((2, 2), dtype=bool)
    result = RM65CapturedRobotObservationFilterResult(
        depth_m=depth,
        removed_mask=removed,
        evidence={"removed_background_points": 0, "removed_target_points": 0},
    )
    depth[0, 0] = 2
    removed[0, 0] = True

    assert result.depth_m[0, 0] == 1
    assert not result.removed_mask[0, 0]
    with pytest.raises(ValueError):
        result.depth_m[0, 0] = 3
    with pytest.raises(TypeError):
        result.evidence["removed_background_points"] = 1
