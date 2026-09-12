from __future__ import annotations

import numpy as np
import pytest

from robot_motion.replay import (
    FIXED_TRAJECTORY_DIRECTIONS,
    available_fixed_trajectory_ids,
    build_fixed_trajectory_targets,
    fixed_trajectory_direction_yaw_rad,
    get_fixed_trajectory,
)

pytestmark = pytest.mark.unit


def _pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def test_digit_catalog_contains_nine_verified_one_stroke_paths() -> None:
    ids = available_fixed_trajectory_ids(family="digit")

    assert ids == tuple(f"digit_{index}_v1" for index in range(1, 10))
    for trajectory_id in ids:
        spec = get_fixed_trajectory(trajectory_id)
        assert spec.stroke_count == 1
        assert len(spec.points) >= 2
        assert spec.bounding_height == pytest.approx(1.0)


def test_greek_catalog_contains_complete_verified_one_stroke_alphabet() -> None:
    names = (
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
        "phi",
        "chi",
        "psi",
        "omega",
    )
    symbols = tuple("ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ")
    ids = available_fixed_trajectory_ids(family="greek")

    assert ids == tuple(f"greek_{name}_v1" for name in names)
    for trajectory_id, name, symbol in zip(ids, names, symbols, strict=True):
        spec = get_fixed_trajectory(trajectory_id)
        assert spec.symbol == symbol
        assert spec.metadata["latin_name"] == name
        assert spec.stroke_count == 1
        assert len(spec.points) >= 2
        assert spec.bounding_height == pytest.approx(1.0)


def test_every_greek_trajectory_reuses_size_direction_and_boundary_contract() -> None:
    boundary = {"right": _pose(0.2, -0.1, 0.4)}

    for trajectory_id in available_fixed_trajectory_ids(family="greek"):
        targets = build_fixed_trajectory_targets(
            boundary,
            trajectory_id,
            scale_m=0.06,
            linear_speed_m_s=0.03,
            direction="right",
        )
        positions = np.stack(
            [
                np.asarray(target.pose_by_side["right"], dtype=float)[:3, 3]
                for target in targets
            ]
        )
        np.testing.assert_allclose(targets[0].pose_by_side["right"], boundary["right"])
        np.testing.assert_allclose(positions[:, 2], 0.4)
        assert targets[0].metadata["family"] == "greek"
        assert targets[0].metadata["size_m"] == pytest.approx(0.06)
        assert targets[0].metadata["direction"] == "right"
        assert targets[-1].timestamp > targets[0].timestamp


def test_fixed_trajectory_anchors_first_point_and_holds_other_side() -> None:
    boundary = {
        "left": _pose(-0.1, 0.2, 0.3),
        "right": _pose(0.4, -0.2, 0.5),
    }
    targets = build_fixed_trajectory_targets(
        boundary,
        "digit_1_v1",
        scale_m=0.08,
        linear_speed_m_s=0.04,
    )

    right = np.stack([np.asarray(target.pose_by_side["right"]) for target in targets])
    left = np.stack([np.asarray(target.pose_by_side["left"]) for target in targets])

    np.testing.assert_allclose(right[0], boundary["right"])
    np.testing.assert_allclose(
        left,
        np.repeat(boundary["left"][None, ...], len(targets), axis=0),
    )
    np.testing.assert_allclose(right[:, 2, 3], 0.5)
    np.testing.assert_allclose(
        right[:, :3, :3],
        np.repeat(boundary["right"][None, :3, :3], len(targets), axis=0),
    )
    assert targets[0].metadata["stroke_count"] == 1
    assert targets[0].metadata["pose_frame"] == "canonical_desk"


def test_fixed_trajectory_size_and_four_directions_are_explicit() -> None:
    boundary = {"right": _pose(0.0, 0.0, 0.4)}
    targets_by_direction = {
        direction: build_fixed_trajectory_targets(
            boundary,
            "digit_1_v1",
            scale_m=0.1,
            linear_speed_m_s=0.05,
            direction=direction,
        )
        for direction in FIXED_TRAJECTORY_DIRECTIONS
    }

    for direction, targets in targets_by_direction.items():
        positions = np.stack(
            [np.asarray(target.pose_by_side["right"], dtype=float)[:3, 3] for target in targets]
        )
        assert targets[0].metadata["direction"] == direction
        assert targets[0].metadata["size_m"] == pytest.approx(0.1)
        assert targets[0].metadata["direction_frame"] == "canonical_desk"
        assert targets[0].metadata["direction_yaw_deg"] == pytest.approx(
            np.rad2deg(fixed_trajectory_direction_yaw_rad(direction))
        )
        assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) > 0.0

    up_delta = np.asarray(
        targets_by_direction["up"][1].pose_by_side["right"][:3, 3]
    ) - np.asarray(targets_by_direction["up"][0].pose_by_side["right"][:3, 3])
    right_delta = np.asarray(
        targets_by_direction["right"][1].pose_by_side["right"][:3, 3]
    ) - np.asarray(targets_by_direction["right"][0].pose_by_side["right"][:3, 3])
    np.testing.assert_allclose(
        right_delta[:2],
        np.asarray((up_delta[1], -up_delta[0])),
    )


def test_fixed_trajectory_speed_is_arc_length_over_duration() -> None:
    targets = build_fixed_trajectory_targets(
        {"right": _pose(0.0, 0.0, 0.4)},
        "digit_8_v1",
        scale_m=0.1,
        linear_speed_m_s=0.05,
        direction="left",
    )
    positions = np.stack(
        [np.asarray(target.pose_by_side["right"], dtype=float)[:3, 3] for target in targets]
    )
    path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    duration = float(targets[-1].timestamp - targets[0].timestamp)

    assert path_length / duration == pytest.approx(0.05)
    assert targets[-1].timestamp == pytest.approx(
        targets[0].metadata["duration_sec"]
    )


def test_fixed_trajectory_rejects_unknown_or_nonpositive_parameters() -> None:
    with pytest.raises(KeyError, match="unknown fixed trajectory"):
        build_fixed_trajectory_targets(
            {"right": _pose(0.0, 0.0, 0.4)},
            "digit_0_v1",
            scale_m=0.1,
            linear_speed_m_s=0.05,
        )
    with pytest.raises(ValueError, match="scale_m"):
        build_fixed_trajectory_targets(
            {"right": _pose(0.0, 0.0, 0.4)},
            "digit_1_v1",
            scale_m=0.0,
            linear_speed_m_s=0.05,
        )
    with pytest.raises(ValueError, match="direction"):
        build_fixed_trajectory_targets(
            {"right": _pose(0.0, 0.0, 0.4)},
            "digit_1_v1",
            scale_m=0.1,
            linear_speed_m_s=0.05,
            direction="diagonal",
        )
