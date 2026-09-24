from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_motion.replay import (
    FIXED_TRAJECTORY_DIRECTIONS,
    available_fixed_trajectory_ids,
    build_fixed_trajectory_targets,
    fixed_trajectory_catalog_hash,
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


def test_reconstructed_digits_keep_clear_single_stroke_geometry() -> None:
    paths = {
        digit: np.asarray(get_fixed_trajectory(f"digit_{digit}_v1").points)
        for digit in (2, 3, 4, 9)
    }
    for digit, points in paths.items():
        steps = np.diff(points, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        turns = np.rad2deg(np.arccos(np.clip(
            np.sum(steps[:-1] * steps[1:], axis=1)
            / (lengths[:-1] * lengths[1:]),
            -1.0,
            1.0,
        )))
        assert np.all(lengths > 0.0), digit
        assert float(lengths.sum()) < 3.0, digit
        assert float(np.max(turns)) < (110.0 if digit == 4 else 95.0), digit

    two = paths[2]
    assert two[np.argmax(two[:, 1]), 0] > two[0, 0]
    np.testing.assert_allclose(two[-2:, 1], 0.0)
    assert two[-1, 0] > two[-2, 0]

    three = paths[3]
    assert np.max(three[three[:, 1] > 0.6, 0]) > 0.3
    assert np.max(three[three[:, 1] < 0.4, 0]) > 0.3
    assert np.min(three[(three[:, 1] > 0.45) & (three[:, 1] < 0.55), 0]) < 0.0

    four = paths[4]
    assert four[0, 1] == pytest.approx(1.0)
    assert four[1, 0] < 0.0
    assert four[-1, 1] == pytest.approx(0.0)
    assert four[-1, 0] == pytest.approx(four[-2, 0])

    nine = paths[9]
    np.testing.assert_allclose(nine[0], nine[-5])  # loop joins its descender
    assert nine[-1, 1] == pytest.approx(0.0)
    assert nine[-1, 0] < nine[-5, 0]


@pytest.mark.parametrize("digit", (2, 3, 4, 9))
def test_reconstructed_digit_targets_preserve_canonical_motion_contract(digit: int) -> None:
    boundary = {"left": _pose(-0.2, 0.1, 0.4), "right": _pose(0.2, -0.1, 0.4)}
    targets = build_fixed_trajectory_targets(
        boundary,
        f"digit_{digit}_v1",
        scale_m=0.08,
        linear_speed_m_s=0.09,
    )
    positions = np.stack([target.pose_by_side["right"][:3, 3] for target in targets])
    np.testing.assert_allclose(positions[0], boundary["right"][:3, 3])
    np.testing.assert_allclose(positions[:, 2], 0.4)
    assert all(
        np.array_equal(target.pose_by_side["left"], boundary["left"])
        for target in targets
    )
    assert np.all(np.diff([target.timestamp for target in targets]) > 0.0)
    assert targets[0].metadata["catalog_hash"] == fixed_trajectory_catalog_hash()
    assert targets[0].metadata["symbol"] == str(digit)


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


def test_latin_catalog_contains_complete_verified_one_stroke_alphabet() -> None:
    letters = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ids = available_fixed_trajectory_ids(family="latin")

    assert ids == tuple(f"latin_{letter.lower()}_v1" for letter in letters)
    for trajectory_id, letter in zip(ids, letters, strict=True):
        spec = get_fixed_trajectory(trajectory_id)
        assert spec.symbol == letter
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


def test_every_latin_trajectory_reuses_size_direction_and_boundary_contract() -> None:
    boundary = {"right": _pose(0.2, -0.1, 0.4)}

    for trajectory_id in available_fixed_trajectory_ids(family="latin"):
        targets = build_fixed_trajectory_targets(
            boundary,
            trajectory_id,
            scale_m=0.06,
            linear_speed_m_s=0.03,
            direction="left",
        )
        positions = np.stack(
            [
                np.asarray(target.pose_by_side["right"], dtype=float)[:3, 3]
                for target in targets
            ]
        )
        np.testing.assert_allclose(targets[0].pose_by_side["right"], boundary["right"])
        np.testing.assert_allclose(positions[:, 2], 0.4)
        assert targets[0].metadata["family"] == "latin"
        assert targets[0].metadata["size_m"] == pytest.approx(0.06)
        assert targets[0].metadata["direction"] == "left"
        assert targets[-1].timestamp > targets[0].timestamp


def _write_additional_catalog(
    path: Path,
    *,
    trajectory_id: str = "custom_triangle_v1",
    stroke_count: int = 1,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "agentic_robot.rule_based_fixed_trajectories.v1",
                "path_frame": "glyph_local_xy",
                "normalization": "bounding_height_is_one",
                "trajectories": [
                    {
                        "trajectory_id": trajectory_id,
                        "family": "custom",
                        "symbol": "triangle",
                        "stroke_count": stroke_count,
                        "points": [[0.0, 0.0], [0.5, 1.0], [1.0, 0.0], [0.0, 0.0]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_additional_catalog_appends_without_replacing_builtins(tmp_path: Path) -> None:
    catalog = tmp_path / "custom-trajectories.json"
    _write_additional_catalog(catalog)

    assert available_fixed_trajectory_ids(catalog_path=catalog) == (
        "custom_triangle_v1",
    )
    ids = available_fixed_trajectory_ids(additional_catalog_path=catalog)
    assert ids[:9] == tuple(f"digit_{index}_v1" for index in range(1, 10))
    assert ids[-1] == "custom_triangle_v1"
    assert get_fixed_trajectory(
        "custom_triangle_v1",
        additional_catalog_path=catalog,
    ).family == "custom"

    targets = build_fixed_trajectory_targets(
        {"right": _pose(0.0, 0.0, 0.4)},
        "custom_triangle_v1",
        scale_m=0.08,
        linear_speed_m_s=0.04,
        additional_catalog_path=catalog,
    )
    assert targets[0].metadata["catalog_hash"] == fixed_trajectory_catalog_hash(
        additional_catalog_path=catalog
    )
    assert targets[0].metadata["catalog_hash"] != fixed_trajectory_catalog_hash()


def test_additional_catalog_cannot_shadow_builtin_id(tmp_path: Path) -> None:
    catalog = tmp_path / "duplicate.json"
    _write_additional_catalog(catalog, trajectory_id="digit_1_v1")

    with pytest.raises(ValueError, match="duplicates existing ids: digit_1_v1"):
        available_fixed_trajectory_ids(additional_catalog_path=catalog)


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
