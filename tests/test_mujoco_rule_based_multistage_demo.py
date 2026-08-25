from __future__ import annotations

import numpy as np
import pytest

from sim.mujoco.replay.rule_based_multistage_demo import (
    _circle_targets,
    parse_stage_span,
)


def _pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    return pose


def test_parse_stage_span_is_half_open_and_bounded() -> None:
    span = parse_stage_span("3:8", total_frames=10)

    assert span.start == 3
    assert span.end == 8
    assert span.frame_count == 5

    with pytest.raises(ValueError, match="exceeds source frame count"):
        parse_stage_span("3:11", total_frames=10)


def test_circle_is_continuous_and_parallel_to_desk_xy() -> None:
    boundary = {"left": _pose(-0.1, 0.2, 0.3), "right": _pose(0.1, -0.2, 0.4)}
    targets = _circle_targets(boundary, radius_m=0.02, steps=9, hz=30.0)

    right = np.stack([np.asarray(target.pose_by_side["right"]) for target in targets])
    left = np.stack([np.asarray(target.pose_by_side["left"]) for target in targets])
    center = np.asarray([0.08, -0.2, 0.4])
    radius = np.linalg.norm(right[:, :3, 3] - center, axis=1)

    np.testing.assert_allclose(right[0], boundary["right"])
    np.testing.assert_allclose(right[-1], boundary["right"])
    np.testing.assert_allclose(left, np.repeat(boundary["left"][None, ...], len(targets), axis=0))
    np.testing.assert_allclose(radius, 0.02, atol=1e-9)
    np.testing.assert_allclose(right[:, 2, 3], 0.4, atol=1e-9)
    np.testing.assert_allclose(
        right[:, :3, :3],
        np.repeat(boundary["right"][None, :3, :3], len(targets), axis=0),
    )
