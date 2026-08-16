from __future__ import annotations

from contextlib import nullcontext
import math

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.mujoco.replay.tcp_pose_envelope_visualization import (  # noqa: E402
    TcpPoseEnvelopeVisualizer,
)
from sim.sim_real_tests.mujoco.view_tcp_pose_envelope import (  # noqa: E402
    build_parser,
    load_relative_preview_profiles,
)


pytestmark = [pytest.mark.integration, pytest.mark.optional]


class _FakeViewerHandle:
    def __init__(self, user_scene) -> None:
        self.user_scn = user_scene

    def lock(self):
        return nullcontext()


def _poses(offset: float) -> tuple[np.ndarray, ...]:
    values = []
    for index in range(3):
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [offset + index * 0.01, 0.0, 0.2]
        values.append(pose)
    return tuple(values)


def test_visualizer_draws_cones_trails_and_common_origin_pose_directions() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco model='envelope_viz'/>")
    user_scene = mujoco.MjvScene(model, 512)
    viewer = _FakeViewerHandle(user_scene)
    visualizer = TcpPoseEnvelopeVisualizer(
        anchor_pose_world=np.eye(4, dtype=float),
        max_direction_angle_rad=math.radians(8.0),
        trajectory_pose_world_by_kind={
            "raw": _poses(0.0),
            "effective": _poses(0.001),
            "actual": _poses(0.002),
        },
    )

    assert visualizer.update(viewer, frame_index=1) is True
    assert user_scene.ngeom == visualizer.geom_count
    labels = {str(user_scene.geoms[index].label) for index in range(user_scene.ngeom)}
    assert "tcp_translation_envelope" not in labels
    assert "tcp_envelope_edge" not in labels
    assert all(
        int(user_scene.geoms[index].type) != int(mujoco.mjtGeom.mjGEOM_BOX)
        for index in range(user_scene.ngeom)
    )
    for axis_name in "xyz":
        label = f"tcp_orientation_{axis_name}_cone"
        assert label in labels
        cone_segments = [
            user_scene.geoms[index]
            for index in range(user_scene.ngeom)
            if str(user_scene.geoms[index].label) == label
        ]
        assert len(cone_segments) == 1
        assert all(
            int(segment.type) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
            for segment in cone_segments
        )
        assert all(0.0 < float(segment.rgba[3]) < 1.0 for segment in cone_segments)
    assert "raw_tcp_trail" not in labels
    assert "effective_tcp_trail" not in labels
    assert "actual_tcp_trail" not in labels
    for kind in ("raw", "effective", "actual"):
        assert f"{kind}_tcp_position" in labels
        for axis_name in "xyz":
            assert f"{kind}_tcp_{axis_name}_direction" not in labels


@pytest.mark.parametrize("angle_deg", [45.0, 90.0, 120.0])
def test_spherical_direction_boundary_supports_wide_angles(
    angle_deg: float,
) -> None:
    origin = np.array([0.2, -0.1, 0.3], dtype=float)
    axis_frame = np.eye(3, dtype=float)
    radius = 0.105
    points = TcpPoseEnvelopeVisualizer._spherical_direction_boundary(
        origin=origin,
        axis_frame=axis_frame,
        angle_rad=math.radians(angle_deg),
        radius=radius,
        segment_count=48,
    )

    assert len(points) == 48
    for point in points:
        direction = (point - origin) / radius
        assert np.linalg.norm(point - origin) == pytest.approx(radius)
        assert direction[2] == pytest.approx(math.cos(math.radians(angle_deg)))


def test_visualizer_optionally_draws_relative_translation_box() -> None:
    model = mujoco.MjModel.from_xml_string("<mujoco model='relative_box'/>")
    user_scene = mujoco.MjvScene(model, 512)
    viewer = _FakeViewerHandle(user_scene)
    frame = np.eye(4, dtype=float)
    frame[:3, 3] = [0.2, -0.1, 0.3]
    visualizer = TcpPoseEnvelopeVisualizer(
        anchor_pose_world=frame,
        max_direction_angle_rad=math.radians(15.0),
        trajectory_pose_world_by_kind={
            "raw": _poses(0.0),
            "effective": _poses(0.001),
            "actual": _poses(0.002),
        },
        translation_frame_world=frame,
        translation_min_xyz_m=(-0.04, -0.04, -0.04),
        translation_max_xyz_m=(0.04, 0.04, 0.04),
    )

    assert visualizer.update(viewer, frame_index=1) is True
    labels = {str(user_scene.geoms[index].label) for index in range(user_scene.ngeom)}
    assert "tcp_relative_translation_envelope" in labels
    assert "tcp_relative_translation_edge" not in labels


def test_relative_preview_profiles_are_ordered_and_have_expected_values() -> None:
    profiles = load_relative_preview_profiles()

    assert profiles["relative_tight"] == {
        "translation_half_extent_m": pytest.approx(0.1),
        "max_rotation_deg": pytest.approx(45.0),
    }
    assert profiles["relative_standard"] == {
        "translation_half_extent_m": pytest.approx(0.2),
        "max_rotation_deg": pytest.approx(90.0),
    }
    assert profiles["relative_loose"] == {
        "translation_half_extent_m": pytest.approx(0.3),
        "max_rotation_deg": pytest.approx(120.0),
    }


def test_viewer_cli_defaults_to_visible_slow_infinite_replay() -> None:
    args = build_parser().parse_args([])

    assert args.speed == pytest.approx(0.35)
    assert args.repeat == 0
    assert args.hide_left_ui is True
    assert args.hide_right_ui is True
