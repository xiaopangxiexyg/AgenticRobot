from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from sim.mujoco.depth_snapshot import MujocoDepthSnapshotSource


class _Scene:
    camera_names = ("front", "wrist")

    def camera_metadata(self):
        return {
            "front": {
                "width": 2,
                "height": 2,
                "fovy_deg": 90.0,
                "camera_pose_world_xyz_quat_xyzw": [0, 0, 1, 0, 0, 0, 1],
            },
            "wrist": {
                "width": 3,
                "height": 1,
                "fovy_deg": 60.0,
                "camera_pose_world_xyz_quat_xyzw": [1, 2, 3, 0, 0, 0, 1],
            },
        }


class _Renderer:
    def __init__(self, _scene, *, width, height):
        self.width = width
        self.height = height
        self.closed = False
        self.calls: list[str] = []

    def render_depth(self, view_id):
        self.calls.append(view_id)
        if view_id == "front":
            return np.asarray([[1.0, np.nan], [-2.0, 3.0]], dtype=np.float64)
        return np.asarray([[0.5, 1.5, 2.5]], dtype=np.float32)

    def render(self, view_id):
        self.calls.append(f"rgb:{view_id}")
        return np.full((self.height, self.width, 3), 64, dtype=np.uint8)

    def close(self):
        self.closed = True


def _source(*, clock=lambda: 12.5, max_renderer_dimensions=4):
    scene = _Scene()
    renderers = []

    def factory(*args, **kwargs):
        renderer = _Renderer(*args, **kwargs)
        renderers.append(renderer)
        return renderer

    source = MujocoDepthSnapshotSource(
        capture_scene=lambda capture: capture(scene),
        clock=clock,
        renderer_factory=factory,
        max_renderer_dimensions=max_renderer_dimensions,
    )
    return source, renderers


def test_depth_snapshot_is_native_metric_float32_with_zero_invalid() -> None:
    source, renderers = _source()
    frame = source.capture_npy_frame(view_id="front")

    decoded = np.load(io.BytesIO(frame.data), allow_pickle=False)
    assert decoded.dtype == np.float32
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(
        decoded,
        np.asarray([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    )
    assert frame.captured_at == 12.5
    assert (frame.width, frame.height) == (2, 2)
    assert frame.intrinsics == {
        "width": 2,
        "height": 2,
        "fx": pytest.approx(1.0),
        "fy": pytest.approx(1.0),
        "cx": 1.0,
        "cy": 1.0,
        "fovy_deg": 90.0,
    }
    assert frame.camera_pose_world_xyz_quat_xyzw == (0, 0, 1, 0, 0, 0, 1)

    again = source.capture_npy_frame(view_id="front")
    assert again.data == frame.data
    assert len(renderers) == 1
    assert renderers[0].calls == ["front", "front"]


def test_rgbd_cut_uses_one_scene_capture_when_scene_changes_between_captures() -> None:
    scene = _Scene()
    scene.generation = 0
    capture_calls = 0

    class _ChangingRenderer(_Renderer):
        def render(self, view_id):
            return np.full(
                (self.height, self.width, 3),
                scene.generation,
                dtype=np.uint8,
            )

        def render_depth(self, view_id):
            return np.full(
                (self.height, self.width),
                scene.generation,
                dtype=np.float32,
            )

    def capture_scene(capture):
        nonlocal capture_calls
        capture_calls += 1
        scene.generation = 10 * capture_calls
        return capture(scene)

    source = MujocoDepthSnapshotSource(
        capture_scene=capture_scene,
        clock=lambda: 12.5,
        renderer_factory=_ChangingRenderer,
    )
    rgb, depth = source.capture_jpeg_npy_cut(view_id="front")

    assert capture_calls == 1
    with Image.open(io.BytesIO(rgb.data)) as image:
        assert np.asarray(image)[0, 0, 0] == 10
    assert np.load(io.BytesIO(depth.data), allow_pickle=False)[0, 0] == 10.0


def test_depth_snapshot_catalog_bounds_cache_and_close() -> None:
    source, renderers = _source(max_renderer_dimensions=1)
    assert source.source_catalog() == (
        {"view_id": "front", "native_width": 2, "native_height": 2},
        {"view_id": "wrist", "native_width": 3, "native_height": 1},
    )
    source.capture_npy_frame(view_id="front")
    with pytest.raises(RuntimeError, match="cache is full"):
        source.capture_npy_frame(view_id="wrist")
    with pytest.raises(ValueError, match="unknown depth view"):
        source.capture_npy_frame(view_id="missing")

    source.close()
    source.close()
    assert renderers[0].closed
    with pytest.raises(RuntimeError, match="closed"):
        source.source_catalog()


@pytest.mark.parametrize("clock", [lambda: float("nan"), lambda: float("inf")])
def test_depth_snapshot_rejects_nonfinite_capture_time(clock) -> None:
    source, _ = _source(clock=clock)
    with pytest.raises(RuntimeError, match="clock is non-finite"):
        source.capture_npy_frame(view_id="front")


@pytest.mark.parametrize("view_id", ["", " front", "front "])
def test_depth_snapshot_rejects_ambiguous_view_ids(view_id) -> None:
    source, _ = _source()
    with pytest.raises(ValueError, match="view_id"):
        source.capture_npy_frame(view_id=view_id)
