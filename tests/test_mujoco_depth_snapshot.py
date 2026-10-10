from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from sim.mujoco.depth_snapshot import MujocoDepthSnapshotSource
from sim.mujoco.rgb_snapshot import MujocoRgbSnapshotSource


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


@pytest.mark.parametrize("kind", ["rgb", "depth", "rgbd"])
def test_snapshot_renderer_lifetime_stays_on_one_thread(kind) -> None:
    scene = _Scene()
    scene_lock = threading.RLock()
    events = []
    fail_render = False

    class AffineRenderer(_Renderer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.owner = threading.current_thread()
            self.check("create")

        def check(self, event):
            assert threading.current_thread() is self.owner
            events.append((event, self.owner))

        def render(self, view_id):
            self.check("rgb")
            if fail_render:
                raise RuntimeError("injected render failure")
            return super().render(view_id)

        def render_all(self, names):
            return {name: self.render(name) for name in names}

        def render_depth(self, view_id):
            self.check("depth")
            if fail_render:
                raise RuntimeError("injected render failure")
            return super().render_depth(view_id)

        def close(self):
            self.check("close")
            super().close()

    def capture_scene(callback):
        # The calling application may already own this RLock. Moving this
        # callback to a worker would deadlock; only rendering may move.
        assert scene_lock._is_owned()
        with scene_lock:
            return callback(scene)

    source_class = MujocoRgbSnapshotSource if kind == "rgb" else MujocoDepthSnapshotSource
    source = source_class(
        capture_scene=capture_scene, clock=lambda: 12.5,
        renderer_factory=AffineRenderer,
    )

    def capture():
        with scene_lock:
            if kind == "rgb":
                return source.capture_jpeg_snapshot(source_ids=("front",), width=2, height=2)
            if kind == "depth":
                return source.capture_npy_frame(view_id="front")
            return source.capture_jpeg_npy_cut(view_id="front")

    try:
        capture()
        with ThreadPoolExecutor(max_workers=1) as caller:
            caller.submit(capture).result(timeout=5)
            fail_render = True
            with pytest.raises(RuntimeError, match="injected render failure"):
                caller.submit(capture).result(timeout=5)
            fail_render = False
            caller.submit(capture).result(timeout=5)
            caller.submit(source.close).result(timeout=5)
        source.close()
        assert [event for event, _ in events].count("create") == 1
        assert [event for event, _ in events].count("close") == 1
        assert len({thread for _, thread in events}) == 1
        assert not events[0][1].is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            capture()
    finally:
        source.close()


def test_snapshot_renderer_factory_and_close_failures_release_worker() -> None:
    scene = _Scene()
    created = []
    closed = []
    fail_create = True

    class FailingRenderer(_Renderer):
        def __init__(self, *args, **kwargs):
            if fail_create:
                raise RuntimeError("injected factory failure")
            super().__init__(*args, **kwargs)
            self.owner = threading.current_thread()
            created.append(self)

        def close(self):
            assert threading.current_thread() is self.owner
            closed.append(self)
            raise RuntimeError("injected close failure")

    source = MujocoDepthSnapshotSource(
        capture_scene=lambda callback: callback(scene), clock=lambda: 1.0,
        renderer_factory=FailingRenderer,
    )
    try:
        with pytest.raises(RuntimeError, match="factory failure"):
            source.capture_npy_frame(view_id="front")
        fail_create = False
        source.capture_npy_frame(view_id="front")
        source.capture_npy_frame(view_id="wrist")
        with pytest.raises(RuntimeError, match="close failure"):
            source.close()
        assert closed == created
        assert len(closed) == 2
        assert not created[0].owner.is_alive()
        source.close()
        with pytest.raises(RuntimeError, match="closed"):
            source.capture_npy_frame(view_id="front")
    finally:
        source.close()


@pytest.mark.optional
@pytest.mark.integration
def test_real_mujoco_rgbd_context_survives_caller_thread_changes() -> None:
    mujoco = pytest.importorskip("mujoco")
    scene = _Scene()
    scene.mujoco = mujoco
    scene.model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><camera name="front" pos="0 0 1" fovy="90"/>'
        '<camera name="wrist" pos="0 0 1" fovy="60"/>'
        '<geom type="plane" size="1 1 .1"/></worldbody></mujoco>'
    )
    scene.data = mujoco.MjData(scene.model)
    mujoco.mj_forward(scene.model, scene.data)
    source = MujocoDepthSnapshotSource(
        capture_scene=lambda callback: callback(scene), clock=lambda: 1.0,
    )
    try:
        source.capture_jpeg_npy_cut(view_id="front")
        # Keep the first caller alive; EGL cannot silently inherit its context.
        with ThreadPoolExecutor(max_workers=1) as caller:
            for view_id in ("front", "wrist", "front"):
                rgb, depth = caller.submit(
                    source.capture_jpeg_npy_cut, view_id=view_id
                ).result(timeout=10)
                assert rgb.data.startswith(b"\xff\xd8")
                values = np.load(io.BytesIO(depth.data), allow_pickle=False)
                assert values.shape == ((2, 2) if view_id == "front" else (1, 3))
                assert np.isfinite(values).all() and (values > 0).all()
            caller.submit(source.close).result(timeout=10)
    finally:
        source.close()
