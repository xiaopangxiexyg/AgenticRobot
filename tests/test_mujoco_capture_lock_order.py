from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.motion_application import MujocoAgentSkillMotionApplication
from sim.mujoco.depth_snapshot import MujocoDepthSnapshotSource


def _application():
    # Real application capture methods and source locks, with no runtime startup.
    app = object.__new__(MujocoAgentSkillMotionApplication)
    app._lock = threading.RLock()
    app._closed = False
    scene = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(12), qvel=np.zeros(12), time=1.0),
        arm_qpos_by_side=lambda: {"left": np.zeros(6), "right": np.zeros(6)},
        camera_names=("front",),
        camera_metadata=lambda: {"front": {
            "width": 2, "height": 2, "fovy_deg": 60.0,
            "camera_pose_world_xyz_quat_xyzw": [0, 0, 1, 0, 0, 0, 1],
            "optical_camera_pose_world_xyz_quat_xyzw": [0, 0, 1, 1, 0, 0, 0],
        }},
    )
    rig_calls = []

    def capture_scene(callback):
        assert app._lock._is_owned()
        rig_calls.append(True)
        return callback(scene)

    app.runtime = SimpleNamespace(rig=SimpleNamespace(capture_scene=capture_scene))
    source = MujocoDepthSnapshotSource(capture_scene=app._capture_visual_scene, clock=lambda: 1.0)

    class Renderer:
        def render(self, view_id):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def render_depth(self, view_id):
            return np.ones((2, 2), dtype=np.float32)

    # Exercise the real scene/encoding path without an OpenGL worker or device.
    class Cache:
        def render(self, scene, *, width, height, draw):
            return draw(Renderer())

        def close(self):
            pass

    source._renderer_cache = Cache()
    app._depth_snapshot_source = source
    return app, source, rig_calls


def test_depth_and_visual_cut_share_source_then_application_lock_order():
    app, source, rig_calls = _application()
    depth_owns_source = threading.Event()
    visual_enters_source = threading.Event()
    errors, results = [], {}
    original_capture = source._capture_scene
    original_lock = source._lock

    class CheckedLock:
        def __enter__(self):
            if threading.current_thread().name == "visual-cut":
                visual_enters_source.set()
            # Fail before waiting, so a regression cannot leave deadlocked threads.
            assert not app._lock._is_owned(), "application -> source lock inversion"
            original_lock.acquire()
            return self

        def __exit__(self, *args):
            original_lock.release()

    source._lock = CheckedLock()

    def capture_scene(callback):
        if threading.current_thread().name == "depth-cut":
            depth_owns_source.set()
            assert visual_enters_source.wait(2)
        return original_capture(callback)

    source._capture_scene = capture_scene

    def run(capture):
        try:
            results[threading.current_thread().name] = capture(view_id="front")
        except BaseException as error:
            errors.append(error)

    depth = threading.Thread(target=run, args=(app.capture_depth_npy_frame,), name="depth-cut", daemon=True)
    visual = threading.Thread(target=run, args=(app.capture_visual_depth_cut,), name="visual-cut", daemon=True)
    try:
        depth.start()
        assert depth_owns_source.wait(2)
        visual.start()
        depth.join(3)
        visual.join(3)
        assert not depth.is_alive() and not visual.is_alive()
        assert not errors
        assert len(results) == len(rig_calls) == 2
        rgb, depth_frame = results["visual-cut"]
        assert (rgb.width, rgb.height) == (depth_frame.width, depth_frame.height) == (2, 2)
        assert depth_frame.captured_at == 1.0
    finally:
        visual_enters_source.set()
        depth.join(3)
        if visual.ident is not None:
            visual.join(3)
        source.close()


def test_visual_cut_close_race_rejects_under_application_lock_before_rig():
    app, source, rig_calls = _application()
    original_capture = source._capture_scene

    def close_before_application_lock(callback):
        app._closed = True
        return original_capture(callback)

    source._capture_scene = close_before_application_lock
    try:
        with pytest.raises(RuntimeError, match="application is closed"):
            app.capture_visual_depth_cut(view_id="front")
        assert not rig_calls
    finally:
        source.close()


@pytest.mark.parametrize("invalid", [None, "not_started", "not_held", "active", "no_planner"])
def test_frozen_capture_requires_started_held_inactive_planner_before_render(invalid):
    app, source, rig_calls = _application()
    app._started = invalid != "not_started"
    app._held = invalid != "not_held"
    app._active_operation_id = "active:other" if invalid == "active" else None
    app.target = SimpleNamespace(planned_tcp_motion=None if invalid == "no_planner" else object())
    renders = []
    render = source._renderer_cache.render

    def record_render(*args, **kwargs):
        renders.append(True)
        return render(*args, **kwargs)

    source._renderer_cache.render = record_render
    try:
        if invalid is not None:
            with pytest.raises(RuntimeError, match="stationary held"):
                app.capture_frozen_rgbd_cut(source_ids=("front",))
            assert not renders
        else:
            cut = app.capture_frozen_rgbd_cut(source_ids=("front",))
            assert cut.simulation_time_sec == 1.0 and len(cut.views) == 1
            assert len(renders) == 1
        assert len(rig_calls) == 1
    finally:
        source.close()
