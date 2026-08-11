from __future__ import annotations

import pytest

import sim.mujoco.diagnostics.cube_drop as mujoco_diagnostics

mujoco_viewer = pytest.importorskip("mujoco.viewer")

from sim.mujoco.scene import MujocoScene
from sim.mujoco.viewer import MujocoPassiveViewerSession


class _FakeViewerHandle:
    def __init__(self) -> None:
        self.running = True
        self.sync_count = 0
        self.close_count = 0

    def is_running(self) -> bool:
        return self.running

    def sync(self) -> None:
        self.sync_count += 1

    def close(self) -> None:
        self.close_count += 1
        self.running = False


@pytest.mark.optional
def test_passive_viewer_session_owns_handle_and_uses_lock_aware_sync() -> None:
    scene = object.__new__(MujocoScene)
    scene.model = object()
    scene.data = object()
    handle = _FakeViewerHandle()
    launches: list[tuple[object, object, bool, bool]] = []

    def launch(model, data, *, show_left_ui, show_right_ui):
        launches.append((model, data, show_left_ui, show_right_ui))
        return handle

    sync_calls: list[object] = []

    def sync(viewer_handle: object) -> None:
        sync_calls.append(viewer_handle)
        viewer_handle.sync()

    viewer = MujocoPassiveViewerSession(
        scene,
        sync_callback=sync,
        show_left_ui=False,
        show_right_ui=True,
        launch_callback=launch,
    )

    assert viewer.status()["state"] == "not_started"
    assert viewer.start()["running"] is True
    assert launches == [(scene.model, scene.data, False, True)]
    assert viewer.sync() is True
    assert sync_calls == [handle]
    assert handle.sync_count == 1
    assert viewer.status()["sync_count"] == 1

    viewer.close()
    assert handle.close_count == 1
    assert viewer.status()["state"] == "closed"
    assert viewer.status()["teardown_confirmed"] is True


@pytest.mark.optional
def test_passive_viewer_window_close_is_observational_not_control_fault() -> None:
    scene = object.__new__(MujocoScene)
    scene.model = object()
    scene.data = object()
    handle = _FakeViewerHandle()
    viewer = MujocoPassiveViewerSession(
        scene,
        launch_callback=lambda *_args, **_kwargs: handle,
    )
    viewer.start()
    handle.running = False

    assert viewer.sync() is False
    assert viewer.status()["state"] == "closed_by_user"
    assert viewer.status()["last_error"] is None


@pytest.mark.optional
def test_passive_viewer_close_failure_is_not_reported_as_drained() -> None:
    scene = object.__new__(MujocoScene)
    scene.model = object()
    scene.data = object()

    class FailingHandle(_FakeViewerHandle):
        def close(self) -> None:
            raise RuntimeError("native close failed")

    viewer = MujocoPassiveViewerSession(
        scene,
        launch_callback=lambda *_args, **_kwargs: FailingHandle(),
    )
    viewer.start()

    with pytest.raises(RuntimeError, match="close\\(\\) failed"):
        viewer.close()

    status = viewer.status()
    assert status["state"] == "close_faulted"
    assert status["teardown_confirmed"] is False


@pytest.mark.optional
def test_default_viewer_launch_does_not_use_terminating_glfw_probe(
    monkeypatch,
) -> None:
    scene = object.__new__(MujocoScene)
    scene.model = object()
    scene.data = object()
    handle = _FakeViewerHandle()
    calls: list[str] = []

    monkeypatch.setattr(
        mujoco_diagnostics,
        "_require_interactive_display",
        lambda: calls.append("display.check"),
    )

    def forbidden_loader():
        raise AssertionError("diagnostic GLFW lifecycle leaked into viewer startup")

    monkeypatch.setattr(
        mujoco_diagnostics,
        "_load_mujoco_gui",
        forbidden_loader,
    )
    monkeypatch.setattr(
        mujoco_viewer,
        "launch_passive",
        lambda *_args, **_kwargs: handle,
    )

    viewer = MujocoPassiveViewerSession(scene)

    assert viewer.start()["running"] is True
    assert calls == ["display.check"]
    viewer.close()
