from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator

from .scene import MujocoScene


def _passive_viewer_sim_alive(viewer_handle: object) -> bool:
    sim_ref = getattr(viewer_handle, "_sim", None)
    if callable(sim_ref):
        try:
            return sim_ref() is not None
        except Exception:
            return False
    is_running = getattr(viewer_handle, "is_running", None)
    if not callable(is_running):
        return False
    try:
        return bool(is_running())
    except Exception:
        return False


def close_passive_viewer(
    viewer_handle: object,
    *,
    timeout_sec: float = 3.0,
    teardown_grace_sec: float = 0.05,
) -> None:
    """Close and drain MuJoCo's Linux GUI thread before scene destruction."""

    close = getattr(viewer_handle, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        return
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        if not _passive_viewer_sim_alive(viewer_handle):
            break
        time.sleep(0.02)
    # launch_passive owns a daemon GUI thread on Linux.  Let its C++/GLFW/GLX
    # teardown finish before MjModel/MjData or the Python interpreter go away.
    time.sleep(max(float(teardown_grace_sec), 0.0))


@contextmanager
def passive_viewer_handle(
    scene: MujocoScene,
    *,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> Iterator[object]:
    """Yield a passive handle without advancing or forwarding the scene.

    Runtime rigs that own ``MjData`` can use this helper and synchronize the
    handle under their own lock.  It intentionally performs no physics work.
    """

    import mujoco.viewer

    handle = mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
        show_left_ui=bool(show_left_ui),
        show_right_ui=bool(show_right_ui),
    )
    try:
        yield handle
    finally:
        close_passive_viewer(handle)


def launch_passive_viewer(
    scene: MujocoScene,
    *,
    duration_sec: float | None = None,
    realtime_hz: float = 30.0,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
    on_step: Callable[[float], None] | None = None,
) -> None:
    """Open a MuJoCo GUI viewer for the current scene state.

    The viewer runs until the user closes the window. If ``duration_sec`` is
    positive, it closes automatically after that many seconds.
    """

    period = 1.0 / max(float(realtime_hz), 1e-6)
    started_at = time.time()
    with passive_viewer_handle(
        scene,
        show_left_ui=bool(show_left_ui),
        show_right_ui=bool(show_right_ui),
    ) as handle:
        while handle.is_running():
            elapsed = time.time() - started_at
            if on_step is not None:
                on_step(elapsed)
            scene.forward()
            handle.sync()
            if duration_sec is not None and duration_sec > 0.0:
                if elapsed >= float(duration_sec):
                    break
            time.sleep(period)
