from __future__ import annotations

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator

from .scene import MujocoScene


class MujocoPassiveViewerSession:
    """Own one passive MuJoCo viewer without owning simulation advancement.

    The simulator package owns the native viewer handle and its teardown.  A
    caller may supply the execution rig's lock-aware ``sync_viewer`` callback;
    this keeps GUI synchronization from racing the lower physics owner while
    leaving application/profile code responsible only for lifecycle
    coordination.
    """

    def __init__(
        self,
        scene: MujocoScene,
        *,
        sync_callback: Callable[[object], None] | None = None,
        key_callback: Callable[[int], None] | None = None,
        show_left_ui: bool = True,
        show_right_ui: bool = True,
        launch_callback: Callable[..., object] | None = None,
    ) -> None:
        if not isinstance(scene, MujocoScene):
            raise TypeError("scene must be a MujocoScene.")
        if sync_callback is not None and not callable(sync_callback):
            raise TypeError("sync_callback must be callable or None.")
        if key_callback is not None and not callable(key_callback):
            raise TypeError("key_callback must be callable or None.")
        if launch_callback is not None and not callable(launch_callback):
            raise TypeError("launch_callback must be callable or None.")
        self.scene = scene
        self.sync_callback = sync_callback
        self.key_callback = key_callback
        self.show_left_ui = bool(show_left_ui)
        self.show_right_ui = bool(show_right_ui)
        self._launch_callback = launch_callback
        self._lock = threading.RLock()
        self._handle: object | None = None
        self._state = "not_started"
        self._last_error: str | None = None
        self._sync_count = 0
        self._closed = False
        self._teardown_confirmed = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._handle is not None and self._state == "running"

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("MuJoCo passive viewer session is closed.")
            if self._handle is not None and self._state == "running":
                return self.status()
            launch = self._launch_callback
            if launch is None:
                # Fail cleanly before entering MuJoCo's native viewer when
                # Linux has no usable interactive display.  Do not use the
                # diagnostics loader here: its probe owns an isolated GLFW
                # init/terminate cycle, while this lifecycle is immediately
                # followed by MuJoCo's long-lived passive viewer.
                from sim.mujoco.diagnostics.cube_drop import (
                    _require_interactive_display,
                )

                _require_interactive_display()
                import mujoco.viewer

                launch = mujoco.viewer.launch_passive
            try:
                handle = launch(
                    self.scene.model,
                    self.scene.data,
                    key_callback=self.key_callback,
                    show_left_ui=self.show_left_ui,
                    show_right_ui=self.show_right_ui,
                )
            except Exception as error:
                self._state = "faulted"
                self._last_error = f"{type(error).__name__}:{error}"
                raise RuntimeError(
                    "MuJoCo passive viewer failed to start: "
                    f"{self._last_error}"
                ) from error
            self._handle = handle
            self._state = "running"
            self._last_error = None
            return self.status()

    def sync(self) -> bool:
        """Synchronize one rendered frame; viewer faults remain observational."""

        with self._lock:
            handle = self._handle
            if handle is None or self._state != "running":
                return False
            is_running = getattr(handle, "is_running", None)
            if not callable(is_running) or not bool(is_running()):
                self._state = "closed_by_user"
                return False
            try:
                if self.sync_callback is None:
                    sync = getattr(handle, "sync", None)
                    if not callable(sync):
                        raise TypeError("MuJoCo viewer handle must expose sync().")
                    sync()
                else:
                    self.sync_callback(handle)
            except Exception as error:
                self._state = "faulted"
                self._last_error = f"{type(error).__name__}:{error}"
                return False
            self._sync_count += 1
            return True

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "state": self._state,
                "running": bool(
                    self._handle is not None and self._state == "running"
                ),
                "sync_count": self._sync_count,
                "last_error": self._last_error,
                "teardown_confirmed": self._teardown_confirmed,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            handle = self._handle
        if handle is not None:
            try:
                close_passive_viewer(handle)
            except Exception as error:
                with self._lock:
                    self._state = "close_faulted"
                    self._last_error = f"{type(error).__name__}:{error}"
                raise
        with self._lock:
            self._closed = True
            self._handle = None
            self._state = "closed"
            self._teardown_confirmed = True


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
) -> bool:
    """Close and drain MuJoCo's Linux GUI thread before scene destruction."""

    close = getattr(viewer_handle, "close", None)
    if not callable(close):
        return True
    try:
        close()
    except Exception as error:
        raise RuntimeError(
            "MuJoCo passive viewer close() failed."
        ) from error
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        if not _passive_viewer_sim_alive(viewer_handle):
            break
        time.sleep(0.02)
    if _passive_viewer_sim_alive(viewer_handle):
        raise TimeoutError(
            "MuJoCo passive viewer did not confirm native teardown before "
            f"{max(float(timeout_sec), 0.0):.3f}s."
        )
    # launch_passive owns a daemon GUI thread on Linux.  Let its C++/GLFW/GLX
    # teardown finish before MjModel/MjData or the Python interpreter go away.
    time.sleep(max(float(teardown_grace_sec), 0.0))
    return True


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
