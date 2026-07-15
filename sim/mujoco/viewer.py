from __future__ import annotations

import time
from collections.abc import Callable

from .scene import MujocoScene


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

    import mujoco.viewer

    period = 1.0 / max(float(realtime_hz), 1e-6)
    started_at = time.time()
    with mujoco.viewer.launch_passive(
        scene.model,
        scene.data,
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
