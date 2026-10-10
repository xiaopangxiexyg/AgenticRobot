"""Thread-affine renderer cache used by passive RGB and depth snapshots."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class SnapshotRendererCache:
    """Caller serializes access and holds the scene cut until render returns.

    Only renderer creation, drawing and destruction run on the worker. Callbacks
    must not acquire the caller's scene/application locks or call source APIs.
    Physics, scene capture and snapshot publication remain caller-owned.
    """

    def __init__(self, factory: Callable[..., Any], max_dimensions: int) -> None:
        self._factory = factory
        self._max_dimensions = max_dimensions
        self._renderers: dict[tuple[int, int], Any] = {}
        self._worker: ThreadPoolExecutor | None = None

    def render(
        self, scene: Any, *, width: int, height: int, draw: Callable[[Any], Any]
    ) -> Any:
        if self._worker is None:
            self._worker = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mujoco-snapshot-render"
            )

        def run() -> Any:
            key = (width, height)
            renderer = self._renderers.get(key)
            if renderer is None:
                if len(self._renderers) >= self._max_dimensions:
                    raise RuntimeError("snapshot renderer dimension cache is full.")
                renderer = self._factory(scene, width=width, height=height)
                self._renderers[key] = renderer
            return draw(renderer)

        return self._worker.submit(run).result()

    def close(self) -> None:
        worker = self._worker
        if worker is None:
            return

        def dispose() -> None:
            renderers = tuple(self._renderers.values())
            self._renderers.clear()
            first_error: Exception | None = None
            for renderer in renderers:
                try:
                    renderer.close()
                except Exception as exc:  # noqa: BLE001 - close every renderer, then raise.
                    if first_error is None:
                        first_error = exc
            if first_error is not None:
                raise first_error

        try:
            worker.submit(dispose).result()
        finally:
            worker.shutdown(wait=True)
            self._worker = None
