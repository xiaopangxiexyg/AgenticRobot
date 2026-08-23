"""Dependency-neutral application extension seam for the RealTeleop owner loop."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

from .components import RealTeleopComponents


class RealTeleopRuntimeExtension(Protocol):
    """Application-leaf hooks around one existing RealTeleop runtime.

    ``attach_components`` must be side-effect free: it may validate and return
    a replacement immutable aggregate, but it must not start a thread, socket,
    device, or motion.  ``start`` runs only inside the normal runtime cleanup
    boundary.  ``close_before_session`` must establish any extension-owned
    lower HOLD before the Session and hardware resources close.
    """

    def attach_components(
        self,
        components: RealTeleopComponents,
    ) -> RealTeleopComponents: ...

    def start(self, *, components: RealTeleopComponents) -> None: ...

    def bind_runtime(
        self,
        *,
        components: RealTeleopComponents,
        clutch_input: Any,
        camera_sources: Mapping[str, Any],
        dataset_recorder: object | None,
        dataset_recorder_provider: Callable[[], object | None],
        episode_recorder: object,
        control_ui: object | None,
        recording_active: Callable[[], bool],
        pause_recording: Callable[..., bool],
    ) -> None: ...

    def after_control_input_poll(self, *, now: float) -> None: ...

    def after_session_step(self, *, step_result: object, now: float) -> None: ...

    def close_before_session(
        self,
        *,
        components: RealTeleopComponents,
        terminal_error: BaseException | None,
    ) -> None: ...


__all__ = ["RealTeleopRuntimeExtension"]
