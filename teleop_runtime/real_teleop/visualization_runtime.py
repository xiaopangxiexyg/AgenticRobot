"""Native camera/tactile preview lifecycle used before and during teleop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VisualizationRuntime:
    close_camera_preview: Callable[..., None]
    open_camera_preview: Callable[[float], None]
    close_tactile_preview: Callable[..., None]
    open_tactile_preview: Callable[[float], None]
    handle_startup_requests: Callable[[float], None]
    update_startup_visualizations: Callable[[float], None]
    current_state: Callable[[], tuple[Any, bool, Any]]
    sync_state: Callable[[Any, bool, Any], None]


def build_visualization_runtime(
    *,
    args: Any,
    components: Any,
    clutch_input: Any,
    control_ui: Any,
    preview_camera_sources: Mapping[str, Any],
    camera_preview_rate: Any,
    tactile_preview_rate: Any,
    initial_camera_previewer: Any,
    initial_camera_preview_active: bool,
    initial_tactile_visualizer: Any,
    set_outer_state: Callable[[Any, bool, Any], None],
    retry_failed_devices: Callable[[], None],
    start_camera_sources: Callable[[Mapping[str, Any]], None],
    camera_previewer_type: type,
    camera_preview_config_type: type,
    build_tactile_visualizer: Callable[..., Any],
    enable_gui_tactile_solving: Callable[..., int],
    tactile_solving_active: Callable[..., bool],
    tactile_visualization_available: Callable[..., bool],
    dmtac_visualizer_types: tuple[type, ...],
    dmtac_visualization_snapshot: Callable[..., tuple[Any, Any]],
    tactile_snapshot: Callable[..., Any],
    gripper_backend_for_side: Callable[..., Any],
    hand_backend_for_side: Callable[..., Any],
    log: Callable[..., None],
) -> VisualizationRuntime:
    camera_previewer = initial_camera_previewer
    camera_preview_active = bool(initial_camera_preview_active)
    tactile_visualizer = initial_tactile_visualizer
    CameraPreviewer = camera_previewer_type
    CameraPreviewConfig = camera_preview_config_type
    DMTacDistributedForceVisualizer = dmtac_visualizer_types[0]
    AsyncDMTacDistributedForceVisualizer = dmtac_visualizer_types[1]
    _start_camera_sources = start_camera_sources
    _build_tactile_visualizer = build_tactile_visualizer
    _enable_gui_tactile_solving = enable_gui_tactile_solving
    _tactile_solving_active = tactile_solving_active
    _tactile_visualization_available = tactile_visualization_available
    dmtac_visualization_snapshot_from_backends = dmtac_visualization_snapshot
    tactile_by_side_from_backends = tactile_snapshot
    _gripper_backend_for_side = gripper_backend_for_side
    _hand_backend_for_side = hand_backend_for_side
    _log = log

    def sync_outer_state() -> None:
        set_outer_state(
            camera_previewer,
            bool(camera_preview_active),
            tactile_visualizer,
        )
    def close_camera_preview(status: str = "") -> None:
        nonlocal camera_previewer, camera_preview_active
        if camera_previewer is not None:
            camera_previewer.close()
            camera_previewer = None
        camera_preview_active = False
        sync_outer_state()
        clutch_input.set_visualization_state(camera_active=False, status=status)


    def open_camera_preview(now: float) -> None:
        nonlocal camera_previewer, camera_preview_active
        if not preview_camera_sources:
            message = "没有可用相机配置；图像窗口未打开。"
            clutch_input.set_visualization_state(camera_active=False, status=message)
            if control_ui is not None:
                control_ui.append_message("Camera", message)
            return
        try:
            _start_camera_sources(preview_camera_sources)
            if camera_previewer is None:
                camera_previewer = CameraPreviewer(CameraPreviewConfig())
            if not camera_previewer.ensure_available():
                reason = camera_previewer.disabled_reason or "display unavailable"
                close_camera_preview(f"图像窗口不可用: {reason}")
                if control_ui is not None:
                    control_ui.append_message("Camera", f"图像窗口不可用：{reason}")
                return
            camera_preview_active = True
            sync_outer_state()
            camera_preview_rate.reset()
            message = f"图像窗口已打开: {', '.join(sorted(preview_camera_sources))}"
            clutch_input.set_visualization_state(camera_active=True, status=message)
            _log("Camera", message, now=now)
        except Exception as exc:  # noqa: BLE001
            close_camera_preview(f"图像窗口启动失败: {type(exc).__name__}: {exc}")
            if control_ui is not None:
                control_ui.append_message("Camera", f"图像窗口启动失败：{exc}")


    def close_tactile_preview(status: str = "") -> None:
        nonlocal tactile_visualizer
        if tactile_visualizer is not None:
            tactile_visualizer.close()
            tactile_visualizer = None
        sync_outer_state()
        clutch_input.set_visualization_state(tactile_active=False, status=status)


    def open_tactile_preview(now: float) -> None:
        nonlocal tactile_visualizer
        if not bool(getattr(args, "tactile_visualization_available", False)):
            message = "没有可用触觉 backend；触觉窗口未打开。"
            clutch_input.set_visualization_state(tactile_active=False, status=message)
            if control_ui is not None:
                control_ui.append_message("TactileVis", message)
            return
        try:
            if tactile_visualizer is None:
                tactile_visualizer = _build_tactile_visualizer(
                    args,
                    components.active_sides,
                    force=True,
                )
            if tactile_visualizer is None:
                raise RuntimeError("tactile visualizer was not created")
            sync_outer_state()
            tactile_preview_rate.reset()
            message = "触觉窗口已打开"
            clutch_input.set_visualization_state(tactile_active=True, status=message)
            _log("TactileVis", message, now=now)
        except Exception as exc:  # noqa: BLE001
            close_tactile_preview(f"触觉窗口启动失败: {type(exc).__name__}: {exc}")
            if control_ui is not None:
                control_ui.append_message("TactileVis", f"触觉窗口启动失败：{exc}")


    def handle_startup_visualization_requests(now: float) -> None:
        """Consume visualization controls while waiting for runtime Home."""

        if clutch_input.consume_device_retry_request():
            retry_failed_devices()
        if clutch_input.consume_camera_visualization_toggle_request():
            if camera_preview_active or camera_previewer is not None:
                close_camera_preview("图像窗口已关闭")
                if control_ui is not None:
                    control_ui.append_message("Camera", "图像窗口已关闭。")
            else:
                open_camera_preview(now)
        if clutch_input.consume_tactile_solving_enable_request():
            try:
                changed = _enable_gui_tactile_solving(args, components)
                message = (
                    f"触觉解算已开启：{changed} 个 worker 已按固定传感器标定重启；"
                    "解算保持开启，关闭窗口不会改变采集字段。"
                )
                clutch_input.set_visualization_state(
                    tactile_solving_active=True,
                    status=message,
                )
                if control_ui is not None:
                    control_ui.append_message("TactileSolve", message)
                _log("TactileSolve", message, now=now)
            except Exception as exc:  # noqa: BLE001
                message = f"触觉解算开启失败: {type(exc).__name__}: {exc}"
                args.tactile_solving_active = _tactile_solving_active(args, components)
                args.tactile_visualization_available = _tactile_visualization_available(
                    args, components
                )
                clutch_input.set_visualization_state(
                    tactile_solving_active=args.tactile_solving_active,
                    status=message,
                )
                if control_ui is not None:
                    control_ui.append_message("TactileSolve", message)
        if clutch_input.consume_tactile_visualization_toggle_request():
            if tactile_visualizer is not None:
                close_tactile_preview("触觉窗口已关闭")
                if control_ui is not None:
                    control_ui.append_message("TactileVis", "触觉窗口已关闭。")
            else:
                open_tactile_preview(now)


    def update_startup_visualizations(now: float) -> None:
        """Keep native preview windows responsive before runtime Home."""

        if tactile_visualizer is not None and tactile_preview_rate.should_update(now):
            try:
                if isinstance(
                    tactile_visualizer,
                    (DMTacDistributedForceVisualizer, AsyncDMTacDistributedForceVisualizer),
                ):
                    tactile_force, tactile_health = dmtac_visualization_snapshot_from_backends(
                        components.active_sides,
                        lambda side: _gripper_backend_for_side(components, side),
                        now=now,
                    )
                    tactile_keys = tactile_visualizer.update(
                        tactile_force,
                        health_by_side=tactile_health,
                    )
                else:
                    tactile_keys = tactile_visualizer.update(
                        tactile_by_side_from_backends(
                            components.active_sides,
                            lambda side: _hand_backend_for_side(components, side),
                        )
                    ) or []
                if any(key in {"q", "\x1b"} for key in tactile_keys):
                    close_tactile_preview("触觉窗口已关闭")
            except Exception as exc:  # noqa: BLE001
                close_tactile_preview(f"触觉窗口刷新失败: {type(exc).__name__}: {exc}")
                if control_ui is not None:
                    control_ui.append_message("TactileVis", f"触觉窗口刷新失败：{exc}")

        if (
            camera_preview_active
            and camera_previewer is not None
            and camera_preview_rate.should_update(now)
        ):
            try:
                camera_keys = camera_previewer.update(preview_camera_sources)
                if any(key in {"q", "\x1b"} for key in camera_keys):
                    close_camera_preview("图像窗口已关闭")
            except Exception as exc:  # noqa: BLE001
                close_camera_preview(f"图像窗口刷新失败: {type(exc).__name__}: {exc}")
                if control_ui is not None:
                    control_ui.append_message("Camera", f"图像窗口刷新失败：{exc}")

    def current_state() -> tuple[Any, bool, Any]:
        return camera_previewer, bool(camera_preview_active), tactile_visualizer

    def sync_state(
        previewer: Any,
        camera_active: bool,
        visualizer: Any,
    ) -> None:
        nonlocal camera_previewer, camera_preview_active, tactile_visualizer
        camera_previewer = previewer
        camera_preview_active = bool(camera_active)
        tactile_visualizer = visualizer
        sync_outer_state()

    sync_outer_state()
    return VisualizationRuntime(
        close_camera_preview=close_camera_preview,
        open_camera_preview=open_camera_preview,
        close_tactile_preview=close_tactile_preview,
        open_tactile_preview=open_tactile_preview,
        handle_startup_requests=handle_startup_visualization_requests,
        update_startup_visualizations=update_startup_visualizations,
        current_state=current_state,
        sync_state=sync_state,
    )


__all__ = ["VisualizationRuntime", "build_visualization_runtime"]

