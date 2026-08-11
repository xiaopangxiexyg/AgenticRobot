"""Device-neutral admission transactions before real-teleop device startup.

This module owns optional operator-UI startup, camera source configuration,
and collection-recorder construction.  It does not start camera or robot
devices, begin an episode, or own steady-state control and recording.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StartupUiRuntime:
    """Resources created by the optional GUI startup transaction."""

    control_ui: Any | None
    progress: Any | None


def initialize_startup_ui(
    args: Any,
    *,
    startup_cleanup: Any,
    ui_type: type,
    progress_type: type,
) -> StartupUiRuntime:
    """Create, start, and register the optional operator UI resources."""

    if not bool(getattr(args, "gui_requested", False)):
        return StartupUiRuntime(control_ui=None, progress=None)

    control_ui = ui_type(
        open_browser=True,
        resizable_sidebars=bool(
            getattr(args, "gui_resizable_sidebars", True)
        ),
        input_activity_indicators=bool(
            getattr(args, "gui_input_activity_indicators", True)
        ),
    )
    startup_cleanup.add("web_ui", control_ui.close)
    control_ui.publish(
        stage="设备连接中",
        control_hint="正在尝试连接全部配置设备；失败设备不会阻止进入 GUI。",
        active_controls=["q"],
    )
    control_ui.start()

    progress = progress_type(control_ui)
    startup_cleanup.add("startup_progress", progress.close)
    progress.start()
    return StartupUiRuntime(control_ui=control_ui, progress=progress)


@dataclass(frozen=True)
class StartupCameraConfiguration:
    """Admitted camera sources and the selected preview subset."""

    sources: Mapping[str, Any]
    preview_sources: Mapping[str, Any]


@dataclass(frozen=True)
class StartupComponentAssembly:
    """Constructed runtime component bundle and optional early visualizer."""

    components: Any
    tactile_visualizer: Any | None


def assemble_startup_components(
    args: Any,
    *,
    startup_cleanup: Any,
    startup_progress: Any | None,
    active_sides_from_args: Callable[[Any], tuple[str, ...]],
    build_tactile_visualizer: Callable[[Any, tuple[str, ...]], Any | None],
    build_components: Callable[[Any], Any],
    register_component_cleanup: Callable[[Any, Any], None],
) -> StartupComponentAssembly:
    """Construct the runtime bundle and register all early cleanup hooks."""

    if startup_progress is not None:
        startup_progress.begin(
            "runtime.components",
            label="遥操运行时组件",
            message="正在构建机械臂、夹爪与 tracker 组件",
            requirement="runtime_core",
        )

    tactile_visualizer = None
    if bool(getattr(args, "visualize_tactile", False)) and not bool(
        getattr(args, "skip_renderer", False)
    ):
        tactile_visualizer = build_tactile_visualizer(
            args,
            active_sides_from_args(args),
        )
        if tactile_visualizer is not None:
            startup_cleanup.add(
                "tactile_visualizer",
                tactile_visualizer.close,
            )

    try:
        components = build_components(args)
    except Exception as exc:  # noqa: BLE001 - preserve component-builder failures.
        if startup_progress is not None:
            startup_progress.fail(
                "runtime.components",
                f"{type(exc).__name__}: {exc}",
                message="运行时组件构建失败",
            )
            startup_progress.close()
        if tactile_visualizer is not None:
            tactile_visualizer.close()
        raise

    if startup_progress is not None:
        startup_progress.succeed(
            "runtime.components",
            message="运行时组件已构建，尚未启动设备",
        )
    register_component_cleanup(startup_cleanup, components)
    return StartupComponentAssembly(
        components=components,
        tactile_visualizer=tactile_visualizer,
    )


def configure_startup_cameras(
    args: Any,
    *,
    startup_cleanup: Any,
    startup_progress: Any | None,
    control_ui: Any | None,
    capability_admission: bool,
    build_sources: Callable[[Any], Mapping[str, Any]],
    select_preview_sources: Callable[[Mapping[str, Any], str], Mapping[str, Any]],
    strict_failure_cleanup: Callable[[], None],
    log: Callable[..., None],
) -> StartupCameraConfiguration:
    """Construct and admit camera configuration without starting devices."""

    if startup_progress is not None:
        startup_progress.begin(
            "camera.config",
            label="相机配置",
            message="正在解析相机注册表与预览配置",
            requirement="required_for_collection",
        )
    try:
        camera_sources = build_sources(args)
        for camera_key, source in camera_sources.items():
            startup_cleanup.add(f"camera[{camera_key}]", source.close)
        try:
            preview_camera_sources = select_preview_sources(
                camera_sources,
                str(getattr(args, "vis_cam_key", "") or ""),
            )
        except ValueError:
            if bool(getattr(args, "vis_cam", False)):
                raise
            log(
                "Camera",
                f"preview camera key {getattr(args, 'vis_cam_key', '')!r} "
                "is unavailable; preview disabled.",
            )
            preview_camera_sources = {}
    except Exception as exc:  # noqa: BLE001 - capability admission isolates config failures.
        if startup_progress is not None:
            startup_progress.fail(
                "camera.config",
                f"{type(exc).__name__}: {exc}",
                message="相机配置加载失败",
            )
        if not capability_admission:
            strict_failure_cleanup()
            if startup_progress is not None:
                startup_progress.close()
            raise
        detail = f"{type(exc).__name__}: {exc}"
        args.camera_startup_error = detail
        camera_sources = {}
        preview_camera_sources = {}
        log(
            "Camera",
            "WARNING: camera configuration/startup was isolated from GUI and "
            f"arm controls: {detail}",
        )
        if control_ui is not None:
            control_ui.append_message("Camera", detail)
    else:
        if startup_progress is not None:
            startup_progress.succeed(
                "camera.config",
                message=f"已加载 {len(camera_sources)} 个相机配置",
            )

    setattr(args, "camera_visualization_available", bool(preview_camera_sources))
    setattr(args, "configured_camera_keys", tuple(sorted(camera_sources)))
    return StartupCameraConfiguration(
        sources=camera_sources,
        preview_sources=preview_camera_sources,
    )


def prepare_startup_dataset_recorder(
    args: Any,
    *,
    components: Any,
    camera_sources: Mapping[str, Any],
    startup_cleanup: Any,
    startup_progress: Any | None,
    control_ui: Any | None,
    capability_admission: bool,
    recorder_type: type,
    log: Callable[..., None],
) -> Any | None:
    """Construct and register the optional collection recorder.

    Successful construction is registered immediately with the early-startup
    cleanup owner.  The writer subprocess is not started here.
    """

    if bool(getattr(args, "no_collection", False)):
        return None

    if startup_progress is not None:
        startup_progress.begin(
            "dataset.writer.prepare",
            label="数采 writer 配置",
            message="正在校验数据集路径、模态声明与 writer 配置",
            requirement="required_for_collection",
        )
    try:
        recorder = recorder_type(
            args=args,
            components=components,
            camera_sources=camera_sources,
        )
        startup_cleanup.add("dataset_recorder", recorder.close)
    except (OSError, RuntimeError, ValueError) as exc:
        if startup_progress is not None:
            startup_progress.fail(
                "dataset.writer.prepare",
                f"{type(exc).__name__}: {exc}",
                message="writer 配置校验失败",
            )
        if not capability_admission:
            if startup_progress is not None:
                startup_progress.close()
            raise
        detail = f"{type(exc).__name__}: {exc}"
        args.dataset_startup_error = detail
        log(
            "Dataset",
            "WARNING: dataset writer construction was isolated from GUI and "
            f"arm controls: {detail}",
        )
        if control_ui is not None:
            control_ui.append_message("Dataset", detail)
        return None

    if startup_progress is not None:
        startup_progress.succeed(
            "dataset.writer.prepare",
            message="writer 配置已就绪，尚未启动子进程",
        )
    return recorder


__all__ = [
    "StartupCameraConfiguration",
    "StartupComponentAssembly",
    "StartupUiRuntime",
    "assemble_startup_components",
    "configure_startup_cameras",
    "initialize_startup_ui",
    "prepare_startup_dataset_recorder",
]
