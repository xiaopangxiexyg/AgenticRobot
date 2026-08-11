from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop.startup_admission import (
    configure_startup_cameras,
)


class _Progress:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def begin(self, key: str, **details: object) -> None:
        self.events.append(("begin", key, details))

    def succeed(self, key: str, **details: object) -> None:
        self.events.append(("succeed", key, details))

    def fail(self, key: str, detail: str, **details: object) -> None:
        self.events.append(("fail", key, detail, details))

    def close(self) -> None:
        self.events.append(("close",))


class _Cleanup:
    def __init__(self) -> None:
        self.entries: list[tuple[str, object]] = []

    def add(self, label: str, callback: object) -> None:
        self.entries.append((label, callback))


class _Source:
    def close(self) -> None:
        pass


def _configure(
    *,
    args: object | None = None,
    capability: bool = False,
    build_sources=None,
    select_preview_sources=None,
):
    resolved_args = args or SimpleNamespace(vis_cam=False, vis_cam_key="front")
    cleanup = _Cleanup()
    progress = _Progress()
    logs: list[tuple[str, str]] = []
    messages: list[tuple[str, str]] = []
    strict_cleanup: list[str] = []
    result = configure_startup_cameras(
        resolved_args,
        startup_cleanup=cleanup,
        startup_progress=progress,
        control_ui=SimpleNamespace(
            append_message=lambda tag, message: messages.append((tag, message))
        ),
        capability_admission=capability,
        build_sources=build_sources or (lambda _: {"front": _Source()}),
        select_preview_sources=(
            select_preview_sources
            or (lambda sources, key: {key: sources[key]})
        ),
        strict_failure_cleanup=lambda: strict_cleanup.append("cleanup"),
        log=lambda tag, message: logs.append((tag, message)),
    )
    return (
        result,
        resolved_args,
        cleanup,
        progress,
        logs,
        messages,
        strict_cleanup,
    )


def test_startup_camera_configuration_registers_sources_and_preview() -> None:
    result, args, cleanup, progress, logs, messages, strict_cleanup = _configure()

    assert tuple(result.sources) == ("front",)
    assert result.preview_sources == {"front": result.sources["front"]}
    assert cleanup.entries == [("camera[front]", result.sources["front"].close)]
    assert args.camera_visualization_available is True
    assert args.configured_camera_keys == ("front",)
    assert progress.events[-1] == (
        "succeed",
        "camera.config",
        {"message": "已加载 1 个相机配置"},
    )
    assert logs == []
    assert messages == []
    assert strict_cleanup == []


def test_missing_optional_preview_key_disables_preview_without_rejecting_sources() -> None:
    def reject_preview(*_: object):
        raise ValueError("unknown preview")

    result, args, cleanup, progress, logs, messages, strict_cleanup = _configure(
        select_preview_sources=reject_preview,
    )

    assert tuple(result.sources) == ("front",)
    assert result.preview_sources == {}
    assert args.camera_visualization_available is False
    assert args.configured_camera_keys == ("front",)
    assert logs == [
        (
            "Camera",
            "preview camera key 'front' is unavailable; preview disabled.",
        )
    ]
    assert progress.events[-1][0] == "succeed"
    assert messages == []
    assert strict_cleanup == []
    assert len(cleanup.entries) == 1


def test_required_preview_failure_preserves_strict_cleanup_and_exception_order() -> None:
    args = SimpleNamespace(vis_cam=True, vis_cam_key="missing")
    cleanup = _Cleanup()
    progress = _Progress()
    strict_cleanup: list[str] = []

    with pytest.raises(ValueError, match="unknown preview"):
        configure_startup_cameras(
            args,
            startup_cleanup=cleanup,
            startup_progress=progress,
            control_ui=None,
            capability_admission=False,
            build_sources=lambda _: {"front": _Source()},
            select_preview_sources=lambda *_: (_ for _ in ()).throw(
                ValueError("unknown preview")
            ),
            strict_failure_cleanup=lambda: strict_cleanup.append("cleanup"),
            log=lambda *_: None,
        )

    assert strict_cleanup == ["cleanup"]
    assert progress.events[-2:] == [
        (
            "fail",
            "camera.config",
            "ValueError: unknown preview",
            {"message": "相机配置加载失败"},
        ),
        ("close",),
    ]


def test_capability_camera_failure_isolated_with_empty_configuration() -> None:
    def fail_build(_: object):
        raise OSError("registry unavailable")

    result, args, cleanup, progress, logs, messages, strict_cleanup = _configure(
        capability=True,
        build_sources=fail_build,
    )

    assert result.sources == {}
    assert result.preview_sources == {}
    assert args.camera_visualization_available is False
    assert args.configured_camera_keys == ()
    assert args.camera_startup_error == "OSError: registry unavailable"
    assert cleanup.entries == []
    assert progress.events[-1][0] == "fail"
    assert logs == [
        (
            "Camera",
            "WARNING: camera configuration/startup was isolated from GUI and "
            "arm controls: OSError: registry unavailable",
        )
    ]
    assert messages == [("Camera", "OSError: registry unavailable")]
    assert strict_cleanup == []
