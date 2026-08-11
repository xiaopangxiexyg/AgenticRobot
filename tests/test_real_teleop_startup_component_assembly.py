from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop.startup_admission import (
    assemble_startup_components,
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


class _Visualizer:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def test_startup_component_assembly_registers_visualizer_and_component_bundle() -> None:
    args = SimpleNamespace(visualize_tactile=True, skip_renderer=False)
    cleanup = _Cleanup()
    progress = _Progress()
    visualizer = _Visualizer()
    registrations: list[tuple[object, object]] = []
    active_side_calls: list[object] = []
    components = object()

    result = assemble_startup_components(
        args,
        startup_cleanup=cleanup,
        startup_progress=progress,
        active_sides_from_args=lambda value: (
            active_side_calls.append(value) or ("left", "right")
        ),
        build_tactile_visualizer=lambda value, sides: (
            visualizer
            if value is args and sides == ("left", "right")
            else None
        ),
        build_components=lambda value: components if value is args else None,
        register_component_cleanup=lambda registry, value: registrations.append(
            (registry, value)
        ),
    )

    assert result.components is components
    assert result.tactile_visualizer is visualizer
    assert active_side_calls == [args]
    assert cleanup.entries == [("tactile_visualizer", visualizer.close)]
    assert registrations == [(cleanup, components)]
    assert progress.events[-1] == (
        "succeed",
        "runtime.components",
        {"message": "运行时组件已构建，尚未启动设备"},
    )


def test_startup_component_assembly_skips_unrequested_visualizer() -> None:
    visualizer_builds: list[str] = []

    result = assemble_startup_components(
        SimpleNamespace(visualize_tactile=False, skip_renderer=False),
        startup_cleanup=_Cleanup(),
        startup_progress=None,
        active_sides_from_args=lambda _: ("left",),
        build_tactile_visualizer=lambda *_: visualizer_builds.append("build"),
        build_components=lambda _: "components",
        register_component_cleanup=lambda *_: None,
    )

    assert result.components == "components"
    assert result.tactile_visualizer is None
    assert visualizer_builds == []


def test_startup_component_failure_closes_progress_and_early_visualizer() -> None:
    progress = _Progress()
    visualizer = _Visualizer()
    registrations: list[object] = []

    with pytest.raises(RuntimeError, match="component failure"):
        assemble_startup_components(
            SimpleNamespace(visualize_tactile=True, skip_renderer=False),
            startup_cleanup=_Cleanup(),
            startup_progress=progress,
            active_sides_from_args=lambda _: ("left",),
            build_tactile_visualizer=lambda *_: visualizer,
            build_components=lambda _: (_ for _ in ()).throw(
                RuntimeError("component failure")
            ),
            register_component_cleanup=lambda *_: registrations.append("registered"),
        )

    assert visualizer.close_count == 1
    assert registrations == []
    assert progress.events[-2:] == [
        (
            "fail",
            "runtime.components",
            "RuntimeError: component failure",
            {"message": "运行时组件构建失败"},
        ),
        ("close",),
    ]
