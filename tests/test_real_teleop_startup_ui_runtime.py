from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.real_teleop.startup_admission import initialize_startup_ui


class _Cleanup:
    def __init__(self) -> None:
        self.entries: list[tuple[str, object]] = []

    def add(self, label: str, callback: object) -> None:
        self.entries.append((label, callback))


class _Ui:
    instances: list["_Ui"] = []

    def __init__(self, **options: object) -> None:
        self.options = options
        self.events: list[tuple[object, ...]] = []
        self.instances.append(self)

    def publish(self, **payload: object) -> None:
        self.events.append(("publish", payload))

    def start(self) -> None:
        self.events.append(("start",))

    def close(self) -> None:
        self.events.append(("close",))


class _Progress:
    instances: list["_Progress"] = []

    def __init__(self, ui: _Ui) -> None:
        self.ui = ui
        self.events: list[str] = []
        self.instances.append(self)

    def start(self) -> None:
        self.events.append("start")

    def close(self) -> None:
        self.events.append("close")


def test_startup_ui_disabled_does_not_construct_resources() -> None:
    _Ui.instances.clear()
    _Progress.instances.clear()
    cleanup = _Cleanup()

    result = initialize_startup_ui(
        SimpleNamespace(gui_requested=False),
        startup_cleanup=cleanup,
        ui_type=_Ui,
        progress_type=_Progress,
    )

    assert result.control_ui is None
    assert result.progress is None
    assert cleanup.entries == []
    assert _Ui.instances == []
    assert _Progress.instances == []


def test_startup_ui_owns_creation_start_and_cleanup_registration() -> None:
    _Ui.instances.clear()
    _Progress.instances.clear()
    cleanup = _Cleanup()

    result = initialize_startup_ui(
        SimpleNamespace(
            gui_requested=True,
            gui_resizable_sidebars=False,
            gui_input_activity_indicators=False,
        ),
        startup_cleanup=cleanup,
        ui_type=_Ui,
        progress_type=_Progress,
    )

    ui = _Ui.instances[-1]
    progress = _Progress.instances[-1]
    assert result.control_ui is ui
    assert result.progress is progress
    assert ui.options == {
        "open_browser": True,
        "resizable_sidebars": False,
        "input_activity_indicators": False,
    }
    assert ui.events == [
        (
            "publish",
            {
                "stage": "设备连接中",
                "control_hint": "正在尝试连接全部配置设备；失败设备不会阻止进入 GUI。",
                "active_controls": ["q"],
            },
        ),
        ("start",),
    ]
    assert progress.ui is ui
    assert progress.events == ["start"]
    assert cleanup.entries == [
        ("web_ui", ui.close),
        ("startup_progress", progress.close),
    ]


def test_startup_ui_preserves_default_ui_options() -> None:
    _Ui.instances.clear()
    cleanup = _Cleanup()

    initialize_startup_ui(
        SimpleNamespace(gui_requested=True),
        startup_cleanup=cleanup,
        ui_type=_Ui,
        progress_type=_Progress,
    )

    assert _Ui.instances[-1].options == {
        "open_browser": True,
        "resizable_sidebars": True,
        "input_activity_indicators": True,
    }
