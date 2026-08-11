from __future__ import annotations

from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop.startup_admission import (
    prepare_startup_dataset_recorder,
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


class _Recorder:
    instances: list["_Recorder"] = []

    def __init__(self, *, args: object, components: object, camera_sources: object) -> None:
        self.args = args
        self.components = components
        self.camera_sources = camera_sources
        self.closed = False
        self.instances.append(self)

    def close(self) -> None:
        self.closed = True


def _prepare(
    *,
    no_collection: bool = False,
    capability: bool = False,
    recorder_type: type = _Recorder,
):
    args = SimpleNamespace(no_collection=no_collection)
    cleanup = _Cleanup()
    progress = _Progress()
    messages: list[tuple[str, str]] = []
    logs: list[tuple[str, str]] = []
    recorder = prepare_startup_dataset_recorder(
        args,
        components="components",
        camera_sources={"front": "camera"},
        startup_cleanup=cleanup,
        startup_progress=progress,
        control_ui=SimpleNamespace(
            append_message=lambda tag, message: messages.append((tag, message))
        ),
        capability_admission=capability,
        recorder_type=recorder_type,
        log=lambda tag, message: logs.append((tag, message)),
    )
    return recorder, args, cleanup, progress, messages, logs


def test_startup_dataset_recorder_success_registers_cleanup_before_receipt() -> None:
    _Recorder.instances.clear()

    recorder, args, cleanup, progress, messages, logs = _prepare()

    assert recorder is _Recorder.instances[-1]
    assert recorder.args is args
    assert recorder.components == "components"
    assert recorder.camera_sources == {"front": "camera"}
    assert cleanup.entries == [("dataset_recorder", recorder.close)]
    assert progress.events == [
        (
            "begin",
            "dataset.writer.prepare",
            {
                "label": "数采 writer 配置",
                "message": "正在校验数据集路径、模态声明与 writer 配置",
                "requirement": "required_for_collection",
            },
        ),
        (
            "succeed",
            "dataset.writer.prepare",
            {"message": "writer 配置已就绪，尚未启动子进程"},
        ),
    ]
    assert messages == []
    assert logs == []


def test_startup_dataset_recorder_disabled_has_no_side_effects() -> None:
    recorder, args, cleanup, progress, messages, logs = _prepare(
        no_collection=True,
    )

    assert recorder is None
    assert vars(args) == {"no_collection": True}
    assert cleanup.entries == []
    assert progress.events == []
    assert messages == []
    assert logs == []


def test_startup_dataset_recorder_strict_failure_closes_progress_and_reraises() -> None:
    class FailingRecorder:
        def __init__(self, **_: object) -> None:
            raise ValueError("invalid contract")

    args = SimpleNamespace(no_collection=False)
    cleanup = _Cleanup()
    progress = _Progress()
    with pytest.raises(ValueError, match="invalid contract"):
        prepare_startup_dataset_recorder(
            args,
            components="components",
            camera_sources={},
            startup_cleanup=cleanup,
            startup_progress=progress,
            control_ui=None,
            capability_admission=False,
            recorder_type=FailingRecorder,
            log=lambda *_: None,
        )

    assert cleanup.entries == []
    assert progress.events[-2:] == [
        (
            "fail",
            "dataset.writer.prepare",
            "ValueError: invalid contract",
            {"message": "writer 配置校验失败"},
        ),
        ("close",),
    ]


def test_startup_dataset_recorder_capability_failure_isolated_with_receipts() -> None:
    class FailingRecorder:
        def __init__(self, **_: object) -> None:
            raise OSError("read-only target")

    recorder, args, cleanup, progress, messages, logs = _prepare(
        capability=True,
        recorder_type=FailingRecorder,
    )

    assert recorder is None
    assert cleanup.entries == []
    assert args.dataset_startup_error == "OSError: read-only target"
    assert progress.events[-1] == (
        "fail",
        "dataset.writer.prepare",
        "OSError: read-only target",
        {"message": "writer 配置校验失败"},
    )
    assert messages == [("Dataset", "OSError: read-only target")]
    assert logs == [
        (
            "Dataset",
            "WARNING: dataset writer construction was isolated from GUI and "
            "arm controls: OSError: read-only target",
        )
    ]
