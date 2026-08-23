"""Startup-only progress configuration and rebuildable UI publication."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from ..operator_ui_ports import OperatorUiPort
from .progress_contracts import (
    ProgressArmControlModeSelectionCommand,
    ProgressEntryStepSelectionCommand,
    ProgressProfileSelectionCommand,
)


class StartupProgressControlPort(Protocol):
    """Narrow control-input surface needed before runtime Home."""

    def build_startup_progress_ui_payload(
        self,
        *,
        dataset_recorder: object | None,
        profile_selection_enabled: bool,
    ) -> dict[str, object]: ...

    def publish_progress_ui_state(self, payload: Mapping[str, object]) -> None: ...

    def consume_progress_profile_selection_intents(
        self,
    ) -> tuple[ProgressProfileSelectionCommand, ...]: ...

    def consume_progress_arm_control_mode_selection_intents(
        self,
    ) -> tuple[ProgressArmControlModeSelectionCommand, ...]: ...

    def consume_progress_entry_step_selection_intents(
        self,
    ) -> tuple[ProgressEntryStepSelectionCommand, ...]: ...

    def select_managed_progress_profile(
        self,
        *,
        dataset_recorder: object | None,
        profile_id: str,
    ) -> Any: ...

    def select_progress_entry_step(
        self,
        *,
        dataset_recorder: object | None,
        progress_step_id: str,
    ) -> str: ...


class StartupProgressLogger(Protocol):
    def __call__(
        self,
        category: str,
        message: str,
        *,
        now: float | None = None,
    ) -> None: ...


class RealTeleopStartupProgressRuntime:
    """Own startup progress selection, publication, and error deduplication.

    This object is configuration-only. It does not start an Episode, mutate a
    dataset binding, control a device, or own the progress workflow used after
    runtime Home.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        control_input: StartupProgressControlPort,
        control_ui: OperatorUiPort | None,
        dataset_recorder: Callable[[], object | None],
        log: StartupProgressLogger,
        profile_selection_enabled: bool = True,
    ) -> None:
        self._args = args
        self._control_input = control_input
        self._control_ui = control_ui
        self._dataset_recorder = dataset_recorder
        self._log = log
        self._profile_selection_enabled = bool(profile_selection_enabled)
        self._published_signature: tuple[object, ...] | None = None
        self._error_signature: tuple[object, ...] | None = None

    def _state_signature(self) -> tuple[object, ...]:
        recorder = self._dataset_recorder()
        recorder_profile = (
            None if recorder is None else getattr(recorder, "progress_profile", None)
        )
        requested_profile = getattr(
            self._args,
            "collection_progress_profile",
            None,
        )
        return (
            bool(getattr(self._args, "collection_progress_gui", False)),
            self._profile_selection_enabled,
            str(getattr(self._args, "collection_task_id", "") or ""),
            str(Path(self._args.dataset_root).expanduser().resolve()),
            id(recorder),
            str(getattr(recorder_profile, "semantic_content_hash", "")),
            str(getattr(requested_profile, "semantic_content_hash", "")),
            bool(
                False
                if recorder is None
                else getattr(
                    recorder,
                    "progress_profile_selection_locked",
                    False,
                )
            ),
            str(
                getattr(
                    self._args,
                    "collection_progress_arm_control_mode",
                    "profile",
                )
            ),
            str(
                getattr(
                    self._args,
                    "collection_progress_entry_step",
                    "",
                )
                or ""
            ),
        )

    def publish(self, *, force: bool = False) -> None:
        """Publish one changed startup projection, isolating UI failures."""

        if self._control_ui is None:
            return
        signature = self._state_signature()
        if not force and signature == self._published_signature:
            return
        try:
            payload = self._control_input.build_startup_progress_ui_payload(
                dataset_recorder=self._dataset_recorder(),
                profile_selection_enabled=self._profile_selection_enabled,
            )
            self._control_input.publish_progress_ui_state(payload)
            self._published_signature = signature
            self._error_signature = None
        except Exception as exc:  # noqa: BLE001 - UI remains observational.
            failure_signature = (
                *signature,
                type(exc).__qualname__,
                str(exc),
            )
            self._published_signature = signature
            if failure_signature == self._error_signature:
                return
            self._log(
                "Progress",
                "启动阶段长程进度 GUI 状态发布失败；原遥操 GUI 保持可用："
                f"{type(exc).__name__}: {exc}",
            )
            self._control_ui.append_message(
                "Progress",
                "长程进度面板初始化失败："
                f"{type(exc).__name__}: {exc}",
            )
            self._error_signature = failure_signature

    def handle_pending_selections(self, *, now: float) -> None:
        """Apply admitted idle selections and republish canonical state."""

        arm_mode_selections = (
            self._control_input.consume_progress_arm_control_mode_selection_intents()
        )
        for selection in arm_mode_selections:
            self._args.collection_progress_arm_control_mode = selection.mode
            if self._control_ui is not None:
                self._control_ui.append_message(
                    "Progress",
                    f"已选择长程进度臂控制模式：{selection.mode}。",
                )
            self.publish(force=True)

        selections = self._control_input.consume_progress_profile_selection_intents()
        for selection in selections:
            try:
                selected = self._control_input.select_managed_progress_profile(
                    dataset_recorder=self._dataset_recorder(),
                    profile_id=selection.profile_id,
                )
                if self._control_ui is not None:
                    self._control_ui.append_message(
                        "Progress",
                        "已选择长程进度 profile："
                        f"{selected.profile_id}@{selected.profile_version}；"
                        "首个 Episode 启动时将按完整 hash 锁定到数据集。",
                    )
            except Exception as exc:  # noqa: BLE001 - configuration remains usable.
                message = (
                    "长程进度 profile 选择失败，原配置保持不变："
                    f"{type(exc).__name__}: {exc}"
                )
                self._log("Progress", message, now=now)
                if self._control_ui is not None:
                    self._control_ui.append_message("Progress", message)
            finally:
                self.publish(force=True)
        entry_selections = (
            self._control_input.consume_progress_entry_step_selection_intents()
        )
        for selection in entry_selections:
            try:
                selected_step_id = self._control_input.select_progress_entry_step(
                    dataset_recorder=self._dataset_recorder(),
                    progress_step_id=selection.progress_step_id,
                )
                if self._control_ui is not None:
                    self._control_ui.append_message(
                        "Progress",
                        "下一个 Episode 将从子任务开始："
                        f"{selected_step_id}；之后沿用正常推进与标注流程。",
                    )
            except Exception as exc:  # noqa: BLE001 - configuration remains usable.
                message = (
                    "长程进度起始子任务选择失败，原配置保持不变："
                    f"{type(exc).__name__}: {exc}"
                )
                self._log("Progress", message, now=now)
                if self._control_ui is not None:
                    self._control_ui.append_message("Progress", message)
            finally:
                self.publish(force=True)
        self.publish()


__all__ = [
    "RealTeleopStartupProgressRuntime",
    "StartupProgressControlPort",
]
