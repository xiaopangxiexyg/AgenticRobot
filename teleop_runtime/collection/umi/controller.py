"""UMI control and recording state mapped onto the existing formal GUI."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Iterable, MutableMapping

import numpy as np

from robot_motion.runtime.contracts import TeleopObservation, merge_teleop_commands

from ...gui.calibration_web_ui import CalibrationWebUi
from ...real_teleop.gui_tracker_status import raw_tracker_pose_state_from_channels
from ..operator_labels import OperatorLabelStateMachine
from ..episode_curation import EpisodeCurationSession, load_task_variant_profile
from .calibration import UmiCalibrationBinding
from .components import UmiComponents
from .config import UmiCollectionConfig
from .recorder import UmiDatasetRecorder
from .snapshot import UmiSnapshotAssembler
from .types import SnapshotRejection


_FOOT_PAD_TEST_LABELS = {
    "fix_arm_left": "踏板 1 / F13 → 固定左臂",
    "fix_arm_right": "踏板 2 / F14 → 固定右臂",
    "init_arm_left": "踏板 3 / F15 → 左臂归位",
    "init_arm_right": "踏板 4 / F16 → 右臂归位",
}


class UmiCollectionController:
    def __init__(
        self,
        *,
        config: UmiCollectionConfig,
        calibration: UmiCalibrationBinding,
        components: UmiComponents,
        recorder: UmiDatasetRecorder,
        ui: CalibrationWebUi,
        labels: OperatorLabelStateMachine,
        snapshot_assembler: UmiSnapshotAssembler,
        last_commands: MutableMapping[str, float] | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.components = components
        self.recorder = recorder
        self.ui = ui
        self.labels = labels
        self.snapshot_assembler = snapshot_assembler
        self.should_quit = False
        self.last_commands = (
            last_commands
            if last_commands is not None
            else {"left": 0.0, "right": 0.0}
        )
        self.last_rejection: tuple[str, ...] = ()
        self._next_sample_at = time.perf_counter()
        self._selected_label_side = {"fix": "left", "init": "left"}
        self._event_status = "ready"
        self._event_message = ""
        self._calibration_expired = False
        self.recalibration_requested = False
        self.angle_recalibration_requested = False
        self.device_retry_requested = False
        self.plus_curation = EpisodeCurationSession(
            enabled=bool(getattr(config, "plus_episode_curation", False)),
            annotation_root=getattr(config, "plus_annotation_root", None),
            task_profile=(
                load_task_variant_profile(getattr(config, "plus_task_profile", None))
                if bool(getattr(config, "plus_episode_curation", False))
                else None
            ),
            source_binding=str(
                getattr(config, "plus_source_binding", "v23_compat_content_locator_v1")
            ),
            source_schema="dexteleop_dataset_v2.3",
            status=("ready" if bool(getattr(config, "plus_episode_curation", False)) else "off"),
        )
        self._plus_curation_request: dict[str, object] | None = None
        self._plus_curation_save_thread: threading.Thread | None = None
        self._foot_pad_test_active = False
        self._foot_pad_test_last_event = ""
        self._foot_pad_test_status = (
            "测试模式关闭；点击“开始测试踏板”后逐个踩踏板。"
        )
        if not bool(getattr(self.config, "foot_pad_enabled", True)):
            self._foot_pad_test_status = (
                "脚踏输入未启用；GUI/键盘 fix/init 标注保持可用。"
            )
        foot_pad_error = str(getattr(self.components, "foot_pad_error", "") or "")
        if foot_pad_error:
            self._foot_pad_test_status = (
                "脚踏设备未连接；仅踏板快捷输入不可用，GUI/键盘标注保持可用。"
            )
        self._last_foot_pad_online = self._foot_pad_online()

    def _foot_pad_online(self) -> bool:
        source = getattr(self.components, "foot_pad", None)
        if source is None:
            return False
        configured_count = getattr(source, "foot_pad_count", None)
        if configured_count is not None:
            try:
                if int(configured_count) not in {2, 4}:
                    return False
            except (TypeError, ValueError):
                return False
        online = getattr(source, "is_online", None)
        # Compatibility fakes used by collection tests do not expose the
        # production adapter property and represent an online input.
        return bool(online if online is not None else True)

    def _foot_pad_count(self) -> int:
        return int(
            getattr(getattr(self.components, "foot_pad", None), "foot_pad_count", 0)
            or 0
        )

    def _foot_pad_mode_enabled(self) -> bool:
        return self._foot_pad_online() and self._foot_pad_count() in {2, 4}

    def _foot_pad_test_ui_state(self) -> dict[str, object]:
        requested = bool(getattr(self.config, "foot_pad_enabled", True))
        online = self._foot_pad_online()
        count = self._foot_pad_count()
        if online and count in {2, 4}:
            status = "online"
            status_label = f"{count} 个已连接"
            status_message = f"已连接并识别 {count} 个踏板。"
        elif requested:
            status = "unavailable"
            status_label = "未连接 / 未就绪"
            status_message = str(
                getattr(self.components, "foot_pad_error", "")
                or "未找到可用的 2/4 踏板虚拟键盘。"
            )
        else:
            status = "disabled"
            status_label = "未启用"
            status_message = "当前配置未请求 foot-pad 控制。"
        return {
            "foot_pad_requested": requested,
            "foot_pad_available": online,
            "foot_pad_count": count,
            "foot_pad_status": status,
            "foot_pad_status_label": status_label,
            "foot_pad_status_message": status_message,
            "foot_pad_test_active": bool(self._foot_pad_test_active),
            "foot_pad_test_last_event": self._foot_pad_test_last_event,
            "foot_pad_test_status": self._foot_pad_test_status,
        }

    def _active_controls(self) -> list[str]:
        recording = bool(getattr(self.recorder, "recording", False))
        plus_save = bool(
            self.plus_curation.enabled
            and self.plus_curation.pending_episode_index is not None
            and self.plus_curation.status not in {"saving"}
        )
        if self._calibration_expired:
            controls = ["q", "umi_recalibrate", "angle_recalibrate"]
            if plus_save:
                controls.append("plus_curation_save")
            return controls

        controls: list[str] = []
        if not recording:
            if not self._startup_collection_blockers():
                controls.append("episode_start")
            controls.extend(["episode_delete", "device_retry"])
        else:
            controls.append("episode_stop")
        controls.append("q")

        # Foot pads are an additive source for the same label events.  They do
        # not own or hide the equivalent keyboard/GUI controls.
        controls.extend(
            [
                "c",
                "g",
                "fix_arm_toggle",
                "init_arm_toggle",
                "fix_arm_select",
                "init_arm_select",
            ]
        )
        if not recording:
            controls.extend(["umi_recalibrate", "angle_recalibrate"])
            if self._foot_pad_online():
                controls.append(
                    "foot_pad_test_stop"
                    if self._foot_pad_test_active
                    else "foot_pad_test_start"
                )
        if plus_save:
            controls.append("plus_curation_save")
        if self._plus_curation_save_in_flight():
            controls = [control for control in controls if control != "episode_delete"]
        return controls

    def _startup_collection_blockers(self) -> tuple[str, ...]:
        ignored_cameras = set(getattr(self.config, "ignored_camera_keys", ()) or ())
        blockers: list[str] = []
        for name in sorted(getattr(self.components, "startup_errors", {}) or {}):
            if name.startswith("camera.") and name.split(".", 1)[1] in ignored_cameras:
                continue
            blockers.append(name)
        return tuple(blockers)

    def _publish_foot_pad_test_state(self, *, message: str = "") -> None:
        self.ui.publish(
            special_arm_control_mode=(
                "foot-pad" if self._foot_pad_mode_enabled() else "manual"
            ),
            active_controls=self._active_controls(),
            **self._foot_pad_test_ui_state(),
        )
        if message:
            append_message = getattr(self.ui, "append_message", None)
            if callable(append_message):
                append_message("FootPadTest", message)

    def _plus_ui_state(self) -> dict[str, object]:
        return self.plus_curation.ui_state()

    def _plus_curation_save_in_flight(self) -> bool:
        save_thread = getattr(self, "_plus_curation_save_thread", None)
        return bool(
            self.plus_curation.enabled
            and (
                self.plus_curation.status == "saving"
                or (
                    save_thread is not None
                    and save_thread.is_alive()
                )
            )
        )

    def _begin_plus_episode(self, episode_index: int, variant_id: str | None = None) -> None:
        if not self.plus_curation.enabled:
            return
        self.plus_curation.begin_episode(
            episode_index=int(episode_index),
            variant_id=variant_id if variant_id is not None else self.plus_curation.variant_id or None,
        )

    def _mark_plus_pending(self, episode_index: int) -> None:
        if not self.plus_curation.enabled:
            return
        self.plus_curation.mark_pending(
            dataset_root=self.config.dataset_root,
            episode_index=int(episode_index),
        )

    def _save_plus_curation(
        self,
        *,
        episode_index: int,
        pending_token: str,
        disposition: str,
        reason_code: str,
    ) -> None:
        if not self.plus_curation.enabled:
            return
        if self._plus_curation_save_thread is not None and self._plus_curation_save_thread.is_alive():
            return
        if not self.plus_curation.matches_pending(
            episode_index=int(episode_index), pending_token=str(pending_token)
        ):
            self.plus_curation.status = "error"
            self.plus_curation.message = "Plus 整理请求已过期：当前待整理 episode 已变化。"
            self.ui.publish(**self._plus_ui_state(), active_controls=self._active_controls())
            return
        try:
            self.plus_curation.set_review(
                disposition=str(disposition), reason_code=str(reason_code)
            )
        except ValueError as exc:
            self.plus_curation.status = "error"
            self.plus_curation.message = f"Plus 整理参数无效: {exc}"
            self.ui.publish(**self._plus_ui_state(), active_controls=self._active_controls())
            return
        request_episode_index = self.plus_curation.pending_episode_index
        request_dataset_root = self.plus_curation.pending_dataset_root
        request_pending_token = self.plus_curation.pending_token
        request_variant_id = self.plus_curation.variant_id
        self.plus_curation.status = "saving"
        self.plus_curation.message = "Plus sidecar 正在后台写入。"
        self.ui.publish(**self._plus_ui_state(), active_controls=self._active_controls())

        def write_sidecar() -> None:
            try:
                record = self.plus_curation.save(
                    disposition=str(disposition),
                    reason_code=str(reason_code),
                    pending_token=request_pending_token,
                    variant_id=request_variant_id,
                    source_episode_index=request_episode_index,
                    source_dataset_root=request_dataset_root,
                )
            except Exception as exc:  # noqa: BLE001 - overlay is non-fatal to UMI data.
                if (
                    self.plus_curation.pending_episode_index == request_episode_index
                    and self.plus_curation.pending_dataset_root == request_dataset_root
                    and self.plus_curation.pending_token == request_pending_token
                ):
                    self.plus_curation.status = "error"
                    self.plus_curation.message = f"Plus sidecar 写入失败: {type(exc).__name__}: {exc}"
                self.ui.append_message(
                    "Plus",
                    f"Plus sidecar 写入失败: {type(exc).__name__}: {exc}",
                )
            else:
                self.ui.append_message("Plus", f"已写入 Plus sidecar: {record.annotation_episode_id}")
            finally:
                self.ui.publish(**self._plus_ui_state(), active_controls=self._active_controls())

        self._plus_curation_save_thread = threading.Thread(
            target=write_sidecar,
            name="plus-episode-curation-save",
            daemon=True,
        )
        self._plus_curation_save_thread.start()

    def _start_foot_pad_test(self) -> None:
        if not self._foot_pad_online():
            self._foot_pad_test_status = (
                "脚踏测试不可用：当前没有在线的 foot-pad 输入。"
            )
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        if self._foot_pad_test_active:
            return
        if bool(getattr(self.recorder, "recording", False)):
            self._foot_pad_test_status = "脚踏测试未开启：请先停止当前 UMI 数采。"
            self._publish_foot_pad_test_state(message=self._foot_pad_test_status)
            return
        self._foot_pad_test_active = True
        self._foot_pad_test_last_event = ""
        if self._foot_pad_count() == 2:
            self._foot_pad_test_status = (
                "脚踏测试进行中：请依次踩踏板 1、2；这里只显示固定左右臂映射，"
                "不执行任何动作。归位仍使用 G。"
            )
        else:
            self._foot_pad_test_status = (
                "脚踏测试进行中：请依次踩踏板 1、2、3、4；这里只显示映射，"
                "不执行任何动作。"
            )
        self._publish_foot_pad_test_state(
            message="已开启 UMI 脚踏测试模式；踏板事件不会写入标注。"
        )

    def _stop_foot_pad_test(self) -> None:
        self._foot_pad_test_active = False
        self._foot_pad_test_status = (
            "脚踏测试已关闭；确认映射后才可使用踏板写入 UMI 标注。"
        )
        self._publish_foot_pad_test_state(message=self._foot_pad_test_status)

    def _record_foot_pad_test_event(self, event: str) -> None:
        label = _FOOT_PAD_TEST_LABELS.get(str(event))
        if label is None:
            return
        self._foot_pad_test_last_event = label
        self._foot_pad_test_status = (
            f"最近检测：{label}。测试模式仍开启；确认无误后点击“停止测试踏板”。"
        )
        self._publish_foot_pad_test_state(
            message=f"检测到 {label}；未写入 UMI 标注。"
        )

    def _calibration_validity_warning(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, object]:
        instant = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
        remaining = float((self.calibration.expires_at - instant).total_seconds())
        warning_sec = float(getattr(self.config, "calibration_warning_sec", 600.0))
        critical_sec = float(getattr(self.config, "calibration_critical_sec", 300.0))
        display_sec = max(0.0, remaining)
        minutes = int(display_sec // 60.0)
        seconds = int(display_sec % 60.0)
        countdown = f"{minutes:02d}:{seconds:02d}"
        if remaining <= 0.0 or self._calibration_expired:
            return {
                "active": True,
                "level": "expired",
                "title": "UMI 标定已过期",
                "message": "当前数采已停止。点击“重新标定左右 UMI”，完成后可直接继续数采。",
                "remaining_sec": 0.0,
                "countdown": "00:00",
            }
        if remaining <= critical_sec:
            return {
                "active": True,
                "level": "critical",
                "title": "UMI 标定即将过期",
                "message": f"仅剩 {countdown}，请尽快结束当前 episode 并重新标定。",
                "remaining_sec": remaining,
                "countdown": countdown,
            }
        if remaining <= warning_sec:
            return {
                "active": True,
                "level": "warning",
                "title": "UMI 标定有效期提醒",
                "message": f"剩余 {countdown}，请安排本轮结束后的重新标定。",
                "remaining_sec": remaining,
                "countdown": countdown,
            }
        return {
            "active": False,
            "level": "valid",
            "title": "UMI 标定有效",
            "message": f"剩余 {countdown}",
            "remaining_sec": remaining,
            "countdown": countdown,
        }

    def _events(self) -> list[tuple[str, str, float]]:
        now = time.time()
        events = [(str(event), "web_gui", now) for event in self.ui.consume_events()]
        foot_pad_state_published = False
        if self.components.foot_pad is not None:
            events.extend(
                (str(event), "foot_pad", now)
                for event in self.components.foot_pad.consume_events()
            )
        if self._foot_pad_test_active and not self._foot_pad_online():
            self._foot_pad_test_active = False
            self._foot_pad_test_status = (
                "脚踏设备已断开，测试模式已安全关闭；GUI/键盘 fix/init 保持可用。"
            )
            self._publish_foot_pad_test_state(
                message="脚踏设备已断开；仅踏板快捷输入暂停，UMI GUI/键盘标注保持可用。"
            )
            foot_pad_state_published = True
        online_now = self._foot_pad_online()
        if online_now != bool(getattr(self, "_last_foot_pad_online", False)):
            self._last_foot_pad_online = online_now
            if not foot_pad_state_published:
                self._publish_foot_pad_test_state(
                    message=(
                        "脚踏设备已重连；已恢复 UMI 踏板 fix/init 映射。"
                        if online_now
                        else "脚踏设备已断开；仅踏板快捷输入暂停，UMI GUI/键盘标注保持可用。"
                    )
                )
        return events

    def _handle_events(
        self,
        events: Iterable[tuple[str, str, float]],
        *,
        gripper_ready: bool,
        gripper_reason: str,
    ) -> None:
        event_batch = list(events)
        # Treat pedal edges sampled in the same cycle as a test start/stop as
        # diagnostics too.  This prevents a browser click racing the input
        # poll from turning one edge into a real annotation.
        foot_pad_test_batch = self._foot_pad_test_active or any(
            event in {"foot_pad_test_start", "foot_pad_test_stop"}
            for event, _source, _timestamp in event_batch
        )
        for event, source, timestamp in event_batch:
            if event == "q":
                self.should_quit = True
            elif source == "foot_pad" and (
                self._foot_pad_test_active or foot_pad_test_batch
            ):
                self._record_foot_pad_test_event(event)
            elif event == "foot_pad_test_start":
                self._start_foot_pad_test()
            elif event == "foot_pad_test_stop":
                self._stop_foot_pad_test()
            elif event == "umi_recalibrate" and not self.recorder.recording:
                self.recalibration_requested = True
                self.should_quit = True
            elif event == "angle_recalibrate" and not self.recorder.recording:
                self.angle_recalibration_requested = True
                self.should_quit = True
            elif event == "device_retry" and not self.recorder.recording:
                self.device_retry_requested = True
                self.should_quit = True
            elif event.startswith("fix_arm_select:"):
                side = event.split(":", 1)[1]
                if side in {"left", "right"}:
                    self._selected_label_side["fix"] = side
            elif event.startswith("init_arm_select:"):
                side = event.split(":", 1)[1]
                if side in {"left", "right"}:
                    self._selected_label_side["init"] = side
            elif event in {"c", "fix_arm_toggle"}:
                self.labels.toggle(
                    self._selected_label_side["fix"], "fixed_hold",
                    source=source, timestamp=timestamp,
                )
            elif event in {"g", "init_arm_toggle"}:
                self.labels.toggle(
                    self._selected_label_side["init"], "init_pose",
                    source=source, timestamp=timestamp,
                )
            elif event in {"fix_arm_left", "fix_arm_right"}:
                self.labels.toggle(
                    event.rsplit("_", 1)[1], "fixed_hold",
                    source=source, timestamp=timestamp,
                )
            elif event in {"init_arm_left", "init_arm_right"}:
                self.labels.toggle(
                    event.rsplit("_", 1)[1], "init_pose",
                    source=source, timestamp=timestamp,
                )
            elif event == "labels_reset":
                self.labels.reset(source=source, timestamp=timestamp)
            elif event.startswith("episode_delete:"):
                try:
                    episode_index = int(event.split(":", 1)[1])
                    if self._plus_curation_save_in_flight():
                        self._event_status = "error"
                        self._event_message = (
                            f"删除 episode {episode_index} 失败：Plus 整理旁路正在写入，"
                            "请等待写入完成后重试。"
                        )
                        continue
                    if self.recorder.recording:
                        self._event_status = "error"
                        self._event_message = (
                            f"删除 episode {episode_index} 失败：当前仍在录制；"
                            "请先停止当前 episode，再重新执行删除。"
                        )
                        continue
                    self._event_status = "finalizing"
                    self._event_message = f"deleting episode {episode_index} and compacting dataset"
                    self.ui.publish(
                        episode_recording=False,
                        episode_event_status="finalizing",
                        episode_event_message=self._event_message,
                        active_controls=["q"],
                    )
                    result = self.recorder.delete_episode(episode_index)
                    self._event_status = "deleted"
                    self._event_message = (
                        f"episode {episode_index} deleted; remaining={result.episode_count}, "
                        f"next={result.next_episode_index}; backup={result.backup_root}"
                    )
                    self.plus_curation.clear_pending(
                        message="episode 删除/重编号后，待整理 Plus 绑定已清除。"
                    )
                except (ValueError, IndexError, OSError, RuntimeError) as exc:
                    self._event_status = "error"
                    self._event_message = f"delete failed: {type(exc).__name__}: {exc}"
            elif event.startswith("plus_curation_save:"):
                try:
                    payload = json.loads(event.split(":", 1)[1])
                    if not isinstance(payload, Mapping):
                        raise ValueError("Plus curation payload must be an object")
                    self._save_plus_curation(
                        episode_index=int(payload.get("episode_index")),
                        pending_token=str(payload.get("pending_token", "") or "").strip(),
                        disposition=str(payload.get("disposition", "unknown")),
                        reason_code=str(payload.get("reason_code", "unknown")),
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.plus_curation.status = "error"
                    self.plus_curation.message = f"Plus 整理参数无效: {exc}"
                    self.ui.publish(**self._plus_ui_state())
            elif event == "episode_start" or event.startswith("episode_start:"):
                if not self.recorder.recording:
                    if event.startswith("episode_start:"):
                        try:
                            payload = json.loads(event.split(":", 1)[1])
                            selected_root = str(
                                payload.get("dataset_base_root", "")
                                if isinstance(payload, dict)
                                else ""
                            ).strip()
                            root = (
                                Path(os.path.expandvars(selected_root)).expanduser()
                                if selected_root
                                else self.config.dataset_root
                            )
                            ignored_cameras = payload.get(
                                "ignored_camera_keys",
                                list(self.config.ignored_camera_keys),
                            )
                            ignored_tactile = payload.get(
                                "ignored_tactile_keys",
                                list(self.config.ignored_tactile_keys),
                            )
                            if not isinstance(ignored_cameras, list) or not isinstance(
                                ignored_tactile, list
                            ):
                                raise ValueError(
                                    "ignored camera/tactile settings must be lists"
                                )
                            next_config = replace(
                                self.config,
                                dataset_root=root,
                                ignored_camera_keys=tuple(ignored_cameras),
                                ignored_tactile_keys=tuple(ignored_tactile),
                            )
                            reconfigure = getattr(
                                self.recorder, "reconfigure_config", None
                            )
                            if callable(reconfigure):
                                reconfigure(next_config)
                            else:
                                self.recorder.reconfigure_dataset_root(root)
                            self.config = next_config
                            self.snapshot_assembler.config = next_config
                            if self.plus_curation.enabled and "plus_variant_id" in payload:
                                variant_id = str(payload.get("plus_variant_id", "") or "").strip()
                                self.plus_curation.set_variant(variant_id)
                        except (OSError, TypeError, ValueError, RuntimeError) as exc:
                            self._event_status = "error"
                            self._event_message = (
                                "切换 UMI 数据集路径失败："
                                f"{type(exc).__name__}: {exc}"
                            )
                            continue
                    if not gripper_ready:
                        self.last_rejection = (gripper_reason or "gripper_input_not_ready",)
                        self._event_status = "ready"
                        self._event_message = (
                            "cannot start: " + ", ".join(self.last_rejection)
                        )
                        continue
                    self.calibration.require_valid(self.config)
                    readiness = self.snapshot_assembler.assemble(
                        episode_index=-1,
                        frame_index=0,
                        timestamp=time.time(),
                    )
                    if isinstance(readiness, SnapshotRejection):
                        self.last_rejection = readiness.reasons
                        self._event_status = "ready"
                        self._event_message = (
                            "cannot start: " + ", ".join(readiness.reasons)
                        )
                        continue
                    self.labels.reset(source="episode_start", timestamp=timestamp)
                    episode_index = self.recorder.start_episode(
                        {
                            "embodiment": "umi_dual_wrist",
                            "embodiment_has_arm": False,
                            "calibration_sha256": self.calibration.sha256,
                            "tcp_extrinsic_by_side": {
                                side: {
                                    "T_tcp_to_tracker": self.calibration.T_tcp_to_tracker_by_side[
                                        side
                                    ].tolist(),
                                    "status": self.config.side(side).tcp_extrinsic_status,
                                    "tcp_profile": self.config.side(side).tcp_profile,
                                    "hand_eye_result_path": (
                                        None
                                        if side not in self.calibration.by_side
                                        else str(self.calibration.by_side[side].path)
                                    ),
                                    "hand_eye_result_sha256": (
                                        None
                                        if side not in self.calibration.by_side
                                        else self.calibration.by_side[side].sha256
                                    ),
                                }
                                for side in ("left", "right")
                            },
                        }
                    )
                    self._begin_plus_episode(int(episode_index))
                    self._next_sample_at = time.perf_counter()
                    self.last_rejection = ()
                    self._event_status = "recording"
                    self._event_message = ""
            elif event in {"episode_stop", "episode_cancel"}:
                if self.recorder.recording:
                    self._event_status = "finalizing"
                    self._event_message = "finalizing episode"
                    self.ui.publish(
                        episode_recording=False,
                        episode_event_status="finalizing",
                        episode_event_message=self._event_message,
                        active_controls=["q"],
                    )
                    stopped_episode_index = self.recorder.stop_episode(
                        reason=event,
                        discard=event == "episode_cancel",
                    )
                    if (
                        stopped_episode_index is not None
                        and str(self.recorder.last_outcome) == "saved"
                    ):
                        self._mark_plus_pending(int(stopped_episode_index))
                    self._event_status = self.recorder.last_outcome
                    self._event_message = self.recorder.last_message
                    self.labels.reset(source=event, timestamp=timestamp)

    def _update_grippers(self, now: float) -> tuple[bool, str]:
        channels = self.components.angle_trigger.get_latest_channels()
        required = {f"{side}_angle_trigger" for side in ("left", "right")}
        if not required.issubset(channels):
            return False, "missing_angle_trigger"
        for name in required:
            channel = channels[name]
            age = now - float(channel.timestamp)
            if (
                not channel.valid
                or age < -1e-3
                or age > self.config.timing.angle_trigger_max_age_sec
            ):
                return False, f"invalid_or_stale_angle_trigger:{name}:{age:.3f}"
        observation = TeleopObservation(channels=channels, timestamp=now)
        merged = merge_teleop_commands(
            [
                self.components.retargeters[side].retarget(observation)
                for side in ("left", "right")
            ]
        )
        if set(merged.commands) != {"left_hand", "right_hand"}:
            return False, "incomplete_gripper_command"
        pending: dict[str, float] = {}
        for side in ("left", "right"):
            scalar = float(
                merged.commands[f"{side}_hand"].metadata["gripper_joint_target"]
            )
            if not np.isfinite(scalar):
                return False, f"nonfinite_gripper_command:{side}"
            pending[side] = scalar
        self.components.rig.send_commands(merged)
        self.last_commands.update(pending)
        return True, ""

    def _sample_if_due(self, wall_now: float) -> None:
        monotonic_now = time.perf_counter()
        if monotonic_now < self._next_sample_at:
            return
        period = 1.0 / self.config.timing.sample_hz
        # Never emit a catch-up burst after idle time or a slow/rejected frame.
        self._next_sample_at = monotonic_now + period
        if not self.recorder.recording:
            return
        assert self.recorder.episode_index is not None
        result = self.snapshot_assembler.assemble(
            episode_index=self.recorder.episode_index,
            frame_index=int(
                getattr(self.recorder, "next_frame_index", self.recorder.frame_count)
            ),
            timestamp=wall_now,
        )
        if isinstance(result, SnapshotRejection):
            note_attempt = getattr(self.recorder, "note_snapshot_attempt", None)
            if callable(note_attempt):
                note_attempt(result.reasons)
            break_continuity = getattr(
                self.recorder, "break_action_continuity", None
            )
            if callable(break_continuity):
                break_continuity(
                    reason="snapshot_rejected:" + ",".join(result.reasons)
                )
            self.last_rejection = result.reasons
            self._event_message = "sample rejected: " + ", ".join(result.reasons)
            return
        note_attempt = getattr(self.recorder, "note_snapshot_attempt", None)
        if callable(note_attempt):
            note_attempt()
        self.last_rejection = ()
        self._event_message = ""
        accepted = self.recorder.submit(result)
        if accepted is False:
            rejection_message = str(
                getattr(self.recorder, "last_message", "sample rejected")
                or "sample rejected"
            )
            self.last_rejection = (rejection_message,)
            self._event_message = rejection_message

    def _publish(self, now: float, *, gripper_ready: bool, gripper_reason: str) -> None:
        remaining = (
            self.calibration.expires_at - datetime.now(timezone.utc)
        ).total_seconds()
        label_state = self.labels.snapshot()
        left_label = label_state.labels_by_side["left"]
        right_label = label_state.labels_by_side["right"]
        vive_channels = self.components.vive.get_latest_channels_at_or_before(now)
        gui_vive_channels = {
            f"{side}_wrist_pose": vive_channels.get(
                self.config.side(side).tracker_channel
            )
            for side in ("left", "right")
        }
        hand_states = self.components.hand_state_poller.latest_at_or_before(now)
        angle_channels = self.components.angle_trigger.get_latest_channels()
        actual_sample_hz = self.recorder.effective_sample_hz(now=now)
        low_frequency = bool(
            self.recorder.recording
            and self.recorder.frame_count > 0
            and actual_sample_hz is not None
            and actual_sample_hz < 0.8 * self.config.timing.sample_hz
        )
        calibration_warning = self._calibration_validity_warning()
        selected_fix_side = self._selected_label_side["fix"]
        selected_init_side = self._selected_label_side["init"]
        event_status = (
            "paused"
            if self.recorder.recording and (gripper_reason or self.last_rejection)
            else "recording"
            if self.recorder.recording
            else self._event_status
        )
        event_message = (
            gripper_reason
            or self._event_message
            or self.recorder.last_message
        )
        recorder_writer = getattr(self.recorder, "writer", None)
        dataset_root = Path(
            getattr(recorder_writer, "root_dir", self.config.dataset_root)
        )
        transmit_gripper_commands = bool(self.config.send_gripper_commands)
        self.ui.publish(
            ui_mode="umi_collection",
            stage=(
                "UMI 标定已过期"
                if self._calibration_expired
                else "UMI 数采中"
                if self.recorder.recording
                else "UMI 就绪"
            ),
            control_hint=(
                gripper_reason
                or (
                    "脚踏设备未连接；仅踏板快捷输入不可用，GUI/键盘 fix/init 保持可用。"
                    if getattr(self.components, "foot_pad_error", "")
                    and not self._foot_pad_online()
                    else ""
                )
            ),
            teleop_running=True,
            home_ready=gripper_ready and not self._calibration_expired,
            episode_recording_available=(
                not self._calibration_expired
                and not self._startup_collection_blockers()
            ),
            episode_recording=self.recorder.recording,
            episode_dir=str(dataset_root),
            episode_frame_count=self.recorder.frame_count,
            episode_attempted_sample_count=getattr(
                self.recorder, "sample_attempt_count", None
            ),
            episode_rejection_counts=dict(
                getattr(self.recorder, "rejection_counts", {}) or {}
            ),
            episode_sample_hz=actual_sample_hz,
            episode_target_hz=self.config.timing.sample_hz,
            episode_frequency_warning={
                "active": low_frequency,
                "message": (
                    f"UMI accepted sample rate {actual_sample_hz:.2f} Hz is below "
                    f"target {self.config.timing.sample_hz:.2f} Hz"
                    if low_frequency and actual_sample_hz is not None
                    else ""
                ),
            },
            calibration_validity_warning=calibration_warning,
            episode_event_status=event_status,
            episode_event_message=event_message,
            episode_last_episode_index=(
                self.recorder.episode_index
                if self.recorder.recording
                else self.recorder.last_episode_index
            ),
            episode_last_frame_count=self.recorder.last_frame_count,
            dataset_base_root=str(dataset_root),
            dataset_root=str(dataset_root),
            dataset_modality_contract=self.config.capture_modality_contract.to_dict(),
            hardware_status={
                name: {
                    "online": name not in (getattr(self.components, "startup_errors", {}) or {}),
                    "health": (
                        "online" if name not in (getattr(self.components, "startup_errors", {}) or {}) else "error"
                    ),
                    "status_level": (
                        "ok" if name not in (getattr(self.components, "startup_errors", {}) or {}) else "error"
                    ),
                    "status_label": (
                        "在线" if name not in (getattr(self.components, "startup_errors", {}) or {}) else "连接失败"
                    ),
                    "requirement": (
                        "collection_core"
                        if name in {"vive", "angle_trigger", "dmtac.rig", "dmtac.state_poller"}
                        else "collection"
                    ),
                    "error": (getattr(self.components, "startup_errors", {}) or {}).get(name, ""),
                }
                for name in (
                    "vive",
                    "angle_trigger",
                    "dmtac.rig",
                    "dmtac.state_poller",
                    *(f"camera.{key}" for key in sorted(self.components.cameras)),
                )
            },
            collection_task_id=self.config.task_id,
            collection_task_name=self.config.task_description,
            vive_calibration_profile_ready=remaining >= 0.0 and not self._calibration_expired,
            vive_calibration_profile_loaded=True,
            vive_calibration_profile_path=str(self.calibration.path),
            vive_calibration_profile_status=(
                f"expires={self.calibration.expires_at.isoformat()} "
                f"remaining={max(0.0, remaining):.1f}s hash={self.calibration.sha256}"
            ),
            raw_tracker_poses=raw_tracker_pose_state_from_channels(
                gui_vive_channels,
                active_sides=("left", "right"),
                now=now,
            ),
            camera_status={
                key: {
                    **camera.device_info(),
                    "requirement": (
                        "ignored_by_dataset_contract"
                        if key in self.config.ignored_camera_keys
                        else "required_for_collection"
                    ),
                    "ignored_by_dataset_contract": (
                        key in self.config.ignored_camera_keys
                    ),
                }
                for key, camera in self.components.cameras.items()
            },
            gripper_control_available=True,
            gripper_control_enabled=gripper_ready and transmit_gripper_commands,
            gripper_status={
                side: {
                    "configured": True,
                    "is_connected": self.components.grippers[side].is_ready(),
                    "input_source": "angle_trigger",
                    "q6_control_available": True,
                    "q6_valid": bool(
                        angle_channels.get(f"{side}_angle_trigger")
                        and angle_channels[f"{side}_angle_trigger"].valid
                    ),
                    "q6_age_sec": (
                        None
                        if angle_channels.get(f"{side}_angle_trigger") is None
                        else now - angle_channels[f"{side}_angle_trigger"].timestamp
                    ),
                    "q6_source_dim": None,
                    "q6_source_value": (
                        None
                        if angle_channels.get(f"{side}_angle_trigger") is None
                        else float(angle_channels[f"{side}_angle_trigger"].value)
                    ),
                    "q6_target_preview": self.last_commands[side],
                    "q6_values": [],
                    "q6_command": {
                        "status": (
                            "sent"
                            if gripper_ready and transmit_gripper_commands
                            else "control_disabled"
                            if gripper_ready
                            else "blocked"
                        ),
                        "sent": gripper_ready and transmit_gripper_commands,
                        "target": self.last_commands[side],
                        "error": (
                            gripper_reason
                            or ("send_gripper_commands=false" if not transmit_gripper_commands else "")
                        ),
                    },
                    "commanded": self.last_commands[side],
                    "measured": hand_states.get(side, {}).get("gripper_measured_scalar"),
                    "state": hand_states.get(side, {}).get("gripper_state_scalar"),
                    "max_action": 1.0,
                    "speed": self.components.grippers[side].config.speed,
                    "torque": self.components.grippers[side].config.torque,
                    "tactile_devices": list(
                        hand_states.get(side, {}).get("tactile_devices", ())
                    ),
                    "tactile_read_hz": float(
                        getattr(
                            self.components.grippers[side].config,
                            "tactile_read_hz",
                            self.config.dmtac_tactile_read_hz,
                        )
                    ),
                    "tactile_snapshot_age_sec": hand_states.get(side, {}).get(
                        "tactile_snapshot_age_sec"
                    ),
                    "telemetry_error": hand_states.get(side, {}).get("telemetry_error"),
                }
                for side in ("left", "right")
            },
            tactile_camera_status={
                side: {
                    "timestamp": state.get("tactile_timestamp"),
                    "is_connected": self.components.grippers[side].is_ready(),
                    "connect_tactile": bool(
                        self.components.grippers[side].config.connect_tactile
                    ),
                    "tactile_devices": list(state.get("tactile_devices", ())),
                    "tactile_age_sec": state.get("tactile_snapshot_age_sec"),
                    "tactile_version": state.get("tactile_version"),
                    "tactile_frame_ids": state.get("tactile_frame_ids", {}),
                    "tactile_shapes": state.get("tactile_shapes", {}),
                    "tactile_read_elapsed_s": state.get("tactile_read_elapsed_s"),
                    "tactile_gate_wait_s": state.get("tactile_gate_wait_s"),
                    "tactile_read_skipped": state.get("tactile_read_skipped"),
                    "tactile_worker_alive": state.get("tactile_worker_alive"),
                    "tactile_worker_receiver_alive": state.get(
                        "tactile_worker_receiver_alive"
                    ),
                    "tactile_process_read": bool(
                        self.components.grippers[side].config.tactile_process_read
                    ),
                    "tactile_read_hz": float(
                        self.components.grippers[side].config.tactile_read_hz
                    ),
                    "tactile_max_fps": int(
                        self.components.grippers[side].config.tactile_max_fps
                    ),
                    "device_count": len(state.get("tactile_devices", ())),
                    "ready_count": len(state.get("tactile_devices", ())),
                    "requirement": (
                        "ignored_by_dataset_contract"
                        if side in self.config.ignored_tactile_keys
                        else "required_for_collection"
                    ),
                    "ignored_by_dataset_contract": (
                        side in self.config.ignored_tactile_keys
                    ),
                }
                for side, state in hand_states.items()
            },
            visualization_status=(
                f"last_rejection={list(self.last_rejection)} "
                f"writer={self.recorder.stats()}"
            ),
            fix_arm_available=True,
            fix_arm_selected_side=selected_fix_side,
            fix_arm_fixed_side=(
                selected_fix_side
                if label_state.labels_by_side[selected_fix_side] == "fixed_hold"
                else ""
            ),
            fix_arm_active=(
                label_state.labels_by_side[selected_fix_side] == "fixed_hold"
            ),
            fix_arm_status=f"UMI annotation only: left={left_label}, right={right_label}",
            init_arm_available=True,
            init_arm_selected_side=selected_init_side,
            init_arm_active_side=(
                selected_init_side
                if label_state.labels_by_side[selected_init_side] == "init_pose"
                else ""
            ),
            init_arm_active=(
                label_state.labels_by_side[selected_init_side] == "init_pose"
            ),
            init_arm_status=f"UMI annotation only: left={left_label}, right={right_label}",
            special_arm_control_mode=(
                "foot-pad" if self._foot_pad_mode_enabled() else "manual"
            ),
            **self._foot_pad_test_ui_state(),
            **self._plus_ui_state(),
            active_controls=self._active_controls(),
        )

    def run(self) -> str:
        period = 1.0 / self.config.timing.control_hz
        last_publish = 0.0
        while not self.should_quit:
            started = time.perf_counter()
            wall_now = time.time()
            events = self._events()
            if any(event == "q" for event, _source, _timestamp in events):
                self.should_quit = True
                break
            if any(
                event == "umi_recalibrate"
                for event, _source, _timestamp in events
            ) and not self.recorder.recording:
                self.recalibration_requested = True
                self.should_quit = True
                break
            if any(
                event == "angle_recalibrate"
                for event, _source, _timestamp in events
            ) and not self.recorder.recording:
                self.angle_recalibration_requested = True
                self.should_quit = True
                break
            if not self._calibration_expired:
                try:
                    self.calibration.require_valid(self.config)
                except RuntimeError as exc:
                    if self.recorder.recording:
                        stopped_episode_index = self.recorder.stop_episode(reason="calibration_expired")
                        if (
                            stopped_episode_index is not None
                            and str(self.recorder.last_outcome) == "saved"
                        ):
                            self._mark_plus_pending(int(stopped_episode_index))
                    self._calibration_expired = True
                    self._event_status = "error"
                    self._event_message = (
                        "UMI 标定已过期；当前 episode 已安全结束。"
                        "请点击界面中的“重新标定左右 UMI”。"
                    )
                    self.last_rejection = ("calibration_expired", str(exc))
            if self._calibration_expired:
                if wall_now - last_publish >= 0.2:
                    self._publish(
                        wall_now,
                        gripper_ready=False,
                        gripper_reason=self._event_message,
                    )
                    last_publish = wall_now
                remaining = period - (time.perf_counter() - started)
                if remaining > 0.0:
                    time.sleep(remaining)
                continue
            gripper_ready, gripper_reason = self._update_grippers(wall_now)
            self._handle_events(
                events,
                gripper_ready=gripper_ready,
                gripper_reason=gripper_reason,
            )
            if self.should_quit:
                # Recalibration/device-retry events transfer ownership back to
                # the outer lifecycle. Do not sample or republish ordinary
                # controls after accepting that one-shot transition.
                break
            # T0 is captured after command generation/event handling so the
            # row never claims a command became active before this process
            # computed and dispatched it.
            sample_now = time.time()
            if gripper_ready:
                self._sample_if_due(sample_now)
            elif self.recorder.recording:
                break_continuity = getattr(
                    self.recorder, "break_action_continuity", None
                )
                if callable(break_continuity):
                    break_continuity(reason=f"gripper_not_ready:{gripper_reason}")
                self.last_rejection = (gripper_reason,)
            if sample_now - last_publish >= 0.2:
                self._publish(
                    sample_now,
                    gripper_ready=gripper_ready,
                    gripper_reason=gripper_reason,
                )
                last_publish = sample_now
            remaining = period - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
        if self.angle_recalibration_requested:
            return "recalibrate_angle"
        if self.device_retry_requested:
            return "retry_devices"
        return "recalibrate" if self.recalibration_requested else "quit"


__all__ = ["UmiCollectionController"]
