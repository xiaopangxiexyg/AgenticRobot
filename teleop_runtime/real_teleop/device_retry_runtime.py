"""Runtime hardware status projection and explicit device retry routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceRetryRuntime:
    hardware_requirement: Callable[[str], str]
    hardware_status_item: Callable[[str], dict[str, object]]
    collection_startup_blockers: Callable[[], dict[str, str]]
    publish_runtime_hardware_status: Callable[[], None]
    sync_foot_pad_runtime_status: Callable[[], None]
    retry_failed_devices: Callable[[], None]


def build_device_retry_runtime(
    *,
    args: Any,
    components: Any,
    control_ui: Any,
    clutch_input: Any,
    camera_sources: Mapping[str, Any],
    vive_source: Any,
    startup_device_errors: dict[str, str],
    position_force_guard_requested: bool,
    dataset_recorder_provider: Callable[[], Any],
    set_dataset_recorder: Callable[[Any], None],
    foot_pad_input_provider: Callable[[], Any],
    set_foot_pad_input: Callable[[Any], None],
    dataset_recorder_type: type,
    dmtac_gripper_backend_type: type,
    foot_pad_input_type: type,
    foot_pad_control_mode: str,
    has_dmtac_gripper_backend: Callable[[Any], bool],
    promote_angle_trigger_gripper_input_to_q6: Callable[..., tuple[bool, str]],
    log: Callable[..., None],
) -> DeviceRetryRuntime:
    RealTeleopDatasetRecorder = dataset_recorder_type
    DMTacGripperBackend = dmtac_gripper_backend_type
    FootPadSpecialArmInput = foot_pad_input_type
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD = foot_pad_control_mode
    _has_dmtac_gripper_backend = has_dmtac_gripper_backend
    _promote_angle_trigger_gripper_input_to_q6 = (
        promote_angle_trigger_gripper_input_to_q6
    )
    _log = log
    def hardware_requirement(key: str) -> str:
        if key.startswith("arm."):
            return "motion_safety"
        if key.startswith("camera."):
            camera_key = key.split(".", 1)[1]
            if camera_key in set(getattr(args, "ignore_camera", ()) or ()):
                return "ignored_by_dataset_contract"
            return "required_for_collection"
        if key.startswith("dataset."):
            return "required_for_collection"
        return "feature_dependent"


    def hardware_status_item(key: str) -> dict[str, object]:
        detail = str(startup_device_errors.get(key, "") or "")
        if not detail:
            return {
                "online": True,
                "health": "online",
                "status_level": "ok",
                "status_label": "在线",
                "requirement": hardware_requirement(key),
                "error": "",
            }
        if key == "foot_pad":
            return {
                "online": False,
                "health": "waiting",
                "status_level": "warn",
                "status_label": "未就绪（可选）",
                "requirement": "optional_input",
                "error": detail,
            }
        return {
            "online": False,
            "health": "error",
            "status_level": "error",
            "status_label": "连接失败",
            "requirement": hardware_requirement(key),
            "error": detail,
        }


    def collection_startup_blockers() -> dict[str, str]:
        ignored_cameras = set(getattr(args, "ignore_camera", ()) or ())
        blockers: dict[str, str] = {}
        for key, detail in startup_device_errors.items():
            # Foot-pad is an operator convenience and never a data modality.
            if key == "foot_pad":
                continue
            # The writer is rebuilt/revalidated on episode start.  Its
            # current startup error must not prevent that recovery attempt.
            if key == "dataset.writer":
                continue
            if key.startswith("camera.") and key.split(".", 1)[1] in ignored_cameras:
                continue
            blockers[str(key)] = str(detail)
        return blockers


    def publish_runtime_hardware_status() -> None:
        if control_ui is None:
            return
        status_keys = {
            "arm.left.qpos",
            "arm.right.qpos",
            *(f"camera.{key}" for key in camera_sources),
            *startup_device_errors,
        }
        if components.left_hand_backend is not None:
            status_keys.add("hand.left")
        if components.right_hand_backend is not None:
            status_keys.add("hand.right")
        status_keys.add("dataset.writer")
        if position_force_guard_requested:
            status_keys.add("arm.force_guard")
        if bool(getattr(args, "foot_pad_requested", False)):
            status_keys.add("foot_pad")
        control_ui.publish(
            hardware_status={
                key: hardware_status_item(key)
                for key in sorted(status_keys)
            },
            active_controls=clutch_input._current_active_controls(),
        )


    def sync_foot_pad_runtime_status() -> None:
        """Mirror manual foot-pad reconnect status into the GUI hardware list."""

        foot_pad_input = foot_pad_input_provider()
        if not bool(getattr(args, "foot_pad_requested", False)):
            return
        online = bool(
            foot_pad_input is not None
            and getattr(foot_pad_input, "is_online", False)
        )
        changed = False
        if online:
            changed = "foot_pad" in startup_device_errors
            startup_device_errors.pop("foot_pad", None)
            args.foot_pad_startup_error = ""
        else:
            detail = str(
                getattr(args, "foot_pad_startup_error", "") or (
                    "FootPadUnavailableError: virtual keyboard disconnected; "
                    "press 'retry failed devices' to scan again"
                )
            )
            if startup_device_errors.get("foot_pad") != detail:
                startup_device_errors["foot_pad"] = detail
                changed = True
            args.foot_pad_startup_error = detail
        if changed:
            publish_runtime_hardware_status()


    def retry_failed_devices() -> None:
        dataset_recorder = dataset_recorder_provider()
        foot_pad_input = foot_pad_input_provider()
        recorder_busy = bool(
            dataset_recorder is not None
            and (
                bool(getattr(dataset_recorder, "is_recording", False))
                or bool(getattr(dataset_recorder, "finalizing", False))
            )
        )
        if recorder_busy:
            rejection_message = (
                "设备重连未执行：正式 dataset recorder 仍在录制或后台保存。"
            )
            _log("Device", rejection_message)
            if control_ui is not None:
                control_ui.append_message("Device", rejection_message)
                control_ui.publish(
                    active_controls=clutch_input._current_active_controls(),
                    control_hint=rejection_message,
                )
            publish_runtime_hardware_status()
            return
        admitted, rejection_message = clutch_input.device_retry_admission()
        if not admitted:
            _log("Device", rejection_message)
            if control_ui is not None:
                control_ui.append_message("Device", rejection_message)
                control_ui.publish(
                    active_controls=clutch_input._current_active_controls(),
                    control_hint=rejection_message,
                )
            publish_runtime_hardware_status()
            return
        q6_retry_handled = False
        if (
            str(getattr(components, "gripper_input_source", "") or "")
            == "angle_trigger"
            and _has_dmtac_gripper_backend(components)
        ):
            q6_retry_handled = True
            teleop_running = bool(
                getattr(clutch_input, "_teleop_started", False)
            ) and not clutch_input.hold_active()
            if (
                clutch_input._episode_busy()
                or teleop_running
            ):
                q6_message = (
                    "Q6 热切换只允许在未数采且遥操尚未开始或已进入 Hold 时执行；"
                    "请先停止 episode 并点击 Hold。"
                )
            else:
                try:
                    promoted, q6_message = (
                        _promote_angle_trigger_gripper_input_to_q6(
                            args,
                            components,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - preserve working fallback.
                    promoted = False
                    q6_message = (
                        "Q6 重连/切换失败，继续使用 angle trigger："
                        f"{type(exc).__name__}: {exc}"
                    )
                if promoted:
                    clutch_input._gripper_control_enabled = False
                    startup_device_errors.pop("angle_trigger", None)
                    args.angle_trigger_startup_error = ""
                    clutch_input.publish_gripper_status(force=True)
            _log("GripperInput", q6_message)
            if control_ui is not None:
                control_ui.append_message("GripperInput", q6_message)
        if not startup_device_errors:
            if control_ui is not None and not q6_retry_handled:
                control_ui.append_message("Device", "当前没有启动失败的可重连设备。")
            publish_runtime_hardware_status()
            return
        retry_backends = {
            "arm.left.qpos": components.left_state_backend,
            "arm.right.qpos": components.right_state_backend,
            "hand.left": components.left_hand_backend,
            "hand.right": components.right_hand_backend,
        }
        for key in tuple(startup_device_errors):
            try:
                if key in retry_backends and retry_backends[key] is not None:
                    retry_backends[key].start()
                    if (
                        key.startswith("hand.")
                        and isinstance(retry_backends[key], DMTacGripperBackend)
                        and not retry_backends[key].is_ready()
                    ):
                        raise RuntimeError("DMTAC backend is still not ready")
                    if key.startswith("arm."):
                        state = retry_backends[key].refresh_state_from_hardware()
                        if state.get("current_qpos") is None:
                            raise RuntimeError("hardware returned no current_qpos")
                elif key.startswith("camera.") and key.split(".", 1)[1] in camera_sources:
                    camera = camera_sources[key.split(".", 1)[1]]
                    camera.close()
                    camera.start()
                elif key == "dataset.writer":
                    if dataset_recorder is None:
                        dataset_recorder = RealTeleopDatasetRecorder(
                            args=args,
                            components=components,
                            camera_sources=camera_sources,
                        )
                        set_dataset_recorder(dataset_recorder)
                    dataset_recorder.start()
                    dataset_recorder.write_camera_device_config()
                elif key == "rig.session":
                    components.session.start()
                    if components.human_shadow_frontend is not None:
                        components.human_shadow_frontend.start()
                    if (
                        vive_source is not None
                        and vive_source.clock_sync_enabled
                        and bool(args.vive_clock_sync_required)
                        and not vive_source.wait_for_clock_sync(
                            float(args.vive_clock_sync_startup_timeout_sec),
                            min_samples=int(args.vive_clock_sync_min_samples),
                        )
                    ):
                        raise RuntimeError(
                            "core rig reconnected but required VIVE clock sync is not ready"
                        )
                elif key == "vive.clock_sync" and vive_source is not None:
                    sync_ready = vive_source.wait_for_clock_sync(
                        float(args.vive_clock_sync_startup_timeout_sec),
                        min_samples=int(args.vive_clock_sync_min_samples),
                    )
                    if not sync_ready:
                        raise RuntimeError("required VIVE clock sync is still not ready")
                elif key == "foot_pad":
                    replacement = FootPadSpecialArmInput.open_online()
                    if foot_pad_input is not None:
                        foot_pad_input.close()
                    # Keep the runtime provider and the control-input object
                    # on the same live adapter.  The provider is also used by
                    # the periodic status synchronizer; updating only
                    # ``clutch_input.foot_pad_input`` leaves the provider
                    # pointing at the old offline placeholder and makes a
                    # successful manual reconnect appear disconnected again
                    # on the next status tick.
                    set_foot_pad_input(replacement)
                    foot_pad_input = replacement
                    clutch_input.foot_pad_input = replacement
                    clutch_input._last_foot_pad_online = True
                    args.special_arm_control_mode = SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
                    args.foot_pad_startup_error = ""
                elif key in {
                    "dmtac.assignment",
                    "camera.config",
                    "angle_trigger",
                }:
                    if control_ui is not None:
                        control_ui.append_message(
                            "Device",
                            f"{key} 涉及组件重建，当前进程不能安全热重连；"
                            "请修复配置后重启运行命令。",
                        )
                    continue
                else:
                    continue
            except Exception as exc:  # noqa: BLE001 - keep the failed status visible.
                startup_device_errors[key] = f"{type(exc).__name__}: {exc}"
                continue
            startup_device_errors.pop(key, None)
            if control_ui is not None:
                control_ui.append_message("Device", f"{key} 重连成功。")
        publish_runtime_hardware_status()

    return DeviceRetryRuntime(
        hardware_requirement=hardware_requirement,
        hardware_status_item=hardware_status_item,
        collection_startup_blockers=collection_startup_blockers,
        publish_runtime_hardware_status=publish_runtime_hardware_status,
        sync_foot_pad_runtime_status=sync_foot_pad_runtime_status,
        retry_failed_devices=retry_failed_devices,
    )


__all__ = ["DeviceRetryRuntime", "build_device_retry_runtime"]
