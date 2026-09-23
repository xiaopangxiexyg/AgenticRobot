"""Runtime hardware status projection and explicit device retry routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any

from robot_motion.runtime.readiness import (
    format_readiness_report,
    readiness_report_blockers,
)
from teleop_runtime.sources.camera import (
    RealSenseRGBSource,
    start_real_sense_source_with_containment,
)

from .foot_pad_controls import (
    apply_live_foot_pad_topology,
    foot_pad_progress_topology_guidance,
)


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
    publish_camera_status: Callable[[], None] | None = None,
    refresh_application_status: Callable[[], None] | None = None,
) -> DeviceRetryRuntime:
    DMTacGripperBackend = dmtac_gripper_backend_type
    FootPadSpecialArmInput = foot_pad_input_type
    del foot_pad_control_mode
    _has_dmtac_gripper_backend = has_dmtac_gripper_backend
    _promote_angle_trigger_gripper_input_to_q6 = (
        promote_angle_trigger_gripper_input_to_q6
    )
    _log = log
    dynamic_readiness_keys: set[str] = set()
    # ``Rig.readiness_report()`` may ask a vendor backend for a bounded state
    # snapshot.  It is an observational status path, not a control-loop
    # operation; keep it out of repeated GUI/status projections while still
    # refreshing promptly after an explicit reconnect/Home boundary.
    readiness_cache_ttl_sec = 0.25
    readiness_cache_report: Mapping[str, object] | None = None
    readiness_cache_at = 0.0
    readiness_cache_key: tuple[object, ...] | None = None
    readiness_cache_error: str | None = None

    def _readiness_context_key() -> tuple[object, ...]:
        session = getattr(components, "session", None)
        rig = getattr(session, "rig", None)
        raw_state = getattr(getattr(session, "state_machine", None), "state", None)
        state = str(getattr(raw_state, "value", raw_state) or "").upper()
        backends = getattr(rig, "backends", None)
        if isinstance(backends, Mapping):
            backend_key = tuple(
                sorted((str(name), id(backend)) for name, backend in backends.items())
            )
        else:
            backend_key = ()
        return (
            id(session),
            id(rig),
            bool(getattr(session, "_started", False)),
            state,
            backend_key,
        )

    def _invalidate_readiness_cache() -> None:
        nonlocal readiness_cache_report, readiness_cache_at
        nonlocal readiness_cache_key, readiness_cache_error
        readiness_cache_report = None
        readiness_cache_at = 0.0
        readiness_cache_key = None
        readiness_cache_error = None

    def _rig_readiness_report(*, force: bool = False) -> Mapping[str, object] | None:
        nonlocal readiness_cache_report, readiness_cache_at
        nonlocal readiness_cache_key, readiness_cache_error
        context_key = _readiness_context_key()
        now = time.monotonic()
        if (
            not force
            and readiness_cache_key == context_key
            and now - readiness_cache_at < readiness_cache_ttl_sec
        ):
            return readiness_cache_report
        session = getattr(components, "session", None)
        rig = getattr(session, "rig", None)
        report_fn = getattr(rig, "readiness_report", None)
        if not callable(report_fn):
            _invalidate_readiness_cache()
            return None
        try:
            report = report_fn()
        except Exception as exc:  # noqa: BLE001 - status must remain available.
            error = f"{type(exc).__name__}: {exc}"
            # Cache an observational failure briefly as well.  A disconnected
            # vendor handle must not turn every hardware-status item into a
            # serial retry storm; explicit retry invalidates this latch.
            readiness_cache_report = None
            readiness_cache_at = now
            readiness_cache_key = context_key
            if readiness_cache_error != error:
                _log(
                    "Device",
                    "Rig readiness report failed; preserving existing device status: "
                    f"{error}",
                )
                readiness_cache_error = error
            return None
        if not isinstance(report, Mapping):
            readiness_cache_report = None
            readiness_cache_at = now
            readiness_cache_key = context_key
            readiness_cache_error = "readiness report returned a non-mapping value"
            return None
        readiness_cache_report = dict(report)
        readiness_cache_at = now
        readiness_cache_key = context_key
        readiness_cache_error = None
        return readiness_cache_report

    def _readiness_key(effector: str) -> str:
        if effector in {"left_arm", "right_arm"}:
            return f"arm.{effector.split('_', 1)[0]}.qpos"
        if effector in {"left_hand", "right_hand"}:
            return f"hand.{effector.split('_', 1)[0]}"
        return f"rig.{effector}"

    def _missing_rig_hand_backends() -> dict[str, str]:
        """Detect an unexpectedly omitted hand in the authoritative Rig.

        The canonical capability startup path preserves every configured
        hand binding while the backend is being repaired.  This independent
        check remains for compatibility builders and older live sessions that
        may still have omitted a failed hand; a plain ``ready=True`` report
        would otherwise erase the hand's startup error and expose a command
        path that fails with ``KeyError``.
        """

        session = getattr(components, "session", None)
        rig = getattr(session, "rig", None)
        rig_backends = getattr(rig, "backends", None)
        blockers: dict[str, str] = {}
        for side in ("left", "right"):
            if not bool(
                getattr(components, f"{side}_hand_control_via_rig", False)
            ):
                continue
            backend = getattr(components, f"{side}_hand_backend", None)
            if backend is None:
                continue
            effector = f"{side}_hand"
            if not isinstance(rig_backends, Mapping) or effector not in rig_backends:
                blockers[effector] = "backend missing from Session Rig"
        return blockers

    def _sync_dynamic_rig_readiness(*, force: bool = False) -> dict[str, str]:
        """Mirror current backend readiness into the shared status map."""

        report = _rig_readiness_report(force=force)
        if report is None:
            return {}
        blockers = readiness_report_blockers(report or {})
        # A removed hand is not present in ``report['backends']``.  Merge this
        # independent binding check before projecting readiness so the UI and
        # collection admission remain fail-closed until retry restores it.
        missing_hand_backends = _missing_rig_hand_backends()
        blockers.update(missing_hand_backends)
        current_dynamic_keys: set[str] = set()
        for effector, reason in blockers.items():
            key = _readiness_key(str(effector))
            startup_device_errors[key] = f"{effector}: {reason}"
            current_dynamic_keys.add(key)
        report_ready = bool(report.get("ready", False))
        if blockers or not report_ready:
            detail_parts: list[str] = []
            if not report_ready or readiness_report_blockers(report or {}):
                detail_parts.append(format_readiness_report(report or {}))
            detail_parts.extend(
                f"{effector} {reason}"
                for effector, reason in missing_hand_backends.items()
            )
            detail = "; ".join(part for part in detail_parts if part)
            if not detail:
                detail = "rig is not ready"
            startup_device_errors["rig.session"] = f"Rig is not ready: {detail}"
            current_dynamic_keys.add("rig.session")
        else:
            # A successful backend reconnect does not re-arm an already
            # started Session.  Keep an explicit, non-device blocker until
            # the operator performs a fresh runtime Home; otherwise the
            # generic hardware projection can briefly claim that collection
            # is ready while Session remains DEBUG_ONLY.
            session = getattr(components, "session", None)
            state_machine = getattr(session, "state_machine", None)
            raw_state = getattr(state_machine, "state", None)
            state = str(getattr(raw_state, "value", raw_state) or "").upper()
            if bool(getattr(session, "_started", False)) and state not in {
                "",
                "ARMED",
                "RUNNING",
            }:
                startup_device_errors["rig.session"] = (
                    f"Session state={state}; runtime Home and hardware arm admission "
                    "are required before rule collection."
                )
                current_dynamic_keys.add("rig.session")
        # Remove only errors that this projection owns.  This matters when one
        # backend recovers while another remains faulted: the old effector
        # blocker must disappear without erasing the remaining one.
        for key in tuple(dynamic_readiness_keys - current_dynamic_keys):
            startup_device_errors.pop(key, None)
        dynamic_readiness_keys.clear()
        dynamic_readiness_keys.update(current_dynamic_keys)
        return blockers

    def _restore_rig_hand_binding(side: str, backend: object) -> None:
        """Restore a hand omitted by a legacy/compatibility startup path.

        A retry must restore the exact same backend object before clearing its
        startup error; otherwise the UI can report a healthy serial device
        while the authoritative Rig still raises ``missing effector`` on the
        first command.  Inspire hand retargeters are similarly restored when
        an older startup path removed them from the Session list.  DMTAC
        scalar retargeters intentionally remain out-of-session and are
        therefore not appended.
        """

        if not bool(
            getattr(components, f"{str(side)}_hand_control_via_rig", False)
        ):
            return
        session = getattr(components, "session", None)
        rig = getattr(session, "rig", None)
        rig_backends = getattr(rig, "backends", None)
        if not isinstance(rig_backends, dict):
            raise RuntimeError("Rig hand recovery requires a mutable backend map")
        effector_name = f"{str(side)}_hand"
        existing = rig_backends.get(effector_name)
        if existing is not None and existing is not backend:
            raise RuntimeError(
                f"Rig already owns a different backend for {effector_name}"
            )
        rig_backends[effector_name] = backend

        retargeter = getattr(components, f"{str(side)}_hand_retargeter", None)
        session_retargeters = getattr(session, "retargeters", None)
        if retargeter is not None and isinstance(session_retargeters, list):
            if all(existing_retargeter is not retargeter for existing_retargeter in session_retargeters):
                session_retargeters.append(retargeter)

    def q6_required_for_collection() -> bool:
        """Whether the current effective input still requires live Q6."""

        return bool(getattr(args, "q6_config_required", False)) and str(
            getattr(components, "gripper_input_source", "") or ""
        ) != "angle_trigger"


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
        if key == "device.preflight":
            return "required_for_collection"
        if key == "angle_trigger":
            return (
                "required_for_collection"
                if str(getattr(components, "gripper_input_source", "") or "")
                == "angle_trigger"
                else "optional_input"
            )
        if key == "gripper.input":
            return (
                "required_for_collection"
                if q6_required_for_collection()
                else "optional_input"
            )
        return "feature_dependent"


    def hardware_status_item(key: str) -> dict[str, object]:
        _sync_dynamic_rig_readiness()
        detail = str(startup_device_errors.get(key, "") or "")
        if not detail:
            if key == "dataset.writer":
                recorder = dataset_recorder_provider()
                if recorder is None:
                    return {
                        "online": False,
                        "health": "deferred",
                        "status_level": "info",
                        "status_label": "等待开始数采",
                        "requirement": hardware_requirement(key),
                        "error": (
                            "writer 将在显式选择数据集/profile 或开始数采时，"
                            "按当前配置异步初始化。"
                        ),
                    }
                writer_started = getattr(recorder, "writer_started", None)
                if writer_started is False:
                    return {
                        "online": False,
                        "health": "waiting",
                        "status_level": "info",
                        "status_label": "等待 writer 握手",
                        "requirement": hardware_requirement(key),
                        "error": "writer 尚未完成 ready 握手。",
                    }
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
        if key == "gripper.input" and not q6_required_for_collection():
            return {
                "online": False,
                "health": "waiting",
                "status_level": "warn",
                "status_label": "Q6 未就绪（angle trigger 回退中）",
                "requirement": "optional_input",
                "error": detail,
            }
        if key == "rig.session" and "runtime home" in detail.lower():
            return {
                "online": False,
                "health": "waiting",
                "status_level": "warn",
                "status_label": "等待 runtime Home",
                "requirement": hardware_requirement(key),
                "error": detail,
            }
        if key == "dataset.writer":
            storage_blocked = any(
                marker in detail.lower()
                for marker in (
                    "insufficient free space",
                    "insufficient_free_space",
                    "free space",
                )
            )
            return {
                "online": False,
                "health": "waiting" if storage_blocked else "error",
                "status_level": "warn" if storage_blocked else "error",
                "status_label": (
                    "存储空间不足（采集阻塞）"
                    if storage_blocked
                    else "writer 准备失败"
                ),
                "requirement": hardware_requirement(key),
                "error": (
                    f"{detail}；释放/更换存储空间后，再次开始数采会在后台"
                    "创建全新 writer 并重试。"
                    if storage_blocked
                    else f"{detail}；再次开始数采会在后台创建全新 writer 并重试。"
                ),
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
        _sync_dynamic_rig_readiness()
        ignored_cameras = set(getattr(args, "ignore_camera", ()) or ())
        blockers: dict[str, str] = {}
        for key, detail in startup_device_errors.items():
            # Foot-pad is an operator convenience and never a data modality.
            if key == "foot_pad":
                continue
            # ``--q6-config-required`` describes the requested Q6 path, but
            # capability admission may have selected a working angle-trigger
            # source instead.  Keep the Q6 fault visible/reconnectable without
            # making that superseded source block formal collection.  The
            # angle-trigger startup error and per-episode freshness checks
            # remain authoritative for the effective input.
            if key == "gripper.input" and not q6_required_for_collection():
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
        _sync_dynamic_rig_readiness()
        status_keys = {
            "arm.left.qpos",
            "arm.right.qpos",
            "rig.session",
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
        # The retry path replaces the live adapter and intentionally updates
        # ``_last_foot_pad_online`` to suppress a second edge event on the
        # next poll.  Therefore the normal edge publisher cannot be relied on
        # to refresh the GUI card.  Include the complete foot-pad test state
        # in every hardware-status projection so a successful reconnect is
        # immediately visible (and stays visible on later status ticks).
        foot_pad_ui_state: dict[str, object] = {}
        foot_pad_state_builder = getattr(clutch_input, "_foot_pad_test_ui_state", None)
        if callable(foot_pad_state_builder):
            try:
                foot_pad_ui_state = dict(foot_pad_state_builder() or {})
            except Exception as exc:  # noqa: BLE001 - status must not break retry.
                _log("FootPad", f"状态刷新失败，保留硬件状态：{type(exc).__name__}: {exc}")
        control_ui.publish(
            hardware_status={
                key: hardware_status_item(key)
                for key in sorted(status_keys)
            },
            startup_blockers=collection_startup_blockers(),
            active_controls=clutch_input._current_active_controls(),
            **foot_pad_ui_state,
        )

    def _publish_retry_completion(*, error: str | None) -> None:
        publish_result = getattr(control_ui, "publish_device_retry_result", None)
        if callable(publish_result):
            publish_result(error=error)

    def _publish_retry_outcome(*, error: str | None = None) -> None:
        """Refresh generic and application-specific readiness in causal order."""

        publish_runtime_hardware_status()
        if callable(refresh_application_status):
            try:
                refresh_application_status()
            except Exception as exc:  # noqa: BLE001 - status is observational only.
                _log(
                    "Device",
                    "post-retry application status refresh failed; preserving "
                    f"fail-closed readiness: {type(exc).__name__}: {exc}",
                )
        _publish_retry_completion(error=error)


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

    def _retry_unready_rig_backends(report: Mapping[str, object]) -> None:
        """Retry explicitly unready command backends of a live Session Rig."""

        session = getattr(components, "session", None)
        rig = getattr(session, "rig", None)
        rig_backends = getattr(rig, "backends", None)
        rig_reports = report.get("backends")
        if not isinstance(rig_backends, Mapping) or not isinstance(
            rig_reports, Mapping
        ):
            return
        for effector, raw_report in rig_reports.items():
            if not isinstance(raw_report, Mapping):
                continue
            report_ready = bool(raw_report.get("ready", False))
            if report_ready:
                continue
            backend = rig_backends.get(str(effector))
            if backend is None:
                continue
            if str(effector) in {"left_hand", "right_hand"}:
                # The dedicated ``hand.<side>`` branch below owns hand
                # reconnect and binding restoration.  Avoid running it twice
                # when the dynamic Rig projection also reports the same hand.
                hand_key = f"hand.{str(effector).split('_', 1)[0]}"
                if hand_key in startup_device_errors:
                    continue
            is_ready = getattr(backend, "is_ready", None)
            if not callable(is_ready):
                continue
            try:
                if bool(is_ready()):
                    continue
                if str(effector) in {"left_hand", "right_hand"} and isinstance(
                    backend, DMTacGripperBackend
                ):
                    reconnect = getattr(backend, "force_reconnect_gripper", None)
                    if callable(reconnect):
                        if not bool(reconnect(reason="manual_device_retry")):
                            raise RuntimeError(
                                "DMTAC gripper force reconnect did not recover"
                            )
                    else:
                        backend.start()
                else:
                    backend.start()
                if not bool(is_ready()):
                    refresh = getattr(backend, "refresh_state_from_hardware", None)
                    if callable(refresh):
                        refresh()
                if not bool(is_ready()):
                    raise RuntimeError(f"{effector} backend is still not ready")
            except Exception as exc:  # noqa: BLE001 - preserve fail-closed status.
                _log(
                    "Device",
                    f"{effector} retry during Rig recovery failed: "
                    f"{type(exc).__name__}: {exc}",
                )


    def retry_failed_devices() -> None:
        # A retry is the explicit freshness boundary for readiness; do not
        # reuse a cached fault from the preceding status tick.
        _invalidate_readiness_cache()
        _sync_dynamic_rig_readiness(force=True)
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
            _publish_retry_completion(error=rejection_message)
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
            _publish_retry_completion(error=rejection_message)
            return
        q6_retry_handled = False
        camera_retry_attempted = False
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
                    startup_device_errors.pop("gripper.input", None)
                    args.angle_trigger_startup_error = ""
                    args.q6_startup_error = ""
                    clutch_input.publish_gripper_status(force=True)
            _log("GripperInput", q6_message)
            if control_ui is not None:
                control_ui.append_message("GripperInput", q6_message)
        if not startup_device_errors:
            if control_ui is not None and not q6_retry_handled:
                control_ui.append_message("Device", "当前没有启动失败的可重连设备。")
            _publish_retry_outcome()
            return
        retry_failures: list[str] = []
        retry_backends = {
            "arm.left.qpos": components.left_state_backend,
            "arm.right.qpos": components.right_state_backend,
            "hand.left": components.left_hand_backend,
            "hand.right": components.right_hand_backend,
        }
        for key in tuple(startup_device_errors):
            try:
                if key in retry_backends and retry_backends[key] is not None:
                    backend = retry_backends[key]
                    side = key.split(".", 1)[1] if key.startswith("hand.") else ""
                    if side:
                        # Reinsert the existing backend before checking its
                        # readiness so a failed reconnect remains visible in
                        # the Rig report instead of looking like a healthy Rig
                        # with an untracked hand.
                        _restore_rig_hand_binding(side, backend)
                    if key.startswith("hand."):
                        if isinstance(backend, DMTacGripperBackend):
                            # ``start()`` is idempotent for a live serial
                            # session; it cannot clear a latched vendor fault
                            # or rebuild a stale serial handle.  Manual retry
                            # owns an explicit close/recreate/clear-fault
                            # cycle, then reads one fresh state before
                            # readiness is evaluated.
                            force_reconnect = getattr(
                                backend, "force_reconnect_gripper", None
                            )
                            if callable(force_reconnect):
                                if not bool(
                                    force_reconnect(reason="manual_device_retry")
                                ):
                                    raise RuntimeError(
                                        "DMTAC gripper force reconnect did not recover"
                                    )
                            else:
                                # Keep compatibility with a lightweight
                                # backend fixture that predates the explicit
                                # reconnect seam, while still requiring a
                                # fresh read below.
                                backend.start()
                            read_now = getattr(backend, "read_gripper_state_now", None)
                            if callable(read_now):
                                read_now()
                        else:
                            # Non-DMTAC hand backends (for example Inspire)
                            # expose the ordinary idempotent start/readiness
                            # contract.  The old retry path accidentally
                            # skipped this branch and cleared the error even
                            # though the hand remained disconnected.
                            backend.start()
                        if not bool(backend.is_ready()):
                            report_fn = getattr(backend, "readiness_report", None)
                            report = report_fn() if callable(report_fn) else {
                                "ready": False,
                                "reason": "backend is not ready",
                            }
                            detail = format_readiness_report(
                                {"backends": {f"{side}_hand": report}}
                            )
                            raise RuntimeError(
                                f"hand backend is still not ready: {detail}"
                            )
                    if key.startswith("arm."):
                        state = retry_backends[key].refresh_state_from_hardware()
                        if state.get("current_qpos") is None:
                            raise RuntimeError("hardware returned no current_qpos")
                elif key.startswith("camera.") and key.split(".", 1)[1] in camera_sources:
                    camera_retry_attempted = True
                    camera = camera_sources[key.split(".", 1)[1]]
                    camera.close()
                    if isinstance(camera, RealSenseRGBSource):
                        start_real_sense_source_with_containment(
                            camera,
                            timeout_sec=15.0,
                            software_retry_attempts=None,
                        )
                    else:
                        camera.start()
                elif key == "dataset.writer":
                    # A formal writer is a collection capability, not a
                    # hardware connection.  Constructing it here would run its
                    # ready handshake synchronously on the realtime/UI thread
                    # and can stall arm command production while a populated
                    # dataset is scanned.  The episode control plane already
                    # owns fresh one-shot candidate construction and clears
                    # this error after a successful async preparation.
                    message = (
                        "dataset.writer 不参与通用设备重连；请再次点击“开始数采”，"
                        "系统会在后台创建全新 writer 并执行 ready 握手。"
                    )
                    _log("Dataset", message)
                    if control_ui is not None:
                        control_ui.append_message("Dataset", message)
                    continue
                elif key == "rig.session":
                    if bool(
                        getattr(
                            components.session,
                            "startup_retry_requires_reconstruction",
                            False,
                        )
                    ):
                        raise RuntimeError(
                            "the failed real-teleop Session owns terminal lifecycle "
                            "resources; reconstruct the complete component graph "
                            "instead of restarting the same Session"
                        )
                    # A started Session's ``start()`` is intentionally a
                    # no-op.  Recheck the live Rig first and repair any exact
                    # command backend that failed during its partial start.
                    report_before = _rig_readiness_report()
                    if report_before is not None and not bool(
                        report_before.get("ready", False)
                    ):
                        _retry_unready_rig_backends(report_before)
                        report_before = _rig_readiness_report(force=True)
                    if report_before is not None and not bool(
                        report_before.get("ready", False)
                    ):
                        raise RuntimeError(
                            "Rig is not ready: "
                            f"{format_readiness_report(report_before)}"
                        )
                    components.session.start()
                    report_after = _rig_readiness_report()
                    if report_after is not None and not bool(
                        report_after.get("ready", False)
                    ):
                        raise RuntimeError(
                            "Rig is not ready after retry: "
                            f"{format_readiness_report(report_after)}"
                        )
                    human_shadow_frontend = getattr(
                        components, "human_shadow_frontend", None
                    )
                    if human_shadow_frontend is not None:
                        human_shadow_frontend.start()
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
                elif key == "vive.clock_sync" and vive_source is None:
                    # Rule-based runtimes intentionally have no VIVE source;
                    # tolerate a stale compatibility diagnostic from an older
                    # startup projection and remove it instead of exposing an
                    # impossible retry action.
                    _log(
                        "ViveClockSync",
                        "VIVE clock-sync diagnostic ignored because this runtime has no VIVE source",
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
                    try:
                        live_count = int(
                            getattr(replacement, "foot_pad_count", 0) or 0
                        )
                        live_mode = apply_live_foot_pad_topology(
                            args,
                            live_count,
                        )
                    except Exception:
                        replacement.close()
                        raise
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
                    # A manual retry bypasses the normal offline->online edge
                    # in ``RealTeleopControl.poll`` (the edge is intentionally
                    # suppressed above so the replacement is not opened twice).
                    # Re-arm the connection-scoped diagnostic and publish its
                    # complete UI payload explicitly; otherwise the hardware
                    # card can turn green while the ``脚踏板安全测试`` control
                    # remains hidden/stale until another unrelated event.
                    clutch_input._foot_pad_test_active = False
                    clutch_input._foot_pad_test_completed = False
                    clutch_input._foot_pad_test_seen_events = set()
                    clutch_input._foot_pad_test_last_event = ""
                    count = int(getattr(replacement, "foot_pad_count", 0) or 0)
                    restored_persisted_test = False
                    if count == 5:
                        restore_test = getattr(
                            clutch_input,
                            "_restore_persisted_five_pedal_test_state",
                            None,
                        )
                        if callable(restore_test):
                            restored_persisted_test = bool(restore_test())
                    if count == 2:
                        clutch_input._foot_pad_test_status = (
                            "已重连并识别 2 个踏板；可开始“脚踏板安全测试”。"
                        )
                    elif count == 4:
                        clutch_input._foot_pad_test_status = (
                            "已重连并识别 4 个踏板；可开始“脚踏板安全测试”。"
                        )
                    elif count == 5:
                        if not restored_persisted_test:
                            clutch_input._foot_pad_test_status = (
                                "已重连并识别 5 个踏板；1/2 为 Fix，"
                                "3/4/5 在 progress profile 就绪后执行 progress；"
                                "可开始“脚踏板安全测试”。"
                            )
                    _log(
                        "FootPad",
                        "manual retry resolved live virtual keyboard topology: "
                        f"count={live_count} mode={live_mode}",
                    )
                    topology_guidance = foot_pad_progress_topology_guidance(
                        live_count,
                        progress_enabled=bool(
                            getattr(args, "collection_progress_profile", None)
                            is not None
                            or getattr(args, "collection_progress_gui", False)
                        ),
                    )
                    if topology_guidance:
                        _log("FootPad", topology_guidance)
                        if control_ui is not None:
                            control_ui.append_message("FootPad", topology_guidance)
                    args.foot_pad_startup_error = ""
                    publish_foot_pad_test = getattr(
                        clutch_input, "_publish_foot_pad_test_state", None
                    )
                    if callable(publish_foot_pad_test):
                        publish_foot_pad_test(
                            message=(
                                "脚踏板已重连；当前五踏板映射已有持久化测试结果，"
                                "无需重测。"
                                if restored_persisted_test
                                else "脚踏板已重连；“脚踏板安全测试”已恢复。"
                            )
                        )
                elif key == "gripper.input":
                    # The reconnect/promotion attempt above is the sole owner
                    # of this logical input resource.  Keep its failed status
                    # visible without also claiming that a process restart is
                    # required.
                    continue
                elif key in {
                    "dmtac.assignment",
                    "device.preflight",
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
                detail = f"{type(exc).__name__}: {exc}"
                startup_device_errors[key] = detail
                retry_failures.append(f"{key}: {detail}")
                if key == "dataset.writer":
                    args.dataset_startup_error = detail
                if key.startswith("camera.") and isinstance(exc, TimeoutError):
                    # The native worker may still own the process-wide SDK
                    # lock; do not attempt another device in this retry pass.
                    break
                continue
            startup_device_errors.pop(key, None)
            dynamic_readiness_keys.discard(key)
            if key == "dataset.writer":
                args.dataset_startup_error = ""
            if control_ui is not None:
                control_ui.append_message("Device", f"{key} 重连成功。")
            # A successful backend/session operation can change the report
            # immediately.  Invalidate before the final projection so a
            # recovered hand is not hidden behind a 250 ms stale cache.
            _invalidate_readiness_cache()
            _sync_dynamic_rig_readiness(force=True)
        if camera_retry_attempted and publish_camera_status is not None:
            try:
                # ``hardware_status`` and the left-column RealSense cards are
                # independent rebuildable projections.  During startup/Home
                # wait there may be no control-loop tick to refresh the latter,
                # so explicitly replace its stale startup-error snapshot after
                # this manual retry attempt.
                publish_camera_status()
            except Exception as exc:  # noqa: BLE001 - display refresh is non-authoritative.
                message = f"相机重连状态刷新失败：{type(exc).__name__}: {exc}"
                _log("Camera", message)
                if control_ui is not None:
                    control_ui.append_message("Camera", message)
        _invalidate_readiness_cache()
        _publish_retry_outcome(
            error="；".join(retry_failures) if retry_failures else None,
        )

    return DeviceRetryRuntime(
        hardware_requirement=hardware_requirement,
        hardware_status_item=hardware_status_item,
        collection_startup_blockers=collection_startup_blockers,
        publish_runtime_hardware_status=publish_runtime_hardware_status,
        sync_foot_pad_runtime_status=sync_foot_pad_runtime_status,
        retry_failed_devices=retry_failed_devices,
    )


__all__ = ["DeviceRetryRuntime", "build_device_retry_runtime"]
