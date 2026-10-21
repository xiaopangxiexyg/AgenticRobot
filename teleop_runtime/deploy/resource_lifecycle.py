"""Ordered resource startup and shutdown for policy deployment."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
)
from robot_motion.runtime import exclude_cpu_ids_from_current_process

from .camera_runtime import run_camera_startup_self_check, start_camera_sources
from .hand_backend_runtime import start_hand_backends_checked


class PolicyDeployResourceLifecycle:
    """Own ordered deployment startup and the resource shutdown transaction."""

    def __init__(
        self,
        *,
        debug_logger: Any,
        rtc_frontend: Any | None,
        rtc_debug_observer: Any | None,
        position_deploy_executor: Any | None,
        position_force_guard_monitor: Any | None,
        impedance_force_guard_monitor: Any | None,
        fixed_init_controller: Any | None,
        hand_state_poller: Any | None,
        arm_backends: Mapping[str, Any],
        hand_backends: Mapping[str, Any],
        camera_sources: Mapping[str, Any],
        camera_previewer: Any | None,
        tactile_visualizer: Any | None,
        submit_legacy_owner_state: Callable[..., Any],
        flush_position_executor_debug_events: Callable[[], None],
    ) -> None:
        self.debug_logger = debug_logger
        self.rtc_frontend = rtc_frontend
        self.rtc_debug_observer = rtc_debug_observer
        self.position_deploy_executor = position_deploy_executor
        self.position_force_guard_monitor = position_force_guard_monitor
        self.impedance_force_guard_monitor = impedance_force_guard_monitor
        self.fixed_init_controller = fixed_init_controller
        self.hand_state_poller = hand_state_poller
        self.arm_backends = dict(arm_backends)
        self.hand_backends = dict(hand_backends)
        self.camera_sources = dict(camera_sources)
        self.camera_previewer = camera_previewer
        self.tactile_visualizer = tactile_visualizer
        self.submit_legacy_owner_state = submit_legacy_owner_state
        self.flush_position_executor_debug_events = (
            flush_position_executor_debug_events
        )

    def start(
        self,
        *,
        args: argparse.Namespace,
        disconnected_gripper_sides: set[str] | tuple[str, ...],
        impedance_force_control_is_deferred: bool,
        warm_start_impedance_force_control_backends: Callable[
            [Mapping[str, Any]], tuple[str, ...]
        ],
    ) -> None:
        """Start resources in the established deployment safety order."""

        for side, backend in self.arm_backends.items():
            backend.start()
            if not backend.is_ready() and not bool(args.allow_arm_connect_failure):
                raise RuntimeError(
                    f"{side} arm backend is not ready: {backend.get_state()}"
                )
        if self.position_deploy_executor is not None:
            self.position_deploy_executor.prepare_collision_planner()
            self.position_deploy_executor.start()
            if args.position_reference_worker_cpu is not None:
                affinity = self.position_deploy_executor.snapshot().get(
                    "reference_worker_affinity_receipt"
                )
                if args.position_continuous_waypoint_worker_cpu is None:
                    print(
                        "[PositionCPU] READY: reference worker affinity "
                        f"verified cpu={args.position_reference_worker_cpu}; "
                        "C2 worker is disabled for this formal A/B run.",
                        flush=True,
                    )
                else:
                    print(
                        "[PositionCPU] READY: reference worker affinity verified "
                        f"cpu={args.position_reference_worker_cpu}; "
                        "C2 worker affinity is verified by its planner startup "
                        f"receipt cpu={args.position_continuous_waypoint_worker_cpu} "
                        "isolation="
                        f"{args.position_continuous_waypoint_worker_isolation}.",
                        flush=True,
                    )
                if not isinstance(affinity, Mapping) or not bool(
                    affinity.get("applied", False)
                ):
                    raise RuntimeError(
                        "position reference worker affinity receipt is missing "
                        "after successful startup"
                    )
                if bool(args.position_worker_cpu_reservation):
                    reference_native_thread_id = affinity.get("native_thread_id")
                    if isinstance(reference_native_thread_id, bool) or not isinstance(
                        reference_native_thread_id, int
                    ):
                        raise RuntimeError(
                            "position reference worker affinity receipt lacks "
                            "a native thread id for CPU reservation"
                        )
                    reservation = exclude_cpu_ids_from_current_process(
                        args.position_worker_cpu_reservation_cpu_ids,
                        role="position_deploy_client_ordinary_threads",
                        exempt_native_thread_ids=(int(reference_native_thread_id),),
                    )
                    print(
                        "[PositionCPU] READY: deployment-client ordinary "
                        "threads exclude reserved CPUs "
                        f"{reservation['excluded_cpu_ids']}; reference_tid="
                        f"{reference_native_thread_id} remains pinned; mode="
                        + (
                            "async_c2_process"
                            if args.position_continuous_waypoint_worker_cpu is not None
                            else "formal_reference_only"
                        )
                        + "; migrated_native_threads="
                        f"{reservation['native_threads_migrated']}.",
                        flush=True,
                    )
                if bool(args.position_continuous_waypoint_no_debug_deadline_soak):
                    print(
                        "[PositionCPU] SOAK: bounded no-debug dry-run exception "
                        f"active max_steps={int(args.max_steps)}; hardware "
                        "dispatch remains forbidden.",
                        flush=True,
                    )
        if self.position_force_guard_monitor is not None:
            guard_snapshot = self.position_force_guard_monitor.start(
                arm_stale_watchdog=False,
            )
            print(
                "[PositionForceGuard] force limits armed; stale watchdog "
                "deferred until camera startup completes "
                f"config={guard_snapshot.get('config', {})} "
                f"sides={sorted(guard_snapshot.get('last_reading_by_side', {}))}",
                flush=True,
            )
        if impedance_force_control_is_deferred:
            warmed_sides = warm_start_impedance_force_control_backends(
                self.arm_backends
            )
            if warmed_sides:
                print(
                    "[Deploy] impedance prepared without periodic output; "
                    "press g/H/A to enable it, or use --auto-start.",
                    flush=True,
                )
        start_hand_backends_checked(
            self.hand_backends,
            disconnected_gripper_sides=disconnected_gripper_sides,
            allow_connect_failure=bool(args.allow_hand_connect_failure),
            debug=bool(args.debug),
        )
        if self.hand_state_poller is not None:
            self.hand_state_poller.start()
        if self.camera_sources:
            print(
                "[Camera] configured "
                f"cameras={sorted(self.camera_sources)} mode={args.camera_mode} "
                f"vis_cam={bool(args.vis_cam)}"
            )
            if self.camera_previewer is not None:
                if self.camera_previewer.ensure_available():
                    print("[Camera] --vis_cam display preflight ok.")
                else:
                    print(
                        "[Camera] preview disabled: "
                        f"{self.camera_previewer.disabled_reason}"
                    )
        start_camera_sources(
            self.camera_sources,
            stagger_sec=float(args.camera_start_stagger_sec),
        )
        run_camera_startup_self_check(self.camera_sources)
        if self.position_force_guard_monitor is not None:
            guard_snapshot = self.position_force_guard_monitor.arm_stale_watchdog()
            print(
                "[PositionForceGuard] armed fail_closed=true latched=true "
                "stop_scope=bimanual "
                f"config={guard_snapshot.get('config', {})}",
                flush=True,
            )
        if self.rtc_frontend is not None:
            self.rtc_frontend.start()
            rtc_shadow = bool(getattr(args, "rtc_real_shadow", False))
            dispatch_status = (
                " live observation-only hardware guard enabled."
                if rtc_shadow and not bool(getattr(args, "send_actions", False))
                else (
                    " live hardware dispatch enabled by explicit --send-actions; "
                    "qpos-init and safety gates remain active."
                    if rtc_shadow and bool(getattr(args, "send_actions", False))
                    else (
                        " explicit action dispatch enabled for synthetic/local "
                        "backends."
                        if bool(getattr(args, "send_actions", False))
                        else " action dispatch disabled."
                    )
                )
            )
            if str(args.rtc_profile) == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
                print(
                    "[raw-RTC] OpenPI/model final-chunk shadow active: "
                    "position_lookahead=false, timed-plan admission enabled."
                    + dispatch_status,
                    flush=True,
                )
            elif str(args.rtc_profile) == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
                print(
                    "[model-RTC] conditioned OpenPI shadow active: "
                    "position_lookahead=false, timed-plan admission and prefix "
                    "conditioning enabled."
                    + dispatch_status,
                    flush=True,
                )
            else:
                print(
                    "[RTC-3c] virtual-policy shadow active: action dispatch "
                    "disabled, position_lookahead=false, timed-plan admission "
                    "enabled.",
                    flush=True,
                )
    def close(
        self,
        *,
        keyboard_interrupt_stop_error: str | None,
    ) -> tuple[str, ...]:
        """Drain owners before backend I/O; diagnostics cannot interrupt cleanup."""
        cleanup_errors: list[str] = []
        control_error: BaseException | None = None

        def emit(message: str) -> None:
            nonlocal control_error
            try:
                print(message, file=sys.stderr)
            except BaseException as exc:
                cleanup_errors.append(f"cleanup_stderr: {type(exc).__name__}: {exc}")
                if not isinstance(exc, Exception) and control_error is None:
                    control_error = exc

        def record_error(label: str, exc: BaseException) -> None:
            nonlocal control_error
            if not isinstance(exc, Exception) and control_error is None:
                control_error = exc
            error = f"{label}: {type(exc).__name__}: {exc}"
            cleanup_errors.append(error)
            emit(f"[DeployCleanup] {error}")

        def log_event(event: str, payload: Any) -> None:
            try:
                self.debug_logger.log_event(event, payload)
            except BaseException as exc:
                record_error(f"debug_event:{event}", exc)

        def close_resource(label: str, resource: Any) -> bool:
            try:
                resource.close()
                return True
            except BaseException as exc:
                record_error(label, exc)
                log_event("cleanup_error", {"resource": label,
                    "error_type": type(exc).__name__, "error": str(exc)})
                return False

        try:
            self.submit_legacy_owner_state(lifecycle_state="stopped",
                execution_gate="hold", reason="policy_deploy_cleanup_started")
        except BaseException as exc:
            record_error("legacy_control_owner_stop", exc)
        if keyboard_interrupt_stop_error is not None:
            cleanup_errors.append("keyboard_interrupt_emergency_stop: " + keyboard_interrupt_stop_error)

        if self.rtc_frontend is not None:
            try:
                rtc_stop = self.rtc_frontend.close(now=time.monotonic())
                if self.rtc_debug_observer is not None:
                    self.rtc_debug_observer.offer("rtc_frontend_close", {
                        "worker_stop_hash": rtc_stop.semantic_content_hash,
                        "worker_session_state": rtc_stop.session_state,
                        "thread_exited": rtc_stop.thread_exited,
                        "execution_drain_evidence": rtc_stop.execution_drain_evidence,
                        "stop_status": rtc_stop.status})
            except BaseException as exc:
                record_error("rtc_frontend", exc)
        if self.rtc_debug_observer is not None:
            try:
                enabled = self.rtc_debug_observer.enabled
                status = self.rtc_debug_observer.close()
                if enabled:
                    log_event("rtc_debug_observer_status", status.to_payload())
            except BaseException as exc:
                record_error("rtc_debug_observer", exc)

        position_workers_stopped = self.position_deploy_executor is None
        if self.position_deploy_executor is not None:
            close_error = None
            try:
                self.position_deploy_executor.close()
            except BaseException as exc:
                close_error = exc
                record_error("position_executor", exc)
            # This positive receipt is mandatory on both success and failure.
            # Missing fields, truthy non-bools and snapshot failures are unknown.
            close_snapshot = None
            try:
                close_snapshot = self.position_deploy_executor.shutdown_snapshot()
                position_workers_stopped = (
                    isinstance(close_snapshot, Mapping)
                    and close_snapshot.get("dispatch_workers_stopped") is True)
            except BaseException as exc:
                record_error("position_executor_snapshot", exc)
            if close_error is not None:
                log_event("position_executor_close_error", {
                    "error_type": type(close_error).__name__, "error": str(close_error),
                    "snapshot": close_snapshot,
                    "backend_close_skipped": not position_workers_stopped})
            try:
                self.flush_position_executor_debug_events()
            except BaseException as exc:
                record_error("position_executor_debug_flush", exc)
            log_event("position_executor_final_snapshot", close_snapshot)

        arm_dependencies_stopped = position_workers_stopped
        if self.position_force_guard_monitor is not None:
            try:
                snapshot = self.position_force_guard_monitor.snapshot()
                log_event("position_force_guard_final_snapshot", snapshot)
            except BaseException as exc:
                record_error("position_force_guard_snapshot", exc)
            if not close_resource("position_force_guard", self.position_force_guard_monitor):
                arm_dependencies_stopped = False
        if self.impedance_force_guard_monitor is not None:
            if not close_resource("impedance_force_guard", self.impedance_force_guard_monitor):
                arm_dependencies_stopped = False
        if self.fixed_init_controller is not None and arm_dependencies_stopped:
            try:
                self.fixed_init_controller.restore_speed(self.arm_backends)
            except BaseException as exc:
                record_error("fixed_init_restore_speed", exc)
                log_event("cleanup_error", {"resource": "fixed_init_restore_speed",
                    "error_type": type(exc).__name__, "error": str(exc)})
        hand_dependencies_stopped = position_workers_stopped
        if self.hand_state_poller is not None:
            if not close_resource("hand_state_poller", self.hand_state_poller):
                hand_dependencies_stopped = False

        if arm_dependencies_stopped:
            for side, backend in self.arm_backends.items():
                close_resource(f"{side}_arm_backend", backend)
        if hand_dependencies_stopped:
            for side, backend in self.hand_backends.items():
                close_resource(f"{side}_hand_backend", backend)
        if not position_workers_stopped:
            cleanup_errors.append("position_backends: close skipped because worker shutdown is unconfirmed")
        elif not arm_dependencies_stopped or not hand_dependencies_stopped:
            cleanup_errors.append("position_backends: dependent backend close skipped because guard/poller shutdown is unconfirmed")
        for camera_key, camera_source in self.camera_sources.items():
            close_resource(f"camera:{camera_key}", camera_source)
        if self.camera_previewer is not None:
            close_resource("camera_previewer", self.camera_previewer)
        if self.tactile_visualizer is not None:
            close_resource("tactile_visualizer", self.tactile_visualizer)
        try:
            self.debug_logger.close()
        except BaseException as exc:
            record_error("debug_logger", exc)
        if cleanup_errors:
            emit(f"[DeployCleanup] completed with {len(cleanup_errors)} error(s).")
        if control_error is not None:
            # Preserve interrupt/exit semantics after the independent owners
            # have been attempted; ordinary failures keep the tuple API.
            control_error.cleanup_errors = tuple(cleanup_errors)
            raise control_error
        return tuple(cleanup_errors)


__all__ = ["PolicyDeployResourceLifecycle"]
