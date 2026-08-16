"""Ordered resource shutdown for the real policy deployment application."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping
from typing import Any


class PolicyDeployResourceLifecycle:
    """Own the deployment resource shutdown transaction.

    Construction and startup remain application-controlled for now.  Once the
    application enters its runtime ``try/finally``, this object is the sole
    owner of ordered resource release and cleanup-error aggregation.
    """

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

    def close(
        self,
        *,
        keyboard_interrupt_stop_error: str | None,
    ) -> tuple[str, ...]:
        cleanup_errors: list[str] = []
        position_workers_stopped = True

        try:
            self.submit_legacy_owner_state(
                lifecycle_state="stopped",
                execution_gate="hold",
                reason="policy_deploy_cleanup_started",
            )
        except Exception as exc:  # noqa: BLE001 - cleanup must continue.
            cleanup_errors.append(
                "legacy_control_owner_stop: " + f"{type(exc).__name__}: {exc}"
            )
        if keyboard_interrupt_stop_error is not None:
            cleanup_errors.append(
                "keyboard_interrupt_emergency_stop: "
                + keyboard_interrupt_stop_error
            )

        def close_resource(label: str, resource: Any) -> None:
            try:
                resource.close()
            except Exception as exc:  # noqa: BLE001 - close every resource.
                error = f"{label}: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                self.debug_logger.log_event(
                    "cleanup_error",
                    {
                        "resource": str(label),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"[DeployCleanup] {error}", file=sys.stderr)

        if self.rtc_frontend is not None:
            try:
                rtc_stop = self.rtc_frontend.close(now=time.monotonic())
                if self.rtc_debug_observer is not None:
                    self.rtc_debug_observer.offer(
                        "rtc_frontend_close",
                        {
                            "worker_stop_hash": rtc_stop.semantic_content_hash,
                            "worker_session_state": rtc_stop.session_state,
                            "thread_exited": rtc_stop.thread_exited,
                            "execution_drain_evidence": (
                                rtc_stop.execution_drain_evidence
                            ),
                            "stop_status": rtc_stop.status,
                        },
                    )
            except Exception as exc:  # noqa: BLE001 - continue all cleanup.
                error = f"rtc_frontend: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if self.rtc_debug_observer is not None:
            try:
                rtc_debug_was_enabled = self.rtc_debug_observer.enabled
                rtc_debug_status = self.rtc_debug_observer.close()
                if rtc_debug_was_enabled:
                    self.debug_logger.log_event(
                        "rtc_debug_observer_status",
                        rtc_debug_status.to_payload(),
                    )
            except Exception as exc:  # noqa: BLE001 - diagnostic shutdown only.
                error = f"rtc_debug_observer: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if self.position_deploy_executor is not None:
            try:
                self.position_deploy_executor.close()
            except Exception as exc:  # noqa: BLE001
                close_snapshot = self.position_deploy_executor.snapshot()
                position_workers_stopped = not bool(
                    close_snapshot.get("reference_worker_alive")
                    or close_snapshot.get("discrete_worker_alive")
                )
                print(
                    f"[PositionDeploy] close/hold error: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                cleanup_errors.append(
                    f"position_executor: {type(exc).__name__}: {exc}"
                )
                self.debug_logger.log_event(
                    "position_executor_close_error",
                    {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "snapshot": close_snapshot,
                        "backend_close_skipped": bool(not position_workers_stopped),
                    },
                )
            self.flush_position_executor_debug_events()
            self.debug_logger.log_event(
                "position_executor_final_snapshot",
                self.position_deploy_executor.snapshot(),
            )
        if self.position_force_guard_monitor is not None:
            self.debug_logger.log_event(
                "position_force_guard_final_snapshot",
                self.position_force_guard_monitor.snapshot(),
            )
            close_resource(
                "position_force_guard",
                self.position_force_guard_monitor,
            )
        if self.impedance_force_guard_monitor is not None:
            close_resource(
                "impedance_force_guard",
                self.impedance_force_guard_monitor,
            )
        if self.fixed_init_controller is not None:
            try:
                self.fixed_init_controller.restore_speed(self.arm_backends)
            except Exception as exc:  # noqa: BLE001
                error = f"fixed_init_restore_speed: {type(exc).__name__}: {exc}"
                cleanup_errors.append(error)
                self.debug_logger.log_event(
                    "cleanup_error",
                    {
                        "resource": "fixed_init_restore_speed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"[DeployCleanup] {error}", file=sys.stderr)
        if self.hand_state_poller is not None:
            close_resource("hand_state_poller", self.hand_state_poller)
        if position_workers_stopped:
            for side, backend in self.arm_backends.items():
                close_resource(f"{side}_arm_backend", backend)
            for side, backend in self.hand_backends.items():
                close_resource(f"{side}_hand_backend", backend)
        else:
            cleanup_errors.append(
                "position_backends: close skipped because a dispatch worker is still alive"
            )
            print(
                "[PositionDeploy] ERROR: a dispatch worker is still alive; "
                "arm/hand backend close was intentionally skipped to avoid "
                "concurrent I/O.",
                file=sys.stderr,
            )
        for camera_key, camera_source in self.camera_sources.items():
            close_resource(f"camera:{camera_key}", camera_source)
        if self.camera_previewer is not None:
            close_resource("camera_previewer", self.camera_previewer)
        if self.tactile_visualizer is not None:
            close_resource("tactile_visualizer", self.tactile_visualizer)
        try:
            self.debug_logger.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"debug_logger: {type(exc).__name__}: {exc}")
            print(
                f"[DeployCleanup] debug_logger: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        if cleanup_errors:
            print(
                f"[DeployCleanup] completed with {len(cleanup_errors)} error(s).",
                file=sys.stderr,
            )
        return tuple(cleanup_errors)


__all__ = ["PolicyDeployResourceLifecycle"]
