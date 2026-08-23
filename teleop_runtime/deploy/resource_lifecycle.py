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
        policy_prediction_worker: Any | None = None,
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
        self.policy_prediction_worker = policy_prediction_worker

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
            if str(args.rtc_profile) == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
                print(
                    "[raw-RTC] OpenPI/model final-chunk shadow active: "
                    "send_actions=false, position_lookahead=false, "
                    "timed-plan admission enabled."
                    + (
                        " live observation-only hardware guard enabled."
                        if bool(getattr(args, "rtc_real_shadow", False))
                        else ""
                    ),
                    flush=True,
                )
            elif str(args.rtc_profile) == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1:
                print(
                    "[model-RTC] conditioned OpenPI shadow active: "
                    "send_actions=false, position_lookahead=false, "
                    "timed-plan admission and prefix conditioning enabled."
                    + (
                        " live observation-only hardware guard enabled."
                        if bool(getattr(args, "rtc_real_shadow", False))
                        else ""
                    ),
                    flush=True,
                )
            else:
                print(
                    "[RTC-3c] virtual-policy shadow active: send_actions=false, "
                    "position_lookahead=false, timed-plan admission enabled.",
                    flush=True,
                )
        if self.policy_prediction_worker is not None:
            self.policy_prediction_worker.start()

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

        if self.policy_prediction_worker is not None:
            close_resource(
                "policy_prediction_worker",
                self.policy_prediction_worker,
            )

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
