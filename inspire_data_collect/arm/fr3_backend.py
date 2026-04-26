from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from exo_teleop.fr3_joint_control_client import FR3JointClient

from ..common import EffectorCommand

ARM_DOF = 7


@dataclass
class FR3ArmBackendConfig:
    host: str
    side: str = "right"
    effector_name: str = "arm"
    port: int = 5558
    timeout_s: float = 5.0
    retry_times: int = 1
    retry_wait_s: float = 0.05
    connect_hardware: bool = True
    state_poll_hz: float = 20.0
    max_state_age_s: float = 0.5
    wait_state_timeout_s: float = 2.0
    close_resend_delay_s: float = 0.05
    close_send_stop: bool = True
    startup_align_enabled: bool = True
    startup_align_duration_s: float = 5.0
    startup_align_threshold_rad: float = 0.02
    silent_status: bool = True

    def __post_init__(self) -> None:
        if int(self.port) <= 0:
            raise ValueError("port must be positive.")
        if float(self.timeout_s) <= 0.0:
            raise ValueError("timeout_s must be positive.")
        if float(self.state_poll_hz) <= 0.0:
            raise ValueError("state_poll_hz must be positive.")
        if float(self.max_state_age_s) <= 0.0:
            raise ValueError("max_state_age_s must be positive.")
        if float(self.wait_state_timeout_s) <= 0.0:
            raise ValueError("wait_state_timeout_s must be positive.")
        if float(self.startup_align_duration_s) < 0.0:
            raise ValueError("startup_align_duration_s must be non-negative.")
        if float(self.startup_align_threshold_rad) < 0.0:
            raise ValueError("startup_align_threshold_rad must be non-negative.")


class FR3ArmBackend:
    def __init__(self, config: FR3ArmBackendConfig):
        self.config = config
        self._client: FR3JointClient | None = None
        self._state_lock = threading.Lock()
        self._last_sent_command: np.ndarray | None = None
        self._last_sent_command_timestamp: float | None = None
        self._latest_qpos: np.ndarray | None = None
        self._latest_dq: np.ndarray | None = None
        self._latest_state_timestamp: float | None = None
        self._latest_state_receive_timestamp: float | None = None
        self._latest_state_age_sec: float | None = None
        self._latest_server_age_s: float | None = None
        self._latest_raw_state: dict[str, Any] | None = None
        self._state_error: str | None = None
        self._startup_error: str | None = None
        self._startup_align_active = False
        self._startup_align_complete = (not bool(config.startup_align_enabled)) or (
            not bool(config.connect_hardware)
        )
        self._startup_align_started_at: float | None = None
        self._startup_align_completed_at: float | None = None
        self._startup_align_progress = 1.0 if self._startup_align_complete else 0.0
        self._startup_align_start_qpos: np.ndarray | None = None
        self._startup_align_initial_target: np.ndarray | None = None
        self._startup_align_latest_target: np.ndarray | None = None
        self._startup_align_start_max_delta_rad: float | None = None
        self._startup_align_last_notice_step: int | None = None

    @property
    def name(self) -> str:
        return str(self.config.effector_name)

    def start(self) -> None:
        if not self.config.connect_hardware or self._client is not None:
            return
        try:
            self._client = FR3JointClient(
                self.config.host,
                self.config.port,
                timeout_s=float(self.config.timeout_s),
                retry_times=int(self.config.retry_times),
                retry_wait_s=float(self.config.retry_wait_s),
            )
            self._client.connect()
            self._client.start_state_poller(
                poll_hz=float(self.config.state_poll_hz),
                warn_stale_after_s=float(self.config.max_state_age_s),
                log_poll_errors=not bool(self.config.silent_status),
            )
            if not self._client.wait_until_state_ready(timeout_s=float(self.config.wait_state_timeout_s)):
                raise RuntimeError("FR3 state cache did not become ready in time.")
            self._startup_error = None
            self._state_error = None
        except Exception as exc:
            self._startup_error = str(exc)
            self._state_error = f"startup_failed: {exc}"
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None

    def is_ready(self) -> bool:
        return (not self.config.connect_hardware) or (
            self._client is not None and self._startup_error is None
        )

    def device_info(self) -> dict[str, Any]:
        return {
            "type": "fr3_joint_control_client",
            "side": str(self.config.side),
            "effector_name": str(self.config.effector_name),
            "host": str(self.config.host),
            "port": int(self.config.port),
            "state_poll_hz": float(self.config.state_poll_hz),
            "max_state_age_s": float(self.config.max_state_age_s),
            "startup_align_enabled": bool(self.config.startup_align_enabled),
            "startup_align_duration_s": float(self.config.startup_align_duration_s),
            "startup_align_threshold_rad": float(self.config.startup_align_threshold_rad),
            "ready": bool(self.is_ready()),
            "startup_error": self._startup_error,
        }

    def _current_qpos_for_startup_alignment(self) -> np.ndarray:
        with self._state_lock:
            qpos = None if self._latest_qpos is None else self._latest_qpos.copy()
        if qpos is not None:
            return qpos.astype(np.float32, copy=True)
        if self._client is None:
            raise RuntimeError("startup alignment requires current FR3 qpos, but client is unavailable.")
        state, age = self._client.get_state_cached(
            max_age_s=float(self.config.max_state_age_s),
            fallback_to_sync=True,
        )
        qpos = np.asarray(state.get("q"), dtype=np.float32).reshape(-1)
        if qpos.shape != (ARM_DOF,):
            raise ValueError(f"FR3 startup alignment q has invalid shape {qpos.shape}.")
        with self._state_lock:
            self._latest_qpos = qpos.copy()
            self._latest_state_receive_timestamp = float(time.time() - max(float(age), 0.0))
            self._latest_state_age_sec = float(age)
            self._latest_raw_state = dict(state)
        return qpos.astype(np.float32, copy=True)

    def _target_after_startup_alignment(
        self,
        raw_target: np.ndarray,
        *,
        command_time: float,
    ) -> np.ndarray:
        if (
            not self.config.connect_hardware
            or not self.config.startup_align_enabled
            or self._startup_align_complete
        ):
            return raw_target.astype(np.float32, copy=True)

        start_qpos: np.ndarray | None = None
        with self._state_lock:
            needs_start = not self._startup_align_active and not self._startup_align_complete
        if needs_start:
            start_qpos = self._current_qpos_for_startup_alignment()

        with self._state_lock:
            if self._startup_align_complete:
                return raw_target.astype(np.float32, copy=True)

            if not self._startup_align_active:
                if start_qpos is None:
                    raise RuntimeError("startup alignment failed to resolve start qpos.")
                max_delta = float(np.max(np.abs(raw_target - start_qpos)))
                self._startup_align_start_qpos = start_qpos.copy()
                self._startup_align_initial_target = raw_target.copy()
                self._startup_align_latest_target = raw_target.copy()
                self._startup_align_start_max_delta_rad = max_delta
                self._startup_align_started_at = float(command_time)
                self._startup_align_completed_at = None
                self._startup_align_progress = 0.0
                self._startup_align_last_notice_step = None
                if (
                    max_delta <= float(self.config.startup_align_threshold_rad)
                    or float(self.config.startup_align_duration_s) <= 0.0
                ):
                    self._startup_align_active = False
                    self._startup_align_complete = True
                    self._startup_align_progress = 1.0
                    self._startup_align_completed_at = float(command_time)
                    print(
                        "[FR3ArmBackend] startup alignment skipped: "
                        f"max_delta={max_delta:.4f} rad, "
                        f"threshold={float(self.config.startup_align_threshold_rad):.4f} rad"
                    )
                    return raw_target.astype(np.float32, copy=True)
                self._startup_align_active = True
                print(
                    "\n[FR3ArmBackend] SAFETY: startup alignment active.\n"
                    "  FR3 is ramping from current qpos to the exoskeleton target.\n"
                    "  Keep the exoskeleton still until alignment completes.\n"
                    f"max_delta={max_delta:.4f} rad, "
                    f"duration={float(self.config.startup_align_duration_s):.2f}s"
                )

            assert self._startup_align_start_qpos is not None
            assert self._startup_align_started_at is not None
            duration = max(float(self.config.startup_align_duration_s), 1e-9)
            elapsed = max(0.0, float(command_time) - float(self._startup_align_started_at))
            alpha = min(1.0, elapsed / duration)
            self._startup_align_latest_target = raw_target.copy()
            self._startup_align_progress = float(alpha)
            notice_step = min(int(duration), int(elapsed))
            if (
                not self.config.silent_status
                and self._startup_align_last_notice_step != notice_step
                and alpha < 1.0
            ):
                remaining = max(0.0, duration - elapsed)
                print(
                    "[FR3ArmBackend] startup alignment progress: "
                    f"{100.0 * alpha:.0f}% "
                    f"elapsed={elapsed:.1f}s remaining={remaining:.1f}s"
                )
                self._startup_align_last_notice_step = notice_step
            target = (
                (1.0 - alpha) * self._startup_align_start_qpos.astype(np.float32)
                + alpha * raw_target.astype(np.float32)
            )
            if alpha >= 1.0:
                self._startup_align_active = False
                self._startup_align_complete = True
                self._startup_align_completed_at = float(command_time)
                if (
                    not self.config.silent_status
                    and self._startup_align_last_notice_step != int(duration)
                ):
                    print("[FR3ArmBackend] startup alignment progress: 100%")
                print(
                    "[FR3ArmBackend] startup alignment complete. "
                    "FR3 is now following the live exoskeleton target."
                )
                self._startup_align_last_notice_step = int(duration)
            return target.astype(np.float32, copy=True)

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for '{command.effector_name}'.")
        raw_target = np.asarray(command.value, dtype=np.float32).reshape(-1)
        if raw_target.shape != (ARM_DOF,):
            raise ValueError(f"{self.name} expects q shape {(ARM_DOF,)}, got {raw_target.shape}.")
        command_time = time.time()
        try:
            target = self._target_after_startup_alignment(raw_target, command_time=command_time)
        except Exception as exc:
            self._state_error = f"startup_align_failed: {exc}"
            raise
        if self._client is not None:
            try:
                self._client.send_q(target.astype(float).tolist())
            except Exception as exc:
                self._state_error = f"send_failed: {exc}"
                raise
        with self._state_lock:
            self._last_sent_command = target.copy()
            self._last_sent_command_timestamp = command_time

    def _update_cached_state_from_client(self) -> None:
        if self._client is None:
            return
        state, age = self._client.get_state_cached(
            max_age_s=float(self.config.max_state_age_s),
            fallback_to_sync=False,
        )
        q = np.asarray(state.get("q"), dtype=np.float32).reshape(-1)
        if q.shape != (ARM_DOF,):
            raise ValueError(f"FR3 cached state q has invalid shape {q.shape}.")
        dq_raw = state.get("dq")
        dq = None
        if dq_raw is not None:
            dq_arr = np.asarray(dq_raw, dtype=np.float32).reshape(-1)
            if dq_arr.shape == (ARM_DOF,):
                dq = dq_arr
        receive_timestamp = float(time.time() - max(float(age), 0.0))
        controller_timestamp = state.get("t")
        server_age_s = state.get("age_s")
        with self._state_lock:
            self._latest_qpos = q.copy()
            self._latest_dq = None if dq is None else dq.copy()
            self._latest_state_timestamp = controller_timestamp
            self._latest_state_receive_timestamp = receive_timestamp
            self._latest_state_age_sec = float(age)
            self._latest_server_age_s = None if server_age_s is None else float(server_age_s)
            self._latest_raw_state = dict(state)
            self._state_error = None

    def get_state(self) -> dict[str, object]:
        if self._client is not None:
            try:
                self._update_cached_state_from_client()
            except Exception as exc:
                self._state_error = str(exc)
        with self._state_lock:
            return {
                "arm_enabled": bool(self.config.connect_hardware),
                "arm_side": str(self.config.side),
                "state_source": "fr3_joint_control_client.cached",
                "last_sent_command": None
                if self._last_sent_command is None
                else self._last_sent_command.copy(),
                "last_sent_command_timestamp": self._last_sent_command_timestamp,
                "qpos": None if self._latest_qpos is None else self._latest_qpos.copy(),
                "dq": None if self._latest_dq is None else self._latest_dq.copy(),
                "controller_timestamp": self._latest_state_timestamp,
                "state_receive_timestamp": self._latest_state_receive_timestamp,
                "state_age_sec": self._latest_state_age_sec,
                "server_age_s": self._latest_server_age_s,
                "raw_state": None if self._latest_raw_state is None else dict(self._latest_raw_state),
                "state_error": self._state_error,
                "startup_alignment_enabled": bool(self.config.startup_align_enabled),
                "startup_alignment_active": bool(self._startup_align_active),
                "startup_alignment_complete": bool(self._startup_align_complete),
                "startup_alignment_progress": float(self._startup_align_progress),
                "startup_alignment_duration_s": float(self.config.startup_align_duration_s),
                "startup_alignment_threshold_rad": float(self.config.startup_align_threshold_rad),
                "startup_alignment_started_at": self._startup_align_started_at,
                "startup_alignment_completed_at": self._startup_align_completed_at,
                "startup_alignment_start_max_delta_rad": self._startup_align_start_max_delta_rad,
                "startup_alignment_start_qpos": None
                if self._startup_align_start_qpos is None
                else self._startup_align_start_qpos.copy(),
                "startup_alignment_initial_target": None
                if self._startup_align_initial_target is None
                else self._startup_align_initial_target.copy(),
                "startup_alignment_latest_target": None
                if self._startup_align_latest_target is None
                else self._startup_align_latest_target.copy(),
            }

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        try:
            if self._last_sent_command is not None:
                try:
                    client.send_q(self._last_sent_command.astype(float).tolist())
                    if float(self.config.close_resend_delay_s) > 0.0:
                        time.sleep(float(self.config.close_resend_delay_s))
                except Exception as exc:
                    print(f"[FR3ArmBackend] warn: resend last q failed during close: {exc}")
            if self.config.close_send_stop:
                try:
                    client.stop()
                except Exception as exc:
                    print(f"[FR3ArmBackend] warn: stop failed during close: {exc}")
        finally:
            client.close()
