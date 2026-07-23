"""Live GUI projection for synchronous hardware startup.

Device ownership and startup ordering remain with the calling runtime.  This
helper only publishes progress while a blocking ``start()`` call is in flight.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Protocol


class StartupProgressUi(Protocol):
    def publish(self, **updates: Any) -> None: ...


class GuiStartupProgress:
    """Publish per-device startup state plus a low-rate liveness heartbeat."""

    def __init__(
        self,
        ui: StartupProgressUi,
        *,
        stage_prefix: str = "设备连接中",
        heartbeat_interval_sec: float | None = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ui = ui
        self._stage_prefix = str(stage_prefix)
        self._heartbeat_interval_sec = (
            None
            if heartbeat_interval_sec is None
            else max(0.05, float(heartbeat_interval_sec))
        )
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._current_key = ""
        self._items: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        with self._lock:
            if self._started_at is None:
                self._started_at = self._clock()
            if self._thread is not None or self._heartbeat_interval_sec is None:
                should_start_thread = False
            else:
                self._stop_event.clear()
                self._thread = threading.Thread(
                    target=self._heartbeat_loop,
                    name="gui-startup-progress",
                    daemon=True,
                )
                should_start_thread = True
        self.publish_now()
        if should_start_thread:
            assert self._thread is not None
            self._thread.start()

    def begin(
        self,
        key: str,
        *,
        label: str,
        message: str,
        requirement: str = "feature_dependent",
    ) -> None:
        now = self._clock()
        with self._lock:
            if self._started_at is None:
                self._started_at = now
            normalized_key = str(key)
            self._current_key = normalized_key
            self._items[normalized_key] = {
                "label": str(label),
                "message": str(message),
                "requirement": str(requirement),
                "status": "waiting",
                "error": "",
                "started_at": now,
                "finished_at": None,
            }
        self.publish_now()

    def update(self, key: str, *, message: str) -> None:
        with self._lock:
            item = self._items.get(str(key))
            if item is None:
                return
            item["message"] = str(message)
            if item["status"] == "waiting":
                self._current_key = str(key)
        self.publish_now()

    def succeed(self, key: str, *, message: str = "连接完成") -> None:
        self._finish(key, status="online", message=message, error="")

    def fail(self, key: str, error: str, *, message: str = "连接失败") -> None:
        self._finish(key, status="error", message=message, error=str(error))

    def warn(self, key: str, warning: str, *, message: str) -> None:
        self._finish(key, status="warning", message=message, error=str(warning))

    def close(self) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def publish_now(self) -> None:
        now = self._clock()
        with self._lock:
            started_at = self._started_at if self._started_at is not None else now
            total_elapsed_sec = max(0.0, now - started_at)
            current_key = self._current_key
            items = {
                key: self._public_item(item, now)
                for key, item in self._items.items()
            }
            current = self._items.get(current_key)
            current_label = "" if current is None else str(current["label"])
            current_message = "" if current is None else str(current["message"])
        if current_label:
            control_hint = (
                f"当前：{current_label}；{current_message}；"
                f"启动总耗时 {total_elapsed_sec:.1f}s。"
            )
        else:
            control_hint = f"正在准备硬件连接；启动总耗时 {total_elapsed_sec:.1f}s。"
        self._ui.publish(
            stage=f"{self._stage_prefix} · {total_elapsed_sec:.1f}s",
            control_hint=control_hint,
            hardware_status=items,
        )

    def _finish(self, key: str, *, status: str, message: str, error: str) -> None:
        now = self._clock()
        with self._lock:
            normalized_key = str(key)
            item = self._items.get(normalized_key)
            if item is None:
                item = {
                    "label": normalized_key,
                    "requirement": "feature_dependent",
                    "started_at": now,
                }
                self._items[normalized_key] = item
            item.update(
                status=str(status),
                message=str(message),
                error=str(error),
                finished_at=now,
            )
            if self._current_key == normalized_key:
                self._current_key = ""
        self.publish_now()

    @staticmethod
    def _public_item(item: dict[str, Any], now: float) -> dict[str, Any]:
        status = str(item.get("status") or "waiting")
        started_at = float(item.get("started_at", now))
        finished_at = item.get("finished_at")
        elapsed_until = now if finished_at is None else float(finished_at)
        elapsed_sec = max(0.0, elapsed_until - started_at)
        if status == "online":
            level = "ok"
            label = "已连接"
            online = True
            health = "online"
        elif status == "error":
            level = "error"
            label = "连接失败"
            online = False
            health = "error"
        elif status == "warning":
            level = "warn"
            label = "未就绪（可继续）"
            online = False
            health = "waiting"
        else:
            level = "waiting"
            label = "连接中"
            online = False
            health = "waiting"
        return {
            "online": online,
            "health": health,
            "status_level": level,
            "status_label": label,
            "requirement": str(item.get("requirement") or "feature_dependent"),
            "label": str(item.get("label") or ""),
            "message": str(item.get("message") or ""),
            "error": str(item.get("error") or ""),
            "elapsed_sec": elapsed_sec,
        }

    def _heartbeat_loop(self) -> None:
        assert self._heartbeat_interval_sec is not None
        while not self._stop_event.wait(self._heartbeat_interval_sec):
            self.publish_now()
