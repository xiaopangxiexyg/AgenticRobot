"""Host-side lifecycle for the optional visualization feature."""

from __future__ import annotations

import threading
from typing import Any

from .protocol import FEATURE_NAME, feedback_error_message


class VisualizationRuntime:
    def __init__(
        self, server: Any, config: dict[str, Any], stop_event: threading.Event
    ) -> None:
        self.server = server
        self.config = config
        self.stop_event = stop_event
        self.reader: Any = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        feedback_cfg = self.config.get("robot_feedback", {})
        if not isinstance(feedback_cfg, dict) or not bool(
            feedback_cfg.get("enabled", False)
        ):
            print("[quest-dev:visualization] robot feedback disabled", flush=True)
            return
        redis_cfg = feedback_cfg.get("redis", {})
        sources_cfg = feedback_cfg.get("sources", {})
        if not isinstance(redis_cfg, dict):
            raise TypeError(
                "features.visualization.robot_feedback.redis must be a mapping"
            )
        if not isinstance(sources_cfg, dict) or not sources_cfg:
            raise TypeError(
                "features.visualization.robot_feedback.sources must be a non-empty mapping"
            )
        from .robot_feedback import connect_redis

        sources = {str(robot): str(key) for robot, key in sources_cfg.items()}
        self.reader = connect_redis(
            str(redis_cfg.get("host", "localhost")),
            int(redis_cfg.get("port", 6379)),
            int(redis_cfg.get("db", 0)),
            float(redis_cfg.get("socket_timeout_s", 0.2)),
        )
        self.thread = threading.Thread(
            target=self._run_feedback_loop,
            kwargs={
                "sources": sources,
                "rate_hz": float(feedback_cfg.get("rate_hz", 30)),
                "max_age_s": float(feedback_cfg.get("max_age_s", 0.25)),
            },
            name="quest-dev-visualization-feedback",
            daemon=True,
        )
        self.thread.start()
        print(
            f"[quest-dev:visualization] read-only robot feedback enabled: {sources}",
            flush=True,
        )

    def stop(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        close_reader = getattr(self.reader, "close", None)
        if callable(close_reader):
            close_reader()

    def _run_feedback_loop(
        self,
        *,
        sources: dict[str, str],
        rate_hz: float,
        max_age_s: float,
    ) -> None:
        from .robot_feedback import poll_robot_feedback

        period_s = 1.0 / max(1.0, float(rate_hz))
        last_error: str | None = None
        while not self.stop_event.is_set():
            try:
                message = poll_robot_feedback(
                    self.reader, sources, max_age_s=float(max_age_s)
                )
                last_error = None
            except Exception as exc:  # noqa: BLE001 - keep the optional feature isolated.
                error = f"{type(exc).__name__}: {exc}"
                message = feedback_error_message(error)
                if error != last_error:
                    print(
                        f"[quest-dev:visualization] feedback error: {error}", flush=True
                    )
                    last_error = error
            self.server.publish_feature_message(FEATURE_NAME, message)
            self.stop_event.wait(period_s)


def create_runtime(
    server: Any, config: dict[str, Any], stop_event: threading.Event
) -> VisualizationRuntime:
    return VisualizationRuntime(server, config, stop_event)
