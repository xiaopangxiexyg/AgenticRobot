"""Host-side lifecycle for the optional visualization feature."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .frame_binding import (
    AcceptedFrameBinding,
    load_accepted_frame_binding,
    parse_accepted_frame_binding,
)
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
        self.frame_bindings: dict[str, AcceptedFrameBinding] = {}

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
        self.frame_bindings = self._load_frame_bindings(feedback_cfg)
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
                "frame_bindings": self.frame_bindings,
            },
            name="quest-dev-visualization-feedback",
            daemon=True,
        )
        self.thread.start()
        binding_text = (
            f"; accepted frame bindings: {tuple(sorted(self.frame_bindings))}"
            if self.frame_bindings
            else "; no accepted frame bindings; TCP remains unrenderable"
        )
        print(
            f"[quest-dev:visualization] read-only robot feedback enabled: "
            f"{sources}{binding_text}",
            flush=True,
        )

    def stop(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        close_reader = getattr(self.reader, "close", None)
        if callable(close_reader):
            close_reader()

    @staticmethod
    def _load_frame_bindings(
        feedback_cfg: Mapping[str, Any],
    ) -> dict[str, AcceptedFrameBinding]:
        raw_bindings = feedback_cfg.get("frame_bindings", {})
        raw_paths = feedback_cfg.get("frame_binding_paths", {})
        if not isinstance(raw_bindings, Mapping):
            raise TypeError(
                "features.visualization.robot_feedback.frame_bindings must be a mapping"
            )
        if not isinstance(raw_paths, Mapping):
            raise TypeError(
                "features.visualization.robot_feedback.frame_binding_paths must be a mapping"
            )
        bindings: dict[str, AcceptedFrameBinding] = {}
        for raw_robot, raw_payload in raw_bindings.items():
            robot = str(raw_robot)
            if robot in bindings:
                raise ValueError(f"duplicate accepted frame binding for {robot!r}")
            if not isinstance(raw_payload, Mapping):
                raise TypeError(
                    f"frame_bindings[{robot!r}] must be an accepted binding mapping"
                )
            binding = parse_accepted_frame_binding(raw_payload)
            if binding.robot.robot_id != robot:
                raise ValueError(
                    f"frame binding robot identity differs from key {robot!r}"
                )
            bindings[robot] = binding
        for raw_robot, raw_path in raw_paths.items():
            robot = str(raw_robot)
            if robot in bindings:
                raise ValueError(
                    f"frame binding {robot!r} is configured both inline and by path"
                )
            path = Path(str(raw_path)).expanduser()
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            binding = load_accepted_frame_binding(path)
            if binding.robot.robot_id != robot:
                raise ValueError(
                    f"frame binding file robot identity differs from key {robot!r}"
                )
            bindings[robot] = binding
        return bindings

    def _run_feedback_loop(
        self,
        *,
        sources: dict[str, str],
        rate_hz: float,
        max_age_s: float,
        frame_bindings: Mapping[str, AcceptedFrameBinding] | None = None,
    ) -> None:
        from .robot_feedback import poll_robot_feedback

        period_s = 1.0 / max(1.0, float(rate_hz))
        last_error: str | None = None
        while not self.stop_event.is_set():
            try:
                message = poll_robot_feedback(
                    self.reader,
                    sources,
                    max_age_s=float(max_age_s),
                    frame_bindings=frame_bindings,
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
