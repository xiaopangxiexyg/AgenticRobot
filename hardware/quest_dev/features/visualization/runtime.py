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


class VisualizationRuntime:
    def __init__(
        self, server: Any, config: dict[str, Any], stop_event: threading.Event
    ) -> None:
        self.server = server
        self.config = config
        self.stop_event = stop_event

    def start(self) -> None:
        feedback_cfg = self.config.get("robot_feedback", {})
        if not isinstance(feedback_cfg, dict):
            raise TypeError("features.visualization.robot_feedback must be a mapping")
        if bool(feedback_cfg.get("enabled", False)):
            raise ValueError(
                "hardware Quest visualization cannot own robot feedback; "
                "an application layer must publish canonical robot_tcp telemetry"
            )
        print(
            "[quest-dev:visualization] awaiting application-published telemetry",
            flush=True,
        )

    def stop(self) -> None:
        return None

    @staticmethod
    def _load_frame_bindings(
        feedback_cfg: Mapping[str, Any],
    ) -> dict[str, AcceptedFrameBinding]:
        """Parse lower-owned binding data without creating a robot source."""

        raw_bindings = feedback_cfg.get("frame_bindings", {})
        raw_paths = feedback_cfg.get("frame_binding_paths", {})
        if not isinstance(raw_bindings, Mapping):
            raise TypeError("frame_bindings must be a mapping")
        if not isinstance(raw_paths, Mapping):
            raise TypeError("frame_binding_paths must be a mapping")
        bindings: dict[str, AcceptedFrameBinding] = {}
        for raw_robot, raw_payload in raw_bindings.items():
            robot = str(raw_robot)
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


def create_runtime(
    server: Any, config: dict[str, Any], stop_event: threading.Event
) -> VisualizationRuntime:
    return VisualizationRuntime(server, config, stop_event)
