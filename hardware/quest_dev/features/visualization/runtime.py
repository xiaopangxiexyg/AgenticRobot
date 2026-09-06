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
        self.frame_bindings = self._load_frame_bindings(feedback_cfg)
        from .realman_feedback import (
            RealManFeedbackReader,
            parse_realman_arm_specs,
        )

        realman_cfg = feedback_cfg.get("realman", {})
        if not isinstance(realman_cfg, dict):
            raise TypeError(
                "features.visualization.robot_feedback.realman must be a mapping"
            )
        specs = parse_realman_arm_specs(realman_cfg)
        self.reader = RealManFeedbackReader(
            specs,
            max_age_s=float(feedback_cfg.get("max_age_s", 0.25)),
            frame_bindings=self.frame_bindings,
            control_domain_id=str(
                realman_cfg.get(
                    "control_domain_id",
                    "quest-dev:realman-rm65-observation",
                )
            ),
        )
        self.reader.start()
        self.thread = threading.Thread(
            target=self._run_feedback_loop,
            kwargs={
                "rate_hz": float(feedback_cfg.get("rate_hz", 30)),
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
            f"[quest-dev:visualization] direct RealMan RM65 read-only feedback "
            f"enabled: {tuple(spec.robot_id for spec in specs)}{binding_text}",
            flush=True,
        )

    def stop(self) -> None:
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                # Never close an RM65 transport while the reader may still be
                # inside a vendor SDK request. Leave the resource owned and
                # report cleanup failure to the host for a later retry.
                raise RuntimeError(
                    "RealMan feedback thread did not stop before the 2s "
                    "cleanup deadline; RM65 transport remains open"
                )
        stop_reader = getattr(self.reader, "stop", None)
        if callable(stop_reader):
            stop_reader()

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
        rate_hz: float,
    ) -> None:
        period_s = 1.0 / max(1.0, float(rate_hz))
        last_error: str | None = None
        while not self.stop_event.is_set():
            try:
                # The active runtime owns a direct RealMan reader. It never
                # accepts browser/control input and never falls back to a
                # cache or transport adapter.
                message = self.reader.poll()
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
