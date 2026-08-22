"""Bounded latest-value admission for authenticated controller pose telemetry."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from ...protocol import PROTOCOL_VERSION
from .protocol import (
    REFERENCE_SPACE_TYPES,
    ControllerPoseSample,
    parse_controller_pose_sample,
)


@dataclass
class _ClientState:
    client_instance_id: str
    xr_session_id: str
    reference_space_type: str
    reference_space_generation: int
    last_sequence: int = -1
    last_received_monotonic_ns: int = 0
    latest: ControllerPoseSample | None = None


class ControllerPoseTelemetryRuntime:
    def __init__(
        self, server: Any, config: dict[str, Any], stop_event: threading.Event
    ) -> None:
        del stop_event
        admission = config.get("admission", {})
        if not isinstance(admission, dict):
            raise TypeError("features.controller_pose.admission must be a mapping")
        self.server = server
        self.max_rate_hz = max(1.0, float(admission.get("max_rate_hz", 40)))
        self.max_controllers = max(0, int(admission.get("max_controllers", 2)))
        self.lock = threading.RLock()
        self.states: dict[int, _ClientState] = {}
        self.accepted_count = 0
        self.rejected_count = 0
        self.last_rejection: str | None = None

    def start(self) -> None:
        self.server.register_client_message_handler(
            "xr_session_state", self._handle_session_state
        )
        self.server.register_client_message_handler(
            "reference_space_reset", self._handle_reference_space_reset
        )
        self.server.register_client_message_handler(
            "controller_pose_sample", self._handle_pose_sample
        )
        self.server.register_client_disconnect_handler(self._handle_disconnect)
        self.server.register_feature_status_provider(
            "controller_pose", self.snapshot_status
        )
        print(
            "[quest-dev:controller_pose] authenticated read-only telemetry enabled",
            flush=True,
        )

    def stop(self) -> None:
        with self.lock:
            self.states.clear()

    @staticmethod
    def _identity(payload: dict[str, Any]) -> tuple[str, str, str, int]:
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError("core protocol version mismatch")
        client_instance_id = payload.get("client_instance_id")
        xr_session_id = payload.get("xr_session_id")
        reference_space_type = payload.get("reference_space_type")
        generation = payload.get("reference_space_generation")
        for name, value in (
            ("client_instance_id", client_instance_id),
            ("xr_session_id", xr_session_id),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 160
                or not all(
                    character.isalnum() or character in "._:-" for character in value
                )
            ):
                raise ValueError(f"{name} is not a valid session identifier")
        if reference_space_type not in REFERENCE_SPACE_TYPES:
            raise ValueError("unsupported reference-space type")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError(
                "reference_space_generation must be a non-negative integer"
            )
        return client_instance_id, xr_session_id, reference_space_type, generation

    def _reject(self, exc: Exception) -> None:
        with self.lock:
            self.rejected_count += 1
            self.last_rejection = f"{type(exc).__name__}: {exc}"

    def _handle_session_state(self, client_id: int, payload: dict[str, Any]) -> None:
        try:
            state = payload.get("state")
            identity = self._identity(payload)
            with self.lock:
                if state == "started":
                    if identity[3] != 0:
                        raise ValueError(
                            "XR session must start at reference-space generation zero"
                        )
                    if client_id in self.states:
                        raise ValueError("XR session already active for client")
                    self.states[client_id] = _ClientState(*identity)
                elif state == "ended":
                    current = self.states.get(client_id)
                    current_identity = (
                        (
                            current.client_instance_id,
                            current.xr_session_id,
                            current.reference_space_type,
                            current.reference_space_generation,
                        )
                        if current is not None
                        else None
                    )
                    if identity != current_identity:
                        raise ValueError("XR session end identity mismatch")
                    self.states.pop(client_id, None)
                else:
                    raise ValueError("unsupported XR session state")
        except Exception as exc:  # noqa: BLE001 - reject malformed telemetry only.
            self._reject(exc)

    def _handle_reference_space_reset(
        self, client_id: int, payload: dict[str, Any]
    ) -> None:
        try:
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                raise ValueError("core protocol version mismatch")
            previous = payload.get("previous_generation")
            new = payload.get("new_generation")
            if (
                isinstance(previous, bool)
                or not isinstance(previous, int)
                or isinstance(new, bool)
                or not isinstance(new, int)
                or new != previous + 1
            ):
                raise ValueError(
                    "reference-space generation must advance by exactly one"
                )
            with self.lock:
                state = self.states.get(client_id)
                if state is None:
                    raise ValueError("reference-space reset has no active XR session")
                if payload.get("client_instance_id") != state.client_instance_id:
                    raise ValueError("reference-space reset client identity mismatch")
                if payload.get("xr_session_id") != state.xr_session_id:
                    raise ValueError("reference-space reset session identity mismatch")
                if payload.get("reference_space_type") != state.reference_space_type:
                    raise ValueError("reference-space reset type mismatch")
                if previous != state.reference_space_generation:
                    raise ValueError("reference-space reset generation mismatch")
                state.reference_space_generation = new
                state.latest = None
                state.last_received_monotonic_ns = 0
        except Exception as exc:  # noqa: BLE001 - reject malformed telemetry only.
            self._reject(exc)

    def _handle_pose_sample(self, client_id: int, payload: dict[str, Any]) -> None:
        try:
            sample = parse_controller_pose_sample(
                payload,
                received_wall_time_s=payload.get("server_timestamp"),
                received_monotonic_ns=payload.get("server_monotonic_ns"),
                max_controllers=self.max_controllers,
            )
            with self.lock:
                state = self.states.get(client_id)
                if state is None:
                    raise ValueError("controller pose has no active XR session")
                if sample.client_instance_id != state.client_instance_id:
                    raise ValueError("controller pose client identity mismatch")
                if sample.xr_session_id != state.xr_session_id:
                    raise ValueError("controller pose session identity mismatch")
                if sample.reference_space_type != state.reference_space_type:
                    raise ValueError("controller pose reference-space type mismatch")
                if (
                    sample.reference_space_generation
                    != state.reference_space_generation
                ):
                    raise ValueError(
                        "controller pose reference-space generation mismatch"
                    )
                if sample.sample_sequence <= state.last_sequence:
                    raise ValueError("controller pose sequence must strictly increase")
                minimum_period_ns = int(1_000_000_000 / self.max_rate_hz)
                if (
                    state.last_received_monotonic_ns
                    and sample.received_monotonic_ns - state.last_received_monotonic_ns
                    < minimum_period_ns
                ):
                    raise ValueError("controller pose admission rate exceeded")
                state.last_sequence = sample.sample_sequence
                state.last_received_monotonic_ns = sample.received_monotonic_ns
                state.latest = sample
                self.accepted_count += 1
                self.last_rejection = None
        except Exception as exc:  # noqa: BLE001 - reject malformed telemetry only.
            self._reject(exc)

    def _handle_disconnect(self, client_id: int) -> None:
        with self.lock:
            self.states.pop(client_id, None)

    def latest_sample(self, *, client_id: int) -> ControllerPoseSample | None:
        with self.lock:
            state = self.states.get(client_id)
            return None if state is None else state.latest

    def snapshot_status(self) -> dict[str, Any]:
        with self.lock:
            clients = {
                str(client_id): {
                    "client_instance_id": state.client_instance_id,
                    "xr_session_id": state.xr_session_id,
                    "reference_space_type": state.reference_space_type,
                    "reference_space_generation": state.reference_space_generation,
                    "last_sequence": state.last_sequence,
                    "controller_count": len(state.latest.controllers)
                    if state.latest is not None
                    else 0,
                }
                for client_id, state in self.states.items()
            }
            return {
                "mode": "latest_only",
                "accepted_count": self.accepted_count,
                "rejected_count": self.rejected_count,
                "last_rejection": self.last_rejection,
                "clients": clients,
            }


def create_runtime(
    server: Any, config: dict[str, Any], stop_event: threading.Event
) -> ControllerPoseTelemetryRuntime:
    return ControllerPoseTelemetryRuntime(server, config, stop_event)
