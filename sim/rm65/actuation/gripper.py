"""Backend-independent fitted DM-Tac normalized position actuator."""

from __future__ import annotations

import hashlib
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


FIT_SCHEMA = "dexteleop.sim.rm65.dmtac_gripper_position_fit.v1"
MODEL_TYPE = "delayed_asymmetric_acceleration_rate_limited_position_v1"
_ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "real2sim" / "artifacts"
_DMTAC_POSITION_V1_ARTIFACT = (
    _ARTIFACTS_DIR / "dmtac_gripper_position_20260722_v1.json"
)


@dataclass(frozen=True)
class DmtacDirectionalPositionParameters:
    delay_sec: float
    max_rate_per_sec: float
    acceleration_per_sec2: float

    def __post_init__(self) -> None:
        for name, positive in (
            ("delay_sec", False),
            ("max_rate_per_sec", True),
            ("acceleration_per_sec2", True),
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
                requirement = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {requirement}")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, float]:
        return {
            "delay_sec": self.delay_sec,
            "max_rate_per_sec": self.max_rate_per_sec,
            "acceleration_per_sec2": self.acceleration_per_sec2,
        }


@dataclass(frozen=True)
class DmtacPositionActuationProfile:
    name: str
    parameters_by_side: Mapping[
        str,
        Mapping[str, DmtacDirectionalPositionParameters],
    ]
    command_hz: float
    send_threshold: float
    source_artifact: str
    source_payload_sha256: str
    source_conditions: Mapping[str, Any]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("profile name must be non-empty")
        command_hz = float(self.command_hz)
        send_threshold = float(self.send_threshold)
        if not math.isfinite(command_hz) or command_hz <= 0.0:
            raise ValueError("command_hz must be finite and positive")
        if not math.isfinite(send_threshold) or send_threshold < 0.0:
            raise ValueError("send_threshold must be finite and non-negative")
        normalized: dict[
            str,
            Mapping[str, DmtacDirectionalPositionParameters],
        ] = {}
        for raw_side, raw_directions in self.parameters_by_side.items():
            side = str(raw_side)
            if side not in {"left", "right"}:
                raise ValueError(f"unsupported gripper side {side!r}")
            if set(raw_directions) != {"close", "open"}:
                raise ValueError(f"{side} must contain close and open parameters")
            directions: dict[str, DmtacDirectionalPositionParameters] = {}
            for direction in ("close", "open"):
                value = raw_directions[direction]
                if not isinstance(value, DmtacDirectionalPositionParameters):
                    raise TypeError(f"{side}.{direction} has invalid parameter type")
                directions[direction] = value
            normalized[side] = MappingProxyType(directions)
        if not normalized:
            raise ValueError("profile must contain at least one fitted side")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "command_hz", command_hz)
        object.__setattr__(self, "send_threshold", send_threshold)
        object.__setattr__(self, "parameters_by_side", MappingProxyType(normalized))
        object.__setattr__(self, "source_artifact", str(self.source_artifact))
        object.__setattr__(
            self,
            "source_payload_sha256",
            str(self.source_payload_sha256),
        )
        object.__setattr__(
            self,
            "source_conditions",
            MappingProxyType(dict(self.source_conditions)),
        )

    @classmethod
    def from_fit_artifact(
        cls,
        path: str | Path,
        *,
        command_hz: float = 20.0,
        send_threshold: float = 0.002,
        name: str | None = None,
        source_artifact_locator: str | None = None,
    ) -> "DmtacPositionActuationProfile":
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != FIT_SCHEMA:
            raise ValueError(f"{source}: unsupported fit artifact schema")
        if payload.get("model_type") != MODEL_TYPE:
            raise ValueError(f"{source}: unsupported gripper position model")
        expected_hash = str(payload.get("payload_sha256", ""))
        without_hash = dict(payload)
        without_hash.pop("payload_sha256", None)
        canonical = json.dumps(
            without_hash,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        observed_hash = hashlib.sha256(canonical).hexdigest()
        if expected_hash != observed_hash:
            raise ValueError(f"{source}: payload_sha256 mismatch")
        raw_fit = payload.get("fit_by_side")
        if not isinstance(raw_fit, Mapping):
            raise ValueError(f"{source}: fit_by_side must be an object")
        parameters: dict[
            str,
            dict[str, DmtacDirectionalPositionParameters],
        ] = {}
        for raw_side, raw_directions in raw_fit.items():
            if not isinstance(raw_directions, Mapping):
                continue
            side = str(raw_side)
            if not all(isinstance(raw_directions.get(item), Mapping) for item in ("close", "open")):
                continue
            parameters[side] = {}
            for direction in ("close", "open"):
                values = raw_directions[direction]
                assert isinstance(values, Mapping)
                parameters[side][direction] = DmtacDirectionalPositionParameters(
                    delay_sec=float(values["delay_sec"]),
                    max_rate_per_sec=float(values["max_rate_per_sec"]),
                    acceleration_per_sec2=float(values["acceleration_per_sec2"]),
                )
        if not parameters:
            raise ValueError(f"{source}: no side has both close and open fits")
        return cls(
            name=(
                str(name)
                if name is not None
                else f"dmtac_position_fit.{expected_hash[:12]}"
            ),
            parameters_by_side=parameters,
            command_hz=command_hz,
            send_threshold=send_threshold,
            source_artifact=(
                str(source)
                if source_artifact_locator is None
                else str(source_artifact_locator)
            ),
            source_payload_sha256=expected_hash,
            source_conditions=dict(payload.get("source_conditions_observed", {})),
        )

    def parameters(
        self,
        side: str,
        *,
        direction: str,
    ) -> DmtacDirectionalPositionParameters:
        return self.parameters_by_side[str(side)][str(direction)]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command_hz": self.command_hz,
            "send_threshold": self.send_threshold,
            "parameters_by_side": {
                side: {
                    direction: parameters.to_dict()
                    for direction, parameters in directions.items()
                }
                for side, directions in self.parameters_by_side.items()
            },
            "source_artifact": self.source_artifact,
            "source_payload_sha256": self.source_payload_sha256,
            "source_conditions": dict(self.source_conditions),
            "scope": "free_motion_normalized_position_only",
        }


@dataclass(frozen=True)
class DmtacPositionActuatorTick:
    physics_time_sec: float
    accepted_by_side: Mapping[str, bool]
    accepted_target_by_side: Mapping[str, float]
    servo_goal_by_side: Mapping[str, float]
    actuator_position_by_side: Mapping[str, float]
    actuator_rate_by_side: Mapping[str, float]


class DmtacPositionActuator:
    """Causal normalized free-motion actuator independent of a physics engine."""

    execution_mode = "dmtac_fitted_normalized_position_v1"

    def __init__(self, profile: DmtacPositionActuationProfile) -> None:
        if not isinstance(profile, DmtacPositionActuationProfile):
            raise TypeError("profile must be DmtacPositionActuationProfile")
        self.profile = profile
        self.active_sides = tuple(profile.parameters_by_side)
        self._pending: dict[str, tuple[float, int]] = {}
        self._accepted: dict[str, float] = {}
        self._servo_goal: dict[str, float] = {}
        self._position: dict[str, float] = {}
        self._rate: dict[str, float] = {}
        self._next_command_time: dict[str, float] = {}
        self._delayed: list[tuple[float, int, str, float]] = []
        self._sequence = 0
        self._applied_sequence_by_side: dict[str, int] = {}
        self._reset = False

    def reset(self, initial_position_by_side: Mapping[str, float]) -> None:
        if set(initial_position_by_side) != set(self.active_sides):
            raise ValueError(
                "initial sides must exactly match profile: "
                f"expected={sorted(self.active_sides)} got={sorted(initial_position_by_side)}"
            )
        values = {
            side: self._validate_position(value, side=side)
            for side, value in initial_position_by_side.items()
        }
        self._pending.clear()
        self._accepted = dict(values)
        self._servo_goal = dict(values)
        self._position = dict(values)
        self._rate = {side: 0.0 for side in values}
        self._next_command_time = {side: 0.0 for side in values}
        self._delayed.clear()
        self._sequence = 0
        self._applied_sequence_by_side = {side: 0 for side in values}
        self._reset = True

    def queue_targets(self, target_by_side: Mapping[str, float]) -> int:
        self._require_reset()
        unknown = set(target_by_side) - set(self.active_sides)
        if unknown:
            raise KeyError(f"unknown gripper side(s): {sorted(unknown)}")
        if not target_by_side:
            return self._sequence
        self._sequence += 1
        for side, value in target_by_side.items():
            self._pending[side] = (
                self._validate_position(value, side=side),
                self._sequence,
            )
        return self._sequence

    def hold_targets(self) -> None:
        """Cancel queued/delayed commands and freeze the current target state."""

        self._require_reset()
        self._pending.clear()
        self._delayed.clear()
        self._accepted = dict(self._position)
        self._servo_goal = dict(self._position)
        self._rate = {side: 0.0 for side in self.active_sides}

    def step(
        self,
        *,
        physics_time_sec: float,
        timestep_sec: float,
    ) -> DmtacPositionActuatorTick:
        self._require_reset()
        now = float(physics_time_sec)
        dt = float(timestep_sec)
        if not math.isfinite(now) or now < 0.0:
            raise ValueError("physics_time_sec must be finite and non-negative")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("timestep_sec must be finite and positive")
        accepted = {side: False for side in self.active_sides}
        period = 1.0 / self.profile.command_hz
        for side in self.active_sides:
            if side not in self._pending or now + 1e-12 < self._next_command_time[side]:
                continue
            requested, sequence = self._pending.pop(side)
            previous = self._accepted[side]
            if abs(requested - previous) >= self.profile.send_threshold:
                direction = "close" if requested > self._position[side] else "open"
                delay = self.profile.parameters(side, direction=direction).delay_sec
                heapq.heappush(
                    self._delayed,
                    (now + delay, sequence, side, requested),
                )
                self._accepted[side] = requested
                accepted[side] = True
            deadline = self._next_command_time[side]
            while deadline <= now + 1e-12:
                deadline += period
            self._next_command_time[side] = deadline

        while self._delayed and self._delayed[0][0] <= now + 1e-12:
            _apply_time, sequence, side, target = heapq.heappop(self._delayed)
            if sequence < self._applied_sequence_by_side[side]:
                continue
            self._servo_goal[side] = target
            self._applied_sequence_by_side[side] = sequence

        for side in self.active_sides:
            goal = self._servo_goal[side]
            current = self._position[side]
            error = goal - current
            if abs(error) <= 1e-12:
                self._position[side] = goal
                self._rate[side] = 0.0
                continue
            direction = "close" if error > 0.0 else "open"
            parameters = self.profile.parameters(side, direction=direction)
            sign = math.copysign(1.0, error)
            braking_rate = math.sqrt(2.0 * parameters.acceleration_per_sec2 * abs(error))
            desired_rate = sign * min(parameters.max_rate_per_sec, braking_rate)
            rate = self._rate[side] + float(
                np.clip(
                    desired_rate - self._rate[side],
                    -parameters.acceleration_per_sec2 * dt,
                    parameters.acceleration_per_sec2 * dt,
                )
            )
            candidate = current + rate * dt
            if (goal - candidate) * error <= 0.0:
                candidate = goal
                rate = 0.0
            self._position[side] = float(np.clip(candidate, 0.0, 1.0))
            self._rate[side] = float(rate)

        return DmtacPositionActuatorTick(
            physics_time_sec=now,
            accepted_by_side=accepted,
            accepted_target_by_side=dict(self._accepted),
            servo_goal_by_side=dict(self._servo_goal),
            actuator_position_by_side=dict(self._position),
            actuator_rate_by_side=dict(self._rate),
        )

    def snapshot(self) -> dict[str, Any]:
        self._require_reset()
        return {
            "execution_mode": self.execution_mode,
            "profile": self.profile.to_metadata(),
            "pending": {
                side: {"target": target, "sequence": sequence}
                for side, (target, sequence) in self._pending.items()
            },
            "pending_delayed_targets": len(self._delayed),
            "accepted_target_by_side": dict(self._accepted),
            "servo_goal_by_side": dict(self._servo_goal),
            "actuator_position_by_side": dict(self._position),
            "actuator_rate_by_side": dict(self._rate),
        }

    @staticmethod
    def _validate_position(value: float, *, side: str) -> float:
        position = float(value)
        if not math.isfinite(position) or not 0.0 <= position <= 1.0:
            raise ValueError(f"{side} target must be finite and in [0, 1]")
        return position

    def _require_reset(self) -> None:
        if not self._reset:
            raise RuntimeError("reset(initial_position_by_side) is required first")


def dmtac_bimanual_position_profile_v1() -> DmtacPositionActuationProfile:
    """Load the frozen 2026-07-22 bilateral free-motion stage profile."""

    return DmtacPositionActuationProfile.from_fit_artifact(
        _DMTAC_POSITION_V1_ARTIFACT,
        command_hz=20.0,
        send_threshold=0.002,
        name="dmtac_bimanual_position_20260722.v1",
        source_artifact_locator=(
            "sim/rm65/real2sim/artifacts/"
            "dmtac_gripper_position_20260722_v1.json"
        ),
    )


def dmtac_bimanual_position_profile() -> DmtacPositionActuationProfile:
    """Return the current simulation-owned DM-Tac position profile."""

    return dmtac_bimanual_position_profile_v1()


__all__ = [
    "FIT_SCHEMA",
    "MODEL_TYPE",
    "DmtacDirectionalPositionParameters",
    "DmtacPositionActuationProfile",
    "DmtacPositionActuator",
    "DmtacPositionActuatorTick",
    "dmtac_bimanual_position_profile",
    "dmtac_bimanual_position_profile_v1",
]
