from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TeleopRunState(str, Enum):
    INIT = "INIT"
    CALIBRATION = "CALIBRATION"
    DEBUG_ONLY = "DEBUG_ONLY"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    FROZEN = "FROZEN"
    FAULT = "FAULT"
    ESTOP = "ESTOP"


class FaultSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class FaultEvent:
    code: str
    message: str
    severity: FaultSeverity = FaultSeverity.ERROR
    timestamp: float = field(default_factory=time.time)
    component: str | None = None
    channel_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RateTracker:
    def __init__(self, max_samples: int = 128):
        self._timestamps: deque[float] = deque(maxlen=max_samples)

    @property
    def num_samples(self) -> int:
        return len(self._timestamps)

    def tick(self, timestamp: float | None = None) -> float:
        now = time.time() if timestamp is None else float(timestamp)
        self._timestamps.append(now)
        return self.current_hz

    @property
    def current_hz(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        dt = self._timestamps[-1] - self._timestamps[0]
        if dt <= 0.0:
            return 0.0
        return float((len(self._timestamps) - 1) / dt)


class TeleopStateMachine:
    def __init__(self, initial_state: TeleopRunState = TeleopRunState.INIT):
        self.state = initial_state
        self.history: list[dict[str, Any]] = [
            {"timestamp": time.time(), "state": initial_state.value, "reason": "init"}
        ]

    def transition(self, new_state: TeleopRunState, reason: str = "") -> None:
        self.state = new_state
        self.history.append(
            {"timestamp": time.time(), "state": new_state.value, "reason": reason}
        )
