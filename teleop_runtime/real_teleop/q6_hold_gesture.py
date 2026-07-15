"""Pure Q6 close-release-close gesture state machine for real teleoperation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Q6HoldRecalibrationGestureConfig:
    enabled: bool = False
    hold_sec: float = 1.0
    close_threshold: float = 0.95
    open_threshold: float = 0.85


class Q6HoldRecalibrationGesture:
    EVENT_HOLD = "hold"
    EVENT_REARMED = "rearmed"
    EVENT_RESTART = "restart"

    _RUNNING_WAIT_CLOSE = "running_wait_close"
    _HOLD_WAIT_OPEN = "hold_wait_open"
    _HOLD_WAIT_CLOSE = "hold_wait_close"
    _RESTART_REQUESTED = "restart_requested"
    _RUNNING_WAIT_RELEASE_TO_REARM = "running_wait_release_to_rearm"

    def __init__(self, config: Q6HoldRecalibrationGestureConfig) -> None:
        self.config = config
        self._state = self._RUNNING_WAIT_CLOSE
        self._condition_since: float | None = None

    def reset_for_running(self, *, require_release: bool = False) -> None:
        self._state = (
            self._RUNNING_WAIT_RELEASE_TO_REARM
            if bool(require_release)
            else self._RUNNING_WAIT_CLOSE
        )
        self._condition_since = None

    def reset_for_hold(self) -> None:
        self._state = self._HOLD_WAIT_OPEN
        self._condition_since = None

    @staticmethod
    def q6_values_from_channel(channel: object | None) -> np.ndarray | None:
        if channel is None or not bool(getattr(channel, "valid", False)):
            return None
        try:
            values = np.asarray(getattr(channel, "value"), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if values.size < 6:
            return None
        tail = values[3:6]
        if not np.all(np.isfinite(tail)):
            return None
        return values

    @staticmethod
    def _tail(values: np.ndarray | None) -> np.ndarray | None:
        if values is None:
            return None
        tail = np.asarray(values, dtype=float).reshape(-1)[3:6]
        if tail.shape != (3,) or not np.all(np.isfinite(tail)):
            return None
        return tail

    def _both_closed(self, left_q6: np.ndarray | None, right_q6: np.ndarray | None) -> bool:
        left_tail = self._tail(left_q6)
        right_tail = self._tail(right_q6)
        if left_tail is None or right_tail is None:
            return False
        threshold = float(self.config.close_threshold)
        return bool(np.all(left_tail > threshold) and np.all(right_tail > threshold))

    def _both_released(self, left_q6: np.ndarray | None, right_q6: np.ndarray | None) -> bool:
        left_tail = self._tail(left_q6)
        right_tail = self._tail(right_q6)
        if left_tail is None or right_tail is None:
            return False
        close_threshold = float(self.config.close_threshold)
        release_threshold = min(float(self.config.open_threshold), close_threshold - 1e-6)
        return bool(np.all(left_tail < release_threshold) and np.all(right_tail < release_threshold))

    def _condition_sustained(self, condition: bool, *, now: float) -> bool:
        if not condition:
            self._condition_since = None
            return False
        if self._condition_since is None:
            self._condition_since = float(now)
        return float(now) - float(self._condition_since) >= max(
            0.0,
            float(self.config.hold_sec),
        )

    def update(
        self,
        *,
        left_q6: np.ndarray | None,
        right_q6: np.ndarray | None,
        teleop_started: bool,
        hold_active: bool,
        now: float,
    ) -> str | None:
        if not bool(self.config.enabled) or not bool(teleop_started):
            self.reset_for_running()
            return None

        closed = self._both_closed(left_q6, right_q6)
        released = self._both_released(left_q6, right_q6)

        if not bool(hold_active):
            if self._state == self._RUNNING_WAIT_RELEASE_TO_REARM:
                if self._condition_sustained(released, now=now):
                    self.reset_for_running()
                return None
            if self._state != self._RUNNING_WAIT_CLOSE:
                self.reset_for_running(require_release=True)
                if released:
                    self.reset_for_running()
            if self._condition_sustained(closed, now=now):
                self._state = self._HOLD_WAIT_OPEN
                self._condition_since = None
                return self.EVENT_HOLD
            return None

        if self._state not in {
            self._HOLD_WAIT_OPEN,
            self._HOLD_WAIT_CLOSE,
            self._RESTART_REQUESTED,
        }:
            self._state = self._HOLD_WAIT_OPEN
            self._condition_since = None

        if self._state == self._HOLD_WAIT_OPEN:
            if self._condition_sustained(released, now=now):
                self._state = self._HOLD_WAIT_CLOSE
                self._condition_since = None
                return self.EVENT_REARMED
            return None

        if self._state == self._HOLD_WAIT_CLOSE:
            if self._condition_sustained(closed, now=now):
                self._state = self._RESTART_REQUESTED
                self._condition_since = None
                return self.EVENT_RESTART
            return None

        return None

    def mark_runtime_restarted(self) -> None:
        self.reset_for_running(require_release=True)


__all__ = ["Q6HoldRecalibrationGesture", "Q6HoldRecalibrationGestureConfig"]
