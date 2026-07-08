from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ...diagnostics import FaultEvent
from ...schemas import EffectorCommand, ObservationChannel, TeleopCommand

FIX_ARM_FAULT_CODES = frozenset(
    {
        "missing_channel",
        "invalid_channel",
        "invalid_pose",
        "stale_channel",
        "translation_jump",
        "rotation_jump",
    }
)


def normalize_fix_arm_side(side: str) -> str:
    normalized = str(side or "").strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"fix-arm side must be 'left' or 'right', got {side!r}.")
    return normalized


@dataclass(frozen=True)
class FixArmTransition:
    ok: bool
    message: str
    side: str = ""
    mode: str = ""
    missing_effectors: tuple[str, ...] = ()
    missing_channels: tuple[str, ...] = ()


@dataclass
class _RecalibrationState:
    started_at: float = 0.0
    timeout_sec: float = 0.0
    min_samples: int = 1
    samples: list[np.ndarray] = field(default_factory=list)


class FixArmCommandController:
    """Freeze one logical side by replaying the last dispatched command."""

    def __init__(
        self,
        *,
        active_sides: Sequence[str],
        required_effectors_by_side: Mapping[str, Sequence[str]],
        selected_side: str = "left",
    ) -> None:
        self.active_sides = tuple(normalize_fix_arm_side(side) for side in active_sides)
        if set(self.active_sides) != {"left", "right"}:
            raise ValueError("--fix-arm requires active-sides both.")
        self.required_effectors_by_side = {
            normalize_fix_arm_side(side): tuple(str(name) for name in effectors if str(name))
            for side, effectors in required_effectors_by_side.items()
        }
        for side in self.active_sides:
            if not self.required_effectors_by_side.get(side):
                self.required_effectors_by_side[side] = (f"{side}_arm",)
        self._selected_side = normalize_fix_arm_side(selected_side)
        self._mode = "live"
        self._fixed_side = ""
        self._latest_by_effector: dict[str, EffectorCommand] = {}
        self._fixed_by_effector: dict[str, EffectorCommand] = {}
        self._latest_by_channel: dict[str, ObservationChannel] = {}
        self._fixed_by_channel: dict[str, ObservationChannel] = {}
        self._recalibration = _RecalibrationState()
        self._status_message = "fix-arm ready"

    @property
    def enabled(self) -> bool:
        return True

    @property
    def selected_side(self) -> str:
        return self._selected_side

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def fixed_side(self) -> str:
        return self._fixed_side

    @property
    def fixed(self) -> bool:
        return self._mode in {"fixed", "recapturing"} and bool(self._fixed_side)

    @property
    def recapturing(self) -> bool:
        return self._mode == "recapturing" and bool(self._fixed_side)

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def sample_count(self) -> int:
        return len(self._recalibration.samples)

    @property
    def min_samples(self) -> int:
        return max(1, int(self._recalibration.min_samples))

    @property
    def recalibration_started_at(self) -> float:
        return float(self._recalibration.started_at)

    @property
    def recalibration_timeout_sec(self) -> float:
        return float(self._recalibration.timeout_sec)

    def recalibration_samples(self) -> list[np.ndarray]:
        return [sample.copy() for sample in self._recalibration.samples]

    def select_side(self, side: str) -> FixArmTransition:
        normalized = normalize_fix_arm_side(side)
        if self.fixed:
            return FixArmTransition(
                ok=False,
                message="fixed side cannot be changed while fix-arm is active",
                side=self._fixed_side,
                mode=self._mode,
            )
        self._selected_side = normalized
        self._status_message = f"selected {normalized}"
        return FixArmTransition(
            ok=True,
            message=f"selected {normalized}",
            side=normalized,
            mode=self._mode,
        )

    def freeze_selected(self, *, now: float) -> FixArmTransition:
        del now
        if self.fixed:
            return FixArmTransition(
                ok=False,
                message=f"{self._fixed_side} is already fixed",
                side=self._fixed_side,
                mode=self._mode,
            )
        side = self._selected_side
        required = self.required_effectors_by_side.get(side, ())
        missing = tuple(name for name in required if name not in self._latest_by_effector)
        required_channels = self._required_channels_for_side(side)
        missing_channels = tuple(
            name
            for name in required_channels
            if name not in self._latest_by_channel
            or not bool(getattr(self._latest_by_channel[name], "valid", False))
        )
        if missing:
            self._status_message = f"cannot fix {side}; missing last command for {', '.join(missing)}"
            return FixArmTransition(
                ok=False,
                message=self._status_message,
                side=side,
                mode=self._mode,
                missing_effectors=missing,
            )
        if missing_channels:
            self._status_message = (
                f"cannot fix {side}; missing valid input channel for "
                f"{', '.join(missing_channels)}"
            )
            return FixArmTransition(
                ok=False,
                message=self._status_message,
                side=side,
                mode=self._mode,
                missing_channels=missing_channels,
            )
        self._fixed_side = side
        self._fixed_by_effector = {
            name: self._latest_by_effector[name].copy()
            for name in required
            if name in self._latest_by_effector
        }
        self._fixed_by_channel = {
            name: self._latest_by_channel[name].copy()
            for name in required_channels
            if name in self._latest_by_channel
        }
        self._mode = "fixed"
        self._recalibration = _RecalibrationState()
        self._status_message = f"{side} fixed"
        return FixArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def start_recalibration(
        self,
        *,
        now: float,
        timeout_sec: float,
        min_samples: int,
    ) -> FixArmTransition:
        if not self.fixed:
            return FixArmTransition(
                ok=False,
                message="fix-arm recalibration requires an active fixed side",
                side=self._selected_side,
                mode=self._mode,
            )
        self._mode = "recapturing"
        self._recalibration = _RecalibrationState(
            started_at=float(now),
            timeout_sec=max(0.1, float(timeout_sec)),
            min_samples=max(1, int(min_samples)),
        )
        self._status_message = f"recapturing {self._fixed_side} home"
        return FixArmTransition(
            ok=True,
            message=self._status_message,
            side=self._fixed_side,
            mode=self._mode,
        )

    def add_recalibration_sample(self, pose_desk: np.ndarray) -> int:
        if not self.recapturing:
            return self.sample_count
        if self.sample_count >= self.min_samples:
            return self.sample_count
        self._recalibration.samples.append(np.asarray(pose_desk, dtype=float).reshape(4, 4).copy())
        return self.sample_count

    def complete_recalibration(self, *, now: float, message: str = "") -> FixArmTransition:
        del now
        side = self._fixed_side
        self._mode = "live"
        self._fixed_side = ""
        self._fixed_by_effector = {}
        self._fixed_by_channel = {}
        self._recalibration = _RecalibrationState()
        self._status_message = message or f"{side} home recaptured; fix-arm released"
        return FixArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def cancel_active(
        self,
        *,
        now: float,
        reason: str = "",
        clear_snapshots: bool = False,
    ) -> FixArmTransition:
        del now
        side = self._fixed_side
        was_active = bool(self.fixed)
        self._mode = "live"
        self._fixed_side = ""
        self._fixed_by_effector = {}
        self._fixed_by_channel = {}
        self._recalibration = _RecalibrationState()
        if clear_snapshots:
            self._latest_by_effector = {}
            self._latest_by_channel = {}
        if was_active:
            suffix = f": {reason}" if reason else ""
            self._status_message = f"{side} fix-arm stopped{suffix}"
        else:
            self._status_message = "fix-arm ready"
        return FixArmTransition(
            ok=was_active,
            message=self._status_message,
            side=side,
            mode=self._mode,
        )

    def fail_recalibration(self, *, now: float, message: str) -> FixArmTransition:
        del now
        side = self._fixed_side
        self._mode = "fixed" if side else "live"
        self._recalibration = _RecalibrationState()
        self._status_message = str(message)
        return FixArmTransition(ok=False, message=self._status_message, side=side, mode=self._mode)

    def filter_channels(
        self,
        channels: Mapping[str, ObservationChannel],
        now: float,
    ) -> dict[str, ObservationChannel]:
        fixed_side = self._fixed_side if self.fixed else ""
        fixed_channel_names = (
            set(self._required_channels_for_side(fixed_side)) if fixed_side else set()
        )
        filtered = {name: channel.copy() for name, channel in channels.items()}
        for channel_name, channel in channels.items():
            if fixed_side and channel_name in fixed_channel_names:
                continue
            if self._side_for_channel(channel_name, channel) in {"left", "right"}:
                self._latest_by_channel[channel_name] = channel.copy()
        if fixed_side:
            for channel_name in self._required_channels_for_side(fixed_side):
                frozen = self._fixed_by_channel.get(channel_name)
                if frozen is None:
                    continue
                next_channel = frozen.copy()
                original_timestamp = float(next_channel.timestamp)
                next_channel.timestamp = float(now)
                next_channel.valid = True
                next_channel.metadata = {
                    **dict(next_channel.metadata),
                    "fix_arm_frozen": True,
                    "fix_arm_fixed_side": fixed_side,
                    "fix_arm_original_timestamp": original_timestamp,
                }
                filtered[channel_name] = next_channel
        return filtered

    def filter_command(
        self,
        command: TeleopCommand,
        _observation,
        _now: float,
        *,
        include_missing_fixed_effectors: bool = False,
        allowed_effectors: Sequence[str] | None = None,
    ) -> TeleopCommand:
        fixed_side = self._fixed_side if self.fixed else ""
        allowed = (
            None
            if allowed_effectors is None
            else {str(name) for name in allowed_effectors if str(name)}
        )
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
        for effector_name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            filtered.commands[effector_name] = next_command
            if not fixed_side or effector_name not in self.required_effectors_by_side.get(
                fixed_side, ()
            ):
                self._latest_by_effector[effector_name] = next_command.copy()
        if fixed_side:
            for effector_name in self.required_effectors_by_side.get(fixed_side, ()):
                if allowed is not None and effector_name not in allowed:
                    continue
                if (
                    not include_missing_fixed_effectors
                    and effector_name not in filtered.commands
                ):
                    continue
                fixed_command = self._fixed_by_effector.get(effector_name)
                if fixed_command is None:
                    continue
                filtered_command = fixed_command.copy()
                filtered_command.metadata = {
                    **dict(filtered_command.metadata),
                    "fix_arm_active": True,
                    "fix_arm_side": fixed_side,
                    "fix_arm_mode": self._mode,
                }
                filtered.commands[effector_name] = filtered_command
                self._latest_by_effector[effector_name] = filtered_command.copy()
            filtered.metadata.update(
                {
                    "fix_arm_active_side": fixed_side,
                    "fix_arm_mode": self._mode,
                }
            )
        return filtered

    def filter_faults(
        self,
        faults: Sequence[FaultEvent],
        _channels,
        _now: float,
    ) -> list[FaultEvent]:
        if not self.fixed:
            return list(faults)
        fixed_channels = set(self._required_channels_for_side(self._fixed_side))
        filtered: list[FaultEvent] = []
        for fault in faults:
            if fault.channel_name in fixed_channels and fault.code in FIX_ARM_FAULT_CODES:
                continue
            filtered.append(fault)
        return filtered

    def status_payload(self) -> dict[str, object]:
        fixed_side = self._fixed_side if self.fixed else ""
        return {
            "fix_arm_available": True,
            "fix_arm_selected_side": self._selected_side,
            "fix_arm_fixed_side": fixed_side,
            "fix_arm_active": bool(fixed_side),
            "fix_arm_recalibrating": self.recapturing,
            "fix_arm_status": self._status_message,
            "fix_arm_sample_count": self.sample_count if self.recapturing else None,
            "fix_arm_min_samples": self.min_samples if self.recapturing else None,
        }

    def _required_channels_for_side(self, side: str) -> tuple[str, ...]:
        if side not in {"left", "right"}:
            return ()
        channels = [f"{side}_wrist_pose"]
        if f"{side}_hand" in self.required_effectors_by_side.get(side, ()):
            channels.append(f"{side}_hand_q6")
        return tuple(channels)

    def _side_for_channel(self, channel_name: str, channel: ObservationChannel) -> str:
        if channel.side in {"left", "right"}:
            return str(channel.side)
        if channel_name.startswith("left_"):
            return "left"
        if channel_name.startswith("right_"):
            return "right"
        return ""

    def _side_for_command(self, effector_name: str, command: EffectorCommand) -> str:
        for side, effectors in self.required_effectors_by_side.items():
            if effector_name in effectors:
                return side
        if command.side in {"left", "right"}:
            return str(command.side)
        if effector_name.startswith("left_"):
            return "left"
        if effector_name.startswith("right_"):
            return "right"
        return ""
