"""GUI-side input adapter for the virtual foot-pad keyboard.

The hardware service remains an independently managed process.  It exposes a
virtual ``DexTeleop foot pads`` keyboard that emits F13--F16; this module only
checks that keyboard and translates press edges into existing GUI actions.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any


SPECIAL_ARM_CONTROL_MODE_MANUAL = "manual"
SPECIAL_ARM_CONTROL_MODE_FOOT_PAD = "foot-pad"
SPECIAL_ARM_CONTROL_MODE_CHOICES = (
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
)

FOOT_PAD_DEVICE_NAME = "DexTeleop foot pads"
FOOT_PAD_START_COMMAND = "python -m hardware.foot_pad --count 4"
FOOT_PAD_REQUIRED_KEY_CODES = frozenset((183, 184, 185, 186))  # F13--F16
FOOT_PAD_EVENT_BY_KEY_CODE = {
    183: "fix_arm_left",
    184: "fix_arm_right",
    185: "init_arm_left",
    186: "init_arm_right",
}


class FootPadUnavailableError(RuntimeError):
    """The independently managed virtual foot-pad keyboard is absent."""


def resolve_special_arm_control_mode(
    requested_mode: str | None,
    *,
    fix_arm_enabled: bool,
    init_arm_enabled: bool,
) -> str:
    """Resolve the default without changing either special-arm controller."""

    requested = str(requested_mode or "").strip()
    if requested and requested not in SPECIAL_ARM_CONTROL_MODE_CHOICES:
        raise ValueError(f"Unsupported special-arm control mode {requested!r}.")
    mode = requested or (
        SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
        if fix_arm_enabled and init_arm_enabled
        else SPECIAL_ARM_CONTROL_MODE_MANUAL
    )
    if mode == SPECIAL_ARM_CONTROL_MODE_FOOT_PAD and not (
        fix_arm_enabled and init_arm_enabled
    ):
        raise ValueError(
            "--special-arm-control-mode foot-pad requires both --fix-arm "
            "and --init-arm-available."
        )
    return mode


def _require_evdev() -> Any:
    try:
        import evdev
    except ImportError as exc:  # pragma: no cover - depends on the host environment.
        raise FootPadUnavailableError(
            "Python package 'evdev' is unavailable; install "
            "hardware/foot_pad/requirements.txt before starting the foot-pad service."
        ) from exc
    return evdev


def _require_four_pedal_config() -> None:
    """Verify that the independently managed service is configured for all pads."""

    try:
        from hardware.foot_pad.service import default_config_path, load_config

        config_path = default_config_path()
        config = load_config(config_path)
    except Exception as exc:
        raise FootPadUnavailableError(
            "a valid four-pedal foot-pad config is unavailable"
        ) from exc
    if int(config.get("count", 0) or 0) != 4:
        raise FootPadUnavailableError(
            f"foot-pad config {config_path} does not contain four calibrated pedals"
        )


class FootPadSpecialArmInput:
    """Read F13--F16 press edges from the foot-pad service's virtual device."""

    def __init__(self, device: Any, evdev: Any) -> None:
        self._device = device
        self._evdev = evdev

    @classmethod
    def open_online(cls) -> "FootPadSpecialArmInput":
        _require_four_pedal_config()
        evdev = _require_evdev()
        expected = set(FOOT_PAD_REQUIRED_KEY_CODES)
        matches: list[Any] = []
        for event_path in evdev.list_devices():
            try:
                device = evdev.InputDevice(event_path)
                if str(device.name or "") != FOOT_PAD_DEVICE_NAME:
                    device.close()
                    continue
                capabilities = device.capabilities(absinfo=False)
                key_codes = {
                    int(code)
                    for code in capabilities.get(evdev.ecodes.EV_KEY, ())
                }
                if expected.issubset(key_codes):
                    matches.append(device)
                else:
                    device.close()
            except (OSError, ValueError):
                continue
        if not matches:
            raise FootPadUnavailableError(
                f"virtual keyboard {FOOT_PAD_DEVICE_NAME!r} with F13--F16 was not found"
            )
        for extra in matches[1:]:
            extra.close()
        return cls(matches[0], evdev)

    @classmethod
    def wait_until_online(
        cls,
        *,
        retry_sec: float = 1.0,
        output: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "FootPadSpecialArmInput":
        """Block before real-teleop construction until the service is healthy."""

        stream = sys.stderr if output is None else output
        last_message = ""
        while True:
            try:
                source = cls.open_online()
            except FootPadUnavailableError as exc:
                message = str(exc)
                if message != last_message:
                    print("[FootPad] foot-pad mode is waiting for its virtual keyboard.", file=stream)
                    print(f"[FootPad] {message}", file=stream)
                    print(f"[FootPad] Start it in another terminal: {FOOT_PAD_START_COMMAND}", file=stream)
                    stream.flush()
                    last_message = message
                sleep(max(0.1, float(retry_sec)))
                continue
            print(
                f"[FootPad] online: {FOOT_PAD_DEVICE_NAME} (F13=fix-left, F14=fix-right, "
                "F15=init-left, F16=init-right).",
                file=stream,
            )
            stream.flush()
            return source

    def close(self) -> None:
        device = self._device
        self._device = None
        if device is None:
            return
        try:
            device.close()
        except OSError:
            pass

    def consume_events(self) -> list[str]:
        """Return new pedal press edges; releases and key-repeat events are ignored."""

        if self._device is None:
            return []
        try:
            events = self._device.read()
        except BlockingIOError:
            return []
        except OSError:
            self.close()
            return []
        event_type = int(self._evdev.ecodes.EV_KEY)
        mapped: list[str] = []
        for event in events:
            if int(getattr(event, "type", -1)) != event_type:
                continue
            if int(getattr(event, "value", 0)) != 1:
                continue
            action = FOOT_PAD_EVENT_BY_KEY_CODE.get(int(getattr(event, "code", -1)))
            if action:
                mapped.append(action)
        return mapped


__all__ = [
    "FOOT_PAD_DEVICE_NAME",
    "FOOT_PAD_EVENT_BY_KEY_CODE",
    "FOOT_PAD_REQUIRED_KEY_CODES",
    "FOOT_PAD_START_COMMAND",
    "FootPadSpecialArmInput",
    "FootPadUnavailableError",
    "SPECIAL_ARM_CONTROL_MODE_CHOICES",
    "SPECIAL_ARM_CONTROL_MODE_FOOT_PAD",
    "SPECIAL_ARM_CONTROL_MODE_MANUAL",
    "resolve_special_arm_control_mode",
]
