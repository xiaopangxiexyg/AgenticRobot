"""GUI-side input adapter for the virtual foot-pad keyboard.

The hardware service remains an independently managed process.  It exposes a
virtual ``DexTeleop foot pads`` keyboard that advertises F13/F14 for two pads
or F13--F16 for four pads; this module only checks that keyboard and translates
press edges into existing GUI actions.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


SPECIAL_ARM_CONTROL_MODE_MANUAL = "manual"
SPECIAL_ARM_CONTROL_MODE_FOOT_PAD = "foot-pad"
SPECIAL_ARM_CONTROL_MODE_CHOICES = (
    SPECIAL_ARM_CONTROL_MODE_MANUAL,
    SPECIAL_ARM_CONTROL_MODE_FOOT_PAD,
)

FOOT_PAD_DEVICE_NAME = "DexTeleop foot pads"
FOOT_PAD_START_COMMAND = "python -m hardware.foot_pad --count 4"
FOOT_PAD_SUPPORTED_COUNTS = frozenset((2, 4))
FOOT_PAD_REQUIRED_KEY_CODES = frozenset((183, 184, 185, 186))  # F13--F16
FOOT_PAD_REQUIRED_KEY_CODES_BY_COUNT = {
    2: frozenset((183, 184)),
    4: FOOT_PAD_REQUIRED_KEY_CODES,
}
FOOT_PAD_EVENT_BY_KEY_CODE = {
    183: "fix_arm_left",
    184: "fix_arm_right",
    185: "init_arm_left",
    186: "init_arm_right",
}
FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT = {
    2: {183: "fix_arm_left", 184: "fix_arm_right"},
    4: FOOT_PAD_EVENT_BY_KEY_CODE,
}


class FootPadUnavailableError(RuntimeError):
    """The independently managed virtual foot-pad keyboard is absent."""


def _input_permission_hint() -> str:
    event_nodes = sorted(Path("/dev/input").glob("event*"))
    uinput = Path("/dev/uinput")
    event_unreadable = bool(event_nodes) and not any(
        os.access(path, os.R_OK) for path in event_nodes
    )
    uinput_unwritable = uinput.exists() and not os.access(uinput, os.W_OK)
    if not (event_unreadable or uinput_unwritable):
        return ""
    python = sys.executable or "python"
    return (
        "physical input/uinput nodes exist but are not accessible to this user; "
        "add the user to the input group or run: "
        f"sudo -E {python} -m teleop_runtime.configure_devices --fix-permissions"
    )


def foot_pad_count_from_key_codes(key_codes: Any) -> int | None:
    """Infer the active topology from the daemon's virtual-key capabilities.

    The service creates a virtual keyboard whose advertised F13--F16 set is
    the authoritative *live* topology.  Looking at the capabilities instead
    of only the saved calibration file lets teleop recover when a two-pad
    daemon was started after argument parsing, or when the saved file is
    stale/missing.  Four pads are checked first because their key set contains
    the two-pad subset.
    """

    try:
        available = {int(code) for code in key_codes}
    except (TypeError, ValueError):
        return None
    for count in (4, 2):
        if FOOT_PAD_REQUIRED_KEY_CODES_BY_COUNT[count].issubset(available):
            return count
    return None


def resolve_special_arm_control_mode(
    requested_mode: str | None,
    *,
    fix_arm_enabled: bool,
    init_arm_enabled: bool,
    foot_pad_count: int | None = None,
) -> str:
    """Resolve the mode for the configured 4-pedal or 2-pedal topology.

    Four pedals retain the original fix-left/fix-right/init-left/init-right
    mapping.  With exactly two configured pedals, the pedals emit only
    fix-left/fix-right.  This function selects the additional pedal mapping;
    it does not enable or disable the Fix/Init controllers or their GUI and
    keyboard controls.  Any other known count selects manual mode so the caller
    can continue without foot-pad hardware.
    """

    requested = str(requested_mode or "").strip()
    if requested and requested not in SPECIAL_ARM_CONTROL_MODE_CHOICES:
        raise ValueError(f"Unsupported special-arm control mode {requested!r}.")
    count = None if foot_pad_count is None else int(foot_pad_count)
    if count is not None and count not in FOOT_PAD_SUPPORTED_COUNTS:
        return SPECIAL_ARM_CONTROL_MODE_MANUAL
    if requested:
        mode = requested
    elif count == 2:
        mode = (
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
            if fix_arm_enabled
            else SPECIAL_ARM_CONTROL_MODE_MANUAL
        )
    else:
        mode = (
            SPECIAL_ARM_CONTROL_MODE_FOOT_PAD
            if fix_arm_enabled and init_arm_enabled
            else SPECIAL_ARM_CONTROL_MODE_MANUAL
        )
    if mode == SPECIAL_ARM_CONTROL_MODE_FOOT_PAD:
        if count == 2:
            if not fix_arm_enabled:
                raise ValueError(
                    "--special-arm-control-mode foot-pad with two pedals "
                    "requires --fix-arm."
                )
        elif not (fix_arm_enabled and init_arm_enabled):
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


def _load_foot_pad_config() -> tuple[Any, int]:
    """Load the service config and return its path and configured count."""

    try:
        from hardware.foot_pad.service import default_config_path, load_config

        config_path = default_config_path()
        config = load_config(config_path)
    except Exception as exc:
        raise FootPadUnavailableError("a valid foot-pad config is unavailable") from exc
    try:
        count = int(config.get("count", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise FootPadUnavailableError("foot-pad config has an invalid pedal count") from exc
    return config_path, count


def configured_foot_pad_count() -> int | None:
    """Return the saved count without touching evdev or starting hardware."""

    try:
        _config_path, count = _load_foot_pad_config()
    except FootPadUnavailableError:
        return None
    return count


def _require_supported_foot_pad_config() -> tuple[Any, int]:
    config_path, count = _load_foot_pad_config()
    if count not in FOOT_PAD_SUPPORTED_COUNTS:
        raise FootPadUnavailableError(
            f"foot-pad config {config_path} has {count} pedals; teleop supports only 2 or 4"
        )
    return config_path, count


def _sysfs_input_device_name(event_path: str | os.PathLike[str]) -> str:
    """Read an event node's kernel name without opening the protected node."""

    event_name = Path(event_path).name
    try:
        return (
            Path("/sys/class/input")
            .joinpath(event_name, "device", "name")
            .read_text(encoding="utf-8")
            .strip()
        )
    except OSError:
        return ""


class FootPadSpecialArmInput:
    """Read F13--F16 press edges from the foot-pad service's virtual device."""

    def __init__(self, device: Any, evdev: Any, *, foot_pad_count: int = 4) -> None:
        count = int(foot_pad_count)
        if count not in FOOT_PAD_SUPPORTED_COUNTS:
            raise ValueError(f"foot_pad_count must be 2 or 4, got {count}")
        self._device = device
        self._evdev = evdev
        self.foot_pad_count = count
        self._event_by_key_code = FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT[count]
        self._closed = False

    @classmethod
    def waiting_for_service(
        cls,
        *,
        foot_pad_count: int | None = None,
    ) -> "FootPadSpecialArmInput":
        """Create an offline adapter that never scans from the poll path.

        The GUI keeps this object so foot-pad status stays visible.  Rebinding
        is intentionally explicit: only the user's "retry failed devices"
        action calls :meth:`open_online`.
        """

        count = (
            configured_foot_pad_count()
            if foot_pad_count is None
            else int(foot_pad_count)
        )
        if count not in FOOT_PAD_SUPPORTED_COUNTS:
            count = 4
        return cls(None, None, foot_pad_count=int(count))

    @classmethod
    def open_online(cls) -> "FootPadSpecialArmInput":
        # A saved assignment is useful for diagnostics and for starting the
        # forwarding daemon, but the live virtual keyboard is authoritative
        # once it exists.  In particular, a two-pad daemon must be accepted
        # even when parsing happened before its config was written or while an
        # old four-pad config is still present.
        configured_count = configured_foot_pad_count()
        evdev = _require_evdev()
        matches: list[tuple[Any, int]] = []
        inaccessible_matches: list[tuple[str, BaseException]] = []
        event_paths = list(evdev.list_devices())
        for event_path in event_paths:
            # Most hosts expose many event nodes, and constructing an evdev
            # InputDevice may perform several blocking ioctls.  The kernel
            # name is available through sysfs without opening the device, so
            # reject known non-foot-pad nodes before the expensive probe.
            sysfs_name = _sysfs_input_device_name(event_path)
            if sysfs_name and sysfs_name != FOOT_PAD_DEVICE_NAME:
                continue
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
                live_count = foot_pad_count_from_key_codes(key_codes)
                if live_count is not None:
                    matches.append((device, live_count))
                else:
                    device.close()
            except PermissionError as exc:
                if _sysfs_input_device_name(event_path) == FOOT_PAD_DEVICE_NAME:
                    inaccessible_matches.append((str(event_path), exc))
                continue
            except (OSError, ValueError):
                continue
        if not matches:
            if inaccessible_matches:
                blocked = ", ".join(path for path, _exc in inaccessible_matches)
                python = sys.executable or "python"
                raise FootPadUnavailableError(
                    f"virtual keyboard {FOOT_PAD_DEVICE_NAME!r} exists at {blocked}, "
                    "but the current user cannot read its event node; run while the "
                    "foot-pad service is active: "
                    f"sudo -E {python} -m teleop_runtime.configure_devices "
                    "--fix-permissions"
                ) from inaccessible_matches[0][1]
            permission_hint = _input_permission_hint()
            if permission_hint:
                raise FootPadUnavailableError(permission_hint)
            if configured_count in FOOT_PAD_SUPPORTED_COUNTS:
                configured_hint = f"configured {configured_count}-pedal"
                remediation = f"python -m hardware.foot_pad --count {configured_count}"
            else:
                configured_hint = "no saved 2/4-pedal config"
                remediation = "python -m hardware.foot_pad --count 2 --recalibrate"
            raise FootPadUnavailableError(
                f"virtual keyboard {FOOT_PAD_DEVICE_NAME!r} with a supported "
                f"2/4-pedal key set was not found ({configured_hint}); "
                f"start/calibrate it with: {remediation}"
            )
        # Prefer the four-key match if a stale duplicate UInput device is
        # present.  ``matches`` is non-empty and every entry has a supported
        # live count, so this also handles a missing/stale saved config.
        selected_device, detected_count = max(matches, key=lambda item: item[1])
        for device, _count in matches:
            if device is not selected_device:
                device.close()
        return cls(selected_device, evdev, foot_pad_count=detected_count)

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
                    configured_count = configured_foot_pad_count()
                    start_command = (
                        f"python -m hardware.foot_pad --count {configured_count}"
                        if configured_count in FOOT_PAD_SUPPORTED_COUNTS
                        else FOOT_PAD_START_COMMAND
                    )
                    print(f"[FootPad] Start it in another terminal: {start_command}", file=stream)
                    stream.flush()
                    last_message = message
                sleep(max(0.1, float(retry_sec)))
                continue
            mapping = (
                "F13=fix-left, F14=fix-right"
                if source.foot_pad_count == 2
                else "F13=fix-left, F14=fix-right, F15=init-left, F16=init-right"
            )
            print(f"[FootPad] online: {FOOT_PAD_DEVICE_NAME} ({mapping}).", file=stream)
            stream.flush()
            return source

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._drop_device()

    @property
    def is_online(self) -> bool:
        """Whether a readable virtual keyboard is currently bound.

        The offline adapter intentionally survives a device disconnect so the
        GUI can keep showing its failed status until a manual retry replaces
        it. Availability must therefore use this property rather than checking
        whether the adapter object itself is non-null.
        """

        return bool(not self._closed and self._device is not None)

    def _drop_device(self, *, expected_device: Any = None) -> None:
        if expected_device is not None and self._device is not expected_device:
            return
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

        if self._closed or self._device is None or self._evdev is None:
            return []
        device = self._device
        evdev = self._evdev
        try:
            events = device.read()
        except BlockingIOError:
            return []
        except OSError:
            # The daemon recreates its UInput device after a physical pad is
            # unplugged. Drop only this stale fd and remain offline; the GUI's
            # explicit retry action is the only path that scans for a replacement.
            self._drop_device(expected_device=device)
            return []
        event_type = int(evdev.ecodes.EV_KEY)
        mapped: list[str] = []
        for event in events:
            if int(getattr(event, "type", -1)) != event_type:
                continue
            if int(getattr(event, "value", 0)) != 1:
                continue
            action = self._event_by_key_code.get(int(getattr(event, "code", -1)))
            if action:
                mapped.append(action)
        return mapped


__all__ = [
    "FOOT_PAD_DEVICE_NAME",
    "FOOT_PAD_EVENT_BY_KEY_CODE",
    "FOOT_PAD_EVENT_BY_KEY_CODE_BY_COUNT",
    "FOOT_PAD_REQUIRED_KEY_CODES",
    "FOOT_PAD_REQUIRED_KEY_CODES_BY_COUNT",
    "FOOT_PAD_START_COMMAND",
    "FOOT_PAD_SUPPORTED_COUNTS",
    "FootPadSpecialArmInput",
    "FootPadUnavailableError",
    "SPECIAL_ARM_CONTROL_MODE_CHOICES",
    "SPECIAL_ARM_CONTROL_MODE_FOOT_PAD",
    "SPECIAL_ARM_CONTROL_MODE_MANUAL",
    "configured_foot_pad_count",
    "foot_pad_count_from_key_codes",
    "resolve_special_arm_control_mode",
]
