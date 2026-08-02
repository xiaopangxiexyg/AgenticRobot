"""Reusable gripper-input selection for teleoperation application profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


GRIPPER_INPUT_AUTO = "auto"
GRIPPER_INPUT_Q6 = "q6"
GRIPPER_INPUT_ANGLE_TRIGGER = "angle_trigger"
GRIPPER_INPUT_CHOICES = (
    GRIPPER_INPUT_AUTO,
    GRIPPER_INPUT_Q6,
    GRIPPER_INPUT_ANGLE_TRIGGER,
)


def _normalized_side(value: str) -> str:
    side = str(value).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right.")
    return side


def _normalized_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in GRIPPER_INPUT_CHOICES:
        raise ValueError(
            "requested_source must be auto, q6, or angle_trigger."
        )
    return source


def _device_port(device: Mapping[str, Any]) -> str | None:
    value = (
        device.get("preferred_port")
        or device.get("port")
        or device.get("device")
    )
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _device_aliases(device: Mapping[str, Any]) -> frozenset[str]:
    aliases = {
        str(value).strip()
        for value in tuple(device.get("port_aliases") or ())
        if str(value).strip()
    }
    port = _device_port(device)
    if port is not None:
        aliases.add(port)
    raw_port = device.get("port")
    if raw_port is not None and str(raw_port).strip():
        aliases.add(str(raw_port).strip())
    return frozenset(aliases)


@dataclass(frozen=True)
class SingleGripperInputSelection:
    """Resolved startup input for one logical gripper side."""

    side: str
    requested_source: str
    resolved_source: str
    q6_port: str | None = None
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        side = _normalized_side(self.side)
        requested = _normalized_source(self.requested_source)
        resolved = _normalized_source(self.resolved_source)
        if resolved == GRIPPER_INPUT_AUTO:
            raise ValueError("resolved_source cannot be auto.")
        q6_port = (
            None
            if self.q6_port is None
            else str(self.q6_port).strip() or None
        )
        if resolved == GRIPPER_INPUT_Q6 and q6_port is None:
            raise ValueError("A resolved Q6 selection requires q6_port.")
        if resolved != GRIPPER_INPUT_Q6 and q6_port is not None:
            raise ValueError("Only a Q6 selection may carry q6_port.")
        fallback_reason = (
            None
            if self.fallback_reason is None
            else str(self.fallback_reason).strip() or None
        )
        if requested != GRIPPER_INPUT_AUTO and fallback_reason is not None:
            raise ValueError(
                "fallback_reason is only valid for an automatic selection."
            )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "requested_source", requested)
        object.__setattr__(self, "resolved_source", resolved)
        object.__setattr__(self, "q6_port", q6_port)
        object.__setattr__(self, "fallback_reason", fallback_reason)


def _load_configured_q6_port(
    *,
    side: str,
    config: str | Path,
) -> tuple[str | None, str | None]:
    from .q6_hardware import load_latest_q6_ports

    try:
        ports = load_latest_q6_ports(
            active_sides=(side,),
            config=config,
            allow_missing=True,
        )
    except Exception as exc:  # noqa: BLE001 - auto selection may continue by scanning.
        return None, f"q6_config_error:{type(exc).__name__}:{exc}"
    port = ports.get(side)
    if port is None or not str(port).strip():
        return None, "q6_config_missing"
    return str(port).strip(), None


def _probe_configured_q6(port: str) -> tuple[bool, str | None]:
    from hardware.teleop_hand.q6_assignment import probe_q6_port

    try:
        result = probe_q6_port(str(port))
    except Exception as exc:  # noqa: BLE001 - auto selection may use its fallback.
        return False, f"q6_probe_error:{type(exc).__name__}:{exc}"
    if bool(result.get("reachable", False)):
        return True, None
    return False, str(result.get("error") or "q6_not_reachable")


def _scan_reachable_q6() -> tuple[list[Mapping[str, Any]], str | None]:
    from hardware.teleop_hand.q6_assignment import scan_q6_devices

    try:
        scan = scan_q6_devices(
            probe_all_serial=True,
            show_raw_lines=False,
        )
    except Exception as exc:  # noqa: BLE001 - auto selection may use its fallback.
        return [], f"q6_scan_error:{type(exc).__name__}:{exc}"
    devices = [
        item
        for item in tuple(scan.get("q6_devices") or ())
        if isinstance(item, Mapping) and _device_port(item) is not None
    ]
    return devices, None


def resolve_single_gripper_input(
    *,
    side: str,
    requested_source: str = GRIPPER_INPUT_AUTO,
    q6_port: str | None = None,
    q6_config: str | Path = "latest",
) -> SingleGripperInputSelection:
    """Prefer one reachable Q6 and otherwise use the angle trigger.

    ``auto`` is the only fallback mode.  An explicit ``q6`` request fails if
    no unambiguous reachable Q6 exists, while an explicit ``angle_trigger``
    request performs no Q6 discovery.  This function probes input devices but
    does not construct or start their long-lived readers.
    """

    normalized_side = _normalized_side(side)
    requested = _normalized_source(requested_source)
    if requested == GRIPPER_INPUT_ANGLE_TRIGGER:
        return SingleGripperInputSelection(
            side=normalized_side,
            requested_source=requested,
            resolved_source=GRIPPER_INPUT_ANGLE_TRIGGER,
        )

    configured_port = (
        None if q6_port is None else str(q6_port).strip() or None
    )
    reasons: list[str] = []
    if configured_port is None:
        configured_port, config_error = _load_configured_q6_port(
            side=normalized_side,
            config=q6_config,
        )
        if config_error is not None:
            reasons.append(config_error)

    if configured_port is not None:
        reachable, probe_error = _probe_configured_q6(configured_port)
        if reachable:
            return SingleGripperInputSelection(
                side=normalized_side,
                requested_source=requested,
                resolved_source=GRIPPER_INPUT_Q6,
                q6_port=configured_port,
            )
        reasons.append(
            f"configured_q6_unavailable:{configured_port}:{probe_error or 'unknown'}"
        )

    devices, scan_error = _scan_reachable_q6()
    if scan_error is not None:
        reasons.append(scan_error)
    if configured_port is not None:
        matches = [
            device
            for device in devices
            if configured_port in _device_aliases(device)
        ]
        if len(matches) == 1:
            return SingleGripperInputSelection(
                side=normalized_side,
                requested_source=requested,
                resolved_source=GRIPPER_INPUT_Q6,
                q6_port=_device_port(matches[0]),
            )
    if len(devices) == 1:
        return SingleGripperInputSelection(
            side=normalized_side,
            requested_source=requested,
            resolved_source=GRIPPER_INPUT_Q6,
            q6_port=_device_port(devices[0]),
        )
    if len(devices) > 1:
        raise ValueError(
            f"Detected {len(devices)} reachable Q6 devices but no unambiguous "
            f"{normalized_side}-side assignment. Configure the side mapping or "
            "pass an explicit Q6 port."
        )

    reason = ";".join(reasons) or "no_reachable_q6"
    if requested == GRIPPER_INPUT_Q6:
        raise RuntimeError(f"Requested Q6 input is unavailable: {reason}")
    return SingleGripperInputSelection(
        side=normalized_side,
        requested_source=requested,
        resolved_source=GRIPPER_INPUT_ANGLE_TRIGGER,
        fallback_reason=reason,
    )


__all__ = [
    "GRIPPER_INPUT_ANGLE_TRIGGER",
    "GRIPPER_INPUT_AUTO",
    "GRIPPER_INPUT_CHOICES",
    "GRIPPER_INPUT_Q6",
    "SingleGripperInputSelection",
    "resolve_single_gripper_input",
]
