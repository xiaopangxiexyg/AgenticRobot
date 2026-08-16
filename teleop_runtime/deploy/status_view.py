"""Read-only console status formatting for policy deployment.

This module may inspect resource snapshots for operator visibility, but it does
not start, advance, command, stop, or close any deployment resource.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def compact_log_value(value: object, *, max_chars: int = 90) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def camera_status_text(
    camera_sources: Mapping[str, Any],
    camera_previewer: Any | None,
) -> str:
    if not camera_sources:
        return "camera=off"
    bits: list[str] = []
    if camera_previewer is not None and camera_previewer.disabled_reason:
        bits.append(
            "preview=disabled:"
            + compact_log_value(camera_previewer.disabled_reason, max_chars=80)
        )
    for camera_key, camera_source in sorted(camera_sources.items()):
        try:
            info = camera_source.device_info()
        except Exception as exc:  # noqa: BLE001 - status must remain observable.
            bits.append(f"{camera_key}=status_error:{type(exc).__name__}")
            continue
        frame_count = int(info.get("frame_count", 0) or 0)
        last_error = info.get("last_error")
        if last_error:
            bits.append(
                f"{camera_key}=frames:{frame_count},err:"
                f"{compact_log_value(last_error, max_chars=70)}"
            )
        else:
            bits.append(f"{camera_key}=frames:{frame_count}")
    return "camera=" + ";".join(bits)


def hand_state_snapshot(
    backends: Mapping[str, Any],
    *,
    history_poller: Any | None = None,
    timestamp: float | None = None,
) -> dict[str, dict[str, Any]]:
    if history_poller is not None and timestamp is not None:
        return history_poller.latest_by_side(float(timestamp))
    return {side: backend.get_state() for side, backend in backends.items()}


def compact_faults(faults: list[str], *, limit: int = 4) -> str:
    if not faults:
        return "none"
    shown = faults[:limit]
    suffix = "" if len(faults) <= limit else f",+{len(faults) - limit}"
    return ";".join(shown) + suffix


def policy_status_text(policy: Any) -> str:
    wrapped_policy = getattr(policy, "policy", None)
    status = getattr(wrapped_policy, "status_text", None)
    if isinstance(status, str) and status:
        return status
    if callable(status):
        text = status()
        if text:
            return str(text)
    return ""


def print_controls(
    *,
    send_actions: bool,
    auto_start: bool,
    wait_for_ready_confirm: bool,
    fixed_init_enabled: bool = False,
    policy_qpos_init_enabled: bool = False,
) -> None:
    print("\n" + "=" * 60)
    print("teleop_runtime policy deployment loop started")
    print(f"  send_actions={'on' if send_actions else 'off(dry-run)'}")
    if (fixed_init_enabled or policy_qpos_init_enabled) and auto_start:
        initial_policy_state = "paused(auto-start after init)"
    else:
        initial_policy_state = "active" if auto_start else "paused"
    print(f"  initial_policy_state={initial_policy_state}")
    if wait_for_ready_confirm:
        print("  ready_confirm=on(A/a required after startup init)")
    if fixed_init_enabled:
        print("  fixed_init=on(H first-frame qpos, H execute T, then startup init)")
        print(
            "Controls: H=pause policy or fixed-init step, g=start after "
            "fixed-init, r=manual safety re-arm, A=confirm, q=quit"
        )
    elif policy_qpos_init_enabled:
        print("  policy_qpos_init=on(g moves to dataset frame qpos, then startup init)")
        print(
            "Controls: g=start qpos init / resume, r=manual safety re-arm, "
            "A=confirm ready, h=pause+clear queue, q=quit"
        )
    else:
        print(
            "Controls: g=start/resume, r=manual safety re-arm, A=confirm "
            "ready/start policy, h=pause+clear queue, q=quit"
        )
    print("=" * 60 + "\n")
