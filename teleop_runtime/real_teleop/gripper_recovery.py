"""Runtime promotion from angle-trigger fallback to reconnectable Q6 input."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import numpy as np

from ..retargeters import ScalarToGripperRetargeter
from ..sources.hand import AngleTriggerSource, HandQ6Source
from .components import RealTeleopComponents
from .hand_runtime import _gripper_retargeter_for_side


def promote_angle_trigger_gripper_input_to_q6(
    args: argparse.Namespace,
    components: RealTeleopComponents,
    *,
    ready_timeout_sec: float = 2.0,
    discover_runtime_q6_ports: Callable[..., dict[str, str]],
    q6_gripper_retargeter_for_side: Callable[..., ScalarToGripperRetargeter],
) -> tuple[bool, str]:
    """Hot-switch a healthy angle-trigger fallback to newly connected Q6 input."""

    if str(getattr(components, "gripper_input_source", "") or "") != "angle_trigger":
        return False, "当前 gripper input 不是 angle trigger fallback，无需切换。"

    old_sources = [
        source
        for source in components.session.sources
        if isinstance(source, AngleTriggerSource)
    ]
    # Capability-admission startup may preserve the angle-trigger contract
    # and retargeters even when the physical angle reader could not be
    # constructed.  In that state the source is intentionally absent from
    # the session, but a later Q6 reconnect is still a valid recovery path.
    # The replacement assembly below inserts Q6 when no old source exists.

    ports = discover_runtime_q6_ports(args, components.active_sides)
    q6_source = HandQ6Source.from_bimanual(
        left_port=ports.get("left"),
        right_port=ports.get("right"),
        baudrate=int(getattr(args, "hand_reader_baudrate", 115200)),
        timeout=float(getattr(args, "hand_reader_timeout", 0.05)),
        show_raw_lines=bool(getattr(args, "show_hand_raw_lines", False)),
        exo_hand_calibration=bool(getattr(args, "exo_hand_calibration", False)),
        calibration_path=str(getattr(args, "exo_hand_calibration_path", "") or "") or None,
        critical=False,
        strict_expected=False,
    )
    original_session_sources = list(components.session.sources)
    original_session_retargeters = list(components.session.retargeters)
    original_component_source = str(components.gripper_input_source)
    original_left_retargeter = components.left_gripper_retargeter
    original_right_retargeter = components.right_gripper_retargeter
    original_arg_source = getattr(args, "gripper_input_source", None)
    original_loaded_ports = getattr(args, "q6_config_loaded_ports", None)
    original_hand_ports = {
        side: getattr(args, f"{side}_hand_port", None)
        for side in components.active_sides
    }
    try:
        q6_source.start()
        deadline = time.monotonic() + max(0.0, float(ready_timeout_sec))
        last_reason = "等待 Q6 首帧。"
        while True:
            now = time.time()
            channels = q6_source.get_latest_channels()
            failures: list[str] = []
            for side in components.active_sides:
                name = f"{side}_hand_q6"
                channel = channels.get(name)
                if channel is None:
                    failures.append(f"{name}:missing")
                    continue
                age_sec = now - float(channel.timestamp)
                values = np.asarray(channel.value, dtype=float).reshape(-1)
                if not bool(channel.valid):
                    failures.append(f"{name}:invalid")
                elif values.shape != (6,) or not np.all(np.isfinite(values)):
                    failures.append(f"{name}:bad_shape")
                elif age_sec < -0.05 or age_sec > 0.5:
                    failures.append(f"{name}:stale({age_sec:.3f}s)")
            if not failures:
                break
            last_reason = ", ".join(failures)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Q6 did not become ready: {last_reason}")
            time.sleep(0.02)

        replacements: dict[int, ScalarToGripperRetargeter] = {}
        new_by_side: dict[str, ScalarToGripperRetargeter] = {}
        for side in components.active_sides:
            old_retargeter = _gripper_retargeter_for_side(components, side)
            if old_retargeter is None:
                continue
            replacement = q6_gripper_retargeter_for_side(args, side)
            replacements[id(old_retargeter)] = replacement
            new_by_side[side] = replacement
        if not new_by_side:
            raise RuntimeError("no active DMTAC gripper retargeter is available for Q6")

        replacement_sources: list[object] = []
        q6_inserted = False
        for source in components.session.sources:
            if isinstance(source, AngleTriggerSource):
                if not q6_inserted:
                    replacement_sources.append(q6_source)
                    q6_inserted = True
                continue
            if isinstance(source, HandQ6Source):
                continue
            replacement_sources.append(source)
        if not q6_inserted:
            replacement_sources.append(q6_source)

        replacement_retargeters: list[object] = []
        for retargeter in components.session.retargeters:
            replacement = replacements.get(id(retargeter))
            if replacement is None:
                replacement_retargeters.append(retargeter)
                continue
            replacement_retargeters.append(replacement)

        # DMTAC ScalarToGripperRetargeters are intentionally out of the
        # session/Rig command path.  They are dispatched by
        # ``_dispatch_q6_gripper_control`` directly through the DMTAC
        # backend, while Inspire hand retargeters are the hand effectors that
        # belong to ``session.retargeters`` and ``Rig``.  During a hot switch,
        # only replace a gripper retargeter if it was already part of the
        # session.  Do not append replacements merely because a component
        # reference exists: doing so makes the next session command contain
        # ``left_hand``/``right_hand`` without corresponding Rig backends.
        # An existing in-session DMTAC-compatible fixture/consumer is still
        # replaced atomically; the normal real DMTAC topology remains
        # unchanged.

        components.session.sources = replacement_sources
        components.session.retargeters = replacement_retargeters
        object.__setattr__(components, "gripper_input_source", "q6")
        if "left" in new_by_side:
            object.__setattr__(
                components,
                "left_gripper_retargeter",
                new_by_side["left"],
            )
        if "right" in new_by_side:
            object.__setattr__(
                components,
                "right_gripper_retargeter",
                new_by_side["right"],
            )
        args.gripper_input_source = "q6"
        args.q6_config_loaded_ports = dict(ports)
        for side, port in ports.items():
            setattr(args, f"{side}_hand_port", str(port))
    except BaseException:
        components.session.sources = original_session_sources
        components.session.retargeters = original_session_retargeters
        object.__setattr__(
            components,
            "gripper_input_source",
            original_component_source,
        )
        object.__setattr__(
            components,
            "left_gripper_retargeter",
            original_left_retargeter,
        )
        object.__setattr__(
            components,
            "right_gripper_retargeter",
            original_right_retargeter,
        )
        args.gripper_input_source = original_arg_source
        args.q6_config_loaded_ports = original_loaded_ports
        for side, port in original_hand_ports.items():
            setattr(args, f"{side}_hand_port", port)
        try:
            q6_source.close()
        except Exception:
            pass
        raise

    close_errors: list[str] = []
    for source in old_sources:
        try:
            source.close()
        except Exception as exc:  # noqa: BLE001 - new Q6 source is already committed.
            close_errors.append(f"{type(exc).__name__}: {exc}")
    message = (
        "Q6 已连接并切换为活动输入；"
        + ("已替换 angle trigger，" if old_sources else "原缺失的 angle trigger fallback 已补齐，")
        + "左右 gripper retargeter 已切换到 hand_q6。"
    )
    if close_errors:
        message += f" angle trigger 关闭告警：{'; '.join(close_errors)}"
    return True, message


__all__ = ["promote_angle_trigger_gripper_input_to_q6"]
