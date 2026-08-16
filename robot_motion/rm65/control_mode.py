"""Shared RM65 arm-control mode names and normalization."""

from __future__ import annotations


RM65_ARM_CONTROL_MODE_POSITION = "position"
RM65_ARM_CONTROL_MODE_IMPEDANCE = "impedance"
RM65_ARM_CONTROL_MODES = (
    RM65_ARM_CONTROL_MODE_POSITION,
    RM65_ARM_CONTROL_MODE_IMPEDANCE,
)


def normalize_rm65_arm_control_mode(
    value: object,
    *,
    error_context: str = "",
) -> str:
    """Normalize historical RM65 control-mode aliases to a canonical mode."""

    text = (
        str(value or RM65_ARM_CONTROL_MODE_POSITION).strip().lower().replace("-", "_")
    )
    aliases = {
        "pos": RM65_ARM_CONTROL_MODE_POSITION,
        "movej": RM65_ARM_CONTROL_MODE_POSITION,
        "canfd": RM65_ARM_CONTROL_MODE_POSITION,
        "force": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "force_control": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "admittance": RM65_ARM_CONTROL_MODE_IMPEDANCE,
        "force_admittance": RM65_ARM_CONTROL_MODE_IMPEDANCE,
    }
    mode = aliases.get(text, text)
    if mode not in RM65_ARM_CONTROL_MODES:
        raise ValueError(
            f"Unsupported {error_context}RM65 arm control mode {value!r}; "
            f"expected one of {RM65_ARM_CONTROL_MODES}."
        )
    return mode


__all__ = [
    "RM65_ARM_CONTROL_MODE_IMPEDANCE",
    "RM65_ARM_CONTROL_MODE_POSITION",
    "RM65_ARM_CONTROL_MODES",
    "normalize_rm65_arm_control_mode",
]
