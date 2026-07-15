"""Read-only fix-arm and init-arm status views for the real-teleop GUI."""

from __future__ import annotations

from collections.abc import Mapping


def fix_arm_ui_state(components: object) -> dict[str, object]:
    controller = getattr(components, "fix_arm_controller", None)
    if controller is None:
        return {
            "fix_arm_available": False,
            "fix_arm_selected_side": "",
            "fix_arm_fixed_side": "",
            "fix_arm_active": False,
            "fix_arm_recalibrating": False,
            "fix_arm_status": "",
            "fix_arm_sample_count": None,
            "fix_arm_min_samples": None,
        }
    return controller.status_payload()


def init_arm_ui_state(components: object) -> dict[str, object]:
    controller = getattr(components, "init_arm_controller", None)
    if controller is None:
        return {
            "init_arm_available": False,
            "init_arm_selected_side": "",
            "init_arm_active_side": "",
            "init_arm_active": False,
            "init_arm_recalibrating": False,
            "init_arm_status": "",
            "init_arm_sample_count": None,
            "init_arm_min_samples": None,
            "init_arm_profile_path": "",
            "init_arm_arrival_status": "inactive",
            "init_arm_arrival_label": "",
            "init_arm_arrival_reason": "",
            "init_arm_target_position_error_m": None,
            "init_arm_target_rotation_error_deg": None,
            "init_arm_stability_dwell_sec": None,
        }
    return controller.status_payload()


def fix_arm_debug_state(components: object) -> dict[str, object]:
    state = fix_arm_ui_state(components)
    return {
        "available": bool(state.get("fix_arm_available", False)),
        "selected_side": str(state.get("fix_arm_selected_side", "") or ""),
        "fixed_side": str(state.get("fix_arm_fixed_side", "") or ""),
        "active": bool(state.get("fix_arm_active", False)),
        "recalibrating": bool(state.get("fix_arm_recalibrating", False)),
        "status": str(state.get("fix_arm_status", "") or ""),
        "sample_count": state.get("fix_arm_sample_count"),
        "min_samples": state.get("fix_arm_min_samples"),
    }


def fix_arm_status_text(state: Mapping[str, object]) -> str:
    if not bool(state.get("available", False)):
        return ""
    fixed_side = str(state.get("fixed_side", "") or "-")
    selected_side = str(state.get("selected_side", "") or "-")
    mode = "recapturing" if state.get("recalibrating") else "fixed" if state.get("active") else "live"
    sample_count = state.get("sample_count")
    min_samples = state.get("min_samples")
    sample_bits = ""
    if sample_count is not None or min_samples is not None:
        sample_bits = f",samples={sample_count or 0}/{min_samples or 0}"
    return f" {mode}(selected={selected_side},fixed={fixed_side}{sample_bits})"


def init_arm_debug_state(components: object) -> dict[str, object]:
    state = init_arm_ui_state(components)
    return {
        "available": bool(state.get("init_arm_available", False)),
        "selected_side": str(state.get("init_arm_selected_side", "") or ""),
        "active_side": str(state.get("init_arm_active_side", "") or ""),
        "active": bool(state.get("init_arm_active", False)),
        "recalibrating": bool(state.get("init_arm_recalibrating", False)),
        "status": str(state.get("init_arm_status", "") or ""),
        "sample_count": state.get("init_arm_sample_count"),
        "min_samples": state.get("init_arm_min_samples"),
        "arrival_status": str(state.get("init_arm_arrival_status", "") or ""),
        "arrival_label": str(state.get("init_arm_arrival_label", "") or ""),
    }


def init_arm_status_text(state: Mapping[str, object]) -> str:
    if not bool(state.get("available", False)):
        return ""
    active_side = str(state.get("active_side", "") or "-")
    selected_side = str(state.get("selected_side", "") or "-")
    mode = "recapturing" if state.get("recalibrating") else "moving" if state.get("active") else "live"
    sample_count = state.get("sample_count")
    min_samples = state.get("min_samples")
    sample_bits = ""
    if sample_count is not None or min_samples is not None:
        sample_bits = f",samples={sample_count or 0}/{min_samples or 0}"
    return f" {mode}(selected={selected_side},active={active_side}{sample_bits})"


__all__ = [
    "fix_arm_debug_state",
    "fix_arm_status_text",
    "fix_arm_ui_state",
    "init_arm_debug_state",
    "init_arm_status_text",
    "init_arm_ui_state",
]
