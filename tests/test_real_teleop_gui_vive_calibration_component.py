from __future__ import annotations

from types import SimpleNamespace

from teleop_runtime.real_teleop.gui_vive_calibration import (
    vive_calibration_profile_ready,
    vive_calibration_profile_status_text,
    vive_calibration_profile_ui_state,
    vive_calibration_tracker_roles,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def test_legacy_vive_calibration_gui_helpers_are_direct_aliases() -> None:
    assert legacy_entrypoint._vive_calibration_tracker_roles is vive_calibration_tracker_roles
    assert legacy_entrypoint._vive_calibration_profile_ready is vive_calibration_profile_ready
    assert (
        legacy_entrypoint._vive_calibration_profile_status_text
        is vive_calibration_profile_status_text
    )
    assert legacy_entrypoint._vive_calibration_profile_ui_state is vive_calibration_profile_ui_state


class _Calibration:
    def __init__(self, *, desk_ready: bool, offsets: set[str]) -> None:
        self._desk_ready = desk_ready
        self.tracker_offsets = {role: object() for role in offsets}

    def world_to_desk_pose(self):
        return object() if self._desk_ready else None


def test_extracted_vive_calibration_gui_payload_reports_missing_offsets_without_side_effects() -> None:
    calibration = _Calibration(desk_ready=True, offsets={"left_wrist"})
    args = SimpleNamespace(
        gui_last_vive_calibration_profile="/tmp/latest.json",
        load_profile="",
    )
    components = SimpleNamespace(
        active_sides=("left", "right"),
        session=SimpleNamespace(calibration=calibration),
    )

    assert vive_calibration_tracker_roles(("left", "right", "ignored")) == (
        "left_wrist",
        "right_wrist",
    )
    assert vive_calibration_profile_ui_state(
        args,
        components,
        status="等待校准",
    ) == {
        "vive_calibration_profile_path": "/tmp/latest.json",
        "vive_calibration_profile_status": "等待校准\ndesk=ok tracker_offsets=missing right_wrist",
        "vive_calibration_profile_ready": False,
        "vive_calibration_profile_loaded": False,
    }
