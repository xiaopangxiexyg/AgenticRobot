"""Pure UMI control-availability projection.

The policy consumes an immutable application snapshot and returns presentation
controls only. It performs no device reads, recorder operations, calibration,
or command dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UmiControlState:
    calibration_expired: bool = False
    recording: bool = False
    collection_blocked: bool = False
    plus_save_available: bool = False
    plus_save_in_flight: bool = False
    foot_pad_online: bool = False
    foot_pad_test_active: bool = False


def active_controls(state: UmiControlState) -> list[str]:
    """Return controls visible for one UMI application snapshot."""

    if not isinstance(state, UmiControlState):
        raise TypeError("state must be a UmiControlState.")
    if state.calibration_expired:
        controls = ["q", "task_select", "umi_recalibrate", "angle_recalibrate"]
        if state.plus_save_available:
            controls.append("plus_curation_save")
        return controls

    controls: list[str] = []
    if not state.recording:
        controls.append("task_select")
        if not state.collection_blocked:
            controls.append("episode_start")
        controls.extend(["episode_delete", "device_retry"])
    else:
        controls.append("episode_stop")
    controls.append("q")

    # Foot pads supplement these operator controls; they do not hide the
    # equivalent keyboard/GUI operations.
    controls.extend(
        [
            "c",
            "g",
            "fix_arm_toggle",
            "init_arm_toggle",
            "fix_arm_select",
            "init_arm_select",
        ]
    )
    if not state.recording:
        controls.extend(["umi_recalibrate", "angle_recalibrate"])
        if state.foot_pad_online:
            controls.append(
                "foot_pad_test_stop"
                if state.foot_pad_test_active
                else "foot_pad_test_start"
            )
    if state.plus_save_available:
        controls.append("plus_curation_save")
    if state.plus_save_in_flight:
        controls = [control for control in controls if control != "episode_delete"]
    return controls


__all__ = ["UmiControlState", "active_controls"]
