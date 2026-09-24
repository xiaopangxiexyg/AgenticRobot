"""Explicit Station policy assembly for offline control-input tests."""

from types import SimpleNamespace

from control_station.real_teleop import control_availability
from control_station.real_teleop.input_mapping import prioritize_input_batch
from control_station.real_teleop_application import StationProgressApplication
from teleop_runtime.real_teleop.operator_control_runtime import RealTeleopControlInput


def bind_control_application(control: RealTeleopControlInput) -> RealTeleopControlInput:
    """Bind real pure policies, without startup/device/HITL worker assembly."""

    application = getattr(control, "_progress_application", None)
    if application is None:
        application = StationProgressApplication(
            getattr(control, "args", SimpleNamespace())
        )
    control.bind_application_ports(
        progress_application=application,
        input_batch_prioritizer=prioritize_input_batch,
        control_policy=control_availability,
    )
    return control


def build_control_input(**kwargs: object) -> RealTeleopControlInput:
    return bind_control_application(RealTeleopControlInput(**kwargs))
