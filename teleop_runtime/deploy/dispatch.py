from __future__ import annotations

from typing import Mapping

from ..action import split_effector_name
from ..interfaces import EffectorBackend
from ..schemas import TeleopCommand


# Real-launcher boundary:
# This file should NOT move to robot_motion. robot_motion should own common
# action/deploy preparation up to TeleopCommand. teleop_runtime owns dispatching
# TeleopCommand to real RM65/Inspire/DMTAC backends. A MuJoCo deploy launcher
# should implement an equivalent dispatcher in sim-specific code.
def dispatch_teleop_command(
    command: TeleopCommand,
    *,
    arm_backends: Mapping[str, EffectorBackend],
    hand_backends: Mapping[str, EffectorBackend],
) -> None:
    for effector_name, effector_command in command.commands.items():
        side, kind = split_effector_name(effector_name)
        if kind == "arm":
            backend = arm_backends.get(side)
        else:
            backend = hand_backends.get(side)
        if backend is None:
            raise KeyError(f"Deploy backend missing for effector {effector_name!r}.")
        backend.send_command(effector_command)
