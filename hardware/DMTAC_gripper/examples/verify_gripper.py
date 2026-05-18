#!/usr/bin/env python3
import _bootstrap  # noqa: F401

from portable_dmtac_gripper import GripperConfig, InspireGripper


def main() -> None:
    cfg = GripperConfig(port="/dev/ttyUSB0", gripper_id=1)
    with InspireGripper(**cfg.__dict__) as gripper:
        state = gripper.get_state()
        print(state)


if __name__ == "__main__":
    main()
