#!/usr/bin/env python3
import _bootstrap  # noqa: F401

from portable_dmtac_gripper import DMTacGripperDevice, TactileConfig, format_frame_summary


def main() -> None:
    cfg = TactileConfig(
        dev_ids=None,  # None means auto-discover first two sensors
        backend="cpu",
        fields=("distributed_force",),
        max_fps=120,
    )
    with DMTacGripperDevice(tactile=cfg, connect_gripper=False, connect_tactile=True) as dev:
        print("ready:", dev.wait_tactile_ready(timeout_s=10))
        for _ in range(10):
            frames = dev.read_tactile(timeout_ms=500)
            for dev_id, frame in frames.items():
                print(dev_id, format_frame_summary(frame))


if __name__ == "__main__":
    main()
