#!/usr/bin/env python3
import _bootstrap  # noqa: F401

from portable_dmtac_gripper import (
    DMTacGripperDevice,
    GripperConfig,
    TactileConfig,
    TactileDenoiseConfig,
    format_snapshot_summary,
)


def main() -> None:
    gripper = GripperConfig(port="/dev/ttyUSB0", gripper_id=1)
    tactile = TactileConfig(
        dev_ids=None,
        backend="cpu",
        fields=("distributed_force",),
        max_fps=120,
        denoise_config=TactileDenoiseConfig(
            mode="delta",
            baseline_frames=20,
            deadband_abs=5e-5,
            temporal_alpha=0.85,
            spatial_blur_ksize=5,
        ),
    )

    with DMTacGripperDevice(gripper=gripper, tactile=tactile) as dev:
        dev.clear_fault()
        dev.set_speed(0.2)
        dev.set_torque(0.2)
        dev.open(wait=True)
        print("tactile ready:", dev.wait_tactile_ready(timeout_s=10))
        dev.warmup_tactile_filter(frames=30, timeout_ms=100)

        dev.close_gripper(speed=0.2, torque=0.2, continuous=True, wait=False)
        for _ in range(100):
            snapshot = dev.read(tactile_timeout_ms=100)
            print(format_snapshot_summary(snapshot))


if __name__ == "__main__":
    main()
