from __future__ import annotations

import argparse

import numpy as np

from teleop.rm65_inspire_tac_portable.retargeting import (
    OfflineVisionProSource,
    RM65InspireTacRetargeter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RM65 Inspire Tac retargeting demo.")
    parser.add_argument("offline_path", help="Path to a recorded AVP stream pickle file.")
    parser.add_argument("--mode", default="both", choices=["left", "right", "both"])
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    retargeter = RM65InspireTacRetargeter(
        mode=args.mode,
        source=OfflineVisionProSource(args.offline_path),
    )

    for step in range(args.steps):
        result = retargeter.retarget(include_keypoints=True)
        print(f"[step {step}] timestamp={result.timestamp}")
        for side in ("left", "right"):
            side_result = getattr(result, side)
            if side_result is None:
                continue
            translation = np.round(side_result.wrist_pose_robot[:3, 3], 4)
            print(
                f"  {side}: wrist={translation.tolist()} "
                f"hand_qpos_shape={side_result.retargeted_hand_qpos.shape} "
                f"hardware_shape={side_result.hardware_hand_qpos.shape}"
            )


if __name__ == "__main__":
    main()
