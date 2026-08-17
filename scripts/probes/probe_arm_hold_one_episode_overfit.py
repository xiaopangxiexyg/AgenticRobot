"""Compatibility launcher for the Arm-Hold overfit experiment."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.policy_learning.arm_hold.commands.probe_arm_hold_one_episode_overfit import (
    main,
)


if __name__ == "__main__":
    main()
