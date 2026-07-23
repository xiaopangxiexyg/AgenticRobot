"""Canonical Cartesian-action replay entrypoint for MuJoCo.

Use ``python -m sim.mujoco.replay.runner`` only for explicit kinematic qpos
debugging of recorded joint state/targets.
"""

from __future__ import annotations

from .umi_runner import main


if __name__ == "__main__":
    main()
