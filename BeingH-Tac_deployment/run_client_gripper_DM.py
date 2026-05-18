from __future__ import annotations

from client_core import run_profile
from profiles import GRIPPER_DM_PROFILE


if __name__ == "__main__":
    raise SystemExit(run_profile(GRIPPER_DM_PROFILE))
