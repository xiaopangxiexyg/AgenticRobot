"""Temporary end-to-end PPO smoke pipeline for the current Cube-Lift runtime.

The package is imported by both the training CLI and its test harness before
those modules import PyTorch.  Load the environment's C++ runtime at this
boundary so optional Pinocchio remains usable regardless of whether callers
import the numerical stack before constructing a MuJoCo environment.  Without
this ordering guarantee, PyTorch can load the system ``libstdc++`` first and
Pinocchio then fails with a missing ``CXXABI`` symbol; the pose adapter falls
back to a deliberately less capable Jacobian solver and produces a different
grasp trajectory.
"""

from __future__ import annotations

import ctypes
import os
import sys


def _preload_environment_cxx_runtime() -> None:
    """Make the conda runtime visible before PyTorch/Pinocchio are imported.

    This is best-effort and intentionally side-effect limited: if the
    interpreter has no environment-local runtime (for example a system-only
    installation), normal dependency discovery and the existing explicit
    pose-IK fallback remain unchanged.  ``RTLD_GLOBAL`` is required because
    Pinocchio's extension resolves C++ ABI symbols from the process global
    namespace.  Loading by absolute path also avoids depending on a caller's
    ``LD_LIBRARY_PATH``.
    """

    runtime_path = os.path.join(sys.prefix, "lib", "libstdc++.so.6")
    if not os.path.isfile(runtime_path):
        return
    try:
        global_flag = int(getattr(os, "RTLD_GLOBAL", 0))
        now_flag = int(getattr(os, "RTLD_NOW", 0))
        ctypes.CDLL(runtime_path, mode=global_flag | now_flag)
    except OSError:
        # Do not make importing the compatibility/training package fail on a
        # platform whose C++ runtime is already provided by the system.
        return


_preload_environment_cxx_runtime()

from .config import PPOConfig, RewardShapingConfig

__all__ = ["PPOConfig", "RewardShapingConfig"]
