"""Compatibility shim for geometry projection dataset processing.

New code should import from dataset_postprocess.wrist_cam_adjust.geometry_mask.dataset.
"""

from __future__ import annotations

from .geometry_mask.dataset import *  # noqa: F401,F403

