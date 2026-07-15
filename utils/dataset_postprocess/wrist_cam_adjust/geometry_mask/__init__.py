"""Geometry-projection masks for wrist RGB background adjustment."""

from __future__ import annotations

from .dataset import DatasetAdjustResult, adjust_dataset
from .mask import (
    GEOMETRY_BACKGROUND_MASK_NAME,
    GEOMETRY_KEEP_MASK_NAME,
    GEOMETRY_MASK_FAMILY,
    STATIC_BODY_MASK_NAME,
    TABLE_GEOMETRY_MASK_NAME,
    RayGrid,
    background_mask_from_keep,
    build_ray_grid,
    combine_keep_mask,
    load_binary_mask,
    mask_stats,
    render_mask_output,
    table_keep_mask,
    table_keep_mask_polygon,
    table_perimeter_points,
)
from .video import (
    TABLE_MASK_MODE_CHOICES,
    FrameMask,
    MaskBuildConfig,
    VideoAdjustResult,
    WristMaskBuilder,
    process_video,
)

__all__ = [
    "DatasetAdjustResult",
    "adjust_dataset",
    "GEOMETRY_BACKGROUND_MASK_NAME",
    "GEOMETRY_KEEP_MASK_NAME",
    "GEOMETRY_MASK_FAMILY",
    "STATIC_BODY_MASK_NAME",
    "TABLE_GEOMETRY_MASK_NAME",
    "RayGrid",
    "background_mask_from_keep",
    "build_ray_grid",
    "combine_keep_mask",
    "load_binary_mask",
    "mask_stats",
    "render_mask_output",
    "table_keep_mask",
    "table_keep_mask_polygon",
    "table_perimeter_points",
    "TABLE_MASK_MODE_CHOICES",
    "FrameMask",
    "MaskBuildConfig",
    "VideoAdjustResult",
    "WristMaskBuilder",
    "process_video",
]
