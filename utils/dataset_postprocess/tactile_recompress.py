"""Compatibility imports for collection-runtime-owned tactile recompression.

Offline callers keep the historic module path while the mutation mechanism,
integrity checks, and receipts are owned by :mod:`collection_runtime`.
"""

from collection_runtime.tactile_recompression import (
    COLLECTION_RECHECK_CHUNKS,
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_MIN_FREE_BYTES,
    LOSSLESS_PREDICTOR_V1,
    LOSSLESS_PROFILE_V1,
    TactileRecompressionCancelled,
    TactileRecompressionPlan,
    TactileRecompressionResult,
    _ZstdCodec,
    _active_collection_processes,
    _atomic_write_json,
    clean_uncommitted_tactile_recompression_artifacts,
    plan_tactile_zstd_recompression,
    recompress_tactile_zstd_in_place,
)


__all__ = [
    "COLLECTION_RECHECK_CHUNKS",
    "DEFAULT_CHUNK_FRAMES",
    "DEFAULT_MIN_FREE_BYTES",
    "LOSSLESS_PREDICTOR_V1",
    "LOSSLESS_PROFILE_V1",
    "TactileRecompressionCancelled",
    "TactileRecompressionPlan",
    "TactileRecompressionResult",
    "clean_uncommitted_tactile_recompression_artifacts",
    "plan_tactile_zstd_recompression",
    "recompress_tactile_zstd_in_place",
]
