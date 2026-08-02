"""Compatibility facade for the organized prepared-training subsystem.

New implementation code lives in :mod:`dataset_interface.prepared_training`.
This stable import surface keeps callers independent from the internal file
organization while the provisional artifact is exercised by real loaders.
"""

from .prepared_training import (
    PREPARED_TRAINING_VIEW_SCHEMA_V1,
    LoadedPreparedTrainingView,
    PreparedSourceDataset,
    PreparedTrainingViewBuildResult,
    PreparedTrainingViewError,
    build_prepared_training_view,
    load_prepared_training_view,
)
from .prepared_training.models import (
    PREPARED_SOURCE_DATASETS_DOMAIN_V1,
)

__all__ = [
    "LoadedPreparedTrainingView",
    "PREPARED_SOURCE_DATASETS_DOMAIN_V1",
    "PREPARED_TRAINING_VIEW_SCHEMA_V1",
    "PreparedSourceDataset",
    "PreparedTrainingViewBuildResult",
    "PreparedTrainingViewError",
    "build_prepared_training_view",
    "load_prepared_training_view",
]
