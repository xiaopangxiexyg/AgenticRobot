"""Prepared training snapshots over immutable source datasets."""

from .builder import build_prepared_training_view
from .models import (
    PREPARED_TRAINING_VIEW_SCHEMA_V1,
    LoadedPreparedTrainingView,
    PreparedSourceDataset,
    PreparedTrainingViewBuildResult,
    PreparedTrainingViewError,
)
from .reader import load_prepared_training_view

__all__ = [
    "LoadedPreparedTrainingView",
    "PREPARED_TRAINING_VIEW_SCHEMA_V1",
    "PreparedSourceDataset",
    "PreparedTrainingViewBuildResult",
    "PreparedTrainingViewError",
    "build_prepared_training_view",
    "load_prepared_training_view",
]
