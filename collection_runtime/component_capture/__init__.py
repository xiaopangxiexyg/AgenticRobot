"""Version-neutral component-stream capture runtime.

The current persisted contracts are provisional v2.5-p schemas, but callers
use capability-oriented runtime names from this package. Versioned contract
names remain confined to ``dataset_contracts`` and compatibility shims.
"""

from .action_timing import (
    ComponentActionTiming,
    attach_projected_component_action_timing,
    attach_projected_component_action_timings,
)
from .episode_store import (
    ComponentCaptureCommitReport,
    ComponentCaptureEpisodeStore,
    ComponentCaptureSealReport,
    ComponentStreamInput,
    validate_component_stream_profile,
)
from .parquet import ParquetChunkEpisodeWriter
from .producer import (
    ComponentStreamBindings,
    ProducedComponentSample,
    validate_produced_values,
    validate_producer_bindings,
)


__all__ = [
    "ComponentActionTiming",
    "ComponentCaptureCommitReport",
    "ComponentCaptureEpisodeStore",
    "ComponentCaptureSealReport",
    "ComponentStreamBindings",
    "ComponentStreamInput",
    "ParquetChunkEpisodeWriter",
    "ProducedComponentSample",
    "attach_projected_component_action_timing",
    "attach_projected_component_action_timings",
    "validate_component_stream_profile",
    "validate_produced_values",
    "validate_producer_bindings",
]
