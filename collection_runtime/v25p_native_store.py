"""Compatibility imports for the former version-named native-store module."""

from .component_capture.episode_store import (
    V25P_RECORD_CJSON_LAYOUT_V1,
    ComponentCaptureCommitReport,
    ComponentCaptureEpisodeStore,
    ComponentCaptureSealReport,
    ComponentStreamInput,
    validate_component_stream_profile,
)


V25pComponentInput = ComponentStreamInput
V25pNativeCommitReport = ComponentCaptureCommitReport
V25pNativeEpisodeStore = ComponentCaptureEpisodeStore
V25pNativeSealReport = ComponentCaptureSealReport
validate_native_stream_profile = validate_component_stream_profile


__all__ = [
    "V25P_RECORD_CJSON_LAYOUT_V1",
    "V25pComponentInput",
    "V25pNativeCommitReport",
    "V25pNativeEpisodeStore",
    "V25pNativeSealReport",
    "validate_native_stream_profile",
]
