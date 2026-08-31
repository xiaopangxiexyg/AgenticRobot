"""Target-independent read-only perception Skill contracts."""

from .definition import (
    CAPTURE_DEPTH_SKILL_ID_V1,
    CAPTURE_RGB_SKILL_ID_V1,
    capture_depth_skill_definition_v1,
    capture_depth_skill_definition_v2,
    capture_rgb_skill_definition_v1,
    capture_rgb_skill_definition_v2,
)
from .schemas import (
    capture_depth_schema_entry_v2,
    capture_rgb_schema_entry_v2,
    validate_capture_depth_result_for_parameters_v2,
    validate_capture_depth_result_v2,
    validate_capture_rgb_result_for_parameters_v2,
    validate_capture_rgb_result_v2,
)

__all__ = [
    "CAPTURE_DEPTH_SKILL_ID_V1",
    "CAPTURE_RGB_SKILL_ID_V1",
    "capture_depth_schema_entry_v2",
    "capture_depth_skill_definition_v1",
    "capture_depth_skill_definition_v2",
    "capture_rgb_schema_entry_v2",
    "capture_rgb_skill_definition_v1",
    "capture_rgb_skill_definition_v2",
    "validate_capture_depth_result_for_parameters_v2",
    "validate_capture_depth_result_v2",
    "validate_capture_rgb_result_for_parameters_v2",
    "validate_capture_rgb_result_v2",
]
