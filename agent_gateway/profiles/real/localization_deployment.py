"""Compatibility exports for the shared localization deployment."""

from agent_gateway.profiles.localization_deployment import (
    REAL_LOCALIZE_OBJECTS_EXECUTOR_ID_V1,
    REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1,
    REAL_LOCALIZE_OBJECTS_MAX_ANNOTATED_BYTES_V1,
    REAL_LOCALIZE_OBJECTS_MAX_DETECTIONS_V1,
    REAL_LOCALIZE_OBJECTS_MAX_IMAGE_BYTES_V1,
    REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1,
    REAL_LOCALIZE_OBJECTS_ROUTE_ID_V1,
    RealObjectLocalizationDeploymentBinding,
    RealObjectLocalizationProfile,
    RealObjectLocalizationRegistration,
    build_real_object_localization_registration,
)

__all__ = [
    "REAL_LOCALIZE_OBJECTS_EXECUTOR_ID_V1",
    "REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1",
    "REAL_LOCALIZE_OBJECTS_MAX_ANNOTATED_BYTES_V1",
    "REAL_LOCALIZE_OBJECTS_MAX_DETECTIONS_V1",
    "REAL_LOCALIZE_OBJECTS_MAX_IMAGE_BYTES_V1",
    "REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1",
    "REAL_LOCALIZE_OBJECTS_ROUTE_ID_V1",
    "RealObjectLocalizationDeploymentBinding",
    "RealObjectLocalizationProfile",
    "RealObjectLocalizationRegistration",
    "build_real_object_localization_registration",
]
