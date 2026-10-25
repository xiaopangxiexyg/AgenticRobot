"""Compatibility exports for the shared supplied-image localization provider."""

from agent_gateway.profiles.localization import (
    ObjectLocalizationProvider,
    PreparedObjectLocalization,
    build_real_object_localization_executor,
    decode_bounded_image,
    read_bounded_regular_file,
    safe_local_path,
)

__all__ = [
    "ObjectLocalizationProvider",
    "PreparedObjectLocalization",
    "build_real_object_localization_executor",
    "decode_bounded_image",
    "read_bounded_regular_file",
    "safe_local_path",
]
