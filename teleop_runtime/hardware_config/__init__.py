"""Device-configuration contracts with no application-host dependency."""

from .rm65_arm_ips import (
    LATEST_CONFIG_NAME,
    SCHEMA_VERSION,
    SIDES,
    arm_ips_for_sides,
    build_assignment,
    default_config_dir,
    empty_assignment,
    hostname,
    iso_timestamp,
    latest_config_path,
    legacy_config_dirs,
    load_assignment,
    normalize_assignment,
    save_assignment,
    save_assignment_for_config,
    timestamp_for_filename,
)

__all__ = [
    "LATEST_CONFIG_NAME",
    "SCHEMA_VERSION",
    "SIDES",
    "arm_ips_for_sides",
    "build_assignment",
    "default_config_dir",
    "empty_assignment",
    "hostname",
    "iso_timestamp",
    "latest_config_path",
    "legacy_config_dirs",
    "load_assignment",
    "normalize_assignment",
    "save_assignment",
    "save_assignment_for_config",
    "timestamp_for_filename",
]
