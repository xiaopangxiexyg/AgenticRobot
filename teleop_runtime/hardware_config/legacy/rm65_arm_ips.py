"""Read-only locator for historical RM65 IP assignment files."""

from pathlib import Path

from project_paths import teleop_records_root


def legacy_rm65_arm_ip_config_dirs() -> tuple[Path, ...]:
    return (teleop_records_root() / "rm65_arm_ip_configs",)


__all__ = ["legacy_rm65_arm_ip_config_dirs"]
