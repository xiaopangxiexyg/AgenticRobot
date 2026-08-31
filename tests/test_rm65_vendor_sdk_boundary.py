from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CURRENT_RM65_DOCS = (
    REPOSITORY_ROOT / "README_ENV.md",
    REPOSITORY_ROOT / "teleop_runtime" / "README.md",
    REPOSITORY_ROOT / "teleop_runtime" / "collection" / "README.md",
)


def test_retired_rm65_api1_vendor_snapshot_is_absent() -> None:
    assert not (REPOSITORY_ROOT / "hardware" / "rm65" / "robotic_arm_package").exists()


def test_current_rm65_docs_reference_the_api2_compatibility_owner() -> None:
    for path in CURRENT_RM65_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "hardware.rm65.robotic_arm_package" not in text
        assert "hardware.rm65.rm_api2_adapter" in text


def test_rm65_api2_vendor_and_compatibility_owner_remain_versioned() -> None:
    expected_paths = (
        REPOSITORY_ROOT / "hardware" / "rm65" / "rm_api2_adapter.py",
        REPOSITORY_ROOT / "hardware" / "rm65" / "Robotic_Arm" / "rm_ctypes_wrap.py",
        REPOSITORY_ROOT / "hardware" / "rm65" / "Robotic_Arm" / "rm_robot_interface.py",
        REPOSITORY_ROOT
        / "hardware"
        / "rm65"
        / "Robotic_Arm"
        / "libs"
        / "linux_x86"
        / "libapi_c.so",
    )
    assert all(path.is_file() for path in expected_paths)
