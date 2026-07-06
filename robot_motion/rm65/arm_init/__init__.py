from __future__ import annotations

from importlib import import_module
from typing import Any

from .schema import (
    ArmQposSpec,
    QPOS_TARGET_DATA_TYPE,
    QposTargetSpec,
    load_qpos_target,
    qpos_target_schema_path,
    save_qpos_target,
)

_LAZY_EXPORTS = {
    "build_rm65_slow_qpos_init_plan_from_files": (
        ".qpos_init",
        "build_rm65_slow_qpos_init_plan_from_files",
    ),
    "QPOS_TEST_SEQUENCE_DATA_TYPE": (".qpos_sequence", "QPOS_TEST_SEQUENCE_DATA_TYPE"),
    "QPOS_TEST_SEQUENCE_REPORT_DATA_TYPE": (
        ".qpos_sequence",
        "QPOS_TEST_SEQUENCE_REPORT_DATA_TYPE",
    ),
    "QposSequenceEntry": (".qpos_sequence", "QposSequenceEntry"),
    "QposTestSequence": (".qpos_sequence", "QposTestSequence"),
    "build_qpos_test_sequence_from_dataset": (
        ".qpos_sequence",
        "build_qpos_test_sequence_from_dataset",
    ),
    "default_sequence_output_path": (".qpos_sequence", "default_sequence_output_path"),
    "dry_run_qpos_test_sequence": (".qpos_sequence", "dry_run_qpos_test_sequence"),
    "load_qpos_test_sequence": (".qpos_sequence", "load_qpos_test_sequence"),
    "qpos_test_sequence_schema_path": (".qpos_sequence", "qpos_test_sequence_schema_path"),
    "save_qpos_test_sequence": (".qpos_sequence", "save_qpos_test_sequence"),
}

__all__ = [
    "ArmQposSpec",
    "QPOS_TARGET_DATA_TYPE",
    "QPOS_TEST_SEQUENCE_DATA_TYPE",
    "QPOS_TEST_SEQUENCE_REPORT_DATA_TYPE",
    "QposSequenceEntry",
    "QposTargetSpec",
    "QposTestSequence",
    "build_rm65_slow_qpos_init_plan_from_files",
    "build_qpos_test_sequence_from_dataset",
    "default_sequence_output_path",
    "dry_run_qpos_test_sequence",
    "load_qpos_target",
    "load_qpos_test_sequence",
    "qpos_target_schema_path",
    "qpos_test_sequence_schema_path",
    "save_qpos_target",
    "save_qpos_test_sequence",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
