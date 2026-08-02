"""Compatibility exports for the real-teleop task layout implementation.

The implementation lives in ``collection_runtime.dataset_layout`` so the GUI
coordinator does not own a second copy of dataset/task policy. This module
intentionally keeps the historic GUI import path and symbols available.
"""

from __future__ import annotations

from collection_runtime.dataset_layout import (
    GENERAL_TASK_TYPE,
    ROUND_TRIP_IN,
    ROUND_TRIP_OUT,
    ROUND_TRIP_PHASES,
    ROUND_TRIP_TASK_TYPE,
    FormalDatasetLayout,
    _default_key_from_name,
    _raw_task_type,
    _round_trip_base_value,
    _task_type_for_layout,
    apply_formal_dataset_layout,
    collection_task_ui_state,
    default_task_id_from_name,
    expand_cli_path,
    load_task_registry,
    normalize_round_trip_phase,
    normalize_task_type,
    opposite_round_trip_phase,
    refresh_formal_dataset_layout,
    resolve_formal_dataset_layout,
    round_trip_variant,
    sanitize_dataset_name_for_path,
    strip_round_trip_suffix,
    task_definition_for_key,
    toggle_round_trip_phase,
)


__all__ = [
    "GENERAL_TASK_TYPE",
    "ROUND_TRIP_IN",
    "ROUND_TRIP_OUT",
    "ROUND_TRIP_PHASES",
    "ROUND_TRIP_TASK_TYPE",
    "FormalDatasetLayout",
    "apply_formal_dataset_layout",
    "collection_task_ui_state",
    "default_task_id_from_name",
    "expand_cli_path",
    "load_task_registry",
    "normalize_round_trip_phase",
    "normalize_task_type",
    "opposite_round_trip_phase",
    "refresh_formal_dataset_layout",
    "resolve_formal_dataset_layout",
    "round_trip_variant",
    "sanitize_dataset_name_for_path",
    "strip_round_trip_suffix",
    "task_definition_for_key",
    "toggle_round_trip_phase",
]
