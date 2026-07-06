from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GENERAL_TASK_TYPE = "general"
ROUND_TRIP_TASK_TYPE = "round_trip"
ROUND_TRIP_IN = "in"
ROUND_TRIP_OUT = "out"
ROUND_TRIP_PHASES = (ROUND_TRIP_IN, ROUND_TRIP_OUT)


@dataclass(frozen=True)
class FormalDatasetLayout:
    dataset_root: Path
    dataset_dir_name: str
    task_name: str
    task_name_zh: str
    task_id: str
    task_description: str
    dataset_name: str
    task_type: str
    task_key: str
    round_trip_phase: str
    round_trip_next_phase: str
    round_trip_base_name: str
    round_trip_base_id: str
    round_trip_base_name_zh: str
    round_trip_base_description: str


def sanitize_dataset_name_for_path(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "task"


def default_task_id_from_name(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "task"


def normalize_task_type(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"roundtrip", "round_trip", "back_and_forth", "in_out", "inout", "往返", "来回"}:
        return ROUND_TRIP_TASK_TYPE
    return GENERAL_TASK_TYPE


def normalize_round_trip_phase(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == ROUND_TRIP_OUT:
        return ROUND_TRIP_OUT
    return ROUND_TRIP_IN


def opposite_round_trip_phase(value: object) -> str:
    return ROUND_TRIP_IN if normalize_round_trip_phase(value) == ROUND_TRIP_OUT else ROUND_TRIP_OUT


def strip_round_trip_suffix(text: str) -> str:
    return re.sub(r"_(in|out)$", "", str(text or "").strip(), flags=re.IGNORECASE)


def round_trip_variant(text: str, phase: object) -> str:
    base = strip_round_trip_suffix(text)
    return f"{base}_{normalize_round_trip_phase(phase)}" if base else f"task_{normalize_round_trip_phase(phase)}"


def _raw_task_type(raw: Mapping[str, Any] | None, fallback: object = "") -> str:
    if raw is None:
        return normalize_task_type(fallback)
    value = raw.get("task_type", raw.get("type", raw.get("kind", fallback)))
    return normalize_task_type(value)


def load_task_registry(
    args: Any,
    *,
    expand_path: Callable[[str], Path],
) -> list[dict[str, str]]:
    cached = getattr(args, "_task_registry", None)
    if isinstance(cached, list):
        return [dict(item) for item in cached if isinstance(item, Mapping)]
    path_text = str(getattr(args, "task_config", "") or "").strip()
    if not path_text:
        setattr(args, "_task_registry", [])
        return []
    path = expand_path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        setattr(args, "_task_registry", [])
        return []
    if not isinstance(payload, Mapping):
        raise ValueError(f"Task config {path} must contain a JSON object.")
    raw_tasks = payload.get("tasks", [])
    if isinstance(raw_tasks, Mapping):
        iterable = []
        for key, value in raw_tasks.items():
            if isinstance(value, Mapping):
                item = dict(value)
                item.setdefault("key", str(key))
                iterable.append(item)
    elif isinstance(raw_tasks, Sequence) and not isinstance(raw_tasks, (str, bytes)):
        iterable = list(raw_tasks)
    else:
        raise ValueError(f"Task config {path} field 'tasks' must be an array or object.")
    tasks: list[dict[str, str]] = []
    for raw in iterable:
        if not isinstance(raw, Mapping):
            continue
        key = str(raw.get("key") or raw.get("task_id") or raw.get("gripper_task") or "").strip()
        english_name = str(raw.get("english_name") or raw.get("name_en") or key).strip()
        chinese_name = str(raw.get("chinese_name") or raw.get("name_zh") or "").strip()
        gripper_task = str(raw.get("gripper_task") or key).strip()
        if not key:
            continue
        tasks.append(
            {
                "key": key,
                "english_name": english_name or key,
                "chinese_name": chinese_name,
                "description": str(raw.get("description") or "").strip(),
                "gripper_task": gripper_task,
                "task_type": _raw_task_type(raw),
            }
        )
    setattr(args, "_task_registry", [dict(item) for item in tasks])
    return tasks


def task_definition_for_key(
    args: Any,
    key: str,
    *,
    expand_path: Callable[[str], Path],
) -> dict[str, str] | None:
    key_text = str(key or "").strip()
    if not key_text:
        return None
    stripped_key = strip_round_trip_suffix(key_text)
    for item in load_task_registry(args, expand_path=expand_path):
        candidates = {
            item.get("key", ""),
            item.get("gripper_task", ""),
            strip_round_trip_suffix(item.get("key", "")),
            strip_round_trip_suffix(item.get("gripper_task", "")),
        }
        if key_text in candidates or stripped_key in candidates:
            return dict(item)
    return None


def _task_type_for_layout(args: Any, task_def: Mapping[str, str] | None) -> str:
    configured = str(getattr(args, "collection_task_type", "") or "").strip()
    if configured:
        return normalize_task_type(configured)
    return _raw_task_type(task_def)


def _round_trip_base_value(args: Any, attr: str, fallback: str) -> str:
    value = str(getattr(args, attr, "") or "").strip()
    if value:
        return strip_round_trip_suffix(value)
    return strip_round_trip_suffix(fallback)


def resolve_formal_dataset_layout(
    args: Any,
    *,
    expand_path: Callable[[str], Path],
    date_prefix: str,
) -> FormalDatasetLayout:
    configured_task_key = str(getattr(args, "collection_task_key", "") or "").strip()
    configured_task_id = str(getattr(args, "collection_task_id", "") or "").strip()
    gripper_task = str(getattr(args, "gripper_task", "") or "").strip()
    task_def = task_definition_for_key(
        args,
        configured_task_key or configured_task_id or gripper_task,
        expand_path=expand_path,
    )
    task_type = _task_type_for_layout(args, task_def)

    default_task_key = ""
    default_task_name = ""
    default_task_name_zh = ""
    default_task_description = ""
    default_task_id = gripper_task
    if task_def is not None:
        default_task_key = str(task_def.get("key") or "").strip()
        default_task_id = default_task_key or default_task_id
        default_task_name = str(task_def.get("english_name") or "").strip()
        default_task_name_zh = str(task_def.get("chinese_name") or "").strip()
        default_task_description = (
            str(task_def.get("description") or "").strip()
            or default_task_name_zh
        )

    configured_name = str(getattr(args, "collection_task_name", "") or "").strip()
    configured_name_zh = str(getattr(args, "collection_task_name_zh", "") or "").strip()
    configured_description = str(getattr(args, "collection_task_description", "") or "").strip()
    task_key = configured_task_key or default_task_key or strip_round_trip_suffix(configured_task_id) or gripper_task

    if task_type == ROUND_TRIP_TASK_TYPE:
        phase = normalize_round_trip_phase(getattr(args, "collection_round_trip_phase", ROUND_TRIP_IN))
        base_name = _round_trip_base_value(
            args,
            "collection_round_trip_base_name",
            configured_name or default_task_name or gripper_task or "vive_rm65_real_teleop",
        )
        base_name_zh = _round_trip_base_value(
            args,
            "collection_round_trip_base_name_zh",
            configured_name_zh or default_task_name_zh,
        )
        base_description = _round_trip_base_value(
            args,
            "collection_round_trip_base_description",
            configured_description or default_task_description or base_name_zh or base_name,
        )
        base_id = _round_trip_base_value(
            args,
            "collection_round_trip_base_id",
            configured_task_id or default_task_id or default_task_key or default_task_id_from_name(base_name),
        )
        base_id = base_id or default_task_id_from_name(base_name)
        task_name = round_trip_variant(base_name, phase)
        task_id = round_trip_variant(base_id, phase)
        task_description = base_description or base_name_zh or task_name
        round_trip_next_phase = opposite_round_trip_phase(phase)
    else:
        phase = ROUND_TRIP_IN
        round_trip_next_phase = ROUND_TRIP_OUT
        task_name = configured_name or default_task_name or gripper_task or "vive_rm65_real_teleop"
        task_id = configured_task_id or default_task_id or _default_key_from_name(task_name)
        base_name = strip_round_trip_suffix(task_name)
        base_id = strip_round_trip_suffix(task_id)
        base_name_zh = configured_name_zh or default_task_name_zh
        base_description = configured_description or default_task_description or base_name_zh or task_name
        task_description = base_description

    task_name_zh = configured_name_zh or default_task_name_zh
    dataset_dir_name = f"{date_prefix}_{sanitize_dataset_name_for_path(task_name)}"
    dataset_root = expand_path(str(getattr(args, "dataset_root", ""))) / dataset_dir_name
    dataset_name_text = str(getattr(args, "dataset_name", "") or "").strip()
    explicit_dataset_name = getattr(args, "_dataset_name_explicit", None)
    dataset_name_is_explicit = (
        bool(dataset_name_text)
        if explicit_dataset_name is None
        else bool(explicit_dataset_name)
    )
    dataset_name = dataset_name_text if dataset_name_is_explicit and dataset_name_text else dataset_dir_name
    return FormalDatasetLayout(
        dataset_root=dataset_root,
        dataset_dir_name=dataset_dir_name,
        task_name=task_name,
        task_name_zh=task_name_zh,
        task_id=task_id,
        task_description=task_description,
        dataset_name=dataset_name,
        task_type=task_type,
        task_key=task_key,
        round_trip_phase=phase,
        round_trip_next_phase=round_trip_next_phase,
        round_trip_base_name=base_name,
        round_trip_base_id=base_id,
        round_trip_base_name_zh=base_name_zh,
        round_trip_base_description=base_description,
    )


def _default_key_from_name(task_name: str) -> str:
    return default_task_id_from_name(task_name)


def apply_formal_dataset_layout(
    args: Any,
    *,
    expand_path: Callable[[str], Path],
    date_prefix: str,
) -> FormalDatasetLayout:
    if not hasattr(args, "_dataset_name_explicit"):
        args._dataset_name_explicit = bool(str(getattr(args, "dataset_name", "") or "").strip())
    dataset_base_root = expand_path(str(getattr(args, "dataset_root", "")))
    layout = resolve_formal_dataset_layout(
        args,
        expand_path=expand_path,
        date_prefix=date_prefix,
    )
    args.dataset_base_root = str(dataset_base_root)
    args.dataset_root = str(layout.dataset_root)
    args.dataset_dir_name = layout.dataset_dir_name
    args.collection_task_name = layout.task_name
    args.collection_task_name_zh = layout.task_name_zh
    args.collection_task_id = layout.task_id
    args.collection_task_description = layout.task_description
    args.dataset_name = layout.dataset_name
    args.collection_task_type = layout.task_type
    args.collection_task_key = layout.task_key
    args.collection_round_trip_phase = layout.round_trip_phase
    args.collection_round_trip_next_phase = layout.round_trip_next_phase
    args.collection_round_trip_base_name = layout.round_trip_base_name
    args.collection_round_trip_base_id = layout.round_trip_base_id
    args.collection_round_trip_base_name_zh = layout.round_trip_base_name_zh
    args.collection_round_trip_base_description = layout.round_trip_base_description
    return layout


def refresh_formal_dataset_layout(
    args: Any,
    *,
    expand_path: Callable[[str], Path],
    date_prefix: str,
) -> FormalDatasetLayout:
    base = str(getattr(args, "dataset_base_root", "") or "").strip()
    if base:
        args.dataset_root = base
    return apply_formal_dataset_layout(args, expand_path=expand_path, date_prefix=date_prefix)


def toggle_round_trip_phase(
    args: Any,
    *,
    expand_path: Callable[[str], Path],
    date_prefix: str,
) -> bool:
    if normalize_task_type(getattr(args, "collection_task_type", "")) != ROUND_TRIP_TASK_TYPE:
        return False
    args.collection_round_trip_phase = opposite_round_trip_phase(
        getattr(args, "collection_round_trip_phase", ROUND_TRIP_IN)
    )
    refresh_formal_dataset_layout(args, expand_path=expand_path, date_prefix=date_prefix)
    return True


def collection_task_ui_state(
    args: Any,
    *,
    expand_path: Callable[[str], Path],
) -> dict[str, object]:
    task_type = normalize_task_type(getattr(args, "collection_task_type", ""))
    phase = normalize_round_trip_phase(getattr(args, "collection_round_trip_phase", ROUND_TRIP_IN))
    return {
        "dataset_base_root": str(getattr(args, "dataset_base_root", "") or getattr(args, "dataset_root", "") or ""),
        "dataset_root": str(getattr(args, "dataset_root", "") or ""),
        "dataset_dir_name": str(getattr(args, "dataset_dir_name", "") or ""),
        "collection_task_id": str(getattr(args, "collection_task_id", "") or ""),
        "collection_task_key": str(getattr(args, "collection_task_key", "") or ""),
        "collection_task_name": str(getattr(args, "collection_task_name", "") or ""),
        "collection_task_name_zh": str(getattr(args, "collection_task_name_zh", "") or ""),
        "collection_task_description": str(getattr(args, "collection_task_description", "") or ""),
        "collection_task_type": task_type,
        "collection_round_trip_phase": phase,
        "collection_round_trip_next_phase": opposite_round_trip_phase(phase),
        "collection_round_trip_base_name": str(getattr(args, "collection_round_trip_base_name", "") or ""),
        "collection_round_trip_base_id": str(getattr(args, "collection_round_trip_base_id", "") or ""),
        "collection_round_trip_active": task_type == ROUND_TRIP_TASK_TYPE,
        "gripper_task": str(getattr(args, "gripper_task", "") or ""),
        "task_registry": load_task_registry(args, expand_path=expand_path),
    }
