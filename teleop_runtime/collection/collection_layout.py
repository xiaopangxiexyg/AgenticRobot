"""Dataset and task layout resolution for the headless collection entrypoint.

The helpers normalize operator-facing task names and populate the parsed
collection namespace.  They only compute paths and metadata; they do not
create directories, open devices, start writers, or mutate collection state.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path


def sanitize_task_name_for_path(task_name: str) -> str:
    """Convert a task label into a stable single path component."""

    text = str(task_name).strip()
    if not text:
        return "task"
    text = re.sub(r"[\\/]+", "-", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("._-")
    return text or "task"


def default_task_id_from_name(task_name: str) -> str:
    """Derive the legacy-compatible task id from a task label."""

    slug = sanitize_task_name_for_path(task_name).lower()
    slug = re.sub(r"[^0-9a-z_-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("._-")
    return slug or "task"


def resolve_collection_layout(
    args: argparse.Namespace,
) -> tuple[Path, str, str, str, str]:
    """Resolve dataset path and task metadata without performing I/O."""

    legacy_task = str(getattr(args, "task", "")).strip()
    task_name = str(getattr(args, "task_name", "")).strip() or legacy_task or "default"
    task_id = (
        str(getattr(args, "task_id", "")).strip()
        or legacy_task
        or default_task_id_from_name(task_name)
    )
    task_description = str(getattr(args, "task_description", "")).strip() or task_name
    date_prefix = datetime.now().strftime(  # noqa: DTZ005 - preserve local date naming.
        "%Y-%m-%d"
    )
    dataset_dir_name = f"{date_prefix}_{sanitize_task_name_for_path(task_name)}"
    dataset_root = Path(args.dataset_root).expanduser() / dataset_dir_name
    dataset_name = str(getattr(args, "dataset_name", "")).strip() or dataset_dir_name
    return dataset_root, task_name, task_id, task_description, dataset_name


def apply_collection_layout(args: argparse.Namespace) -> None:
    """Populate the parsed namespace with the resolved collection layout."""

    dataset_base_root = Path(args.dataset_root).expanduser()
    dataset_root, task_name, task_id, task_description, dataset_name = (
        resolve_collection_layout(args)
    )
    args.dataset_base_root = str(dataset_base_root)
    args.dataset_root = str(dataset_root)
    args.dataset_dir_name = dataset_root.name
    args.task_name = task_name
    args.task_id = task_id
    args.task = task_id
    args.task_description = task_description
    args.dataset_name = dataset_name


__all__ = [
    "apply_collection_layout",
    "default_task_id_from_name",
    "resolve_collection_layout",
    "sanitize_task_name_for_path",
]
