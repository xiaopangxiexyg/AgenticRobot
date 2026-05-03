from __future__ import annotations

from typing import Dict, Iterable, List

from Datasets.base import DatasetManifest


def build_task_name_to_id(manifests: Iterable[DatasetManifest]) -> Dict[str, int]:
    task_names = {"unknown_task"}
    for manifest in manifests:
        task_names.update(str(task_text) for task_text in manifest.task_lookup.values() if str(task_text).strip())
        task_names.update(str(task_text) for task_text in manifest.episode_task_lookup.values() if str(task_text).strip())
    return {
        task_name: task_index
        for task_index, task_name in enumerate(sorted(task_names))
    }


def list_task_names(manifests: Iterable[DatasetManifest]) -> List[str]:
    task_name_to_id = build_task_name_to_id(manifests)
    return [task_name for task_name, _ in sorted(task_name_to_id.items(), key=lambda item: item[1])]
