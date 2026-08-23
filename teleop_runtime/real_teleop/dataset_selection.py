"""Real-teleop dataset/task selection transaction.

This module owns the device-neutral mutation of the pending collection
configuration.  The operator coordinator remains responsible for admission,
rollback, UI publication, Plus curation, and applying gripper policy to live
backends.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from collection_runtime.dataset_layout import (
    ROUND_TRIP_IN,
    ROUND_TRIP_TASK_TYPE,
    default_task_id_from_name,
    normalize_round_trip_phase,
    normalize_task_type,
    strip_round_trip_suffix,
)
from collection_runtime.progress.profiles import (
    inspect_dataset_progress_profile,
    inspect_dataset_task_stage_profile,
)
from dataset_contracts.capture_modalities import CaptureModalityContract

from .progress_adapter import (
    resolve_real_teleop_default_progress_profile,
    resolve_real_teleop_task_stage_profile,
)


class RealTeleopDatasetSelectionError(ValueError):
    """A rejected operator dataset-selection request."""


def apply_real_teleop_dataset_selection(
    args: Any,
    text: str,
    *,
    bind_dataset_contract: bool,
    task_definition_for_key: Callable[[Any, str], Mapping[str, Any] | None],
    refresh_formal_dataset_layout: Callable[[Any], object],
    expand_cli_path: Callable[[str], Path],
    bind_capture_modality_contract: Callable[
        [Path, CaptureModalityContract],
        object,
    ],
) -> Mapping[str, Any]:
    """Apply one pending selection and return its decoded payload.

    Mutation is intentionally performed on ``args`` because the canonical
    runtime configuration is still an argparse namespace.  The caller owns a
    full pre-call snapshot and restores it when this function raises
    ``RealTeleopDatasetSelectionError`` or a later application-side effect is
    rejected.
    """

    try:
        payload = json.loads(str(text or "{}"))
    except json.JSONDecodeError as exc:
        raise RealTeleopDatasetSelectionError(f"数采参数无效: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RealTeleopDatasetSelectionError("数采参数无效：payload 必须是对象。")

    dataset_base_root = str(payload.get("dataset_base_root") or "").strip()
    task_key = str(payload.get("task_key") or "").strip()
    english_name = str(payload.get("task_english_name") or "").strip()
    chinese_name = str(payload.get("task_chinese_name") or "").strip()
    payload_task_type = str(payload.get("task_type") or "").strip()
    payload_round_trip_phase = str(payload.get("round_trip_phase") or "").strip()

    if "ignored_camera_keys" in payload or "ignored_tactile_keys" in payload:
        ignored_camera_keys = payload.get("ignored_camera_keys", ())
        ignored_tactile_keys = payload.get("ignored_tactile_keys", ())
        if not isinstance(ignored_camera_keys, list) or not isinstance(
            ignored_tactile_keys,
            list,
        ):
            raise RealTeleopDatasetSelectionError(
                "数据集忽视规则无效：camera/tactile 必须是列表。"
            )
        camera_keys = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in ignored_camera_keys
                    if str(value).strip()
                }
            )
        )
        configured_camera_keys = {
            str(value)
            for value in (getattr(args, "configured_camera_keys", ()) or ())
        }
        unknown_camera_keys = sorted(set(camera_keys) - configured_camera_keys)
        if configured_camera_keys and unknown_camera_keys:
            raise RealTeleopDatasetSelectionError(
                "相机忽视规则包含当前配置中不存在的视角："
                + ", ".join(unknown_camera_keys)
            )
        tactile_keys = tuple(
            sorted({str(value).strip() for value in ignored_tactile_keys})
        )
        if any(value not in {"left", "right"} for value in tactile_keys):
            raise RealTeleopDatasetSelectionError(
                "触觉忽视规则仅支持 left/right。"
            )
        args.ignore_camera = list(camera_keys)
        args.ignore_tactile = list(tactile_keys)

    task_definition = task_definition_for_key(args, task_key)
    if task_definition is not None:
        if not english_name:
            english_name = str(task_definition.get("english_name") or task_key)
        if not chinese_name:
            chinese_name = str(task_definition.get("chinese_name") or "")
        gripper_task = str(task_definition.get("gripper_task") or task_key)
        task_type = normalize_task_type(
            payload_task_type
            if payload_task_type
            else task_definition.get("task_type", "")
        )
    else:
        gripper_task = task_key
        task_type = normalize_task_type(
            payload_task_type
            if payload_task_type
            else getattr(args, "collection_task_type", "")
        )

    if dataset_base_root:
        args.dataset_root = dataset_base_root
        args.dataset_base_root = str(expand_cli_path(dataset_base_root))
    if gripper_task:
        args.gripper_task = gripper_task
    task_name = (
        english_name
        or gripper_task
        or strip_round_trip_suffix(
            str(getattr(args, "collection_task_name", "") or "")
        )
    )
    if task_name:
        args.collection_task_name = task_name
    if chinese_name:
        args.collection_task_name_zh = chinese_name
    elif task_definition is None and "task_chinese_name" in payload:
        # A custom task has no registry label to inherit. Do not leak the
        # previous task's Chinese display name into the new dataset.
        args.collection_task_name_zh = ""
    base_task_id = task_key or default_task_id_from_name(task_name or gripper_task)
    args.collection_task_id = base_task_id
    args.collection_task_key = task_key
    args.collection_task_type = task_type
    task_description = (
        str(task_definition.get("description") or "").strip()
        if task_definition is not None
        else ""
    )
    args.collection_task_description = task_description or chinese_name or task_name
    if task_type == ROUND_TRIP_TASK_TYPE:
        phase = (
            normalize_round_trip_phase(payload_round_trip_phase)
            if payload_round_trip_phase
            else ROUND_TRIP_IN
        )
        args.collection_round_trip_phase = phase
        args.collection_round_trip_base_name = task_name
        args.collection_round_trip_base_id = base_task_id
        args.collection_round_trip_base_name_zh = chinese_name
        args.collection_round_trip_base_description = (
            args.collection_task_description
        )
    else:
        args.collection_round_trip_phase = ROUND_TRIP_IN
        args.collection_round_trip_base_name = strip_round_trip_suffix(task_name)
        args.collection_round_trip_base_id = strip_round_trip_suffix(base_task_id)
        args.collection_round_trip_base_name_zh = chinese_name
        args.collection_round_trip_base_description = (
            args.collection_task_description
        )

    previous_progress_profile_hash = str(
        getattr(
            getattr(args, "collection_progress_profile", None),
            "semantic_content_hash",
            "",
        )
        or ""
    )
    previous_stage_profile_hash = str(
        getattr(
            getattr(args, "collection_stage_profile", None),
            "semantic_content_hash",
            "",
        )
        or ""
    )
    refresh_formal_dataset_layout(args)
    try:
        target_dataset_root = Path(args.dataset_root)
        requested_progress_profile = getattr(
            args,
            "collection_progress_profile",
            None,
        )
        # A task/root change first adopts a target's immutable binding. A
        # blank target may carry the operator's current draft profile.
        target_binding = inspect_dataset_progress_profile(
            target_dataset_root,
            requested_profile=None,
        )
        if (
            requested_progress_profile is None
            and not target_binding.locked
            and target_binding.profile is None
        ):
            requested_progress_profile = resolve_real_teleop_default_progress_profile(
                dataset_root=target_dataset_root,
                requested_profile=None,
                gui_enabled=bool(getattr(args, "collection_progress_gui", False)),
            )
        progress_admission = (
            target_binding
            if target_binding.locked or target_binding.profile is not None
            else inspect_dataset_progress_profile(
                target_dataset_root,
                requested_profile=requested_progress_profile,
            )
        )
    except Exception as exc:  # noqa: BLE001 - caller restores its full snapshot.
        raise RealTeleopDatasetSelectionError(
            "数据集 progress profile 校验失败，已恢复原配置："
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if progress_admission.profile is not None:
        args.collection_progress_profile = progress_admission.profile
        try:
            requested_stage_profile = (
                None
                if target_binding.profile is not None
                else resolve_real_teleop_task_stage_profile(
                    progress_admission.profile,
                    getattr(args, "collection_stage_profile", None),
                )
            )
            stage_admission = inspect_dataset_task_stage_profile(
                Path(args.dataset_root),
                progress_profile=progress_admission.profile,
                requested_profile=requested_stage_profile,
            )
        except Exception as exc:  # noqa: BLE001 - caller restores its full snapshot.
            raise RealTeleopDatasetSelectionError(
                "数据集 stage profile 校验失败，已恢复原配置："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        args.collection_stage_profile = stage_admission.profile
    else:
        # A target may be permanently profile-less; do not retain a draft
        # profile from the previously selected task.
        args.collection_progress_profile = None
        args.collection_stage_profile = None

    selected_progress_profile_hash = str(
        getattr(
            getattr(args, "collection_progress_profile", None),
            "semantic_content_hash",
            "",
        )
        or ""
    )
    selected_stage_profile_hash = str(
        getattr(
            getattr(args, "collection_stage_profile", None),
            "semantic_content_hash",
            "",
        )
        or ""
    )
    if (
        selected_progress_profile_hash != previous_progress_profile_hash
        or selected_stage_profile_hash != previous_stage_profile_hash
    ):
        # Entry selection is profile-local and Episode-local. Adopting another
        # dataset's immutable profile must not retain a stale coarse/fine entry
        # from the previous task.
        args.collection_progress_entry_step = ""
        args.collection_progress_entry_stage = ""
        args.collection_progress_entry_condition = "nominal"

    if bind_dataset_contract:
        try:
            bind_capture_modality_contract(
                Path(args.dataset_root),
                CaptureModalityContract(
                    ignored_camera_keys=tuple(
                        getattr(args, "ignore_camera", ()) or ()
                    ),
                    ignored_tactile_keys=tuple(
                        getattr(args, "ignore_tactile", ()) or ()
                    ),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - caller restores its full snapshot.
            raise RealTeleopDatasetSelectionError(
                "数据集忽视规则绑定失败，已恢复原配置："
                f"{type(exc).__name__}: {exc}"
            ) from exc

    return payload


__all__ = [
    "RealTeleopDatasetSelectionError",
    "apply_real_teleop_dataset_selection",
]
