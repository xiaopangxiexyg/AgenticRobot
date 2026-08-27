"""Reusable detailed-stage annotation capability and status projection.

The detailed annotation overlay is optional and additive.  This module owns
profile discovery, immutable overlay selection, and the derived sidebar state;
the application remains responsible for editing locks, draft persistence, and
commit orchestration.  Keeping those boundaries explicit lets other Control
Station surfaces reuse detail availability without copying GUI-specific logic.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from collection_runtime.progress.profiles.catalog import (
    load_task_progress_profile_file,
    load_task_stage_profile_file,
)
from collection_runtime.progress.stage_annotation_store import (
    LoadedStageAnnotationSet,
)
from dataset_contracts.progress_detail_annotations import (
    validate_detail_stage_annotation_set,
    validate_detail_stage_profile_compatibility,
)
from dataset_contracts.progress_stages import TaskStageProfileV1
from dataset_interface.progress_annotation_jobs import ProgressAnnotationWorkItem
from dataset_interface.progress_detail_annotation_artifacts import (
    DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT,
    DetailStageAnnotationArtifactReadError,
    LoadedDetailStageAnnotationSet,
    load_detail_stage_annotation_set,
)

_DETAIL_STATUS_CACHE_LIMIT = 128
_PROFILE_SELECTION_CACHE_LIMIT = 32
_PROFILE_INVENTORY_PROBE_INTERVAL_SEC = 5.0


@dataclass(frozen=True)
class DetailProfileSelection:
    """One unambiguous detail profile bound to a coarse stage profile."""

    selection_id: str
    stage_profile: Any
    target_progress_step_id: str
    progress_profile_path: Path
    stage_profile_path: Path


@dataclass(frozen=True)
class DetailAnnotationStatus:
    """Stable, UI-neutral status for an optional immutable detail overlay."""

    state: str
    complete: bool
    applicable: bool
    target_progress_step_id: str | None = None
    reason: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "complete": self.complete,
            "applicable": self.applicable,
            "target_progress_step_id": self.target_progress_step_id,
            "reason": self.reason,
        }


class DetailAnnotationModule:
    """Discover and resolve optional detail overlays for one dataset view."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        profile_root: str | Path,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.profile_root = Path(profile_root).expanduser().resolve()
        self._lock = threading.RLock()
        self._profile_inventory_scan_lock = threading.Lock()
        self._profile_inventory_checked_at = 0.0
        self._profile_inventory_fingerprint: tuple[object, ...] = ()
        self._profile_pairs: tuple[tuple[Path, Path], ...] = ()
        self._profile_inventory_error: str | None = None
        self._profile_selection_cache: OrderedDict[
            tuple[str, tuple[object, ...]],
            tuple[tuple[DetailProfileSelection, ...], str | None],
        ] = OrderedDict()
        self._status_cache: OrderedDict[
            str,
            tuple[tuple[object, ...], DetailAnnotationStatus],
        ] = OrderedDict()

    def close(self) -> None:
        with self._lock:
            self._profile_pairs = ()
            self._profile_selection_cache.clear()
            self._status_cache.clear()

    def prune(self, eligible_episode_uuids: set[str]) -> None:
        with self._lock:
            for episode_uuid in tuple(self._status_cache):
                if episode_uuid not in eligible_episode_uuids:
                    self._status_cache.pop(episode_uuid, None)

    def invalidate(self, episode_uuid: str) -> None:
        with self._lock:
            self._status_cache.pop(str(episode_uuid), None)

    def refresh_profiles(self) -> None:
        """Force the next lookup to observe newly added editable profiles."""

        with self._lock:
            self._profile_inventory_checked_at = 0.0

    def _profile_inventory(
        self,
    ) -> tuple[
        tuple[tuple[Path, Path], ...],
        tuple[object, ...],
        str | None,
    ]:
        """Reuse one profile-tree scan across a visible Episode batch."""

        now = time.monotonic()
        with self._lock:
            if (
                self._profile_inventory_checked_at > 0.0
                and now - self._profile_inventory_checked_at
                < _PROFILE_INVENTORY_PROBE_INTERVAL_SEC
            ):
                return (
                    self._profile_pairs,
                    self._profile_inventory_fingerprint,
                    self._profile_inventory_error,
                )

        with self._profile_inventory_scan_lock:
            now = time.monotonic()
            with self._lock:
                if (
                    self._profile_inventory_checked_at > 0.0
                    and now - self._profile_inventory_checked_at
                    < _PROFILE_INVENTORY_PROBE_INTERVAL_SEC
                ):
                    return (
                        self._profile_pairs,
                        self._profile_inventory_fingerprint,
                        self._profile_inventory_error,
                    )
            try:
                progress_paths = tuple(
                    sorted(self.profile_root.rglob("*.progress_detailed.json"))
                )
                fingerprints: list[tuple[object, ...]] = []
                pairs: list[tuple[Path, Path]] = []
                for progress_path in progress_paths:
                    stage_path = progress_path.with_name(
                        progress_path.name.replace(
                            ".progress_detailed.json",
                            ".stages_detailed.json",
                        )
                    )
                    try:
                        progress_stat = progress_path.stat()
                        stage_stat = stage_path.stat()
                    except FileNotFoundError:
                        continue
                    if not stage_path.is_file():
                        continue
                    fingerprints.append(
                        (
                            str(progress_path),
                            progress_stat.st_size,
                            progress_stat.st_mtime_ns,
                            stage_stat.st_size,
                            stage_stat.st_mtime_ns,
                        )
                    )
                    pairs.append((progress_path, stage_path))
                inventory_pairs = tuple(pairs)
                fingerprint: tuple[object, ...] = tuple(fingerprints)
                error = None
            except OSError as exc:
                inventory_pairs = ()
                fingerprint = (
                    "error",
                    type(exc).__name__,
                    getattr(exc, "errno", None),
                )
                error = f"无法读取详细 stage profile 目录：{exc}"

            with self._lock:
                if fingerprint != self._profile_inventory_fingerprint:
                    self._profile_selection_cache.clear()
                self._profile_pairs = inventory_pairs
                self._profile_inventory_fingerprint = fingerprint
                self._profile_inventory_error = error
                self._profile_inventory_checked_at = now
                return inventory_pairs, fingerprint, error

    def profile_selections(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> tuple[tuple[DetailProfileSelection, ...], str | None]:
        """Find every compatible additive detail target.

        A detail source may refine more than one coarse step.  The annotation
        contract still publishes one target per immutable overlay, so each
        changed step is projected into an independent selection.  This keeps
        target status and user choice independent without changing the
        collection profile or the existing V1 overlay contract.
        """

        base_stage_profile = item.progress.stage_profile
        if base_stage_profile is None:
            return (), "当前 Episode 没有可用的粗粒度 stage profile。"
        pairs, inventory_fingerprint, inventory_error = (
            self._profile_inventory()
        )
        if inventory_error is not None:
            return (), inventory_error
        cache_key = (
            base_stage_profile.semantic_content_hash,
            inventory_fingerprint,
        )
        with self._lock:
            cached = self._profile_selection_cache.get(cache_key)
            if cached is not None:
                self._profile_selection_cache.move_to_end(cache_key)
        if cached is not None:
            return cached

        candidates: list[DetailProfileSelection] = []
        errors: list[str] = []
        base_step_ids = {
            step.progress_step_id for step in item.progress.profile.steps
        }
        for progress_path, stage_path in pairs:
            try:
                detail_progress = load_task_progress_profile_file(progress_path)
                if (
                    detail_progress.semantic_content_hash
                    != base_stage_profile.task_progress_profile_hash
                ):
                    continue
                detail_stage_profile = load_task_stage_profile_file(
                    stage_path,
                    progress_profile=detail_progress,
                )
                detail_step_ids = {
                    stage.progress_step_id
                    for stage in detail_stage_profile.stages
                }
                if detail_step_ids != base_step_ids:
                    continue
                changed_steps = tuple(
                    step_id
                    for step_id in sorted(base_step_ids)
                    if base_stage_profile.stages_for_step(step_id)
                    != detail_stage_profile.stages_for_step(step_id)
                )
                if not changed_steps:
                    errors.append(
                        f"{stage_path.name}: 详细 profile 没有可补充的粗阶段"
                    )
                    continue
                relative_progress = progress_path.relative_to(
                    self.profile_root
                ).as_posix()
                for target_step_id in changed_steps:
                    # Keep the current V1 overlay contract (one target per
                    # artifact) while allowing one source profile to expose
                    # multiple independent targets.  Non-target steps remain
                    # byte-for-byte equivalent to the coarse profile.
                    projected_stages = tuple(
                        stage
                        for step_id in sorted(base_step_ids)
                        for stage in (
                            detail_stage_profile.stages_for_step(step_id)
                            if step_id == target_step_id
                            else base_stage_profile.stages_for_step(step_id)
                        )
                    )
                    projected_profile = TaskStageProfileV1(
                        profile_id=detail_stage_profile.profile_id,
                        profile_version=detail_stage_profile.profile_version,
                        task_progress_profile_hash=(
                            detail_stage_profile.task_progress_profile_hash
                        ),
                        stages=projected_stages,
                    )
                    validate_detail_stage_profile_compatibility(
                        base_stage_profile=base_stage_profile,
                        detail_stage_profile=projected_profile,
                        target_progress_step_id=target_step_id,
                    )
                    candidates.append(
                        DetailProfileSelection(
                            selection_id=(
                                f"detail:{relative_progress}:"
                                f"{target_step_id}"
                            ),
                            stage_profile=projected_profile,
                            target_progress_step_id=target_step_id,
                            progress_profile_path=progress_path,
                            stage_profile_path=stage_path,
                        )
                    )
            except (OSError, TypeError, ValueError) as exc:
                errors.append(f"{stage_path.name}: {exc}")

        if candidates:
            result: tuple[tuple[DetailProfileSelection, ...], str | None] = (
                tuple(candidates),
                None,
            )
        elif errors:
            result = ((), errors[0])
        else:
            result = (
                (),
                "当前数据集没有匹配的详细 stage profile；继续使用粗粒度标注。",
            )
        with self._lock:
            self._profile_selection_cache[cache_key] = result
            self._profile_selection_cache.move_to_end(cache_key)
            while (
                len(self._profile_selection_cache)
                > _PROFILE_SELECTION_CACHE_LIMIT
            ):
                self._profile_selection_cache.popitem(last=False)
        return result

    def profile_selection(
        self,
        item: ProgressAnnotationWorkItem,
        selection_id: str | None = None,
    ) -> tuple[DetailProfileSelection | None, str | None]:
        """Resolve one selected target while keeping legacy single-candidate behavior."""

        candidates, reason = self.profile_selections(item)
        if selection_id is not None:
            normalized = str(selection_id).strip()
            if not normalized:
                return None, None
            for candidate in candidates:
                if candidate.selection_id == normalized:
                    return candidate, None
            return None, "所选详细 profile 与当前 Episode 不兼容。"
        if len(candidates) == 1:
            return candidates[0], reason
        if len(candidates) > 1:
            return None, "当前 Episode 有多个可用详细 profile，请先选择。"
        return None, reason

    def annotation_root(self, item: ProgressAnnotationWorkItem) -> Path:
        return (
            self.dataset_root
            / DETAIL_STAGE_ANNOTATION_RELATIVE_ROOT
            / item.episode_uuid
        )

    def annotation_root_fingerprint(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> tuple[object, ...]:
        root = self.annotation_root(item)
        try:
            value = root.stat()
        except FileNotFoundError:
            return ("missing",)
        except OSError as exc:
            return ("error", type(exc).__name__, getattr(exc, "errno", None))
        return (
            "directory",
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def current_annotation_set(
        self,
        item: ProgressAnnotationWorkItem,
        base_annotation_set: LoadedStageAnnotationSet,
        selection: DetailProfileSelection,
    ) -> tuple[LoadedDetailStageAnnotationSet | None, str | None]:
        """Load only the newest overlay bound to the exact current base set."""

        root = self.annotation_root(item)
        if not root.is_dir():
            return None, None
        assert item.progress.stage_profile is not None
        matches: list[LoadedDetailStageAnnotationSet] = []
        try:
            children = tuple(sorted(root.iterdir(), key=lambda value: value.name))
        except OSError as exc:
            return None, f"无法读取 detail 标注目录：{exc}"
        for child in children:
            if not child.is_dir() or not (child / "manifest.cjson").is_file():
                continue
            try:
                loaded = load_detail_stage_annotation_set(child)
            except DetailStageAnnotationArtifactReadError as exc:
                return None, f"detail 标注版本 {child.name!r} 无法读取：{exc}"
            binding = loaded.binding
            if (
                binding.episode_capture_identity_hash
                != item.episode_identity_hash
                or binding.source_progress_manifest_hash
                != item.progress_manifest_hash
                or binding.base_annotation_set_id
                != base_annotation_set.binding.annotation_set_id
                or binding.base_annotation_set_manifest_hash
                != base_annotation_set.manifest.semantic_content_hash
                or binding.base_task_stage_profile_hash
                != item.progress.stage_profile.semantic_content_hash
                or binding.detail_task_stage_profile_hash
                != selection.stage_profile.semantic_content_hash
                or binding.target_progress_step_id
                != selection.target_progress_step_id
            ):
                continue
            try:
                validate_detail_stage_annotation_set(
                    source_progress_manifest=item.progress.manifest,
                    base_binding=base_annotation_set.binding,
                    base_bundle=base_annotation_set.bundle,
                    base_manifest=base_annotation_set.manifest,
                    base_stage_profile=item.progress.stage_profile,
                    detail_stage_profile=selection.stage_profile,
                    binding=binding,
                    bundle=loaded.bundle,
                )
            except (TypeError, ValueError) as exc:
                return None, f"当前 detail 标注版本校验失败：{exc}"
            matches.append(loaded)
        if not matches:
            return None, None
        return max(
            matches,
            key=lambda value: (
                value.binding.created_at,
                value.binding.annotation_set_id,
            ),
        ), None

    def status(
        self,
        item: ProgressAnnotationWorkItem,
        annotation_sets: Sequence[LoadedStageAnnotationSet],
        *,
        selection_id: str | None = None,
    ) -> DetailAnnotationStatus:
        """Resolve a user-facing state without changing coarse fallback."""

        selection, profile_reason = self.profile_selection(
            item,
            selection_id=selection_id,
        )
        if selection is None:
            return DetailAnnotationStatus(
                state="NOT_APPLICABLE",
                complete=False,
                applicable=False,
                reason=(
                    "当前仅使用粗粒度标注（已关闭详细补充）。"
                    if selection_id == ""
                    else profile_reason
                ),
            )
        target = selection.target_progress_step_id
        if not annotation_sets:
            return DetailAnnotationStatus(
                state="PENDING",
                complete=False,
                applicable=True,
                target_progress_step_id=target,
                reason=(
                    "详细 profile 已匹配；粗、细标注可以独立编辑，"
                    "详细 overlay 会在粗标注版本确定后绑定。"
                ),
            )
        base_annotation_set = annotation_sets[-1]
        if not any(
            occurrence.progress_step_id == target
            for occurrence in base_annotation_set.bundle.occurrences
        ):
            return DetailAnnotationStatus(
                state="NOT_APPLICABLE",
                complete=False,
                applicable=False,
                target_progress_step_id=target,
                reason="当前 Episode 不包含详细 profile 的目标子任务。",
            )

        root_fingerprint = self.annotation_root_fingerprint(item)
        cache_key = (
            root_fingerprint,
            base_annotation_set.binding.annotation_set_id,
            base_annotation_set.manifest.semantic_content_hash,
            selection.stage_profile.semantic_content_hash,
            target,
            selection.selection_id,
        )
        with self._lock:
            cached = self._status_cache.get(item.episode_uuid)
            if cached is not None and cached[0] == cache_key:
                self._status_cache.move_to_end(item.episode_uuid)
                return cached[1]

        if root_fingerprint and root_fingerprint[0] == "error":
            result = DetailAnnotationStatus(
                state="ERROR",
                complete=False,
                applicable=True,
                target_progress_step_id=target,
                reason="无法读取详细标注目录。",
            )
        else:
            try:
                detail_set, load_reason = self.current_annotation_set(
                    item,
                    base_annotation_set,
                    selection,
                )
            except (OSError, TypeError, ValueError) as exc:
                detail_set = None
                load_reason = str(exc)
            if detail_set is not None:
                result = DetailAnnotationStatus(
                    state="COMPLETE",
                    complete=True,
                    applicable=True,
                    target_progress_step_id=target,
                )
            elif load_reason is not None:
                result = DetailAnnotationStatus(
                    state="ERROR",
                    complete=False,
                    applicable=True,
                    target_progress_step_id=target,
                    reason=load_reason,
                )
            elif root_fingerprint != ("missing",):
                result = DetailAnnotationStatus(
                    state="STALE",
                    complete=False,
                    applicable=True,
                    target_progress_step_id=target,
                    reason="已有详细标注未绑定当前粗标注版本，需要重新精标。",
                )
            else:
                result = DetailAnnotationStatus(
                    state="PENDING",
                    complete=False,
                    applicable=True,
                    target_progress_step_id=target,
                    reason="当前粗标注尚未发布对应的详细 overlay。",
                )

        with self._lock:
            self._status_cache[item.episode_uuid] = (cache_key, result)
            self._status_cache.move_to_end(item.episode_uuid)
            while len(self._status_cache) > _DETAIL_STATUS_CACHE_LIMIT:
                self._status_cache.popitem(last=False)
        return result


__all__ = [
    "DetailAnnotationModule",
    "DetailAnnotationStatus",
    "DetailProfileSelection",
]
