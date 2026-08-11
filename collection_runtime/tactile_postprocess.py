"""State-aware orchestration for lossless tactile dataset maintenance.

This module does not define another tactile format or storage transaction.  It
only composes the existing verified recompression and modality-storage
operations into idempotent target states.  Every mutating step keeps its own
canonical receipt, so an interrupted workflow can be safely planned again from
the dataset's current physical state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.tactile import tactile_stream_compression
from dataset_interface.modality_artifacts import DETACHED_VERIFIED, SELF_CONTAINED

from .modality_storage import (
    ModalityStorageError,
    modality_storage_status,
    move_modality,
    plan_modality_detach,
    plan_modality_rehydrate,
    rehydrate_modality,
)
from .tactile_recompression import (
    DEFAULT_CHUNK_FRAMES,
    DEFAULT_MIN_FREE_BYTES,
    plan_tactile_zstd_recompression,
    recompress_tactile_zstd_in_place,
)


TACTILE_TARGET_HOT = "hot"
TACTILE_TARGET_COMPRESSED_HOT = "compressed-hot"
TACTILE_TARGET_DETACHED = "detached"
TACTILE_MAINTENANCE_TARGETS = (
    TACTILE_TARGET_HOT,
    TACTILE_TARGET_COMPRESSED_HOT,
    TACTILE_TARGET_DETACHED,
)
_HOT_STATUSES = {SELF_CONTAINED, "SELF_CONTAINED_LEGACY"}
_SUPPORTED_COMPRESSIONS = {"none", "lz4", "zstd"}


@dataclass(frozen=True)
class TactilePostprocessPlan:
    dataset_root: str
    target: str
    current_status: str
    inventory_hash: str | None
    raw_manifest_count: int | None
    raw_stream_count: int | None
    raw_bytes: int | None
    compressed_stream_count: int | None
    actions: tuple[dict[str, Any], ...]
    ready: bool
    target_satisfied: bool
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TactilePostprocessResult:
    dataset_root: str
    target: str
    operations: tuple[dict[str, Any], ...]
    final_plan: TactilePostprocessPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "target": self.target,
            "operations": list(self.operations),
            "final_plan": self.final_plan.to_dict(),
        }


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _catalog_tactile_manifests(root: Path) -> tuple[Path, ...]:
    catalog = root / "meta" / "episodes.jsonl"
    if not catalog.is_file():
        return ()
    manifests: list[Path] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        catalog.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid Episode catalog JSON at {catalog}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, Mapping):
            raise ValueError(
                f"Episode catalog record is not an object at {catalog}:{line_number}"
            )
        tactile_paths = record.get("tactile_paths", {})
        if not isinstance(tactile_paths, Mapping):
            raise ValueError(
                f"Episode tactile_paths is not an object at {catalog}:{line_number}"
            )
        for value in tactile_paths.values():
            relative = Path(str(value or ""))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe tactile manifest path: {value!r}")
            text = relative.as_posix()
            if text not in seen:
                manifests.append(root / relative)
                seen.add(text)
    return tuple(manifests)


def _validate_hot_compressions(manifests: tuple[Path, ...]) -> None:
    for manifest_path in manifests:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Referenced tactile manifest is missing: {manifest_path}")
        manifest = _read_json_object(manifest_path)
        streams = manifest.get("streams")
        if not isinstance(streams, Mapping) or not streams:
            raise ValueError(f"Tactile manifest has no stream map: {manifest_path}")
        for sensor_key, raw_spec in streams.items():
            if not isinstance(raw_spec, Mapping):
                raise ValueError(
                    f"Invalid tactile stream {sensor_key!r} in {manifest_path}"
                )
            compression = tactile_stream_compression(raw_spec)
            if compression not in _SUPPORTED_COMPRESSIONS:
                raise ValueError(
                    f"Unsupported tactile compression {compression!r} for "
                    f"{manifest_path}:{sensor_key}"
                )


def _recompression_summary(
    root: Path,
    *,
    min_free_bytes: int,
) -> tuple[Any, tuple[str, ...]]:
    plan = plan_tactile_zstd_recompression(
        root,
        min_free_bytes=int(min_free_bytes),
    )
    errors: list[str] = []
    if (
        plan.stream_count > 0
        and plan.free_bytes - plan.largest_manifest_raw_bytes < plan.min_free_bytes
    ):
        errors.append(
            "insufficient free space to stage the largest raw tactile manifest: "
            f"free={plan.free_bytes}, largest={plan.largest_manifest_raw_bytes}, "
            f"reserve={plan.min_free_bytes}"
        )
    return plan, tuple(errors)


def plan_tactile_postprocess(
    dataset_root: str | Path,
    *,
    target: str = TACTILE_TARGET_COMPRESSED_HOT,
    cold_root: str | Path | None = None,
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    compress_before_detach: bool = True,
) -> TactilePostprocessPlan:
    """Create a read-only plan from mixed raw/compressed/detached state."""

    root = Path(dataset_root).expanduser().resolve()
    requested_target = str(target).strip().lower()
    if requested_target not in TACTILE_MAINTENANCE_TARGETS:
        raise ValueError(
            f"target must be one of {TACTILE_MAINTENANCE_TARGETS}, got {target!r}"
        )
    manifests = _catalog_tactile_manifests(root)
    if not manifests:
        tactile_root = root / "tactile"
        unowned = (
            tuple(path for path in tactile_root.rglob("*") if path.is_file())
            if tactile_root.is_dir()
            else ()
        )
        if unowned:
            return TactilePostprocessPlan(
                dataset_root=str(root),
                target=requested_target,
                current_status="UNOWNED_TACTILE_TREE",
                inventory_hash=None,
                raw_manifest_count=None,
                raw_stream_count=None,
                raw_bytes=None,
                compressed_stream_count=None,
                actions=(),
                ready=False,
                target_satisfied=False,
                errors=(
                    "tactile files exist but no Episode catalog tactile_paths own them: "
                    + ", ".join(
                        path.relative_to(root).as_posix() for path in unowned[:5]
                    ),
                ),
            )
        return TactilePostprocessPlan(
            dataset_root=str(root),
            target=requested_target,
            current_status="NO_EXTERNAL_TACTILE",
            inventory_hash=None,
            raw_manifest_count=0,
            raw_stream_count=0,
            raw_bytes=0,
            compressed_stream_count=0,
            actions=(),
            ready=True,
            target_satisfied=True,
        )

    status = modality_storage_status(root)
    current_status = str(status.get("status") or "UNKNOWN")
    inventory_hash = status.get("inventory_hash")
    pending = bool(status.get("pending_transaction", False))
    errors: list[str] = []
    actions: list[dict[str, Any]] = []
    raw_plan: Any | None = None

    if pending:
        errors.append("a modality-storage transaction is pending; recover it first")
    if current_status not in _HOT_STATUSES | {DETACHED_VERIFIED}:
        errors.append(
            f"tactile state is not safely actionable: {current_status}: "
            f"{status.get('reason', '')}"
        )

    is_hot = current_status in _HOT_STATUSES
    is_detached = current_status == DETACHED_VERIFIED
    if is_hot and not errors:
        try:
            _validate_hot_compressions(manifests)
            raw_plan, space_errors = _recompression_summary(
                root,
                min_free_bytes=int(min_free_bytes),
            )
            errors.extend(space_errors)
        except Exception as exc:
            errors.append(f"cannot inspect hot tactile streams: {type(exc).__name__}: {exc}")

    if requested_target == TACTILE_TARGET_HOT:
        if is_hot:
            target_satisfied = True
        elif is_detached:
            target_satisfied = False
            if cold_root is None:
                errors.append("--cold-root is required to rehydrate detached tactile")
            elif not inventory_hash:
                errors.append("detached tactile has no unambiguous current inventory hash")
            else:
                try:
                    rehydrate = plan_modality_rehydrate(
                        root,
                        cold_root,
                        str(inventory_hash),
                        require_distinct_device=bool(require_distinct_device),
                        min_free_bytes=int(min_free_bytes),
                    )
                    actions.append({"name": "rehydrate", "plan": rehydrate.to_dict()})
                    if not rehydrate.ready:
                        errors.extend(rehydrate.errors)
                except Exception as exc:
                    errors.append(f"cannot plan rehydrate: {type(exc).__name__}: {exc}")
        else:
            target_satisfied = False
    elif requested_target == TACTILE_TARGET_COMPRESSED_HOT:
        if is_hot and raw_plan is not None:
            target_satisfied = raw_plan.stream_count == 0
            if raw_plan.stream_count:
                actions.append({"name": "recompress-raw-zstd", "plan": raw_plan.to_dict()})
        elif is_detached:
            target_satisfied = False
            if cold_root is None:
                errors.append(
                    "--cold-root is required to rehydrate before normalizing detached tactile"
                )
            elif not inventory_hash:
                errors.append("detached tactile has no unambiguous current inventory hash")
            else:
                try:
                    rehydrate = plan_modality_rehydrate(
                        root,
                        cold_root,
                        str(inventory_hash),
                        require_distinct_device=bool(require_distinct_device),
                        min_free_bytes=int(min_free_bytes),
                    )
                    actions.extend(
                        (
                            {"name": "rehydrate", "plan": rehydrate.to_dict()},
                            {
                                "name": "recompress-raw-zstd",
                                "plan": "recomputed after verified rehydrate",
                            },
                        )
                    )
                    if not rehydrate.ready:
                        errors.extend(rehydrate.errors)
                except Exception as exc:
                    errors.append(f"cannot plan rehydrate: {type(exc).__name__}: {exc}")
        else:
            target_satisfied = False
    else:
        if is_detached:
            target_satisfied = True
        elif is_hot:
            target_satisfied = False
            if cold_root is None:
                errors.append("--cold-root is required to detach tactile")
            else:
                if bool(compress_before_detach) and raw_plan is not None and raw_plan.stream_count:
                    actions.append(
                        {"name": "recompress-raw-zstd", "plan": raw_plan.to_dict()}
                    )
                try:
                    detach = plan_modality_detach(
                        root,
                        cold_root,
                        detachment_scope="whole-prefix",
                        require_distinct_device=bool(require_distinct_device),
                        min_free_bytes=int(min_free_bytes),
                    )
                    actions.append(
                        {
                            "name": "move-whole-tactile-prefix",
                            "plan": detach.to_dict(),
                            "replanned_after_recompression": bool(
                                compress_before_detach
                                and raw_plan is not None
                                and raw_plan.stream_count
                            ),
                        }
                    )
                    if not detach.ready:
                        errors.extend(detach.errors)
                except Exception as exc:
                    errors.append(f"cannot plan detach: {type(exc).__name__}: {exc}")
        else:
            target_satisfied = False

    return TactilePostprocessPlan(
        dataset_root=str(root),
        target=requested_target,
        current_status=current_status,
        inventory_hash=None if inventory_hash is None else str(inventory_hash),
        raw_manifest_count=(None if raw_plan is None else int(raw_plan.manifest_count)),
        raw_stream_count=(None if raw_plan is None else int(raw_plan.stream_count)),
        raw_bytes=(None if raw_plan is None else int(raw_plan.raw_bytes)),
        compressed_stream_count=(
            None if raw_plan is None else int(raw_plan.skipped_compressed_streams)
        ),
        actions=tuple(actions),
        ready=not errors,
        target_satisfied=bool(target_satisfied and not errors),
        errors=tuple(dict.fromkeys(str(error) for error in errors if str(error))),
    )


def apply_tactile_postprocess(
    dataset_root: str | Path,
    *,
    target: str = TACTILE_TARGET_COMPRESSED_HOT,
    cold_root: str | Path | None = None,
    level: int = 1,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    backend: str = "auto",
    require_distinct_device: bool = True,
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES,
    compress_before_detach: bool = True,
    confirm_no_active_readers: bool = False,
) -> TactilePostprocessResult:
    """Apply a target state using existing resumable, receipt-backed steps."""

    root = Path(dataset_root).expanduser().resolve()
    requested_target = str(target).strip().lower()
    plan = plan_tactile_postprocess(
        root,
        target=requested_target,
        cold_root=cold_root,
        require_distinct_device=bool(require_distinct_device),
        min_free_bytes=int(min_free_bytes),
        compress_before_detach=bool(compress_before_detach),
    )
    if not plan.ready:
        raise RuntimeError("tactile postprocess plan is not ready: " + "; ".join(plan.errors))
    if plan.target_satisfied:
        return TactilePostprocessResult(
            dataset_root=str(root),
            target=requested_target,
            operations=(),
            final_plan=plan,
        )

    operations: list[dict[str, Any]] = []

    def rehydrate_current() -> None:
        if cold_root is None or not plan.inventory_hash:
            raise RuntimeError("rehydrate requires cold_root and inventory_hash")
        proposal = plan_modality_rehydrate(
            root,
            cold_root,
            plan.inventory_hash,
            require_distinct_device=bool(require_distinct_device),
            min_free_bytes=int(min_free_bytes),
        )
        result = rehydrate_modality(
            root,
            cold_root,
            plan.inventory_hash,
            expected_plan_hash=proposal.plan_hash,
            confirm_dataset_fingerprint=proposal.dataset_fingerprint,
            require_distinct_device=bool(require_distinct_device),
            min_free_bytes=int(min_free_bytes),
            confirm_no_active_readers=bool(confirm_no_active_readers),
        )
        operations.append({"name": "rehydrate", "result": result.to_dict()})

    def recompress_raw() -> None:
        current, errors = _recompression_summary(root, min_free_bytes=int(min_free_bytes))
        if errors:
            raise RuntimeError("; ".join(errors))
        if current.stream_count <= 0:
            return
        result = recompress_tactile_zstd_in_place(
            root,
            level=int(level),
            chunk_frames=int(chunk_frames),
            min_free_bytes=int(min_free_bytes),
            delete_originals=True,
            backend=str(backend),
        )
        operations.append(
            {"name": "recompress-raw-zstd", "result": result.to_dict()}
        )

    if requested_target == TACTILE_TARGET_HOT:
        rehydrate_current()
    elif requested_target == TACTILE_TARGET_COMPRESSED_HOT:
        if plan.current_status == DETACHED_VERIFIED:
            if not confirm_no_active_readers:
                raise ModalityStorageError(
                    "rehydrate apply requires --confirm-no-active-readers"
                )
            rehydrate_current()
        recompress_raw()
    else:
        if bool(compress_before_detach):
            recompress_raw()
        if cold_root is None:
            raise RuntimeError("detach requires cold_root")
        if not confirm_no_active_readers:
            raise ModalityStorageError("detach apply requires --confirm-no-active-readers")
        result = move_modality(
            root,
            cold_root,
            require_distinct_device=bool(require_distinct_device),
            min_free_bytes=int(min_free_bytes),
            confirm_no_active_readers=True,
        )
        operations.append(
            {"name": "move-whole-tactile-prefix", "result": result.to_dict()}
        )

    final_plan = plan_tactile_postprocess(
        root,
        target=requested_target,
        cold_root=cold_root,
        require_distinct_device=bool(require_distinct_device),
        min_free_bytes=int(min_free_bytes),
        compress_before_detach=bool(compress_before_detach),
    )
    if not final_plan.ready or not final_plan.target_satisfied:
        raise RuntimeError(
            "tactile postprocess completed its steps but did not reach the target: "
            + "; ".join(final_plan.errors or (final_plan.current_status,))
        )
    return TactilePostprocessResult(
        dataset_root=str(root),
        target=requested_target,
        operations=tuple(operations),
        final_plan=final_plan,
    )


__all__ = [
    "TACTILE_MAINTENANCE_TARGETS",
    "TACTILE_TARGET_COMPRESSED_HOT",
    "TACTILE_TARGET_DETACHED",
    "TACTILE_TARGET_HOT",
    "TactilePostprocessPlan",
    "TactilePostprocessResult",
    "apply_tactile_postprocess",
    "plan_tactile_postprocess",
]
