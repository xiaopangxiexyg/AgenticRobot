from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .tactile_recompress import _active_collection_processes


DEFAULT_MIN_FILE_BYTES = 1024 * 1024
DEFAULT_IMMUTABLE_PREFIXES = ("videos", "tactile", "wrist_depth")


@dataclass(frozen=True)
class HardlinkReplacement:
    reference_path: str
    target_path: str
    size: int
    allocated_bytes: int
    sha256: str
    reference_device: int
    reference_inode: int
    reference_mtime_ns: int
    target_device: int
    target_inode: int
    target_mtime_ns: int


@dataclass(frozen=True)
class HardlinkDedupePlan:
    target_root: str
    reference_roots: tuple[str, ...]
    scanned_reference_files: int
    scanned_target_files: int
    hashed_files: int
    replacement_count: int
    logical_duplicate_bytes: int
    estimated_reclaimable_bytes: int
    already_hardlinked_bytes: int
    cross_device_duplicate_bytes: int
    replacements: tuple[HardlinkReplacement, ...]

    def to_dict(self, *, include_replacements: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_replacements:
            payload.pop("replacements", None)
        return payload


@dataclass(frozen=True)
class HardlinkDedupeResult:
    target_root: str
    replacement_count: int
    logical_duplicate_bytes: int
    estimated_reclaimed_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _File:
    path: Path
    size: int
    allocated_bytes: int
    device: int
    inode: int
    links: int
    mtime_ns: int


def _regular_files(
    root: Path,
    *,
    min_file_bytes: int,
    include_prefixes: tuple[str, ...],
) -> list[_File]:
    files: list[_File] = []
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if include_prefixes and not any(part in include_prefixes for part in relative.parts[:-1]):
            continue
        stat = path.stat()
        if stat.st_size < min_file_bytes:
            continue
        files.append(
            _File(
                path=path,
                size=int(stat.st_size),
                allocated_bytes=int(stat.st_blocks * 512),
                device=int(stat.st_dev),
                inode=int(stat.st_ino),
                links=int(stat.st_nlink),
                mtime_ns=int(stat.st_mtime_ns),
            )
        )
    return files


def _sha256(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolved_roots(
    target_root: str | Path,
    reference_roots: Iterable[str | Path],
) -> tuple[Path, tuple[Path, ...]]:
    target = Path(target_root).expanduser().resolve()
    references = tuple(Path(value).expanduser().resolve() for value in reference_roots)
    if not target.is_dir():
        raise NotADirectoryError(target)
    if not references:
        raise ValueError("At least one reference_root is required.")
    for reference in references:
        if not reference.is_dir():
            raise NotADirectoryError(reference)
        if reference == target:
            raise ValueError("target_root cannot also be a reference_root.")
        if reference in target.parents or target in reference.parents:
            raise ValueError("target_root and reference_roots must not overlap.")
    return target, references


def plan_hardlink_dedupe(
    target_root: str | Path,
    reference_roots: Iterable[str | Path],
    *,
    min_file_bytes: int = DEFAULT_MIN_FILE_BYTES,
    include_prefixes: Iterable[str] = DEFAULT_IMMUTABLE_PREFIXES,
) -> HardlinkDedupePlan:
    """Hash duplicate candidates and plan target-only hardlink replacements.

    Size is used only as a candidate filter. A replacement is planned only
    after both files have the same complete SHA-256 digest.
    """

    active = _active_collection_processes()
    if active:
        raise RuntimeError(
            "Refusing I/O-heavy hardlink dedupe scan while formal collection is active: "
            + "; ".join(active[:3])
        )
    if int(min_file_bytes) < 0:
        raise ValueError("min_file_bytes must be non-negative.")
    prefixes = tuple(str(value).strip("/") for value in include_prefixes if str(value).strip("/"))
    if not prefixes:
        raise ValueError("At least one immutable artifact prefix is required.")
    target, references = _resolved_roots(target_root, reference_roots)
    reference_files = [
        item
        for root in references
        for item in _regular_files(
            root,
            min_file_bytes=int(min_file_bytes),
            include_prefixes=prefixes,
        )
    ]
    target_files = _regular_files(
        target,
        min_file_bytes=int(min_file_bytes),
        include_prefixes=prefixes,
    )
    references_by_size: dict[int, list[_File]] = {}
    for item in reference_files:
        references_by_size.setdefault(item.size, []).append(item)
    candidate_targets = [item for item in target_files if item.size in references_by_size]
    digest_cache: dict[tuple[int, int], str] = {}

    def digest(item: _File) -> str:
        key = (item.device, item.inode)
        value = digest_cache.get(key)
        if value is None:
            value = _sha256(item.path)
            digest_cache[key] = value
        return value

    references_by_digest: dict[tuple[int, str], list[_File]] = {}
    for size in {item.size for item in candidate_targets}:
        for item in references_by_size[size]:
            references_by_digest.setdefault((size, digest(item)), []).append(item)

    replacements: list[HardlinkReplacement] = []
    already_hardlinked = 0
    cross_device = 0
    for target_file in candidate_targets:
        target_digest = digest(target_file)
        matches = references_by_digest.get((target_file.size, target_digest), ())
        if not matches:
            continue
        linked = next(
            (
                item
                for item in matches
                if (item.device, item.inode) == (target_file.device, target_file.inode)
            ),
            None,
        )
        if linked is not None:
            already_hardlinked += target_file.size
            continue
        same_device = next((item for item in matches if item.device == target_file.device), None)
        if same_device is None:
            cross_device += target_file.size
            continue
        replacements.append(
            HardlinkReplacement(
                reference_path=str(same_device.path),
                target_path=str(target_file.path),
                size=target_file.size,
                allocated_bytes=target_file.allocated_bytes,
                sha256=target_digest,
                reference_device=same_device.device,
                reference_inode=same_device.inode,
                reference_mtime_ns=same_device.mtime_ns,
                target_device=target_file.device,
                target_inode=target_file.inode,
                target_mtime_ns=target_file.mtime_ns,
            )
        )
    # Most dataset artifacts have one link. For multiply-linked target inodes,
    # replacing one directory entry may not release blocks, so do not promise it.
    reclaimable = sum(
        replacement.allocated_bytes
        for replacement in replacements
        if next(item for item in target_files if item.path == Path(replacement.target_path)).links == 1
    )
    return HardlinkDedupePlan(
        target_root=str(target),
        reference_roots=tuple(str(path) for path in references),
        scanned_reference_files=len(reference_files),
        scanned_target_files=len(target_files),
        hashed_files=len(digest_cache),
        replacement_count=len(replacements),
        logical_duplicate_bytes=sum(item.size for item in replacements),
        estimated_reclaimable_bytes=reclaimable,
        already_hardlinked_bytes=already_hardlinked,
        cross_device_duplicate_bytes=cross_device,
        replacements=tuple(replacements),
    )


def _same_snapshot(path: Path, replacement: HardlinkReplacement, *, reference: bool) -> bool:
    stat = path.stat()
    expected = (
        (
            replacement.reference_device,
            replacement.reference_inode,
            replacement.reference_mtime_ns,
        )
        if reference
        else (replacement.target_device, replacement.target_inode, replacement.target_mtime_ns)
    )
    return (
        int(stat.st_size) == replacement.size
        and (int(stat.st_dev), int(stat.st_ino), int(stat.st_mtime_ns)) == expected
    )


def apply_hardlink_dedupe(
    plan: HardlinkDedupePlan,
    *,
    confirm_replace_targets: bool = False,
) -> HardlinkDedupeResult:
    if not confirm_replace_targets:
        raise ValueError("apply requires confirm_replace_targets=True.")
    active = _active_collection_processes()
    if active:
        raise RuntimeError(
            "Refusing hardlink dedupe while formal collection is active: "
            + "; ".join(active[:3])
        )
    target_root = Path(plan.target_root).resolve()
    reference_roots = tuple(Path(value).resolve() for value in plan.reference_roots)
    replaced = 0
    logical = 0
    reclaimed = 0
    for replacement in plan.replacements:
        reference = Path(replacement.reference_path)
        target = Path(replacement.target_path)
        try:
            target.resolve().relative_to(target_root)
        except ValueError as exc:
            raise RuntimeError(f"Planned target escapes target_root: {target}") from exc
        if not any(
            reference.resolve() == root or root in reference.resolve().parents
            for root in reference_roots
        ):
            raise RuntimeError(f"Planned reference escapes reference_roots: {reference}")
        if not _same_snapshot(reference, replacement, reference=True) or not _same_snapshot(
            target, replacement, reference=False
        ):
            raise RuntimeError(f"Duplicate changed after planning; refusing replacement: {target}")
        target_stat = target.stat()
        if reference.stat().st_dev != target_stat.st_dev:
            raise RuntimeError(f"Duplicate moved across filesystems: {target}")
        fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(fd)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            os.link(reference, temporary)
            os.replace(temporary, target)
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        replaced += 1
        logical += replacement.size
        if target_stat.st_nlink == 1:
            reclaimed += replacement.allocated_bytes
    return HardlinkDedupeResult(
        target_root=plan.target_root,
        replacement_count=replaced,
        logical_duplicate_bytes=logical,
        estimated_reclaimed_bytes=reclaimed,
    )


__all__ = [
    "DEFAULT_MIN_FILE_BYTES",
    "DEFAULT_IMMUTABLE_PREFIXES",
    "HardlinkDedupePlan",
    "HardlinkDedupeResult",
    "HardlinkReplacement",
    "apply_hardlink_dedupe",
    "plan_hardlink_dedupe",
]
