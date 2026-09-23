"""Bind a dataset-wide capture modality contract to a dataset root."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import fcntl

from dataset_contracts.capture_modalities import CaptureModalityContract


MODALITY_VALIDITY_SCHEMA_VERSION = "dexteleop.modality_validity.v1"
CAPTURE_MODALITY_METADATA_KEY = "capture"


class CaptureModalityBindingError(ValueError):
    """Raised when an append would mix different ignore rules."""


def _read_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CaptureModalityBindingError(
            f"cannot read capture modality declaration {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CaptureModalityBindingError(
            f"capture modality declaration must be a JSON object: {path}"
        )
    return value


def _has_nonempty_file(root: Path) -> bool:
    if not root.exists():
        return False
    for directory_name in ("data", "videos", "tactile", "wrist_depth"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for current_root, _directories, files in os.walk(directory):
            if any((Path(current_root) / name).stat().st_size > 0 for name in files):
                return True
    return False


def dataset_is_blank(dataset_root: str | Path) -> bool:
    """Return whether no episode/sample payload has been committed or staged."""

    root = Path(dataset_root).expanduser()
    episodes_path = root / "meta" / "episodes.jsonl"
    if episodes_path.exists() and episodes_path.read_text(encoding="utf-8").strip():
        return False
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = _read_object(info_path)
        if int(info.get("episode_count", 0) or 0) > 0:
            return False
        if int(info.get("frame_count", 0) or 0) > 0:
            return False
    return not _has_nonempty_file(root)


def read_capture_modality_contract(
    dataset_root: str | Path,
) -> CaptureModalityContract | None:
    path = Path(dataset_root).expanduser() / "meta" / "modality_validity.json"
    payload = _read_object(path)
    capture = payload.get(CAPTURE_MODALITY_METADATA_KEY)
    if capture is None:
        return None
    if not isinstance(capture, Mapping):
        raise CaptureModalityBindingError(f"{path}.capture must be an object")
    try:
        return CaptureModalityContract.from_dict(capture)
    except ValueError as exc:
        raise CaptureModalityBindingError(f"invalid {path}.capture: {exc}") from exc


def bind_capture_modality_contract(
    dataset_root: str | Path,
    contract: CaptureModalityContract,
) -> Path:
    """Persist *contract*, rejecting policy changes on a populated dataset.

    A populated dataset must already carry an explicit declaration. Payload
    presence alone cannot distinguish a legacy tactile dataset from a legacy
    no-tactile dataset, so guessing here could silently create a mixed dataset.
    """

    root = Path(dataset_root).expanduser()
    meta_dir = root / "meta"
    path = meta_dir / "modality_validity.json"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lock_path = meta_dir / ".modality_validity.lock"
    # Multiple GUI/headless processes may target the same new dataset.  Keep
    # read/blank-check/write under one process-shared lock so two different
    # ignore policies cannot both observe an unbound root and race to commit.
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = _read_object(path)
            existing = read_capture_modality_contract(root)
            blank = dataset_is_blank(root)
            if existing is None and not blank:
                raise CaptureModalityBindingError(
                    "target dataset already contains data but has no capture modality "
                    "contract; migrate meta/modality_validity.json explicitly before "
                    "appending so tactile and non-tactile episodes cannot be mixed"
                )
            effective_existing = existing
            if not blank and effective_existing != contract:
                raise CaptureModalityBindingError(
                    "target dataset already contains data with a different modality ignore "
                    f"contract: existing={effective_existing.to_dict() if effective_existing else None}, "
                    f"requested={contract.to_dict()}"
                )

            schema = payload.get("schema_version")
            if schema not in (None, MODALITY_VALIDITY_SCHEMA_VERSION):
                raise CaptureModalityBindingError(
                    f"unsupported modality_validity schema_version={schema!r} in {path}"
                )
            payload["schema_version"] = MODALITY_VALIDITY_SCHEMA_VERSION
            payload[CAPTURE_MODALITY_METADATA_KEY] = contract.to_dict()
            temporary = path.with_name(
                f"{path.name}.{os.getpid()}.tmp"
            )
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return path


__all__ = [
    "CAPTURE_MODALITY_METADATA_KEY",
    "CaptureModalityBindingError",
    "bind_capture_modality_contract",
    "dataset_is_blank",
    "read_capture_modality_contract",
]
