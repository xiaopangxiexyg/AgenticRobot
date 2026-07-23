"""Bind the optional dataset-wide DM-Tac gripper force profile."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import fcntl

from dataset_contracts.gripper_force import GripperForceDatasetContract
from .capture_modalities import dataset_is_blank


GRIPPER_FORCE_PROFILE_METADATA_KEY = "gripper_force_profile"


def _metadata_path(dataset_root: str | Path) -> Path:
    return Path(dataset_root).expanduser() / "meta" / "modality_validity.json"


def _read_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset modality declaration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"dataset modality declaration must be an object: {path}")
    return value


def read_gripper_force_contract(
    dataset_root: str | Path,
) -> GripperForceDatasetContract | None:
    path = _metadata_path(dataset_root)
    payload = _read_payload(path)
    value = payload.get(GRIPPER_FORCE_PROFILE_METADATA_KEY)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{GRIPPER_FORCE_PROFILE_METADATA_KEY} must be an object")
    try:
        return GripperForceDatasetContract.from_dict(value)
    except ValueError as exc:
        raise ValueError(f"invalid {path}.{GRIPPER_FORCE_PROFILE_METADATA_KEY}: {exc}") from exc


def bind_gripper_force_contract(
    dataset_root: str | Path,
    contract: GripperForceDatasetContract,
) -> Path:
    """Persist a force profile, allowing writes only to a blank dataset.

    A populated dataset is intentionally not backfilled or changed by this
    button.  Existing datasets can still be read, but a new permanent force
    selection requires a new/empty dataset root.
    """

    root = Path(dataset_root).expanduser()
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / "modality_validity.json"
    lock_path = meta_dir / ".modality_validity.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = _read_payload(path)
            existing = read_gripper_force_contract(root)
            if not dataset_is_blank(root):
                if existing is None:
                    raise ValueError(
                        "permanent gripper force can only be set on a new/empty dataset; "
                        "the populated dataset has no force profile declaration"
                    )
                if existing != contract:
                    raise ValueError(
                        "target dataset already contains a different gripper force profile: "
                        f"existing={existing.to_dict()}, requested={contract.to_dict()}"
                    )
                raise ValueError(
                    "permanent gripper force can only be set on a new/empty dataset; "
                    "create a new dataset before changing it"
                )
            payload[GRIPPER_FORCE_PROFILE_METADATA_KEY] = contract.to_dict()
            payload.setdefault("schema_version", "dexteleop.modality_validity.v1")
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return path


__all__ = [
    "GRIPPER_FORCE_PROFILE_METADATA_KEY",
    "bind_gripper_force_contract",
    "read_gripper_force_contract",
]
