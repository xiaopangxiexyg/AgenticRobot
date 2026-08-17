"""Load canonical policy-I/O layout contracts from dataset metadata files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dataset_contracts.v22 import PolicyIOLayout


def load_policy_io_layout(path: str | Path | None) -> PolicyIOLayout | None:
    """Load a canonical layout JSON without changing default v2.1 collection.

    The file may contain either a layout object itself or an object with a
    top-level ``policy_io_layout`` key copied from dataset/checkpoint metadata.
    """

    if path is None or not str(path).strip():
        return None
    layout_path = Path(path).expanduser().resolve()
    root_payload: Any = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(root_payload, Mapping):
        raise ValueError(f"policy I/O layout file must contain a JSON object: {layout_path}")
    payload: Mapping[str, Any] = root_payload
    nested = root_payload.get("policy_io_layout")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError(f"policy_io_layout in {layout_path} must be a JSON object.")
        payload = nested
    layout = PolicyIOLayout.from_dict(payload)
    declared_hash = (
        root_payload.get("policy_io_layout_hash")
        or payload.get("policy_io_layout_hash")
        or payload.get("layout_hash")
    )
    if declared_hash is not None and str(declared_hash) != layout.layout_hash:
        raise ValueError(
            f"policy I/O layout hash mismatch in {layout_path}: "
            f"expected {layout.layout_hash}, got {declared_hash}."
        )
    return layout


__all__ = ["load_policy_io_layout"]
