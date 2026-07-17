"""Pluggable row encoding boundary for collection embodiments."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from dataset_contracts.v22 import PolicyIOLayout


class CollectionRowEncoder(Protocol):
    """Encode one causally assembled sample without owning media I/O."""

    def encode_row(
        self,
        sample: Any,
        *,
        layout: PolicyIOLayout | None,
        camera_video_paths: Mapping[str, str] | None,
        tactile_storage_refs: Mapping[str, Mapping[str, Any]] | None,
        inline_tactile: bool,
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        """Return the parquet row and per-camera frame expectations."""


__all__ = ["CollectionRowEncoder"]
