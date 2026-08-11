"""Legacy debug-session recording path and writer configuration."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..recording import RecordingConfig
from collection_runtime.dataset_layout import expand_cli_path as _expand_cli_path

def _session_record_root_dir(args: argparse.Namespace) -> Path | None:
    cached = str(getattr(args, "_session_record_root_dir", "") or "").strip()
    if cached:
        return Path(cached)
    session_record_root = str(getattr(args, "session_record_root", "") or "").strip()
    canonical_session_dir = str(
        getattr(args, "_runtime_artifact_session_dir", "") or ""
    ).strip()
    if not session_record_root:
        if not canonical_session_dir:
            return None
        root_dir = Path(canonical_session_dir).expanduser()
        setattr(args, "_session_record_root_dir", str(root_dir))
        return root_dir
    stamp = str(getattr(args, "_session_record_stamp", "") or "").strip()
    if not stamp:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        setattr(args, "_session_record_stamp", stamp)
    root_dir = _expand_cli_path(session_record_root) / f"{stamp}_vive_rm65_real_teleop"
    setattr(args, "_session_record_root_dir", str(root_dir))
    return root_dir


def _session_recording_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "no_record", True))


def _session_recording_config(args: argparse.Namespace) -> RecordingConfig:
    return RecordingConfig(
        enabled=True,
        save_commands=not bool(getattr(args, "no_session_record_commands", False)),
        flush_every=max(1, int(getattr(args, "session_record_flush_every", 1) or 1)),
        compact_json=bool(getattr(args, "session_record_compact_json", False)),
    )



__all__ = [
    "_session_record_root_dir",
    "_session_recording_config",
    "_session_recording_enabled",
]
