from __future__ import annotations

import errno
import importlib.util
from pathlib import Path


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "todos/probes/export_realman_tactile_to_nas.py"
)
_SPEC = importlib.util.spec_from_file_location("export_realman_tactile_to_nas", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_export_auto_resume_classifies_transient_cifs_error() -> None:
    assert _MODULE._retryable_export_error(OSError(errno.EBUSY, "busy"))


def test_export_auto_resume_rejects_integrity_and_permanent_io_errors() -> None:
    assert not _MODULE._retryable_export_error(RuntimeError("hash mismatch"))
    assert not _MODULE._retryable_export_error(OSError(errno.EIO, "I/O error"))


def test_export_run_state_tracks_only_completed_entries(tmp_path: Path) -> None:
    state = tmp_path / "results.jsonl"
    state.write_text(
        '{"entry_index": 3, "status": "retry_scheduled"}\n'
        '{"entry_index": 4, "status": "complete"}\n'
        'partial trailing line',
        encoding="utf-8",
    )

    assert _MODULE._completed_entry_indices(state) == {4}
