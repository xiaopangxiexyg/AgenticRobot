"""Characterize the JSON publication boundary used by collection stores."""

from __future__ import annotations

import errno
from functools import partial
import json
import os
from pathlib import Path
import stat

import pytest

from collection_runtime import (
    dataset_capture_recovery,
    modality_inventory,
    modality_storage,
    tactile_bundle_materialize,
    tactile_bundle_store,
    tactile_recompression,
)


_WRITERS = {
    "inventory": modality_inventory._atomic_write,
    "recompression": tactile_recompression._atomic_write_json,
    "materialize": tactile_bundle_materialize._atomic_write_json,
    # This leaf does not consume store state; the store workflows have their
    # own recovery tests in test_tactile_bundle_native.py.
    "bundle_recovery": partial(
        tactile_bundle_store.NativeTactileBundleStore._atomic_write_recovery_json,
        None,
    ),
    "capture_recovery": dataset_capture_recovery._atomic_write_json,
    "modality_storage": modality_storage._atomic_write_json,
}


@pytest.fixture(params=tuple(_WRITERS))
def writer(request):
    return request.param, _WRITERS[request.param]


def _assert_no_temporary(path: Path) -> None:
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_publication_bytes_order_and_fresh_inode(writer, tmp_path, monkeypatch):
    name, publish = writer
    path = tmp_path / "manifest.json"
    path.write_bytes(b"old contents")
    retained = tmp_path / "retained.json"
    os.link(path, retained)
    payload = {"z": [None, True], "a": "触觉", "n": float("nan")}
    expected = (
        json.dumps(payload, ensure_ascii=name != "capture_recovery", sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")
    events = []
    opened_directories = []
    real_open, real_sync, real_replace = os.open, os.fsync, os.replace

    def open_file(filename, flags, *args, **kwargs):
        fd = real_open(filename, flags, *args, **kwargs)
        if Path(filename) == path.parent:
            opened_directories.append((fd, flags))
        return fd

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            events.append("directory_sync")
            assert path.read_bytes() == expected
        else:
            events.append("file_sync")
            assert os.pread(fd, len(expected) + 1, 0) == expected
            assert path.read_bytes() == b"old contents"
        real_sync(fd)

    def replace(source, target):
        events.append("replace")
        assert Path(source).parent == path.parent
        assert Path(source).name.startswith(f".{path.name}.")
        assert Path(source).suffix == ".tmp"
        assert Path(target) == path
        real_replace(source, target)

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    assert publish(path, payload) is None
    assert events == ["file_sync", "replace", "directory_sync"]
    directory_flag = (
        getattr(os, "O_DIRECTORY", 0)
        if name in {"materialize", "bundle_recovery", "capture_recovery"}
        else 0
    )
    assert len(opened_directories) == 1
    directory_fd, flags = opened_directories[0]
    assert flags == os.O_RDONLY | directory_flag
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(directory_fd)
    assert path.read_bytes() == expected
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.stat().st_ino != retained.stat().st_ino
    assert retained.read_bytes() == b"old contents"
    _assert_no_temporary(path)


@pytest.mark.parametrize("failure", ["file_sync", "replace", "directory_open", "directory_sync"])
def test_publication_failure_keeps_the_correct_visible_version(
    writer, failure, tmp_path, monkeypatch
):
    _, publish = writer
    path = tmp_path / "manifest.json"
    path.write_bytes(b"old contents")
    injected = OSError(errno.EIO, failure)
    directory_fds = []
    real_open, real_sync, real_replace = os.open, os.fsync, os.replace

    def open_file(filename, flags, *args, **kwargs):
        if Path(filename) == path.parent and failure == "directory_open":
            raise injected
        fd = real_open(filename, flags, *args, **kwargs)
        if Path(filename) == path.parent:
            directory_fds.append(fd)
        return fd

    def sync(fd):
        stage = "directory_sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_sync"
        if stage == failure:
            raise injected
        real_sync(fd)

    def replace(source, target):
        if failure == "replace":
            raise injected
        real_replace(source, target)

    monkeypatch.setattr(os, "open", open_file)
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError) as caught:
        publish(path, {"version": 2})
    assert caught.value is injected
    if failure in {"file_sync", "replace"}:
        assert path.read_bytes() == b"old contents"
    else:
        assert json.loads(path.read_bytes()) == {"version": 2}
    for fd in directory_fds:
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(fd)
    _assert_no_temporary(path)


def test_serialization_failure_does_not_publish_or_leave_a_temporary(writer, tmp_path):
    _, publish = writer
    path = tmp_path / "new-parent" / "manifest.json"
    with pytest.raises(TypeError):
        publish(path, {"unsupported": object()})
    assert path.parent.is_dir()
    assert not path.exists()
    _assert_no_temporary(path)


def test_only_modality_storage_retries_transient_directory_sync(writer, tmp_path, monkeypatch):
    name, publish = writer
    path = tmp_path / "manifest.json"
    attempts = []
    sleeps = []
    injected = BlockingIOError(errno.EAGAIN, "try again")
    real_sync = os.fsync

    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            attempts.append(fd)
            if len(attempts) == 1:
                raise injected
        real_sync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(modality_storage.time, "sleep", sleeps.append)
    if name == "modality_storage":
        assert publish(path, {"version": 2}) is None
        assert len(attempts) == 2
        assert sleeps == [modality_storage.FSYNC_INITIAL_RETRY_DELAY_SEC]
    else:
        with pytest.raises(BlockingIOError) as caught:
            publish(path, {"version": 2})
        assert caught.value is injected
        assert len(attempts) == 1
        assert sleeps == []
    assert json.loads(path.read_bytes()) == {"version": 2}
    _assert_no_temporary(path)


def test_capture_recovery_preserves_missing_directory_sync_policy(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    real_is_dir, real_sync, real_replace = Path.is_dir, os.fsync, os.replace
    synced = []

    def is_dir(value):
        return False if value == path.parent else real_is_dir(value)

    def sync(fd):
        synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real_sync(fd)

    def replace(source, target):
        real_replace(source, target)
        monkeypatch.setattr(Path, "is_dir", is_dir)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "fsync", sync)
    dataset_capture_recovery._atomic_write_json(path, {"version": 2})
    assert synced == [False]
    assert json.loads(path.read_bytes()) == {"version": 2}
