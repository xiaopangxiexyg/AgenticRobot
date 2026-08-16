from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactMaterializationReceipt,
    AgentArtifactRef,
)
from agent_gateway.runtime.artifact_store import (
    ArtifactStoreCapacityError,
    ArtifactStoreConflictError,
    ArtifactStoreIntegrityError,
    RecoverableAgentArtifactStore,
)


def _artifact(data: bytes, *, media_type: str = "application/octet-stream"):
    return AgentArtifactRef(
        media_type=media_type,
        schema_id=None,
        encoding="identity",
        compression="none",
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _store(root: Path, *, quota_bytes: int = 1024):
    return RecoverableAgentArtifactStore(
        root=root,
        quota_bytes=quota_bytes,
        max_artifact_bytes=quota_bytes,
        max_chunk_bytes=16,
        staging_ttl_sec=30.0,
        minimum_retention_sec=10.0,
        handoff_retention_sec=100.0,
    )


def _upload(
    store: RecoverableAgentArtifactStore,
    *,
    data: bytes,
    key: str = "upload:key",
    now: float = 0.0,
) -> AgentArtifactRef:
    artifact = _artifact(data)
    lease = store.begin_upload(
        client_id="rpent:local",
        runtime_session_id="runtime:test",
        idempotency_key=key,
        artifact=artifact,
        now=now,
    )
    offset = 0
    while offset < len(data):
        chunk = data[offset : offset + 8]
        lease = store.write_upload_chunk(
            upload_id=lease.upload_id,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            offset=offset,
            data=chunk,
            now=now + 1.0,
        )
        offset += len(chunk)
    return store.commit_upload(
        upload_id=lease.upload_id,
        client_id=lease.client_id,
        runtime_session_id=lease.runtime_session_id,
        now=now + 2.0,
    )


def test_store_requires_a_private_absolute_managed_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute path"):
        _store(Path("relative/store"))

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        _store(link)


def test_upload_is_resumable_idempotent_and_available_only_after_commit(
    tmp_path: Path,
) -> None:
    data = b"0123456789abcdef"
    artifact = _artifact(data)
    with _store(tmp_path / "store") as store:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:one",
            artifact=artifact,
            now=0.0,
        )
        assert not lease.committed
        assert not hasattr(lease, "path")
        with pytest.raises(KeyError):
            store.artifact(artifact.artifact_id)

        first = store.write_upload_chunk(
            upload_id=lease.upload_id,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            offset=0,
            data=data[:8],
            now=1.0,
        )
        replay = store.write_upload_chunk(
            upload_id=lease.upload_id,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            offset=0,
            data=data[:8],
            now=1.1,
        )
        assert replay.received_bytes == first.received_bytes == 8
        with pytest.raises(ArtifactStoreConflictError, match="differs"):
            store.write_upload_chunk(
                upload_id=lease.upload_id,
                client_id=lease.client_id,
                runtime_session_id=lease.runtime_session_id,
                offset=0,
                data=b"XXXXXXXX",
                now=1.2,
            )
        store.write_upload_chunk(
            upload_id=lease.upload_id,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            offset=8,
            data=data[8:],
            now=2.0,
        )
        committed = store.commit_upload(
            upload_id=lease.upload_id,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            now=3.0,
        )
        assert committed == artifact
        assert store.artifact(artifact.artifact_id).available
        assert store.managed_read_path(artifact.artifact_id).read_bytes() == data

        same = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:one",
            artifact=artifact,
            now=4.0,
        )
        assert same.upload_id == lease.upload_id
        assert same.committed
        with pytest.raises(ArtifactStoreConflictError, match="another Artifact"):
            store.begin_upload(
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                idempotency_key="upload:one",
                artifact=_artifact(b"different"),
                now=4.0,
            )


def test_hash_mismatch_quarantines_staging_and_never_publishes_ref(
    tmp_path: Path,
) -> None:
    expected = _artifact(b"expected")
    with _store(tmp_path / "store") as store:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:bad",
            artifact=expected,
            now=0.0,
        )
        store.write_upload_chunk(
            upload_id=lease.upload_id,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            offset=0,
            data=b"tampered",
            now=1.0,
        )
        with pytest.raises(ArtifactStoreIntegrityError, match="size/hash"):
            store.commit_upload(
                upload_id=lease.upload_id,
                client_id=lease.client_id,
                runtime_session_id=lease.runtime_session_id,
                now=2.0,
            )
        assert store.upload(lease.upload_id).state == "quarantined"
        with pytest.raises(KeyError):
            store.artifact(expected.artifact_id)


def test_quota_is_reserved_before_bytes_are_accepted(tmp_path: Path) -> None:
    with _store(tmp_path / "store", quota_bytes=12) as store:
        store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:first",
            artifact=_artifact(b"12345678"),
            now=0.0,
        )
        with pytest.raises(ArtifactStoreCapacityError, match="quota"):
            store.begin_upload(
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                idempotency_key="upload:second",
                artifact=_artifact(b"abcdefgh"),
                now=0.0,
            )


def test_operation_pin_and_materialization_gate_gc(tmp_path: Path) -> None:
    data = b"depth-frame"
    with _store(tmp_path / "store") as store:
        artifact = _upload(store, data=data)
        store.pin(
            pin_id="pin:operation",
            artifact_id=artifact.artifact_id,
            pin_kind="operation_output",
            owner_id="operation:one",
            now=3.0,
            expires_at=20.0,
        )
        store.record_materialization(
            AgentArtifactMaterializationReceipt(
                receipt_id="receipt:materialized",
                artifact_id=artifact.artifact_id,
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                durable_store_id="rpent:experience",
                durable_object_ref="experience-artifact:one",
                materialized_at=5.0,
                verified_sha256=artifact.sha256,
            )
        )

        before_pin_expiry = store.garbage_collect(now=15.0)
        assert not before_pin_expiry.expired_artifact_ids
        assert store.artifact(artifact.artifact_id).available

        after_pin_expiry = store.garbage_collect(now=21.0)
        assert after_pin_expiry.released_pin_ids == ("pin:operation",)
        assert after_pin_expiry.expired_artifact_ids == (artifact.artifact_id,)
        assert after_pin_expiry.reclaimed_bytes == len(data)
        tombstone = store.artifact(artifact.artifact_id)
        assert tombstone.state == "expired"
        assert tombstone.deletion_reason == (
            "materialized_and_minimum_retention_elapsed"
        )
        with pytest.raises(ArtifactStoreConflictError, match="not available"):
            store.managed_read_path(artifact.artifact_id)


def test_without_materialization_output_survives_until_handoff_deadline(
    tmp_path: Path,
) -> None:
    with _store(tmp_path / "store") as store:
        artifact = _upload(store, data=b"handoff")
        assert not store.garbage_collect(now=99.0).expired_artifact_ids
        report = store.garbage_collect(now=102.0)
        assert report.expired_artifact_ids == (artifact.artifact_id,)
        assert store.artifact(artifact.artifact_id).deletion_reason == (
            "handoff_deadline_elapsed"
        )


def test_store_reopens_and_quarantines_corrupted_committed_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "store"
    store = _store(root)
    artifact = _upload(store, data=b"persistent")
    path = store.managed_read_path(artifact.artifact_id)
    store.close()

    reopened = _store(root)
    try:
        assert reopened.artifact(artifact.artifact_id).available
        assert reopened.managed_read_path(artifact.artifact_id).read_bytes() == (
            b"persistent"
        )
        os.chmod(path, 0o600)
        path.write_bytes(b"corrupted!")
        quarantined = reopened.recover(now=50.0)
        assert quarantined == (artifact.artifact_id,)
        assert reopened.artifact(artifact.artifact_id).state == "quarantined"
    finally:
        reopened.close()


def test_expired_staging_is_aborted_and_reclaims_its_private_inode(
    tmp_path: Path,
) -> None:
    with _store(tmp_path / "store") as store:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:expires",
            artifact=_artifact(b"unfinished"),
            now=0.0,
        )
        report = store.garbage_collect(now=31.0)
        assert report.aborted_upload_ids == (lease.upload_id,)
        assert store.upload(lease.upload_id).state == "aborted"
