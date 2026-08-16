from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_gateway.protocol.artifact_contracts import AgentArtifactRef
from agent_gateway.protocol.session_contracts import (
    AgentGatewaySessionCapabilities,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker


def _artifact(data: bytes) -> AgentArtifactRef:
    return AgentArtifactRef(
        media_type="application/octet-stream",
        schema_id=None,
        encoding="identity",
        compression="none",
        byte_length=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _runtime(tmp_path: Path):
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "store",
        quota_bytes=1024,
        max_artifact_bytes=1024,
        max_chunk_bytes=4,
        staging_ttl_sec=60.0,
        minimum_retention_sec=10.0,
        handoff_retention_sec=100.0,
    )
    capabilities = AgentGatewaySessionCapabilities(
        runtime_session_id="runtime:test",
        artifact_transfer_enabled=True,
        artifact_access_mechanisms=("chunked_rpc", "local_path"),
        max_control_request_bytes=1024,
        max_control_response_bytes=1024,
        max_artifact_bytes=1024,
        max_artifact_chunk_bytes=4,
        max_artifact_refs_per_operation=4,
        max_concurrent_uploads=2,
        max_concurrent_downloads=2,
        artifact_store_quota_bytes=1024,
        event_payload_max_bytes=1024,
        event_retention_count=16,
    )
    broker = AgentArtifactTransferBroker(
        store=store,
        session_capabilities=capabilities,
        max_chunk_bytes=4,
    )
    return store, broker


def test_chunked_upload_download_and_ack_are_session_bound(tmp_path: Path) -> None:
    data = b"abcdefghij"
    store, broker = _runtime(tmp_path)
    try:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:one",
            artifact=_artifact(data),
            now=0.0,
        )
        write_grant = broker.issue_write_grant(
            lease=lease,
            mechanism="chunked_rpc",
            now=1.0,
        )
        assert "absolute_path" not in write_grant.to_wire()["locator"]
        for offset in range(0, len(data), 4):
            broker.write_chunk(
                grant_id=write_grant.grant_id,
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                offset=offset,
                data=data[offset : offset + 4],
                now=2.0 + offset,
            )
        artifact, upload_receipt = broker.commit_write(
            grant_id=write_grant.grant_id,
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            now=20.0,
        )
        assert upload_receipt.verified_sha256 == artifact.sha256

        read_grant = broker.issue_read_grant(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            artifact_id=artifact.artifact_id,
            mechanism="chunked_rpc",
            now=21.0,
        )
        received = b"".join(
            broker.read_chunk(
                grant_id=read_grant.grant_id,
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                offset=offset,
                max_bytes=4,
                now=22.0 + offset,
            )
            for offset in range(0, len(data), 4)
        )
        assert received == data
        receipt = broker.acknowledge_read(
            grant_id=read_grant.grant_id,
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            transferred_bytes=len(data),
            verified_sha256=hashlib.sha256(received).hexdigest(),
            now=40.0,
        )
        assert receipt.status == "completed"
        with pytest.raises(PermissionError, match="already completed"):
            broker.read_chunk(
                grant_id=read_grant.grant_id,
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                offset=0,
                max_bytes=4,
                now=41.0,
            )
    finally:
        store.close()


def test_local_path_is_read_only_and_only_in_the_ephemeral_grant(
    tmp_path: Path,
) -> None:
    data = b"local-path"
    store, broker = _runtime(tmp_path)
    try:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:local",
            artifact=_artifact(data),
            now=0.0,
        )
        with pytest.raises(ValueError, match="uploads require"):
            broker.issue_write_grant(
                lease=lease,
                mechanism="local_path",
                now=1.0,
            )
        grant = broker.issue_write_grant(
            lease=lease,
            mechanism="chunked_rpc",
            now=1.0,
        )
        for offset in range(0, len(data), 4):
            broker.write_chunk(
                grant_id=grant.grant_id,
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                offset=offset,
                data=data[offset : offset + 4],
                now=1.0 + offset,
            )
        artifact, _ = broker.commit_write(
            grant_id=grant.grant_id,
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            now=2.0,
        )
        read = broker.issue_read_grant(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            artifact_id=artifact.artifact_id,
            mechanism="local_path",
            now=3.0,
        )
        locator = read.to_wire()["locator"]
        path = Path(locator["absolute_path"])
        assert path.is_relative_to(store.root)
        assert locator["read_only"] is True
        assert "absolute_path" not in read.artifact.to_wire()
        assert path.read_bytes() == data
    finally:
        store.close()


def test_retried_write_grant_does_not_consume_another_concurrency_slot(
    tmp_path: Path,
) -> None:
    store, broker = _runtime(tmp_path)
    try:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:retry",
            artifact=_artifact(b"retry"),
            now=0.0,
        )
        first = broker.issue_write_grant(
            lease=lease,
            mechanism="chunked_rpc",
            now=1.0,
        )
        second = broker.issue_write_grant(
            lease=store.upload(lease.upload_id),
            mechanism="chunked_rpc",
            now=2.0,
        )
        assert second == first
    finally:
        store.close()


def test_grants_reject_wrong_client_expiry_and_partial_identity(tmp_path: Path) -> None:
    data = b"bounded"
    store, broker = _runtime(tmp_path)
    try:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:bounded",
            artifact=_artifact(data),
            now=0.0,
        )
        grant = broker.issue_write_grant(
            lease=lease,
            mechanism="chunked_rpc",
            now=1.0,
            ttl_sec=2.0,
        )
        with pytest.raises(PermissionError, match="another client"):
            broker.write_chunk(
                grant_id=grant.grant_id,
                client_id="other:client",
                runtime_session_id="runtime:test",
                offset=0,
                data=data[:4],
                now=2.0,
            )
        with pytest.raises(PermissionError, match="expired"):
            broker.write_chunk(
                grant_id=grant.grant_id,
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                offset=0,
                data=data[:4],
                now=3.0,
            )

        artifact = _artifact(b"ready")
        other = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:ready",
            artifact=artifact,
            now=5.0,
        )
        store.write_upload_chunk(
            upload_id=other.upload_id,
            client_id=other.client_id,
            runtime_session_id=other.runtime_session_id,
            offset=0,
            data=b"read",
            now=6.0,
        )
        store.write_upload_chunk(
            upload_id=other.upload_id,
            client_id=other.client_id,
            runtime_session_id=other.runtime_session_id,
            offset=4,
            data=b"y",
            now=7.0,
        )
        store.commit_upload(
            upload_id=other.upload_id,
            client_id=other.client_id,
            runtime_session_id=other.runtime_session_id,
            now=8.0,
        )
        with pytest.raises(ValueError, match="complete Artifact"):
            broker.issue_read_grant(
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                artifact_id=artifact.artifact_id,
                mechanism="chunked_rpc",
                now=9.0,
                byte_offset=1,
            )
    finally:
        store.close()


def test_transfer_grants_enforce_advertised_concurrency(tmp_path: Path) -> None:
    store, broker = _runtime(tmp_path)
    try:
        leases = tuple(
            store.begin_upload(
                client_id="rpent:local",
                runtime_session_id="runtime:test",
                idempotency_key=f"upload:{name}",
                artifact=_artifact(name.encode("ascii")),
                now=0.0,
            )
            for name in ("first", "second", "third")
        )
        for lease in leases[:2]:
            broker.issue_write_grant(
                lease=lease,
                mechanism="chunked_rpc",
                now=1.0,
                ttl_sec=5.0,
            )
        with pytest.raises(RuntimeError, match="limit is exhausted"):
            broker.issue_write_grant(
                lease=leases[2],
                mechanism="chunked_rpc",
                now=2.0,
            )
        broker.issue_write_grant(
            lease=leases[2],
            mechanism="chunked_rpc",
            now=6.0,
        )
    finally:
        store.close()


def test_empty_artifact_can_commit_and_ack_without_data_chunks(
    tmp_path: Path,
) -> None:
    store, broker = _runtime(tmp_path)
    try:
        lease = store.begin_upload(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            idempotency_key="upload:empty",
            artifact=_artifact(b""),
            now=0.0,
        )
        write_grant = broker.issue_write_grant(
            lease=lease,
            mechanism="chunked_rpc",
            now=1.0,
        )
        artifact, _ = broker.commit_write(
            grant_id=write_grant.grant_id,
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            now=2.0,
        )
        read_grant = broker.issue_read_grant(
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            artifact_id=artifact.artifact_id,
            mechanism="chunked_rpc",
            now=3.0,
        )
        receipt = broker.acknowledge_read(
            grant_id=read_grant.grant_id,
            client_id="rpent:local",
            runtime_session_id="runtime:test",
            transferred_bytes=0,
            verified_sha256=artifact.sha256,
            now=4.0,
        )
        assert receipt.transferred_bytes == 0
    finally:
        store.close()
