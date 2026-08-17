from __future__ import annotations

import pytest

from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore


def _store(tmp_path):
    return RecoverableAgentArtifactStore(
        root=tmp_path / "store",
        quota_bytes=4096,
        max_artifact_bytes=2048,
        max_chunk_bytes=3,
    )


def test_provider_commit_chunks_and_replays_idempotently(tmp_path) -> None:
    store = _store(tmp_path)
    data = b"depth-bytes"
    artifact = artifact_ref_for_bytes(
        data,
        media_type="application/x-npy",
        schema_id="depth.v1",
        encoding="npy.float32.c_order.v1",
        compression="none",
    )
    first = commit_artifact_bytes(
        store=store,
        data=data,
        artifact=artifact,
        producer_id="provider.depth",
        runtime_session_id="session:1",
        idempotency_key="operation:1:depth",
        now=1.0,
    )
    replay = commit_artifact_bytes(
        store=store,
        data=data,
        artifact=artifact,
        producer_id="provider.depth",
        runtime_session_id="session:1",
        idempotency_key="operation:1:depth",
        now=2.0,
    )
    assert first == replay == artifact
    assert store.managed_read_path(artifact.artifact_id).read_bytes() == data
    store.close()


def test_provider_commit_rejects_descriptor_byte_mismatch(tmp_path) -> None:
    store = _store(tmp_path)
    artifact = artifact_ref_for_bytes(
        b"expected",
        media_type="application/octet-stream",
        schema_id=None,
        encoding="raw",
        compression="none",
    )
    with pytest.raises(ValueError, match="do not match"):
        commit_artifact_bytes(
            store=store,
            data=b"different",
            artifact=artifact,
            producer_id="provider.test",
            runtime_session_id="session:1",
            idempotency_key="operation:1:output",
            now=1.0,
        )
    assert not tuple((tmp_path / "store" / "staging").iterdir())
    store.close()
