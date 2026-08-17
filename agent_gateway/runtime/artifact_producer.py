"""Small provider-side helper for committing finite bytes to the v2 CAS."""

from __future__ import annotations

import hashlib

from agent_gateway.protocol.artifact_contracts import AgentArtifactRef

from .artifact_store import RecoverableAgentArtifactStore


def artifact_ref_for_bytes(
    data: bytes | bytearray | memoryview,
    *,
    media_type: str,
    schema_id: str | None,
    encoding: str,
    compression: str,
) -> AgentArtifactRef:
    payload = bytes(data)
    return AgentArtifactRef(
        media_type=media_type,
        schema_id=schema_id,
        encoding=encoding,
        compression=compression,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def commit_artifact_bytes(
    *,
    store: RecoverableAgentArtifactStore,
    data: bytes | bytearray | memoryview,
    artifact: AgentArtifactRef,
    producer_id: str,
    runtime_session_id: str,
    idempotency_key: str,
    now: float,
) -> AgentArtifactRef:
    """Commit exact bytes through the same resumable store path as uploads."""

    if not isinstance(store, RecoverableAgentArtifactStore):
        raise TypeError("store must be a RecoverableAgentArtifactStore.")
    payload = bytes(data)
    expected = artifact_ref_for_bytes(
        payload,
        media_type=artifact.media_type,
        schema_id=artifact.schema_id,
        encoding=artifact.encoding,
        compression=artifact.compression,
    )
    if expected != artifact:
        raise ValueError("Provider bytes do not match the Artifact descriptor.")
    lease = store.begin_upload(
        client_id=producer_id,
        runtime_session_id=runtime_session_id,
        idempotency_key=idempotency_key,
        artifact=artifact,
        now=now,
    )
    if not lease.committed:
        offset = lease.received_bytes
        while offset < len(payload):
            chunk = payload[offset : offset + store.max_chunk_bytes]
            lease = store.write_upload_chunk(
                upload_id=lease.upload_id,
                client_id=producer_id,
                runtime_session_id=runtime_session_id,
                offset=offset,
                data=chunk,
                now=now,
            )
            offset = lease.received_bytes
        committed = store.commit_upload(
            upload_id=lease.upload_id,
            client_id=producer_id,
            runtime_session_id=runtime_session_id,
            now=now,
        )
    else:
        committed = lease.artifact
    if committed != artifact:
        raise RuntimeError("Artifact store committed another descriptor.")
    return committed


__all__ = [
    "artifact_ref_for_bytes",
    "commit_artifact_bytes",
]
