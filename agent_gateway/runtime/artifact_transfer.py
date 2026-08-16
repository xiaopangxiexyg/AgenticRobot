"""Session-bound access-grant broker for the v2 Artifact data plane."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from typing import Any
import uuid

from agent_gateway.protocol._contract_validation import (
    finite_number,
    identifier,
    nonnegative_integer,
    positive_integer,
    sha256_digest,
)
from agent_gateway.protocol.artifact_contracts import (
    AgentArtifactAccessGrant,
    AgentArtifactRef,
    AgentArtifactTransferReceipt,
)
from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities

from .artifact_store import (
    ArtifactUploadLease,
    RecoverableAgentArtifactStore,
)


@dataclass
class _GrantState:
    grant: AgentArtifactAccessGrant
    use_count: int = 0
    upload_id: str | None = None
    completed: bool = False


class ArtifactTransferCapacityError(RuntimeError):
    """Raised before issuing a grant beyond advertised session concurrency."""


class AgentArtifactTransferBroker:
    """Issue ephemeral locators while keeping Artifact identity transport-free."""

    def __init__(
        self,
        *,
        store: RecoverableAgentArtifactStore,
        session_capabilities: AgentGatewaySessionCapabilities,
        max_chunk_bytes: int,
    ) -> None:
        if not isinstance(store, RecoverableAgentArtifactStore):
            raise TypeError("store must be a RecoverableAgentArtifactStore.")
        if not isinstance(
            session_capabilities,
            AgentGatewaySessionCapabilities,
        ):
            raise TypeError(
                "session_capabilities must be AgentGatewaySessionCapabilities."
            )
        self.store = store
        self.session_capabilities = session_capabilities
        self.max_chunk_bytes = positive_integer(
            max_chunk_bytes,
            field_name="max_chunk_bytes",
        )
        if self.max_chunk_bytes > store.max_chunk_bytes:
            raise ValueError("Broker chunk size cannot exceed the store chunk size.")
        if session_capabilities.artifact_store_quota_bytes != store.quota_bytes:
            raise ValueError("Session capability quota does not match the store.")
        if session_capabilities.max_artifact_bytes != store.max_artifact_bytes:
            raise ValueError(
                "Session max_artifact_bytes does not match the store."
            )
        self._lock = threading.RLock()
        self._grants: dict[str, _GrantState] = {}

    def _mechanism(self, value: str) -> str:
        if value not in self.session_capabilities.artifact_access_mechanisms:
            raise ValueError(
                f"Artifact access mechanism is not advertised: {value!r}."
            )
        return value

    def _register_grant(
        self,
        *,
        grant: AgentArtifactAccessGrant,
        upload_id: str | None,
        now: float,
    ) -> None:
        """Atomically expire stale grants and reserve one advertised slot."""

        timestamp = finite_number(now, field_name="now")
        maximum = (
            self.session_capabilities.max_concurrent_uploads
            if grant.direction == "write"
            else self.session_capabilities.max_concurrent_downloads
        )
        with self._lock:
            stale = tuple(
                grant_id
                for grant_id, state in self._grants.items()
                if state.completed or timestamp >= state.grant.expires_at
            )
            for grant_id in stale:
                self._grants.pop(grant_id, None)
            active = sum(
                state.grant.direction == grant.direction
                for state in self._grants.values()
            )
            if active >= maximum:
                raise ArtifactTransferCapacityError(
                    f"Concurrent Artifact {grant.direction} grant limit is exhausted."
                )
            self._grants[grant.grant_id] = _GrantState(
                grant=grant,
                upload_id=upload_id,
            )

    @staticmethod
    def _ttl(*, now: float, ttl_sec: float) -> tuple[float, float]:
        issued = finite_number(now, field_name="now")
        ttl = finite_number(ttl_sec, field_name="ttl_sec")
        if ttl <= 0 or ttl > 3600.0:
            raise ValueError("ttl_sec must be in (0, 3600].")
        return issued, issued + ttl

    def issue_read_grant(
        self,
        *,
        client_id: str,
        runtime_session_id: str,
        artifact_id: str,
        mechanism: str,
        now: float,
        ttl_sec: float = 60.0,
        byte_offset: int = 0,
        max_bytes: int | None = None,
    ) -> AgentArtifactAccessGrant:
        client = identifier(client_id, field_name="client_id")
        session = identifier(runtime_session_id, field_name="runtime_session_id")
        if session != self.session_capabilities.runtime_session_id:
            raise PermissionError("Artifact grant session does not match capabilities.")
        selected = self._mechanism(mechanism)
        if selected not in {"chunked_rpc", "local_path"}:
            raise NotImplementedError(
                f"Artifact mechanism is negotiated but not implemented: {selected}."
            )
        record = self.store.artifact(artifact_id)
        if not record.available:
            raise ValueError("Artifact is not available for transfer.")
        path = self.store.managed_read_path(record.artifact.artifact_id)
        issued, expires = self._ttl(now=now, ttl_sec=ttl_sec)
        remaining = record.artifact.byte_length - nonnegative_integer(
            byte_offset,
            field_name="byte_offset",
        )
        if remaining < 0:
            raise ValueError("byte_offset exceeds Artifact byte_length.")
        bounded = (
            remaining
            if max_bytes is None
            else nonnegative_integer(max_bytes, field_name="max_bytes")
        )
        if bounded > remaining:
            raise ValueError("max_bytes exceeds the remaining Artifact bytes.")
        if byte_offset != 0 or bounded != record.artifact.byte_length:
            raise ValueError(
                "v2 read grants cover the complete Artifact; resume uses chunk offsets."
            )
        grant_id = f"grant:{uuid.uuid4().hex}"
        chunk_uses = math.ceil(bounded / self.max_chunk_bytes) + 1
        locator: dict[str, Any]
        max_uses: int
        if selected == "local_path":
            locator = {
                "absolute_path": str(path),
                "mount_namespace": "gateway_host",
                "read_only": True,
            }
            max_uses = 1
        else:
            locator = {"transfer_id": grant_id}
            max_uses = chunk_uses
        grant = AgentArtifactAccessGrant(
            grant_id=grant_id,
            artifact=record.artifact,
            direction="read",
            mechanism=selected,
            client_id=client,
            runtime_session_id=session,
            issued_at=issued,
            expires_at=expires,
            max_uses=max_uses,
            locator=locator,
            byte_offset=byte_offset,
            max_bytes=bounded,
        )
        self._register_grant(grant=grant, upload_id=None, now=issued)
        return grant

    def issue_write_grant(
        self,
        *,
        lease: ArtifactUploadLease,
        mechanism: str,
        now: float,
        ttl_sec: float = 60.0,
    ) -> AgentArtifactAccessGrant:
        if not isinstance(lease, ArtifactUploadLease):
            raise TypeError("lease must be an ArtifactUploadLease.")
        if lease.committed or lease.state != "staging":
            raise ValueError("Only a staging upload may receive a write grant.")
        if lease.runtime_session_id != self.session_capabilities.runtime_session_id:
            raise PermissionError("Upload session does not match capabilities.")
        selected = self._mechanism(mechanism)
        if selected not in {"chunked_rpc", "local_path"}:
            raise NotImplementedError(
                f"Artifact mechanism is negotiated but not implemented: {selected}."
            )
        issued, requested_expiry = self._ttl(now=now, ttl_sec=ttl_sec)
        expires = min(requested_expiry, lease.expires_at)
        if expires <= issued:
            raise ValueError("Upload lease expires before the access grant.")
        grant_id = f"grant:{uuid.uuid4().hex}"
        remaining = lease.artifact.byte_length - lease.received_bytes
        max_uses = math.ceil(remaining / self.max_chunk_bytes) + 1
        if selected == "local_path":
            path = self.store.managed_upload_path(
                upload_id=lease.upload_id,
                client_id=lease.client_id,
                runtime_session_id=lease.runtime_session_id,
            )
            locator: dict[str, Any] = {
                "absolute_path": str(path),
                "mount_namespace": "gateway_host",
                "server_owned_staging": True,
            }
            max_uses = 1
        else:
            locator = {"transfer_id": grant_id, "upload_id": lease.upload_id}
        grant = AgentArtifactAccessGrant(
            grant_id=grant_id,
            artifact=lease.artifact,
            direction="write",
            mechanism=selected,
            client_id=lease.client_id,
            runtime_session_id=lease.runtime_session_id,
            issued_at=issued,
            expires_at=expires,
            max_uses=max_uses,
            locator=locator,
            byte_offset=lease.received_bytes,
            max_bytes=remaining,
        )
        self._register_grant(
            grant=grant,
            upload_id=lease.upload_id,
            now=issued,
        )
        return grant

    def _authorize(
        self,
        *,
        grant_id: str,
        client_id: str,
        runtime_session_id: str,
        direction: str,
        now: float,
        consume_use: bool,
    ) -> _GrantState:
        normalized_grant = identifier(grant_id, field_name="grant_id")
        client = identifier(client_id, field_name="client_id")
        session = identifier(runtime_session_id, field_name="runtime_session_id")
        timestamp = finite_number(now, field_name="now")
        with self._lock:
            state = self._grants.get(normalized_grant)
            if state is None:
                raise KeyError(f"Unknown Artifact access grant: {normalized_grant!r}.")
            grant = state.grant
            if state.completed:
                raise PermissionError("Artifact access grant is already completed.")
            if grant.client_id != client or grant.runtime_session_id != session:
                raise PermissionError("Artifact access grant belongs to another client.")
            if grant.direction != direction:
                raise PermissionError("Artifact access grant direction mismatch.")
            if timestamp >= grant.expires_at:
                raise PermissionError("Artifact access grant has expired.")
            if consume_use:
                if state.use_count >= grant.max_uses:
                    raise PermissionError("Artifact access grant use limit is exhausted.")
                state.use_count += 1
            return state

    def read_chunk(
        self,
        *,
        grant_id: str,
        client_id: str,
        runtime_session_id: str,
        offset: int,
        max_bytes: int,
        now: float,
    ) -> bytes:
        state = self._authorize(
            grant_id=grant_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            direction="read",
            now=now,
            consume_use=True,
        )
        grant = state.grant
        if grant.mechanism != "chunked_rpc":
            raise ValueError("read_chunk requires a chunked_rpc grant.")
        requested_offset = nonnegative_integer(offset, field_name="offset")
        requested_bytes = positive_integer(max_bytes, field_name="max_bytes")
        if requested_bytes > self.max_chunk_bytes:
            raise ValueError("max_bytes exceeds the broker chunk bound.")
        grant_end = grant.byte_offset + int(grant.max_bytes)
        if not grant.byte_offset <= requested_offset < grant_end:
            raise ValueError("offset lies outside the access grant.")
        size = min(requested_bytes, grant_end - requested_offset)
        path = self.store.managed_read_path(grant.artifact.artifact_id)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            return os.pread(descriptor, size, requested_offset)
        finally:
            os.close(descriptor)

    def write_chunk(
        self,
        *,
        grant_id: str,
        client_id: str,
        runtime_session_id: str,
        offset: int,
        data: bytes | bytearray | memoryview,
        now: float,
    ) -> ArtifactUploadLease:
        state = self._authorize(
            grant_id=grant_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            direction="write",
            now=now,
            consume_use=True,
        )
        if state.grant.mechanism != "chunked_rpc" or state.upload_id is None:
            raise ValueError("write_chunk requires a chunked_rpc upload grant.")
        return self.store.write_upload_chunk(
            upload_id=state.upload_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            offset=offset,
            data=data,
            now=now,
        )

    def commit_write(
        self,
        *,
        grant_id: str,
        client_id: str,
        runtime_session_id: str,
        now: float,
    ) -> tuple[AgentArtifactRef, AgentArtifactTransferReceipt]:
        state = self._authorize(
            grant_id=grant_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            direction="write",
            now=now,
            consume_use=True,
        )
        if state.upload_id is None:
            raise ValueError("Write grant is not bound to an upload.")
        artifact = self.store.commit_upload(
            upload_id=state.upload_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            now=now,
        )
        receipt = AgentArtifactTransferReceipt(
            receipt_id=f"receipt:{uuid.uuid4().hex}",
            grant_id=state.grant.grant_id,
            artifact_id=artifact.artifact_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            direction="write",
            status="completed",
            transferred_bytes=artifact.byte_length,
            verified_sha256=artifact.sha256,
            completed_at=now,
            reason_code="upload_committed_and_verified",
        )
        with self._lock:
            state.completed = True
        return artifact, receipt

    def acknowledge_read(
        self,
        *,
        grant_id: str,
        client_id: str,
        runtime_session_id: str,
        transferred_bytes: int,
        verified_sha256: str,
        now: float,
    ) -> AgentArtifactTransferReceipt:
        state = self._authorize(
            grant_id=grant_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            direction="read",
            now=now,
            consume_use=False,
        )
        grant = state.grant
        count = nonnegative_integer(
            transferred_bytes,
            field_name="transferred_bytes",
        )
        digest = sha256_digest(verified_sha256, field_name="verified_sha256")
        if count != grant.max_bytes or digest != grant.artifact.sha256:
            raise ValueError("Read acknowledgement does not match the grant bytes.")
        if grant.mechanism == "chunked_rpc" and grant.max_bytes > 0 and (
            state.use_count == 0
        ):
            raise ValueError("Read acknowledgement requires at least one data read.")
        receipt = AgentArtifactTransferReceipt(
            receipt_id=f"receipt:{uuid.uuid4().hex}",
            grant_id=grant.grant_id,
            artifact_id=grant.artifact.artifact_id,
            client_id=client_id,
            runtime_session_id=runtime_session_id,
            direction="read",
            status="completed",
            transferred_bytes=count,
            verified_sha256=digest,
            completed_at=now,
            reason_code="download_acknowledged_and_verified",
        )
        with self._lock:
            state.completed = True
        return receipt

    def expire_grants(self, *, now: float) -> tuple[str, ...]:
        timestamp = finite_number(now, field_name="now")
        with self._lock:
            expired = tuple(
                sorted(
                    grant_id
                    for grant_id, state in self._grants.items()
                    if state.completed or timestamp >= state.grant.expires_at
                )
            )
            for grant_id in expired:
                self._grants.pop(grant_id, None)
            return expired


__all__ = [
    "AgentArtifactTransferBroker",
    "ArtifactTransferCapacityError",
]
