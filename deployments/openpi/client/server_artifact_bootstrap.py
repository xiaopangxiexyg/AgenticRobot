"""Bootstrap a deployment artifact from the selected OpenPI server.

The server owns model and checkpoint selection.  A real deployment client only
needs the resulting tensor, camera, normalization, and continuation contracts;
it must not repeat the server's model identity in its command line.
"""

from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol

from dataset_contracts.policy_artifact import (
    POLICY_ARTIFACT_METADATA_KEY,
    PolicyArtifactContract,
    policy_artifact_from_metadata,
)

from .transport import OpenPIWebsocketTransport


class _MetadataTransport(Protocol):
    @property
    def metadata(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


TransportFactory = Callable[..., _MetadataTransport]


def fetch_server_policy_artifact(
    *,
    host: str,
    port: int,
    api_key: str | None = None,
    retry_interval_s: float = 2.0,
    transport_factory: TransportFactory = OpenPIWebsocketTransport,
) -> PolicyArtifactContract:
    """Connect to one server, validate its advertised artifact, then close.

    This negotiation performs network I/O only.  It deliberately happens
    before cameras, robot backends, or the real deployment runtime are built.
    """

    transport = transport_factory(
        host=str(host),
        port=int(port),
        api_key=api_key,
        retry_interval_s=float(retry_interval_s),
    )
    try:
        metadata = transport.metadata
        if not isinstance(metadata, Mapping):
            raise ValueError("OpenPI server metadata must be a mapping.")
        if POLICY_ARTIFACT_METADATA_KEY not in metadata:
            raise ValueError(
                "OpenPI server metadata does not publish policy_artifact; "
                "start a prepared deployment server that advertises its contract."
            )
        return policy_artifact_from_metadata(metadata)
    finally:
        transport.close()


@contextmanager
def materialize_runtime_policy_artifact(
    artifact: PolicyArtifactContract,
) -> Iterator[Path]:
    """Bridge a server artifact into the existing real-runtime file boundary.

    The private temporary sidecar lives only for the delegated runtime call. It
    is not a client checkpoint selection or a durable model binding.
    """

    if not isinstance(artifact, PolicyArtifactContract):
        raise TypeError("artifact must be a PolicyArtifactContract.")
    with tempfile.TemporaryDirectory(prefix="openpi-server-artifact-") as root:
        path = Path(root) / "policy_artifact.json"
        path.write_text(
            json.dumps(artifact.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        yield path


__all__ = [
    "fetch_server_policy_artifact",
    "materialize_runtime_policy_artifact",
]
