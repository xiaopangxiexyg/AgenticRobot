"""Allocate short, private project-local Unix endpoints for Agent Gateway.

Long-lived logs, history, Artifact data, and Experience records should use
their normal descriptive paths.  AF_UNIX pathnames are a separate, bounded
transport namespace, so callers allocate one short endpoint and explicitly
hand the returned path to both the lower Gateway and its upper client.
"""

from __future__ import annotations

import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .unix_transport import (
    MAX_AGENT_GATEWAY_UNIX_SOCKET_PATH_BYTES,
    validate_agent_gateway_unix_server_preflight,
)


@dataclass(frozen=True)
class AgentGatewayUnixEndpointAllocation:
    """One caller-owned directory reservation for a future Gateway socket."""

    allocation_id: str
    endpoint_dir: Path
    socket_path: Path
    encoded_path_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": "agent_gateway_unix_endpoint_allocated",
            "allocation_id": self.allocation_id,
            "endpoint_dir": str(self.endpoint_dir),
            "socket_path": str(self.socket_path),
            "encoded_path_bytes": self.encoded_path_bytes,
            "max_path_bytes": MAX_AGENT_GATEWAY_UNIX_SOCKET_PATH_BYTES,
        }


def allocate_agent_gateway_unix_endpoint(
    *,
    endpoint_root: str | os.PathLike[str],
) -> AgentGatewayUnixEndpointAllocation:
    """Create a private unique directory and return its absent socket path.

    The allocation reserves the parent directory, not the socket inode.  The
    server remains the sole owner of binding and unlinking that inode and
    repeats its normal validation immediately before bind.
    """

    root = Path(os.path.abspath(os.fspath(endpoint_root)))
    if not root.is_absolute():  # pragma: no cover - abspath is defensive.
        raise ValueError("endpoint_root must resolve to an absolute path.")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError("endpoint_root must be a real directory, not a symlink.")
    if root_stat.st_mode & stat.S_IWOTH:
        raise ValueError("endpoint_root cannot be world-writable.")

    endpoint_dir = Path(tempfile.mkdtemp(prefix="g-", dir=root))
    os.chmod(endpoint_dir, 0o700)
    socket_path = endpoint_dir / "g.sock"
    try:
        validated_path, _ = validate_agent_gateway_unix_server_preflight(
            socket_path=socket_path,
        )
    except Exception:
        # This directory was created by this call and is still empty because
        # validation precedes publication.  Never remove caller-owned roots.
        os.rmdir(endpoint_dir)
        raise

    return AgentGatewayUnixEndpointAllocation(
        allocation_id=endpoint_dir.name,
        endpoint_dir=endpoint_dir,
        socket_path=Path(validated_path),
        encoded_path_bytes=len(os.fsencode(validated_path)),
    )
