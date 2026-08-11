"""Backend-neutral incremental integrity for persisted artifact bytes.

The helper deliberately owns no dataset schema or filesystem lifecycle.  A
concrete persistence adapter feeds it the exact byte strings passed to its
file handle and publishes the sealed result only after the file is durable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path


STORED_BYTES_SHA256_PROFILE_V1 = "stored_bytes_sha256_v1"


@dataclass(frozen=True)
class SealedArtifactDigest:
    size_bytes: int
    sha256: str
    profile: str = STORED_BYTES_SHA256_PROFILE_V1


class IncrementalArtifactDigest:
    """Incrementally hash the exact representation written to one artifact."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._digest = hashlib.sha256() if self._enabled else None
        self._size_bytes = 0
        self._sealed = False
        self._sealed_result: SealedArtifactDigest | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def size_bytes(self) -> int:
        return int(self._size_bytes)

    def update(self, payload: bytes | bytearray | memoryview) -> None:
        if self._sealed:
            raise RuntimeError("artifact digest is already sealed")
        if not self._enabled:
            return
        view = memoryview(payload)
        assert self._digest is not None
        self._digest.update(view)
        self._size_bytes += int(view.nbytes)

    def seal(self) -> SealedArtifactDigest | None:
        if self._sealed:
            return self._sealed_result
        self._sealed = True
        if not self._enabled:
            return None
        assert self._digest is not None
        self._sealed_result = SealedArtifactDigest(
            size_bytes=int(self._size_bytes),
            sha256=self._digest.hexdigest(),
        )
        return self._sealed_result


def digest_stored_file(
    path: str | Path,
    *,
    block_size: int = 4 * 1024 * 1024,
) -> SealedArtifactDigest:
    """Hash the exact stored bytes of a finalized regular file.

    This is the offline counterpart of :class:`IncrementalArtifactDigest`.
    Explicit transforms use it after their durable replacement so integrity
    metadata can never describe the pre-transform payload.
    """

    target = Path(path)
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"stored artifact must be a regular non-symlink file: {target}")
    digest = hashlib.sha256()
    size_bytes = 0
    with target.open("rb") as handle:
        while True:
            block = handle.read(int(block_size))
            if not block:
                break
            digest.update(block)
            size_bytes += len(block)
    stat_size = int(target.stat().st_size)
    if size_bytes != stat_size:
        raise RuntimeError(
            f"stored artifact changed while hashing: read={size_bytes}, stat={stat_size}"
        )
    return SealedArtifactDigest(size_bytes=size_bytes, sha256=digest.hexdigest())


__all__ = [
    "IncrementalArtifactDigest",
    "SealedArtifactDigest",
    "STORED_BYTES_SHA256_PROFILE_V1",
    "digest_stored_file",
]
