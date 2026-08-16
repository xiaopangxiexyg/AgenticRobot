from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .remote_request import SHA256_PATTERN, _sha256
from .workspace import LabError, _atomic_write_text, validate_task_id

ENDPOINT_LEASE_SCHEMA_ID = "lab.endpoint-lease.v1"
ENDPOINT_RELEASE_SCHEMA_ID = "lab.endpoint-release.v1"
ENDPOINT_STATE_SCHEMA_ID = "lab.endpoint-lease-state.v2"
_ENDPOINT_SCHEMES = frozenset({"http", "https", "ws", "wss"})
_LOOPBACK_HOST = "127.0.0.1"
_MAX_STATE_BYTES = 64 * 1024


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise LabError(f"{label} has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None:
        raise LabError(f"{label} timestamp requires a timezone")
    return parsed


def _validate_endpoint(scheme: object, host: object, port: object) -> tuple[str, int]:
    if scheme not in _ENDPOINT_SCHEMES:
        raise LabError(f"unsupported endpoint lease scheme: {scheme}")
    if host != _LOOPBACK_HOST:
        raise LabError("endpoint lease must remain on IPv4 loopback")
    if (
        not isinstance(port, int)
        or isinstance(port, bool)
        or not 1024 <= port <= 65535
    ):
        raise LabError("endpoint lease port is invalid")
    return str(scheme), port


def validate_endpoint_lease_receipt(
    receipt: object,
    *,
    host_id: str | None = None,
    host_profile_sha256: str | None = None,
    request_id: str | None = None,
    slot_id: str | None = None,
    require_active: bool = True,
    now: datetime | None = None,
) -> dict[str, object]:
    expected = {
        "schema",
        "lease_id",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "scheme",
        "host",
        "port",
        "acquired_at",
        "expires_at",
        "active",
        "namespace_reserved",
        "socket_bound",
        "process_started",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("endpoint lease receipt has an unexpected shape")
    if receipt.get("schema") != ENDPOINT_LEASE_SCHEMA_ID:
        raise LabError("unsupported endpoint lease receipt schema")
    for field in ("lease_id", "request_id", "slot_id", "host_id"):
        value = receipt.get(field)
        if not isinstance(value, str):
            raise LabError(f"endpoint lease receipt has invalid {field}")
        validate_task_id(value)
    if not SHA256_PATTERN.fullmatch(str(receipt.get("host_profile_sha256", ""))):
        raise LabError("endpoint lease receipt has invalid host Profile hash")
    scheme, port = _validate_endpoint(
        receipt.get("scheme"), receipt.get("host"), receipt.get("port")
    )
    bindings = {
        "host_id": host_id,
        "host_profile_sha256": host_profile_sha256,
        "request_id": request_id,
        "slot_id": slot_id,
    }
    for field, expected_value in bindings.items():
        if expected_value is not None and receipt.get(field) != expected_value:
            raise LabError(f"endpoint lease receipt identity mismatch: {field}")
    acquired = _timestamp(receipt.get("acquired_at"), label="endpoint lease")
    expires = _timestamp(receipt.get("expires_at"), label="endpoint lease")
    validity = (expires - acquired).total_seconds()
    if not 30 <= validity <= 3600:
        raise LabError("endpoint lease validity is outside the protected range")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise LabError("endpoint lease validation clock requires a timezone")
    if acquired > current:
        raise LabError("endpoint lease was acquired in the future")
    if require_active and current >= expires:
        raise LabError("endpoint lease has expired")
    if (
        receipt.get("active") is not True
        or receipt.get("namespace_reserved") is not True
        or receipt.get("socket_bound") is not False
        or receipt.get("process_started") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("endpoint lease has an unsafe authority envelope")
    identity_content = {
        "request_id": receipt["request_id"],
        "slot_id": receipt["slot_id"],
        "host_id": receipt["host_id"],
        "host_profile_sha256": receipt["host_profile_sha256"],
        "scheme": scheme,
        "host": _LOOPBACK_HOST,
        "port": port,
        "acquired_at": receipt["acquired_at"],
        "expires_at": receipt["expires_at"],
    }
    identity_hash = _sha256(identity_content)
    if receipt.get("lease_id") != f"EP-{identity_hash[:24]}":
        raise LabError("endpoint lease receipt ID mismatch")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("endpoint lease receipt hash mismatch")
    return receipt


def validate_endpoint_release_receipt(
    receipt: object,
    *,
    lease_receipt: object,
) -> dict[str, object]:
    lease = validate_endpoint_lease_receipt(lease_receipt, require_active=False)
    expected = {
        "schema",
        "lease_id",
        "lease_receipt_sha256",
        "request_id",
        "slot_id",
        "host_id",
        "host_profile_sha256",
        "scheme",
        "host",
        "port",
        "released_at",
        "release_reason",
        "process_absent",
        "socket_unbound",
        "active",
        "device_access",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected:
        raise LabError("endpoint release receipt has an unexpected shape")
    bindings = {
        "schema": ENDPOINT_RELEASE_SCHEMA_ID,
        "lease_id": lease["lease_id"],
        "lease_receipt_sha256": lease["receipt_sha256"],
        "request_id": lease["request_id"],
        "slot_id": lease["slot_id"],
        "host_id": lease["host_id"],
        "host_profile_sha256": lease["host_profile_sha256"],
        "scheme": lease["scheme"],
        "host": lease["host"],
        "port": lease["port"],
    }
    for field, expected_value in bindings.items():
        if receipt.get(field) != expected_value:
            raise LabError(f"endpoint release receipt identity mismatch: {field}")
    released = _timestamp(receipt.get("released_at"), label="endpoint release")
    acquired = _timestamp(lease.get("acquired_at"), label="endpoint lease")
    if released < acquired:
        raise LabError("endpoint release predates its lease")
    if (
        receipt.get("release_reason")
        not in {"nonstarted-plan-discarded", "expired-nonstarted-reconciled"}
        or receipt.get("process_absent") is not True
        or receipt.get("socket_unbound") is not True
        or receipt.get("active") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("endpoint release has an unsafe evidence envelope")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _sha256(content):
        raise LabError("endpoint release receipt hash mismatch")
    return receipt


class EndpointLeaseStore:
    """Persist framework-local loopback endpoint ownership without binding sockets.

    This v1 store prevents two managed Lab requests from selecting the same port.
    It deliberately cannot authorize process launch: a future trusted launcher must
    bind and attest the socket before the plan can become executable.
    """

    def __init__(
        self,
        root: Path,
        *,
        host_id: str,
        host_profile_sha256: str,
    ) -> None:
        validate_task_id(host_id)
        if not SHA256_PATTERN.fullmatch(host_profile_sha256):
            raise LabError("endpoint lease store requires a host Profile hash")
        requested = root.expanduser()
        resolved = requested.resolve()
        if (
            not requested.is_absolute()
            or requested.is_symlink()
            or requested.absolute() != resolved
        ):
            raise LabError(f"endpoint lease state root is unsafe: {root}")
        self.root = resolved
        self.host_id = host_id
        self.host_profile_sha256 = host_profile_sha256
        self.state_root = self.root / "states"
        self.locks_root = self.root / "locks"

    def _ensure_roots(self) -> None:
        if self.root.is_symlink():
            raise LabError(f"endpoint lease state root is a symlink: {self.root}")
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.locks_root.mkdir(parents=True, exist_ok=True)
        for path in (self.root, self.state_root, self.locks_root):
            if path.is_symlink() or path.resolve() != path.absolute():
                raise LabError(f"endpoint lease state path is unsafe: {path}")

    def _paths(self, port: int) -> tuple[Path, Path]:
        _validate_endpoint("http", _LOOPBACK_HOST, port)
        suffix = hashlib.sha256(f"tcp:{port}".encode()).hexdigest()
        return self.state_root / f"{suffix}.json", self.locks_root / f"{suffix}.lock"

    @contextmanager
    def _locked(self, port: int) -> Iterator[Path]:
        self._ensure_roots()
        state_path, lock_path = self._paths(port)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise LabError(f"cannot open endpoint lease lock: {lock_path}") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield state_path
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load(self, state_path: Path, *, port: int) -> dict[str, object]:
        if not state_path.exists():
            return {
                "schema": ENDPOINT_STATE_SCHEMA_ID,
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "port": port,
                "active": None,
                "last_lease": None,
                "last_release": None,
            }
        if state_path.is_symlink():
            raise LabError(f"endpoint lease state is a symlink: {state_path}")
        try:
            if state_path.stat().st_size > _MAX_STATE_BYTES:
                raise LabError("endpoint lease state exceeds its protected size")
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError(f"cannot read endpoint lease state: {state_path}") from exc
        if not isinstance(state, dict) or set(state) != {
            "schema",
            "host_id",
            "host_profile_sha256",
            "port",
            "active",
            "last_lease",
            "last_release",
        }:
            raise LabError("endpoint lease state has an unexpected shape")
        if (
            state.get("schema") != ENDPOINT_STATE_SCHEMA_ID
            or state.get("host_id") != self.host_id
            or state.get("host_profile_sha256") != self.host_profile_sha256
            or state.get("port") != port
        ):
            raise LabError("endpoint lease state identity is invalid")
        active = state.get("active")
        last_lease = state.get("last_lease")
        last_release = state.get("last_release")
        if active is not None:
            validate_endpoint_lease_receipt(
                active,
                host_id=self.host_id,
                host_profile_sha256=self.host_profile_sha256,
                require_active=False,
            )
            if last_lease is not None or last_release is not None:
                raise LabError("active endpoint state cannot retain release history")
        elif (last_lease is None) != (last_release is None):
            raise LabError("endpoint lease release history is incomplete")
        elif last_lease is not None:
            validated_last_lease = validate_endpoint_lease_receipt(
                last_lease,
                host_id=self.host_id,
                host_profile_sha256=self.host_profile_sha256,
                require_active=False,
            )
            if validated_last_lease.get("port") != port:
                raise LabError("endpoint lease release history has the wrong port")
            validate_endpoint_release_receipt(
                last_release,
                lease_receipt=validated_last_lease,
            )
        return state

    def acquire(
        self,
        *,
        request_id: str,
        slot_id: str,
        scheme: str,
        port: int,
        valid_for_seconds: int = 600,
        now: datetime | None = None,
    ) -> dict[str, object]:
        validate_task_id(request_id)
        validate_task_id(slot_id)
        scheme, port = _validate_endpoint(scheme, _LOOPBACK_HOST, port)
        if (
            not isinstance(valid_for_seconds, int)
            or isinstance(valid_for_seconds, bool)
            or not 30 <= valid_for_seconds <= 3600
        ):
            raise LabError("endpoint lease validity must be between 30 and 3600 seconds")
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise LabError("endpoint lease clock requires a timezone")
        with self._locked(port) as state_path:
            state = self._load(state_path, port=port)
            active = state.get("active")
            if isinstance(active, dict):
                expires = _timestamp(active.get("expires_at"), label="endpoint lease")
                same_owner = (
                    active.get("request_id") == request_id
                    and active.get("slot_id") == slot_id
                    and active.get("scheme") == scheme
                )
                if current < expires and same_owner:
                    return validate_endpoint_lease_receipt(
                        active,
                        host_id=self.host_id,
                        host_profile_sha256=self.host_profile_sha256,
                        request_id=request_id,
                        slot_id=slot_id,
                        now=current,
                    )
                if current >= expires:
                    raise LabError(
                        "expired endpoint lease requires explicit nonstarted reconciliation"
                    )
                raise LabError(
                    f"endpoint 127.0.0.1:{port} is leased by request "
                    f"{active.get('request_id')}"
                )
            expires = current + timedelta(seconds=valid_for_seconds)
            identity = {
                "request_id": request_id,
                "slot_id": slot_id,
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "scheme": scheme,
                "host": _LOOPBACK_HOST,
                "port": port,
                "acquired_at": current.isoformat(),
                "expires_at": expires.isoformat(),
            }
            content: dict[str, object] = {
                "schema": ENDPOINT_LEASE_SCHEMA_ID,
                "lease_id": f"EP-{_sha256(identity)[:24]}",
                **identity,
                "active": True,
                "namespace_reserved": True,
                "socket_bound": False,
                "process_started": False,
                "device_access": False,
            }
            lease = {**content, "receipt_sha256": _sha256(content)}
            validate_endpoint_lease_receipt(
                lease,
                host_id=self.host_id,
                host_profile_sha256=self.host_profile_sha256,
                request_id=request_id,
                slot_id=slot_id,
                now=current,
            )
            new_state = {
                "schema": ENDPOINT_STATE_SCHEMA_ID,
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "port": port,
                "active": lease,
                "last_lease": None,
                "last_release": None,
            }
            _atomic_write_text(
                state_path, json.dumps(new_state, indent=2, sort_keys=True) + "\n"
            )
            return lease

    def release_nonstarted(
        self,
        lease_receipt: object,
        *,
        expired_reconciliation: bool = False,
        now: datetime | None = None,
    ) -> dict[str, object]:
        lease = validate_endpoint_lease_receipt(
            lease_receipt,
            host_id=self.host_id,
            host_profile_sha256=self.host_profile_sha256,
            require_active=False,
        )
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise LabError("endpoint release clock requires a timezone")
        expires = _timestamp(lease.get("expires_at"), label="endpoint lease")
        if expired_reconciliation != (current >= expires):
            raise LabError("endpoint release reason does not match lease expiry")
        raw_port = lease["port"]
        assert isinstance(raw_port, int) and not isinstance(raw_port, bool)
        port = raw_port
        with self._locked(port) as state_path:
            state = self._load(state_path, port=port)
            active = state.get("active")
            if not isinstance(active, dict) or active != lease:
                last_release = state.get("last_release")
                if (
                    isinstance(last_release, dict)
                    and last_release.get("lease_receipt_sha256")
                    == lease.get("receipt_sha256")
                ):
                    return validate_endpoint_release_receipt(
                        last_release,
                        lease_receipt=lease,
                    )
                raise LabError("endpoint lease is no longer the active lease")
            content: dict[str, object] = {
                "schema": ENDPOINT_RELEASE_SCHEMA_ID,
                "lease_id": lease["lease_id"],
                "lease_receipt_sha256": lease["receipt_sha256"],
                "request_id": lease["request_id"],
                "slot_id": lease["slot_id"],
                "host_id": lease["host_id"],
                "host_profile_sha256": lease["host_profile_sha256"],
                "scheme": lease["scheme"],
                "host": lease["host"],
                "port": lease["port"],
                "released_at": current.isoformat(),
                "release_reason": (
                    "expired-nonstarted-reconciled"
                    if expired_reconciliation
                    else "nonstarted-plan-discarded"
                ),
                "process_absent": True,
                "socket_unbound": True,
                "active": False,
                "device_access": False,
            }
            release = {**content, "receipt_sha256": _sha256(content)}
            validate_endpoint_release_receipt(release, lease_receipt=lease)
            new_state = {
                "schema": ENDPOINT_STATE_SCHEMA_ID,
                "host_id": self.host_id,
                "host_profile_sha256": self.host_profile_sha256,
                "port": port,
                "active": None,
                "last_lease": lease,
                "last_release": release,
            }
            _atomic_write_text(
                state_path, json.dumps(new_state, indent=2, sort_keys=True) + "\n"
            )
            return release

    def active_lease(
        self,
        *,
        port: int,
        now: datetime | None = None,
    ) -> dict[str, object] | None:
        with self._locked(port) as state_path:
            state = self._load(state_path, port=port)
            active = state.get("active")
            if active is None:
                return None
            return validate_endpoint_lease_receipt(
                active,
                host_id=self.host_id,
                host_profile_sha256=self.host_profile_sha256,
                now=now,
            )
