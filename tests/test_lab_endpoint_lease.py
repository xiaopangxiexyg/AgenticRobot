from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from lab_control.endpoint_lease import (
    EndpointLeaseStore,
    validate_endpoint_lease_receipt,
    validate_endpoint_release_receipt,
)
from lab_control.workspace import LabError

HOST_HASH = "a" * 64
NOW = datetime(2026, 8, 14, 8, 0, tzinfo=timezone.utc)


def _store(tmp_path):
    return EndpointLeaseStore(
        (tmp_path / "endpoint-leases").resolve(),
        host_id="execution-node",
        host_profile_sha256=HOST_HASH,
    )


def test_endpoint_lease_is_persistent_idempotent_and_exclusive(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.acquire(
        request_id="Q-request-one",
        slot_id="slot-01",
        scheme="http",
        port=18101,
        now=NOW,
    )
    replay = store.acquire(
        request_id="Q-request-one",
        slot_id="slot-01",
        scheme="http",
        port=18101,
        now=NOW + timedelta(seconds=1),
    )

    assert replay == first
    assert first["namespace_reserved"] is True
    assert first["socket_bound"] is False
    assert first["process_started"] is False
    assert store.active_lease(port=18101, now=NOW) == first
    with pytest.raises(LabError, match="leased by request Q-request-one"):
        store.acquire(
            request_id="Q-request-two",
            slot_id="slot-02",
            scheme="http",
            port=18101,
            now=NOW + timedelta(seconds=2),
        )


def test_endpoint_lease_rejects_non_loopback_low_port_and_tampering(tmp_path) -> None:
    store = _store(tmp_path)
    with pytest.raises(LabError, match="port is invalid"):
        store.acquire(
            request_id="Q-request",
            slot_id="slot-01",
            scheme="http",
            port=80,
            now=NOW,
        )
    lease = store.acquire(
        request_id="Q-request",
        slot_id="slot-01",
        scheme="wss",
        port=18443,
        now=NOW,
    )
    tampered = copy.deepcopy(lease)
    tampered["host"] = "0.0.0.0"
    with pytest.raises(LabError, match="loopback"):
        validate_endpoint_lease_receipt(tampered, now=NOW)


def test_nonstarted_release_is_idempotent_and_allows_reuse(tmp_path) -> None:
    store = _store(tmp_path)
    lease = store.acquire(
        request_id="Q-request-one",
        slot_id="slot-01",
        scheme="ws",
        port=18142,
        now=NOW,
    )
    release = store.release_nonstarted(
        lease,
        now=NOW + timedelta(seconds=5),
    )
    replay = store.release_nonstarted(
        lease,
        now=NOW + timedelta(seconds=6),
    )

    assert replay == release
    assert validate_endpoint_release_receipt(release, lease_receipt=lease) == release
    assert store.active_lease(port=18142, now=NOW + timedelta(seconds=6)) is None
    replacement = store.acquire(
        request_id="Q-request-two",
        slot_id="slot-02",
        scheme="ws",
        port=18142,
        now=NOW + timedelta(seconds=7),
    )
    assert replacement["request_id"] == "Q-request-two"


def test_expired_lease_requires_explicit_nonstarted_reconciliation(tmp_path) -> None:
    store = _store(tmp_path)
    lease = store.acquire(
        request_id="Q-request-one",
        slot_id="slot-01",
        scheme="http",
        port=18199,
        valid_for_seconds=30,
        now=NOW,
    )
    expired = NOW + timedelta(seconds=31)

    with pytest.raises(LabError, match="explicit nonstarted reconciliation"):
        store.acquire(
            request_id="Q-request-two",
            slot_id="slot-02",
            scheme="http",
            port=18199,
            now=expired,
        )
    with pytest.raises(LabError, match="release reason"):
        store.release_nonstarted(lease, now=expired)
    release = store.release_nonstarted(
        lease,
        expired_reconciliation=True,
        now=expired,
    )
    assert release["release_reason"] == "expired-nonstarted-reconciled"


def test_endpoint_lease_store_refuses_symlink_state_root(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(LabError, match="state root is unsafe"):
        EndpointLeaseStore(
            linked,
            host_id="execution-node",
            host_profile_sha256=HOST_HASH,
        )


def test_endpoint_lease_store_rejects_tampered_release_history(tmp_path) -> None:
    store = _store(tmp_path)
    lease = store.acquire(
        request_id="Q-request",
        slot_id="slot-01",
        scheme="http",
        port=18222,
        now=NOW,
    )
    store.release_nonstarted(lease, now=NOW + timedelta(seconds=1))
    state_path = next((store.root / "states").glob("*.json"))
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["last_release"]["released_at"] = (
        NOW + timedelta(seconds=2)
    ).isoformat()
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="receipt hash mismatch"):
        store.active_lease(port=18222, now=NOW + timedelta(seconds=2))
