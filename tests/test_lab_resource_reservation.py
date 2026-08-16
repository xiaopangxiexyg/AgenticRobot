from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from lab_control import project_cli
from lab_control.gpu_lease import (
    GpuLeaseStore,
    collect_nvidia_compute_process_inventory,
    collect_nvidia_gpu_inventory,
)
from lab_control.project_cli import main as project_main
from lab_control.remote_preflight import load_host_profile
from lab_control.remote_request import build_remote_request, load_remote_profile
from lab_control.resource_reservation import validate_gpu_resource_reservation
from lab_control.runner_state import RequestStateStore, validate_request_state_identity
from lab_control.workspace import LabError

CONTROL_VERSION = "0.3.0-m3-bootstrap"
HOST_ID = "remote-gpu-hardware"
HOST_PROFILE_SHA = "a" * 64
GPU_UUID = "GPU-resource-reservation-test-0001"


def _inventory(tmp_path: Path) -> dict[str, object]:
    probe = tmp_path / "nvidia-smi"
    probe.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = '--query-gpu=index,uuid,name' || exit 7\n"
        f"printf '0, {GPU_UUID}, Test GPU\\n'\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)
    return collect_nvidia_gpu_inventory(nvidia_smi=probe)


def _process_inventory(
    tmp_path: Path,
    *,
    process_line: str = "",
) -> dict[str, object]:
    probe = tmp_path / "nvidia-smi-processes"
    probe.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = "
        "'--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' "
        "|| exit 7\n"
        f"printf '%b' '{process_line}'\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)
    return collect_nvidia_compute_process_inventory(nvidia_smi=probe)


def _request(*, task_id: str, snapshot_id: str) -> tuple[dict[str, object], object]:
    repository = Path(__file__).resolve().parents[1]
    profile = load_remote_profile(
        repository / "integration" / "remote_profiles",
        "agenticrobot-openpi-gpu",
    )
    request = build_remote_request(
        task_id=task_id,
        snapshot_id=snapshot_id,
        repositories={
            "AgenticRobot": {
                "commit": "1" * 40,
                "tree_oid": "2" * 40,
                "dirty": False,
                "submodules": {},
            },
            "openpi": {
                "commit": "3" * 40,
                "tree_oid": "4" * 40,
                "dirty": False,
                "submodules": {
                    "third_party/aloha": "5" * 40,
                    "third_party/libero": "6" * 40,
                },
            },
        },
        assets={
            "openpi-checkpoint": {
                "asset_id": "checkpoint-v1",
                "sha256": "a" * 64,
            },
            "openpi-config": {"asset_id": "config-v1", "sha256": "b" * 64},
            "openpi-norm-stats": {
                "asset_id": "norm-stats-v1",
                "sha256": "c" * 64,
            },
        },
        asset_lock={
            "lock_id": "resource-assets-v1",
            "workload_id": "resource-workload-v1",
            "sha256": "d" * 64,
        },
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )
    return request, profile


def test_slot_and_gpu_are_reserved_in_one_persistent_request_transition(
    tmp_path: Path,
) -> None:
    request, profile = _request(task_id="AR-0601", snapshot_id="snapshot-0601")
    state_store = RequestStateStore(tmp_path / "state")
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )
    inventory = _inventory(tmp_path)
    state_store.submit(request)

    reserved = state_store.reserve_gpu(
        request,
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
        remote_profile_sha256=profile.sha256,
        inventory_receipt=inventory,
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )
    replayed = state_store.reserve_gpu(
        request,
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
        remote_profile_sha256=profile.sha256,
        inventory_receipt=inventory,
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )

    assert replayed == reserved
    assert reserved["state"] == "RESERVED"
    receipt = reserved["resource_reservation_receipt"]
    assert isinstance(receipt, dict)
    assert receipt["execution_eligible"] is False
    assert receipt["device_access"] is False
    assert validate_gpu_resource_reservation(
        receipt,
        request=request,
        slot_id="gpu-slot-01",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
        remote_profile_sha256=profile.sha256,
    ) == receipt
    assert validate_request_state_identity(reserved, request=request) == reserved


def test_cancel_completion_releases_gpu_and_binds_exact_receipts(
    tmp_path: Path,
) -> None:
    request, profile = _request(task_id="AR-0601-C", snapshot_id="snapshot-0601-c")
    state_store = RequestStateStore(tmp_path / "state")
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )
    state_store.submit(request)
    state_store.reserve_gpu(
        request,
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
        remote_profile_sha256=profile.sha256,
        inventory_receipt=_inventory(tmp_path),
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )
    state_store.request_cancel(str(request["request_id"]))

    cancelled = state_store.complete_cancel(
        request,
        gpu_uuid=GPU_UUID,
        gpu_process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )

    reservation = cancelled["resource_reservation_receipt"]
    release = cancelled["resource_release_receipt"]
    teardown = cancelled["cancel_teardown_receipt"]
    assert isinstance(reservation, dict)
    assert isinstance(release, dict)
    assert isinstance(teardown, dict)
    assert cancelled["state"] == "CANCELLED"
    assert teardown["resource_reservation_receipt_sha256"] == reservation[
        "receipt_sha256"
    ]
    assert teardown["gpu_release_receipt_sha256"] == release["receipt_sha256"]
    assert teardown["gpu_lease_acquired"] is True
    assert teardown["service_started"] is False
    assert teardown["managed_service_teardown_receipt"] is None
    assert gpu_store.active_lease(gpu_uuid=GPU_UUID) is None
    assert state_store.complete_cancel(
        request,
        gpu_uuid=GPU_UUID,
        gpu_process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    ) == cancelled
    assert validate_request_state_identity(cancelled, request=request) == cancelled

    forged = copy.deepcopy(cancelled)
    forged_teardown = forged["cancel_teardown_receipt"]
    assert isinstance(forged_teardown, dict)
    forged_teardown["gpu_release_receipt_sha256"] = "e" * 64
    content = {
        key: value
        for key, value in forged_teardown.items()
        if key != "receipt_sha256"
    }
    forged_teardown["receipt_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(LabError, match="resource identity mismatch"):
        validate_request_state_identity(forged, request=request)


def test_resource_reservation_rejects_contention_and_tampering(
    tmp_path: Path,
) -> None:
    first, profile = _request(task_id="AR-0602", snapshot_id="snapshot-0602")
    second, _ = _request(task_id="AR-0603", snapshot_id="snapshot-0603")
    state_store = RequestStateStore(tmp_path / "state")
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )
    inventory = _inventory(tmp_path)
    for request in (first, second):
        state_store.submit(request)
    reserved = state_store.reserve_gpu(
        first,
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
        remote_profile_sha256=profile.sha256,
        inventory_receipt=inventory,
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )

    with pytest.raises(LabError, match="already reserved"):
        state_store.reserve_gpu(
            second,
            slot_id="gpu-slot-01",
            allowed_slots=("gpu-slot-01",),
            host_id=HOST_ID,
            host_profile_sha256=HOST_PROFILE_SHA,
            remote_profile_sha256=profile.sha256,
            inventory_receipt=inventory,
            process_inventory_receipt=_process_inventory(tmp_path),
            gpu_store=gpu_store,
        )
    receipt = reserved["resource_reservation_receipt"]
    assert isinstance(receipt, dict)
    forged = copy.deepcopy(receipt)
    forged["slot_id"] = "gpu-slot-02"
    with pytest.raises(LabError, match="identity mismatch|hash mismatch"):
        validate_gpu_resource_reservation(
            forged,
            request=first,
            slot_id="gpu-slot-01",
        )


def test_gpu_reservation_requires_an_accepted_request_state(tmp_path: Path) -> None:
    request, profile = _request(task_id="AR-0604", snapshot_id="snapshot-0604")
    state_store = RequestStateStore(tmp_path / "state")
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )

    with pytest.raises(LabError, match="state does not exist"):
        state_store.reserve_gpu(
            request,
            slot_id="gpu-slot-01",
            allowed_slots=("gpu-slot-01",),
            host_id=HOST_ID,
            host_profile_sha256=HOST_PROFILE_SHA,
            remote_profile_sha256=profile.sha256,
            inventory_receipt=_inventory(tmp_path),
            process_inventory_receipt=_process_inventory(tmp_path),
            gpu_store=gpu_store,
        )
    assert list((tmp_path / "gpu-leases").glob("*.json")) == []


def test_existing_compute_process_blocks_before_lease_is_written(tmp_path: Path) -> None:
    request, profile = _request(task_id="AR-0606", snapshot_id="snapshot-0606")
    state_store = RequestStateStore(tmp_path / "state")
    state_store.submit(request)
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )
    process_line = f"{GPU_UUID}, 4321, python, 4096\\n"

    with pytest.raises(LabError, match="existing compute processes"):
        state_store.reserve_gpu(
            request,
            slot_id="gpu-slot-01",
            allowed_slots=("gpu-slot-01",),
            host_id=HOST_ID,
            host_profile_sha256=HOST_PROFILE_SHA,
            remote_profile_sha256=profile.sha256,
            inventory_receipt=_inventory(tmp_path),
            process_inventory_receipt=_process_inventory(
                tmp_path,
                process_line=process_line,
            ),
            gpu_store=gpu_store,
        )

    assert state_store.load(str(request["request_id"]))["state"] == "SUBMITTED"
    assert list((tmp_path / "gpu-leases").glob("*.json")) == []


def test_failed_preflight_lease_requires_explicit_nonstarted_reconciliation(
    tmp_path: Path,
) -> None:
    request, profile = _request(task_id="AR-0607", snapshot_id="snapshot-0607")
    state_store = RequestStateStore(tmp_path / "state")
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )
    state_store.submit(request)
    reserved = state_store.reserve_gpu(
        request,
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
        remote_profile_sha256=profile.sha256,
        inventory_receipt=_inventory(tmp_path),
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )
    state_store.fail(
        str(request["request_id"]),
        classification="BLOCKED_PRECONDITION",
        detail="preflight fixture failed",
    )
    reservation = reserved["resource_reservation_receipt"]
    assert isinstance(reservation, dict)
    lease = reservation["gpu_lease_receipt"]
    assert isinstance(lease, dict)
    assert gpu_store.active_lease(gpu_uuid=GPU_UUID) == lease

    reconciled = state_store.reconcile_nonstarted_gpu(
        request,
        gpu_uuid=GPU_UUID,
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )

    assert reconciled["state"] == "FAILED"
    assert reconciled["resource_release_receipt"]["status"] == "RELEASED"
    assert gpu_store.active_lease(gpu_uuid=GPU_UUID) is None
    assert validate_request_state_identity(reconciled, request=request) == reconciled


def test_orphan_lease_is_reconciled_only_when_request_never_started(
    tmp_path: Path,
) -> None:
    request, _ = _request(task_id="AR-0608", snapshot_id="snapshot-0608")
    state_store = RequestStateStore(tmp_path / "state")
    state_store.submit(request)
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )
    gpu_store.acquire(
        gpu_id="gpu-0",
        gpu_uuid=GPU_UUID,
        request_id=str(request["request_id"]),
        slot_id="gpu-slot-01",
    )

    reconciled = state_store.reconcile_nonstarted_gpu(
        request,
        gpu_uuid=GPU_UUID,
        process_inventory_receipt=_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )

    assert reconciled["state"] == "FAILED"
    assert reconciled["failure"]["classification"] == "INFRASTRUCTURE_FAILURE"
    assert reconciled["resource_reservation_receipt"] is None
    assert reconciled["resource_release_receipt"]["status"] == "RELEASED"
    assert gpu_store.active_lease(gpu_uuid=GPU_UUID) is None


def test_projectctl_gpu_reserve_uses_accepted_state_and_no_payload(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    request, profile = _request(task_id="AR-0605", snapshot_id="snapshot-0605")
    repository = Path(__file__).resolve().parents[1]
    configured_host = load_host_profile(
        repository / "integration" / "host_profiles",
        "remote-gpu-hardware",
    )
    host = replace(configured_host, slot_root=(tmp_path / "slots").resolve())
    state_root = tmp_path / "state"
    RequestStateStore(state_root).submit(request)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    inventory = _inventory(tmp_path)
    monkeypatch.setattr(project_cli, "load_host_profile", lambda *_args: host)
    monkeypatch.setattr(
        project_cli,
        "collect_nvidia_gpu_inventory",
        lambda: inventory,
    )
    monkeypatch.setattr(
        project_cli,
        "collect_nvidia_compute_process_inventory",
        lambda: _process_inventory(tmp_path),
    )

    exit_code = project_main(
        [
            "gpu-lease",
            "reserve",
            "--request",
            str(request_path),
            "--profile",
            profile.profile_id,
            "--host-profile",
            host.host_id,
            "--slot",
            "gpu-slot-01",
            "--state-root",
            str(state_root),
        ]
    )

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "lab.gpu-reservation.v2"
    assert result["service_started"] is False
    assert result["execution_eligible"] is False
    assert result["device_access"] is False
    state = result["request_state"]
    assert state["state"] == "RESERVED"
    assert state["resource_reservation_receipt"]["request_id"] == request["request_id"]

    RequestStateStore(state_root).fail(
        str(request["request_id"]),
        classification="BLOCKED_PRECONDITION",
        detail="test preflight did not start a service",
    )
    exit_code = project_main(
        [
            "gpu-lease",
            "reconcile-nonstarted",
            "--request",
            str(request_path),
            "--state-root",
            str(state_root),
            "--host-profile",
            host.host_id,
        ]
    )

    assert exit_code == 0
    reconciled = json.loads(capsys.readouterr().out)
    assert reconciled["schema"] == "lab.gpu-reconciliation.v1"
    assert reconciled["resource_release"]["status"] == "RELEASED"
    assert reconciled["request_state"]["state"] == "FAILED"
