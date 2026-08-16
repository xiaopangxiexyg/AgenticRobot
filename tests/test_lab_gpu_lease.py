from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control.gpu_lease import (
    GpuLeaseStore,
    build_gpu_cleanup_receipt,
    collect_nvidia_compute_process_inventory,
    collect_nvidia_gpu_inventory,
    require_gpu_compute_processes_absent,
    select_single_gpu,
    validate_gpu_inventory,
    validate_gpu_lease_receipt,
    validate_gpu_process_inventory,
)
from lab_control.workspace import LabError

HOST_ID = "remote-gpu-test"
HOST_PROFILE_SHA = "a" * 64
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"


def _store(tmp_path: Path) -> GpuLeaseStore:
    return GpuLeaseStore(
        (tmp_path / "gpu-leases").resolve(),
        host_id=HOST_ID,
        host_profile_sha256=HOST_PROFILE_SHA,
    )


def _process_inventory(
    tmp_path: Path,
    *,
    line: str = "",
) -> dict[str, object]:
    probe = tmp_path / "nvidia-smi-processes-helper"
    probe.write_text(
        "#!/bin/sh\n" f"printf '%b' '{line}'\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)
    return collect_nvidia_compute_process_inventory(nvidia_smi=probe)


def _acquire(store: GpuLeaseStore, *, request_id: str = "Q-request-01"):
    return store.acquire(
        gpu_id="gpu-0",
        gpu_uuid=GPU_UUID,
        request_id=request_id,
        slot_id="gpu-slot-01",
    )


def test_gpu_lease_is_persistent_exclusive_and_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)

    first = _acquire(store)
    repeated = _acquire(store)

    assert repeated == first
    assert first["execution_eligible"] is False
    assert first["device_access"] is False
    assert store.observation(first, gpu_uuid=GPU_UUID) == {
        "reserved_count": 1,
        "lease_id": first["lease_id"],
        "gpu_id": "gpu-0",
        "gpu_uuid": GPU_UUID,
        "lease_receipt_sha256": first["receipt_sha256"],
        "lease_receipt": first,
    }
    with pytest.raises(LabError, match="leased by request"):
        _acquire(store, request_id="Q-request-02")


def test_gpu_lease_rejects_tampering_and_expiry_without_reconciliation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    lease = _acquire(store)
    forged = copy.deepcopy(lease)
    forged["request_id"] = "Q-forged"

    with pytest.raises(LabError, match="hash mismatch"):
        validate_gpu_lease_receipt(
            forged,
            host_id=HOST_ID,
            gpu_uuid=GPU_UUID,
            require_active=False,
        )
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    with pytest.raises(LabError, match="expired"):
        validate_gpu_lease_receipt(
            lease,
            host_id=HOST_ID,
            gpu_uuid=GPU_UUID,
            require_active=True,
            now=future,
        )


def test_gpu_release_requires_cleanup_then_allows_new_owner(tmp_path: Path) -> None:
    store = _store(tmp_path)
    lease = _acquire(store)
    occupied = _process_inventory(
        tmp_path,
        line=f"{GPU_UUID}, 1234, python, 2048\\n",
    )
    with pytest.raises(LabError, match="existing compute processes"):
        build_gpu_cleanup_receipt(
            lease,
            host_id=HOST_ID,
            gpu_uuid=GPU_UUID,
            service_started=True,
            service_stopped=False,
            process_inventory_receipt=occupied,
        )

    cleanup = build_gpu_cleanup_receipt(
        lease,
        host_id=HOST_ID,
        gpu_uuid=GPU_UUID,
        service_started=True,
        service_stopped=True,
        process_inventory_receipt=_process_inventory(tmp_path),
    )
    released = store.release(
        lease,
        gpu_uuid=GPU_UUID,
        cleanup_receipt=cleanup,
    )

    assert released["status"] == "RELEASED"
    assert store.release(
        lease,
        gpu_uuid=GPU_UUID,
        cleanup_receipt=cleanup,
    ) == released
    replacement = _acquire(store, request_id="Q-request-02")
    assert replacement["lease_id"] != lease["lease_id"]


def test_gpu_lease_state_root_refuses_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(LabError, match="state root is unsafe"):
        GpuLeaseStore(
            linked,
            host_id=HOST_ID,
            host_profile_sha256=HOST_PROFILE_SHA,
        )

    assert list(outside.iterdir()) == []


def test_gpu_inventory_uses_fixed_read_only_query_and_binds_uuid(tmp_path: Path) -> None:
    probe = tmp_path / "nvidia-smi"
    probe.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = '--query-gpu=index,uuid,name' || exit 7\n"
        "test \"$2\" = '--format=csv,noheader,nounits' || exit 8\n"
        "printf '0, GPU-test-inventory-0001, Test GPU\\n'\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)

    receipt = collect_nvidia_gpu_inventory(nvidia_smi=probe)

    assert validate_gpu_inventory(receipt) == receipt
    assert select_single_gpu(receipt) == (
        "gpu-0",
        "GPU-test-inventory-0001",
    )
    assert receipt["read_only"] is True
    assert receipt["device_control"] is False
    assert receipt["network_used"] is False


def test_gpu_inventory_rejects_duplicate_or_multiple_selection(tmp_path: Path) -> None:
    probe = tmp_path / "nvidia-smi"
    probe.write_text(
        "#!/bin/sh\n"
        "printf '0, GPU-test-inventory-0001, First GPU\\n'\n"
        "printf '1, GPU-test-inventory-0002, Second GPU\\n'\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)

    receipt = collect_nvidia_gpu_inventory(nvidia_smi=probe)

    with pytest.raises(LabError, match="exactly one"):
        select_single_gpu(receipt)


def test_gpu_process_inventory_is_read_only_and_blocks_existing_compute(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "nvidia-smi-processes"
    probe.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = "
        "'--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' "
        "|| exit 7\n"
        f"printf '{GPU_UUID}, 1234, python, 2048\\n'\n",
        encoding="utf-8",
    )
    probe.chmod(0o700)

    receipt = collect_nvidia_compute_process_inventory(nvidia_smi=probe)

    assert validate_gpu_process_inventory(receipt) == receipt
    assert receipt["read_only"] is True
    assert receipt["process_control"] is False
    assert receipt["network_used"] is False
    with pytest.raises(LabError, match="existing compute processes"):
        require_gpu_compute_processes_absent(receipt, gpu_uuid=GPU_UUID)


def test_empty_gpu_process_inventory_proves_no_compute_process(tmp_path: Path) -> None:
    probe = tmp_path / "nvidia-smi-processes"
    probe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    probe.chmod(0o700)

    receipt = collect_nvidia_compute_process_inventory(nvidia_smi=probe)

    assert require_gpu_compute_processes_absent(receipt, gpu_uuid=GPU_UUID) == receipt
