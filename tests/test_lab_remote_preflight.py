from __future__ import annotations

import copy
import hashlib
import json
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lab_control.asset_selection import RUNTIME_ASSET_VERIFICATION_SCHEMA_ID
from lab_control.cancellation import build_cancel_teardown_receipt
from lab_control.gpu_lease import (
    GpuLeaseStore,
    collect_nvidia_compute_process_inventory,
    collect_nvidia_gpu_inventory,
)
from lab_control.host_observation import (
    build_runtime_receipts,
    collect_environment_observation,
    collect_host_observation,
    collect_repository_observation,
    validate_runtime_receipts,
)
from lab_control.project_cli import main as project_main
from lab_control.remote_preflight import (
    HOST_OBSERVATION_SCHEMA_ID,
    EnvironmentPlacement,
    HostNetworkPolicy,
    HostProfile,
    RepositoryPlacement,
    load_host_profile,
    preflight_remote_request,
    validate_preflight_receipt,
)
from lab_control.remote_request import build_remote_request, load_remote_profile
from lab_control.remote_runner import prepare_remote_request
from lab_control.runner_state import RequestStateStore
from lab_control.workspace import LabError

CONTROL_VERSION = "0.3.0-m3-bootstrap"
ASSET_LOCK = {
    "lock_id": "preflight-assets-v1",
    "workload_id": "preflight-workload-v1",
    "sha256": "d" * 64,
}
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000401"


def _runtime_asset_verification(
    assets: dict[str, dict[str, object]],
) -> dict[str, object]:
    content: dict[str, object] = {
        "schema": RUNTIME_ASSET_VERIFICATION_SCHEMA_ID,
        "observation_sha256": "1" * 64,
        "artifact_manifest_receipt_sha256": "2" * 64,
        "selection_sha256": "3" * 64,
        "asset_lock_sha256": "4" * 64,
        "asset_lock_id": "test-runtime-assets",
        "workload_id": "test-runtime-workload",
        "assets": assets,
        "verified_at": "2026-08-14T12:00:00+08:00",
        "content_rehashed": True,
        "network_used": False,
        "downloads_permitted": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _gpu_inventory(tmp_path: Path) -> dict[str, object]:
    probe = tmp_path / "fake-nvidia-smi"
    if not probe.exists():
        probe.write_text(
            "#!/bin/sh\n"
            "test \"$1\" = '--query-gpu=index,uuid,name' || exit 7\n"
            "printf '0, " + GPU_UUID + ", Test GPU\\n'\n",
            encoding="utf-8",
        )
        probe.chmod(0o700)
    return collect_nvidia_gpu_inventory(nvidia_smi=probe)


def _gpu_process_inventory(tmp_path: Path) -> dict[str, object]:
    probe = tmp_path / "fake-nvidia-smi-processes"
    if not probe.exists():
        probe.write_text(
            "#!/bin/sh\n"
            "test \"$1\" = "
            "'--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory' "
            "|| exit 7\n"
            "exit 0\n",
            encoding="utf-8",
        )
        probe.chmod(0o700)
    return collect_nvidia_compute_process_inventory(nvidia_smi=probe)


def _reserve_gpu_resources(
    request: dict[str, object],
    *,
    remote,
    host,
    state_store: RequestStateStore,
    tmp_path: Path,
) -> dict[str, object]:
    state_store.submit(request)
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
    )
    return state_store.reserve_gpu(
        request,
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
        remote_profile_sha256=remote.sha256,
        inventory_receipt=_gpu_inventory(tmp_path),
        process_inventory_receipt=_gpu_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )


def _cancel_receipt(
    request: dict[str, object],
    *,
    slot_id: str = "gpu-slot-01",
    resource_reservation_receipt: object | None = None,
    resource_release_receipt: object | None = None,
) -> dict[str, object]:
    return build_cancel_teardown_receipt(
        request,
        slot_id=slot_id,
        resource_reservation_receipt=resource_reservation_receipt,
        resource_release_receipt=resource_release_receipt,
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Host Observer Test")
    _git(path, "config", "user.email", "host-observer@example.invalid")
    (path / "README.md").write_text("# observed\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    _git(
        path,
        "remote",
        "add",
        "origin",
        "git@example.invalid:TestOrg/Observed.git",
    )
    _git(path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return path.resolve()


@pytest.fixture
def profiles():
    repository = Path(__file__).resolve().parents[1]
    remote = load_remote_profile(
        repository / "integration" / "remote_profiles",
        "agenticrobot-openpi-gpu",
    )
    host = load_host_profile(
        repository / "integration" / "host_profiles",
        "remote-gpu-hardware",
    )
    return remote, host


@pytest.fixture
def remote_request_payload(profiles):
    remote, _ = profiles
    return build_remote_request(
        task_id="AR-0401",
        snapshot_id="snapshot-0401",
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
        asset_lock=ASSET_LOCK,
        profile=remote,
        control_plane_version=CONTROL_VERSION,
    )


@pytest.fixture
def observation(remote_request_payload, profiles, tmp_path: Path):
    _, host = profiles
    repositories = remote_request_payload["repositories"]
    assets = remote_request_payload["assets"]
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
    )
    gpu_lease = gpu_store.acquire(
        gpu_id="gpu-0",
        gpu_uuid=GPU_UUID,
        request_id=str(remote_request_payload["request_id"]),
        slot_id="gpu-slot-01",
    )
    return {
        "schema": HOST_OBSERVATION_SCHEMA_ID,
        "host_id": host.host_id,
        "hostname": host.hostname,
        "capabilities": list(host.capabilities),
        "device_groups": [],
        "gpu": gpu_store.observation(
            gpu_lease,
            gpu_uuid=GPU_UUID,
        ),
        "repositories": {
            name: {
                **identity,
                "canonical_identity": host.repositories[name].canonical_identity,
                "path": str(host.repositories[name].path),
                "realpath": str(host.repositories[name].path),
                "is_symlink": False,
                "exists": True,
                "submodules_initialized": True,
                "published_ref_contains": True,
            }
            for name, identity in repositories.items()
        },
        "environments": {
            environment_id: {
                "path": str(placement.path),
                "realpath": str(placement.path),
                "is_symlink": False,
                "exists": True,
                "python_major_minor": placement.python_major_minor,
                "import_origins": {
                    module: str(
                        host.repositories[placement.repository].path
                        / "src"
                        / module
                        / "__init__.py"
                    )
                    for module in placement.import_modules
                },
            }
            for environment_id, placement in host.environments.items()
        },
        "assets": {
            role: {
                **identity,
                "exists": True,
                "readable": True,
                "realpath": str(host.asset_roots[0] / role / identity["asset_id"]),
            }
            for role, identity in assets.items()
        },
        "asset_verification_receipt_sha256": "7" * 64,
    }


def test_preflight_binds_source_environment_asset_gpu_and_no_device(
    remote_request_payload,
    profiles,
    observation,
) -> None:
    remote, host = profiles

    receipt = preflight_remote_request(
        remote_request_payload,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
    )

    assert receipt["request_id"] == remote_request_payload["request_id"]
    assert receipt["gpu_lease_id"] == observation["gpu"]["lease_id"]
    assert (
        receipt["gpu_lease_receipt_sha256"]
        == observation["gpu"]["lease_receipt_sha256"]
    )
    assert receipt["gpu_uuid"] == "GPU-00000000-0000-0000-0000-000000000401"
    assert receipt["source_verified"] is True
    assert receipt["environment_verified"] is True
    assert receipt["assets_verified"] is True
    assert receipt["device_access"] is False
    assert len(receipt["receipt_sha256"]) == 64
    assert (
        validate_preflight_receipt(
            receipt,
            request_id=remote_request_payload["request_id"],
            idempotency_key=remote_request_payload["idempotency_key"],
        )
        == receipt
    )


@pytest.mark.parametrize(
    ("section", "key", "field", "value", "message"),
    [
        ("repositories", "openpi", "dirty", True, "missing or dirty"),
        ("repositories", "openpi", "is_symlink", True, "real path"),
        ("repositories", "openpi", "published_ref_contains", False, "not published"),
        (
            "environments",
            "openpi-py311-cuda",
            "import_origins",
            {
                "openpi": "/wrong/source/openpi/__init__.py",
                "openpi_client": "/wrong/source/openpi_client/__init__.py",
            },
            "environment provenance",
        ),
        (
            "assets",
            "openpi-checkpoint",
            "sha256",
            "d" * 64,
            "asset identity",
        ),
    ],
)
def test_preflight_fails_closed_on_dynamic_fact_mismatch(
    remote_request_payload,
    profiles,
    observation,
    section,
    key,
    field,
    value,
    message,
) -> None:
    remote, host = profiles
    invalid = copy.deepcopy(observation)
    invalid[section][key][field] = value

    with pytest.raises(LabError, match=message):
        preflight_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=host,
            observation=invalid,
            control_plane_version=CONTROL_VERSION,
        )


def test_preflight_rejects_device_groups_and_missing_gpu_lease(
    remote_request_payload,
    profiles,
    observation,
) -> None:
    remote, host = profiles
    without_asset_verification = copy.deepcopy(observation)
    without_asset_verification["asset_verification_receipt_sha256"] = None
    with pytest.raises(LabError, match="content-rehashed asset verification"):
        preflight_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=host,
            observation=without_asset_verification,
            control_plane_version=CONTROL_VERSION,
        )
    with_devices = copy.deepcopy(observation)
    with_devices["device_groups"] = ["dialout"]
    with pytest.raises(LabError, match="refuses device groups"):
        preflight_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=host,
            observation=with_devices,
            control_plane_version=CONTROL_VERSION,
        )


def test_preflight_rejects_host_policy_that_allows_large_downloads(
    remote_request_payload,
    profiles,
    observation,
) -> None:
    remote, host = profiles
    unsafe = replace(
        host,
        network_policy=replace(host.network_policy, large_file_downloads=True),
    )

    with pytest.raises(LabError, match="network policies"):
        preflight_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=unsafe,
            observation=observation,
            control_plane_version=CONTROL_VERSION,
        )

    missing_capability = copy.deepcopy(observation)
    missing_capability["capabilities"] = ["gpu:cuda"]
    with pytest.raises(LabError, match="lacks required capabilities"):
        preflight_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=host,
            observation=missing_capability,
            control_plane_version=CONTROL_VERSION,
        )

    no_lease = copy.deepcopy(observation)
    no_lease["gpu"]["lease_id"] = None
    with pytest.raises(LabError, match="GPU lease"):
        preflight_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=host,
            observation=no_lease,
            control_plane_version=CONTROL_VERSION,
        )


def test_request_store_is_idempotent_and_enforces_transitions(
    remote_request_payload,
    profiles,
    observation,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    receipt = preflight_remote_request(
        remote_request_payload,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
    )
    store = RequestStateStore(tmp_path / "state")

    submitted = store.submit(remote_request_payload)
    assert store.submit(remote_request_payload) == submitted
    reserved = store.reserve(
        remote_request_payload["request_id"],
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
    )
    assert reserved["state"] == "RESERVED"
    assert (
        store.reserve(
            remote_request_payload["request_id"],
            slot_id="gpu-slot-01",
            allowed_slots=host.slots,
        )
        == reserved
    )
    preparing = store.attach_preflight(remote_request_payload["request_id"], receipt)
    assert preparing["state"] == "PREPARING"
    assert (
        store.attach_preflight(remote_request_payload["request_id"], receipt)
        == preparing
    )
    with pytest.raises(LabError, match="RUNNING is disabled"):
        store.transition(remote_request_payload["request_id"], "RUNNING")
    cancelled = store.request_cancel(remote_request_payload["request_id"])
    assert cancelled["state"] == "CANCEL_REQUESTED"
    with pytest.raises(LabError, match="requires an explicit teardown receipt"):
        store.transition(remote_request_payload["request_id"], "CANCELLED")
    teardown = _cancel_receipt(remote_request_payload)
    cancelled = store.acknowledge_cancel(remote_request_payload, teardown)
    assert (
        store.acknowledge_cancel(remote_request_payload, teardown)
        == cancelled
    )
    assert [entry["state"] for entry in cancelled["history"]] == [
        "SUBMITTED",
        "RESERVED",
        "PREPARING",
        "CANCEL_REQUESTED",
        "CANCELLED",
    ]
    with pytest.raises(LabError, match="invalid request transition"):
        store.transition(remote_request_payload["request_id"], "RUNNING")


def test_slot_is_exclusive_and_cancel_path_is_explicit(
    remote_request_payload,
    profiles,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    store = RequestStateStore(tmp_path / "state")
    first = store.submit(remote_request_payload)
    second_request = build_remote_request(
        task_id="AR-0402",
        snapshot_id="snapshot-0402",
        repositories=remote_request_payload["repositories"],
        assets=remote_request_payload["assets"],
        asset_lock=remote_request_payload["asset_lock"],
        profile=remote,
        control_plane_version=CONTROL_VERSION,
    )
    second = store.submit(second_request)
    store.reserve(first["request_id"], slot_id="gpu-slot-01", allowed_slots=host.slots)

    with pytest.raises(LabError, match="already reserved"):
        store.reserve(
            second["request_id"],
            slot_id="gpu-slot-01",
            allowed_slots=host.slots,
        )
    cancel_requested = store.transition(first["request_id"], "CANCEL_REQUESTED")
    assert cancel_requested["state"] == "CANCEL_REQUESTED"
    assert (
        store.acknowledge_cancel(
            remote_request_payload, _cancel_receipt(remote_request_payload)
        )["state"]
        == "CANCELLED"
    )
    assert (
        store.reserve(
            second["request_id"],
            slot_id="gpu-slot-01",
            allowed_slots=host.slots,
        )["state"]
        == "RESERVED"
    )


def test_request_cancel_is_idempotent_and_never_claims_teardown(
    remote_request_payload,
    profiles,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    store = RequestStateStore(tmp_path / "state")
    submitted = store.submit(remote_request_payload)

    cancelled = store.request_cancel(submitted["request_id"])

    assert cancelled["state"] == "CANCELLED"
    assert store.request_cancel(submitted["request_id"]) == cancelled
    assert cancelled["cancel_teardown_receipt"] is None
    changed = store.watch(
        submitted["request_id"],
        after_sequence=0,
        timeout_seconds=0,
    )
    assert changed["changed"] is True
    assert changed["observed_sequence"] == 1
    assert changed["terminal"] is True
    unchanged_terminal = store.watch(
        submitted["request_id"],
        after_sequence=1,
        timeout_seconds=0,
    )
    assert unchanged_terminal["changed"] is False
    assert unchanged_terminal["terminal"] is True
    with pytest.raises(LabError, match="timeout must be between"):
        store.watch(submitted["request_id"], timeout_seconds=61)

    active_request = build_remote_request(
        task_id="AR-0405",
        snapshot_id="snapshot-0405",
        repositories=remote_request_payload["repositories"],
        assets=remote_request_payload["assets"],
        asset_lock=remote_request_payload["asset_lock"],
        profile=remote,
        control_plane_version=CONTROL_VERSION,
    )
    active = store.submit(active_request)
    store.reserve(
        active["request_id"],
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
    )
    requested = store.request_cancel(active["request_id"])
    assert requested["state"] == "CANCEL_REQUESTED"
    assert store.request_cancel(active["request_id"]) == requested

    overclaim = _cancel_receipt(active_request)
    overclaim["hardware_safety_claimed"] = True
    overclaim_content = {
        key: value for key, value in overclaim.items() if key != "receipt_sha256"
    }
    overclaim["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            overclaim_content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(LabError, match="overclaims safety"):
        store.acknowledge_cancel(active_request, overclaim)


def test_preflight_receipt_cannot_attach_to_another_request(
    remote_request_payload,
    profiles,
    observation,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    receipt = preflight_remote_request(
        remote_request_payload,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
    )
    store = RequestStateStore(tmp_path / "state")
    other = build_remote_request(
        task_id="AR-0403",
        snapshot_id="snapshot-0403",
        repositories=remote_request_payload["repositories"],
        assets=remote_request_payload["assets"],
        asset_lock=remote_request_payload["asset_lock"],
        profile=remote,
        control_plane_version=CONTROL_VERSION,
    )
    store.submit(other)
    store.reserve(other["request_id"], slot_id="gpu-slot-01", allowed_slots=host.slots)

    with pytest.raises(LabError, match="different request"):
        store.attach_preflight(other["request_id"], receipt)


def test_preflight_receipt_tampering_is_rejected(
    remote_request_payload,
    profiles,
    observation,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    receipt = preflight_remote_request(
        remote_request_payload,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
    )
    tampered = copy.deepcopy(receipt)
    tampered["gpu_uuid"] = "GPU-forged"
    store = RequestStateStore(tmp_path / "state")
    store.submit(remote_request_payload)
    store.reserve(
        remote_request_payload["request_id"],
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
    )

    with pytest.raises(LabError, match="receipt hash mismatch"):
        store.attach_preflight(remote_request_payload["request_id"], tampered)


def test_state_file_is_json_and_contains_no_execution_command(
    remote_request_payload,
    tmp_path: Path,
) -> None:
    store = RequestStateStore(tmp_path / "state")
    state = store.submit(remote_request_payload)
    path = tmp_path / "state" / "requests" / state["request_id"] / "state.json"

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == state
    assert "argv" not in path.read_text(encoding="utf-8")


def test_request_store_rejects_forged_idempotency_identity(
    remote_request_payload,
    tmp_path: Path,
) -> None:
    forged = copy.deepcopy(remote_request_payload)
    forged["idempotency_key"] = "9" * 64
    forged["request_id"] = "Q-" + "9" * 24

    with pytest.raises(LabError, match="idempotency hash mismatch"):
        RequestStateStore(tmp_path / "state").submit(forged)


def test_projectctl_remote_preflight_emits_receipt(
    remote_request_payload,
    observation,
    tmp_path: Path,
    capsys,
) -> None:
    request_path = tmp_path / "request.json"
    observation_path = tmp_path / "observation.json"
    request_path.write_text(
        json.dumps(remote_request_payload),
        encoding="utf-8",
    )
    observation_path.write_text(json.dumps(observation), encoding="utf-8")

    exit_code = project_main(
        [
            "remote-preflight",
            "--request",
            str(request_path),
            "--observation",
            str(observation_path),
            "--profile",
            "agenticrobot-openpi-gpu",
            "--host-profile",
            "remote-gpu-hardware",
            "--control-version",
            CONTROL_VERSION,
        ]
    )

    assert exit_code == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["request_id"] == remote_request_payload["request_id"]
    assert receipt["source_verified"] is True


def test_projectctl_remote_preflight_reports_blocked_observation(
    remote_request_payload,
    observation,
    tmp_path: Path,
    capsys,
) -> None:
    blocked_observation = copy.deepcopy(observation)
    blocked_observation["repositories"]["openpi"]["path"] = "/home/user/zc/openpi"
    request_path = tmp_path / "request.json"
    observation_path = tmp_path / "observation.json"
    request_path.write_text(
        json.dumps(remote_request_payload),
        encoding="utf-8",
    )
    observation_path.write_text(
        json.dumps(blocked_observation),
        encoding="utf-8",
    )

    exit_code = project_main(
        [
            "remote-preflight",
            "--request",
            str(request_path),
            "--observation",
            str(observation_path),
            "--profile",
            "agenticrobot-openpi-gpu",
            "--host-profile",
            "remote-gpu-hardware",
            "--control-version",
            CONTROL_VERSION,
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "repository path mismatch: openpi" in captured.err
    assert captured.out == ""


def test_repository_observer_reads_exact_revision_without_mutation(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "Observed")
    before_status = _git(repository, "status", "--porcelain=v1")

    observed = collect_repository_observation(
        RepositoryPlacement(
            name="Observed",
            canonical_identity="TestOrg/Observed",
            path=repository,
        )
    )

    assert observed["canonical_identity"] == "TestOrg/Observed"
    assert observed["commit"] == _git(repository, "rev-parse", "HEAD")
    assert observed["tree_oid"] == _git(repository, "rev-parse", "HEAD^{tree}")
    assert observed["dirty"] is False
    assert observed["submodules"] == {}
    assert observed["submodules_initialized"] is True
    assert observed["published_ref_contains"] is True
    assert _git(repository, "status", "--porcelain=v1") == before_status


def test_environment_observer_reports_interpreter_and_import_origin() -> None:
    environment_root = Path(sys.prefix).resolve()
    placement = EnvironmentPlacement(
        environment_id="test-python",
        path=environment_root,
        python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
        repository="AgenticRobot",
        import_modules=("json",),
    )

    observed = collect_environment_observation(placement)

    assert observed["exists"] is True
    assert observed["python_major_minor"] == placement.python_major_minor
    assert Path(observed["import_origins"]["json"]).is_file()


def test_host_observer_combines_live_facts_with_runtime_receipts(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "Observed")
    asset_root = tmp_path / "assets"
    asset = asset_root / "checkpoint" / "checkpoint-v1"
    asset.parent.mkdir(parents=True)
    asset.write_text("immutable test artifact\n", encoding="utf-8")
    host_profile = HostProfile(
        host_id="test-host",
        hostname=socket.gethostname(),
        capabilities=("gpu:cuda",),
        repositories={
            "Observed": RepositoryPlacement(
                name="Observed",
                canonical_identity="TestOrg/Observed",
                path=repository,
            )
        },
        environments={},
        asset_roots=(asset_root,),
        slots=("gpu-slot-01",),
        slot_root=tmp_path / "slots",
        control_plane_projectctl=tmp_path / "control-plane/current/projectctl",
        device_access=False,
        network_policy=HostNetworkPolicy(
            task_payload_egress="loopback-only",
            trusted_prepare_egress="explicit-only",
            implicit_downloads=False,
            large_file_downloads=False,
        ),
        sha256="f" * 64,
    )
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
    )
    gpu_uuid = "GPU-runtime-receipt-test"
    gpu_lease = gpu_store.acquire(
        gpu_id="gpu-0",
        gpu_uuid=gpu_uuid,
        request_id="request-test",
        slot_id="gpu-slot-01",
    )
    runtime_assets = {
        "checkpoint": {
            "asset_id": "checkpoint-v1",
            "sha256": "a" * 64,
            "path": str(asset),
        }
    }
    runtime_receipts = build_runtime_receipts(
        host_profile,
        gpu=gpu_store.observation(gpu_lease, gpu_uuid=gpu_uuid),
        assets=runtime_assets,
        asset_verification=_runtime_asset_verification(runtime_assets),
    )

    observed = collect_host_observation(host_profile, runtime_receipts)

    assert observed["hostname"] == socket.gethostname()
    assert observed["repositories"]["Observed"]["dirty"] is False
    assert observed["assets"]["checkpoint"]["exists"] is True
    assert observed["assets"]["checkpoint"]["readable"] is True
    assert observed["gpu"]["lease_id"] == gpu_lease["lease_id"]

    tampered = copy.deepcopy(runtime_receipts)
    tampered["capabilities"].append("device:robot")
    with pytest.raises(LabError, match="identity or authority"):
        validate_runtime_receipts(tampered, host_profile=host_profile)

    outside_asset = tmp_path / "outside-checkpoint"
    outside_asset.write_text("not in the protected asset root\n", encoding="utf-8")
    with pytest.raises(LabError, match="asset path is unsafe"):
        build_runtime_receipts(
            host_profile,
            gpu=gpu_store.observation(gpu_lease, gpu_uuid=gpu_uuid),
            assets={
                "checkpoint": {
                    "asset_id": "checkpoint-v1",
                    "sha256": "a" * 64,
                    "path": str(outside_asset),
                }
            },
        )


def test_projectctl_host_observe_is_local_and_read_only(
    tmp_path: Path,
    capsys,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    host_profile = load_host_profile(
        repository_root / "integration" / "host_profiles",
        "remote-gpu-hardware",
    )
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=host_profile.host_id,
        host_profile_sha256=host_profile.sha256,
    )
    gpu_uuid = "GPU-offline-test"
    gpu_lease = gpu_store.acquire(
        gpu_id="gpu-0",
        gpu_uuid=gpu_uuid,
        request_id="request-offline-test",
        slot_id="gpu-slot-01",
    )
    runtime_receipts = build_runtime_receipts(
        host_profile,
        gpu=gpu_store.observation(gpu_lease, gpu_uuid=gpu_uuid),
        assets={},
    )
    receipts_path = tmp_path / "runtime-receipts.json"
    receipts_path.write_text(
        json.dumps(runtime_receipts),
        encoding="utf-8",
    )

    exit_code = project_main(
        [
            "host-observe",
            "--host-profile",
            "remote-gpu-hardware",
            "--runtime-receipts",
            str(receipts_path),
        ]
    )

    assert exit_code == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed["schema"] == HOST_OBSERVATION_SCHEMA_ID
    assert observed["host_id"] == "remote-gpu-hardware"
    assert observed["hostname"] == socket.gethostname()


def test_prepare_lifecycle_is_idempotent_and_stops_before_execution(
    remote_request_payload,
    profiles,
    observation,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    store = RequestStateStore(tmp_path / "state")
    reserved = _reserve_gpu_resources(
        remote_request_payload,
        remote=remote,
        host=host,
        state_store=store,
        tmp_path=tmp_path,
    )

    prepared = prepare_remote_request(
        remote_request_payload,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        slot_id="gpu-slot-01",
    )
    replayed = prepare_remote_request(
        remote_request_payload,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        slot_id="gpu-slot-01",
    )

    assert prepared == replayed
    assert reserved["resource_reservation_receipt"] is not None
    assert prepared["state"] == "PREPARING"
    assert prepared["preflight_receipt"]["source_verified"] is True
    assert "argv" not in json.dumps(prepared)


def test_failed_prepare_records_precondition_and_keeps_gpu_fail_closed(
    remote_request_payload,
    profiles,
    observation,
    tmp_path: Path,
) -> None:
    remote, host = profiles
    store = RequestStateStore(tmp_path / "state")
    _reserve_gpu_resources(
        remote_request_payload,
        remote=remote,
        host=host,
        state_store=store,
        tmp_path=tmp_path,
    )
    blocked = copy.deepcopy(observation)
    blocked["repositories"]["openpi"]["path"] = "/home/user/zc/openpi"

    with pytest.raises(LabError, match="repository path mismatch"):
        prepare_remote_request(
            remote_request_payload,
            remote_profile=remote,
            host_profile=host,
            observation=blocked,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            slot_id="gpu-slot-01",
        )

    failed = store.load(remote_request_payload["request_id"])
    assert failed["state"] == "FAILED"
    assert failed["failure"]["classification"] == "BLOCKED_PRECONDITION"

    second = build_remote_request(
        task_id="AR-0404",
        snapshot_id="snapshot-0404",
        repositories=remote_request_payload["repositories"],
        assets=remote_request_payload["assets"],
        asset_lock=remote_request_payload["asset_lock"],
        profile=remote,
        control_plane_version=CONTROL_VERSION,
    )
    store.submit(second)
    with pytest.raises(LabError, match="leased by request"):
        _reserve_gpu_resources(
            second,
            remote=remote,
            host=host,
            state_store=store,
            tmp_path=tmp_path,
        )
    assert store.load(str(second["request_id"]))["state"] == "SUBMITTED"


def test_projectctl_exposes_prepare_status_and_cancel_without_execution(
    remote_request_payload,
    observation,
    tmp_path: Path,
    capsys,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    remote = load_remote_profile(
        repository / "integration" / "remote_profiles",
        "agenticrobot-openpi-gpu",
    )
    host = load_host_profile(
        repository / "integration" / "host_profiles",
        "remote-gpu-hardware",
    )
    request_path = tmp_path / "request.json"
    observation_path = tmp_path / "observation.json"
    state_root = tmp_path / "runner-state"
    request_path.write_text(json.dumps(remote_request_payload), encoding="utf-8")
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    _reserve_gpu_resources(
        remote_request_payload,
        remote=remote,
        host=host,
        state_store=RequestStateStore(state_root),
        tmp_path=tmp_path,
    )

    assert (
        project_main(
            [
                "remote-runner",
                "prepare",
                "--request",
                str(request_path),
                "--observation",
                str(observation_path),
                "--profile",
                "agenticrobot-openpi-gpu",
                "--host-profile",
                "remote-gpu-hardware",
                "--slot",
                "gpu-slot-01",
                "--state-root",
                str(state_root),
                "--control-version",
                CONTROL_VERSION,
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["state"] == "PREPARING"
    assert "argv" not in json.dumps(prepared)

    request_id = remote_request_payload["request_id"]
    assert (
        project_main(
            [
                "remote-runner",
                "status",
                request_id,
                "--state-root",
                str(state_root),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == prepared

    assert (
        project_main(
            [
                "remote-runner",
                "cancel",
                request_id,
                "--state-root",
                str(state_root),
            ]
        )
        == 0
    )
    cancelled = json.loads(capsys.readouterr().out)
    assert cancelled["state"] == "CANCEL_REQUESTED"
    assert cancelled["history"][-1]["state"] == "CANCEL_REQUESTED"

    store = RequestStateStore(state_root)
    gpu_store = GpuLeaseStore(
        tmp_path / "gpu-leases",
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
    )
    released = store.reconcile_nonstarted_gpu(
        remote_request_payload,
        gpu_uuid=GPU_UUID,
        process_inventory_receipt=_gpu_process_inventory(tmp_path),
        gpu_store=gpu_store,
    )

    receipt_path = tmp_path / "cancel-receipt.json"
    receipt_path.write_text(
        json.dumps(
            _cancel_receipt(
                remote_request_payload,
                resource_reservation_receipt=released[
                    "resource_reservation_receipt"
                ],
                resource_release_receipt=released["resource_release_receipt"],
            )
        ),
        encoding="utf-8",
    )
    assert (
        project_main(
            [
                "remote-runner",
                "ack-cancel",
                request_id,
                "--request",
                str(request_path),
                "--receipt",
                str(receipt_path),
                "--state-root",
                str(state_root),
            ]
        )
        == 0
    )
    acknowledged = json.loads(capsys.readouterr().out)
    assert acknowledged["state"] == "CANCELLED"
    assert acknowledged["cancel_teardown_receipt"]["hardware_safety_claimed"] is False

    assert (
        project_main(
            [
                "remote-runner",
                "watch",
                request_id,
                "--state-root",
                str(state_root),
                "--after-sequence",
                "3",
                "--timeout-seconds",
                "0",
            ]
        )
        == 0
    )
    watched = json.loads(capsys.readouterr().out)
    assert watched["changed"] is True
    assert watched["observed_sequence"] == 4
    assert watched["terminal"] is True
    assert watched["request"]["state"] == "CANCELLED"


def test_projectctl_sandbox_probe_uses_protected_host_profile(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    observed: dict[str, object] = {}

    def fake_probe(host: HostProfile) -> dict[str, object]:
        observed["host_id"] = host.host_id
        observed["backend"] = host.sandbox_backend
        return {
            "schema": "lab.sandbox-backend-probe.test-double",
            "host_id": host.host_id,
            "task_payload_executed": False,
            "device_access": False,
        }

    monkeypatch.setattr(
        "lab_control.project_cli.collect_sandbox_backend_probe",
        fake_probe,
    )

    assert (
        project_main(
            [
                "sandbox-probe",
                "--host-profile",
                "remote-gpu-hardware",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["host_id"] == "remote-gpu-hardware"
    assert payload["task_payload_executed"] is False
    assert observed["host_id"] == "remote-gpu-hardware"
    assert observed["backend"] is not None
