from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control.clean_slot import materialize_clean_request_sources
from lab_control.command_registry import (
    TrustedCommandRegistry,
    load_command_registry,
)
from lab_control.execution_sandbox import (
    build_execution_sandbox_receipt,
    build_sandbox_backend_probe_receipt,
    validate_execution_sandbox_for_launch,
    validate_sandbox_backend_probe_receipt,
)
from lab_control.gpu_lease import GpuLeaseStore
from lab_control.network_namespace import NetworkNamespaceProbePolicy
from lab_control.remote_execution import (
    TrustedCommandOutcome,
    collect_attempt_artifacts,
)
from lab_control.remote_preflight import (
    PREFLIGHT_RECEIPT_SCHEMA_ID,
    EnvironmentPlacement,
    HostNetworkPolicy,
    HostProfile,
    RepositoryPlacement,
    SandboxBackendPolicy,
    preflight_remote_request,
)
from lab_control.remote_request import (
    RemoteProfile,
    ServicePolicy,
    build_remote_request,
)
from lab_control.remote_runner import (
    execute_remote_request_attempt,
    stage_remote_network_namespace_evidence,
    stage_remote_request_readiness,
    stage_remote_request_sources,
    stage_remote_service_launch_plan,
    stage_remote_service_launch_spec,
    stage_remote_service_receipt,
)
from lab_control.runner_state import RequestStateStore, validate_request_state_identity
from lab_control.service_identity import SERVICE_HEALTH_SCHEMA_ID, build_service_receipt
from lab_control.slot_readiness import prepare_slot_readiness
from lab_control.workspace import LabError

CONTROL_VERSION = "0.3.0-m3-bootstrap"
ASSET_LOCK = {
    "lock_id": "clean-slot-assets-v1",
    "workload_id": "clean-slot-workload-v1",
    "sha256": "d" * 64,
}


def _sandbox_receipt(
    staged: dict[str, object],
    *,
    observed_at: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    launch_spec = staged.get("service_launch_spec")
    service_receipt = staged.get("service_receipt")
    assert isinstance(launch_spec, dict)
    assert isinstance(service_receipt, dict)
    preflight = staged.get("preflight_receipt")
    assert isinstance(preflight, dict)
    host_profile_sha256 = str(preflight["host_profile_sha256"])
    policy = SandboxBackendPolicy(
        backend_id="bubblewrap-v1",
        executable=Path("/usr/bin/bwrap"),
        allowed_gpu_devices=(
            Path("/dev/nvidia0"),
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia-uvm"),
        ),
        max_probe_age_seconds=300,
    )
    backend = build_sandbox_backend_probe_receipt(
        host_id="remote-gpu-hardware",
        host_profile_sha256=host_profile_sha256,
        backend_sha256="b" * 64,
        allowed_devices=("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"),
        user_namespace=True,
        network_namespace=True,
        loopback_available=True,
        external_network_unreachable=True,
        filesystem_isolated=True,
        forbidden_devices_absent=True,
        process_namespace=True,
        no_new_privileges=True,
        observed_at=observed_at,
    )
    return build_execution_sandbox_receipt(
        launch_spec=launch_spec,
        service_receipt=service_receipt,
        backend_probe_receipt=backend,
        host_profile_sha256=host_profile_sha256,
        sandbox_backend_policy=policy,
        control_plane_version=CONTROL_VERSION,
        now=now,
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip()


def _repository(path: Path, content: str) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Clean Slot Test")
    _git(path, "config", "user.email", "clean-slot@example.invalid")
    (path / "source.txt").write_text(content, encoding="utf-8")
    _git(path, "add", "source.txt")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _write_network_process(
    proc_root: Path,
    *,
    pid: int,
    parent_pid: int,
    process_group_id: int,
    session_id: int,
    start_ticks: int,
) -> None:
    process = proc_root / str(pid)
    (process / "ns").mkdir(parents=True)
    fields = [
        "S",
        str(parent_pid),
        str(process_group_id),
        str(session_id),
        *(["0"] * 15),
        str(start_ticks),
    ]
    (process / "stat").write_text(
        f"{pid} (network process) {' '.join(fields)}\n",
        encoding="utf-8",
    )
    (process / "status").write_text(
        "Name:\tnetwork-process\nNoNewPrivs:\t1\n",
        encoding="utf-8",
    )


def _network_evidence_fixture(
    tmp_path: Path,
) -> tuple[Path, int, int, NetworkNamespaceProbePolicy]:
    proc_root = (tmp_path / "proc").resolve()
    collector_pid = 4100
    anchor_pid = 4200
    _write_network_process(
        proc_root,
        pid=collector_pid,
        parent_pid=1,
        process_group_id=collector_pid,
        session_id=collector_pid,
        start_ticks=100,
    )
    _write_network_process(
        proc_root,
        pid=anchor_pid,
        parent_pid=collector_pid,
        process_group_id=anchor_pid,
        session_id=anchor_pid,
        start_ticks=200,
    )
    (proc_root / str(collector_pid) / "ns" / "net").write_text(
        "collector-net\n",
        encoding="utf-8",
    )
    (proc_root / str(anchor_pid) / "ns" / "net").write_text(
        "anchor-net\n",
        encoding="utf-8",
    )

    protected = (tmp_path / "protected-network-probe").resolve()
    protected.mkdir()
    enter = protected / "nsenter"
    python = protected / "python"
    enter.write_text(
        "#!/usr/bin/python3\n"
        "import json, sys\n"
        "assert sys.argv[1].startswith('--net=')\n"
        "assert sys.argv[2] == '--'\n"
        "assert sys.argv[4:6] == ['-I', '-c']\n"
        "inode = int(sys.argv[-1])\n"
        "print(json.dumps({\n"
        "  'external_routes_absent': True,\n"
        "  'loopback_available': True,\n"
        "  'network_namespace_inode': inode,\n"
        "  'only_loopback_interfaces': True,\n"
        "  'requested_namespace_matched': True,\n"
        "}, sort_keys=True))\n",
        encoding="utf-8",
    )
    enter.chmod(0o700)
    python.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    python.chmod(0o700)
    return (
        proc_root,
        collector_pid,
        anchor_pid,
        NetworkNamespaceProbePolicy(
            policy_id="fixed-netns-probe-v1",
            enter_executable=enter,
            python_executable=python,
            max_probe_age_seconds=300,
        ),
    )


def _profiles(tmp_path: Path) -> tuple[RemoteProfile, HostProfile, dict[str, Path]]:
    repositories = {
        "AgenticRobot": _repository(tmp_path / "AgenticRobot", "agenticrobot\n"),
        "openpi": _repository(tmp_path / "openpi", "openpi\n"),
    }
    modules = {
        "AgenticRobot": "staged_agenticrobot_probe",
        "openpi": "staged_openpi_probe",
    }
    for name, module in modules.items():
        (repositories[name] / f"{module}.py").write_text(
            f"REPOSITORY = {name!r}\n", encoding="utf-8"
        )
        _git(repositories[name], "add", f"{module}.py")
        _git(repositories[name], "commit", "-m", "add staged import probe")
    asset_root = (tmp_path / "assets").resolve()
    asset_root.mkdir()
    (asset_root / "checkpoint-test").write_text(
        "small identity-only test asset\n", encoding="utf-8"
    )
    remote = RemoteProfile(
        profile_id="clean-slot-test",
        target_host="offline-host",
        risk_level="R1",
        repositories=("AgenticRobot", "openpi"),
        required_capabilities=("gpu:cuda",),
        required_environments=("agentic-env", "openpi-env"),
        required_assets=("checkpoint",),
        command_ids=("gpu-smoke",),
        gpu_count=1,
        device_access=False,
        network_egress="loopback-only",
        downloads_permitted=False,
        service_policy=ServicePolicy(
            service_name="openpi-policy",
            repository="openpi",
            environment_id="openpi-env",
            protocol="openpi-policy-service.v1",
            launch_mode="ephemeral",
            launch_command_id="openpi-policy-service",
            endpoint_scheme="http",
        ),
        sha256="a" * 64,
    )
    host = HostProfile(
        host_id="offline-host",
        hostname="offline-host",
        capabilities=("gpu:cuda",),
        repositories={
            name: RepositoryPlacement(
                name=name,
                canonical_identity=f"test/{name}",
                path=path,
            )
            for name, path in repositories.items()
        },
        environments={
            "agentic-env": EnvironmentPlacement(
                environment_id="agentic-env",
                path=Path(sys.prefix).resolve(),
                python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                repository="AgenticRobot",
                import_modules=(modules["AgenticRobot"],),
                source_roots=(".",),
            ),
            "openpi-env": EnvironmentPlacement(
                environment_id="openpi-env",
                path=Path(sys.prefix).resolve(),
                python_major_minor=f"{sys.version_info.major}.{sys.version_info.minor}",
                repository="openpi",
                import_modules=(modules["openpi"],),
                source_roots=(".",),
            ),
        },
        asset_roots=(asset_root,),
        slots=("gpu-slot-01",),
        slot_root=(tmp_path / "slots").resolve(),
        control_plane_projectctl=(
            tmp_path / "control-plane/current/projectctl"
        ).resolve(),
        device_access=False,
        network_policy=HostNetworkPolicy(
            task_payload_egress="loopback-only",
            trusted_prepare_egress="explicit-only",
            implicit_downloads=False,
            large_file_downloads=False,
        ),
        sha256="b" * 64,
        sandbox_backend=SandboxBackendPolicy(
            backend_id="bubblewrap-v1",
            executable=Path("/usr/bin/bwrap"),
            allowed_gpu_devices=(
                Path("/dev/nvidia0"),
                Path("/dev/nvidiactl"),
                Path("/dev/nvidia-uvm"),
            ),
            max_probe_age_seconds=300,
        ),
    )
    return remote, host, repositories


def _request(
    remote: RemoteProfile,
    repositories: dict[str, Path],
    *,
    task_id: str = "AR-CLEAN-01",
) -> dict[str, object]:
    identities = {
        name: {
            "commit": _git(path, "rev-parse", "HEAD"),
            "tree_oid": _git(path, "rev-parse", "HEAD^{tree}"),
            "dirty": False,
            "submodules": {},
        }
        for name, path in repositories.items()
    }
    return build_remote_request(
        task_id=task_id,
        snapshot_id="clean-snapshot-01",
        repositories=identities,
        assets={"checkpoint": {"asset_id": "checkpoint-test", "sha256": "c" * 64}},
        asset_lock=ASSET_LOCK,
        profile=remote,
        control_plane_version=CONTROL_VERSION,
    )


def _preflight_receipt(
    request: dict[str, object], remote: RemoteProfile, host: HostProfile
) -> dict[str, object]:
    content: dict[str, object] = {
        "schema": PREFLIGHT_RECEIPT_SCHEMA_ID,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "host_id": host.host_id,
        "host_profile_sha256": host.sha256,
        "remote_profile_sha256": remote.sha256,
        "control_plane_version": CONTROL_VERSION,
        "asset_lock": request["asset_lock"],
        "observation_sha256": "d" * 64,
        "gpu_lease_id": "offline-test-lease",
        "gpu_lease_receipt_sha256": "e" * 64,
        "gpu_uuid": "GPU-offline-test",
        "source_verified": True,
        "environment_verified": True,
        "assets_verified": True,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical(content)).hexdigest(),
    }


def _observation(
    request: dict[str, object], remote: RemoteProfile, host: HostProfile
) -> dict[str, object]:
    gpu_store = GpuLeaseStore(
        host.slot_root / ".gpu-leases",
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
    )
    gpu_lease = gpu_store.acquire(
        gpu_id="gpu-0",
        gpu_uuid="GPU-offline-test",
        request_id=str(request["request_id"]),
        slot_id="gpu-slot-01",
    )
    return {
        "schema": "lab.host-observation.v1",
        "host_id": host.host_id,
        "hostname": host.hostname,
        "capabilities": list(host.capabilities),
        "device_groups": [],
        "gpu": gpu_store.observation(gpu_lease, gpu_uuid="GPU-offline-test"),
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
            for name, identity in request["repositories"].items()
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
                        host.repositories[placement.repository].path / f"{module}.py"
                    )
                    for module in placement.import_modules
                },
            }
            for environment_id, placement in host.environments.items()
        },
        "assets": {
            "checkpoint": {
                **request["assets"]["checkpoint"],
                "exists": True,
                "readable": True,
                "realpath": str(host.asset_roots[0] / "checkpoint-test"),
            }
        },
        "asset_verification_receipt_sha256": "7" * 64,
    }


def test_clean_request_source_staging_is_exact_inert_and_idempotent(
    tmp_path: Path,
) -> None:
    remote, host, repositories = _profiles(tmp_path)
    request = _request(remote, repositories)
    store = RequestStateStore(tmp_path / "state")
    store.submit(request)
    store.reserve(
        str(request["request_id"]),
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
    )
    store.attach_preflight(
        str(request["request_id"]),
        _preflight_receipt(request, remote, host),
    )
    anchor_state = {
        name: (_git(path, "rev-parse", "HEAD"), _git(path, "status", "--short"))
        for name, path in repositories.items()
    }

    staged = stage_remote_request_sources(
        request,
        remote_profile=remote,
        host_profile=host,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
    )

    assert staged["state"] == "PREPARING"
    receipt = staged["source_receipt"]
    assert receipt["source_prepared"] is True
    assert receipt["execution_eligible"] is False
    assert receipt["environment_prepared"] is False
    assert receipt["assets_prepared"] is False
    assert receipt["service_started"] is False
    for name, identity in request["repositories"].items():
        checkout = Path(receipt["repositories"][name]["path"])
        assert _git(checkout, "branch", "--show-current") == ""
        assert _git(checkout, "rev-parse", "HEAD") == identity["commit"]
        assert _git(checkout, "rev-parse", "HEAD^{tree}") == identity["tree_oid"]
        assert _git(checkout, "status", "--short") == ""
        assert _git(checkout, "config", "--get", "lfs.fetchexclude") == "*"
        assert anchor_state[name] == (
            _git(repositories[name], "rev-parse", "HEAD"),
            _git(repositories[name], "status", "--short"),
        )
    assert (
        stage_remote_request_sources(
            request,
            remote_profile=remote,
            host_profile=host,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
        )
        == staged
    )
    with pytest.raises(LabError, match="RUNNING is disabled"):
        store.transition(str(request["request_id"]), "RUNNING")

    forged = copy.deepcopy(receipt)
    forged["control_plane_version"] = "forged-control"
    forged_content = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = hashlib.sha256(_canonical(forged_content)).hexdigest()
    with pytest.raises(
        LabError, match="does not match preflight control_plane_version"
    ):
        store.attach_source_preparation(str(request["request_id"]), forged)

    extra_field = copy.deepcopy(receipt)
    extra_field["grant_device"] = True
    extra_content = {
        key: value for key, value in extra_field.items() if key != "receipt_sha256"
    }
    extra_field["receipt_sha256"] = hashlib.sha256(
        _canonical(extra_content)
    ).hexdigest()
    with pytest.raises(LabError, match="unexpected shape"):
        store.attach_source_preparation(str(request["request_id"]), extra_field)


def test_clean_slot_rejects_dirty_anchor_and_materialized_source_drift(
    tmp_path: Path,
) -> None:
    remote, host, repositories = _profiles(tmp_path)
    request = _request(remote, repositories)
    (repositories["openpi"] / "source.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(LabError, match="anchor identity mismatch"):
        materialize_clean_request_sources(
            request,
            remote_profile=remote,
            host_profile=host,
            control_plane_version=CONTROL_VERSION,
            slot_id="gpu-slot-01",
        )
    _git(repositories["openpi"], "restore", "source.txt")
    receipt = materialize_clean_request_sources(
        request,
        remote_profile=remote,
        host_profile=host,
        control_plane_version=CONTROL_VERSION,
        slot_id="gpu-slot-01",
    )
    checkout = Path(receipt["repositories"]["AgenticRobot"]["path"])
    (checkout / "source.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(LabError, match="clean slot repository drifted"):
        materialize_clean_request_sources(
            request,
            remote_profile=remote,
            host_profile=host,
            control_plane_version=CONTROL_VERSION,
            slot_id="gpu-slot-01",
        )


def test_readiness_proves_staged_imports_and_assets_but_cannot_execute(
    tmp_path: Path,
) -> None:
    remote, host, repositories = _profiles(tmp_path)
    request = _request(remote, repositories)
    observation = _observation(request, remote, host)
    preflight = preflight_remote_request(
        request,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
    )
    store = RequestStateStore(tmp_path / "readiness-state")
    store.submit(request)
    store.reserve(
        str(request["request_id"]),
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
    )
    store.attach_preflight(str(request["request_id"]), preflight)
    staged = stage_remote_request_sources(
        request,
        remote_profile=remote,
        host_profile=host,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
    )

    ready = stage_remote_request_readiness(
        request,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
    )

    receipt = ready["readiness_receipt"]
    assert receipt["source_prepared"] is True
    assert receipt["environment_prepared"] is True
    assert receipt["assets_prepared"] is True
    assert receipt["service_started"] is False
    assert receipt["execution_eligible"] is False
    assert receipt["device_access"] is False
    assert (
        receipt["source_receipt_sha256"] == staged["source_receipt"]["receipt_sha256"]
    )
    for environment in receipt["environments"].values():
        repository = environment["repository"]
        staged_path = Path(staged["source_receipt"]["repositories"][repository]["path"])
        assert environment["first_party_from_staged_source"] is True
        assert all(
            Path(origin).is_relative_to(staged_path)
            for origin in environment["import_origins"].values()
        )
    assert receipt["assets"]["checkpoint"]["content_rehashed"] is False
    assert (
        stage_remote_request_readiness(
            request,
            remote_profile=remote,
            host_profile=host,
            observation=observation,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
        )
        == ready
    )
    service_staged = stage_remote_service_launch_spec(
        request,
        remote_profile=remote,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        launcher_nonce="offline-launcher-nonce-0001",
    )
    service_spec = service_staged["service_launch_spec"]
    assert service_spec["source"]["repository"] == "openpi"
    assert (
        service_spec["source"]["commit"] == request["repositories"]["openpi"]["commit"]
    )
    assert service_spec["execution_eligible"] is False
    assert service_spec["device_access"] is False
    assert (
        stage_remote_service_launch_spec(
            request,
            remote_profile=remote,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            launcher_nonce="offline-launcher-nonce-0001",
        )
        == service_staged
    )
    with pytest.raises(LabError, match="different service launch spec"):
        stage_remote_service_launch_spec(
            request,
            remote_profile=remote,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            launcher_nonce="offline-launcher-nonce-0002",
        )
    with pytest.raises(LabError, match="RUNNING is disabled"):
        store.transition(str(request["request_id"]), "RUNNING")
    forged = copy.deepcopy(receipt)
    forged["observation_sha256"] = "f" * 64
    forged_content = {
        key: value for key, value in forged.items() if key != "receipt_sha256"
    }
    forged["receipt_sha256"] = hashlib.sha256(_canonical(forged_content)).hexdigest()
    with pytest.raises(LabError, match="binding mismatch: observation_sha256"):
        store.attach_readiness(str(request["request_id"]), forged)
    with pytest.raises(LabError, match="RUNNING is disabled"):
        store.transition(str(request["request_id"]), "RUNNING")


def _service_health(spec: dict[str, object]) -> dict[str, object]:
    return {
        "schema": SERVICE_HEALTH_SCHEMA_ID,
        "service_id": spec["service_id"],
        "launcher_nonce": spec["launcher_nonce"],
        "managed": True,
        "mode": spec["mode"],
        "protocol": spec["protocol"],
        "launch_command_id": spec["launch_command_id"],
        "source": spec["source"],
        "environment": spec["environment"],
        "asset_lock": spec["asset_lock"],
        "assets": spec["assets"],
        "gpu": spec["gpu"],
        "endpoint": {"scheme": "http", "host": "127.0.0.1", "port": 18427},
        "process": {
            "pid": 4242,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _prepared_service_state(
    tmp_path: Path,
    *,
    task_id: str,
) -> tuple[
    RemoteProfile,
    dict[str, object],
    RequestStateStore,
    dict[str, object],
    TrustedCommandRegistry,
]:
    remote, host, repositories = _profiles(tmp_path)
    request = _request(remote, repositories, task_id=task_id)
    observation = _observation(request, remote, host)
    preflight = preflight_remote_request(
        request,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
    )
    store = RequestStateStore(tmp_path / "execution-state")
    store.submit(request)
    store.reserve(
        str(request["request_id"]),
        slot_id="gpu-slot-01",
        allowed_slots=host.slots,
    )
    store.attach_preflight(str(request["request_id"]), preflight)
    stage_remote_request_sources(
        request,
        remote_profile=remote,
        host_profile=host,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
    )
    stage_remote_request_readiness(
        request,
        remote_profile=remote,
        host_profile=host,
        observation=observation,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
    )
    staged = stage_remote_service_launch_spec(
        request,
        remote_profile=remote,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        launcher_nonce="trusted-launcher-nonce-0001",
    )
    spec = staged["service_launch_spec"]
    assert isinstance(spec, dict)
    adapter_root = tmp_path / "protected-adapters"
    adapter_root.mkdir()
    (adapter_root / "openpi-policy-service.json").write_text(
        json.dumps(
            {
                "schema": "lab.service-adapter.v1",
                "adapter_id": "clean-slot-service-v1",
                "launch_command_id": "openpi-policy-service",
                "argv_template": [
                    "{python}",
                    "-I",
                    "-c",
                    "print('ready')",
                    "{workload_id}",
                    "{port}",
                ],
                "working_directory": "/workspace/source/0",
                "environment": {},
                "supported_workload_ids": ["clean-slot-workload-v1"],
                "asset_projections": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    backend_probe = build_sandbox_backend_probe_receipt(
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
        backend_sha256="b" * 64,
        backend_executable="/usr/bin/bwrap",
        allowed_devices=("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm"),
        user_namespace=True,
        network_namespace=True,
        loopback_available=True,
        external_network_unreachable=True,
        filesystem_isolated=True,
        forbidden_devices_absent=True,
        process_namespace=True,
        no_new_privileges=True,
    )
    runtime_root = host.slot_root / "gpu-slot-01" / "runtime"
    runtime_root.mkdir()
    staged = stage_remote_service_launch_plan(
        request,
        remote_profile=remote,
        host_profile=host,
        backend_probe_receipt=backend_probe,
        runtime_root=runtime_root,
        adapter_root=adapter_root,
        endpoint_port=18427,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
    )
    plan = staged["service_launch_plan"]
    endpoint_lease = staged["endpoint_lease_receipt"]
    network_plan = staged["network_namespace_plan"]
    assert isinstance(plan, dict)
    assert isinstance(endpoint_lease, dict)
    assert isinstance(network_plan, dict)
    assert endpoint_lease["namespace_reserved"] is True
    assert endpoint_lease["socket_bound"] is False
    assert (
        plan["endpoint_lease_receipt_sha256"]
        == endpoint_lease["receipt_sha256"]
    )
    assert (
        plan["network_namespace_plan_sha256"]
        == network_plan["receipt_sha256"]
    )
    assert network_plan["namespace_created"] is False
    assert network_plan["launch_authorized"] is False
    assert plan["process_started"] is False
    assert plan["launch_authorized"] is False
    assert (
        stage_remote_service_launch_plan(
            request,
            remote_profile=remote,
            host_profile=host,
            backend_probe_receipt=backend_probe,
            runtime_root=runtime_root,
            adapter_root=adapter_root,
            endpoint_port=18427,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
        )
        == staged
    )
    with pytest.raises(LabError, match="different service launch plan"):
        stage_remote_service_launch_plan(
            request,
            remote_profile=remote,
            host_profile=host,
            backend_probe_receipt=backend_probe,
            runtime_root=runtime_root,
            adapter_root=adapter_root,
            endpoint_port=18428,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
        )
    proc_root, collector_pid, anchor_pid, network_policy = (
        _network_evidence_fixture(tmp_path)
    )
    staged = stage_remote_network_namespace_evidence(
        request,
        remote_profile=remote,
        host_profile=host,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        anchor_pid=anchor_pid,
        policy=network_policy,
        proc_root=proc_root,
        collector_pid=collector_pid,
    )
    anchor_observation = staged["network_namespace_anchor_observation"]
    probe_receipt = staged["network_namespace_probe_receipt"]
    assert isinstance(anchor_observation, dict)
    assert isinstance(probe_receipt, dict)
    assert probe_receipt["network_ready"] is True
    assert probe_receipt["launch_authorized"] is False
    assert (
        stage_remote_network_namespace_evidence(
            request,
            remote_profile=remote,
            host_profile=host,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            anchor_pid=anchor_pid,
            policy=network_policy,
            proc_root=proc_root,
            collector_pid=collector_pid,
        )
        == staged
    )
    with pytest.raises(LabError, match="different namespace anchor"):
        stage_remote_network_namespace_evidence(
            request,
            remote_profile=remote,
            host_profile=host,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            anchor_pid=anchor_pid + 1,
            policy=network_policy,
            proc_root=proc_root,
            collector_pid=collector_pid,
        )
    health = _service_health(spec)
    process = health["process"]
    endpoint = health["endpoint"]
    assert isinstance(process, dict)
    service_receipt = build_service_receipt(
        health,
        launch_spec=spec,
        managed_process_receipt={
            **process,
            "proc_start_time_ticks": 123456,
            "endpoint": endpoint,
            "receipt_sha256": "6" * 64,
        },
    )
    staged = stage_remote_service_receipt(
        request,
        remote_profile=remote,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        service_receipt=service_receipt,
    )
    registry_path = tmp_path / "command-registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "lab.command-registry.v1",
                "registry_id": "clean-slot-test-v1",
                "commands": {
                    "gpu-smoke": {
                        "adapter_id": "gpu-smoke-v1",
                        "service_required": True,
                        "device_access": False,
                        "network_egress": "loopback-only",
                        "downloads_permitted": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    command_registry = load_command_registry(registry_path)
    return remote, request, store, staged, command_registry


def test_trusted_attempt_requires_service_identity_and_records_artifacts(
    tmp_path: Path,
) -> None:
    remote, request, store, staged, command_registry = _prepared_service_state(
        tmp_path,
        task_id="AR-EXEC-01",
    )
    assert staged["service_receipt"]["service_identity_verified"] is True
    with pytest.raises(LabError, match="RUNNING is disabled"):
        store.transition(str(request["request_id"]), "RUNNING")

    def passing_adapter(
        start: dict[str, object],
        attempt_root: Path,
    ) -> TrustedCommandOutcome:
        assert start["command_id"] == "gpu-smoke"
        (attempt_root / "stdout.log").write_text(
            "offline smoke passed\n", encoding="utf-8"
        )
        return TrustedCommandOutcome(
            exit_code=0,
            assertions_passed=True,
            task_success=True,
            service_acknowledged=True,
            artifact_paths=("stdout.log",),
        )

    completed = execute_remote_request_attempt(
        request,
        remote_profile=remote,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        command_registry=command_registry,
        command_id="gpu-smoke",
        executor_nonce="trusted-executor-nonce-0001",
        execution_sandbox_receipt=_sandbox_receipt(staged),
        runtime_root=tmp_path / "runs",
        adapter=passing_adapter,
    )

    assert completed["state"] == "COMPLETED"
    assert completed["outcome"] == {
        "attempt_id": completed["attempts"][0]["result"]["attempt_id"],
        "classification": "PASSED",
        "task_success": True,
        "evidence_complete": True,
    }
    assert completed["attempts"][0]["result"]["artifacts"]["stdout.log"][
        "size_bytes"
    ] == len("offline smoke passed\n")
    assert [entry["state"] for entry in completed["history"]][-3:] == [
        "RUNNING",
        "COLLECTING",
        "COMPLETED",
    ]
    assert validate_request_state_identity(completed, request=request) == completed
    without_plan = copy.deepcopy(completed)
    without_plan["service_sandbox_receipt"] = None
    without_plan["service_launch_plan"] = None
    with pytest.raises(LabError, match="requires launch plan"):
        validate_request_state_identity(without_plan, request=request)
    assert "argv" not in json.dumps(request)
    assert completed["service_launch_plan"]["task_command_eligible"] is False
    assert completed["attempts"][0]["start"]["device_access"] is False
    assert completed["attempts"][0]["start"]["network_egress"] == "loopback-only"
    assert completed["attempts"][0]["start"]["downloads_permitted"] is False
    assert completed["attempts"][0]["start"]["asset_lock"] == ASSET_LOCK
    assert (
        completed["attempts"][0]["start"]["command_adapter_id"]
        == "gpu-smoke-v1"
    )
    assert (
        completed["attempts"][0]["start"]["command_registry_sha256"]
        == command_registry.sha256
    )


def test_trusted_attempt_failure_is_not_reported_as_success(tmp_path: Path) -> None:
    remote, request, store, staged, command_registry = _prepared_service_state(
        tmp_path,
        task_id="AR-EXEC-02",
    )

    def failing_adapter(
        _start: dict[str, object],
        attempt_root: Path,
    ) -> TrustedCommandOutcome:
        (attempt_root / "assertions.json").write_text(
            '{"passed": false}\n',
            encoding="utf-8",
        )
        return TrustedCommandOutcome(
            exit_code=1,
            assertions_passed=False,
            task_success=False,
            service_acknowledged=True,
            artifact_paths=("assertions.json",),
        )

    failed = execute_remote_request_attempt(
        request,
        remote_profile=remote,
        control_plane_version=CONTROL_VERSION,
        state_store=store,
        command_registry=command_registry,
        command_id="gpu-smoke",
        executor_nonce="trusted-executor-nonce-0002",
        execution_sandbox_receipt=_sandbox_receipt(staged),
        runtime_root=tmp_path / "runs",
        adapter=failing_adapter,
    )

    assert failed["state"] == "FAILED"
    assert failed["failure"]["classification"] == "FAILED_ASSERTION"
    assert failed["outcome"]["task_success"] is False
    assert failed["attempts"][0]["result"]["execution_completed"] is True
    assert validate_request_state_identity(failed, request=request) == failed


def test_trusted_attempt_refuses_missing_or_device_leaking_sandbox(
    tmp_path: Path,
) -> None:
    remote, request, store, _, command_registry = _prepared_service_state(
        tmp_path,
        task_id="AR-EXEC-SANDBOX",
    )

    def must_not_run(
        _start: dict[str, object],
        _attempt_root: Path,
    ) -> TrustedCommandOutcome:
        raise AssertionError("adapter must not run without an enforced sandbox")

    with pytest.raises(LabError, match="sandbox receipt"):
        execute_remote_request_attempt(
            request,
            remote_profile=remote,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            command_registry=command_registry,
            command_id="gpu-smoke",
            executor_nonce="trusted-executor-sandbox-defense",
            execution_sandbox_receipt={},
            runtime_root=tmp_path / "runs",
            adapter=must_not_run,
        )
    assert store.load(str(request["request_id"]))["state"] == "PREPARING"

    leaking_probe = build_sandbox_backend_probe_receipt(
        host_id="remote-gpu-hardware",
        host_profile_sha256="a" * 64,
        backend_sha256="b" * 64,
        allowed_devices=("/dev/nvidia0", "/dev/video0"),
        user_namespace=True,
        network_namespace=True,
        loopback_available=True,
        external_network_unreachable=True,
        filesystem_isolated=True,
        forbidden_devices_absent=True,
        process_namespace=True,
        no_new_privileges=True,
    )
    with pytest.raises(LabError, match="device policy"):
        validate_sandbox_backend_probe_receipt(leaking_probe)


def test_execution_sandbox_rejects_stale_backend_probe(tmp_path: Path) -> None:
    _, _, _, staged, _ = _prepared_service_state(
        tmp_path,
        task_id="AR-EXEC-STALE-SANDBOX",
    )
    now = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(LabError, match="stale or from the future"):
        _sandbox_receipt(
            staged,
            observed_at=now - timedelta(seconds=301),
            now=now,
        )


def test_execution_sandbox_cannot_launch_after_probe_expiry(tmp_path: Path) -> None:
    _, _, _, staged, _ = _prepared_service_state(
        tmp_path,
        task_id="AR-EXEC-EXPIRED-SANDBOX",
    )
    created = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
    receipt = _sandbox_receipt(staged, observed_at=created, now=created)

    with pytest.raises(LabError, match="no longer launch-valid"):
        validate_execution_sandbox_for_launch(
            receipt,
            now=created + timedelta(seconds=301),
        )


def test_trusted_attempt_rejects_symlinked_request_runtime_parent(
    tmp_path: Path,
) -> None:
    remote, request, store, staged, command_registry = _prepared_service_state(
        tmp_path,
        task_id="AR-EXEC-LINK",
    )
    runtime_root = tmp_path / "runs"
    outside = tmp_path / "outside-run"
    runtime_root.mkdir()
    outside.mkdir()
    (runtime_root / str(request["request_id"])).symlink_to(
        outside,
        target_is_directory=True,
    )

    def must_not_run(
        _start: dict[str, object],
        _attempt_root: Path,
    ) -> TrustedCommandOutcome:
        raise AssertionError("adapter must not run")

    with pytest.raises(LabError, match="runtime path is a symlink"):
        execute_remote_request_attempt(
            request,
            remote_profile=remote,
            control_plane_version=CONTROL_VERSION,
            state_store=store,
            command_registry=command_registry,
            command_id="gpu-smoke",
            executor_nonce="trusted-executor-link-defense",
            execution_sandbox_receipt=_sandbox_receipt(staged),
            runtime_root=runtime_root,
            adapter=must_not_run,
        )

    assert list(outside.iterdir()) == []


def test_attempt_artifacts_reject_symlinked_path_components(tmp_path: Path) -> None:
    attempt_root = tmp_path / "attempt"
    outside = tmp_path / "outside"
    attempt_root.mkdir()
    outside.mkdir()
    (outside / "result.log").write_text("not trusted\n", encoding="utf-8")
    (attempt_root / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(LabError, match="symlink"):
        collect_attempt_artifacts(attempt_root, ("nested/result.log",))


def test_attempt_artifacts_enforce_total_limit_while_hashing(tmp_path: Path) -> None:
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    (attempt_root / "first.log").write_bytes(b"1234")
    (attempt_root / "second.log").write_bytes(b"5678")

    with pytest.raises(LabError, match="too large"):
        collect_attempt_artifacts(
            attempt_root,
            ("first.log", "second.log"),
            max_total_bytes=7,
        )


def test_clean_slot_rejects_a_symlinked_slot_parent(tmp_path: Path) -> None:
    remote, host, repositories = _profiles(tmp_path)
    request = _request(remote, repositories)
    outside = tmp_path / "outside-slot"
    outside.mkdir()
    host.slot_root.mkdir(parents=True)
    (host.slot_root / "gpu-slot-01").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(LabError, match="slot path is a symlink"):
        materialize_clean_request_sources(
            request,
            remote_profile=remote,
            host_profile=host,
            control_plane_version=CONTROL_VERSION,
            slot_id="gpu-slot-01",
        )

    assert list(outside.iterdir()) == []


def test_readiness_rejects_anchor_import_provenance_for_staged_probe(
    tmp_path: Path,
) -> None:
    remote, host, repositories = _profiles(tmp_path)
    request = _request(remote, repositories)
    observation = _observation(request, remote, host)
    source = materialize_clean_request_sources(
        request,
        remote_profile=remote,
        host_profile=host,
        control_plane_version=CONTROL_VERSION,
        slot_id="gpu-slot-01",
    )
    agentic_checkout = Path(source["repositories"]["AgenticRobot"]["path"])
    (agentic_checkout / "staged_agenticrobot_probe.py").unlink()

    with pytest.raises(
        LabError,
        match="clean slot repository drifted|staged import",
    ):
        prepare_slot_readiness(
            request,
            remote_profile=remote,
            host_profile=host,
            observation=observation,
            source_receipt=source,
            control_plane_version=CONTROL_VERSION,
            slot_id="gpu-slot-01",
        )
