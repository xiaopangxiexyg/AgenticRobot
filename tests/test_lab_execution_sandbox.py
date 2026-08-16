from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control import execution_sandbox
from lab_control.endpoint_lease import EndpointLeaseStore
from lab_control.execution_sandbox import (
    build_sandbox_backend_probe_receipt,
    build_service_sandbox_receipt,
    collect_sandbox_backend_probe,
    validate_service_sandbox_for_launch,
    validate_service_sandbox_receipt,
)
from lab_control.managed_service import (
    build_attested_service_receipt,
    collect_managed_service_process_receipt,
    collect_managed_service_teardown_receipt,
    validate_managed_service_process_receipt,
    validate_managed_service_teardown_receipt,
)
from lab_control.network_namespace import build_network_namespace_plan
from lab_control.project_cli import main as project_main
from lab_control.remote_preflight import (
    EnvironmentPlacement,
    HostNetworkPolicy,
    HostProfile,
    SandboxBackendPolicy,
    load_host_profile,
)
from lab_control.remote_request import (
    RemoteProfile,
    ServicePolicy,
    build_remote_request,
)
from lab_control.service_adapters import load_trusted_service_adapter
from lab_control.service_identity import (
    SERVICE_HEALTH_SCHEMA_ID,
    build_service_launch_spec,
)
from lab_control.service_launch_plan import (
    TrustedServiceAdapter,
    build_service_launch_plan,
    validate_service_launch_plan_bindings,
    validate_service_launch_plan_receipt,
)
from lab_control.slot_readiness import SLOT_READINESS_SCHEMA_ID
from lab_control.workspace import LabError

_SERVICE_ADAPTER_ROOT = (
    Path(__file__).resolve().parents[1] / "integration" / "service_adapters"
)


def _service_adapter(command_id: str = "openpi-policy-service") -> TrustedServiceAdapter:
    return load_trusted_service_adapter(_SERVICE_ADAPTER_ROOT, command_id)


def _fake_host(backend: Path) -> HostProfile:
    return HostProfile(
        host_id="sandbox-test-host",
        hostname="sandbox-test-host",
        capabilities=("gpu:cuda",),
        repositories={},
        environments={},
        asset_roots=(Path("/tmp/assets"),),
        slots=("gpu-slot-01",),
        slot_root=Path("/tmp/slots"),
        control_plane_projectctl=Path("/tmp/projectctl"),
        device_access=False,
        network_policy=HostNetworkPolicy(
            task_payload_egress="loopback-only",
            trusted_prepare_egress="explicit-only",
            implicit_downloads=False,
            large_file_downloads=False,
        ),
        sha256="a" * 64,
        sandbox_backend=SandboxBackendPolicy(
            backend_id="bubblewrap-v1",
            executable=backend,
            allowed_gpu_devices=(Path("/dev/nvidia0"), Path("/dev/nvidiactl")),
            max_probe_age_seconds=300,
        ),
    )


def _write_fake_bwrap(path: Path, *, external_network_unreachable: bool) -> None:
    facts = {
        "allowed_devices_present": True,
        "external_network_unreachable": external_network_unreachable,
        "filesystem_isolated": True,
        "forbidden_devices_absent": True,
        "loopback_available": True,
        "network_namespace": True,
        "no_new_privileges": True,
        "process_namespace": True,
        "user_namespace": True,
    }
    path.write_text(
        "#!/usr/bin/python3\n"
        "import json, sys\n"
        "assert '--unshare-net' in sys.argv\n"
        "assert '--unshare-user' in sys.argv\n"
        "assert '--clearenv' in sys.argv\n"
        "assert '/home' not in sys.argv\n"
        "allowed = json.loads(sys.argv[-1])\n"
        f"facts = {facts!r}\n"
        "facts['allowed_devices'] = allowed\n"
        "print(json.dumps(facts, sort_keys=True))\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def test_fixed_sandbox_probe_binds_backend_policy_and_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = tmp_path / "bwrap"
    _write_fake_bwrap(backend, external_network_unreachable=True)
    host = _fake_host(backend)
    monkeypatch.setattr(
        execution_sandbox,
        "_validate_allowed_device_nodes",
        lambda _devices: None,
    )

    receipt = collect_sandbox_backend_probe(host)

    assert receipt["backend_executable"] == str(backend)
    assert receipt["host_profile_sha256"] == host.sha256
    assert receipt["allowed_devices"] == ["/dev/nvidia0", "/dev/nvidiactl"]
    assert receipt["external_network_unreachable"] is True
    assert receipt["device_control"] is False


def test_fixed_sandbox_probe_rejects_failed_network_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = tmp_path / "bwrap"
    _write_fake_bwrap(backend, external_network_unreachable=False)
    monkeypatch.setattr(
        execution_sandbox,
        "_validate_allowed_device_nodes",
        lambda _devices: None,
    )

    with pytest.raises(LabError, match="external_network_unreachable"):
        collect_sandbox_backend_probe(_fake_host(backend))


def test_fixed_sandbox_probe_rejects_symlinked_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual = tmp_path / "actual-bwrap"
    _write_fake_bwrap(actual, external_network_unreachable=True)
    backend = tmp_path / "bwrap"
    backend.symlink_to(actual)
    monkeypatch.setattr(
        execution_sandbox,
        "_validate_allowed_device_nodes",
        lambda _devices: None,
    )

    with pytest.raises(LabError, match="cannot safely open"):
        collect_sandbox_backend_probe(_fake_host(backend))


def test_host_profile_rejects_non_gpu_sandbox_device(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = (
        repository
        / "integration"
        / "host_profiles"
        / "remote-gpu-hardware.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["sandbox_backend"]["allowed_gpu_devices"] = ["/dev/video0"]
    target = tmp_path / "remote-gpu-hardware.json"
    target.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="non-GPU device"):
        load_host_profile(tmp_path, "remote-gpu-hardware")


def _sha256(payload: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _service_sandbox_fixture(
    tmp_path: Path,
    *,
    workload_id: str | None = "dual_rm65_dmgripper_latest",
) -> tuple[
    HostProfile,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    Path,
]:
    environment_root = (tmp_path / "env").resolve()
    environment_root.mkdir()
    environment_bin = environment_root / "bin"
    environment_bin.mkdir()
    environment_python = environment_bin / "python"
    environment_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    environment_python.chmod(0o700)
    slot_root = (tmp_path / "slots").resolve()
    openpi_root = (
        slot_root / "gpu-slot-01" / "prepared" / "request" / "openpi"
    )
    source_root = openpi_root / "src"
    client_source_root = openpi_root / "packages" / "openpi-client" / "src"
    source_root.mkdir(parents=True)
    client_source_root.mkdir(parents=True)
    runtime_root = slot_root / "gpu-slot-01" / "runtime" / "service"
    runtime_root.mkdir(parents=True)
    asset_root = (tmp_path / "assets").resolve()
    asset_root.mkdir()
    asset_paths: dict[str, Path] = {}
    for role in ("openpi-checkpoint", "openpi-config"):
        path = asset_root / role
        path.mkdir()
        asset_paths[role] = path
    norm_stats = asset_root / "openpi-norm-stats"
    norm_stats.write_text('{"norm": true}\n', encoding="utf-8")
    asset_paths["openpi-norm-stats"] = norm_stats
    backend = (tmp_path / "bwrap").resolve()
    _write_fake_bwrap(backend, external_network_unreachable=True)
    host = HostProfile(
        host_id="sandbox-service-host",
        hostname="sandbox-service-host",
        capabilities=("gpu:cuda",),
        repositories={},
        environments={
            "openpi-env": EnvironmentPlacement(
                environment_id="openpi-env",
                path=environment_root,
                python_major_minor="3.11",
                repository="openpi",
                import_modules=("openpi",),
                source_roots=(".", "src", "packages/openpi-client/src"),
            )
        },
        asset_roots=(asset_root,),
        slots=("gpu-slot-01",),
        slot_root=slot_root,
        control_plane_projectctl=tmp_path / "projectctl",
        device_access=False,
        network_policy=HostNetworkPolicy(
            task_payload_egress="loopback-only",
            trusted_prepare_egress="explicit-only",
            implicit_downloads=False,
            large_file_downloads=False,
        ),
        sha256="a" * 64,
        sandbox_backend=SandboxBackendPolicy(
            backend_id="bubblewrap-v1",
            executable=backend,
            allowed_gpu_devices=(Path("/dev/nvidia0"), Path("/dev/nvidiactl")),
            max_probe_age_seconds=300,
        ),
    )
    profile = RemoteProfile(
        profile_id="sandbox-service-profile",
        target_host=host.host_id,
        risk_level="R1",
        repositories=("AgenticRobot", "openpi"),
        required_capabilities=("gpu:cuda",),
        required_environments=("openpi-env",),
        required_assets=(
            "openpi-checkpoint",
            "openpi-config",
            "openpi-norm-stats",
        ),
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
            endpoint_scheme="ws",
        ),
        sha256="b" * 64,
    )
    assets = {
        role: {"asset_id": f"{role}-v1", "sha256": digit * 64}
        for role, digit in zip(
            profile.required_assets,
            ("6", "7", "8"),
            strict=True,
        )
    }
    request = build_remote_request(
        task_id="AR-SERVICE-SANDBOX",
        snapshot_id="service-sandbox-snapshot",
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
                "submodules": {},
            },
        },
        assets=assets,
        asset_lock={
            "lock_id": "sandbox-assets-v1",
            "workload_id": workload_id,
            "sha256": "9" * 64,
        },
        profile=profile,
        control_plane_version="0.3.0-m3-bootstrap",
    )
    environment_receipt = {
        "environment_id": "openpi-env",
        "path": str(environment_root),
        "python_executable": str(environment_root / "bin" / "python"),
        "python_prefix": str(environment_root),
        "python_major_minor": "3.11",
        "repository": "openpi",
        "source_roots": [
            str(openpi_root),
            str(source_root),
            str(client_source_root),
        ],
        "import_origins": {"openpi": str(source_root / "openpi" / "__init__.py")},
        "first_party_from_staged_source": True,
    }
    readiness_content: dict[str, object] = {
        "schema": SLOT_READINESS_SCHEMA_ID,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "task_id": request["task_id"],
        "snapshot_id": request["snapshot_id"],
        "slot_id": "gpu-slot-01",
        "host_profile_sha256": host.sha256,
        "remote_profile_sha256": profile.sha256,
        "control_plane_version": "0.3.0-m3-bootstrap",
        "preflight_receipt_sha256": "c" * 64,
        "source_receipt_sha256": "d" * 64,
        "observation_sha256": "e" * 64,
        "environments": {"openpi-env": environment_receipt},
        "assets": {
            role: {
                **identity,
                "realpath": str(asset_paths[role]),
                "identity_attested_by_runtime_receipt": True,
                "content_rehashed": False,
            }
            for role, identity in assets.items()
        },
        "gpu_lease_id": "gpu-lease-sandbox",
        "gpu_lease_receipt_sha256": "f" * 64,
        "gpu_uuid": "GPU-sandbox-test",
        "source_prepared": True,
        "environment_prepared": True,
        "assets_prepared": True,
        "service_started": False,
        "execution_eligible": False,
        "device_access": False,
    }
    readiness = {
        **readiness_content,
        "receipt_sha256": _sha256(readiness_content),
    }
    launch_spec = build_service_launch_spec(
        request,
        remote_profile=profile,
        readiness_receipt=readiness,
        control_plane_version="0.3.0-m3-bootstrap",
        slot_id="gpu-slot-01",
        launcher_nonce="sandbox-launcher-nonce-0001",
    )
    observed_at = datetime.now(timezone.utc)
    probe = build_sandbox_backend_probe_receipt(
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
        backend_sha256="0" * 64,
        backend_executable=str(backend),
        allowed_devices=("/dev/nvidia0", "/dev/nvidiactl"),
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
    return host, launch_spec, readiness, probe, runtime_root


def _endpoint_lease(
    tmp_path: Path,
    host: HostProfile,
    launch_spec: dict[str, object],
    *,
    port: int = 18427,
) -> dict[str, object]:
    endpoint_policy = launch_spec["endpoint_policy"]
    assert isinstance(endpoint_policy, dict)
    return EndpointLeaseStore(
        (tmp_path / "endpoint-leases").resolve(),
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
    ).acquire(
        request_id=str(launch_spec["request_id"]),
        slot_id=str(launch_spec["slot_id"]),
        scheme=str(endpoint_policy["scheme"]),
        port=port,
    )


def _network_plan(
    endpoint_lease: dict[str, object],
    host: HostProfile,
) -> dict[str, object]:
    return build_network_namespace_plan(
        endpoint_lease,
        host_profile=host,
    )


def _fake_managed_proc(
    root: Path,
    *,
    collector_pid: int,
    service_pid: int,
    executable: Path,
    no_new_privileges: bool = True,
    isolate_namespaces: bool = True,
) -> None:
    collector_ns = root / str(collector_pid) / "ns"
    service_root = root / str(service_pid)
    service_ns = service_root / "ns"
    collector_ns.mkdir(parents=True)
    service_ns.mkdir(parents=True)
    collector_fields = [
        "S",
        "1",
        str(collector_pid),
        str(collector_pid),
        *(["0"] * 15),
        "123456",
    ]
    (collector_ns.parent / "stat").write_text(
        f"{collector_pid} (trusted launcher) {' '.join(collector_fields)}\n",
        encoding="utf-8",
    )
    stat_fields = [
        "S",
        str(collector_pid),
        str(service_pid),
        str(service_pid),
        *(["0"] * 15),
        "987654",
    ]
    (service_root / "stat").write_text(
        f"{service_pid} (managed service) {' '.join(stat_fields)}\n",
        encoding="utf-8",
    )
    (service_root / "status").write_text(
        f"Name:\tmanaged-service\nNoNewPrivs:\t{int(no_new_privileges)}\n",
        encoding="utf-8",
    )
    (service_root / "exe").symlink_to(executable)
    for name in ("user", "net", "pid", "mnt"):
        parent = collector_ns / name
        parent.write_text(f"collector-{name}\n", encoding="utf-8")
        child = service_ns / name
        if isolate_namespaces:
            child.write_text(f"service-{name}\n", encoding="utf-8")
        else:
            child.hardlink_to(parent)


def _managed_service_fixture(
    tmp_path: Path,
) -> tuple[
    HostProfile,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    proc_root = (tmp_path / "proc").resolve()
    proc_root.mkdir()
    environment = host.environments["openpi-env"]
    collector_pid = 4100
    service_pid = 4200
    _fake_managed_proc(
        proc_root,
        collector_pid=collector_pid,
        service_pid=service_pid,
        executable=environment.path / "bin" / "python",
    )
    started_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    process = collect_managed_service_process_receipt(
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        host_profile=host,
        pid=service_pid,
        endpoint_port=18427,
        control_plane_version="0.3.0-m3-bootstrap",
        started_at=started_at,
        proc_root=proc_root,
        collector_pid=collector_pid,
    )
    endpoint_policy = launch_spec["endpoint_policy"]
    assert isinstance(endpoint_policy, dict)
    health = {
        "schema": SERVICE_HEALTH_SCHEMA_ID,
        "service_id": launch_spec["service_id"],
        "launcher_nonce": launch_spec["launcher_nonce"],
        "managed": True,
        "mode": launch_spec["mode"],
        "protocol": launch_spec["protocol"],
        "launch_command_id": launch_spec["launch_command_id"],
        "source": launch_spec["source"],
        "environment": launch_spec["environment"],
        "asset_lock": launch_spec["asset_lock"],
        "assets": launch_spec["assets"],
        "gpu": launch_spec["gpu"],
        "endpoint": {
            "scheme": endpoint_policy["scheme"],
            "host": "127.0.0.1",
            "port": 18427,
        },
        "process": {
            "pid": service_pid,
            "started_at": started_at.isoformat(),
        },
    }
    return host, launch_spec, sandbox, process, health


def test_service_sandbox_binds_exact_inputs_without_task_authority(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )

    receipt = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )

    assert validate_service_sandbox_receipt(receipt) == receipt
    assert (
        validate_service_sandbox_for_launch(
            receipt,
            launch_spec=launch_spec,
            host_profile=host,
        )
        == receipt
    )
    assert receipt["service_launch_eligible"] is True
    assert receipt["task_command_eligible"] is False
    assert receipt["device_access"] is False
    assert receipt["network_egress"] == "loopback-only"
    mounts = receipt["mounts"]
    assert isinstance(mounts, list)
    assert {mount["target"] for mount in mounts} >= {
        "/opt/lab/environment",
        "/workspace/source/0",
        "/assets/openpi-checkpoint",
        "/run/lab",
    }
    assert next(
        mount for mount in mounts if mount["kind"] == "runtime"
    )["read_only"] is False
    assert all(
        mount["read_only"] is True
        for mount in mounts
        if mount["kind"] != "runtime"
    )


def test_service_sandbox_rejects_asset_escape_and_stale_probe(tmp_path: Path) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    config = host.asset_roots[0] / "openpi-config"
    config.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    config.symlink_to(outside, target_is_directory=True)
    with pytest.raises(LabError, match="symlink"):
        build_service_sandbox_receipt(
            launch_spec=launch_spec,
            readiness_receipt=readiness,
            backend_probe_receipt=probe,
            host_profile=host,
            control_plane_version="0.3.0-m3-bootstrap",
            runtime_root=runtime_root,
        )

    sandbox_policy = host.sandbox_backend
    assert sandbox_policy is not None
    stale_probe = build_sandbox_backend_probe_receipt(
        host_id=host.host_id,
        host_profile_sha256=host.sha256,
        backend_sha256="0" * 64,
        backend_executable=str(sandbox_policy.executable),
        allowed_devices=("/dev/nvidia0", "/dev/nvidiactl"),
        user_namespace=True,
        network_namespace=True,
        loopback_available=True,
        external_network_unreachable=True,
        filesystem_isolated=True,
        forbidden_devices_absent=True,
        process_namespace=True,
        no_new_privileges=True,
        observed_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    with pytest.raises(LabError, match="stale"):
        build_service_sandbox_receipt(
            launch_spec=launch_spec,
            readiness_receipt=readiness,
            backend_probe_receipt=stale_probe,
            host_profile=host,
            control_plane_version="0.3.0-m3-bootstrap",
            runtime_root=runtime_root,
        )


def test_service_sandbox_launch_rechecks_live_asset_paths(tmp_path: Path) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    receipt = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    checkpoint = host.asset_roots[0] / "openpi-checkpoint"
    checkpoint.rmdir()

    with pytest.raises(LabError, match="missing"):
        validate_service_sandbox_for_launch(
            receipt,
            launch_spec=launch_spec,
            host_profile=host,
        )


def test_projectctl_builds_inert_service_sandbox_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    launch_path = tmp_path / "launch.json"
    readiness_path = tmp_path / "readiness.json"
    probe_path = tmp_path / "probe.json"
    launch_path.write_text(json.dumps(launch_spec), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    monkeypatch.setattr(
        "lab_control.project_cli.load_host_profile",
        lambda _root, _host_id: host,
    )

    result = project_main(
        [
            "service-identity",
            "sandbox-receipt",
            "--launch-spec",
            str(launch_path),
            "--readiness",
            str(readiness_path),
            "--backend-probe",
            str(probe_path),
            "--host-profile",
            host.host_id,
            "--runtime-root",
            str(runtime_root),
            "--control-version",
            "0.3.0-m3-bootstrap",
        ]
    )

    assert result == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["service_id"] == launch_spec["service_id"]
    assert receipt["service_launch_eligible"] is True
    assert receipt["task_command_eligible"] is False


def test_managed_service_receipt_binds_os_process_and_health(
    tmp_path: Path,
) -> None:
    host, launch_spec, sandbox, process, health = _managed_service_fixture(tmp_path)

    assert (
        validate_managed_service_process_receipt(
            process,
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
        )
        == process
    )
    receipt = build_attested_service_receipt(
        health,
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        managed_process_receipt=process,
        host_profile=host,
    )

    assert receipt["managed_process_receipt_sha256"] == process["receipt_sha256"]
    assert receipt["process"] == {
        "pid": 4200,
        "started_at": process["started_at"],
        "proc_start_time_ticks": 987654,
    }
    assert receipt["endpoint"] == process["endpoint"]
    assert receipt["service_identity_verified"] is True
    assert receipt["device_access"] is False


def test_managed_service_collection_rejects_weak_process_isolation(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    proc_root = (tmp_path / "proc").resolve()
    proc_root.mkdir()
    _fake_managed_proc(
        proc_root,
        collector_pid=4100,
        service_pid=4200,
        executable=host.environments["openpi-env"].path / "bin" / "python",
        no_new_privileges=False,
    )

    with pytest.raises(LabError, match="NoNewPrivs"):
        collect_managed_service_process_receipt(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            pid=4200,
            endpoint_port=18427,
            control_plane_version="0.3.0-m3-bootstrap",
            started_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            proc_root=proc_root,
            collector_pid=4100,
        )


def test_attested_service_rejects_payload_process_or_endpoint_claim(
    tmp_path: Path,
) -> None:
    host, launch_spec, sandbox, process, health = _managed_service_fixture(tmp_path)
    asserted_process = health["process"]
    assert isinstance(asserted_process, dict)
    asserted_process["pid"] = 4201

    with pytest.raises(LabError, match="process differs"):
        build_attested_service_receipt(
            health,
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            managed_process_receipt=process,
            host_profile=host,
        )


def test_projectctl_verifies_health_only_with_process_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    host, launch_spec, sandbox, process, health = _managed_service_fixture(tmp_path)
    paths = {
        "launch-spec": launch_spec,
        "service-sandbox": sandbox,
        "process-receipt": process,
        "health": health,
    }
    for name, payload in paths.items():
        (tmp_path / f"{name}.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
    monkeypatch.setattr(
        "lab_control.project_cli.load_host_profile",
        lambda _root, _host_id: host,
    )

    result = project_main(
        [
            "service-identity",
            "verify-health",
            "--launch-spec",
            str(tmp_path / "launch-spec.json"),
            "--service-sandbox",
            str(tmp_path / "service-sandbox.json"),
            "--process-receipt",
            str(tmp_path / "process-receipt.json"),
            "--health",
            str(tmp_path / "health.json"),
            "--host-profile",
            host.host_id,
        ]
    )

    assert result == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["managed_process_receipt_sha256"] == process["receipt_sha256"]
    assert receipt["execution_eligible"] is False


def test_managed_service_teardown_requires_waitpid_and_process_group_absence(
    tmp_path: Path,
) -> None:
    _host, launch_spec, sandbox, process, _health = _managed_service_fixture(
        tmp_path
    )
    shutil.rmtree(tmp_path / "proc" / "4200")

    receipt = collect_managed_service_teardown_receipt(
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        managed_process_receipt=process,
        proc_root=(tmp_path / "proc").resolve(),
        collector_pid=4100,
        waitpid=lambda pid, options: (pid, 0) if options else (0, 0),
    )

    assert (
        validate_managed_service_teardown_receipt(
            receipt,
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            managed_process_receipt=process,
        )
        == receipt
    )
    assert receipt["reaped_by_original_launcher"] is True
    assert receipt["original_process_absent"] is True
    assert receipt["process_group_absent"] is True
    assert receipt["exit_code"] == 0
    assert receipt["device_access"] is False
    assert receipt["hardware_safety_claimed"] is False


def test_managed_service_teardown_rejects_running_or_unreaped_child(
    tmp_path: Path,
) -> None:
    _host, launch_spec, sandbox, process, _health = _managed_service_fixture(
        tmp_path
    )

    with pytest.raises(LabError, match="remains present"):
        collect_managed_service_teardown_receipt(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            managed_process_receipt=process,
            proc_root=(tmp_path / "proc").resolve(),
            collector_pid=4100,
            waitpid=lambda pid, _options: (pid, 0),
        )

    shutil.rmtree(tmp_path / "proc" / "4200")
    with pytest.raises(LabError, match="has not exited and been reaped"):
        collect_managed_service_teardown_receipt(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            managed_process_receipt=process,
            proc_root=(tmp_path / "proc").resolve(),
            collector_pid=4100,
            waitpid=lambda _pid, _options: (0, 0),
        )


def test_service_launch_plan_compiles_only_sandbox_local_fixed_command(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    adapter = _service_adapter()
    endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)

    plan = build_service_launch_plan(
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        host_profile=host,
        adapter=adapter,
        endpoint_lease_receipt=endpoint_lease,
        network_namespace_plan=_network_plan(endpoint_lease, host),
    )

    assert validate_service_launch_plan_receipt(plan) == plan
    assert (
        validate_service_launch_plan_bindings(
            plan,
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=adapter,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=_network_plan(endpoint_lease, host),
        )
        == plan
    )
    payload_argv = plan["payload_argv"]
    payload_environment = plan["payload_environment"]
    launcher_argv = plan["launcher_argv"]
    assert isinstance(payload_argv, list)
    assert isinstance(payload_environment, dict)
    assert isinstance(launcher_argv, list)
    assert payload_argv[0] == "/opt/lab/environment/bin/python"
    assert "/workspace/source/0/scripts/serve_dexteleop_policy.py" in payload_argv
    assert "dual_rm65_dmgripper_latest" in payload_argv
    assert payload_argv[-2:] == [
        "--checkpoint-dir",
        "/assets/openpi-checkpoint",
    ]
    assert payload_environment["HF_HUB_OFFLINE"] == "1"
    assert "HTTP_PROXY" not in payload_environment
    assert "--unshare-net" not in launcher_argv
    assert "--clearenv" in launcher_argv
    assert [
        "--ro-bind",
        str((tmp_path / "assets" / "openpi-norm-stats").resolve()),
        "/assets/openpi-checkpoint/assets/dual_rm65_dmgripper/norm_stats.json",
    ] == launcher_argv[
        launcher_argv.index(
            "/assets/openpi-checkpoint/assets/dual_rm65_dmgripper/norm_stats.json"
        )
        - 2 : launcher_argv.index(
            "/assets/openpi-checkpoint/assets/dual_rm65_dmgripper/norm_stats.json"
        )
        + 1
    ]
    assert "/dev/video0" not in launcher_argv
    assert plan["process_started"] is False
    assert plan["endpoint_namespace_reserved"] is True
    assert plan["endpoint_reservation_required"] is True
    assert plan["network_topology"] == "shared-isolated-loopback"
    assert plan["outer_network_namespace_required"] is True
    assert plan["outer_network_namespace_joined"] is False
    assert plan["direct_launcher_execution_permitted"] is False
    assert plan["launch_authorized"] is False
    assert plan["service_launch_eligible"] is False
    assert plan["task_command_eligible"] is False

    with pytest.raises(LabError, match="no service adapter"):
        _service_adapter("unregistered-service")


def test_service_launch_plan_allows_project_without_workload_identity(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path,
        workload_id=None,
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    generic = TrustedServiceAdapter(
        adapter_id="generic-service-v1",
        launch_command_id="openpi-policy-service",
        manifest_sha256="0" * 64,
        argv_template=("{python}", "-I", "-c", "print('ready')", "{port}"),
    )

    endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)
    plan = build_service_launch_plan(
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        host_profile=host,
        adapter=generic,
        endpoint_lease_receipt=endpoint_lease,
        network_namespace_plan=_network_plan(endpoint_lease, host),
    )

    assert plan["adapter"]["supported_workload_ids"] == []
    assert plan["payload_argv"][-1] == "18427"
    requiring_workload = TrustedServiceAdapter(
        adapter_id="workload-service-v1",
        launch_command_id="openpi-policy-service",
        manifest_sha256="1" * 64,
        argv_template=("{python}", "-I", "-c", "print('ready')"),
        supported_workload_ids=("some-workload",),
    )
    with pytest.raises(LabError, match="requires a workload identity"):
        requiring_lease = _endpoint_lease(tmp_path, host, launch_spec)
        build_service_launch_plan(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=requiring_workload,
            endpoint_lease_receipt=requiring_lease,
            network_namespace_plan=_network_plan(requiring_lease, host),
        )


@pytest.mark.parametrize(
    "adapter",
    [
        TrustedServiceAdapter(
            adapter_id="bad-shell-v1",
            launch_command_id="openpi-policy-service",
            manifest_sha256="0" * 64,
            argv_template=("/bin/bash", "-c", "echo unsafe"),
        ),
        TrustedServiceAdapter(
            adapter_id="bad-proxy-v1",
            launch_command_id="openpi-policy-service",
            manifest_sha256="0" * 64,
            argv_template=("{python}", "-m", "openpi.serving"),
            environment=(("HTTP_PROXY", "http://proxy.invalid"),),
        ),
        TrustedServiceAdapter(
            adapter_id="bad-token-v1",
            launch_command_id="openpi-policy-service",
            manifest_sha256="0" * 64,
            argv_template=("{python}", "{host_path}"),
        ),
        TrustedServiceAdapter(
            adapter_id="bad-offline-override-v1",
            launch_command_id="openpi-policy-service",
            manifest_sha256="0" * 64,
            argv_template=("{python}", "-m", "openpi.serving"),
            environment=(("HF_HUB_OFFLINE", "0"),),
        ),
    ],
)
def test_service_launch_plan_rejects_shell_proxy_or_host_token(
    tmp_path: Path,
    adapter: TrustedServiceAdapter,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )

    with pytest.raises(LabError, match="trusted service adapter"):
        endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)
        build_service_launch_plan(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=adapter,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=_network_plan(endpoint_lease, host),
        )


def test_service_launch_plan_rejects_wrong_workload_or_projection_escape(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    unsupported = TrustedServiceAdapter(
        adapter_id="wrong-workload-v1",
        launch_command_id="openpi-policy-service",
        manifest_sha256="0" * 64,
        argv_template=("{python}", "-m", "openpi.serving"),
        supported_workload_ids=("another-workload",),
    )
    escaping = TrustedServiceAdapter(
        adapter_id="escaping-projection-v1",
        launch_command_id="openpi-policy-service",
        manifest_sha256="0" * 64,
        argv_template=("{python}", "-m", "openpi.serving"),
        asset_projections=(
            ("openpi-norm-stats", "openpi-checkpoint", "../model.json"),
        ),
    )

    with pytest.raises(LabError, match="does not support selected workload"):
        endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)
        build_service_launch_plan(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=unsupported,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=_network_plan(endpoint_lease, host),
        )
    with pytest.raises(LabError, match="unsafe asset projection"):
        endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)
        build_service_launch_plan(
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=escaping,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=_network_plan(endpoint_lease, host),
        )


def test_service_launch_plan_rejects_rehashed_asset_projection_drift(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    adapter = _service_adapter()
    endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)
    plan = build_service_launch_plan(
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        host_profile=host,
        adapter=adapter,
        endpoint_lease_receipt=endpoint_lease,
        network_namespace_plan=_network_plan(endpoint_lease, host),
    )
    launcher_argv = plan["launcher_argv"]
    assert isinstance(launcher_argv, list)
    target = "/assets/openpi-checkpoint/assets/dual_rm65_dmgripper/norm_stats.json"
    launcher_argv[launcher_argv.index(target)] = (
        "/assets/openpi-checkpoint/assets/other-model/norm_stats.json"
    )
    plan["launcher_argv_sha256"] = _sha256(launcher_argv)
    content = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    plan["receipt_sha256"] = _sha256(content)

    assert validate_service_launch_plan_receipt(plan) == plan
    with pytest.raises(LabError, match="mount.*drifted"):
        validate_service_launch_plan_bindings(
            plan,
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=adapter,
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=_network_plan(endpoint_lease, host),
        )


def test_service_launch_plan_rejects_rehashed_device_mount_expansion(
    tmp_path: Path,
) -> None:
    host, launch_spec, readiness, probe, runtime_root = _service_sandbox_fixture(
        tmp_path
    )
    sandbox = build_service_sandbox_receipt(
        launch_spec=launch_spec,
        readiness_receipt=readiness,
        backend_probe_receipt=probe,
        host_profile=host,
        control_plane_version="0.3.0-m3-bootstrap",
        runtime_root=runtime_root,
    )
    endpoint_lease = _endpoint_lease(tmp_path, host, launch_spec)
    plan = build_service_launch_plan(
        launch_spec=launch_spec,
        service_sandbox_receipt=sandbox,
        host_profile=host,
        adapter=TrustedServiceAdapter(
            adapter_id="openpi-policy-service-v1",
            launch_command_id="openpi-policy-service",
            manifest_sha256="0" * 64,
            argv_template=("{python}", "-m", "openpi.serving"),
        ),
        endpoint_lease_receipt=endpoint_lease,
        network_namespace_plan=_network_plan(endpoint_lease, host),
    )
    launcher_argv = plan["launcher_argv"]
    assert isinstance(launcher_argv, list)
    separator = launcher_argv.index("--")
    launcher_argv[separator:separator] = [
        "--dev-bind",
        "/dev/nvidia9",
        "/dev/nvidia9",
    ]
    plan["launcher_argv_sha256"] = _sha256(launcher_argv)
    content = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    plan["receipt_sha256"] = _sha256(content)

    assert validate_service_launch_plan_receipt(plan) == plan
    with pytest.raises(LabError, match="device.*drifted"):
        validate_service_launch_plan_bindings(
            plan,
            launch_spec=launch_spec,
            service_sandbox_receipt=sandbox,
            host_profile=host,
            adapter=TrustedServiceAdapter(
                adapter_id="openpi-policy-service-v1",
                launch_command_id="openpi-policy-service",
                manifest_sha256="0" * 64,
                argv_template=("{python}", "-m", "openpi.serving"),
            ),
            endpoint_lease_receipt=endpoint_lease,
            network_namespace_plan=_network_plan(endpoint_lease, host),
        )
