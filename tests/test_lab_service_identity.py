from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lab_control.project_cli import main as project_main
from lab_control.remote_request import (
    RemoteProfile,
    ServicePolicy,
    build_remote_request,
    load_remote_profile,
)
from lab_control.service_identity import (
    SERVICE_HEALTH_SCHEMA_ID,
    build_service_launch_spec,
    build_service_receipt,
    validate_service_launch_spec_identity,
    validate_service_receipt,
    verify_attach_existing_service,
)
from lab_control.slot_readiness import SLOT_READINESS_SCHEMA_ID
from lab_control.workspace import LabError

CONTROL_VERSION = "0.3.0-m3-bootstrap"
ASSET_LOCK = {
    "lock_id": "service-assets-v1",
    "workload_id": "service-workload-v1",
    "sha256": "9" * 64,
}


def _sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _profile(*, mode: str = "ephemeral") -> RemoteProfile:
    return RemoteProfile(
        profile_id="service-identity-test",
        target_host="offline-host",
        risk_level="R1",
        repositories=("AgenticRobot", "openpi"),
        required_capabilities=("gpu:cuda",),
        required_environments=("agentic-env", "openpi-env"),
        required_assets=("checkpoint", "config", "norm-stats"),
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
            launch_mode=mode,
            launch_command_id="openpi-policy-service",
            endpoint_scheme="http",
        ),
        sha256=("a" if mode == "ephemeral" else "b") * 64,
    )


def _request(profile: RemoteProfile) -> dict[str, object]:
    return build_remote_request(
        task_id="AR-SERVICE-01",
        snapshot_id="service-snapshot-01",
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
                "submodules": {"third_party/aloha": "5" * 40},
            },
        },
        assets={
            role: {"asset_id": f"{role}-v1", "sha256": digit * 64}
            for role, digit in zip(
                profile.required_assets,
                ("6", "7", "8"),
                strict=True,
            )
        },
        asset_lock=ASSET_LOCK,
        profile=profile,
        control_plane_version=CONTROL_VERSION,
    )


def _environment(environment_id: str, repository: str) -> dict[str, object]:
    return {
        "environment_id": environment_id,
        "path": f"/trusted/envs/{environment_id}",
        "python_executable": f"/trusted/envs/{environment_id}/bin/python",
        "python_prefix": f"/trusted/envs/{environment_id}",
        "python_major_minor": "3.11",
        "repository": repository,
        "source_roots": [f"/trusted/slots/source/{repository}"],
        "import_origins": {
            repository.lower(): f"/trusted/slots/source/{repository}/module.py"
        },
        "first_party_from_staged_source": True,
    }


def _readiness(request: dict[str, object], profile: RemoteProfile) -> dict[str, object]:
    content: dict[str, object] = {
        "schema": SLOT_READINESS_SCHEMA_ID,
        "request_id": request["request_id"],
        "idempotency_key": request["idempotency_key"],
        "task_id": request["task_id"],
        "snapshot_id": request["snapshot_id"],
        "slot_id": "gpu-slot-01",
        "host_profile_sha256": "9" * 64,
        "remote_profile_sha256": profile.sha256,
        "control_plane_version": CONTROL_VERSION,
        "preflight_receipt_sha256": "a" * 64,
        "source_receipt_sha256": "b" * 64,
        "observation_sha256": "c" * 64,
        "environments": {
            environment_id: _environment(
                environment_id,
                (
                    profile.service_policy.repository
                    if environment_id == profile.service_policy.environment_id
                    else "AgenticRobot"
                ),
            )
            for environment_id in profile.required_environments
        },
        "assets": {
            role: {
                **identity,
                "realpath": f"/trusted/assets/{identity['asset_id']}",
                "identity_attested_by_runtime_receipt": True,
                "content_rehashed": False,
            }
            for role, identity in request["assets"].items()
        },
        "gpu_lease_id": "gpu-lease-service-01",
        "gpu_lease_receipt_sha256": "e" * 64,
        "gpu_uuid": "GPU-offline-service-01",
        "source_prepared": True,
        "environment_prepared": True,
        "assets_prepared": True,
        "service_started": False,
        "execution_eligible": False,
        "device_access": False,
    }
    return {**content, "receipt_sha256": _sha256(content)}


def _spec(*, mode: str = "ephemeral") -> tuple[dict[str, object], dict[str, object]]:
    profile = _profile(mode=mode)
    request = _request(profile)
    spec = build_service_launch_spec(
        request,
        remote_profile=profile,
        readiness_receipt=_readiness(request, profile),
        control_plane_version=CONTROL_VERSION,
        slot_id="gpu-slot-01",
        launcher_nonce="launcher-nonce-service-0001",
    )
    return spec, request


def _health(spec: dict[str, object]) -> dict[str, object]:
    endpoint_policy = spec["endpoint_policy"]
    assert isinstance(endpoint_policy, dict)
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
        "endpoint": {
            "scheme": endpoint_policy["scheme"],
            "host": "127.0.0.1",
            "port": 18427,
        },
        "process": {"pid": 12345, "started_at": "2026-08-14T12:00:00+08:00"},
    }


def _process_attestation(spec: dict[str, object]) -> dict[str, object]:
    endpoint_policy = spec["endpoint_policy"]
    assert isinstance(endpoint_policy, dict)
    return {
        "pid": 12345,
        "started_at": "2026-08-14T12:00:00+08:00",
        "proc_start_time_ticks": 987654,
        "endpoint": {
            "scheme": endpoint_policy["scheme"],
            "host": "127.0.0.1",
            "port": 18427,
        },
        "receipt_sha256": "d" * 64,
    }


def test_launch_spec_is_content_addressed_protected_and_inert() -> None:
    profile = _profile()
    request = _request(profile)
    readiness = _readiness(request, profile)

    first = build_service_launch_spec(
        request,
        remote_profile=profile,
        readiness_receipt=readiness,
        control_plane_version=CONTROL_VERSION,
        slot_id="gpu-slot-01",
        launcher_nonce="launcher-nonce-service-0001",
    )
    second = build_service_launch_spec(
        request,
        remote_profile=profile,
        readiness_receipt=readiness,
        control_plane_version=CONTROL_VERSION,
        slot_id="gpu-slot-01",
        launcher_nonce="launcher-nonce-service-0001",
    )

    assert first == second
    assert validate_service_launch_spec_identity(first) == first
    assert first["source"] == {
        "repository": "openpi",
        "commit": request["repositories"]["openpi"]["commit"],
        "tree_oid": request["repositories"]["openpi"]["tree_oid"],
        "submodules": request["repositories"]["openpi"]["submodules"],
    }
    assert first["endpoint_policy"] == {
        "scheme": "http",
        "host": "127.0.0.1",
        "dynamic_port": True,
    }
    assert first["execution_eligible"] is False
    serialized = json.dumps(first)
    assert "argv" not in serialized
    assert '"port"' not in serialized


def test_matching_health_builds_strict_non_executing_service_receipt() -> None:
    spec, _ = _spec()

    receipt = build_service_receipt(
        _health(spec),
        launch_spec=spec,
        managed_process_receipt=_process_attestation(spec),
    )

    assert validate_service_receipt(receipt) == receipt
    assert receipt["service_identity_verified"] is True
    assert receipt["asset_lock"] == ASSET_LOCK
    assert receipt["endpoint"]["port"] == 18427
    assert receipt["source"]["commit"] == "3" * 40
    assert receipt["device_access"] is False
    assert receipt["execution_eligible"] is False


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("launcher_nonce", "stale-launcher-nonce-9999", "launcher_nonce"),
        ("source", {"repository": "openpi", "commit": "f" * 40}, "source"),
        ("environment", {"environment_id": "wrong"}, "environment"),
        ("assets", {}, "assets"),
        (
            "gpu",
            {
                "lease_id": "old",
                "lease_receipt_sha256": "f" * 64,
                "gpu_uuid": "old",
            },
            "gpu",
        ),
    ],
)
def test_health_identity_mismatch_is_rejected(
    field: str, replacement: object, message: str
) -> None:
    spec, _ = _spec()
    health = _health(spec)
    health[field] = replacement

    with pytest.raises(LabError, match=message):
        build_service_receipt(
            health,
            launch_spec=spec,
            managed_process_receipt=_process_attestation(spec),
        )


def test_unmanaged_and_non_loopback_services_are_rejected() -> None:
    spec, _ = _spec()
    unmanaged = _health(spec)
    unmanaged["managed"] = False
    with pytest.raises(LabError, match="unmanaged service is forbidden"):
        build_service_receipt(
            unmanaged,
            launch_spec=spec,
            managed_process_receipt=_process_attestation(spec),
        )

    external = _health(spec)
    external["endpoint"]["host"] = "0.0.0.0"
    with pytest.raises(LabError, match="violates protected policy"):
        build_service_receipt(
            external,
            launch_spec=spec,
            managed_process_receipt=_process_attestation(spec),
        )


def test_attach_existing_requires_protected_mode_and_exact_managed_receipt() -> None:
    attach_spec, _ = _spec(mode="attach-existing")
    receipt = build_service_receipt(
        _health(attach_spec),
        launch_spec=attach_spec,
        managed_process_receipt=_process_attestation(attach_spec),
    )

    assert verify_attach_existing_service(attach_spec, receipt) == receipt

    stale = copy.deepcopy(receipt)
    stale["source"]["commit"] = "e" * 40
    content = {key: value for key, value in stale.items() if key != "receipt_sha256"}
    stale["receipt_sha256"] = _sha256(content)
    with pytest.raises(LabError, match="identity mismatch: source"):
        verify_attach_existing_service(attach_spec, stale)

    ephemeral_spec, _ = _spec()
    with pytest.raises(LabError, match="protected attach-existing Profile"):
        verify_attach_existing_service(ephemeral_spec, receipt)


def test_receipt_tampering_and_extra_authority_fail_closed() -> None:
    spec, _ = _spec()
    receipt = build_service_receipt(
        _health(spec),
        launch_spec=spec,
        managed_process_receipt=_process_attestation(spec),
    )
    tampered = copy.deepcopy(receipt)
    tampered["endpoint"]["port"] = 18428
    with pytest.raises(LabError, match="hash mismatch"):
        validate_service_receipt(tampered)

    extra = copy.deepcopy(receipt)
    extra["device_authority"] = "control"
    content = {key: value for key, value in extra.items() if key != "receipt_sha256"}
    extra["receipt_sha256"] = _sha256(content)
    with pytest.raises(LabError, match="unexpected shape"):
        validate_service_receipt(extra)


def test_projectctl_builds_and_verifies_identity_without_starting_service(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).resolve().parents[1]
    profile = load_remote_profile(
        repository / "integration" / "remote_profiles",
        "agenticrobot-openpi-gpu",
    )
    request = _request(profile)
    readiness = _readiness(request, profile)
    request_path = tmp_path / "request.json"
    readiness_path = tmp_path / "readiness.json"
    spec_path = tmp_path / "launch-spec.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")

    result = project_main(
        [
            "service-identity",
            "launch-spec",
            "--request",
            str(request_path),
            "--readiness",
            str(readiness_path),
            "--profile",
            profile.profile_id,
            "--control-version",
            CONTROL_VERSION,
            "--slot",
            "gpu-slot-01",
            "--launcher-nonce",
            "launcher-nonce-service-0001",
        ]
    )
    assert result == 0
    spec = json.loads(capsys.readouterr().out)
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    assert spec["service_id"].startswith("SVC-")
    assert spec["execution_eligible"] is False
