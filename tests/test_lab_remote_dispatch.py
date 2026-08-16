from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.project_cli import main as project_main
from lab_control.remote_dispatch import RemoteDispatchPlanner, validate_remote_outbox
from lab_control.remote_inbox import RemoteInbox
from lab_control.remote_request import load_remote_profile
from lab_control.runner_state import RequestStateStore
from lab_control.ssh_transport import (
    control_remote_request,
    dispatch_remote_outbox,
    install_pinned_known_host,
    verify_pinned_known_host,
)
from lab_control.transport import (
    build_transport_receipt,
    load_transport_profile,
)
from lab_control.workspace import LabError, WorkspaceManager


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, name: str) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Remote Dispatch Test")
    _git(path, "config", "user.email", "remote-dispatch@example.invalid")
    (path / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    _git(path, "remote", "add", "origin", f"git@example.invalid:TestOrg/{name}.git")
    _git(path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return path.resolve()


def _write_remote_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "lab.remote-profile.v1",
                "profile_id": "remote-dispatch-test",
                "description": "Offline dispatch planning contract",
                "execution": "remote-gpu",
                "target_host": "offline-host",
                "risk_level": "R1",
                "requires_clean": True,
                "repositories": ["AgenticRobot", "openpi"],
                "required_capabilities": ["gpu:cuda"],
                "required_environments": ["agentic-env", "openpi-env"],
                "required_assets": ["checkpoint", "config", "norm-stats"],
                "command_ids": ["gpu-smoke"],
                "resources": {
                    "gpu_count": 1,
                    "device_access": False,
                    "network_egress": "loopback-only",
                    "downloads_permitted": False,
                },
                "service": {
                    "service_name": "openpi-policy",
                    "repository": "openpi",
                    "environment_id": "openpi-env",
                    "protocol": "openpi-policy-service.v1",
                    "launch_mode": "ephemeral",
                    "launch_command_id": "openpi-policy-service",
                    "endpoint_scheme": "http",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_transport_profile(
    path: Path,
    *,
    fingerprint: str | None,
) -> None:
    status = "ACTIVE" if fingerprint is not None else "UNRESOLVED"
    path.write_text(
        json.dumps(
            {
                "schema": "lab.transport-profile.v1",
                "transport_id": "offline-host-ssh",
                "target_host": "offline-host",
                "kind": "ssh-stdio",
                "status": status,
                "ssh_alias": "offline-host",
                "ssh_host": "offline-host",
                "ssh_port": 22,
                "ssh_user": "runner",
                "host_key_fingerprints": (
                    [fingerprint] if fingerprint is not None else []
                ),
                "remote_projectctl": "/opt/agenticrobot/projectctl",
                "remote_inbox_root": "/var/lib/agenticrobot/inbox",
                "remote_state_root": "/var/lib/agenticrobot/state",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def dispatch_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    repository_root = tmp_path / "repositories"
    repositories = {
        name: _repository(repository_root / name, name)
        for name in ("AgenticRobot", "RPent", "openpi")
    }
    profile_root = tmp_path / "remote-profiles"
    profile_root.mkdir()
    _write_remote_profile(profile_root / "remote-dispatch-test.json")
    key_path = tmp_path / "offline-host-key"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(key_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    public_fields = (
        (tmp_path / "offline-host-key.pub").read_text(encoding="utf-8").split()
    )
    fingerprint_output = subprocess.run(
        ["ssh-keygen", "-E", "sha256", "-lf", str(key_path) + ".pub"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    fingerprint = fingerprint_output[1]
    known_hosts_path = tmp_path / "known_hosts"
    known_hosts_path.write_text(
        f"offline-host {public_fields[0]} {public_fields[1]}\n",
        encoding="utf-8",
    )
    transport_profile_root = tmp_path / "transport-profiles"
    transport_profile_root.mkdir()
    _write_transport_profile(
        transport_profile_root / "offline-host-ssh.json",
        fingerprint=fingerprint,
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v1",
                "baseline_id": "dispatch-baseline-v1",
                "repositories": {
                    "openpi": {
                        "canonical_identity": "TestOrg/openpi",
                        "commit": _git(repositories["openpi"], "rev-parse", "HEAD"),
                        "tree_oid": _git(
                            repositories["openpi"],
                            "rev-parse",
                            "HEAD^{tree}",
                        ),
                        "submodules": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    asset_lock_path = tmp_path / "remote-assets.json"
    asset_lock_path.write_text(
        json.dumps(
            {
                "schema": "lab.asset-lock.v1",
                "lock_id": "dispatch-assets-v1",
                "workload_id": "dispatch-workload-v1",
                "assets": {
                    "checkpoint": {
                        "asset_id": "checkpoint-v1",
                        "sha256": "a" * 64,
                    },
                    "config": {"asset_id": "config-v1", "sha256": "b" * 64},
                    "norm-stats": {
                        "asset_id": "norm-stats-v1",
                        "sha256": "c" * 64,
                    },
                },
                "unresolved": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAB_REPOSITORY_AGENTICROBOT", str(repositories["AgenticRobot"]))
    monkeypatch.setenv("LAB_REPOSITORY_RPENT", str(repositories["RPent"]))
    monkeypatch.setenv("LAB_REPOSITORY_OPENPI", str(repositories["openpi"]))
    monkeypatch.setenv(
        "LAB_PROJECT_MANIFEST",
        str(Path(__file__).resolve().parents[1] / "integration" / "project.json"),
    )
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_REMOTE_PROFILE_ROOT", str(profile_root))
    monkeypatch.setenv("LAB_BASELINE_LOCK", str(baseline_path))
    monkeypatch.setenv("LAB_REMOTE_ASSET_LOCK", str(asset_lock_path))
    monkeypatch.setenv(
        "LAB_REMOTE_ASSET_OBSERVATIONS",
        str(tmp_path / "asset-observations-not-recorded.json"),
    )
    monkeypatch.setenv(
        "LAB_TRANSPORT_PROFILE_ROOT",
        str(transport_profile_root),
    )
    monkeypatch.setenv("LAB_KNOWN_HOSTS_FILE", str(known_hosts_path))
    return {
        **repositories,
        "profile_root": profile_root,
        "transport_profile_root": transport_profile_root,
        "known_hosts": known_hosts_path,
        "baseline": baseline_path,
        "asset_lock": asset_lock_path,
        "lab_home": tmp_path / "lab-home",
    }


def test_task_remote_plan_is_exact_idempotent_and_not_dispatched(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-REMOTE-01", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "remote",
                "plan",
                "AR-REMOTE-01",
                "remote-dispatch-test",
                "--json",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert first["dispatch"] == {
        "status": "PENDING_DISPATCH",
        "attempts": [],
        "remote_ack": None,
    }
    assert first["execution_eligible"] is False
    assert set(first["request"]["repositories"]) == {"AgenticRobot", "openpi"}
    assert (
        first["effective_workspace"]["repositories"]["AgenticRobot"]["source"]
        == "task_override"
    )
    assert first["effective_workspace"]["repositories"]["openpi"]["source"] == (
        "baseline"
    )
    assert "management_path" not in json.dumps(first["request"])
    assert "argv" not in json.dumps(first["request"])
    request_id = first["request_id"]
    envelope_path = (
        dispatch_environment["lab_home"]
        / "State"
        / "remote-outbox"
        / request_id
        / "envelope.json"
    )
    assert envelope_path.is_file()

    assert (
        main(
            [
                "remote",
                "dispatch",
                request_id,
                "--transport",
                "offline-host-ssh",
                "--dry-run",
                "--json",
            ]
        )
        == 0
    )
    dispatch_plan = json.loads(capsys.readouterr().out)
    assert dispatch_plan["network_used"] is False
    assert dispatch_plan["execution_eligible"] is False
    assert "StrictHostKeyChecking=yes" in dispatch_plan["argv"]
    assert "accept-new" not in dispatch_plan["argv"]

    assert (
        main(
            [
                "remote",
                "plan",
                "AR-REMOTE-01",
                "remote-dispatch-test",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == first
    assert main(["remote", "status", request_id, "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "PENDING_DISPATCH"
    assert status["remote_acknowledged"] is False
    assert status["execution_eligible"] is False
    assert status["envelope"] == first
    assert main(["remote", "list", "AR-REMOTE-01", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == [status]


def test_pinned_host_key_installs_offline_into_dedicated_trust_file(
    dispatch_environment: dict[str, Path],
) -> None:
    profile = load_transport_profile(
        dispatch_environment["transport_profile_root"],
        "offline-host-ssh",
    )
    target = dispatch_environment["known_hosts"].parent / "lab-trust" / "known_hosts"

    first = install_pinned_known_host(
        profile,
        source_known_hosts=dispatch_environment["known_hosts"],
        target_known_hosts=target,
    )
    second = install_pinned_known_host(
        profile,
        source_known_hosts=dispatch_environment["known_hosts"],
        target_known_hosts=target,
    )

    assert first["changed"] is True
    assert first["network_used"] is False
    assert first["accept_new_used"] is False
    assert second["changed"] is False
    assert target.stat().st_mode & 0o777 == 0o600
    assert verify_pinned_known_host(profile, target) == first["host_key_fingerprint"]


def test_remote_doctor_reports_static_readiness_without_network(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["remote", "doctor", "remote-dispatch-test", "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "READY"
    assert receipt["planning_ready"] is True
    assert receipt["dispatch_ready"] is True
    assert receipt["network_checked"] is False
    assert receipt["execution_eligible"] is False
    assert receipt["asset_observations"] == {
        "status": "NOT_RECORDED",
        "activation_authorized": False,
    }
    assert receipt["transports"] == [
        {
            "transport_id": "offline-host-ssh",
            "status": "ACTIVE",
            "ssh_alias": "offline-host",
            "host_key_pin_count": 1,
            "sha256": receipt["transports"][0]["sha256"],
        }
    ]


def test_production_remote_doctor_reports_pin_but_blocks_unresolved_assets() -> None:
    repository = Path(__file__).resolve().parents[1]
    manager = WorkspaceManager(control_repository=repository, environ={})

    receipt = RemoteDispatchPlanner(manager).doctor("agenticrobot-openpi-gpu")

    assert receipt["status"] == "BLOCKED"
    assert receipt["planning_ready"] is False
    assert receipt["dispatch_ready"] is False
    assert receipt["network_checked"] is False
    assert receipt["asset_observations"]["status"] == "OBSERVED_NOT_AUTHORIZED"
    assert receipt["asset_observations"]["candidates"] == [
        {
                "candidate_id": "dual_rm65_dmgripper_latest",
                "blockers": [
                    "content-manifest",
                    "explicit-selection",
                ],
        }
    ]
    assert receipt["asset_observations"]["activation_authorized"] is False
    assert any("unresolved" in blocker for blocker in receipt["blockers"])
    assert not any(
        "ACTIVE pinned Profile" in blocker for blocker in receipt["blockers"]
    )
    assert receipt["transports"] == [
        {
            "transport_id": "remote-gpu-hardware-ssh",
            "status": "ACTIVE",
                "ssh_alias": "agenticrobot-remote-192-168-70-116",
            "host_key_pin_count": 1,
            "sha256": receipt["transports"][0]["sha256"],
        }
    ]


def test_remote_plan_rejects_unpublished_task_commit_before_outbox(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-REMOTE-02", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    checkout = (
        dispatch_environment["lab_home"] / "Worktrees" / "AR-REMOTE-02" / "AgenticRobot"
    )
    (checkout / "change.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(checkout, "add", "change.py")
    _git(checkout, "commit", "-m", "unpublished checkpoint")

    assert main(["remote", "plan", "AR-REMOTE-02", "remote-dispatch-test"]) == 2
    assert "checkpoint and push it" in capsys.readouterr().err
    outbox = dispatch_environment["lab_home"] / "State" / "remote-outbox"
    assert not outbox.exists()


def test_remote_plan_fails_closed_for_unresolved_asset_lock(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    dispatch_environment["asset_lock"].write_text(
        json.dumps(
            {
                "schema": "lab.asset-lock.v1",
                "lock_id": "dispatch-assets-unresolved",
                "workload_id": None,
                "assets": {},
                "unresolved": ["checkpoint", "config", "norm-stats"],
            }
        ),
        encoding="utf-8",
    )
    assert main(["task", "create", "AR-REMOTE-03", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    assert main(["remote", "plan", "AR-REMOTE-03", "remote-dispatch-test"]) == 2
    error = capsys.readouterr().err
    assert "cannot satisfy Profile" in error
    assert "unresolved=checkpoint,config,norm-stats" in error
    assert not (dispatch_environment["lab_home"] / "State" / "remote-outbox").exists()


def test_remote_outbox_tampering_is_rejected(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-REMOTE-04", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "remote",
                "plan",
                "AR-REMOTE-04",
                "remote-dispatch-test",
                "--json",
            ]
        )
        == 0
    )
    envelope = json.loads(capsys.readouterr().out)
    tampered = copy.deepcopy(envelope)
    tampered["dispatch"]["status"] = "DISPATCHED"
    with pytest.raises(LabError, match="must remain PENDING_DISPATCH"):
        validate_remote_outbox(tampered)

    mismatched_assets = copy.deepcopy(envelope)
    mismatched_assets["asset_lock"]["sha256"] = "f" * 64
    with pytest.raises(LabError, match="asset lock mismatch"):
        validate_remote_outbox(mismatched_assets)

    request_id = envelope["request_id"]
    envelope_path = (
        dispatch_environment["lab_home"]
        / "State"
        / "remote-outbox"
        / request_id
        / "envelope.json"
    )
    envelope_path.write_text(json.dumps(tampered), encoding="utf-8")
    manager = WorkspaceManager(control_repository=Path(__file__).resolve().parents[1])
    with pytest.raises(LabError, match="must remain PENDING_DISPATCH"):
        RemoteDispatchPlanner(manager).load(request_id)


def test_remote_inbox_accepts_exact_request_idempotently_without_execution(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-REMOTE-05", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "remote",
                "plan",
                "AR-REMOTE-05",
                "remote-dispatch-test",
                "--json",
            ]
        )
        == 0
    )
    envelope = json.loads(capsys.readouterr().out)
    envelope_path = dispatch_environment["lab_home"] / "envelope-for-target.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")
    inbox_root = dispatch_environment["lab_home"] / "target-inbox"
    state_root = dispatch_environment["lab_home"] / "target-state"

    arguments = [
        "remote-inbox",
        "accept",
        "--envelope",
        str(envelope_path),
        "--profile",
        "remote-dispatch-test",
        "--inbox-root",
        str(inbox_root),
        "--state-root",
        str(state_root),
    ]
    assert project_main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["accepted_state"] == "SUBMITTED"
    assert first["execution_eligible"] is False
    assert first["outbox_envelope_sha256"] == envelope["envelope_sha256"]
    assert project_main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == first

    request_id = envelope["request_id"]
    state_store = RequestStateStore(state_root)
    state = state_store.load(request_id)
    assert state["state"] == "SUBMITTED"
    assert state["slot_id"] is None
    persisted_request = json.loads(
        (inbox_root / request_id / "request.json").read_text(encoding="utf-8")
    )
    assert persisted_request == envelope["request"]

    profile = load_remote_profile(
        dispatch_environment["profile_root"],
        "remote-dispatch-test",
    )
    direct = RemoteInbox(inbox_root, RequestStateStore(state_root)).accept(
        envelope,
        remote_profile=profile,
        control_plane_version="0.3.0-m3-bootstrap",
    )
    assert direct == first

    transport_profile = load_transport_profile(
        dispatch_environment["transport_profile_root"],
        "offline-host-ssh",
    )
    with pytest.raises(LabError, match="host key is not pinned"):
        build_transport_receipt(
            envelope=envelope,
            acceptance=first,
            profile=transport_profile,
            observed_host_key_fingerprint="SHA256:" + "B" * 43,
        )
    transport_receipt = build_transport_receipt(
        envelope=envelope,
        acceptance=first,
        profile=transport_profile,
        observed_host_key_fingerprint=transport_profile.host_key_fingerprints[0],
    )
    assert transport_receipt["transport_authenticated"] is True
    captured: dict[str, object] = {}

    def fake_ssh(
        argv: list[str],
        stdin_payload: str,
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        captured.update(
            {
                "argv": argv,
                "stdin": stdin_payload,
                "timeout": timeout_seconds,
            }
        )
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(first),
            stderr="",
        )

    dispatched_acceptance, dispatched_transport = dispatch_remote_outbox(
        envelope=envelope,
        profile=transport_profile,
        known_hosts_path=dispatch_environment["known_hosts"],
        ssh_runner=fake_ssh,
    )
    assert dispatched_acceptance == first
    assert dispatched_transport["remote_acknowledged"] is True
    assert captured["timeout"] == 30
    assert json.loads(str(captured["stdin"])) == envelope
    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "StrictHostKeyChecking=yes" in argv
    assert "GlobalKnownHostsFile=/dev/null" in argv
    assert "accept-new" not in argv
    assert "--envelope" in argv
    assert "-" in argv
    manager = WorkspaceManager(control_repository=Path(__file__).resolve().parents[1])
    dispatch_status = RemoteDispatchPlanner(manager).attach_acknowledgement(
        request_id,
        acceptance=first,
        transport_receipt=dispatched_transport,
    )
    assert dispatch_status["status"] == "ACKNOWLEDGED"
    assert dispatch_status["remote_acknowledged"] is True
    assert dispatch_status["execution_eligible"] is False
    assert RemoteDispatchPlanner(manager).status(request_id) == dispatch_status

    def fake_control(
        argv: list[str],
        stdin_payload: str,
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        assert stdin_payload == ""
        assert "StrictHostKeyChecking=yes" in argv
        runner_index = argv.index("remote-runner")
        action = argv[runner_index + 1]
        assert argv[runner_index + 2] == request_id
        if action == "status":
            payload: object = state_store.load(request_id)
        elif action == "watch":
            after = int(argv[argv.index("--after-sequence") + 1])
            payload = state_store.watch(
                request_id,
                after_sequence=after,
                timeout_seconds=0,
            )
        elif action == "cancel":
            payload = state_store.request_cancel(request_id)
        else:  # pragma: no cover - fixed argv contract
            raise AssertionError(action)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    observed, observed_fingerprint = control_remote_request(
        envelope=envelope,
        profile=transport_profile,
        known_hosts_path=dispatch_environment["known_hosts"],
        action="status",
        ssh_runner=fake_control,
    )
    observed_status = RemoteDispatchPlanner(manager).attach_remote_state(
        request_id,
        remote_state=observed,
        transport_id=transport_profile.transport_id,
        observed_host_key_fingerprint=observed_fingerprint,
    )
    assert observed_status["remote_observation"]["remote_state_sequence"] == 0
    assert observed_status["remote_observation"]["remote_state"]["state"] == "SUBMITTED"

    reserved = state_store.reserve(
        request_id,
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
    )
    watched, watched_fingerprint = control_remote_request(
        envelope=envelope,
        profile=transport_profile,
        known_hosts_path=dispatch_environment["known_hosts"],
        action="watch",
        after_sequence=0,
        watch_timeout_seconds=0,
        ssh_timeout_seconds=5,
        ssh_runner=fake_control,
    )
    assert watched == reserved
    watched_status = RemoteDispatchPlanner(manager).attach_remote_state(
        request_id,
        remote_state=watched,
        transport_id=transport_profile.transport_id,
        observed_host_key_fingerprint=watched_fingerprint,
    )
    assert watched_status["remote_observation"]["remote_state_sequence"] == 1
    with pytest.raises(LabError, match="state regression"):
        RemoteDispatchPlanner(manager).attach_remote_state(
            request_id,
            remote_state=observed,
            transport_id=transport_profile.transport_id,
            observed_host_key_fingerprint=observed_fingerprint,
        )
    forked = copy.deepcopy(reserved)
    forked["slot_id"] = "gpu-slot-02"
    with pytest.raises(LabError, match="state fork"):
        RemoteDispatchPlanner(manager).attach_remote_state(
            request_id,
            remote_state=forked,
            transport_id=transport_profile.transport_id,
            observed_host_key_fingerprint=watched_fingerprint,
        )

    cancelled, cancelled_fingerprint = control_remote_request(
        envelope=envelope,
        profile=transport_profile,
        known_hosts_path=dispatch_environment["known_hosts"],
        action="cancel",
        ssh_runner=fake_control,
    )
    assert cancelled["state"] == "CANCEL_REQUESTED"
    cancelled_status = RemoteDispatchPlanner(manager).attach_remote_state(
        request_id,
        remote_state=cancelled,
        transport_id=transport_profile.transport_id,
        observed_host_key_fingerprint=cancelled_fingerprint,
    )
    assert cancelled_status["remote_observation"]["remote_state_sequence"] == 2
    assert (
        cancelled_status["remote_observation"]["remote_state"]["state"]
        == "CANCEL_REQUESTED"
    )


def test_remote_inbox_rejects_request_that_already_advanced(
    dispatch_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-REMOTE-06", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=Path(__file__).resolve().parents[1])
    envelope = RemoteDispatchPlanner(manager).plan(
        "AR-REMOTE-06",
        "remote-dispatch-test",
    )
    profile = load_remote_profile(
        dispatch_environment["profile_root"],
        "remote-dispatch-test",
    )
    state_store = RequestStateStore(dispatch_environment["lab_home"] / "target-state")
    inbox = RemoteInbox(dispatch_environment["lab_home"] / "target-inbox", state_store)
    receipt = inbox.accept(
        envelope,
        remote_profile=profile,
        control_plane_version="0.3.0-m3-bootstrap",
    )
    state_store.reserve(
        envelope["request_id"],
        slot_id="gpu-slot-01",
        allowed_slots=("gpu-slot-01",),
    )

    with pytest.raises(LabError, match="already advanced beyond acceptance"):
        inbox.accept(
            envelope,
            remote_profile=profile,
            control_plane_version="0.3.0-m3-bootstrap",
        )
    assert receipt["accepted_state"] == "SUBMITTED"
