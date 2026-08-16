from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.orchestration import (
    OrchestrationManager,
    load_distributed_profile,
)
from lab_control.run_requests import RunRequestManager
from lab_control.runner_protocol import RunnerCoordinator, load_runner_profile
from lab_control.runs import control_plane_identity
from lab_control.workspace import LabError, WorkspaceManager


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, *, title: str) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Orchestration Test")
    _git(path, "config", "user.email", "orchestration-test@example.invalid")
    (path / "README.md").write_text(f"# {title}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _write_runner(
    root: Path,
    *,
    runner_id: str,
    host_id: str,
    target_id: str,
    slot_id: str,
    capability: str,
    resource: str,
) -> None:
    (root / f"{runner_id}.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-profile.v2",
                "runner_id": runner_id,
                "description": f"Temporary {runner_id}",
                "host_id": host_id,
                "target_ids": [target_id],
                "slots": [slot_id],
                "capabilities": [capability],
                "resources": [resource],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def orchestration_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceManager, RunRequestManager, OrchestrationManager, Path]:
    repositories_root = tmp_path / "repositories"
    controller = _repository(repositories_root / "Controller", title="Controller")
    component = _repository(repositories_root / "Component", title="Component")
    project_manifest = tmp_path / "project.json"
    project_manifest.write_text(
        json.dumps(
            {
                "schema": "lab.project.v3",
                "project_id": "distributed-stack",
                "control_repository": "Controller",
                "default_development_host": "developer-a",
                "repositories": {
                    "Controller": {
                        "aliases": [],
                        "checkout_mode": "stable_branch",
                        "default_path": ".",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_CONTROLLER",
                        "stable_branch": "main",
                    },
                    "Component": {
                        "aliases": [],
                        "checkout_mode": "stable_branch",
                        "default_path": "Component",
                        "path_base": "control_repository_parent",
                        "path_environment": "LAB_REPOSITORY_COMPONENT",
                        "stable_branch": "main",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    queue_root = tmp_path / "queue-profiles"
    runner_root = tmp_path / "runner-profiles"
    distributed_root = tmp_path / "distributed-profiles"
    queue_root.mkdir()
    runner_root.mkdir()
    distributed_root.mkdir()
    (queue_root / "distributed-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v3",
                "profile_id": "distributed-check",
                "description": "Aggregate protected command authorization",
                "target_id": "distributed-target",
                "risk_level": "R0",
                "snapshot_modes": ["clean"],
                "execution_class": "compute",
                "execution_topology": "distributed",
                "repositories": ["Controller", "Component"],
                "command_ids": ["upper-command", "lower-command"],
                "capabilities": ["cpu-runtime", "accelerator-runtime"],
            }
        ),
        encoding="utf-8",
    )
    (distributed_root / "two-node-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.distributed-profile.v1",
                "profile_id": "two-node-check",
                "description": "Two generic nodes joined by one versioned service",
                "queue_profile_id": "distributed-check",
                "target_id": "distributed-target",
                "nodes": [
                    {
                        "node_id": "lower-node",
                        "target_id": "compute-target",
                        "repositories": ["Component"],
                        "command_ids": ["lower-command"],
                        "capabilities": ["accelerator-runtime"],
                        "resources": ["accelerator-a"],
                        "produces_services": [
                            {
                                "service_id": "lower-api",
                                "contract_sha256": "a" * 64,
                            }
                        ],
                    },
                    {
                        "node_id": "upper-node",
                        "target_id": "control-target",
                        "repositories": ["Controller"],
                        "command_ids": ["upper-command"],
                        "capabilities": ["cpu-runtime"],
                        "resources": ["runtime-a"],
                        "produces_services": [],
                    },
                ],
                "dependencies": [
                    {
                        "producer_node": "lower-node",
                        "consumer_node": "upper-node",
                        "service_id": "lower-api",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_runner(
        runner_root,
        runner_id="runner-lower",
        host_id="host-b",
        target_id="compute-target",
        slot_id="slot-b1",
        capability="accelerator-runtime",
        resource="accelerator-a",
    )
    _write_runner(
        runner_root,
        runner_id="runner-upper",
        host_id="host-a",
        target_id="control-target",
        slot_id="slot-a1",
        capability="cpu-runtime",
        resource="runtime-a",
    )
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(project_manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(controller))
    monkeypatch.setenv("LAB_REPOSITORY_COMPONENT", str(component))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_QUEUE_PROFILE_ROOT", str(queue_root))
    monkeypatch.setenv("LAB_RUNNER_PROFILE_ROOT", str(runner_root))
    monkeypatch.setenv("LAB_DISTRIBUTED_PROFILE_ROOT", str(distributed_root))
    monkeypatch.setenv(
        "LAB_ORCHESTRATION_STATE_ROOT", str(tmp_path / "orchestration-state")
    )
    monkeypatch.setenv("LAB_REQUEST_STATE_ROOT", str(tmp_path / "request-state"))
    manager = WorkspaceManager(control_repository=Path(__file__).resolve().parents[1])
    requests = RunRequestManager(manager)
    orchestration = OrchestrationManager(
        requests.store,
        distributed_profile_root=distributed_root,
        queue_profile_root=queue_root,
        runner_profile_root=runner_root,
        state_root=tmp_path / "orchestration-state",
    )
    return manager, requests, orchestration, distributed_root


def _request(
    manager: WorkspaceManager,
    requests: RunRequestManager,
    task_id: str,
) -> str:
    manager.create_task(
        manager.plan_task(
            task_id=task_id,
            repository_names=["Controller", "Component"],
            scope_assignments=["Controller=**", "Component=**"],
        )
    )
    submission = requests.submit_clean_profile(
        task_id, "distributed-check", evidence_tier="Development"
    )
    request = submission["request"]
    assert isinstance(request, dict)
    return str(request["request_id"])


def test_composite_reservation_is_all_or_nothing_and_releases_together(
    orchestration_environment: tuple[
        WorkspaceManager, RunRequestManager, OrchestrationManager, Path
    ],
) -> None:
    manager, requests, orchestration, _ = orchestration_environment
    first_request = _request(manager, requests, "T701")
    second_request = _request(manager, requests, "T702")
    observed = datetime(2026, 8, 16, 10, 0, tzinfo=timezone.utc)
    request = requests.store.load_request(first_request)
    assert request["schema"] == "lab.run-request.v4"
    coordinator = RunnerCoordinator(
        requests.store,
        runner_profile_root=orchestration.runner_profile_root,
        queue_profile_root=orchestration.queue_profile_root,
        control_plane=control_plane_identity(manager.control_repository),
    )
    runner_inspection = coordinator.inspect("runner-lower")
    assert runner_inspection["eligible_count"] == 0
    assert "composite orchestrator" in runner_inspection["pending"][0]["reasons"][0]

    plan = orchestration.plan(first_request, "two-node-check", now=observed)
    assert plan["schema"] == "lab.orchestration-plan.v1"
    assignments = {item["node_id"]: item for item in plan["assignments"]}
    assert assignments["lower-node"]["runner_id"] == "runner-lower"
    assert assignments["upper-node"]["runner_id"] == "runner-upper"
    reservation = orchestration.reserve(
        first_request,
        "two-node-check",
        orchestrator_id="controller-a",
        now=observed,
    )
    assert len(reservation["assignments"]) == 2
    assert (
        orchestration.reserve(
            first_request,
            "two-node-check",
            orchestrator_id="controller-a",
            now=observed + timedelta(seconds=1),
        )
        == reservation
    )

    with pytest.raises(LabError, match="cannot atomically reserve all"):
        orchestration.reserve(
            second_request,
            "two-node-check",
            orchestrator_id="controller-a",
            now=observed + timedelta(seconds=2),
        )
    reservation_files = list((orchestration.root / "reservations").glob("*.json"))
    assert len(reservation_files) == 1

    release = orchestration.release(
        first_request,
        str(reservation["reservation_id"]),
        actor="controller-a",
        reason="registered teardown completed",
        now=observed + timedelta(seconds=3),
    )
    assert release["schema"] == "lab.composite-reservation-release.v1"
    replacement = orchestration.reserve(
        second_request,
        "two-node-check",
        orchestrator_id="controller-a",
        now=observed + timedelta(seconds=4),
    )
    assert replacement["request_id"] == second_request


def test_service_receipt_binds_source_runtime_and_ready_barrier(
    orchestration_environment: tuple[
        WorkspaceManager, RunRequestManager, OrchestrationManager, Path
    ],
) -> None:
    manager, requests, orchestration, _ = orchestration_environment
    request_id = _request(manager, requests, "T703")
    observed = datetime(2026, 8, 16, 11, 0, tzinfo=timezone.utc)
    reservation = orchestration.reserve(
        request_id,
        "two-node-check",
        orchestrator_id="controller-a",
        now=observed,
    )
    reservation_id = str(reservation["reservation_id"])

    with pytest.raises(LabError, match="missing services: lower-api"):
        orchestration.ready_barrier(
            request_id,
            reservation_id,
            now=observed + timedelta(seconds=1),
        )
    with pytest.raises(LabError, match="must not embed credentials"):
        orchestration.record_service_ready(
            request_id,
            reservation_id,
            "lower-api",
            runner_id="runner-lower",
            endpoint="http://user:password@127.0.0.1:18080",
            process_id=123,
            process_start_token="boot-1",
            environment_sha256="b" * 64,
            configuration_sha256="c" * 64,
            now=observed + timedelta(seconds=1),
        )
    receipt = orchestration.record_service_ready(
        request_id,
        reservation_id,
        "lower-api",
        runner_id="runner-lower",
        endpoint="http://127.0.0.1:18080",
        process_id=123,
        process_start_token="boot-1",
        environment_sha256="b" * 64,
        configuration_sha256="c" * 64,
        now=observed + timedelta(seconds=2),
    )
    request = requests.store.load_request(request_id)
    component = request["snapshot"]["repositories"]["Component"]
    assert receipt["source"]["Component"] == {
        "commit": component["commit"],
        "tree_oid": component["tree_oid"],
    }
    assert receipt["contract_sha256"] == "a" * 64
    barrier = orchestration.ready_barrier(
        request_id,
        reservation_id,
        now=observed + timedelta(seconds=3),
    )
    assert barrier["schema"] == "lab.ready-barrier.v1"
    assert barrier["services"][0]["service_instance_id"] == receipt["service_instance_id"]
    inspection = orchestration.inspect(
        request_id, now=observed + timedelta(seconds=4)
    )
    assert inspection["ready_barrier"] == barrier

    service_path = (
        orchestration.root / "services" / reservation_id / "lower-api.json"
    )
    tampered = json.loads(service_path.read_text(encoding="utf-8"))
    tampered["contract_sha256"] = "d" * 64
    service_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(LabError, match="producer identity mismatch"):
        orchestration.ready_barrier(
            request_id,
            reservation_id,
            now=observed + timedelta(seconds=5),
        )
    requests.store.cancel(
        request_id,
        actor="controller-a",
        reason="stop before any payload execution",
    )
    with pytest.raises(LabError, match="no longer active for orchestration"):
        orchestration.record_service_ready(
            request_id,
            reservation_id,
            "lower-api",
            runner_id="runner-lower",
            endpoint="http://127.0.0.1:18080",
            process_id=123,
            process_start_token="boot-1",
            environment_sha256="b" * 64,
            configuration_sha256="c" * 64,
            now=observed + timedelta(seconds=6),
        )


def test_profiles_reject_cycles_and_duplicate_resource_ownership(
    orchestration_environment: tuple[
        WorkspaceManager, RunRequestManager, OrchestrationManager, Path
    ],
) -> None:
    manager, requests, orchestration, distributed_root = orchestration_environment
    profile_path = distributed_root / "two-node-check.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["nodes"][1]["produces_services"] = [
        {"service_id": "upper-api", "contract_sha256": "e" * 64}
    ]
    payload["dependencies"].append(
        {
            "producer_node": "upper-node",
            "consumer_node": "lower-node",
            "service_id": "upper-api",
        }
    )
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabError, match="contain a cycle"):
        load_distributed_profile(distributed_root, "two-node-check")

    payload["dependencies"].pop()
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    runner_root = orchestration.runner_profile_root
    duplicate = json.loads((runner_root / "runner-upper.json").read_text())
    duplicate.update(
        {
            "runner_id": "runner-duplicate",
            "host_id": "host-c",
            "target_ids": ["other-target"],
            "slots": ["slot-c1"],
            "resources": ["runtime-a"],
        }
    )
    (runner_root / "runner-duplicate.json").write_text(
        json.dumps(duplicate), encoding="utf-8"
    )
    request_id = _request(manager, requests, "T704")
    with pytest.raises(LabError, match="belongs to multiple runners"):
        orchestration.plan(request_id, "two-node-check")


def test_runner_profile_v1_remains_resource_free_and_v2_exposes_host_resources(
    orchestration_environment: tuple[
        WorkspaceManager, RunRequestManager, OrchestrationManager, Path
    ],
    tmp_path: Path,
) -> None:
    _, _, orchestration, _ = orchestration_environment
    current = load_runner_profile(orchestration.runner_profile_root, "runner-lower")
    assert current.host_id == "host-b"
    assert current.resources == ("accelerator-a",)

    legacy_root = tmp_path / "legacy-runner"
    legacy_root.mkdir()
    (legacy_root / "runner-old.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-profile.v1",
                "runner_id": "runner-old",
                "description": "Legacy runner",
                "target_ids": ["legacy-target"],
                "slots": ["legacy-slot"],
                "capabilities": ["python"],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )
    legacy = load_runner_profile(legacy_root, "runner-old")
    assert legacy.host_id == "runner-old"
    assert legacy.resources == ()


def test_cli_exposes_plan_reserve_and_inspection_without_execution(
    orchestration_environment: tuple[
        WorkspaceManager, RunRequestManager, OrchestrationManager, Path
    ],
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, requests, _, _ = orchestration_environment
    request_id = _request(manager, requests, "T705")

    assert (
        main(
            [
                "orchestrate",
                "plan",
                request_id,
                "two-node-check",
                "--json",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == "lab.orchestration-plan.v1"
    assert len(plan["assignments"]) == 2

    assert (
        main(
            [
                "orchestrate",
                "reserve",
                request_id,
                "two-node-check",
                "--orchestrator",
                "controller-a",
                "--json",
            ]
        )
        == 0
    )
    reservation = json.loads(capsys.readouterr().out)
    assert reservation["schema"] == "lab.composite-reservation.v1"
    assert main(["orchestrate", "inspect", request_id, "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["active_reservations"][0] == reservation


def test_protected_profile_drift_invalidates_service_readiness(
    orchestration_environment: tuple[
        WorkspaceManager, RunRequestManager, OrchestrationManager, Path
    ],
) -> None:
    manager, requests, orchestration, _ = orchestration_environment
    request_id = _request(manager, requests, "T706")
    observed = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
    reservation = orchestration.reserve(
        request_id,
        "two-node-check",
        orchestrator_id="controller-a",
        now=observed,
    )
    profile_path = orchestration.queue_profile_root / "distributed-check.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["description"] = "drifted after reservation"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(LabError, match="Queue Profile changed after reservation"):
        orchestration.record_service_ready(
            request_id,
            str(reservation["reservation_id"]),
            "lower-api",
            runner_id="runner-lower",
            endpoint="http://127.0.0.1:18080",
            process_id=123,
            process_start_token="boot-1",
            environment_sha256="b" * 64,
            configuration_sha256="c" * 64,
            now=observed + timedelta(seconds=1),
        )
