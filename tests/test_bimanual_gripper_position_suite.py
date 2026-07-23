from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import identify_bimanual_gripper_position as suite


def test_complete_suite_freezes_fit_validation_and_preemption_programs() -> None:
    phases = suite.phase_specs()

    assert [phase.role for phase in phases] == [
        "fit",
        "validation",
        "diagnostic_only",
    ]
    assert phases[0].targets[:4] == (1.0, 0.0, 1.0, 0.0)
    assert {0.51, 0.52, 0.55}.issubset(phases[0].targets)
    assert phases[0].repeats == 3
    assert phases[1].repeats == 2
    assert phases[2].dwell_sec == 0.5
    assert all(phase.targets[-1] == 0.0 for phase in phases)


def test_plan_is_bimanual_and_does_not_call_hardware(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        suite.collector,
        "run",
        lambda args: pytest.fail("plan must not call collector.run"),
    )
    args = suite.build_parser().parse_args(["plan"])

    assert suite.plan(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["mode"] == "plan_no_hardware"
    assert payload["side_order"] == ["left", "right"]
    assert payload["source_conditions"]["force"] == 0.2
    assert payload["source_conditions"]["continuous_grasp"] is True
    assert payload["target_segments_by_role_both_sides"] == {
        "fit": 120,
        "validation": 44,
        "diagnostic_only": 36,
    }
    assert payload["total_target_segments_both_sides"] == 200
    assert payload["analysis"]["canonical_profile_update"] is False


def test_run_requires_suite_acknowledgement_before_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        suite.collector,
        "run",
        lambda args: pytest.fail("missing acknowledgement must fail before hardware"),
    )
    args = suite.build_parser().parse_args(["run"])

    with pytest.raises(RuntimeError, match="requires --apply"):
        suite.run_suite(args)


def test_child_collector_args_match_main_position_conditions(tmp_path: Path) -> None:
    args = suite.build_parser().parse_args(
        [
            "run",
            "--apply",
            "--confirmation-token",
            suite.SUITE_CONFIRMATION,
        ]
    )
    phase = suite.phase_specs()[0]
    confirmation = {"suite_id": "test_suite", "side": "left"}
    endpoint = suite.collector.Endpoint(
        "left", "/dev/test-left", 7, 57600, 0.08, "test"
    )

    child = suite._collector_args(
        args,
        side="left",
        endpoint=endpoint,
        phase=phase,
        output_root=tmp_path,
        suite_confirmation=confirmation,
    )
    specs = suite.collector.build_trial_specs(child)

    assert child.command_mode == "runtime-equivalent"
    assert child.port == endpoint.port
    assert child.gripper_id == endpoint.gripper_id
    assert child.baudrate == endpoint.baudrate
    assert child.timeout == endpoint.timeout
    assert child.sample_hz == 50.0
    assert child.max_temperature_c == 55.0
    assert child._suite_operator_confirmation == confirmation
    assert len(specs) == 3
    assert all(spec.force == 0.2 for spec in specs)
    assert all(spec.continuous_grasp is True for spec in specs)
    assert all(spec.motion_targets == phase.targets for spec in specs)


def test_endpoint_preflight_rejects_same_serial_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        suite.collector,
        "resolve_endpoint",
        lambda args: (
            suite.collector.Endpoint(
                args.side, "/dev/test-shared", 1, 115200, 0.05, "test"
            ),
            {},
        ),
    )
    monkeypatch.setattr(suite.collector, "validate_endpoint", lambda endpoint: None)
    args = suite.build_parser().parse_args(["plan"])

    with pytest.raises(RuntimeError, match="same serial device"):
        suite._resolved_endpoints(args)


def test_suite_preflight_rejects_unfinished_recovery_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints = {
        "left": suite.collector.Endpoint(
            "left", "/dev/test-left", 1, 115200, 0.05, "test"
        ),
        "right": suite.collector.Endpoint(
            "right", "/dev/test-right", 1, 115200, 0.05, "test"
        ),
    }
    unfinished = tmp_path / "unfinished" / "recovery.json"
    monkeypatch.setattr(suite, "_resolved_endpoints", lambda args: endpoints)
    monkeypatch.setattr(
        suite.collector,
        "_unfinished_journals",
        lambda root, endpoint: [unfinished] if endpoint.side == "left" else [],
    )
    monkeypatch.setattr(suite.collector, "find_device_users", lambda port: [])
    args = suite.build_parser().parse_args(["plan", "--output-root", str(tmp_path)])

    with pytest.raises(suite.collector.RecoveryError, match="must be restored"):
        suite._preflight_suite(args, suite_root=tmp_path / "suite")


def test_fit_coverage_requires_both_directions_and_held_out_groups() -> None:
    coverage = suite._fit_coverage(
        {
            "fit_by_side": {
                "left": {"close": {"status": "identified"}},
                "right": {
                    "close": {"status": "provisional"},
                    "open": {"status": "identified"},
                },
            },
            "validation": {"groups": {"left.close": {}}},
        }
    )

    assert coverage["status"] == "incomplete"
    assert coverage["missing_fit_groups"] == ["left.open"]
    assert coverage["provisional_fit_groups"] == ["right.close"]
    assert coverage["missing_validation_groups"] == [
        "left.open",
        "right.close",
        "right.open",
    ]


def test_orchestrator_writes_complete_manifest_and_analysis_without_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(suite.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        suite,
        "_confirm_side",
        lambda side, suite_id: {"suite_id": suite_id, "side": side},
    )
    monkeypatch.setattr(
        suite,
        "_preflight_suite",
        lambda args, suite_root: {
            "left": suite.collector.Endpoint(
                "left", "/dev/test-left", 1, 115200, 0.05, "test"
            ),
            "right": suite.collector.Endpoint(
                "right", "/dev/test-right", 1, 115200, 0.05, "test"
            ),
        },
    )

    def fake_run_phase(
        args: argparse.Namespace,
        *,
        suite_root: Path,
        side: str,
        endpoint: suite.collector.Endpoint,
        phase: suite.PhaseSpec,
        suite_confirmation,
    ) -> Path:
        calls.append((side, phase.name))
        assert endpoint.side == side
        run_directory = suite_root / "fake" / side / phase.name
        run_directory.mkdir(parents=True)
        (run_directory / "manifest.json").write_text("{}\n", encoding="utf-8")
        (run_directory / "summary.json").write_text("{}\n", encoding="utf-8")
        return run_directory

    monkeypatch.setattr(suite, "_run_phase", fake_run_phase)
    monkeypatch.setattr(
        suite.fitter,
        "build_fit_artifact",
        lambda *args, **kwargs: {
            "payload_sha256": "a" * 64,
            "fit_by_side": {
                side: {
                    direction: {"status": "identified"}
                    for direction in ("close", "open")
                }
                for side in ("left", "right")
            },
            "validation": {
                "status": "evaluated",
                "groups": {
                    f"{side}.{direction}": {}
                    for side in ("left", "right")
                    for direction in ("close", "open")
                },
            },
            "fidelity_claims": {
                "free_motion_position_response": "provisional",
                "contact_force_newtons": "unavailable",
            },
        },
    )
    args = suite.build_parser().parse_args(
        [
            "run",
            "--output-root",
            str(tmp_path),
            "--apply",
            "--confirmation-token",
            suite.SUITE_CONFIRMATION,
        ]
    )

    assert suite.run_suite(args) == 0

    suite_roots = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(suite_roots) == 1
    manifest = json.loads(
        (suite_roots[0] / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert calls == [
        (side, phase.name)
        for side in ("left", "right")
        for phase in suite.phase_specs()
    ]
    assert manifest["normal_completion"] is True
    assert manifest["status"] == "complete"
    assert manifest["resolved_endpoints"]["left"]["port"] == "/dev/test-left"
    assert len(manifest["environment"]["suite_script_sha256"]) == 64
    assert manifest["analysis"]["coverage"]["status"] == "complete"
    assert manifest["analysis"]["canonical_profile_updated"] is False
    assert manifest["analysis"]["fidelity_claims"]["contact_force_newtons"] == "unavailable"


def test_resume_reuses_verified_phase_and_runs_only_pending_phases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_id = "dmtac_bimanual_position_resume_test"
    suite_root = tmp_path / suite_id
    completed = suite_root / "runs" / "left" / "fit_steps" / "completed"
    completed.mkdir(parents=True)
    child_manifest_path = completed / "manifest.json"
    child_summary_path = completed / "summary.json"
    child_manifest_path.write_text(
        json.dumps({"normal_completion": True, "settings_restored": True}) + "\n",
        encoding="utf-8",
    )
    child_summary_path.write_text("{}\n", encoding="utf-8")
    endpoints = {
        "left": suite.collector.Endpoint(
            "left", "/dev/test-left", 1, 115200, 0.05, "test"
        ),
        "right": suite.collector.Endpoint(
            "right", "/dev/test-right", 1, 115200, 0.05, "test"
        ),
    }
    base_args = suite.build_parser().parse_args(["plan"])
    manifest = {
        "schema": suite.SUITE_SCHEMA,
        "suite_id": suite_id,
        "status": "failed",
        "normal_completion": False,
        "suite_root": str(suite_root),
        "plan": suite._normalized_json(suite._plan_payload(base_args)),
        "resolved_endpoints": {
            side: suite.asdict(endpoint) for side, endpoint in endpoints.items()
        },
        "environment": {},
        "side_confirmations": {"left": {"side": "left", "old": True}},
        "runs": {
            "left": {
                "fit_steps": {
                    "run_directory": str(completed),
                    "manifest_sha256": suite._sha256_file(child_manifest_path),
                    "summary_sha256": suite._sha256_file(child_summary_path),
                }
            },
            "right": {},
        },
        "analysis": None,
        "error": "transient preflight status",
    }
    manifest["plan"]["source_conditions"].pop("phase_handoff_preflight")
    (suite_root / "suite_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(suite.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(suite, "_preflight_suite", lambda args, suite_root: endpoints)
    monkeypatch.setattr(
        suite,
        "_confirm_side",
        lambda side, suite_id: {"suite_id": suite_id, "side": side, "new": True},
    )

    def fake_run_phase(
        args: argparse.Namespace,
        *,
        suite_root: Path,
        side: str,
        endpoint: suite.collector.Endpoint,
        phase: suite.PhaseSpec,
        suite_confirmation,
    ) -> Path:
        calls.append((side, phase.name))
        run_directory = suite_root / "new" / side / phase.name
        run_directory.mkdir(parents=True)
        (run_directory / "manifest.json").write_text("{}\n", encoding="utf-8")
        (run_directory / "summary.json").write_text("{}\n", encoding="utf-8")
        return run_directory

    monkeypatch.setattr(suite, "_run_phase", fake_run_phase)
    monkeypatch.setattr(
        suite.fitter,
        "build_fit_artifact",
        lambda *args, **kwargs: {
            "payload_sha256": "b" * 64,
            "fit_by_side": {
                side: {
                    direction: {"status": "identified"}
                    for direction in ("close", "open")
                }
                for side in ("left", "right")
            },
            "validation": {
                "status": "evaluated",
                "groups": {
                    f"{side}.{direction}": {}
                    for side in ("left", "right")
                    for direction in ("close", "open")
                },
            },
            "fidelity_claims": {"contact_force_newtons": "unavailable"},
        },
    )
    args = suite.build_parser().parse_args(
        [
            "run",
            "--resume-suite",
            str(suite_root),
            "--apply",
            "--confirmation-token",
            suite.SUITE_CONFIRMATION,
        ]
    )

    assert suite.run_suite(args) == 0
    assert ("left", "fit_steps") not in calls
    assert calls == [
        ("left", "held_out_steps"),
        ("left", "fast_preemption"),
        ("right", "fit_steps"),
        ("right", "held_out_steps"),
        ("right", "fast_preemption"),
    ]
    resumed = json.loads(
        (suite_root / "suite_manifest.json").read_text(encoding="utf-8")
    )
    assert resumed["status"] == "complete"
    assert len(resumed["resumptions"]) == 1
    assert resumed["side_confirmation_history"] == [{"side": "left", "old": True}]
