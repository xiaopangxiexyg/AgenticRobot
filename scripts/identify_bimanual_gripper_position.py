#!/usr/bin/env python3
"""Collect and fit the complete bimanual DM-Tac position-only test suite.

One interactive invocation runs three free-motion phases on the left gripper,
then the same phases on the right gripper, and finally creates a position-only
fit artifact with independent held-out evaluation. The RM65 arms, tactile
sensors, teleoperation runtime, replay, and policy deployment are never
initialized by this script.

The suite intentionally asks for one strong confirmation per side. Individual
free-motion phases reuse that side-scoped acknowledgement, while every phase
retains the standalone collector's serial exclusivity, guardian, recovery
journal, temperature limit, trace, summary, and evidence manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.DMTAC_gripper.examples import identify_gripper_actuator as collector
from sim.rm65.real2sim import gripper_position_fit as fitter


SUITE_SCHEMA = "dexteleop.hardware.dmtac_bimanual_position_suite.v1"
SUITE_CONFIRMATION = "BIMANUAL_GRIPPER_POSITION_SUITE"
SIDE_CONFIRMATIONS = {
    "left": "LEFT_GRIPPER_PATH_CLEAR",
    "right": "RIGHT_GRIPPER_PATH_CLEAR",
}
SIDES = ("left", "right")


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    role: str
    description: str
    targets: tuple[float, ...]
    dwell_sec: float
    repeats: int

    @property
    def estimated_motion_sec(self) -> float:
        return len(self.targets) * self.dwell_sec * self.repeats


def phase_specs() -> tuple[PhaseSpec, ...]:
    """Frozen position-only probe program used identically on both sides."""

    return (
        PhaseSpec(
            name="fit_steps",
            role="fit",
            description=(
                "full strokes, partial positions, small steps, hysteresis, and settled reversals"
            ),
            targets=(
                1.0,
                0.0,
                1.0,
                0.0,
                0.25,
                0.50,
                0.75,
                1.0,
                0.75,
                0.50,
                0.25,
                0.0,
                0.50,
                0.51,
                0.50,
                0.52,
                0.50,
                0.55,
                0.50,
                0.0,
            ),
            dwell_sec=1.1,
            repeats=3,
        ),
        PhaseSpec(
            name="held_out_steps",
            role="validation",
            description="independent unseen amplitudes and direction changes",
            targets=(0.90, 0.10, 0.70, 0.30, 1.0, 0.0, 0.40, 0.60, 0.20, 0.80, 0.0),
            dwell_sec=1.0,
            repeats=2,
        ),
        PhaseSpec(
            name="fast_preemption",
            role="diagnostic_only",
            description="short-dwell target replacement and reversal evidence",
            targets=(0.80, 0.20, 1.0, 0.0, 0.60, 0.40, 0.90, 0.10, 0.0),
            dwell_sec=0.5,
            repeats=2,
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _plan_payload(args: argparse.Namespace) -> dict[str, Any]:
    phases = phase_specs()
    estimated_motion = sum(phase.estimated_motion_sec for phase in phases) * len(SIDES)
    segments_by_role = {
        role: sum(
            len(phase.targets) * phase.repeats
            for phase in phases
            if phase.role == role
        )
        * len(SIDES)
        for role in ("fit", "validation", "diagnostic_only")
    }
    return {
        "schema": SUITE_SCHEMA,
        "mode": "plan_no_hardware",
        "side_order": list(SIDES),
        "source_conditions": {
            "speed": 1.0,
            "force": float(args.force),
            "continuous_grasp": True,
            "command_mode": "runtime-equivalent",
            "host_position_hold": "one-shot_per_target",
            "phase_handoff_preflight": (
                "read_only_status_idle_or_stable_safe_open_low_current"
            ),
            "sample_hz": float(args.sample_hz),
            "max_temperature_c": float(args.max_temperature_c),
        },
        "phases_per_side": [
            {
                **asdict(phase),
                "target_count": len(phase.targets),
                "estimated_motion_sec": phase.estimated_motion_sec,
            }
            for phase in phases
        ],
        "target_segments_by_role_both_sides": segments_by_role,
        "total_target_segments_both_sides": sum(segments_by_role.values()),
        "estimated_motion_sec_both_sides": estimated_motion,
        "estimated_wall_time_note": (
            "Add opening, cooldown, configuration, serial reads, two side confirmations, "
            "and offline fitting; Ctrl-C aborts the active phase."
        ),
        "analysis": {
            "fit_phases": [phase.name for phase in phases if phase.role == "fit"],
            "validation_phases": [
                phase.name for phase in phases if phase.role == "validation"
            ],
            "diagnostic_only_phases": [
                phase.name for phase in phases if phase.role == "diagnostic_only"
            ],
            "contact_force_claim": "unavailable_without_external_force_gauge",
            "canonical_profile_update": False,
        },
    }


def plan(args: argparse.Namespace) -> int:
    print(json.dumps(_plan_payload(args), ensure_ascii=False, indent=2))
    return 0


def _confirm_side(side: str, *, suite_id: str) -> dict[str, Any]:
    token = SIDE_CONFIRMATIONS[side]
    print(
        "\n" + "=" * 72 + "\n"
        f"Next side: {side.upper()} gripper, suite={suite_id}\n"
        "Confirm RM65 arm/replay/teleop/policy are stopped; both hands and all objects are "
        "outside the full jaw path; only this gripper may move; its serial assignment is "
        "correct; emergency power removal is reachable.\n"
        f"Type exactly {token} to run all three {side} free-motion phases."
    )
    observed = input("confirmation: ").strip()
    if observed != token:
        raise RuntimeError(f"{side} side confirmation did not match {token}")
    return {
        "suite_id": suite_id,
        "side": side,
        "token": token,
        "confirmed_wall_time_ns": time.time_ns(),
        "scope": "all_position_only_phases_for_one_side",
    }


def _resolved_endpoints(args: argparse.Namespace) -> dict[str, collector.Endpoint]:
    endpoints: dict[str, collector.Endpoint] = {}
    for side in SIDES:
        inspect_args = collector.build_parser().parse_args(
            ["inspect", "--side", side, "--config", str(args.config)]
        )
        endpoint, _config = collector.resolve_endpoint(inspect_args)
        collector.validate_endpoint(endpoint)
        endpoints[side] = endpoint
    real_ports = {
        side: os.path.realpath(endpoint.port)
        for side, endpoint in endpoints.items()
    }
    if len(set(real_ports.values())) != len(SIDES):
        raise RuntimeError(f"left/right resolve to the same serial device: {real_ports}")
    return endpoints


def _preflight_suite(
    args: argparse.Namespace,
    *,
    suite_root: Path,
) -> dict[str, collector.Endpoint]:
    """Run read-only cross-phase checks before either side may move."""

    endpoints = _resolved_endpoints(args)
    journal_roots = (
        PROJECT_ROOT / "logs" / "gripper_actuator_identification",
        args.output_root.expanduser().resolve(),
        suite_root,
    )
    unfinished: list[Path] = []
    occupied: dict[str, list[dict[str, Any]]] = {}
    for side, endpoint in endpoints.items():
        for root in journal_roots:
            if root.exists():
                unfinished.extend(collector._unfinished_journals(root, endpoint))
        users = collector.find_device_users(endpoint.port)
        if users:
            occupied[side] = users
    unique_unfinished = sorted({path.resolve() for path in unfinished})
    if unique_unfinished:
        raise collector.RecoveryError(
            "unfinished gripper recovery journal(s) must be restored before the suite: "
            + ", ".join(str(path) for path in unique_unfinished)
        )
    if occupied:
        raise RuntimeError(f"gripper serial device(s) already in use: {occupied}")
    return endpoints


def _collector_args(
    args: argparse.Namespace,
    *,
    side: str,
    endpoint: collector.Endpoint,
    phase: PhaseSpec,
    output_root: Path,
    suite_confirmation: Mapping[str, Any],
) -> argparse.Namespace:
    argv = [
        "run",
        "--side",
        side,
        "--config",
        str(args.config),
        "--port",
        endpoint.port,
        "--gripper-id",
        str(endpoint.gripper_id),
        "--baudrate",
        str(endpoint.baudrate),
        "--timeout",
        f"{endpoint.timeout:.12g}",
        "--protocol",
        "free",
        "--force-levels",
        f"{float(args.force):.12g}",
        "--free-continuous-grasp",
        "--command-mode",
        "runtime-equivalent",
        "--target-sequence",
        ",".join(f"{target:.12g}" for target in phase.targets),
        "--free-duration-sec",
        f"{phase.dwell_sec:.12g}",
        "--sample-hz",
        f"{float(args.sample_hz):.12g}",
        "--repeats",
        str(phase.repeats),
        "--cooldown-sec",
        f"{float(args.cooldown_sec):.12g}",
        "--max-temperature-c",
        f"{float(args.max_temperature_c):.12g}",
        "--motion-timeout-sec",
        f"{float(args.motion_timeout_sec):.12g}",
        "--queue-size",
        str(int(args.queue_size)),
        "--output-root",
        str(output_root),
        "--apply",
        "--confirmation-token",
        collector.MOTION_CONFIRMATION,
    ]
    child_args = collector.build_parser().parse_args(argv)
    setattr(child_args, "_suite_operator_confirmation", dict(suite_confirmation))
    return child_args


def _new_run_directory(output_root: Path, before: set[Path]) -> Path:
    candidates = {
        path.resolve()
        for path in output_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    } - before
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one completed run under {output_root}, got {sorted(candidates)}"
        )
    return next(iter(candidates))


def _run_phase(
    args: argparse.Namespace,
    *,
    suite_root: Path,
    side: str,
    endpoint: collector.Endpoint,
    phase: PhaseSpec,
    suite_confirmation: Mapping[str, Any],
) -> Path:
    output_root = suite_root / "runs" / side / phase.name
    output_root.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in output_root.iterdir() if path.is_dir()}
    print(
        f"\nSTART side={side} phase={phase.name} role={phase.role} "
        f"targets={len(phase.targets)} repeats={phase.repeats}",
        flush=True,
    )
    result = collector.run(
        _collector_args(
            args,
            side=side,
            endpoint=endpoint,
            phase=phase,
            output_root=output_root,
            suite_confirmation=suite_confirmation,
        )
    )
    if result != 0:
        raise RuntimeError(f"collector returned {result} for {side}/{phase.name}")
    run_directory = _new_run_directory(output_root, before)
    print(f"PHASE_COMPLETE side={side} phase={phase.name} output={run_directory}")
    return run_directory


def _phase_record(phase: PhaseSpec, run_directory: Path) -> dict[str, Any]:
    manifest_path = run_directory / "manifest.json"
    summary_path = run_directory / "summary.json"
    return {
        "phase": asdict(phase),
        "run_directory": str(run_directory),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
    }


def _fit_coverage(artifact: Mapping[str, Any]) -> dict[str, Any]:
    expected = [f"{side}.{direction}" for side in SIDES for direction in ("close", "open")]
    fit_by_side = artifact.get("fit_by_side", {})
    validation_groups = artifact.get("validation", {}).get("groups", {})
    missing_fit: list[str] = []
    provisional_fit: list[str] = []
    missing_validation: list[str] = []
    for side in SIDES:
        directions = fit_by_side.get(side, {})
        for direction in ("close", "open"):
            key = f"{side}.{direction}"
            result = directions.get(direction)
            if not isinstance(result, Mapping):
                missing_fit.append(key)
            elif result.get("status") != "identified":
                provisional_fit.append(key)
            if key not in validation_groups:
                missing_validation.append(key)
    complete = not (missing_fit or provisional_fit or missing_validation)
    return {
        "status": "complete" if complete else "incomplete",
        "expected_groups": expected,
        "missing_fit_groups": missing_fit,
        "provisional_fit_groups": provisional_fit,
        "missing_validation_groups": missing_validation,
    }


def _normalized_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _validate_resume_plan(manifest: Mapping[str, Any], plan_payload: Mapping[str, Any]) -> None:
    previous = manifest.get("plan")
    if not isinstance(previous, Mapping):
        raise ValueError("resume manifest has no suite plan")
    if previous.get("phases_per_side") != _normalized_json(
        plan_payload["phases_per_side"]
    ):
        raise ValueError("resume suite phase program differs from the current script")
    previous_conditions = previous.get("source_conditions", {})
    current_conditions = plan_payload["source_conditions"]
    required_conditions = (
        "speed",
        "force",
        "continuous_grasp",
        "command_mode",
        "host_position_hold",
        "sample_hz",
        "max_temperature_c",
    )
    mismatches = {
        key: (previous_conditions.get(key), current_conditions.get(key))
        for key in required_conditions
        if previous_conditions.get(key) != current_conditions.get(key)
    }
    if mismatches:
        raise ValueError(f"resume suite source conditions differ: {mismatches}")


def _existing_run_directories(
    manifest: Mapping[str, Any],
    *,
    suite_root: Path,
) -> dict[str, dict[str, Path]]:
    run_directories: dict[str, dict[str, Path]] = {side: {} for side in SIDES}
    runs = manifest.get("runs", {})
    for side in SIDES:
        side_records = runs.get(side, {}) if isinstance(runs, Mapping) else {}
        for phase in phase_specs():
            record = side_records.get(phase.name)
            if not isinstance(record, Mapping):
                continue
            run_directory = Path(str(record.get("run_directory", ""))).resolve()
            if not run_directory.is_relative_to(suite_root):
                raise ValueError(f"resume run escapes suite root: {run_directory}")
            manifest_path = run_directory / "manifest.json"
            summary_path = run_directory / "summary.json"
            if not manifest_path.is_file() or not summary_path.is_file():
                raise ValueError(f"resume run evidence is incomplete: {run_directory}")
            if _sha256_file(manifest_path) != record.get("manifest_sha256"):
                raise ValueError(f"resume run manifest hash mismatch: {manifest_path}")
            if _sha256_file(summary_path) != record.get("summary_sha256"):
                raise ValueError(f"resume run summary hash mismatch: {summary_path}")
            child_manifest = _load_json(manifest_path)
            if not child_manifest.get("normal_completion") or not child_manifest.get(
                "settings_restored"
            ):
                raise ValueError(f"resume run was not safely completed: {run_directory}")
            run_directories[side][phase.name] = run_directory
    return run_directories


def _start_or_resume_suite(
    args: argparse.Namespace,
) -> tuple[str, Path, Path, dict[str, Any], dict[str, Any], bool]:
    plan_payload = _plan_payload(args)
    if args.resume_suite is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suite_id = f"dmtac_bimanual_position_{timestamp}"
        suite_root = args.output_root.expanduser().resolve() / suite_id
        suite_root.mkdir(parents=True, exist_ok=False)
        manifest_path = suite_root / "suite_manifest.json"
        plan_path = suite_root / "suite_plan.json"
        _atomic_write_json(plan_path, plan_payload)
        manifest: dict[str, Any] = {
            "schema": SUITE_SCHEMA,
            "suite_id": suite_id,
            "status": "running",
            "normal_completion": False,
            "started_wall_time_ns": time.time_ns(),
            "suite_root": str(suite_root),
            "plan": plan_payload,
            "resolved_endpoints": None,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "suite_script_sha256": _sha256_file(Path(__file__).resolve()),
                "collector_script_sha256": _sha256_file(
                    Path(collector.__file__).resolve()
                ),
                "fitter_script_sha256": _sha256_file(Path(fitter.__file__).resolve()),
                "suite_plan_sha256": _sha256_file(plan_path),
            },
            "side_confirmations": {},
            "side_confirmation_history": [],
            "runs": {side: {} for side in SIDES},
            "resumptions": [],
            "analysis": None,
            "error": None,
        }
        _atomic_write_json(manifest_path, manifest)
        return suite_id, suite_root, manifest_path, plan_payload, manifest, False

    suite_root = args.resume_suite.expanduser().resolve()
    manifest_path = suite_root / "suite_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"resume suite manifest does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != SUITE_SCHEMA:
        raise ValueError("resume suite has an unsupported schema")
    if manifest.get("normal_completion") or manifest.get("status") == "complete":
        raise ValueError("completed suite must not be resumed")
    if manifest.get("status") not in {"failed", "running"}:
        raise ValueError(f"suite status cannot be resumed: {manifest.get('status')!r}")
    _validate_resume_plan(manifest, plan_payload)
    suite_id = str(manifest.get("suite_id", "")).strip()
    if not suite_id or suite_root.name != suite_id:
        raise ValueError("resume suite id does not match its directory")
    manifest.setdefault("resumptions", []).append(
        {
            "resumed_wall_time_ns": time.time_ns(),
            "previous_status": manifest.get("status"),
            "previous_error": manifest.get("error"),
            "suite_script_sha256": _sha256_file(Path(__file__).resolve()),
            "collector_script_sha256": _sha256_file(Path(collector.__file__).resolve()),
            "fitter_script_sha256": _sha256_file(Path(fitter.__file__).resolve()),
        }
    )
    manifest["status"] = "running"
    manifest["normal_completion"] = False
    manifest["error"] = None
    manifest["analysis"] = None
    manifest.setdefault("side_confirmation_history", [])
    _atomic_write_json(manifest_path, manifest)
    return suite_id, suite_root, manifest_path, plan_payload, manifest, True


def run_suite(args: argparse.Namespace) -> int:
    if not args.apply or args.confirmation_token != SUITE_CONFIRMATION:
        raise RuntimeError(
            f"run requires --apply --confirmation-token {SUITE_CONFIRMATION}"
        )
    if not sys.stdin.isatty():
        raise RuntimeError("bimanual gripper motion suite requires an interactive terminal")
    if not 0.0 <= float(args.force) <= 0.5:
        raise ValueError("--force must be within [0,0.5] for this suite")
    if not 10.0 <= float(args.sample_hz) <= 100.0:
        raise ValueError("--sample-hz must be within [10,100]")
    suite_id, suite_root, manifest_path, plan_payload, manifest, resumed = (
        _start_or_resume_suite(args)
    )
    try:
        endpoints = _preflight_suite(args, suite_root=suite_root)
        if resumed:
            recorded_endpoints = manifest.get("resolved_endpoints")
            current_endpoints = {
                side: asdict(endpoint) for side, endpoint in endpoints.items()
            }
            if recorded_endpoints != current_endpoints:
                raise RuntimeError(
                    "resolved endpoints changed since the suite started: "
                    f"recorded={recorded_endpoints}, current={current_endpoints}. "
                    "Do not reuse side-labelled phases across a changed assignment; "
                    "start a new suite with the latest config."
                )
        manifest["resolved_endpoints"] = {
            side: asdict(endpoint) for side, endpoint in endpoints.items()
        }
        manifest["environment"]["endpoint_source_sha256"] = {
            side: (
                _sha256_file(Path(endpoint.source))
                if Path(endpoint.source).is_file()
                else None
            )
            for side, endpoint in endpoints.items()
        }
        _atomic_write_json(manifest_path, manifest)
        run_directories = _existing_run_directories(
            manifest,
            suite_root=suite_root,
        )
        for side in SIDES:
            if resumed:
                for phase in phase_specs():
                    if phase.name in run_directories[side]:
                        print(
                            f"RESUME_REUSE side={side} phase={phase.name} "
                            f"output={run_directories[side][phase.name]}",
                            flush=True,
                        )
            pending_phases = [
                phase
                for phase in phase_specs()
                if phase.name not in run_directories[side]
            ]
            if not pending_phases:
                print(f"RESUME_SKIP side={side} all phases already complete", flush=True)
                continue
            confirmation = _confirm_side(side, suite_id=suite_id)
            previous_confirmation = manifest["side_confirmations"].get(side)
            if previous_confirmation is not None:
                manifest["side_confirmation_history"].append(previous_confirmation)
            manifest["side_confirmations"][side] = confirmation
            _atomic_write_json(manifest_path, manifest)
            for phase in pending_phases:
                run_directory = _run_phase(
                    args,
                    suite_root=suite_root,
                    side=side,
                    endpoint=endpoints[side],
                    phase=phase,
                    suite_confirmation=confirmation,
                )
                run_directories[side][phase.name] = run_directory
                manifest["runs"][side][phase.name] = _phase_record(
                    phase,
                    run_directory,
                )
                _atomic_write_json(manifest_path, manifest)

        fit_inputs = [
            run_directories[side][phase.name]
            for side in SIDES
            for phase in phase_specs()
            if phase.role == "fit"
        ]
        validation_inputs = [
            run_directories[side][phase.name]
            for side in SIDES
            for phase in phase_specs()
            if phase.role == "validation"
        ]
        artifact = fitter.build_fit_artifact(
            fit_inputs,
            validation_sources=validation_inputs,
            expected_speed=1.0,
            expected_force=float(args.force),
            expected_continuous=True,
            expected_command_mode="runtime-equivalent",
        )
        artifact_path = suite_root / "analysis" / "dmtac_gripper_position_fit.json"
        _atomic_write_json(artifact_path, artifact)
        coverage = _fit_coverage(artifact)
        manifest["analysis"] = {
            "artifact_path": str(artifact_path),
            "artifact_sha256": _sha256_file(artifact_path),
            "payload_sha256": artifact["payload_sha256"],
            "fit_by_side": artifact["fit_by_side"],
            "validation": artifact["validation"],
            "coverage": coverage,
            "fidelity_claims": artifact["fidelity_claims"],
            "diagnostic_only_runs": {
                side: str(run_directories[side]["fast_preemption"])
                for side in SIDES
            },
            "canonical_profile_updated": False,
        }
        _atomic_write_json(manifest_path, manifest)
        if coverage["status"] != "complete":
            raise RuntimeError(f"position fit coverage is incomplete: {coverage}")
        manifest["status"] = "complete"
        manifest["normal_completion"] = True
        manifest["finished_wall_time_ns"] = time.time_ns()
        _atomic_write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["normal_completion"] = False
        manifest["finished_wall_time_ns"] = time.time_ns()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write_json(manifest_path, manifest)
        raise
    print(
        f"SUITE_COMPLETE normal_completion=true manifest={manifest_path} "
        f"artifact={manifest['analysis']['artifact_path']}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (("plan", plan), ("run", run_suite)):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", default="latest")
        subparser.add_argument("--force", type=float, default=0.2)
        subparser.add_argument("--sample-hz", type=float, default=50.0)
        subparser.add_argument("--cooldown-sec", type=float, default=2.0)
        subparser.add_argument("--max-temperature-c", type=float, default=55.0)
        subparser.add_argument("--motion-timeout-sec", type=float, default=5.0)
        subparser.add_argument("--queue-size", type=int, default=16384)
        subparser.add_argument(
            "--output-root",
            type=Path,
            default=PROJECT_ROOT / "logs" / "gripper_position_suite",
        )
        if name == "run":
            subparser.add_argument("--apply", action="store_true")
            subparser.add_argument("--confirmation-token", default="")
            subparser.add_argument(
                "--resume-suite",
                type=Path,
                default=None,
                help="Resume a failed suite after verifying completed phase evidence.",
            )
        subparser.set_defaults(func=function)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("ABORTED: operator interrupted suite", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PhaseSpec",
    "SUITE_CONFIRMATION",
    "SUITE_SCHEMA",
    "build_parser",
    "phase_specs",
    "plan",
    "run_suite",
]
