"""M5.8-b1-real-A idle-worker epoch/drain plus one-shot stop preflight.

This is an isolated hardware diagnostic, not a production deploy entrypoint.
It starts real RM65 background worker threads with no target source, proves
that no joint-target SDK call occurs, advances one execution epoch, drains the
three arms-only participants, writes the drain lineage to a dedicated WAL, and
then sends at most one permit-gated ``Arm_Slow_Stop`` per stationary arm.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dataset_contracts.completeness import trace_requirement_profile
from dataset_contracts.control_trace import ControlTraceFact
from robot_motion.action.authoritative_trace import (
    AuthoritativeTraceContext,
    project_execution_cancellation,
    project_execution_cancellation_ack,
)
from robot_motion.deploy import (
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
)
from robot_motion.rm65.constants import (
    RM65_ABSOLUTE_Q_MAX_RAD,
    RM65_ABSOLUTE_Q_MIN_RAD,
)
from robot_motion.runtime import (
    CancellationAcknowledgementTracker,
    ExecutionCancellationRequest,
    acknowledge_worker_epoch_drain,
)
from runtime_artifact_paths import (
    LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
    RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
    ArtifactWriteResolution,
    RuntimeArtifactLayout,
    configured_runtime_artifact_layout,
    resolve_artifact_write_path,
)
from teleop_runtime.arm_frames import TOOL_TCP_FRAME
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.backends.arms.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_capability,
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.backends.arms.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)
from teleop_runtime.backends.arms.rm65_idle_worker_preflight_binding import (
    RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1,
    RM65IdleWorkerDrainHardwareAuthorization,
    RM65IdleWorkerDrainPreflightBinding,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    drain_rm65_arms_only_workers,
)
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceWriterHealthLedger,
    RuntimeTraceSession,
)
from teleop_runtime.control_trace.rm65_worker_drain import (
    admit_rm65_worker_drain_trace,
)
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)


def legacy_preflight_evidence_root() -> Path:
    return Path("teleop_records/preflight_evidence/m5_8b1_real_a")


def canonical_preflight_evidence_root() -> Path:
    return (
        RuntimeArtifactLayout.configured().evidence_dir(kind="preflight")
        / "rm65_idle_worker_drain_stop_only"
    )


def resolve_preflight_evidence_root(
    args: argparse.Namespace,
) -> ArtifactWriteResolution:
    return resolve_artifact_write_path(
        explicit_path=getattr(args, "evidence_root", None),
        artifact_layout=getattr(args, "artifact_layout", None),
        legacy_path=legacy_preflight_evidence_root(),
        canonical_path=canonical_preflight_evidence_root(),
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "M5.8-b1-real-A: run target-free RM65 workers, drain one execution "
            "epoch, persist ControlTrace WAL evidence, and send exactly one "
            "Arm_Slow_Stop per stationary arm. This is NOT read-only."
        )
    )
    parser.add_argument("--left-arm-ip", required=True)
    parser.add_argument("--right-arm-ip", required=True)
    parser.add_argument("--operator-authorization-id", required=True)
    parser.add_argument("--execute-idle-worker-drain-stop-only", action="store_true")
    parser.add_argument("--confirm-arms-stationary", action="store_true")
    parser.add_argument("--confirm-estop-ready", action="store_true")
    parser.add_argument("--idle-observation-sec", type=float, default=0.5)
    parser.add_argument("--state-interval-sec", type=float, default=0.25)
    parser.add_argument("--max-stationary-drift-rad", type=float, default=0.005)
    parser.add_argument("--authorization-valid-sec", type=float, default=90.0)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help=(
            "Exact evidence root. When provided it overrides --artifact-layout."
        ),
    )
    parser.add_argument(
        "--artifact-layout",
        choices=(
            LEGACY_ARTIFACT_LAYOUT_CLI_NAME,
            RUNTIME_ARTIFACT_LAYOUT_CLI_NAME,
        ),
        default=configured_runtime_artifact_layout(),
        help=(
            "Select the managed evidence root when --evidence-root is omitted. "
            "The repository default is legacy; local_paths.json can coordinate "
            "a project-wide canonical cutover."
        ),
    )
    return parser.parse_args(argv)


def _require_explicit_scope(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in (
            "execute_idle_worker_drain_stop_only",
            "confirm_arms_stationary",
            "confirm_estop_ready",
        )
        if not bool(getattr(args, name, False))
    ]
    if missing:
        raise RuntimeError(
            "Refusing M5.8-b1 hardware preflight; missing confirmations: "
            + ", ".join(missing)
        )
    for name in (
        "idle_observation_sec",
        "state_interval_sec",
        "max_stationary_drift_rad",
        "authorization_valid_sec",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be finite and positive.")


def _read_fresh_state(backend: RM65ArmBackend, *, side: str) -> np.ndarray:
    state = backend.refresh_state_from_hardware()
    if state.get("hardware_result_code") != 0:
        raise RuntimeError(
            f"{side}: RM65 state read failed: {state.get('hardware_result_text')}"
        )
    if state.get("hardware_arm_err") != 0 or state.get("hardware_sys_err") != 0:
        raise RuntimeError(
            f"{side}: RM65 error state arm_err={state.get('hardware_arm_err')} "
            f"sys_err={state.get('hardware_sys_err')}"
        )
    qpos = state.get("hardware_qpos")
    if qpos is None:
        raise RuntimeError(f"{side}: no fresh hardware qpos was returned.")
    q = np.asarray(qpos, dtype=float).reshape(-1)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise RuntimeError(f"{side}: hardware qpos is not six finite values.")
    q_min = np.asarray(RM65_ABSOLUTE_Q_MIN_RAD, dtype=float)
    q_max = np.asarray(RM65_ABSOLUTE_Q_MAX_RAD, dtype=float)
    if np.any(q < q_min) or np.any(q > q_max):
        raise RuntimeError(f"{side}: fresh qpos violates the active URDF limits.")
    return q


def _arm_only_pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("left", "right"),
                arms_enabled=True,
                hands_enabled=False,
                arm_joint_lower_rad=tuple(RM65_ABSOLUTE_Q_MIN_RAD),
                arm_joint_upper_rad=tuple(RM65_ABSOLUTE_Q_MAX_RAD),
            )
        ),
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
    )


def _close_without_hiding_primary_error(
    position_executor: PositionDeployExecutor | None,
    backends: list[RM65ArmBackend],
    spool: ControlTraceSpool | None,
    health: ControlTraceWriterHealthLedger | None,
) -> list[str]:
    errors: list[str] = []
    if position_executor is not None:
        try:
            position_executor.close()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            # Hardware SDK cleanup failures must not hide the primary error.
            errors.append(f"position_executor:{type(exc).__name__}:{exc}")
    for backend in reversed(backends):
        try:
            backend.close()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            # Hardware SDK cleanup failures must not hide the primary error.
            errors.append(f"{backend.name}:{type(exc).__name__}:{exc}")
    if spool is not None:
        try:
            spool.close()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            # Storage cleanup failures must not hide the primary error.
            errors.append(f"control_trace_spool:{type(exc).__name__}:{exc}")
    if health is not None:
        try:
            health.close()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            # Storage cleanup failures must not hide the primary error.
            errors.append(f"writer_health_ledger:{type(exc).__name__}:{exc}")
    return errors


def main() -> None:
    args = _parse_args()
    _require_explicit_scope(args)
    storage_resolution = resolve_preflight_evidence_root(args)
    print(
        "[M5.8-b1-real-A] AUTHORIZED STATE CHANGE: idle workers + one epoch "
        "drain + one Arm_Slow_Stop per stationary arm"
    )
    print(
        "[M5.8-b1-real-A] Movej_CANFD/targets/home/enable/rearm/force-position/"
        "replay/owner-switch are denied"
    )
    print(
        "[M5.8-b1-real-A] isolated preflight profile; canonical deploy defaults "
        "and the M6 candidate manifest are not modified"
    )

    manifest = build_rm65_policy_arm_qpos_candidate_manifest()
    capability = build_rm65_policy_arm_qpos_candidate_capability()
    evidence_bundle = build_rm65_execution_evidence_bundle(manifest)
    now = time.monotonic()
    authorization = RM65IdleWorkerDrainHardwareAuthorization.create(
        operator_authorization_id=args.operator_authorization_id,
        execution_manifest=manifest,
        dispatch_inventory=evidence_bundle.dispatch_inventory,
        worker_drain_plan=evidence_bundle.worker_drain_plan,
        control_domain_id="rm65-arms",
        embodiment_instance_id="rm65-bimanual-0",
        allowed_sides=("left", "right"),
        issued_at=now,
        expires_at=now + float(args.authorization_valid_sec),
    )
    runtime_session_id = f"rm65-idle-worker-preflight-{uuid.uuid4()}"
    binding = RM65IdleWorkerDrainPreflightBinding(
        runtime_session_id=runtime_session_id,
        owner_runtime_id="legacy-rm65-idle-worker-preflight",
        evidence_bundle=evidence_bundle,
        authorization=authorization,
    )

    trace_session_id = f"trace-{runtime_session_id}"
    profile = trace_requirement_profile(RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1)
    trace_runtime = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id=trace_session_id,
            runtime_session_id=runtime_session_id,
            control_domain_id=authorization.control_domain_id,
            profile_id=profile.profile_id,
            started_at=now,
            clock_domain_id="control-monotonic",
        ),
        profile=profile,
        capacity=256,
    )
    trace_context = AuthoritativeTraceContext(
        runtime_session_id=runtime_session_id,
        control_domain_id=authorization.control_domain_id,
        embodiment_instance_id=authorization.embodiment_instance_id,
        clock_domain_id="control-monotonic",
    )

    session_root = (
        storage_resolution.path.expanduser().resolve() / runtime_session_id
    )
    session_root.mkdir(parents=True, exist_ok=False)
    spool: ControlTraceSpool | None = ControlTraceSpool(
        session_root / "control_trace.wal",
        trace_session_id=trace_session_id,
        max_bytes=4 << 20,
        physical_preallocation=False,
    )
    health: ControlTraceWriterHealthLedger | None = ControlTraceWriterHealthLedger(
        session_root / "writer_health.ledger",
        trace_session_id=trace_session_id,
    )
    durable_sink = ControlTraceSpoolSink(
        sink_id="rm65-idle-worker-preflight-spool",
        runtime=trace_runtime,
        spool=spool,
        health_ledger=health,
    )

    def admit(fact: ControlTraceFact) -> None:
        receipt = trace_runtime.issue(fact)
        if not receipt.accepted:
            raise RuntimeError(
                f"ControlTrace rejected {fact.event_type}: {receipt.rejection_code}"
            )

    position_executor: PositionDeployExecutor | None = None
    backends: list[RM65ArmBackend] = []
    evidence: dict[str, object] = {
        "profile_id": RM65_IDLE_WORKER_PREFLIGHT_PROFILE_V1,
        "runtime_session_id": runtime_session_id,
        "trace_session_id": trace_session_id,
        "evidence_root": str(session_root),
        "artifact_storage": {
            "requested_layout": str(args.artifact_layout),
            "selection_source": storage_resolution.source,
            "resolved_layout": storage_resolution.layout_name,
            "resolved_family_root": str(
                storage_resolution.path.expanduser().resolve()
            ),
            "semantics": "preflight_control_trace_wal_and_terminal_evidence",
        },
        "authorization": asdict(authorization),
        "manifest_id": manifest.manifest_id,
        "manifest_hash": manifest.semantic_content_hash,
        "inventory_hash": evidence_bundle.dispatch_inventory.semantic_content_hash,
        "worker_drain_plan_hash": evidence_bundle.worker_drain_plan.semantic_content_hash,
        "canonical_defaults_modified": False,
        "motion_authorized": False,
        "lifecycle_authorized": False,
        "sides": {},
    }
    primary_error: BaseException | None = None
    terminal_issued = False
    try:
        admit(binding.project_authorization_fact(trace_context))
        position_executor = PositionDeployExecutor(
            config=PositionDeployExecutorConfig(
                sample_hz=30.0,
                reference_hz=float(capability.reference_resample_hz),
                command_hz=float(capability.command_hz),
            ),
            canonical_pipeline=_arm_only_pipeline(),
            dispatch_callback=None,
            hold_callback=None,
            **binding.position_executor_kwargs(),
        )
        position_executor.start()

        by_side: dict[str, RM65ArmBackend] = {}
        for side, arm_ip in (
            ("left", args.left_arm_ip),
            ("right", args.right_arm_ip),
        ):
            backend = RM65ArmBackend(
                RM65ArmBackendConfig(
                    arm_ip=str(arm_ip),
                    side=side,
                    default_tcp_frame=TOOL_TCP_FRAME,
                    connect_hardware=True,
                    initial_qpos=None,
                    configure_singularity_thresholds=False,
                    force_position_control=False,
                    background_streaming_enabled=True,
                    background_streaming_hz=float(capability.command_hz),
                ),
                hardware_idle_worker_preflight_binding=(
                    binding.backend_binding(side)
                ),
            )
            backend.start()
            backends.append(backend)
            by_side[side] = backend
            if not callable(getattr(backend.arm, "Arm_Slow_Stop", None)):
                # This is a runtime SDK capability failure, not bad input.
                raise RuntimeError(  # noqa: TRY004
                    f"{side}: SDK does not expose Arm_Slow_Stop; refusing fallback."
                )

        before: dict[str, np.ndarray] = {}
        for side, backend in by_side.items():
            before[side] = _read_fresh_state(backend, side=side)
        time.sleep(float(args.state_interval_sec))
        for side, backend in by_side.items():
            second = _read_fresh_state(backend, side=side)
            drift = float(np.max(np.abs(second - before[side])))
            if drift > float(args.max_stationary_drift_rad):
                raise RuntimeError(
                    f"{side}: arm is not stationary; max drift {drift:.6f} rad exceeds "
                    f"{float(args.max_stationary_drift_rad):.6f} rad."
                )
            evidence["sides"][side] = {
                "qpos_before": second.tolist(),
                "stationary_drift_rad": drift,
            }

        time.sleep(float(args.idle_observation_sec))
        for side, backend in by_side.items():
            stream = dict(backend.get_state()["background_streaming"])
            if not stream.get("thread_alive") or int(stream.get("tick_sequence", 0)) <= 0:
                raise RuntimeError(f"{side}: idle background worker did not run.")
            if stream.get("target_active") or int(stream.get("send_count", 0)) != 0:
                raise RuntimeError(
                    f"{side}: target/send activity occurred in the no-motion preflight."
                )
            if stream.get("fault") is not None or not stream.get("healthy"):
                raise RuntimeError(f"{side}: idle background worker is unhealthy: {stream}")
            evidence["sides"][side]["idle_worker_before_drain"] = {
                "tick_sequence": int(stream["tick_sequence"]),
                "send_count": int(stream["send_count"]),
                "target_active": bool(stream["target_active"]),
                "binding_id": stream["hardware_idle_worker_preflight_binding_id"],
            }

        cancellation = binding.context.execution_epochs.cancel(
            ExecutionCancellationRequest(
                reason="hold",
                authority_epoch=0,
                requested_scope="control_domain",
                source_event_ids=(authorization.authorization_id,),
            ),
            capability=capability,
        )
        binding.context.observe_execution_cancellation(cancellation)
        admit(project_execution_cancellation(trace_context, cancellation))

        drain_result = drain_rm65_arms_only_workers(
            plan=binding.worker_drain_plan,
            cancellation=cancellation,
            position_reference_worker=position_executor,
            left_background_worker=by_side["left"],
            right_background_worker=by_side["right"],
        )
        binding.accept_completed_worker_drain(
            drain_result,
            cancellation=cancellation,
        )
        drain_trace_report = admit_rm65_worker_drain_trace(
            drain_result,
            cancellation=cancellation,
            embodiment_instance_id=authorization.embodiment_instance_id,
            fact_sink=admit,
        )
        if not drain_result.aggregate.complete or not drain_trace_report.complete:
            raise RuntimeError("Worker drain or its trace admission is incomplete.")

        tracker = CancellationAcknowledgementTracker()
        tracker.register(cancellation)
        acknowledgement = acknowledge_worker_epoch_drain(
            drain_result.aggregate,
            tracker=tracker,
            clock_domain_id="control-monotonic",
        )
        admit(project_execution_cancellation_ack(trace_context, acknowledgement))

        for side in ("left", "right"):
            backend = by_side[side]
            started = time.monotonic()
            backend.hold_motion("m5_8b1_idle_worker_drain:operator_confirmed")
            latency = time.monotonic() - started
            after = _read_fresh_state(backend, side=side)
            side_evidence = evidence["sides"][side]
            side_evidence.update(
                {
                    "slow_stop_latency_sec": latency,
                    "qpos_after": after.tolist(),
                    "post_stop_delta_rad": float(
                        np.max(
                            np.abs(
                                after
                                - np.asarray(side_evidence["qpos_before"], dtype=float)
                            )
                        )
                    ),
                    "send_result_code": backend.get_state().get("send_result_code"),
                    "send_result_text": backend.get_state().get("send_result_text"),
                }
            )

        for fact in binding.project_stop_dispatch_facts(trace_context):
            admit(fact)

        terminal_at = time.monotonic()
        admit(
            ControlTraceFact.create(
                fact_id=f"runtime-terminal:{runtime_session_id}",
                event_type="runtime_trace_terminal",
                runtime_session_id=runtime_session_id,
                control_domain_id=authorization.control_domain_id,
                embodiment_instance_id=authorization.embodiment_instance_id,
                occurred_at=terminal_at,
                clock_domain_id="control-monotonic",
                payload={
                    "terminal_status": "completed",
                    "reason": "m5_8b1_idle_worker_drain_stop_only_completed",
                    "occurred_at": terminal_at,
                },
            )
        )
        terminal_issued = True
        flush_report = durable_sink.flush_pending()
        if flush_report.failure_code is not None:
            raise RuntimeError(
                f"ControlTrace WAL flush failed: {flush_report.failure_code}"
            )
        completeness = trace_runtime.completeness()
        if not completeness.in_memory_complete or not completeness.durable_complete:
            raise RuntimeError(
                "M5.8-b1 ControlTrace is not complete/durable: "
                f"missing={sorted(completeness.missing_required_event_types)}"
            )

        evidence["binding"] = asdict(binding.snapshot())
        evidence["drain_receipts"] = [
            asdict(receipt) for receipt in drain_result.receipts
        ]
        evidence["drain_aggregate"] = asdict(drain_result.aggregate)
        evidence["drain_trace_report"] = asdict(drain_trace_report)
        evidence["cancellation_acknowledgement"] = asdict(acknowledgement)
        evidence["stop_receipts"] = [
            asdict(receipt) for receipt in binding.stop_receipts
        ]
        evidence["trace"] = {
            "wal_path": str(spool.path),
            "writer_health_path": str(health.path),
            "issued_sequence": trace_runtime.snapshot().journal_issued_sequence,
            "durable_high_watermark": (
                trace_runtime.snapshot().sink_durable_high_watermark
            ),
            "required_event_types": sorted(profile.required_event_types),
            "in_memory_complete": completeness.in_memory_complete,
            "durable_complete": completeness.durable_complete,
            "capture_commit_eligible": completeness.capture_commit_eligible,
        }
        evidence["status"] = "idle_worker_drain_stop_only_completed"
    except BaseException as exc:  # noqa: BLE001
        # Preserve terminal evidence and cleanup even for process-level exits.
        primary_error = exc
        evidence["status"] = "failed"
        evidence["error"] = f"{type(exc).__name__}:{exc}"
        if not terminal_issued:
            try:
                terminal_at = time.monotonic()
                admit(
                    ControlTraceFact.create(
                        fact_id=f"runtime-terminal:{runtime_session_id}",
                        event_type="runtime_trace_terminal",
                        runtime_session_id=runtime_session_id,
                        control_domain_id=authorization.control_domain_id,
                        embodiment_instance_id=authorization.embodiment_instance_id,
                        occurred_at=terminal_at,
                        clock_domain_id="control-monotonic",
                        payload={
                            "terminal_status": "failed",
                            "reason": f"{type(exc).__name__}:{exc}",
                            "occurred_at": terminal_at,
                        },
                    )
                )
                durable_sink.flush_pending()
            except Exception as trace_exc:  # noqa: BLE001  # pragma: no cover
                # Evidence failure is secondary to the primary runtime error.
                evidence["terminal_trace_error"] = (
                    f"{type(trace_exc).__name__}:{trace_exc}"
                )
    finally:
        close_errors = _close_without_hiding_primary_error(
            position_executor,
            backends,
            spool,
            health,
        )
        if close_errors:
            evidence["close_errors"] = close_errors

    evidence_path = session_root / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        "M5_8B1_REAL_A_EVIDENCE="
        + json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    )
    if primary_error is not None:
        raise primary_error
    if evidence.get("close_errors"):
        raise RuntimeError("RM65 close reported errors; inspect evidence.json.")


if __name__ == "__main__":
    main()
