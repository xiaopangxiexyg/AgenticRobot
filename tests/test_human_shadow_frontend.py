from __future__ import annotations

import argparse
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanLegacyCommandObservation,
)
from teleop_runtime.action.human_shadow_frontend import (
    HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
    HumanShadowFrontend,
    RealTeleopHumanShadowProfile,
    add_human_shadow_arguments,
    build_real_teleop_human_shadow_frontend,
    validate_human_shadow_arguments,
)
from teleop_runtime.action.human_safety_shadow import (
    build_human_legacy_aligned_safety_profile,
    evaluate_human_legacy_aligned_safety,
)
from teleop_runtime.real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyConfig,
    TemporaryDualArmSafetyLimiter,
)


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "teleop_runtime/action/human_shadow_frontend.py"
COMPONENT_BUILDER = ROOT / "teleop_runtime/real_teleop/component_builder.py"
SHUTDOWN_RUNTIME = ROOT / "teleop_runtime/real_teleop/shutdown_runtime.py"


def profile(**overrides) -> RealTeleopHumanShadowProfile:
    values = {
        "mode": "shadow",
        "pose_source_id": "rm65_real_single_left_v1",
        "active_sides": ("left",),
        "arm_reference_frame_by_side": {"left": "left_arm_base"},
        "decision_hz": 60.0,
        "trace_capacity": 256,
    }
    values.update(overrides)
    return RealTeleopHumanShadowProfile(**values)


def arm_command(value: float = 0.1) -> TeleopCommand:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = value
    command = TeleopCommand(timestamp=10.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="cartesian_pose",
            value=pose,
            frame_name="left_arm_base",
        )
    )
    return command


def emission(
    command: TeleopCommand,
    *,
    sequence: int = 0,
    emitted_at: float = 10.0,
) -> HumanCommandEmission:
    return HumanCommandEmission.capture(
        emission_sequence=sequence,
        emitted_at=emitted_at,
        contributions=(("left_pose", command),),
        merged_command=command,
    )


def arm_qpos_command() -> TeleopCommand:
    command = TeleopCommand(timestamp=10.0)
    command.add_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.arange(6, dtype=np.float32),
            frame_name=None,
            metadata={"target_pose_base": np.eye(4, dtype=np.float64)},
        )
    )
    return command


def test_default_off_profile_does_not_allocate_runtime_identity() -> None:
    called = False

    def forbidden_uuid() -> uuid.UUID:
        nonlocal called
        called = True
        raise AssertionError("disabled frontend must not allocate identity")

    disabled = profile(mode="off")
    assert build_real_teleop_human_shadow_frontend(
        disabled,
        uuid_factory=forbidden_uuid,
    ) is None
    assert not called


def test_frontend_builds_profile_derived_identity_catalog_and_specs() -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(
            hand_backend_by_side={"left": "inspire"},
            session_hand_sides=("left",),
            control_domain_id="station-control-1",
            embodiment_instance_id="rm65-inspire-left-1",
        ),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=1),
    )

    assert isinstance(frontend, HumanShadowFrontend)
    assert frontend.runtime_session_id == "human-gui-runtime-00000000000000000000000000000001"
    assert frontend.control_domain_id == "station-control-1"
    adapter = frontend.runtime.adapter
    assert adapter.config.embodiment_instance_id == "rm65-inspire-left-1"
    assert tuple(adapter.config.component_catalog.component_by_id) == (
        "left_arm",
        "left_hand",
    )
    assert adapter.config.action_adapter.arm_specs["left"].reference_frame == (
        "left_arm_base"
    )
    assert adapter.config.action_adapter.hand_specs["left"].hand_model == "inspire_tac"
    assert frontend.trace_runtime.episode_membership is None


def test_frontend_end_to_end_trace_remains_shadow_and_in_memory_only() -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=2),
    )
    assert frontend is not None
    source = emission(arm_command(0.2))
    observed = frontend.observe_emission_sync_for_test(source)
    assert observed.shadow_step.evaluation is not None
    assert observed.shadow_step.evaluation.non_authoritative
    frontend.observe_legacy_execution_sync_for_test(
        HumanLegacyCommandObservation.capture(
            emission=source,
            observed_at=10.0,
            outcome="dispatched",
            command_dispatched=True,
            run_state="RUNNING",
            command=source.to_merged_command(),
        )
    )
    frontend.close(reason="test_complete")

    assert frontend.close_report is not None
    assert frontend.close_report.in_memory_complete
    assert not frontend.close_report.durable_complete
    assert frontend.close_report.wal_directory is None
    event_types = {
        record.fact.event_type for record in frontend.trace_runtime.records()
    }
    assert "arbitration_evaluation" in event_types
    assert "arbitration_decision" not in event_types
    assert frontend.trace_runtime.episode_membership is None


def test_position_ik_output_is_explicit_absolute_qpos_shadow_not_lossy_pose() -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(action_space="bimanual_qpos_v1", arm_joint_policy="qpos"),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=5),
    )
    assert frontend is not None
    observed = frontend.observe_emission_sync_for_test(emission(arm_qpos_command()))

    assert observed.adaptation.source_adapter_receipt.lossiness == "lossless"
    assert observed.adaptation.source_adapter_receipt.authoritative_eligible
    assert observed.adaptation.robot_action.action_space == "bimanual_qpos_v1"
    assert observed.shadow_step.evaluation is not None
    np.testing.assert_array_equal(
        observed.shadow_step.evaluation.would_select_action.value(
            "left_arm"
        ).values,
        np.arange(6, dtype=np.float32),
    )
    frontend.close(reason="qpos_shadow_fixture")


def test_frontend_binds_exact_legacy_limiter_anchor_without_claiming_commit() -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=8),
    )
    assert frontend is not None
    limiter = TemporaryDualArmSafetyLimiter(
        TemporaryDualArmSafetyConfig(
            enabled=True,
            active_sides=("left", "right"),
            max_xyz_step_m=0.1,
            max_rot_step_deg=15.0,
            evidence_enabled=True,
        )
    )

    first = emission(arm_command(0.0), sequence=0, emitted_at=10.0)
    frontend.observe_emission_sync_for_test(first)
    first_filtered = limiter.filter_command(first.to_merged_command(), None, 10.0)
    first_bound = frontend.observe_filtered_command_sync_for_test(
        first,
        first_filtered,
        10.0,
        limiter.last_evaluation,
    )
    assert not first_bound.binding_evidence.comparable
    assert "legacy_previous_anchor_unavailable" in first_bound.binding_evidence.reasons

    second = emission(arm_command(0.3), sequence=1, emitted_at=10.1)
    second_source = frontend.observe_emission_sync_for_test(second)
    second_filtered = limiter.filter_command(second.to_merged_command(), None, 10.1)
    second_bound = frontend.observe_filtered_command_sync_for_test(
        second,
        second_filtered,
        10.1,
        limiter.last_evaluation,
    )

    evidence = second_bound.binding_evidence
    assert evidence.comparable
    assert evidence.input_matches_emission
    assert evidence.output_matches_post_filter
    assert evidence.previous_anchor_available
    assert evidence.non_authoritative
    filtered_adaptation = frontend.runtime.adapter.adapt_filter_observation(
        second_bound.observation
    )
    assert second_source.adaptation.robot_action is not None
    assert filtered_adaptation.robot_action is not None
    comparison = evaluate_human_legacy_aligned_safety(
        source_action=second_source.adaptation.robot_action,
        legacy_filtered_action=filtered_adaptation.robot_action,
        binding_evidence=evidence,
        limiter_evaluation=limiter.last_evaluation,
        arbitration_evaluation_id="human-eval-fixture-1",
        profile=build_human_legacy_aligned_safety_profile(
            source_action=second_source.adaptation.robot_action,
            limiter_evaluation=limiter.last_evaluation,
        ),
    )
    assert comparison.status == "matched"
    assert comparison.shared_safety_result is not None
    assert comparison.shared_safety_result.anchor_kind == "previous_commanded"
    assert comparison.non_authoritative
    # Limiter filter-stage state is evidence only; no target/dispatch fact is emitted.
    assert {
        record.fact.event_type for record in frontend.trace_runtime.records()
    }.isdisjoint({"execution_target_admission", "arbitration_decision"})
    frontend.close(reason="human_legacy_safety_evidence_fixture")


def test_qpos_legacy_limiter_matches_shared_joint_step_without_claiming_position_parity() -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(action_space="bimanual_qpos_v1", arm_joint_policy="qpos"),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=9),
    )
    assert frontend is not None
    limiter = TemporaryDualArmSafetyLimiter(
        TemporaryDualArmSafetyConfig(
            enabled=True,
            active_sides=("left", "right"),
            max_joint_step_deg=6.0,
            evidence_enabled=True,
        )
    )

    first_command = arm_qpos_command()
    first = emission(first_command, sequence=0, emitted_at=10.0)
    frontend.observe_emission_sync_for_test(first)
    first_filtered = limiter.filter_command(first.to_merged_command(), None, 10.0)
    first_bound = frontend.observe_filtered_command_sync_for_test(
        first,
        first_filtered,
        10.0,
        limiter.last_evaluation,
    )
    assert not first_bound.binding_evidence.comparable

    second_command = arm_qpos_command()
    second_command.commands["left_arm"].value = (
        np.arange(6, dtype=np.float32) + 0.3
    )
    second = emission(second_command, sequence=1, emitted_at=10.1)
    second_source = frontend.observe_emission_sync_for_test(second)
    second_filtered = limiter.filter_command(second.to_merged_command(), None, 10.1)
    second_bound = frontend.observe_filtered_command_sync_for_test(
        second,
        second_filtered,
        10.1,
        limiter.last_evaluation,
    )
    filtered_adaptation = frontend.runtime.adapter.adapt_filter_observation(
        second_bound.observation
    )
    assert second_bound.binding_evidence.comparable
    assert second_source.adaptation.robot_action is not None
    assert filtered_adaptation.robot_action is not None
    comparison_profile = build_human_legacy_aligned_safety_profile(
        source_action=second_source.adaptation.robot_action,
        limiter_evaluation=limiter.last_evaluation,
    )
    assert comparison_profile.profile_id == (
        "human-temporary-legacy-aligned-qpos-v1"
    )
    comparison = evaluate_human_legacy_aligned_safety(
        source_action=second_source.adaptation.robot_action,
        legacy_filtered_action=filtered_adaptation.robot_action,
        binding_evidence=second_bound.binding_evidence,
        limiter_evaluation=limiter.last_evaluation,
        arbitration_evaluation_id="human-qpos-eval-fixture-1",
        profile=comparison_profile,
    )
    assert comparison.status == "matched"
    assert comparison.shared_safety_result is not None
    assert comparison.shared_safety_result.anchor_kind == "previous_commanded"
    assert comparison.non_authoritative
    assert {
        record.fact.event_type for record in frontend.trace_runtime.records()
    }.isdisjoint(
        {
            "reference_segment_admission",
            "execution_target_admission",
            "arbitration_decision",
        }
    )
    frontend.close(reason="human_qpos_legacy_safety_evidence_fixture")


def test_optional_scoped_wal_is_durable_but_not_an_episode(tmp_path: Path) -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(trace_root=tmp_path, wal_max_bytes=2 * 1024 * 1024),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=3),
    )
    assert frontend is not None
    source = emission(arm_command())
    frontend.observe_emission_sync_for_test(source)
    frontend.observe_legacy_execution_sync_for_test(
        HumanLegacyCommandObservation.capture(
            emission=source,
            observed_at=10.0,
            outcome="dispatched",
            command_dispatched=True,
            run_state="RUNNING",
            command=source.to_merged_command(),
        )
    )
    frontend.close(reason="wal_fixture_complete")

    assert frontend.close_report is not None
    assert frontend.close_report.durable_complete
    assert frontend.trace_runtime.episode_membership is None
    assert (frontend.wal_directory / "control_trace.wal").is_file()
    assert (frontend.wal_directory / "writer_health.ledger").is_file()


def test_formal_human_shadow_episode_seals_without_terminating_session(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    trace_root = dataset_root / "meta/v24/control_trace/sessions"
    frontend = build_real_teleop_human_shadow_frontend(
        profile(
            trace_root=trace_root,
            wal_max_bytes=2 * 1024 * 1024,
            trace_profile_id=HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
        ),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=6),
    )
    assert frontend is not None
    frontend.start()
    assert frontend._worker is not None

    episode_sink = frontend.build_formal_episode_trace_sink(
        dataset_root=dataset_root,
    )
    prepared = episode_sink.prepare_episode(legacy_episode_index=0, prepared_at=10.0)
    episode_sink.bind_prepared_episode(prepared, bound_at=10.0)
    source = emission(arm_command(0.2))
    frontend.observe_emission_sync_for_test(source)
    frontend.observe_legacy_execution_sync_for_test(
        HumanLegacyCommandObservation.capture(
            emission=source,
            observed_at=10.0,
            outcome="dispatched",
            command_dispatched=True,
            run_state="RUNNING",
            command=source.to_merged_command(),
        )
    )
    release_handle = episode_sink.release_episode(
        prepared,
        unbound_at=10.0,
        reason="legacy_finalize_requested",
    )
    finalized = episode_sink.legacy_finalize_completed(
        release_handle,
        completed_at=20.0,
    )
    released = episode_sink.resolve_release(release_handle)
    assert released is not None

    assert released.seal.trace_sealed
    assert released.seal.capture_commit_eligible
    assert finalized.capture_commit_status == "committed"
    assert frontend.trace_runtime.episode_membership is None
    # A per-episode commit must not close the session admission gate.
    second = episode_sink.prepare_episode(legacy_episode_index=1, prepared_at=10.0)
    episode_sink.cancel_prepared_episode(second)
    frontend.close(reason="formal_human_fixture_complete")
    assert frontend.close_report is not None
    assert frontend.close_report.durable_complete


def test_formal_episode_release_barrier_preserves_fifo_membership_boundary(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    frontend = build_real_teleop_human_shadow_frontend(
        profile(
            trace_root=dataset_root / "meta/v24/control_trace/sessions",
            wal_max_bytes=2 * 1024 * 1024,
            trace_profile_id=HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
        ),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=7),
    )
    assert frontend is not None
    frontend.start()
    episode_sink = frontend.build_formal_episode_trace_sink(dataset_root=dataset_root)
    prepared = episode_sink.prepare_episode(legacy_episode_index=0, prepared_at=10.0)
    episode_sink.bind_prepared_episode(prepared, bound_at=10.0)

    worker_entered = threading.Event()
    allow_first_emission = threading.Event()
    original_observe_emission = frontend.runtime.observe_emission

    def block_first_emission(value: HumanCommandEmission):
        if value.emission_sequence == 0:
            worker_entered.set()
            assert allow_first_emission.wait(2.0)
        return original_observe_emission(value)

    frontend.runtime.observe_emission = block_first_emission
    first = emission(arm_command(0.2), sequence=0)
    assert frontend.observe_emission(first)
    assert worker_entered.wait(1.0)
    assert frontend.observe_legacy_execution(
        HumanLegacyCommandObservation.capture(
            emission=first,
            observed_at=10.0,
            outcome="dispatched",
            command_dispatched=True,
            run_state="RUNNING",
            command=first.to_merged_command(),
        )
    )
    release_handle = episode_sink.release_episode(
        prepared,
        unbound_at=10.0,
        reason="legacy_finalize_requested",
    )
    assert not release_handle.done
    second = emission(arm_command(0.3), sequence=1, emitted_at=11.0)
    assert frontend.observe_emission(second)

    allow_first_emission.set()
    released = episode_sink.resolve_release(release_handle)
    assert released is not None
    frontend._observer_queue.join()

    source_entries = {
        int(entry.fact.payload["source_emission_sequence"]): entry
        for entry in frontend.trace_runtime.journal_entries()
        if entry.fact.event_type == "human_command_emission"
    }
    assert source_entries[0].episode_membership is not None
    assert source_entries[0].episode_membership.episode_uuid == prepared.episode_uuid
    assert source_entries[1].episode_membership is None
    assert (
        released.release.trace_range.last_journal_sequence
        < source_entries[1].journal_sequence
    )
    assert released.seal.trace_sealed

    finalized = episode_sink.legacy_finalize_completed(
        released,
        completed_at=20.0,
    )
    assert finalized.capture_commit_status == "committed"
    frontend.close(reason="formal_fifo_boundary_fixture_complete")


def test_out_of_session_dmtac_branch_is_rejected_instead_of_omitted() -> None:
    with pytest.raises(ValueError, match="outside TeleopSession"):
        build_real_teleop_human_shadow_frontend(
            profile(
                hand_backend_by_side={"left": "dmtac-gripper"},
                out_of_session_effectors=("left_hand",),
            )
        )


def test_gui_observer_queue_is_nonblocking_and_drop_invalidates_completeness() -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        profile(observer_queue_capacity=1),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=4),
    )
    assert frontend is not None
    frontend.start()
    worker_entered = threading.Event()
    release_worker = threading.Event()

    def blocked_observer(_emission) -> None:
        worker_entered.set()
        assert release_worker.wait(2.0)

    frontend.runtime.observe_emission = blocked_observer
    source = emission(arm_command())
    started_at = time.perf_counter()
    assert frontend.observe_emission(source)
    assert worker_entered.wait(1.0)
    assert frontend.observe_emission(source)
    assert not frontend.observe_emission(source)
    assert time.perf_counter() - started_at < 0.1
    release_worker.set()
    frontend.close(reason="observer_overflow_fixture")

    assert frontend.close_report is not None
    assert frontend.close_report.observer_drop_count == 1
    assert not frontend.close_report.in_memory_complete


def test_cli_defaults_off_and_rejects_incomplete_or_conflicting_modes() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand-backend", default="inspire")
    parser.add_argument("--left-hand-backend", default=None)
    parser.add_argument("--right-hand-backend", default=None)
    parser.add_argument("--active-sides", default=("left", "right"))
    parser.add_argument("--no-debug-disk-write", action="store_true")
    add_human_shadow_arguments(parser)

    defaults = parser.parse_args([])
    validate_human_shadow_arguments(defaults, parser)
    assert defaults.human_action_arbiter_mode == "off"
    assert defaults.human_action_arbiter_hz == 60.0
    assert defaults.human_action_shadow_observer_queue_size == 4096

    with pytest.raises(SystemExit):
        args = parser.parse_args(
            ["--human-action-shadow-trace-root", "/tmp/trace-without-shadow"]
        )
        validate_human_shadow_arguments(args, parser)
    with pytest.raises(SystemExit):
        args = parser.parse_args(
            [
                "--human-action-arbiter-mode",
                "shadow",
                "--human-action-shadow-trace-root",
                "/tmp/trace-disabled",
                "--no-debug-disk-write",
            ]
        )
        validate_human_shadow_arguments(args, parser)
    with pytest.raises(SystemExit):
        args = parser.parse_args(
            [
                "--human-action-arbiter-mode",
                "shadow",
                "--hand-backend",
                "dmtac-gripper",
            ]
        )
        validate_human_shadow_arguments(args, parser)


def test_canonical_gui_wiring_is_passive_default_off_and_closes_first() -> None:
    frontend_source = FRONTEND.read_text(encoding="utf-8")
    component_source = COMPONENT_BUILDER.read_text(encoding="utf-8")
    shutdown_source = SHUTDOWN_RUNTIME.read_text(encoding="utf-8")

    assert 'default="off"' in frontend_source
    assert "human_command_observer=(" in component_source
    assert "human_execution_observer=(" in component_source
    assert "human_shadow_frontend.observe_emission" in component_source
    assert "human_shadow_frontend.observe_legacy_execution" in component_source
    assert shutdown_source.index(
        '("human_action_shadow", components.human_shadow_frontend.close)'
    ) < shutdown_source.index(
        '("teleop_session", components.session.close)'
    )
    assert "DispatchOwnerToken" not in frontend_source
    assert ".send_commands(" not in frontend_source
    assert ".send_command(" not in frontend_source
