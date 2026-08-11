from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import pytest

from dataset_interface.control_trace import (
    ControlTraceReadError,
    load_episode_control_trace,
    materialize_shadow_evaluation_view,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanLegacyCommandObservation,
)
from teleop_runtime.action.human_shadow_frontend import (
    HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
    RealTeleopHumanShadowProfile,
    build_real_teleop_human_shadow_frontend,
)


pytestmark = pytest.mark.integration


def _command(value: float = 0.1) -> TeleopCommand:
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


def _write_human_shadow_capture(
    dataset_root: Path,
    *,
    finalize: bool,
) -> tuple[str, object]:
    trace_root = dataset_root / "meta/v24/control_trace/sessions"
    frontend = build_real_teleop_human_shadow_frontend(
        RealTeleopHumanShadowProfile(
            mode="shadow",
            pose_source_id="rm65_real_single_left_v1",
            active_sides=("left",),
            arm_reference_frame_by_side={"left": "left_arm_base"},
            decision_hz=60.0,
            trace_capacity=256,
            trace_root=trace_root,
            wal_max_bytes=2 * 1024 * 1024,
            trace_profile_id=HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
        ),
        clock=lambda: 10.0,
        uuid_factory=lambda: uuid.UUID(int=11),
    )
    assert frontend is not None
    frontend.start()
    sink = frontend.build_formal_episode_trace_sink(dataset_root=dataset_root)
    prepared = sink.prepare_episode(legacy_episode_index=0, prepared_at=10.0)
    sink.bind_prepared_episode(prepared, bound_at=10.0)

    command = _command(0.2)
    emission = HumanCommandEmission.capture(
        emission_sequence=0,
        emitted_at=10.0,
        contributions=(("left_pose", command),),
        merged_command=command,
    )
    frontend.observe_emission_sync_for_test(emission)
    frontend.observe_legacy_execution_sync_for_test(
        HumanLegacyCommandObservation.capture(
            emission=emission,
            observed_at=10.0,
            outcome="dispatched",
            command_dispatched=True,
            run_state="RUNNING",
            command=emission.to_merged_command(),
        )
    )
    release_handle = sink.release_episode(
        prepared,
        unbound_at=10.0,
        reason="legacy_finalize_requested",
    )
    released = sink.resolve_release(release_handle)
    assert released is not None and released.seal.trace_sealed
    if finalize:
        report = sink.legacy_finalize_completed(released, completed_at=20.0)
        assert report.capture_commit_status == "committed"
    frontend.close(reason="reader_fixture_complete")
    return prepared.episode_uuid, released


def test_committed_shadow_trace_rebuilds_deterministic_non_authoritative_view(
    tmp_path: Path,
) -> None:
    episode_uuid, _released = _write_human_shadow_capture(tmp_path, finalize=True)

    trace = load_episode_control_trace(tmp_path, episode_uuid)
    first = materialize_shadow_evaluation_view(trace)
    second = materialize_shadow_evaluation_view(trace)

    assert trace.committed
    assert trace.capabilities == ("control_trace_core",)
    assert trace.wal_relative_path.endswith("/control_trace.wal")
    assert len(first) == 1
    assert first == second
    row = first[0]
    assert row.legacy_execution_outcomes == ("dispatched",)
    assert not row.conditional_evaluation
    assert row.binding_assessment_ids == ()
    payload = row.to_payload()
    assert "would_select_action" in payload
    assert "selected_action" not in payload
    assert "action" not in payload


def test_sealed_without_commit_is_audit_readable_but_not_public_capture(
    tmp_path: Path,
) -> None:
    episode_uuid, _released = _write_human_shadow_capture(tmp_path, finalize=False)

    with pytest.raises(ControlTraceReadError, match="no capture commit"):
        load_episode_control_trace(tmp_path, episode_uuid)

    audit_trace = load_episode_control_trace(
        tmp_path,
        episode_uuid,
        require_commit=False,
    )
    assert not audit_trace.committed
    assert len(materialize_shadow_evaluation_view(audit_trace)) == 1


def test_formal_human_capture_rejects_noncanonical_session_wal_location(
    tmp_path: Path,
) -> None:
    frontend = build_real_teleop_human_shadow_frontend(
        RealTeleopHumanShadowProfile(
            mode="shadow",
            pose_source_id="rm65_real_single_left_v1",
            active_sides=("left",),
            arm_reference_frame_by_side={"left": "left_arm_base"},
            trace_root=tmp_path / "custom-trace-root",
            trace_profile_id=HUMAN_FORMAL_SHADOW_TRACE_PROFILE_V1,
        ),
        uuid_factory=lambda: uuid.UUID(int=12),
    )
    assert frontend is not None
    frontend.start()
    try:
        with pytest.raises(ValueError, match="canonical dataset session path"):
            frontend.build_formal_episode_trace_sink(dataset_root=tmp_path)
    finally:
        frontend.close(reason="noncanonical_path_fixture_complete")
