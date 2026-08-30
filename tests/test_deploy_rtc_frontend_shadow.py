from __future__ import annotations

import argparse
import inspect
import json
import queue
import threading
import time
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from control_station import (
    BoundedStationEventLog,
    SideEffectStationOperationCoordinator,
    StationIntentDraft,
)
from control_station.composition import (
    CS3_RTC_POLICY_PASSIVE_SHADOW_MANIFEST_ID_V1,
    RtcPolicyPassiveStationBootstrap,
)
from control_station.web import (
    LocalIntentAccessPolicy,
    build_scoped_passive_shadow_control_station_server,
)
from deployments.client.policy import load_policy
from robot_motion.deploy import (
    ACTION_ARM_POSE_SLICES,
    DEPLOY_ACTION_DIM,
    POLICY_ASYNC_PRODUCER_FAULT,
    POLICY_ASYNC_PRODUCER_HOLD,
    POLICY_WORKER_SESSION_ABANDONED_UNDRAINED,
    POLICY_WORKER_SESSION_DRAINED,
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    ActionSafetyFilter,
    ActionSafetyFilterConfig,
    CanonicalDeployActionPipeline,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    build_deployment_observation,
    build_rtc_policy_emission_source_binding,
    nan_vector,
)
from teleop_runtime.deploy.rtc_frontend import (
    RtcDecodedPolicyExecutor,
    RtcPolicyShadowConfig,
    RtcPolicyShadowFrontend,
)
from teleop_runtime.deploy.rtc_step_fanout import (
    PolicyActionEmissionFanout,
    RtcFrontendStepFanout,
    RtcPolicyEmissionPublisher,
    build_rtc_policy_emission_binding_ack,
)
from teleop_runtime.deploy import run_policy


class _ChunkPolicy:
    name = "rtc-3c-virtual-chunk"

    def __init__(self, chunk_size: int = 4) -> None:
        self.chunk_size = int(chunk_size)

    def predict(self, _observation):
        actions = []
        for index in range(self.chunk_size):
            action = nan_vector(DEPLOY_ACTION_DIM)
            action[ACTION_ARM_POSE_SLICES["right"]] = np.asarray(
                [
                    0.10 + 0.001 * index,
                    0.20,
                    0.30,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            actions.append(action)
        return np.stack(actions)


def _observation(*, timestamp: float, step_index: int):
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [0.10, 0.20, 0.30]
    return build_deployment_observation(
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
        step_index=step_index,
        arm_states={
            "right": {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
        },
        hand_states={},
        timestamp=timestamp,
    )


def _pipeline() -> CanonicalDeployActionPipeline:
    return CanonicalDeployActionPipeline(
        safety_filter=ActionSafetyFilter(
            ActionSafetyFilterConfig(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
                max_arm_xyz_step_m=0.05,
                max_arm_rot_step_deg=20.0,
                max_arm_joint_step_rad=0.2,
            )
        ),
        active_sides=("right",),
        arms_enabled=True,
        hands_enabled=False,
    )


def _runtime() -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            arms_enabled=True,
            hands_enabled=False,
            action_keyframe_period_sec=1.0 / 30.0,
            action_maximum_hold_sec=1.0 / 30.0,
        )
    )


def _config(*, session: str = "rtc-3c-test") -> RtcPolicyShadowConfig:
    return RtcPolicyShadowConfig(
        runtime_session_id=session,
        control_domain_id="deploy-control:test",
        producer_id="policy:virtual-test",
        policy_node_id="single_policy",
        artifact_hash="artifact:virtual-test",
        action_codec_hash="codec:deploy-absolute-v1",
        frame_contract_hash="frame:runtime-raw-v1",
        layout_hash="layout:deploy-v23-v1",
        normalization_hash="normalization:none-v1",
        prediction_horizon_steps=4,
        request_validity_steps=10,
        request_trigger_coverage_steps=2,
    )


def _wait_for(
    predicate,
    *,
    timeout_sec: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError("condition timed out")


def _parse_deploy_args(*argv: str):
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser.parse_args(list(argv))


def test_rtc_profile_is_default_off_while_position_splice_is_default_on() -> None:
    args = _parse_deploy_args()
    assert args.rtc_profile == "off"
    assert args.rtc_debug_level is None
    assert args.position_chunk_lookahead is True
    assert args.position_plan_splice_governor is True
    assert args.position_stream_start_acquisition is True
    run_policy.validate_args(args)
    assert args._rtc_debug_level == "off"
    assert args.position_chunk_lookahead is True


def test_stream_start_acquisition_accepts_rtc_dry_run_and_real_dispatch() -> None:
    profile_id = "rtc_3c_virtual_shadow_15_30_60_120_v1"
    args = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--position-plan-splice-governor",
        "--position-stream-start-acquisition",
    )
    run_policy.validate_args(args)
    assert args.position_chunk_lookahead is False
    assert args.position_stream_start_acquisition is True
    assert args.send_actions is False

    active = _parse_deploy_args(
        "--send-actions",
        "--start-init_qpos",
        "--left-arm-ip",
        "192.0.2.1",
        "--right-arm-ip",
        "192.0.2.2",
        "--arm-control-mode",
        "position",
        "--debug",
    )
    run_policy.validate_args(active)
    assert active.position_plan_splice_governor is True
    assert active.position_stream_start_acquisition is True
    assert active.send_actions is True


def test_position_plan_splice_and_stream_start_have_atomic_cli_rollback() -> None:
    args = _parse_deploy_args(
        "--no-position-plan-splice-governor",
        "--no-position-stream-start-acquisition",
    )
    run_policy.validate_args(args)

    assert args.position_plan_splice_governor is False
    assert args.position_stream_start_acquisition is False


def test_rtc_debug_defaults_to_summary_and_requires_explicit_debug() -> None:
    profile_id = "rtc_3c_virtual_shadow_15_30_60_120_v1"
    args = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--debug",
    )
    run_policy.validate_args(args)
    assert args._rtc_debug_level == "summary"

    trace_args = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--debug",
        "--rtc-debug-level",
        "trace",
        "--rtc-debug-queue-capacity",
        "8",
    )
    run_policy.validate_args(trace_args)
    assert trace_args._rtc_debug_level == "trace"

    missing_debug = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--rtc-debug-level",
        "trace",
    )
    with pytest.raises(ValueError, match="requires --debug"):
        run_policy.validate_args(missing_debug)


def test_rtc_profile_forces_dry_run_cadence_and_disables_lookahead() -> None:
    profile_id = "rtc_3c_virtual_shadow_15_30_60_120_v1"
    args = _parse_deploy_args("--rtc-profile", profile_id)
    run_policy.validate_args(
        args,
        raw_argv=("--rtc-profile", profile_id),
    )
    assert args.action_hz == 30.0
    assert args.policy_hz == 30.0
    assert args.position_reference_hz == 60.0
    assert args.arm_control_hz == 120.0
    assert args.position_chunk_lookahead is False
    assert args.position_path_lift_backend == run_policy.PINK_POSE_LIFT_BACKEND
    assert args.position_continuous_waypoint_profile_effective == (
        "formal_third_order"
    )
    assert args.position_continuous_waypoint_async_precompute is False
    assert not args.send_actions

    explicit_c2 = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--debug",
        "--position-continuous-waypoint-profile",
        "async_c2",
    )
    with pytest.raises(ValueError, match="async_c2 requires the effective"):
        run_policy.validate_args(
            explicit_c2,
            raw_argv=(
                "--rtc-profile",
                profile_id,
                "--debug",
                "--position-continuous-waypoint-profile",
                "async_c2",
            ),
        )

    with pytest.raises(ValueError, match="forbids --send-actions"):
        run_policy.validate_args(
            _parse_deploy_args(
                "--rtc-profile",
                profile_id,
                "--send-actions",
            )
        )
    with pytest.raises(ValueError, match="builtin hold"):
        run_policy.validate_args(
            _parse_deploy_args(
                "--rtc-profile",
                profile_id,
                "--policy",
                "some.module:Policy",
            )
        )


def test_raw_rtc_profile_is_explicit_model_adapter_shadow_only() -> None:
    profile_id = RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1
    args = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--policy",
        "some.module:Policy",
        "--policy-artifact-json",
        "/tmp/raw-rtc-policy-artifact.json",
    )
    run_policy.validate_args(
        args,
        raw_argv=(
            "--rtc-profile",
            profile_id,
            "--policy-artifact-json",
            "/tmp/raw-rtc-policy-artifact.json",
        ),
    )
    assert args.action_hz == 30.0
    assert args.policy_hz == 30.0
    assert args.position_reference_hz == 60.0
    assert args.arm_control_hz == 120.0
    assert args.position_chunk_lookahead is False
    assert not args.send_actions

    with pytest.raises(ValueError, match="forbids --send-actions"):
        run_policy.validate_args(
            _parse_deploy_args(
                "--rtc-profile",
                profile_id,
                "--policy-artifact-json",
                "/tmp/raw-rtc-policy-artifact.json",
                "--send-actions",
            )
        )


def test_conditioned_model_rtc_profile_reuses_shadow_guards() -> None:
    profile_id = MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
    args = _parse_deploy_args(
        "--rtc-profile",
        profile_id,
        "--policy",
        "some.module:Policy",
        "--policy-artifact-json",
        "/tmp/model-rtc-policy-artifact.json",
        "--rtc-real-shadow",
    )
    run_policy.validate_args(
        args,
        raw_argv=(
            "--rtc-profile",
            profile_id,
            "--policy-artifact-json",
            "/tmp/model-rtc-policy-artifact.json",
            "--rtc-real-shadow",
        ),
    )
    assert args.action_hz == 30.0
    assert args.policy_hz == 30.0
    assert args.position_chunk_lookahead is False
    assert not args.send_actions

    with pytest.raises(ValueError, match="forbids --send-actions"):
        run_policy.validate_args(
            _parse_deploy_args(
                "--rtc-profile",
                profile_id,
                "--policy-artifact-json",
                "/tmp/model-rtc-policy-artifact.json",
                "--send-actions",
            )
        )


def test_rtc_step_fanout_preserves_identity_and_isolates_passive_branches() -> None:
    class FailingQueue(queue.Queue):
        def put_nowait(self, _item: object) -> None:
            raise RuntimeError("passive observer failed")

    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-fanout"),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        preparer=_pipeline(),
        created_at=time.monotonic(),
    )
    healthy: queue.Queue = queue.Queue(maxsize=2)
    full: queue.Queue = queue.Queue(maxsize=1)
    full.put_nowait(object())
    failing: queue.Queue = FailingQueue(maxsize=1)
    fanout = RtcFrontendStepFanout(
        {"control_station": healthy, "full": full, "failing": failing}
    )
    binding = fanout.bind_frontend(frontend)
    assert fanout.wait_for_source_binding(timeout_sec=0.0) is binding
    with pytest.raises(RuntimeError, match="already bound"):
        fanout.bind_frontend(frontend)
    frontend.start()
    base = frontend.state.plan.cursor_at
    step = None
    try:
        for index in range(120):
            candidate = frontend.step(
                _observation(timestamp=40.0 + index / 30.0, step_index=index),
                now=base + index / 30.0,
            )
            if candidate.action_preparation.emission is not None:
                step = candidate
                break
            time.sleep(0.001)
    finally:
        frontend.close(now=base + 5.0)
    assert step is not None

    fanout.put_nowait(step)

    assert healthy.get_nowait() is step
    status = fanout.status()
    assert status["authoritative"] is False
    assert status["producer_lifecycle_owned"] is False
    assert status["source_binding_hash"] == binding.semantic_content_hash
    assert status["input_offer_count"] == 1
    assert status["emitting_step_count"] == 1
    assert status["subscribers"]["control_station"]["offer_count"] == 1
    assert status["subscribers"]["full"]["drop_count"] == 1
    assert status["subscribers"]["failing"]["error_count"] == 1


def test_rtc_step_fanout_is_default_off_and_only_called_at_exact_step_seam() -> None:
    with pytest.raises(ValueError, match="bounded queue"):
        RtcFrontendStepFanout({"unbounded": queue.Queue()})

    fanout = RtcFrontendStepFanout(
        {"control_station": queue.Queue(maxsize=1)}
    )
    emission_fanout = PolicyActionEmissionFanout(
        {"control_station": queue.Queue(maxsize=1)}
    )
    emission_publisher = RtcPolicyEmissionPublisher(
        observers={"control_station": queue.Queue(maxsize=1)}
    )
    with pytest.raises(ValueError, match="explicit RTC-3c profile"):
        run_policy.main([], rtc_step_fanout=fanout)
    with pytest.raises(ValueError, match="explicit RTC profile"):
        run_policy.main([], rtc_emission_fanout=emission_fanout)
    with pytest.raises(ValueError, match="explicit RTC profile"):
        run_policy.main([], rtc_emission_publisher=emission_publisher)

    source = inspect.getsource(run_policy.main)
    assert source.count("rtc_step_fanout.put_nowait(rtc_step)") == 1
    assert source.count("rtc_step_fanout.bind_frontend(rtc_frontend)") == 1
    assert source.count("rtc_emission_fanout.put_nowait(emission)") == 1
    assert source.count("rtc_emission_fanout.bind_frontend(rtc_frontend)") == 1
    assert source.count(
        "rtc_emission_publisher.publish_committed_emission("
    ) == 1
    assert source.count(
        "rtc_emission_publisher.bind_frontend(rtc_frontend)"
    ) == 1
    assert "control_station" not in inspect.getsource(
        __import__(
            "teleop_runtime.deploy.rtc_step_fanout",
            fromlist=["RtcFrontendStepFanout"],
        )
    )
    assert "teleop_runtime" not in inspect.getsource(
        __import__(
            "control_station.composition.rtc_policy_shadow",
            fromlist=["RtcPolicyPassiveStationBootstrap"],
        )
    )


def test_policy_emission_fanout_is_bounded_bound_and_emission_only() -> None:
    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-emission-fanout"),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        preparer=_pipeline(),
        created_at=time.monotonic(),
    )
    output: queue.Queue = queue.Queue(maxsize=2)
    fanout = PolicyActionEmissionFanout({"consumer": output})
    with pytest.raises(RuntimeError, match="not bound"):
        fanout.put_nowait(object())
    binding = fanout.bind_frontend(frontend)
    frontend.start()
    base = frontend.state.plan.cursor_at
    try:
        step = None
        for index in range(120):
            candidate = frontend.step(
                _observation(timestamp=50.0 + index / 30.0, step_index=index),
                now=base + index / 30.0,
            )
            if candidate.emission_result.emission is not None:
                step = candidate
                break
            time.sleep(0.001)
        assert step is not None
        emission = step.emission_result.emission
        assert emission is not None
        assert emission.source_binding_hash == binding.semantic_content_hash
        with pytest.raises(ValueError, match="source binding identity"):
            binding.validate_emission(
                replace(emission, source_binding_hash="binding:another-epoch")
            )
        assert step.emission_commit_transition is not None
        assert step.emission_commit_transition.receipt.accepted
        fanout.put_nowait(emission)
        assert output.get_nowait() is emission
        assert fanout.status()["source_binding_hash"] == binding.semantic_content_hash
        with pytest.raises(TypeError, match="PolicyActionEmission"):
            fanout.put_nowait(step)
    finally:
        frontend.close(now=base + 5.0)


def test_committed_emission_publisher_validates_commit_ttl_and_idempotency() -> None:
    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-committed-publisher"),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        preparer=_pipeline(),
        created_at=time.monotonic(),
    )
    accepted_queue: queue.Queue = queue.Queue(maxsize=1)
    full_queue: queue.Queue = queue.Queue(maxsize=1)
    full_queue.put_nowait(object())
    publisher = RtcPolicyEmissionPublisher(
        observers={
            "accepted_observer": accepted_queue,
            "dropping_observer": full_queue,
        }
    )
    binding = publisher.bind_frontend(frontend)
    assert publisher.active_binding is binding
    frontend.start()
    base = frontend.state.plan.cursor_at
    try:
        step = None
        for index in range(120):
            candidate = frontend.step(
                _observation(timestamp=55.0 + index / 30.0, step_index=index),
                now=base + index / 30.0,
            )
            if candidate.emission_result.emission is not None:
                step = candidate
                break
            time.sleep(0.001)
        assert step is not None
        emission = step.emission_result.emission
        commit = step.emission_commit_receipt
        assert emission is not None
        assert commit is not None
        assert commit.emission_semantic_content_hash == emission.semantic_content_hash
        publication_now = commit.committed_at
        assert publication_now <= emission.effective_until

        receipt = publisher.publish_committed_emission(
            emission,
            commit,
            now=publication_now,
        )
        assert receipt.accepted
        assert not receipt.idempotent_replay
        assert accepted_queue.get_nowait() is emission
        assert receipt.observer_outcomes[1].dropped

        replay = publisher.publish_committed_emission(
            emission,
            commit,
            now=publication_now,
        )
        assert replay.accepted
        assert replay.idempotent_replay
        assert accepted_queue.empty()
        assert replay.publication_key == receipt.publication_key

        mismatched_emission = replace(
            emission,
            semantic_content_hash="emission:forged-different-content",
        )
        mismatch = publisher.publish_committed_emission(
            mismatched_emission,
            commit,
            now=publication_now,
        )
        assert not mismatch.accepted
        assert "producer_commit_emission_hash_mismatch" in mismatch.reasons

        expired_publisher = RtcPolicyEmissionPublisher(
            observers={"expired_observer": queue.Queue(maxsize=1)}
        )
        expired_publisher.bind_frontend(frontend)
        expired = expired_publisher.publish_committed_emission(
            emission,
            commit,
            now=emission.effective_until + 1e-6,
        )
        assert not expired.accepted
        assert "emission_ttl_expired" in expired.reasons
    finally:
        frontend.close(now=base + 5.0)


def test_required_control_subscriber_failure_is_not_silently_dropped() -> None:
    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-required-control-publisher"),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        preparer=_pipeline(),
        created_at=time.monotonic(),
    )
    required_queue: queue.Queue = queue.Queue(maxsize=1)
    required_queue.put_nowait(object())
    observer_queue: queue.Queue = queue.Queue(maxsize=1)
    publisher = RtcPolicyEmissionPublisher(
        observers={"passive_observer": observer_queue},
        required_control_subscriber=("control_adapter", required_queue),
    )
    publisher.bind_frontend(frontend)
    frontend.start()
    base = frontend.state.plan.cursor_at
    try:
        step = None
        for index in range(120):
            candidate = frontend.step(
                _observation(timestamp=57.0 + index / 30.0, step_index=index),
                now=base + index / 30.0,
            )
            if candidate.emission_result.emission is not None:
                step = candidate
                break
            time.sleep(0.001)
        assert step is not None
        emission = step.emission_result.emission
        commit = step.emission_commit_receipt
        assert emission is not None
        assert commit is not None
        receipt = publisher.publish_committed_emission(
            emission,
            commit,
            now=emission.emitted_at,
        )
        assert not receipt.accepted
        assert receipt.required_control_outcome is not None
        assert not receipt.required_control_outcome.accepted
        assert not receipt.required_control_outcome.dropped
        assert observer_queue.empty()
        status = publisher.status()
        assert status["suspended"]
        assert status["accepted_publication_count"] == 0
        assert status["rejected_publication_count"] == 1
    finally:
        frontend.close(now=base + 5.0)


def test_cs3_passive_station_observes_committed_emission_without_owning_producer() -> None:
    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-cs3-passive"),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        preparer=_pipeline(),
        created_at=time.monotonic(),
    )
    bootstrap = RtcPolicyPassiveStationBootstrap(queue_capacity=16)
    binding = bootstrap.install_source_binding(
        build_rtc_policy_emission_source_binding(frontend)
    )
    observed: list[object] = []
    station = bootstrap.build_station_after_source_binding(
        observed.append,
        station_clock=time.monotonic,
        timeout_sec=0.0,
    )
    assert bootstrap.station is station
    assert station.consumer.emission_queue is bootstrap.emission_queue

    manifest = station.station.manifest
    assert manifest.manifest_id == CS3_RTC_POLICY_PASSIVE_SHADOW_MANIFEST_ID_V1
    assert manifest.execution_binding.target_kind == "null"
    assert manifest.source_slots[0].supported_uses == ("shadow",)
    assert "request_candidate" not in manifest.allowed_intent_kinds
    assert "request_motion_authority" not in manifest.allowed_intent_kinds
    assert "control_station_rtc_producer_lifecycle" in manifest.explicit_exclusions

    source_state_before = frontend.state.semantic_content_hash
    selected_before = frontend.selected_emission_count
    queue_pop_before = runtime.action_queue_pop_count()

    sequence = 0

    def submit(kind: str, subject_id: str | None = None):
        nonlocal sequence
        state = station.station.state
        transition = station.station.submit_intent(
            StationIntentDraft(
                client_id="operator:cs3-test",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=time.monotonic(),
                produced_clock_domain_id="test-monotonic",
                expected_state_revision=state.state_revision,
                manifest_hash=manifest.semantic_content_hash,
                subject_id=subject_id,
                reason="cs3 passive lifecycle fixture",
            )
        )
        sequence += 1
        completion = None
        if transition.receipt.status == "operation_requested":
            completion = station.complete_pending_operation(transition)
            assert completion.completion_transition.accepted
        return transition, completion

    target_id = manifest.execution_binding.binding_id
    source_id = binding.source_slot_id
    submit("acquire_control_lease")
    submit("connect_target", target_id)
    submit("connect_source", source_id)
    submit("run_preflight", target_id)
    submit("run_preflight", source_id)

    # Station setup starts only its passive queue consumer.  RTC plan, worker,
    # pop count, and selected-emission count remain untouched.
    assert frontend.state.semantic_content_hash == source_state_before
    assert frontend.selected_emission_count == selected_before
    assert runtime.action_queue_pop_count() == queue_pop_before

    frontend.start()
    base = frontend.state.plan.cursor_at
    emitting_step = None
    try:
        for index in range(120):
            candidate = frontend.step(
                _observation(timestamp=60.0 + index / 30.0, step_index=index),
                now=base + index / 30.0,
            )
            if candidate.action_preparation.emission is not None:
                emitting_step = candidate
                break
            time.sleep(0.001)
        assert emitting_step is not None
        emission = emitting_step.emission_result.emission
        assert emission is not None

        # A pre-activation item is ordered before the activation fence and is
        # therefore discarded as inactive, never replayed after start_shadow.
        bootstrap.emission_queue.put_nowait(emission)
        _wait_for(
            lambda: station.consumer.status().inactive_emission_count == 1
        )
        submit("start_shadow", source_id)
        bootstrap.emission_queue.put_nowait(emission)
        _wait_for(lambda: station.consumer.status().observed_emission_count == 1)
        assert observed == [emission]
        assert observed[0] is emission

        rtc_state_at_station_stop = frontend.state.semantic_content_hash
        rtc_selected_at_station_stop = frontend.selected_emission_count
        submit("stop_shadow", source_id)
        assert frontend.state.semantic_content_hash == rtc_state_at_station_stop
        assert frontend.selected_emission_count == rtc_selected_at_station_stop

        bootstrap.emission_queue.put_nowait(emission)
        _wait_for(
            lambda: station.consumer.status().inactive_emission_count == 2
        )
        assert observed == [emission]
        status = station.consumer.status()
        assert not status.shadow_enabled
        assert status.observed_emission_count == 1
        assert status.rejected_emission_count == 0
        assert status.error_count == 0

        rejected, _completion = submit("request_candidate", source_id)
        assert not rejected.receipt.accepted
        assert rejected.receipt.status == "intent_not_allowed_by_manifest"
    finally:
        station.close()
        frontend.close(now=base + 5.0)


def test_cs3_bootstrap_fails_closed_until_lower_source_binding_exists() -> None:
    bootstrap = RtcPolicyPassiveStationBootstrap(queue_capacity=2)
    with pytest.raises(TimeoutError, match="source binding"):
        bootstrap.build_station_after_source_binding(
            lambda _step: None,
            station_clock=time.monotonic,
            timeout_sec=0.0,
        )
    assert bootstrap.station is None


def test_cs3_loopback_api_only_coordinates_passive_shadow_lifecycle() -> None:
    token = "cs3a-passive-rtc-test-token-0123456789-abcdef"
    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-cs3-web"),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        preparer=_pipeline(),
        created_at=time.monotonic(),
    )
    bootstrap = RtcPolicyPassiveStationBootstrap(queue_capacity=8)
    bootstrap.install_source_binding(
        build_rtc_policy_emission_source_binding(frontend)
    )
    station = bootstrap.build_station_after_source_binding(
        lambda _step: None,
        station_clock=time.monotonic,
        timeout_sec=0.0,
    )
    event_log = BoundedStationEventLog(capacity=64)
    coordinator = SideEffectStationOperationCoordinator(
        coordinator_id="cs3a-passive-web-worker",
        runtime=station.station,
        adapters=station.adapters,
        operation_prepare=station.prepare_pending_operation,
        operation_executor=station.complete_pending_operation,
        event_log=event_log,
        queue_capacity=4,
    )
    policy = LocalIntentAccessPolicy(
        access_policy_id="cs3a-passive-web-v1",
        bearer_token=token,
        allowed_intent_kinds=station.station.manifest.allowed_intent_kinds,
    )
    server = build_scoped_passive_shadow_control_station_server(
        station.station,
        access_policy=policy,
        event_log=event_log,
        operation_coordinator=coordinator,
        passive_status_provider=station.passive_status_payload,
        port=0,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    coordinator.start()
    server_thread.start()
    host, port = server.server_address[:2]
    root = f"http://{host}:{port}"

    def get_state():
        with urlopen(f"{root}/api/state", timeout=2.0) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))

    sequence = 0

    def post(kind: str, subject_id: str | None = None, **extra):
        nonlocal sequence
        state = station.station.state
        payload = {
            "client_id": "browser:cs3a-test",
            "client_sequence": sequence,
            "intent_kind": kind,
            "produced_at": time.monotonic(),
            "produced_clock_domain_id": "browser-performance",
            "expected_state_revision": state.state_revision,
            "manifest_hash": station.station.manifest.semantic_content_hash,
            "reason": "cs3a passive web fixture",
            **extra,
        }
        if subject_id is not None:
            payload["subject_id"] = subject_id
        sequence += 1
        request = Request(
            f"{root}/api/intents",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            response = urlopen(request, timeout=2.0)  # noqa: S310
        except HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))
        with response:
            return response.status, json.loads(response.read().decode("utf-8"))

    try:
        initial = get_state()
        assert initial["maturity"] == "cs3a_rtc_policy_passive_shadow_api"
        assert initial["target"]["target_kind"] == "null"
        assert initial["target"]["motion_authorized"] is False
        assert initial["capabilities"]["browser_raw_action_submission"] is False

        target_id = station.station.manifest.execution_binding.binding_id
        source_id = station.source_adapter.subject_id
        for kind, subject_id in (
            ("acquire_control_lease", None),
            ("connect_target", target_id),
            ("connect_source", source_id),
            ("run_preflight", target_id),
            ("run_preflight", source_id),
            ("start_shadow", source_id),
        ):
            status, body = post(kind, subject_id)
            assert status == 200
            assert body["motion_authorized"] is False
            assert body["adapter_executed_inline"] is False
            assert coordinator.wait_until_idle(timeout_sec=1.0)

        active = get_state()
        assert active["sources"][0]["runtime_state"] == "shadow"
        assert active["control_facts"]["station_publication"][
            "shadow_enabled"
        ] is True
        assert active["target"]["motion_authorized"] is False

        status, body = post(
            "request_candidate",
            source_id,
        )
        assert status == 403
        assert body["error"] == "intent_not_enabled"

        status, body = post(
            "stop_shadow",
            source_id,
            action={"right_arm_qpos": [0.0] * 6},
        )
        assert status == 422
        assert body["error"] == "invalid_typed_intent"

        status, body = post("stop_shadow", source_id)
        assert status == 200
        assert body["motion_authorized"] is False
        assert coordinator.wait_until_idle(timeout_sec=1.0)
        assert not station.consumer.status().shadow_enabled
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2.0)
        assert coordinator.stop(timeout_sec=1.0)
        station.close()
        frontend.start()
        frontend.close(now=time.monotonic())


def test_virtual_policy_long_run_emits_each_timed_keyframe_once_without_lookahead() -> None:
    runtime = _runtime()
    emission_adapter = _pipeline().action_adapter
    frontend = RtcPolicyShadowFrontend(
        config=_config(),
        deploy_runtime=runtime,
        policy=_ChunkPolicy(),
        created_at=time.monotonic(),
        emission_adapter=emission_adapter,
    )
    frontend.start()
    emissions = []
    durations = []
    request_steps = []
    base = frontend.state.plan.cursor_at
    try:
        for step_index in range(240):
            now = base + step_index / 30.0
            step = frontend.step(
                _observation(timestamp=100.0 + step_index / 30.0, step_index=step_index),
                now=now,
            )
            durations.append(step.control_duration_sec)
            if (
                step.request_transition is not None
                and step.request_transition.receipt.accepted
            ):
                request_steps.append(step_index)
            if step.emission_result.emission is not None:
                emissions.append(step.emission_result.emission)
                assert step.emission_commit_transition is not None
                assert step.emission_commit_transition.receipt.accepted
                assert step.emission_result.raw_action is not None
                assert step.action_preparation.prepared_action is None
                assert step.action_preparation.raw_lookahead_action is None
                assert step.action_preparation.next_step_execution_signal is None
                assert (
                    step.emission_result.emission.emitted_at
                    >= step.emission_result.emission.produced_at
                )
                assert step.state.plan.producer_committed_through == (
                    step.emission_result.emission.target_effective_at
                )
                assert all(
                    keyframe.target_effective_at
                    > step.state.plan.producer_committed_through
                    for keyframe in step.state.plan.keyframes
                )
            else:
                assert step.emission_commit_transition is None
            time.sleep(0.0005)
    finally:
        frontend.close(now=base + 9.0)

    identities = [
        (
            emission.inference_id,
            emission.chunk_id,
            emission.chunk_keyframe_index,
        )
        for emission in emissions
    ]
    assert len(emissions) >= 20
    assert len(identities) == len(set(identities))
    assert frontend.selected_emission_count == len(emissions)
    assert len(request_steps) >= 5
    assert all(
        later - earlier >= 2
        for earlier, later in zip(request_steps, request_steps[1:])
    )
    assert runtime.action_queue_length() == 0
    assert runtime.action_queue_pop_count() == 0
    assert float(np.quantile(durations, 0.99)) < 1.0 / 30.0


def test_blocking_worker_never_blocks_control_step_and_late_completion_cannot_rearm() -> None:
    runtime = _runtime()
    policy = _ChunkPolicy(chunk_size=2)
    started = threading.Event()
    release = threading.Event()

    def executor_factory(_worker_session_id: str):
        decoded = RtcDecodedPolicyExecutor(
            policy=policy,
            deploy_runtime=runtime,
        )

        def execute(request, actor_observation):
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test worker release timed out")
            return decoded(request, actor_observation)

        return execute

    base = time.monotonic()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-blocking"),
        deploy_runtime=runtime,
        policy=policy,
        preparer=_pipeline(),
        created_at=base,
        executor_factory=executor_factory,
    )
    frontend.start()
    binding_before_rearm = build_rtc_policy_emission_source_binding(frontend)
    publisher = RtcPolicyEmissionPublisher(
        required_control_subscriber=(
            "control_adapter",
            queue.Queue(maxsize=2),
        )
    )
    assert publisher.bind_frontend(frontend) is binding_before_rearm
    first = frontend.step(_observation(timestamp=10.0, step_index=0), now=base)
    assert first.worker_submission is not None
    assert first.worker_submission.accepted
    assert started.wait(timeout=1.0)

    durations = []
    for index in range(1, 20):
        step = frontend.step(
            _observation(timestamp=10.0 + index / 30.0, step_index=index),
            now=base + index / 30.0,
        )
        durations.append(step.control_duration_sec)
        assert step.action_preparation.emission is None
    assert max(durations) < 0.02

    held = frontend.hold(now=base + 1.0, reason="operator_hold")
    publisher.suspend_publication(reason="producer_rearm")
    assert frontend.state.lifecycle_state == POLICY_ASYNC_PRODUCER_HOLD
    assert held.worker_stop_receipt.session_state == (
        POLICY_WORKER_SESSION_ABANDONED_UNDRAINED
    )
    with pytest.raises(RuntimeError, match="has not drained"):
        frontend.rearm(now=base + 1.01, reason="too_early")

    old_session = frontend.state.worker_session_id
    release.set()
    late_steps = []

    def late_completion_drained() -> bool:
        step = frontend.step(
            _observation(timestamp=11.1, step_index=100),
            now=base + 1.1,
            allow_output=False,
        )
        late_steps.append(step)
        return frontend.state.worker_session_state == POLICY_WORKER_SESSION_DRAINED

    _wait_for(late_completion_drained)
    late = next(step for step in late_steps if step.completion is not None)
    assert late.result_transition is not None
    assert not late.result_transition.receipt.accepted
    assert "producer_not_eligible" in late.result_transition.receipt.reasons
    assert frontend.late_completion_rejection_count == 1
    rearmed = frontend.rearm(now=base + 1.2, reason="explicit_rearm")
    assert rearmed.receipt.accepted
    assert frontend.state.worker_session_id != old_session
    binding_after_rearm = build_rtc_policy_emission_source_binding(frontend)
    assert binding_after_rearm.producer_epoch > binding_before_rearm.producer_epoch
    assert (
        binding_after_rearm.descriptor_hash
        == binding_before_rearm.descriptor_hash
    )
    assert (
        binding_after_rearm.semantic_content_hash
        != binding_before_rearm.semantic_content_hash
    )
    readiness = frontend.last_activation_readiness_receipt
    assert readiness is not None
    assert readiness.ready
    assert readiness.old_producer_epoch == binding_before_rearm.producer_epoch
    assert readiness.new_producer_epoch == binding_after_rearm.producer_epoch
    with pytest.raises(ValueError, match="ACK is missing"):
        publisher.activate_rebound_binding(
            binding_after_rearm,
            readiness_receipt=readiness,
            required_control_ack=None,
        )
    ack = build_rtc_policy_emission_binding_ack(
        binding_after_rearm,
        subscriber_id="control_adapter",
        accepted=True,
        acknowledged_at=base + 1.2,
    )
    assert (
        publisher.activate_rebound_binding(
            binding_after_rearm,
            readiness_receipt=readiness,
            required_control_ack=ack,
        )
        is binding_after_rearm
    )
    assert publisher.active_binding is binding_after_rearm
    assert not publisher.status()["suspended"]
    frontend.close(now=base + 1.3)


def test_expired_matching_result_settles_only_itself_and_reports_underrun() -> None:
    runtime = _runtime()
    policy = _ChunkPolicy(chunk_size=1)
    completed = threading.Event()

    def executor_factory(_worker_session_id: str):
        decoded = RtcDecodedPolicyExecutor(
            policy=policy,
            deploy_runtime=runtime,
        )

        def execute(request, actor_observation):
            result = decoded(request, actor_observation)
            completed.set()
            return result

        return execute

    base = time.monotonic()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-expired"),
        deploy_runtime=runtime,
        policy=policy,
        preparer=_pipeline(),
        created_at=base,
        executor_factory=executor_factory,
    )
    frontend.start()
    first = frontend.step(_observation(timestamp=20.0, step_index=0), now=base)
    assert first.underrun
    request = frontend.state.active_request
    assert request is not None
    assert completed.wait(timeout=1.0)
    _wait_for(lambda: frontend.pending_completion_count == 1)

    expired = frontend.step(
        _observation(timestamp=21.0, step_index=1),
        now=request.valid_until + 0.01,
    )
    assert expired.result_transition is not None
    assert not expired.result_transition.receipt.accepted
    assert "request_deadline_expired" in (
        expired.result_transition.receipt.reasons
    )
    debug_payload = expired.to_debug_payload()
    assert debug_payload["result_transition_accepted"] is False
    assert debug_payload["result_transition_reasons"] == [
        "request_deadline_expired"
    ]
    assert debug_payload["plan_merge_accepted"] is False
    assert debug_payload["plan_merge_reasons"] == ["request_deadline_expired"]
    assert debug_payload["plan_merge_admitted_keyframe_count"] == 0
    assert expired.action_preparation.emission is None
    assert expired.underrun
    frontend.close(now=request.valid_until + 0.02)


def test_worker_fault_enters_fault_and_requires_explicit_rearm() -> None:
    runtime = _runtime()
    policy = load_policy("hold")

    def executor_factory(_worker_session_id: str):
        def fail(_request, _actor_observation):
            raise RuntimeError("synthetic rtc fault")

        return fail

    base = time.monotonic()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-3c-fault"),
        deploy_runtime=runtime,
        policy=policy,
        preparer=_pipeline(),
        created_at=base,
        executor_factory=executor_factory,
    )
    frontend.start()
    frontend.step(_observation(timestamp=30.0, step_index=0), now=base)

    observed = []

    def faulted() -> bool:
        step = frontend.step(
            _observation(timestamp=30.1, step_index=1),
            now=base + 0.1,
        )
        observed.append(step)
        return frontend.state.lifecycle_state == POLICY_ASYNC_PRODUCER_FAULT

    _wait_for(faulted)
    fault_step = next(step for step in observed if step.completion is not None)
    assert fault_step.result_transition is not None
    assert "worker_result_fault" in fault_step.result_transition.receipt.reasons
    assert fault_step.lifecycle_transition is not None

    def fault_worker_drained() -> bool:
        frontend.step(
            _observation(timestamp=30.2, step_index=2),
            now=base + 0.2,
            allow_output=False,
        )
        return frontend.state.worker_session_state == POLICY_WORKER_SESSION_DRAINED

    _wait_for(fault_worker_drained)
    frontend.rearm(now=base + 0.2, reason="operator_rearm")
    frontend.close(now=base + 0.3)
