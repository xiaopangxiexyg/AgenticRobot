"""Non-executable RTC preflight and worker-session handoff regressions."""
from __future__ import annotations

import time
import threading

import pytest

from robot_motion.deploy import POLICY_ASYNC_PRODUCER_FAULT
from teleop_runtime.deploy.rtc_frontend import RtcDecodedPolicyExecutor, RtcPolicyShadowFrontend
from test_deploy_rtc_frontend_shadow import (
    _ChunkPolicy, _config, _observation, _pipeline, _runtime, _wait_for,
)
from test_openpi_rtc_final_chunk_adapter import (
    _conditioned_artifact_and_metadata, _conditioned_base_backend,
    _conditioned_bundle_and_request, _CounterClock,
)
from deployments.openpi.client.rtc_conditioned import (
    OpenPIConditionedRtcBackendV3, OpenPIConditionedRtcConfig,
)
from robot_motion.deploy.rtc import RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1


def _frontend(*, executor_factory=None):
    runtime = _runtime()
    frontend = RtcPolicyShadowFrontend(
        config=_config(session="rtc-preflight-test"), deploy_runtime=runtime,
        policy=_ChunkPolicy(), preparer=_pipeline(), created_at=time.monotonic(),
        executor_factory=executor_factory,
    )
    frontend.start()
    return frontend


def test_preflight_outlives_action_ttl_without_merging_and_reuses_worker():
    frontend = _frontend()
    base = frontend.state.plan.cursor_at
    worker_id = frontend.state.worker_session_id
    try:
        for index in range(4):
            now = base + index * 10.0
            submitted = frontend.step(
                _observation(timestamp=now, step_index=index * 2), now=now,
                allow_output=False, allow_request=True, preflight_timeout_sec=60.0,
            )
            assert submitted.worker_submission.accepted
            assert submitted.request_transition is None
            assert frontend.state.active_request is None
            _wait_for(lambda: frontend.pending_completion_count == 1)
            observed = frontend.step(
                _observation(timestamp=now + 9.0, step_index=index * 2 + 1),
                now=now + 9.0, allow_output=False, allow_request=True, preflight_timeout_sec=60.0,
            )
            assert observed.preflight_completion
            assert observed.result_transition is None
            assert observed.emission_result.emission is None
            assert observed.state.plan.keyframes == ()
            assert observed.state.plan.in_flight_request_id is None
            assert observed.state.worker_session_id == worker_id
        assert frontend.selected_emission_count == 0
        formal = frontend.step(
            _observation(timestamp=base + 40.0, step_index=9), now=base + 40.0,
        )
        assert formal.request_transition.receipt.accepted
        request = formal.state.active_request
        assert request.observation_step_index == 9
        assert request.valid_until - request.first_target_effective_at == pytest.approx(
            frontend.config.effective_request_validity_steps(request.forecast_delay_steps)
            * request.action_period_sec
        )
        _wait_for(lambda: frontend.pending_completion_count == 1)
        expired = frontend.step(
            _observation(timestamp=base + 41.0, step_index=10),
            now=request.valid_until + 0.01, allow_request=False,
        )
        assert not expired.result_transition.receipt.accepted
        assert "request_deadline_expired" in expired.result_transition.receipt.reasons
        assert expired.emission_result.emission is None
    finally:
        frontend.close()


def test_pending_preflight_cannot_become_executable_and_timeout_is_bounded():
    started, release = threading.Event(), threading.Event()

    def factory(_session):
        def execute(_request, _actor):
            started.set()
            release.wait(timeout=2.0)
            raise RuntimeError("synthetic cold-start failure")
        return execute

    frontend = _frontend(executor_factory=factory)
    base = frontend.state.plan.cursor_at
    try:
        frontend.step(_observation(timestamp=base, step_index=0), now=base,
                      allow_output=False, allow_request=True, preflight_timeout_sec=0.5)
        assert started.wait(timeout=1.0)
        with pytest.raises(ValueError, match="preflight must finish"):
            frontend.step(_observation(timestamp=base, step_index=1), now=base)
        expired = frontend.step(
            _observation(timestamp=base + 0.6, step_index=2), now=base + 0.6,
            allow_output=False, allow_request=True, preflight_timeout_sec=0.5,
        )
        assert expired.state.lifecycle_state == POLICY_ASYNC_PRODUCER_FAULT
        assert expired.to_debug_payload()["lifecycle_transition_reason"] == "rtc_preflight_timeout"
        assert not expired.preflight_completion
        assert expired.control_duration_sec < 0.1
        assert expired.emission_result.emission is None
    finally:
        release.set()
        frontend.close()


def test_preflight_worker_fault_is_not_a_latency_sample_and_summary_explains_it():
    def factory(_session):
        def execute(_request, _actor):
            raise RuntimeError("transport session refused " + "x" * 5000)
        return execute
    frontend = _frontend(executor_factory=factory)
    base = frontend.state.plan.cursor_at
    try:
        frontend.step(_observation(timestamp=base, step_index=0), now=base,
                      allow_output=False, allow_request=True, preflight_timeout_sec=60.0)
        _wait_for(lambda: frontend.pending_completion_count == 1)
        failed = frontend.step(
            _observation(timestamp=base + 0.1, step_index=1), now=base + 0.1,
            allow_output=False, allow_request=True, preflight_timeout_sec=60.0,
        )
        assert not failed.preflight_completion
        assert failed.state.lifecycle_state == POLICY_ASYNC_PRODUCER_FAULT
        payload = failed.to_debug_payload()
        assert payload["completion_fault_code"] == "policy_worker_exception:RuntimeError"
        assert payload["completion_fault_detail"].startswith("transport session refused")
        assert len(payload["completion_fault_detail"]) <= 512
        assert payload["plan_merge_admitted_keyframe_count"] == 0
    finally:
        frontend.close()


def test_conditioned_session_recovery_runs_on_worker_and_checks_false():
    artifact, metadata = _conditioned_artifact_and_metadata(inference_only_v2=True)
    base, transport = _conditioned_base_backend(artifact, metadata)
    calls = []
    transport.begin_session = lambda: calls.append(threading.get_ident()) or False
    conditioned = OpenPIConditionedRtcBackendV3(
        base_backend=base, config=OpenPIConditionedRtcConfig.from_artifact(
            artifact, selected_prefix_schedule=RTC_PREFIX_SCHEDULE_EXPLICIT_SELECTED_STEPS_V1,
        ),
        clock=_CounterClock(),
    )
    executor = conditioned.create_worker_executor()
    assert calls == []  # Factory runs on the control thread without network wait.
    _contract, _bundle, request = _conditioned_bundle_and_request(base)
    errors = []
    def run():
        try:
            executor.prepare_input(request, _observation(timestamp=1.0, step_index=0))
        except RuntimeError as error:
            errors.append(str(error))
    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert calls and calls[0] != threading.get_ident()
    assert "declined a new worker session" in errors[0]
