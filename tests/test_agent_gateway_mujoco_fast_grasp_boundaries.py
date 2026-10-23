"""Offline native consumers with fake inference and controlled failure boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_agent_gateway_mujoco_fast_grasp import CLIENT, _capture, _plan_request, _session

from agent_gateway.profiles.mujoco import grasp_fast_plan
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected


@pytest.mark.integration
@pytest.mark.parametrize("stage", ["primary", "auxiliary"])
@pytest.mark.parametrize("failure", ["budget", "cancel", "backend"])
def test_native_preparation_uses_caller_guard_without_publishing_or_holding(
    tmp_path, monkeypatch, stage, failure,
):
    session = _session(tmp_path)
    app = session.application
    try:
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        source, pixel, _ = _capture(session, adapter)
        request = _plan_request(session, adapter, source, pixel)
        operation = SimpleNamespace(operation_id="planning:guard", deadline_at=float(app.runtime.clock()) + 100)
        provider = session.fast_grasp_provider
        elapsed, cancelled, calls = [0.0], [False], []
        provider.monotonic = lambda: elapsed[0]

        def transition():
            calls.append(stage)
            if failure == "backend":
                raise LookupError("injected inference failure")
            elapsed[0] = 20.0
            cancelled[0] = failure == "cancel"

        def segment(*args, **kwargs):
            transition()
            return np.ones((2, 2), dtype=bool)

        def fuse(**kwargs):
            kwargs["segment_point"](object(), pixel_uv=(0, 0))
            pytest.fail("An expired/cancelled auxiliary call returned to fusion")

        def plan(image, depth, **kwargs):
            kwargs["preparation_guard"]()
            if stage == "primary":
                transition()
                kwargs["preparation_guard"]()
            else:
                kwargs["visible_scene_factory"](
                    depth_m=depth, target_mask=np.ones(depth.shape, dtype=bool),
                    intrinsics={}, geometry=None,
                )
            pytest.fail("Preparation crossed its original caller boundary")

        monkeypatch.setattr(provider.model, "plan", plan)
        monkeypatch.setattr(provider.model.point_segmenter, "segment_point", segment)
        monkeypatch.setattr(grasp_fast_plan, "fuse_grasp_views", fuse)
        monkeypatch.setattr(app, "request_hold", lambda **kwargs: pytest.fail("Read-only preparation called HOLD"))
        before = app.capture_grasp_scene_state()
        if failure == "cancel":
            with pytest.raises(MujocoTcpPlanRejected, match="safe_tcp_planning_cancelled"):
                provider.plan_request(request, operation, cancelled=lambda: cancelled[0])
        elif failure == "backend":
            with pytest.raises(LookupError, match="injected inference failure"):
                provider.plan_request(request, operation)
        else:
            result = provider.plan_request(request, operation).execution_result
            assert result["planning_status"] == "rejected"
            assert result["reason_code"] == "grasp_fast_search_budget_exhausted"
            assert result["grasp_approach_plan_id"] is None and not result["eligible_for_execution"]
        assert calls == [stage]
        assert provider._retained is None and not app.target.planned_tcp_motion.active
        assert app.capture_grasp_scene_state() == before
        assert not provider.model.client.calls
    finally:
        monkeypatch.undo()
        assert session.close().accepted


@pytest.mark.integration
def test_fast_resources_remain_open_until_passive_reads_quiesce(tmp_path, monkeypatch):
    session = _session(tmp_path)
    model = session.fast_grasp_planner
    try:
        session.application.start_observation_ready()
        session.start_perception_resources()
        monkeypatch.setattr(session.fast_grasp_executor, "wait_all", lambda **kwargs: False)
        with pytest.raises(RuntimeError, match="fast_grasp reads did not quiesce"):
            session.close()
        assert not model.client.closed_for_test and model.point_segmenter.closes == 0
        assert not session.application.closed
        with pytest.raises(RuntimeError, match="Closed Gateway"):
            session.start_perception_resources()
    finally:
        monkeypatch.undo()
        assert session.close().accepted
        assert model.client.closed_for_test and model.point_segmenter.closes == 1


@pytest.mark.integration
@pytest.mark.parametrize("startup_error", [LookupError, KeyboardInterrupt])
def test_partial_fast_startup_cleanup_failure_is_owned_and_retryable(tmp_path, monkeypatch, startup_error):
    session = _session(tmp_path)
    model = session.fast_grasp_planner
    close_calls = []
    close = model.client.close

    def fail_start():
        raise startup_error("injected model startup failure")

    def fail_close_once():
        close_calls.append(True)
        if len(close_calls) == 1:
            raise OSError("injected startup cleanup failure")
        close()

    try:
        session.application.start_observation_ready()
        monkeypatch.setattr(model.client, "start", fail_start)
        monkeypatch.setattr(model.client, "close", fail_close_once)
        with pytest.raises(RuntimeError, match="startup cleanup failed") as error:
            session.start_perception_resources()
        assert isinstance(error.value.__cause__, startup_error)
        assert not session._fast_grasp_model_closed and session._fast_grasp_model_started
        assert model.point_segmenter.starts == 1
        with pytest.raises(RuntimeError, match="Failed fast grasp startup"):
            session.start_perception_resources()
        assert session.close().accepted
        assert session.close().accepted
        assert len(close_calls) == 2 and model.point_segmenter.closes == 1
    finally:
        monkeypatch.undo()
        assert session.close().accepted
