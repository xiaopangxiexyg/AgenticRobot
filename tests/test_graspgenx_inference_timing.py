"""Observational inference diagnostics; runtime/model boundaries are mocked."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from deployments.perception import graspgenx_client as client_module

_SWEEP = {
    "extents_open": (0.1, 0.03, 0.1),
    "offset_open": (0.0, 0.0, 0.02),
    "extents_mid": (0.07, 0.03, 0.1),
    "offset_mid": (0.0, 0.0, 0.02),
}
_POINTS = np.zeros((16, 3), dtype=np.float32)


def _records(caplog) -> list[dict[str, object]]:
    return [
        json.loads(record.getMessage().split(": ", 1)[1])
        for record in caplog.records
        if record.name == client_module.__name__
        and record.getMessage().startswith("GraspGenX inference diagnostics: ")
    ]


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _Runtime:
    """No torch import, CUDA initialization, network, or model execution."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.sampler_builds = 0
        self.runner_calls: list[dict[str, object]] = []
        self.device_entries = 0
        self.cpu_transfers = 0
        self.locked = False
        self.lock_entries = self.lock_exits = 0
        self.coerce_error = None
        self.runner_error = None
        self.model = object()
        self.grasps = np.repeat(np.eye(4)[None], 4, axis=0)
        self.grasps[:, 0, 3] = np.arange(4) * 0.01
        self.scores = np.array([0.8, 0.4, 0.8, 0.1])
        self.tags = ["diffusion", "obb_top", "obb_side", "obb_top"]

    def lock(self):
        runtime = self

        class _Lock:
            def __enter__(self):
                runtime.clock.advance(0.06)
                assert not runtime.locked
                runtime.locked = True
                runtime.lock_entries += 1

            def __exit__(self, *_args):
                runtime.lock_exits += 1
                runtime.locked = False
                return False

        return _Lock()

    def tensor(self, value, delay):
        runtime = self

        class _Tensor:
            def detach(self):
                return self

            def cpu(self):
                assert not runtime.locked
                runtime.cpu_transfers += 1
                runtime.clock.advance(delay)
                return self

            def numpy(self):
                return value

        return _Tensor()

    def load(self):
        runtime = self

        class _CudaContext:
            def __enter__(self):
                assert runtime.locked
                runtime.device_entries += 1
                runtime.clock.advance(0.02)

            def __exit__(self, *_args):
                runtime.clock.advance(0.03)
                return False

        class _Params:
            @classmethod
            def coerce(cls, _value):
                assert runtime.locked
                runtime.clock.advance(0.1)
                if runtime.coerce_error is not None:
                    raise runtime.coerce_error
                return cls()

            def cache_key(self):
                return "fixed-sweep"

        class _Sampler:
            @classmethod
            def from_sweep_volume(cls, _cfg, _params, *, model):
                assert runtime.locked and model is runtime.model
                runtime.sampler_builds += 1
                runtime.clock.advance(0.2)
                return cls()

        def runner(points, sampler, **kwargs):
            assert runtime.locked and isinstance(sampler, _Sampler)
            np.testing.assert_array_equal(points, _POINTS)
            runtime.runner_calls.append(kwargs)
            runtime.clock.advance(0.3)
            if runtime.runner_error is not None:
                raise runtime.runner_error
            return (
                runtime.tensor(runtime.grasps, 0.4),
                runtime.tensor(runtime.scores, 0.5),
                runtime.tags, None,
            )

        # No synchronize method exists: adding one to inference breaks the test.
        torch = SimpleNamespace(
            cuda=SimpleNamespace(device=lambda _device: _CudaContext())
        )
        return (
            torch, "cuda", "cfg", runtime.model, _Sampler, _Params, runner, False,
        )


@pytest.fixture
def backend(tmp_path, monkeypatch, caplog):
    clock = _Clock()
    runtime = _Runtime(clock)
    model_spec = SimpleNamespace(
        semantic_content_hash="0" * 64, bundle_root=tmp_path,
        checkpoint_root=tmp_path,
    )
    backend = client_module.GraspGenXInProcessBackend(model_spec=model_spec)
    monkeypatch.setattr(client_module, "perf_counter", clock)
    monkeypatch.setattr(
        backend, "_runtime_verifier", lambda **_kwargs: SimpleNamespace(passed=True)
    )
    monkeypatch.setattr(backend, "_load_runtime", runtime.load)
    backend.start()
    monkeypatch.setattr(backend, "_lock", runtime.lock())
    caplog.set_level(logging.INFO, logger=client_module.__name__)
    try:
        yield backend, runtime, clock
    finally:
        backend.close()


def test_phase_times_and_sources_preserve_stable_top_k_and_sampler_reuse(
    backend, monkeypatch, caplog,
) -> None:
    inference, runtime, clock = backend
    original_contiguous = np.ascontiguousarray
    original_argsort = np.argsort

    def contiguous(value, *args, **kwargs):
        if value is _POINTS:
            clock.advance(0.01)
        return original_contiguous(value, *args, **kwargs)

    def argsort(*args, **kwargs):
        clock.advance(0.07)
        return original_argsort(*args, **kwargs)

    monkeypatch.setattr(np, "ascontiguousarray", contiguous)
    monkeypatch.setattr(np, "argsort", argsort)
    semantic_before = dict(inference.semantic_config)
    expected = runtime.grasps[[0, 2]]
    for _ in range(2):
        actual, scores = inference.infer_object(
            _POINTS, sweep_volume_params=_SWEEP, score_threshold=0.3, top_k=2,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scores, [0.8, 0.8])
        assert actual.dtype == scores.dtype == np.float64
        assert actual.flags.c_contiguous and scores.flags.c_contiguous

    first, second = _records(caplog)
    assert first["status"] == second["status"] == "ok"
    assert first["phase_wall_sec"] == pytest.approx({
        "validation": 0.01, "lock_wait": 0.06,
        "sampler_preparation": 0.3, "planner_call": 0.35,
        "output_conversion": 0.9, "top_k_postprocess": 0.07,
        "diagnostic_summary": 0.0,
    })
    assert second["phase_wall_sec"]["sampler_preparation"] == pytest.approx(0.1)
    assert first["total_wall_sec"] == pytest.approx(1.69)
    assert second["total_wall_sec"] == pytest.approx(1.49)
    assert first["timing_basis"] == "host_wall_without_added_cuda_synchronization"
    assert first["sampler_created"] is True and second["sampler_created"] is False
    assert first["branch_tags"] == {
        "status": "available",
        "raw_counts": {"diffusion": 1, "obb_top": 2, "obb_side": 1},
        "after_threshold_counts": {"diffusion": 1, "obb_top": 1, "obb_side": 1},
        "returned_counts": {"diffusion": 1, "obb_side": 1},
    }
    assert (first["raw_candidate_count"], first["after_threshold_count"],
            first["returned_candidate_count"]) == (4, 3, 2)
    assert runtime.sampler_builds == 1 and len(runtime.runner_calls) == 2
    assert runtime.device_entries == 2 and runtime.cpu_transfers == 4
    assert runtime.lock_entries == runtime.lock_exits == 2
    assert all(call["grasp_threshold"] == -1.0 for call in runtime.runner_calls)
    assert all(call["topk_num_grasps"] == -1 for call in runtime.runner_calls)
    assert dict(inference.semantic_config) == semantic_before


@pytest.mark.parametrize("planner", ["graspmoe", "diffusion"])
@pytest.mark.parametrize("tags", [None, [], np.array([0, 1, 0, 1])])
def test_unknown_sources_do_not_change_all_candidate_return_or_invent_counts(
    backend, caplog, tags, planner,
) -> None:
    inference, runtime, _clock = backend
    runtime.tags = tags
    actual, scores = inference.infer_object(
        _POINTS, sweep_volume_params=_SWEEP, top_k=-1, planner=planner,
    )
    np.testing.assert_array_equal(actual, runtime.grasps[[0, 2, 1, 3]])
    np.testing.assert_array_equal(scores, runtime.scores[[0, 2, 1, 3]])
    record = _records(caplog)[0]
    assert record["branch_tags"]["status"] == "unavailable"
    assert "raw_counts" not in record["branch_tags"]
    expected = -1 if planner == "graspmoe" else 200
    assert runtime.runner_calls[0]["topk_num_grasps"] == expected


def test_diagnostic_tags_never_transfer_device_data(backend, caplog) -> None:
    inference, runtime, _clock = backend

    class _DeviceTags:
        def detach(self):
            pytest.fail("diagnostics must not detach or transfer branch tags")

        def __array__(self, *_args, **_kwargs):
            pytest.fail("diagnostics must not coerce a device-like tag object")

    runtime.tags = _DeviceTags()
    actual, _scores = inference.infer_object(
        _POINTS, sweep_volume_params=_SWEEP, top_k=-1,
    )
    assert len(actual) == 4
    assert _records(caplog)[0]["branch_tags"] == {
        "status": "unavailable", "reason": "unsupported_host_labels",
    }
    assert runtime.cpu_transfers == 2


@pytest.mark.parametrize("failure_phase", [
    "validation", "sampler_preparation", "planner_call", "output_conversion",
    "top_k_postprocess",
])
def test_failures_retain_exception_and_incomplete_phase_context(
    backend, monkeypatch, caplog, failure_phase,
) -> None:
    inference, runtime, _clock = backend
    sentinel = RuntimeError("retained failure context")
    points = _POINTS
    if failure_phase == "validation":
        points = np.zeros((1, 3))
    elif failure_phase == "sampler_preparation":
        runtime.coerce_error = sentinel
    elif failure_phase == "planner_call":
        runtime.runner_error = sentinel
    elif failure_phase == "output_conversion":
        runtime.scores = [object()]
    else:
        def fail_sort(*_args, **_kwargs):
            raise sentinel

        monkeypatch.setattr(np, "argsort", fail_sort)
    with pytest.raises((ValueError, RuntimeError)) as captured:
        inference.infer_object(points, sweep_volume_params=_SWEEP)
    if failure_phase in {"sampler_preparation", "planner_call", "top_k_postprocess"}:
        assert captured.value is sentinel
    if failure_phase == "output_conversion":
        assert isinstance(captured.value.__cause__, TypeError)
        assert str(captured.value) == "GraspGenX returned malformed confidence scores."
    record = _records(caplog)[0]
    assert record["status"] == "failed"
    assert record["failed_phase"] == failure_phase
    assert record["exception_type"] == type(captured.value).__name__
    assert record["phase_wall_sec"][failure_phase] is not None
    assert record["phase_wall_sec"]["diagnostic_summary"] is None
    assert runtime.lock_entries == runtime.lock_exits
    assert not runtime.locked


def test_interrupt_is_logged_and_not_swallowed(backend, caplog) -> None:
    inference, runtime, _clock = backend
    interruption = KeyboardInterrupt("cancelled by operator")
    runtime.runner_error = interruption
    with pytest.raises(KeyboardInterrupt) as captured:
        inference.infer_object(_POINTS, sweep_volume_params=_SWEEP)
    assert captured.value is interruption
    record = _records(caplog)[0]
    assert record["failed_phase"] == "planner_call"
    assert record["exception_type"] == "KeyboardInterrupt"
    assert not runtime.locked


def test_mismatched_outputs_keep_counts_and_original_failure(backend, caplog):
    inference, runtime, _clock = backend
    runtime.scores = np.array([0.8])
    with pytest.raises(RuntimeError, match="grasp and confidence counts differ"):
        inference.infer_object(_POINTS, sweep_volume_params=_SWEEP)
    record = _records(caplog)[0]
    assert record["raw_candidate_count"] == 4
    assert record["converted_score_count"] == 1
    assert record["failed_phase"] == "output_conversion"
    assert record["phase_wall_sec"]["top_k_postprocess"] is None


def test_empty_output_still_returns_empty_arrays_and_exact_empty_counts(
    backend, caplog,
):
    inference, runtime, _clock = backend
    runtime.grasps = np.empty((0, 4, 4))
    runtime.scores = np.empty(0)
    runtime.tags = []
    actual, scores = inference.infer_object(_POINTS, sweep_volume_params=_SWEEP)
    assert actual.shape == (0, 4, 4) and scores.shape == (0,)
    record = _records(caplog)[0]
    assert record["status"] == "ok"
    assert record["raw_candidate_count"] == record["returned_candidate_count"] == 0
    assert record["branch_tags"] == {
        "status": "available", "raw_counts": {},
        "after_threshold_counts": {}, "returned_counts": {},
    }


@pytest.mark.parametrize(("tags", "count", "reason"), [
    (np.array([["diffusion"]]), 1, "non_vector_labels"),
    (["x" * 65], 1, "unsupported_host_labels"),
    ([f"branch-{index}" for index in range(33)], 33, "distinct_label_limit"),
    (["diffusion"] * 32_769, 32_769, "label_count_limit"),
])
def test_branch_observation_has_fixed_work_and_record_bounds(tags, count, reason):
    summary = client_module._branch_tag_summary(
        tags, np.ones(count, dtype=bool), np.empty(0, dtype=int),
    )
    assert summary == {"status": "unavailable", "reason": reason}


def test_valid_host_tag_vectors_and_empty_union_have_exact_counts():
    for tags in (["diffusion", "obb"], ("diffusion", "obb"),
                 np.array(["diffusion", "obb"], dtype="U")):
        summary = client_module._branch_tag_summary(
            tags, np.array([True, False]), np.array([0]),
        )
        assert summary["raw_counts"] == {"diffusion": 1, "obb": 1}
        assert summary["returned_counts"] == {"diffusion": 1}
    assert client_module._branch_tag_summary(
        [], np.empty(0, dtype=bool), np.empty(0, dtype=int),
    ) == {
        "status": "available", "raw_counts": {},
        "after_threshold_counts": {}, "returned_counts": {},
    }


class _FailingOutputHandler(logging.Handler):
    def emit(self, record):
        raise OSError("diagnostic sink unavailable")


def _infer_with_original_outcome(inference, runtime, outcome):
    original = {"normal": None, "error": ValueError("original model failure"),
                "interrupted": KeyboardInterrupt("original operator interruption")}[outcome]
    runtime.runner_error = original
    if original is None:
        grasps, scores = inference.infer_object(
            _POINTS, sweep_volume_params=_SWEEP, top_k=2)
        np.testing.assert_array_equal(grasps, runtime.grasps[[0, 2]])
        np.testing.assert_array_equal(scores, [.8, .8])
        assert runtime.cpu_transfers == 2
    else:
        with pytest.raises(type(original)) as caught:
            inference.infer_object(_POINTS, sweep_volume_params=_SWEEP, top_k=2)
        assert caught.value is original
        assert runtime.cpu_transfers == 0
    assert runtime.lock_entries == runtime.lock_exits == 1 and not runtime.locked


@pytest.mark.parametrize("outcome", ("normal", "error", "interrupted"))
@pytest.mark.parametrize("fault", ("handler", "serialization"))
def test_diagnostic_output_failure_preserves_primary_outcome(
    backend, monkeypatch, capsys, outcome, fault,
):
    inference, runtime, _clock = backend
    logger = logging.getLogger(client_module.__name__)
    handler = _FailingOutputHandler()
    if fault == "handler":
        logger.addHandler(handler)
    else:
        def fail_serialization(*_args, **_kwargs):
            raise ValueError("diagnostic JSON unavailable")
        monkeypatch.setattr(client_module, "json", SimpleNamespace(dumps=fail_serialization))
    try:
        _infer_with_original_outcome(inference, runtime, outcome)
        assert "GraspGenX inference diagnostics unavailable" in capsys.readouterr().err
    finally:
        logger.removeHandler(handler)


@pytest.mark.parametrize("outcome", ("normal", "error", "interrupted"))
@pytest.mark.parametrize("fallback", ("absent", "broken"))
def test_both_optional_output_sinks_unavailable_preserve_primary_outcome(
    backend, monkeypatch, outcome, fallback,
):
    inference, runtime, _clock = backend
    logger = logging.getLogger(client_module.__name__)
    handler = _FailingOutputHandler()
    logger.addHandler(handler)
    def broken_write(_text):
        raise OSError("stderr unavailable")
    stream = None if fallback == "absent" else SimpleNamespace(write=broken_write)
    try:
        with monkeypatch.context() as context:
            context.setattr(client_module.sys, "stderr", stream)
            _infer_with_original_outcome(inference, runtime, outcome)
    finally:
        logger.removeHandler(handler)


def test_new_operator_interruption_during_logging_is_not_swallowed(backend):
    inference, runtime, _clock = backend
    interruption = KeyboardInterrupt("new operator interruption")
    class InterruptingHandler(logging.Handler):
        def emit(self, record):
            raise interruption
    logger = logging.getLogger(client_module.__name__)
    handler = InterruptingHandler()
    logger.addHandler(handler)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            inference.infer_object(_POINTS, sweep_volume_params=_SWEEP)
        assert caught.value is interruption
        assert runtime.lock_entries == runtime.lock_exits == 1 and not runtime.locked
    finally:
        logger.removeHandler(handler)
