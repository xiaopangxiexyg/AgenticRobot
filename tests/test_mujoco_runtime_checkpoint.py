"""Actual runtime watermarks at a caller-fenced manual target boundary."""

from dataclasses import replace

import pytest

from sim.mujoco.deploy.runtime_checkpoint import capture_runtime
from tests.test_mujoco_agent_control_checkpoint import _native_target
from tests.test_rm65_ik_checkpoint import _move

pytestmark = [pytest.mark.integration, pytest.mark.slow_sim]


@pytest.fixture(name="source")
def target_source():
    config, prepared, profile, clock, target = _native_target()
    try:
        yield config, prepared, profile, clock, target
    finally:
        target.runtime.close()


@pytest.mark.parametrize("moved", (False, True))
def test_runtime_captures_complete_status_and_nonzero_cursors(source, moved):
    _config, _prepared, profile, clock, target = source
    if moved:
        _move((target, profile, clock))
    else:
        target.runtime.start()
    target.runtime.capture_policy_observation()
    target.runtime.capture_policy_observation(policy_node_id="review")
    checkpoint = capture_runtime(target.runtime)
    assert len(checkpoint.observations) == 2
    assert checkpoint.state["_next_direct_component_proposal_sequence"] == int(moved)
    assert [fact.semantic_content_hash for fact in checkpoint.core.facts] == [
        fact.semantic_content_hash for fact in target.runtime.core.trace_facts
    ]


@pytest.mark.parametrize(
    "field", ("latest", "context", "reference", "observation", "receipt", "history")
)
def test_journal_rejects_cache_and_sequence_rewind(source, field):
    _config, _prepared, profile, clock, target = source
    _move((target, profile, clock))
    runtime = target.runtime
    runtime.capture_policy_observation()
    saved = capture_runtime(runtime)
    if field == "latest":
        runtime._latest_target_execution_step = None
    elif field == "context":
        runtime._last_dispatch_context = None
    elif field == "reference":
        runtime._immediate_target_by_reference_sequence = {}
    elif field == "observation":
        runtime.observation_adapter._sequence = 0
    elif field == "receipt":
        latest = runtime._latest_target_execution_step
        dispatch = latest.target_dispatch_result
        receipts = {
            name: replace(value, status="fabricated")
            for name, value in dispatch.dispatch_receipts.items()
        }
        runtime._latest_target_execution_step = replace(
            latest, target_dispatch_result=replace(dispatch, dispatch_receipts=receipts)
        )
    else:
        runtime._immediate_component_targets = []
        runtime._immediate_component_target_admissions = []
        runtime._next_immediate_target_admission_sequence = 0
        runtime._projected_reference_fact_ids = set()
        runtime._reference_trace_cursor_by_stream = {}
    with pytest.raises(ValueError):
        capture_runtime(runtime)
    # Validation never truncates or repairs the authoritative journal.
    assert [fact.semantic_content_hash for fact in runtime.core.trace_facts] == [
        fact.semantic_content_hash for fact in saved.core.facts
    ]
