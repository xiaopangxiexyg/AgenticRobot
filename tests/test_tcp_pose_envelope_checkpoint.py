"""Continuation of the original lower envelope, without replaying lifecycle."""

from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from robot_motion.target_safety import (
    TCPPoseEnvelopeLifecycleError,
    TCPPoseEnvelopeSafetyRuntime,
)
from robot_motion.target_safety.checkpoint import (
    _copy,
    _hash,
    capture_safety_envelope,
    restore_safety_envelope,
)
from tests.test_tcp_pose_envelope_safety_runtime import (
    ROUTE_ID,
    _entry,
    _pose,
    _runtime,
    _selection,
)

pytestmark = pytest.mark.unit


def _source(*, relative=True):
    entry = _entry("envelope.standard", 0.2)
    if not relative:
        profile = replace(
            entry.profile,
            side_config_by_side={
                "right": replace(
                    entry.profile.side_config_by_side["right"],
                    relative_min_xyz_m=None,
                    relative_max_xyz_m=None,
                    max_rotation_rad=None,
                )
            },
        )
        entry = replace(entry, profile=profile)
    runtime = _runtime(entry, _entry("envelope.strict", 0.05))
    for _ in range(2):
        runtime.invalidate_reference(
            reason="source-reset", source_event_ids=("reset",), lower_state="hold"
        )
    if relative:
        runtime.commit_reference(
            reference_id="original-measured-reference",
            tcp_pose_by_side={"right": _pose(0.1)},
            captured_at_by_side={"right": 2.5},
            source_event_ids=("original-initialization",),
            lower_state="hold",
        )
    runtime.activate(
        selection=_selection(entry, safety_profile_id="caller-selection"),
        operation_id="original-operation",
        route_id=ROUTE_ID,
        execution_epoch=7,
        source_event_ids=("original-ready-transition",),
        lower_state="hold",
    )
    runtime.project({"right": _pose(0.6)})
    return runtime


@pytest.mark.parametrize("relative", (False, True))
def test_projection_continues_with_original_reference_and_no_lifecycle_replay(
    monkeypatch, relative
):
    source = _source(relative=relative)
    saved = capture_safety_envelope(source, captured_at=10.0)

    def no_lifecycle(*args, **kwargs):
        raise AssertionError("Restoration must not replay a lifecycle transition.")

    with monkeypatch.context() as patch:
        for name in (
            "commit_reference",
            "invalidate_reference",
            "activate",
            "deactivate_after_hold",
        ):
            patch.setattr(TCPPoseEnvelopeSafetyRuntime, name, no_lifecycle)
        branch = restore_safety_envelope(saved, captured_at=10.0)
    assert branch.snapshot() == source.snapshot()
    assert branch._lock is not source._lock
    assert branch._entry_by_hash is not source._entry_by_hash
    assert (
        branch._active.entry
        is branch._entry_by_hash[branch._active.selection.deployment_envelope_hash]
    )
    assert branch._active.entry is not source._active.entry
    assert (
        capture_safety_envelope(branch, captured_at=10.0).content_hash
        == saved.content_hash
    )
    if relative:
        assert branch._reference.captured_at_by_side["right"] == 2.5
        assert branch._reference.safety_reference_epoch == 2
        assert branch._active.execution_epoch == 7
        assert not np.shares_memory(
            branch._reference.tcp_pose_by_side["right"],
            source._reference.tcp_pose_by_side["right"],
        )
    else:
        assert branch._reference is branch._active.anchor is None
    result = branch.project({"right": _pose(0.4)})
    assert (
        capture_safety_envelope(source, captured_at=10.0).content_hash
        == saved.content_hash
    )
    expected = source.project({"right": _pose(0.4)})
    assert result.semantic_content_hash == expected.semantic_content_hash
    branch.deactivate_after_hold(
        operation_id="original-operation",
        next_execution_epoch=8,
        hold_evidence_ids=("branch-hold",),
    )
    assert (
        branch.snapshot().safety_reference_hash
        == source.snapshot().safety_reference_hash
    )
    assert branch._active is branch._last_projection is None
    assert source._active is not None
    assert (
        restore_safety_envelope(
            capture_safety_envelope(branch, captured_at=10.0), captured_at=10.0
        ).snapshot()
        == branch.snapshot()
    )


def test_invalidated_reference_and_fresh_preview_keep_independent_identity():
    source = _runtime(_entry("envelope.standard", 0.2))
    original_preview = source.planning_projector_for_route(route_id=ROUTE_ID)
    branch = restore_safety_envelope(
        capture_safety_envelope(source, captured_at=10.0), captured_at=10.0
    )
    branch_preview = branch.planning_projector_for_route(route_id=ROUTE_ID)
    expected = branch_preview({"right": _pose(0.4)})
    source.invalidate_reference(
        reason="new-scene", source_event_ids=("reset",), lower_state="hold"
    )
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="reference changed"):
        original_preview({"right": _pose(0.4)})
    assert (
        branch_preview({"right": _pose(0.4)}).semantic_content_hash
        == expected.semantic_content_hash
    )
    restored = restore_safety_envelope(
        capture_safety_envelope(source, captured_at=10.0), captured_at=10.0
    )
    assert restored._safety_reference_epoch == 1
    assert restored._reference is restored._active is restored._last_projection is None
    with pytest.raises(TCPPoseEnvelopeLifecycleError, match="no committed"):
        restored.planning_projector_for_route(route_id=ROUTE_ID)


def test_saved_time_and_digest_cannot_be_reinterpreted():
    saved = capture_safety_envelope(_source(), captured_at=10.0)
    with pytest.raises(ValueError, match="cut"):
        restore_safety_envelope(saved, captured_at=11.0)
    with pytest.raises(ValueError, match="content"):
        restore_safety_envelope(replace(saved, content_hash="wrong"), captured_at=10.0)


def test_both_sides_keep_distinct_reference_times_and_projection():
    entry = _entry("envelope.both", 0.2)
    sides = ("left", "right")
    profile = replace(
        entry.profile,
        tool_frame_binding_hash_by_side={side: side + "-tool" for side in sides},
        reference_frame_by_side={side: side + "-base" for side in sides},
        side_config_by_side={
            side: entry.profile.side_config_by_side["right"] for side in sides
        },
    )
    runtime = TCPPoseEnvelopeSafetyRuntime(
        catalog_entries=(replace(entry, profile=profile),),
        target_frame_binding_hash=profile.target_frame_binding_hash,
        tool_frame_binding_hash_by_side=profile.tool_frame_binding_hash_by_side,
        reference_frame_by_side=profile.reference_frame_by_side,
        clock_domain_id="shared-sim-clock",
    )
    runtime.commit_reference(
        reference_id="paired-reference",
        tcp_pose_by_side={"left": _pose(-0.1), "right": _pose(0.1)},
        captured_at_by_side={"left": 2.0, "right": 2.1},
        source_event_ids=("paired-init",),
        lower_state="hold",
    )
    runtime.activate(
        selection=runtime.default_selection(),
        operation_id="paired-operation",
        route_id=ROUTE_ID,
        execution_epoch=5,
        source_event_ids=("paired-ready",),
        lower_state="hold",
    )
    runtime.project({"left": _pose(-0.6), "right": _pose(0.8)})
    checkpoint = capture_safety_envelope(runtime, captured_at=10.0)
    branch = restore_safety_envelope(checkpoint, captured_at=10.0)
    assert dict(branch._reference.captured_at_by_side) == {"left": 2.0, "right": 2.1}
    assert branch.snapshot() == runtime.snapshot()
    assert (
        capture_safety_envelope(branch, captured_at=10.0).content_hash
        == checkpoint.content_hash
    )


@pytest.mark.parametrize(
    "damage",
    (
        "reference_epoch",
        "reference_clock",
        "reference_frame",
        "future_reference",
        "operation",
        "route",
        "selection",
        "constraint",
        "anchor",
        "projection",
        "inactive_projection",
        "cached_hash",
        "unknown_field",
    ),
)
def test_self_consistent_payload_hash_does_not_bypass_owner_links(damage):
    saved = capture_safety_envelope(_source(), captured_at=10.0)
    state = _copy(saved.state)
    active = state["_active"]
    reference = state["_reference"]
    if damage == "reference_epoch":
        state["_safety_reference_epoch"] += 1
    elif damage == "reference_clock":
        state["_reference"] = replace(reference, clock_domain_id="other-clock")
    elif damage == "reference_frame":
        state["_reference"] = replace(
            reference, reference_frame_by_side={"right": "other"}
        )
    elif damage == "future_reference":
        state["_reference"] = replace(reference, captured_at_by_side={"right": 11.0})
    elif damage == "operation":
        state["_active"] = replace(active, operation_id="other-operation")
    elif damage == "route":
        state["_active"] = replace(
            active, receipt=replace(active.receipt, route_id="other-route")
        )
    elif damage == "selection":
        state["_active"] = replace(
            active, selection=replace(active.selection, safety_profile_id="other")
        )
    elif damage == "constraint":
        state["_active"] = replace(
            active, selection=replace(active.selection, constraint_set_hash="other")
        )
    elif damage == "anchor":
        state["_active"] = replace(
            active, anchor=replace(active.anchor, anchor_id="other")
        )
    elif damage == "projection":
        state["_last_projection"] = replace(
            state["_last_projection"], effective_tcp_pose_by_side={"right": _pose(0.8)}
        )
    elif damage == "inactive_projection":
        state["_active"] = None
    elif damage == "cached_hash":
        object.__setattr__(active.selection, "semantic_content_hash", "forged")
    else:
        object.__setattr__(active.receipt, "unexpected", True)
    changed = replace(
        saved,
        state=state,
        content_hash=_hash(saved.configuration, saved.captured_at, state),
    )
    with pytest.raises(ValueError):
        restore_safety_envelope(changed, captured_at=10.0)


@pytest.mark.parametrize("damage", ("extra", "catalog_alias", "active_alias"))
def test_capture_rejects_unknown_or_detached_owner_graph_without_repair(damage):
    source = _source()
    before = source.snapshot()
    if damage == "extra":
        source.foreign_state = object()
    elif damage == "catalog_alias":
        source._entry_by_hash = MappingProxyType(
            {key: replace(entry) for key, entry in source._entry_by_hash.items()}
        )
    else:
        source._active = replace(source._active, entry=replace(source._active.entry))
    with pytest.raises(ValueError):
        capture_safety_envelope(source, captured_at=10.0)
    assert source.snapshot() == before
