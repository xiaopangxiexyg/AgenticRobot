from __future__ import annotations

import numpy as np
import pytest

from robot_motion.target_safety.tcp_pose_envelope import (
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
)
from sim.mujoco.adapters.tcp_pose_envelope import (
    MujocoTCPPoseEnvelopeError,
    MujocoTCPPoseEnvelopeTargetFilter,
    install_mujoco_tcp_pose_envelope_filter,
    mujoco_target_frame_binding_hash,
    mujoco_tool_frame_binding_hashes,
)
from sim.mujoco.adapters.tcp_pose_command_guard import MujocoTCPPoseCommandGuard
from sim.mujoco.control import DirectQposCommand


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[0, 3] = x
    return pose


class _Retargeter:
    def __init__(self) -> None:
        self.pre_ik_target_safety_filter = None

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side: str,
        solver_world_target: np.ndarray,
    ) -> np.ndarray:
        del side
        return np.asarray(solver_world_target, dtype=float).copy()

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side: str,
        tool_tcp_world_target: np.ndarray,
    ) -> np.ndarray:
        del side
        return np.asarray(tool_tcp_world_target, dtype=float).copy()


def _fixture():
    retargeter = _Retargeter()
    target_hash = mujoco_target_frame_binding_hash(
        reference_frame="mujoco_world",
        world_from_desk=np.eye(4),
        world_to_base_by_side={
            "left": np.eye(4),
            "right": np.eye(4),
        },
    )
    tool_hashes = mujoco_tool_frame_binding_hashes(
        retargeter,
        sides=("left", "right"),
    )
    profile = TCPPoseEnvelopeProfile(
        profile_id="mujoco-adapter-test",
        target_frame_binding_hash=target_hash,
        tool_frame_binding_hash_by_side=tool_hashes,
        reference_frame_by_side={
            "left": "mujoco_world",
            "right": "mujoco_world",
        },
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.1, -0.1),
                relative_max_xyz_m=(0.1, 0.1, 0.1),
            )
        },
    )
    anchor = TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id="sim-start",
        profile=profile,
        execution_epoch=3,
        clock_domain_id="mujoco_physics_time",
        tcp_pose_by_side={"left": _pose()},
        captured_at_by_side={"left": 0.0},
    )
    return retargeter, target_hash, profile, anchor


def _command_guard_fixture():
    retargeter, target_hash, profile, anchor = _fixture()
    target_filter = install_mujoco_tcp_pose_envelope_filter(
        retargeter=retargeter, profile=profile, target_frame_binding_hash=target_hash,
        execution_epoch=3, anchor_receipt=anchor, apply_to_target=True,
    )
    guard = MujocoTCPPoseCommandGuard(
        target_filter=target_filter,
        fk_by_side=lambda pair: {s: _pose(q[0]) for s, q in pair.items()},
    )
    return target_filter, guard, profile


def test_final_command_guard_checks_stopping_response_as_well_as_endpoint() -> None:
    target_filter, guard, _ = _command_guard_fixture()
    start = {s: np.zeros(6) for s in ("left", "right")}
    requested = {s: q.copy() for s, q in start.items()}
    requested["left"][0] = 0.2
    command = DirectQposCommand(1, requested, source="final-guard-test")
    observed = []

    def stopping_query(candidate, *, qpos_check, max_preview_time_sec):
        assert candidate.sequence == command.sequence
        assert candidate.source == command.source
        assert max_preview_time_sec > 0
        assert qpos_check(candidate.qpos_by_side)
        safe = candidate.qpos_by_side["left"][0] <= 0.08
        observed.append((candidate.qpos_by_side["left"][0], safe))
        return safe

    alpha = guard.admission_fraction(
        command, previous_qpos_by_side=start, preview_held_command=stopping_query,
    )
    assert 0.399 < alpha <= 0.4
    accepted = alpha * command.qpos_by_side["left"][0]
    assert any(x == accepted and safe for x, safe in observed)
    np.testing.assert_array_equal(command.qpos_by_side["left"], requested["left"])
    assert target_filter.projection_count == 0
    assert target_filter.last_projection is None
    assert guard.snapshot()["modified_count"] == 1


def test_final_command_guard_rejects_an_unproved_stop() -> None:
    _, guard, _ = _command_guard_fixture()
    pair = {s: np.zeros(6) for s in ("left", "right")}
    with pytest.raises(MujocoTCPPoseEnvelopeError, match="no safe held-control"):
        guard.admission_fraction(
            DirectQposCommand(1, pair), previous_qpos_by_side=pair,
            preview_held_command=lambda *a, **k: False,
        )


def test_final_command_guard_rejects_rebound_anchor_epoch() -> None:
    target_filter, guard, profile = _command_guard_fixture()
    anchor = TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id="new-activation", profile=profile, execution_epoch=4,
        clock_domain_id="mujoco_physics_time", tcp_pose_by_side={"left": _pose()},
        captured_at_by_side={"left": 0.0},
    )
    target_filter.rebind(execution_epoch=4, anchor_receipt=anchor)
    with pytest.raises(MujocoTCPPoseEnvelopeError, match="binding changed"):
        guard.check_qpos_pair({s: np.zeros(6) for s in ("left", "right")})


def test_final_command_guard_rejects_shadow_and_nonfinite_fk() -> None:
    target_filter, guard, _ = _command_guard_fixture()
    target_filter.apply_to_target = False
    with pytest.raises(MujocoTCPPoseEnvelopeError, match="Active"):
        MujocoTCPPoseCommandGuard(target_filter=target_filter, fk_by_side=lambda q: {})
    target_filter.apply_to_target = True
    pair = {s: np.zeros(6) for s in ("left", "right")}
    pair["left"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        guard.check_qpos_pair(pair)


def test_active_mujoco_adapter_projects_only_configured_side() -> None:
    retargeter, target_hash, profile, anchor = _fixture()
    target_filter = MujocoTCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        target_frame_binding_hash=target_hash,
        execution_epoch=3,
        anchor_receipt=anchor,
        apply_to_target=True,
    )
    right = _pose(-0.4)

    result = target_filter({"left": _pose(0.25), "right": right})

    assert result["left"][0, 3] == pytest.approx(0.1)
    np.testing.assert_array_equal(result["right"], right)
    assert target_filter.projection_count == 1
    assert target_filter.last_projection is not None
    feedback = target_filter.feedback_snapshot()
    assert feedback.mode == "active"
    assert feedback.state == "limited"
    assert feedback.modified_sides == ("left",)


def test_shadow_mujoco_adapter_observes_but_preserves_input_identity() -> None:
    retargeter, target_hash, profile, anchor = _fixture()
    target_filter = MujocoTCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        target_frame_binding_hash=target_hash,
        execution_epoch=3,
        anchor_receipt=anchor,
        apply_to_target=False,
    )
    original = {"left": _pose(0.25), "right": _pose(-0.4)}

    result = target_filter(original)

    assert result is original
    assert target_filter.feedback_snapshot().state == "limited"


def test_mujoco_adapter_install_refuses_to_replace_safety_owner() -> None:
    retargeter, target_hash, profile, anchor = _fixture()
    installed = install_mujoco_tcp_pose_envelope_filter(
        retargeter=retargeter,
        profile=profile,
        target_frame_binding_hash=target_hash,
        execution_epoch=3,
        anchor_receipt=anchor,
        apply_to_target=True,
    )
    assert retargeter.pre_ik_target_safety_filter is installed

    with pytest.raises(MujocoTCPPoseEnvelopeError, match="already has"):
        install_mujoco_tcp_pose_envelope_filter(
            retargeter=retargeter,
            profile=profile,
            target_frame_binding_hash=target_hash,
            execution_epoch=3,
            anchor_receipt=anchor,
            apply_to_target=True,
        )
