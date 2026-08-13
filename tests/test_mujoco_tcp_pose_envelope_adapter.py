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
