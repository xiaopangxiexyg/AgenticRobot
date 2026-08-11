from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from robot_motion.position_control.tcp_pose_envelope import (
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
)
from teleop_runtime.position_control.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeRuntimeError,
    TCPPoseEnvelopeTargetFilter,
    capture_tcp_pose_envelope_anchor,
    install_tcp_pose_envelope_filter,
    target_frame_binding_hash_from_calibration,
    tool_frame_binding_hashes_from_retargeter,
)


def _pose(xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = xyz
    return result


@dataclass
class _ArmCalibration:
    T_D_B: np.ndarray


class _Calibration:
    def __init__(self, transforms: dict[str, np.ndarray]):
        self.transforms = transforms

    def get_arm_for_side(self, side: str) -> _ArmCalibration:
        return _ArmCalibration(self.transforms[side])


class _Retargeter:
    def __init__(self, tool_from_solver_by_side: dict[str, np.ndarray]):
        self.tool_from_solver_by_side = tool_from_solver_by_side
        self.target_override_filter = None
        self.solver_to_tool_calls = 0
        self.tool_to_solver_calls = 0

    def tool_tcp_world_target_from_solver_world_target(
        self,
        *,
        side: str,
        solver_world_target: np.ndarray,
    ) -> np.ndarray:
        self.solver_to_tool_calls += 1
        return solver_world_target @ self.tool_from_solver_by_side[side]

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side: str,
        tool_tcp_world_target: np.ndarray,
    ) -> np.ndarray:
        self.tool_to_solver_calls += 1
        return tool_tcp_world_target @ np.linalg.inv(
            self.tool_from_solver_by_side[side]
        )


def _fixture():
    sides = ("left", "right")
    references = {side: "collision_world" for side in sides}
    calibration = _Calibration(
        {
            "left": _pose((1.0, 0.0, 0.0)),
            "right": _pose((-1.0, 0.0, 0.0)),
        }
    )
    retargeter = _Retargeter(
        {
            "left": _pose((0.0, 0.0, 0.14)),
            "right": _pose((0.0, 0.0, 0.15)),
        }
    )
    target_hash = target_frame_binding_hash_from_calibration(
        calibration,
        sides=sides,
        reference_frame_by_side=references,
    )
    tool_hashes = tool_frame_binding_hashes_from_retargeter(
        retargeter,
        sides=sides,
    )
    profile = TCPPoseEnvelopeProfile(
        profile_id="runtime-test",
        target_frame_binding_hash=target_hash,
        tool_frame_binding_hash_by_side=tool_hashes,
        reference_frame_by_side=references,
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.1, -0.1),
                relative_max_xyz_m=(0.1, 0.1, 0.1),
            ),
            "right": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-2.0, -1.0, -1.0),
                absolute_max_xyz_m=(0.0, 1.0, 1.0),
            ),
        },
    )
    return calibration, retargeter, profile


def _anchor(calibration, profile):
    return capture_tcp_pose_envelope_anchor(
        anchor_id="after-home",
        profile=profile,
        calibration=calibration,
        actual_state_by_side={
            "left": {
                "actual_tcp_pose": _pose((0.2, 0.0, 0.3)),
                "actual_tcp_pose_source": "hardware_qpos_fk",
                "actual_tcp_pose_timestamp": 10.0,
            }
        },
        execution_epoch=7,
        clock_domain_id="runtime-wall",
    )


def test_anchor_consumes_existing_tcp_snapshot_and_only_changes_reference_frame() -> None:
    calibration, _, profile = _fixture()

    anchor = _anchor(calibration, profile)

    np.testing.assert_allclose(
        anchor.tcp_pose_by_side["left"][:3, 3],
        (1.2, 0.0, 0.3),
    )
    assert anchor.captured_at_by_side["left"] == pytest.approx(10.0)
    assert anchor.execution_epoch == 7


def test_anchor_rejects_non_authoritative_source_and_pair_skew() -> None:
    calibration, _, profile = _fixture()
    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="source mismatch"):
        capture_tcp_pose_envelope_anchor(
            anchor_id="bad-source",
            profile=profile,
            calibration=calibration,
            actual_state_by_side={
                "left": {
                    "actual_tcp_pose": _pose(),
                    "actual_tcp_pose_source": "current_qpos_fk",
                    "actual_tcp_pose_timestamp": 1.0,
                }
            },
            execution_epoch=7,
            clock_domain_id="runtime-wall",
        )

    dual_profile = TCPPoseEnvelopeProfile(
        profile_id="dual-relative",
        target_frame_binding_hash=profile.target_frame_binding_hash,
        tool_frame_binding_hash_by_side=profile.tool_frame_binding_hash_by_side,
        reference_frame_by_side=profile.reference_frame_by_side,
        side_config_by_side={
            side: TCPPoseEnvelopeSideConfig(max_rotation_rad=0.2)
            for side in ("left", "right")
        },
    )
    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="pair skew"):
        capture_tcp_pose_envelope_anchor(
            anchor_id="skewed",
            profile=dual_profile,
            calibration=calibration,
            actual_state_by_side={
                side: {
                    "actual_tcp_pose": _pose(),
                    "actual_tcp_pose_source": "hardware_qpos_fk",
                    "actual_tcp_pose_timestamp": timestamp,
                }
                for side, timestamp in (("left", 1.0), ("right", 1.2))
            },
            execution_epoch=7,
            clock_domain_id="runtime-wall",
            max_pair_skew_sec=0.05,
        )


def test_shadow_observes_outermost_target_but_preserves_previous_filter_result() -> None:
    calibration, retargeter, profile = _fixture()
    anchor = _anchor(calibration, profile)
    prior = {
        "left": _pose((1.5, 0.0, 0.16)),
        "right": _pose((-1.2, 0.0, -0.15)),
    }

    def previous_filter(targets, observation, selected_calibration):
        assert selected_calibration is calibration
        return prior

    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        execution_epoch=7,
        anchor_receipt=anchor,
        previous_filter=previous_filter,
        apply_to_target=False,
    )
    original = {"left": _pose(), "right": _pose()}

    returned = observer(original, object(), calibration)

    assert returned is prior
    snapshot = observer.snapshot()
    assert snapshot.last_error is None
    assert snapshot.projection is not None
    assert snapshot.projection["side_projection_by_side"]["left"]["modified"] is True


def test_active_projects_tcp_then_converts_back_without_ik_or_fk() -> None:
    calibration, retargeter, profile = _fixture()
    anchor = _anchor(calibration, profile)
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        execution_epoch=7,
        anchor_receipt=anchor,
        previous_filter=None,
        apply_to_target=True,
    )
    before_solver_to_tool = retargeter.solver_to_tool_calls
    before_tool_to_solver = retargeter.tool_to_solver_calls

    result = observer(
        {
            "left": _pose((1.5, 0.0, 0.16)),
            "right": _pose((-1.2, 0.0, -0.15)),
        },
        object(),
        calibration,
    )

    assert result is not None
    # left TCP anchor is world x=1.2 and its +x limit is 1.3.  The fixed TCP
    # transform only affects z, so the projected solver x is also 1.3.
    assert result["left"][0, 3] == pytest.approx(1.3)
    assert retargeter.solver_to_tool_calls - before_solver_to_tool == 2
    assert retargeter.tool_to_solver_calls - before_tool_to_solver == 2


def test_active_fails_closed_when_calibration_binding_changes() -> None:
    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        previous_filter=None,
        apply_to_target=True,
    )
    changed = _Calibration(
        {
            "left": _pose((1.01, 0.0, 0.0)),
            "right": _pose((-1.0, 0.0, 0.0)),
        }
    )

    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="binding hash mismatch"):
        observer({"left": _pose(), "right": _pose()}, object(), changed)


def test_installer_wraps_existing_filter_without_replacing_it() -> None:
    calibration, retargeter, profile = _fixture()
    previous = lambda targets, observation, selected_calibration: None
    retargeter.target_override_filter = previous

    observer = install_tcp_pose_envelope_filter(
        retargeter=retargeter,
        profile=profile,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=False,
    )

    assert retargeter.target_override_filter is observer
    assert observer.previous_filter is previous
