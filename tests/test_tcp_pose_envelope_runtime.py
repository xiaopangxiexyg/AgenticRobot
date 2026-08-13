from __future__ import annotations

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.target_safety.tcp_pose_envelope import (
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
)
from teleop_runtime.target_safety.tcp_pose_envelope_runtime import (
    TCP_POSE_ENVELOPE_FILE_SCHEMA_V1,
    TCPPoseEnvelopeRuntimeError,
    TCPPoseEnvelopeTargetFilter,
    capture_tcp_pose_envelope_anchor,
    install_tcp_pose_envelope_filter,
    load_tcp_pose_envelope_profile,
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
    home_pose: np.ndarray | None
    workspace_min: np.ndarray | None = None
    workspace_max: np.ndarray | None = None

    def home_pose_for_tcp_mode(self, tcp_mode: str) -> np.ndarray | None:
        assert tcp_mode == "tool_tcp"
        return None if self.home_pose is None else self.home_pose.copy()


class _Calibration:
    def __init__(
        self,
        transforms: dict[str, np.ndarray],
        homes: dict[str, np.ndarray | None],
        workspaces: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        self.transforms = transforms
        self.homes = homes
        self.workspaces = {} if workspaces is None else workspaces
        self.desk_frame = SimpleNamespace(name="collision_world")

    def get_arm_for_side(self, side: str) -> _ArmCalibration:
        workspace = self.workspaces.get(side, (None, None))
        return _ArmCalibration(
            self.transforms[side],
            self.homes[side],
            workspace_min=workspace[0],
            workspace_max=workspace[1],
        )


class _Retargeter:
    def __init__(self, tool_from_solver_by_side: dict[str, np.ndarray]):
        self.tool_from_solver_by_side = tool_from_solver_by_side
        self.target_override_filter = None
        self.pre_ik_target_safety_filter = None
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
        },
        {
            "left": _pose((1.2, 0.0, 0.3)),
            "right": _pose((-1.2, 0.0, 0.3)),
        },
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
        execution_epoch=7,
        clock_domain_id="runtime-wall",
        captured_at=10.0,
    )


def test_anchor_consumes_committed_tool_tcp_home_pose_without_qpos() -> None:
    calibration, _, profile = _fixture()

    anchor = _anchor(calibration, profile)

    np.testing.assert_allclose(
        anchor.tcp_pose_by_side["left"][:3, 3],
        (1.2, 0.0, 0.3),
    )
    assert anchor.captured_at_by_side["left"] == pytest.approx(10.0)
    assert anchor.execution_epoch == 7


def test_anchor_rejects_missing_home_and_preserves_uncaptured_side() -> None:
    calibration, _, profile = _fixture()
    calibration.homes["left"] = None
    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="Home pose"):
        capture_tcp_pose_envelope_anchor(
            anchor_id="missing-home",
            profile=profile,
            calibration=calibration,
            execution_epoch=7,
            clock_domain_id="runtime-wall",
            captured_at=11.0,
        )

    calibration, _, profile = _fixture()
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
    first = capture_tcp_pose_envelope_anchor(
        anchor_id="initial",
        profile=dual_profile,
        calibration=calibration,
        execution_epoch=7,
        clock_domain_id="runtime-wall",
        captured_at=11.0,
    )
    old_right = first.tcp_pose_by_side["right"].copy()
    calibration.homes["left"] = _pose((1.4, 0.0, 0.3))
    calibration.homes["right"] = _pose((-9.0, 0.0, 0.3))
    rebound = capture_tcp_pose_envelope_anchor(
        anchor_id="left-only",
        profile=dual_profile,
        calibration=calibration,
        execution_epoch=8,
        clock_domain_id="runtime-wall",
        captured_at=12.0,
        captured_sides=("left",),
        previous_anchor_receipt=first,
    )

    np.testing.assert_allclose(rebound.tcp_pose_by_side["left"], calibration.homes["left"])
    np.testing.assert_allclose(rebound.tcp_pose_by_side["right"], old_right)
    assert rebound.captured_at_by_side == {"left": 12.0, "right": 11.0}


def test_home_on_unconfigured_side_preserves_existing_limited_side_anchor() -> None:
    calibration, _, profile = _fixture()
    first = _anchor(calibration, profile)
    old_left = first.tcp_pose_by_side["left"].copy()
    calibration.homes["left"] = _pose((9.0, 0.0, 0.3))

    rebound = capture_tcp_pose_envelope_anchor(
        anchor_id="right-only-home",
        profile=profile,
        calibration=calibration,
        execution_epoch=8,
        clock_domain_id="runtime-wall",
        captured_at=12.0,
        captured_sides=("right",),
        previous_anchor_receipt=first,
    )

    np.testing.assert_allclose(rebound.tcp_pose_by_side["left"], old_left)
    assert rebound.captured_at_by_side["left"] == pytest.approx(10.0)
    assert rebound.execution_epoch == 8


def test_shadow_observes_final_target_but_returns_it_unchanged() -> None:
    calibration, retargeter, profile = _fixture()
    anchor = _anchor(calibration, profile)
    original = {
        "left": _pose((1.5, 0.0, 0.16)),
        "right": _pose((-1.2, 0.0, -0.15)),
    }

    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=anchor,
        apply_to_target=False,
    )

    returned = observer(original)

    assert returned is original
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
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=anchor,
        apply_to_target=True,
    )
    before_solver_to_tool = retargeter.solver_to_tool_calls
    before_tool_to_solver = retargeter.tool_to_solver_calls

    original_right = _pose((-1.2, 0.0, -0.15))
    result = observer(
        {
            "left": _pose((1.5, 0.0, 0.16)),
            "right": original_right,
        },
    )

    assert result is not None
    # left TCP anchor is world x=1.2 and its +x limit is 1.3.  The fixed TCP
    # transform only affects z, so the projected solver x is also 1.3.
    assert result["left"][0, 3] == pytest.approx(1.3)
    np.testing.assert_array_equal(result["right"], original_right)
    assert retargeter.solver_to_tool_calls - before_solver_to_tool == 2
    assert retargeter.tool_to_solver_calls - before_tool_to_solver == 1
    command_metadata = observer.command_metadata()
    assert command_metadata["mode"] == "active"
    assert command_metadata["controlled_frame"] == "tool_tcp"
    assert command_metadata["profile_id"] == "runtime-test"
    assert command_metadata["profile_hash"] == profile.semantic_content_hash
    assert command_metadata["execution_epoch"] == 7
    assert command_metadata["anchor_hash"] == anchor.semantic_content_hash
    assert command_metadata["modified_sides"] == ["left"]
    assert command_metadata["side_adjustments"]["left"]["modified"] is True
    assert command_metadata["side_adjustments"]["left"][
        "translation_adjustment_m"
    ] == pytest.approx(0.2)
    assert command_metadata["projection_hash"]
    assert command_metadata["error"] is None
    assert command_metadata["state"] == "limited"
    assert command_metadata["target_role"] == "final_pre_ik"
    assert command_metadata["feedback_schema_version"] == (
        "robot_motion.tcp_pose_envelope_feedback.v1"
    )
    assert command_metadata["receipt_ref"]
    feedback = observer.feedback_snapshot()
    assert feedback.state == "limited"
    assert feedback.receipt_ref == command_metadata["receipt_ref"]
    json.dumps(command_metadata)


def test_active_projects_native_impedance_composed_tcp_in_arm_base() -> None:
    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )

    # left desk_from_base translates +1 m in x.  Base x=0.5 therefore requests
    # desk x=1.5, while the anchored upper bound is desk x=1.3.
    result = observer.project_arm_base_tool_tcp(
        "left",
        _pose((0.5, 0.0, 0.3)),
    )

    np.testing.assert_allclose(result[:3, 3], (0.3, 0.0, 0.3), atol=1e-12)
    metadata = observer.command_metadata()
    assert metadata["modified_sides"] == ["left"]
    assert metadata["side_adjustments"]["left"]["translation_adjustment_m"] == pytest.approx(0.2)


def test_shadow_native_impedance_projection_returns_base_pose_bit_exact() -> None:
    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=False,
    )
    raw = _pose((0.5, 0.0, 0.3))

    result = observer.project_arm_base_tool_tcp("left", raw)

    np.testing.assert_array_equal(result, raw)
    assert observer.command_metadata()["modified_sides"] == ["left"]


def test_active_fails_closed_when_calibration_binding_changes() -> None:
    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )
    calibration.transforms["left"] = _pose((1.01, 0.0, 0.0))

    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="binding hash mismatch"):
        observer({"left": _pose(), "right": _pose()})


def test_active_boundary_is_memoryless_and_releases_inward_without_catch_up() -> None:
    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )
    right = _pose((-1.2, 0.0, -0.15))

    emitted_x = []
    for requested_x in (1.20, 1.25, 1.35, 1.50, 1.50, 1.29, 1.24):
        projected = observer(
            {
                "left": _pose((requested_x, 0.0, 0.16)),
                "right": right,
            }
        )
        assert projected is not None
        emitted_x.append(float(projected["left"][0, 3]))
        np.testing.assert_array_equal(projected["right"], right)

    assert emitted_x == pytest.approx(
        (1.20, 1.25, 1.30, 1.30, 1.30, 1.29, 1.24)
    )
    # An inward target is admitted immediately.  No rejected target residual
    # is stored, so there is no delayed boundary catch-up.
    assert emitted_x[-2:] == pytest.approx((1.29, 1.24))


def test_installer_uses_dedicated_final_pre_ik_slot() -> None:
    calibration, retargeter, profile = _fixture()
    previous = lambda targets, observation, selected_calibration: None
    retargeter.target_override_filter = previous

    observer = install_tcp_pose_envelope_filter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=False,
    )

    assert retargeter.target_override_filter is previous
    assert retargeter.pre_ik_target_safety_filter is observer


def test_profile_loader_reuses_calibration_workspace_and_relative_limits(
    tmp_path,
) -> None:
    calibration, retargeter, _ = _fixture()
    calibration.workspaces["left"] = (
        np.array((-0.6, -0.7, 0.0), dtype=float),
        np.array((0.8, 0.7, 0.9), dtype=float),
    )
    path = tmp_path / "tcp_envelope.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": TCP_POSE_ENVELOPE_FILE_SCHEMA_V1,
                "profile_id": "left-runtime-box",
                "sides": {
                    "left": {
                        "absolute_bounds": "calibration_workspace",
                        "relative_bounds": {
                            "min_xyz_m": [-0.1, -0.2, -0.3],
                            "max_xyz_m": [0.1, 0.2, 0.3],
                        },
                        "relative_axes": "anchor_local_axes",
                        "max_total_rotation_deg": 20.0,
                    },
                    "right": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )

    profile = load_tcp_pose_envelope_profile(
        path,
        calibration=calibration,
        retargeter=retargeter,
        active_sides=("left", "right"),
    )

    config = profile.side_config_by_side["left"]
    assert config.absolute_min_xyz_m == pytest.approx((-0.6, -0.7, 0.0))
    assert config.absolute_max_xyz_m == pytest.approx((0.8, 0.7, 0.9))
    assert config.relative_min_xyz_m == pytest.approx((-0.1, -0.2, -0.3))
    assert config.relative_axes == "anchor_local_axes"
    assert config.max_rotation_rad == pytest.approx(np.deg2rad(20.0))
    assert profile.enabled_sides == ("left",)
    anchor = capture_tcp_pose_envelope_anchor(
        anchor_id="loaded-profile-home",
        profile=profile,
        calibration=calibration,
        execution_epoch=1,
        clock_domain_id="test",
        captured_at=1.0,
    )
    assert tuple(anchor.tcp_pose_by_side) == ("left",)


@pytest.mark.parametrize(
    ("payload_update", "match"),
    (
        ({"unexpected": True}, "unknown fields"),
        (
            {
                "sides": {
                    "right": {
                        "absolute_bounds": {
                            "min_xyz_m": [-1.0, -1.0, -1.0],
                            "max_xyz_m": [1.0, 1.0, 1.0],
                        }
                    }
                }
            },
            "inactive sides",
        ),
    ),
)
def test_profile_loader_rejects_unknown_fields_and_inactive_sides(
    tmp_path,
    payload_update,
    match,
) -> None:
    calibration, retargeter, _ = _fixture()
    payload = {
        "schema_version": TCP_POSE_ENVELOPE_FILE_SCHEMA_V1,
        "profile_id": "invalid-profile",
        "sides": {
            "left": {
                "absolute_bounds": {
                    "min_xyz_m": [-1.0, -1.0, -1.0],
                    "max_xyz_m": [1.0, 1.0, 1.0],
                }
            }
        },
    }
    payload.update(payload_update)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_tcp_pose_envelope_profile(
            path,
            calibration=calibration,
            retargeter=retargeter,
            active_sides=("left",),
        )
