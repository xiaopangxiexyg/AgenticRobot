from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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
    TCPPoseEnvelopeTargetFrameBinding,
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


@pytest.mark.parametrize("apply", (False, True))
def test_planning_projection_preserves_outputs_and_only_reports_modified_targets(apply):
    calibration, retargeter, profile = _fixture()
    def observer():
        return TCPPoseEnvelopeTargetFilter(retargeter=retargeter, profile=profile,
            calibration=calibration, execution_epoch=7,
            anchor_receipt=_anchor(calibration, profile), apply_to_target=apply)
    normal, planning = observer(), observer()
    for x in (1.2, 1.3, 1.3 + 1e-10, 1.3 + 2e-9, 1.5):
        raw = {"left": _pose((x, 0., 0.16)), "right": _pose((-1.2, 0., -0.15))}
        expected = normal(raw)
        prior_count = planning.snapshot().frame_count
        actual = planning.project_for_planning(raw)
        for side, value in raw.items():
            np.testing.assert_array_equal(actual[side], expected[side])
            if apply:
                assert not np.shares_memory(actual[side], value)
        if normal.snapshot().projection["side_projection_by_side"]["left"]["modified"] or not apply:
            assert planning.snapshot().frame_count == prior_count + 1
            assert planning.snapshot().projection == normal.snapshot().projection
        else:
            assert planning.snapshot().frame_count == prior_count


def test_planning_projection_and_planner_fail_closed_after_binding_changes():
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
        RM65GraspJointTrajectoryPlanner,
        RM65GraspTrajectoryPlanningError,
    )
    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(retargeter=retargeter, profile=profile,
        calibration=calibration, execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile), apply_to_target=True)
    planner = object.__new__(RM65GraspJointTrajectoryPlanner)
    planner.target_envelope_filter = observer
    raw = {"left": _pose((1.2, 0., 0.16)), "right": _pose((-1.2, 0., -0.15))}
    planner._require_unmodified_target_envelope(raw, sides=("left", "right"), phase="path_target")
    calibration.transforms["left"][0, 3] += 0.01
    with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
        planner._require_unmodified_target_envelope(raw, sides=("left", "right"), phase="path_target")
    assert raised.value.reason_code == "path_target_safety_envelope_unavailable"


def test_planning_query_retains_rejection_diagnostics_and_custom_filter_port():
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
        RM65GraspJointTrajectoryPlanner,
        RM65GraspTrajectoryPlanningError,
    )
    calibration, retargeter, profile = _fixture()
    class CustomFilter(TCPPoseEnvelopeTargetFilter):
        def project_for_planning(self, _targets):
            pytest.fail("Subclass must keep its callable contract")
    for filter_type in (TCPPoseEnvelopeTargetFilter, CustomFilter):
        observer = filter_type(retargeter=retargeter, profile=profile,
            calibration=calibration, execution_epoch=7,
            anchor_receipt=_anchor(calibration, profile), apply_to_target=True)
        planner = object.__new__(RM65GraspJointTrajectoryPlanner)
        planner.target_envelope_filter = observer
        raw = {"left": _pose((1.5, 0., 0.16)), "right": _pose((-1.2, 0., -0.15))}
        with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
            planner._require_unmodified_target_envelope(raw, sides=("left", "right"), phase="path_target")
        assert raised.value.reason_code == "path_target_safety_envelope_blocked"
        assert raised.value.details["modified_sides"] == ("left",)
        assert raised.value.details["envelope_projection_by_side"]["left"]["translation_adjustment_m"] == pytest.approx(0.2)


def test_planning_projection_matches_canonical_for_rotated_paired_targets():
    from scipy.spatial.transform import Rotation

    calibration, retargeter, profile = _fixture()
    normal = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter, profile=profile, calibration=calibration,
        execution_epoch=7, anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )
    planning = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter, profile=profile, calibration=calibration,
        execution_epoch=7, anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )
    random = np.random.default_rng(37)
    for _ in range(80):
        raw = {
            "left": _pose(random.uniform((0.9, -0.2, 0.0), (1.5, 0.2, 0.5))),
            "right": _pose(random.uniform((-2.2, -1.2, -1.2), (0.2, 1.2, 1.2))),
        }
        for pose in raw.values():
            pose[:3, :3] = Rotation.from_rotvec(random.normal(size=3)).as_matrix()
        before = {side: value.copy() for side, value in raw.items()}
        expected = normal(raw)
        count = planning.snapshot().frame_count
        actual = planning.project_for_planning(raw)
        for side, value in raw.items():
            np.testing.assert_array_equal(actual[side], expected[side])
            np.testing.assert_array_equal(value, before[side])
        projection = normal.snapshot().projection
        if any(
            side["modified"] for side in projection["side_projection_by_side"].values()
        ):
            assert planning.snapshot().frame_count == count + 1
            assert planning.snapshot().projection == projection
        else:
            assert planning.snapshot().frame_count == count


def test_native_planning_query_omits_report_without_mutating_control_snapshot(monkeypatch):
    from robot_motion.target_safety import tcp_pose_envelope as envelope_module
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
        RM65GraspJointTrajectoryPlanner,
    )

    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter, profile=profile, calibration=calibration,
        execution_epoch=7, anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )
    planner = object.__new__(RM65GraspJointTrajectoryPlanner)
    planner.target_envelope_filter = observer
    raw = {"left": _pose((1.2, 0.0, 0.16)), "right": _pose((-1.2, 0.0, -0.15))}
    before = observer.snapshot()

    def unexpected_report(**_kwargs):
        pytest.fail("An unmodified planning target must not allocate a lineage report")

    monkeypatch.setattr(envelope_module, "_projection_result", unexpected_report)
    planner._require_unmodified_target_envelope(
        raw, sides=("left", "right"), phase="path_target",
    )
    assert observer.snapshot() == before
    assert retargeter.tool_to_solver_calls == 0
    projected = observer.project_for_planning(raw)
    for side, value in raw.items():
        np.testing.assert_array_equal(projected[side], value)
        assert not np.shares_memory(projected[side], value)


def test_native_planning_query_uses_new_home_and_epoch_without_cached_admission():
    from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
        RM65GraspJointTrajectoryPlanner,
        RM65GraspTrajectoryPlanningError,
    )

    calibration, retargeter, profile = _fixture()
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter, profile=profile, calibration=calibration,
        execution_epoch=7, anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )
    planner = object.__new__(RM65GraspJointTrajectoryPlanner)
    planner.target_envelope_filter = observer
    raw = {"left": _pose((1.2, 0.0, 0.16)), "right": _pose((-1.2, 0.0, -0.15))}
    planner._require_unmodified_target_envelope(
        raw, sides=("left", "right"), phase="path_target",
    )
    assert observer.snapshot().frame_count == 0
    calibration.homes["left"] = _pose((1.5, 0.0, 0.3))
    anchor = capture_tcp_pose_envelope_anchor(
        anchor_id="new-home", profile=profile, calibration=calibration,
        execution_epoch=8, clock_domain_id="runtime-wall", captured_at=11.0,
    )
    observer.rebind(execution_epoch=8, anchor_receipt=anchor)
    with pytest.raises(RM65GraspTrajectoryPlanningError) as raised:
        planner._require_unmodified_target_envelope(
            raw, sides=("left", "right"), phase="path_target",
        )
    assert raised.value.reason_code == "path_target_safety_envelope_blocked"
    assert raised.value.details["modified_sides"] == ("left",)
    assert raised.value.details["envelope_projection_by_side"]["left"][
        "translation_adjustment_m"
    ] == pytest.approx(0.2)
    snapshot = observer.snapshot()
    assert snapshot.execution_epoch == 8
    assert snapshot.anchor_hash == anchor.semantic_content_hash
    assert snapshot.frame_count == 1


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


def test_explicit_target_frame_binding_matches_legacy_projection_and_is_immutable() -> None:
    calibration, retargeter, profile = _fixture()
    binding = TCPPoseEnvelopeTargetFrameBinding(
        reference_frame_by_side=profile.reference_frame_by_side,
        target_world_from_base_by_side=calibration.transforms,
    )
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        target_frame_binding=binding,
        execution_epoch=7,
        anchor_receipt=_anchor(calibration, profile),
        apply_to_target=True,
    )

    result = observer.project_arm_base_tool_tcp(
        "left",
        _pose((0.5, 0.0, 0.3)),
    )

    np.testing.assert_allclose(result[:3, 3], (0.3, 0.0, 0.3), atol=1e-12)
    assert binding.semantic_hash(sides=("left", "right")) == (
        profile.target_frame_binding_hash
    )
    assert not binding.target_world_from_base_by_side["left"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        binding.target_world_from_base_by_side["left"][0, 3] = 99.0


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


@pytest.mark.parametrize("side", ("left", "right"))
def test_real_agent_primitive_relative_tight_profiles_are_strict_runtime_specs(
    side,
) -> None:
    calibration, retargeter, _ = _fixture()
    path = (
        Path(__file__).resolve().parents[1]
        / "control_station"
        / "real_teleop"
        / "configs"
        / f"agent_primitive_{side}_relative_tight_v1.json"
    )

    profile = load_tcp_pose_envelope_profile(
        path,
        calibration=calibration,
        retargeter=retargeter,
        active_sides=(side,),
    )

    assert profile.enabled_sides == (side,)
    config = profile.side_config_by_side[side]
    assert config.relative_min_xyz_m == pytest.approx((-0.1, -0.1, -0.1))
    assert config.relative_max_xyz_m == pytest.approx((0.1, 0.1, 0.1))
    assert config.relative_axes == "canonical_axes"
    assert config.max_rotation_rad == pytest.approx(np.deg2rad(45.0))


def test_real_agent_grasp_profile_uses_station_absolute_workspace_without_anchor(
) -> None:
    calibration, retargeter, _ = _fixture()
    path = (
        Path(__file__).resolve().parents[1]
        / "teleop_runtime/configs/skill_deploy/agent_grasp_right_workspace_v1.json"
    )

    profile = load_tcp_pose_envelope_profile(
        path,
        calibration=calibration,
        retargeter=retargeter,
        active_sides=("right",),
    )

    assert profile.enabled_sides == ("right",)
    assert profile.anchor_required_sides == ()
    config = profile.side_config_by_side["right"]
    assert config.absolute_min_xyz_m == pytest.approx((-0.39, -0.6, 0.08))
    assert config.absolute_max_xyz_m == pytest.approx((0.39, 0.1, 0.85))
    assert config.relative_min_xyz_m is None
    assert config.max_rotation_rad is None


def test_fixed_binding_hash_reuse_preserves_subsets_and_numeric_identity(monkeypatch) -> None:
    from teleop_runtime.target_safety import tcp_pose_envelope_runtime as runtime

    transforms = {"left": _pose((1., 0., 0.)), "right": _pose((0., 1., 0.))}
    solver_transform = _pose((0., 0., 2.))
    references = {"left": "station", "right": "station"}
    binding = TCPPoseEnvelopeTargetFrameBinding(
        reference_frame_by_side=references,
        target_world_from_base_by_side=transforms,
        target_world_from_solver_world=solver_transform,
    )
    sides_to_check = (("left",), ("right",), ("left", "right"), ("right", "left"))
    expected = [runtime.target_frame_binding_hash_from_transforms(
        target_world_from_base_by_side=transforms, sides=sides,
        reference_frame_by_side=references, target_world_from_solver_world=solver_transform,
    ) for sides in sides_to_check]
    calls = []
    original = runtime.target_frame_binding_hash_from_transforms

    def counted(**kwargs):
        calls.append(kwargs["sides"])
        return original(**kwargs)

    monkeypatch.setattr(runtime, "target_frame_binding_hash_from_transforms", counted)
    for _ in range(2):
        assert [binding.semantic_hash(sides=sides) for sides in sides_to_check] == expected
    assert calls == []
    transforms["left"][0, 3] = 99.
    solver_transform[2, 3] = 99.
    references["left"] = "other-frame"
    assert binding.semantic_hash(sides=("left",)) == expected[0]
    for matrix in (*binding.target_world_from_base_by_side.values(),
                   binding.target_world_from_solver_world):
        with pytest.raises(ValueError):
            matrix.setflags(write=True)
    with pytest.raises(ValueError, match="non-empty and unique"):
        binding.semantic_hash(sides=("left", "left"))
    with pytest.raises(ValueError, match="reference_frame_by_side"):
        binding.semantic_hash(sides=("missing",))


def test_fixed_binding_replacement_still_fails_closed() -> None:
    calibration, retargeter, profile = _fixture()
    binding = TCPPoseEnvelopeTargetFrameBinding(
        reference_frame_by_side=profile.reference_frame_by_side,
        target_world_from_base_by_side=calibration.transforms,
    )
    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter, profile=profile, target_frame_binding=binding,
        execution_epoch=7, anchor_receipt=_anchor(calibration, profile), apply_to_target=True,
    )
    changed = dict(calibration.transforms)
    changed["left"] = _pose((1.01, 0., 0.))
    observer.target_frame_binding = TCPPoseEnvelopeTargetFrameBinding(
        reference_frame_by_side=profile.reference_frame_by_side,
        target_world_from_base_by_side=changed,
    )
    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="binding hash mismatch"):
        observer({"left": _pose(), "right": _pose()})
    assert observer.snapshot().last_error is not None
