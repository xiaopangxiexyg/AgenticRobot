from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.pose_frames import PoseFrameContract
from robot_motion.position_control.tcp_pose_envelope import (
    TCPPoseEnvelopeSideConfig,
)
from teleop_runtime.position_control.deploy_tcp_pose_envelope import (
    DeployTCPPoseEnvelopeProjector,
)
from teleop_runtime.position_control.tcp_pose_envelope_runtime import (
    TCPPoseEnvelopeFileSpec,
    TCPPoseEnvelopeRuntimeError,
    load_tcp_pose_envelope_file_spec,
)


def _pose(xyz=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, 3] = xyz
    return result


def _contract() -> PoseFrameContract:
    left = _pose((1.0, 0.0, 0.0))
    right = _pose((-1.0, 0.0, 0.0))
    return PoseFrameContract.legacy_v22_raw(
        active_sides=("left", "right"),
        raw_to_desk_transform_by_side={"left": left, "right": right},
    )


def _state(
    pose: np.ndarray,
    *,
    timestamp: float = 10.0,
    tcp_z: float = 0.15,
) -> dict[str, object]:
    return {
        "actual_tcp_pose": np.asarray(pose, dtype=float).copy(),
        "actual_tcp_pose_source": "hardware_qpos_fk",
        "actual_tcp_pose_timestamp": float(timestamp),
        "tool_tcp_source": "active_tool_profile",
        "tool_tcp_pose_flange_m_rad": np.array(
            [0.0, 0.0, tcp_z, 0.0, 0.0, 0.0], dtype=float
        ),
        "active_tool_name": "DMTAC_M",
        "active_tool_profile_path": "/profiles/DMTAC_M.json",
    }


def _observation(*, timestamp: float = 10.0) -> SimpleNamespace:
    return SimpleNamespace(
        arm_states={
            "left": _state(_pose((0.0, 0.0, 0.3)), timestamp=timestamp),
            "right": _state(_pose((0.0, 0.0, 0.3)), timestamp=timestamp),
        }
    )


def test_deploy_projector_clamps_in_canonical_desk_before_ik() -> None:
    spec = TCPPoseEnvelopeFileSpec(
        profile_id="deploy-absolute",
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(0.8, -1.0, 0.0),
                absolute_max_xyz_m=(1.2, 1.0, 1.0),
            )
        },
    )
    projector = DeployTCPPoseEnvelopeProjector(
        file_spec=spec,
        pose_frame_contract=_contract(),
        active_sides=("left", "right"),
        apply_to_target=True,
    )
    observation = _observation()
    assert projector.bind_from_observation(
        observation,
        execution_epoch=3,
        captured_at=10.0,
    ) is None
    right = _pose((0.125, 0.25, 0.3))
    result = projector(
        {
            # canonical x is 1.5 and must clamp to 1.2, hence raw x 0.2.
            "left": _pose((0.5, 0.0, 0.3)),
            "right": right,
        },
        observation=observation,
        target_role="current",
    )

    assert result["left"][0, 3] == pytest.approx(0.2)
    # Unconfigured sides never take a canonical->raw round trip.
    assert np.array_equal(result["right"], right)
    snapshot = projector.snapshot()
    assert snapshot.last_target_role == "current"
    assert snapshot.projection is not None
    assert snapshot.projection["side_projection_by_side"]["left"]["modified"] is True
    metadata = projector.command_metadata()
    assert metadata["state"] == "limited"
    assert metadata["mode"] == "active"
    assert metadata["target_role"] == "current"
    assert metadata["modified_sides"] == ["left"]
    assert metadata["feedback_schema_version"] == (
        "robot_motion.tcp_pose_envelope_feedback.v1"
    )
    assert projector.feedback_snapshot().receipt_ref == metadata["receipt_ref"]
    json.dumps(metadata)


def test_deploy_projector_clamps_native_impedance_composed_tcp() -> None:
    spec = TCPPoseEnvelopeFileSpec(
        profile_id="deploy-impedance-absolute",
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(0.8, -1.0, 0.0),
                absolute_max_xyz_m=(1.2, 1.0, 1.0),
            )
        },
    )
    projector = DeployTCPPoseEnvelopeProjector(
        file_spec=spec,
        pose_frame_contract=_contract(),
        active_sides=("left", "right"),
        apply_to_target=True,
    )
    projector.bind_from_observation(
        _observation(),
        execution_epoch=3,
        captured_at=10.0,
    )

    result = projector.project_arm_base_tool_tcp(
        "left",
        _pose((0.5, 0.0, 0.3)),
    )

    assert result[0, 3] == pytest.approx(0.2)
    snapshot = projector.snapshot()
    assert snapshot.last_target_role == "final_composed_impedance"
    assert snapshot.projection is not None
    assert snapshot.projection["side_projection_by_side"]["left"]["modified"] is True


def test_deploy_relative_anchor_uses_one_fresh_pose_and_releases_inward() -> None:
    spec = TCPPoseEnvelopeFileSpec(
        profile_id="deploy-relative",
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.1, -0.1),
                relative_max_xyz_m=(0.1, 0.1, 0.1),
            )
        },
    )
    projector = DeployTCPPoseEnvelopeProjector(
        file_spec=spec,
        pose_frame_contract=_contract(),
        active_sides=("left", "right"),
        apply_to_target=True,
    )
    observation = _observation()
    anchor = projector.bind_from_observation(
        observation,
        execution_epoch=4,
        captured_at=10.0,
    )
    assert anchor is not None

    emitted = []
    for raw_x in (0.05, 0.15, 0.25, 0.25, 0.09, 0.04):
        projected = projector(
            {"left": _pose((raw_x, 0.0, 0.3))},
            observation=observation,
            target_role="current",
        )
        emitted.append(float(projected["left"][0, 3]))
    assert emitted == pytest.approx((0.05, 0.1, 0.1, 0.1, 0.09, 0.04))


def test_deploy_active_fails_closed_on_stale_anchor_or_tool_binding_change() -> None:
    spec = TCPPoseEnvelopeFileSpec(
        profile_id="deploy-relative",
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(max_rotation_rad=0.2)
        },
    )
    projector = DeployTCPPoseEnvelopeProjector(
        file_spec=spec,
        pose_frame_contract=_contract(),
        active_sides=("left", "right"),
        apply_to_target=True,
    )
    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="stale"):
        projector.bind_from_observation(
            _observation(timestamp=9.0),
            execution_epoch=1,
            captured_at=10.0,
        )

    observation = _observation()
    projector.bind_from_observation(
        observation,
        execution_epoch=2,
        captured_at=10.0,
    )
    changed = _observation()
    changed.arm_states["left"] = _state(_pose((0.0, 0.0, 0.3)), tcp_z=0.16)
    with pytest.raises(TCPPoseEnvelopeRuntimeError, match="tool-frame binding"):
        projector(
            {"left": _pose((0.0, 0.0, 0.3))},
            observation=changed,
        )


def test_deploy_shadow_never_changes_target_when_unbound() -> None:
    spec = TCPPoseEnvelopeFileSpec(
        profile_id="deploy-shadow",
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-0.1, -0.1, -0.1),
                absolute_max_xyz_m=(0.1, 0.1, 0.1),
            )
        },
    )
    projector = DeployTCPPoseEnvelopeProjector(
        file_spec=spec,
        pose_frame_contract=_contract(),
        active_sides=("left", "right"),
        apply_to_target=False,
    )
    pending = projector.feedback_snapshot()
    assert pending.state == "unbound"
    assert pending.profile_hash is None
    raw = _pose((4.0, 0.0, 0.0))
    result = projector(
        {"left": raw},
        observation=_observation(),
    )
    assert result["left"] is raw
    assert projector.snapshot().last_error is not None
    failed = projector.feedback_snapshot()
    assert failed.state == "fault"
    assert failed.error is not None


def test_deploy_profile_requires_explicit_canonical_absolute_bounds(
    tmp_path,
) -> None:
    profile_path = tmp_path / "tcp-envelope.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "teleop_runtime.tcp_pose_envelope.v1",
                "profile_id": "no-implicit-teleop-frame",
                "sides": {
                    "left": {
                        "absolute_bounds": "calibration_workspace",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicit canonical-frame bounds"):
        load_tcp_pose_envelope_file_spec(
            profile_path,
            active_sides=("left", "right"),
        )
