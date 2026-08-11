from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rotation

from robot_motion.position_control.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeBindingError,
    TCPPoseEnvelopeError,
    TCPPoseEnvelopeInfeasibleError,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
    project_tcp_pose_envelope,
)


TARGET_HASH = "target-frame-binding-hash"
TOOL_HASH_BY_SIDE = {
    "left": "left-tool-frame-binding-hash",
    "right": "right-tool-frame-binding-hash",
}
REFERENCE_BY_SIDE = {"left": "left_arm_base", "right": "right_arm_base"}


def _pose(
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    rotation: Rotation | None = None,
) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = xyz
    if rotation is not None:
        pose[:3, :3] = rotation.as_matrix()
    return pose


def _profile(
    config_by_side: dict[str, TCPPoseEnvelopeSideConfig],
) -> TCPPoseEnvelopeProfile:
    return TCPPoseEnvelopeProfile(
        profile_id="test-envelope",
        target_frame_binding_hash=TARGET_HASH,
        tool_frame_binding_hash_by_side={
            side: TOOL_HASH_BY_SIDE[side]
            for side in config_by_side
        },
        reference_frame_by_side={
            side: REFERENCE_BY_SIDE[side]
            for side in config_by_side
        },
        side_config_by_side=config_by_side,
    )


def _anchor(
    profile: TCPPoseEnvelopeProfile,
    pose_by_side: dict[str, np.ndarray],
    *,
    execution_epoch: int = 4,
) -> TCPPoseEnvelopeAnchorReceipt:
    return TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id="measured-after-init",
        profile=profile,
        execution_epoch=execution_epoch,
        clock_domain_id="runtime-monotonic",
        tcp_pose_by_side=pose_by_side,
        captured_at_by_side={
            side: 10.0 + index * 0.01
            for index, side in enumerate(sorted(pose_by_side))
        },
    )


def _project(
    raw_by_side: dict[str, np.ndarray],
    profile: TCPPoseEnvelopeProfile,
    *,
    anchor: TCPPoseEnvelopeAnchorReceipt | None = None,
    execution_epoch: int = 4,
    reference_by_side: dict[str, str] | None = None,
):
    references = (
        {side: REFERENCE_BY_SIDE[side] for side in raw_by_side}
        if reference_by_side is None
        else reference_by_side
    )
    return project_tcp_pose_envelope(
        raw_by_side,
        profile=profile,
        target_frame_binding_hash=TARGET_HASH,
        tool_frame_binding_hash_by_side={
            side: TOOL_HASH_BY_SIDE[side]
            for side in raw_by_side
        },
        controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
        reference_frame_by_side=references,
        execution_epoch=execution_epoch,
        anchor_receipt=anchor,
    )


def test_side_config_rejects_incomplete_or_empty_enabled_bounds() -> None:
    with pytest.raises(TCPPoseEnvelopeError, match="provided together"):
        TCPPoseEnvelopeSideConfig(absolute_min_xyz_m=(-1.0, -1.0, -1.0))
    with pytest.raises(TCPPoseEnvelopeError, match="must not exceed"):
        TCPPoseEnvelopeSideConfig(
            relative_min_xyz_m=(1.0, 0.0, 0.0),
            relative_max_xyz_m=(0.0, 1.0, 1.0),
        )
    with pytest.raises(TCPPoseEnvelopeError, match="needs an absolute"):
        TCPPoseEnvelopeSideConfig()
    with pytest.raises(TCPPoseEnvelopeError, match=r"within \[0, pi\]"):
        TCPPoseEnvelopeSideConfig(max_rotation_rad=math.pi + 0.01)


def test_profile_rejects_no_active_side_and_non_tcp_controlled_frame() -> None:
    disabled = TCPPoseEnvelopeSideConfig(enabled=False)
    with pytest.raises(TCPPoseEnvelopeError, match="enable at least one"):
        _profile({"left": disabled})
    with pytest.raises(TCPPoseEnvelopeError, match="controlled_frame_id='tool_tcp'"):
        TCPPoseEnvelopeProfile(
            profile_id="wrong-frame",
            target_frame_binding_hash=TARGET_HASH,
            tool_frame_binding_hash_by_side={
                "left": TOOL_HASH_BY_SIDE["left"]
            },
            controlled_frame_id="flange",
            reference_frame_by_side={"left": "left_arm_base"},
            side_config_by_side={
                "left": TCPPoseEnvelopeSideConfig(max_rotation_rad=0.1)
            },
        )


def test_profile_requires_one_distinct_tool_binding_for_each_configured_side() -> None:
    with pytest.raises(TCPPoseEnvelopeError, match="missing configured sides"):
        TCPPoseEnvelopeProfile(
            profile_id="missing-right-tool",
            target_frame_binding_hash=TARGET_HASH,
            tool_frame_binding_hash_by_side={
                "left": TOOL_HASH_BY_SIDE["left"]
            },
            reference_frame_by_side=REFERENCE_BY_SIDE,
            side_config_by_side={
                "left": TCPPoseEnvelopeSideConfig(max_rotation_rad=0.1),
                "right": TCPPoseEnvelopeSideConfig(max_rotation_rad=0.1),
            },
        )


def test_absolute_box_clips_only_configured_side_without_anchor() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-0.5, -0.4, 0.1),
                absolute_max_xyz_m=(0.5, 0.4, 0.8),
            )
        }
    )
    raw = {
        "left": _pose((0.8, -0.6, 0.5)),
        "right": _pose((2.0, 3.0, 4.0)),
    }

    result = _project(raw, profile)

    np.testing.assert_allclose(
        result.effective_tcp_pose_by_side["left"][:3, 3],
        (0.5, -0.4, 0.5),
    )
    np.testing.assert_allclose(result.effective_tcp_pose_by_side["right"], raw["right"])
    assert result.modified_sides == ("left",)
    assert result.anchor_hash is None
    assert set(
        result.side_projection_by_side["left"].violated_translation_constraints
    ) == {"absolute:x:max", "absolute:y:min"}


def test_relative_canonical_box_uses_anchor_origin_but_not_anchor_axes() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.2, -0.3),
                relative_max_xyz_m=(0.1, 0.2, 0.3),
                relative_axes="canonical_axes",
            )
        }
    )
    anchor_pose = _pose(
        (1.0, 2.0, 3.0),
        rotation=Rotation.from_euler("z", 90.0, degrees=True),
    )
    result = _project(
        {"left": _pose((1.3, 1.7, 3.1))},
        profile,
        anchor=_anchor(profile, {"left": anchor_pose}),
    )

    np.testing.assert_allclose(
        result.effective_tcp_pose_by_side["left"][:3, 3],
        (1.1, 1.8, 3.1),
        atol=1e-12,
    )


def test_relative_local_box_rotates_with_anchor_tcp() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.2, -0.05, -0.1),
                relative_max_xyz_m=(0.2, 0.05, 0.1),
                relative_axes="anchor_local_axes",
            )
        }
    )
    anchor_pose = _pose(
        (0.4, -0.2, 0.5),
        rotation=Rotation.from_euler("z", 90.0, degrees=True),
    )
    raw = _pose((0.4, 0.1, 0.5))
    result = _project(
        {"left": raw},
        profile,
        anchor=_anchor(profile, {"left": anchor_pose}),
    )

    effective = result.effective_tcp_pose_by_side["left"][:3, 3]
    local_delta = anchor_pose[:3, :3].T @ (effective - anchor_pose[:3, 3])
    np.testing.assert_allclose(local_delta, (0.2, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(effective, (0.4, 0.0, 0.5), atol=1e-12)


def test_absolute_and_local_boxes_are_projected_as_one_intersection() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-0.4, -0.4, -0.2),
                absolute_max_xyz_m=(0.4, 0.4, 0.2),
                relative_min_xyz_m=(-0.25, -0.08, -0.1),
                relative_max_xyz_m=(0.25, 0.08, 0.1),
                relative_axes="anchor_local_axes",
            )
        }
    )
    anchor_pose = _pose(
        (0.28, 0.0, 0.0),
        rotation=Rotation.from_euler("z", 45.0, degrees=True),
    )
    result = _project(
        {"left": _pose((0.8, 0.8, 0.0))},
        profile,
        anchor=_anchor(profile, {"left": anchor_pose}),
    )

    effective = result.effective_tcp_pose_by_side["left"][:3, 3]
    local_delta = anchor_pose[:3, :3].T @ (effective - anchor_pose[:3, 3])
    assert np.all(effective >= np.array((-0.4, -0.4, -0.2)) - 1e-10)
    assert np.all(effective <= np.array((0.4, 0.4, 0.2)) + 1e-10)
    assert np.all(local_delta >= np.array((-0.25, -0.08, -0.1)) - 1e-10)
    assert np.all(local_delta <= np.array((0.25, 0.08, 0.1)) + 1e-10)
    assert result.side_projection_by_side["left"].translation_modified is True


def test_empty_absolute_relative_intersection_fails_closed() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-1.0, -1.0, -1.0),
                absolute_max_xyz_m=(-0.5, 1.0, 1.0),
                relative_min_xyz_m=(0.5, -0.1, -0.1),
                relative_max_xyz_m=(1.0, 0.1, 0.1),
            )
        }
    )
    anchor = _anchor(profile, {"left": _pose()})

    with pytest.raises(TCPPoseEnvelopeInfeasibleError, match="empty intersection"):
        _project({"left": _pose()}, profile, anchor=anchor)


def test_invalid_tcp_pose_is_rejected_before_projection() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-1.0, -1.0, -1.0),
                absolute_max_xyz_m=(1.0, 1.0, 1.0),
            )
        }
    )
    non_finite = _pose()
    non_finite[0, 3] = np.nan
    with pytest.raises(TCPPoseEnvelopeError, match="finite 4x4 pose"):
        _project({"left": non_finite}, profile)

    distorted = _pose()
    distorted[0, 0] = 2.0
    with pytest.raises(TCPPoseEnvelopeError, match="orthonormal"):
        _project({"left": distorted}, profile)


def test_total_rotation_is_clipped_on_so3_shortest_path() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                max_rotation_rad=np.deg2rad(30.0),
            )
        }
    )
    anchor_pose = _pose(
        rotation=Rotation.from_euler(
            "xyz",
            (20.0, -10.0, 5.0),
            degrees=True,
        )
    )
    requested_delta = Rotation.from_rotvec(
        np.asarray((1.0, 2.0, -0.5), dtype=float)
        / np.linalg.norm((1.0, 2.0, -0.5))
        * np.deg2rad(120.0)
    )
    raw = _pose(
        rotation=Rotation.from_matrix(anchor_pose[:3, :3]) * requested_delta,
    )
    result = _project(
        {"left": raw},
        profile,
        anchor=_anchor(profile, {"left": anchor_pose}),
    )

    effective_delta = (
        Rotation.from_matrix(anchor_pose[:3, :3]).inv()
        * Rotation.from_matrix(
            np.array(
                result.effective_tcp_pose_by_side["left"][:3, :3],
                copy=True,
            )
        )
    )
    assert np.rad2deg(effective_delta.magnitude()) == pytest.approx(30.0)
    diagnostic = result.side_projection_by_side["left"]
    assert diagnostic.raw_anchor_rotation_rad == pytest.approx(np.deg2rad(120.0))
    assert diagnostic.effective_anchor_rotation_rad == pytest.approx(np.deg2rad(30.0))


def test_rotation_wrap_at_plus_minus_pi_does_not_create_full_turn() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                max_rotation_rad=np.deg2rad(5.0),
            )
        }
    )
    anchor_pose = _pose(
        rotation=Rotation.from_euler("z", 179.0, degrees=True),
    )
    raw = _pose(rotation=Rotation.from_euler("z", -179.0, degrees=True))
    result = _project(
        {"left": raw},
        profile,
        anchor=_anchor(profile, {"left": anchor_pose}),
    )

    diagnostic = result.side_projection_by_side["left"]
    assert np.rad2deg(diagnostic.raw_anchor_rotation_rad) == pytest.approx(2.0)
    assert diagnostic.rotation_modified is False
    np.testing.assert_allclose(result.effective_tcp_pose_by_side["left"], raw)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"target_frame_binding_hash": "wrong"}, "target-frame binding"),
        (
            {"tool_frame_binding_hash_by_side": {"left": "wrong"}},
            "tool-frame binding",
        ),
        ({"execution_epoch": 5}, "execution epoch"),
    ),
)
def test_anchor_identity_mismatch_is_rejected(
    change: dict[str, object],
    message: str,
) -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.1, -0.1),
                relative_max_xyz_m=(0.1, 0.1, 0.1),
            )
        }
    )
    anchor = replace(
        _anchor(profile, {"left": _pose()}),
        **change,
    )

    with pytest.raises(TCPPoseEnvelopeBindingError, match=message):
        _project({"left": _pose()}, profile, anchor=anchor)


def test_missing_anchor_side_and_target_reference_mismatch_are_rejected() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.1, -0.1),
                relative_max_xyz_m=(0.1, 0.1, 0.1),
            ),
            "right": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-1.0, -1.0, -1.0),
                absolute_max_xyz_m=(1.0, 1.0, 1.0),
            ),
        }
    )
    anchor = _anchor(profile, {"right": _pose()})
    with pytest.raises(TCPPoseEnvelopeBindingError, match="missing required side"):
        _project(
            {"left": _pose(), "right": _pose()},
            profile,
            anchor=anchor,
        )

    with pytest.raises(TCPPoseEnvelopeBindingError, match="reference frame mismatch"):
        _project(
            {"left": _pose(), "right": _pose()},
            profile,
            anchor=_anchor(profile, {"left": _pose()}),
            reference_by_side={"left": "wrong-frame", "right": "right_arm_base"},
        )


def test_anchor_preserves_per_side_capture_skew_and_outputs_are_immutable() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                max_rotation_rad=0.1,
            ),
            "right": TCPPoseEnvelopeSideConfig(
                max_rotation_rad=0.1,
            ),
        }
    )
    anchor = TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id="dual-measured-anchor",
        profile=profile,
        execution_epoch=4,
        clock_domain_id="runtime-monotonic",
        tcp_pose_by_side={"left": _pose(), "right": _pose()},
        captured_at_by_side={"left": 10.0, "right": 10.025},
    )
    result = _project(
        {"left": _pose(), "right": _pose()},
        profile,
        anchor=anchor,
    )

    assert anchor.pair_skew_sec == pytest.approx(0.025)
    assert result.semantic_content_hash == _project(
        {"right": _pose(), "left": _pose()},
        profile,
        anchor=anchor,
    ).semantic_content_hash
    assert result.raw_tcp_pose_by_side["left"].flags.writeable is False
    with pytest.raises(ValueError):
        result.effective_tcp_pose_by_side["left"][0, 3] = 1.0
