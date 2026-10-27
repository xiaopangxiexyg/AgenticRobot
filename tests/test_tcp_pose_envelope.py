from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rotation

from robot_motion.target_safety.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeBindingError,
    TCPPoseEnvelopeError,
    TCPPoseEnvelopeInfeasibleError,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeRotationCutLocusError,
    TCPPoseEnvelopeSideConfig,
    project_tcp_pose_envelope,
    project_tcp_pose_envelope_side,
    tcp_pose_envelope_requires_projection,
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


@pytest.mark.parametrize("local_axes", (False, True))
@pytest.mark.parametrize("offset,angle,translation,rotation,expected", (
    ((0., 0., 0.), 0., 0., 0., True),
    ((.105, 0., 0.), 0., .01, .02, True),
    ((.105, 0., 0.), 0., .003, .02, False),
    ((.108, .108, 0.), 0., .01, .02, False),
    ((.11, 0., 0.), 0., .01, 0., True),
    ((0., 0., 0.), .32, .01, .03, True),
    ((0., 0., 0.), .32, .01, .01, False),
))
def test_goal_tolerance_intersection_uses_projection_distance(
    local_axes, offset, angle, translation, rotation, expected,
):
    anchor_pose = _pose((.2, -.1, .3), rotation=Rotation.from_euler("z", .7))
    profile = _profile({"right": TCPPoseEnvelopeSideConfig(
        relative_min_xyz_m=(-.1, -.1, -.1), relative_max_xyz_m=(.1, .1, .1),
        relative_axes="anchor_local_axes" if local_axes else "canonical_axes",
        max_rotation_rad=.3,
    )})
    pose = anchor_pose.copy()
    pose[:3, 3] += (anchor_pose[:3, :3] @ offset if local_axes else np.asarray(offset))
    pose[:3, :3] = anchor_pose[:3, :3] @ Rotation.from_euler("x", angle).as_matrix()
    result = _project({"right": pose}, profile, anchor=_anchor(profile, {"right": anchor_pose}))
    before_hash = result.semantic_content_hash
    assert result.intersects_goal_tolerance(
        translation_tolerance_m=translation, rotation_tolerance_rad=rotation,
    ) is expected
    np.testing.assert_array_equal(result.raw_tcp_pose_by_side["right"], pose)
    assert result.semantic_content_hash == before_hash


@pytest.mark.parametrize("field", ("translation_tolerance_m", "rotation_tolerance_rad"))
@pytest.mark.parametrize("bad", (-.1, float("nan"), float("inf"), True, np.bool_(False)))
def test_goal_tolerance_query_rejects_invalid_tolerances(field, bad):
    profile = _profile({"right": TCPPoseEnvelopeSideConfig(
        absolute_min_xyz_m=(-1., -1., -1.), absolute_max_xyz_m=(1., 1., 1.))})
    projection = _project({"right": _pose()}, profile)
    with pytest.raises(TCPPoseEnvelopeError, match="finite and non-negative"):
        projection.intersects_goal_tolerance(**{
            "translation_tolerance_m": .01, "rotation_tolerance_rad": .03, field: bad,
        })


@pytest.mark.parametrize("relative,rotation,disabled", (
    (False, False, False), (True, False, False),
    (True, True, False), (True, True, True),
))
def test_planning_projection_query_matches_canonical_projection(relative, rotation, disabled):
    configs = {
        "left": TCPPoseEnvelopeSideConfig(
            enabled=not disabled,
            absolute_min_xyz_m=(-0.4, -0.7, 0.03),
            absolute_max_xyz_m=(0.75, 0.7, 0.85),
            relative_min_xyz_m=(-0.15, -0.15, -0.15) if relative else None,
            relative_max_xyz_m=(0.15, 0.15, 0.15) if relative else None,
            max_rotation_rad=0.5 if rotation else None,
        ),
    }
    if disabled:
        configs["right"] = TCPPoseEnvelopeSideConfig(
            absolute_min_xyz_m=(-5., -5., -5.), absolute_max_xyz_m=(5., 5., 5.))
    profile = _profile(configs)
    anchor = _anchor(profile, {"left": _pose((0.2, 0.1, 0.4),
                    rotation=Rotation.from_euler("xyz", (0.2, -0.3, 0.1)))}) if relative else None
    bindings = {"profile": profile, "target_frame_binding_hash": TARGET_HASH,
        "tool_frame_binding_hash_by_side": TOOL_HASH_BY_SIDE,
        "controlled_frame_id": TCP_CONTROLLED_FRAME_ID, "reference_frame_by_side": REFERENCE_BY_SIDE,
        "execution_epoch": 4, "anchor_receipt": anchor}
    random = np.random.default_rng(19)
    poses = [_pose(random.uniform((-0.5, -0.8, 0.), (0.8, 0.8, 0.9)),
                   rotation=Rotation.from_rotvec(random.normal(size=3))) for _ in range(150)]
    for x in (-0.4, 0.75):
        for offset in (0., 1e-10, 1e-9, 2e-9, -2e-9):
            poses.append(_pose((x + offset, 0., 0.4)))
    for pose in poses:
        raw = {"left": pose, "right": _pose((4., 4., 4.))}
        before = {s: p.copy() for s, p in raw.items()}
        projection = project_tcp_pose_envelope(raw, **bindings)
        assert tcp_pose_envelope_requires_projection(raw, **bindings) == bool(projection.modified_sides)
        for side, value in raw.items():
            np.testing.assert_array_equal(value, before[side])


@pytest.mark.parametrize("fault", ("target", "tool", "frame", "epoch", "missing", "nan", "rotation", "cut_locus"))
def test_planning_projection_query_retains_all_binding_and_numeric_errors(fault):
    profile = _profile({"left": TCPPoseEnvelopeSideConfig(
        relative_min_xyz_m=(-0.1, -0.1, -0.1),
        relative_max_xyz_m=(0.1, 0.1, 0.1), max_rotation_rad=0.5)})
    raw = {"left": _pose()}
    bindings = {"profile": profile, "target_frame_binding_hash": TARGET_HASH,
        "tool_frame_binding_hash_by_side": TOOL_HASH_BY_SIDE,
        "controlled_frame_id": TCP_CONTROLLED_FRAME_ID, "reference_frame_by_side": REFERENCE_BY_SIDE,
        "execution_epoch": 4, "anchor_receipt": _anchor(profile, raw)}
    if fault == "target":
        bindings["target_frame_binding_hash"] = "wrong"
    elif fault == "tool":
        bindings["tool_frame_binding_hash_by_side"] = {"left": "wrong"}
    elif fault == "frame":
        bindings["reference_frame_by_side"] = {"left": "wrong"}
    elif fault == "epoch":
        bindings["execution_epoch"] = 5
    elif fault == "missing":
        raw = {"right": _pose()}
    elif fault == "nan":
        raw["left"][0, 3] = np.nan
    elif fault == "rotation":
        raw["left"][0, 0] = 2.
    else:
        raw["left"] = _pose(rotation=Rotation.from_rotvec((math.pi, 0., 0.)))
    with pytest.raises(TCPPoseEnvelopeError) as original:
        project_tcp_pose_envelope(raw, **bindings)
    with pytest.raises(type(original.value)) as checked:
        tcp_pose_envelope_requires_projection(raw, **bindings)
    assert str(checked.value) == str(original.value)


def test_planning_query_checks_second_arm_after_first_arm_needs_projection():
    profile = _profile({
        "left": TCPPoseEnvelopeSideConfig(
            absolute_min_xyz_m=(-0.1, -0.1, -0.1),
            absolute_max_xyz_m=(0.1, 0.1, 0.1),
        ),
        "right": TCPPoseEnvelopeSideConfig(max_rotation_rad=0.5),
    })
    raw = {
        "left": _pose((0.5, 0.0, 0.0)),
        "right": _pose(rotation=Rotation.from_rotvec((math.pi, 0.0, 0.0))),
    }
    bindings = {
        "profile": profile,
        "target_frame_binding_hash": TARGET_HASH,
        "tool_frame_binding_hash_by_side": TOOL_HASH_BY_SIDE,
        "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
        "reference_frame_by_side": REFERENCE_BY_SIDE,
        "execution_epoch": 4,
        "anchor_receipt": _anchor(profile, {"left": _pose(), "right": _pose()}),
    }
    with pytest.raises(TCPPoseEnvelopeRotationCutLocusError) as original:
        project_tcp_pose_envelope(raw, **bindings)
    with pytest.raises(type(original.value)) as checked:
        tcp_pose_envelope_requires_projection(raw, **bindings)
    assert str(checked.value) == str(original.value)


def test_single_side_projection_preserves_pair_contract_and_profile_identity() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.01, -0.01, -0.01),
                relative_max_xyz_m=(0.01, 0.01, 0.01),
            ),
            "right": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.02, -0.02, -0.02),
                relative_max_xyz_m=(0.02, 0.02, 0.02),
            ),
        }
    )
    anchor = _anchor(
        profile,
        {"left": _pose(), "right": _pose((0.2, 0.0, 0.0))},
    )

    with pytest.raises(TCPPoseEnvelopeBindingError, match="missing enabled sides"):
        _project({"left": _pose((0.03, 0.0, 0.0))}, profile, anchor=anchor)

    result = project_tcp_pose_envelope_side(
        "left",
        _pose((0.03, 0.0, 0.0)),
        profile=profile,
        target_frame_binding_hash=TARGET_HASH,
        tool_frame_binding_hash_by_side=TOOL_HASH_BY_SIDE,
        controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
        reference_frame_by_side=REFERENCE_BY_SIDE,
        execution_epoch=4,
        anchor_receipt=anchor,
    )

    assert tuple(result.raw_tcp_pose_by_side) == ("left",)
    assert result.envelope_profile_hash == profile.semantic_content_hash
    assert result.anchor_hash == anchor.semantic_content_hash
    assert result.modified_sides == ("left",)
    np.testing.assert_allclose(
        result.effective_tcp_pose_by_side["left"][:3, 3],
        (0.01, 0.0, 0.0),
    )


def test_single_side_projection_rejects_side_absent_from_profile() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                absolute_min_xyz_m=(-1.0, -1.0, -1.0),
                absolute_max_xyz_m=(1.0, 1.0, 1.0),
            )
        }
    )

    with pytest.raises(TCPPoseEnvelopeBindingError, match="absent"):
        project_tcp_pose_envelope_side(
            "right",
            _pose(),
            profile=profile,
            target_frame_binding_hash=TARGET_HASH,
            tool_frame_binding_hash_by_side=TOOL_HASH_BY_SIDE,
            controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
            reference_frame_by_side=REFERENCE_BY_SIDE,
            execution_epoch=4,
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


def test_rotation_cut_locus_fails_closed_instead_of_flipping_axis() -> None:
    profile = _profile(
        {
            "left": TCPPoseEnvelopeSideConfig(
                max_rotation_rad=np.deg2rad(30.0),
            )
        }
    )
    anchor_pose = _pose()
    raw = _pose(rotation=Rotation.from_euler("x", 180.0, degrees=True))

    with pytest.raises(TCPPoseEnvelopeRotationCutLocusError, match="180 degrees"):
        _project(
            {"left": raw},
            profile,
            anchor=_anchor(profile, {"left": anchor_pose}),
        )


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


@pytest.mark.parametrize("owner", ("domain", "runtime"))
def test_small_pose_validation_matches_original_allclose_at_boundaries(owner):
    from robot_motion.target_safety import tcp_pose_envelope as domain
    from teleop_runtime.target_safety import tcp_pose_envelope_runtime as runtime
    validate = domain._pose if owner == "domain" else runtime._pose
    error_type = domain.TCPPoseEnvelopeError if owner == "domain" else ValueError
    matrices = []
    for delta in (0., np.nextafter(1e-9, 0.), 1e-9, np.nextafter(1e-9, 1.), 1e-8):
        homogeneous = np.eye(4)
        homogeneous[3, 0] = delta
        matrices.append(homogeneous)
        orthogonal = np.eye(4)
        orthogonal[0, 0] = np.sqrt(1. + delta)
        matrices.append(orthogonal)
    for value in (np.nan, np.inf, 1e100, -1.):
        pose = np.eye(4)
        pose[0, 0] = value
        matrices.append(pose)
    random = np.random.default_rng(7)
    for _ in range(32):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec(random.normal(size=3)).as_matrix()
        pose[:3, 3] = random.normal(size=3)
        matrices.append(pose)
    for pose in matrices:
        expected_error = None
        if not np.isfinite(pose).all():
            expected_error = "finite"
        elif not np.allclose(pose[3], (0., 0., 0., 1.), atol=1e-9, rtol=0.):
            expected_error = "homogeneous"
        elif not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-9, rtol=0.):
            expected_error = "orthonormal"
        elif not math.isclose(float(np.linalg.det(pose[:3, :3])), 1., abs_tol=1e-9,
                              rel_tol=0. if owner == "domain" else 1e-9):
            expected_error = "determinant"
        if expected_error:
            with pytest.raises(error_type, match=expected_error):
                validate(pose, name="test-pose")
        else:
            result = validate(pose, name="test-pose")
            np.testing.assert_array_equal(result, pose)
            assert not np.shares_memory(result, pose)
