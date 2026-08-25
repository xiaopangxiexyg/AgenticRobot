"""Sim-HIL human-source selection and profile binding.

These helpers connect lower-owned teleoperation producers to one Station
profile.  They do not initialize resources at import time and do not own
device lifecycle, composition, safety, target execution, or dataset writing.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class _SimHILHumanAttachment:
    source: Any
    gripper_input_selection: Any
    calibration_binding: Mapping[str, Any]
    runtime_home_pose_desk_by_side_capture: Callable[
        [],
        Mapping[str, Any],
    ]


def _file_sha256(path: Path) -> str | None:
    """Return a stable file identity for Station status, if the file exists."""

    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_vive_coordinate_profile(
    calibration: Any,
    *,
    coordinate_profile_path: Path | None,
    home_profile_path: Path | None = None,
    required_sides: tuple[str, ...] = ("left", "right"),
) -> tuple[Any, dict[str, Any]]:
    """Bind Station VIVE coordinates to a stored human-Home profile.

    ``TeleopCalibration`` remains owned by ``teleop_runtime``.  The Station
    profile only selects two immutable inputs: the compatibility profile
    supplies stored human Home, while the independent Control Station profile
    supplies the VIVE world-to-retarget transform and tracker offsets. Target
    geometry and robot Home are bound later from the concrete target. No data
    is written back to either source profile.
    """

    home_profile_value = getattr(calibration, "metadata", {}).get(
        "source_profile_path"
    )
    binding: dict[str, Any] = {
        "mode": "home_profile_only_compatibility",
        "home_profile_path": (
            None if not str(home_profile_value or "").strip() else str(home_profile_value)
        ),
        "home_profile_sha256": None,
        "vive_coordinate_profile_path": None,
        "vive_coordinate_profile_sha256": None,
        "coordinate_profile_kind": None,
        "coordinate_profile_schema": None,
        "active_sides": None,
    }
    if coordinate_profile_path is None:
        return calibration, binding

    from ..vive_calibration_profiles import (
        load_coordinate_profile,
        validate_coordinate_profile,
    )

    coordinate_source = Path(coordinate_profile_path).expanduser()
    if (
        str(coordinate_source).strip().lower() != "latest"
        and not coordinate_source.is_absolute()
        and coordinate_source.exists()
    ):
        coordinate_source = coordinate_source.resolve()
    coordinate, resolved_path = load_coordinate_profile(coordinate_source)
    active_sides = validate_coordinate_profile(coordinate, path=resolved_path)
    required = tuple(str(side).strip().lower() for side in required_sides)
    if (
        not required
        or any(side not in {"left", "right"} for side in required)
        or len(required) != len(set(required))
    ):
        raise ValueError("required_sides must contain unique known sides.")
    missing = tuple(side for side in required if side not in active_sides)
    if missing:
        raise ValueError(
            "The selected Sim-HIL profile requires VIVE coordinate sides "
            f"{required}, but the profile provides {active_sides}."
        )
    world_to_desk = coordinate.world_to_desk_pose()
    if world_to_desk is None:
        raise ValueError("VIVE coordinate profile is missing world_to_vive_retarget.")
    coordinate_source_hash = _file_sha256(resolved_path)
    home_profile_hash = (
        None
        if home_profile_path is None
        else _file_sha256(Path(home_profile_path).expanduser())
    )
    derived_home_hash = str(
        coordinate.metadata.get("source_profile_sha256", "")
    ).strip()
    if (
        derived_home_hash
        and home_profile_hash is not None
        and derived_home_hash != home_profile_hash
    ):
        raise ValueError(
            "Stored human Home belongs to another VIVE coordinate profile; "
            "capture a new session Home instead of reinterpreting it."
        )

    merged = copy.deepcopy(calibration)
    merged.set_world_to_desk(world_to_desk)
    for side in active_sides:
        role = f"{side}_wrist"
        merged.set_tracker_offset(role, coordinate.tracker_offsets[role])
    metadata = dict(getattr(merged, "metadata", {}))
    metadata.update(
        {
            "sim_hil_coordinate_profile_path": str(resolved_path),
            "sim_hil_coordinate_profile_kind": str(
                coordinate.metadata.get("profile_kind", "")
            ),
            "sim_hil_coordinate_profile_schema": str(
                coordinate.metadata.get("profile_schema", "")
            ),
            "sim_hil_coordinate_profile_name": str(
                coordinate.metadata.get("profile_name", resolved_path.stem)
            ),
            "canonical_target_frame_contract_id": coordinate.metadata.get(
                "canonical_target_frame_contract_id"
            ),
            "sim_hil_calibration_binding": (
                "independent_vive_coordinate_plus_stored_human_home_v1"
            ),
        }
    )
    merged.metadata = metadata
    binding.update(
        {
            "mode": (
                "independent_vive_coordinate_plus_stored_human_home_v1"
            ),
            "vive_coordinate_profile_path": str(resolved_path),
            "vive_coordinate_profile_sha256": coordinate_source_hash,
            "vive_coordinate_profile_id": str(
                coordinate.metadata.get("profile_name", resolved_path.stem)
            ),
            "coordinate_profile_kind": coordinate.metadata.get("profile_kind"),
            "coordinate_profile_schema": coordinate.metadata.get("profile_schema"),
            "active_sides": list(active_sides),
        }
    )
    return merged, binding


def _build_single_vive_gripper_component_source_spec(
    *,
    arm_reference_frame: str,
    target_action_spec_hash_by_component: Mapping[str, str],
) -> Any:
    """Declare the stable human component contract before devices exist."""

    from control_composition.composition import (
        hand_action_spec_semantic_hash,
    )
    from control_station.sources.teleop_pose import (
        PassiveTeleopComponentSourceSpec,
        PassiveTeleopComponentSpec,
    )
    from robot_motion.action import ArmActionSpec, HandActionSpec

    hashes = {
        str(component): str(value).strip()
        for component, value in dict(
            target_action_spec_hash_by_component
        ).items()
    }
    if set(hashes) != {"right_arm", "right_gripper"} or any(
        not value for value in hashes.values()
    ):
        raise ValueError(
            "Single-side human source requires right_arm/right_gripper "
            "target payload identities."
        )
    reference_frame = str(arm_reference_frame).strip()
    if not reference_frame:
        raise ValueError("arm_reference_frame must be non-empty.")
    hand_spec = HandActionSpec.dmtac_gripper("right")
    return PassiveTeleopComponentSourceSpec(
        components=(
            PassiveTeleopComponentSpec.arm_pose(
                "right_arm",
                "right_arm",
                action_spec_hash=hashes["right_arm"],
                arm_action_spec=ArmActionSpec.tool_tcp(
                    "right",
                    reference_frame=reference_frame,
                ),
            ),
            PassiveTeleopComponentSpec.hand_qpos(
                "right_gripper",
                "right_hand",
                action_spec_hash=hand_action_spec_semantic_hash(hand_spec),
                hand_action_spec=hand_spec,
            ),
        ),
        source_id="sim-hil-human-passive-right",
        observer_binding_id="sim-hil-human-passive-right-v1",
        source_profile_id="single-human-right-mujoco-target-v1",
        units_convention_id=(
            "si_m_rad_xyzw_plus_normalized_hand_v1"
        ),
    )


def _build_passive_single_vive_gripper_component_source(
    *,
    calibration: Any,
    source_config: Any,
    target_action_spec_hash_by_component: dict[str, str],
    target_units_convention_id: str,
    source_spec: Any | None = None,
) -> tuple[Any, Any]:
    """Bind one right-side VIVE/gripper producer without motion ownership."""

    if not str(target_units_convention_id).strip():
        raise ValueError("target_units_convention_id must be non-empty.")

    from control_station.sources.teleop_pose import (
        PassiveTeleopComponentClockBridge,
        PassiveTeleopComponentSessionSource,
        PassiveTeleopComponentSourceSpec,
    )
    from teleop_runtime.configs import (
        build_passive_single_vive_gripper_session,
    )
    from teleop_runtime.real_teleop.defaults import (
        DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG,
        DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M,
    )
    from teleop_runtime.safety import TeleopSafetyConfig

    hashes = {
        str(component): str(value).strip()
        for component, value in dict(
            target_action_spec_hash_by_component
        ).items()
    }
    if set(hashes) != {"right_arm", "right_gripper"} or any(
        not value for value in hashes.values()
    ):
        raise ValueError(
            "Single-side VIVE/gripper source requires right_arm/right_gripper "
            "target payload identities."
        )
    session = build_passive_single_vive_gripper_session(
        calibration=calibration,
        source_config=source_config,
        safety_config=TeleopSafetyConfig(
            # Sim-HIL consumes the same physical VIVE stream as formal
            # teleoperation.  Do not silently apply the much tighter generic
            # 10 cm / 45 degree fixture defaults at this application binding.
            max_translation_jump_m=(
                DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M
            ),
            max_rotation_jump_deg=(
                DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG
            ),
            # The receiver is latest-value while composition is lower-rate.
            # Never compare poses across a consumer stall as though they were
            # adjacent tracker samples.
            pose_jump_baseline_max_gap_sec=0.25,
        ),
    )
    # The passive source contract must be derived from the retargeter that
    # will actually emit commands.  The single-side profile currently binds
    # ``right_arm_base`` through ``seed_vive_home_anchors``; deriving it here
    # keeps the producer contract correct if a future target profile changes
    # the output frame, and avoids a second hard-coded frame identity.
    arm_retargeters = tuple(
        retargeter
        for retargeter in tuple(getattr(session, "retargeters", ()) or ())
        if str(
            getattr(getattr(retargeter, "config", None), "effector_name", "")
        ).strip()
        == "right_arm"
    )
    if len(arm_retargeters) != 1:
        raise ValueError(
            "Single-side VIVE/gripper session must expose exactly one "
            f"right_arm pose retargeter; found {len(arm_retargeters)}."
        )
    arm_reference_frame = str(
        getattr(
            getattr(arm_retargeters[0], "config", None),
            "output_frame_name",
            "",
        )
    ).strip()
    if not arm_reference_frame:
        raise ValueError(
            "The right_arm pose retargeter must declare output_frame_name."
        )
    expected_spec = (
        _build_single_vive_gripper_component_source_spec(
            arm_reference_frame=arm_reference_frame,
            target_action_spec_hash_by_component=hashes,
        )
        if source_spec is None
        else source_spec
    )
    if not isinstance(
        expected_spec,
        PassiveTeleopComponentSourceSpec,
    ):
        raise TypeError(
            "source_spec must be a PassiveTeleopComponentSourceSpec."
        )
    component_by_id = {
        item.component_id: item for item in expected_spec.components
    }
    if set(component_by_id) != {"right_arm", "right_gripper"}:
        raise ValueError(
            "The single-side source contract must expose the target component "
            "IDs right_arm/right_gripper."
        )
    arm_components = tuple(
        item
        for item in expected_spec.components
        if item.effector_name == "right_arm"
    )
    if (
        len(arm_components) != 1
        or arm_components[0].arm_action_spec.reference_frame
        != arm_reference_frame
    ):
        raise ValueError(
            "The attached VIVE retargeter frame does not match the reserved "
            "human source contract."
        )
    source = PassiveTeleopComponentSessionSource(session, expected_spec)
    return PassiveTeleopComponentClockBridge(source), expected_spec
