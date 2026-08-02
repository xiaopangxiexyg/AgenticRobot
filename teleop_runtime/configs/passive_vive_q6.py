"""Passive VIVE/Q6 session construction owned by :mod:`teleop_runtime`.

The session created here polls the real teleoperation inputs and retargets
them into the existing human emission seam, but it has no arm rig and cannot
dispatch commands.  Application profiles may bind that session to a target
and a composition source; they must not recreate the device or Home setup.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from robot_motion.coordinate_profiles import (
    STATION_RETARGET_FRAME_CONTRACT_V1,
    SessionAnchorReceipt,
    SessionPoseAnchor,
    TargetFrameProfile,
    ToolFrameProfile,
)
from ..backends.grippers import DMTacGripperBackendConfig
from ..calibration import TeleopCalibration
from .builders import (
    build_bimanual_passive_session,
    build_single_side_passive_session,
    default_vive_tracker_configs,
)
from ..session import TeleopSession, TeleopSessionConfig


@dataclass(frozen=True)
class PassiveViveQ6SessionConfig:
    """Connection and observation-only settings for a passive VIVE/Q6 input."""

    vive_port: int = 9999
    vive_clock_sync_port: int | None = None
    left_q6_port: str = ""
    right_q6_port: str = ""

    def __post_init__(self) -> None:
        vive_port = int(self.vive_port)
        if not 0 <= vive_port <= 65535:
            raise ValueError("vive_port must be in [0, 65535].")
        object.__setattr__(self, "vive_port", vive_port)
        if self.vive_clock_sync_port is not None:
            sync_port = int(self.vive_clock_sync_port)
            if not 0 <= sync_port <= 65535:
                raise ValueError("vive_clock_sync_port must be in [0, 65535].")
            object.__setattr__(self, "vive_clock_sync_port", sync_port)
        for name in ("left_q6_port", "right_q6_port"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PassiveSingleViveQ6SessionConfig:
    """One logical side backed by packet-slot-zero VIVE and one Q6."""

    side: str
    vive_port: int = 9999
    vive_clock_sync_port: int | None = None
    q6_port: str = ""
    vive_source_tracker_role: str = "left_wrist"

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        vive_port = int(self.vive_port)
        if not 0 <= vive_port <= 65535:
            raise ValueError("vive_port must be in [0, 65535].")
        object.__setattr__(self, "vive_port", vive_port)
        if self.vive_clock_sync_port is not None:
            sync_port = int(self.vive_clock_sync_port)
            if not 0 <= sync_port <= 65535:
                raise ValueError("vive_clock_sync_port must be in [0, 65535].")
            object.__setattr__(self, "vive_clock_sync_port", sync_port)
        q6_port = str(self.q6_port).strip()
        if not q6_port:
            raise ValueError("q6_port must be non-empty.")
        object.__setattr__(self, "q6_port", q6_port)
        tracker_role = str(self.vive_source_tracker_role).strip()
        if tracker_role not in {"left_wrist", "right_wrist"}:
            raise ValueError(
                "vive_source_tracker_role must be left_wrist or right_wrist."
            )
        object.__setattr__(
            self,
            "vive_source_tracker_role",
            tracker_role,
        )


def seed_vive_home_anchors(
    session: TeleopSession,
    calibration: TeleopCalibration,
    *,
    active_sides: tuple[str, ...] = ("left", "right"),
) -> None:
    """Bind calibrated VIVE Home poses to the existing pose retargeters."""

    metadata = getattr(calibration, "metadata", {})
    if not isinstance(metadata, Mapping) or metadata.get("home_ready") is not True:
        raise ValueError(
            "Passive VIVE/Q6 Sim-HIL requires calibration metadata.home_ready=true."
        )
    capture = metadata.get("home_capture")
    if not isinstance(capture, Mapping):
        raise ValueError("Calibration profile is missing metadata.home_capture.")
    all_required = {
        "left": ("left_human_home_virtual", "left_robot_home_desk"),
        "right": ("right_human_home_virtual", "right_robot_home_desk"),
    }
    sides = tuple(str(side).strip().lower() for side in active_sides)
    if (
        not sides
        or any(side not in all_required for side in sides)
        or len(sides) != len(set(sides))
    ):
        raise ValueError("active_sides must contain unique known sides.")
    required = {side: all_required[side] for side in sides}
    anchors: dict[str, tuple[Any, Any]] = {}
    for side, (human_key, robot_key) in required.items():
        if human_key not in capture or robot_key not in capture:
            raise ValueError(f"Calibration Home capture is incomplete for {side}.")
        anchors[side] = (capture[human_key], capture[robot_key])

    seen: set[str] = set()
    for retargeter in session.retargeters:
        config = getattr(retargeter, "config", None)
        side = str(getattr(config, "side", "")).strip().lower()
        setter = getattr(retargeter, "set_home_anchors", None)
        if side in anchors and callable(setter):
            if hasattr(config, "output_frame_name"):
                config.output_frame_name = f"{side}_arm_base"
            human_home, robot_home = anchors[side]
            setter(
                human_pose_desk=human_home,
                robot_pose_desk=robot_home,
            )
            seen.add(side)
    if seen != set(required):
        raise ValueError(
            "Passive VIVE source did not expose both pose retargeter Home anchors: "
            f"seen={sorted(seen)}."
        )


def bind_passive_vive_target_session_anchors(
    calibration: TeleopCalibration,
    *,
    session_id: str,
    coordinate_profile_id: str,
    coordinate_profile_hash: str,
    target_frame_profile: TargetFrameProfile,
    tool_frame_profile: ToolFrameProfile,
    side_by_component: Mapping[str, str],
    target_home_pose_reference_by_component: Mapping[str, Any],
    captured_at: float,
    clock_domain_id: str,
    source_anchor_origin: str,
    target_anchor_origin: str,
) -> tuple[TeleopCalibration, SessionAnchorReceipt]:
    """Bind stored human Home to target-specific measured TCP poses.

    The source and target profiles remain immutable.  The returned calibration
    is an activation-local copy used only by the passive retargeter.
    """

    if not isinstance(calibration, TeleopCalibration):
        raise TypeError("calibration must be TeleopCalibration.")
    if not isinstance(target_frame_profile, TargetFrameProfile):
        raise TypeError("target_frame_profile must be TargetFrameProfile.")
    if not isinstance(tool_frame_profile, ToolFrameProfile):
        raise TypeError("tool_frame_profile must be ToolFrameProfile.")
    coordinate_frame_contract = str(
        calibration.metadata.get("canonical_target_frame_contract_id", "")
    ).strip()
    target_frame_contract = str(
        target_frame_profile.semantic_parameters.get(
            "canonical_target_frame_contract_id",
            "",
        )
    ).strip()
    if (
        coordinate_frame_contract != STATION_RETARGET_FRAME_CONTRACT_V1
        or target_frame_contract != coordinate_frame_contract
    ):
        raise ValueError(
            "VIVE coordinate profile and target frame profile do not share "
            "the canonical station-frame contract."
        )

    merged = copy.deepcopy(calibration)
    capture = dict(merged.metadata.get("home_capture", {}))
    measured = {
        str(component_id): np.asarray(value, dtype=np.float64)
        for component_id, value in dict(
            target_home_pose_reference_by_component
        ).items()
    }
    expected = set(target_frame_profile.reference_frame_by_component)
    if set(measured) != expected:
        raise ValueError(
            "Target Home measurements must exactly cover target pose components."
        )
    sides = {
        str(component_id): str(side).strip()
        for component_id, side in dict(side_by_component).items()
    }
    if set(sides) != expected or any(not side for side in sides.values()):
        raise ValueError(
            "side_by_component must exactly cover target pose components."
        )
    entries = []
    for component_id in sorted(expected):
        side = sides[component_id]
        human_key = f"{side}_human_home_virtual"
        if human_key not in capture:
            raise ValueError(
                f"Home profile is missing stored human anchor {human_key}."
            )
        reference_frame = (
            target_frame_profile.reference_frame_by_component[component_id]
        )
        canonical_from_reference = (
            target_frame_profile.frame_graph.resolve(
                reference_frame,
                target_frame_profile.canonical_reference_frame_id,
            ).matrix_target_source
        )
        target_home_reference = measured[component_id]
        target_home_canonical = (
            canonical_from_reference @ target_home_reference
        )
        arm = merged.get_arm_for_side(side)
        arm.T_D_B = np.asarray(
            canonical_from_reference,
            dtype=np.float64,
        ).copy()
        capture[f"{side}_robot_home_desk"] = target_home_canonical.tolist()
        entries.append(
            SessionPoseAnchor(
                component_id=component_id,
                controlled_frame_id=(
                    tool_frame_profile.binding_by_component[
                        component_id
                    ].controlled_frame_id
                ),
                target_reference_frame_id=reference_frame,
                source_home_pose_canonical=capture[human_key],
                target_home_pose_canonical=target_home_canonical,
                target_home_pose_reference=target_home_reference,
            )
        )

    metadata = dict(merged.metadata)
    metadata["home_capture"] = capture
    metadata["home_ready"] = True
    metadata["target_home_origin"] = str(target_anchor_origin)
    merged.metadata = metadata
    receipt = SessionAnchorReceipt.capture(
        session_id=session_id,
        coordinate_profile_id=coordinate_profile_id,
        coordinate_profile_hash=coordinate_profile_hash,
        target_frame_profile=target_frame_profile,
        tool_frame_profile=tool_frame_profile,
        source_anchor_origin=source_anchor_origin,
        target_anchor_origin=target_anchor_origin,
        captured_at=captured_at,
        clock_domain_id=clock_domain_id,
        anchors=entries,
    )
    return merged, receipt


def build_passive_vive_q6_session(
    *,
    calibration: TeleopCalibration,
    source_config: PassiveViveQ6SessionConfig,
    session_config: TeleopSessionConfig | None = None,
) -> TeleopSession:
    """Build the observation-only VIVE/Q6 session used by application profiles.

    The existing passive builder remains the source of truth for VIVE, Q6,
    retargeter, and tracker configuration.  This wrapper only fixes the
    Sim-HIL lifecycle policy: no recording, no arm eligibility, observation-
    only Q6 backends, and Home anchors bound before the application starts.
    """

    backend_configs = {
        side: DMTacGripperBackendConfig(
            side=side,
            connect_hardware=False,
            send_commands=False,
            observation_only=True,
        )
        for side in ("left", "right")
    }
    config = session_config or TeleopSessionConfig(
        mode="control-station-sim-hil-vive-q6-passive",
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
        tracker_loss_hold_enabled=True,
        tracker_channel_names=("left_wrist_pose", "right_wrist_pose"),
        tracker_auto_hold_sec=0.15,
        tracker_manual_hold_sec=1.0,
    )
    session = build_bimanual_passive_session(
        calibration=calibration,
        session_config=config,
        vive_port=source_config.vive_port,
        vive_clock_sync_port=source_config.vive_clock_sync_port,
        vive_tracker_configs=default_vive_tracker_configs(),
        left_hand_serial_port=source_config.left_q6_port,
        right_hand_serial_port=source_config.right_q6_port,
        left_hand_backend_config=backend_configs["left"],
        right_hand_backend_config=backend_configs["right"],
        load_default_hand_serial_ports=False,
    )
    seed_vive_home_anchors(session, calibration)
    return session


def build_passive_single_vive_q6_session(
    *,
    calibration: TeleopCalibration,
    source_config: PassiveSingleViveQ6SessionConfig,
    session_config: TeleopSessionConfig | None = None,
) -> TeleopSession:
    """Build one-side passive VIVE/Q6 input without arm authority."""

    side = source_config.side
    backend_config = DMTacGripperBackendConfig(
        side=side,
        connect_hardware=False,
        send_commands=False,
        observation_only=True,
    )
    config = session_config or TeleopSessionConfig(
        mode=f"control-station-sim-hil-vive-q6-passive-{side}",
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
        tracker_loss_hold_enabled=True,
        tracker_channel_names=(f"{side}_wrist_pose",),
        tracker_auto_hold_sec=0.15,
        tracker_manual_hold_sec=1.0,
    )
    session = build_single_side_passive_session(
        side=side,
        calibration=calibration,
        session_config=config,
        vive_port=source_config.vive_port,
        vive_clock_sync_port=source_config.vive_clock_sync_port,
        vive_source_tracker_role=source_config.vive_source_tracker_role,
        hand_serial_port=source_config.q6_port,
        hand_backend_config=backend_config,
    )
    seed_vive_home_anchors(
        session,
        calibration,
        active_sides=(side,),
    )
    return session


__all__ = [
    "PassiveSingleViveQ6SessionConfig",
    "PassiveViveQ6SessionConfig",
    "bind_passive_vive_target_session_anchors",
    "build_passive_vive_q6_session",
    "build_passive_single_vive_q6_session",
    "seed_vive_home_anchors",
]
