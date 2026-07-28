"""Passive VIVE/Q6 session construction owned by :mod:`teleop_runtime`.

The session created here polls the real teleoperation inputs and retargets
them into the existing human emission seam, but it has no arm rig and cannot
dispatch commands.  Application profiles may bind that session to a target
and a composition source; they must not recreate the device or Home setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.grippers import DMTacGripperBackendConfig
from ..calibration import TeleopCalibration
from .builders import build_bimanual_passive_session, default_vive_tracker_configs
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


def seed_vive_home_anchors(
    session: TeleopSession,
    calibration: TeleopCalibration,
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
    required = {
        "left": ("left_human_home_virtual", "left_robot_home_desk"),
        "right": ("right_human_home_virtual", "right_robot_home_desk"),
    }
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


__all__ = [
    "PassiveViveQ6SessionConfig",
    "build_passive_vive_q6_session",
    "seed_vive_home_anchors",
]
