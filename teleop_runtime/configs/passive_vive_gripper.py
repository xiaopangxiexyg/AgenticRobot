"""Passive single-VIVE input with a selected Q6 or angle trigger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..backends.grippers import DMTacGripperBackendConfig
from ..calibration import TeleopCalibration
from ..session import TeleopSession, TeleopSessionConfig
from ..safety import TeleopSafetyConfig
from ..sources.hand import AngleTriggerSource
from .builders import (
    PassiveHandInputBinding,
    build_single_side_passive_session,
)
from .gripper_input import (
    GRIPPER_INPUT_ANGLE_TRIGGER,
    GRIPPER_INPUT_Q6,
)
from .passive_vive_q6 import seed_vive_home_anchors


@dataclass(frozen=True)
class PassiveSingleViveGripperSessionConfig:
    """Resolved device settings for one passive pose-plus-gripper source."""

    side: str
    gripper_input_source: str
    vive_port: int = 9999
    vive_clock_sync_port: int | None = None
    vive_source_tracker_role: str = "left_wrist"
    q6_port: str | None = None
    angle_trigger_config_root: Path | None = None
    angle_trigger_rate_hz: float = 30.0
    # Optional profile-local tracker-loss grace.  Real-teleop callers retain
    # the historical default; Sim-HIL may widen it for a network-only gap.
    tracker_auto_hold_sec: float = 0.15
    tracker_manual_hold_sec: float = 1.0

    def __post_init__(self) -> None:
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        source = str(self.gripper_input_source).strip().lower()
        if source not in {GRIPPER_INPUT_Q6, GRIPPER_INPUT_ANGLE_TRIGGER}:
            raise ValueError(
                "gripper_input_source must be q6 or angle_trigger."
            )
        vive_port = int(self.vive_port)
        if not 0 <= vive_port <= 65535:
            raise ValueError("vive_port must be in [0, 65535].")
        sync_port = self.vive_clock_sync_port
        if sync_port is not None:
            sync_port = int(sync_port)
            if not 0 <= sync_port <= 65535:
                raise ValueError(
                    "vive_clock_sync_port must be in [0, 65535]."
                )
        q6_port = (
            None if self.q6_port is None else str(self.q6_port).strip() or None
        )
        if source == GRIPPER_INPUT_Q6 and q6_port is None:
            raise ValueError("Q6 input requires q6_port.")
        if source != GRIPPER_INPUT_Q6 and q6_port is not None:
            raise ValueError("Angle-trigger input must not carry q6_port.")
        rate_hz = float(self.angle_trigger_rate_hz)
        if rate_hz <= 0.0:
            raise ValueError("angle_trigger_rate_hz must be positive.")
        tracker_hold = float(self.tracker_auto_hold_sec)
        if tracker_hold <= 0.0:
            raise ValueError("tracker_auto_hold_sec must be positive.")
        tracker_manual_hold = float(self.tracker_manual_hold_sec)
        if tracker_manual_hold <= 0.0:
            raise ValueError("tracker_manual_hold_sec must be positive.")
        tracker_role = str(self.vive_source_tracker_role).strip()
        if tracker_role not in {"left_wrist", "right_wrist"}:
            raise ValueError(
                "vive_source_tracker_role must be left_wrist or right_wrist."
            )
        root = (
            None
            if self.angle_trigger_config_root is None
            else Path(self.angle_trigger_config_root).expanduser()
        )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "gripper_input_source", source)
        object.__setattr__(self, "vive_port", vive_port)
        object.__setattr__(self, "vive_clock_sync_port", sync_port)
        object.__setattr__(self, "q6_port", q6_port)
        object.__setattr__(self, "angle_trigger_rate_hz", rate_hz)
        object.__setattr__(self, "tracker_auto_hold_sec", tracker_hold)
        object.__setattr__(self, "tracker_manual_hold_sec", tracker_manual_hold)
        object.__setattr__(self, "angle_trigger_config_root", root)
        object.__setattr__(
            self,
            "vive_source_tracker_role",
            tracker_role,
        )


def build_passive_single_vive_gripper_session(
    *,
    calibration: TeleopCalibration,
    source_config: PassiveSingleViveGripperSessionConfig,
    session_config: TeleopSessionConfig | None = None,
    safety_config: TeleopSafetyConfig | None = None,
) -> TeleopSession:
    """Build a passive single-side VIVE plus resolved gripper input."""

    side = source_config.side
    backend_config = DMTacGripperBackendConfig(
        side=side,
        connect_hardware=False,
        send_commands=False,
        observation_only=True,
    )
    config = session_config or TeleopSessionConfig(
        mode=(
            "control-station-sim-hil-vive-"
            f"{source_config.gripper_input_source}-passive-{side}"
        ),
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
        tracker_loss_hold_enabled=True,
        tracker_channel_names=(f"{side}_wrist_pose",),
        tracker_auto_hold_sec=source_config.tracker_auto_hold_sec,
        tracker_manual_hold_sec=source_config.tracker_manual_hold_sec,
    )
    kwargs = {}
    if source_config.gripper_input_source == GRIPPER_INPUT_Q6:
        kwargs["hand_serial_port"] = source_config.q6_port
    else:
        from hardware.angle_trigger.angle_trigger import (
            DeferredAngleTriggerReader,
        )

        reader = DeferredAngleTriggerReader(
            active_sides=(side,),
            config_root=source_config.angle_trigger_config_root,
        )
        angle_source = AngleTriggerSource(
            reader,
            rate_hz=source_config.angle_trigger_rate_hz,
            config_root=source_config.angle_trigger_config_root,
            active_sides=(side,),
        )
        kwargs["hand_input_binding"] = PassiveHandInputBinding(
            source=angle_source,
            input_channel_name=f"{side}_angle_trigger",
            source_dim=None,
        )
    session = build_single_side_passive_session(
        side=side,
        calibration=calibration,
        session_config=config,
        vive_port=source_config.vive_port,
        vive_clock_sync_port=source_config.vive_clock_sync_port,
        vive_source_tracker_role=source_config.vive_source_tracker_role,
        hand_backend_config=backend_config,
        safety_config=safety_config,
        **kwargs,
    )
    seed_vive_home_anchors(
        session,
        calibration,
        active_sides=(side,),
    )
    return session


__all__ = [
    "PassiveSingleViveGripperSessionConfig",
    "build_passive_single_vive_gripper_session",
]
