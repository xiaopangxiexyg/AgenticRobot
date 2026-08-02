from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from ..backends.grippers import DMTacGripperBackend, DMTacGripperBackendConfig
from ..backends.hands import InspireTacHandBackend, InspireTacHandBackendConfig
from ..calibration import TeleopCalibration
from ..hand_tcp_alignment import (
    DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG,
    tracker_hand_to_rm65_tcp_rotation,
    vive_tracker_height_correction_offset,
)
from ..rigs import BimanualRig
from ..session import TeleopSession, TeleopSessionConfig
from ..safety import TeleopSafetyConfig
from ..sources.hand import HandQ6Source
from ..sources.misc import KeyboardButtonSource
from ..sources.pose import VivePoseSource, ViveTrackerConfig
from ..retargeters import (
    DualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    JointsToDexHandRetargeter,
    JointsToDexHandRetargeterConfig,
    PoseToArmRetargeter,
    PoseToArmRetargeterConfig,
    ScalarToGripperRetargeter,
    ScalarToGripperRetargeterConfig,
)
from ..interfaces import EffectorBackend, ObservationSource
from .q6_hardware import load_latest_q6_ports
from .pose_smoothing import build_bimanual_pose_smoother, load_default_pose_smoothing_config


def default_vive_tracker_configs(
    *,
    left_tracker_name: str = "tracker_1",
    right_tracker_name: str = "tracker_2",
    auxiliary_tracker_name: str | None = None,
) -> list[ViveTrackerConfig]:
    configs = [
        ViveTrackerConfig(
            name=left_tracker_name,
            side="left",
            role="left_wrist",
            channel_name="left_wrist_pose",
            legacy_index=0,
        ),
        ViveTrackerConfig(
            name=right_tracker_name,
            side="right",
            role="right_wrist",
            channel_name="right_wrist_pose",
            legacy_index=1,
        ),
    ]
    if auxiliary_tracker_name is not None:
        configs.append(
            ViveTrackerConfig(
                name=auxiliary_tracker_name,
                side=None,
                role="auxiliary",
                channel_name="auxiliary_tracker_pose",
                legacy_index=2,
                critical=False,
            )
        )
    return configs


HandBackendConfig = InspireTacHandBackendConfig | DMTacGripperBackendConfig


@dataclass(frozen=True)
class PassiveHandInputBinding:
    """Bind one already-constructed observation source to a hand retargeter.

    The source owns device I/O.  This value only declares which observation
    channel feeds the passive session and, for scalar-to-gripper mappings,
    whether the scalar is the whole channel or one element of a vector.
    """

    source: ObservationSource
    input_channel_name: str
    source_dim: int | None

    def __post_init__(self) -> None:
        channel = str(self.input_channel_name).strip()
        if not channel:
            raise ValueError("input_channel_name must be non-empty.")
        object.__setattr__(self, "input_channel_name", channel)
        if self.source_dim is not None:
            source_dim = int(self.source_dim)
            if not 0 <= source_dim < 6:
                raise ValueError("source_dim must be in [0, 5], or None.")
            object.__setattr__(self, "source_dim", source_dim)


def _build_hand_backend(config: HandBackendConfig) -> EffectorBackend:
    if isinstance(config, DMTacGripperBackendConfig):
        return DMTacGripperBackend(config)
    if isinstance(config, InspireTacHandBackendConfig):
        return InspireTacHandBackend(config)
    raise TypeError(f"Unsupported hand backend config: {type(config).__name__}")


def _hand_effector_name(
    config: HandBackendConfig | None,
    *,
    side: str,
) -> str:
    if config is not None and config.effector_name is not None:
        return str(config.effector_name)
    return f"{side}_hand"


def _build_hand_retargeter(
    config: HandBackendConfig | None,
    *,
    side: str,
    input_channel_name: str,
    source_dim: int | None = 2,
):
    effector_name = _hand_effector_name(config, side=side)
    if isinstance(config, DMTacGripperBackendConfig):
        return ScalarToGripperRetargeter(
            ScalarToGripperRetargeterConfig(
                side=side,
                input_channel_name=input_channel_name,
                effector_name=effector_name,
                source_dim=source_dim,
                output_dim=0,
                output_max=float(config.max_action),
            )
        )
    return JointsToDexHandRetargeter(
        JointsToDexHandRetargeterConfig(
            side=side,
            input_channel_name=input_channel_name,
            effector_name=effector_name,
        )
    )


def build_bimanual_session(
    *,
    calibration: TeleopCalibration,
    session_config: TeleopSessionConfig | None = None,
    vive_port: int = 9999,
    vive_clock_sync_port: int | None = None,
    vive_tracker_configs: list[ViveTrackerConfig] | None = None,
    left_hand_serial_port: str | None = None,
    right_hand_serial_port: str | None = None,
    left_arm_backend_config: RM65ArmBackendConfig | None = None,
    right_arm_backend_config: RM65ArmBackendConfig | None = None,
    left_hand_backend_config: HandBackendConfig | None = None,
    right_hand_backend_config: HandBackendConfig | None = None,
    hand_serial_config: str = "latest",
    load_default_hand_serial_ports: bool = True,
    keyboard_source: KeyboardButtonSource | None = None,
    enable_dual_arm_collision_avoidance: bool = True,
    dual_arm_collision_config: DualArmCollisionAwareRetargeterConfig | None = None,
    rm65_ik_framework: str = "robot_motion",
    vive_tracker_height_correction: bool = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_enabled
    ),
    vive_tracker_height_correction_m: float = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m
    ),
    safety_config: TeleopSafetyConfig | None = None,
    _build_rig: bool = True,
) -> TeleopSession:
    sources = []
    retargeters = []
    backends = {}

    if bool(load_default_hand_serial_ports):
        missing_hand_sides = []
        if left_hand_serial_port is None and left_hand_backend_config is not None:
            missing_hand_sides.append("left")
        if right_hand_serial_port is None and right_hand_backend_config is not None:
            missing_hand_sides.append("right")
        if missing_hand_sides:
            default_ports = load_latest_q6_ports(
                active_sides=tuple(missing_hand_sides),
                config=hand_serial_config,
                allow_missing=True,
            )
            left_hand_serial_port = left_hand_serial_port or default_ports.get("left")
            right_hand_serial_port = right_hand_serial_port or default_ports.get("right")

    vive_source = VivePoseSource(
        tracker_configs=vive_tracker_configs or default_vive_tracker_configs(),
        port=vive_port,
        clock_sync_port=vive_clock_sync_port,
    )
    sources.append(vive_source)

    if keyboard_source is not None:
        sources.append(keyboard_source)

    both_arm_backends_present = (
        left_arm_backend_config is not None and right_arm_backend_config is not None
    )
    if float(vive_tracker_height_correction_m) < 0.0:
        raise ValueError("vive_tracker_height_correction_m must be >= 0.")
    tracker_translation_offset = (
        vive_tracker_height_correction_offset(vive_tracker_height_correction_m)
        if bool(vive_tracker_height_correction)
        else None
    )
    left_pose_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="left",
            input_channel_name="left_wrist_pose",
            effector_name="left_arm",
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=True,
        )
    )
    right_pose_retargeter = PoseToArmRetargeter(
        PoseToArmRetargeterConfig(
            side="right",
            input_channel_name="right_wrist_pose",
            effector_name="right_arm",
            input_local_translation_offset=tracker_translation_offset,
            input_local_rotation_offset=tracker_hand_to_rm65_tcp_rotation(),
            tracker_motion_intent_enabled=True,
        )
    )
    if enable_dual_arm_collision_avoidance and both_arm_backends_present:
        left_execution_step = tuple(
            (
                np.asarray(left_arm_backend_config.max_angvel, dtype=float)
                * float(left_arm_backend_config.control_dt)
            ).tolist()
        )
        right_execution_step = tuple(
            (
                np.asarray(right_arm_backend_config.max_angvel, dtype=float)
                * float(right_arm_backend_config.control_dt)
            ).tolist()
        )
        enforce_realistic_joint_step = bool(
            left_arm_backend_config.connect_hardware
            or right_arm_backend_config.connect_hardware
            or (not left_arm_backend_config.dry_run_fast)
            or (not right_arm_backend_config.dry_run_fast)
        )
        collision_config = (
            dual_arm_collision_config
            if dual_arm_collision_config is not None
            else DualArmCollisionAwareRetargeterConfig(
                enforce_realistic_joint_step=enforce_realistic_joint_step,
                left_execution_max_joint_step_rad=left_execution_step,
                right_execution_max_joint_step_rad=right_execution_step,
                left_initial_qpos=(
                    None
                    if left_arm_backend_config is None
                    else left_arm_backend_config.initial_qpos
                ),
                right_initial_qpos=(
                    None
                    if right_arm_backend_config is None
                    else right_arm_backend_config.initial_qpos
                ),
                left_tool_tcp_pose_flange_m_rad=(
                    None
                    if left_arm_backend_config is None
                    else left_arm_backend_config.tcp_pose_flange_m_rad
                ),
                right_tool_tcp_pose_flange_m_rad=(
                    None
                    if right_arm_backend_config is None
                    else right_arm_backend_config.tcp_pose_flange_m_rad
                ),
                ik_framework=str(rm65_ik_framework),
                target_pose_smoother=build_bimanual_pose_smoother(
                    load_default_pose_smoothing_config()
                ),
            )
        )
        retargeters.append(
            DualArmCollisionAwareRetargeter(
                collision_config,
                left_pose_retargeter=left_pose_retargeter,
                right_pose_retargeter=right_pose_retargeter,
            )
        )
    else:
        retargeters.extend((left_pose_retargeter, right_pose_retargeter))

    if left_hand_serial_port is not None or right_hand_serial_port is not None:
        sources.append(
            HandQ6Source.from_bimanual(
                left_port=left_hand_serial_port,
                right_port=right_hand_serial_port,
            )
        )

    if left_hand_serial_port is not None:
        retargeters.append(
            _build_hand_retargeter(
                left_hand_backend_config,
                side="left",
                input_channel_name="left_hand_q6",
            )
        )
    if right_hand_serial_port is not None:
        retargeters.append(
            _build_hand_retargeter(
                right_hand_backend_config,
                side="right",
                input_channel_name="right_hand_q6",
            )
        )

    if not _build_rig and (
        left_arm_backend_config is not None or right_arm_backend_config is not None
    ):
        raise ValueError("Passive bimanual session cannot receive arm backend configs.")
    if _build_rig:
        if left_arm_backend_config is not None:
            backends["left_arm"] = RM65ArmBackend(left_arm_backend_config)
        if right_arm_backend_config is not None:
            backends["right_arm"] = RM65ArmBackend(right_arm_backend_config)
        if left_hand_backend_config is not None:
            left_backend = _build_hand_backend(left_hand_backend_config)
            backends[left_backend.name] = left_backend
        if right_hand_backend_config is not None:
            right_backend = _build_hand_backend(right_hand_backend_config)
            backends[right_backend.name] = right_backend
        rig = BimanualRig(backends=backends)
    else:
        rig = None
    return TeleopSession(
        config=session_config or TeleopSessionConfig(mode="bimanual"),
        sources=sources,
        retargeters=retargeters,
        rig=rig,
        calibration=calibration,
        safety_config=safety_config,
    )


def build_bimanual_passive_session(
    *,
    calibration: TeleopCalibration,
    session_config: TeleopSessionConfig | None = None,
    vive_port: int = 9999,
    vive_clock_sync_port: int | None = None,
    vive_tracker_configs: list[ViveTrackerConfig] | None = None,
    left_hand_serial_port: str | None = None,
    right_hand_serial_port: str | None = None,
    left_hand_backend_config: HandBackendConfig | None = None,
    right_hand_backend_config: HandBackendConfig | None = None,
    hand_serial_config: str = "latest",
    load_default_hand_serial_ports: bool = True,
    keyboard_source: KeyboardButtonSource | None = None,
    vive_tracker_height_correction: bool = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_enabled
    ),
    vive_tracker_height_correction_m: float = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m
    ),
    safety_config: TeleopSafetyConfig | None = None,
) -> TeleopSession:
    """Build the collection-profile input side without a rig or backend.

    This reuses the same Vive/hand source, tracker alignment, calibration and
    hand retargeter construction as :func:`build_bimanual_session`.  It never
    creates an arm backend or ``BimanualRig``; the caller must still start the
    returned session explicitly, and a passive wrapper must keep it outside
    ``ARMED/RUNNING``.
    """

    config = session_config or TeleopSessionConfig(
        mode="bimanual_passive_pose",
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
    )
    if config.record_enabled:
        raise ValueError("Passive bimanual session forbids session recording.")
    if config.allow_hardware_after_debug_pass:
        raise ValueError("Passive bimanual session forbids hardware-arm eligibility.")
    return build_bimanual_session(
        calibration=calibration,
        session_config=config,
        vive_port=vive_port,
        vive_clock_sync_port=vive_clock_sync_port,
        vive_tracker_configs=vive_tracker_configs,
        left_hand_serial_port=left_hand_serial_port,
        right_hand_serial_port=right_hand_serial_port,
        left_hand_backend_config=left_hand_backend_config,
        right_hand_backend_config=right_hand_backend_config,
        hand_serial_config=hand_serial_config,
        load_default_hand_serial_ports=load_default_hand_serial_ports,
        keyboard_source=keyboard_source,
        enable_dual_arm_collision_avoidance=False,
        vive_tracker_height_correction=vive_tracker_height_correction,
        vive_tracker_height_correction_m=vive_tracker_height_correction_m,
        safety_config=safety_config,
        _build_rig=False,
    )


def build_single_side_passive_session(
    *,
    side: str,
    calibration: TeleopCalibration,
    session_config: TeleopSessionConfig | None = None,
    vive_port: int = 9999,
    vive_clock_sync_port: int | None = None,
    vive_tracker_name: str = "tracker_1",
    vive_source_tracker_role: str | None = None,
    hand_serial_port: str | None = None,
    hand_input_binding: PassiveHandInputBinding | None = None,
    hand_backend_config: HandBackendConfig | None = None,
    vive_tracker_height_correction: bool = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_enabled
    ),
    vive_tracker_height_correction_m: float = (
        DEFAULT_VIVE_TRACKER_ALIGNMENT_CONFIG.height_correction_m
    ),
    safety_config: TeleopSafetyConfig | None = None,
) -> TeleopSession:
    """Build one observation-only tracker/hand source without a robot rig.

    The physical tracker occupies legacy packet slot zero but is explicitly
    mapped to the requested logical side.  This mapping is an application
    source profile, not a change to the VIVE sender's compatibility behavior.
    """

    normalized_side = str(side).strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError("side must be left or right.")
    config = session_config or TeleopSessionConfig(
        mode=f"single_{normalized_side}_passive_pose",
        record_enabled=False,
        allow_hardware_after_debug_pass=False,
    )
    if config.record_enabled:
        raise ValueError("Passive single-side session forbids session recording.")
    if config.allow_hardware_after_debug_pass:
        raise ValueError(
            "Passive single-side session forbids hardware-arm eligibility."
        )
    if float(vive_tracker_height_correction_m) < 0.0:
        raise ValueError("vive_tracker_height_correction_m must be >= 0.")
    if hand_serial_port is not None and hand_input_binding is not None:
        raise ValueError(
            "hand_serial_port and hand_input_binding are mutually exclusive."
        )
    tracker_channel = f"{normalized_side}_wrist_pose"
    source_tracker_role = (
        f"{normalized_side}_wrist"
        if vive_source_tracker_role is None
        else str(vive_source_tracker_role).strip()
    )
    if source_tracker_role not in {"left_wrist", "right_wrist"}:
        raise ValueError(
            "vive_source_tracker_role must be left_wrist or right_wrist."
        )
    tracker = ViveTrackerConfig(
        name=str(vive_tracker_name),
        side=normalized_side,
        role=source_tracker_role,
        channel_name=tracker_channel,
        legacy_index=0,
    )
    sources: list[ObservationSource] = [
        VivePoseSource(
            tracker_configs=[tracker],
            port=vive_port,
            clock_sync_port=vive_clock_sync_port,
        )
    ]
    tracker_translation_offset = (
        vive_tracker_height_correction_offset(
            vive_tracker_height_correction_m
        )
        if bool(vive_tracker_height_correction)
        else None
    )
    retargeters = [
        PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side=normalized_side,
                input_channel_name=tracker_channel,
                effector_name=f"{normalized_side}_arm",
                # Calibration profiles are keyed by the logical output side.
                # The possibly different upstream sender role remains in the
                # observation and command lineage.
                tracker_role=f"{normalized_side}_wrist",
                input_local_translation_offset=tracker_translation_offset,
                input_local_rotation_offset=(
                    tracker_hand_to_rm65_tcp_rotation()
                ),
                tracker_motion_intent_enabled=True,
            )
        )
    ]
    if hand_serial_port is not None:
        sources.append(
            HandQ6Source.from_single_hand(
                port=str(hand_serial_port),
                side=normalized_side,
            )
        )
        retargeters.append(
            _build_hand_retargeter(
                hand_backend_config,
                side=normalized_side,
                input_channel_name=f"{normalized_side}_hand_q6",
            )
        )
    elif hand_input_binding is not None:
        sources.append(hand_input_binding.source)
        retargeters.append(
            _build_hand_retargeter(
                hand_backend_config,
                side=normalized_side,
                input_channel_name=hand_input_binding.input_channel_name,
                source_dim=hand_input_binding.source_dim,
            )
        )
    return TeleopSession(
        config=config,
        sources=sources,
        retargeters=retargeters,
        rig=None,
        calibration=calibration,
        safety_config=safety_config,
    )


def vive_source_role_mapping(
    session: TeleopSession,
) -> dict[str, str]:
    """Inspect the unique VIVE source's upstream-role/logical-channel map."""

    sources = [
        source
        for source in tuple(getattr(session, "sources", ()) or ())
        if isinstance(source, VivePoseSource)
    ]
    if len(sources) != 1:
        raise RuntimeError(
            "Expected exactly one VivePoseSource in the teleop session; "
            f"found {len(sources)}."
        )
    return sources[0].source_role_by_output_channel


def swap_vive_left_right_sources(
    session: TeleopSession,
) -> dict[str, object]:
    """Swap the unique VIVE source's left/right upstream role binding."""

    sources = [
        source
        for source in tuple(getattr(session, "sources", ()) or ())
        if isinstance(source, VivePoseSource)
    ]
    if len(sources) != 1:
        raise RuntimeError(
            "Expected exactly one VivePoseSource in the teleop session; "
            f"found {len(sources)}."
        )
    return sources[0].swap_left_right_source_roles()
