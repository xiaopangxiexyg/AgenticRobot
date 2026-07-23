"""No-device synthetic human source for the first Control Station sim slice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime import EffectorCommand, TeleopCommand
from teleop_runtime.action.human_proposal import HumanCommandEmission


SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1 = (
    "control_station_synthetic_human_bimanual_qpos_v1"
)
SYNTHETIC_HUMAN_QPOS_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.synthetic_human_qpos_source.v1"
)
SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1 = (
    "control_station_synthetic_human_bimanual_pose_v1"
)
SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1 = (
    "control_station_synthetic_human_pose_v1"
)
SYNTHETIC_HUMAN_POSE_SOURCE_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.synthetic_human_pose_source.v1"
)


@dataclass(frozen=True)
class SyntheticHumanQposSourceSpec:
    """Exact source-side shape contract for the scoped CS2a1 fixture."""

    source_id: str = "synthetic-human-qpos"
    joint_dof_by_effector: tuple[tuple[str, int], ...] = (
        ("left_arm", 6),
        ("right_arm", 6),
    )
    contribution_name: str = "control_station_synthetic_human_qpos_v1"

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        contribution_name = str(self.contribution_name).strip()
        if not source_id or not contribution_name:
            raise ValueError("source_id and contribution_name must be non-empty.")
        normalized: list[tuple[str, int]] = []
        for effector_name, dof in self.joint_dof_by_effector:
            name = str(effector_name).strip()
            count = int(dof)
            if not name or count <= 0:
                raise ValueError("Each synthetic effector needs a name and positive DOF.")
            normalized.append((name, count))
        if not normalized or len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("Synthetic effector names must be non-empty and unique.")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "contribution_name", contribution_name)
        object.__setattr__(self, "joint_dof_by_effector", tuple(normalized))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "contract_id": SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1,
                "source_id": self.source_id,
                "joint_dof_by_effector": self.joint_dof_by_effector,
                "contribution_name": self.contribution_name,
            },
            domain=SYNTHETIC_HUMAN_QPOS_SOURCE_HASH_DOMAIN_V1,
        )


class SyntheticHumanQposSource:
    """Produce one immutable human emission per explicit caller submission.

    This object deliberately has no worker and no buffering.  It is a typed
    fixture source, not an alternate teleoperation runtime.
    """

    def __init__(self, spec: SyntheticHumanQposSourceSpec) -> None:
        if not isinstance(spec, SyntheticHumanQposSourceSpec):
            raise TypeError("spec must be a SyntheticHumanQposSourceSpec.")
        self.spec = spec
        self._next_sequence = 0

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def emit(
        self,
        qpos_by_effector: dict[str, np.ndarray | tuple[float, ...] | list[float]],
        *,
        emitted_at: float,
    ) -> HumanCommandEmission:
        timestamp = float(emitted_at)
        if not math.isfinite(timestamp):
            raise ValueError("emitted_at must be finite.")
        expected = dict(self.spec.joint_dof_by_effector)
        if set(qpos_by_effector) != set(expected):
            raise ValueError(
                "Synthetic qpos effectors must exactly match the source spec."
            )
        command = TeleopCommand(timestamp=timestamp)
        for effector_name, dof in self.spec.joint_dof_by_effector:
            value = np.asarray(qpos_by_effector[effector_name], dtype=np.float64)
            if value.shape != (dof,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{effector_name} must be a finite qpos vector with shape ({dof},)."
                )
            side = (
                "left"
                if effector_name == "left_arm"
                else "right" if effector_name == "right_arm" else None
            )
            command.add_command(
                EffectorCommand(
                    effector_name=effector_name,
                    side=side,
                    command_type="joint_vector",
                    value=value.copy(),
                )
            )
        emission = HumanCommandEmission.capture(
            emission_sequence=self._next_sequence,
            emitted_at=timestamp,
            contributions=((self.spec.contribution_name, command),),
            merged_command=command,
        )
        self._next_sequence += 1
        return emission


@dataclass(frozen=True)
class SyntheticHumanPoseSourceSpec:
    """Device-free pose source matching the mainline mixed-action boundary.

    The historical default remains bilateral for compatibility; custom
    effectors are represented by the same tuple/hash contract rather than a
    second source implementation.
    """

    source_id: str = "synthetic-human-pose"
    reference_frame_by_effector: tuple[tuple[str, str], ...] = (
        ("left_arm", "left_arm_base"),
        ("right_arm", "right_arm_base"),
    )
    pose_frame_contract_hash: str = "rm65-arm-base-tool-tcp-pose-frame-v1"
    action_spec_hash_by_effector: tuple[tuple[str, str], ...] = (
        ("left_arm", "rm65-tool-tcp-pose-v1"),
        ("right_arm", "rm65-tool-tcp-pose-v1"),
    )
    units_convention_id: str = "si_m_rad_xyzw_v1"
    contribution_name: str = "control_station_synthetic_human_pose_v1"

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        contribution_name = str(self.contribution_name).strip()
        pose_frame_hash = str(self.pose_frame_contract_hash).strip()
        units = str(self.units_convention_id).strip()
        if not source_id or not contribution_name or not pose_frame_hash or not units:
            raise ValueError(
                "source_id, contribution_name, pose frame hash, and units must be non-empty."
            )
        normalized: list[tuple[str, str]] = []
        for effector_name, frame_name in self.reference_frame_by_effector:
            effector = str(effector_name).strip()
            frame = str(frame_name).strip()
            if not effector or not frame:
                raise ValueError("Pose effectors require non-empty names and frames.")
            normalized.append((effector, frame))
        if not normalized or len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("Synthetic pose effector names must be unique.")
        action_specs = tuple(
            (str(effector).strip(), str(spec_hash).strip())
            for effector, spec_hash in self.action_spec_hash_by_effector
        )
        if (
            any(not effector or not spec_hash for effector, spec_hash in action_specs)
            or len({effector for effector, _ in action_specs}) != len(action_specs)
            or {effector for effector, _ in action_specs}
            != {effector for effector, _ in normalized}
        ):
            raise ValueError(
                "Pose action-spec hashes must exactly cover the source effectors."
            )
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "contribution_name", contribution_name)
        object.__setattr__(self, "pose_frame_contract_hash", pose_frame_hash)
        object.__setattr__(self, "units_convention_id", units)
        object.__setattr__(
            self,
            "reference_frame_by_effector",
            tuple(normalized),
        )
        object.__setattr__(
            self,
            "action_spec_hash_by_effector",
            action_specs,
        )

    @property
    def effector_names(self) -> tuple[str, ...]:
        return tuple(effector for effector, _ in self.reference_frame_by_effector)

    @property
    def contract_id(self) -> str:
        if self.effector_names == ("left_arm", "right_arm"):
            return SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1
        return SYNTHETIC_HUMAN_POSE_GENERIC_SOURCE_CONTRACT_ID_V1

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "contract_id": self.contract_id,
                "source_id": self.source_id,
                "reference_frame_by_effector": self.reference_frame_by_effector,
                "pose_frame_contract_hash": self.pose_frame_contract_hash,
                "action_spec_hash_by_effector": self.action_spec_hash_by_effector,
                "units_convention_id": self.units_convention_id,
                "contribution_name": self.contribution_name,
            },
            domain=SYNTHETIC_HUMAN_POSE_SOURCE_HASH_DOMAIN_V1,
        )


class SyntheticHumanPoseSource:
    """Produce exact pose emissions without owning a device or worker."""

    def __init__(self, spec: SyntheticHumanPoseSourceSpec) -> None:
        if not isinstance(spec, SyntheticHumanPoseSourceSpec):
            raise TypeError("spec must be a SyntheticHumanPoseSourceSpec.")
        self.spec = spec
        self._next_sequence = 0

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    def emit(
        self,
        pose_matrix_by_effector: dict[
            str,
            np.ndarray | tuple[tuple[float, ...], ...] | list[list[float]],
        ],
        *,
        emitted_at: float,
    ) -> HumanCommandEmission:
        timestamp = float(emitted_at)
        if not math.isfinite(timestamp):
            raise ValueError("emitted_at must be finite.")
        expected = dict(self.spec.reference_frame_by_effector)
        if set(pose_matrix_by_effector) != set(expected):
            raise ValueError("Synthetic pose effectors must exactly match the source spec.")
        command = TeleopCommand(timestamp=timestamp)
        for effector_name, frame_name in self.spec.reference_frame_by_effector:
            value = np.asarray(
                pose_matrix_by_effector[effector_name],
                dtype=np.float64,
            )
            if value.shape != (4, 4) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{effector_name} must be a finite homogeneous 4x4 pose."
                )
            if not np.allclose(value[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
                raise ValueError(f"{effector_name} pose has an invalid homogeneous row.")
            side = (
                "left"
                if effector_name == "left_arm"
                else "right" if effector_name == "right_arm" else None
            )
            command.add_command(
                EffectorCommand(
                    effector_name=effector_name,
                    side=side,
                    command_type="cartesian_pose",
                    value=value.copy(),
                    frame_name=frame_name,
                )
            )
        emission = HumanCommandEmission.capture(
            emission_sequence=self._next_sequence,
            emitted_at=timestamp,
            contributions=((self.spec.contribution_name, command),),
            merged_command=command,
        )
        self._next_sequence += 1
        return emission


__all__ = [
    "SYNTHETIC_HUMAN_QPOS_SOURCE_CONTRACT_ID_V1",
    "SYNTHETIC_HUMAN_POSE_SOURCE_CONTRACT_ID_V1",
    "SyntheticHumanPoseSource",
    "SyntheticHumanPoseSourceSpec",
    "SyntheticHumanQposSource",
    "SyntheticHumanQposSourceSpec",
]
