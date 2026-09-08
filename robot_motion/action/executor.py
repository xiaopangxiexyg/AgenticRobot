from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from robot_motion.runtime import EffectorCommand, Rig, TeleopCommand
from robot_motion.runtime.dispatch import CommandDispatchReceipt

from .arm_spec import ArmActionSpec, arm_action_specs_from_metadata
from .hand_spec import hand_action_specs_from_metadata
from .schema import (
    ActionExecutionResult,
    HAND16_SLOT_DOF,
    RobotAction,
    effector_name,
)


@dataclass(frozen=True)
class ActionResolutionFailure:
    """Expected reason why an action could not become an executable command.

    This is target-resolution evidence, not a canonical safety verdict.  The
    resolver may expose a small, backend-neutral diagnostic summary without
    leaking its private planner state into the action contract.
    """

    reason_code: str
    resolver_status: str
    message: str
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reason_code = str(self.reason_code).strip()
        resolver_status = str(self.resolver_status).strip()
        message = str(self.message).strip()
        if not reason_code or any(
            not (character.islower() or character.isdigit() or character == "_")
            for character in reason_code
        ):
            raise ValueError(
                "reason_code must use non-empty lower_snake_case tokens."
            )
        if not resolver_status:
            raise ValueError("resolver_status must be non-empty.")
        if not message:
            raise ValueError("message must be non-empty.")
        diagnostics = {
            str(name): value
            for name, value in dict(self.diagnostics).items()
            if str(name).strip()
        }
        unsupported = {
            name: type(value).__name__
            for name, value in diagnostics.items()
            if value is not None
            and not isinstance(value, (bool, int, float, str))
        }
        if unsupported:
            raise TypeError(
                "Action resolution diagnostics must be bounded scalar "
                f"values; unsupported={unsupported}."
            )
        if any(
            isinstance(value, float) and not np.isfinite(value)
            for value in diagnostics.values()
        ):
            raise ValueError(
                "Action resolution diagnostics must contain finite floats."
            )
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "resolver_status", resolver_status)
        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(diagnostics),
        )

    @property
    def outcome_code(self) -> str:
        return f"target_private_resolution_failure.{self.reason_code}"


class ActionResolutionError(RuntimeError):
    """Typed expected failure raised by a concrete action resolver."""

    def __init__(self, failure: ActionResolutionFailure) -> None:
        if not isinstance(failure, ActionResolutionFailure):
            raise TypeError("failure must be an ActionResolutionFailure.")
        self.failure = failure
        super().__init__(
            f"{failure.reason_code}: {failure.message} "
            f"(resolver_status={failure.resolver_status})"
        )


def xyzquat_to_matrix(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"Expected xyzquat pose shape (7,), got {pose.shape}.")
    xyz = pose[:3].astype(np.float32, copy=True)
    quat = np.asarray(pose[3:7], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("Pose quaternion norm is zero.")
    x, y, z, w = quat / norm
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    matrix[:3, 3] = xyz
    return matrix


class ActionResolver(Protocol):
    def resolve(
        self,
        action: RobotAction,
        *,
        observation: Any = None,
        context: Any = None,
    ) -> TeleopCommand:
        ...


class CommandDispatcher(Protocol):
    def dispatch(self, command: TeleopCommand, *, context: Any = None) -> object | None:
        ...


class RobotActionResolver:
    """Resolve canonical actions into the common command schema."""

    def resolve(
        self,
        action: RobotAction,
        *,
        observation: Any = None,
        context: Any = None,
    ) -> TeleopCommand:
        command = TeleopCommand(
            timestamp=action.timestamp,
            metadata={
                "action_space": action.action_space,
                "action_source": action.source,
                **dict(action.metadata),
            },
        )
        active = set(action.active_effectors)
        arm_specs = arm_action_specs_from_metadata(action.metadata)
        hand_specs = hand_action_specs_from_metadata(action.metadata)
        for side, pose in action.arm_pose_by_side.items():
            effector = effector_name(side, "arm")
            if effector not in active:
                continue
            pose_matrix = xyzquat_to_matrix(pose)
            arm_spec = arm_specs.get(side)
            if arm_spec is None:
                arm_spec = ArmActionSpec.tool_tcp(
                    side,
                    reference_frame=action.arm_frame(side),
                )
            controlled_frame = action.arm_controlled_frame(side)
            command.add_command(
                EffectorCommand(
                    effector_name=effector,
                    side=side,
                    command_type="cartesian_pose",
                    value=pose_matrix,
                    frame_name=action.arm_frame(side),
                    metadata={
                        "source": "robot_action",
                        "action_space": action.action_space,
                        "action_source": action.source,
                        "action_timestamp": action.timestamp,
                        "target_pose_base": pose_matrix.copy(),
                        "target_pose_controlled_frame": controlled_frame,
                        "arm_controlled_frame": controlled_frame,
                        "arm_reference_frame": action.arm_frame(side),
                        "arm_action_spec": arm_spec.to_metadata(),
                        "target_xyzquat": np.asarray(pose, dtype=np.float32).copy(),
                    },
                )
            )
        for side, qpos in action.arm_qpos_by_side.items():
            effector = effector_name(side, "arm")
            if effector not in active:
                continue
            command.add_command(
                EffectorCommand(
                    effector_name=effector,
                    side=side,
                    command_type="joint_vector",
                    value=np.asarray(qpos, dtype=np.float32).copy(),
                    frame_name=None,
                    metadata={
                        "source": "robot_action",
                        "action_space": action.action_space,
                        "action_source": action.source,
                        "action_timestamp": action.timestamp,
                        "arm_controlled_frame": action.arm_controlled_frame(side),
                        "arm_reference_frame": action.arm_frame(side),
                        "arm_action_spec": (
                            arm_specs.get(side)
                            or ArmActionSpec.tool_tcp(side, reference_frame=action.arm_frame(side))
                        ).to_metadata(),
                    },
                )
            )
        for side, qpos in action.hand_qpos_by_side.items():
            effector = effector_name(side, "hand")
            if effector not in active:
                continue
            qpos_array = np.asarray(qpos, dtype=np.float32).reshape(-1)
            spec = hand_specs.get(side)
            if spec is not None:
                qpos_array = spec.validate_vector(qpos_array)
                command_value = spec.active_values(qpos_array)
                active_mask = list(spec.active_mask or ())
                spec_metadata = spec.to_metadata()
            else:
                if qpos_array.size == HAND16_SLOT_DOF or not np.all(np.isfinite(qpos_array)):
                    raise ValueError(
                        f"{effector} action requires hand_action_specs_by_side metadata for q16/partial hand slots."
                    )
                command_value = qpos_array.copy()
                active_mask = np.isfinite(qpos_array).tolist()
                spec_metadata = None
            command.add_command(
                EffectorCommand(
                    effector_name=effector,
                    side=side,
                    command_type="joint_vector",
                    value=command_value.astype(np.float32, copy=True),
                    frame_name=None,
                    metadata={
                        "source": "robot_action",
                        "action_space": action.action_space,
                        "action_source": action.source,
                        "action_timestamp": action.timestamp,
                        "hand_slot_dof": int(qpos_array.size),
                        "hand_active_mask": active_mask,
                        "hand_qpos_full": qpos_array.copy(),
                        "hand_invalid_slot_value": (
                            spec.invalid_slot_value if spec is not None else "none"
                        ),
                        "hand_slot_semantics": (
                            spec.semantics
                            if spec is not None
                            else (
                                "Every finite slot is an active actuator target; "
                                "no inactive padding slots are present."
                            )
                        ),
                        "hand_action_spec": spec_metadata,
                    },
                )
            )
        for side, target in action.end_effector_targets_by_side.items():
            effector = effector_name(side, "hand")
            if effector not in active:
                continue
            command.add_command(
                EffectorCommand(
                    effector_name=effector,
                    side=side,
                    command_type=f"end_effector_{target.value_type}",
                    value=target.values.copy(),
                    frame_name=None,
                    metadata={
                        "source": "robot_action",
                        "action_space": action.action_space,
                        "action_source": action.source,
                        "action_timestamp": action.timestamp,
                        "end_effector_adapter": target.adapter,
                        "end_effector_value_type": target.value_type,
                        "end_effector_names": list(target.names),
                        **dict(target.metadata),
                    },
                )
            )
        return command


def teleop_command_semantic_payload(command: TeleopCommand) -> dict[str, Any]:
    """Return the transport-neutral semantic payload of a resolved command."""

    if not isinstance(command, TeleopCommand):
        raise TypeError("command must be a TeleopCommand.")
    return {
        "timestamp": command.timestamp,
        "commands": {
            name: {
                "effector_name": item.effector_name,
                "side": item.side,
                "command_type": item.command_type,
                "value": item.value,
                "frame_name": item.frame_name,
                "metadata": item.metadata,
            }
            for name, item in sorted(command.commands.items())
        },
        "metadata": command.metadata,
    }


class NullCommandDispatcher:
    def __init__(self):
        self.commands: list[TeleopCommand] = []

    def dispatch(self, command: TeleopCommand, *, context: Any = None) -> None:
        self.commands.append(command)


class RigCommandDispatcher:
    def __init__(self, rig: Rig):
        self.rig = rig

    def start(self) -> None:
        self.rig.start()

    def dispatch(self, command: TeleopCommand, *, context: Any = None) -> object | None:
        phase_recorder = getattr(context, "record_phase", None)
        return self.rig.send_commands(command, phase_recorder=phase_recorder)

    def close(self) -> None:
        self.rig.close()


@dataclass
class ActionExecutorConfig:
    dispatch_enabled: bool = True
    raise_on_error: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class ActionExecutor:
    def __init__(
        self,
        *,
        resolver: ActionResolver | None = None,
        dispatcher: CommandDispatcher | None = None,
        config: ActionExecutorConfig | None = None,
    ):
        self.resolver = resolver if resolver is not None else RobotActionResolver()
        self.dispatcher = dispatcher
        self.config = config if config is not None else ActionExecutorConfig()

    def start(self) -> None:
        start = getattr(self.dispatcher, "start", None)
        if callable(start):
            start()

    def execute(
        self,
        action: RobotAction,
        *,
        observation: Any = None,
        context: Any = None,
    ) -> ActionExecutionResult:
        try:
            command = self.resolver.resolve(
                action,
                observation=observation,
                context=context,
            )
            sent = False
            if self.config.dispatch_enabled and self.dispatcher is not None:
                dispatch_result = self.dispatcher.dispatch(command, context=context)
                receipts = (
                    tuple(dispatch_result.values())
                    if isinstance(dispatch_result, dict)
                    else (dispatch_result,)
                )
                sent = all(
                    receipt.accepted
                    for receipt in receipts
                    if isinstance(receipt, CommandDispatchReceipt)
                )
            return ActionExecutionResult(
                timestamp=time.time(),
                command=command,
                sent=sent,
                metadata={
                    "dispatch_enabled": bool(self.config.dispatch_enabled),
                    **dict(self.config.metadata),
                },
            )
        except Exception as exc:
            if self.config.raise_on_error:
                raise
            return ActionExecutionResult(
                timestamp=time.time(),
                command=None,
                sent=False,
                faults=(f"{type(exc).__name__}:{exc}",),
                metadata={
                    "dispatch_enabled": bool(self.config.dispatch_enabled),
                    **dict(self.config.metadata),
                },
            )

    def close(self) -> None:
        close = getattr(self.dispatcher, "close", None)
        if callable(close):
            close()


__all__ = [
    "ActionExecutor",
    "ActionExecutorConfig",
    "ActionResolutionError",
    "ActionResolutionFailure",
    "ActionResolver",
    "CommandDispatcher",
    "NullCommandDispatcher",
    "RigCommandDispatcher",
    "RobotActionResolver",
    "teleop_command_semantic_payload",
    "xyzquat_to_matrix",
]
