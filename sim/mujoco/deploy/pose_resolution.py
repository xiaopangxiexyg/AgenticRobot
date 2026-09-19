"""Scoped shared-safe pose to qpos execution resolution for MuJoCo.

This adapter sits *after* canonical shared safety and before the qpos
reference receiver.  It never changes the replay proposal or selected/safe
action stage.  The coupled RM65 planner result is a separate immutable
execution-target value whose lineage can be checked by the receiver.

The arm resolver intentionally accepts only exact per-arm base-frame
``tool_tcp`` poses.  Already-safe hand qpos may accompany those poses and is
carried unchanged into the resolved command under an explicit derived
resolver contract.  Canonical-desk replay requires a separately versioned and
verified transform; this module must not guess or silently apply one.
"""

from __future__ import annotations

import gc
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    BIMANUAL_EE_POSE_HAND16_V1,
    BIMANUAL_EE_POSE_V1,
    BIMANUAL_QPOS_HAND16_V1,
    BIMANUAL_QPOS_V1,
    ActionResolutionError,
    ActionResolutionFailure,
    AuthoritativeCommandDispatchContext,
    AuthoritativeTargetDispatchContext,
    CanonicalDeskToRuntimePoseTransformer,
    ComponentExecutionTarget,
    CompoundDispatchManifest,
    ResolvedExecutionTarget,
    RobotAction,
    TargetProcessingFactReference,
    build_component_execution_target,
    build_resolved_execution_target,
    hand_action_specs_from_metadata,
    project_robot_action_to_components,
    validate_compound_dispatch_manifest,
)
from robot_motion.action.executor import RobotActionResolver, xyzquat_to_matrix
from robot_motion.action.proposal import (
    freeze_robot_action,
    robot_action_semantic_payload,
)
from robot_motion.action.safety_anchor import safety_anchor_action_hash
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.pose_smoothing import BimanualPoseSmoother
from robot_motion.rm65.constants import TOOL_TCP_FRAME
from robot_motion.runtime import TeleopCommand

from ..ik import IKResult, PoseTarget, RM65CanonicalIK, RM65RetargeterIK

POSE_TO_QPOS_RESOLUTION_HASH_DOMAIN_V1 = (
    "dexteleop.mujoco.pose_to_qpos_execution_resolution.v1"
)
POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1 = (
    "mujoco-rm65-coupled-arm-base-pose-to-qpos-v1"
)
POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1,
        "input_action_space": BIMANUAL_EE_POSE_V1,
        "required_components": ("left_arm", "right_arm"),
        "reference_frame_by_side": {
            "left": "left_arm_base",
            "right": "right_arm_base",
        },
        "controlled_frame": TOOL_TCP_FRAME,
        "output_action_space": "bimanual_qpos_v1",
        "source_side_transform": "forbidden",
        "solver": "robot_motion.rm65.position_collision",
    },
    domain="dexteleop.mujoco.pose_to_qpos_resolver_contract.v1",
)

RETARGETER_POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1 = (
    "mujoco-rm65-dual-arm-retargeter-arm-base-pose-to-qpos-v1"
)
RETARGETER_POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": RETARGETER_POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1,
        "input_action_space": BIMANUAL_EE_POSE_V1,
        "required_components": ("left_arm", "right_arm"),
        "reference_frame_by_side": {
            "left": "left_arm_base",
            "right": "right_arm_base",
        },
        "controlled_frame": TOOL_TCP_FRAME,
        "output_action_space": "bimanual_qpos_v1",
        "source_side_transform": "forbidden",
        "solver": "robot_motion.rm65.dual_arm_collision_aware",
    },
    domain="dexteleop.mujoco.pose_to_qpos_resolver_contract.v1",
)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_plain(item) for item in value)
    return value


@dataclass(frozen=True)
class MujocoPoseToQposResolution:
    """Immutable evidence for one successful coupled IK resolution."""

    resolution_id: str
    input_safe_action: RobotAction
    input_safe_action_hash: str
    solver_input_pose_action: RobotAction
    solver_input_pose_action_hash: str
    output_qpos_action: RobotAction
    output_qpos_action_hash: str
    resolver_contract_id: str
    resolver_contract_hash: str
    transform_receipt_id: str | None
    transform_contract_hash: str | None
    solver_id: str
    solver_status: str
    solver_hold: bool
    position_error_m_by_side: Mapping[str, float]
    orientation_error_rad_by_side: Mapping[str, float]
    solver_metadata: Mapping[str, Any]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "resolution_id",
            "input_safe_action_hash",
            "output_qpos_action_hash",
            "resolver_contract_id",
            "resolver_contract_hash",
            "solver_id",
            "solver_status",
            "semantic_content_hash",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        source = freeze_robot_action(self.input_safe_action)
        solver_input = freeze_robot_action(self.solver_input_pose_action)
        output = freeze_robot_action(self.output_qpos_action)
        if safety_anchor_action_hash(source) != self.input_safe_action_hash:
            raise ValueError("Pose resolution input safe-action hash mismatch.")
        if safety_anchor_action_hash(output) != self.output_qpos_action_hash:
            raise ValueError("Pose resolution qpos-action hash mismatch.")
        if safety_anchor_action_hash(solver_input) != self.solver_input_pose_action_hash:
            raise ValueError("Pose resolution solver-input action hash mismatch.")
        valid_pairs = {
            BIMANUAL_EE_POSE_V1: BIMANUAL_QPOS_V1,
            BIMANUAL_EE_POSE_HAND16_V1: BIMANUAL_QPOS_HAND16_V1,
        }
        if source.action_space not in valid_pairs:
            raise ValueError("Pose resolution input must use a pose action space.")
        if output.action_space != valid_pairs[source.action_space]:
            raise ValueError("Pose resolution output action space does not match input.")
        if solver_input.action_space != source.action_space:
            raise ValueError("Pose resolution solver input action space mismatch.")
        if (self.transform_receipt_id is None) != (
            self.transform_contract_hash is None
        ):
            raise ValueError("Pose transform receipt id/hash must be paired.")
        if self.solver_hold:
            raise ValueError("A held IK result cannot be an executable resolution.")
        object.__setattr__(self, "input_safe_action", source)
        object.__setattr__(self, "solver_input_pose_action", solver_input)
        object.__setattr__(self, "output_qpos_action", output)
        object.__setattr__(
            self,
            "position_error_m_by_side",
            MappingProxyType(
                {
                    str(side): float(value)
                    for side, value in self.position_error_m_by_side.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "orientation_error_rad_by_side",
            MappingProxyType(
                {
                    str(side): float(value)
                    for side, value in self.orientation_error_rad_by_side.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "solver_metadata",
            MappingProxyType(dict(_plain(self.solver_metadata))),
        )


class MujocoCoupledPoseToQposResolver:
    """Resolve shared-safe pose actions through explicit frame evidence and IK."""

    def __init__(
        self,
        ik: RM65CanonicalIK | RM65RetargeterIK,
        *,
        pose_transformer: CanonicalDeskToRuntimePoseTransformer | None = None,
        target_pose_smoother: BimanualPoseSmoother | None = None,
        pose_target_safety_filter: Callable[[RobotAction], RobotAction] | None = None,
        support_only_arm_sides: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(ik, (RM65CanonicalIK, RM65RetargeterIK)):
            raise TypeError(
                "ik must be RM65CanonicalIK or RM65RetargeterIK."
            )
        self.ik = ik
        self.pose_transformer = pose_transformer
        if target_pose_smoother is not None and not isinstance(
            target_pose_smoother,
            BimanualPoseSmoother,
        ):
            raise TypeError(
                "target_pose_smoother must be a BimanualPoseSmoother or None."
            )
        self.target_pose_smoother = target_pose_smoother
        if pose_target_safety_filter is not None and not callable(
            pose_target_safety_filter
        ):
            raise TypeError("pose_target_safety_filter must be callable or None.")
        safety_filter_hash = (
            None
            if pose_target_safety_filter is None
            else str(
                getattr(
                    pose_target_safety_filter,
                    "filter_contract_hash",
                    "",
                )
            ).strip()
        )
        if pose_target_safety_filter is not None and not safety_filter_hash:
            raise ValueError(
                "pose_target_safety_filter requires filter_contract_hash."
            )
        self.pose_target_safety_filter = pose_target_safety_filter
        support_sides = tuple(
            str(side).strip().lower() for side in support_only_arm_sides
        )
        if (
            len(support_sides) != len(set(support_sides))
            or set(support_sides) - {"left", "right"}
            or set(support_sides) == {"left", "right"}
        ):
            raise ValueError(
                "support_only_arm_sides must be a unique proper arm-side subset."
            )
        self.support_only_arm_sides = support_sides
        self.active_arm_sides = tuple(
            side for side in ("left", "right") if side not in support_sides
        )
        transform_hash = (
            None
            if pose_transformer is None
            else pose_transformer.binding_receipt.semantic_content_hash
        )
        collision_constraints_enabled = bool(
            getattr(self.ik.planner, "collision_pair_count", 0) > 0
        )
        retargeter_backend = isinstance(self.ik, RM65RetargeterIK)
        if retargeter_backend:
            if not collision_constraints_enabled:
                raise ValueError(
                    "RM65RetargeterIK requires an auditable collision profile."
                )
            base_contract_id = RETARGETER_POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1
            base_contract_hash = RETARGETER_POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1
        elif collision_constraints_enabled:
            base_contract_id = POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1
            base_contract_hash = POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1
        else:
            base_contract_id = (
                "mujoco-rm65-coupled-arm-base-pose-to-qpos-no-collision-v1"
            )
            base_contract_hash = canonical_content_hash(
                {
                    "base_resolver_contract_hash": (
                        POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1
                    ),
                    "collision_constraints_enabled": False,
                    "joint_constraints_unchanged": True,
                },
                domain=(
                    "dexteleop.mujoco.pose_to_qpos_no_collision_resolver.v1"
                ),
            )
        if transform_hash is None:
            transformed_contract_id = base_contract_id
        elif retargeter_backend:
            transformed_contract_id = (
                "mujoco-rm65-dual-arm-retargeter-canonical-desk-"
                "pose-to-qpos-v1"
            )
        elif collision_constraints_enabled:
            transformed_contract_id = (
                "mujoco-rm65-coupled-canonical-desk-pose-to-qpos-v1"
            )
        else:
            transformed_contract_id = (
                "mujoco-rm65-coupled-canonical-desk-pose-to-qpos-"
                "no-collision-v1"
            )
        transformed_contract_hash = (
            base_contract_hash
            if transform_hash is None
            else canonical_content_hash(
                {
                    "base_resolver_contract_hash": (
                        base_contract_hash
                    ),
                    "transform_contract_hash": transform_hash,
                    "transform_stage": "after_shared_safety_before_coupled_ik",
                },
                domain="dexteleop.mujoco.transformed_pose_to_qpos_resolver.v1",
            )
        )
        safety_contract_id = (
            transformed_contract_id
            if safety_filter_hash is None
            else f"{transformed_contract_id}+tcp-envelope-v1"
        )
        safety_contract_hash = (
            transformed_contract_hash
            if safety_filter_hash is None
            else canonical_content_hash(
                {
                    "base_resolver_contract_hash": transformed_contract_hash,
                    "pose_target_safety_filter_hash": safety_filter_hash,
                    "filter_stage": "final_after_pose_transform_before_coupled_ik",
                },
                domain="dexteleop.mujoco.safe_pose_to_qpos_resolver.v1",
            )
        )
        smoothing_contract = None
        if target_pose_smoother is not None:
            smoothing_contract = dict(target_pose_smoother.config.to_metadata())
            # Host-specific paths are provenance, not control semantics.
            smoothing_contract.pop("source_path", None)
        self.resolver_contract_id = (
            safety_contract_id
            if smoothing_contract is None
            else f"{safety_contract_id}+target-pose-smoothing-v1"
        )
        self.resolver_contract_hash = (
            safety_contract_hash
            if smoothing_contract is None
            else canonical_content_hash(
                {
                    "base_resolver_contract_hash": safety_contract_hash,
                    "target_pose_smoothing": smoothing_contract,
                    "smoothing_stage": (
                        "after_pose_transform_before_final_pose_safety_and_coupled_ik"
                    ),
                },
                domain=(
                    "dexteleop.mujoco.smoothed_pose_to_qpos_resolver.v1"
                ),
            )
        )
        if self.support_only_arm_sides:
            self.resolver_contract_id = (
                f"{self.resolver_contract_id}+inactive-support-hold-v1"
            )
            self.resolver_contract_hash = canonical_content_hash(
                {
                    "base_resolver_contract_hash": self.resolver_contract_hash,
                    "active_arm_sides": self.active_arm_sides,
                    "support_only_arm_sides": self.support_only_arm_sides,
                    "support_semantics": (
                        "measured_pose_for_coupled_context_then_exact_qpos_hold"
                    ),
                },
                domain=(
                    "dexteleop.mujoco.pose_to_qpos_support_arm_hold_resolver.v1"
                ),
            )
        self._command_resolver = RobotActionResolver()
        self._lock = threading.RLock()
        self._resolution_by_id: dict[str, MujocoPoseToQposResolution] = {}
        # Runtime-only timing evidence.  These values are deliberately kept
        # outside the semantic resolution payload/hash: measuring a resolve
        # must never change command identity or acceptance behavior.
        self._last_resolve_diagnostics: dict[str, object] = {}

    @property
    def last_resolve_diagnostics(self) -> Mapping[str, object]:
        """Return timing/status evidence for the most recent resolve call."""

        with self._lock:
            return dict(self._last_resolve_diagnostics)

    def clear_last_resolve_diagnostics(self) -> None:
        """Clear the previous timing sample before a new runtime step."""

        with self._lock:
            self._last_resolve_diagnostics = {}

    def reset_temporal_state(self) -> None:
        """Clear stateful target shaping at HOLD/rebuild boundaries."""

        if self.target_pose_smoother is None:
            return
        with self._lock:
            self.target_pose_smoother.reset()

    def resolve(
        self,
        action: RobotAction,
        *,
        observation: Any = None,
        context: Any = None,
    ) -> TeleopCommand:
        del observation, context
        resolve_started = time.perf_counter()
        timing: dict[str, object] = {"resolve_invoked": True}
        with self._lock:
            self._last_resolve_diagnostics = {}
        safe_action = freeze_robot_action(action)
        if self.pose_transformer is None:
            solver_input = safe_action
            transform_result = None
            timing["pose_transform_elapsed_sec"] = 0.0
        else:
            transform_started = time.perf_counter()
            transform_result = self.pose_transformer.transform(safe_action)
            solver_input = transform_result.transformed_action
            timing["pose_transform_elapsed_sec"] = max(
                0.0, time.perf_counter() - transform_started
            )
        target_pose_smoothing: dict[str, object] = {
            "enabled": False,
            "status": "not_configured",
        }
        if self.target_pose_smoother is not None:
            smoothing_started = time.perf_counter()
            with self._lock:
                smoothed_targets, target_pose_smoothing = (
                    self.target_pose_smoother.smooth_pair(
                        left_target=xyzquat_to_matrix(
                            solver_input.arm_pose_by_side["left"]
                        ),
                        right_target=xyzquat_to_matrix(
                            solver_input.arm_pose_by_side["right"]
                        ),
                        timestamp=solver_input.timestamp,
                    )
                )
            smoothed_vectors = {
                side: pose_matrix_to_xyzquat(smoothed_targets[side])
                for side in ("left", "right")
            }
            if any(value is None for value in smoothed_vectors.values()):
                raise ValueError("Target pose smoother produced an invalid pose.")
            solver_input = freeze_robot_action(
                replace(
                    solver_input,
                    arm_pose_by_side=smoothed_vectors,
                )
            )
            timing["pose_smoothing_elapsed_sec"] = max(
                0.0, time.perf_counter() - smoothing_started
            )
        else:
            timing["pose_smoothing_elapsed_sec"] = 0.0
        if self.pose_target_safety_filter is not None:
            safety_filter_started = time.perf_counter()
            solver_input = freeze_robot_action(
                self.pose_target_safety_filter(solver_input)
            )
            timing["pose_safety_filter_elapsed_sec"] = max(
                0.0, time.perf_counter() - safety_filter_started
            )
        else:
            timing["pose_safety_filter_elapsed_sec"] = 0.0
        self._validate_arm_base_input(solver_input)
        targets = {
            side: PoseTarget(
                side=side,
                pose=xyzquat_to_matrix(solver_input.arm_pose_by_side[side]),
                controlled_frame=solver_input.arm_controlled_frame(side),
            )
            for side in ("left", "right")
        }
        timing["pre_ik_elapsed_sec"] = max(
            0.0, time.perf_counter() - resolve_started
        )
        ik_started = time.perf_counter()
        try:
            result = self.ik.solve(targets, active_sides=self.active_arm_sides)
        except BaseException as error:
            timing.update(
                {
                    "ik_elapsed_sec": max(0.0, time.perf_counter() - ik_started),
                    "ik_status": "exception",
                    "ik_error_type": type(error).__name__,
                    "resolve_status": "error",
                    "resolve_total_elapsed_sec": max(
                        0.0, time.perf_counter() - resolve_started
                    ),
                }
            )
            with self._lock:
                self._last_resolve_diagnostics = dict(timing)
            raise
        timing.update(
            {
                "ik_elapsed_sec": max(0.0, time.perf_counter() - ik_started),
                "ik_status": str(result.status),
                "ik_solver": str(result.solver),
                "ik_hold": bool(result.hold),
            }
        )
        post_ik_started = time.perf_counter()
        # These are runtime-only observations.  They deliberately stay out of
        # the semantic resolution payload so profiling cannot change command
        # identity or acceptance behavior.
        timing["gc_count_before"] = list(gc.get_count())
        with self._lock:
            timing["resolution_cache_size_before"] = len(self._resolution_by_id)
        if not result.success or result.hold:
            metadata = dict(result.metadata or {})
            failure = dict(metadata.get("failure") or {})
            reason_code = str(
                failure.get("code")
                or (
                    "resolver_hold"
                    if result.hold
                    else "resolver_failed"
                )
            ).strip()
            attempts = tuple(
                dict(value)
                for value in metadata.get("collision_check_attempts", ())
                if isinstance(value, Mapping)
            )
            finite_distances = tuple(
                float(value["distance_m"])
                for value in attempts
                if value.get("distance_m") is not None
                and np.isfinite(float(value["distance_m"]))
            )
            best_attempt = (
                None
                if not finite_distances
                else max(
                    (
                        value
                        for value in attempts
                        if value.get("distance_m") is not None
                        and np.isfinite(float(value["distance_m"]))
                    ),
                    key=lambda value: float(value["distance_m"]),
                )
            )
            collision_min_distance = metadata.get(
                "collision_min_distance_m"
            )
            collision_epsilon = metadata.get(
                "collision_distance_epsilon_m"
            )
            diagnostics = {
                "solver": str(result.solver),
                "hold": bool(result.hold),
                "collision_min_distance_m": collision_min_distance,
                "collision_distance_epsilon_m": collision_epsilon,
                "collision_threshold_m": (
                    None
                    if collision_min_distance is None
                    or collision_epsilon is None
                    else float(collision_min_distance)
                    - float(collision_epsilon)
                ),
                "hold_reference_distance_m": metadata.get(
                    "collision_distance_m"
                ),
                "best_candidate_distance_m": (
                    None if not finite_distances else max(finite_distances)
                ),
                "collision_attempt_count": len(attempts),
                "collision_pair_index": (
                    None
                    if best_attempt is None
                    else best_attempt.get("collision_pair_index")
                ),
                "collision_geometry_1": (
                    None
                    if best_attempt is None
                    else best_attempt.get("collision_geometry_1")
                ),
                "collision_geometry_2": (
                    None
                    if best_attempt is None
                    else best_attempt.get("collision_geometry_2")
                ),
                "target_pose_smoothing_enabled": bool(
                    target_pose_smoothing.get("enabled", False)
                ),
                "target_pose_smoothing_left_status": str(
                    _plain(target_pose_smoothing.get("left", {})).get(
                        "status",
                        target_pose_smoothing.get("status", "unknown"),
                    )
                ),
                "target_pose_smoothing_right_status": str(
                    _plain(target_pose_smoothing.get("right", {})).get(
                        "status",
                        target_pose_smoothing.get("status", "unknown"),
                    )
                ),
            }
            timing.update(
                {
                    "resolve_status": "rejected",
                    "resolve_failure_reason_code": reason_code,
                    "post_ik_elapsed_sec": max(
                        0.0, time.perf_counter() - post_ik_started
                    ),
                    "resolve_total_elapsed_sec": max(
                        0.0, time.perf_counter() - resolve_started
                    ),
                }
            )
            with self._lock:
                self._last_resolve_diagnostics = dict(timing)
            raise ActionResolutionError(
                ActionResolutionFailure(
                    reason_code=reason_code,
                    resolver_status=str(result.status or "unknown"),
                    message=(
                        str(result.message).strip()
                        or "Coupled pose IK produced no executable qpos target."
                    ),
                    diagnostics=diagnostics,
                )
            )
        resolution_build_started = time.perf_counter()
        resolution = self._build_resolution(
            safe_action,
            solver_input,
            result,
            transform_receipt_id=(
                None
                if transform_result is None
                else transform_result.binding_receipt_id
            ),
            transform_contract_hash=(
                None
                if transform_result is None
                else transform_result.binding_receipt_hash
            ),
            target_pose_smoothing=target_pose_smoothing,
        )
        timing["resolution_build_elapsed_sec"] = max(
            0.0, time.perf_counter() - resolution_build_started
        )
        command_resolve_started = time.perf_counter()
        command = self._command_resolver.resolve(resolution.output_qpos_action)
        timing["command_resolve_elapsed_sec"] = max(
            0.0, time.perf_counter() - command_resolve_started
        )
        command_metadata_started = time.perf_counter()
        command.metadata.update(
            {
                "pose_to_qpos_resolution_id": resolution.resolution_id,
                "pose_to_qpos_resolution_hash": resolution.semantic_content_hash,
                "input_safe_action_hash": resolution.input_safe_action_hash,
                "output_qpos_action_hash": resolution.output_qpos_action_hash,
                "resolver_contract_id": resolution.resolver_contract_id,
                "resolver_contract_hash": resolution.resolver_contract_hash,
                "transform_receipt_id": resolution.transform_receipt_id,
                "transform_contract_hash": resolution.transform_contract_hash,
                "target_pose_smoothing": resolution.solver_metadata.get(
                    "target_pose_smoothing"
                ),
            }
        )
        timing["command_metadata_elapsed_sec"] = max(
            0.0, time.perf_counter() - command_metadata_started
        )
        cache_insert_started = time.perf_counter()
        with self._lock:
            self._resolution_by_id[resolution.resolution_id] = resolution
            timing["resolution_cache_size_after"] = len(self._resolution_by_id)
        timing["resolution_cache_insert_elapsed_sec"] = max(
            0.0, time.perf_counter() - cache_insert_started
        )
        timing["gc_count_after"] = list(gc.get_count())
        with self._lock:
            timing.update(
                {
                    "resolve_status": "accepted",
                    "post_ik_elapsed_sec": max(
                        0.0, time.perf_counter() - post_ik_started
                    ),
                    "resolve_total_elapsed_sec": max(
                        0.0, time.perf_counter() - resolve_started
                    ),
                }
            )
            self._last_resolve_diagnostics = dict(timing)
        return command

    def resolution_for_command(
        self,
        command: TeleopCommand,
    ) -> MujocoPoseToQposResolution:
        resolution_id = str(
            command.metadata.get("pose_to_qpos_resolution_id") or ""
        ).strip()
        resolution_hash = str(
            command.metadata.get("pose_to_qpos_resolution_hash") or ""
        ).strip()
        if not resolution_id or not resolution_hash:
            raise ValueError("Command lacks typed pose-to-qpos resolution lineage.")
        with self._lock:
            resolution = self._resolution_by_id.get(resolution_id)
        if resolution is None:
            raise ValueError("Command references an unknown pose-to-qpos resolution.")
        if resolution.semantic_content_hash != resolution_hash:
            raise ValueError("Command pose-to-qpos resolution hash mismatch.")
        return resolution

    @staticmethod
    def _validate_arm_base_input(action: RobotAction) -> None:
        if action.action_space not in {
            BIMANUAL_EE_POSE_V1,
            BIMANUAL_EE_POSE_HAND16_V1,
        }:
            raise ValueError("Pose resolver requires a bimanual pose action space.")
        expected_effectors = {"left_arm", "right_arm"} | {
            f"{side}_hand" for side in action.hand_qpos_by_side
        }
        if set(action.active_effectors) != expected_effectors:
            raise ValueError(
                "Pose resolver requires coupled arms plus only explicit hand-qpos "
                "components."
            )
        if action.end_effector_targets_by_side:
            raise ValueError("Pose resolver does not execute native hand targets.")
        if action.hand_qpos_by_side:
            hand_specs = hand_action_specs_from_metadata(action.metadata)
            if set(hand_specs) != set(action.hand_qpos_by_side):
                raise ValueError(
                    "Pose resolver hand-qpos passthrough requires one explicit "
                    "HandActionSpec per side."
                )
            for side, values in action.hand_qpos_by_side.items():
                hand_specs[side].validate_vector(values)
        for side in ("left", "right"):
            expected_frame = f"{side}_arm_base"
            if action.arm_frame(side) != expected_frame:
                raise ValueError(
                    "Pose resolver requires exact arm-base input at the solver; "
                    f"{side} frame={action.arm_frame(side)!r}, expected={expected_frame!r}."
                )
            if action.arm_controlled_frame(side) != TOOL_TCP_FRAME:
                raise ValueError("Pose resolver v1 requires tool_tcp controlled frames.")

    def _build_resolution(
        self,
        safe_action: RobotAction,
        solver_input: RobotAction,
        result: IKResult,
        *,
        transform_receipt_id: str | None,
        transform_contract_hash: str | None,
        target_pose_smoothing: Mapping[str, object],
    ) -> MujocoPoseToQposResolution:
        output_action_space = (
            BIMANUAL_QPOS_HAND16_V1
            if safe_action.action_space == BIMANUAL_EE_POSE_HAND16_V1
            else BIMANUAL_QPOS_V1
        )
        hand_specs_metadata = (
            {}
            if not safe_action.hand_qpos_by_side
            else {
                "hand_action_specs_by_side": dict(
                    safe_action.metadata["hand_action_specs_by_side"]
                )
            }
        )
        qpos_action = RobotAction(
            action_space=output_action_space,
            arm_qpos_by_side={
                "left": np.asarray(result.qpos_by_side["left"], dtype=np.float32),
                "right": np.asarray(result.qpos_by_side["right"], dtype=np.float32),
            },
            hand_qpos_by_side=safe_action.hand_qpos_by_side,
            active_effectors=safe_action.active_effectors,
            timestamp=safe_action.timestamp,
            source="mujoco_coupled_pose_execution_target",
            metadata=hand_specs_metadata,
        )
        resolver_contract_id = self.resolver_contract_id
        resolver_contract_hash = self.resolver_contract_hash
        if safe_action.hand_qpos_by_side:
            resolver_contract_id = (
                f"{self.resolver_contract_id}+safe-hand-qpos-passthrough-v1"
            )
            resolver_contract_hash = canonical_content_hash(
                {
                    "arm_pose_resolver_contract_hash": self.resolver_contract_hash,
                    "input_action_space": safe_action.action_space,
                    "output_action_space": output_action_space,
                    "hand_component_mode": (
                        "identity_passthrough_after_canonical_safety"
                    ),
                    "hand_action_specs_by_side": hand_specs_metadata[
                        "hand_action_specs_by_side"
                    ],
                },
                domain=(
                    "dexteleop.mujoco.pose_to_qpos_with_hand_passthrough.v1"
                ),
            )
        source_hash = safety_anchor_action_hash(safe_action)
        solver_input_hash = safety_anchor_action_hash(solver_input)
        output_hash = safety_anchor_action_hash(qpos_action)
        payload = {
            "input_safe_action": robot_action_semantic_payload(safe_action),
            "input_safe_action_hash": source_hash,
            "solver_input_pose_action": robot_action_semantic_payload(solver_input),
            "solver_input_pose_action_hash": solver_input_hash,
            "output_qpos_action": robot_action_semantic_payload(qpos_action),
            "output_qpos_action_hash": output_hash,
            "resolver_contract_id": resolver_contract_id,
            "resolver_contract_hash": resolver_contract_hash,
            "transform_receipt_id": transform_receipt_id,
            "transform_contract_hash": transform_contract_hash,
            "solver_id": result.solver,
            "solver_status": str(result.status),
            "solver_hold": bool(result.hold),
            "position_error_m_by_side": {
                side: float(value)
                for side, value in result.position_error_m.items()
            },
            "orientation_error_rad_by_side": {
                side: float(value)
                for side, value in result.orientation_error_rad.items()
            },
            "solver_metadata": {
                **dict(_plain(result.metadata or {})),
                "target_pose_smoothing": _plain(target_pose_smoothing),
            },
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=POSE_TO_QPOS_RESOLUTION_HASH_DOMAIN_V1,
        )
        return MujocoPoseToQposResolution(
            resolution_id=f"mujoco-pose-to-qpos-resolution:{semantic_hash}",
            **{
                key: value
                for key, value in payload.items()
                if key not in {
                    "input_safe_action",
                    "solver_input_pose_action",
                    "output_qpos_action",
                }
            },
            input_safe_action=safe_action,
            solver_input_pose_action=solver_input,
            output_qpos_action=qpos_action,
            semantic_content_hash=semantic_hash,
        )


def build_mujoco_pose_qpos_execution_target(
    resolution: MujocoPoseToQposResolution,
    context: AuthoritativeCommandDispatchContext,
    *,
    resolved_at: float,
    target_validity_sec: float,
    clock_domain_id: str,
    input_action_schema_hash: str,
    output_target_schema_hash: str,
    compound_manifest: CompoundDispatchManifest | None = None,
) -> ResolvedExecutionTarget:
    """Bind one coupled IK result to its authoritative safe-pose lineage."""

    if not isinstance(resolution, MujocoPoseToQposResolution):
        raise TypeError("resolution must be MujocoPoseToQposResolution.")
    if not isinstance(context, AuthoritativeCommandDispatchContext):
        raise TypeError("context must be AuthoritativeCommandDispatchContext.")
    if safety_anchor_action_hash(context.safe_action) != (
        resolution.input_safe_action_hash
    ):
        raise ValueError("Pose execution resolution references another safe action.")
    if context.safe_action.action_space not in {
        BIMANUAL_EE_POSE_V1,
        BIMANUAL_EE_POSE_HAND16_V1,
    }:
        raise ValueError("Pose execution target requires a shared-safe pose action.")
    if "arms" not in context.group_segment_generation_by_group:
        raise ValueError("Pose execution target requires the coupled arms group.")
    if compound_manifest is not None:
        validate_compound_dispatch_manifest(compound_manifest, context)
    validity = float(target_validity_sec)
    if not np.isfinite(validity) or validity <= 0.0:
        raise ValueError("target_validity_sec must be finite and positive.")
    target_at = float(context.target_effective_at)
    arm_output_action = project_robot_action_to_components(
        resolution.output_qpos_action,
        ("left_arm", "right_arm"),
    )
    arm_output_hash = safety_anchor_action_hash(arm_output_action)
    target_payload = {
        "resolution_id": resolution.resolution_id,
        "resolution_hash": resolution.semantic_content_hash,
        "solver_input_pose_action": robot_action_semantic_payload(
            resolution.solver_input_pose_action
        ),
        "solver_input_pose_action_hash": (
            resolution.solver_input_pose_action_hash
        ),
        "output_qpos_action": robot_action_semantic_payload(arm_output_action),
        "output_qpos_action_hash": arm_output_hash,
        "solver_id": resolution.solver_id,
        "solver_status": resolution.solver_status,
        "transform_receipt_id": resolution.transform_receipt_id,
        "transform_contract_hash": resolution.transform_contract_hash,
    }
    if resolution.output_qpos_action.hand_qpos_by_side:
        if compound_manifest is None:
            raise ValueError(
                "Resolved pose+hand target requires a compound dispatch manifest."
            )
        target_payload["compound_output_qpos_action_hash"] = (
            resolution.output_qpos_action_hash
        )
        target_payload["compound_dispatch_manifest_id"] = (
            compound_manifest.manifest_id
        )
        target_payload["compound_dispatch_manifest_hash"] = (
            compound_manifest.semantic_content_hash
        )
    elif compound_manifest is not None:
        raise ValueError("Arms-only resolved target cannot carry a compound manifest.")
    return build_resolved_execution_target(
        control_domain_id=context.control_domain_id,
        selection_group_id="arms",
        execution_epoch=context.execution_epoch,
        group_segment_generation=context.group_segment_generation_by_group["arms"],
        source_safety_result_id=context.safety_result_id,
        safe_action=context.safe_action,
        target_kind="qpos_reference",
        component_dependencies=("left_arm", "right_arm"),
        resolved_at=float(resolved_at),
        scheduled_from=target_at,
        scheduled_until=target_at + validity,
        clock_domain_id=str(clock_domain_id),
        input_action_schema_hash=str(input_action_schema_hash),
        output_target_schema_hash=str(output_target_schema_hash),
        resolver_contract_id=resolution.resolver_contract_id,
        resolver_contract_hash=resolution.resolver_contract_hash,
        transform_contract_hash=resolution.transform_contract_hash,
        planner_contract_hash=resolution.resolver_contract_hash,
        payload=target_payload,
    )


def build_mujoco_component_execution_target(
    resolution: MujocoPoseToQposResolution,
    context: AuthoritativeTargetDispatchContext,
    *,
    resolved_at: float,
    output_target_schema_hash: str,
    component_id_by_output_effector: Mapping[str, str] | None = None,
    support_only_output_effectors: tuple[str, ...] = (),
) -> ComponentExecutionTarget:
    """Build the component-keyed MuJoCo target without a universal safe claim."""

    if not isinstance(resolution, MujocoPoseToQposResolution):
        raise TypeError("resolution must be MujocoPoseToQposResolution.")
    if not isinstance(context, AuthoritativeTargetDispatchContext):
        raise TypeError(
            "context must be an AuthoritativeTargetDispatchContext."
        )
    output = resolution.output_qpos_action
    raw_component_values: dict[str, object] = {
        f"{side}_arm": np.asarray(values, dtype=float).tolist()
        for side, values in output.arm_qpos_by_side.items()
    }
    raw_component_values.update(
        {
            f"{side}_hand": np.asarray(values, dtype=float).tolist()
            for side, values in output.hand_qpos_by_side.items()
        }
    )
    output_effectors = tuple(output.active_effectors)
    if set(raw_component_values) != set(output_effectors):
        raise ValueError(
            "Resolved qpos payload and output active-effectors disagree."
        )
    support_only = tuple(str(value).strip() for value in support_only_output_effectors)
    if (
        any(not value for value in support_only)
        or len(support_only) != len(set(support_only))
    ):
        raise ValueError("support_only_output_effectors must be unique identifiers.")
    if component_id_by_output_effector is None:
        output_to_component = {
            effector: effector for effector in output_effectors
        }
    else:
        output_to_component = {
            str(effector).strip(): str(component_id).strip()
            for effector, component_id in dict(
                component_id_by_output_effector
            ).items()
        }
        if any(
            not effector or not component_id
            for effector, component_id in output_to_component.items()
        ):
            raise ValueError(
                "component_id_by_output_effector must contain identifiers."
            )
    if set(output_to_component) & set(support_only):
        raise ValueError(
            "An output effector cannot be authoritative and support-only."
        )
    if set(output_effectors) != set(output_to_component) | set(support_only):
        raise ValueError(
            "Every resolved output effector must be explicitly authoritative "
            "or support-only."
        )
    if (
        len(output_to_component.values())
        != len(set(output_to_component.values()))
    ):
        raise ValueError("Output effectors must map to unique component ids.")
    authoritative_component_to_output_effector = {
        component_id: effector
        for effector, component_id in output_to_component.items()
    }
    if set(authoritative_component_to_output_effector) != set(
        context.component_coverage
    ):
        raise ValueError(
            "Resolved qpos payload does not exactly cover the target context."
        )
    resolved = float(resolved_at)
    processing_fact = TargetProcessingFactReference(
        stage="mujoco.coupled_pose_to_qpos",
        fact_id=resolution.resolution_id,
        fact_hash=resolution.semantic_content_hash,
        status=resolution.solver_status,
        processing_started_at=resolved,
        processing_completed_at=resolved,
        clock_domain_id=context.target_execution_request.clock_domain_id,
    )
    target_at = float(context.target_effective_at)
    return build_component_execution_target(
        context.target_execution_request,
        context.submission_receipt,
        component_dependencies=context.component_coverage,
        output_target_schema_hash=str(output_target_schema_hash),
        target_kind="mujoco.component_qpos_reference",
        resolved_at=resolved,
        temporal_semantics="point_target",
        scheduled_for=target_at,
        scheduled_until=None,
        payload={
            "command_id": context.command_id,
            "resolution_id": resolution.resolution_id,
            "resolution_hash": resolution.semantic_content_hash,
            "resolver_contract_id": resolution.resolver_contract_id,
            "resolver_contract_hash": resolution.resolver_contract_hash,
            "transform_receipt_id": resolution.transform_receipt_id,
            "transform_contract_hash": resolution.transform_contract_hash,
            "input_processed_action_hash": (
                resolution.input_safe_action_hash
            ),
            "output_qpos_action_hash": (
                resolution.output_qpos_action_hash
            ),
            # These are downstream resolver outputs, not the native source or
            # composition component values.  Keep the representation edge
            # explicit so a target-private six-slot hand vector cannot be
            # mistaken for a physical gripper-qpos component.
            "output_values_by_effector": raw_component_values,
            "authoritative_component_to_output_effector": (
                authoritative_component_to_output_effector
            ),
            "support_only_output_effectors": support_only,
            "group_segment_generation_by_group": dict(
                context.group_segment_generation_by_group
            ),
        },
        processing_fact_references=(processing_fact,),
    )


__all__ = [
    "POSE_TO_QPOS_RESOLUTION_HASH_DOMAIN_V1",
    "POSE_TO_QPOS_RESOLVER_CONTRACT_HASH_V1",
    "POSE_TO_QPOS_RESOLVER_CONTRACT_ID_V1",
    "MujocoCoupledPoseToQposResolver",
    "MujocoPoseToQposResolution",
    "build_mujoco_component_execution_target",
    "build_mujoco_pose_qpos_execution_target",
]
