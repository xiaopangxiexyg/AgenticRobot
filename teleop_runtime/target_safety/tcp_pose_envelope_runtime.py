"""Real-runtime adapter for the control-mode-neutral TCP-pose envelope.

The target filter sits on the existing coupled pre-IK target-filter seam.  It
converts the retargeter's solver-frame target to the configured ``tool_tcp``,
projects that pose once, then converts the result back to the solver frame.
It never calls IK or FK and it does not participate in 120 Hz qpos dispatch.
The same adapter also projects the final composed host-side impedance TCP.

Anchor capture consumes the ``tool_tcp`` Home pose already committed by the
runtime calibration contract.  It never reconstructs that anchor from qpos.
"""

from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.target_safety.tcp_pose_envelope import (
    TCP_CONTROLLED_FRAME_ID,
    TCPPoseEnvelopeAnchorReceipt,
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeProjectionFeedback,
    TCPPoseEnvelopeProjectionResult,
    TCPPoseEnvelopeSideConfig,
    project_tcp_pose_envelope,
    project_tcp_pose_envelope_side,
)


TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.real_tcp_pose_target_frame_binding.development"
)
TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.real_tcp_pose_tool_frame_binding.development"
)
TCP_POSE_ENVELOPE_FILE_SCHEMA_V1 = "teleop_runtime.tcp_pose_envelope.v1"


class TCPPoseEnvelopeRuntimeError(RuntimeError):
    """Fail-closed runtime error raised by an Active envelope filter."""


def _required_text(value: object, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} must be non-empty")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _pose(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite rigid 4x4 pose") from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite rigid 4x4 pose")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} must be homogeneous")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9, rtol=0.0):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix.copy()


def _normalized_sides(sides: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(_required_text(side, name="side") for side in sides)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("sides must be non-empty and unique")
    return normalized


def target_frame_binding_hash_from_calibration(
    calibration: Any,
    *,
    sides: Sequence[str],
    reference_frame_by_side: Mapping[str, str],
) -> str:
    """Hash the exact transforms used to express coupled targets in world."""

    normalized = _normalized_sides(sides)
    world_from_base_by_side: dict[str, np.ndarray] = {}
    for side in normalized:
        try:
            arm = calibration.get_arm_for_side(side)
        except Exception as exc:  # noqa: BLE001 - convert to one binding failure.
            raise ValueError(f"calibration is missing side {side!r}") from exc
        world_from_base_by_side[side] = _pose(
            getattr(arm, "T_D_B", None),
            name=f"calibration.{side}.T_D_B",
        )
    return target_frame_binding_hash_from_transforms(
        target_world_from_base_by_side=world_from_base_by_side,
        sides=normalized,
        reference_frame_by_side=reference_frame_by_side,
    )


def target_frame_binding_hash_from_transforms(
    *,
    target_world_from_base_by_side: Mapping[str, Any],
    sides: Sequence[str],
    reference_frame_by_side: Mapping[str, str],
) -> str:
    """Hash an explicit, lower-owned base-to-target-world frame binding."""

    normalized = _normalized_sides(sides)
    payload: dict[str, object] = {
        "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
        "reference_frame_by_side": {},
        "target_world_from_base_by_side": {},
    }
    references = payload["reference_frame_by_side"]
    transforms = payload["target_world_from_base_by_side"]
    assert isinstance(references, dict)
    assert isinstance(transforms, dict)
    for side in normalized:
        frame = _required_text(
            reference_frame_by_side.get(side),
            name=f"reference_frame_by_side.{side}",
        )
        try:
            raw_transform = target_world_from_base_by_side[side]
        except KeyError as exc:
            raise ValueError(
                f"target frame binding is missing side {side!r}"
            ) from exc
        transform = _pose(
            raw_transform,
            name=f"target_world_from_base_by_side.{side}",
        )
        references[side] = frame
        transforms[side] = transform.tolist()
    return canonical_content_hash(
        payload,
        domain=TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
    )


def tool_frame_binding_hashes_from_retargeter(
    retargeter: Any,
    *,
    sides: Sequence[str],
) -> dict[str, str]:
    """Hash the exact fixed solver→tool_tcp transforms used by the retargeter."""

    normalized = _normalized_sides(sides)
    convert = getattr(
        retargeter,
        "tool_tcp_world_target_from_solver_world_target",
        None,
    )
    if not callable(convert):
        raise TypeError("retargeter lacks solver-to-tool_tcp conversion")
    result: dict[str, str] = {}
    identity = np.eye(4, dtype=float)
    for side in normalized:
        transform = _pose(
            convert(side=side, solver_world_target=identity),
            name=f"retargeter.{side}.solver_to_tool_tcp",
        )
        result[side] = canonical_content_hash(
            {
                "side": side,
                "controlled_frame_id": TCP_CONTROLLED_FRAME_ID,
                "solver_to_tool_tcp": transform.tolist(),
            },
            domain=TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT,
        )
    return result


def _strict_keys(
    payload: Mapping[str, object],
    *,
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")


def _bounds_from_payload(
    value: object,
    *,
    name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be null or an object")
    _strict_keys(
        value,
        allowed={"min_xyz_m", "max_xyz_m"},
        name=name,
    )
    if set(value) != {"min_xyz_m", "max_xyz_m"}:
        raise ValueError(f"{name} requires min_xyz_m and max_xyz_m")
    try:
        lower = tuple(float(item) for item in value["min_xyz_m"])
        upper = tuple(float(item) for item in value["max_xyz_m"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} bounds must contain three finite numbers") from exc
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError(f"{name} bounds must contain three values")
    return lower, upper


@dataclass(frozen=True)
class TCPPoseEnvelopeFileSpec:
    """Frame-independent contents of one strict envelope JSON profile."""

    profile_id: str
    side_config_by_side: Mapping[str, TCPPoseEnvelopeSideConfig]

    def __post_init__(self) -> None:
        profile_id = _required_text(self.profile_id, name="profile_id")
        configs = dict(self.side_config_by_side)
        if not configs:
            raise ValueError("side_config_by_side must not be empty")
        if not all(
            isinstance(config, TCPPoseEnvelopeSideConfig)
            for config in configs.values()
        ):
            raise TypeError(
                "side_config_by_side values must be TCPPoseEnvelopeSideConfig"
            )
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(
            self,
            "side_config_by_side",
            MappingProxyType(dict(sorted(configs.items()))),
        )


@dataclass(frozen=True)
class TCPPoseEnvelopeTargetFrameBinding:
    """Immutable lower-owned geometry used by a TCP envelope runtime."""

    reference_frame_by_side: Mapping[str, str]
    target_world_from_base_by_side: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        references: dict[str, str] = {}
        for raw_side, frame in dict(self.reference_frame_by_side).items():
            side = str(raw_side).strip().lower()
            if side not in {"left", "right"} or str(raw_side) != side:
                raise ValueError(
                    "target frame binding side keys must be exact 'left' or 'right'"
                )
            references[side] = _required_text(
                frame,
                name=f"reference_frame_by_side.{side}",
            )
        transforms: dict[str, np.ndarray] = {}
        for raw_side, transform in dict(
            self.target_world_from_base_by_side
        ).items():
            side = str(raw_side).strip().lower()
            if side not in {"left", "right"} or str(raw_side) != side:
                raise ValueError(
                    "target frame binding side keys must be exact 'left' or 'right'"
                )
            transforms[side] = _pose(
                transform,
                name=f"target_world_from_base_by_side.{side}",
            )
        if not references or set(references) != set(transforms):
            raise ValueError(
                "reference_frame_by_side and target_world_from_base_by_side "
                "must contain the same non-empty side set"
            )
        frozen_transforms: dict[str, np.ndarray] = {}
        for side, transform in transforms.items():
            transform.setflags(write=False)
            frozen_transforms[side] = transform
        object.__setattr__(
            self,
            "reference_frame_by_side",
            MappingProxyType(dict(sorted(references.items()))),
        )
        object.__setattr__(
            self,
            "target_world_from_base_by_side",
            MappingProxyType(dict(sorted(frozen_transforms.items()))),
        )

    def semantic_hash(self, *, sides: Sequence[str]) -> str:
        return target_frame_binding_hash_from_transforms(
            target_world_from_base_by_side=self.target_world_from_base_by_side,
            sides=sides,
            reference_frame_by_side=self.reference_frame_by_side,
        )


def load_tcp_pose_envelope_file_spec(
    path: str | Path,
    *,
    active_sides: Sequence[str],
    calibration_workspace_resolver: Any | None = None,
) -> TCPPoseEnvelopeFileSpec:
    """Parse the shared strict JSON without choosing a runtime frame binding.

    ``calibration_workspace`` is intentionally unavailable unless the caller
    provides an explicit resolver.  Deploy callers therefore cannot silently
    reinterpret a teleop calibration workspace in another reference frame.
    """

    profile_path = Path(path).expanduser()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("TCP pose envelope profile root must be an object")
    _strict_keys(
        payload,
        allowed={"schema_version", "profile_id", "sides"},
        name="TCP pose envelope profile",
    )
    if payload.get("schema_version") != TCP_POSE_ENVELOPE_FILE_SCHEMA_V1:
        raise ValueError(
            "TCP pose envelope profile schema_version must be "
            f"{TCP_POSE_ENVELOPE_FILE_SCHEMA_V1!r}"
        )
    sides_payload = payload.get("sides")
    if not isinstance(sides_payload, Mapping) or not sides_payload:
        raise ValueError("TCP pose envelope profile sides must be a non-empty object")
    active = set(_normalized_sides(active_sides))
    requested = {str(side) for side in sides_payload}
    inactive = sorted(requested - active)
    if inactive:
        raise ValueError(
            f"TCP pose envelope profile configures inactive sides: {inactive}"
        )

    configs: dict[str, TCPPoseEnvelopeSideConfig] = {}
    for side in sorted(requested):
        raw = sides_payload[side]
        if not isinstance(raw, Mapping):
            raise ValueError(f"TCP pose envelope side {side!r} must be an object")
        _strict_keys(
            raw,
            allowed={
                "enabled",
                "absolute_bounds",
                "relative_bounds",
                "relative_axes",
                "max_total_rotation_deg",
            },
            name=f"TCP pose envelope side {side!r}",
        )
        enabled = raw.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError(f"TCP pose envelope side {side!r} enabled must be bool")
        absolute_value = raw.get("absolute_bounds")
        if absolute_value == "calibration_workspace":
            if not callable(calibration_workspace_resolver):
                raise ValueError(
                    "absolute_bounds='calibration_workspace' requires an explicit "
                    "calibration workspace resolver; deploy profiles must use "
                    "explicit canonical-frame bounds"
                )
            absolute = calibration_workspace_resolver(side)
        else:
            absolute = _bounds_from_payload(
                absolute_value,
                name=f"sides.{side}.absolute_bounds",
            )
        relative = _bounds_from_payload(
            raw.get("relative_bounds"),
            name=f"sides.{side}.relative_bounds",
        )
        max_rotation_deg = raw.get("max_total_rotation_deg")
        max_rotation_rad = (
            None
            if max_rotation_deg is None
            else math.radians(float(max_rotation_deg))
        )
        configs[side] = TCPPoseEnvelopeSideConfig(
            enabled=enabled,
            absolute_min_xyz_m=None if absolute is None else absolute[0],
            absolute_max_xyz_m=None if absolute is None else absolute[1],
            relative_min_xyz_m=None if relative is None else relative[0],
            relative_max_xyz_m=None if relative is None else relative[1],
            relative_axes=str(raw.get("relative_axes", "canonical_axes")),
            max_rotation_rad=max_rotation_rad,
        )
    return TCPPoseEnvelopeFileSpec(
        profile_id=_required_text(payload.get("profile_id"), name="profile_id"),
        side_config_by_side=configs,
    )


def load_tcp_pose_envelope_profile(
    path: str | Path,
    *,
    calibration: Any,
    retargeter: Any,
    active_sides: Sequence[str],
) -> TCPPoseEnvelopeProfile:
    """Load one strict, default-off runtime profile bound to live calibration."""

    def calibration_workspace(side: str):
        arm = calibration.get_arm_for_side(side)
        workspace_min = getattr(arm, "workspace_min", None)
        workspace_max = getattr(arm, "workspace_max", None)
        if workspace_min is None or workspace_max is None:
            raise ValueError(
                f"calibration workspace is incomplete for side {side!r}"
            )
        return (
            tuple(np.asarray(workspace_min, dtype=float).reshape(3)),
            tuple(np.asarray(workspace_max, dtype=float).reshape(3)),
        )

    spec = load_tcp_pose_envelope_file_spec(
        path,
        active_sides=active_sides,
        calibration_workspace_resolver=calibration_workspace,
    )
    configs = dict(spec.side_config_by_side)

    references = {
        side: _required_text(
            getattr(getattr(calibration, "desk_frame", None), "name", None),
            name="calibration.desk_frame.name",
        )
        for side in configs
    }
    enabled_sides = tuple(
        side for side, config in configs.items() if config.enabled
    )
    return TCPPoseEnvelopeProfile(
        profile_id=spec.profile_id,
        target_frame_binding_hash=target_frame_binding_hash_from_calibration(
            calibration,
            sides=enabled_sides,
            reference_frame_by_side=references,
        ),
        tool_frame_binding_hash_by_side=tool_frame_binding_hashes_from_retargeter(
            retargeter,
            sides=tuple(configs),
        ),
        reference_frame_by_side=references,
        side_config_by_side=configs,
    )


def load_tcp_pose_envelope_profile_from_target_frame_binding(
    path: str | Path,
    *,
    target_frame_binding: TCPPoseEnvelopeTargetFrameBinding,
    retargeter: Any,
    active_sides: Sequence[str],
) -> TCPPoseEnvelopeProfile:
    """Load a deploy envelope from explicit Station geometry, not teleop state."""

    if not isinstance(target_frame_binding, TCPPoseEnvelopeTargetFrameBinding):
        raise TypeError("target_frame_binding must be typed")
    spec = load_tcp_pose_envelope_file_spec(path, active_sides=active_sides)
    configs = dict(spec.side_config_by_side)
    references = {
        side: target_frame_binding.reference_frame_by_side[side]
        for side in configs
    }
    enabled_sides = tuple(
        side for side, config in configs.items() if config.enabled
    )
    return TCPPoseEnvelopeProfile(
        profile_id=spec.profile_id,
        target_frame_binding_hash=target_frame_binding.semantic_hash(
            sides=enabled_sides
        ),
        tool_frame_binding_hash_by_side=tool_frame_binding_hashes_from_retargeter(
            retargeter,
            sides=tuple(configs),
        ),
        reference_frame_by_side=references,
        side_config_by_side=configs,
    )


def capture_tcp_pose_envelope_anchor(
    *,
    anchor_id: str,
    profile: TCPPoseEnvelopeProfile,
    calibration: Any,
    execution_epoch: int,
    clock_domain_id: str,
    captured_at: float,
    captured_sides: Sequence[str] | None = None,
    previous_anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None = None,
) -> TCPPoseEnvelopeAnchorReceipt:
    """Bind relative limits to committed runtime-Home ``tool_tcp`` poses.

    ``captured_sides`` supports a single-side runtime Home.  Sides not captured
    in this Home retain the previous receipt's pose and timestamp.  A first
    anchor must capture every side required by the profile.
    """

    capture_time = float(captured_at)
    if not math.isfinite(capture_time):
        raise ValueError("captured_at must be finite")
    required_sides = profile.anchor_required_sides
    if not required_sides:
        raise ValueError("profile does not require an anchor")
    requested_capture_sides = (
        set(profile.enabled_sides)
        if captured_sides is None
        else set(_normalized_sides(captured_sides))
    )
    unsupported_capture_sides = sorted(
        requested_capture_sides - {"left", "right"}
    )
    if unsupported_capture_sides:
        raise TCPPoseEnvelopeRuntimeError(
            "runtime Home contains unsupported sides: "
            f"{unsupported_capture_sides}"
        )
    # A single-side Home outside this profile must not invalidate the still
    # valid anchor of a limited side.  Reissue the receipt in the new epoch and
    # preserve every required side that this Home did not capture.
    updated_sides = requested_capture_sides.intersection(required_sides)

    actual_target_hash = target_frame_binding_hash_from_calibration(
        calibration,
        sides=profile.enabled_sides,
        reference_frame_by_side=profile.reference_frame_by_side,
    )
    if actual_target_hash != profile.target_frame_binding_hash:
        raise TCPPoseEnvelopeRuntimeError("target-frame binding changed before anchor capture")

    world_pose_by_side: dict[str, np.ndarray] = {}
    captured_at_by_side: dict[str, float] = {}
    for side in required_sides:
        if side not in updated_sides:
            if (
                previous_anchor_receipt is None
                or side not in previous_anchor_receipt.tcp_pose_by_side
            ):
                raise TCPPoseEnvelopeRuntimeError(
                    f"runtime Home did not capture required side {side!r} and no "
                    "previous anchor is available"
                )
            world_pose_by_side[side] = np.asarray(
                previous_anchor_receipt.tcp_pose_by_side[side],
                dtype=float,
            ).copy()
            captured_at_by_side[side] = float(
                previous_anchor_receipt.captured_at_by_side[side]
            )
            continue
        try:
            arm = calibration.get_arm_for_side(side)
        except Exception as exc:  # noqa: BLE001 - one fail-closed binding error.
            raise TCPPoseEnvelopeRuntimeError(
                f"calibration is missing side {side!r}"
            ) from exc
        home_pose = getattr(arm, "home_pose_for_tcp_mode", None)
        if not callable(home_pose):
            raise TCPPoseEnvelopeRuntimeError(
                f"calibration side {side!r} cannot provide a tool_tcp Home pose"
            )
        committed_home = home_pose(TCP_CONTROLLED_FRAME_ID)
        if committed_home is None:
            raise TCPPoseEnvelopeRuntimeError(
                f"calibration side {side!r} has no committed tool_tcp Home pose"
            )
        try:
            world_pose_by_side[side] = _pose(
                committed_home,
                name=f"calibration.{side}.tool_tcp_home_pose",
            )
        except ValueError as exc:
            raise TCPPoseEnvelopeRuntimeError(str(exc)) from exc
        captured_at_by_side[side] = capture_time

    receipt = TCPPoseEnvelopeAnchorReceipt.capture(
        anchor_id=anchor_id,
        profile=profile,
        execution_epoch=execution_epoch,
        clock_domain_id=clock_domain_id,
        tcp_pose_by_side=world_pose_by_side,
        captured_at_by_side=captured_at_by_side,
    )
    return receipt


@dataclass(frozen=True)
class TCPPoseEnvelopeFilterSnapshot:
    frame_count: int
    apply_to_target: bool
    execution_epoch: int
    anchor_hash: str | None
    last_error: str | None
    projection: dict[str, object] | None


class TCPPoseEnvelopeTargetFilter:
    """Composable pre-IK TCP envelope observer or Active target projector."""

    def __init__(
        self,
        *,
        retargeter: Any,
        profile: TCPPoseEnvelopeProfile,
        calibration: Any | None = None,
        target_frame_binding: TCPPoseEnvelopeTargetFrameBinding | None = None,
        execution_epoch: int,
        anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
        apply_to_target: bool = False,
    ) -> None:
        if not isinstance(profile, TCPPoseEnvelopeProfile):
            raise TypeError("profile must be TCPPoseEnvelopeProfile")
        if (calibration is None) == (target_frame_binding is None):
            raise ValueError(
                "provide exactly one of calibration or target_frame_binding"
            )
        if target_frame_binding is not None and not isinstance(
            target_frame_binding,
            TCPPoseEnvelopeTargetFrameBinding,
        ):
            raise TypeError("target_frame_binding must be typed")
        self.retargeter = retargeter
        self.profile = profile
        self.calibration = calibration
        self.target_frame_binding = target_frame_binding
        self.apply_to_target = bool(apply_to_target)
        self._tool_frame_binding_hash_by_side = (
            tool_frame_binding_hashes_from_retargeter(
                retargeter,
                sides=profile.enabled_sides,
            )
        )
        for side in profile.enabled_sides:
            if (
                self._tool_frame_binding_hash_by_side.get(side)
                != profile.tool_frame_binding_hash_by_side[side]
            ):
                raise TCPPoseEnvelopeRuntimeError(
                    f"tool-frame binding changed before filter installation for {side!r}"
                )
        self._lock = threading.Lock()
        self._execution_epoch = int(execution_epoch)
        self._anchor_receipt = anchor_receipt
        self._frame_count = 0
        self._last_target_role: str | None = None
        self._last_error: str | None = None
        self._last_projection: TCPPoseEnvelopeProjectionResult | None = None

    @property
    def execution_epoch(self) -> int:
        with self._lock:
            return int(self._execution_epoch)

    @property
    def anchor_receipt(self) -> TCPPoseEnvelopeAnchorReceipt | None:
        with self._lock:
            return self._anchor_receipt

    def _current_target_frame_binding_hash(self) -> str:
        if self.target_frame_binding is not None:
            return self.target_frame_binding.semantic_hash(
                sides=self.profile.enabled_sides
            )
        return target_frame_binding_hash_from_calibration(
            self.calibration,
            sides=self.profile.enabled_sides,
            reference_frame_by_side=self.profile.reference_frame_by_side,
        )

    def _target_world_from_base(self, side: str) -> np.ndarray:
        if self.target_frame_binding is not None:
            return _pose(
                self.target_frame_binding.target_world_from_base_by_side[side],
                name=f"target_frame_binding.target_world_from_base_by_side.{side}",
            )
        arm = self.calibration.get_arm_for_side(side)
        return _pose(
            getattr(arm, "T_D_B", None),
            name=f"calibration.{side}.T_D_B",
        )

    def rebind(
        self,
        *,
        execution_epoch: int,
        anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
    ) -> None:
        """Atomically replace the Home/epoch binding while motion is held."""

        epoch = int(execution_epoch)
        if isinstance(execution_epoch, bool) or epoch < 0:
            raise ValueError("execution_epoch must be a non-negative integer")
        required_sides = set(self.profile.anchor_required_sides)
        if required_sides:
            if anchor_receipt is None:
                raise TCPPoseEnvelopeRuntimeError(
                    f"TCP envelope anchor is required for sides {sorted(required_sides)}"
                )
            if anchor_receipt.execution_epoch != epoch:
                raise TCPPoseEnvelopeRuntimeError(
                    "TCP envelope anchor execution epoch does not match rebind epoch"
                )
            if not required_sides.issubset(anchor_receipt.tcp_pose_by_side):
                raise TCPPoseEnvelopeRuntimeError(
                    "TCP envelope anchor is missing one or more required sides"
                )
        with self._lock:
            self._execution_epoch = epoch
            self._anchor_receipt = anchor_receipt
            self._last_error = None
            self._last_target_role = None
            self._last_projection = None

    def rebind_from_calibration_home(
        self,
        *,
        execution_epoch: int,
        captured_at: float,
        captured_sides: Sequence[str] | None = None,
        anchor_id: str = "runtime_home",
        clock_domain_id: str = "python_monotonic",
    ) -> TCPPoseEnvelopeAnchorReceipt | None:
        """Rebind after runtime Home without reading qpos or invoking FK."""

        if self.calibration is None:
            raise TCPPoseEnvelopeRuntimeError(
                "this envelope is not bound to a teleop calibration"
            )
        previous = self.anchor_receipt
        anchor = (
            None
            if not self.profile.anchor_required_sides
            else capture_tcp_pose_envelope_anchor(
                anchor_id=anchor_id,
                profile=self.profile,
                calibration=self.calibration,
                execution_epoch=execution_epoch,
                clock_domain_id=clock_domain_id,
                captured_at=captured_at,
                captured_sides=captured_sides,
                previous_anchor_receipt=previous,
            )
        )
        self.rebind(
            execution_epoch=execution_epoch,
            anchor_receipt=anchor,
        )
        return anchor

    def _tool_targets(
        self,
        targets: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        convert = getattr(
            self.retargeter,
            "tool_tcp_world_target_from_solver_world_target",
            None,
        )
        if not callable(convert):
            raise TypeError("retargeter lacks solver-to-tool_tcp conversion")
        return {
            side: _pose(
                convert(side=side, solver_world_target=targets[side]),
                name=f"tool_tcp_target.{side}",
            )
            for side in targets
        }

    def _solver_targets(
        self,
        targets: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        convert = getattr(
            self.retargeter,
            "solver_world_target_from_tool_tcp_world_target",
            None,
        )
        if not callable(convert):
            raise TypeError("retargeter lacks tool_tcp-to-solver conversion")
        return {
            side: _pose(
                convert(side=side, tool_tcp_world_target=targets[side]),
                name=f"solver_target.{side}",
            )
            for side in targets
        }

    def __call__(
        self,
        targets: Mapping[str, np.ndarray],
        *_unused: Any,
    ) -> Mapping[str, np.ndarray] | None:
        effective_targets = targets
        with self._lock:
            execution_epoch = int(self._execution_epoch)
            anchor_receipt = self._anchor_receipt
        try:
            target_hash = self._current_target_frame_binding_hash()
            tool_targets = self._tool_targets(effective_targets)
            projection = project_tcp_pose_envelope(
                tool_targets,
                profile=self.profile,
                target_frame_binding_hash=target_hash,
                tool_frame_binding_hash_by_side=(
                    self._tool_frame_binding_hash_by_side
                ),
                controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
                reference_frame_by_side=self.profile.reference_frame_by_side,
                execution_epoch=execution_epoch,
                anchor_receipt=anchor_receipt,
            )
            # Preserve every unmodified solver target bit-for-bit.  This is
            # important for a single-side envelope: an unrestricted arm must
            # not pick up even round-trip floating-point noise from the other
            # side's safety projection.
            applied_targets = {
                side: np.asarray(target, dtype=float).copy()
                for side, target in effective_targets.items()
            }
            modified_tool_targets = {
                side: projection.effective_tcp_pose_by_side[side]
                for side in projection.modified_sides
            }
            if modified_tool_targets:
                applied_targets.update(
                    self._solver_targets(modified_tool_targets)
                )
        except Exception as exc:  # noqa: BLE001 - Shadow records; Active fails closed.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._frame_count += 1
                self._last_target_role = "final_pre_ik"
                self._last_error = error
                self._last_projection = None
            if self.apply_to_target:
                if isinstance(exc, TCPPoseEnvelopeRuntimeError):
                    raise
                raise TCPPoseEnvelopeRuntimeError(error) from exc
            return effective_targets

        with self._lock:
            self._frame_count += 1
            self._last_target_role = "final_pre_ik"
            self._last_error = None
            self._last_projection = projection
        if not self.apply_to_target:
            return effective_targets
        return applied_targets

    def project_arm_base_tool_tcp(
        self,
        side: str,
        arm_base_tool_tcp: np.ndarray,
    ) -> np.ndarray:
        """Project one native-impedance composed TCP without IK or FK.

        Native RM65 impedance workers operate independently in each arm base.
        The envelope remains expressed in its bound target-world frame, so this
        adapter performs only the fixed base/target-world transform around the
        same pure projector used by position control.
        """

        normalized_side = _required_text(side, name="side")
        if normalized_side not in self.profile.side_config_by_side:
            raise TCPPoseEnvelopeRuntimeError(
                f"side {normalized_side!r} is absent from the TCP envelope profile"
            )
        raw_base_pose = _pose(
            arm_base_tool_tcp,
            name=f"arm_base_tool_tcp.{normalized_side}",
        )
        with self._lock:
            execution_epoch = int(self._execution_epoch)
            anchor_receipt = self._anchor_receipt
        try:
            target_world_from_base = self._target_world_from_base(normalized_side)
            target_hash = self._current_target_frame_binding_hash()
            raw_target_world_pose = target_world_from_base @ raw_base_pose
            projection = project_tcp_pose_envelope_side(
                normalized_side,
                raw_target_world_pose,
                profile=self.profile,
                target_frame_binding_hash=target_hash,
                tool_frame_binding_hash_by_side=(
                    self._tool_frame_binding_hash_by_side
                ),
                controlled_frame_id=TCP_CONTROLLED_FRAME_ID,
                reference_frame_by_side=self.profile.reference_frame_by_side,
                execution_epoch=execution_epoch,
                anchor_receipt=anchor_receipt,
            )
            effective_target_world_pose = projection.effective_tcp_pose_by_side[
                normalized_side
            ]
            effective_base_pose = _pose(
                np.linalg.inv(target_world_from_base) @ effective_target_world_pose,
                name=f"effective_arm_base_tool_tcp.{normalized_side}",
            )
        except Exception as exc:  # noqa: BLE001 - Shadow records; Active fails closed.
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._frame_count += 1
                self._last_target_role = "final_composed_impedance"
                self._last_error = error
                self._last_projection = None
            if self.apply_to_target:
                if isinstance(exc, TCPPoseEnvelopeRuntimeError):
                    raise
                raise TCPPoseEnvelopeRuntimeError(error) from exc
            return raw_base_pose

        with self._lock:
            self._frame_count += 1
            self._last_target_role = "final_composed_impedance"
            self._last_error = None
            self._last_projection = projection
        if not self.apply_to_target:
            return raw_base_pose
        return effective_base_pose

    def snapshot(self) -> TCPPoseEnvelopeFilterSnapshot:
        with self._lock:
            projection = self._last_projection
            return TCPPoseEnvelopeFilterSnapshot(
                frame_count=int(self._frame_count),
                apply_to_target=bool(self.apply_to_target),
                execution_epoch=int(self._execution_epoch),
                anchor_hash=(
                    None
                    if self._anchor_receipt is None
                    else self._anchor_receipt.semantic_content_hash
                ),
                last_error=self._last_error,
                projection=(
                    None
                    if projection is None
                    else copy.deepcopy(projection.to_payload())
                ),
            )

    def feedback_snapshot(self) -> TCPPoseEnvelopeProjectionFeedback:
        """Return the latest compact downstream receipt without control authority."""

        with self._lock:
            anchor_hash = (
                None
                if self._anchor_receipt is None
                else self._anchor_receipt.semantic_content_hash
            )
            bound = bool(
                not self.profile.anchor_required_sides
                or self._anchor_receipt is not None
            )
            return TCPPoseEnvelopeProjectionFeedback.from_evaluation(
                apply_to_target=self.apply_to_target,
                bound=bound,
                profile_id=self.profile.profile_id,
                profile_hash=self.profile.semantic_content_hash,
                execution_epoch=self._execution_epoch,
                anchor_hash=anchor_hash,
                frame_count=self._frame_count,
                target_role=self._last_target_role,
                projection=self._last_projection,
                error=self._last_error,
            )

    def command_metadata(self) -> dict[str, object]:
        """Return compact JSON-safe lineage for the just-evaluated IK target.

        Full raw/effective poses remain available through :meth:`snapshot` for
        deliberate diagnostics.  The per-command path carries only the fields
        needed to prove whether the final pre-IK boundary was evaluated and
        whether it changed either side, avoiding large 60 Hz metadata payloads.
        """

        payload = self.feedback_snapshot().to_payload()
        # Preserve the existing per-command schema identity for current trace
        # consumers while exposing the shared receipt schema explicitly.
        payload["feedback_schema_version"] = payload["schema_version"]
        payload["schema_version"] = (
            "teleop_runtime.tcp_pose_envelope_command.v1"
        )
        return payload


def install_tcp_pose_envelope_filter(
    *,
    retargeter: Any,
    profile: TCPPoseEnvelopeProfile,
    calibration: Any | None = None,
    target_frame_binding: TCPPoseEnvelopeTargetFrameBinding | None = None,
    execution_epoch: int,
    anchor_receipt: TCPPoseEnvelopeAnchorReceipt | None,
    apply_to_target: bool = False,
) -> TCPPoseEnvelopeTargetFilter:
    """Install one canonical final pre-IK filter without replacing modifiers."""

    existing = getattr(retargeter, "pre_ik_target_safety_filter", None)
    if existing is not None:
        raise TCPPoseEnvelopeRuntimeError(
            "retargeter already has a final pre-IK target safety owner"
        )

    observer = TCPPoseEnvelopeTargetFilter(
        retargeter=retargeter,
        profile=profile,
        calibration=calibration,
        target_frame_binding=target_frame_binding,
        execution_epoch=execution_epoch,
        anchor_receipt=anchor_receipt,
        apply_to_target=apply_to_target,
    )
    retargeter.pre_ik_target_safety_filter = observer
    return observer


__all__ = [
    "TCP_POSE_ENVELOPE_FILE_SCHEMA_V1",
    "TARGET_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "TOOL_FRAME_BINDING_HASH_DOMAIN_DEVELOPMENT",
    "TCPPoseEnvelopeFileSpec",
    "TCPPoseEnvelopeFilterSnapshot",
    "TCPPoseEnvelopeRuntimeError",
    "TCPPoseEnvelopeTargetFrameBinding",
    "TCPPoseEnvelopeTargetFilter",
    "capture_tcp_pose_envelope_anchor",
    "install_tcp_pose_envelope_filter",
    "load_tcp_pose_envelope_file_spec",
    "load_tcp_pose_envelope_profile",
    "load_tcp_pose_envelope_profile_from_target_frame_binding",
    "target_frame_binding_hash_from_calibration",
    "target_frame_binding_hash_from_transforms",
    "tool_frame_binding_hashes_from_retargeter",
]
