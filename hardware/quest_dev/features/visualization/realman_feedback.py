"""Direct, read-only RealMan RM65 feedback for Quest visualization.

The Quest host is an observer of the physical arm.  It attaches to the
existing RM65 backend in ``observation_only`` mode, calls only the backend's
state refresh path, derives the TCP pose from the measured joint state, and
publishes the result to the browser.  No Redis publisher, command loop,
trajectory, enable, stop, or reset path is involved here.

When a reviewed Quest/robot frame binding is configured, the source is joined
to the canonical passive pose publication path from ``teleop_runtime``.  A
binding mismatch, stale read, missing attestation, or failed SDK read removes
the Quest-space matrix instead of guessing a transform.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np

from .frame_binding import AcceptedFrameBinding
from .pose_projection import project_pose_publication
from .protocol import robot_tcp_message
from .transforms import matrix_to_column_major, matrix_to_pose_payload

REALMAN_SOURCE_NAME = "realman_rm65_direct"
DEFAULT_CONTROL_DOMAIN_ID = "quest-dev:realman-rm65-observation"
DEFAULT_TCP_FRAME = "tool_tcp"


def _required_text(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _finite_positive(value: object, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True)
class RealManArmSpec:
    """One explicitly configured RealMan RM65 observation endpoint."""

    robot_id: str
    side: str
    arm_ip: str
    arm_urdf_path: str
    tool_profile: str | None = None
    pose_source_instance_id: str | None = None

    def __post_init__(self) -> None:
        robot_id = _required_text(self.robot_id, name="robot_id")
        side = _required_text(self.side, name="side").lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be exactly 'left' or 'right'")
        arm_ip = _required_text(self.arm_ip, name="arm_ip")
        arm_urdf_path = _required_text(self.arm_urdf_path, name="arm_urdf_path")
        tool_profile = self.tool_profile
        if tool_profile is not None:
            tool_profile = _required_text(tool_profile, name="tool_profile")
        source_instance_id = self.pose_source_instance_id
        if source_instance_id is not None:
            source_instance_id = _required_text(
                source_instance_id,
                name="pose_source_instance_id",
            )
        object.__setattr__(self, "robot_id", robot_id)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "arm_ip", arm_ip)
        object.__setattr__(self, "arm_urdf_path", arm_urdf_path)
        object.__setattr__(self, "tool_profile", tool_profile)
        object.__setattr__(self, "pose_source_instance_id", source_instance_id)


def _default_arm_urdf_path() -> str:
    from robot_motion.rm65.assets import require_asset_path

    return str(require_asset_path("rm65_arm.urdf"))


def _assignment_ips(value: object) -> dict[str, str]:
    if value is None or str(value).strip() == "":
        return {}
    from teleop_runtime.hardware_config.rm65_arm_ips import load_assignment

    payload = load_assignment(str(value))
    assignments = payload.get("assignments")
    if not isinstance(assignments, Mapping):
        raise TypeError("RM65 IP assignment has no assignments mapping")
    resolved: dict[str, str] = {}
    for side in ("left", "right"):
        side_payload = assignments.get(side)
        if isinstance(side_payload, Mapping):
            ip = str(side_payload.get("arm_ip") or "").strip()
            if ip:
                resolved[side] = ip
    return resolved


def parse_realman_arm_specs(realman_cfg: Mapping[str, Any]) -> tuple[RealManArmSpec, ...]:
    """Parse direct RM65 source configuration without opening a device."""

    if not isinstance(realman_cfg, Mapping):
        raise TypeError("robot_feedback.realman must be a mapping")
    raw_arms = realman_cfg.get("arms", {})
    if not isinstance(raw_arms, Mapping) or not raw_arms:
        raise ValueError("robot_feedback.realman.arms must be a non-empty mapping")
    assignment_value = realman_cfg.get("ip_assignment")
    assignment_ips: dict[str, str] | None = None
    default_urdf = str(realman_cfg.get("arm_urdf_path") or "").strip()

    specs: list[RealManArmSpec] = []
    seen_sides: set[str] = set()
    for raw_robot_id, raw_payload in raw_arms.items():
        robot_id = _required_text(str(raw_robot_id), name="robot_id")
        if not isinstance(raw_payload, Mapping):
            raise TypeError(f"realman.arms[{robot_id!r}] must be a mapping")
        if raw_payload.get("enabled", True) is not True:
            continue
        side = _required_text(raw_payload.get("side"), name=f"{robot_id}.side").lower()
        if side in seen_sides:
            raise ValueError(f"duplicate enabled RealMan side {side!r}")
        seen_sides.add(side)
        env_ip = os.environ.get(f"QUEST_REALMAN_{side.upper()}_IP", "").strip()
        if not raw_payload.get("arm_ip") and not env_ip and assignment_ips is None:
            assignment_ips = _assignment_ips(assignment_value)
        arm_ip = str(
            raw_payload.get("arm_ip")
            or env_ip
            or ({} if assignment_ips is None else assignment_ips).get(side)
            or ""
        ).strip()
        if not arm_ip:
            raise ValueError(
                f"realman.arms[{robot_id!r}] has no arm_ip; pass it explicitly, "
                f"set QUEST_REALMAN_{side.upper()}_IP, or configure ip_assignment"
            )
        arm_urdf_path = str(raw_payload.get("arm_urdf_path") or default_urdf).strip()
        if not arm_urdf_path:
            arm_urdf_path = _default_arm_urdf_path()
        specs.append(
            RealManArmSpec(
                robot_id=robot_id,
                side=side,
                arm_ip=arm_ip,
                arm_urdf_path=arm_urdf_path,
                tool_profile=(
                    None
                    if raw_payload.get("tool_profile") in (None, "")
                    else str(raw_payload["tool_profile"])
                ),
                pose_source_instance_id=(
                    None
                    if raw_payload.get("pose_source_instance_id") in (None, "")
                    else str(raw_payload["pose_source_instance_id"])
                ),
            )
        )
    if not specs:
        raise ValueError("realman.arms has no enabled arm")
    return tuple(specs)


def _build_backend(spec: RealManArmSpec) -> Any:
    """Construct the existing RM65 backend with its no-motion profile."""

    from teleop_runtime.backends.arms.rm65 import (
        RM65ArmBackend,
        RM65ArmBackendConfig,
    )

    return RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip=spec.arm_ip,
            side=spec.side,
            default_tcp_frame=DEFAULT_TCP_FRAME,
            arm_urdf_path=spec.arm_urdf_path,
            connect_hardware=True,
            tool_profile=spec.tool_profile,
            background_streaming_enabled=False,
            force_position_control=False,
            configure_singularity_thresholds=False,
            observation_only=True,
        ),
        pose_source_instance_id=spec.pose_source_instance_id,
    )


def _close_backend_without_motion(backend: Any) -> None:
    """Close an observation-only backend and its transport without a stop call."""

    arm = getattr(backend, "arm", None)
    try:
        backend.close()
    finally:
        # The shared observation-only close path intentionally avoids all
        # safety-control calls.  The API2 compatibility wrapper's
        # RM_API_UnInit is a no-op, so close the transport explicitly here.
        close_socket = getattr(arm, "Arm_Socket_Close", None)
        if callable(close_socket) and not bool(getattr(arm, "_closed", False)):
            close_socket()


def _pose_from_state(state: Mapping[str, Any]) -> np.ndarray | None:
    value = state.get("current_tcp_pose")
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    if (
        not np.all(np.isfinite(pose))
        or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
    ):
        return None
    return pose


def _coordinate_metadata(robot_id: str, coordinate_binding: Any) -> dict[str, Any]:
    fields = (
        "source_instance_id",
        "source_epoch_id",
        "source_binding_hash",
        "component_id",
        "stream_id",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "moving_frame_id",
        "tool_profile_hash",
        "pose_schema_hash",
        "fk_contract_id",
        "fk_contract_hash",
        "source_kind",
    )
    metadata = {field: getattr(coordinate_binding, field) for field in fields}
    metadata.update(
        {
            "robot_id": robot_id,
            "coordinate_binding_hash": coordinate_binding.semantic_content_hash,
        }
    )
    return metadata


class RealManFeedbackReader:
    """Own direct RM65 observation resources for one Quest host."""

    def __init__(
        self,
        specs: Sequence[RealManArmSpec],
        *,
        max_age_s: float,
        frame_bindings: Mapping[str, AcceptedFrameBinding] | None = None,
        control_domain_id: str = DEFAULT_CONTROL_DOMAIN_ID,
        clock: Callable[[], float] = time.time,
        monotonic_ns_clock: Callable[[], int] = time.monotonic_ns,
        backend_factory: Callable[[RealManArmSpec], Any] = _build_backend,
    ) -> None:
        normalized_specs = tuple(specs)
        if not normalized_specs:
            raise ValueError("at least one RealMan arm spec is required")
        if len({spec.robot_id for spec in normalized_specs}) != len(normalized_specs):
            raise ValueError("RealMan robot_id values must be unique")
        if len({spec.side for spec in normalized_specs}) != len(normalized_specs):
            raise ValueError("RealMan sides must be unique")
        self.specs = normalized_specs
        self.max_age_s = _finite_positive(max_age_s, name="max_age_s")
        self.frame_bindings = dict(frame_bindings or {})
        self.control_domain_id = _required_text(
            control_domain_id,
            name="control_domain_id",
        )
        if not callable(clock) or not callable(monotonic_ns_clock):
            raise TypeError("clock functions must be callable")
        if not callable(backend_factory):
            raise TypeError("backend_factory must be callable")
        self.clock = clock
        self.monotonic_ns_clock = monotonic_ns_clock
        self.backend_factory = backend_factory
        self.backends_by_side: dict[str, Any] = {}
        self.spec_by_robot: dict[str, RealManArmSpec] = {
            spec.robot_id: spec for spec in normalized_specs
        }
        self.pose_reader: Any = None
        self.projection_by_side: dict[str, Any] = {}
        self.receiver_clock: Any = None
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        from teleop_runtime.deploy.arm_state_reader import ArmStateReader
        from teleop_runtime.deploy.rm65_pose_observation import (
            assemble_rm65_pose_observation,
            build_policy_deploy_receiver_clock,
        )

        started: list[Any] = []
        try:
            for spec in self.specs:
                backend = self.backend_factory(spec)
                backend.start()
                self.backends_by_side[spec.side] = backend
                started.append(backend)

            # A reviewed frame binding must be checked against the exact
            # source/tool/FK epoch created by this process.  Without a
            # binding, direct state diagnostics still work but Quest-space
            # rendering remains disabled.
            if self.frame_bindings:
                self.receiver_clock = build_policy_deploy_receiver_clock(
                    runtime_session_id=f"quest-dev:realman:{uuid4().hex}"
                )
                assembly = assemble_rm65_pose_observation(
                    self.backends_by_side,
                    receipt_clock=self.receiver_clock,
                    control_domain_id=self.control_domain_id,
                    bound_at=float(self.clock()),
                    receipt_monotonic_ns_clock=self.monotonic_ns_clock,
                )
                self.projection_by_side = dict(assembly.projection_by_side)
                self.pose_reader = ArmStateReader(
                    self.backends_by_side,
                    refresh_hz=0.0,
                    compute_tcp_pose=True,
                    clock=self.clock,
                    pose_projection_by_side=assembly.projection_by_side,
                    pose_state_read_by_side=assembly.state_read_by_side,
                    receipt_clock=assembly.receipt_clock,
                    receipt_monotonic_ns_clock=assembly.receipt_monotonic_ns_clock,
                )
            self.started = True
        except BaseException:
            for backend in reversed(started):
                try:
                    _close_backend_without_motion(backend)
                except Exception as cleanup_error:  # noqa: BLE001
                    print(
                        "[quest-dev:visualization] RealMan startup cleanup error: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}",
                        flush=True,
                    )
            self.backends_by_side.clear()
            self.pose_reader = None
            self.projection_by_side.clear()
            raise

    def _state_record(
        self,
        spec: RealManArmSpec,
        state: Mapping[str, Any] | None,
        *,
        now_monotonic_ns: int,
    ) -> dict[str, Any]:
        base = {
            "robot": spec.robot_id,
            "key": f"realman:{spec.robot_id}",
            "source": REALMAN_SOURCE_NAME,
            "fresh": False,
            "control_ready": False,
            "alignment_ready": False,
        }
        if not isinstance(state, Mapping):
            return {**base, "error": "hardware_state_unavailable"}
        if state.get("hardware_read_success") is not True:
            result = state.get("hardware_result_text") or "read_failed"
            return {**base, "error": f"hardware_read_failed:{result}"}
        raw_read_ns = state.get("hardware_read_timestamp_monotonic")
        try:
            read_ns = int(float(raw_read_ns) * 1_000_000_000.0)
        except (TypeError, ValueError, OverflowError):
            return {**base, "error": "missing_hardware_read_clock"}
        age_s = (int(now_monotonic_ns) - read_ns) / 1_000_000_000.0
        if age_s < -0.1:
            return {**base, "error": "hardware_read_timestamp_in_future"}
        if age_s > self.max_age_s:
            return {**base, "age_s": age_s, "error": "stale"}
        pose = _pose_from_state(state)
        if pose is None:
            return {**base, "age_s": age_s, "error": "invalid_tcp_pose"}
        normalized = matrix_to_pose_payload(pose)
        result = {
            **base,
            "fresh": True,
            "age_s": age_s,
            "mode": "observation_only",
            "frame": f"{spec.side}_arm_base",
            "position": pose[:3, 3].tolist(),
            "orientation_quat_xyzw": normalized["orientation_quat_xyzw"],
            "matrix_robot_from_tcp": matrix_to_column_major(pose),
            "source_sequence": int(state.get("hardware_measurement_sequence") or 0),
        }
        projection = self.projection_by_side.get(spec.side)
        if projection is not None:
            result["pose_binding"] = _coordinate_metadata(
                spec.robot_id,
                projection.coordinate_binding,
            )
        return result

    def poll(self) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("RealManFeedbackReader is not started")
        now_monotonic_ns = int(self.monotonic_ns_clock())
        state_by_side: dict[str, Mapping[str, Any] | None] = {}
        for spec in self.specs:
            backend = self.backends_by_side[spec.side]
            try:
                state_by_side[spec.side] = backend.refresh_state_from_hardware(
                    include_derived_pose=True,
                    include_timing_percentiles=False,
                    include_pose_source_attestation=self.pose_reader is not None,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one arm read.
                state_by_side[spec.side] = {
                    "hardware_read_success": False,
                    "hardware_result_text": f"{type(exc).__name__}: {exc}",
                }

        if self.pose_reader is not None:
            try:
                self.pose_reader.snapshot(float(self.clock()), cut_after_refresh=True)
            except Exception as exc:  # noqa: BLE001 - fail closed per arm below.
                print(
                    f"[quest-dev:visualization] RealMan pose publication error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

        robots: dict[str, dict[str, Any]] = {}
        for spec in self.specs:
            record = self._state_record(
                spec,
                state_by_side.get(spec.side),
                now_monotonic_ns=now_monotonic_ns,
            )
            binding = self.frame_bindings.get(spec.robot_id)
            if (
                binding is not None
                and record.get("fresh") is True
                and self.pose_reader is not None
            ):
                publication = self.pose_reader.latest_pose_measurement(side=spec.side)
                if publication is None:
                    record["alignment_error"] = "missing_pose_publication"
                else:
                    projected = project_pose_publication(
                        publication,
                        expected_robot=spec.robot_id,
                        key=f"realman:{spec.robot_id}",
                        max_age_s=self.max_age_s,
                        now_monotonic_ns=now_monotonic_ns,
                        frame_binding=binding,
                    )
                    projected["source"] = REALMAN_SOURCE_NAME
                    # Retain the direct state error/age gate even if a stale
                    # journal publication happens to remain cached.
                    if record.get("fresh") is not True:
                        projected.pop("matrix_quest_from_tcp", None)
                        projected["fresh"] = False
                    record = projected
            robots[spec.robot_id] = record
        return robot_tcp_message(robots)

    def stop(self) -> None:
        if not self.started and not self.backends_by_side:
            return
        if self.pose_reader is not None:
            try:
                self.pose_reader.invalidate_pose_measurements(
                    reason="realman_feedback_stopped",
                    terminal=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "[quest-dev:visualization] RealMan journal invalidation error: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        errors: list[str] = []
        for backend in reversed(tuple(self.backends_by_side.values())):
            try:
                _close_backend_without_motion(backend)
            except Exception as exc:  # noqa: BLE001 - aggregate cleanup evidence.
                errors.append(f"{type(exc).__name__}: {exc}")
        self.backends_by_side.clear()
        self.pose_reader = None
        self.projection_by_side.clear()
        self.started = False
        if errors:
            raise RuntimeError("; ".join(errors))


__all__ = [
    "DEFAULT_CONTROL_DOMAIN_ID",
    "REALMAN_SOURCE_NAME",
    "RealManArmSpec",
    "RealManFeedbackReader",
    "parse_realman_arm_specs",
]
