from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...diagnostics import FaultEvent
from ...schemas import EffectorCommand, TeleopCommand
from ...transforms import as_pose_matrix, invert_pose

INIT_ARM_FAULT_CODES = frozenset(
    {
        "missing_channel",
        "invalid_channel",
        "invalid_pose",
        "stale_channel",
        "translation_jump",
        "rotation_jump",
    }
)


def normalize_init_arm_side(side: str) -> str:
    normalized = str(side or "").strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"init-arm side must be 'left' or 'right', got {side!r}.")
    return normalized


def _finite_vector(value: Any, *, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite {length}D vector, got shape {array.shape}.")
    return array.copy()


def _pose_or_none(value: Any, *, label: str) -> np.ndarray | None:
    if value is None:
        return None
    pose = as_pose_matrix(value)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} must be a finite 4x4 pose matrix.")
    return pose.copy()


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, Any] | tuple[None, None]:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return key, mapping[key]
    return None, None


def _side_payload(payload: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    candidates: list[Any] = []
    sides = payload.get("sides")
    if isinstance(sides, Mapping):
        candidates.extend([sides.get(side), sides.get(f"{side}_arm")])
    candidates.extend(
        [
            payload.get(side),
            payload.get(f"{side}_arm"),
            payload.get(f"{side}_init"),
            payload.get(f"{side}_init_pose"),
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


@dataclass(frozen=True)
class InitArmResolvedTarget:
    side: str
    tcp_pose_base: np.ndarray
    tcp_pose_desk: np.ndarray | None = None
    solver_pose_desk: np.ndarray | None = None
    arm_qpos: np.ndarray | None = None
    source_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", normalize_init_arm_side(self.side))
        object.__setattr__(self, "tcp_pose_base", as_pose_matrix(self.tcp_pose_base))
        if self.tcp_pose_desk is not None:
            object.__setattr__(self, "tcp_pose_desk", as_pose_matrix(self.tcp_pose_desk))
        if self.solver_pose_desk is not None:
            object.__setattr__(self, "solver_pose_desk", as_pose_matrix(self.solver_pose_desk))
        if self.arm_qpos is not None:
            object.__setattr__(self, "arm_qpos", _finite_vector(self.arm_qpos, length=6, label=f"{self.side}.arm_qpos"))


@dataclass(frozen=True)
class _RawInitArmTarget:
    side: str
    tcp_pose_base: np.ndarray | None = None
    tcp_pose_desk: np.ndarray | None = None
    arm_qpos: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitArmTransition:
    ok: bool
    message: str
    side: str = ""
    mode: str = ""


@dataclass
class _RecalibrationState:
    started_at: float = 0.0
    timeout_sec: float = 0.0
    min_samples: int = 1
    samples: list[np.ndarray] = field(default_factory=list)


def _parse_raw_side_target(
    payload: Mapping[str, Any],
    side: str,
    *,
    source_path: str,
) -> _RawInitArmTarget:
    side_payload = _side_payload(payload, side)
    metadata = dict(side_payload.get("metadata") or {})
    metadata.setdefault("source_path", source_path)

    q_key, q_value = _first_present(
        side_payload,
        (
            "arm_qpos",
            "qpos",
            "joint_qpos",
            "target_qpos",
            "init_qpos",
            "arm_qpos_rad",
            "qpos_rad",
            "joint_rad",
            "arm_qpos_deg",
            "qpos_deg",
            "joint_deg",
        ),
    )
    qpos = None
    if q_value is not None:
        qpos = _finite_vector(q_value, length=6, label=f"{side}.{q_key}")
        unit = str(side_payload.get("qpos_unit") or payload.get("qpos_unit") or "").strip().lower()
        if q_key and (str(q_key).endswith("_deg") or unit in {"deg", "degree", "degrees"}):
            qpos = np.deg2rad(qpos)

    pose_base = _pose_or_none(
        _first_present(
            side_payload,
            ("tcp_pose_base", "target_pose_base", "pose_base", "init_pose_base"),
        )[1],
        label=f"{side}.tcp_pose_base",
    )
    pose_desk = _pose_or_none(
        _first_present(
            side_payload,
            ("tcp_pose_desk", "target_pose_desk", "pose_desk", "init_pose_desk"),
        )[1],
        label=f"{side}.tcp_pose_desk",
    )
    if qpos is None and pose_base is None and pose_desk is None:
        raise ValueError(
            f"init-arm profile side {side!r} needs one of arm_qpos, tcp_pose_base, or tcp_pose_desk."
        )
    return _RawInitArmTarget(
        side=side,
        tcp_pose_base=pose_base,
        tcp_pose_desk=pose_desk,
        arm_qpos=qpos,
        metadata=metadata,
    )


def _raw_targets_from_profile(path: str | Path) -> dict[str, _RawInitArmTarget]:
    source_path = str(path)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("init-arm profile must be a JSON object.")
    return {
        side: _parse_raw_side_target(payload, side, source_path=source_path)
        for side in ("left", "right")
    }


def _calibration_desk_to_base(calibration: Any, pose_desk: np.ndarray, side: str) -> np.ndarray:
    desk_to_base = getattr(calibration, "desk_to_base", None)
    if not callable(desk_to_base):
        raise ValueError("init-arm tcp_pose_desk requires a TeleopCalibration with desk_to_base().")
    return as_pose_matrix(desk_to_base(pose_desk, side=side))


def _calibration_base_to_desk(calibration: Any, pose_base: np.ndarray, side: str) -> np.ndarray | None:
    get_arm = getattr(calibration, "get_arm_for_side", None)
    if not callable(get_arm):
        return None
    arm = get_arm(side)
    transform = getattr(arm, "T_D_B", None)
    if transform is None:
        return None
    return as_pose_matrix(transform) @ as_pose_matrix(pose_base)


def _tool_tcp_from_solver(dual_arm_retargeter: Any, side: str) -> np.ndarray | None:
    raw = getattr(dual_arm_retargeter, "_tool_tcp_from_solver", None)
    if isinstance(raw, Mapping) and raw.get(side) is not None:
        return as_pose_matrix(raw[side])
    return None


def _split_current_qpos(dual_arm_retargeter: Any) -> tuple[np.ndarray, np.ndarray]:
    current_q = np.asarray(dual_arm_retargeter.ik_solver.current_q, dtype=float).reshape(-1)
    splitter = getattr(dual_arm_retargeter, "_split_solution", None)
    if callable(splitter):
        left, right = splitter(current_q)
        return np.asarray(left, dtype=float).reshape(6), np.asarray(right, dtype=float).reshape(6)
    if current_q.shape[0] == 12:
        return current_q[0:6].copy(), current_q[6:12].copy()
    if current_q.shape[0] == 14:
        return current_q[1:7].copy(), current_q[8:14].copy()
    if current_q.shape[0] >= 24:
        return current_q[0:6].copy(), current_q[18:24].copy()
    raise ValueError(f"cannot split dual-arm solver qpos shape {current_q.shape}.")


def _embed_qpos(
    dual_arm_retargeter: Any,
    *,
    left_qpos: np.ndarray,
    right_qpos: np.ndarray,
) -> np.ndarray:
    current_q = np.asarray(dual_arm_retargeter.ik_solver.current_q, dtype=float).reshape(-1)
    embedder = getattr(dual_arm_retargeter, "_embed_arm_qpos", None)
    if callable(embedder):
        return np.asarray(
            embedder(
                current_full_q=current_q,
                left_qpos=np.asarray(left_qpos, dtype=float).reshape(6),
                right_qpos=np.asarray(right_qpos, dtype=float).reshape(6),
            ),
            dtype=float,
        )
    full_q = current_q.copy()
    if full_q.shape[0] == 12:
        full_q[0:6] = left_qpos
        full_q[6:12] = right_qpos
        return full_q
    if full_q.shape[0] == 14:
        full_q[1:7] = left_qpos
        full_q[8:14] = right_qpos
        return full_q
    if full_q.shape[0] >= 24:
        full_q[0:6] = left_qpos
        full_q[18:24] = right_qpos
        return full_q
    raise ValueError(f"cannot embed dual-arm qpos into solver qpos shape {full_q.shape}.")


def load_init_arm_targets(
    path: str | Path,
    *,
    calibration: Any,
    dual_arm_retargeter: Any | None = None,
) -> dict[str, InitArmResolvedTarget]:
    raw_targets = _raw_targets_from_profile(path)
    current_left_qpos: np.ndarray | None = None
    current_right_qpos: np.ndarray | None = None
    if dual_arm_retargeter is not None:
        current_left_qpos, current_right_qpos = _split_current_qpos(dual_arm_retargeter)

    resolved: dict[str, InitArmResolvedTarget] = {}
    for side in ("left", "right"):
        raw = raw_targets[side]
        tcp_pose_base = None if raw.tcp_pose_base is None else as_pose_matrix(raw.tcp_pose_base)
        tcp_pose_desk = None if raw.tcp_pose_desk is None else as_pose_matrix(raw.tcp_pose_desk)
        solver_pose_desk = None

        if raw.arm_qpos is not None:
            if dual_arm_retargeter is None:
                if tcp_pose_base is None and tcp_pose_desk is None:
                    raise ValueError(
                        f"init-arm profile side {side!r} only has arm_qpos, "
                        "but dual-arm FK is unavailable. Add tcp_pose_base/tcp_pose_desk."
                    )
            else:
                left_qpos = raw_targets["left"].arm_qpos
                right_qpos = raw_targets["right"].arm_qpos
                if left_qpos is None:
                    if current_left_qpos is None:
                        raise ValueError("cannot resolve left fallback qpos for init-arm FK.")
                    left_qpos = current_left_qpos
                if right_qpos is None:
                    if current_right_qpos is None:
                        raise ValueError("cannot resolve right fallback qpos for init-arm FK.")
                    right_qpos = current_right_qpos
                full_q = _embed_qpos(
                    dual_arm_retargeter,
                    left_qpos=np.asarray(left_qpos, dtype=float).reshape(6),
                    right_qpos=np.asarray(right_qpos, dtype=float).reshape(6),
                )
                frame_name = (
                    getattr(dual_arm_retargeter.profile, "left_frame_name")
                    if side == "left"
                    else getattr(dual_arm_retargeter.profile, "right_frame_name")
                )
                solver_pose_desk = as_pose_matrix(
                    dual_arm_retargeter.ik_solver.compute_frame_pose(frame_name, full_q)
                )
                tool_tcp = _tool_tcp_from_solver(dual_arm_retargeter, side)
                tcp_pose_desk = (
                    solver_pose_desk.copy()
                    if tool_tcp is None
                    else solver_pose_desk @ tool_tcp
                )
                tcp_pose_base = _calibration_desk_to_base(calibration, tcp_pose_desk, side)

        if tcp_pose_base is None and tcp_pose_desk is not None:
            tcp_pose_base = _calibration_desk_to_base(calibration, tcp_pose_desk, side)
        if tcp_pose_desk is None and tcp_pose_base is not None:
            tcp_pose_desk = _calibration_base_to_desk(calibration, tcp_pose_base, side)
        if solver_pose_desk is None and tcp_pose_desk is not None and dual_arm_retargeter is not None:
            tool_tcp = _tool_tcp_from_solver(dual_arm_retargeter, side)
            solver_pose_desk = (
                tcp_pose_desk.copy()
                if tool_tcp is None
                else tcp_pose_desk @ invert_pose(tool_tcp)
            )
        if tcp_pose_base is None:
            raise ValueError(f"init-arm profile side {side!r} could not be resolved to tcp_pose_base.")

        resolved[side] = InitArmResolvedTarget(
            side=side,
            tcp_pose_base=tcp_pose_base,
            tcp_pose_desk=tcp_pose_desk,
            solver_pose_desk=solver_pose_desk,
            arm_qpos=raw.arm_qpos,
            source_path=str(path),
            metadata=dict(raw.metadata),
        )
    return resolved


class InitArmCommandController:
    """Route one selected arm to an init pose without changing dataset schemas."""

    def __init__(
        self,
        *,
        targets_by_side: Mapping[str, InitArmResolvedTarget],
        selected_side: str = "left",
        use_dual_target_override: bool = False,
    ) -> None:
        self.targets_by_side = {
            normalize_init_arm_side(side): target
            for side, target in targets_by_side.items()
        }
        if set(self.targets_by_side) != {"left", "right"}:
            raise ValueError("init-arm requires resolved left and right targets.")
        self._selected_side = normalize_init_arm_side(selected_side)
        self._mode = "live"
        self._active_side = ""
        self._recalibration = _RecalibrationState()
        self._status_message = "init-arm ready"
        self._use_dual_target_override = bool(use_dual_target_override)

    @property
    def enabled(self) -> bool:
        return True

    @property
    def selected_side(self) -> str:
        return self._selected_side

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def active_side(self) -> str:
        return self._active_side

    @property
    def fixed_side(self) -> str:
        return self._active_side

    @property
    def active(self) -> bool:
        return self._mode in {"moving", "recapturing"} and bool(self._active_side)

    @property
    def recapturing(self) -> bool:
        return self._mode == "recapturing" and bool(self._active_side)

    @property
    def status_message(self) -> str:
        return self._status_message

    @property
    def sample_count(self) -> int:
        return len(self._recalibration.samples)

    @property
    def min_samples(self) -> int:
        return max(1, int(self._recalibration.min_samples))

    @property
    def recalibration_started_at(self) -> float:
        return float(self._recalibration.started_at)

    @property
    def recalibration_timeout_sec(self) -> float:
        return float(self._recalibration.timeout_sec)

    def recalibration_samples(self) -> list[np.ndarray]:
        return [sample.copy() for sample in self._recalibration.samples]

    def set_dual_target_override_enabled(self, enabled: bool) -> None:
        self._use_dual_target_override = bool(enabled)

    def select_side(self, side: str) -> InitArmTransition:
        normalized = normalize_init_arm_side(side)
        if self.active:
            return InitArmTransition(
                ok=False,
                message="init side cannot be changed while init-arm is active",
                side=self._active_side,
                mode=self._mode,
            )
        self._selected_side = normalized
        self._status_message = f"selected {normalized}"
        return InitArmTransition(ok=True, message=self._status_message, side=normalized, mode=self._mode)

    def start_init(self, *, now: float) -> InitArmTransition:
        del now
        if self.active:
            return InitArmTransition(
                ok=False,
                message=f"{self._active_side} is already in init-arm mode",
                side=self._active_side,
                mode=self._mode,
            )
        side = self._selected_side
        if side not in self.targets_by_side:
            self._status_message = f"cannot init {side}; target missing"
            return InitArmTransition(ok=False, message=self._status_message, side=side, mode=self._mode)
        self._active_side = side
        self._mode = "moving"
        self._recalibration = _RecalibrationState()
        self._status_message = f"{side} moving to init pose"
        return InitArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def start_recalibration(
        self,
        *,
        now: float,
        timeout_sec: float,
        min_samples: int,
    ) -> InitArmTransition:
        if not self.active:
            return InitArmTransition(
                ok=False,
                message="init-arm recalibration requires an active init side",
                side=self._selected_side,
                mode=self._mode,
            )
        self._mode = "recapturing"
        self._recalibration = _RecalibrationState(
            started_at=float(now),
            timeout_sec=max(0.1, float(timeout_sec)),
            min_samples=max(1, int(min_samples)),
        )
        self._status_message = f"recapturing {self._active_side} home at init pose"
        return InitArmTransition(ok=True, message=self._status_message, side=self._active_side, mode=self._mode)

    def add_recalibration_sample(self, pose_desk: np.ndarray) -> int:
        if not self.recapturing:
            return self.sample_count
        if self.sample_count >= self.min_samples:
            return self.sample_count
        self._recalibration.samples.append(np.asarray(pose_desk, dtype=float).reshape(4, 4).copy())
        return self.sample_count

    def complete_recalibration(self, *, now: float, message: str = "") -> InitArmTransition:
        del now
        side = self._active_side
        self._mode = "live"
        self._active_side = ""
        self._recalibration = _RecalibrationState()
        self._status_message = message or f"{side} home recaptured; init-arm released"
        return InitArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def cancel_active(self, *, now: float, reason: str = "") -> InitArmTransition:
        del now
        side = self._active_side
        was_active = bool(self.active)
        self._mode = "live"
        self._active_side = ""
        self._recalibration = _RecalibrationState()
        if was_active:
            suffix = f": {reason}" if reason else ""
            self._status_message = f"{side} init-arm stopped{suffix}"
        else:
            self._status_message = "init-arm ready"
        return InitArmTransition(
            ok=was_active,
            message=self._status_message,
            side=side,
            mode=self._mode,
        )

    def fail_recalibration(self, *, now: float, message: str) -> InitArmTransition:
        del now
        side = self._active_side
        self._mode = "moving" if side else "live"
        self._recalibration = _RecalibrationState()
        self._status_message = str(message)
        return InitArmTransition(ok=False, message=self._status_message, side=side, mode=self._mode)

    def active_target(self) -> InitArmResolvedTarget | None:
        if not self.active:
            return None
        return self.targets_by_side.get(self._active_side)

    def filter_dual_arm_targets(
        self,
        targets: Mapping[str, np.ndarray],
        _observation,
        _calibration,
    ) -> dict[str, np.ndarray] | None:
        target = self.active_target()
        if target is None or target.solver_pose_desk is None:
            return None
        filtered = {
            side: as_pose_matrix(pose).copy()
            for side, pose in targets.items()
        }
        filtered[target.side] = target.solver_pose_desk.copy()
        return filtered

    def filter_command(self, command: TeleopCommand, _observation, _now: float) -> TeleopCommand:
        target = self.active_target()
        if target is None:
            return command
        effector_name = f"{target.side}_arm"
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
        filtered.metadata["init_arm_active_side"] = target.side
        filtered.metadata["init_arm_mode"] = self._mode
        for name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            if name == effector_name:
                next_command = self._filter_arm_command(next_command, target)
            filtered.commands[name] = next_command
        return filtered

    def filter_faults(
        self,
        faults: Sequence[FaultEvent],
        _channels,
        _now: float,
    ) -> list[FaultEvent]:
        if not self.active:
            return list(faults)
        active_channels = {f"{self._active_side}_wrist_pose"}
        filtered: list[FaultEvent] = []
        for fault in faults:
            if fault.channel_name in active_channels and fault.code in INIT_ARM_FAULT_CODES:
                continue
            filtered.append(fault)
        return filtered

    def status_payload(self) -> dict[str, object]:
        active_side = self._active_side if self.active else ""
        return {
            "init_arm_available": True,
            "init_arm_selected_side": self._selected_side,
            "init_arm_active_side": active_side,
            "init_arm_active": bool(active_side),
            "init_arm_recalibrating": self.recapturing,
            "init_arm_status": self._status_message,
            "init_arm_sample_count": self.sample_count if self.recapturing else None,
            "init_arm_min_samples": self.min_samples if self.recapturing else None,
            "init_arm_profile_path": next(iter(self.targets_by_side.values())).source_path,
        }

    def _filter_arm_command(
        self,
        command: EffectorCommand,
        target: InitArmResolvedTarget,
    ) -> EffectorCommand:
        metadata = dict(command.metadata)
        metadata.update(
            {
                "init_arm_active": True,
                "init_arm_side": target.side,
                "init_arm_mode": self._mode,
                "init_arm_profile_path": target.source_path,
                "init_arm_target_pose_base": target.tcp_pose_base.copy(),
            }
        )
        if target.tcp_pose_desk is not None:
            metadata["init_arm_target_pose_desk"] = target.tcp_pose_desk.copy()
        if target.arm_qpos is not None:
            metadata["init_arm_target_qpos"] = target.arm_qpos.copy()

        if self._use_dual_target_override and target.solver_pose_desk is not None:
            command.metadata = metadata
            return command

        metadata.update(
            {
                "arm_controlled_frame": "tool_tcp",
                "arm_reference_frame": "arm_base",
                "target_pose_base": target.tcp_pose_base.copy(),
                "target_pose_tcp_base": target.tcp_pose_base.copy(),
            }
        )
        if target.tcp_pose_desk is not None:
            metadata["target_pose_desk"] = target.tcp_pose_desk.copy()
            metadata["target_pose_tcp_desk"] = target.tcp_pose_desk.copy()
        if target.arm_qpos is not None:
            metadata["solver_output_qpos"] = target.arm_qpos.copy()
        command.command_type = "cartesian_pose"
        command.value = target.tcp_pose_base.copy()
        command.frame_name = "arm_base"
        command.metadata = metadata
        return command
