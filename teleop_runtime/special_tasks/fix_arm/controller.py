from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Mapping, Sequence

import numpy as np

from ...diagnostics import FaultEvent
from ...schemas import EffectorCommand, ObservationChannel, TeleopCommand
from ..home_capture_stability import HomeCaptureStabilityState

FIX_ARM_FAULT_CODES = frozenset(
    {
        "missing_channel",
        "invalid_channel",
        "invalid_pose",
        "stale_channel",
        "translation_jump",
        "rotation_jump",
    }
)

# A profile-driven arm handoff may use the latest tracker sample as an atomic
# zero-delta rebase when a stillness-qualified passive Home is unavailable.
# Keep this substantially tighter than the ordinary 0.5 s Home freshness
# bound: a moving hand must not turn an older sample into queued catch-up.
PROFILE_TRANSITION_FRESH_REBASE_MAX_AGE_SEC = 0.075


def normalize_fix_arm_side(side: str) -> str:
    normalized = str(side or "").strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"fix-arm side must be 'left' or 'right', got {side!r}.")
    return normalized


@dataclass(frozen=True)
class FixArmTransition:
    ok: bool
    message: str
    side: str = ""
    mode: str = ""
    missing_effectors: tuple[str, ...] = ()
    missing_channels: tuple[str, ...] = ()


@dataclass
class _RecalibrationState:
    started_at: float = 0.0
    timeout_sec: float = 0.0
    min_samples: int = 1
    samples: list[np.ndarray] = field(default_factory=list)
    stability: HomeCaptureStabilityState = field(
        default_factory=HomeCaptureStabilityState
    )


@dataclass
class _PassiveHomeCache:
    stability: HomeCaptureStabilityState = field(
        default_factory=HomeCaptureStabilityState.transition_profile
    )
    pose: np.ndarray | None = None
    source_timestamp: float = 0.0
    last_observed_at: float = 0.0
    unavailable_reason: str = "waiting_for_valid_fresh_tracker_sample"


class FixArmCommandController:
    """Freeze every configured effector on one or both logical sides."""

    def __init__(
        self,
        *,
        active_sides: Sequence[str],
        required_effectors_by_side: Mapping[str, Sequence[str]],
        selected_side: str = "left",
    ) -> None:
        self.active_sides = tuple(normalize_fix_arm_side(side) for side in active_sides)
        if set(self.active_sides) != {"left", "right"}:
            raise ValueError("--fix-arm requires active-sides both.")
        self.required_effectors_by_side = {}
        for side in self.active_sides:
            configured = tuple(
                dict.fromkeys(
                    str(name)
                    for name in required_effectors_by_side.get(side, ())
                    if str(name)
                )
            )
            arm_name = f"{side}_arm"
            if arm_name not in configured:
                configured = (arm_name, *configured)
            self.required_effectors_by_side[side] = configured
        self._scope_by_side = {
            side: (
                "arm_and_end_effector"
                if any(name != f"{side}_arm" for name in effectors)
                else "arm_only"
            )
            for side, effectors in self.required_effectors_by_side.items()
        }
        self._warning_by_side = {
            side: (
                "end_effector_not_configured; fix_arm is arm_only"
                if self._scope_by_side[side] == "arm_only"
                else ""
            )
            for side in self.active_sides
        }
        self._selected_side = normalize_fix_arm_side(selected_side)
        self._mode = "live"
        self._fixed_side = ""
        self._fixed_sides: set[str] = set()
        self._recapturing_side = ""
        self._post_recalibration_fixed_side: str | None = None
        self._two_pedal_transition_pending = False
        self._latest_by_effector: dict[str, EffectorCommand] = {}
        self._fixed_by_effector: dict[str, EffectorCommand] = {}
        self._latest_by_channel: dict[str, ObservationChannel] = {}
        self._fixed_by_channel: dict[str, ObservationChannel] = {}
        self._runtime_home_epoch: int | None = None
        self._runtime_home_arm_by_side: dict[str, EffectorCommand] = {}
        self._recalibration = _RecalibrationState()
        self._passive_home_cache = {
            side: _PassiveHomeCache() for side in self.active_sides
        }
        self._status_message = "fix-arm ready"

    def scope_for_side(self, side: str) -> str:
        return self._scope_by_side.get(normalize_fix_arm_side(side), "arm_only")

    def warning_for_side(self, side: str) -> str:
        return self._warning_by_side.get(normalize_fix_arm_side(side), "")

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
    def fixed_side(self) -> str:
        if self._recapturing_side:
            return self._recapturing_side
        if self._selected_side in self._fixed_sides:
            return self._selected_side
        return self._fixed_side if self._fixed_side in self._fixed_sides else ""

    @property
    def fixed_sides(self) -> tuple[str, ...]:
        return tuple(side for side in self.active_sides if side in self._fixed_sides)

    def is_fixed(self, side: str) -> bool:
        return normalize_fix_arm_side(side) in self._fixed_sides

    @property
    def fixed(self) -> bool:
        return bool(self._fixed_sides)

    @property
    def recapturing(self) -> bool:
        return self._mode == "recapturing" and bool(self._recapturing_side)

    @property
    def two_pedal_transition_pending(self) -> bool:
        return bool(self._two_pedal_transition_pending)

    @property
    def two_pedal_transition_target_fixed_side(self) -> str | None:
        return self._post_recalibration_fixed_side

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

    @property
    def recalibration_stability_reason(self) -> str:
        return str(self._recalibration.stability.reason or "")

    def note_recalibration_unavailable(self, reason: str) -> None:
        if not self.recapturing:
            return
        self._recalibration.stability.reason = (
            str(reason).strip() or "waiting_for_valid_fresh_tracker_sample"
        )

    def note_two_pedal_home_unavailable(
        self,
        side: str,
        *,
        reason: str,
        observed_at: float,
    ) -> None:
        normalized = normalize_fix_arm_side(side)
        cache = self._passive_home_cache[normalized]
        cache.last_observed_at = float(observed_at)
        cache.unavailable_reason = (
            str(reason).strip() or "waiting_for_valid_fresh_tracker_sample"
        )

    def passive_home_cache_status(
        self,
        side: str,
        *,
        now: float,
        allow_fresh_rebase: bool = False,
    ) -> dict[str, object]:
        normalized = normalize_fix_arm_side(side)
        cache = self._passive_home_cache[normalized]
        age_sec = (
            None
            if cache.pose is None or cache.source_timestamp <= 0.0
            else max(0.0, float(now) - float(cache.source_timestamp))
        )
        stable_ready = bool(age_sec is not None and age_sec <= 0.5)
        latest_timestamp = cache.stability.last_timestamp
        latest_age_sec = (
            None
            if cache.stability.last_pose is None or latest_timestamp is None
            else float(now) - float(latest_timestamp)
        )
        fresh_rebase_ready = bool(
            latest_age_sec is not None
            and -0.05 <= latest_age_sec <= PROFILE_TRANSITION_FRESH_REBASE_MAX_AGE_SEC
        )
        ready = bool(stable_ready or (allow_fresh_rebase and fresh_rebase_ready))
        quality = (
            "stable"
            if stable_ready
            else (
                "fresh_rebase"
                if allow_fresh_rebase and fresh_rebase_ready
                else "unavailable"
            )
        )
        return {
            "side": normalized,
            "ready": ready,
            "stable_ready": stable_ready,
            "fresh_rebase_ready": fresh_rebase_ready,
            "quality": quality,
            "age_sec": age_sec,
            "latest_age_sec": latest_age_sec,
            "reason": (
                "stable_home_cache_ready"
                if stable_ready
                else (
                    "fresh_tracker_rebase_ready"
                    if allow_fresh_rebase and fresh_rebase_ready
                    else str(cache.unavailable_reason)
                )
            ),
        }

    def select_side(self, side: str) -> FixArmTransition:
        normalized = normalize_fix_arm_side(side)
        if self.recapturing:
            return FixArmTransition(
                ok=False,
                message="fixed side cannot be changed while Home recapture is active",
                side=self._recapturing_side,
                mode=self._mode,
            )
        self._selected_side = normalized
        self._status_message = f"selected {normalized}"
        return FixArmTransition(
            ok=True,
            message=f"selected {normalized}",
            side=normalized,
            mode=self._mode,
        )

    def bind_runtime_home_arm_snapshot(
        self,
        qpos_by_side: Mapping[str, object],
        *,
        epoch: int,
    ) -> None:
        """Cache one admitted runtime-Home arm snapshot for source-free startup Fix."""

        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError("runtime Home epoch must be a non-negative integer")
        snapshots: dict[str, EffectorCommand] = {}
        for side, raw_qpos in qpos_by_side.items():
            normalized = normalize_fix_arm_side(side)
            if normalized not in self.active_sides:
                raise ValueError(
                    f"runtime Home snapshot contains inactive side {normalized!r}"
                )
            qpos = np.asarray(raw_qpos, dtype=float).reshape(-1)
            if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
                raise ValueError(
                    f"runtime Home snapshot for {normalized} must be finite qpos[6]"
                )
            snapshots[normalized] = EffectorCommand(
                effector_name=f"{normalized}_arm",
                side=normalized,
                command_type="joint_vector",
                value=qpos.copy(),
                frame_name="arm_base",
                metadata={
                    "source": "atomic_runtime_home_snapshot",
                    "runtime_home_epoch": int(epoch),
                },
            )
        missing = tuple(side for side in self.active_sides if side not in snapshots)
        if missing:
            raise ValueError(
                "runtime Home snapshot is missing active side(s): " + ", ".join(missing)
            )
        self._runtime_home_epoch = int(epoch)
        self._runtime_home_arm_by_side = snapshots

    def invalidate_runtime_home_arm_snapshot(self) -> None:
        """Prevent a later startup Fix from reusing an older Home epoch."""

        self._runtime_home_epoch = None
        self._runtime_home_arm_by_side = {}

    def freeze_selected_from_runtime_home(
        self,
        *,
        now: float,
        expected_epoch: int,
    ) -> FixArmTransition:
        """Establish an arm-only Fix from the current admitted runtime Home."""

        del now
        side = self._selected_side
        if side in self._fixed_sides:
            return FixArmTransition(False, f"{side} is already fixed", side, self._mode)
        if self.scope_for_side(side) != "arm_only":
            return FixArmTransition(
                False,
                f"runtime Home startup Fix requires arm_only scope for {side}",
                side,
                self._mode,
            )
        if self._runtime_home_epoch is None or int(expected_epoch) != self._runtime_home_epoch:
            return FixArmTransition(
                False,
                f"runtime Home snapshot epoch mismatch for {side}: "
                f"cached={self._runtime_home_epoch!r} expected={int(expected_epoch)!r}",
                side,
                self._mode,
            )
        arm_name = f"{side}_arm"
        snapshot = self._runtime_home_arm_by_side.get(side)
        if snapshot is None:
            return FixArmTransition(
                False,
                f"runtime Home arm snapshot is unavailable for {side}",
                side,
                self._mode,
                missing_effectors=(arm_name,),
            )
        fixed = snapshot.copy()
        fixed.metadata = {**dict(fixed.metadata), "fix_arm_snapshot_origin": "runtime_home"}
        self._latest_by_effector[arm_name] = fixed.copy()
        self._fixed_by_effector[arm_name] = fixed
        self._fixed_by_channel = {
            name: value
            for name, value in self._fixed_by_channel.items()
            if self._side_for_channel(name, value) != side
        }
        self._fixed_side = side
        self._fixed_sides.add(side)
        self._mode = "fixed"
        self._recalibration = _RecalibrationState()
        warning = self.warning_for_side(side)
        self._status_message = (
            f"{side} fixed from runtime Home epoch {self._runtime_home_epoch}"
            if not warning
            else f"{side} fixed from runtime Home epoch {self._runtime_home_epoch}; warning: {warning}"
        )
        return FixArmTransition(True, self._status_message, side, self._mode)

    def freeze_arm_only_from_command(
        self,
        *,
        side: str,
        command: EffectorCommand,
        now: float,
        snapshot_origin: str,
    ) -> FixArmTransition:
        """Establish an arm-only Fix from a verified, arrived command."""

        normalized = normalize_fix_arm_side(side)
        arm_name = f"{normalized}_arm"
        origin = str(snapshot_origin or "").strip()
        if self.recapturing:
            return FixArmTransition(
                False,
                "arrived-command Fix cannot start while Home recapture is active",
                normalized,
                self._mode,
            )
        if normalized not in self.active_sides:
            return FixArmTransition(False, f"inactive arm side {normalized}", normalized, self._mode)
        if normalized in self._fixed_sides:
            return FixArmTransition(False, f"{normalized} is already fixed", normalized, self._mode)
        if self.scope_for_side(normalized) != "arm_only":
            return FixArmTransition(
                False,
                f"arrived-command Fix requires arm_only scope for {normalized}",
                normalized,
                self._mode,
            )
        if not isinstance(command, EffectorCommand):
            return FixArmTransition(
                False,
                f"arrived-command Fix requires an EffectorCommand for {arm_name}",
                normalized,
                self._mode,
                missing_effectors=(arm_name,),
            )
        if (
            command.effector_name != arm_name
            or command.side != normalized
            or command.command_type != "joint_vector"
        ):
            return FixArmTransition(
                False,
                f"arrived-command Fix requires {arm_name} joint_vector; "
                f"got name={command.effector_name!r} side={command.side!r} "
                f"type={command.command_type!r}",
                normalized,
                self._mode,
                missing_effectors=(arm_name,),
            )
        try:
            qpos = np.asarray(command.value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            qpos = np.asarray([], dtype=float)
        if qpos.shape != (6,) or not np.all(np.isfinite(qpos)):
            return FixArmTransition(
                False,
                f"arrived-command Fix requires finite {arm_name} qpos[6]",
                normalized,
                self._mode,
                missing_effectors=(arm_name,),
            )
        if not origin:
            return FixArmTransition(
                False,
                "arrived-command Fix requires a non-empty snapshot origin",
                normalized,
                self._mode,
            )

        fixed = command.copy()
        fixed.value = qpos.copy()
        fixed.metadata = {
            **dict(fixed.metadata),
            "fix_arm_snapshot_origin": origin,
            "fix_arm_snapshot_captured_at": float(now),
        }
        self._latest_by_effector[arm_name] = fixed.copy()
        self._fixed_by_effector[arm_name] = fixed
        self._fixed_by_channel = {
            name: value
            for name, value in self._fixed_by_channel.items()
            if self._side_for_channel(name, value) != normalized
        }
        self._selected_side = normalized
        self._fixed_side = normalized
        self._fixed_sides.add(normalized)
        self._mode = "fixed"
        self._recalibration = _RecalibrationState()
        warning = self.warning_for_side(normalized)
        self._status_message = (
            f"{normalized} fixed from arrived command ({origin})"
            if not warning
            else f"{normalized} fixed from arrived command ({origin}); warning: {warning}"
        )
        return FixArmTransition(True, self._status_message, normalized, self._mode)

    def freeze_selected(self, *, now: float) -> FixArmTransition:
        del now
        side = self._selected_side
        if side in self._fixed_sides:
            return FixArmTransition(
                ok=False,
                message=f"{side} is already fixed",
                side=side,
                mode=self._mode,
            )
        required = self.required_effectors_by_side.get(side, ())
        missing = tuple(name for name in required if name not in self._latest_by_effector)
        required_channels = self._required_channels_for_side(side)
        missing_channels = tuple(
            name
            for name in required_channels
            if name not in self._latest_by_channel
            or not bool(getattr(self._latest_by_channel[name], "valid", False))
        )
        if missing:
            self._status_message = f"cannot fix {side}; missing last command for {', '.join(missing)}"
            return FixArmTransition(
                ok=False,
                message=self._status_message,
                side=side,
                mode=self._mode,
                missing_effectors=missing,
            )
        if missing_channels:
            self._status_message = (
                f"cannot fix {side}; missing valid input channel for "
                f"{', '.join(missing_channels)}"
            )
            return FixArmTransition(
                ok=False,
                message=self._status_message,
                side=side,
                mode=self._mode,
                missing_channels=missing_channels,
            )
        self._fixed_side = side
        self._fixed_sides.add(side)
        self._fixed_by_effector.update(
            {
                name: self._latest_by_effector[name].copy()
                for name in required
                if name in self._latest_by_effector
            }
        )
        self._fixed_by_channel.update(
            {
                name: self._latest_by_channel[name].copy()
                for name in required_channels
                if name in self._latest_by_channel
            }
        )
        self._mode = "fixed"
        self._recalibration = _RecalibrationState()
        warning = self.warning_for_side(side)
        self._status_message = f"{side} fixed" if not warning else f"{side} fixed; warning: {warning}"
        return FixArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def start_recalibration(
        self,
        *,
        now: float,
        timeout_sec: float,
        min_samples: int,
    ) -> FixArmTransition:
        side = self._selected_side
        if side not in self._fixed_sides:
            return FixArmTransition(
                ok=False,
                message=f"fix-arm recalibration requires {side} to be fixed",
                side=side,
                mode=self._mode,
            )
        if self.recapturing:
            return FixArmTransition(
                ok=False,
                message=f"{self._recapturing_side} Home recapture is already active",
                side=self._recapturing_side,
                mode=self._mode,
            )
        self._fixed_side = side
        self._recapturing_side = side
        self._post_recalibration_fixed_side = None
        self._two_pedal_transition_pending = False
        self._mode = "recapturing"
        self._recalibration = _RecalibrationState(
            started_at=float(now),
            timeout_sec=max(0.1, float(timeout_sec)),
            min_samples=max(1, int(min_samples)),
        )
        self._status_message = f"recapturing {side} home"
        return FixArmTransition(
            ok=True,
            message=self._status_message,
            side=side,
            mode=self._mode,
        )

    def start_two_pedal_transition(
        self,
        *,
        fixed_side: str | None,
        now: float,
        timeout_sec: float = 1.0,
        min_samples: int = 1,
        allow_fresh_home_rebase: bool = False,
    ) -> FixArmTransition:
        """Prepare one atomic two-pedal Fix transition.

        The currently fixed side remains fixed while its Home is refreshed.
        Once that refresh is applied, ``complete_recalibration`` replaces the
        fixed-side set in one mutation, so collection can never observe both
        sides fixed during a left-only/right-only switch.
        """

        target = None if fixed_side is None else normalize_fix_arm_side(fixed_side)
        current = self.fixed_side
        if not current:
            return FixArmTransition(
                ok=False,
                message="two-pedal transition requires one currently fixed side",
                side="",
                mode=self._mode,
            )

        # A newer pedal event owns the requested destination.  If it asks to
        # remain in the current single-arm state, cancel any older pending
        # transition instead of allowing that stale target to commit later.
        if target == current:
            if self.recapturing:
                self._recapturing_side = ""
                self._post_recalibration_fixed_side = None
                self._two_pedal_transition_pending = False
                self._selected_side = current
                self._fixed_side = current
                self._mode = "fixed"
                self._recalibration = _RecalibrationState()
                self._status_message = (
                    f"two-pedal transition cancelled; {current} remains fixed"
                )
                return FixArmTransition(
                    ok=True,
                    message=self._status_message,
                    side=current,
                    mode=self._mode,
                )
            return FixArmTransition(
                ok=True,
                message=f"{current} is already the two-pedal fixed side",
                side=current,
                mode=self._mode,
            )
        if target is not None:
            required = self.required_effectors_by_side.get(target, ())
            missing = tuple(name for name in required if name not in self._latest_by_effector)
            required_channels = self._required_channels_for_side(target)
            missing_channels = tuple(
                name
                for name in required_channels
                if name not in self._latest_by_channel
                or not bool(getattr(self._latest_by_channel[name], "valid", False))
            )
            if missing or missing_channels:
                if self.recapturing:
                    # The newest pedal event supersedes the older pending
                    # destination even when its own target cannot be prepared.
                    # Remain in the current single-arm state; never allow the
                    # stale older destination to commit later.
                    self._recapturing_side = ""
                    self._post_recalibration_fixed_side = None
                    self._two_pedal_transition_pending = False
                    self._selected_side = current
                    self._fixed_side = current
                    self._mode = "fixed"
                    self._recalibration = _RecalibrationState()
                return FixArmTransition(
                    ok=False,
                    message=f"cannot prepare atomic switch to {target}",
                    side=current,
                    mode=self._mode,
                    missing_effectors=missing,
                    missing_channels=missing_channels,
                )

        # Do not discard a new pedal/B request while the current fixed-side
        # Home is being refreshed.  Reuse the in-flight stable sample window
        # and atomically commit only the newest requested destination.
        if self.recapturing:
            self._selected_side = current
            self._post_recalibration_fixed_side = target
            self._two_pedal_transition_pending = True
            self._recalibration.started_at = float(now)
            self._recalibration.timeout_sec = min(
                1.0,
                max(0.1, float(timeout_sec)),
            )
            self._recalibration.min_samples = max(1, int(min_samples))
            target_text = target or "live"
            self._status_message = (
                f"updated two-pedal target to {target_text}; "
                f"waiting for stable {current} Home"
            )
            return FixArmTransition(
                ok=True,
                message=self._status_message,
                side=current,
                mode=self._mode,
            )

        # GUI selection is display/input state and may legitimately differ
        # from the side that is actually fixed.  Home must always be captured
        # for the real fixed side, otherwise the transition is permanently
        # rejected after a selection-only change.
        self._selected_side = current
        transition = self.start_recalibration(
            now=now,
            timeout_sec=min(1.0, max(0.1, float(timeout_sec))),
            min_samples=max(1, int(min_samples)),
        )
        if transition.ok:
            # Two-pedal handoffs use a bounded transition profile.  Regular
            # C/Home recapture keeps the strict default profile.
            self._recalibration.stability = (
                HomeCaptureStabilityState.transition_profile()
            )
            self._post_recalibration_fixed_side = target
            self._two_pedal_transition_pending = True
            cache = self._passive_home_cache.get(current)
            cache_status = self.passive_home_cache_status(
                current,
                now=now,
                allow_fresh_rebase=allow_fresh_home_rebase,
            )
            seed_pose: np.ndarray | None = None
            seed_timestamp: float | None = None
            seed_quality = str(cache_status["quality"])
            if cache is not None and bool(cache_status["stable_ready"]):
                seed_pose = None if cache.pose is None else cache.pose.copy()
                seed_timestamp = float(cache.source_timestamp)
            elif (
                cache is not None
                and allow_fresh_home_rebase
                and bool(cache_status["fresh_rebase_ready"])
                and cache.stability.last_pose is not None
                and cache.stability.last_timestamp is not None
            ):
                seed_pose = cache.stability.last_pose.copy()
                seed_timestamp = float(cache.stability.last_timestamp)
            if seed_pose is not None and seed_timestamp is not None:
                self._recalibration.samples = [seed_pose]
                if seed_quality == "stable":
                    # The passive cache has already passed the stillness dwell.
                    # Carry that state into the active capture rather than
                    # waiting through a duplicate dwell window.
                    self._recalibration.stability.prime_from_stable_sample(
                        seed_pose,
                        source_timestamp=seed_timestamp,
                        reason="stable_home_cache_reused",
                    )
                else:
                    # This is an instantaneous rebase, not a claim that the
                    # hand passed the ordinary stillness contract.  The sample
                    # is consumed immediately by the atomic progress handoff.
                    self._recalibration.stability.reason = (
                        "fresh_tracker_rebase_reused"
                    )
            target_text = target or "live"
            if self._recalibration.samples:
                self._status_message = (
                    f"fresh {seed_quality} {current} Home cache ready; "
                    f"committing atomic switch to {target_text}"
                )
            else:
                self._status_message = (
                    f"waiting for stable {current} Home before atomic switch to "
                    f"{target_text}; reason={cache_status['reason']}"
                )
            transition = FixArmTransition(
                ok=True,
                message=self._status_message,
                side=current,
                mode=self._mode,
            )
        return transition

    def observe_two_pedal_home_cache(
        self,
        side: str,
        pose_desk: np.ndarray,
        *,
        source_timestamp: float,
    ) -> bool:
        """Maintain stable per-side Home candidates throughout single-arm mode.

        Both trackers remain live while exactly one robot arm is fixed.  Cache
        both sides so the arm that becomes fixed after an atomic switch already
        has a recent candidate if the operator immediately switches again.
        These passive candidates never change an active retargeter Home; only
        the side being released is applied by the transition path.
        """

        normalized = normalize_fix_arm_side(side)
        if (
            normalized not in self.active_sides
            or not self._fixed_sides
            or self.recapturing
        ):
            return False
        cache = self._passive_home_cache[normalized]
        cache.last_observed_at = float(source_timestamp)
        decision = cache.stability.observe(
            pose_desk,
            source_timestamp=float(source_timestamp),
            first_accepted_pose=cache.pose,
        )
        if decision.reset_samples:
            cache.pose = None
            cache.source_timestamp = 0.0
            cache.unavailable_reason = str(decision.reason)
        if decision.accepted:
            cache.pose = np.asarray(pose_desk, dtype=float).reshape(4, 4).copy()
            cache.source_timestamp = float(source_timestamp)
            cache.unavailable_reason = "stable_home_cache_ready"
            return True
        cache.unavailable_reason = str(decision.reason)
        return False

    def add_recalibration_sample(self, pose_desk: np.ndarray) -> int:
        if not self.recapturing:
            return self.sample_count
        if self.sample_count >= self.min_samples:
            return self.sample_count
        self._recalibration.samples.append(np.asarray(pose_desk, dtype=float).reshape(4, 4).copy())
        return self.sample_count

    def observe_recalibration_sample(
        self,
        pose_desk: np.ndarray,
        *,
        source_timestamp: float,
    ) -> int:
        """Accept only fresh, continuously still tracker samples."""

        if not self.recapturing:
            return self.sample_count
        first_pose = (
            self._recalibration.samples[0]
            if self._recalibration.samples
            else None
        )
        decision = self._recalibration.stability.observe(
            pose_desk,
            source_timestamp=source_timestamp,
            first_accepted_pose=first_pose,
        )
        if decision.reset_samples:
            self._recalibration.samples.clear()
        if decision.accepted:
            self.add_recalibration_sample(pose_desk)
        return self.sample_count

    def complete_recalibration(self, *, now: float, message: str = "") -> FixArmTransition:
        del now
        side = self._recapturing_side or self._fixed_side
        target = self._post_recalibration_fixed_side
        if not self._two_pedal_transition_pending:
            self._fixed_sides.discard(side)
            for name in self.required_effectors_by_side.get(side, ()):
                self._fixed_by_effector.pop(name, None)
            for name in self._required_channels_for_side(side):
                self._fixed_by_channel.pop(name, None)
            self._recapturing_side = ""
            self._fixed_side = next(iter(self.fixed_sides), "")
            self._mode = "fixed" if self._fixed_sides else "live"
            self._recalibration = _RecalibrationState()
            self._status_message = message or f"{side} home recaptured; fix-arm released"
            return FixArmTransition(
                ok=True,
                message=self._status_message,
                side=side,
                mode=self._mode,
            )
        replacement_effectors: dict[str, EffectorCommand] = {}
        replacement_channels: dict[str, ObservationChannel] = {}
        if target is not None:
            replacement_effectors = {
                name: self._latest_by_effector[name].copy()
                for name in self.required_effectors_by_side.get(target, ())
                if name in self._latest_by_effector
            }
            replacement_channels = {
                name: self._latest_by_channel[name].copy()
                for name in self._required_channels_for_side(target)
                if name in self._latest_by_channel
            }
        self._fixed_sides = set() if target is None else {target}
        self._fixed_by_effector = replacement_effectors
        self._fixed_by_channel = replacement_channels
        self._recapturing_side = ""
        self._post_recalibration_fixed_side = None
        self._two_pedal_transition_pending = False
        self._fixed_side = target or ""
        self._selected_side = target or side
        self._mode = "fixed" if self._fixed_sides else "live"
        self._recalibration = _RecalibrationState()
        self._status_message = message or f"{side} home recaptured; fix-arm released"
        return FixArmTransition(ok=True, message=self._status_message, side=side, mode=self._mode)

    def cancel_active(
        self,
        *,
        now: float,
        reason: str = "",
        clear_snapshots: bool = False,
    ) -> FixArmTransition:
        del now
        side = self._fixed_side
        was_active = bool(self.fixed)
        self._mode = "live"
        self._fixed_side = ""
        self._fixed_sides.clear()
        self._recapturing_side = ""
        self._post_recalibration_fixed_side = None
        self._two_pedal_transition_pending = False
        self._fixed_by_effector = {}
        self._fixed_by_channel = {}
        self._recalibration = _RecalibrationState()
        if clear_snapshots:
            self._latest_by_effector = {}
            self._latest_by_channel = {}
            self.invalidate_runtime_home_arm_snapshot()
        if was_active:
            suffix = f": {reason}" if reason else ""
            self._status_message = f"{side} fix-arm stopped{suffix}"
        else:
            self._status_message = "fix-arm ready"
        return FixArmTransition(
            ok=was_active,
            message=self._status_message,
            side=side,
            mode=self._mode,
        )

    def fail_recalibration(self, *, now: float, message: str) -> FixArmTransition:
        del now
        side = self._recapturing_side or self._fixed_side
        self._recapturing_side = ""
        self._post_recalibration_fixed_side = None
        self._two_pedal_transition_pending = False
        self._mode = "fixed" if self._fixed_sides else "live"
        self._recalibration = _RecalibrationState()
        self._status_message = str(message)
        return FixArmTransition(ok=False, message=self._status_message, side=side, mode=self._mode)

    def filter_channels(
        self,
        channels: Mapping[str, ObservationChannel],
        now: float,
    ) -> dict[str, ObservationChannel]:
        fixed_sides = set(self.fixed_sides)
        fixed_channel_names = {
            name
            for side in fixed_sides
            for name in self._required_channels_for_side(side)
        }
        filtered = {name: channel.copy() for name, channel in channels.items()}
        for channel_name, channel in channels.items():
            if channel_name in fixed_channel_names:
                continue
            if self._side_for_channel(channel_name, channel) in {"left", "right"}:
                self._latest_by_channel[channel_name] = channel.copy()
        for fixed_side in self.fixed_sides:
            for channel_name in self._required_channels_for_side(fixed_side):
                frozen = self._fixed_by_channel.get(channel_name)
                if frozen is None:
                    continue
                next_channel = frozen.copy()
                original_timestamp = float(next_channel.timestamp)
                next_channel.timestamp = float(now)
                next_channel.valid = True
                next_channel.metadata = {
                    **dict(next_channel.metadata),
                    "fix_arm_frozen": True,
                    "fix_arm_fixed_side": fixed_side,
                    "fix_arm_original_timestamp": original_timestamp,
                }
                filtered[channel_name] = next_channel
        return filtered

    def filter_command(
        self,
        command: TeleopCommand,
        _observation,
        _now: float,
        *,
        include_missing_fixed_effectors: bool = False,
        allowed_effectors: Sequence[str] | None = None,
    ) -> TeleopCommand:
        fixed_sides = set(self.fixed_sides)
        allowed = (
            None
            if allowed_effectors is None
            else {str(name) for name in allowed_effectors if str(name)}
        )
        filtered = TeleopCommand(timestamp=command.timestamp, metadata=dict(command.metadata))
        for effector_name, effector_command in command.commands.items():
            next_command = effector_command.copy()
            filtered.commands[effector_name] = next_command
            if not any(
                effector_name in self.required_effectors_by_side.get(side, ())
                for side in fixed_sides
            ):
                self._latest_by_effector[effector_name] = next_command.copy()
        for fixed_side in self.fixed_sides:
            for effector_name in self.required_effectors_by_side.get(fixed_side, ()):
                if allowed is not None and effector_name not in allowed:
                    continue
                if (
                    not include_missing_fixed_effectors
                    and effector_name not in filtered.commands
                ):
                    continue
                fixed_command = self._fixed_by_effector.get(effector_name)
                if fixed_command is None:
                    continue
                filtered_command = fixed_command.copy()
                filtered_command.metadata = {
                    **dict(filtered_command.metadata),
                    "fix_arm_active": True,
                    "fix_arm_side": fixed_side,
                    "fix_arm_mode": self._mode,
                    "fix_arm_scope": self.scope_for_side(fixed_side),
                    "fix_arm_warning": self.warning_for_side(fixed_side),
                    "fix_arm_effectors": list(self.required_effectors_by_side.get(fixed_side, ())),
                }
                filtered.commands[effector_name] = filtered_command
                self._latest_by_effector[effector_name] = filtered_command.copy()
        if fixed_sides:
            primary_side = self.fixed_side or self.fixed_sides[0]
            filtered.metadata.update(
                {
                    "fix_arm_active_side": primary_side,
                    "fix_arm_active_sides": list(self.fixed_sides),
                    "fix_arm_mode": self._mode,
                    # Keep the historical singular fields for downstream
                    # consumers while exposing the complete per-side state.
                    "fix_arm_scope": self.scope_for_side(primary_side),
                    "fix_arm_warning": self.warning_for_side(primary_side),
                    "fix_arm_effectors": list(
                        self.required_effectors_by_side.get(primary_side, ())
                    ),
                }
            )
        return filtered

    def filter_faults(
        self,
        faults: Sequence[FaultEvent],
        _channels,
        _now: float,
    ) -> list[FaultEvent]:
        if not self.fixed:
            return list(faults)
        fixed_channels = {
            name
            for side in self.fixed_sides
            for name in self._required_channels_for_side(side)
        }
        filtered: list[FaultEvent] = []
        for fault in faults:
            if fault.channel_name in fixed_channels and fault.code in FIX_ARM_FAULT_CODES:
                continue
            filtered.append(fault)
        return filtered

    def status_payload(self) -> dict[str, object]:
        fixed_side = self.fixed_side
        return {
            "fix_arm_available": True,
            "fix_arm_selected_side": self._selected_side,
            "fix_arm_fixed_side": fixed_side,
            "fix_arm_fixed_sides": list(self.fixed_sides),
            "fix_arm_active": bool(fixed_side),
            "fix_arm_recalibrating": self.recapturing,
            "fix_arm_status": self._status_message,
            "fix_arm_scope": self.scope_for_side(fixed_side) if fixed_side else "",
            "fix_arm_warning": self.warning_for_side(fixed_side) if fixed_side else "",
            "fix_arm_effectors": list(self.required_effectors_by_side.get(fixed_side, ())),
            "fix_arm_sample_count": self.sample_count if self.recapturing else None,
            "fix_arm_min_samples": self.min_samples if self.recapturing else None,
            "fix_arm_home_cache": {
                side: self.passive_home_cache_status(side, now=time.time())
                for side in self.active_sides
            },
        }

    def _required_channels_for_side(self, side: str) -> tuple[str, ...]:
        if side not in {"left", "right"}:
            return ()
        channels = [f"{side}_wrist_pose"]
        if f"{side}_hand" in self.required_effectors_by_side.get(side, ()):
            channels.append(f"{side}_hand_q6")
        return tuple(channels)

    def _side_for_channel(self, channel_name: str, channel: ObservationChannel) -> str:
        if channel.side in {"left", "right"}:
            return str(channel.side)
        if channel_name.startswith("left_"):
            return "left"
        if channel_name.startswith("right_"):
            return "right"
        return ""
