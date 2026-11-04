from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from robot_motion.position_control import (
    POSITION_15_60_120_PROFILE,
    POSITION_45_150_PROFILE,
    POSITION_PROFILE,
    PositionControlRates,
)
from robot_motion.rm65 import RM65_POSITION_TRACKING_FOLLOW_ALPHA

from ..defaults import (
    DEFAULT_RM65_ARM_CONTROL_HZ,
    DEFAULT_RM65_POSITION_ARM_CONTROL_HZ,
    DEFAULT_TELEOP_POSITION_REFERENCE_HZ,
    DEFAULT_TELEOP_SAMPLE_HZ,
)


IMPEDANCE_40HZ_PROFILE_NAME = "impedance_40hz"
POSITION_45_150HZ_PROFILE_NAME = POSITION_45_150_PROFILE.name
POSITION_15_60_120HZ_PROFILE_NAME = POSITION_15_60_120_PROFILE.name
POSITION_PROFILE_NAME = POSITION_PROFILE.name
POSITION_PROFILES_BY_NAME = {
    POSITION_PROFILE_NAME: POSITION_PROFILE,
    POSITION_15_60_120HZ_PROFILE_NAME: POSITION_15_60_120_PROFILE,
    POSITION_45_150HZ_PROFILE_NAME: POSITION_45_150_PROFILE,
}
ARM_EXECUTION_PROFILE_CHOICES = (
    IMPEDANCE_40HZ_PROFILE_NAME,
    POSITION_PROFILE_NAME,
    POSITION_15_60_120HZ_PROFILE_NAME,
    POSITION_45_150HZ_PROFILE_NAME,
)

CONTROL_DEBUG_OFF = "off"
CONTROL_DEBUG_SUMMARY = "summary"
CONTROL_DEBUG_FULL = "full"
CONTROL_DEBUG_LEVELS = (
    CONTROL_DEBUG_OFF,
    CONTROL_DEBUG_SUMMARY,
    CONTROL_DEBUG_FULL,
)
DEFAULT_CONTROL_DEBUG_ROOT = Path("teleop_records/control_debug")
DEFAULT_CONTROL_DEBUG_QUEUE_SIZE = 8192
DEFAULT_CONTROL_DEBUG_FLUSH_EVERY = 30


def _is_option_present(raw_argv: Sequence[str], *options: str) -> bool:
    return any(
        str(item) == option or str(item).startswith(f"{option}=")
        for item in raw_argv
        for option in options
    )


def _finite_positive(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _finite_non_negative(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return resolved


@dataclass(frozen=True)
class ControlDebugConfig:
    """Purely observational control-debug request.

    This contract intentionally contains no safety, smoothing, hold, or control
    switches.  The asynchronous audit recorder consumes it, but resolving or
    enabling it must never change generated references or robot commands.
    """

    level: str = CONTROL_DEBUG_OFF
    root: str = str(DEFAULT_CONTROL_DEBUG_ROOT)
    queue_size: int = DEFAULT_CONTROL_DEBUG_QUEUE_SIZE
    flush_every: int = DEFAULT_CONTROL_DEBUG_FLUSH_EVERY

    def __post_init__(self) -> None:
        level = str(self.level).strip().lower()
        if level not in CONTROL_DEBUG_LEVELS:
            raise ValueError(
                f"control debug level must be one of {CONTROL_DEBUG_LEVELS}, got {self.level!r}."
            )
        root = str(self.root).strip()
        if not root:
            raise ValueError("control debug root must be non-empty.")
        queue_size = int(self.queue_size)
        flush_every = int(self.flush_every)
        if queue_size <= 0:
            raise ValueError("control debug queue_size must be positive.")
        if flush_every <= 0:
            raise ValueError("control debug flush_every must be positive.")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "queue_size", queue_size)
        object.__setattr__(self, "flush_every", flush_every)

    @property
    def enabled(self) -> bool:
        return self.level != CONTROL_DEBUG_OFF

    @property
    def record_summary(self) -> bool:
        return self.enabled

    @property
    def record_full_trace(self) -> bool:
        return self.level == CONTROL_DEBUG_FULL

    def to_metadata(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "enabled": self.enabled,
            "recording_requested": self.enabled,
            "record_summary": self.record_summary,
            "record_full_trace": self.record_full_trace,
            "root": self.root,
            "queue_size": self.queue_size,
            "flush_every": self.flush_every,
            "observational_only": True,
        }


@dataclass(frozen=True)
class RealArmExecutionConfig:
    requested_profile: str | None
    resolved_profile: str
    mode: str
    sample_hz: float
    outer_control_hz: float
    position_reference_hz: float
    dual_arm_ik_hz: float
    arm_control_hz: float
    arm_canfd_follow: bool
    arm_canfd_follow_stream_hz: float | None
    position_background_streaming_enabled: bool
    position_background_streaming_hz: float
    position_post_ik_qpos_smoothing_alpha: float
    arm_follow_qpos_alpha: float
    arm_canfd_follow_warning_ratio: float
    arm_canfd_trajectory_mode: int
    arm_canfd_radio: int
    position_reference_max_age_sec: float
    explicit_overrides: tuple[str, ...] = ()
    profile_deviations: tuple[str, ...] = ()
    compatibility_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requested_profile = (
            None
            if self.requested_profile in (None, "")
            else str(self.requested_profile).strip()
        )
        resolved_profile = str(self.resolved_profile).strip()
        if not resolved_profile:
            raise ValueError("resolved_profile must be non-empty.")
        object.__setattr__(self, "requested_profile", requested_profile)
        object.__setattr__(self, "resolved_profile", resolved_profile)
        if self.mode not in {"impedance", "position"}:
            raise ValueError("mode must be 'impedance' or 'position'.")
        for field_name in (
            "sample_hz",
            "outer_control_hz",
            "position_reference_hz",
            "dual_arm_ik_hz",
            "arm_control_hz",
            "position_background_streaming_hz",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_positive(getattr(self, field_name), name=field_name),
            )
        object.__setattr__(
            self,
            "position_reference_max_age_sec",
            _finite_non_negative(
                self.position_reference_max_age_sec,
                name="position_reference_max_age_sec",
            ),
        )
        stream_hz = self.arm_canfd_follow_stream_hz
        if stream_hz is not None:
            stream_hz = _finite_non_negative(
                stream_hz,
                name="arm_canfd_follow_stream_hz",
            )
        object.__setattr__(self, "arm_canfd_follow_stream_hz", stream_hz)
        post_ik_alpha = float(self.position_post_ik_qpos_smoothing_alpha)
        alpha = float(self.arm_follow_qpos_alpha)
        warning_ratio = float(self.arm_canfd_follow_warning_ratio)
        trajectory_mode = int(self.arm_canfd_trajectory_mode)
        radio = int(self.arm_canfd_radio)
        if not 0.0 < post_ik_alpha <= 1.0:
            raise ValueError("position_post_ik_qpos_smoothing_alpha must be in (0, 1].")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("arm_follow_qpos_alpha must be in (0, 1].")
        if not 0.0 < warning_ratio <= 1.0:
            raise ValueError("arm_canfd_follow_warning_ratio must be in (0, 1].")
        if trajectory_mode not in (0, 1, 2):
            raise ValueError("arm_canfd_trajectory_mode must be 0, 1, or 2.")
        if not 0 <= radio <= 100:
            raise ValueError("arm_canfd_radio must be in [0, 100].")
        object.__setattr__(self, "arm_canfd_follow", bool(self.arm_canfd_follow))
        object.__setattr__(
            self,
            "position_background_streaming_enabled",
            bool(self.position_background_streaming_enabled),
        )
        object.__setattr__(
            self,
            "position_post_ik_qpos_smoothing_alpha",
            post_ik_alpha,
        )
        object.__setattr__(self, "arm_follow_qpos_alpha", alpha)
        object.__setattr__(self, "arm_canfd_follow_warning_ratio", warning_ratio)
        object.__setattr__(self, "arm_canfd_trajectory_mode", trajectory_mode)
        object.__setattr__(self, "arm_canfd_radio", radio)
        object.__setattr__(
            self,
            "explicit_overrides",
            tuple(str(value) for value in self.explicit_overrides),
        )
        object.__setattr__(
            self,
            "profile_deviations",
            tuple(str(value) for value in self.profile_deviations),
        )
        object.__setattr__(
            self,
            "compatibility_warnings",
            tuple(str(value) for value in self.compatibility_warnings),
        )

    @property
    def position_rates(self) -> PositionControlRates | None:
        if self.mode != "position":
            return None
        return PositionControlRates(
            sample_hz=self.sample_hz,
            reference_hz=self.position_reference_hz,
            ik_hz=self.dual_arm_ik_hz,
            command_hz=self.position_background_streaming_hz,
        )

    def to_metadata(self) -> dict[str, Any]:
        rates = {
            "sample_hz": self.sample_hz,
            "outer_control_hz": self.outer_control_hz,
            "position_reference_hz": self.position_reference_hz,
            "dual_arm_ik_hz": self.dual_arm_ik_hz,
            "arm_control_hz": self.arm_control_hz,
            "position_background_streaming_hz": self.position_background_streaming_hz,
        }
        position_rates = self.position_rates
        return {
            "requested_profile": self.requested_profile,
            "resolved_profile": self.resolved_profile,
            "mode": self.mode,
            "rates": rates,
            "rate_semantics": {
                "sample_hz": "dataset_sample_clock",
                "outer_control_hz": "reference_and_ik_call_clock",
                "position_reference_hz": "position_outer_reference_clock",
                "dual_arm_ik_hz": "solver_nominal_budget_not_independent_scheduler",
                "arm_control_hz": "backend_command_clock",
            },
            "position_rates": None if position_rates is None else position_rates.to_metadata(),
            "arm_canfd_follow": self.arm_canfd_follow,
            "arm_canfd_follow_stream_hz_requested": self.arm_canfd_follow_stream_hz,
            "position_background_streaming_enabled": self.position_background_streaming_enabled,
            "position_post_ik_qpos_smoothing_alpha": (
                self.position_post_ik_qpos_smoothing_alpha
            ),
            "arm_follow_qpos_alpha": self.arm_follow_qpos_alpha,
            "arm_canfd_follow_warning_ratio": self.arm_canfd_follow_warning_ratio,
            "arm_canfd_trajectory_mode": self.arm_canfd_trajectory_mode,
            "arm_canfd_radio": self.arm_canfd_radio,
            "position_reference_max_age_sec": self.position_reference_max_age_sec,
            "explicit_overrides": list(self.explicit_overrides),
            "profile_deviations": list(self.profile_deviations),
            "compatibility_warnings": list(self.compatibility_warnings),
        }


def _profile_mode(profile: str) -> str:
    if profile == IMPEDANCE_40HZ_PROFILE_NAME:
        return "impedance"
    if profile in POSITION_PROFILES_BY_NAME:
        return "position"
    raise ValueError(
        f"arm execution profile must be one of {ARM_EXECUTION_PROFILE_CHOICES}, got {profile!r}."
    )


def _changed(label: str, actual: float | int | bool, expected: float | int | bool) -> str | None:
    if isinstance(actual, float) or isinstance(expected, float):
        return None if math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-9) else label
    return None if actual == expected else label


def resolve_real_arm_execution_config(
    args: Any,
    *,
    raw_argv: Sequence[str] = (),
    impedance_default_outer_control_hz: float | None = None,
) -> RealArmExecutionConfig:
    requested_profile_value = getattr(args, "arm_execution_profile", None)
    requested_profile = (
        None
        if requested_profile_value in (None, "")
        else str(requested_profile_value).strip()
    )
    requested_position_profile = POSITION_PROFILES_BY_NAME.get(requested_profile)
    mode_explicit = _is_option_present(raw_argv, "--arm-control-mode")
    raw_mode = str(getattr(args, "arm_control_mode", "impedance"))
    if requested_profile is not None:
        expected_mode = _profile_mode(requested_profile)
        if mode_explicit and raw_mode != expected_mode:
            raise ValueError(
                f"--arm-execution-profile {requested_profile} requires "
                f"--arm-control-mode {expected_mode}, got {raw_mode}."
            )
        mode = expected_mode
    else:
        mode = raw_mode
    if mode not in {"impedance", "position"}:
        raise ValueError(f"Unsupported arm control mode: {mode!r}.")
    position_profile_metadata = getattr(
        args,
        "position_control_profile_metadata",
        None,
    )
    position_source_shaping_applied = bool(
        isinstance(position_profile_metadata, Mapping)
        and position_profile_metadata.get("source_shaping_applied", False)
    )
    position_source_shaping_metadata = (
        position_profile_metadata.get("source_shaping")
        if isinstance(position_profile_metadata, Mapping)
        else None
    )
    profile_post_ik_alpha: float | None = None
    if isinstance(position_source_shaping_metadata, Mapping):
        profile_post_ik_alpha_value = position_source_shaping_metadata.get(
            "post_ik_qpos_smoothing_alpha"
        )
        if profile_post_ik_alpha_value is not None:
            profile_post_ik_alpha = float(profile_post_ik_alpha_value)
    default_impedance_outer_control_hz = (
        DEFAULT_TELEOP_SAMPLE_HZ
        if impedance_default_outer_control_hz is None
        else _finite_positive(
            impedance_default_outer_control_hz,
            name="impedance_default_outer_control_hz",
        )
    )

    explicit_labels: list[str] = []
    if requested_profile is not None:
        explicit_labels.append("arm_execution_profile")
    if mode_explicit:
        explicit_labels.append("arm_control_mode")

    def explicit(label: str, *options: str) -> bool:
        present = _is_option_present(raw_argv, *options)
        if present:
            explicit_labels.append(label)
        return present

    sample_explicit = explicit("sample_hz", "--hz")
    impedance_control_explicit = explicit(
        "impedance_teleop_control_hz",
        "--impedance-teleop-control-hz",
    )
    reference_explicit = explicit("position_reference_hz", "--position-reference-hz")
    ik_explicit = explicit("dual_arm_ik_hz", "--dual-arm-ik-hz")
    control_explicit = explicit("arm_control_hz", "--arm-control-hz")
    # The bare calls below only record explicit overrides; the resolved values
    # honor any non-None argument, including formal-profile injected defaults.
    explicit("arm_canfd_follow", "--arm-canfd-follow", "--no-arm-canfd-follow")
    stream_explicit = explicit("arm_canfd_follow_stream_hz", "--arm-canfd-follow-stream-hz")
    post_ik_alpha_explicit = explicit(
        "position_post_ik_qpos_smoothing_alpha",
        "--position-post-ik-qpos-smoothing-alpha",
    )
    explicit("arm_follow_qpos_alpha", "--arm-follow-qpos-alpha")
    warning_explicit = explicit("arm_canfd_follow_warning_ratio", "--arm-canfd-follow-warning-ratio")
    trajectory_explicit = explicit("arm_canfd_trajectory_mode", "--arm-canfd-trajectory-mode")
    radio_explicit = explicit("arm_canfd_radio", "--arm-canfd-radio")
    explicit("position_reference_max_age_sec", "--position-reference-max-age-sec")

    sample_hz = _finite_positive(
        getattr(args, "hz", DEFAULT_TELEOP_SAMPLE_HZ),
        name="sample_hz",
    )
    position_reference_hz = _finite_positive(
        getattr(args, "position_reference_hz", DEFAULT_TELEOP_POSITION_REFERENCE_HZ),
        name="position_reference_hz",
    )
    if requested_position_profile is not None:
        if not sample_explicit:
            sample_hz = requested_position_profile.rates.sample_hz
        if not reference_explicit:
            position_reference_hz = requested_position_profile.rates.reference_hz
    elif requested_profile == IMPEDANCE_40HZ_PROFILE_NAME and not sample_explicit:
        sample_hz = DEFAULT_TELEOP_SAMPLE_HZ

    if mode == "position":
        outer_control_hz = position_reference_hz
    else:
        raw_impedance_control_hz = getattr(args, "impedance_teleop_control_hz", None)
        if impedance_control_explicit and raw_impedance_control_hz is not None:
            outer_control_hz = _finite_positive(
                raw_impedance_control_hz,
                name="impedance_teleop_control_hz",
            )
        else:
            outer_control_hz = default_impedance_outer_control_hz
    raw_ik_hz = getattr(args, "dual_arm_ik_hz", None)
    if ik_explicit and raw_ik_hz is not None:
        dual_arm_ik_hz = _finite_positive(raw_ik_hz, name="dual_arm_ik_hz")
    else:
        dual_arm_ik_hz = outer_control_hz

    raw_control_hz = getattr(args, "arm_control_hz", None)
    if control_explicit and raw_control_hz is not None:
        arm_control_hz = _finite_positive(raw_control_hz, name="arm_control_hz")
    elif requested_position_profile is not None:
        arm_control_hz = requested_position_profile.rates.command_hz
    elif requested_profile == IMPEDANCE_40HZ_PROFILE_NAME:
        arm_control_hz = DEFAULT_RM65_ARM_CONTROL_HZ
    elif raw_control_hz is not None:
        arm_control_hz = _finite_positive(raw_control_hz, name="arm_control_hz")
    else:
        arm_control_hz = (
            DEFAULT_RM65_POSITION_ARM_CONTROL_HZ
            if mode == "position"
            else DEFAULT_RM65_ARM_CONTROL_HZ
        )

    raw_follow = getattr(args, "arm_canfd_follow", None)
    if raw_follow is not None:
        arm_canfd_follow = bool(raw_follow)
    elif requested_position_profile is not None:
        arm_canfd_follow = True
    elif requested_profile == IMPEDANCE_40HZ_PROFILE_NAME:
        arm_canfd_follow = False
    else:
        arm_canfd_follow = mode == "position"

    raw_stream_hz = getattr(args, "arm_canfd_follow_stream_hz", None)
    stream_hz = (
        None
        if raw_stream_hz is None
        else _finite_non_negative(raw_stream_hz, name="arm_canfd_follow_stream_hz")
    )
    position_background_streaming_enabled = bool(
        mode == "position"
        and arm_canfd_follow
        and (stream_hz is None or stream_hz > 0.0)
    )
    position_background_streaming_hz = (
        arm_control_hz if stream_hz is None or stream_hz <= 0.0 else stream_hz
    )

    raw_post_ik_alpha = getattr(args, "position_post_ik_qpos_smoothing_alpha", None)
    position_post_ik_qpos_smoothing_alpha = (
        float(raw_post_ik_alpha)
        if (
            mode == "position"
            and position_background_streaming_enabled
            and (
                post_ik_alpha_explicit
                or position_source_shaping_applied
            )
            and raw_post_ik_alpha is not None
        )
        else 0.9
        if mode == "position" and position_background_streaming_enabled
        else 1.0
    )
    raw_alpha = getattr(args, "arm_follow_qpos_alpha", None)
    arm_follow_qpos_alpha = (
        (
            float(raw_alpha)
            if raw_alpha is not None
            else 0.1
        )
        if mode == "position" and position_background_streaming_enabled
        else 1.0
    )
    raw_warning_ratio = getattr(args, "arm_canfd_follow_warning_ratio", None)
    warning_ratio = (
        float(raw_warning_ratio)
        if warning_explicit and raw_warning_ratio is not None
        else 0.9
    )
    raw_trajectory_mode = getattr(args, "arm_canfd_trajectory_mode", None)
    trajectory_mode = (
        int(raw_trajectory_mode)
        if trajectory_explicit and raw_trajectory_mode is not None
        else 0
    )
    raw_radio = getattr(args, "arm_canfd_radio", None)
    radio = int(raw_radio) if radio_explicit and raw_radio is not None else 0
    max_age_sec = _finite_non_negative(
        getattr(args, "position_reference_max_age_sec", 0.2),
        name="position_reference_max_age_sec",
    )

    expected_position_profile = requested_position_profile or POSITION_PROFILE
    expected_position_follow_alpha = (
        RM65_POSITION_TRACKING_FOLLOW_ALPHA
        if expected_position_profile.name == POSITION_15_60_120HZ_PROFILE_NAME
        else 0.1
    )
    expected_position_post_ik_alpha = (
        profile_post_ik_alpha
        if position_source_shaping_applied and profile_post_ik_alpha is not None
        else 0.9
    )
    expected = (
        {
            "sample_hz": DEFAULT_TELEOP_SAMPLE_HZ,
            "outer_control_hz": default_impedance_outer_control_hz,
            "dual_arm_ik_hz": default_impedance_outer_control_hz,
            "arm_control_hz": DEFAULT_RM65_ARM_CONTROL_HZ,
            "arm_canfd_follow": False,
            "position_background_streaming_enabled": False,
            "position_background_streaming_hz": DEFAULT_RM65_ARM_CONTROL_HZ,
            "position_post_ik_qpos_smoothing_alpha": 1.0,
            "arm_follow_qpos_alpha": 1.0,
            "arm_canfd_trajectory_mode": 0,
            "arm_canfd_radio": 0,
        }
        if mode == "impedance"
        else {
            "sample_hz": expected_position_profile.rates.sample_hz,
            "outer_control_hz": expected_position_profile.rates.reference_hz,
            "position_reference_hz": expected_position_profile.rates.reference_hz,
            "dual_arm_ik_hz": expected_position_profile.rates.ik_hz,
            "arm_control_hz": expected_position_profile.rates.command_hz,
            "arm_canfd_follow": True,
            "position_background_streaming_enabled": True,
            "position_background_streaming_hz": expected_position_profile.rates.command_hz,
            "position_post_ik_qpos_smoothing_alpha": expected_position_post_ik_alpha,
            "arm_follow_qpos_alpha": expected_position_follow_alpha,
            "arm_canfd_trajectory_mode": 0,
            "arm_canfd_radio": 0,
        }
    )
    actual = {
        "sample_hz": sample_hz,
        "outer_control_hz": outer_control_hz,
        "position_reference_hz": position_reference_hz,
        "dual_arm_ik_hz": dual_arm_ik_hz,
        "arm_control_hz": arm_control_hz,
        "arm_canfd_follow": arm_canfd_follow,
        "position_background_streaming_enabled": position_background_streaming_enabled,
        "position_background_streaming_hz": position_background_streaming_hz,
        "position_post_ik_qpos_smoothing_alpha": (
            position_post_ik_qpos_smoothing_alpha
        ),
        "arm_follow_qpos_alpha": arm_follow_qpos_alpha,
        "arm_canfd_trajectory_mode": trajectory_mode,
        "arm_canfd_radio": radio,
    }
    deviations = tuple(
        label
        for label, expected_value in expected.items()
        if _changed(label, actual[label], expected_value) is not None
    )
    canonical_profile = (
        IMPEDANCE_40HZ_PROFILE_NAME
        if mode == "impedance"
        else (
            requested_position_profile.name
            if requested_position_profile is not None
            else POSITION_PROFILE_NAME
        )
    )
    resolved_profile = canonical_profile if not deviations else f"{mode}_custom"

    compatibility_warnings: list[str] = []
    if (
        stream_explicit
        and stream_hz is not None
        and stream_hz > 0.0
        and not math.isclose(stream_hz, arm_control_hz, rel_tol=0.0, abs_tol=1e-9)
    ):
        compatibility_warnings.append(
            "legacy --arm-canfd-follow-stream-hz overrides --arm-control-hz for the "
            "position background stream"
        )
    if mode == "position" and not position_background_streaming_enabled:
        compatibility_warnings.append(
            "position background streaming is disabled; the canonical "
            f"{expected_position_profile.rates.reference_hz:g}->"
            f"{expected_position_profile.rates.command_hz:g} chain is not active"
        )
    if mode == "impedance" and arm_canfd_follow:
        compatibility_warnings.append(
            "impedance CANFD follow was explicitly enabled; this differs from impedance_40hz"
        )
    if mode == "position":
        position_rates = PositionControlRates(
            sample_hz=sample_hz,
            reference_hz=position_reference_hz,
            ik_hz=dual_arm_ik_hz,
            command_hz=position_background_streaming_hz,
        )
        compatibility_warnings.extend(position_rates.ordering_warnings)
    if not math.isclose(
        dual_arm_ik_hz,
        outer_control_hz,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        compatibility_warnings.append(
            "dual_arm_ik_hz configures the solver nominal budget; actual IK calls "
            "run on the outer control/reference clock"
        )

    return RealArmExecutionConfig(
        requested_profile=requested_profile,
        resolved_profile=resolved_profile,
        mode=mode,
        sample_hz=sample_hz,
        outer_control_hz=outer_control_hz,
        position_reference_hz=position_reference_hz,
        dual_arm_ik_hz=dual_arm_ik_hz,
        arm_control_hz=arm_control_hz,
        arm_canfd_follow=arm_canfd_follow,
        arm_canfd_follow_stream_hz=stream_hz,
        position_background_streaming_enabled=position_background_streaming_enabled,
        position_background_streaming_hz=position_background_streaming_hz,
        position_post_ik_qpos_smoothing_alpha=(
            position_post_ik_qpos_smoothing_alpha
        ),
        arm_follow_qpos_alpha=arm_follow_qpos_alpha,
        arm_canfd_follow_warning_ratio=warning_ratio,
        arm_canfd_trajectory_mode=trajectory_mode,
        arm_canfd_radio=radio,
        position_reference_max_age_sec=max_age_sec,
        explicit_overrides=tuple(dict.fromkeys(explicit_labels)),
        profile_deviations=deviations,
        compatibility_warnings=tuple(dict.fromkeys(compatibility_warnings)),
    )


def resolve_control_debug_config(args: Any) -> ControlDebugConfig:
    return ControlDebugConfig(
        level=str(getattr(args, "control_debug", CONTROL_DEBUG_OFF)),
        root=str(getattr(args, "control_debug_root", DEFAULT_CONTROL_DEBUG_ROOT)),
        queue_size=int(
            getattr(args, "control_debug_queue_size", DEFAULT_CONTROL_DEBUG_QUEUE_SIZE)
        ),
        flush_every=int(
            getattr(args, "control_debug_flush_every", DEFAULT_CONTROL_DEBUG_FLUSH_EVERY)
        ),
    )


__all__ = [
    "ARM_EXECUTION_PROFILE_CHOICES",
    "CONTROL_DEBUG_FULL",
    "CONTROL_DEBUG_LEVELS",
    "CONTROL_DEBUG_OFF",
    "CONTROL_DEBUG_SUMMARY",
    "DEFAULT_CONTROL_DEBUG_FLUSH_EVERY",
    "DEFAULT_CONTROL_DEBUG_QUEUE_SIZE",
    "DEFAULT_CONTROL_DEBUG_ROOT",
    "IMPEDANCE_40HZ_PROFILE_NAME",
    "POSITION_15_60_120HZ_PROFILE_NAME",
    "POSITION_45_150HZ_PROFILE_NAME",
    "POSITION_PROFILE_NAME",
    "ControlDebugConfig",
    "RealArmExecutionConfig",
    "resolve_control_debug_config",
    "resolve_real_arm_execution_config",
]
