"""Passive supported-release evidence, independent of a robot or simulator.

The caller binds one object, pose frame and monotonic measurement clock per
window. Samples must be measured, not interpolated or inferred from an open
command. Stability describes the observed window only: it is neither holding
proof, destination membership nor a physical task success decision.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} is outside its finite range")
    return result


@dataclass(frozen=True)
class ObjectReleasePolicy:
    duration_s: float
    min_measurements: int
    max_sample_gap_s: float
    position_tolerance_m: float
    rotation_tolerance_rad: float

    def __post_init__(self):
        for name in ("duration_s", "max_sample_gap_s", "position_tolerance_m",
                     "rotation_tolerance_rad"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        if (isinstance(self.min_measurements, bool)
                or not isinstance(self.min_measurements, int)
                or not 3 <= self.min_measurements <= 4096
                or self.max_sample_gap_s > self.duration_s / 2
                or self.rotation_tolerance_rad > math.pi):
            raise ValueError("Release evidence requires a bounded, sufficiently sampled window")


@dataclass(frozen=True)
class ObjectReleaseMeasurement:
    measured_at: float
    pose_xyz_quat_xyzw: tuple[float, ...]
    support_contact: bool | None
    gripper_contact: bool | None
    other_contact: bool | None

    def __post_init__(self):
        object.__setattr__(self, "measured_at", _number(self.measured_at, "measured_at"))
        pose = self.pose_xyz_quat_xyzw
        if not isinstance(pose, (tuple, list)) or len(pose) != 7:
            raise ValueError("Release measurement requires one XYZ/XYZW quaternion pose")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in pose):
            raise ValueError("Release pose must be finite")
        pose = tuple(float(v) for v in pose)
        if not math.isclose(math.hypot(*pose[3:]), 1., rel_tol=0., abs_tol=1e-6):
            raise ValueError("Release quaternion must have unit length")
        object.__setattr__(self, "pose_xyz_quat_xyzw", pose)
        for name in ("support_contact", "gripper_contact", "other_contact"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError("Release contact flags must be measured bool or None")


class ObjectReleaseWindow:
    """Bounded observation history; no sampling, waiting, physics or dispatch.

    A sliding window retains the cut immediately before its duration boundary.
    Drift is measured against that cut, never just consecutive poses. Unknown
    contacts, an actual contact violation or a sampling gap break the window.
    Repeated reads cannot add time or samples. A later reading can revoke a
    previously stable result; this is intentionally not a latched success.
    """

    def __init__(self, policy: ObjectReleasePolicy):
        if not isinstance(policy, ObjectReleasePolicy):
            raise TypeError("A release observation policy is required")
        self.policy = policy
        self._samples = deque(maxlen=4096)
        self._last = None
        self._reason = "not_observed"

    def observe(self, sample: ObjectReleaseMeasurement) -> None:
        if not isinstance(sample, ObjectReleaseMeasurement):
            raise TypeError("A measured release sample is required")
        if self._last is not None:
            if sample.measured_at < self._last.measured_at:
                raise ValueError("Release measurement time went backwards")
            if sample.measured_at == self._last.measured_at:
                if sample != self._last:
                    raise ValueError("Conflicting release measurements at the same time")
                return
        gap = (self._last is not None and sample.measured_at - self._last.measured_at
               > self.policy.max_sample_gap_s + 1e-9)
        self._last = sample
        if sample.other_contact is True:
            reason = "unexpected_contact"
        elif sample.gripper_contact is True:
            reason = "not_released"
        elif sample.support_contact is False:
            reason = "not_supported"
        elif any(v is None for v in (
                sample.support_contact, sample.gripper_contact, sample.other_contact)):
            reason = "contact_evidence_unavailable"
        else:
            reason = "observation_gap" if gap else "collecting"
        if reason != "collecting":
            self._samples.clear()
        self._reason = reason
        if reason not in {"collecting", "observation_gap"}:
            return
        self._samples.append(sample)
        while (len(self._samples) > 2 and sample.measured_at - self._samples[1].measured_at
               >= self.policy.duration_s):
            self._samples.popleft()

    def snapshot(self, *, now: float) -> dict:
        """Return current evidence without renewing its age or window."""
        now = _number(now, "now")
        if self._last is not None and now < self._last.measured_at:
            raise ValueError("Release observation time is in the future")
        age = None if self._last is None else now - self._last.measured_at
        fresh = age is not None and age <= self.policy.max_sample_gap_s + 1e-9
        distance = angle = duration = 0.
        if self._samples:
            first = self._samples[0]
            duration = self._samples[-1].measured_at - first.measured_at
            reference = first.pose_xyz_quat_xyzw
            for sample in self._samples:
                pose = sample.pose_xyz_quat_xyzw
                distance = max(distance, math.dist(reference[:3], pose[:3]))
                cosine = abs(sum(a * b for a, b in zip(reference[3:], pose[3:])))
                cosine /= math.hypot(*reference[3:]) * math.hypot(*pose[3:])
                angle = max(angle, 2 * math.acos(min(1., cosine)))
        if not math.isfinite(distance):
            raise ValueError("Derived release position drift overflowed")
        status = self._reason
        if self._samples:
            if (distance > self.policy.position_tolerance_m
                    or angle > self.policy.rotation_tolerance_rad):
                status = "moving"
            elif (duration + 1e-9 >= self.policy.duration_s
                    and len(self._samples) >= self.policy.min_measurements):
                status = "stable_supported_release"
            else:
                status = "insufficient_window"
        if self._last is not None and not fresh:
            status = "stale"
        return {
            "status": status, "stable_supported_release": status == "stable_supported_release",
            "measured_at": None if self._last is None else self._last.measured_at,
            "evidence_age_s": age, "evidence_fresh": fresh,
            "window_start_s": None if not self._samples else self._samples[0].measured_at,
            "observed_duration_s": duration, "measurement_count": len(self._samples),
            "max_position_drift_m": distance, "max_rotation_drift_rad": angle,
            "max_observed_sample_gap_s": max(
                (b.measured_at - a.measured_at
                 for a, b in zip(self._samples, tuple(self._samples)[1:])),
                default=None),
            "window_reason": "complete" if status == "stable_supported_release" else self._reason,
            "policy": asdict(self.policy),
        }
