"""Operator annotations for haptic transparency trials, without control authority.

The dynamics trace remains the sample source. This journal only marks trial
windows and operator observations on the same host monotonic clock. It neither
submits intents nor renews the motion lease. Manual force-gauge readings are
explicitly unsynchronised observations, not motor torque or live force feedback.
"""

from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

TRIAL_KINDS = ("background_drag", "cue_perception", "unloaded_response")
CONTACT_STATES = ("unknown", "none", "holding", "moving")
AXES = ("base_x", "base_y", "base_z", "joint")
OBSERVATION_FIELDS = {"perceived_strength", "perceived_direction", "manual_force_n",
                      "force_instrument", "force_observation", "note"}
_SNAPSHOT_FIELDS = (
    "phase", "owner_ready", "controller_mode", "operator_armed", "joint_name", "motion_control_enabled",
    "qpos_rad", "qvel_fd_rad_s", "feedback_age_sec", "dispatch_age_sec",
    "desired_assist_nm", "effective_assist_nm", "target_offset_rad", "fault",
    "last_dispatch",
)


def finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def six_numbers(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise ValueError(f"{name} must contain six numbers")
    return [finite_number(item, name) for item in value]


def _choice(value: Any, choices: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"invalid {name}")
    return value


def _text(value: Any, name: str, *, required: bool = False, maximum: int = 240) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ValueError(f"{name} must be text of at most {maximum} characters")
    text = value.strip()
    if required and not text:
        raise ValueError(f"{name} is required")
    return text


def _fields(payload: dict[str, Any], expected: set[str]) -> None:
    if set(payload) != expected | {"action"}:
        raise ValueError("invalid assessment request fields")


def validate_trial_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    _fields(payload, {"kind", "pair_label", "condition_label", "axis", "direction",
                      "entry_direction", "contact", "note"})
    return {
        "kind": _choice(payload["kind"], TRIAL_KINDS, "kind"),
        "pair_label": _text(payload["pair_label"], "pair_label", required=True, maximum=80),
        "condition_label": _text(payload["condition_label"], "condition_label", required=True, maximum=80),
        "axis": _choice(payload["axis"], AXES, "axis"),
        "direction": _choice(payload["direction"], ("positive", "negative"), "direction"),
        "entry_direction": _choice(payload["entry_direction"], ("positive", "negative", "unknown"), "entry_direction"),
        "contact": _choice(payload["contact"], CONTACT_STATES, "contact"),
        "note": _text(payload["note"], "note"),
    }


def validate_observation(payload: dict[str, Any], *, axis: str) -> dict[str, Any]:
    _fields(payload, OBSERVATION_FIELDS)
    force = payload["manual_force_n"]
    instrument = _text(payload["force_instrument"], "force_instrument", maximum=120)
    observation = _choice(payload["force_observation"],
                          ("none", "breakaway_peak", "steady_drag", "cue_hold"), "force_observation")
    if force is not None:
        force = finite_number(force, "manual_force_n")
        if axis == "joint" or not instrument or observation == "none":
            raise ValueError("force reading requires a base-frame axis, instrument and observation type")
    elif instrument or observation != "none":
        raise ValueError("force observation metadata requires an actual reading; leave all three empty otherwise")
    return {
        "perceived_strength": _choice(payload["perceived_strength"], ("unknown", "none", "weak", "clear"), "perceived_strength"),
        "perceived_direction": _choice(payload["perceived_direction"], ("unknown", "aligned", "opposed", "unclear"), "perceived_direction"),
        "manual_force_n": force,
        "force_instrument": instrument,
        "force_observation": observation,
        "force_source": "not_measured" if force is None else "operator_entered_external_gauge",
        "force_time_synchronised": False,
        "note": _text(payload["note"], "note"),
    }


class TransparencyAssessment:
    """Small, durable annotation journal; never called by the servo loop."""

    def __init__(self, path: Path, *, automatic: bool = False) -> None:
        self.path = path.resolve()
        self._lock = threading.Lock()
        self._stream = self.path.open("x", encoding="utf-8")
        self._active: dict[str, Any] | None = None
        self._recent: deque[dict[str, Any]] = deque(maxlen=8)
        self._sequence = 0
        self._next_trial = 0
        self._error: str | None = None
        self._closed = False
        self.automatic = automatic
        self._human_intervention: bool | None = None
        self._context_key: tuple[Any, ...] | None = None
        self._context_id = 0
        self._context: dict[str, Any] | None = None

    def _append(self, event: dict[str, Any]) -> dict[str, Any]:
        if self._closed or self._error:
            raise OSError(self._error or "assessment journal is closed")
        row = {"schema": "b601.transparency-annotation.v1", "sequence": self._sequence + 1,
               "recorded_monotonic_sec": time.monotonic(), **event}
        encoded = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        try:
            self._stream.write(encoded)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except OSError as exc:
            self._error = f"annotation persistence failed: {exc}"
            raise
        self._sequence += 1
        return row

    def record(self, payload: dict[str, Any], snapshot: dict[str, Any], *, declaration_source=None) -> None:
        """Record a user-labelled window against cached owner state only."""
        with self._lock:
            action = payload.get("action")
            sampled = {key: copy.deepcopy(snapshot.get(key)) for key in _SNAPSHOT_FIELDS}
            if self.automatic:
                _fields(payload, {"human_intervention"})
                human = payload.get("human_intervention")
                if action != "context" or not isinstance(human, bool):
                    raise ValueError("context requires human_intervention=true or false")
                if declaration_source not in {None, "managed_cycle"}:
                    raise ValueError("unknown declaration source")
                self._append({"schema": "b601.joint-diagnostic-context.v1", "event": "declaration",
                              "human_intervention": human, "owner_snapshot": sampled,
                              **({"declaration_source": declaration_source} if declaration_source else {})})
                self._human_intervention = human
                self._observe_locked(snapshot)
                return
            if action == "begin":
                meta = validate_trial_metadata(payload)
                if self._active is not None:
                    raise ValueError("finish or abort the current recording first")
                age = finite_number(snapshot.get("feedback_age_sec"), "feedback_age_sec")
                if not snapshot.get("owner_ready") or snapshot.get("fault") or not 0 <= age <= 0.15:
                    raise ValueError("recording requires ready owner and fresh feedback")
                six_numbers(snapshot.get("qpos_rad"), "qpos_rad")
                if snapshot.get("joint_name") not in tuple(f"joint{i}" for i in range(1, 7)):
                    raise ValueError("recording requires a selected arm joint")
                if meta["kind"] == "background_drag":
                    zero = all(abs(finite_number(snapshot.get(key, 0.0), key)) <= 1e-9
                               for key in ("desired_assist_nm", "effective_assist_nm"))
                    if snapshot.get("phase") != "inactive" or snapshot.get("operator_armed") or not zero:
                        raise ValueError("background recording requires withdrawn, unarmed zero assist")
                trial_id = self._next_trial + 1
                row = self._append({"event": "begin", "trial_id": trial_id, "metadata": meta,
                                    "owner_snapshot": sampled})
                self._next_trial = trial_id
                self._active = {"trial_id": trial_id, "metadata": meta,
                                "joint_name": snapshot["joint_name"], "contact": meta["contact"],
                                "start_qpos_rad": sampled["qpos_rad"],
                                "start_sec": row["recorded_monotonic_sec"]}
            elif action in ("contact", "finish", "abort"):
                if self._active is None:
                    raise ValueError("there is no active recording")
                event: dict[str, Any] = {"event": action, "trial_id": self._active["trial_id"],
                                         "owner_snapshot": sampled}
                if action == "contact":
                    _fields(payload, {"contact"})
                    event["contact"] = _choice(payload["contact"], CONTACT_STATES, "contact")
                elif action == "finish":
                    event["observation"] = validate_observation(payload, axis=self._active["metadata"]["axis"])
                else:
                    _fields(payload, {"note"})
                    event["note"] = _text(payload["note"], "note")
                self._append(event)
                if action == "contact":
                    self._active["contact"] = event["contact"]
                else:
                    self._recent.append({**self._active, "end": action,
                                         "observation": event.get("observation")})
                    self._active = None
            else:
                raise ValueError("unknown assessment action")

    def _observe_locked(self, snapshot: dict[str, Any]) -> None:
        if not self.automatic or self._closed or self._error:
            return
        q = snapshot.get("qpos_rad")
        if q is None:
            return
        six_numbers(q, "qpos_rad")
        phase = snapshot.get("phase")
        drive_enabled = bool(snapshot.get("operator_armed")) or phase == "armed"
        # Requested torque is a setpoint only in direct mode. The staged
        # effective overlay changes on every cycle and must not split windows.
        target = (snapshot.get("target_offset_rad") if snapshot.get("controller_mode") == "staged_assist" or snapshot.get("motion_control_enabled") is True
                  else snapshot.get("desired_assist_nm"))
        age = snapshot.get("feedback_age_sec")
        fresh = isinstance(age, (int, float)) and not isinstance(age, bool) and 0 <= age <= 0.15
        key = (self._human_intervention, drive_enabled, snapshot.get("joint_name"),
               target, bool(snapshot.get("owner_ready")), fresh, snapshot.get("fault"))
        if key == self._context_key:
            return
        sampled = {name: copy.deepcopy(snapshot.get(name)) for name in _SNAPSHOT_FIELDS}
        row = self._append({"schema": "b601.joint-diagnostic-context.v1", "event": "context",
                            "context_id": self._context_id + 1,
                            "human_intervention": self._human_intervention,
                            "drive_enabled": drive_enabled, "feedback_fresh": fresh,
                            "owner_snapshot": sampled})
        self._context_id += 1
        self._context_key = key
        self._context = {"context_id": self._context_id,
                         "human_intervention": self._human_intervention,
                         "drive_enabled": drive_enabled, "joint_name": snapshot.get("joint_name"),
                         "start_sec": row["recorded_monotonic_sec"], "start_qpos_rad": sampled["qpos_rad"]}

    def observe(self, snapshot: dict[str, Any]) -> None:
        """Observe cached state off the servo thread; never renew an intent."""
        with self._lock:
            self._observe_locked(snapshot)

    def observation_failed(self, error: Exception) -> None:
        with self._lock:
            self._error = self._error or f"automatic recording stopped: {error}"

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {"available": not self._closed and self._error is None,
                    "error": self._error, "journal_path": str(self.path),
                    "active": copy.deepcopy(self._active), "recent": list(self._recent),
                    "sequence": self._sequence, "live_force_sensor": False,
                    "control_authority": False, "automatic": self.automatic,
                    "human_intervention": self._human_intervention,
                    "context": copy.deepcopy(self._context)}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self.automatic and self._context is not None and not self._error:
                    self._append({"schema": "b601.joint-diagnostic-context.v1", "event": "end",
                                  "context_id": self._context_id, "reason": "owner_console_closed"})
                if self._active is not None and not self._error:
                    self._append({"event": "interrupted", "trial_id": self._active["trial_id"],
                                  "reason": "owner_console_closed"})
                    self._recent.append({**self._active, "end": "interrupted"})
                    self._active = None
            finally:
                self._closed = True
                self._stream.close()
