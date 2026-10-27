"""Frozen experimental command-field input; no device or motion authority."""

from __future__ import annotations

import hashlib
import json
import stat
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from .calibration import CalibrationCondition, CalibrationObservation, finite, six
from .command_field import LocalCommandField, evaluate_local_command_field


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class CommandFieldSource:
    """Read-only local models bound to one declared physical/gravity cohort.

    A new run ID is allowed only with otherwise identical source conditions.
    Recorded setup labels are declarations, not physical payload measurements.
    Covered held-out anchors restrict queries; no report grants control rights.
    """

    def __init__(self, report, *, source_sha256, source_path, total_assist_cap_nm):
        cap = finite(total_assist_cap_nm, "command field source cap")
        if (cap <= 0 or not isinstance(report, dict) or not _digest(source_sha256)
            or report.get("schema") != "b601.workspace-command-identification.v1"
            or report.get("reduction_method") != "closed_loop_average"
            or report.get("auto_apply") is not False
            or report.get("source_batch_state") not in {"complete", "failed", "cancelled"}
            or not all(_digest(report.get(k)) for k in ("source_trace_sha256", "source_report_sha256"))):
            raise ValueError("command field requires a closed provenance-bound command report")
        self.condition = CalibrationCondition.from_mapping(report["condition"], run_id=report["condition"]["run_id"])
        context = report.get("command_field_context") or {}
        if not isinstance(context, dict):
            raise TypeError("command field context must be an object")
        self.setup_label = context.get("setup_label")
        if not isinstance(self.setup_label, str) or not self.setup_label:
            raise ValueError("command field requires the recorded physical setup label")
        centres = context.get("pose_centres_rad")
        if not isinstance(centres, list) or not 1 <= len(centres) <= 64:
            raise ValueError("command field requires recorded pose centres")
        self.centres = tuple(tuple(finite(v, "command field pose centre") for v in pair) for pair in centres)
        if any(len(pair) != 2 for pair in self.centres) or len(set(self.centres)) != len(self.centres):
            raise ValueError("command field pose centres must be distinct J2/J3 pairs")
        windows = report.get("windows")
        if not isinstance(windows, list) or len(windows) > 65536:
            raise ValueError("invalid or oversized command field windows")
        groups = defaultdict(list)
        for window in windows:
            if not isinstance(window, dict):
                raise TypeError("command field windows must be objects")
            if window.get("model_eligible") is not True or window.get("native_phase_verified") is not True:
                continue
            pose, role = window.get("pose_index"), window.get("role")
            if type(pose) is not int or not 0 <= pose < len(self.centres) or role not in {"training", "validation"}:
                raise ValueError("invalid command field pose or split")
            sample = CalibrationObservation(condition=self.condition, **window["observation"])
            if sample.joint_name not in {"joint2", "joint3"} or abs(sample.command_residual_nm) > cap+1e-9:
                raise ValueError("command field sample exceeds the J2/J3 command envelope")
            groups[pose, sample.joint_name, sample.active_driver_kd_nm_s_per_rad].append((role, sample))
        self._regions = {}
        for key, samples in groups.items():
            if len(samples) > 512:
                raise ValueError("command field region exceeds the bounded query bank")
            training = tuple(o for role, o in samples if role == "training")
            validation = tuple(o for role, o in samples if role == "validation")
            score = evaluate_local_command_field(training, validation)
            if not (score["validation_checks"]["independent_bidirectional_training"]
                    and score["validation_checks"]["independent_bidirectional_validation"]
                    and score["directional_validation"]["covered_subset_error_passed"]):
                continue
            anchors = tuple(o for o, p in zip(validation, score["directional_validation"]["predictions"])
                            if p["predicted"] and abs(p["error_nm"]) <= .05)
            if anchors:
                self._regions[key] = (LocalCommandField(training), anchors)
        if not self._regions:
            raise ValueError("no validated native-track command field regions; reprocess the closed raw trace")
        self.source_sha256, self.source_path = source_sha256, str(source_path)
        self.trace_sha256 = report["source_trace_sha256"]
        self.total_assist_cap_nm = cap

    @classmethod
    def load(cls, path, *, total_assist_cap_nm):
        path = Path(path).expanduser().resolve(strict=True)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 64*1024*1024:
            raise ValueError("command field report must be a regular file up to 64 MiB")
        raw = path.read_bytes()
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("command field report changed while loading")
        try:
            return cls(json.loads(raw), source_sha256=hashlib.sha256(raw).hexdigest(), source_path=path,
                       total_assist_cap_nm=total_assist_cap_nm)
        except (TypeError, KeyError) as error:
            raise ValueError(f"invalid command field report: {error}") from error

    def validate_context(self, condition, *, setup_label, pose_centres_rad):
        if not isinstance(condition, CalibrationCondition) or replace(condition, run_id=self.condition.run_id) != self.condition:
            raise ValueError("command field gravity/configuration/source condition mismatch")
        if setup_label != self.setup_label:
            raise ValueError("command field physical setup label mismatch")
        indices = []
        for centre in pose_centres_rad:
            match = next((i for i, source in enumerate(self.centres)
                          if len(centre) == 2 and max(abs(a-b) for a, b in zip(centre, source)) < 1e-9), None)
            if match is None:
                raise ValueError("command field does not cover the declared pose centre")
            indices.append(match)
        return tuple(indices)

    def predict(self, *, pose_index, joint_name, qpos_rad, velocity_rad_s, active_driver_kd_nm_s_per_rad):
        if type(pose_index) is not int or not 0 <= pose_index < len(self.centres) or joint_name not in {"joint2", "joint3"}:
            raise ValueError("command field query requires a recorded pose and J2/J3")
        q = six(qpos_rad, "command field position")
        velocity = finite(velocity_rad_s, "command field velocity")
        kd = six(active_driver_kd_nm_s_per_rad, "command field damping")
        region = self._regions.get((pose_index, joint_name, kd))
        if region is None or velocity == 0:
            return {"predicted": False, "reason": "source_region_or_resolved_motion_unavailable"}
        field, anchors = region
        local_anchors = [a for a in anchors if a.velocity_rad_s*velocity > 0
            and max(abs(x-y) for x, y in zip(q, a.qpos_rad)) <= .0045]
        covered = any(max(abs(velocity), abs(a.velocity_rad_s))/min(abs(velocity), abs(a.velocity_rad_s)) <= 1.25
                      for a in local_anchors)
        if not covered:
            return {"predicted": False, "reason": "outside_validated_speed_neighbourhood" if local_anchors
                    else "outside_validated_pose_or_direction_neighbourhood"}
        return field.predict(condition=self.condition, qpos_rad=q, velocity_rad_s=velocity,
                             active_driver_kd_nm_s_per_rad=kd)

    def metadata(self):
        return {"source_path": self.source_path, "source_sha256": self.source_sha256,
            "source_trace_sha256": self.trace_sha256, "source_run_id": self.condition.run_id,
            "setup_label": self.setup_label, "regions": len(self._regions),
            "quantity": "software_command_not_measured_force", "hardware_approved": False,
            "partial_validation_scope": "covered_held_out_query_neighbourhoods_only"}
