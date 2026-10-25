"""Offline, streaming trial report for B601 torque-console dynamics traces.

Only owner-confirmed ``active_dispatch`` rows are treated as sent commands.
This module never opens CAN, a motor backend, or the torque-console server.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_JOINTS = tuple(f"joint{i}" for i in range(1, 7))


def _number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _six(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{field_name} must contain six numbers")
    return tuple(_number(item, field_name) for item in value)


@dataclass
class _Trial:
    joint_name: str
    target_nm: float
    started_at_sec: float
    anchor_rad: float
    first_qpos_rad: tuple[float, ...]
    first_gravity_nm: float
    onset_displacement_rad: float
    onset_samples_required: int
    samples: int = 0
    max_abs_sent_assist_nm: float = 0.0
    max_abs_velocity_rad_s: float = 0.0
    min_qpos_rad: float = math.inf
    max_qpos_rad: float = -math.inf
    final_qpos_rad: float = 0.0
    final_gravity_nm: float = 0.0
    final_total_command_nm: float = 0.0
    torque_limited_samples: int = 0
    slew_limited_samples: int = 0
    onset: dict[str, float | int | bool] | None = None
    _onset_candidate: dict[str, float | int | bool] | None = None
    _onset_candidate_count: int = 0
    end_reason: str = "trace_end"
    ended_at_sec: float | None = None
    target_zero_at_sec: float | None = None
    confirmed_zero_at_sec: float | None = None
    zero_qpos_rad: float | None = None
    post_zero_qpos_rad: float | None = None
    post_zero_observation_sec: float | None = None
    post_zero_window_complete: bool = False
    guard_reason: str | None = None
    guard_qpos_rad: float | None = None
    guard_velocity_rad_s: float | None = None

    @property
    def joint_index(self) -> int:
        return _JOINTS.index(self.joint_name)

    def add_dispatch(self, row: dict[str, object], *, time_sec: float) -> None:
        index = self.joint_index
        q = _six(row.get("qpos_rad"), "qpos_rad")[index]
        velocity = _six(row.get("qvel_fd_rad_s"), "qvel_fd_rad_s")[index]
        assist = _six(row.get("overlay_effective_tau_nm"), "overlay_effective_tau_nm")[index]
        gravity = _six(row.get("gravity_tau_nm"), "gravity_tau_nm")[index]
        total = _six(row.get("conditioned_tau_nm"), "conditioned_tau_nm")[index]
        if not isinstance(row.get("torque_limited"), bool) or not isinstance(
            row.get("slew_limited"), bool
        ):
            raise TypeError("dispatch limit indicators must be boolean")
        self.samples += 1
        self.final_qpos_rad = q
        self.final_gravity_nm = gravity
        self.final_total_command_nm = total
        self.min_qpos_rad = min(self.min_qpos_rad, q)
        self.max_qpos_rad = max(self.max_qpos_rad, q)
        self.max_abs_sent_assist_nm = max(self.max_abs_sent_assist_nm, abs(assist))
        self.max_abs_velocity_rad_s = max(self.max_abs_velocity_rad_s, abs(velocity))
        self.torque_limited_samples += int(row["torque_limited"])
        self.slew_limited_samples += int(row["slew_limited"])
        displacement = q - self.anchor_rad
        if self.onset is not None or abs(displacement) < self.onset_displacement_rad:
            self._onset_candidate = None
            self._onset_candidate_count = 0
            return
        direction = 1 if displacement > 0 else -1
        candidate = self._onset_candidate
        if candidate is None or candidate["direction"] != direction:
            self._onset_candidate = {
                "time_sec": time_sec,
                "displacement_rad": displacement,
                "sent_assist_nm": assist,
                "velocity_rad_s": velocity,
                "direction": direction,
                "direction_matches_target": displacement * self.target_nm > 0,
            }
            self._onset_candidate_count = 1
        else:
            self._onset_candidate_count += 1
        if self._onset_candidate_count >= self.onset_samples_required:
            self.onset = self._onset_candidate

    def observe_withdrawal(
        self, *, time_sec: float, qpos_rad: float, sent_assist_nm: float, settling_sec: float
    ) -> bool:
        if sent_assist_nm != 0.0:
            self.confirmed_zero_at_sec = None
            self.zero_qpos_rad = None
            self.post_zero_qpos_rad = None
            self.post_zero_observation_sec = None
            return False
        if self.confirmed_zero_at_sec is None:
            self.confirmed_zero_at_sec = time_sec
            self.zero_qpos_rad = qpos_rad
        else:
            self.post_zero_qpos_rad = qpos_rad
            self.post_zero_observation_sec = time_sec
        self.post_zero_window_complete = time_sec >= self.confirmed_zero_at_sec + settling_sec
        return self.post_zero_window_complete

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_name": self.joint_name,
            "target_assist_nm": self.target_nm,
            "started_at_sec": self.started_at_sec,
            "ended_at_sec": self.ended_at_sec,
            "end_reason": self.end_reason,
            "guard_reason": self.guard_reason,
            "guard_qpos_rad": self.guard_qpos_rad,
            "guard_velocity_rad_s": self.guard_velocity_rad_s,
            "first_qpos_rad": self.first_qpos_rad,
            "anchor_rad": self.anchor_rad,
            "first_gravity_nm": self.first_gravity_nm,
            "final_gravity_nm": self.final_gravity_nm,
            "final_total_command_nm": self.final_total_command_nm,
            "dispatch_samples": self.samples,
            "max_abs_sent_assist_nm": self.max_abs_sent_assist_nm,
            "max_abs_velocity_rad_s": self.max_abs_velocity_rad_s,
            "min_displacement_rad": self.min_qpos_rad - self.anchor_rad,
            "max_displacement_rad": self.max_qpos_rad - self.anchor_rad,
            "final_displacement_rad": self.final_qpos_rad - self.anchor_rad,
            "torque_limited_samples": self.torque_limited_samples,
            "slew_limited_samples": self.slew_limited_samples,
            "sustained_motion_onset": self.onset,
            "onset_observation": (
                "observed" if self.onset is not None else "not_observed_within_sent_assist"
            ),
            "target_zero_at_sec": self.target_zero_at_sec,
            "confirmed_zero_at_sec": self.confirmed_zero_at_sec,
            "zero_qpos_rad": self.zero_qpos_rad,
            "post_zero_qpos_rad": self.post_zero_qpos_rad,
            "post_zero_observation_sec": self.post_zero_observation_sec,
            "post_zero_window_complete": self.post_zero_window_complete,
        }


def summarize_trace(
    records: Iterable[dict[str, object]],
    *,
    onset_displacement_rad: float = 0.001,
    onset_consecutive_samples: int = 2,
    post_zero_sec: float = 0.5,
) -> dict[str, object]:
    """Segment target changes; never call a non-moving trial a torque threshold."""

    threshold = _number(onset_displacement_rad, "onset_displacement_rad")
    settling = _number(post_zero_sec, "post_zero_sec")
    if threshold <= 0 or settling <= 0 or onset_consecutive_samples < 1:
        raise ValueError("onset displacement, sample count, and post-zero time must be positive")
    trials: list[_Trial] = []
    active: _Trial | None = None
    pending: _Trial | None = None
    previous_time: float | None = None
    dispatch_count = 0
    for row in records:
        kind = row.get("kind")
        if kind not in ("active_dispatch", "torque_console_guard"):
            continue
        time_sec = _number(row.get("recorded_monotonic_sec"), "recorded_monotonic_sec")
        if previous_time is not None and time_sec < previous_time:
            raise ValueError("relevant trace timestamps must be monotonic")
        previous_time = time_sec
        if kind == "torque_console_guard":
            guarded = active or pending
            if guarded is not None:
                guarded.end_reason = "guard"
                guarded.ended_at_sec = time_sec
                guarded.guard_reason = str(row.get("fault") or "unknown_guard")
                guarded.guard_qpos_rad = _six(row.get("qpos_rad"), "guard qpos_rad")[
                    guarded.joint_index
                ]
                guarded.guard_velocity_rad_s = _six(
                    row.get("qvel_fd_rad_s"), "guard qvel_fd_rad_s"
                )[guarded.joint_index]
                trials.append(guarded)
                active = None
                pending = None
            continue
        audit = row.get("torque_console")
        if audit is None:
            if active is not None:
                active.end_reason = "console_unavailable"
                active.ended_at_sec = time_sec
                trials.append(active)
                active = None
            continue
        if not isinstance(audit, dict):
            raise TypeError("torque_console audit must be an object or null")
        dispatch_count += 1
        joint_name = audit.get("joint_name")
        if joint_name not in _JOINTS:
            raise ValueError("torque_console joint_name is invalid")
        qpos = _six(row.get("qpos_rad"), "qpos_rad")
        desired = _number(audit.get("desired_assist_nm"), "desired_assist_nm")
        phase = audit.get("phase")
        if pending is not None:
            if phase == "armed" and desired != 0:
                trials.append(pending)
                pending = None
            else:
                sent = _six(row.get("overlay_effective_tau_nm"), "overlay_effective_tau_nm")
                if pending.observe_withdrawal(
                    time_sec=time_sec,
                    qpos_rad=qpos[pending.joint_index],
                    sent_assist_nm=sent[pending.joint_index],
                    settling_sec=settling,
                ):
                    trials.append(pending)
                    pending = None
        if phase == "armed" and desired != 0:
            if active is not None and (active.joint_name, active.target_nm) != (
                joint_name,
                desired,
            ):
                active.end_reason = "target_changed"
                active.ended_at_sec = time_sec
                trials.append(active)
                active = None
            if active is None:
                anchor = _number(audit.get("anchor_rad"), "anchor_rad")
                index = _JOINTS.index(joint_name)
                gravity = _six(row.get("gravity_tau_nm"), "gravity_tau_nm")[index]
                active = _Trial(
                    joint_name=joint_name,
                    target_nm=desired,
                    started_at_sec=time_sec,
                    anchor_rad=anchor,
                    first_qpos_rad=qpos,
                    first_gravity_nm=gravity,
                    onset_displacement_rad=threshold,
                    onset_samples_required=onset_consecutive_samples,
                )
            active.add_dispatch(row, time_sec=time_sec)
        elif active is not None:
            active.end_reason = "zero" if desired == 0 else "inactive"
            active.ended_at_sec = time_sec
            if desired == 0:
                active.target_zero_at_sec = time_sec
            pending = active
            active = None
            sent = _six(row.get("overlay_effective_tau_nm"), "overlay_effective_tau_nm")
            if pending.observe_withdrawal(
                time_sec=time_sec,
                qpos_rad=qpos[pending.joint_index],
                sent_assist_nm=sent[pending.joint_index],
                settling_sec=settling,
            ):
                trials.append(pending)
                pending = None
    if active is not None:
        trials.append(active)
    if pending is not None:
        trials.append(pending)
    if not trials:
        raise ValueError("no non-zero torque-console dispatch found in trace")
    return {
        "schema": "b601.torque-console-offline-trials.v1",
        "source_semantics": "owner-confirmed software dispatch; not measured motor torque",
        "onset_definition": {
            "absolute_displacement_rad": threshold,
            "consecutive_samples_same_direction": onset_consecutive_samples,
            "post_zero_observation_sec": settling,
        },
        "console_dispatch_rows": dispatch_count,
        "trials": [trial.as_dict() for trial in trials],
    }


def _resolve_jsonl_path(path: Path) -> Path:
    # Old run-directory references remain readable after lossless archival.
    if not path.exists() and path.suffix != ".gz":
        compressed = path.with_name(path.name+".gz")
        if compressed.exists():
            return compressed
    return path


def _jsonl(path: Path) -> Iterable[dict[str, object]]:
    path = _resolve_jsonl_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number}: expected JSON object")
                yield row
    except EOFError as error:
        raise ValueError(f"{path}: incomplete compressed trace") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--onset-displacement-rad", type=float, default=0.001)
    parser.add_argument("--onset-consecutive-samples", type=int, default=2)
    parser.add_argument("--post-zero-sec", type=float, default=0.5)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--transparency", action="store_true", help="join operator-labelled transparency trials; do not infer hand force")
    mode.add_argument("--calibration", action="store_true", help="offline, gravity-conditioned command identification; never applies compensation")
    mode.add_argument("--automatic-calibration", action="store_true", help="automatic onset, return and running-resistance acquisition summary")
    mode.add_argument("--organize-identification", action="store_true", help="organize retained automatic reports without reading or rewriting raw traces")
    mode.add_argument("--closed-loop-identification", action="store_true",
        help="reprocess a closed continuous run with time-integrated dispatched inputs; no device I/O or automatic application")
    parser.add_argument("--position-bin-width-rad", type=float,
        help="with --closed-loop-identification, reduce complete measured-position cells instead of time windows")
    parser.add_argument("--include-run-dir", type=Path, action="append", default=[],
        help="additional closed run for --organize-identification; conditions remain separate")
    parser.add_argument("--pair-pose-tolerance-rad", type=float, default=0.03)
    args = parser.parse_args(argv)
    try:
        if args.include_run_dir and not args.organize_identification:
            raise ValueError("--include-run-dir requires --organize-identification")
        if args.position_bin_width_rad is not None and not args.closed_loop_identification:
            raise ValueError("--position-bin-width-rad requires --closed-loop-identification")
        if args.organize_identification:
            from .workspace_identification import organize_automatic_reports

            trace = args.run_dir.resolve() / "automatic_calibration_report.json"
            report = organize_automatic_reports([directory / "automatic_calibration_report.json"
                for directory in [args.run_dir, *args.include_run_dir]])
        else:
            trace = _resolve_jsonl_path(args.run_dir.resolve() / "b601_owner.debug.jsonl")
        if args.organize_identification:
            pass
        elif args.closed_loop_identification:
            from robot_motion.rebot_b601_rs import (
                calibration,
                command_field,
                gravity_identification,
            )

            from . import workspace_identification

            source = trace.parent / "automatic_calibration_report.json"
            raw_report = source.read_bytes()
            original = json.loads(raw_report)
            if not isinstance(original, dict):
                raise TypeError("automatic report must be an object")
            workspace = original.get("workspace")
            condition = original.get("condition")
            if (original.get("schema") != "b601.automatic-calibration-report.v1"
                or not isinstance(workspace, dict) or workspace.get("holding_sampling_plan") != "identification"
                or original.get("batch_state") not in {"complete", "failed", "cancelled"}):
                raise ValueError("closed-loop identification requires a continuous automatic report")
            if not isinstance(condition, dict) or not isinstance(condition.get("run_id"), str):
                raise ValueError("closed-loop identification requires source/gravity condition")
            before = trace.stat()
            digest = hashlib.sha256()
            with trace.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            report = workspace_identification.summarize_workspace_identification(_jsonl(trace),
                condition=condition, workspace=workspace,
                reduction_method="closed_loop_average", position_bin_width_rad=args.position_bin_width_rad)
            after = trace.stat()
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or source.read_bytes() != raw_report):
                raise ValueError("source changed during closed-loop reprocessing")
            baseline = original.get("command_identification") or {}
            report.update(trials=[], source_trace_sha256=digest.hexdigest(),
                source_report=str(source), source_report_sha256=hashlib.sha256(raw_report).hexdigest(),
                source_batch_state=original.get("batch_state"),
                source_acquisition_complete=original.get("acquisition_complete"),
                baseline={"reduction_method": baseline.get("reduction_method", "quasisteady"),
                    "admitted_motion_windows": baseline.get("admitted_motion_windows"),
                    "recorded_command_windows": len(baseline.get("windows", [])),
                    "gravity_status": (baseline.get("gravity_fit") or {}).get("status"),
                    "offline_candidates": (baseline.get("running_resistance") or {}).get("offline_candidates")},
                processor_sources_sha256={Path(module.__file__).name:
                    hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                    for module in (workspace_identification, calibration, gravity_identification, command_field)},
                assumptions=["latest complete six-axis dispatch approximates the held input, not device actuation time",
                    "linear encoder motion approximates cruise; inertia and transmission preload are not measured",
                    "only approximately odd repeatable resistance cancels in matched opposite-direction pairs"])
            report["processor_sources_sha256"][Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        elif args.automatic_calibration:
            from .automatic_report import build_automatic_report

            report = build_automatic_report(trace.parent)
        elif args.calibration:
            from .calibration_report import summarize_calibration

            annotations = trace.with_name("b601_owner.transparency.jsonl")
            report = summarize_calibration(_jsonl(annotations), _jsonl(trace), run_id=str(trace.parent))
            report["source_annotations"] = str(annotations)
        elif args.transparency:
            from .transparency_report import summarize_transparency

            annotations = trace.with_name("b601_owner.transparency.jsonl")
            report = summarize_transparency(
                _jsonl(annotations), _jsonl(trace),
                pair_pose_tolerance_rad=args.pair_pose_tolerance_rad,
            )
            report["source_annotations"] = str(annotations)
        else:
            report = summarize_trace(
                _jsonl(trace),
                onset_displacement_rad=args.onset_displacement_rad,
                onset_consecutive_samples=args.onset_consecutive_samples,
                post_zero_sec=args.post_zero_sec,
            )
        if not args.organize_identification:
            report["source_trace"] = str(trace)
        output_root = (args.output_root or trace.parent.parent).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="b601-torque-report-", dir=output_root))
        output = output_dir / "report.json"
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, TypeError, ValueError) as error:
        print(f"[B601-REPORT] ERROR: {error}", file=sys.stderr)
        return 2
    print(f"[B601-REPORT] trials={len(report['trials'])} report={output}")
    if args.closed_loop_identification:
        print(f"[B601-REPORT] windows={len(report['windows'])} admitted={report['admitted_motion_windows']} "
              f"candidates={report['running_resistance']['offline_candidates']} "
              f"gravity={report['gravity_fit']['status']} "
              f"paired_resistance={report['paired_resistance_fit']['status']}; "
              "offline command evidence only; no compensation applied")
        return 0
    if args.organize_identification:
        print(f"[B601-REPORT] runs={report['run_count']} observations={report['record_count']}; no source changed or compensation applied")
        return 0
    if args.automatic_calibration:
        print(f"[B601-REPORT] acquisition={report['status']}; commanded torque only; no compensation applied")
        windows = report["motion_data_assessment"]
        print(
            f"[B601-REPORT] command_windows={windows['recorded_command_windows']} "
            f"steady_windows={windows['observed_steady_motion_windows']} "
            f"model_windows={windows['model_eligible_windows']} "
            f"retained_partial_leg_windows={windows['retained_partial_leg_windows']}"
        )
        return 0
    if args.calibration:
        print(f"[B601-REPORT] calibration={report['status']} candidates={report['offline_candidates']}; no compensation applied")
        return 0 if report["status"] == "OFFLINE_CANDIDATE" else 1
    if args.transparency:
        print("[B601-REPORT] motion and command torque are not hand-force measurements; details saved to report")
        return 0
    for index, trial in enumerate(report["trials"], start=1):
        onset = trial["sustained_motion_onset"]
        onset_text = (
            "none observed" if onset is None else f"at sent {onset['sent_assist_nm']:+.3f} Nm"
        )
        print(
            f"  {index:02d} {trial['joint_name']} target={trial['target_assist_nm']:+.3f} Nm "
            f"max_sent={trial['max_abs_sent_assist_nm']:.3f} Nm "
            f"onset={onset_text} end={trial['end_reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
