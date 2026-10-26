"""Audit a fixed geometry matrix without running models, planners, or devices.

This report consumes the existing private model probes. It is deliberately a
geometry-stage report: retained proposals are not grasp or task successes.
All evidence and machine-local paths live in an external manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile


def _read(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def _unique(items, name):
    result = {}
    for item in items:
        key = item["id"]
        if not isinstance(key, str) or not key or key in result:
            raise ValueError(f"Nonempty unique {name} id required")
        result[key] = item
    if not result:
        raise ValueError(f"At least one {name} required")
    return result


def _verify_file(binding, root):
    path = Path(binding["path"])
    if not path.is_absolute():
        path = root / path
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != binding["sha256"]:
        raise ValueError(f"Evidence changed: {path}")
    return path


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": ordered[0], "max": ordered[-1]}


def condition_identity(condition):
    """Producer and reviewer bind semantic parameters and content, not paths."""
    payload = {"geometry_profile": condition["geometry_profile"],
               "parameters": condition["parameters"],
               "fixed_sha256": sorted({item["sha256"] for item in condition["fixed_files"]})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def review(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    root, manifest = manifest_path.parent, _read(manifest_path)
    if manifest["schema"] != "agenticrobot.grasp_geometry_benchmark.v1":
        raise ValueError("Unsupported geometry benchmark schema")
    cases = _unique(manifest["cases"], "case")
    conditions = _unique(manifest["conditions"], "condition")
    slots = _unique(manifest["slots"], "slot")
    repeats = _count(manifest["repeats"], "repeats")
    if repeats < 1:
        raise ValueError("At least one fixed repeat required")
    if manifest["baseline"] not in conditions:
        raise ValueError("Baseline must name a declared condition")
    input_digests = set()
    for case in cases.values():
        if case["domain"] not in ("real_rgbd", "mujoco_rgbd") or not case["source_group"]:
            raise ValueError("Case requires an explicit sensor domain and source group")
        if set(case["input_files"]) != {"npz", "metadata", "mask"}:
            raise ValueError("Every case must bind RGB-D, camera metadata and measured mask")
        digest = (case["view"], case["prompt"], *(case["input_files"][key]["sha256"] for key in sorted(case["input_files"])))
        if digest in input_digests:
            raise ValueError("The same measured input cannot become multiple unique cases")
        input_digests.add(digest)
        for binding in case["input_files"].values():
            _verify_file(binding, root)
    for condition in conditions.values():
        if not condition["fixed_files"] or not condition["geometry_profile"]:
            raise ValueError("Condition must bind its source/model/configuration evidence")
        for binding in condition["fixed_files"]:
            _verify_file(binding, root)

    reports, used_records, coordinates, rows = {}, set(), set(), []
    for slot_id, slot in slots.items():
        case, condition = cases[slot["case"]], conditions[slot["condition"]]
        repeat = _count(slot["repeat"], "repeat")
        if repeat >= repeats:
            raise ValueError("Repeat index outside the fixed matrix")
        coordinate = (slot["case"], slot["condition"], repeat)
        if coordinate in coordinates:
            raise ValueError("Duplicate case/condition/repeat would inflate the denominator")
        coordinates.add(coordinate)
        row = {"slot": slot_id, "case": slot["case"], "condition": slot["condition"],
               "repeat": repeat, "domain": case["domain"], "source_group": case["source_group"],
               "status": "unstarted", "cleanup_confirmed": False, "retained_pool": None,
               "geometry_wall_sec": None, "candidate_present": False, "execution_identity_verified": False}
        evidence = slot.get("evidence")
        if evidence is None:
            if slot.get("start_evidence") is not None:
                _verify_file(slot["start_evidence"], root)
                row["status"] = "missing_result"
            rows.append(row)
            continue
        path = _verify_file(evidence, root)
        frozen_path = _verify_file(evidence["freeze"], root)
        frozen = _read(frozen_path)
        frozen_cases = _unique(frozen["cases"], "frozen case")
        source_case = frozen_cases[evidence["case_id"]]
        if (case["view"], case["prompt"]) != (source_case["view"], source_case["prompt"]):
            raise ValueError("Case view/prompt differs from the original frozen trial")
        for key, binding in case["input_files"].items():
            if frozen["fixed_files"].get(source_case[key]) != binding["sha256"]:
                raise ValueError("Input identity differs from the original frozen trial")
        identity = (evidence["sha256"], evidence["record_slot"])
        if identity in used_records:
            raise ValueError("One evidence record cannot count as multiple repetitions")
        used_records.add(identity)
        if path not in reports:
            report = _read(path)
            if report.get("native_rig_started") is not False or report.get("public_candidate_identity_created") is not False:
                raise ValueError("Use only geometry-stage probe reports, not native execution reports")
            records = report["records"]
            ids = [item["slot"] for item in records]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate source record slot")
            reports[path] = (report, {item["slot"]: item for item in records})
        report, records = reports[path]
        record = records[evidence["record_slot"]]
        source_binding = report.get("benchmark_binding")
        if source_binding is not None:
            if (source_binding.get("schema") != "agenticrobot.grasp_geometry_execution_binding.v1"
                    or source_binding.get("freeze_sha256") != evidence["freeze"]["sha256"]):
                raise ValueError("Report is not bound to this frozen input set")
            if record.get("condition_sha256") != condition_identity(condition):
                raise ValueError("Declared condition differs from the executed source/model/configuration")
            row["execution_identity_verified"] = True
        if not (evidence["record_slot"] == evidence["case_id"] or evidence["record_slot"].startswith(evidence["case_id"] + "-")):
            raise ValueError("Source record does not belong to the bound frozen case")
        profile = record.get("geometry_profile", record.get("profile"))
        if profile is not None and profile != condition["geometry_profile"]:
            raise ValueError("Source geometry profile differs from the declared condition")
        row["status"] = record["status"]
        row["cleanup_confirmed"] = report.get("normal_cleanup") is True
        if row["status"] == "completed":
            row["retained_pool"] = _count(record["retained_pool"], "retained_pool")
            elapsed = record["wall_sec"]
            if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Geometry time must be a finite non-negative number")
            row["geometry_wall_sec"] = elapsed
            row["candidate_present"] = row["retained_pool"] > 0
        elif row["status"] not in ("failed", "error", "rejected"):
            raise ValueError("Unknown recorded outcome; do not silently classify it")
        rows.append(row)

    expected = {(case, condition, repeat) for case in cases for condition in conditions for repeat in range(repeats)}
    if coordinates != expected:
        raise ValueError("Matrix must retain every declared case/condition/repeat, including unstarted slots")
    totals = {}
    for name in conditions:
        selected = [row for row in rows if row["condition"] == name]
        descriptive = [row for row in selected if row["status"] == "completed" and row["cleanup_confirmed"]]
        qualified = [row for row in descriptive if row["execution_identity_verified"]]
        totals[name] = {
            "declared": len(selected), "attempted": sum(row["status"] != "unstarted" for row in selected),
            "status_counts": dict(Counter(row["status"] for row in selected)),
            "cleanup_confirmed": sum(row["cleanup_confirmed"] for row in selected),
            "execution_identity_verified": sum(row["execution_identity_verified"] for row in selected),
            "candidate_present_descriptive_only": sum(row["candidate_present"] for row in descriptive),
            "candidate_present": sum(row["candidate_present"] for row in qualified),
            "unique_inputs": len({row["case"] for row in selected}),
            "unique_inputs_with_candidates": len({row["case"] for row in qualified if row["candidate_present"]}),
            "independent_source_groups": len({row["source_group"] for row in selected}),
            "geometry_wall_sec_completed_only": _distribution([row["geometry_wall_sec"] for row in qualified]),
        }
    baseline = {(row["case"], row["repeat"]): row for row in rows if row["condition"] == manifest["baseline"]}
    pairs = {}
    for name in sorted(conditions.keys() - {manifest["baseline"]}):
        eligible, case_deltas = [], defaultdict(list)
        for row in rows:
            previous = baseline.get((row["case"], row["repeat"]))
            if row["condition"] != name or previous is None:
                continue
            if not all(item["status"] == "completed" and item["cleanup_confirmed"] and item["execution_identity_verified"]
                       for item in (previous, row)):
                continue
            eligible.append((previous, row))
            case_deltas[row["case"]].append(row["geometry_wall_sec"] - previous["geometry_wall_sec"])
        pairs[name] = {"paired_completed": len(eligible), "paired_unique_inputs": len(case_deltas),
                       "candidate_coverage_gained": sum(not a["candidate_present"] and b["candidate_present"] for a, b in eligible),
                       "candidate_coverage_lost": sum(a["candidate_present"] and not b["candidate_present"] for a, b in eligible),
                       "per_input_median_wall_delta_sec": {key: statistics.median(value) for key, value in case_deltas.items()},
                       "faster_pairs": sum(b["geometry_wall_sec"] < a["geometry_wall_sec"] for a, b in eligible)}
    return {"schema": "agenticrobot.grasp_geometry_benchmark_report.v1", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "baseline": manifest["baseline"], "conditions": totals, "paired_to_baseline": pairs, "rows": rows,
            "scope": "geometry only; no model latency, reachability, physical task success or independent-trial claim",
            "task_success_assessed": False, "source_files_modified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = json.dumps(review(args.manifest), indent=2, allow_nan=False) + "\n"
    # Encode and validate all evidence before publishing, never overwrite reports.
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, prefix=".benchmark-", suffix=".tmp") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.link(stream.name, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
