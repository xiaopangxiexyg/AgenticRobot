"""Fixed evidence, paired denominators, and explicit geometry-only scoring."""
import copy
import hashlib
import json

import pytest

from scripts.probes.benchmark_grasp_geometry import condition_identity, main, review


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def matrix(tmp_path):
    cases = []
    frozen_cases, pins = [], {}
    for index in range(2):
        files = {key: write(tmp_path / f"{index}-{key}.json", [index, key])
                 for key in ("npz", "metadata", "mask")}
        cases.append({"id": f"case{index}", "domain": "mujoco_rgbd", "source_group": "same_layout",
                      "view": "front", "prompt": "blue cube", "input_files": files})
        pins.update({item["path"]: item["sha256"] for item in files.values()})
        for repeat in range(2):
            frozen_cases.append({"id": f"case{index}-r{repeat}", "view": "front", "prompt": "blue cube",
                                 **{key: item["path"] for key, item in files.items()}})
    frozen = write(tmp_path / "freeze.json", {"cases": frozen_cases, "fixed_files": pins})
    manifest = {"schema": "agenticrobot.grasp_geometry_benchmark.v1", "baseline": "base", "repeats": 2,
                "cases": cases, "conditions": [], "slots": []}
    for name in ("base", "new"):
        condition = {"id": name, "geometry_profile": "v5", "fixed_files": [frozen],
                     "parameters": {"variant": name}}
        records = [{"slot": case["id"], "status": "completed", "geometry_profile": "v5",
                    "condition_sha256": condition_identity(condition),
                    "retained_pool": int(name == "new" or "case0" in case["id"]),
                    "wall_sec": 2. if name == "base" else 1.} for case in frozen_cases]
        evidence = write(tmp_path / f"{name}.json", {"native_rig_started": False,
                         "public_candidate_identity_created": False, "normal_cleanup": True,
                         "benchmark_binding": {"schema": "agenticrobot.grasp_geometry_execution_binding.v1",
                                               "freeze_sha256": frozen["sha256"]}, "records": records})
        manifest["conditions"].append(condition)
        for case in frozen_cases:
            case_id, repeat = case["id"].rsplit("-r", 1)
            manifest["slots"].append({"id": f"{name}-{case['id']}", "case": case_id,
                "condition": name, "repeat": int(repeat),
                "evidence": {**evidence, "record_slot": case["id"], "case_id": case["id"], "freeze": frozen}})
    path = tmp_path / "matrix.json"
    write(path, manifest)
    return path, manifest


def test_fixed_paired_report_separates_repeats_sources_and_task_success(tmp_path):
    path, _ = matrix(tmp_path)
    result = review(path)
    assert not result["task_success_assessed"] and not result["source_files_modified"]
    assert result["conditions"]["base"]["declared"] == 4
    assert result["conditions"]["base"]["unique_inputs_with_candidates"] == 1
    assert result["conditions"]["new"]["unique_inputs_with_candidates"] == 2
    assert result["conditions"]["new"]["independent_source_groups"] == 1
    assert result["paired_to_baseline"]["new"] == {
        "paired_completed": 4, "paired_unique_inputs": 2, "candidate_coverage_gained": 2,
        "candidate_coverage_lost": 0, "per_input_median_wall_delta_sec": {"case0": -1., "case1": -1.},
        "faster_pairs": 4,
    }


def test_started_missing_result_and_unstarted_stay_in_denominator(tmp_path):
    path, manifest = matrix(tmp_path)
    manifest["slots"][4]["evidence"] = None
    manifest["slots"][4]["start_evidence"] = write(tmp_path / "started.json", {"slot": "new-case0-r0"})
    manifest["slots"][5]["evidence"] = None
    write(path, manifest)
    result = review(path)
    assert result["conditions"]["new"]["declared"] == 4
    assert result["conditions"]["new"]["attempted"] == 3
    assert result["conditions"]["new"]["status_counts"] == {"missing_result": 1, "unstarted": 1, "completed": 2}
    assert result["paired_to_baseline"]["new"]["paired_completed"] == 2


def test_omitting_a_failed_or_unstarted_slot_cannot_shrink_the_matrix(tmp_path):
    path, manifest = matrix(tmp_path)
    manifest["slots"].pop()
    write(path, manifest)
    with pytest.raises(ValueError, match="retain every declared"):
        review(path)


def test_changed_source_is_rejected_before_report_publication(tmp_path):
    path, manifest = matrix(tmp_path)
    source = tmp_path / "0-npz.json"
    source.write_text("changed")
    output = tmp_path / "output.json"
    with pytest.raises(ValueError, match="Evidence changed"):
        main(["--manifest", str(path), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("mutation,match", [
    ("reuse_record", "multiple repetitions"),
    ("duplicate_repeat", "Duplicate case/condition/repeat"),
    ("changed_prompt", "view/prompt differs"),
    ("swap_input", "Input identity differs"),
    ("duplicate_case", "same measured input"),
    ("wrong_profile", "Declared condition differs"),
])
def test_invalid_comparison_cannot_silently_become_a_pair(tmp_path, mutation, match):
    path, manifest = matrix(tmp_path)
    if mutation == "reuse_record":
        manifest["slots"][5]["evidence"] = copy.deepcopy(manifest["slots"][4]["evidence"])
    elif mutation == "duplicate_repeat":
        manifest["slots"][5]["repeat"] = 0
    elif mutation == "changed_prompt":
        manifest["cases"][0]["prompt"] = "red cylinder"
    elif mutation == "swap_input":
        manifest["cases"][0]["input_files"]["npz"] = manifest["cases"][1]["input_files"]["npz"]
    elif mutation == "duplicate_case":
        manifest["cases"][1]["input_files"] = copy.deepcopy(manifest["cases"][0]["input_files"])
    else:
        manifest["conditions"][1]["geometry_profile"] = "v4"
    write(path, manifest)
    with pytest.raises(ValueError, match=match):
        review(path)


@pytest.mark.parametrize("field,value,match", [
    ("wall_sec", float("nan"), "Non-finite"),
    ("wall_sec", -1, "finite non-negative"),
    ("retained_pool", True, "non-negative integer"),
    ("status", "physical_success", "Unknown recorded outcome"),
])
def test_invalid_metrics_are_not_coerced_to_success(tmp_path, field, value, match):
    path, manifest = matrix(tmp_path)
    report_path = tmp_path / "new.json"
    report = json.loads(report_path.read_text())
    report["records"][0][field] = value
    binding = write(report_path, report)
    for slot in manifest["slots"][4:]:
        slot["evidence"].update(binding)
    write(path, manifest)
    with pytest.raises(ValueError, match=match):
        review(path)


def test_missing_cleanup_excluded_from_comparable_subset_not_from_declared(tmp_path):
    path, manifest = matrix(tmp_path)
    report_path = tmp_path / "new.json"
    report = json.loads(report_path.read_text())
    report["normal_cleanup"] = False
    binding = write(report_path, report)
    for slot in manifest["slots"][4:]:
        slot["evidence"].update(binding)
    write(path, manifest)
    result = review(path)
    assert result["conditions"]["new"]["declared"] == result["conditions"]["new"]["attempted"] == 4
    assert result["conditions"]["new"]["cleanup_confirmed"] == 0
    assert result["paired_to_baseline"]["new"]["paired_completed"] == 0


def test_cli_does_not_overwrite_existing_report(tmp_path):
    path, _ = matrix(tmp_path)
    output = tmp_path / "out.json"
    assert main(["--manifest", str(path), "--output", str(output)]) == 0
    first = output.read_bytes()
    with pytest.raises(FileExistsError):
        main(["--manifest", str(path), "--output", str(output)])
    assert output.read_bytes() == first


def test_replacing_inputs_and_freeze_cannot_relabel_an_old_execution(tmp_path):
    path, manifest = matrix(tmp_path)
    old_freeze = json.loads((tmp_path / "freeze.json").read_text())
    replacement = write(tmp_path / "other-input.json", ["different pixels"])
    manifest["cases"][0]["input_files"]["npz"] = replacement
    for case in old_freeze["cases"]:
        if case["id"].startswith("case0-"):
            case["npz"] = replacement["path"]
    old_freeze["fixed_files"][replacement["path"]] = replacement["sha256"]
    changed_freeze = write(tmp_path / "other-freeze.json", old_freeze)
    for slot in manifest["slots"]:
        slot["evidence"]["freeze"] = changed_freeze
    write(path, manifest)
    with pytest.raises(ValueError, match="not bound to this frozen input"):
        review(path)


def test_byte_identical_report_copy_is_not_a_new_repeat(tmp_path):
    path, manifest = matrix(tmp_path)
    clone = tmp_path / "clone.json"
    clone.write_bytes((tmp_path / "new.json").read_bytes())
    replacement = copy.deepcopy(manifest["slots"][4]["evidence"])
    replacement["path"] = str(clone)
    manifest["slots"][5]["evidence"] = replacement
    write(path, manifest)
    with pytest.raises(ValueError, match="multiple repetitions"):
        review(path)


def test_unrelated_condition_pins_are_rejected(tmp_path):
    path, manifest = matrix(tmp_path)
    manifest["conditions"][1]["fixed_files"] = [write(tmp_path / "unrelated.json", {"version": "other"})]
    write(path, manifest)
    with pytest.raises(ValueError, match="Declared condition differs"):
        review(path)


def test_legacy_unbound_reports_are_descriptive_only_not_formal_pairs(tmp_path):
    path, manifest = matrix(tmp_path)
    report_path = tmp_path / "new.json"
    report = json.loads(report_path.read_text())
    report.pop("benchmark_binding")
    binding = write(report_path, report)
    for slot in manifest["slots"][4:]:
        slot["evidence"].update(binding)
    write(path, manifest)
    result = review(path)
    assert result["conditions"]["new"]["declared"] == 4
    assert result["conditions"]["new"]["candidate_present_descriptive_only"] == 4
    assert result["conditions"]["new"]["candidate_present"] == 0
    assert result["conditions"]["new"]["execution_identity_verified"] == 0
    assert result["paired_to_baseline"]["new"]["paired_completed"] == 0
