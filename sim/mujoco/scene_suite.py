"""Materialize opt-in Skill scenes into the existing task/scene entrypoints.

This module generates configuration only. It owns no renderer, simulator,
observation, execution loop, success oracle, or model prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from project_paths import resolve_path
from sim.contracts import FreeBodyObjectSpec, SimulationContract
from sim.contracts.scene import SCENE_SPEC_VERSION_V2


DEFAULT_SUITE = "sim/configs/scene_suites/rm65_dmtac_skill_v1.json"
_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_PHASES = {"observation", "safe_motion", "payload_pending"}


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lower-case identifier: {value!r}")
    return value


def load_suite(path: str | Path = DEFAULT_SUITE) -> dict[str, Any]:
    """Read declarative fixtures; source geometry is never an Agent input."""

    suite = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list):
        raise ValueError("Scene suite must contain a cases list.")
    _name(suite["name"], "suite name")
    identifiers = [_name(case["id"], "case id") for case in suite["cases"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Scene suite case identifiers must be unique.")
    return suite


def build_case(
    case_id: str, *, variant: int = 0, suite_path: str | Path = DEFAULT_SUITE,
) -> tuple[SimulationContract, dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    """Build an ordinary SceneSpec v2 contract and existing task overrides.

    Positions are authored in MuJoCo world, as in the source scene contract.
    Runtime goals still come from RGB-D through the shared frame graph.
    """

    from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

    suite = load_suite(suite_path)
    _name(case_id, "case id")
    case = next((item for item in suite["cases"] if item["id"] == case_id), None)
    if case is None:
        raise ValueError(f"Unknown scene case {case_id!r}.")
    variants = suite["variants_xy_m"]
    if type(variant) is not int or not 0 <= variant < len(variants):
        raise ValueError("Variant must index the suite's fixed offsets.")
    offset_x, offset_y = variants[variant]
    if not all(math.isfinite(float(x)) for x in (offset_x, offset_y)):
        raise ValueError("Variant offsets must be finite.")
    base_path = resolve_path(suite["base_task"])
    base_task = CubeLiftTaskConfig.load(base_path)
    base = base_task.base_contract()
    material = base.scene.objects[0].contact
    focus = _name(case["focus"], "focus object")
    objects: dict[str, FreeBodyObjectSpec] = {}
    support_heights: dict[str, float] = {}
    yaws: dict[str, float] = {}
    for entry in case["objects"]:
        name = _name(entry["name"], "object name")
        if name in objects:
            raise ValueError(f"Duplicate scene object {name!r}.")
        prototype = suite["prototypes"][entry["prototype"]]
        shape = prototype["shape"]
        size = tuple(prototype["size_m"])
        x, y = entry["xy_m"]
        yaw = math.radians(float(entry.get("yaw_deg", 0.0)))
        if not math.isfinite(yaw):
            raise ValueError("Object yaw must be finite.")
        support = entry.get("support", "table")
        height = float(base_task.table_top_z_m)
        if support != "table":
            if support not in objects:
                raise ValueError("Object support must precede the supported object.")
            top = objects[support]
            if top.shape != "box" or yaws[support] != 0.0:
                raise ValueError("Fixture supports must be axis-aligned boxes.")
            height = top.initial_position_m[2] + top.size_m[2] / 2.0
            # The complete horizontal projection must fit on the support.
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            half = ((c * size[0] + s * size[1]) / 2.0,
                    (s * size[0] + c * size[1]) / 2.0)
            for coordinate, center, radius, extent in zip(
                (x + offset_x, y + offset_y), top.initial_position_m[:2],
                half, top.size_m[:2],
            ):
                if abs(coordinate - center) + radius > extent / 2.0 + 1e-12:
                    raise ValueError(f"Object {name!r} extends beyond its support.")
        # Keep the original contact-profile body binding behind the scene
        # adapter. A semantic label does not need to be a simulator body ID.
        native_name = base_task.object_body_name if name == focus else name
        obj = FreeBodyObjectSpec(
            name=native_name, shape=shape, size_m=size,
            initial_position_m=(float(x) + offset_x, float(y) + offset_y,
                                height + size[2] / 2.0 + base_task.spawn_clearance_m),
            initial_quaternion_wxyz=(math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)),
            mass_kg=prototype["mass_kg"], rgba=tuple(prototype["rgba"]),
            contact=material,
        )
        objects[name] = obj
        support_heights[name] = height
        yaws[name] = yaw
    if focus not in objects:
        raise ValueError("Scene focus must reference an existing object.")
    if yaws[focus] != 0.0:
        raise ValueError("The existing task reset requires an unrotated focus object.")
    primary = objects[focus]
    label = f"{suite['name']}_{case_id}_v{variant}"
    contract = replace(
        base, name=label,
        scene=replace(base.scene, schema_version=SCENE_SPEC_VERSION_V2,
                      profile_name=label, objects=tuple(objects.values())),
        metadata={**base.metadata, "fixture_suite": suite["name"],
                  "fixture_case": case_id, "fixture_variant": variant,
                  "layout_frame": "mujoco_world", "parameter_role": "synthetic_fixtures",
                  "physical_task_success": "not_assessed"},
    )
    overrides = {
        "extends": str(base_path), "name": label,
        "object_body_name": primary.name,
        "object_freejoint_name": primary.name + "_freejoint",
        "object_size_m": list(primary.size_m),
        "cube_center_xy_m": list(primary.initial_position_m[:2]),
        "table_top_z_m": support_heights[focus],
        "cube_spawn_xy_jitter_m": [0.0, 0.0], "cube_spawn_yaw_jitter_rad": 0.0,
        "scripted_pregrasp_reset": False, "scripted_contact_reset": False,
    }
    public_tasks = []
    evaluators = {}
    for task in case["tasks"]:
        task_id = _name(task["id"], "task id")
        if task_id in evaluators:
            raise ValueError("Task identifiers must be unique within a case.")
        phase = task["phase"]
        if phase not in _PHASES:
            raise ValueError(f"Unknown task phase {phase!r}.")
        instruction = task["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Task instruction must be non-empty text.")
        checks = dict(task["evaluation"])
        for key in ("target", "reference", "destination"):
            if key in checks and checks[key] not in objects:
                raise ValueError(f"Evaluation {key} names an absent object.")
            if key in checks:
                checks[key] = objects[checks[key]].name
        public_tasks.append({"id": task_id, "phase": phase, "instruction": instruction})
        evaluators[task_id] = checks
    return contract, overrides, public_tasks, evaluators


def materialize_case(
    case_id: str, output: str | Path, *, variant: int = 0,
    suite_path: str | Path = DEFAULT_SUITE,
) -> Path:
    """Write fresh configs, keeping evaluation geometry outside public tasks."""

    contract, task, public, evaluators = build_case(
        case_id, variant=variant, suite_path=suite_path,
    )
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene_bytes = (json.dumps(contract.to_dict(), indent=2, allow_nan=False) + "\n").encode()
    (output / "scene.json").write_bytes(scene_bytes)
    task.update(base_contract_path=str(output / "scene.json"),
                base_contract_sha256=hashlib.sha256(scene_bytes).hexdigest())
    for name, payload in (("task.json", task), ("public_tasks.json", public),
                          ("evaluation_only.json", evaluators)):
        (output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return output / "task.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path(DEFAULT_SUITE))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--variant", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.list:
        for case in load_suite(args.suite)["cases"]:
            print(case["id"])
        return 0
    if args.case is None or args.output is None:
        parser.error("--case and --output are required when generating a case")
    print(materialize_case(args.case, args.output, variant=args.variant, suite_path=args.suite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
