"""Read-only audit of one compiled MuJoCo collision/contact model.

The report is intentionally task-reward agnostic. It describes the compiled
geometry, contact parameters, solver configuration, CCD state, and forbidden
pair oracle without stepping policy actions or changing safety authority.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sim.mujoco.modeling.collision import (
    ENVIRONMENT_GROUP,
    GRIPPER_GROUP,
    OBJECT_GROUP,
    ROBOT_GROUP,
)
from sim.mujoco.tasks.cube_lift import FastCubeLiftEnv


COLLISION_GROUP_BY_BIT = {
    1: ENVIRONMENT_GROUP,
    2: ROBOT_GROUP,
    4: GRIPPER_GROUP,
    8: OBJECT_GROUP,
}
COLLISION_AUDIT_SCHEMA = "dexteleop.sim.mujoco.collision_model_audit.v1"


def _enum_name(enum_type: Any, value: int) -> str:
    try:
        return str(enum_type(int(value)).name)
    except (TypeError, ValueError):
        return f"unknown_{int(value)}"


def _rounded_tuple(values: Any) -> tuple[float, ...]:
    return tuple(float(value) for value in np.asarray(values).reshape(-1))


def audit_collision_model(
    mujoco: Any,
    model: Any,
    *,
    forbidden_pair_count: int | None = None,
) -> dict[str, object]:
    """Return JSON-safe facts for an already compiled MuJoCo model."""

    geom_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    collision_mesh_geoms: dict[str, list[str]] = defaultdict(list)
    material_counts: dict[
        str,
        Counter[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], int]],
    ] = defaultdict(Counter)
    collision_geom_count = 0
    visual_only_geom_count = 0

    for geom_id in range(int(model.ngeom)):
        contype = int(model.geom_contype[geom_id])
        if contype == 0:
            visual_only_geom_count += 1
            continue
        collision_geom_count += 1
        group = COLLISION_GROUP_BY_BIT.get(contype, f"custom_{contype}")
        geom_type = _enum_name(mujoco.mjtGeom, int(model.geom_type[geom_id]))
        geom_type_counts[group][geom_type] += 1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_type == "mjGEOM_MESH":
            collision_mesh_geoms[group].append(
                f"<geom:{geom_id}>" if name is None else str(name)
            )
        key = (
            _rounded_tuple(model.geom_friction[geom_id]),
            _rounded_tuple(model.geom_solref[geom_id]),
            _rounded_tuple(model.geom_solimp[geom_id]),
            int(model.geom_condim[geom_id]),
        )
        material_counts[group][key] += 1

    disableflags = int(model.opt.disableflags)
    multiccd_bit = getattr(mujoco.mjtDisableBit, "mjDSBL_MULTICCD", None)
    multiccd_disabled = (
        None
        if multiccd_bit is None
        else bool(disableflags & int(multiccd_bit))
    )
    material_profiles: dict[str, list[dict[str, object]]] = {}
    for group, counter in sorted(material_counts.items()):
        material_profiles[group] = [
            {
                "geom_count": int(count),
                "friction": list(key[0]),
                "solref": list(key[1]),
                "solimp": list(key[2]),
                "condim": int(key[3]),
            }
            for key, count in sorted(
                counter.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

    return {
        "schema": COLLISION_AUDIT_SCHEMA,
        "mujoco_version": str(getattr(mujoco, "__version__", "unknown")),
        "model": {
            "body_count": int(model.nbody),
            "geom_count": int(model.ngeom),
            "collision_geom_count": collision_geom_count,
            "visual_only_geom_count": visual_only_geom_count,
            "forbidden_pair_count": (
                None if forbidden_pair_count is None else int(forbidden_pair_count)
            ),
        },
        "solver": {
            "timestep_sec": float(model.opt.timestep),
            "integrator": _enum_name(
                mujoco.mjtIntegrator,
                int(model.opt.integrator),
            ),
            "solver": _enum_name(mujoco.mjtSolver, int(model.opt.solver)),
            "iterations": int(model.opt.iterations),
            "line_search_iterations": int(model.opt.ls_iterations),
            "ccd_iterations": int(model.opt.ccd_iterations),
            "ccd_tolerance": float(model.opt.ccd_tolerance),
            "disableflags": disableflags,
            "multiccd_runtime_flag_available": multiccd_bit is not None,
            "multiccd_enabled": (
                None if multiccd_disabled is None else not multiccd_disabled
            ),
        },
        "collision_geom_type_counts": {
            group: dict(sorted(counts.items()))
            for group, counts in sorted(geom_type_counts.items())
        },
        "collision_mesh_geoms": {
            group: sorted(names)
            for group, names in sorted(collision_mesh_geoms.items())
        },
        "contact_material_profiles": material_profiles,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile one task scene and print a read-only, task-reward-agnostic "
            "MuJoCo collision/contact audit."
        )
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        required=True,
        help="Task config used only to compile the exact runtime scene/profile.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-cache-dir", type=Path, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    import mujoco

    env = FastCubeLiftEnv(
        args.task_config,
        info_mode="compact",
        model_cache_dir=args.model_cache_dir,
    )
    try:
        # Reset applies the exact post-settle simulator-only contact profile
        # used by policy rollouts; no policy action is executed.
        env.reset(seed=int(args.seed))
        projector = env.kernel.scene.collision_projector()
        report = audit_collision_model(
            mujoco,
            env.kernel.model,
            forbidden_pair_count=(
                None if projector is None else projector.oracle.forbidden_pair_count
            ),
        )
        return {
            **report,
            "task_config": str(args.task_config),
            "task_name": env.config.name,
            "policy_action_layout": env.config.policy_action_layout,
            "actor_observation_mode": env.config.observation_mode,
        }
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (FileNotFoundError, KeyError, ModuleNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COLLISION_AUDIT_SCHEMA",
    "audit_collision_model",
    "build_arg_parser",
    "run",
]
