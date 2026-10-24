"""Declarative scene expansion, existing entrypoints and evidence separation."""

import hashlib
import json
import numpy as np
import pytest

from project_paths import resolve_path
from sim.mujoco.scene_suite import build_case, load_suite, materialize_case
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig


SUITE = load_suite()
CASES = [item["id"] for item in SUITE["cases"]]
VARIANTS = range(len(SUITE["variants_xy_m"]))


@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_materialized_suite_uses_existing_contract_and_task(tmp_path, case_id, variant):
    path = materialize_case(case_id, tmp_path / case_id, variant=variant)
    task = CubeLiftTaskConfig.load(path)
    contract = task.base_contract()
    original = CubeLiftTaskConfig.load(SUITE["base_task"])
    assert contract.scene.schema_version == "dexteleop.sim.scene_spec.v2"
    assert contract.scene.collision == original.base_contract().scene.collision
    assert contract.timing == original.base_contract().timing
    assert task.left_home_qpos == original.left_home_qpos
    assert task.right_home_qpos == original.right_home_qpos
    assert task.position_actuation_profile_name == original.position_actuation_profile_name
    assert task.gripper_actuation_profile_name == original.gripper_actuation_profile_name
    assert task.object_body_name == original.object_body_name
    assert not task.scripted_contact_reset and not task.scripted_pregrasp_reset
    assert not task.randomizes_cube_spawn
    target = next(o for o in contract.scene.objects if o.name == task.object_body_name)
    assert task.cube_spawn_qpos == pytest.approx(
        (*target.initial_position_m, *target.initial_quaternion_wxyz)
    )
    public = json.loads((path.parent / "public_tasks.json").read_text())
    private = json.loads((path.parent / "evaluation_only.json").read_text())
    assert len(public) == len(private) == 3
    assert all(set(item) == {"id", "phase", "instruction"} for item in public)
    assert {item["id"] for item in public} == set(private)
    names = {obj.name for obj in contract.scene.objects}
    for check in private.values():
        assert all(check[key] in names for key in ("target", "reference", "destination")
                   if key in check)
    assert task.base_contract_sha256 == hashlib.sha256(
        (path.parent / "scene.json").read_bytes()
    ).hexdigest()


def test_suite_adds_distinct_shapes_layouts_and_task_families():
    assert {p["shape"] for p in SUITE["prototypes"].values()} == {
        "box", "cylinder", "sphere", "ellipsoid",
    }
    built = [build_case(case_id, variant=v) for case_id in CASES for v in VARIANTS]
    scenes = [json.dumps(c.to_dict(), sort_keys=True) for c, *_ in built]
    assert len(scenes) == len(set(scenes)) == 39
    phases = {t["phase"] for _, _, tasks, _ in built for t in tasks}
    assert phases == {"observation", "safe_motion", "payload_pending"}
    assert any("sequential_bilateral_hovers" == t["kind"]
               for *_, checks in built for t in checks.values())
    assert any("stable_placement" == t["kind"]
               for *_, checks in built for t in checks.values())


def test_materializer_refuses_to_overwrite_retained_case(tmp_path):
    folder = tmp_path / "case"
    materialize_case(CASES[0], folder)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(FileExistsError):
        materialize_case(CASES[0], folder)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def _modified_suite(tmp_path, edit):
    suite = json.loads(json.dumps(SUITE))
    suite["base_task"] = str(resolve_path(suite["base_task"]))
    edit(suite)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite))
    return path


@pytest.mark.parametrize("variant", [-1, 3, True, 1.5])
def test_invalid_variant_fails_before_output_creation(tmp_path, variant):
    with pytest.raises(ValueError, match="Variant"):
        materialize_case(CASES[0], tmp_path / "output", variant=variant)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fault,match", [
    ("focus_yaw", "unrotated"),
    ("missing_evaluation_target", "absent object"),
    ("duplicate_objects", "Duplicate"),
    ("duplicate_cases", "unique"),
    ("traversal", "identifier"),
])
def test_invalid_case_cannot_silently_change_reset_or_public_task(tmp_path, fault, match):
    def edit(suite):
        case = suite["cases"][0]
        if fault == "focus_yaw":
            case["objects"][0]["yaw_deg"] = 10
        elif fault == "missing_evaluation_target":
            case["tasks"][0]["evaluation"]["target"] = "absent"
        elif fault == "duplicate_objects":
            case["objects"].append(case["objects"][0])
        elif fault == "duplicate_cases":
            suite["cases"].append(case)
        else:
            case["id"] = "../outside"

    path = _modified_suite(tmp_path, edit)
    with pytest.raises(ValueError, match=match):
        build_case(CASES[0], suite_path=path)


def test_support_placement_is_explicit_and_cannot_overhang(tmp_path):
    contract, task, *_ = build_case("raised_support")
    target, support = (next(o for o in contract.scene.objects if o.name == name)
                       for name in ("debug_cube", "gray_support"))
    assert target.initial_position_m[2] - target.size_m[2] / 2.0 == pytest.approx(
        support.initial_position_m[2] + support.size_m[2] / 2.0 + .001
    )
    assert task["table_top_z_m"] == pytest.approx(.066)

    def edit(suite):
        case = next(c for c in suite["cases"] if c["id"] == "raised_support")
        case["objects"][1]["xy_m"][0] -= .08

    with pytest.raises(ValueError, match="beyond its support"):
        build_case("raised_support", suite_path=_modified_suite(tmp_path, edit))


@pytest.mark.integration
@pytest.mark.parametrize("case_id", CASES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_native_suite_initial_geometry_uses_original_qualification(
    tmp_path, monkeypatch, record_property, case_id, variant,
):
    import mujoco
    from scripts.probes.qualify_mujoco_scene import inspect_prepared_scene
    from sim.mujoco.deploy.target_config import MujocoSimHILTargetConfig
    from sim.mujoco.deploy.target_preparation import prepare_mujoco_sim_hil_target
    from sim.mujoco.scene import MujocoCameraRenderer

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Initial qualification must not step or render.")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(MujocoCameraRenderer, "__init__", forbidden)
    path = materialize_case(case_id, tmp_path / "case", variant=variant)
    prepared = prepare_mujoco_sim_hil_target(MujocoSimHILTargetConfig(
        task_config_path=path, realtime_worker=False, allow_scripted_task_reset=False,
    ))
    report = inspect_prepared_scene(prepared)
    record_property("initial_geometry", json.dumps(report))
    assert report["qualified"], report
    assert report["time_qpos_qvel_unchanged"]
    assert not report["execution_owner_created"] and not report["renderer_created"]
    assert not report["motion_executed"]
    for obj in prepared.task_config.base_contract().scene.objects:
        np.testing.assert_allclose(
            prepared.scene.body_pose(obj.name)[:3, 3], obj.initial_position_m,
            rtol=0, atol=1e-12,
        )


@pytest.mark.integration
@pytest.mark.parametrize("case_id", ["shape_palette", "dense_clutter"])
def test_unanchored_ellipsoid_settles_through_original_physical_reset(tmp_path, case_id):
    from sim.mujoco.tasks.cube_lift.kernel import CubeLiftKernel

    path = materialize_case(case_id, tmp_path / "case")
    kernel = CubeLiftKernel(path, use_settled_snapshot=False, use_model_cache=False)
    try:
        assert kernel._grasp_latch is None
        kernel.reset(seed=0)
        scene = kernel.scene
        obj = next(o for o in kernel.config.base_contract().scene.objects
                   if o.name == "yellow_ellipsoid")
        position = scene.body_pose(obj.name)[:3, 3]
        assert np.linalg.norm(position - obj.initial_position_m) <= .005
        dof = int(scene.model.jnt_dofadr[scene.model.joint(obj.name + "_freejoint").id])
        velocity = scene.data.qvel[dof:dof + 6]
        assert np.linalg.norm(velocity[:3]) <= .01
        assert np.linalg.norm(velocity[3:]) <= .1
        assert kernel._episode_steps == 0
        assert kernel.last_physics_ticks == kernel.config.settle_physics_ticks
        assert not np.any(scene.model.eq_type == scene.mujoco.mjtEq.mjEQ_WELD)
        assert scene.collision_projector().oracle.check_contacts(scene.data).safe
        for contact in scene.contacts():
            if "yellow_ellipsoid_collision" in (contact.geom1_name, contact.geom2_name):
                assert contact.distance_m >= -scene.collision_projector().policy.hard_penetration_m
    finally:
        kernel.close()
        assert kernel._actuated_rig is None
