"""Diagnostic contacts retain forces and geometry without granting pad validity."""
from types import SimpleNamespace

import numpy as np
import pytest

from sim.mujoco.grasp_contact import GripperContactSurfaceSpec
from sim.mujoco.grasp_contact_diagnostics import MujocoGraspContactSampler

pytestmark = pytest.mark.unit


def _fixture(*, point=(0.01, 0., 0.), force=2., efc_address=0, distance=-.0001):
    model = SimpleNamespace(nbody=3, ngeom=2, geom_bodyid=np.array([1, 2]),
        geom_type=np.array([6, 6]), geom_size=np.array([[.01, .02, .03], [.025]*3]))
    contact = SimpleNamespace(geom1=0, geom2=1,
        pos=np.array(point) + .5*distance*np.array([1., 0., 0.]),
        frame=np.eye(3).reshape(-1), dist=distance, efc_address=efc_address)
    data = SimpleNamespace(ncon=1, contact=[contact], xpos=np.zeros((3, 3)),
        xmat=np.tile(np.eye(3).reshape(-1), (3, 1)), geom_xpos=np.zeros((2, 3)),
        geom_xmat=np.tile(np.eye(3).reshape(-1), (2, 1)), qpos=np.array([.2]))
    force_calls = []
    def read_force(_model, _data, index, output):
        force_calls.append(index)
        output[:] = [force, .1, .2, 0., 0., 0.]
    mj = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1, mjOBJ_GEOM=2),
        mjtGeom=SimpleNamespace(mjGEOM_BOX=6), mj_contactForce=read_force,
        mj_id2name=lambda _m, kind, i: (["world", "finger", "object"][i]
            if kind == 1 else ["finger_box", "object_box"][i]))
    scene = SimpleNamespace(model=model, data=data, mujoco=mj,
        body_id=lambda name: {"object": 2, "finger": 1}[name])
    spec = GripperContactSurfaceSpec("finger", "left", (1., 0., 0.))
    sampler = MujocoGraspContactSampler(scene, object_body_name="object", surface_specs=(spec,))
    return scene, sampler, force_calls


def test_readonly_sampler_records_actual_force_without_claiming_a_tactile_pad():
    scene, sampler, calls = _fixture()
    before = scene.data.qpos.copy()
    record, = sampler.sample()
    assert record["region"] == "inward_collision_box_face"
    assert record["load_bearing"] is True and record["normal_force_n"] == 2.
    assert record["physical_tactile_pad_membership"] == "unknown"
    assert record["point_other_body_local_m"] == [.01, 0., 0.]
    assert record["position_world_m"] != record["surface_point_other_world_m"]
    assert record["contact_frame_rows_world"] == np.eye(3).tolist()
    assert record["force_on_object_world_n"] == [2., .1, .2]
    assert calls == [0]
    assert np.array_equal(before, scene.data.qpos)


@pytest.mark.parametrize("point,expected", [
    ((-.01, 0., 0.), "other_collision_box_face"),
    ((.01, .02, 0.), "collision_box_edge"),
    ((0., 0., 0.), "collision_box_surface_unresolved"),
    ((.01, .05, 0.), "collision_box_surface_unresolved"),
])
def test_finite_box_face_is_not_an_unbounded_inner_half_space(point, expected):
    _, sampler, _ = _fixture(point=point)
    assert sampler.sample()[0]["region"] == expected


def test_inactive_contact_does_not_count_as_load_bearing_or_read_constraint_force():
    _, sampler, calls = _fixture(efc_address=-1)
    record, = sampler.sample()
    assert record["active_constraint"] is False and record["load_bearing"] is False
    assert record["normal_force_n"] == 0. and calls == []


def test_reversed_geom_order_retains_the_same_finger_identity():
    scene, sampler, _ = _fixture()
    scene.data.contact[0].geom1, scene.data.contact[0].geom2 = 1, 0
    scene.data.contact[0].frame = np.diag([-1., -1., 1.]).reshape(-1)
    record, = sampler.sample()
    assert record["finger_side"] == "left" and record["other_geom"] == "finger_box"
    assert record["region"] == "inward_collision_box_face"
    assert record["point_other_body_local_m"] == [.01, 0., 0.]
    assert record["force_on_object_world_n"] == [2., .1, -.2]


@pytest.mark.parametrize("distance", [-.006, -.00249, 0., .002])
def test_signed_midpoint_correction_retains_the_surface_under_penetration_and_gap(distance):
    _, sampler, _ = _fixture(distance=distance)
    record, = sampler.sample()
    assert record["region"] == "inward_collision_box_face"
    assert record["surface_point_other_world_m"] == pytest.approx([.01, 0., 0.])


def test_rotated_translated_box_retains_local_surface_and_world_wrench():
    scene, sampler, _ = _fixture(distance=-.003)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([.3, .4, .5])
    scene.data.xpos[1] = scene.data.geom_xpos[0] = translation
    scene.data.xmat[1] = scene.data.geom_xmat[0] = rotation.reshape(-1)
    contact = scene.data.contact[0]
    contact.pos = rotation @ contact.pos + translation
    contact.frame = rotation.T.reshape(-1)
    record, = sampler.sample()
    assert record["point_other_body_local_m"] == pytest.approx([.01, 0., 0.])
    assert record["region"] == "inward_collision_box_face"
    assert record["force_on_object_world_n"] == pytest.approx(rotation @ [2., .1, .2])


def test_other_body_is_reported_but_never_relabelled_as_a_finger():
    scene, sampler, _ = _fixture()
    scene.model.geom_bodyid[0] = 0
    record, = sampler.sample()
    assert record["region"] == "not_configured_finger" and record["finger_side"] is None


@pytest.mark.parametrize("force", [float("nan"), float("inf")])
def test_nonfinite_native_force_is_a_diagnostic_fault(force):
    _, sampler, _ = _fixture(force=force)
    with pytest.raises(ValueError, match="nonfinite"):
        sampler.sample()


@pytest.mark.parametrize("value", [True, 0., -1., float("nan")])
def test_invalid_force_threshold_is_rejected(value):
    scene, _, _ = _fixture()
    with pytest.raises(ValueError, match="minimum_normal_force"):
        MujocoGraspContactSampler(scene, object_body_name="object", surface_specs=(
            GripperContactSurfaceSpec("finger", "left", (1., 0., 0.)),),
            minimum_normal_force_n=value)


def test_crossing_grasps_preserves_the_declared_model_pivot_not_the_old_tcp():
    from scipy.spatial.transform import Rotation

    from scripts.probes.diagnose_grasp_closure import crossed_model_target
    center, orientation, tool_from_model = np.eye(4), np.eye(4), np.eye(4)
    center[:3, 3] = [.1, .2, .3]
    orientation[:3, :3] = Rotation.from_euler("y", 20., degrees=True).as_matrix()
    orientation[:3, 3] = [.5, .6, .7]
    tool_from_model[:3, 3] = [.02, 0., .03]
    tool_from_model[:3, :3] = Rotation.from_euler("xyz", [90., 20., -30.], degrees=True).as_matrix()
    model, tcp = crossed_model_target(center, orientation, tool_from_model)
    assert np.allclose(model[:3, 3], center[:3, 3])
    assert np.allclose(model[:3, :3], orientation[:3, :3])
    assert np.allclose(tcp @ tool_from_model, model)
    assert not np.allclose(tcp[:3, 3], center[:3, 3])
    assert orientation[:3, 3].tolist() == [.5, .6, .7]


def test_factorial_fixture_rejects_nonrigid_or_nonfinite_poses():
    from scripts.probes.diagnose_grasp_closure import crossed_model_target
    bad = np.eye(4)
    bad[0, 0] = 2.
    with pytest.raises(ValueError, match="pose"):
        crossed_model_target(bad, np.eye(4), np.eye(4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="pose"):
        crossed_model_target(bad, np.eye(4), np.eye(4))


@pytest.mark.parametrize("fault", [ValueError("Fixture FK failed"), KeyboardInterrupt()])
def test_probe_retains_early_failure_and_interrupt_without_hiding_it(tmp_path, monkeypatch, fault):
    import json

    from scripts.probes import diagnose_grasp_closure as probe
    def fail(*_, **__):
        raise fault
    monkeypatch.setattr(probe, "_build_scene", fail)
    fixture = tmp_path/"fixture.json"
    fixture.write_text("{}")
    with pytest.raises(type(fault)):
        probe.run_case(fixture, tmp_path/"out", center_key="a", orientation_key="b")
    report = json.loads((tmp_path/"out/result.json").read_text())
    assert report["error"]["type"] == type(fault).__name__
    assert report["cleanup_complete"] is True
    assert report["status"] == ("incomplete_or_interrupted" if isinstance(fault, KeyboardInterrupt) else "diagnostic_fault")


@pytest.mark.parametrize("prefix", [
    "MuJoCo physical execution found forbidden hard contact",
    "MuJoCo physical execution entered forbidden hard contact after physics step",
])
def test_native_rejections_retain_original_outcome_and_all_cleanup_faults(tmp_path, monkeypatch, prefix):
    import json

    from scripts.probes import diagnose_grasp_closure as probe
    closed = []
    class Resource:
        def __init__(self, name):
            self.name = name
        def close(self):
            closed.append(self.name)
            raise OSError(self.name + " close failed")
    def fail(_fixture, _output, _report, register, **_kwargs):
        for name in ("rig", "trace", "renderer"):
            register(name, Resource(name))
        raise RuntimeError(prefix + ": original reason")
    monkeypatch.setattr(probe, "_run_case_contents", fail)
    report = probe.run_case("fixture.json", tmp_path/"out", center_key="a", orientation_key="b")
    assert report["primary_outcome"] == "native_execution_rejected"
    assert report["status"] == "diagnostic_fault" and report["cleanup_complete"] is False
    assert report["error"]["message"] == prefix + ": original reason"
    assert closed == ["renderer", "trace", "rig"]
    assert len(json.loads((tmp_path/"out/result.json").read_text())["cleanup_errors"]) == 3


def test_unexpected_fault_is_not_replaced_by_cleanup_error(tmp_path, monkeypatch):
    from scripts.probes import diagnose_grasp_closure as probe
    class BadRenderer:
        def close(self):
            raise OSError("close failed")
    def fail(_fixture, _output, _report, register, **_kwargs):
        register("renderer", BadRenderer())
        raise ValueError("primary render failed")
    monkeypatch.setattr(probe, "_run_case_contents", fail)
    with pytest.raises(ValueError, match="primary render failed"):
        probe.run_case("fixture.json", tmp_path/"out", center_key="a", orientation_key="b")
