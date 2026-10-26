"""Real 3.10 native components; no robot hardware, model policy, or Gateway RPC."""

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.mujoco.native_checkpoint import (
    NativeCheckpointError,
    capture_native_state,
    restore_native_state,
)

pytestmark = pytest.mark.integration
_BUDGET = 2 * 1024 * 1024
_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <size nuserdata="3"/>
  <worldbody>
    <geom type="plane" size="1 1 .1"/>
    <body pos="-.2 0 .1"><joint name="left" type="slide" axis="1 0 0" damping=".1"/>
      <geom type="sphere" size=".06" mass=".2"/></body>
    <body pos=".2 0 .1"><joint name="right" type="slide" axis="1 0 0" damping=".1"/>
      <geom type="sphere" size=".06" mass=".2"/></body>
    <body pos="0 0 .2"><freejoint/><geom type="box" size=".03 .03 .03" mass=".1"/></body>
  </worldbody>
  <actuator>
    <general joint="left" dyntype="filterexact" dynprm=".02"/>
    <general joint="right" dyntype="filterexact" dynprm=".03"/>
  </actuator>
</mujoco>
"""


def _continuation_witness(data):
    # Future collision generation leaves C contact tail padding unspecified.
    # Compare every public field instead of allocator bytes. Immediate native
    # roundtrip below still requires byte-for-byte equality, including padding.
    omitted = {"model", "signature", "parena"}
    if not mujoco.mj_isSparse(data.model):
        # 3.10 engine_core_constraint.c does not write/read these in dense mode.
        omitted.update(
            ("efc_J_colind", "efc_J_rowadr", "efc_J_rownnz", "efc_J_rowsuper")
        )
    if not mujoco.mj_isDual(data.model):
        # Match the native serializer's MJDATA_ARENA_POINTERS_DUAL condition.
        # Python array getters over absent native pointers are not live state.
        omitted.update(
            (
                "efc_Y_rownnz",
                "efc_Y_rowadr",
                "efc_Y_colind",
                "efc_Y",
                "efc_AR_rownnz",
                "efc_AR_rowadr",
                "efc_AR_colind",
                "efc_AR",
            )
        )
    if not data.nisland:
        # Match MJDATA_ARENA_POINTERS_ISLAND, not a generic cache exclusion.
        omitted.update(
            (
                "tree_island",
                "island_ntree",
                "island_itreeadr",
                "map_itree2tree",
                "dof_island",
                "island_nv",
                "island_idofadr",
                "island_dofadr",
                "map_dof2idof",
                "map_idof2dof",
                "ifrc_smooth",
                "iacc_smooth",
                "iacc",
                "efc_island",
                "island_ne",
                "island_nf",
                "island_nefc",
                "island_iefcadr",
                "map_efc2iefc",
                "map_iefc2efc",
                "iefc_type",
                "iefc_id",
                "iefc_frictionloss",
                "iefc_D",
                "iefc_R",
                "iefc_aref",
                "iefc_state",
                "iefc_force",
                "ifrc_constraint",
            )
        )

    def array(value):
        return [
            value.dtype.str,
            list(value.shape),
            base64.b64encode(value.tobytes()).decode("ascii"),
        ]

    result = {"model_sha256": hashlib.sha256(data.model.__getstate__()).hexdigest()}
    for name in dir(data):
        if name.startswith("_") or name in omitted:
            continue
        value = getattr(data, name)
        if callable(value):
            continue
        if isinstance(value, np.ndarray):
            result[name] = array(value)
        elif type(value) in (int, float, bool):
            result[name] = value
        else:
            assert name in {"contact", "solver", "warning", "timer"}, name
            result[name] = {
                field: array(getattr(value, field))
                for field in dir(value)
                if not field.startswith("_")
                and not callable(getattr(value, field))
                and not (name == "timer" and field == "duration")
            }
    return result


@pytest.fixture(params=["dense", "sparse"])
def native(request):
    model = mujoco.MjModel.from_xml_string(
        _XML.replace("<option ", f'<option jacobian="{request.param}" ')
    )
    data = mujoco.MjData(model)
    data.userdata[:] = [1.0, 2.0, 3.0]
    data.qfrc_applied[:2] = [0.02, -0.01]
    for i in range(90):
        data.ctrl[:] = [np.sin(i * 0.1), np.cos(i * 0.1)]
        mujoco.mj_step(model, data)
    return data


def test_real_native_roundtrip_preserves_both_sides_and_pipeline_without_reset(
    native, monkeypatch
):
    before = native.__getstate__()
    state = capture_native_state(native, max_native_bytes=_BUDGET)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A restore must not recompute the native pipeline.")

    with monkeypatch.context() as patch:
        patch.setattr(mujoco, "mj_resetData", forbidden)
        patch.setattr(mujoco, "mj_forward", forbidden)
        patch.setattr(mujoco, "mj_step", forbidden)
        restored = restore_native_state(state, max_native_bytes=_BUDGET)
    assert restored is not native and restored.model is not native.model
    assert before == native.__getstate__() == restored.__getstate__()
    assert state.source_compilation_signatures == (
        native.model.signature,
        native.signature,
    )
    assert state.source_compilation_signatures[0] != 0
    assert (restored.model.signature, restored.signature) == (0, 0)
    assert not np.shares_memory(native.qpos, restored.qpos)
    assert (
        capture_native_state(restored, max_native_bytes=_BUDGET).auxiliary
        == state.auxiliary
    )
    for i in range(60):
        controls = [np.sin(i * 0.13), np.cos(i * 0.17)]
        native.ctrl[:] = restored.ctrl[:] = controls
        mujoco.mj_step(native.model, native)
        mujoco.mj_step(restored.model, restored)
        assert _continuation_witness(native) == _continuation_witness(restored)
    assert native.ncon > 0 and native.nefc > 0


def test_two_branches_do_not_mutate_source_or_each_other(native):
    state = capture_native_state(native, max_native_bytes=_BUDGET)
    first = restore_native_state(state, max_native_bytes=_BUDGET)
    second = restore_native_state(state, max_native_bytes=_BUDGET)
    source_bytes = native.__getstate__()
    for branch, value in ((first, 1.0), (second, -1.0)):
        for _ in range(20):
            branch.ctrl[:] = [value, -value]
            mujoco.mj_step(branch.model, branch)
    assert not np.array_equal(first.qpos, second.qpos)
    assert native.__getstate__() == source_bytes == state.data
    first.model.opt.gravity[:] = 0
    assert np.array_equal(native.model.opt.gravity, second.model.opt.gravity)
    assert not np.array_equal(first.model.opt.gravity, second.model.opt.gravity)


def test_native_component_survives_fresh_interpreter(native, tmp_path):
    state = capture_native_state(native, max_native_bytes=_BUDGET)
    payload = asdict(state)
    payload["data"] = base64.b64encode(state.data).decode("ascii")
    path = tmp_path / "native.json"
    path.write_text(json.dumps(payload))
    for i in range(12):
        native.ctrl[:] = [i / 20, -i / 20]
        mujoco.mj_step(native.model, native)
    expected = _continuation_witness(native)
    program = """
import base64, json, sys
import mujoco
from sim.mujoco.native_checkpoint import (
    NativeMujocoState, NativePipelineAuxiliary, restore_native_state,
)
from tests.test_mujoco_native_checkpoint import _continuation_witness
with open(sys.argv[1]) as stream:
    fields = json.load(stream)
fields['data'] = base64.b64decode(fields['data'], validate=True)
fields['integration'] = tuple(fields['integration'])
fields['native_abi'] = tuple(fields['native_abi'])
fields['source_compilation_signatures'] = tuple(fields['source_compilation_signatures'])
fields['auxiliary'] = NativePipelineAuxiliary(**{
    key: tuple(value) for key, value in fields['auxiliary'].items()
})
data = restore_native_state(NativeMujocoState(**fields), max_native_bytes=2 * 1024 * 1024)
for i in range(12):
    data.ctrl[:] = [i / 20, -i / 20]
    mujoco.mj_step(data.model, data)
print(json.dumps(_continuation_witness(data)))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    actual = json.loads(result.stdout)
    assert actual.keys() == expected.keys()
    assert not [name for name in expected if actual[name] != expected[name]]


def test_scene_rebind_preserves_pipeline_and_uses_no_source_caches(native, monkeypatch):
    from sim.contracts import SceneSpec
    from sim.mujoco.mjcf import MJCFConversion
    from sim.mujoco.scene import MujocoScene, SceneLoadResult

    metadata = SceneLoadResult(
        conversion=MJCFConversion(
            urdf_path=Path("scene.urdf"),
            mjcf_path=Path("scene.xml"),
            camera_mounts=(),
            joint_names=("left", "right"),
            body_names=(),
        ),
        cameras=(),
        arms={},
        grippers={},
        scene_spec=SceneSpec(visual_rgba_by_link={"world": (0.1, 0.2, 0.3, 1.0)}),
    )
    state = capture_native_state(native, max_native_bytes=_BUDGET)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("A scene rebind must not re-evaluate captured physics.")

    with monkeypatch.context() as patch:
        for name in ("mj_forward", "mj_resetData", "mj_step"):
            patch.setattr(mujoco, name, forbidden)
        scene = MujocoScene.from_native_checkpoint(
            state,
            load_result=metadata,
            max_native_bytes=_BUDGET,
        )
    assert scene.data is not native and scene.model is scene.data.model
    assert scene.data.__getstate__() == native.__getstate__() == state.data
    assert (
        capture_native_state(scene.data, max_native_bytes=_BUDGET).auxiliary
        == state.auxiliary
    )
    assert scene.joint_id("left") == mujoco.mj_name2id(
        native.model, mujoco.mjtObj.mjOBJ_JOINT, "left"
    )
    assert scene._collision_projector is None
    assert scene._collision_projectors_by_gripper_self_contact == {}
    assert scene.arm_specs is not metadata.arms
    assert scene.gripper_specs is not metadata.grippers
    assert scene.scene_spec.to_dict() == metadata.scene_spec.to_dict()
    assert scene.scene_spec is not metadata.scene_spec
    assert (
        scene.scene_spec.visual_rgba_by_link
        is not metadata.scene_spec.visual_rgba_by_link
    )
    assert not np.shares_memory(scene.data.qpos, native.qpos)


def test_scene_restore_rejects_untyped_metadata_before_native_parse(native):
    from sim.mujoco.scene import MujocoScene

    state = capture_native_state(native, max_native_bytes=_BUDGET)
    with pytest.raises(ValueError, match="metadata type"):
        MujocoScene.from_native_checkpoint(
            state, load_result={}, max_native_bytes=_BUDGET
        )
    assert native.__getstate__() == state.data


def test_actual_dual_arm_cad_scene_rebind_keeps_native_state_and_all_camera_bindings(
    monkeypatch,
):
    from sim.contracts import SimulationContract
    from sim.mujoco.scene import MujocoScene

    source = MujocoScene.from_contract(
        SimulationContract.load("sim/configs/rm65_bimanual_physics_v1.json")
    )
    budget = 128 * 1024 * 1024
    state = capture_native_state(source.data, max_native_bytes=budget)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Native CAD scene restore must preserve its pipeline.")

    with monkeypatch.context() as patch:
        for name in ("mj_forward", "mj_resetData", "mj_step"):
            patch.setattr(mujoco, name, forbidden)
        restored = MujocoScene.from_native_checkpoint(
            state,
            load_result=source.load_result,
            max_native_bytes=budget,
        )
    assert restored.data.__getstate__() == state.data == source.data.__getstate__()
    assert (
        capture_native_state(restored.data, max_native_bytes=budget).auxiliary
        == state.auxiliary
    )
    assert set(restored.arm_specs) == {"left", "right"}
    assert set(restored.gripper_specs) == set(source.gripper_specs)
    assert set(restored.camera_names) == {"front", "left_wrist", "right_wrist"}
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            restored.get_arm_qpos(side), source.get_arm_qpos(side)
        )
    for name in source.camera_names:
        assert restored.camera_id(name) == source.camera_id(name)
        np.testing.assert_array_equal(
            restored.camera_pose(name), source.camera_pose(name)
        )
    assert restored.scene_spec.to_dict() == source.scene_spec.to_dict()
    assert restored.load_result is not source.load_result
    assert (
        restored.scene_spec.visual_rgba_by_link
        is not source.scene_spec.visual_rgba_by_link
    )
    assert restored._collision_projector is None


def test_native_codec_omissions_are_preserved_without_recomputation(native):
    # Non-default constraint/lazy-stage state makes omitted fields observable.
    native.model.jnt_limited[0] = 1
    native.model.jnt_range[0] = [-0.1, 0.1]
    native.qpos[0] = 0.2
    mujoco.mj_forward(native.model, native)
    mujoco.mj_energyPos(native.model, native)
    mujoco.mj_energyVel(native.model, native)
    mujoco.mj_subtreeVel(native.model, native)
    mujoco.mj_rnePostConstraint(native.model, native)
    # Solver diagnostics need not be recomputed to be checkpointed.
    native.solver_fwdinv[:] = [0.125, 0.25]
    state = capture_native_state(native, max_native_bytes=_BUDGET)
    assert state.auxiliary.counts[0] > 0
    assert all(state.auxiliary.flags)
    restored = restore_native_state(state, max_native_bytes=_BUDGET)
    assert _continuation_witness(native) == _continuation_witness(restored)
    assert (
        capture_native_state(restored, max_native_bytes=_BUDGET).auxiliary
        == state.auxiliary
    )
    raw = mujoco.MjData.__new__(mujoco.MjData)
    raw.__setstate__(state.data)
    assert raw.nl != native.nl
    assert not np.array_equal(raw.solver_fwdinv, native.solver_fwdinv)


@pytest.mark.parametrize(
    "field,value",
    [
        ("counts", (True, 0, 0, 0, 0)),
        ("counts", (0, -1, 0, 0, 0)),
        ("flags", (1, 1, 1, 1)),
        ("solver_niter", ()),
        ("solver_fwdinv", (float("nan"), 0.0)),
        ("solver_fwdinv", (1.0, 2.0)),
    ],
)
def test_auxiliary_layout_and_checksum_are_checked(native, field, value):
    state = capture_native_state(native, max_native_bytes=_BUDGET)
    changed = replace(state, auxiliary=replace(state.auxiliary, **{field: value}))
    with pytest.raises(NativeCheckpointError):
        restore_native_state(changed, max_native_bytes=_BUDGET)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mujoco_version", "3.9.0"),
        ("native_abi", ("different", "little", 8, 4)),
        ("state_signature", True),
        ("state_sha256", "0" * 64),
        ("model_sha256", "0" * 64),
        ("data", b"corrupt"),
        ("integration", (float("nan"),)),
        ("integration", (0.0,)),
        ("source_compilation_signatures", (True, 0)),
    ],
)
def test_rejects_corrupt_or_incompatible_component_without_touching_source(
    native, field, value
):
    state = capture_native_state(native, max_native_bytes=_BUDGET)
    with pytest.raises(NativeCheckpointError):
        restore_native_state(replace(state, **{field: value}), max_native_bytes=_BUDGET)
    assert native.__getstate__() == state.data


@pytest.mark.parametrize("budget", [True, 0, -1, 100.0])
def test_requires_an_explicit_integer_budget(native, budget):
    with pytest.raises(NativeCheckpointError):
        capture_native_state(native, max_native_bytes=budget)


def test_rejects_budget_overflow_callbacks_and_unverified_library(native, monkeypatch):
    state = capture_native_state(native, max_native_bytes=_BUDGET)
    for operation, argument in (
        (capture_native_state, native),
        (restore_native_state, state),
    ):
        with pytest.raises(NativeCheckpointError):
            operation(argument, max_native_bytes=len(state.data) - 1)
        with monkeypatch.context() as patch:
            patch.setattr(mujoco, "__version__", "3.11.0")
            with pytest.raises(NativeCheckpointError, match="verified"):
                operation(argument, max_native_bytes=_BUDGET)
        mujoco.set_mjcb_control(lambda *_: None)
        try:
            with pytest.raises(NativeCheckpointError, match="callbacks"):
                operation(argument, max_native_bytes=_BUDGET)
        finally:
            mujoco.set_mjcb_control(None)
    assert native.__getstate__() == state.data


def test_rejects_nonfinite_physics_and_wrong_owner(native):
    with pytest.raises(NativeCheckpointError, match="MjData"):
        capture_native_state(object(), max_native_bytes=_BUDGET)
    native.qpos[0] = np.nan
    with pytest.raises(NativeCheckpointError, match="Non-finite"):
        capture_native_state(native, max_native_bytes=_BUDGET)
