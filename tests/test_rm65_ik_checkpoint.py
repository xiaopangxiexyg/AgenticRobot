"""Actual coupled Pink/Pinocchio owner restoration, without a restored target."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.action import ComponentAction
from robot_motion.deploy.schema import pose_matrix_to_xyzquat
from robot_motion.rm65.dual_arm_target_solver import DualArmCollisionAwareRetargeter
from robot_motion.rm65.ik_checkpoint import (
    capture_ik_solver,
    copy_data,
    pack_data,
    restore_ik_solver,
    state_hash,
)
from robot_motion.rm65.retargeter_checkpoint import (
    capture_retargeter,
    decode_retargeter_checkpoint,
    encode_retargeter_checkpoint,
    restore_retargeter,
)
from sim.mujoco.deploy.target_measurement import (
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
)
from tests.test_mujoco_agent_control_checkpoint import _native_target

pytestmark = [pytest.mark.integration, pytest.mark.slow_sim]
BUDGET = 128 * 1024 * 1024


def test_closed_scalar_domain_rejects_finite_complex_before_encoding():
    for operation in (copy_data, pack_data):
        with pytest.raises(ValueError, match="scalar"):
            operation(np.complex128(1 + 2j))


@pytest.fixture
def source():
    _config, _prepared, profile, clock, target = _native_target()
    try:
        yield target, profile, clock
    finally:
        target.runtime.close()


def _move(source):
    target, profile, _clock = source
    runtime = target.runtime
    runtime.start()
    pose = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
        target.scene, target.task_config
    )["right"]
    value = pose_matrix_to_xyzquat(pose)
    value[0] += 0.005
    assert runtime.submit_component_action(
        ComponentAction(
            action_schema=profile.action_schema,
            values_by_component={"right_arm": value},
        ),
        source_event_id="checkpoint:nonzero-ik",
    ).accepted
    assert runtime.step(physics_ticks_after_dispatch=30).status == "dispatched"


@pytest.mark.parametrize("moved", (False, True))
def test_native_solver_and_controller_state_restore_without_reset_or_solve(
    source, monkeypatch, moved
):
    if moved:
        _move(source)
    original = source[0].runtime.pose_resolver.ik.retargeter
    saved = capture_retargeter(original, max_native_bytes=BUDGET)

    def no_reset(*_args, **_kwargs):
        raise AssertionError("Cold restore must not reset tracking")

    monkeypatch.setattr(
        DualArmCollisionAwareRetargeter, "set_current_qpos_pair", no_reset
    )
    branch = restore_retargeter(saved, config=original.config, max_native_bytes=BUDGET)
    assert (
        capture_retargeter(branch, max_native_bytes=BUDGET).content_hash
        == saved.content_hash
    )
    assert (
        capture_retargeter(original, max_native_bytes=BUDGET).content_hash
        == saved.content_hash
    )
    assert branch.ik_solver.robot is not original.ik_solver.robot
    assert branch.ik_solver.configuration.data is not branch.ik_solver.robot.data
    assert (
        branch.ik_solver.configuration.collision_data
        is branch.ik_solver.robot.collision_data
    )
    assert not np.shares_memory(branch._last_safe_full_q, original._last_safe_full_q)
    if moved:
        assert original._last_target_poses["right"] is not None
        assert original.ik_solver.barriers[0]._Barrier__q_cache is not None


def test_first_native_query_and_qp_continue_from_the_same_caches(source):
    _move(source)
    original = source[0].runtime.pose_resolver.ik.retargeter
    saved = capture_retargeter(original, max_native_bytes=BUDGET)
    encoded = encode_retargeter_checkpoint(saved)
    decoded = decode_retargeter_checkpoint(encoded, max_bytes=8 * 1024 * 1024)
    assert encode_retargeter_checkpoint(decoded) == encoded
    branch = restore_retargeter(
        decoded, config=original.config, max_native_bytes=BUDGET
    )
    a, b = original.ik_solver, branch.ik_solver
    for solver in (a, b):
        solver.pin.computeDistances(
            solver.robot.collision_model, solver.robot.collision_data
        )
    for left, right in zip(
        a.robot.collision_data.distanceResults, b.robot.collision_data.distanceResults
    ):
        assert left.min_distance == right.min_distance
        np.testing.assert_array_equal(left.normal, right.normal)
        np.testing.assert_array_equal(left.getNearestPoint1(), right.getNearestPoint1())
        np.testing.assert_array_equal(left.getNearestPoint2(), right.getNearestPoint2())
        np.testing.assert_array_equal(left.cached_gjk_guess, right.cached_gjk_guess)
        np.testing.assert_array_equal(
            left.cached_support_func_guess, right.cached_support_func_guess
        )
    # Exercise the original one-QP primitive, not the wall-clock adaptive
    # outer loop. No target, collision guard or reasoning budget is changed.
    for _ in range(3):
        np.testing.assert_array_equal(a.solve(), b.solve())
        first = capture_ik_solver(a, max_native_bytes=BUDGET)
        second = capture_ik_solver(b, max_native_bytes=BUDGET)
        assert first.content_hash == second.content_hash


def test_live_barrier_q_alias_and_array_writeability_survive(source):
    _move(source)
    original = source[0].runtime.pose_resolver.ik.retargeter
    solver = original.ik_solver
    # A completed barrier query can retain the current q by identity. Keep
    # the original public query and its cache behavior, rather than fabricating it.
    solver.barriers[0].compute_qp_inequalities(solver.configuration, solver.dt)
    assert solver.barriers[0]._Barrier__q_cache is solver.configuration.q
    saved = capture_retargeter(original, max_native_bytes=BUDGET)
    saved = decode_retargeter_checkpoint(
        encode_retargeter_checkpoint(saved), max_bytes=8 * 1024 * 1024
    )
    branch = restore_retargeter(saved, config=original.config, max_native_bytes=BUDGET)
    restored = branch.ik_solver
    assert restored.barriers[0]._Barrier__q_cache is restored.configuration.q
    assert (
        restored.configuration.q.flags.writeable
        == solver.configuration.q.flags.writeable
    )
    assert not np.shares_memory(restored.configuration.q, solver.configuration.q)


def test_corrupt_native_data_wrong_config_and_small_budget_reject(source):
    original = source[0].runtime.pose_resolver.ik.retargeter
    saved = capture_retargeter(original, max_native_bytes=BUDGET)
    damaged_solver = replace(
        saved.solver, native=(saved.solver.native[0][:-1], *saved.solver.native[1:])
    )
    with pytest.raises(ValueError, match="content changed"):
        restore_retargeter(
            replace(saved, solver=damaged_solver),
            config=original.config,
            max_native_bytes=BUDGET,
        )
    with pytest.raises(ValueError, match="configuration"):
        restore_retargeter(
            saved,
            config=replace(original.config, damping=original.config.damping * 2),
            max_native_bytes=BUDGET,
        )
    with pytest.raises(ValueError, match="budget"):
        capture_retargeter(original, max_native_bytes=32)
    assert (
        capture_retargeter(original, max_native_bytes=BUDGET).content_hash
        == saved.content_hash
    )


def test_integrity_detects_a_signed_zero_bit_change_and_codec_retains_it(source):
    original = source[0].runtime.pose_resolver.ik.retargeter
    saved = capture_retargeter(original, max_native_bytes=BUDGET)
    velocity = saved.state["_last_admitted_velocity_full"]
    assert velocity[0] == 0.0
    velocity[0] = -0.0 if not np.signbit(velocity[0]) else 0.0
    with pytest.raises(ValueError, match="content changed"):
        restore_retargeter(saved, config=original.config, max_native_bytes=BUDGET)
    # A new legitimate capture retains signed zero in a real controller field.
    original._last_admitted_velocity_full[0] = -0.0
    source_bytes = original._last_admitted_velocity_full.tobytes()
    current = capture_retargeter(original, max_native_bytes=BUDGET)
    decoded = decode_retargeter_checkpoint(
        encode_retargeter_checkpoint(current), max_bytes=8 * 1024 * 1024
    )
    assert decoded.state["_last_admitted_velocity_full"].tobytes() == source_bytes
    assert (
        decoded.state["_last_admitted_velocity_full"].flags.writeable
        == original._last_admitted_velocity_full.flags.writeable
    )
    assert state_hash(0.0) != state_hash(-0.0)
    assert state_hash(np.float64(0.0)) != state_hash(np.float64(-0.0))


def test_native_barrier_preserves_initial_and_post_qp_displacement_dimensions(source):
    _move(source)
    original = source[0].runtime.pose_resolver.ik.retargeter
    solver = original.ik_solver

    # Standalone Pink supports selecting a subset of nearest collision pairs.
    # Exercise its real dim != nv lifecycle; do not dispatch this test variant.
    def barrier(owner):
        return owner.SelfCollisionBarrier(
            6,
            gain=owner.collision_barrier_gain,
            safe_displacement_gain=owner.collision_safe_displacement_gain,
            d_min=owner.collision_min_distance_m,
        )

    solver.barriers = [barrier(solver)]
    assert solver.barriers[0].safe_displacement.shape == (6,)
    for after_qp in (False, True):
        if after_qp:
            solver.solve()
            assert solver.barriers[0].safe_displacement.shape == (
                solver.robot.model.nv,
            )
        saved = capture_ik_solver(solver, max_native_bytes=BUDGET)
        fresh = DualArmCollisionAwareRetargeter(original.config).ik_solver
        fresh.barriers = [barrier(fresh)]
        restore_ik_solver(fresh, saved, max_native_bytes=BUDGET)
        assert (
            capture_ik_solver(fresh, max_native_bytes=BUDGET).content_hash
            == saved.content_hash
        )


@pytest.mark.parametrize("owner", ("retargeter", "solver"))
def test_restore_rejects_unknown_alias_even_when_values_and_hash_are_unchanged(
    source, owner
):
    original = source[0].runtime.pose_resolver.ik.retargeter
    saved = capture_retargeter(original, max_native_bytes=BUDGET)
    if owner == "retargeter":
        assert (
            saved.state["_singularity_escape_direction_latch"]
            == saved.state["_singularity_escape_direction_failures"]
            == {}
        )
        saved.state["_singularity_escape_direction_failures"] = saved.state[
            "_singularity_escape_direction_latch"
        ]
    else:
        assert (
            saved.solver.state["last_solve_profile"]
            == saved.solver.state["last_collision_distance_profile"]
            == {}
        )
        saved.solver.state["last_solve_profile"] = saved.solver.state[
            "last_collision_distance_profile"
        ]
    with pytest.raises(ValueError, match="alias"):
        restore_retargeter(saved, config=original.config, max_native_bytes=BUDGET)


def test_capture_rejects_unknown_partial_alias_before_copying(source):
    original = source[0].runtime.pose_resolver.ik.retargeter
    original._last_raw_solver_q = original._last_safe_full_q.view()
    with pytest.raises(ValueError, match="alias"):
        capture_retargeter(original, max_native_bytes=BUDGET)


@pytest.mark.parametrize(
    "binding", ("configuration_limit", "velocity_limit", "tangent")
)
def test_unsupported_default_limit_owner_rejects_even_with_identical_attributes(
    source, binding
):
    solver = source[0].runtime.pose_resolver.ik.retargeter.ik_solver
    replacement = SimpleNamespace(**vars(getattr(solver.robot.model, binding)))
    setattr(solver.robot.model, binding, replacement)
    if binding == "tangent":
        solver.configuration.tangent = replacement
    with pytest.raises(ValueError, match="default Pink"):
        capture_ik_solver(solver, max_native_bytes=BUDGET)


def test_data_only_checkpoint_resumes_the_first_qp_in_a_fresh_interpreter(
    source, tmp_path
):
    _move(source)
    original = source[0].runtime.pose_resolver.ik.retargeter
    saved = capture_retargeter(original, max_native_bytes=BUDGET)
    encoded = encode_retargeter_checkpoint(saved)
    path = tmp_path / "retargeter.json"
    path.write_bytes(encoded)
    code = r"""
import json,sys
from pathlib import Path
from robot_motion.rm65.retargeter_checkpoint import decode_retargeter_checkpoint,restore_retargeter,capture_retargeter
from robot_motion.rm65.ik_checkpoint import capture_ik_solver
from tests.test_mujoco_agent_control_checkpoint import _native_target
_,_,_,_,target = _native_target()
try:
    source = decode_retargeter_checkpoint(Path(sys.argv[1]).read_bytes(),max_bytes=8*1024*1024)
    branch = restore_retargeter(source,config=target.runtime.pose_resolver.ik.retargeter.config,max_native_bytes=128*1024*1024)
    initial = capture_retargeter(branch,max_native_bytes=128*1024*1024).content_hash
    q = branch.ik_solver.solve()
    after = capture_ik_solver(branch.ik_solver,max_native_bytes=128*1024*1024).content_hash
    print(json.dumps({"initial":initial,"q":q.tolist(),"after":after}))
finally:
    target.runtime.close()
"""
    child = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    result = json.loads(child.stdout)
    assert result["initial"] == saved.content_hash
    np.testing.assert_array_equal(original.ik_solver.solve(), result["q"])
    assert (
        capture_ik_solver(original.ik_solver, max_native_bytes=BUDGET).content_hash
        == result["after"]
    )
    assert path.read_bytes() == encoded
    with pytest.raises(ValueError, match="budget"):
        decode_retargeter_checkpoint(encoded, max_bytes=len(encoded) - 1)
    with pytest.raises(ValueError, match="duplicate key"):
        decode_retargeter_checkpoint(b'{"schema":1,"schema":2}', max_bytes=1024)


@pytest.mark.parametrize("binding", ("extra", "filter", "timing", "constraint"))
def test_unsupported_bindings_reject_without_silently_disabling_them(source, binding):
    original = source[0].runtime.pose_resolver.ik.retargeter
    if binding == "extra":
        original.unrecognized_history = 1
    elif binding == "filter":
        original.pre_ik_target_safety_filter = lambda x: x
    elif binding == "timing":
        original.ik_solver.robot.collision_data.distanceRequests[
            0
        ].enable_timings = True
    else:
        original.ik_solver.configure_velocity_limit(
            np.ones(original.ik_solver.robot.model.nv)
        )
    with pytest.raises(ValueError):
        capture_retargeter(original, max_native_bytes=BUDGET)
