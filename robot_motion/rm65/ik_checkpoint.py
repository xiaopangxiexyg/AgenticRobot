"""Trusted cold state for the standard coupled Pink solver, without solving.

The caller owns exclusion and the unpublished destination. Native blobs are
accepted only with the same constructor-built model and SDK. This is not an
untrusted file importer, a native memory sandbox, or a whole target checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import platform
import struct
import sys
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash

from .dual_arm_ik import MultiTaskIKSolver

_OPTIONAL = {
    "limits",
    "explicit_velocity_limit",
    "admitted_reference_motion_limit",
    "normalized_sigma_escape_task",
    "structural_branch_guard_limit",
    "analytic_branch_anchor_task",
    "_analytic_branch_anchor_q",
    "branch_continuity_anchor_task",
    "_branch_continuity_anchor_q",
    "_branch_continuity_anchor_cost_spec",
    "_branch_continuity_anchor_cost_vector",
    "acceleration_limit",
    "viz",
}
_STATIC = {
    "debug_profile",
    "debug_viz_requested",
    "collision_barrier_gain",
    "collision_safe_displacement_gain",
    "collision_min_distance_m",
    "damping",
    "solver",
}
_BINDINGS = {
    "pin",
    "pink",
    "solve_ik_fn",
    "FrameTask",
    "PostureTask",
    "AccelerationLimit",
    "SelfCollisionBarrier",
    "qpsolvers",
    "robot",
    "configuration",
    "tasks",
    "barriers",
}
_STATE = {"dt", "frequency", "last_solve_profile", "last_collision_distance_profile"}
_CACHE = ("_Barrier__q_cache", "_Barrier__jac_cache", "_Barrier__barrier_cache")
_FRAME = {"cost", "gain", "lm_damping", "frame", "transform_target_to_world"}
_POSTURE = {"cost", "gain", "lm_damping", "target_q"}
_BARRIER = {
    "dim",
    "gain",
    "gain_function",
    "safe_displacement",
    "safe_displacement_gain",
    "d_min",
    *_CACHE,
}


def copy_data(value, memo=None):
    """Detach closed numeric state while retaining exact arrays and aliases."""
    memo = {} if memo is None else memo
    if value is None or type(value) in (str, bool, int, bytes):
        return value
    if type(value) is float or isinstance(value, np.generic):
        if (
            isinstance(value, np.generic) and value.dtype.kind not in "biuf"
        ) or not np.isfinite(value):
            raise ValueError("Nonfinite IK checkpoint scalar.")
        return value
    if id(value) in memo:
        if memo[id(value)] is None:
            raise ValueError("Cyclic IK checkpoint state is unsupported.")
        return memo[id(value)]
    if type(value) is np.ndarray:
        if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
            raise ValueError("IK checkpoint requires finite numeric arrays.")
        result = value.copy()
        result.setflags(write=value.flags.writeable)
    elif type(value) in (dict, list, tuple):
        # Owner state is acyclic. Unknown object graphs are not admitted.
        memo[id(value)] = None
        if type(value) is dict:
            result = {copy_data(k, memo): copy_data(v, memo) for k, v in value.items()}
        else:
            result = type(value)(copy_data(v, memo) for v in value)
    else:
        raise ValueError(f"Unsupported IK checkpoint data: {type(value).__name__}.")
    memo[id(value)] = result
    return result


def check_data_aliases(value, *, barrier_q_alias=False):
    """Reject unknown mutable aliases; only the documented barrier-q alias exists."""
    seen, arrays = {}, []

    def visit(item, path=()):
        if type(item) in (dict, list, np.ndarray):
            if id(item) in seen:
                if barrier_q_alias and {seen[id(item)], path} == {
                    ("q",),
                    ("barrier", 0),
                }:
                    return
                raise ValueError("Unknown mutable alias in IK checkpoint data.")
            seen[id(item)] = path
        if type(item) is np.ndarray:
            if any(np.shares_memory(item, other) for other in arrays):
                raise ValueError("Unknown partial array alias in IK checkpoint data.")
            arrays.append(item)
        elif type(item) is dict:
            for index, (key, part) in enumerate(item.items()):
                visit(key, (*path, "key", index))
                visit(part, (*path, key))
        elif type(item) in (tuple, list):
            for index, part in enumerate(item):
                visit(part, (*path, index))

    visit(value)


def pack_data(value):
    """Use the existing canonical envelope while retaining exact numeric bits."""
    if value is None or type(value) in (str, bool, int, bytes):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Nonfinite IK checkpoint scalar.")
        return ("float", value.hex())
    if isinstance(value, np.generic):
        copy_data(value)
        return ("scalar", value.dtype.str, value.tobytes())
    if type(value) is np.ndarray:
        copy_data(value)
        return (
            "array",
            value.dtype.str,
            value.shape,
            value.tobytes(),
            value.flags.writeable,
        )
    if type(value) is dict:
        return (
            "mapping",
            tuple((pack_data(k), pack_data(v)) for k, v in value.items()),
        )
    if type(value) in (list, tuple):
        return (type(value).__name__, tuple(pack_data(v) for v in value))
    raise ValueError("Unknown IK checkpoint data type.")


def unpack_data(value, depth=0):
    if depth > 64:
        raise ValueError("IK checkpoint data is too deeply nested.")
    if value is None or type(value) in (str, bool, int, bytes):
        return value
    if type(value) is not tuple or not value:
        raise ValueError("Invalid IK data encoding.")
    tag = value[0]
    if tag == "float" and len(value) == 2 and type(value[1]) is str:
        result = float.fromhex(value[1])
        if math.isfinite(result):
            return result
    elif tag in ("scalar", "array") and len(value) == (3 if tag == "scalar" else 5):
        dtype = np.dtype(value[1])
        shape = () if tag == "scalar" else value[2]
        raw = value[2] if tag == "scalar" else value[3]
        if (
            dtype.kind not in "biuf"
            or type(shape) is not tuple
            or len(shape) > 32
            or any(type(n) is not int or n < 0 for n in shape)
            or type(raw) is not bytes
            or math.prod(shape) * dtype.itemsize != len(raw)
        ):
            raise ValueError("Invalid IK numeric payload.")
        result = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if not np.all(np.isfinite(result)):
            raise ValueError("Nonfinite IK numeric payload.")
        if tag == "scalar":
            return result[()]
        if type(value[4]) is not bool:
            raise ValueError("Invalid IK array flags.")
        result.setflags(write=value[4])
        return result
    elif (
        tag in ("mapping", "tuple", "list")
        and len(value) == 2
        and type(value[1]) is tuple
    ):
        if tag == "mapping":
            result = {}
            for entry in value[1]:
                if type(entry) is not tuple or len(entry) != 2:
                    raise ValueError("Invalid IK mapping entry.")
                key = unpack_data(entry[0], depth + 1)
                if key in result:
                    raise ValueError("Duplicate IK mapping key.")
                result[key] = unpack_data(entry[1], depth + 1)
            return result
        return (list if tag == "list" else tuple)(
            unpack_data(v, depth + 1) for v in value[1]
        )
    raise ValueError("Invalid IK data encoding.")


def _payload(value):
    if type(value) is float or isinstance(value, np.generic):
        return pack_data(value)
    if is_dataclass(value):
        return (
            type(value).__qualname__,
            tuple((f.name, _payload(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, Mapping):
        return ("mapping", tuple((_payload(k), _payload(v)) for k, v in value.items()))
    if isinstance(value, Path):
        return ("path", str(value))
    if type(value) is np.ndarray:
        return pack_data(value)
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_payload(v) for v in value))
    return value


def state_hash(value):
    return canonical_content_hash(_payload(value), domain="rm65.ik-checkpoint-state.v1")


def _budget(limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError("A positive native byte budget is required.")


def _native_bytes(solver, obj, limit):
    buffer = solver.pin.serialization.StreamBuffer()
    obj.saveToBinary(buffer)
    if buffer.size() > limit:
        raise ValueError("IK native state exceeds byte budget.")
    return buffer.tobytes()


def _sdk(solver):
    import coal

    if (solver.pin.__version__, version("pin-pink"), coal.__version__) != (
        "3.7.0",
        "4.2.0",
        "3.0.1",
    ):
        raise ValueError(
            "IK native checkpoint supports only the verified Pin/Pink/Coal stack."
        )
    extension = Path(
        importlib.import_module(solver.pin.Model.__module__).__file__
    ).resolve()
    coal_extension = Path(
        importlib.import_module(coal.DistanceRequest.__module__).__file__
    ).resolve()
    libraries = [
        p / "libcoal.so.3.0.1"
        for p in coal_extension.parents
        if (p / "libcoal.so.3.0.1").is_file()
    ]
    if len(libraries) != 1:
        raise ValueError("Cannot bind the verified native collision library.")
    return (
        solver.pin.__version__,
        version("pin-pink"),
        coal.__version__,
        version("qpsolvers"),
        version("osqp"),
        np.__version__,
        version("scipy"),
        sys.version_info[:2],
        sys.byteorder,
        platform.machine(),
        struct.calcsize("P"),
        *(
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (extension, coal_extension, libraries[0])
        ),
    )


def _qualify(s):
    if (
        type(s) is not MultiTaskIKSolver
        or set(vars(s)) != _OPTIONAL | _STATIC | _BINDINGS | _STATE
    ):
        raise ValueError("Unknown IK solver owner or field layout.")
    if any(getattr(s, n) is not None for n in _OPTIONAL) or s.debug_viz_requested:
        raise ValueError(
            "IK checkpoint does not support additional limits/tasks/viewers."
        )
    import pink
    import pinocchio
    import qpsolvers
    from pink import Configuration, solve_ik
    from pink.barriers import SelfCollisionBarrier
    from pink.limits import AccelerationLimit, ConfigurationLimit, VelocityLimit
    from pink.tasks import FrameTask, PostureTask
    from pink.utils import VectorSpace

    if (
        s.pin is not pinocchio
        or s.pink is not pink
        or s.qpsolvers is not qpsolvers
        or s.solve_ik_fn is not solve_ik
        or s.FrameTask is not FrameTask
        or s.PostureTask is not PostureTask
        or s.SelfCollisionBarrier is not SelfCollisionBarrier
        or s.AccelerationLimit is not AccelerationLimit
    ):
        raise ValueError("IK checkpoint does not accept substituted solver bindings.")
    if tuple(type(t) for t in s.tasks) != (FrameTask, FrameTask, PostureTask):
        raise ValueError("IK checkpoint requires the original ordered task set.")
    if len(s.barriers) != 1 or type(s.barriers[0]) is not SelfCollisionBarrier:
        raise ValueError("IK checkpoint requires the original collision barrier.")
    for task, layout in zip(s.tasks, (_FRAME, _FRAME, _POSTURE)):
        if set(vars(task)) != layout:
            raise ValueError("Unknown IK task state.")
    b = s.barriers[0]
    default = SelfCollisionBarrier(b.dim)
    if (
        set(vars(b)) != _BARRIER
        or not hasattr(b.gain_function, "__code__")
        or b.gain_function.__code__ is not default.gain_function.__code__
        or b.gain_function.__closure__ is not None
    ):
        raise ValueError("Unknown IK barrier state or gain function.")
    c, r = s.configuration, s.robot
    if (
        type(r) is not pinocchio.RobotWrapper
        or type(r.model) is not pinocchio.Model
        or type(r.collision_model) is not pinocchio.GeometryModel
        or any(type(x) is not pinocchio.Data or vars(x) for x in (r.data, c.data))
        or type(r.collision_data) is not pinocchio.GeometryData
        or vars(r.collision_data)
    ):
        raise ValueError("Unknown IK native owner implementation or Python state.")
    if type(c) is not Configuration or set(vars(c)) != {
        "data",
        "model",
        "q",
        "tangent",
        "collision_model",
        "collision_data",
    }:
        raise ValueError("Unknown Pink configuration state.")
    if (
        c.model is not r.model
        or c.collision_model is not r.collision_model
        or c.collision_data is not r.collision_data
        or c.data is r.data
        or c.tangent is not r.model.tangent
    ):
        raise ValueError("IK native owner bindings differ.")
    if (
        set(vars(r))
        != {
            "model",
            "collision_model",
            "visual_model",
            "data",
            "collision_data",
            "visual_data",
            "v0",
            "q0",
            "viz",
        }
        or r.viz is not None
    ):
        raise ValueError("Unknown IK robot wrapper state.")
    if (
        set(vars(r.model))
        != {
            "tangent",
            "configuration_limit",
            "velocity_limit",
            "floating_base_velocity_limit",
        }
        or r.model.floating_base_velocity_limit is not None
    ):
        raise ValueError("Unknown default Pink model limits.")
    for owner, kind, layout in (
        (
            r.model.tangent,
            VectorSpace,
            {"_VectorSpace__eye", "_VectorSpace__ones", "_VectorSpace__zeros"},
        ),
        (
            r.model.configuration_limit,
            ConfigurationLimit,
            {"config_limit_gain", "indices", "joints", "model", "projection_matrix"},
        ),
        (
            r.model.velocity_limit,
            VelocityLimit,
            {"indices", "joints", "model", "projection_matrix"},
        ),
    ):
        if type(owner) is not kind or set(vars(owner)) != layout:
            raise ValueError("Unknown default Pink limit or tangent implementation.")
    for req in (
        *r.collision_data.collisionRequests,
        *r.collision_data.distanceRequests,
    ):
        # Pin 3.7 sets this legacy bit even with DefaultGuess in the newer
        # enum. Read that exact binding; never switch query modes on restore.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="enable_cached_gjk_guess has been marked as deprecated",
                category=DeprecationWarning,
            )
            cached = req.enable_cached_gjk_guess
        if req.enable_timings or not cached:
            raise ValueError("Unsupported native query timing or warm-start contract.")
    if (
        type(s.dt) is not float
        or type(s.frequency) is not float
        or not math.isfinite(s.dt)
        or s.dt <= 0
        or not math.isclose(s.frequency * s.dt, 1.0, rel_tol=1e-14)
    ):
        raise ValueError("IK integration period and frequency differ.")


def _binding(s, limit):
    # Native model bytes bind geometry, pair order, frames and limits. The
    # enclosing profile additionally owns asset/source identity and relocation.
    default_limits = {}
    for name in ("configuration_limit", "velocity_limit"):
        owner = getattr(s.robot.model, name)
        if owner.model is not s.robot.model:
            raise ValueError("Default Pink limits bind a different model.")
        default_limits[name] = {
            k: v for k, v in vars(owner).items() if k not in ("model", "joints")
        }
        default_limits[name]["joints"] = tuple(
            (j.id, j.idx_q, j.idx_v, j.nq, j.nv) for j in owner.joints
        )
    return state_hash(
        {
            "sdk": _sdk(s),
            "solver": {n: getattr(s, n) for n in sorted(_STATIC)},
            "models": tuple(
                hashlib.sha256(_native_bytes(s, m, limit)).hexdigest()
                for m in (s.robot.model, s.robot.collision_model)
            ),
            "tangent": vars(s.configuration.tangent),
            "default_limits": default_limits,
            "q0": s.robot.q0,
            "v0": s.robot.v0,
            "tasks": tuple(
                {
                    k: v
                    for k, v in vars(t).items()
                    if k not in ("transform_target_to_world", "target_q")
                }
                for t in s.tasks
            ),
            "barrier": {
                k: v
                for k, v in vars(s.barriers[0]).items()
                if k not in {*_CACHE, "safe_displacement", "gain_function"}
            },
        }
    )


@dataclass(frozen=True)
class IKSolverCheckpoint:
    binding_hash: str
    state: dict
    native: tuple[bytes, bytes, bytes]
    content_hash: str


def _contents(binding, state, native):
    return state_hash(
        (binding, state, tuple(hashlib.sha256(v).hexdigest() for v in native))
    )


def capture_ik_solver(s, *, max_native_bytes):
    _budget(max_native_bytes)
    _qualify(s)
    c, b = s.configuration, s.barriers[0]
    caches = tuple(getattr(b, n) for n in _CACHE)
    raw_state = {
        **{n: getattr(s, n) for n in sorted(_STATE)},
        "q": c.q,
        "targets": tuple(
            t.transform_target_to_world.homogeneous.copy()
            if t.transform_target_to_world is not None
            else None
            for t in s.tasks[:2]
        )
        + (s.tasks[2].target_q,),
        "barrier": caches,
        "barrier_q_is_current": caches[0] is c.q,
        "safe_displacement": b.safe_displacement,
    }
    check_data_aliases(raw_state, barrier_q_alias=raw_state["barrier_q_is_current"])
    state = copy_data(raw_state)
    native = tuple(
        _native_bytes(s, x, max_native_bytes)
        for x in (s.robot.data, c.data, c.collision_data)
    )
    if sum(map(len, native)) > max_native_bytes:
        raise ValueError("IK native state exceeds byte budget.")
    binding = _binding(s, max_native_bytes)
    return IKSolverCheckpoint(binding, state, native, _contents(binding, state, native))


def restore_ik_solver(s, checkpoint, *, max_native_bytes):
    """Install validated data in a constructor-owned, unpublished fresh solver."""
    _budget(max_native_bytes)
    _qualify(s)
    if type(checkpoint) is not IKSolverCheckpoint:
        raise ValueError("Expected a typed IK solver checkpoint.")
    if type(checkpoint.state) is not dict:
        raise ValueError("Invalid IK checkpoint state fields.")
    if (
        type(checkpoint.native) is not tuple
        or len(checkpoint.native) != 3
        or any(type(v) is not bytes or not v for v in checkpoint.native)
        or sum(map(len, checkpoint.native)) > max_native_bytes
    ):
        raise ValueError("Invalid IK native state or byte budget.")
    if checkpoint.content_hash != _contents(
        checkpoint.binding_hash, checkpoint.state, checkpoint.native
    ):
        raise ValueError("IK checkpoint content changed.")
    if checkpoint.binding_hash != _binding(s, max_native_bytes):
        raise ValueError("IK model, SDK or solver binding differs.")
    check_data_aliases(
        checkpoint.state,
        barrier_q_alias=checkpoint.state.get("barrier_q_is_current") is True,
    )
    data = copy_data(checkpoint.state)
    if set(data) != _STATE | {
        "q",
        "targets",
        "barrier",
        "barrier_q_is_current",
        "safe_displacement",
    }:
        raise ValueError("Unknown IK state fields.")
    nq, nv, dim = s.robot.model.nq, s.robot.model.nv, s.barriers[0].dim

    def array(value, shape):
        if (
            type(value) is not np.ndarray
            or value.shape != shape
            or value.dtype != np.dtype("float64")
            or not np.all(np.isfinite(value))
        ):
            raise ValueError("Invalid IK checkpoint array.")

    array(data["q"], (nq,))
    if (
        type(data["targets"]) is not tuple
        or len(data["targets"]) != 3
        or type(data["barrier"]) is not tuple
        or len(data["barrier"]) != 3
    ):
        raise ValueError("Invalid IK target/cache group.")
    for target, shape in zip(data["targets"], ((4, 4), (4, 4), (nq,))):
        if target is not None:
            array(target, shape)
    for cache, shape in zip(data["barrier"], ((nq,), (dim, nv), (dim,))):
        if cache is not None:
            array(cache, shape)
    if any(x is None for x in data["barrier"]) and not all(
        x is None for x in data["barrier"]
    ):
        raise ValueError("Incomplete IK barrier cache group.")
    if type(data["barrier_q_is_current"]) is not bool or (
        data["barrier_q_is_current"]
        and (
            data["barrier"][0] is None
            or data["barrier"][0].tobytes() != data["q"].tobytes()
            or data["barrier"][0].flags.writeable != data["q"].flags.writeable
        )
    ):
        raise ValueError("Invalid IK barrier cache alias.")
    # Pink initializes this vector in barrier space, then its default QP
    # objective sets a tangent-space vector. Preserve the actual legal phase.
    if type(data["safe_displacement"]) is not np.ndarray or data[
        "safe_displacement"
    ].shape not in ((dim,), (nv,)):
        raise ValueError("Invalid IK safe-displacement phase.")
    array(data["safe_displacement"], data["safe_displacement"].shape)
    if (
        not math.isfinite(data["dt"])
        or data["dt"] <= 0
        or not math.isclose(data["frequency"] * data["dt"], 1.0, rel_tol=1e-14)
    ):
        raise ValueError("Invalid IK checkpoint time step.")
    native = (
        s.pin.Data(s.robot.model),
        s.pin.Data(s.robot.model),
        s.pin.GeometryData(s.robot.collision_model),
    )
    # Only fixed native classes consume authenticated-by-parent blobs. Fresh
    # GeometryData first binds functors to the destination geometry objects.
    with tempfile.TemporaryDirectory(prefix="rm65-ik-restore-") as directory:
        for index, (obj, blob) in enumerate(zip(native, checkpoint.native)):
            path = Path(directory) / f"{index}.bin"
            path.write_bytes(blob)
            obj.loadFromBinary(str(path))
            if _native_bytes(s, obj, max_native_bytes) != blob:
                raise ValueError("Native IK state did not round trip exactly.")
    s.robot.data, cdata, s.robot.collision_data = native
    # Do not recompute FK, collision, Jacobians, or q: their stages can differ.
    s.configuration.data = cdata
    s.configuration.collision_data = s.robot.collision_data
    s.configuration.q = data["q"]
    for task, value in zip(s.tasks[:2], data["targets"][:2]):
        task.transform_target_to_world = None if value is None else s.pin.SE3(value)
    s.tasks[2].target_q = data["targets"][2]
    for name, value in zip(_CACHE, data["barrier"]):
        setattr(s.barriers[0], name, value)
    if data["barrier_q_is_current"]:
        s.barriers[0]._Barrier__q_cache = s.configuration.q
    s.barriers[0].safe_displacement = data["safe_displacement"]
    for name in _STATE:
        setattr(s, name, data[name])
    _qualify(s)
