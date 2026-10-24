"""Native physics component for an owner-held, quiescent MuJoCo scene.

This is not an application or Gateway checkpoint. Controllers, task state,
clocks, RNG, authority and retained operations require their own owner state.
There is no pickle interpreter, filesystem store, reset, forward, or step here.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import sys
from dataclasses import dataclass

import numpy as np

_VERSION = "3.10.0"
_CALLBACKS = (
    "act_bias",
    "act_dyn",
    "act_gain",
    "contactfilter",
    "control",
    "passive",
    "sensor",
    "time",
)
_COUNTS = ("nl", "ntree_awake", "nbody_awake", "nparent_awake", "nv_awake")
_FLAGS = ("flg_energypos", "flg_energyvel", "flg_subtreevel", "flg_rnepost")


class NativeCheckpointError(ValueError):
    """The component cannot be safely captured or restored by this codec."""


@dataclass(frozen=True)
class NativePipelineAuxiliary:
    """Public 3.10 fields omitted by its native MjData serializer.

    Allocation cursors, pointers and compilation signatures are not writable
    state here. Deserialize owns allocation; MjSpec bindings are unsupported.
    """

    counts: tuple[int, ...]
    flags: tuple[bool, ...]
    solver_niter: tuple[int, ...]
    solver_nnz: tuple[int, ...]
    solver_fwdinv: tuple[float, float]


@dataclass(frozen=True)
class NativeMujocoState:
    """Immutable native bytes plus independent integration-state evidence.

    A storage owner persists these fields and verifies its own provenance.
    Checksums detect corruption; they do not authenticate an untrusted writer.
    """

    mujoco_version: str
    native_abi: tuple[str, str, int, int]
    state_signature: int
    source_compilation_signatures: tuple[int, int]
    model_sha256: str
    state_sha256: str
    auxiliary: NativePipelineAuxiliary
    integration: tuple[float, ...]
    data: bytes


def _abi() -> tuple[str, str, int, int]:
    return (
        platform.machine(),
        sys.byteorder,
        struct.calcsize("P"),
        np.dtype(np.intc).itemsize,
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _auxiliary(data) -> NativePipelineAuxiliary:
    return NativePipelineAuxiliary(
        counts=tuple(int(getattr(data, name)) for name in _COUNTS),
        flags=tuple(bool(getattr(data, name)) for name in _FLAGS),
        solver_niter=tuple(int(value) for value in data.solver_niter),
        solver_nnz=tuple(int(value) for value in data.solver_nnz),
        solver_fwdinv=tuple(float(value) for value in data.solver_fwdinv),
    )


def _auxiliary_bytes(aux: NativePipelineAuxiliary, mujoco) -> bytes:
    if (
        type(aux) is not NativePipelineAuxiliary
        or type(aux.counts) is not tuple
        or len(aux.counts) != len(_COUNTS)
        or type(aux.flags) is not tuple
        or len(aux.flags) != len(_FLAGS)
        or any(type(value) is not bool for value in aux.flags)
        or any(
            type(values) is not tuple or len(values) != mujoco.mjNISLAND
            for values in (aux.solver_niter, aux.solver_nnz)
        )
        or any(
            type(value) is not int or not 0 <= value < 2**31
            for values in (aux.counts, aux.solver_niter, aux.solver_nnz)
            for value in values
        )
        or type(aux.solver_fwdinv) is not tuple
        or len(aux.solver_fwdinv) != 2
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in aux.solver_fwdinv
        )
    ):
        raise NativeCheckpointError("Invalid auxiliary native pipeline state.")
    return json.dumps(
        (aux.counts, aux.flags, aux.solver_niter, aux.solver_nnz, aux.solver_fwdinv),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _library():
    import mujoco

    if mujoco.__version__ != _VERSION:
        raise NativeCheckpointError(
            "Native checkpoint requires verified MuJoCo 3.10.0."
        )
    if any(getattr(mujoco, "get_mjcb_" + name)() is not None for name in _CALLBACKS):
        raise NativeCheckpointError(
            "Native checkpoint cannot capture process callbacks."
        )
    return mujoco


def _budget(max_native_bytes: int) -> None:
    if type(max_native_bytes) is not int or max_native_bytes < 1:
        raise NativeCheckpointError(
            "max_native_bytes must be an explicit positive integer."
        )


def _native_owner(mujoco, data) -> None:
    if type(data) is not mujoco.MjData:
        raise NativeCheckpointError("Expected native MuJoCo MjData.")
    if data.model.nplugin or data.pstack or data.pbase or data.threadpool:
        raise NativeCheckpointError(
            "Native checkpoint requires an idle, plugin-free owner."
        )


def _integration(mujoco, data) -> tuple[float, ...]:
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    values = np.empty(mujoco.mj_stateSize(data.model, signature), dtype=np.float64)
    mujoco.mj_getState(data.model, data, values, signature)
    if not np.isfinite(values).all():
        raise NativeCheckpointError("Non-finite MuJoCo integration state.")
    return tuple(float(value) for value in values)


def capture_native_state(data, *, max_native_bytes: int) -> NativeMujocoState:
    """Copy native state while the caller owns the scene/physics exclusion.

    The native serializer retains the compiled model, integrator and solver
    caches. The independently read vector is a consistency check, not a lock.
    No thread is paused here and no observation implies quiescence.
    """
    _budget(max_native_bytes)
    mujoco = _library()
    _native_owner(mujoco, data)
    before = _integration(mujoco, data)
    payload = data.__getstate__()
    if not payload or len(payload) > max_native_bytes:
        raise NativeCheckpointError(
            "Native payload exceeds the caller's storage budget."
        )
    auxiliary = _auxiliary(data)
    state = NativeMujocoState(
        mujoco_version=mujoco.__version__,
        native_abi=_abi(),
        state_signature=int(mujoco.mjtState.mjSTATE_INTEGRATION),
        source_compilation_signatures=(int(data.model.signature), int(data.signature)),
        model_sha256=_digest(data.model.__getstate__()),
        state_sha256=_digest(payload + _auxiliary_bytes(auxiliary, mujoco)),
        auxiliary=auxiliary,
        integration=before,
        data=payload,
    )
    if before != _integration(mujoco, data) or auxiliary != _auxiliary(data):
        raise NativeCheckpointError("Native owner changed during capture.")
    return state


def restore_native_state(state: NativeMujocoState, *, max_native_bytes: int):
    """Validate and build an independent native candidate, never mutate a live owner.

    Use only with a verified, trusted-source component. Native MuJoCo parsing
    is not an untrusted-code sandbox. Application factories must separately
    bind the candidate to the exact profile and restore all non-native owners.

    MuJoCo's native codec does not retain MjSpec compilation signatures. They
    are recorded as source evidence but the independent candidate has zero
    signatures. This component cannot restore MjSpec.bind/edit relationships.
    """
    _budget(max_native_bytes)
    mujoco = _library()
    if (
        type(state) is not NativeMujocoState
        or state.mujoco_version != _VERSION
        or state.native_abi != _abi()
        or type(state.state_signature) is not int
        or state.state_signature != int(mujoco.mjtState.mjSTATE_INTEGRATION)
        or type(state.source_compilation_signatures) is not tuple
        or len(state.source_compilation_signatures) != 2
        or any(
            type(value) is not int or not 0 <= value < 2**64
            for value in state.source_compilation_signatures
        )
        or type(state.data) is not bytes
        or not state.data
        or len(state.data) > max_native_bytes
        or type(state.integration) is not tuple
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in state.integration
        )
    ):
        raise NativeCheckpointError("Incompatible native component or storage budget.")
    aux = state.auxiliary
    if _digest(state.data + _auxiliary_bytes(aux, mujoco)) != state.state_sha256:
        raise NativeCheckpointError("Native state checksum differs.")
    try:
        candidate = mujoco.MjData.__new__(mujoco.MjData)
        candidate.__setstate__(state.data)
    except (ValueError, RuntimeError) as error:
        raise NativeCheckpointError("Invalid native MuJoCo payload.") from error
    _native_owner(mujoco, candidate)
    if (candidate.model.signature, candidate.signature) != (0, 0):
        raise NativeCheckpointError("Unexpected native compilation-binding layout.")
    if _digest(candidate.model.__getstate__()) != state.model_sha256:
        raise NativeCheckpointError("Native model identity differs.")
    if _integration(mujoco, candidate) != state.integration:
        raise NativeCheckpointError("Native data and integration evidence differ.")
    if candidate.__getstate__() != state.data:
        raise NativeCheckpointError(
            "Native roundtrip changed the captured pipeline state."
        )
    limits = (
        candidate.nefc,
        candidate.model.ntree,
        candidate.model.nbody,
        candidate.model.nbody,
        candidate.model.nv,
    )
    if any(value > limit for value, limit in zip(aux.counts, limits)):
        raise NativeCheckpointError("Auxiliary counts exceed the native layout.")
    for name, value in zip(_COUNTS + _FLAGS, aux.counts + aux.flags):
        setattr(candidate, name, value)
    candidate.solver_niter[:] = aux.solver_niter
    candidate.solver_nnz[:] = aux.solver_nnz
    candidate.solver_fwdinv[:] = aux.solver_fwdinv
    if _auxiliary(candidate) != aux or candidate.__getstate__() != state.data:
        raise NativeCheckpointError("Restoring auxiliary state changed native layout.")
    return candidate
