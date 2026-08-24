# Robosuite environment boundary (A1)

A1 binds one deliberately narrow simulation profile:

- robosuite `1.5.2`, MuJoCo `3.9.0`, NumPy `1.26.4`, SciPy `1.13.1`;
- Python 3.10;
- `Lift`, `Panda`, BASIC composite controller with right-arm `OSC_POSE`;
- 7-D normalized delta-pose plus gripper action at 20 Hz;
- no renderer, offscreen renderer, camera observation, GUI, or hardware.

The repository's main simulation extra pins a different MuJoCo version. This
project is therefore an independent uv environment and must not be installed
into `dexteleop310`. Keep its virtual environment and cache outside every Git
worktree:

```bash
repo_root="$(git rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT=/tmp/agenticrobot-robosuite-a1
export UV_CACHE_DIR=/tmp/agenticrobot-robosuite-uv-cache-a1
uv sync --project "$repo_root/environments/robosuite" --frozen
PYTHONPATH="$repo_root" \
  "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest --noconftest -q \
  "$repo_root/tests/test_robosuite_backend_a1.py" \
  "$repo_root/tests/test_streaming_intervention_robosuite_a1.py"
```

`--noconftest` is required here because the repository-wide test fixture owns
the main MuJoCo 3.10 log configuration. These two A1 files are self-contained
and must not import that incompatible main-environment hook into MuJoCo 3.9.

`sim/robosuite/` imports without loading the optional robosuite package. The
real package is loaded only when `PandaLiftOscPoseBackend.reset()` or restore is
called, and exact version drift is rejected before that import.

## Reset and replay boundary

The canonical episode start is not the raw observation returned by
`env.reset()`. A1 captures compiled XML, flattened initial MuJoCo state,
the source effective-XML hash, episode metadata, RNG state,
construction/requested-controller hashes, the observed active-controller
profile, action bounds, model dimensions, and runtime fingerprint. The active
profile closes the actual BASIC action split and OSC_POSE delta/base scaling,
gains, interpolation, impedance, and control-frequency semantics instead of
trusting only the requested JSON. A1 then reapplies that episode start and
force-refreshes the pinned robosuite 1.5.2 observation seam. That normalized
observation is the reset receipt's observation.

Exact seek means creating a fresh environment, restoring that captured episode
start, and replaying every final executed action from index zero. A flattened
mid-episode MuJoCo state plus an action suffix is explicitly unsupported: it
does not restore stateful OSC controller/interpolator state. Bit-exact evidence
is limited to the same locked runtime and platform fingerprint. Installation
paths embedded in compiled XML are rewritten with robosuite's
`edit_model_xml()`; the replay receipt records the effective XML hash rather
than pretending it must equal the original XML bytes. Source and replay
effective hashes must match before an exact-prefix receipt can pass.

The runtime compatibility fingerprint binds the complete A1 `uv.lock` digest,
Python SOABI, libc identity, direct wheel tags and package-owned RECORD
manifests, plus verified bytes for the MuJoCo native engine and Python binding.
It intentionally excludes virtual-environment paths. Equality is a
fail-closed compatibility gate; it is not proof of bit-exact behavior across
hosts or CPUs. Exactness is established only by the recorded reset hash,
effective-model hash, selection hashes, and every replayed backend-step hash in
the tested same-host runtime.

This profile is a headless simulation/runtime and protocol boundary only. It
does not establish policy quality, task success rate, data-learning value,
cross-platform determinism, GUI behavior, or any real-robot readiness. The
runner exercises the existing A0 base-only, absolute-takeover, and additive
relative-overlay modes; it does not yet provide a policy, correction provider,
dataset writer, learner, or benchmark result.
