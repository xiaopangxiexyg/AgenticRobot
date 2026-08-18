# Sim/Real common test laboratory

`sim/sim_real_tests` contains executable acceptance scenarios for motion logic
that must behave the same in simulation and on a real robot, while never
connecting to real hardware. It is a diagnostic/test package, not a production
teleoperation entrypoint.

## Layout

```text
sim/sim_real_tests/
├── core/       # backend-neutral result and future scenario protocols
├── configs/    # versioned deterministic scenario inputs
├── rm65/       # model-only RM65 FK/IK/collision/qpos scenarios
├── mujoco/     # physics, actuator and contact scenarios
└── tests/      # pytest acceptance checks over the reusable scenarios
```

Dependency direction is strict:

```text
sim.sim_real_tests -> sim backend (optional) + robot_motion + pure contracts
```

Do not import `teleop_runtime`, vendor SDKs, USB devices, cameras, or real
network backends. Shared motion behavior missing from `robot_motion` should be
moved there first instead of copied into this directory.

## Current RM65 qpos-init scenario

The first scenario loads the measured collision-safe init qpos, generates a
small deterministic random reachable TCP pose through FK, solves that pose
back to qpos with the shared IK implementation, and plans from the previous
dual-arm qpos to the solved qpos. Every emitted qpos sample is checked against
the same strict collision proxy and hold distance used by real qpos init.

Run the lightweight structure checks in any project environment:

```bash
python -m pytest sim/sim_real_tests/tests/test_core_contracts.py -q
```

The package-local suite uses the repository-wide pytest markers, but remains
outside the default root `testpaths`. Activate the shared project environment
once from the selected worktree root:

```bash
conda activate "${DEXTELEOP_ENV_NAME:-dexteleop310}"
```

Run its fast feedback tier while editing:

```bash
python -m pytest sim/sim_real_tests/tests -q -m "not slow_sim"
```

Run only the long physical closed-loop checks on the designated integration
worktree:

```bash
python -m pytest sim/sim_real_tests/tests -q -m slow_sim
```

Run the complete Pinocchio/Pink/MuJoCo acceptance suite before integration:

```bash
python -m pytest sim/sim_real_tests/tests -q
```

`not slow_sim` is a development feedback selector, not a replacement for the
complete suite. The slow marker is attached to exact closed-loop test
functions; it does not reduce their sample counts, physical ticks, thresholds,
or inclusion in the unfiltered command.

Git does not populate ignored robot assets in a linked worktree. Configure the
clone-shared development asset store once and use the standard worktree claim
tool documented in [`CONTRIBUTING.md`](../../CONTRIBUTING.md). For a worktree
created before that tool, run:

```bash
python scripts/dev_asset_store.py link-worktree --worktree "$PWD"
python scripts/dev_environment_doctor.py --require-development-assets
```

The ignored `assets` symlink is a compatibility mapping, not proof that the
target filesystem is read-only. Do not edit the canonical store through it.
Treat a missing or mismatched asset as a preflight failure; do not copy assets
into Git or change scenario semantics to make the test pass. Runtime asset
identity and semantic hashes remain owned by the TargetRuntime contract rather
than this development-path mechanism.

The random stream is fixed by the JSON config. Increase `sample_count` for a
longer offline soak test; do not replace deterministic CI coverage with an
unbounded random loop.

## Adding scenarios

1. Put reusable scenario code under `rm65/` or the relevant backend package.
2. Add a versioned JSON input under `configs/`.
3. Return `SimRealScenarioResult`; do not print-only or hide failures in logs.
4. Add a bounded deterministic pytest under `tests/`.
5. Record minimum clearance, convergence errors, seed and executed step count.
6. Keep real-hardware confirmation and dispatch outside this package.
