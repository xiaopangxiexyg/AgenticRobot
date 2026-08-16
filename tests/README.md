# DexTeleop test suite

The current compatibility layout keeps general pytest modules at
`tests/test_<owner-or-domain>_<feature>.py` and classifies execution semantics
with markers.  The primary goal is to make tests easy to find, extend, and run;
directory symmetry is not a goal by itself.  Keep behavior changes, test
refactors, marker changes, and file moves in separate reviewable batches.

## Concurrent development mode

When production code or tests are being changed in parallel, use logical
classification instead of physical migration.  Do not move or split a test
module while it is modified, untracked, imported by another test, or named by
an active plan or command.  Marker-only governance changes may proceed when
they are isolated from the files under active ownership.

Treat existing root-level test paths as compatibility interfaces during
parallel development.  New general tests should also use a clear root-level
basename unless an owner already has a stable, reviewed test subtree.  Do not
create a new directory merely to classify one or two files.

A physical migration batch may start only when its exact file list has a short
exclusive ownership window, every file is tracked, and the relevant pre-move
baseline is green (or has a reviewed, exact list of unrelated failures).  A
new failure must be resolved before the move continues; it must not be folded
into the migration diff.

## Default suite

From the repository root:

```bash
python -m pytest --collect-only -q
python -m pytest -q -m "unit and not optional and not hardware and not slow_sim"
python -m pytest -q -m "unit and not optional and not hardware"
python -m pytest -q -m "contract and not optional and not hardware"
python -m pytest -q -m "integration and not optional and not hardware"
```

The current `pytest.ini` default discovers the root `tests/` directory.
Package-local suites under `sim/sim_real_tests/tests` and `hardware/*/tests`
remain separate until their dependency and hardware semantics have been
audited. The sim-real suite now carries the same marker vocabulary and exposes
its own fast/full commands in `sim/sim_real_tests/README.md`, but it is still
selected only by an explicit path. Do not treat a successful default run as
coverage of those suites.

## Cost gates in parallel worktrees

Use the fast unit selector plus the exact tests for the owner being changed on
ordinary feature worktrees. Run the relevant exact `slow_sim` node there only
when the change can affect it. Run the broad `slow_sim` selection and the
package-local sim-real full suite once on the designated integration worktree;
do not make every parallel worktree repeat the same physical closed loops.

`slow_sim` assignment is evidence-based and node-specific. Measure the exact
node on the designated worktree, keep its unfiltered acceptance behavior
unchanged, and add the marker only when simulation, shared IK replay,
simulation-side model loading, or a physical closed loop creates material
repeated cost. A module-level marker is valid only when every collected node in
that module has been measured into the same tier. A filename, loop bound,
optional dependency, or `mujoco` import alone is not sufficient evidence. Do
not share `.pytest_cache` between worktrees or enable repository-wide test
parallelism without first proving that the selected tests are hermetic and
resource-independent.

## Marker semantics

- `unit`: deterministic in-process behavior.  This is the transitional default
  for modules not yet listed in `tests/conftest.py`.
- `slow_sim`: an orthogonal cost marker for long-running deterministic
  simulation, IK replay, or physical closed-loop verification.  It remains in
  the complete `unit`/`integration` gates and is excluded only by an explicit
  fast selector.
- `integration`: multi-component, filesystem, worker, writer, or loopback
  behavior.
- `contract`: an orthogonal public schema, dependency-boundary, action, or data
  compatibility assertion; it may accompany `unit` or `integration`.
- `optional`: requires an optional dependency such as MuJoCo, OpenCV, Torch,
  PyArrow, or OpenPI.  Import optional packages with `pytest.importorskip` before
  importing project modules that require them so collection remains usable in
  smaller environments.
- `hardware`: requires a real device, SDK service, or operator-provided hardware
  environment.

The filename sets in `tests/conftest.py` are a transitional compatibility
mechanism that preserves current pytest node IDs.  New tests should carry the
most precise marker possible; unclassified tests must not silently grow into
multi-component or optional-dependency tests.

Use the `not slow_sim` selector for local edit feedback only.  Marker changes
must not reduce the complete merge or pre-push gate, and a slow test may move
to this tier only when an equivalent behavior is still executed by the full
suite.

## Finding and adding tests

Use the production owner or domain in the basename, followed by the behavior
under test.  Prefer adding a related case to an existing feature module; create
a new `test_<owner-or-domain>_<feature>.py` only when the responsibility is
distinct or the existing module is no longer maintainable.

Before adding a test, identify:

1. the production owner or cross-owner flow it verifies;
2. whether it is unit, integration, optional, contract, or hardware;
3. whether an existing test module already owns that behavior.

Keep static inputs under `tests/fixtures/`.  Introduce `tests/support/` helpers
only when they remove real duplication and have a reviewed import contract.
Do not add wrapper test modules, path aliases, symlinks, or a second pytest
entrypoint to simulate an organizational hierarchy.

## Real-hardware gate

Plain pytest runs must never connect to or change a real device.  Hardware tests
must have a `hardware` marker, perform no device I/O while their module is being
imported, and are skipped unless the operator supplies `--run-hardware`.

The explicit invocation shape is:

```bash
python -m pytest -q -m hardware --run-hardware <exact-test-node>
```

That flag is permission, not a safety check.  Before running it, follow the
device-specific documentation, name the exact device and test node, verify
workspace clearance and stop behavior, and obtain the required motion or
device-state authorization.  Mocked backend tests must not use the `hardware`
marker.

Operator scripts that connect to devices or can perform calibration or motion
are not pytest tests.  Keep them out of both `test_*.py` and `*_test.py` naming
and migrate them to the owning package's explicit `tools/field_validation/`
directory in a separate batch with documentation and caller checks.  The
project config intentionally limits default discovery to `test_*.py` so a
legacy operator tool cannot be imported merely because its name ends in
`_test.py`.

The tracked field-validation scripts formerly kept in `tests/` were migrated
on 2026-08-09.  RM65 state/force/configuration tools now live under
`hardware/rm65/tools/field_validation/`; the combined camera/tactile health
probe lives under `teleop_runtime/tools/field_validation/`.  Each destination
README records the exact old-to-new path map and hardware gates.  These tools
are outside pytest collection and were verified offline only during the move.

## Optional long-term layout

Execution risk is the primary directory axis; package ownership is the second
axis.  `contract` and `optional` remain orthogonal markers rather than competing
top-level directories.  This is an optional long-term direction for owners
that become stable enough to migrate as a complete, reviewable group; it is not
the required location for new tests during active parallel development.

```text
tests/
├── unit/
│   ├── control_composition/
│   ├── control_station/
│   ├── robot_motion/
│   ├── teleop_runtime/
│   ├── deployments/
│   ├── dataset_contracts/
│   ├── dataset_interface/
│   ├── sim/
│   └── hardware/          # mocked backends only
├── integration/
│   ├── control_station/
│   ├── teleop_runtime/
│   ├── policy_deploy/
│   ├── dataset_pipeline/
│   ├── sim/
│   └── sim_real/
├── performance/
├── support/
└── fixtures/
```

Real-device/HIL pytest suites, if introduced, require a separately reviewed
location and retain the `hardware` marker plus explicit execution gate.

## Safe migration protocol

If a physical migration has a concrete maintenance benefit, migrate one domain
at a time only after its files are tracked, stable, and not being edited by
another owner:

1. Record collection count, test names, parameter IDs, markers, skips, and
   errors.
2. Remove directory-depth assumptions such as
   `Path(__file__).resolve().parents[1]` before moving the file.
3. Record the exact old-to-new path map, then use `git mv`; update exact
   test-node references in documentation and scripts.
4. Compare pre/post collection while allowing only the documented path-prefix
   change.
5. Run the smallest affected tier plus `git diff --check`.
6. Never use an ordinary test migration as authorization for hardware access.

The optional layout above supersedes the narrower simulation-era example in
`sim/STRUCTURE_PLAN.md`.  Logical classification, old-path compatibility, and
reliable collection take priority over visual directory uniformity.
