# AgenticRobot test suite

The current compatibility layout keeps general pytest modules at
`tests/test_<owner-or-domain>_<feature>.py` and classifies execution semantics
with markers.  The primary goal is to make tests easy to find, extend, and run;
directory symmetry is not a goal by itself.  Keep behavior changes, test
refactors, marker changes, and file moves in separate reviewable batches.

## Concurrent development mode

When production code or tests are being changed in parallel, use logical
classification instead of physical migration.  Do not move or split a test
module while another owner is editing it. Update current imports, scripts, and
documented commands together when a move is justified; an old plan is not a
permanent path freeze. Marker-only governance changes may proceed when
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
bash scripts/run_pytest.sh --collect-only -q
# Bounded edit feedback:
bash scripts/run_pytest.sh -q -m "unit and not optional and not hardware and not slow_sim"
bash scripts/run_pytest.sh -q -m "contract and not optional and not hardware and not slow_sim"
bash scripts/run_pytest.sh -q -m "integration and not optional and not hardware and not slow_sim"
# Parallel bounded edit feedback (unit + integration, isolated caches/reports):
bash scripts/run_fast_feedback.sh
# Complete compatibility and integration-commit acceptance:
bash scripts/run_full_acceptance.sh --durations=50
```

The current `pytest.ini` default discovers the root `tests/` directory.
Package-local suites under `sim/sim_real_tests/tests` and `hardware/*/tests`
remain separate until their dependency and hardware semantics have been
audited. The sim-real suite now carries the same marker vocabulary and exposes
its own fast/full commands in `sim/sim_real_tests/README.md`, but it is still
selected only by an explicit path. Do not treat a successful default run as
coverage of those suites.

## Selecting evidence for a change

| Change | Development feedback | Before integration |
| --- | --- | --- |
| Documentation or Skill wording | Links, examples, relevant contract checks | Review the diff; no new text-matching tests by default |
| Behavior or bug fix | Focused owner tests, including the demonstrated failure | Affected consumers and failure/state transitions |
| Public contract, dependency boundary, refactor | Contract and owner tests; compare collection if tests move | Relevant cross-package suites and full non-hardware acceptance on the combined commit |
| Simulation/control dynamics | Deterministic replay or simulator cases with declared inputs | Relevant complete simulation acceptance; report hardware separately |

Use coverage, when available in the test environment, to locate missed changed
branches. For a bug fix, reproduce the fault when practical and check that the
regression assertion distinguishes the faulty behavior from the intended result.
Existing behavioral tests can suffice for a refactor; mock external boundaries
without mocking away the behavior being verified. Select relevant failure
branches: invalid input, cancellation, timeout, stale state, persistence failure,
and recovery. Do not prescribe an arbitrary repository percentage or create
duplicate tests to raise it. A minimal environment's optional skips are useful
feedback but do not satisfy acceptance for changed optional behavior. Record
missing dependencies and known unrelated failures explicitly.

## Cost gates in parallel worktrees

Use the fast unit and integration selectors plus the exact tests for the owner
being changed on ordinary feature worktrees. Run the relevant exact
`slow_sim` node there only when the change can affect it. Run the full
`not hardware` gate, the broad `slow_sim` selection, and the package-local
sim-real full suite once for the exact designated integration commit; do not
make every parallel worktree repeat the same high-cost acceptance behavior.

`slow_sim` is the compatibility name for the measured high-cost tier. Assign
it from node-level timing evidence while keeping the unfiltered acceptance
behavior unchanged. Simulation, shared IK replay, simulation-side model
loading, repository-wide source scans, physical closed loops, and high-volume
persistence soaks may qualify.
A module-level marker is valid only when every collected node in that module
has been measured into the same tier. A filename, loop bound, optional
dependency, or `mujoco` import alone is not sufficient evidence. Do not share
`.pytest_cache` between worktrees or invoke independent pytest processes over
overlapping modules. `scripts/run_fast_feedback.sh` is the reviewed parallel
entrypoint for the two bounded selectors. It splits the unit selector into two
duration-weighted, deterministic test-module shards by default, starts those
shards alongside the integration child, and merges the shard reports back to
the historical `unit.log` and `unit.xml` evidence paths. Each child gets its
own pytest cache and JUnit report under `/tmp`. Set
`DEXTELEOP_FAST_FEEDBACK_UNIT_SHARDS=1` to
reproduce one unit process.  A three-shard run has been validated on the
current 32-CPU development host; choose it only when the machine's CPU and
memory budget supports the extra pytest process.  Four shards remain
experimental and are not part of the current integration default.  Explicit
test paths or node IDs
automatically disable unit sharding for that targeted invocation so a test is
never selected twice.  If a clean worktree does not already have the reviewed
`assets/` link, set `DEXTELEOP_ASSET_ROOT` to an absolute shared asset root
before invoking it; the wrapper creates one shared temporary link for all
children and removes only that link after the run.  Additional pytest options
may be appended, but the marker, cache, JUnit, and basetemp options remain
owned by the wrapper.

`scripts/run_full_acceptance.sh` is the complete non-hardware integration gate.
Before allocating an artifact root or starting any shard, it verifies that the
host can create both loopback TCP and Unix sockets required by the complete
suite. An incapable sandbox therefore fails immediately instead of running the
remaining modules and reporting predictable transport errors at the end.
It assigns every discovered `tests/test_*.py` module exactly once using a
checked-in sparse timing profile plus a deterministic source-size fallback.
The sparse full-suite weights come from testcase durations in a complete,
socket-capable green Run and cover only modules that contributed at least five
seconds there. Refresh them only from another exact green full-suite artifact;
failed, capability-limited, or focused runs are not representative inputs.
The default remains four shards. Each shard has separate pytest, Python, XDG,
Matplotlib, Numba, `TMPDIR`, basetemp, log, and JUnit state. Shards in one exact
Run share only its content-addressed MuJoCo model cache; that cache uses process
locking, atomic publication, and hash validation, and it is never reused across
Run roots. The wrapper waits for every child, rejects duplicate JUnit
identities, and writes `full-acceptance.xml` plus `shard-plan.json` under a new
artifact root. Set
`DEXTELEOP_FULL_ACCEPTANCE_SHARDS=1` for serial reproduction. Explicit five-
and six-shard runs are supported only on hosts whose CPU and memory headroom
has been validated; values above six are rejected because the measured suite
is already limited by its slowest module. Keep the deterministic default
instead of deriving this setting automatically from `nproc`. Harness callers
may set new absolute `DEXTELEOP_FULL_ACCEPTANCE_RUN_ROOT` and
`DEXTELEOP_FULL_ACCEPTANCE_JUNIT` paths so the aggregate evidence remains in
the protected Run directory. Explicit test targets are rejected because this
entrypoint represents complete-suite acceptance.

## Marker semantics

- `unit`: deterministic in-process behavior.  This is the transitional default
  for modules not yet listed in `tests/conftest.py`.
- `slow_sim`: the legacy-compatible orthogonal cost marker for measured
  long-running deterministic simulation, IK replay, repository-wide source
  scans, physical closed-loop, or high-volume soak verification. It remains in
  the complete `unit`/`integration` and `not hardware` gates and is excluded
  only by an explicit fast selector.
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

Use the `not slow_sim` selector for local edit feedback only. Marker changes
must not reduce the complete integration-commit gate, and a slow test may move
to this tier only when the same behavior is still executed by the full suite.

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
bash scripts/run_pytest.sh -q -m hardware --run-hardware <exact-test-node>
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
