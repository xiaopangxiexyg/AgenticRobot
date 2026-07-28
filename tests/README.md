# DexTeleop test suite

The test suite is being classified before any bulk path migration.  Keep
behavior changes, test refactors, and file moves in separate reviewable
batches.

## Concurrent development mode

When production code or tests are being changed in parallel, use logical
classification instead of physical migration.  Do not move or split a test
module while it is modified, untracked, imported by another test, or named by
an active plan or command.  Marker-only governance changes may proceed when
they are isolated from the files under active ownership.

A physical migration batch may start only when its exact file list has a short
exclusive ownership window, every file is tracked, and the relevant pre-move
baseline is green (or has a reviewed, exact list of unrelated failures).  A
new failure must be resolved before the move continues; it must not be folded
into the migration diff.

## Default suite

From the repository root:

```bash
python -m pytest --collect-only -q
python -m pytest -q -m "unit and not optional and not hardware"
python -m pytest -q -m "contract and not optional and not hardware"
python -m pytest -q -m "integration and not optional and not hardware"
```

The current `pytest.ini` default discovers the root `tests/` directory.
Package-local suites under `sim/sim_real_tests/tests` and `hardware/*/tests`
remain separate until their dependency and hardware semantics have been
audited.  Do not treat a successful default run as coverage of those suites.

## Marker semantics

- `unit`: deterministic in-process behavior.  This is the transitional default
  for modules not yet listed in `tests/conftest.py`.
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

## Target layout

Execution risk is the primary directory axis; package ownership is the second
axis.  `contract` and `optional` remain orthogonal markers rather than competing
top-level directories.

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

Migrate one domain at a time only after its files are tracked, stable, and not
being edited by another owner:

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

The project-wide target above supersedes the narrower simulation-era example
in `sim/STRUCTURE_PLAN.md`.  Until migration is complete, classification and
reliable collection take priority over visual directory uniformity.
