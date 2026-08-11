# DexTeleop Project Rules

DexTeleop contains shared robot motion, real-robot teleoperation, data
collection, policy deployment, simulation, and dataset contracts. Keep the
directory boundaries stable and do not add a new production entrypoint without
checking the existing README and architecture plan.

## Canonical entrypoints

- GUI human teleop and formal collection:
  `python -m teleop_runtime.debug.run_vive_rm65_real_teleop --gui`
- Headless/compatibility collection:
  `python -m teleop_runtime.collection`
- Policy deployment:
  `python -m teleop_runtime.deploy`

`teleop_runtime.collection` is headless/compatibility only. `teleop_runtime/core/kernel.py`
is experimental.

## Directory ownership

- `control_composition/`: reusable, backend-neutral multi-source control
  composition, lifecycle reducers, causal lineage, and non-authoritative
  difference/feedback. It may depend on `robot_motion` and pure data contracts,
  but must not import concrete sim, hardware, teleop, deployment, or
  `control_station` modules. It owns no authority, canonical safety, execution,
  device/model worker, replay cursor, or episode lifecycle.
- `control_station/`: GUI-facing application control plane. It owns typed
  operator intents, application lifecycle coordination, rebuildable UI state,
  profile selection, and thin source/target integration. It may coordinate
  lower-owned runtimes through typed ports and receipts, but must not
  reimplement reusable composition algorithms, source/model workers, replay
  cursors, simulator/backend execution, dataset writers, canonical safety,
  reference generation, or dispatch. Concrete application bindings may depend
  on active lower-level packages, but none of those packages may import
  `control_station`. Existing migration exceptions must remain explicit in
  boundary tests or documentation.
- `robot_motion/`: shared motion semantics, Rig, Safety, Planner, position
  control, and runtime contracts; no vendor-device initialization or training.
- `hardware/`: vendor SDKs and device backends; no dataset schema or learner.
- `sim/`: simulation-only backends, tasks, and adapters; may use `robot_motion`,
  but must not depend on `teleop_runtime` or real hardware.
- `collection_runtime/`: device-, control-, and UI-neutral live collection
  lifecycle and persistence mechanics. It may depend on `dataset_contracts`,
  `dataset_interface`, and pure domain contracts, but must not import
  `teleop_runtime`, `control_station`, concrete hardware, simulation, or GUI
  modules. It owns no device production, robot control, operator intent, or
  dataset schema definition.
- `teleop_runtime/`: real-robot teleop, GUI/headless orchestration, source-bound
  collection adapters, compatibility collection paths, and policy deployment
  frontends. Compatibility shims may re-export migrated collection runtime
  APIs but must not duplicate their implementations.
- `deployments/`: model client, protocol, and artifact integration.
- `dataset_contracts/`: pure schemas, identities, hash domains, and validation
  contracts; no runtime, device, GUI, or control ownership.
- `dataset_interface/`: data adaptation, projection, reader, window/batch, and
  validation interfaces. It may depend on `dataset_contracts` and pure domain
  contracts, but must not become a device/control runtime or authoritative
  episode owner, and must not import `control_station`.
- `model/`: future policy/learner/DAgger/RL package, not a current runtime entrypoint.
- `legacy/`: confirmed-archived code/docs only; production code must not import it.
- `tests/` and `todos/`: verification and planning; neither is runtime code.

Shared code should depend toward `control_composition`, `robot_motion`, and pure
data contracts. `control_composition` depends downward on `robot_motion`; the
reverse dependency is forbidden. `sim` must not reverse-depend on
`teleop_runtime`. `control_station` is an application leaf: dependencies point
into it from no lower-level package.

Import legality does not imply semantic ownership. Code belongs to the package
that owns its reusable domain semantics or resource lifecycle, not merely to
the highest-level caller that currently uses it.

For one logical runtime resource or stream, source production, cursor or plan
advancement, authority, canonical safety, dispatch, and episode commitment each
have one authoritative owner. Passive bounded observers, fanout queues, and UI
caches are allowed, but they must not mutate or independently reconstruct the
owned state.

## Delivery scope and evidence

- Judge completion against the user-requested outcome. A contract, module,
  fixture, or profile may be complete in its own requested scope, but does not
  prove downstream runtime integration. When broader integration could
  reasonably be inferred, state the actual caller or its absence, the
  environment and furthest boundary exercised, and the next relevant unverified
  boundary. Do not impose a fixed evidence sequence.
- New plans, profiles, adapters, fixtures, wrappers, exports, or modules are not
  progress by themselves. Add one only when requested or needed for the current
  implementation, compatibility or rollback, or relevant verification. Describe
  structural progress by the responsibility, dependency, or call edge changed,
  not merely by files moved, abstractions added, or test counts.

## Safety and change protocol

- Prefer read-only, dry-run, mock, contract, and `--help` checks.
- Do not start real devices or execute motion, homing, replay, servo,
  force-control, reboot, or other device-state changes without explicit scope
  and the relevant safety check.
- Remote hardware work must use the wrappers and rules in `codex_remote/`.
- Inspect status/diff first, preserve concurrent work, keep edits scoped, and
  do not reset or overwrite unrelated changes.
- Before any dataset transform that combines hardlinks, partial selection, or
  overwrite, verify source/target device, inode, link count, and independent
  backup receipts. A differently named path is not an independent backup.
- Dataset video transforms must write a fresh temporary inode, validate the
  declared/decoded frame count and tail decode, then atomically replace the
  target. Stop at the first decoder warning or count mismatch; never accept
  process exit status alone as data-integrity evidence.
- Ask for clarification when an unresolved choice affects safety, public
  control/data semantics, ownership, or destructive behavior; otherwise state
  assumptions and proceed.
- For any stateful control modifier, bound the final composed output including
  persistent terms at the last authoritative boundary. Component and rate
  limits are not total-action limits; persistent feedback must use an explicit
  incremental baseline rather than an absolute target-to-command residual.
- Shadow evidence cannot authorize Active-only feedback or state transitions;
  exercise them offline in closed loop, including nonzero nominal residuals,
  before any real-motion command is provided.
- After unexpected real motion, freeze that Active path and preserve its exact
  command, configuration, logs, and process evidence. Diagnose read-only and do
  not tune or retry until both the cause and the missing safety gate are closed.
- Use `AGENT_TEMP_NOTE:` only for active collaboration and remove obsolete notes.

## Verification and environment metadata

Run the smallest relevant tests and report them. Use `git diff --check`; add
targeted tests, `py_compile`, and entrypoint `--help` checks as appropriate.
Hardware tests and remote runs require explicit scope.

Only dependency, Python-version, package-discovery, or installation changes
require updates to `pyproject.toml`, `environment.yml`, `uv.lock`,
`scripts/install_conda_env.sh`, or `README_ENV.md` as applicable. Source-only
and documentation changes do not require lock updates. Never hand-edit
`uv.lock`; regenerate it after dependency changes.

## Legacy policy

Move code to `legacy/` only after confirming there are no production, import,
test, or documentation callers, a replacement exists, and a migration note
records the reason, replacement, and deletion condition. Keep required
compatibility shims in their active import paths.
