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
- `control_station/`: top-level application/control plane for composing human,
  replay, policy, dataset, sim, and real-target modules. It may depend on the
  active lower-level packages, but none of those packages may import it.
- `robot_motion/`: shared motion semantics, Rig, Safety, Planner, position
  control, and runtime contracts; no vendor-device initialization or training.
- `hardware/`: vendor SDKs and device backends; no dataset schema or learner.
- `sim/`: simulation-only backends, tasks, and adapters; may use `robot_motion`,
  but must not depend on `teleop_runtime` or real hardware.
- `teleop_runtime/`: real////-robot teleop, GUI/headless orchestration, collection,
  and policy deployment frontends.
- `deployments/`: model client, protocol, and artifact integration.
- `dataset_contracts/` and `dataset_interface/`: pure data contracts and
  trajectory/window/batch adapters; no hardware or training implementation.
- `model/`: future policy/learner/DAgger/RL package, not a current runtime entrypoint.
- `legacy/`: confirmed-archived code/docs only; production code must not import it.
- `tests/` and `todos/`: verification and planning; neither is runtime code.

Shared code should depend toward `control_composition`, `robot_motion`, and pure
data contracts. `control_composition` depends downward on `robot_motion`; the
reverse dependency is forbidden. `sim` must not reverse-depend on
`teleop_runtime`. `control_station` is an application leaf: dependencies point
into it from no lower-level package.

## Safety and change protocol

- Prefer read-only, dry-run, mock, contract, and `--help` checks.
- Do not start real devices or execute motion, homing, replay, servo,
  force-control, reboot, or other device-state changes without explicit scope
  and the relevant safety check.
- Remote hardware work must use the wrappers and rules in `codex_remote/`.
- Inspect status/diff first, preserve concurrent work, keep edits scoped, and
  do not reset or overwrite unrelated changes.
- Ask for clarification when an unresolved choice affects safety, public
  control/data semantics, ownership, or destructive behavior; otherwise state
  assumptions and proceed.
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
