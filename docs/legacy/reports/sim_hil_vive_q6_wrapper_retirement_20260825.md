# Sim-HIL VIVE/Q6 wrapper retirement

Date: 2026-08-25

## Decision

The unused private `_build_passive_vive_q6_component_source` wrapper was
removed from `control_station/sim_hil_profile/human.py`, together with its
stale import in `control_station/sim_hil.py`. The wrapper had no caller: its
only repository hits before this change were its own definition and the
unused application-root import.

This is a dead-wrapper removal only. The active single-side Sim-HIL human
source path, control/safety/device lifecycle ownership, and lower-owned
VIVE/Q6 builders remain unchanged.

## Audit evidence

- A repository-wide search for
  `_build_passive_vive_q6_component_source` found only the two removed hits;
  there were no tests, exports, dynamic lookups, or documentation callers.
- The public
  `teleop_runtime.configs.passive_vive_q6.build_passive_vive_q6_session`
  builder remains versioned and unchanged, as does
  `build_passive_single_vive_q6_session`.
- The active Control Station binding continues to use
  `_build_passive_single_vive_gripper_component_source`, which delegates
  construction to the lower `teleop_runtime` owner.
- No production code imports the root `legacy/` directory. Shared retirement
  history is recorded here under `docs/legacy/`; the root `legacy/` directory
  remains strictly local and untracked.

## Preserved behavior and safety boundary

The removal does not change the public passive VIVE/Q6 session builders, the
single-side VIVE/gripper component contract, target action-spec validation,
VIVE coordinate/Home binding, source admission, control composition, safety,
dispatch, or device lifecycle behavior. The bimanual private wrapper was not
reachable from the Sim-HIL application graph, so no active path depended on
its component construction.

No VIVE, Q6, arm, gripper, MuJoCo, or other simulator/device process was
started by this Task. The evidence below is static/offline repository evidence
only and does not establish device readiness, simulator acceptance, or physical
teleoperation readiness.

## Verification

```bash
git grep -n -w _build_passive_vive_q6_component_source -- . ':!docs/legacy/**'
PYTHONDONTWRITEBYTECODE=1 python -c 'from pathlib import Path; [compile(Path(p).read_text(), p, "exec") for p in ("control_station/sim_hil_profile/human.py", "control_station/sim_hil.py")]'
git diff --check
git ls-tree -r --name-only HEAD legacy
python scripts/dev_local_resources.py audit
```

The reference audit should be empty, both targeted modules should compile
without writing bytecode, the root `legacy/` tree should be empty, and the
resource policy audit should pass. These checks are structural evidence only.
