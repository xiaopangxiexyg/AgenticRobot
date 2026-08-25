# Tracker-hand legacy RPY predicate retirement

Date: 2026-08-25

## Decision

The unused `canonical_hand_rpy_is_legacy` predicate was removed from
`teleop_runtime/tracker_hand.py`. A repository-wide search found only the
function definition; there were no production, import, test, documentation, or
dynamic-lookup callers.

This is a dead-predicate removal. The active tracker-hand frame contract is
unchanged: `canonical_hand_rpy_default`,
`canonical_tracker_to_hand_rotation`, `TRACKER_OFFSET_CALIBRATION_VERSION`,
and `apply_tracker_hand_frame_correction` remain available. The canonical
defaults, tracker-to-hand rotation, calibration version, pose conversion, and
all real calibration/control behavior are preserved.

## Evidence and boundaries

- `git grep -n -w canonical_hand_rpy_is_legacy -- . ':!docs/legacy/**'` found
  only the removed definition before the edit.
- The symbol was not present in a checked-in `__all__`, package lazy-export
  list, or documented/registered entrypoint. This proves only that the
  versioned repository has no known caller; because the name has no leading
  underscore, an external private import cannot be ruled out by repository
  evidence alone.
- The remaining public helpers and calibration constants stay in their active
  import path; no replacement shim is needed because the retired predicate had
  no callers.
- The root `legacy/` directory remains strictly local and untracked. This
  report is the versioned historical explanation under `docs/legacy/`.
- No device, simulator, hardware probe, calibration run, or motion command was
  started. The checks below are static/offline evidence only and do not claim
  hardware readiness.

## Verification

```bash
git grep -n -w canonical_hand_rpy_is_legacy -- . ':!docs/legacy/**'
python -m py_compile teleop_runtime/tracker_hand.py
git diff --check
python scripts/check_docs_navigation.py
python scripts/dev_local_resources.py audit
git ls-tree -r --name-only HEAD legacy
```

The target source compiles, the retired symbol is absent from active tracked
paths, the new report link resolves from `docs/real/README.md`,
local-resource policy remains valid, and the root `legacy/` tree remains empty.
The repository-wide documentation navigation check is currently blocked by the
separate concurrent HIL retirement report, which is not yet linked on this
Task's base (`docs/legacy/reports/sim_hil_vive_q6_wrapper_retirement_20260825.md`).
That unrelated navigation issue must be rechecked after the HIL Task is
integrated. These checks do not constitute real-device or physical
teleoperation acceptance.
