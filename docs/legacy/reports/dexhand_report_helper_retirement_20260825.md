# DexHand report helper retirement

Date: 2026-08-25

## Decision

`DexHandTester._generate_report` was removed from
`hardware/pyzlg_dexhand/tools/hardware_test/test_dexhand.py`. The helper was a
large, private report writer that had no active caller: `run_tests` retained
only a commented-out call, while the live test sequence proceeds through reset,
individual-joint, simultaneous-movement, and consecutive-movement tests.

This change removes only the unused report-generation helper and its stale
comment. The `DexHandTester` constructor, device-test entry point, result
collection, `_plot_simultaneous_results`, and all four live test phases remain
unchanged. No replacement report path or compatibility shim is introduced.

## Audit evidence

- Before the edit, a repository-wide search for `_generate_report` and
  `generate_report` found only the private definition and the commented-out
  `run_tests` call in the target file; there were no imports, exports, tests, or
  documentation invoking the helper.
- After the edit, the same search excluding `docs/legacy/**` is empty.
- The report is versioned under `docs/legacy/` as historical explanation only.
  The root `legacy/` directory remains strictly local and untracked.

## Preserved behavior and safety boundary

The hardware test entry point still constructs `DexHandTester` and calls
`run_tests`, including reset, individual, simultaneous, and consecutive test
flows. `_plot_simultaneous_results` remains part of the active simultaneous
test path. This retirement does not change hand commands, feedback collection,
logging, plotting, cleanup, or CLI options.

Validation is intentionally limited to static evidence for this retirement.
An attempted run of the package test directory was not safe offline: the
`test_dexhand_interface.py::test_config_loading` path reached the native
`zcan.open_device` call and terminated with a segmentation fault before
assertions completed. The suite was stopped and will not be rerun here; this
result is neither hardware evidence nor an acceptance result. No deliberate
DexHand command, motion, calibration, or acceptance procedure was run by this
Task.

## Verification

```bash
git grep -n -E '_generate_report|generate_report' -- . ':!docs/legacy/**'
PYTHONDONTWRITEBYTECODE=1 python -c 'from pathlib import Path; compile(Path("hardware/pyzlg_dexhand/tools/hardware_test/test_dexhand.py").read_text(), "hardware/pyzlg_dexhand/tools/hardware_test/test_dexhand.py", "exec")'
# The package test directory is hardware-touching and is not a safe offline
# validation command; keep it out of this retirement's checks.
git diff --check
git ls-tree -r --name-only HEAD legacy
python scripts/dev_local_resources.py audit
```

The no-write source compilation, reference audit, diff check, and resource
policy audit provide static repository evidence only; they do not establish
real-device readiness.
