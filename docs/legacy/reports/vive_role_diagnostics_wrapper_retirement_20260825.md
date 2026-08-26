# VIVE role-diagnostics wrapper retirement

Date: 2026-08-25

## Decision

The private `VivePoseSource._parse_upstream_role_diagnostics` bytes-wrapper was
removed from `teleop_runtime/sources/pose/vive.py`. A repository-wide search found
only its definition; no production code, tests, documentation, export, or dynamic
lookup used it.

The active receive path already calls
`_parse_decoded_upstream_role_diagnostics` after packet decoding. That decoder,
upstream-role channel publication, role statistics, and the VIVE source lifecycle
remain unchanged. This is a dead-wrapper removal, not a change to the VIVE wire
format, role mapping, or observation contract.

## Evidence and boundaries

- `git grep -w _parse_upstream_role_diagnostics` had only the removed definition.
- `_process_received_batch` continues to call
  `_parse_decoded_upstream_role_diagnostics` directly.
- The change does not initialize devices, open sockets, or claim hardware
  readiness; focused tests are offline contract evidence only.
- The root `legacy/` directory remains strictly local and untracked. This report
  is the versioned historical explanation under `docs/legacy/`.

## Verification

```bash
git grep -n -w _parse_upstream_role_diagnostics -- . ':!docs/legacy/**'
bash scripts/run_pytest.sh tests/test_vive_pose_source.py -q
python -m py_compile teleop_runtime/sources/pose/vive.py
python scripts/check_docs_navigation.py
python scripts/dev_local_resources.py audit
```

The focused VIVE tests and source compilation must pass before integration. They
do not constitute real-device or physical teleoperation acceptance.
