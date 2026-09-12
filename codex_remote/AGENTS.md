# Remote Wrapper Instructions

Use `$remote-access` when invoking, diagnosing, or changing this wrapper set.
This scoped file governs changes to `codex_remote/` itself.

- Wrapper maintenance necessary for an authorized task may proceed within its
  declared scope. Changes to safety policy, allowed effects, or authorization
  boundaries require an explicit decision covering that behavior. Inspect the
  affected wrapper, helper, focused tests, and relevant README section.
- Preserve explicit target selection, fail-closed control-risk classification,
  read-only preflight semantics, bounded timeouts, and wrapper-owned quoting.
  Generic execution wrappers must not acquire device or motion authority from
  a token or command-line override.
  Do not add an ad hoc bypass for remote writes, process control, device state,
  or real motion.
- Never store passwords, tokens, private keys, host secrets, credentials, or
  machine-local absolute project paths in the repository.
- Run syntax checks and the focused wrapper tests applicable to the change.
  Do not execute a hardware or control command merely to validate formatting.
