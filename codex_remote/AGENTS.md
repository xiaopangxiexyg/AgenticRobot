# Remote Wrapper Instructions

Use `$remote-access` when invoking or diagnosing this wrapper set. This scoped
file governs changes to `codex_remote/` itself.

- Modify wrappers, shared safety logic, or safety policy only when the user
  explicitly requests that surface. Read `codex_remote/README.md`, state the
  safety impact, and keep the change narrowly scoped.
- Preserve explicit target selection, fail-closed control-risk classification,
  fresh safety-token validation, bounded timeouts, and wrapper-owned quoting.
  Do not add an ad hoc bypass for remote writes, process control, device state,
  or real motion.
- Never store passwords, tokens, private keys, host secrets, credentials, or
  machine-local absolute project paths in the repository.
- Run syntax checks and the focused wrapper tests applicable to the change.
  Do not execute a hardware or control command merely to validate formatting.
