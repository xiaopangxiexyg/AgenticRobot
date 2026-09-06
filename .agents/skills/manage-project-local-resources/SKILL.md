---
name: manage-project-local-resources
description: Audit, add, link, copy, migrate, or rebind AgenticRobot ignored local resources across worktrees, including datasets, records, assets, model or vendor bundles, machine configuration, secrets, generated output, and runtime artifacts. Use when .gitignore, the local-resources manifest, asset-store links, or worktree-local resource ownership is involved.
---

# Manage Project Local Resources

Treat ignored state as owned data, not disposable Git noise. Read
[the local-resource policy](../../../docs/setup/LOCAL_RESOURCES.md) and
`docs/setup/local_resources_manifest_v1.json` before changing paths or links.

## Workflow

1. **Inventory.** Run `python scripts/dev_local_resources.py audit`. Record the
   source and target worktrees, physical paths, Git classification, resource
   kind, mutability, writer, and current consumers.
2. **Select one canonical store.** Reuse the manifest's approved store or add a
   reviewed entry. Do not infer ownership from `.gitignore`, path similarity,
   age, or size.
3. **Choose transfer semantics.**
   - Link only an approved read-mostly asset store.
   - Keep mutable runtime, collection, database, socket, log, and generated
     evidence writer-isolated.
   - Use inventory plus a verified copy for writer handoff; do not silently
     rebind a mutable producer.
4. **Protect repository boundaries.** Never replace a mixed tracked/ignored
   root with a whole-root symlink. Never make tracked code, tests, profiles, or
   documentation depend on a developer-local absolute path.
5. **Update the contract together.** When the resource policy changes, update
   the manifest, policy documentation, ignore rules, and focused policy test in
   the same commit.
6. **Verify.** Re-run the audit, inspect `git status --ignored` for the focused
   paths, resolve symlink targets physically, and verify that no peer worktree
   gained untracked or writable content unexpectedly.

## Data safety

- Before copying or transforming hardlinked or overwrite-prone data, verify
  source/target device, inode, link count, and an independent backup receipt.
- Write media transforms to a fresh inode, validate declared and decoded counts
  plus tail decode, then replace atomically.
- Stop at the first decoder warning, count/hash mismatch, changed link target,
  ambiguous writer, or unavailable backup.
- Never delete or clean a resource merely to make a worktree appear clean.

Report what is versioned, ignored, linked, copied, mutable, and externally
stored separately.
