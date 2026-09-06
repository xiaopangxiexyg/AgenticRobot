---
name: manage-project-local-resources
description: Audit, add, link, copy, migrate, recreate, or rebind AgenticRobot resources intentionally kept outside Git, including datasets, records, assets, model or vendor bundles, machine-local configuration, development certificates, generated output, and runtime artifacts. Use when .gitignore, the local-resources manifest, asset-store links, physical storage, or cross-worktree resource placement changes; do not use for tracked source refactors, dataset semantics, or remote command execution itself.
---

# Manage Project Local Resources

Own where ignored local resources live and how they are linked, copied,
recreated, or rebound. `$remote-access` owns any remote transfer command; the
domain package owns dataset schemas and runtime semantics.

Treat ignored state as project data, not disposable Git noise. Inspect the
relevant class in
[the local-resource policy](../../../docs/setup/LOCAL_RESOURCES.md) and only the
matching entries in `docs/setup/local_resources_manifest_v1.json`; do not load
the full manifest for a focused task.

## Workflow

1. **Inventory.** Run `python scripts/dev_local_resources.py audit`. Record the
   source and target worktrees, physical paths, Git classification, resource
   kind, mutability, writer, and current consumers.
2. **Choose the operation by resource behavior.**
   - Link stable read-mostly assets when the approved asset store is canonical.
   - Bind or copy external datasets according to the owning dataset workflow.
   - Snapshot or stop the writer before copying mutable runtime or collection
     output, then switch its binding explicitly.
   - Copy, recreate, or rebind machine-local configuration and development
     certificates according to the target machine. Exact copying is acceptable
     in a trusted development environment when it is intentional; keep these
     files outside Git and preserve required file permissions.
   - Rebuild generated caches and environments when that is cheaper than moving
     them.
3. **Select one canonical location.** Reuse the manifest's approved store or add
   a reviewed entry. Do not infer ownership from `.gitignore`, path similarity,
   age, or size.
4. **Protect repository boundaries.** Never replace a mixed tracked/ignored
   root with a whole-root symlink. Never make tracked code, tests, profiles, or
   documentation depend on a developer-local absolute path.
5. **Update the index together.** When the resource policy changes, update
   the manifest, policy documentation, ignore rules, and focused policy test in
   the same commit.
6. **Verify.** Re-run the audit, inspect `git status --ignored` for the focused
   paths, resolve symlink targets physically, and verify that no peer worktree
   gained untracked or writable content unexpectedly.

## Copy integrity

- Before transforming hardlinked or overwrite-prone data, verify source and
  target identity and keep a recoverable source copy.
- Write media transforms to a fresh inode, validate declared and decoded counts
  plus tail decode, then replace atomically.
- Stop at a decoder warning, count/hash mismatch, changed link target, or active
  competing writer.
- Never delete or clean a resource merely to make a worktree appear clean.

Report what is versioned, ignored, linked, copied, mutable, and externally
stored separately.
