# Skill projection authoring assets

This directory contains reviewed, editable mappings used by
`dataset_interface` to compile strict `SkillTrainingContractV1` and
`ProgressSkillProjectionV1` objects.

These files are not capture profiles and are not imported by
`collection_runtime`. A mapping is a read-time training assertion bound to the
exact semantic hashes of its source Progress and Stage profiles. Editing a
managed source profile changes those hashes and intentionally makes the old
mapping ineligible until it is reviewed and updated.

When a reviewed source profile is versioned, legacy hash-bound projections may
remain alongside the current mapping. They are separate immutable assertions;
the loader never aliases an old hash to a new profile interpretation.

The projection layer never rewrites source profiles, Progress facts, stage
annotations, Parquet, or video. It emits interval rows and deterministic
provenance receipts only. Unknown predicates, components, context, stages, or
profile hashes fail closed.
