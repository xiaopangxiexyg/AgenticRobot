# Collection progress runtime

`collection_runtime.progress` owns device-, control- and UI-neutral progress
runtime mechanics. It does not own operator widgets, robot Hold behavior,
sample production, dataset schemas, or training policy.

## Modules

```text
requests.py / reducer.py / controller.py
    Typed online progress requests, pure state reduction and effect requests.

store.py
    Append-only capture-time progress facts, restart recovery and seal.

profiles/
    Managed task progress/stage profile assets and dataset-level uniqueness
    binding. LINEAR is the current application gate; ROOTED_TREE remains a
    contract/runtime capability behind a separate application gate.

annotation_store.py
    Existing immutable refinement-boundary chunk store. It preserves the
    earlier boundary/salvage-anchor contract and remains available to its
    readers; it is not the current fine-stage GUI store.

stage_annotation_store.py
    Current ordered-stage annotation persistence: one shared legacy guard,
    cross-machine Episode lock directories, atomic editable drafts and
    immutable annotation-set seal. Distinct Episodes may be edited by upgraded
    programs on the same SSHFS/NFS-backed dataset. The Control Station
    refinement application coordinates this store through public APIs but the
    store has no GUI dependency.
```

The two annotation stores are intentionally not merged. They persist different
frozen contracts: sparse refinement boundaries versus ordered stage
occurrences/outcomes and abnormal intervals. A future migration may project
one read view into another, but must not reinterpret or silently rewrite
already sealed artifacts.

## Dependency boundary

This package may import `dataset_contracts`, `dataset_interface` and pure
domain contracts. It must not import `teleop_runtime`, `control_station`,
concrete hardware, simulation or GUI code.

Episode identity, sample time and final episode publication continue to come
from the application Episode owner. Progress is a subordinate participant and
cannot independently commit the parent Episode.
