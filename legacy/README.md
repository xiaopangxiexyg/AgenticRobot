# Legacy archive

## `teleop_runtime/collection/replay_dataset_sapien.py`

The historical SAPIEN collection replay entrypoint is archived here because it
has no production or import callers and is outside the supported collection,
simulation, and release routes. Its implementation is retained for forensic
reproduction only; `legacy/` is not an active Python package and production code
must not import it.

Use `python -m utils.dataset_postprocess validate-v23 DATASET_ROOT` for the
current read-only dataset acceptance path. For new simulation replay, use the
backend-neutral or MuJoCo replay tools documented under `docs/sim/`.

Deletion condition: remove this archive only after the historical SAPIEN
experiment receipt and any external reproduction request have been explicitly
closed. Until then, this file is evidence preservation, not a supported
runtime entrypoint.
