# Superseded OpenPI integration patches

These patches are retained only as historical review artifacts. Do not apply
them to the current remote OpenPI checkout: in particular,
`openpi_final_keep_mask_default_v1.patch` rewrites existing config semantics,
which is incompatible with checkpoints already trained under those names.

The landed implementation freezes every historical config and adds one
visibly versioned `-corrected-v1` config for each affected training profile.
See the remote OpenPI document
`docs/dexteleop_corrected_training_configs.md`.

`openpi_epoch_worker_sync_v1.patch` originally targeted the remote OpenPI
`src/openpi/training/data_loader.py`. It keeps the configured `num_workers` and
`persistent_workers=True`, while sharing the loader epoch with every worker
and refreshing the worker-local dataset/transform state before the first item
of a new epoch.

`openpi_epoch_worker_sync_v1_test.patch` adds the regression test for
`num_workers=1,2,4` and two consecutive epochs.

`openpi_final_keep_mask_default_v1.patch` makes both DexTeleop v2.2 and v2.3
readers final-only by default and removes every training-config override that
selected `wrist_depth_mask_paths`. Missing final keep-masks then fail closed.
An explicit depth-mask key remains available only for a named rollback
experiment.

Rollback now means selecting the original unsuffixed config. It does not mean
reapplying or reversing these patches.
