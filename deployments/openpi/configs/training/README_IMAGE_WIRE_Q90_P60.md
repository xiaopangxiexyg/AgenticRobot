# OpenPI JPEG q90 60/40 training variants

The `*_jpeg_q90_p60_v1.json` profiles are opt-in siblings of the existing
0804–0806 writing stage 1–4 profiles. They do not change the existing profile
files or their default training behavior.

For every prepared sample and epoch, the v2 image-wire training contract makes
one deterministic decision from the prepared organized-episode ordinal, row
index, epoch, and contract hash:

- 60% probability: apply the exact deployment path to every configured image
  in the sample — OpenPI 224×224 resize-with-pad, JPEG q90 4:2:0 encode, then
  server-equivalent decode.
- 40% probability: preserve the complete upstream sample unchanged. This keeps
  the existing raw/quality-augmentation distribution instead of introducing a
  second legacy-quality branch.

The decision is shared-worker safe and changes deterministically with epoch.
`sample-check` reports the configured probability and exact epoch-0 codec and
bypass counts under `openpi_image_wire_training_selection`.

The variants intentionally reuse the corresponding prepared dataset root and
state/action normalization asset because neither binding nor numeric
normalization changes. They use distinct OpenPI config names and experiment
names so their checkpoints cannot collide with the original profiles.

Rollback is profile selection: choose the original profile without the
`_jpeg_q90_p60_v1` suffix. Absence of `openpi_image_wire_training` keeps the
historical training path exactly as before. A checkpoint trained with one
image-wire contract is rejected when loaded with a profile declaring a
different contract.
