# DexTeleop corrected OpenPI training configs v1

Historical OpenPI config names are frozen for existing checkpoints. Corrected
training must use the corresponding `-corrected-v1` name; selecting the
unsuffixed name is the explicit rollback.

| Frozen config | Corrected config |
| --- | --- |
| `dual_rm65_dmgripper_v22_masked` | `dual_rm65_dmgripper_v22_masked-corrected-v1` |
| `0715-cube-front-ready-mix-4to1` | `0715-cube-front-ready-mix-4to1-corrected-v1` |
| `0715-cube-front-ready-mix-4to1-relative` | `0715-cube-front-ready-mix-4to1-relative-corrected-v1` |
| `0715-cube-front-ready-mix-4to1-h16-absolute` | `0715-cube-front-ready-mix-4to1-h16-absolute-corrected-v1` |
| `0715-cube-front-ready-mix-4to1-h16-absolute-v23` | `0715-cube-front-ready-mix-4to1-h16-absolute-v23-corrected-v1` |
| `0715-cube-front-ready-mix-4to1-h16-relative` | `0715-cube-front-ready-mix-4to1-h16-relative-corrected-v1` |
| `0715-cube-front-ready-full-h16-relative` | `0715-cube-front-ready-full-h16-relative-corrected-v1` |
| `0715-cube-front-ready-full-h16-absolute-v23` | `0715-cube-front-ready-full-h16-absolute-v23-corrected-v1` |
| `0715-cube-front-ready-single-view-h16-relative` | `0715-cube-front-ready-single-view-h16-relative-corrected-v1` |
| `0715-cube-front-ready-single-view-h16-absolute-v23` | `0715-cube-front-ready-single-view-h16-absolute-v23-corrected-v1` |
| `writing-phase1-grap-pen-h16-relative` | `writing-phase1-grap-pen-h16-relative-corrected-v1` |
| `writing-phase1-grap-pen-h16-absolute` | `writing-phase1-grap-pen-h16-absolute-corrected-v1` |
| `0712-cube` | `0712-cube-corrected-v1` |
| `0712-wrist-cube` | `0712-wrist-cube-corrected-v1` |

The frozen configs explicitly retain 100% procedural, camera-independent
background augmentation and `legacy_parent_only` epoch behavior. Every
corrected config binds the following input contract:

- mask source: `final_keep_mask` only;
- background distribution: 10% original, 40% real image, 50% procedural;
- wrist pairing: shared source/canvas with camera-independent view transforms;
- loader: `num_workers=4`, `epoch_worker_mode=shared_worker_v1`;
- media admission: exact declared RGB frame count plus final-frame decode;
- manifest:
  `/home/user/zc/DexTeleop/assets/random_real_image/openimages_v7_validation/real_background_manifest_corrected_v1.json`;
- manifest SHA256:
  `f8ef91e130aa55c9bcb27732392e4ba52ccd5563339ac4daea844e428a7690b7`.

Media admission runs before parquet loading or the first training step.
Historical names retain lazy media reads for reproducibility.

Do not resume an unsuffixed checkpoint with a corrected config. Start a new
experiment/checkpoint directory so the config identity and artifact receipt
remain unambiguous.
