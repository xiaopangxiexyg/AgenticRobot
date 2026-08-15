# Random Background Augmentation

可复用增强、mask reader、photometric、occlusion 与 real-image provider 的 canonical
实现位于 `dataset_interface.random_background`。本目录只保留离线下载、索引、预览和样例生成
命令，以及旧 Python import path 的薄兼容导出；新库代码不要再依赖 `utils` 路径。

用于把 wrist/depth mask 之外的背景替换为随机像素纹理。默认生成的是“空间相关的乱序像素”，不是纯白噪声：包含低频光照、块级乱序、patch mosaic、轻微相机噪声和模糊。

原因：

- 具身智能策略不应该依赖固定房间、墙面、桌边或杂物背景。
- 纯逐像素噪声与真实相机分布差距太大，容易让模型学到异常高频伪影。
- 块级/patch 级乱序能打破语义背景，同时保留接近真实 RGB 的颜色、曝光和空间频率。

生成离线背景：

```bash
python -m utils.random_background.generate_assets \
  --output-dir utils/random_background/generated/color_geometry_v2 \
  --width 640 \
  --height 480 \
  --count 256 \
  --seed 20260709
```

默认还会写入 `real_background_manifest.json`，可以直接交给 `LocalImageBackgroundProvider`。静态池和在线
`procedural` 均包含任意大小、重叠、可部分出界的 `color_blocks`，以及由色块、圆/椭圆、三角形和线条组成的
`shape_collage`；它们是无语义几何扰动，不局限于模拟普通背景纹理。

默认训练使用 `provider="mixed"`：10% 保留原始背景、40% 使用真实图片时间流、50% 使用在线程序化背景，
黑背景为 0%。如果没有显式传入 provider，增强器会自动读取仓库默认图片池；图片池不可用时直接报错，
不会静默退化成 10% original / 90% procedural。可通过 `DEXTELEOP_REAL_IMAGE_ROOT` 或显式
`LocalImageBackgroundProvider(...)` 回退到另一套图片池。

真实图片池默认放在 `assets/random_real_image/openimages_v7_validation/images/`。真实图片流会对每个
episode/camera/epoch 以确定性 seed 采样独立的缩放、横向拉伸和纵向拉伸，再进行 crop、时间连续平移、旋转、缩放、
轻度色彩扰动和模糊。生成可查看样例：

```bash
python -m utils.random_background.generate_samples \
  --real-image-root assets/random_real_image/openimages_v7_validation/images \
  --output-dir utils/random_background/samples/color_geometry_real_v2
```

训练中启用新增真实图片池：

```python
from dataset_interface.random_background import (
    LocalImageBackgroundProvider,
    RandomBackgroundAugmentConfig,
    RandomBackgroundVideoAugmenter,
)

real_images = LocalImageBackgroundProvider(
    image_root="assets/random_real_image/openimages_v7_validation/images"
)
augmenter = RandomBackgroundVideoAugmenter(
    RandomBackgroundAugmentConfig(
        provider="mixed",
        mixed_original_prob=0.1,
        real_image_prob=0.4,
        mixed_black_prob=0.0,
    ),
    real_image_provider=real_images,
)
```

在线合成：

```python
from dataset_interface.random_background import apply_random_background

# keep_mask > 0 is foreground to preserve.
augmented_bgr = apply_random_background(frame_bgr, keep_mask, seed=episode_index * 100000 + frame_index)
```

视频序列建议使用时间一致背景，而不是每帧独立随机：

```python
from dataset_interface.random_background import temporal_background_from_episode

stream = temporal_background_from_episode(
    width=640,
    height=480,
    episode_index=episode_index,
    camera_key="right_wrist",
)

for frame_index, frame_bgr in enumerate(frames):
    augmented_bgr = stream.apply(frame_bgr, keep_mask, frame_index)
```

这个模式先生成一张大画布，再对每帧做平滑 crop、轻微旋转和缩放，模拟腕部相机移动时背景在图像中的连续滑动。它不会精确重建真实三维视差，但比“每帧独立噪声”和“背景贴在相机坐标系不动”都更接近实际数据。

默认时间运动也会做增广：

- 每个 episode/camera 随机方向。
- 程序流默认速度从 `0–6.0 px/frame` 采样；真实图片流默认从 `0–7.0 px/frame` 采样。
- 叠加轻微加速度、低频速度变化和亚像素抖动。
- 旋转和缩放速度也按 seed 随机化。

需要更强相机运动时可以直接构造 spec：

```python
from dataset_interface.random_background import TemporalBackgroundSpec, TemporalRandomBackground

stream = TemporalRandomBackground(
    TemporalBackgroundSpec(
        width=640,
        height=480,
        seed=42,
        min_drift_px_per_frame=0.3,
        max_drift_px_per_frame=6.0,
        acceleration_px_per_frame2=0.025,
        rotation_deg_per_frame=0.08,
    )
)
```

`patch_mosaic` 不是固定 tile size：每行、每块都会随机尺寸，并额外叠加多尺度 patch，避免形成可预测网格。

如果已经有一批背景 PNG，可以显式读取并传入：

```python
from dataset_interface.random_background import apply_random_background, load_background

bg = load_background("utils/random_background/generated/random_bg_0000_block_shuffle.png")
augmented_bgr = apply_random_background(frame_bgr, keep_mask, background=bg)
```

## 大规模训练接口

正式训练建议在线合成，不要把每个增强结果离线写成新视频。这样不会把数据集放大数倍，也能每个 epoch 改变背景。
对下游模型的通用接入建议优先使用 `dataset_interface`；本目录提供随机背景生成和合成后端。

推荐输入是 `utils.dataset_postprocess.wrist_cam_adjust compose-final-mask` 生成的 wrist final keep-mask：

```text
final_keep_mask = postprocessed wrist depth mask OR table geometry keep-mask
final 合成阶段不额外添加 proxy/body mask
```

也就是 `mask > 0` 的像素保留原始 RGB，`mask == 0` 的背景替换为随机背景。

```python
from dataset_interface.random_background import (
    MaskReaderCache,
    RandomBackgroundAugmentConfig,
    RandomBackgroundVideoAugmenter,
)

augmenter = RandomBackgroundVideoAugmenter(
    RandomBackgroundAugmentConfig(
        global_seed=20260709,
        provider="mixed",          # procedural | original | black | mixed
        mixed_original_prob=0.1,
        real_image_prob=0.4,
        mixed_black_prob=0.0,
        wrist_pair_mode="shared_scene_family",  # or "independent" for rollback
        mask_role="keep",          # keep: mask>0 保留; replace: mask>0 替换
        edge_blur_px=1,
        enable_epoch_variation=True,
    )
)
mask_cache = MaskReaderCache(max_size=32)

# DataLoader worker 内部:
mask = mask_cache.read(mask_manifest_path, frame_index)
augmented_bgr = augmenter.apply(
    rgb_bgr,
    mask,
    episode_index=episode_index,
    camera_key="right_wrist",
    frame_index=frame_index,
    epoch=current_epoch,
)
```

也可以直接用 manifest helper：

```python
from dataset_interface.random_background import augment_sample_from_paths

augmented_bgr = augment_sample_from_paths(
    rgb_bgr,
    mask_manifest_path=mask_manifest_path,
    episode_index=episode_index,
    camera_key="right_wrist",
    frame_index=frame_index,
    augmenter=augmenter,
    mask_cache=mask_cache,
    epoch=current_epoch,
)
```

效率要点：

- RGB 视频 reader 应在训练 Dataset/DataLoader worker 里缓存，不要每帧重新打开。
- bitpack mask 用 `MaskReaderCache` 缓存文件句柄。
- 背景流在 `RandomBackgroundVideoAugmenter` 内按 episode/camera/epoch LRU 缓存；默认左右腕共享
  `wrist_pair_scene` 的 provider/family 和 epoch，程序背景共享 coarse canvas，真实图片背景则使用
  独立 donor/crop/运动流；两者都使用独立 view seed，避免左右腕看到完全相同的画面。
- mask 的 dilation、hole filling、时序去闪烁应提前离线完成；训练时只读最终 mask。
- 多相机用不同 `camera_key`，左右腕默认共享 provider/family；传入
  `wrist_pair_mode="independent"` 可回退到旧的逐相机独立 seed 行为。
- `provider="mixed"` 默认严格使用 10% original / 40% real_image / 50% procedural；默认
  `mixed_black_prob=0`，避免把黑背景作为额外变量。real-image provider 缺失时直接失败，防止训练
  输入悄悄变成 10% / 90%。

## 全图非几何图像增强

`RandomBackgroundVideoAugmenter` 默认还会启用 `PhotometricAugmentConfig`。它在 wrist 背景替换或
front 固定边界替换**完成后**作用于完整 BGR 图像，随后才交给 layout/tensor 转换。因此它不会改变
mask 的几何、不会新增离线视频或 Parquet 字段，也不会把 mask 当作模型输入的一部分。

```python
from dataset_interface.random_background import PhotometricAugmentConfig, RandomBackgroundAugmentConfig

augmentation = RandomBackgroundAugmentConfig(
    provider="mixed",
    photometric=PhotometricAugmentConfig(
        enabled=True,  # training default
        enable_epoch_variation=None,  # inherit RandomBackgroundAugmentConfig
    ),
)
```

当前默认候选及上限如下；每个类别均按独立概率采样，未命中时保持恒等。相机质量类别一次最多选中一种，
不会把多种强退化叠加到同一图像。

| 类别 | 默认概率 | 采样范围/候选 |
| --- | ---: | --- |
| 曝光、对比度、Gamma | 0.55 / 0.45 / 0.30 | gain `0.55–1.55`、`0.65–1.40`；Gamma `0.65–1.55` |
| 白平衡 | 0.35 | warm BGR `(0.85, 1.00, 1.15)`；cool BGR `(1.15, 1.00, 0.50)` |
| 饱和度、色相 | 0.25 / 0.10 | saturation `0.70–1.00`；hue `±7°` |
| 非均匀照明 | 0.35 | vignette 最大 `0.38`、soft shadow 最大 `0.46`、低频光照幅度最大 `0.24` |
| 相机质量 | 0.45 | Gaussian noise σ≤`8`、Poisson peak `45–160`、JPEG Q `25–90`、defocus σ≤`1.8`、motion blur `3–11 px`、down/up-scale `0.45–0.80`，或固定 combo（σ=`4`、blur σ=`0.70`、JPEG Q=`35`） |

Photometric 子模块本身刻意不包含 crop、平移、旋转、透视、翻转、cutout、随机黑块或 random drop；
抓取任务的对象位置和图像尺寸保持不变。需要模拟部分或完整视角缺失时，使用下面独立的
view-occlusion 增强。

## View occlusion（高分辨率视角遮挡）

`ViewOcclusionConfig` 接受每个相机的 `occlusion_ratio`：`0` 保持进入该增强的图像不变，`1` 在像素域
返回严格全黑图像，中间值按目标面积生成黑色区域。遮挡在背景替换和 photometric 完成后执行，仍位于
模型 resize 和 normalization 之前。训练预处理不得额外使用 random crop、random resized crop、随机
平移、翻转或透视；固定 resize 或保持宽高比的 letterbox 应在遮挡后执行。

```python
from dataset_interface.random_background import RandomBackgroundAugmentConfig, ViewOcclusionConfig

augmentation = RandomBackgroundAugmentConfig(
    view_occlusion=ViewOcclusionConfig(enabled=True),
)

image = augmenter.apply(
    rgb,
    keep_mask,
    episode_index=episode_index,
    camera_key="left_wrist",
    frame_index=frame_index,
    occlusion_ratio=0.45,
)
```

对于 `0 < ratio < 1`，默认以 `40% / 40% / 20%` 选择 random rectangle、Gaussian blob 或 random
grid。Rectangle 使用 `1~5` 个候选块、`0.05~0.5` 候选面积比例和 log-uniform `0.1~10` 长宽比；
blob 叠加旋转 Gaussian 与低频场；grid 使用相对原始分辨率的 patch fraction，因此不会把 `16x16`
错误地硬编码到任意相机分辨率。为避免高分辨率训练的数据加载瓶颈，几何计划在最长边不超过 192 的
工作网格生成，再以 nearest-neighbor 映射到 source resolution；默认配置的实测面积误差低于 0.2%。
黑色像素仍然直接写入 source-resolution 图像。clip 默认复用同一 mask；需要模拟逐帧丢包时才设置
`temporal_scope="frame"`。

采样计划在每个 DataLoader worker 内以 `episode + camera + epoch` 缓存；照明增益图和 motion-blur
kernel 也只准备一次。`apply_batch(...)` 复用同一计划，不会额外打开视频或 mask 文件。Gaussian/Poisson
像素噪声按 `frame_index` 确定性生成，以保留真实传感器噪声的逐帧细节，但其类型和强度在该计划内固定。
将 `enable_epoch_variation=False` 设在父配置或 photometric 子配置上，可让一个 episode/camera 在整个
训练中保持同一计划。

`MaskedFrameSample.metadata["photometric_augmentation"]` 仅保存审计信息；`TorchMaskedAugmentDataset`
只写入增强图像，不会把该元数据加入训练 batch 或模型 observation。验证、replay、部署应使用未启用
训练增强的输入，例如 `PhotometricAugmentConfig(enabled=False)`。样例视频脚本也显式关闭该项，以保持其
原有的背景预览语义。

## Front 固定边界 mask

`FixedBorderMaskConfig` 是与 wrist final keep-mask 并列的在线 mask 类型，当前只作用于
`front`。它不读取 wrist 的 bitpack mask：训练接口会为 front 合成一个全保留的基础 mask，
然后将被选中的边界替换为随机背景。

```python
from dataset_interface.random_background import FixedBorderMaskConfig, RandomBackgroundAugmentConfig

augmentation = RandomBackgroundAugmentConfig(
    provider="mixed",
    fixed_mask=FixedBorderMaskConfig(
        enabled=True,
        side_probability=0.25,      # top/bottom/left/right independently
        thickness_fraction=0.05,    # 1/20: height for top/bottom; width for left/right
        provider="inherit",        # reuse the configured random-background provider
        enable_epoch_variation=None,  # inherit RandomBackgroundAugmentConfig
    ),
)
```

约定如下：

- 对同一 `episode + epoch`，边界选择和每条边的背景图都固定；所有 frame 和 clip 都一致。
- epoch 变化时会按 seed 确定性重采样。若要整次训练都固定，设
  `enable_epoch_variation=False`。
- 每条被选中的边各自使用独立的静态背景；背景取现有 provider 流的 `frame(0)`，所以不会
  随 frame 移动。角落重叠处按 `top, bottom, left, right` 的固定顺序归属一条边。
- `provider="inherit"` 会复用 `procedural`、`real_image` 或 `mixed`。对于 `mixed`，固定
  边界只在 procedural/real-image 随机背景之间选择，不会静默退化成 original 或 black。
  若主 provider 是 `original` 或 `black`，需要显式把 fixed-mask provider 设为
  `procedural` 或 `real_image`。
- 该类型只改变训练期图像；`fixed_mask` 选择和背景来源仅写入 `MaskedFrameSample.metadata`，
  不会加入模型 observation 或 batch 字段。

## 样例视频检查

生成一条轨迹的左右腕 random background 预览：

```bash
SESSION=datasets/2026-07-09_vive_rm65_real_teleop
FINAL_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask

python -m utils.random_background.generate_wrist_sample_video \
  --session "$SESSION" \
  --final-mask-root "$FINAL_MASK_ROOT" \
  --episode 0 \
  --max-frames 160 \
  --output-dir utils/dataset_postprocess/wrist_cam_adjust/outputs/${SESSION##*/}_random_background_sample \
  --provider procedural \
  --seed 20260710 \
  --output-tag proc_a \
  --side-by-side \
  --overwrite
```

真实图片背景：

```bash
python -m utils.random_background.generate_wrist_sample_video \
  --session "$SESSION" \
  --final-mask-root "$FINAL_MASK_ROOT" \
  --episode 0 \
  --max-frames 160 \
  --output-dir utils/dataset_postprocess/wrist_cam_adjust/outputs/${SESSION##*/}_random_background_sample \
  --provider real_image \
  --real-image-manifest utils/random_background/real_images/openimages_v7_sample/real_background_manifest.json \
  --output-tag real_img \
  --side-by-side \
  --overwrite
```

预览脚本默认读取 canonical `final_keep_mask` 根目录；没有 final mask 时直接失败。仅用于诊断
component 合成时，可以显式传 `--use-component-masks --table-mask-root ...`
和 postprocessed `--depth-mask-root ...`，脚本才会临时执行
`postprocessed_depth_keep_mask OR table_geometry_keep_mask`。正式训练不使用这个回退。

真实图片背景池的下载、索引和过滤建议见：

```text
utils/random_background/REAL_IMAGE_BACKGROUNDS.md
```

## 是否需要海量真实图片

默认配置已经纳入真实图片，但不完全替代程序生成背景。

- 程序生成背景适合快速、无版权依赖、可复现地去除固定背景捷径。
- 大规模真实图片更接近自然图像统计，适合提升 sim/real 和跨场景泛化。
- 真实图片应做 crop、resize、color jitter、blur、随机仿射，并避免包含机器人、手、文字标签、任务物体或会泄漏语义的场景。
- 对腕部相机任务，优先混合桌面、地面、墙面、实验室、工业台面、布料、木纹、塑料箱等纹理类图片；少用人物、动物、强文字、屏幕 UI。

推荐训练配比起点：

```text
10% original background
40% real texture/photo backgrounds
50% generated block/patch shuffled backgrounds
```

如果验证集在真实不同场地表现仍差，再提高真实图片背景占比；如果前景边界出现伪影依赖，则增大 mask dilation/edge blur。
