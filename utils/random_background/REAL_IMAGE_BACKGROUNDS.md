# Real Image Background Pool

真实图片背景建议作为程序生成背景的补充，而不是替代。训练时应继续保持时间一致 crop/warp，避免每帧随机换图造成不真实闪烁。

## 推荐来源

优先级：

1. 自采实验室/桌面/地面/墙面/储物箱/布料/木纹/塑料纹理图片。
2. Places365 场景图，适合提供室内、走廊、桌面、工业环境等自然场景统计。
3. Open Images 子集，适合大规模、多样化真实图片，但需要过滤人物、文字、机器人、任务物体。
4. COCO 子集，只建议小比例混入；COCO 物体语义强，容易引入与任务无关的对象先验。

不建议直接使用未过滤的大规模 web scrape 图片库作为主背景源。背景增强需要“打断原始背景捷径”，不是给模型额外引入人物、文字、屏幕 UI 或目标物体语义。

## 下载策略

图片不要放进 git。建议放在：

```text
<large_storage>/dexteleop_backgrounds/
  places365/
  openimages/
  coco/
  lab_captured/
```

本仓库只保存索引脚本和训练代码。下载完成后生成 manifest：

```bash
python -m utils.random_background.index_real_backgrounds \
  /path/to/dexteleop_backgrounds/places365 \
  --manifest /path/to/dexteleop_backgrounds/places365_manifest.json \
  --source places365
```

训练中使用：

```python
from utils.random_background import (
    LocalImageBackgroundProvider,
    RandomBackgroundAugmentConfig,
    RandomBackgroundVideoAugmenter,
)

provider = LocalImageBackgroundProvider(
    manifest_path="/path/to/dexteleop_backgrounds/places365_manifest.json"
)

augmenter = RandomBackgroundVideoAugmenter(
    RandomBackgroundAugmentConfig(
        provider="mixed",
        real_image_prob=0.4,
        mixed_original_prob=0.1,
        mixed_black_prob=0.0,
    ),
    real_image_provider=provider,
)
```

## Open Images

Open Images V7 官方页面说明最新 V7 于 2022 年发布，并提供 dense annotations 子集、image-label 全量集合和全部 9,178,275 张图片入口。官方页面还提供了 `downloader.py` 和按 image id 下载子集的方法。

推荐流程：

1. 先下载 image IDs / labels / boxes metadata。
2. 过滤掉不适合做背景的类别，例如 person、hand、robot、screen、text-heavy scene、tool、目标物体类别。
3. 只下载筛选后的 image IDs。
4. 下载后用 `index_real_backgrounds.py` 生成 manifest。

官方页面给出的子集下载形式大致是：

```bash
wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
python downloader.py image_ids.txt --download_folder=/path/to/openimages --num_processes=8
```

实际运行前需要根据 Open Images 官方说明安装依赖，并确认当前许可和用途要求。

本仓库也提供一个小规模试跑脚本，不依赖 `boto3`，直接从官方公开 GCS validation 集抽样下载：

```bash
python -m utils.random_background.download_openimages_sample \
  --output-dir utils/random_background/real_images/openimages_v7_sample \
  --count 64 \
  --max-bytes 52428800
```

输出：

```text
utils/random_background/real_images/openimages_v7_sample/
  images/
  real_background_manifest.json
  download_report.json
```

这只适合小规模流程验证。正式大规模下载仍建议先基于 metadata 做类别过滤，再按筛选后的 image id 下载。

## Places365

Places365 是场景识别数据集，官方仓库说明 Places365-Standard train 约 1.8M 张、365 个场景类别；Places365-Challenge 约 8M 张。它比 COCO/Open Images 更适合作为背景库，因为图像以场景为主。

推荐优先选：

```text
office
workshop
garage
storage_room
utility_room
corridor
laboratory-like indoor scenes when available
floor
wall
table/desk related classes
```

避免：

```text
people-heavy scenes
restaurant crowds
screens/signage/text-heavy scenes
highly semantic object closeups
```

## COCO

COCO 官方站点提供下载入口。COCO 图片物体语义很强，不建议作为主背景库；可以小比例用于增加自然图像统计。需要优先过滤人、动物、手持工具、任务物体、强文字场景。

## 质量过滤

最低过滤规则：

- 分辨率短边 >= 240，优先 >= 480。
- 删除灰度、损坏、极端长宽比图片。
- 删除强文字/屏幕 UI/二维码/水印图片。
- 删除包含机器人、机械臂、手、任务物体的图片。
- 对真实背景做随机 crop、resize、轻微 blur/color jitter，不直接 center crop 固定区域。

训练配比起点：

```text
10% original background
40% real-image temporal backgrounds
50% procedural temporal backgrounds
0% black/neutral masked background
```

如果真实部署场景变化很大，提高 real-image 比例；如果模型开始依赖真实背景里的语义物体，降低 COCO/OpenImages 比例，提高 Places365/自采纹理比例。
