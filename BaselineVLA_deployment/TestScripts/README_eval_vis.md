# Evaluation And Visualization

这两个脚本面向训练后检查：

- [`evaluate_checkpoint.py`](/home/zc/HandMotion/ManipRep/TestScripts/evaluate_checkpoint.py)
  - 批量评估一个 checkpoint / run 目录 / run 名
  - 保存 split 指标、样本 dump、以及可视化结果
- [`evaluate_gtmed_pipeline.py`](/home/zc/HandMotion/ManipRep/TestScripts/evaluate_gtmed_pipeline.py)
  - 在同一份数据和同一套 GTMed 指标口径下，同时评估：
    - GTMed 原生测试：`ground-truth med -> GTMed -> action`
    - 组合测试：`observation -> GenMed -> predicted med -> GTMed -> action`
  - 组合测试支持两种模式：
    - `inline`
      - 单次前向里直接串联 `GenMed -> GTMed`
    - `cached`
      - 先把 `GenMed` 预测的 `med` 落盘到 `Data/MedData/...`，再由 `GTMed` 按样本加载
  - 适合直接比较 “真实 med” 和 “预测 med” 对 GTMed 动作输出的影响
- [`visualize_genmed_predictions.py`](/home/zc/HandMotion/ManipRep/TestScripts/visualize_genmed_predictions.py)
  - 轻量查看单个 batch 里的一个样本
  - 适合快速 spot check `genmed` 的预测质量

## 支持的输入

两者都支持以下三种输入方式：

- 直接给 checkpoint 文件
- 直接给 run 目录
- 直接给 `Outputs/` 下的 run 名

例如下面三种都可以：

```bash
python TestScripts/evaluate_checkpoint.py Outputs/genmed/rawhand_point_genmed/best.pt
python TestScripts/evaluate_checkpoint.py Outputs/genmed/rawhand_point_genmed
python TestScripts/evaluate_checkpoint.py rawhand_point_genmed
```

默认 `--checkpoint-kind auto`，会优先找 `best.pt`，找不到再退到 `last.pt`。

## 常用命令

评估一个点云 `genmed` run，并输出指标与可视化：

```bash
python TestScripts/evaluate_checkpoint.py rawhand_point_genmed \
  --split val \
  --loader-mode safe \
  --device cpu \
  --max-batches 4 \
  --save-samples 1 \
  --visualize all
```

只做评估，不生成可视化：

```bash
python TestScripts/evaluate_checkpoint.py rawhand_point_genmed \
  --split val \
  --visualize none
```

快速看一个单样本：

```bash
python TestScripts/visualize_genmed_predictions.py rawhand_point_genmed \
  --split val \
  --loader-mode safe \
  --device cpu \
  --visualize all
```

切换到其他数据 preset：

```bash
python TestScripts/evaluate_checkpoint.py rawhand_point_genmed \
  --data-config desktasktac_rawhand_point_real \
  --set data.raw.loader.batch_size=1
```

对同一组 `GenMed + GTMed` checkpoint 做组合评估：

```bash
python TestScripts/evaluate_gtmed_pipeline.py \
  rawhand_explainable_point_cloud_rawhand_point_gtmed_mlp_gtmed_right_point_cloud_seed7 \
  --genmed-checkpoint rawhand_explainable_point_cloud_rawhand_point_genmed_mlp_genmed_right_point_cloud_seed7 \
  --split val \
  --loader-mode safe \
  --max-batches 4 \
  --visualize all
```

默认 `--pipeline-mode inline`，会同时输出两组结果：

- `gtmed_ground_truth_med`
- `gtmed_with_genmed_predicted_med`

并在 `summary.json` 里额外保存 action 指标差值，便于直接比较。

如果显存或内存紧张，先试：

```bash
python TestScripts/evaluate_gtmed_pipeline.py \
  <gtmed_checkpoint> \
  --genmed-checkpoint <genmed_checkpoint> \
  --pipeline-mode inline \
  --loader-mode safe \
  --max-batches 1 \
  --device cuda \
  --genmed-device cpu \
  --visualize none
```

这会把 `GenMed` stage 放在 CPU，只把 `GTMed` stage 放在 CUDA，通常能显著降低组合评估时的峰值显存。

如果还希望把两阶段彻底拆开，改成 cached 模式：

```bash
python TestScripts/evaluate_gtmed_pipeline.py \
  <gtmed_checkpoint> \
  --genmed-checkpoint <genmed_checkpoint> \
  --pipeline-mode cached \
  --loader-mode safe \
  --visualize none
```

cached 模式会先在 `Data/MedData/GTMedPipelinePredictions/...` 下保存逐样本 `predicted med`，再用一条瘦身后的 data pipeline 只加载 observation + action target，并从磁盘读取 `med` 给 `GTMed`。

常用附加参数：

- `--predicted-med-cache-dir <path>`
  - 指定 cached 模式的落盘目录
- `--reuse-predicted-med-cache`
  - 已有 `summary.json` 时直接复用缓存，不重新跑 `GenMed`

当前评测脚本还做了几层默认的内存收敛：

- 只构建请求的 split，不再为单独的 `val` 评估额外构建 `train` dataset/loader
- 默认把 `data.raw.video.emit_frames=false` 写回评测配置，避免把原始 `video_frames` 张量塞进 batch；点云 projection 仍会按需从源视频重新读取帧
- `GTMed / GenMed / FullTrain` 的 input adapter 只搬运 `intermediates["med"]`，不再整包搬运全部 intermediates
- cached 模式内部会把 `data.raw.sequence.med.emit_intermediate=false`，让第二阶段不再构造 raw future-med 张量
- 评测时 `train` split 会固定成不 shuffle 的 loader，避免 `max-batches` 下 baseline / pipeline 取到不同子集

## 点云投影可视化

当 `med_spec.kind=point_cloud` 且 `--visualize` 包含 `projection` 时，会额外生成：

- `*_projection.png`
  - 原视频帧、GT 投影、Pred 投影的对比图
- `*_projection.mp4`
  - future steps 的逐帧对比视频

相关参数：

- `--projection-point-radius`
- `--projection-frame-height`
- `--projection-max-steps`
- `--camera-serial`
- `--camera-intrinsics-json`

如果数据集里已有 `meta/camera_intrinsics*.json`，通常不需要手动指定内参。

## 输出内容

`evaluate_checkpoint.py` 默认会在 checkpoint 同级目录下创建 `evaluation/...`，里面通常包括：

- `summary.json`
- `resolved_eval_config.yaml`
- `val/summary.json`
- `val/samples/sample_000.pt`
- `val/sample_000_med.png`
- `val/sample_000_med_projection.png`
- `val/sample_000_med_projection.mp4`

`evaluate_gtmed_pipeline.py` 默认会在 GTMed checkpoint 同级目录下创建 `evaluation/...__from_<genmed_run>/`，里面通常包括：

- `resolved_gtmed_eval_config.yaml`
- `resolved_genmed_eval_config.yaml`
- `resolved_cached_pipeline_eval_config.yaml`
  - 仅 cached 模式存在
- `summary.json`
- `gtmed_ground_truth_med/val/...`
- `gtmed_with_genmed_predicted_med/val/...`

cached 模式还会额外在 `Data/MedData/GTMedPipelinePredictions/...` 下生成：

- `summary.json`
- `<split>/<dataset_id>/episode_xxxxxx/frame_xxxxxx.pt`

`visualize_genmed_predictions.py` 默认输出到 checkpoint 同级目录下的 `visualizations/`，通常包括：

- `med_prediction_sample.pt`
- `med_vector.png` 或 `med_point_cloud.png`
- `med_prediction_sample_projection.png`
- `med_prediction_sample_projection.mp4`

## 依赖

建议在 `ContactRep` 环境里运行。常见依赖：

- `pyarrow`
- `opencv-python`
- `matplotlib`
- `ffmpeg`

其中：

- 没有 `matplotlib` 时，脚本仍可保存 `.pt`，但不会生成静态图
- 没有点云投影所需视频/内参时，会跳过 projection，并在终端输出 note
