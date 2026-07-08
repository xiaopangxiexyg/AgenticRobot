# Wrist Camera Background Mask Adjustment

用于后处理腕部相机视频：保留腕部本体和 URDF 桌面，其余像素视为背景。

`keep_mask = painted wrist/body mask OR projected table mask`

输出可以是 masked RGB、overlay、background-mask 或 keep-mask。

## 快速使用

快速生成某个 episode 的 RGB + depth mask 可视化：

```bash
dataset_postprocess/wrist_cam_adjust/scripts/visualize_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

输出写到：

```text
dataset_postprocess/wrist_cam_adjust/outputs/<dataset>/<episode>/
```

打开同步播放器：

```bash
OPEN_GUI=1 dataset_postprocess/wrist_cam_adjust/scripts/visualize_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

对已保存的 depth mask 做去空洞和平滑：

```bash
dataset_postprocess/wrist_cam_adjust/scripts/postprocess_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

输出写到：

```text
dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess/<dataset>/<episode>/
```

常用调参：

```bash
CLOSE_RADIUS_PX=4 \
FILL_HOLE_MAX_AREA_PX=2500 \
MEDIAN_KSIZE=5 \
TEMPORAL_GAP_FILL_FRAMES=2 \
dataset_postprocess/wrist_cam_adjust/scripts/postprocess_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

可视化已保存的 wrist depth mask：

```bash
python -m dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --session datasets/2026-07-04_vive_rm65_real_teleop \
  --episode 0 \
  --output dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4 \
  --overwrite
```

这会从 `meta/episodes.jsonl` 自动找到 `left_wrist` / `right_wrist` 的
`wrist_depth_mask_paths`，横向拼成一个黑白预览视频。也可以直接输入 mask manifest
或 bit-packed stream：

```bash
python -m dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --left-mask /path/to/left/manifest.json \
  --right-mask /path/to/right/manifest.json \
  --output dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4 \
  --overwrite
```

如果直接传没有 manifest 的 `.bin`，需要补充分辨率：

```bash
python -m dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --mask /path/to/mask.bitpack.bin \
  --shape 480x640 \
  --output dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4 \
  --overwrite
```

批量处理一个数据集：

```bash
python -m dataset_postprocess.wrist_cam_adjust dataset \
  --dataset-root datasets/2026-07-04_vive_rm65_real_teleop \
  --output-root datasets/2026-07-04_vive_rm65_real_teleop_wrist_masked \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2.urdf \
  --left-body-mask utils/wrist_cam_fix_mask/painting/left_dmtac_m_mask_01.png \
  --right-body-mask utils/wrist_cam_fix_mask/painting/right_dmtac_m_mask_01.png \
  --output-kind masked \
  --jobs 2 \
  --overwrite
```

快速检查单个 episode/camera：

```bash
python -m dataset_postprocess.wrist_cam_adjust dataset \
  --dataset-root datasets/2026-07-04_vive_rm65_real_teleop \
  --output-root dataset_postprocess/wrist_cam_adjust/outputs/wrist_mask_debug \
  --urdf assets/rm65_inspire_gripper_v2/rm65_dual_arm_wrist_cameras_v2.urdf \
  --camera right_wrist \
  --episode 0 \
  --max-frames 80 \
  --output-kind overlay \
  --right-body-mask utils/wrist_cam_fix_mask/painting/right_dmtac_m_mask_01.png \
  --overwrite
```

输出保持原数据集相对视频路径；manifest 写到：

```text
<output-root>/meta/wrist_cam_adjust.json
```

## 输入约定

- `--urdf` 需要包含 `table`、左右腕部相机 frame、左右机械臂链路。
- 相机内参默认读取 `<dataset-root>/meta/devices/realsense.json`，也可用 `--devices-json` 指定。
- painted wrist/body mask 是静态腕部本体 mask，用 `--left-body-mask` / `--right-body-mask` 传入。
- 默认 pose 来源优先使用 `observation.arm.{side}.actual_tcp_pose`。
- `--pose-frame auto` 会读取 dataset metadata；当前真机数据通常是 `tool_tcp`，代码会用 tool profile 转到机械臂末端后再套 URDF 相机外参。

如果数据集里某列已经是真正的相机位姿：

```bash
--pose-column 'observation.wrist_camera.{side}.pose_base' --pose-frame camera
```

如果要强制从 qpos 做 FK：

```bash
--pose-source qpos --qpos-column 'observation.arm.{side}.actual_qpos'
```

## 桌面 Mask

默认桌面计算方式：

```bash
--table-mask-mode sampled-raycast --table-raycast-stride 2
```

这是稳健的快速路径：在降采样像素网格上做 ray/table 求交，再上采样。常用调参：

```bash
--table-boundary-extend-px 16   # 图像上扩张桌面边界
--table-margin-m 0.02           # URDF 桌面长宽外扩
--table-height-offset-m 0.02    # 桌面平面沿 URDF/table z 轴上移
```

需要精确对照时：

```bash
--table-mask-mode raycast
```

`--table-mask-mode polygon` 是实验性最快路径，只适合桌面完全在相机近裁剪面前方的视角。

## 机械臂与其它前景

本模块不再通过 URDF 粗几何保留机械臂或手臂区域。当前只负责静态腕部本体和桌面；机械臂、手臂或其它非背景前景建议放到 depth 后处理流程里解决。

## Depth Mask

采集端的 depth mask 是 D405 对齐到 RGB 后的 z16 深度区间 mask：

```text
foreground = valid_depth && depth_min_m <= depth_m <= depth_max_m
```

默认有效距离在 `dataset_postprocess/wrist_cam_adjust/depth_mask/core.py` 中定义：

```text
depth_min_m = 0.05
depth_max_m = 0.5
```

正式采集入口也可以通过 `--wrist-depth-min-m` / `--wrist-depth-max-m` 覆盖。已采集的 `datasets/2026-07-05_vive_rm65_real_teleop` metadata 仍记录旧范围 `0.095m` 到 `0.4m`，并且只保存了 `wrist_depth_mask`，没有保存 `wrist_depth_raw`；因此当前后处理从 bit-packed 0/1 mask 开始，而不是从 raw metric depth 开始。新默认值只会影响之后的采集或重新生成。

主体代码放在：

```text
dataset_postprocess/wrist_cam_adjust/depth_mask/
```

其中：

- `core.py`: depth range mask、bitpack/unpack、manifest schema。
- `visualize.py`: mask stream 读取和预览视频生成。
- `postprocess.py`: 去空洞、形态学平滑、去小连通域、可选 Gaussian 平滑、视频时序短断裂补洞。

后处理默认参数偏向补洞和时序稳定：

```text
close_radius_px = 5
fill_hole_max_area_px = 4096
median_ksize = 5
temporal_gap_fill_frames = 2
```

`temporal_gap_fill_frames=2` 的含义是：同一个像素如果前后都是前景，中间最多连续 2 帧突然掉成背景，就把这段短空洞补回，减少视频里跳跃的空洞。若出现明显拖影，再降到 `1` 或 `0`。

后处理结果 JSON 中会记录：

- `spatial_foreground_fraction_mean`: 只做空间去洞/平滑后的前景比例。
- `output_foreground_fraction_mean`: 加上时间连续性后的最终前景比例。
- `temporal_changed_fraction_mean`: 时间连续性相对于空间结果额外改动的比例。
- `input_frame_change_fraction_mean` / `output_frame_change_fraction_mean`: 相邻帧 mask 跳变率，用来判断闪烁是否降低。

## 输出模式

```bash
--output-kind masked          # 正式背景置黑，最快
--output-kind overlay         # 调试，背景红色叠加
--output-kind background-mask # 输出背景二值 mask
--output-kind keep-mask       # 输出保留区域二值 mask
```

软边界：

```bash
--mask-blur-px 5 --mask-blur-sigma 0
```

## 速度与调试

显示进度默认开启；脚本中可关闭：

```bash
--no-progress
```

记录分阶段耗时：

```bash
--debug-timing
```

JSON/manifest 中会出现 `timing_summary.phases`，包括：

- `read_frame`
- `build_mask`
- `render_output`
- `write_frame`
- `collect_stats`

并行处理多个独立视频：

```bash
--jobs 4
```

建议从 `--jobs 2` 或 `--jobs 4` 开始。若 `read_frame` / `write_frame` 占比高，瓶颈在磁盘或编码；若 `build_mask` 占比高，优先调大 `--table-raycast-stride`。

## 常用组合

只看桌面和腕部本体：

```bash
--output-kind overlay \
--table-boundary-extend-px 16
```

正式批处理：

```bash
--output-kind masked \
--table-mask-mode sampled-raycast \
--table-raycast-stride 2 \
--jobs 4
```
