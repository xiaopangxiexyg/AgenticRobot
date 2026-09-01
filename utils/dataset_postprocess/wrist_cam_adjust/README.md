# Wrist Camera Background Mask Adjustment

用于后处理腕部相机视频。当前目录里有两条容易混淆但来源不同的
mask 流程：

- 几何投影 mask：由 URDF 桌面几何、相机位姿和相机内参生成。
- 深度范围 mask：由 D405 对齐到 RGB 的 z16 depth 是否落在距离区间内生成。
- final keep mask：默认把后处理后的深度 mask 和桌面几何 keep-mask 合并，
  作为训练和随机背景替换的最终输入。

## Mask 类型

### 几何投影 Mask

CLI：`python -m utils.dataset_postprocess.wrist_cam_adjust dataset ...` 或
`python -m utils.dataset_postprocess.wrist_cam_adjust video ...`

语义：

```text
table_geometry_projection_mask = URDF finite table top projected into RGB
static_wrist_body_mask = optional painted wrist/body mask
geometry_keep_mask = table_geometry_projection_mask OR static_wrist_body_mask
geometry_background_mask = inverse(geometry_keep_mask)
```

输出可以是 masked RGB、overlay、background-mask 或 keep-mask。这里的
`keep-mask` 指 `geometry_keep_mask`，不读取 `wrist_depth/mask`。

主体代码放在：

```text
utils/dataset_postprocess/wrist_cam_adjust/geometry_mask/
```

其中：

- `mask.py`: URDF table 几何投影、静态 body mask 合并、输出渲染。
- `video.py`: 按帧读取位姿/qpos，构建 `geometry_keep_mask` 并写视频。
- `dataset.py`: 数据集级批处理和 `meta/wrist_cam_adjust.json` manifest。

几何投影相关代码只保留在 `geometry_mask/` 下；请直接 import
`utils.dataset_postprocess.wrist_cam_adjust.geometry_mask.*`。

### 深度范围 Mask

CLI：`visualize-depth-mask`、`postprocess-depth-mask`，以及对应脚本
`visualize_wrist_depth_dataset.sh` / `postprocess_wrist_depth_dataset.sh`

语义：

```text
depth_range_mask = valid_depth && depth_min_m <= depth_m <= depth_max_m
depth_range_mask_postprocessed = depth_range_mask with spatial/temporal repair
```

这条流程读取数据集里的 `wrist_depth/mask/.../manifest.json` 和
`stream.bitpack.bin`，不使用 URDF table geometry。

### Final Keep Mask

CLI：`python -m utils.dataset_postprocess.wrist_cam_adjust compose-final-mask ...`

默认语义：

```text
final_keep_mask = depth_range_mask_postprocessed OR table_geometry_keep_mask
final 合成阶段不额外添加 proxy/body mask
```

`value_1` 表示保留原 RGB 的前景/桌面像素；`value_0` 表示可替换背景。
输出仍使用 bit-packed 0/1 mask manifest，便于训练和 random background 代码直接读取。
默认 table keep-mask 应只包含桌面；如果上游 table keep-mask 已经渲入 painted body mask，
final keep-mask 会按输入保留这部分像素。
默认输出位置：

```text
utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask/<dataset>/episode_000000/<camera>/manifest.json
utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask/<dataset>/episode_000000/<camera>/final_keep.bitpack.bin
```

## 快速使用

快速生成某个 episode 的 RGB + depth mask 可视化：

```bash
utils/dataset_postprocess/wrist_cam_adjust/scripts/visualize_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

输出写到：

```text
utils/dataset_postprocess/wrist_cam_adjust/outputs/<dataset>/<episode>/
```

打开同步播放器：

```bash
OPEN_GUI=1 utils/dataset_postprocess/wrist_cam_adjust/scripts/visualize_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

对已保存的 depth mask 做去空洞和平滑：

```bash
utils/dataset_postprocess/wrist_cam_adjust/scripts/postprocess_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

输出写到：

```text
utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess/<dataset>/<episode>/
```

批量处理整个 session 的 depth mask：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust postprocess-depth-mask \
  --session datasets/2026-07-09_vive_rm65_real_teleop \
  --all-episodes \
  --output-root utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess \
  --jobs 4 \
  --no-preview \
  --overwrite
```

不加 `--all-episodes` 时默认只处理 `meta/episodes.jsonl` 里的第一个 episode；
调试单条轨迹时可加 `--episode 0 --max-frames 160`。

常用调参：

```bash
CLOSE_RADIUS_PX=4 \
FILL_HOLE_MAX_AREA_PX=2500 \
MEDIAN_KSIZE=5 \
TEMPORAL_GAP_FILL_FRAMES=2 \
utils/dataset_postprocess/wrist_cam_adjust/scripts/postprocess_wrist_depth_dataset.sh \
  datasets/2026-07-05_vive_rm65_real_teleop \
  0
```

可视化已保存的 wrist depth mask：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --session datasets/2026-07-04_vive_rm65_real_teleop \
  --episode 0 \
  --output utils/dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4 \
  --overwrite
```

这会从 `meta/episodes.jsonl` 自动找到 `left_wrist` / `right_wrist` 的
`wrist_depth_mask_paths`，横向拼成一个黑白预览视频。也可以直接输入 mask manifest
或 bit-packed stream：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --left-mask /path/to/left/manifest.json \
  --right-mask /path/to/right/manifest.json \
  --output utils/dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4 \
  --overwrite
```

如果直接传没有 manifest 的 `.bin`，需要补充分辨率：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust visualize-depth-mask \
  --mask /path/to/mask.bitpack.bin \
  --shape 480x640 \
  --output utils/dataset_postprocess/wrist_cam_adjust/outputs/wrist_depth_mask_preview.mp4 \
  --overwrite
```

用几何投影 mask 批量处理一个数据集：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust dataset \
  --dataset-root datasets/2026-07-04_vive_rm65_real_teleop \
  --output-root datasets/2026-07-04_vive_rm65_real_teleop_wrist_masked \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --left-body-mask utils/wrist_cam_fixed_mask/painting/left_dmtac_m_mask_01.png \
  --right-body-mask utils/wrist_cam_fixed_mask/painting/right_dmtac_m_mask_01.png \
  --output-kind masked \
  --jobs 2 \
  --overwrite
```

用几何投影 mask 快速检查单个 episode/camera：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust dataset \
  --dataset-root datasets/2026-07-04_vive_rm65_real_teleop \
  --output-root utils/dataset_postprocess/wrist_cam_adjust/outputs/wrist_mask_debug \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --camera right_wrist \
  --episode 0 \
  --max-frames 80 \
  --output-kind overlay \
  --right-body-mask utils/wrist_cam_fixed_mask/painting/right_dmtac_m_mask_01.png \
  --overwrite
```

输出保持原数据集相对视频路径；manifest 写到：

```text
<output-root>/meta/wrist_cam_adjust.json
```

如果只需要桌面 keep-mask 供 final mask 合成，使用：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust dataset \
  --dataset-root datasets/2026-07-09_vive_rm65_real_teleop \
  --output-root utils/dataset_postprocess/wrist_cam_adjust/outputs/2026-07-09_vive_rm65_real_teleop_table_keep_mask \
  --urdf assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --output-kind keep-mask \
  --table-mask-mode sampled-raycast \
  --table-raycast-stride 2 \
  --jobs 4 \
  --overwrite
```

当前默认 final mask 不使用 proxy，也不额外合并 painted body mask。需要只保留桌面时不要传
`--left-body-mask` / `--right-body-mask`。

## Stage1 多 Source 正式 Final Mask

Stage1 all-valid profile 必须使用 Prepared binding 中的有效 episode，而不是按目录名
猜测数据范围。正式流水线为：

```text
raw wrist_depth_mask_paths (只读)
  -> 默认空间/时序 depth 后处理
  -> OR table sampled-raycast geometry keep-mask
  -> source_dataset_id 分区的 final_keep.bitpack.bin
```

训练只读取最后一层；`wrist_depth_mask_paths` 只是生成输入，不能作为训练时的静默
fallback。同名数据集（例如来自不同根目录但 basename 相同的分片）按
`source_dataset_id` 分区，输出布局为：

prepared wrist 训练 profile 还必须显式固定 corrected 背景契约：10% 保留原图、40%
使用固定 real-image 资产、其余 50% 使用 procedural 背景，并以
`shared_scene_family` 绑定同一条样本中的 wrist 背景族。这个替换只作用于
profile 选择的 `left_wrist`/`right_wrist`；`front` 不读取 wrist mask。real-image root/manifest 必须来自本次
训练的独立资产快照，不能继续依赖可变的开发 checkout。
正式 profile 同时固定 `openpi_checkout` 与 `openpi_source_revision`；启动器要求 checkout
中的 `OPENPI_SOURCE_IDENTITY.json` 与 revision 一致，并拒绝进程中已从其他路径导入的
`openpi` 包。

```text
<final-mask-root>/<source_dataset_id>/<dataset-name>/episode_XXXXXX/{left_wrist,right_wrist}/
```

其中 `<dataset-name>` 必须逐字来自 source 的 `meta/info.json.dataset_name`，不是
source 路径的 basename。流水线会把该值限制为单个安全路径分量；新生成的
postprocessed depth 与 final keep-mask 都直接写入这个 canonical namespace。

如果已有、带 source receipt 的旧产物曾按 source basename 生成，必须显式运行一次：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline \
  --profile <profile.json> \
  --binding-root <binding-root> \
  --urdf <pinned-urdf> \
  --migrate-dataset-name-layout \
  --jobs 4
```

迁移先按原 namespace 完整验证 episode/frame/bitpack 契约，再在同一个
`source_dataset_id` 分区和同一文件系统内原子重命名 final 目录；目录 device/inode、
episode/manifest 计数、profile/binding/source receipt SHA、未修改 raw/mask/manifest bytes
以及反向 rename 的回滚前提都会写入
`layout_migrations/<source_dataset_id>.json`。为了保留旧 final manifest 中精确的
depth 来源路径，旧 postprocessed 中间层留在 generation namespace；训练只读取迁移后的
canonical final namespace。目标路径碰撞、无迁移 receipt 的半迁移状态、symlink、跨设备
或计数不一致都会 fail closed。迁移完成后必须另跑一次 `--verify-only`；迁移 flag 本身不能
与 `--verify-only` 同时使用。

先运行 `prepared_skill_launch prepare` 并记录其 `openpi_binding_root`，再执行：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline \
  --profile deployments/openpi/configs/training/writing_stage1_all_valid_through_2026_08_21_front_right_wrist_h32_absolute_non_rtc_v1.json \
  --binding-root /absolute/path/from/openpi_binding_root \
  --urdf /absolute/pinned/assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf \
  --jobs 4
```

物理整合集合不假定单一相机内参。流水线同时读取 `meta/build_audit.json` 的
episode plan 与 `meta/provenance.jsonl`，要求两者对每个 output episode 的
source root/index 精确一致，再逐 episode 使用
`meta/source_devices/<source-name>/realsense.json`。受保护副本必须是整合集合内的
普通文件并把 SHA-256 写入新 receipt；原 source 仍存在时必须逐字节哈希相等。原 source
已按整合流程退役时，receipt 必须显式记录
`original_absent_protected_copy_bound_by_audit_and_provenance`，不能把“找不到原文件”
静默当作已验证相等。

如果早期 source 只保存 serial、没有保存数值内参，单右腕旧 profile 仍可使用
`right_wrist_intrinsics_overrides`；选择左右双腕的新 profile 必须使用
`wrist_intrinsics_overrides`，为每个训练 source 和每个选中的 wrist camera 显式声明
target camera、expected serial、donor source、donor source 内的相对 devices 路径和 donor
camera。donor devices 必须是其固定 dataset root 内的普通文件，不能用绝对路径或越界
路径。流水线逐 target manifest 同时验证 target 与 donor serial，再把 donor 数值内参写入
全新的 `derived/calibration_overlays/<source_dataset_id>/<target-hash>/realsense.json`。
物理整合集合会按 build audit/provenance 的逐 episode devices 绑定分别生成 overlay；receipt
记录 target/donor/overlay 路径与 SHA-256、episode 绑定和两个 wrist camera 的 serial。
不能按 donor 的 `left_wrist`/`right_wrist` 名字猜测，也不能回写 source metadata。

depth 后处理按 manifest 声明帧数精确读取。完整的额外尾随帧会记录在 receipt 中，
但不会进入派生结果；短流、半帧尾随、table 视频提前结束或额外可解码尾帧都会停止
流水线。每个 bitpack episode 先写独立临时目录并原子发布，原始数据始终只读。
receipt 位于：

```text
<workspace>/receipts/prepared_wrist_final_masks/<profile_id>/<source_dataset_id>.json
<workspace>/receipts/prepared_wrist_final_masks/<profile_id>/summary.json
```

当 final root 位于 CIFS/NAS 时，可以把需要被 FFmpeg 复读的 table-geometry 与
postprocessed 中间层固定到本机文件系统，final 与 receipt 仍写入全新的 NAS workspace：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline \
  --profile <profile.json> \
  --binding-root <binding-root> \
  --urdf <pinned-urdf> \
  --workspace-root <fresh-final-and-receipt-workspace> \
  --intermediate-workspace-root <fresh-local-intermediate-workspace> \
  --jobs 4
```

该选项不改变 mask 语义、source/episode 选择或 final 布局，只分离中间层存储。
本机空间不足以同时保留全部 source 的中间层时，可再加
`--remove-validated-intermediates`。流水线会逐 source 严格执行：完整生成 table/post/final，
重新核对 post 与 final 的 episode、frame、bitpack 字节数，删除该 source 在独立本机
workspace 下的 table/post 目录，最后原子发布包含验证摘要和清理状态的 source receipt。
删除失败或进程中断时不会发布 receipt，续跑会重新生成并验证该 source；final 与 receipt
始终保留在新 NAS workspace，raw source 和 calibration overlay 不会被删除。已发布清理
receipt 的 source 在续跑时用其中的固定 post 验证摘要并重新读取、验证 final，不会静默
跳过 final 验证。未使用该参数时，中间层必须保留到 full verify-only 完成。两种模式都
不得绕过 decoder warning 或 strict-media 失败。

两帧管线 smoke 必须使用独立 workspace，不能写入正式 final-mask root：

```bash
python -m utils.dataset_postprocess.wrist_cam_adjust.stage1_final_mask_pipeline \
  --profile <profile.json> \
  --binding-root <binding-root> \
  --urdf <pinned-urdf> \
  --workspace-root <fresh-smoke-workspace> \
  --source-id <one-source-id> \
  --max-frames 2 \
  --jobs 1
```

## 推荐大批量 Wrist Mask 路径

```bash
SESSION=datasets/2026-07-09_vive_rm65_real_teleop
NAME=${SESSION##*/}
URDF=assets/rm65_inspire_gripper_v3/rm65_dual_arm_wrist_cameras_v3_head_camera.urdf
TABLE_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/${NAME}_table_keep_mask
DEPTH_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/depth_mask_postprocess
FINAL_MASK_ROOT=utils/dataset_postprocess/wrist_cam_adjust/outputs/final_keep_mask

python -m utils.dataset_postprocess.wrist_cam_adjust dataset \
  --dataset-root "$SESSION" \
  --output-root "$TABLE_MASK_ROOT" \
  --urdf "$URDF" \
  --output-kind keep-mask \
  --table-mask-mode sampled-raycast \
  --table-raycast-stride 2 \
  --jobs 4 \
  --overwrite

python -m utils.dataset_postprocess.wrist_cam_adjust postprocess-depth-mask \
  --session "$SESSION" \
  --all-episodes \
  --output-root "$DEPTH_MASK_ROOT" \
  --jobs 4 \
  --no-preview \
  --overwrite

python -m utils.dataset_postprocess.wrist_cam_adjust compose-final-mask \
  --session "$SESSION" \
  --table-mask-root "$TABLE_MASK_ROOT" \
  --depth-mask-root "$DEPTH_MASK_ROOT" \
  --output-root "$FINAL_MASK_ROOT" \
  --jobs 4 \
  --overwrite
```

`compose-final-mask` 会检查 postprocessed depth manifest 和 table keep-mask video，
逐 camera/episode 生成最终 bitpack，并写 summary：

```text
<FINAL_MASK_ROOT>/<dataset>/final_mask_summary.json
```

用于抽样检查 random background 的视频：

```bash
python -m utils.random_background.generate_wrist_sample_video \
  --session "$SESSION" \
  --final-mask-root "$FINAL_MASK_ROOT" \
  --episode 0 \
  --max-frames 160 \
  --output-dir utils/dataset_postprocess/wrist_cam_adjust/outputs/${NAME}_random_background_sample \
  --provider procedural \
  --seed 20260710 \
  --output-tag proc_a \
  --side-by-side \
  --overwrite
```

真实图片背景池准备好后可替换为：

```bash
--provider real_image \
--real-image-manifest utils/random_background/real_images/openimages_v7_sample/real_background_manifest.json
```

最终训练建议直接读取 `FINAL_MASK_ROOT` 下的 manifest 在线合成背景，不建议为每个随机背景
离线复制一份完整视频 dataset。

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

## 桌面几何投影 Mask

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

## 深度范围 Mask

采集端的 depth mask 是 D405 对齐到 RGB 后的 z16 深度区间 mask：

```text
foreground = valid_depth && depth_min_m <= depth_m <= depth_max_m
```

默认有效距离在 `utils/dataset_postprocess/wrist_cam_adjust/depth_mask/core.py` 中定义：

```text
depth_min_m = 0.05
depth_max_m = 0.5
```

正式采集入口也可以通过 `--wrist-depth-min-m` / `--wrist-depth-max-m` 覆盖。已采集的 `datasets/2026-07-05_vive_rm65_real_teleop` metadata 仍记录旧范围 `0.095m` 到 `0.4m`，并且只保存了 `wrist_depth_mask`，没有保存 `wrist_depth_raw`；因此当前后处理从 bit-packed 0/1 mask 开始，而不是从 raw metric depth 开始。新默认值只会影响之后的采集或重新生成。

主体代码放在：

```text
utils/dataset_postprocess/wrist_cam_adjust/depth_mask/
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

## 几何投影输出模式

```bash
--output-kind masked          # 正式背景置黑，最快
--output-kind overlay         # 调试，背景红色叠加
--output-kind background-mask # 输出背景二值 mask
--output-kind keep-mask       # 输出 geometry_keep_mask
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

建议从 `--jobs 2` 或 `--jobs 4` 开始：

- `dataset --jobs N` 并行处理独立 camera/episode 视频，适合 table keep-mask。
- `postprocess-depth-mask --all-episodes --jobs N` 并行处理独立 bitpack mask stream。
- `compose-final-mask --jobs N` 并行合并独立 camera/episode 的 final keep-mask。

若 `read_frame` / `write_frame` 占比高，瓶颈在磁盘或编码；若 `build_mask` 占比高，优先调大
`--table-raycast-stride`。大批量正式处理建议加 `--no-preview`，只在抽样 episode 上生成 preview。

结构类 dataset 后处理工具已经避免全量复制大视频：可变的 `meta/` / `data/` 会复制，
视频、触觉、depth 等大文件优先 hardlink，跨文件系统时自动退回 copy。

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
