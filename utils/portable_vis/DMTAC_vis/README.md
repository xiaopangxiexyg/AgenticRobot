# DM-Tac Four-Part Viewer

只读可视化一个 DexTeleop dataset episode 的四路 DM-Tac 触觉变化。典型双夹爪数据按物理
顺序从左到右显示为 `LEFT OUTER | LEFT INNER || RIGHT INNER | RIGHT OUTER`：左右夹爪各自位于
画面左右两侧，inner 相邻、outer 位于外侧。每块传感器都转成竖向；outer 顺时针 90° 后再
上下镜像、inner 逆时针 90°。`Fx/Fy` 使用同一坐标变换，压力位置与箭头不会错位。

槽位首先从带 `inner/outer` 的 sensor key 解析；对真机的不透明序列号，则只读仓库正式
`teleop_runtime/configs/hardware/dmtac/latest.json` 映射。`--inspect` 会显示每路的
`tactile_slot`、`slot_source` 和最终方向，便于在导出前审计，不会按 manifest 偶然顺序猜测。

## 使用

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset --episode 0
```

如果数据集已经有 `tactile_force/` 派生流，`auto` 模式会优先显示四路三维力；可显式切换：

```bash
# 查看 SDK distributed_force
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --tactile-source force

# 强制查看采集时保存的 raw 图像
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --tactile-source raw
```

只有 raw 数据时，可以在打开查看器前直接完成离线解算：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 \
  --derive-force \
  --force-backend cpu \
  --force-sdk-root assets/DM_Tac_SDK_Publish_1.2.10
```

`--derive-force` 默认按传感器序列号自动加载已有固定校准；没有匹配配置时不会选择 episode
首帧，而是保留 SDK 原始解算。用 `--force-calibration-policy required` 可拒绝缺失配置，
`off` 可做未校准诊断。只有明确确认某帧无接触时才传
`--force-base-frame-index INDEX --force-save-calibration`，将四路 base frame 保存供以后自动
复用。已有且校准摘要一致的输出默认复用；校准发生变化时会自动重新解算。

`--tactile-source force` 只读取已生成的派生流，不会隐式写入数据集。如果指定 episode
尚未解算，命令行会打印一条包含当前数据集路径和 episode 的可直接复制命令；执行一次
该命令后，后续查看可以省略 `--derive-force`。

新解算的 `tactile_force` 默认以无损 Zstd level 3、每块 16 帧保存。查看器会依据 manifest
透明解压，不改变命令或 float32 数值。如需兼容未压缩输出，可显式传
`--force-compression none`。

也可直接输入 episode parquet、任意一侧的 `manifest.json` 或 tactile episode 目录：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py \
  /path/to/dataset/tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json
```

SSH 环境使用 `--no-open`，然后转发终端输出的端口。生成单帧图片而不启动服务：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --snapshot /tmp/dmtac_ep0.png --frame 120 --mode delta --style neon
```

## 视频导出

当前正式展示方案已经固化为 `standard/parallel_jaw_top_v1` profile。后续推荐直接使用：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --profile standard \
  --export-video /tmp/dmtac_standard.mp4
```

`standard` 锁定 `tactile-source=force`、`surface_turbo_top + current`、2560×1440、0.18 对称
时序平滑，以及当前确认的夹爪对应几何和 outer 上下镜像。即使命令行同时给出其他
`--style`、`--mode`、`--width`、`--temporal-smoothing` 或 `--export-styles/modes`，也以 profile
为准。输出路径、episode、帧裁剪和编码参数仍可按任务指定。若数据只有 raw，可在同一命令中
增加 `--derive-force --force-sdk-root assets/DM_Tac_SDK_Publish_1.2.10`。

纯画面交付使用 `--no-text`：它隐藏左右/内外标签、峰值、时间和色标数字，顶部标题留白也会
移除；仅保留曲面、网格、夹爪结构和底部窄颜色条：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --profile standard --no-text \
  --export-video /tmp/dmtac_standard_clean.mp4
```

同一 profile 也可用于浏览器和快照：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --profile standard

python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --profile standard --snapshot /tmp/contact.png --frame 120
```

导出一个视频时，输出路径保持不变。视频逐帧调用与浏览器、快照相同的渲染入口，数值模式、
episode 全局力尺度和四路物理空间对应关系完全一致：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --tactile-source force \
  --export-video /tmp/dmtac_show.mp4 \
  --style neon --mode current
```

一次导出多种风格或数值模式时，使用逗号分隔；工具会按
`文件名--风格--模式.mp4` 自动生成多个文件：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --export-video /tmp/dmtac_show.mp4 \
  --export-styles neon,neon_focus,neon_contour,components,surface,surface_left,surface_front,split \
  --export-modes current,delta,baseline \
  --export-fps 30
```

预备方案评审建议直接导出经过筛选的 30 版，而不是生成全部笛卡尔积：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --export-video /tmp/dmtac_review.mp4 --export-showcase-30
```

输出使用 `--v01--...` 到 `--v30--...` 编号。`--export-styles all --export-modes all`
仍可导出全部注册风格与模式组合，但通常没有评审价值。常用裁剪参数：

只导出精进后的 turbo 3D 十版候选：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset \
  --episode 0 --export-video /tmp/dmtac_3d.mp4 --export-showcase-3d-10
```

- `--start-frame 100 --end-frame 300`：首尾均包含；
- `--frame-step 2`：每两帧取一帧；未显式指定 FPS 时会同步减半，保持真实播放时长；
- `--width 2560`：推荐的 3D 展示宽度；3D 视频固定为 16:9，因此输出为 2560×1440；
- `--video-codec mp4v`：四字符 OpenCV FourCC；`.avi` 默认使用 `MJPG`，其他后缀默认
  `mp4v`。
- `--temporal-smoothing 0.18`：显示场使用 82% 当前帧、18% 相邻帧均值。它不修改文件、
  数值文本、轨迹尺度或 CoP 判定，并且对称混合不会产生播放滞后；建议仅用于离线展示视频。

若只想快速对比所有风格而不批量导出，直接打开浏览器界面的 Style 下拉框即可实时切换。

## 展示风格

所有风格仅改变绘制方式，不修改 raw、distributed force、差分、基线或 episode 归一化结果：

- `classic`：原有离散压力热图 + `Fx/Fy` 白色箭头；raw uint8 保持原图；
- `neon`：平滑、高对比的展示色图和轻微辉光，适合演示屏与视频；箭头使用低亮度
  阴影和彩色主体，不再使用过曝的纯白色；
- `neon_focus`：将低于固定噪声门槛的区域压暗，仅突出有效接触边界和切向箭头；
- `neon_contour`：在 neon 背景上叠加等值线，突出压力层级和接触边界；
- `components`：使用相同物理尺度并排显示有符号 `Fx`、有符号 `Fy`、`-Fz`；
- `contour`：压力热图叠加等值线，便于看接触边界、传播范围和细小形变；
- `vectors`：深色矢量场，压力用光点面积/颜色表示，切向力用箭头表示；
- `surface`：右斜视 3D；每只夹爪的 inner/outer 同时放在一个三维场景中，形成空间分离、
  相互平行的两片夹持面，而不是四块彼此无关或铺在同一平面的曲面。压力使表面向各自夹爪
  实体内部凹陷，曲面采样点箭头表示投影到传感器平面的 `Fx/Fy`；
- `surface_left`：同一数值的左斜视 3D，便于检查被右侧透视遮挡的结构；
- `surface_front`：低纵深正视 3D，便于比较压力高度和横向剖面；
- `split`：每路左侧热图/箭头、右侧 3D 曲面，同时看平面位置和立体形状。

3D 曲面先把固定的 600-cell 显示网格旋转为 30×20 竖向布局，再沿两轴各插值 2 倍绘制
高密度网格。插值只用于几何与颜色，
统计数值、CoP 门槛和 episode 归一化仍使用未变的原显示网格。当前提供 `magma`、
`viridis`、`turbo`、`cividis`、`ocean` 五套 3D 压力色板，以及 cyan、amber、mint、coral
四套克制的箭头主题。

同一夹爪的 inner/outer 是像真实夹爪一样相对放置的平行面：中间保留夹持间隙，以底部基座
和外侧薄背板提示两者属于同一夹爪，而不是两张共面的图。第二夹持面从相反边缘参数化，因而
局部坐标中互为镜像的 taxel 会落到共同的夹爪空间位置；这只改变几何映射，不修改传感器数值。
三条低亮度横向参照线用于检查对应高度。两面的压力位移都远离中央夹持间隙，方向与 v10 候选
相反。左右夹爪分别占据画面左右半区；每半区只保留
`LEFT/RIGHT`、`OUTER/INNER`、紧凑峰值和共享 `-Fz` 色标，不显示 episode、序列号、schema、
风格名或调试轨迹等冗余文字。

力场面板还会显示：`Fx/Fy/Fz` 空间均值、三维合力峰值、`-Fz` 均值/峰值、切向力
均值/峰值、接触 taxel 比例、归一化压力中心 `CoP (x, y)` 和整段轨迹参考尺度 `ref`。
低亮度细环标出 CoP，竖直色标明确当前数值范围。CoP 不直接使用全部正压力：先去掉低于整个
episode 参考尺度 4% 的噪声，再要求峰值至少达到 10%、去噪压力总量至少达到 20%，并至少有
3 个有效网格；门槛未满足时明确显示 `CoP hidden`，不绘制会随噪声抖动的主受力点。Delta 和
Baseline 用法向变化绝对值做相同判定，因此加载、卸载事件都可定位。该门槛只控制展示，
不改写 tactile 数值。

非力数值显示 min/max/mean/RMS、逐帧变化和显示尺度。底部四路时间曲线仍是原有
`mean(abs(current - previous))`，用于在展示中突出细微接触发生的时刻。

### 建议的一般展示组合

面向演示时不建议把所有风格平均使用。通常保留以下组合即可覆盖空间位置、灵敏变化、分量
解释和立体形状：

1. `neon + current`：主画面，表达当前压力和切向力；
2. `neon + delta`：灵敏变化画面，突出接触瞬态；
3. `neon_focus + delta`：噪声较明显时替代第 2 项；
4. `components + current`：需要解释 `Fx/Fy/-Fz` 来源时使用；
5. `surface + current`：默认 3D 形状和曲面切向力；
6. `surface_left + current`：默认视角遮挡接触峰时切换；
7. `surface_front + current`：比较高度剖面时切换；
8. `split + current`：只能保留一个综合视频时使用。

`baseline` 仅在 episode 首帧已确认是稳定、无接触基准时使用，否则它容易把首帧偏置包装成
有意义的触觉变化。一般展示默认以 `current + delta` 为主。

只检查工具识别出的 schema、通道、shape 和 dtype：

```bash
python utils/portable_vis/DMTAC_vis/dmtac_viewer.py /path/to/dataset --episode 0 --inspect
```

## 兼容格式

- v2.2、v2.3 dataset root 和 episode 定位方式；
- `binary_stream_v1` 外置 tactile manifest；
- 未压缩 raw C-order、chunked LZ4、chunked Zstd；
- parquet inline tactile 数值向量（需要 `pyarrow`）；
- `uint8` 原始灰度/RGB 触觉图像；
- 一维、二维和多通道整数/浮点触觉数值。

当 stream 的字段为 `distributed_force`（通常 shape 为 `H × W × 3`、float32）时，
查看器自动切换到三维力场显示。依据 SDK 1.2.10 自带的 distforce 对比定义，第 0/1 通道
对应 shear X/Y，第 2 通道对应 depth；因此颜色表示压缩方向的 `-Fz` 法向压力，克制的彩色
箭头表示触觉平面内的 `Fx/Fy` 方向和相对强度。旋转传感器画面时会对 Fx/Fy 使用同一旋转
矩阵，Fz 和力幅值不变；面板底部显示的是最终物理显示坐标中的三个分量空间均值。
Delta/Baseline 模式下，法向背景改为有符号差分色图，箭头表示 `Fx/Fy` 的变化方向。

力场色标按整个 episode 归一化，而不是每帧单独归一化。查看器启动时会扫描四路完整轨迹，
为 Current、相邻帧 Delta、首帧 Baseline 分别计算一个共享的三维力绝对峰值 `ref`。
同一个 `ref` 同时控制 `-Fz` 热图和 `Fx/Fy` 箭头，不再对较小的切向分量单独放大。
播放期间固定使用对应尺度，因此 `1e-6`/`1e-5` 级微小量不会仅因为是当帧或某个分量的
最大值就被显示成满量程，四个触觉区域之间也可以直接比较强弱。每个力面板底部会显示
局部 `peak` 和当前使用的 `traj ref`。

`raw uint8` 图像本身还不是 `(Fx, Fy, Fz)`。查看器不会在 HTTP 请求线程中运行 SDK；使用
`--derive-force` 时，会在服务启动前调用仓库的离线处理器，生成可重复读取的
`tactile_force/`。DexTeleop 保留 raw 的设计目的之一正是离线解算：供应商 SDK 支持重建
`DMTacImage`，再调用 `Sensor.process(...)` 得到 `distributed_force`。也可单独运行后处理入口：

```bash
python -m utils.dataset_postprocess.DM_force_calculate \
  --dataset-root /path/to/dataset \
  --episode 0 \
  --backend cpu \
  --sdk-root assets/DM_Tac_SDK_Publish_1.2.10
```

该工具默认不设置 base frame，直接保存 SDK 的原始解算结果为 float32
`[288, 384, 3]` stream，并写到 `tactile_force/`。只有显式传入
`--base-frame-index` 时，才会把指定的已确认无接触帧设置为 base frame。sensor key/serial、
SDK 版本、mode、backend，以及可选的 base-frame 选择都属于离线派生条件，应随结果一同审计。

也可以在采集时直接请求实时解算结果，但这不是 raw-only 离线设计的必要条件：

```bash
--dmtac-tactile-fields distributed_force \
--dmtac-tactile-compute-profile dist-only
```

界面提供 Current、相邻帧差分和首帧基线差分。底部曲线是每路传感器逐帧
`mean(abs(current - previous))`，用于快速定位接触变化。原始图像保持灰度/RGB 显示；
数值数据转为热力图；差分使用有正负方向的色图。

工具最多展示四路。如果输入单侧 manifest，只显示该侧的两个物理槽位，其余面板明确标为空；
若数据包含超过四路，当前版本读取前四路后再按物理槽位排序。
