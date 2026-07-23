# UMI Vive→desk 标定

本工具标定一套刚性连接的 RealSense 相机和 Vive tracker。标定板固定不动，且其
坐标系直接定义为 v2.3 canonical `desk`。因此标定板必须固定为已标定 RM65 station
URDF 的 `workspace_near_boundary_frame`：该原点在桌面表面上左右居中，并沿“桌面中心到
操作者/机械臂”的方向位于有效操作空间最靠近操作者/机械臂一侧的边界。标定板原点的
物理位置和 xyz 轴方向都必须与其重合，不能只是把任意板坐标系命名为 `desk`。采样时手持整套 UMI 改变位置和姿态；不要
让相机与 tracker 发生相对移动。

矩阵统一采用 `p_B = T_A_to_B @ p_A`。每组样本满足：

```text
T_tracker_to_vive_world @ T_camera_to_tracker @ T_board_to_camera
    = T_board_to_vive_world
```

因此输出的 v2.3 变换为：

```text
T_vive_world_to_desk = inverse(T_board_to_vive_world)
```

同一结果还会从双臂 URDF 与命名 TCP profile 派生：

```text
T_tcp_to_camera  = inverse(T_camera_to_endpoint) @ T_tcp_to_endpoint
T_tcp_to_tracker = T_camera_to_tracker @ T_tcp_to_camera
```

默认 `tcp_profile=dmgripper`，解析到左右各自的 RM65 `DMTAC_M` profile；其正式
`tool_tcp` 相对 `arm_endpoint` 为 `[0, 0, 0.14] m`、单位旋转。相机关系来自
`rm65_dual_arm_wrist_cameras_v3_head_camera.urdf` 的左右 `camera_rgb` optical frame。
结果会记录 URDF/tool-profile 路径和 SHA-256、各段矩阵及最终
`T_tcp_to_tracker` 及其逆 `T_tracker_to_tcp`，不把这些几何数值复制到 UMI 配置中。真机采集仍使用自身的
URDF desk 配置。

## 配置

默认配置是 `configs/rm65_dual_wrist_umi_v1.json`：

- 左腕相机：`261122273924`，tracker：`tracker_1`
- 右腕相机：`260522278055`，tracker：`tracker_2`
- 相机请求分辨率：`1280×720@15 FPS`（与 `utils/head_eye` 默认值一致；不支持时由
  RealSense profile fallback 选择兼容模式）
- 标定有效期：3600 秒
- 默认 TCP profile：`dmgripper`（左右分别解析 `DMTAC_M`）

两套 UMI 都在配置中声明。可用 `--side left/right` 单侧求解，也可用 `--side both`
在一个可缩放窗口中同时监看并分别标定左右两套 UMI。Vive sender 仍发送现有两只
tracker，不需要第三只 tracker。

## 标定板参数（当前默认值）

默认配置使用 ChArUco 标定板，参数来自
`configs/rm65_dual_wrist_umi_v1.json` 的 `target` 节，不是运行时猜测：

| 参数 | 值 |
| --- | --- |
| ArUco dictionary | `DICT_6X6_100` |
| 棋盘方格数 | `5 × 7` |
| 单个方格边长 | `35 mm` |
| 单个 ArUco marker 边长 | `26 mm` |
| 棋盘主体尺寸（不含纸张/外边白边） | `175 × 245 mm` |
| ChArUco 内角点总数 | `4 × 6 = 24` |
| 允许采样的最少内角点 | `12` |

UMI 默认先尝试 ChArUco 内角点插值；如果内角点少于 12，但检测到的 marker 属于这块
5×7 板，则启用 ArUco board fallback，使用整块板的 marker 坐标估计同一个固定 board
坐标系。它不是随机选择一个 marker，因此不会让不同 sample 使用不同坐标系。可用
`--no-allow-aruco-board-fallback` 强制关闭该 fallback。UMI 采样层还要求 fallback
至少匹配 4 个 board marker；少于 4 个时会明确显示
`reject aruco-board markers=N/4`，避免单 marker 位姿被当成有效样本。该门槛可用
`--min-aruco-board-markers` 调整。

打印板上的方格数、字典和边长必须与上表一致。`target_type=charuco` 时，配置中
遗留的 `marker_length_m=0.1` 不参与 ChArUco 姿态估计；真正使用的是
`charuco_square_length_m=0.035` 和 `charuco_marker_length_m=0.026`。

启动时终端会再次打印实际生效的 target 参数，画面左上角会显示详细检测状态，例如：

```text
target: charuco corners=10/12 markers=17
target: ok markers=17 charuco=24/24, reproj=0.32px
```

前一种状态表示虽然检测到了 17 个 ArUco marker，但有效 ChArUco 内角点只有 10 个；
如果这些 marker 能匹配当前 5×7 board，程序会进一步显示类似
`target: ok aruco-board markers=17 charuco=10/12 (board_max=24)`，该状态可以被采样器接受。
若仍然没有 `ok`，说明 marker 字典/板参数或图像质量也没有满足 board pose 估计。

## 在线采集

先启动现有 Vive sender，然后在采集电脑运行：

```bash
python -m utils.umi_hand_eye --side left
python -m utils.umi_hand_eye --side right
python -m utils.umi_hand_eye --side both
```

默认启用自动采样：tracker 在一个新姿态保持静止约 0.5 秒，并且标定板已经得到有效
ChArUco 姿态后，程序自动执行一个 2 秒 burst 并保存 sample。需要调整识别板参数时，
也可以从命令行显式覆盖配置，例如：

```bash
python -m utils.umi_hand_eye \
  --side left \
  --target-type aruco \
  --dictionary DICT_6X6_100 \
  --marker-id 0 \
  --marker-length-m 0.0997
```

若需要完全手动采样，传 `--no-auto-record`；这时必须在 OpenCV 画面获得键盘焦点后
按 `r`。

单侧模式只开启所选 camera serial；`--side both` 会依次启动左右 pipeline，并在采集
期间让两台相机同时保持流状态。左右自动判断、稳定计时、姿态门控和样本集合完全独立：
任意一侧单独满足条件时就立即只记录该侧，不会等待另一侧也满足。若两侧恰好在同一轮
同时触发，两个 2 秒 burst 会由每侧独立 worker 并行读取各自相机，不再串行冻结 GUI。
正在 burst 的一侧显示最后一帧与 `RECORDING BURST`，另一侧仍持续刷新；样本校验、状态机
更新和 JSON 落盘统一回到主线程执行。启动时默认先读取 5 帧做 warmup。若没有任何帧到达或遇到
USB/UVC 连接错误，会停止 pipeline，仅对所选 serial 执行一次 RealSense hardware
reset，等待设备重新枚举后再启动一次。可用 `--no-camera-reset-on-failure` 禁止该恢复；
reset 始终按失败侧 serial 执行，不会 reset 另一侧相机。启动日志会打印 requested/actual profile；如果
`actual` fallback 到较低分辨率，需根据画面中的 ChArUco 角点数量进一步调整相机距离或
profile。

UMI 相机请求原生 `yuyv` color transport，并在硬件边界转换为 `CameraFrame.color_bgr`；
因此标定板检测仍接收原来的 BGR 图像，不会把 YUYV 字节流直接传给 OpenCV。若要排查
底层链路，可在不执行 reset 的情况下使用 `--no-camera-reset-on-failure` 做单 serial
读取测试。

成功求解并保存后，程序默认继续进入实时检查窗口：相机画面会投影标定得到的 desk
坐标轴，右侧面板显示所选 tracker 在 desk 坐标系下的实时 XYZ（米）和 RPY（度），
同时显示相机在 desk 坐标系下的实时 XYZ（米）和 RPY（度），以及当前检测到标定板时
的 desk/board 位姿残差。面板俯视图用 `T`、`C` 分别标出 tracker 和 camera；tracker 自身坐标轴若位于相机视野
前方也会投影在画面中。按 `q`/`Esc` 完成检查；自动化或无需检查时可传
`--no-visualize-after-solve`。

标定程序退出后也可以只读方式重新打开已保存结果，不重新采样或求解：

```bash
python -m utils.umi_hand_eye --view-result --side left
python -m utils.umi_hand_eye --view-result --side right
python -m utils.umi_hand_eye --view-result --side both
```

默认分别读取 `--output-dir` 下的 `latest_left.json` / `latest_right.json`。单侧可用
`--result PATH` 指定结果；双侧可用 `--left-result PATH --right-result PATH` 分别覆盖。
查看模式会严格检查结果中的 side、camera serial 和 tracker name 是否与配置一致，但只
读取相机与 Vive tracker，不执行机械臂运动或设备控制。

采集窗口和标定后检查窗口都使用 OpenCV `WINDOW_NORMAL`，可以拖动自由缩放。双侧画面
会保持宽高比并自动缩放到默认 `1600×900` 范围；程序每帧读取窗口实际客户区尺寸并
重新缩放 RGB 画布，因此拖大或拖小窗口时图像内容会同步变化，而不只是窗口外框变化。
窗口创建时会先用完整占位画布初始化 HighGUI，再设置实际客户区尺寸，并使用 Qt HighGUI
的 `WINDOW_FREERATIO` 让图像控件铺满客户区；RGB 宽高比由程序生成的画布负责保持。
这避免 `WINDOW_KEEPRATIO` 在大窗口中央保留一个固定的小图像控件。双侧画面比例与窗口
不一致时使用黑色留边，输出画布本身仍与窗口客户区等大。可用 `--window-width`、
`--window-height` 修改初始画布大小。

采集界面中 RGB 区域只显示相机画面和 target 坐标轴，不再叠加状态文字。每侧 RGB 下方
都有独立的深色 dashboard，显示 target 检测、自动采样状态、样本数、覆盖跨度和 `NEXT`
提示。文字在最终窗口分辨率上绘制，使用较大字号和 2 像素线宽，避免随 RGB 整体缩放后
变小或模糊。

窗口按键：

- 自动模式：新姿态静止且 target 有效后自动记录一个 2 秒 burst；
- `r`：单侧模式记录当前侧；双侧模式依次尝试记录左右两侧；
- `1` / `2`：双侧模式仅尝试记录左侧 / 右侧；
- `c`：两侧分别通过样本数和位姿覆盖门槛后求解；
- `q` / `Esc`：退出。

默认最低样本数降为 10，但不是任意 10 个姿态都可求解：每个新样本必须相对所有已保存
样本同时变化至少 `40 mm` 和 `10°`；最终还必须达到至少 `100 mm` 总平移跨度、`30°`
总旋转跨度，并包含至少两个相差 `25°` 的非平行旋转轴。画面底部持续显示 count、span、
rotation axes 和 `NEXT` 提示；覆盖不足时按 `c` 会拒绝求解。上述门槛可通过
`--auto-new-pose-*` 和 `--coverage-*` 参数调整，但不建议为了凑满 10 个样本而降低。

所有采样、覆盖度和窗口参数都暴露为命令行参数。例如：

```bash
python -m utils.umi_hand_eye \
  --side both \
  --min-samples 15 \
  --auto-stable-sec 0.3 \
  --auto-target-still-translation-m 0.003 \
  --auto-target-still-rotation-deg 0.8 \
  --auto-new-pose-translation-m 0.04 \
  --auto-new-pose-rotation-deg 10 \
  --pose-change-require-both \
  --coverage-translation-m 0.10 \
  --coverage-rotation-deg 30 \
  --coverage-axis-separation-deg 25 \
  --coverage-axis-observation-rotation-deg 8 \
  --window-width 1600 \
  --window-height 900 \
  --target-type aruco \
  --dictionary DICT_6X6_100 \
  --marker-id 0 \
  --marker-length-m 0.0997
```

还可调节 burst 的 `--capture-seconds`、`--min-detections`，以及静止判断的
tracker 门槛 `--auto-still-translation-m`、`--auto-still-rotation-deg`、标定板门槛
`--auto-target-still-translation-m`、`--auto-target-still-rotation-deg` 和
`--auto-cooldown-sec`。自动计时期间 tracker 或标定板任一方移动、或标定板短暂丢失，
都会重新开始 0.5 秒计时，避免过早进入一次必然被 burst 运动过滤拒绝的采集。
传 `--no-pose-change-require-both` 会将单样本门槛放宽为平移或
旋转满足其一；这会降低十样本求解的姿态多样性，默认不建议使用。

结果默认写入 `teleop_records/umi_hand_eye/`：

- `*_samples.json`：原始配对样本，可离线重算；
- `*.json`：标定结果、残差、设备绑定和过期时间；
- `latest_left.json` / `latest_right.json`：每次成功求解后更新，供默认数采配置绑定；
- `*_pose_frame_contract.json`：v2.3 UMI 数采使用的 `desk_absolute` contract。

## 离线重算与有效期检查

```bash
python -m utils.umi_hand_eye \
  --side left \
  --load-samples teleop_records/umi_hand_eye/example_samples.json

python -m utils.umi_hand_eye \
  --side left \
  --check-result teleop_records/umi_hand_eye/example.json
```

`--check-result` 会同时检查 1 小时有效期、camera serial、tracker name、TCP profile
和 `T_tcp_to_tracker` 是否存在。过期、SteamVR tracking space 重建、基站移动、
相机/tracker 装配变化或 TCP profile 变化后都必须重新标定。

## 时间配对限制

第一版不是运动中硬同步。每个视觉检测帧读取最新 Vive pose，并要求其接收年龄默认不
超过 250 ms；随后在整个 burst 上检查 tracker 和视觉 target 都保持静止。该策略适合
人工摆位后的静止标定，不适合边移动边连续采样。自动状态机只负责判断“新姿态已静止”，
真正保存 sample 仍然经过 burst 的角点、重投影、运动稳定性和全历史姿态差异检查。
双侧模式是共享 UI/生命周期、左右独立求解，不是两个相机帧与两个 tracker 的硬件级同步
联合优化。
