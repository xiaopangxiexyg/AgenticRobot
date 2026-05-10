# Teleop Collection README

## 1. 目的

这份 README 只说明当前仓库里 `teleop_runtime.collection` 的真实可用数据采集链路，不讨论更早期的 blueprint 规划。

当前推荐主线是：

- 先打通 `--no-camera` 的双臂双手采集
- 确认双臂 arm / 双手 q6 / InspireTac telemetry 都正常后，再接 RealSense RGB
- 单臂模式或不含手模式不是单独一套实现，而是复用同一套双臂 schema，未启用部分留空

当前入口脚本是：

- [run_collect.py](run_collect.py)

## 2. 当前支持范围

当前 collector 已支持：

- `active_sides=left|right|both`
- `VIVE wrist pose -> arm retargeting -> RM65`
- `hand q6 serial -> hand retargeting -> InspireTac`
- `InspireTac` 关节 telemetry 与 tactile 采集
- 可选单路 `RealSense RGB`
- LeRobot v2.1 风格的 `parquet + meta + mp4` 输出
- 同一 `dataset-root` 下自动续写 episode index，不从 `episode_000000` 重新覆盖

当前推荐优先跑的配置是：

- 双臂
- 双手 q6
- 有真实 InspireTac hand 就接 `--left-hand-ip/--right-hand-ip`
- 先 `--no-camera`

## 3. 环境与依赖

推荐环境：

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy
```

采集链路按需依赖：

- RM65 SDK：需要能导入 `hardware.rm65.robotic_arm_package.robotic_arm`
- hand q6 串口读取：需要 `hardware.teleop_hand.hand_q6_reader`
- InspireTac：需要 `hardware.inspire_tac.inspire_tac_modbus`
- parquet 导出：需要 `pyarrow`
- 视频导出：需要 `opencv-python`
- RealSense 相机：需要 `pyrealsense2`

如果你只跑 `--no-camera`，就不需要 RealSense 相关依赖。

## 4. 采集前准备

正式启动 collector 前，先确认这几件事：

1. VIVE 发送端已经在发左右 wrist tracker，且 `tracker_role` 分别对应 `left_wrist` 和 `right_wrist`。
2. 左右 RM65 已经上电、联网，并处于安全起始姿态。
3. 如果启用手 teleop，左右 q6 串口已经确认可读。
4. 如果启用真实 InspireTac，左右 hand IP 已经确认可连。
5. 首次联调时，把机器人周围清空，并优先给 `--max-steps` 一个小值。

VIVE 侧如果沿用当前默认配置，receiver 默认读取：

- `tracker_1 -> left_wrist_pose`
- `tracker_2 -> right_wrist_pose`

## 4.1 加载标定文件

如果已经通过 calibration tool 保存过标定 profile，用 `--load-profile` 加载：

```bash
--load-profile teleop_records/calibration_profiles/20260420_150620_vive_rm65_virtual_desk.json
```

完整位置建议放在 `--active-sides` 后面，示例：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root datasets/demo_bimanual_no_camera \
  --task grasp_demo \
  --active-sides both \
  --load-profile teleop_records/calibration_profiles/20260420_150620_vive_rm65_virtual_desk.json \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --right-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --no-camera \
  --no-video
```

这里要区分两层标定：

- `--load-profile`
  加载保存好的固定标定，例如 virtual desk、tracker offsets、desk/base 关系等。
- 运行时按 `p`
  采集本次 session 的起始位置 home anchor。即使加载了 `--load-profile`，每次采集前仍然要按 `p`。

当前 `run_collect.py` 会在加载 profile 后清掉旧的 home state，只保留固定标定；这样不会误用上一次 session 的起始位。

## 5. 最小启动命令

### 5.1 双臂双手，无相机

这是当前推荐的第一条真机采集路径：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /path/to/datasets/demo_bimanual_no_camera \
  --task grasp_demo \
  --task-description "bimanual no-camera collection" \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --no-camera \
  --no-video \
  --hz 30 \
  --calibration-timeout 15 \
  --calibration-samples 20
```

说明：

- `--left-hand-port/--right-hand-port` 决定是否读取 q6，并决定是否生成 hand retargeting command。
- `--left-hand-ip/--right-hand-ip` 决定是否真的连接 InspireTac hand 并记录 hand telemetry/tactile。
- `--inspire-hand-max-force N` 设置 InspireTac 启动时写入每个手指的 force，默认 `800`；兼容旧参数 `--max-fingertip-force`，也可用简写 `--max-force`。
- `--no-camera` 后，视频链路完全不启动。

### 5.2 双臂，无真实手，只采 q6

如果你想先验证双臂 + q6 输入，但暂时不接真实 InspireTac：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /path/to/datasets/demo_bimanual_q6_only \
  --active-sides both \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --no-camera \
  --no-video
```

这时会有：

- q6 输入
- hand target command
- hand backend 的本地 last command state

但不会有真实 hand telemetry/tactile。

### 5.3 双臂，无手

如果你只想先测 arm collector：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /path/to/datasets/demo_bimanual_arm_only \
  --active-sides both \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --no-camera \
  --no-video
```

### 5.4 单臂退化模式

单臂不是另一份 schema，只是未启用 side 留空：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /path/to/datasets/demo_left_only \
  --active-sides left \
  --left-arm-ip 192.168.101.21 \
  --no-camera \
  --no-video
```

此时：

- `right` 侧字段仍然存在
- 但低维槽位会留空
- dataset meta 里会写 `active_sides=["left"]`

## 6. 运行时行为与按键

当前 collector 的运行流程已经改成显式按键控制。所有按键都不区分大小写，例如 `p` 和 `P` 等价。

按键语义：

- `p`
  采集当前起始位置标定。它会读取当前真机 arm qpos、采样 wrist tracker home、写入 runtime home，并让机械臂进入随动。
- `z`
  开始录制一个新 episode。必须先按过 `p`，否则不会开始录制。
- `c`
  结束当前录制并保存为一条 episode。
- `d`
  中止当前录制并丢弃这条 episode。典型用法是先按 `z` 开始，发现动作不合格后按 `d` 丢弃。
- `h`
  锁定机械臂，也就是停止继续发送随动命令，并清除之前的 `p` 标定。再次按 `p` 才会重新标定并恢复随动。
- `q`
  退出 collector。如果退出时仍有 active recording，会先保存当前 episode。

### 6.1 预录轨迹回放采集

如果已经有一条 LeRobot 风格数据集或单个 parquet，可以让 collector 不再读取 VIVE 轨迹，而是按预录 `action` 自动执行，并继续使用当前 RealSense/telemetry writer 采集新的实时观测。

示例：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /path/to/new_datasets/replay_collect \
  --task-name replay_collect_demo \
  --active-sides both \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --replay-dataset /home/ai/Desktop/DexTeleop/datasets/2026-05-06_right_arm_dual_data_soft_hard \
  --replay-group a:0-8 \
  --replay-group d:9-17 \
  --replay-episode 0 \
  --replay-init-time 1.0 \
  --replay-randomize \
  --replay-random-xyz-m 0.01 \
  --replay-random-rot-deg 2.0
```

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /path/to/new_datasets/replay_collect \
  --task-name replay_collect_demo_right \
  --active-sides both \
  --right-arm-ip 192.168.1.20 \
  --right-hand-ip 192.168.11.210 \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --replay-dataset /home/ai/Desktop/DexTeleop/datasets/2026-05-06_right_arm_dual_data_soft_hard \
  --replay-group a:0-8 \
  --replay-group d:9-17 \
  --replay-init-time 1.0 \
  --replay-randomize \
  --replay-random-xyz-m 0.01 \
  --replay-random-rot-deg 2.0
```


默认按键分组：

- `a`: 轮转执行数据集中全部 source episode 的完整轨迹。
- `x`: 中止当前自动轨迹并丢弃本 episode。

也可以手动定义更多组：

```bash
--replay-group a:0-49 \
--replay-group s:50-99 \
--replay-group d:100-149
```

每个组绑定源数据集里的 episode index 范围，并完整执行被抽到的 source episode：

```bash
--replay-group a:0-49 \
--replay-group d:50-99
```

这里 `a:0-49` 表示：

- 按 `a` 启动。
- 只在源数据集 `episode_index=0..49` 之间 round-robin 轮转。
- 每次抽到某个 source episode 后，从该 episode 第一帧执行到最后一帧。

episode 范围支持：

- `all` 或省略：使用可用源 episode 全部范围。
- `x-y`：闭区间，例如 `10-29`。
- `x..y`：同样是闭区间。
- `x,y,z`：显式 episode 列表。

回放模式行为：

- 按轨迹组键后自动开始一个新 episode。
- 如果使用 `--replay-dataset`，默认会在该数据集的全部 source episode 间按 round-robin 轮转，尽量保证每条轨迹被均等抽到；不追求严格随机。
- 如果只想固定使用一条源轨迹，可以加 `--replay-single-episode --replay-episode N`。
- 先用 `--replay-init-time`，默认 `1.0s`，从当前硬件状态缓慢插值到该轨迹组第一帧 action；这段也会写入新 episode。
- 初始移动完成后按源轨迹推断的 action Hz 逐帧发送动作。
- 轨迹段结束后自动保存 episode。
- 相机仍然实时采集；新 dataset 保存的是当前执行过程中的实时相机帧，不是源数据集的视频。

轨迹随机化：

- `--replay-randomize` 只随机 arm pose，不随机手部关节。
- 默认 `--replay-random-mode both`，会叠加 episode 级固定偏移和低频平滑扰动；不会逐帧加白噪声。
- `--replay-random-xyz-m` 和 `--replay-random-rot-deg` 控制 episode 级最大偏移，建议先从 `0.005~0.01m` 和 `1~2deg` 开始。
- smooth noise 默认使用一半幅度；也可以显式设置 `--replay-random-noise-xyz-m` 和 `--replay-random-noise-rot-deg`。
- `--replay-random-seed N` 可复现随机化；默认负数使用系统随机数。
- 随机化后的第一帧会作为 `--replay-init-time` 的目标，因此 init 段和 replay 段不会在开始处跳变。
- 每条 episode metadata 会写入随机种子、幅度和 episode 级偏移；每帧 sample metadata 会写入当前随机偏移。

init_episode 预备模式：

如果每次正式录制 A/D 前，需要先无录制地执行一条固定准备轨迹 `T`，可以加 `--replay-init-episode`。

`T` 可以来自主 `--replay-dataset`，也可以用 `--replay-init-dataset` 或 `--replay-init-parquet` 单独指定。这样 A/D 正式采集轨迹和 H 预备轨迹可以来自不同数据集。

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /home/ai/Desktop/DexTeleop/datasets/2026-05-06_right_arm_dual_data_soft_hard_extend \
  --task-name replay_collect_demo_right \
  --active-sides right \
  --right-arm-ip 192.168.1.20 \
  --right-hand-ip 192.168.11.210 \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --replay-dataset /home/ai/Desktop/DexTeleop/datasets/2026-05-06_right_arm_dual_data_soft_hard \
  --replay-init-dataset /home/ai/Desktop/DexTeleop/datasets/2026-05-06_right_fixed_flip_hand \
  --replay-init-episode 0:0:end \
  --replay-init-episode-move-time 3.0 \
  --replay-init-episode-to-record-time 1.0 \
  --replay-group a:0-8 \
  --replay-group d:9-17 \
  --replay-randomize \
  --replay-random-mode both \
  --replay-random-xyz-m 0.03 \
  --replay-random-rot-deg 3.0
```

按键流程：

1. 按 `h`：机械臂按照 `T` 第一帧的 arm qpos 归位，保证和源轨迹初始关节状态严格一致；同时发送 `T` 第一帧对应的手部目标位置。这段不录制。
2. 再按 `h`：按 `T` 的原始 action Hz 逐帧执行完整 `T`，不录制；arm 和 hand 都直接按 `T` 的原始轨迹发送。
3. 按 `a` 或 `d`：先用 `--replay-init-episode-to-record-time`，默认 `1s`，无录制移动到所选 A/D 轨迹第一帧；这一步会把 arm 对齐到首帧，同时发送对应 hand 目标。
4. 然后自动开始真正录制，并按所选 A/D source episode 的原始 action Hz 执行 replay + 随机化逻辑。
5. A/D 轨迹结束后自动停止录制并保存。
6. 下一轮再次从按 `h` 开始。

`--replay-init-episode` 格式：

- `N`：使用源数据集 `episode_index=N` 的完整轨迹作为 `T`。
- `N:start:end`：只使用该 episode 的 `[start, end)` 帧作为 `T`。注意：这是 `init_episode` 自己的帧切片语法，和 `--replay-group` 的 episode index 分组语法不同。

init 轨迹来源：

- 不写 `--replay-init-dataset/--replay-init-parquet`：`T` 默认从主 `--replay-dataset` 或 `--replay-parquet` 读取。
- `--replay-init-dataset /path/to/init_dataset --replay-init-episode 3:20:180`：从指定 init 数据集读取 `episode_index=3` 的第 `20..179` 帧作为 `T`。
- `--replay-init-parquet /path/to/episode_000003.parquet --replay-init-episode 0:20:180`：从指定 parquet 读取指定帧段。单 parquet 模式下 episode index 主要用于 metadata 和解析路径兼容，通常写 `0` 即可。

时间参数含义：

- `--replay-init-episode-move-time`
  历史上用于第一次按 `h` 的末端插值时间。当前第一次 `h` 已改为 qpos 严格归位，不再使用末端插值；实际速度主要由 `--replay-arm-max-angvel` 和 RM65 后端跟踪能力决定。这段不录制，也不消耗 `T` 的帧。
- `--replay-init-episode-to-record-time`
  只作用于按 `a/d` 后、正式录制前。它控制机械臂从执行完 `T` 后的位置，无录制移动到所选 A/D 轨迹第一帧 arm pose 的时间；这段会持续发送 A/D 第一帧 hand action，但不会写入新 episode。移动完成后才开始 writer episode。

qpos 数据要求：

- 第一次 `h` 需要源 init episode parquet 中包含 `action.replay.<side>_arm_joint_target`，例如 `action.replay.right_arm_joint_target`。该列记录采集源轨迹时实际发送给 RM65 的 6 维关节目标，用于严格复刻 `T` 的初始关节状态。
- 这个 qpos 要求只针对 `--replay-init-episode` 的第一次 `h`。普通 A/D replay group 如果没有这些列，仍会按 arm pose replay 执行。

固定轨迹执行速度：

- 第二次 `h` 执行 `T` 时，直接使用 `T` 的原始 action Hz，不受 `--replay-init-episode-move-time` 影响。
- A/D 正式 replay 轨迹执行时，也直接使用被抽到 source episode 的原始 action Hz。
- `--replay-max-arm-xyz-step-m` 和 `--replay-max-arm-rot-step-deg` 不会限制这些固定轨迹执行段；它们只保留给非固定轨迹安全调试场景。
- 固定轨迹执行期间，RM65 backend 会临时使用 `--replay-arm-max-angvel`（默认约 `3/3/3/4/4/4 rad/s`）提高跟踪速度，结束后恢复普通 teleop/return-home 的配置。

默认 `--replay-init-episode-require-ready` 会要求完成 `H -> H` 后才能按 A/D 开始正式录制；如果调试时想跳过，可以加 `--no-replay-init-episode-require-ready`。

启动顺序：

1. 启动 sources、rig、writer pool
2. 启动非 rig backend，比如 telemetry-only hand backend
3. 对启用的 arm backend 做一次启动阶段硬件状态同步
4. 等待按键
5. 按 `p` 后执行 runtime home capture，并进入 arm following
6. 按 `z` 后开始写当前 episode 的 samples
7. 按 `c` 后 finalize 当前 episode，写 parquet、meta 和可选视频

需要特别注意：

- collector 启动后不会自动标定，也不会自动录制。
- 在按 `p` 后的 home capture 阶段，操作者需要把启用 side 的 wrist tracker 保持在期望 home pose 附近。
- 按 `h` 会清除当前 home，机械臂停止随动；此后必须再次按 `p` 才会恢复随动。
- 按 `z` 后只有按 `c` 才会保存 episode；按 `d` 会丢弃当前 episode。
- `run_collect.py` 只在启动同步和 `p` 标定阶段读取真实 arm 当前状态；进入随动/采集主循环后，不再持续调用 `refresh_state_from_hardware()`。

## 7. 当前数据语义

### 7.1 arm state 语义

当前 arm observation 语义已经固定为：

- `arm_state_source = "last_commanded_action"`

也就是：

- 采集主循环中的 arm `current_qpos/current_tcp_pose` 以当前 backend 的本地状态为准
- 本地状态由启动同步和后续已发送 command 维护
- 不在每一步重新读真机硬件状态

这正是当前实现里为了降低真机 IO 压力而采用的口径。

### 7.2 action / observation.state 维度

当前主低维字段是：

- `action`: 26 维
- `observation.state`: 38 维
- `observation.last_action`: 26 维

具体展开为：

- `action = left_arm_pose(7) + right_arm_pose(7) + left_hand_q6(6) + right_hand_q6(6)`
- `observation.state = left_arm_qpos(6) + left_arm_pose(7) + left_hand_qpos(6) + right_arm_qpos(6) + right_arm_pose(7) + right_hand_qpos(6)`

其中：

- arm pose 使用 `xyz + xyzw quaternion`
- arm action 的 pose frame 是 robot base frame
- 未启用 side 的槽位按空值写出

### 7.3 相机语义

当前相机策略是：

- 视频帧频率通常高于控制频率
- 每条 sample 只保留与该 sample 时间最接近的一帧
- 不额外保留其他冗余帧

如果启用相机：

- row 中会写 `observation.images.<camera_key>.*`
- 真正的视频帧按 sample 顺序写入对应 mp4

## 8. 关键参数

最常用的参数是：

- `--dataset-root`
  基础输出目录；实际数据集目录会自动命名为 `<dataset-root>/<YYYY-MM-DD>_<task-name>`。未显式传入时读取 `local_paths.json` 的 `dataset_root`，再回退到仓库内 `datasets/`。
- `--task-name`
  任务名称，用于数据集目录命名和 episode metadata。旧的 `--task` 仍兼容，会在未提供 `--task-name` 时作为任务名和 task id。
- `--active-sides left|right|both`
  启用哪些 arm side。
- `--load-profile`
  加载已有 `TeleopCalibration` JSON。加载后仍需按 `p` 做本次 session 的起始位置标定。
- `--left-arm-ip --right-arm-ip`
  启用 side 对应的 RM65 IP 必填。
- `--left-hand-port --right-hand-port`
  q6 serial 输入端口，可选。
- `--left-hand-ip --right-hand-ip`
  InspireTac IP，可选。
- `--no-camera`
  完全禁用相机链路。
- `--camera-config`
  RealSense 配置 JSON，默认读取 `teleop_runtime/configs/devices/realsense.json`。
- `--vis_cam`
  采集时弹出 RealSense RGB 预览窗口；默认预览 `front`，没有 `front` 时预览配置里的第一路。如果当前 OpenCV/显示环境不可用，会在终端打印原因并继续采集。
- `--vis-cam-key`
  指定 `--vis_cam` 要预览的 camera key，例如 `front`、`wrist`。
- `--no-video`
  禁止视频导出。当前如果同时保留相机但关闭视频，只会保留相机时间相关 metadata，不会保存图像内容。
- `--no-parquet`
  禁止 parquet 导出，通常只在调 writer 行为时使用。
- `--calibration-timeout`
  runtime home capture 超时时间。
- `--calibration-samples`
  每个启用 side 要累计多少帧有效 wrist pose。
- `--max-steps`
  最多运行多少个随动控制 step。它不包括按 `p` 之前的等待时间。首次联调建议设置成小值。
- `--status-sec`
  状态打印周期。

## 9. 输出目录结构

一个典型输出目录大致如下：

```text
dataset_root/
  data/
    chunk-000/
      episode_000000.parquet
  meta/
    info.json
    tasks.jsonl
    episodes.jsonl
    episodes_stats.jsonl
    calibration/
      teleop_profile.json
    devices/
      realsense.json
  videos/
    observation.images.front/
      chunk-000/
        episode_000000.mp4
```

说明：

- `meta/info.json` 保存 dataset 级别信息与 feature 维度。
- `meta/episodes.jsonl` 保存每个 episode 的 parquet 路径、video 路径、metadata、summary。
- `meta/episodes_stats.jsonl` 保存每个 episode 的 frame 数、时长、平均频率等统计。
- `meta/calibration/teleop_profile.json` 保存本次采集实际使用的 calibration 快照。
- `meta/devices/realsense.json` 只在启用相机时写出。
- `data/chunk-xxx/episode_xxxxxx.parquet` 是低维主表。
- `videos/observation.images.<camera_key>/...` 是每路 RGB 的 mp4。

## 10. SAPIEN 数据集回放

采集完成后，可以用 [replay_dataset_sapien.py](replay_dataset_sapien.py) 读取 dataset root + episode，并在 SAPIEN 中按 parquet 里的关节序列复刻动作，同时逐帧打印左右手 tactile sum。

推荐命令：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root datasets/demo_bimanual_no_camera \
  --episode 1
```

如果该 episode 含相机视频，可以同步打开采集视频窗口：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root /path/to/dataset_root \
  --episode 0 \
  --video-mode window \
  --video-camera-key front
```

如果只想检查 parquet 和触觉统计，不启动 SAPIEN：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root datasets/demo_bimanual_no_camera \
  --episode 1 \
  --skip-renderer \
  --no-sleep
```

常用参数：

- `--arm-source action|observation`
  默认 `action`，使用 `action.replay.<side>_arm_joint_target` 复刻当时发出的 replay target；缺失时回退到 observation qpos。
- `--hand-source action|observation|teleop|open`
  默认 `action`，使用 `action.replay.<side>_hand_joint_target` 渲染手；缺失时回退到 observation hand qpos。`teleop` 会读取 `teleop.action_source.<side>_hand_q6`，并按当前 collector 默认 invert 语义转成 InspireTac command convention。
- `--print-every N`
  每 N 帧打印一次 tactile sum，默认每帧打印。
- `--max-frames N`
  只回放前 N 帧，`0` 表示完整 episode。
- `--headless`
  不打开 SAPIEN viewer window，但仍创建 renderer。
- `--skip-renderer`
  完全不创建 SAPIEN，仅用于轻量数据检查。
- `--video-mode auto|window|off`
  控制采集视频回放。`auto` 会在数据集有视频且有显示环境时打开 OpenCV 窗口，`window` 强制打开，`off` 禁用。
- `--video-camera-key front`
  只回放指定 camera key；不传则回放该 episode 中所有视频。

打印格式示例：

```text
[Tactile] replay_i=000000 frame=000000 t=0.000s left_sum=0.022 right_sum=0.242 total=0.264
```

脚本结束时还会打印当前回放范围内的累计 tactile sum。

## 11. 推荐操作流程

当前建议按下面顺序做：

1. 先启动 VIVE sender，确认左右 wrist tracker 在发包。
2. 先把双臂摆到安全起始姿态，再确认左右 q6 和左右 hand 网络都正常。
3. 首次联调先加 `--no-camera --no-video --max-steps 30`。
4. collector 启动后，先把 wrist tracker 放在期望 home pose，按 `p` 做起始位置标定。
5. 看到状态行 `home=ready` 且机械臂随动正常后，先不录制，做几秒空跑确认动作安全。
6. 按 `z` 开始录制一条 episode。
7. 动作完成且质量合格时按 `c` 保存 episode。
8. 如果动作失败，按 `d` 丢弃当前 episode。
9. 如果要重新摆起始位或暂停随动，按 `h` 锁定机械臂并清除 home；再次按 `p` 才会重新标定并恢复随动。
10. 结束后检查 `meta/episodes.jsonl`、`meta/episodes_stats.jsonl` 和 `data/chunk-*/episode_*.parquet`。
11. 无相机链路稳定后，再打开 RealSense。

## 12. 常见问题

### 12.1 按 p 后卡在 calibration

通常是下面几类问题：

- VIVE sender 没有真正发到 `--host/--port`
- `tracker_name` 或 `tracker_role` 不匹配
- 某一侧 wrist tracker 数据持续 invalid

优先检查：

- `left_wrist_pose`
- `right_wrist_pose`
- `tracker_role`
- `frame=vive_world`

如果还没有按 `p`，collector 等待按键是正常状态，不代表卡死。

### 12.2 终端按键没有反应

collector 启动后会打印按键输入来源：

- `keyboard input active source=stdin`：直接读取当前终端。
- `keyboard input active source=/dev/tty`：当前 `stdin` 不是 TTY，已回退读取控制终端。
- `keyboard input is disabled`：当前进程没有可读交互终端；需要从真实 terminal 启动，或调整 IDE/launch 配置让程序获得 TTY。

如果开启了 `--vis_cam`，焦点在 OpenCV 预览窗口时也可以直接按 `p/z/c/d/h/q`；焦点在终端时同样可用。

### 12.3 `--vis_cam` 没有窗口

`--vis_cam` 依赖当前 Python 环境的 OpenCV HighGUI 和桌面显示。启动时如果看到 `preview is disabled`，优先检查：

- 当前 shell/IDE 是否能连接 `DISPLAY` 或 `WAYLAND_DISPLAY`
- `cv2` 是否是可用的 GUI 构建
- RealSense 是否已经开始产帧，状态行里的 `camera=front=frames:<N>` 是否增长

即使预览窗口禁用，只要相机正常产帧，视频采集仍会继续写入 dataset。

### 12.4 只想采 arm，不想采手

直接不要传任何 `--left-hand-port/--right-hand-port/--left-hand-ip/--right-hand-ip` 即可。

### 12.5 只想采 q6，不接真实 InspireTac

只传 `--left-hand-port/--right-hand-port`，不要传 hand IP。

### 12.6 为什么 arm observation 不是实时真机反馈

这是当前实现的有意选择。原因是：

- 视频和控制循环通常都比真机状态刷新更频繁
- 持续读取真机状态会明显增加 IO 和不稳定性
- 当前 collector 已固定按 `last_commanded_action` 维护 arm state

如果后续要引入独立的真机状态回读语义，需要作为新的采集模式单独设计，而不是偷偷改掉当前 collector 的默认口径。

## 13. 当前实现与测试流程

### 13.1 目前已经实现

截至当前仓库代码，collector 已经具备下面这些能力：

- [run_collect.py](run_collect.py)
  已能直接组装 `VivePoseSource / HandQ6Source / RM65ArmBackend / InspireTacHandBackend / TeleopSession / CollectionRuntime / LeRobotV21Writer`
  并支持按键式 `p/z/c/d/h/q` 采集状态机
- [sample_assembler.py](sample_assembler.py)
  已按固定双臂 schema 生成 sample，并把 arm state 固定为 `last_commanded_action`
- [schema_v21.py](schema_v21.py)
  已把 sample 转成 LeRobot 风格 row
- [writer_v21.py](writer_v21.py)
  已支持 async row/video/finalize job、episode parquet、meta 文件、dataset 续写，以及 active episode discard
- [realsense_rgb.py](../sources/camera/realsense_rgb.py)
  已支持 RealSense RGB 后台采集与 nearest-frame buffer
- [realsense.json](../configs/devices/realsense.json)
  已提供默认单相机配置
- [replay_dataset_sapien.py](replay_dataset_sapien.py)
  已支持读取 LeRobot 风格采集 dataset，并用 SAPIEN 复刻双臂双手动作，同时打印触觉总和

当前还没有完全闭环验证的部分是：

- 双臂双手带相机的长时间真机 smoke test
- collector 与旧 `EpisodeRecorder` sidecar 的联动

### 13.2 已完成的轻量验证

当前代码已经做过这些轻量检查：

1. `python -m py_compile teleop_runtime/session.py teleop_runtime/collection/sample_assembler.py teleop_runtime/collection/writer_v21.py teleop_runtime/collection/run_collect.py`
2. `python -m teleop_runtime.collection.run_collect --help`
3. 静态 `build_collect_app(...)` 组装检查：
   - 双臂双手 `--no-camera`
   - 单臂 arm only
   - 单臂 arm + telemetry-only hand backend
4. writer 续写检查：
   - 已验证不会重新从 `episode_000000` 覆盖
5. writer discard 检查：
   - 已验证 active episode 丢弃后，下一次 `start_episode()` 会复用未保存的 episode index
6. dataset replay 检查：
   - 已验证 `replay_dataset_sapien.py --skip-renderer` 可以读取 `demo_bimanual_no_camera` 的 parquet 并打印触觉总和

这些检查只覆盖：

- 语法
- import
- CLI 参数
- 组件装配
- writer 元数据逻辑

不覆盖：

- 长时间真机稳定性
- 大量视频编码压力
- 网络/串口抖动下的极限表现

### 13.3 建议的详尽测试流程

如果要从零开始系统验证当前 collector，建议按下面顺序做，不要一开始就上双臂双手带相机长时间采集。

#### A. 静态检查

先在 `grasp_py310_copy` 环境里跑：

```bash
python -m py_compile \
  teleop_runtime/session.py \
  teleop_runtime/collection/sample_assembler.py \
  teleop_runtime/collection/writer_v21.py \
  teleop_runtime/collection/run_collect.py
python -m teleop_runtime.collection.run_collect --help
```

#### B. 组件装配检查

不连真机主循环，只先确认参数能完成 collector 组装：

```bash
python - <<'PY'
import sys
from teleop_runtime.collection.run_collect import parse_args, build_collect_app

sys.argv = [
    "prog",
    "--dataset-root", ".tmp_collect_probe_readme",
    "--active-sides", "both",
    "--left-arm-ip", "192.168.101.21",
    "--right-arm-ip", "192.168.1.20",
    "--left-hand-port", "/dev/ttyUSB0",
    "--right-hand-port", "/dev/ttyUSB1",
    "--no-camera",
    "--no-video",
]
args = parse_args()
app = build_collect_app(args)
print(sorted(app.components.backends.keys()))
app.runtime.close()
PY
```

#### C. 无相机、双臂 arm only 真机短测

第一条真机链路建议先只带 arm：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /tmp/collect_arm_only_short \
  --active-sides both \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --no-camera \
  --no-video \
  --max-steps 30 \
  --calibration-timeout 15 \
  --calibration-samples 20
```

检查点：

- 启动后处于等待按键状态，未按 `p` 前不会随动，也不会写 sample
- 按 `p` 后能完成 runtime home capture，并进入随动
- 按 `z` 后开始录制，状态行显示 `recording=on`
- 按 `c` 后保存 episode，生成 parquet 和 meta 文件

#### D. 无相机、双臂双手真机短测

arm only 通过后，再上双手：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /tmp/collect_bimanual_bihand_short \
  --active-sides both \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --no-camera \
  --no-video \
  --max-steps 30
```

检查点：

- 按 `p -> z -> c` 后能保存 episode
- 按 `p -> z -> d` 后能丢弃当前 episode，且不会写入 `episodes.jsonl`
- parquet 中 `teleop.action_source.left_hand_q6/right_hand_q6` 非空
- hand telemetry 存在时，`observation.hand.<side>.joint_timestamp/telemetry_timestamp` 非空
- 如果启用了 tactile，`observation.hand.<side>.tactile` 有值

#### E. 数据检查

短测结束后，至少检查这些文件：

```bash
find /tmp/collect_bimanual_bihand_short -maxdepth 3 -type f | sort
```

重点看：

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `meta/calibration/teleop_profile.json`
- `data/chunk-000/episode_000000.parquet`

#### F. 加相机短测

无相机链路稳定后，再启用相机：

```bash
python -m teleop_runtime.collection.run_collect \
  --dataset-root /tmp/collect_bimanual_with_camera_short \
  --active-sides both \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --vis_cam \
  --vis-cam-key front \
  --max-steps 30
```

检查点：

- 按 `p -> z -> c` 完成保存
- `meta/devices/realsense.json` 已写出
- `videos/observation.images.front/.../episode_000000.mp4` 存在
- parquet 中 `observation.images.front.source_timestamp/age_sec/video_frame_index` 已写出

#### G. 压力与长时间测试

最后才做长时间测试：

1. 把 `--max-steps` 去掉
2. 先跑无相机长测
3. 再跑带相机长测
4. 同时观察 CPU、内存、磁盘写入和 ffmpeg/opencv 编码负载

当前机器如果已经出现过 smoke test 期间卡死，建议：

- 先保持 `writer-workers=2`
- 先跑 `--no-camera`
- 每次只扩大一个变量，不要同时增加相机、时长和 tactile 负载
