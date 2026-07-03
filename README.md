# Dex Inspire Teleop 数据采集 Pipeline

本文档是当前仓库内从设备连接、虚拟验证、真机 teleop 验证、带三路相机采集，到 SAPIEN+视频回放的完整流程。

默认假设在采集电脑上执行，工作目录固定为：

```bash
cd /home/zc/HandMotion/Developing/DexTeleop
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310
```

## 00. 机械臂末端重力标定和TCP标定见 hardware/rm65/utils/README.md
## 01. 遥操作开始时用 hardware/rm65/utils/init_arm/README.md 让双臂归位，起始状态手部自然贴合机械臂姿态双手与肩同宽

```bash
python hardware/rm65/utils/init_arm/rm65_move_to_standard_pose.py \
  --arm both \
  --pose-json hardware/rm65/utils/init_arm/standard_pose.json \
  --execute \
  --confirm INIT_RM65_STANDARD_POSE
```

## 02. Q6 控制双臂 1D 夹爪

Q6 夹爪控制已接入 `teleop_runtime.debug.run_vive_rm65_real_teleop` 和 `teleop_runtime.collection.run_collect`。先保存 Q6 左右手配置，再保存 gripper / DM TAC USB 配置。

采集命令中追加：

```bash
--hand-backend dmtac-gripper \
--dmtac-config latest \
--gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json
```

如果没有 DM TAC 触觉，只接 1D gripper，追加 `--dmtac-no-tactile`。夹爪默认读取 Q6 `source_dim=2`，把 Q6 的 `0.1..0.9` 映射到夹爪 `0..0.99`，并写到 hand action `output_dim=1`；可用 `--gripper-source-dim` / `--gripper-output-dim` 调整。

### 03. USB 权限修复和 GUI 启动预检

USB 设备重插后，先用 sudo 运行统一修复脚本。它会先检查/处理 brltty 对 CH340/CP210x USB-Serial 的抢占，再处理当前 USB 串口、`/dev/video*`、`/dev/media*` 和 `/dev/bus/usb/*/*` 的权限，把这些设备节点临时设为 `666`；不检查 Q6/DMTAC/RealSense 数量，所以不会因为某个设备暂时没插而失败。

```bash
sudo -E "$CONDA_PREFIX/bin/python" -m teleop_runtime.configure_devices --fix-serial-conflicts --fix-permissions
```

如果之前是 brltty/USB-Serial 抢占导致串口节点本身不出现，执行后重新拔插设备；新出现节点若仍不可读写，再重复上面这条命令。

### 03a. DMTAC 触觉 USB HUB 分布检查

开启 DMTAC 触觉后如果 teleop / 数采明显卡顿，先确认触觉相机是否真的分布在不同 USB root controller 上。左右各接一个外置 HUB 不等于已经分到不同 root bus；如果两个 HUB 仍挂在同一个 `480M` root bus 下，四个 DMTAC UVC 相机会和同 root 上的其他 USB2 视频设备竞争带宽/中断。

快速检查：

```bash
python hardware/DMTAC_gripper/check_usb_hub_distribution.py
```

脚本只读取 `/sys/bus/usb/devices` 和 DMTAC latest assignment，不启动 DMTAC SDK，也不会打开触觉相机。输出重点看最后的 `Verdict`：

- `OK tactile sensors are split across USB root buses.`：触觉相机已经分散到不同 root bus，USB 拓扑基本合理。
- `WARN All 4 tactile sensors share root usbX`：左右 HUB 仍在同一个 USB root bus 下；建议把其中一个 HUB 移到另一个主板 root controller，或接到独立 PCIe USB 卡。
- `WARN USB video neighbor ... is also on usbX at 480M`：同一个 DMTAC root bus 上还有 USB2 视频设备，例如 D405；建议把该设备移到 USB3 root bus。

需要机器可读输出时：

```bash
python hardware/DMTAC_gripper/check_usb_hub_distribution.py --json
```

## 0. 本地路径配置

机器相关路径集中放在仓库根目录的 `local_paths.json`，该文件已被 `.gitignore` 忽略。没有该文件时，代码默认使用仓库内的 `datasets/` 和 `teleop_records/`。

```bash
cp local_paths.example.json local_paths.json
```

`local_paths.json` 支持相对仓库根目录的路径，也支持绝对路径。常用键：

```json
{
  "dataset_root": "datasets",
  "teleop_records_root": "teleop_records",
  "real_robot_dataset_root": "../real_robot_dataset/datasets",
  "concat_video_input_dir": "../real_robot_dataset/datasets/2026-04-27_chips_changed/videos/observation.images.front/chunk-000"
}
```

## 2. 固定设备配置

当前 README 中使用的设备配置如下：

| 设备 | 参数 |
| --- | --- |
| 左 RM65 | `192.168.101.21` |
| 右 RM65 | `192.168.1.20` |
| 左 InspireTac | `192.168.12.210` |
| 右 InspireTac | `192.168.11.210` |
| 左 q6 串口 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00` |
| 右 q6 串口 | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00` |
| RealSense RGB | `front=213222079798`，`left_wrist=261122273924`，`right_wrist=260522278055` |
| VIVE UDP | receiver 监听 `0.0.0.0:9999`，sender 目标使用采集电脑 IP，例如 `192.168.20.90` |

推荐先设置这些变量，后面命令直接复用：

```bash
export REPO=$(pwd)
export PROFILE=$REPO/teleop_records/calibration_profiles/your_profile.json
export CAMERA_CONFIG=$REPO/teleop_runtime/configs/devices/realsense.json
export DATASET_ROOT=$REPO/datasets
export VIVE_TARGET_HOST=192.168.20.90

export LEFT_ARM_IP=192.168.101.21
export RIGHT_ARM_IP=192.168.1.20
export LEFT_HAND_IP=192.168.12.210
export RIGHT_HAND_IP=192.168.11.210
export LEFT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00
export RIGHT_HAND_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00
```

RM65 arm IP 也会像 Q6 / DMTAC 一样保存 latest：第一次启动时显式传
`--left-arm-ip` / `--right-arm-ip`，程序会自动更新：

```text
teleop_records/rm65_arm_ip_configs/latest.json
```

后续同一台设备可以省略 arm IP 参数，默认从 latest 读取。需要临时禁用时使用
`--no-arm-ip-default-config`；需要显式传 IP 但不更新 latest 时使用
`--no-arm-ip-save-latest`。

## 2a. 频率语义：数采 15Hz，RM65 控制默认 40Hz

这里有两层频率，不能混在一起调：

- `--hz`：外层 action / 数据采样频率。采集时决定图像、状态、统一 action 空间等写盘节拍；单臂和双臂默认都是 `15Hz`。
- `--nominal-fps`：数据集 metadata 的 fps。采集入口不传时默认跟随 `--hz`，所以默认也是 `15Hz`。
- 部署入口优先用 `--action-hz`；如果没有显式传，也没有从 policy/参考数据集推断到频率，就回退到 `--hz=15`。
- `--arm-control-hz`：RM65 backend 收到外层 action 后的机械臂控制/streaming 频率，默认 `40Hz`。这和采集/部署 action 频率无关，正常不要为了 15Hz 数采去降低它。

因此默认行为是：数据和 policy action 以 `15Hz` 对齐，RM65 真机执行层仍按 `40Hz` 持续跟随最新 action。

```bash
cd "$REPO"

python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --load-profile "$PROFILE" \
  --hz 15 \
  --arm-max-angvel 1.2,1.2,1.2,1.8,1.8,1.8
```

### 遥操作手 zero pose 固定标定

q6 遥操作手默认读取固定 zero pose 标定，不再每次启动都要求保持 zero pose。固定标定文件是：

```text
hardware/teleop_hand/exo_hand_zero_pose.json
```

当前文件由 18 次 zero pose 记录汇总得到，稳定性摘要：

| 侧别 | 记录数 | 最大 span |
| --- | ---: | ---: |
| left | 9 | `0.0315 deg` |
| right | 9 | `0.2700 deg` |

正常 teleop 和采集命令不需要额外参数。如果更换外骨骼、重新装配手指连杆、或希望更新固定值，再显式加 `--exo-hand-calibration`。该模式会重新做 zero pose 稳定检查，保存本次结果，并更新上面的固定标定 JSON。

最简重新标定命令：

```bash
python hardware/teleop_hand/check_zero_pose_calibration.py \
  --left-port "$LEFT_HAND_PORT" \
  --right-port "$RIGHT_HAND_PORT" \
  --rounds 1 \
  --exo-hand-calibration
```

如果只想多次检查稳定性而不更新固定值，去掉 `--exo-hand-calibration`，并把 `--rounds` 调大即可。

## 3. 三路摄像头配置

默认 RealSense 配置文件已经固定到：

```text
teleop_runtime/configs/devices/realsense.json
```

当前内容是三路 `front`、`left_wrist`、`right_wrist`：

```json
{
  "front": {
    "serial": "213222079798",
    "width": 640,
    "height": 480,
    "fps": 30,
    "metadata": {
      "mount": "single_front"
    }
  },
  "left_wrist": {
    "serial": "261122273924",
    "width": 640,
    "height": 480,
    "fps": 30,
    "metadata": {
      "mount": "single_left_wrist"
    }
  },
  "right_wrist": {
    "serial": "260522278055",
    "width": 640,
    "height": 480,
    "fps": 30,
    "metadata": {
      "mount": "single_right_wrist"
    }
  }
}
```

采集时不要传 `--no-camera`，并保留默认 `--write-video`，writer 会输出：

```text
videos/observation.images.front/chunk-000/episode_000000.mp4
videos/observation.images.left_wrist/chunk-000/episode_000000.mp4
videos/observation.images.right_wrist/chunk-000/episode_000000.mp4
```

如果需要采集时同时查看实时 RealSense RGB 画面，给采集命令追加 `--vis_cam`。预览窗口默认显示所有已配置相机；需要只看一路时追加 `--vis-cam-key front`、`--vis-cam-key left_wrist` 或 `--vis-cam-key right_wrist`。`--vis_cam` 不能和 `--no-camera` 同时使用；传 `--no-camera` 时程序从启动阶段就不会连接任何 RealSense，也不会写出视频。

如果相机启动失败，先确认系统能看到三路 serial：

```bash
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
for dev in ctx.query_devices():
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

## 4. 启动 VIVE Sender

在 VIVE 机器或同一台采集电脑的独立终端(Windows)启动 sender：

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310

python hardware/VIVE/VIVE_sender/data_sender_new.py \
  --host "$VIVE_TARGET_HOST" \
  --port 9999 \
  --hz 60 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist
```

推荐使用 `hardware/VIVE/VIVE_sender/data_sender_new.py`：它每帧只取一次 OpenVR pose snapshot，再复用到所有 tracker，并会打印 loop/get_pose/json/send 耗时。第三只 `auxiliary` tracker 需要时再显式追加。不要用 `hardware/VIVE/VIVE_sender/data_sender.py` 跑当前 teleop；它近似满速发 UDP，容易让 receiver 忙于清包。`data_time_sender.py` 发送的是 `double timestamp + 36 float` 到默认 `6666`，不是当前主链路协议。

如果 sender 和 receiver 在同一台机器上临时自测，可以把 `VIVE_TARGET_HOST` 改成 `127.0.0.1`。



## 5. Virtual SAPIEN 验证

这一步只验证 VIVE profile、左右 wrist 映射、q6 手部映射和双臂 SAPIEN preview，不向真机发送运动。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310

python -m teleop_runtime.debug.run_vive_sapien_virtual \
  --host 0.0.0.0 \
  --port 9999 \
  --arm-mode real \
  --control-mode absolute \
  --load-profile "$PROFILE" \
  --left-hand-port "$LEFT_HAND_PORT" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --calibration-timeout 15 \
  --calibration-samples 20
```

检查点：

- 启动后按 `p` 做本次 session home capture。
- 左右 wrist tracker 映射正确，左手控制左臂，右手控制右臂。
- 桌面坐标方向正确，前/左/上不会反。
- q6 控制的 SAPIEN 手部开合方向正确。
- 如果渲染设备不可用，先加 `--skip-renderer` 验证主链路日志。

## 6. Real Teleop 真机短测

这一步接入双臂双手真机，但还不采集 LeRobot dataset。先做短时 teleop 验证，确认随动、触觉读取和频率稳定。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310

python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --load-profile "$PROFILE" \
  --left-hand-port "$LEFT_HAND_PORT" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-telemetry-hz 10 \
  --hand-tactile-regions thumb_1,index_1,middle_1,ring_1,little_1,palm \
  --hz 15 \
  --status-sec 1 \
  --calibration-timeout 15 \
  --calibration-samples 20 \
  --visualize-tactile \
  --skip-renderer
```

检查点：

- 默认使用固定 q6 zero pose 标定；只有需要更新固定值时才追加 `--exo-hand-calibration`。
- 按 `p` 前双臂必须在安全起始位。
- `p` 后才进入本次 session 的 home 对齐和随动。
- 先小范围移动 wrist tracker，确认机械臂方向和速度安全。
- 手部 q6 能下发到 InspireTac。
- 触觉读取不要一开始就全 17 区域高频压力测试；先用上面少量区域和 `10Hz`。

## 7. GUI 正式采集数据

确认上面两步稳定后，再启动正式采集。关键点是使用 `run_vive_rm65_real_teleop --gui`，不要传 `--no-camera`，并显式传入三路相机配置。采集写盘频率默认 `15Hz`；机械臂真机控制流仍按 `--arm-control-hz` 默认 `40Hz` 执行，不被采集频率限制。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310

python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --gui \
  --dataset-root "$DATASET_ROOT" \
  --collection-task-name dmtac_bimanual_demo \
  --collection-task-description "bimanual DMTAC gripper teleop with three RealSense cameras" \
  --active-sides both \
  --load-profile "$PROFILE" \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --gripper-task-profile teleop_runtime/configs/gripper_task_profile.example.json \
  --dmtac-with-tactile \
  --camera-config "$CAMERA_CONFIG" \
  --write-video \
  --hz 15 \
  --collection-warning-hz 15 \
  --arm-control-hz 40 \
  --calibration-timeout 15 \
  --calibration-samples 20 \
  --status-sec 1 \
  --visualize-tactile \
  --vis_cam \
  --storage-queue-size 64
```

GUI 会显示实际数采频率；如果正式记录样本的滚动频率低于 `--collection-warning-hz`，数采面板会出现 `DATA WARNING`。单臂采集时把 `--active-sides both` 改成 `left` 或 `right`，并只保留对应侧的 arm IP 和 tool profile。相机仍可按三路一起录；如果只想预览一路，追加 `--vis-cam-key front` 等，不影响其他相机写盘。

首次带相机短测可以临时追加：

```bash
--max-steps 300
```

如果当前终端没有显示环境，删除 `--vis_cam`；这不会影响视频写盘。需要完全不占用相机时才使用 `--no-camera`，此时也不会写出 RealSense 视频。

采集按键流程：

| 按键 | 行为 |
| --- | --- |
| `p` | 起始位置标定，进入随动 |
| GUI“开始数采” / `z` | 开始录制当前 episode |
| GUI“停止数采” / `c` | 保存当前 episode；后台 finalize，GUI 会显示正在收尾 |
| `d` | 丢弃当前 episode |
| `h` | 锁定机械臂并清除本次 `p` 标定 |
| `q` | 退出；若正在录制会先保存 |

采集后检查：

```bash
find "$DATASET_ROOT" -maxdepth 5 -type f | sort
```

至少应看到：

```text
meta/info.json
meta/episodes.jsonl
meta/episodes_stats.jsonl
meta/calibration/teleop_profile.json
meta/devices/realsense.json
data/chunk-000/episode_000000.parquet
videos/observation.images.front/chunk-000/episode_000000.mp4
videos/observation.images.left_wrist/chunk-000/episode_000000.mp4
videos/observation.images.right_wrist/chunk-000/episode_000000.mp4
tactile/observation.hand.left.tactile/chunk-000/episode_000000/manifest.json
tactile/observation.hand.right.tactile/chunk-000/episode_000000/manifest.json
```

## 8. 回放采集数据

SAPIEN 复刻动作、同步播放采集视频，并逐帧打印左右手 tactile sum：

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dexteleop310

python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --video-mode window \
  --video-camera-key front
```

如果只想看采集视频和触觉，不启动 SAPIEN：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --skip-renderer \
  --video-mode window \
  --video-camera-key front
```

如果当前机器没有显示环境，只检查 parquet 和触觉：

```bash
python -m teleop_runtime.collection.replay_dataset_sapien \
  --dataset-root "$DATASET_ROOT" \
  --episode 0 \
  --skip-renderer \
  --video-mode off \
  --max-frames 20 \
  --no-sleep
```

回放时默认使用：

- arm：`action.replay.<side>_arm_joint_target`
- hand：`action.replay.<side>_hand_joint_target`
- tactile：`observation.hand.<side>.tactile`
- video：`meta/episodes.jsonl` 中的 `video_paths`，或 parquet 中各路 `observation.images.<camera_key>.path`

Frame 兼容约定：

- 新采集、部署和 policy pose action 的正式 arm frame 是 `tool_tcp`。
- 旧 profile 中显式写入的 `hand_base` 仍可加载，用于调试或历史兼容。
- 旧 parquet/row 如果没有 `canonical_action_record` / `arm_action_specs_by_side`，回放入口会按历史
  `arm_endpoint` pose contract 解释；带 canonical metadata 的数据始终按 metadata 解释。

每帧日志示例：

```text
[Tactile] replay_i=000000 frame=000000 t=0.000s left_sum=0.022 right_sum=0.242 total=0.264
```

## 9. 常见阻塞点

| 现象 | 优先检查 |
| --- | --- |
| 按 `p` 超时 | VIVE sender 是否在发 `tracker_1:left_wrist` 和 `tracker_2:right_wrist`，receiver `--host/--port` 是否匹配 |
| 左右臂/VIVE tracker 输入反了 | tracker 名称、`--left-tracker-name`、`--right-tracker-name`、profile 是否匹配；只需在开始前交换 VIVE tracker 时用 `--swap-operator-sides` |
| 机械臂方向不对 | 先回到 virtual SAPIEN 验证 profile，不要直接在真机上调 |
| 相机启动失败 | `realsense.json` 的三路 serial 是否正确，`pyrealsense2` 是否能枚举设备 |
| 没有 mp4 | 是否误传 `--no-camera` 或 `--no-video`，`meta/episodes.jsonl` 的 `video_paths` 是否为空 |
| q6 手部开合零位不对 | 先检查是否更换过外骨骼或连杆；需要更新固定值时运行 `--exo-hand-calibration` |
| `--vis_cam` 没窗口 | 当前终端是否有 `DISPLAY` 或 `WAYLAND_DISPLAY`；无显示环境时删除该参数 |
| 回放没有视频窗口 | 使用 `--video-mode window`，并确认当前终端有 `DISPLAY` 或 `WAYLAND_DISPLAY` |
| 采集频率下降 | 降低 tactile 区域/频率，先用 `--hand-telemetry-hz 10`，相机长测前先做短测 |

## 10. 当前实现状态

已接入：

- 三路 RealSense RGB 配置：`front`、`left_wrist`、`right_wrist`
- q6 遥操作手固定 zero pose 标定，支持 `--exo-hand-calibration` 更新固定值
- 采集时 RealSense 实时预览：`--vis_cam` 默认显示全部相机，`--vis-cam-key` 可只看一路
- `--no-camera` 从启动阶段跳过 RealSense 连接、预览和视频写盘
- GUI 正式采集写出：parquet、meta、calibration、device config、三路 mp4、外置 tactile bin/manifest
- SAPIEN 回放：双臂双手 action replay
- 视频回放：OpenCV window，按 parquet 的 `video_frame_index` 同步
- 触觉统计：逐帧左右手 sum 和 total，结束时累计 sum

未覆盖的高风险测试：

- 长时间带相机真机采集压力测试
- 全 17 tactile 区域高频读取压力测试
- 多相机同步采集
