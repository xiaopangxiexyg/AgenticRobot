# Dex Inspire Teleop 数据采集 Pipeline

本文档是当前仓库内从设备连接、虚拟验证、真机 teleop 验证、带单相机采集，到 SAPIEN+视频回放的完整流程。

默认假设在采集电脑上执行，工作目录固定为：

```bash
cd /path/to/dex_inspire_teleop_new
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy
```

### 00. 机械臂末端重力标定和TCP标定见 hardware/rm65/utils/README.md

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

## 1. 下载解压assets到项目路径: https://disk.pku.edu.cn/link/AA669C31FE27A74A01A2A3BADD61ADF559
文件名：assets.zip
有效期限：2026-06-08 14:10
提取码：YIlq


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
| RealSense RGB | serial `213222079798`，camera key `front` |
| VIVE UDP | receiver 监听 `0.0.0.0:9999`，sender 目标使用采集电脑 IP，例如 `192.168.20.90` |

推荐先设置这些变量，后面命令直接复用：

```bash
export REPO=$(pwd)
export PROFILE=/home/ai/Desktop/DexTeleop/teleop_records/calibration_profiles/20260506_122501_vive_rm65_virtual_desk.json
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

```bash
cd /home/ai/zc/DexTeleop

python -m teleop_runtime.debug.run_vive_rm65_real_teleop \
  --active-sides right \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --load-profile "$PROFILE" \
  --hz 50 \
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

## 3. 单摄像头配置

默认单相机配置文件已经固定到：

```text
teleop_runtime/configs/devices/realsense.json
```

当前内容是单路 `front`：

```json
{
  "front": {
    "serial": "213222079798",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "metadata": {
      "mount": "single_front"
    }
  }
}
```

采集时不要传 `--no-camera`，并保留默认 `--write-video`，writer 会输出：

```text
videos/observation.images.front/chunk-000/episode_000000.mp4
```

如果需要采集时同时查看实时 RealSense RGB 画面，给采集命令追加 `--vis_cam`。预览窗口使用实际采集的帧和分辨率，例如当前 `front` 是 `1280x720@30`。`--vis_cam` 不能和 `--no-camera` 同时使用；传 `--no-camera` 时程序从启动阶段就不会连接任何 RealSense。

如果相机启动失败，先确认系统能看到该 serial：

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
conda activate grasp_py310_copy

python hardware/VIVE/mocap/vive_data_sender.py \
  --host "$VIVE_TARGET_HOST" \
  --port 9999 \
  --tracker tracker_1:left_wrist \
  --tracker tracker_2:right_wrist \
  --tracker tracker_3:auxiliary
```

如果 sender 和 receiver 在同一台机器上临时自测，可以把 `VIVE_TARGET_HOST` 改成 `127.0.0.1`。



## 5. Virtual SAPIEN 验证

这一步只验证 VIVE profile、左右 wrist 映射、q6 手部映射和双臂 SAPIEN preview，不向真机发送运动。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

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
conda activate grasp_py310_copy

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
  --hz 30 \
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

## 7. 带单相机采集数据

确认上面两步稳定后，再启动正式采集。关键点是不要传 `--no-camera`，并显式传入单相机配置。

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python -m teleop_runtime.collection \
  --dataset-root "$DATASET_ROOT" \
  --task right_arm_dual_data \
  --task-description "bimanual InspireTac teleop with single RealSense front camera" \
  --active-sides both \
  --load-profile "$PROFILE" \
  --host 0.0.0.0 \
  --port 9999 \
  --left-arm-ip "$LEFT_ARM_IP" \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --left-hand-port "$LEFT_HAND_PORT" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --left-hand-ip "$LEFT_HAND_IP" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-telemetry-hz 10 \
  --hand-tactile-regions all \
  --visualize-tactile \
  --tactile-vis-value-scale 0.02 \
  --camera-config "$CAMERA_CONFIG" \
  --vis_cam \
  --write-video \
  --hz 30 \
  --nominal-fps 30 \
  --calibration-timeout 15 \
  --calibration-samples 20 \
  --status-sec 1 \
  --vis_cam \
  --vis-cam-key front
```

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

python -m teleop_runtime.collection \
  --dataset-root "$DATASET_ROOT" \
  --task right_arm_new_soft_hard \
  --task-description "right InspireTac teleop with single RealSense front camera" \
  --active-sides right \
  --load-profile "$PROFILE" \
  --host 0.0.0.0 \
  --port 9999 \
  --right-arm-ip "$RIGHT_ARM_IP" \
  --right-hand-port "$RIGHT_HAND_PORT" \
  --right-hand-ip "$RIGHT_HAND_IP" \
  --hand-telemetry-hz 10 \
  --hand-tactile-regions all \
  --visualize-tactile \
  --tactile-vis-value-scale 0.02 \
  --camera-config "$CAMERA_CONFIG" \
  --vis_cam \
  --write-video \
  --hz 30 \
  --nominal-fps 30 \
  --calibration-timeout 15 \
  --calibration-samples 20 \
  --status-sec 1 \
  --vis_cam \
  --vis-cam-key front \
  --max-force 500 \
  --return-right-qpos-deg "-25.67,54.27,76.92,-110.70,75.83,139.68" \
  --hand-tactile-regions all

```

首次带相机短测可以临时追加：

```bash
--max-steps 300
```

如果当前终端没有显示环境，删除 `--vis_cam`；这不会影响视频写盘。需要完全不占用相机时才使用 `--no-camera`，此时也不会写出 RealSense 视频。

采集按键流程：

| 按键 | 行为 |
| --- | --- |
| `p` | 起始位置标定，进入随动 |
| `z` | 开始录制当前 episode |
| `c` | 保存当前 episode |
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
```

## 8. 回放采集数据

SAPIEN 复刻动作、同步播放采集视频，并逐帧打印左右手 tactile sum：

```bash
cd "$REPO"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate grasp_py310_copy

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
- video：`meta/episodes.jsonl` 中的 `video_paths` 或 parquet 中的 `observation.images.front.path`

每帧日志示例：

```text
[Tactile] replay_i=000000 frame=000000 t=0.000s left_sum=0.022 right_sum=0.242 total=0.264
```

## 9. 常见阻塞点

| 现象 | 优先检查 |
| --- | --- |
| 按 `p` 超时 | VIVE sender 是否在发 `tracker_1:left_wrist` 和 `tracker_2:right_wrist`，receiver `--host/--port` 是否匹配 |
| 左右臂反了 | tracker 名称、`--left-tracker-name`、`--right-tracker-name`、profile 是否匹配 |
| 机械臂方向不对 | 先回到 virtual SAPIEN 验证 profile，不要直接在真机上调 |
| 相机启动失败 | `realsense.json` serial 是否为 `213222079798`，`pyrealsense2` 是否能枚举设备 |
| 没有 mp4 | 是否误传 `--no-camera` 或 `--no-video`，`meta/episodes.jsonl` 的 `video_paths` 是否为空 |
| q6 手部开合零位不对 | 先检查是否更换过外骨骼或连杆；需要更新固定值时运行 `--exo-hand-calibration` |
| `--vis_cam` 没窗口 | 当前终端是否有 `DISPLAY` 或 `WAYLAND_DISPLAY`；无显示环境时删除该参数 |
| 回放没有视频窗口 | 使用 `--video-mode window`，并确认当前终端有 `DISPLAY` 或 `WAYLAND_DISPLAY` |
| 采集频率下降 | 降低 tactile 区域/频率，先用 `--hand-telemetry-hz 10`，相机长测前先做短测 |

## 10. 当前实现状态

已接入：

- 单 RealSense RGB 配置：`front`，serial `213222079798`
- q6 遥操作手固定 zero pose 标定，支持 `--exo-hand-calibration` 更新固定值
- 采集时 RealSense 实时预览：`--vis_cam`
- `--no-camera` 从启动阶段跳过 RealSense 连接和视频写盘
- 采集写出：parquet、meta、calibration、device config、mp4
- SAPIEN 回放：双臂双手 action replay
- 视频回放：OpenCV window，按 parquet 的 `video_frame_index` 同步
- 触觉统计：逐帧左右手 sum 和 total，结束时累计 sum

未覆盖的高风险测试：

- 长时间带相机真机采集压力测试
- 全 17 tactile 区域高频读取压力测试
- 多相机同步采集




## 11. Being-H-Tac 真机部署

```DexTeleop/teleop_runtime/deploy```里面写的是deploy原始所需的文件和部署脚本。




# 左臂非操作者实验模式

这里放两条复用 `teleop_runtime.collection` 主采集链路的左臂非操作者采集链路：

- `left-arm-fixed`：左臂固定在按 `p` 时读取到的当前位置。
- `left-arm-jitter`：左臂以按 `p` 时的位置为中心，末端做小范围随机平滑抖动，并和右臂一起走双臂 IK。

## 固定左臂行为

- 左臂：按 `p` 时同步真实 RM65 当前关节和 TCP pose，之后每帧发送同一个 `left_arm` joint hold command。
- 右臂：继续使用 `right_wrist_pose -> right_arm` 遥操作。
- 左手/右手：继续使用现有 q6 到 InspireTac 的遥操作和 telemetry/tactile 采集。
- 数据：仍使用双臂 schema，`action` 26 维、`observation.state` 38 维，左臂 qpos、pose、replay joint target 都保留，不会退化成单臂空槽。

固定左臂模式会关闭双臂 joint-space collision-aware retargeter，右臂走独立 `PoseToArmRetargeter`。运行前需要确认固定左臂不会处在右臂工作路径里。

## 固定左臂启动

当前只保留主采集入口，显式传入 `--left-arm-fixed`：

```bash
python -m teleop_runtime.collection \
  --left-arm-fixed \
  --dataset-root /path/to/datasets \
  --task-name grasp_demo_left_arm_fixed \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --right-tracker-name tracker_2 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --no-camera \
  --no-video
```

## 相机可视化

这两个实验模式都会直接复用 `teleop_runtime.collection` 的相机链路，因此支持：

- `--camera-config`：加载 RealSense 配置。
- `--vis_cam` / `--vis-cam`：打开实时相机预览窗口。
- `--vis-cam-key` / `--vis-camera-key`：指定预览哪一路 camera key；不指定时优先预览 `front`，没有 `front` 时预览配置里的第一路。
- `--no-video`：只预览、不导出视频；保留视频时会继续按双臂 schema 写入数据集。

固定左臂模式带相机预览示例：

```bash
python -m teleop_runtime.collection \
  --left-arm-fixed \
  --dataset-root /path/to/datasets \
  --task-name grasp_demo_left_arm_fixed_camera \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --right-tracker-name tracker_2 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/ttyUSB0 \
  --right-hand-port /dev/ttyUSB1 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --camera-config teleop_runtime/configs/devices/realsense.json \
  --vis_cam \
  --vis-cam-key front
```

抖动左臂模式同样支持相机预览，只需要在主采集命令里加入相同的 `--camera-config --vis_cam --vis-cam-key front` 参数。

## 左臂小范围抖动 + 双臂 IK

显式打开 `--left-arm-jitter`：

```bash
python -m teleop_runtime.collection \
  --left-arm-jitter \
  --dataset-root ./test_left_fix_datasets \
  --task-name grasp_demo_left_arm_jitter \
  --active-sides both \
  --host 0.0.0.0 \
  --port 9999 \
  --load-profile "$PROFILE" \
  --right-tracker-name tracker_2 \
  --left-arm-ip 192.168.101.21 \
  --right-arm-ip 192.168.1.20 \
  --left-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130705-if00 \
  --right-hand-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5ABA130707-if00 \
  --left-hand-ip 192.168.12.210 \
  --right-hand-ip 192.168.11.210 \
  --left-arm-jitter-position-m 0.01 \
  --left-arm-jitter-rotation-deg 2 \
  --left-arm-jitter-refresh-sec 1.2 \
  --left-arm-jitter-smoothing-sec 0.45 \
  --vis_cam \
  --vis-cam-key front \
  --visualize-tactile \
  --hand-tactile-regions all \
  --max-force 800
```

折纸 max force 1000, 薯片 400, 轻重 200

`--dataset-root` 是基础输出目录，实际数据集目录会自动命名为
`<dataset-root>/<YYYY-MM-DD>_<task-name>`；旧的 `--task` 仍可用作兼容入口。

抖动模式行为：

- 左臂：按 `p` 时记录当前 TCP pose 作为 anchor，之后生成围绕 anchor 的随机平滑 cartesian target。
- 抖动启停：只有按 `z` 开始录制后才推进随机抖动；录制前以及按 `c`/`d` 结束后，左臂保持静止。`c`/`d` 会把左臂停在当时当前位置，不会拉回 `p` 时的 anchor。
- 双臂 IK：左臂 jitter target 和右臂 wrist target 一起输入 `DualArmCollisionAwareRetargeter`，输出左右 arm joint command。
- 左手：继续用左 q6 控制左 InspireTac hand。
- 右臂：继续用右 wrist tracker 遥操作。
- 右手：继续用右 q6 控制右 InspireTac hand。
- InspireTac 启动 force：继承 collector 参数 `--inspire-hand-max-force N`，默认 `800`；兼容 `--max-fingertip-force` 和 `--max-force`。
- 数据：仍是双臂双手 schema，左臂 action/observation/replay joint target 来自双臂 IK 后的左臂命令和 backend 状态。

抖动参数：

- `--left-arm-jitter-position-m`：TCP 平移抖动最大半径，单位米。
- `--left-arm-jitter-rotation-deg`：TCP 姿态抖动最大角度，单位度。
- `--left-arm-jitter-refresh-sec`：多久采样一个新的随机目标。
- `--left-arm-jitter-smoothing-sec`：一阶平滑时间常数；越大越慢。
- `--left-arm-jitter-seed`：随机种子；负数使用系统随机源。

抖动模式需要双臂 IK，因此不能和 `--disable-dual-arm-collision-avoidance` 同时使用。

## 按键流程

按键和普通 collector 一致：

- `p`：同步左右臂当前真机状态；右臂采集 wrist home；左臂锁定当前位置。
- `z`：开始 episode。
- `c`：保存 episode。
- `d`：丢弃当前 episode。
- `h`：停止跟随并清除 home/fixed/jitter state。
- `q`：退出。

这两种模式都只需要右臂 wrist tracker 正常输入；左 wrist tracker 不参与运行时 home，也不是安全必需 channel。如果发送端仍然提供左 wrist tracker，collector 会把它作为非关键 channel 记录下来。
